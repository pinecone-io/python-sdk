# How Pinecone Works

A vector database stores numerical representations of data, called vectors or embeddings,
and retrieves the entries most similar to a query vector. Unlike a relational database
that matches rows by exact field values, a vector database uses approximate
nearest-neighbor algorithms to rank results by geometric closeness in high-dimensional
space.

This page defines the vocabulary the rest of the documentation uses, and shows where each
concept appears in the SDK.


## Indexes

An index is the unit you create, query, and delete. It holds the data you store and
serves searches over it.

Every index is created with a **schema**: a map of field names to typed field
configurations. The schema declares the fields that get *searched*, and it cannot change
after the index is created.

```python
from pinecone import Pinecone

pc = Pinecone()
pc.indexes.create(
    name="support-articles",
    schema={
        "fields": {
            "embedding": {"type": "dense_vector", "dimension": 1024, "metric": "cosine"},
            "keywords": {"type": "sparse_vector"},
            "body": {"type": "string", "full_text_search": {"language": "en"}},
        }
    },
    deployment={"deployment_type": "managed", "cloud": "aws", "region": "us-east-1"},
)
```

Three field types can be declared at creation time:

| Field type | What it does |
|---|---|
| `dense_vector` | Similarity search over embeddings. At most one per schema. Takes `dimension` and `metric`. |
| `sparse_vector` | Term-based scoring, such as BM25. Has no fixed dimension. |
| `string` with `full_text_search` | Full-text search over a text field. |

Everything else you store is **metadata**. Metadata fields are not declared in the
schema; Pinecone indexes them for filtering the first time they appear on a record you
write.

{doc}`SchemaBuilder </reference/schema-builder>` builds the same dict with a fluent,
validating API, so a bad field name or an out-of-range dimension fails before the
request goes out:

```python
from pinecone import Pinecone, SchemaBuilder

pc = Pinecone()
schema = (
    SchemaBuilder()
    .add_dense_vector_field("embedding", dimension=1024, metric="cosine")
    .add_string_field("body", full_text_search={"language": "en"})
    .build()
)
pc.indexes.create(name="movie-recommendations", schema=schema)
```

:::{note}
`dimension=`, `metric=`, `vector_type=`, and `spec=` on `pc.indexes.create()` are
deprecated sugar retained for callers written against 9.x. They translate into a
single-field `schema=` and a `deployment=`, and cannot be combined with them. New code
should declare `schema=` directly. See the
[v10 migration guide](../migration/v10-migration.md).
:::


### Dimension and metric

- **Dimension** is the length of the vectors stored in a `dense_vector` field. Every
  dense vector written to that field must have exactly this many values.
- **Metric** is the similarity function used when ranking results from that field:
  `cosine`, `euclidean`, or `dotproduct`.

Both belong to a *field*, not to the index — an index can carry a dense field and
a sparse field, and the sparse field has neither. Read them off the schema:

```python
from pinecone import Pinecone

pc = Pinecone()
desc = pc.indexes.describe("movie-recommendations")

for field_name, field in desc.schema.fields.items():
    print(field_name, type(field).__name__)
```

`IndexModel.dimension`, `.metric`, and `.vector_type` survive as deprecated properties
that resolve only when the schema has exactly one vector field. On a schema with more than
one — or with none — reading them raises `AttributeError` naming the fields it found. Read
`desc.schema.fields["embedding"].dimension` instead.


## Deployment types

The `deployment` argument decides where and how the index runs. It is a dict
discriminated on `deployment_type`, and it defaults to a managed index on AWS
`us-east-1`.

| `deployment_type` | Product name | Capacity | Also configured by |
|---|---|---|---|
| `"managed"` | Serverless | Scales automatically | `read_capacity` |
| `"pod"` | Pod-based | Fixed by `pod_type`, `replicas`, and `shards` | — |
| `"byoc"` | BYOC (bring your own compute) | Runs in your own infrastructure | `read_capacity` |

A managed index takes `cloud` and `region`. A pod-based index takes `environment`,
`pod_type`, `replicas`, and `shards` — all four are required. A BYOC index takes the
`environment` identifier of your provisioned environment.

`read_capacity` tunes reads on managed and BYOC indexes. `{"mode": "OnDemand"}` is the
default and scales with traffic, with nothing to size. `{"mode": "Dedicated", ...}`
provisions a node type and a shard and replica count you control.

See [Serverless indexes](../how-to/indexes/serverless.md) and
[Pod-based indexes](../how-to/indexes/pod.md) for the full set of options.


## Namespaces

A namespace is a logical partition within an index. Data in different namespaces is
isolated: writes, searches, fetches, and deletes in one namespace never touch another.
Namespaces are the usual way to separate data by tenant, language, or environment without
creating separate indexes.

The vector methods `Index.upsert` and `Index.query` default to the empty string `""`,
which is the default namespace. The document and record methods take no default — they
require a non-empty namespace argument.

See [Working with namespaces](../how-to/vectors/namespaces.md).


## Records

A record is one stored entry: an ID plus the data attached to it. **How the index was
created decides which data-plane interface reads and writes its records.**

| Index created with | Records are | Read and write with |
|---|---|---|
| `schema=` naming your own fields | **documents** — an `_id` plus your fields | `index.documents` |
| the deprecated `dimension=`/`metric=` | **vectors** — an ID plus coordinates | `index.upsert`, `index.query` |
| `pc.indexes.create_for_model(...)` | **records** with text Pinecone embeds for you | `index.upsert_records`, `index.search` |

The interfaces are not interchangeable: calling the vector methods on a schema-based index
is rejected by the server.

### Documents

A document is a JSON object carrying the reserved `_id` key. Every other key is a field
of your own — either declared in the schema, or free-form metadata indexed for filtering
on first write.

```python
from pinecone import Pinecone

pc = Pinecone()
with pc.index(name="support-articles") as index:
    index.documents.upsert(
        namespace="published",
        documents=[
            {"_id": "article-101", "body": "Roman aqueducts", "views": 12},
        ],
    )

    hits = index.documents.search(
        namespace="published",
        top_k=5,
        score_by=[{"type": "text", "query": "aqueducts", "fields": ["body"]}],
    )
    for match in hits.matches:
        print(match.id)
```

`score_by` is what makes a document search a *search*: each clause names one scoring
method — a dense vector, a sparse vector, a BM25 text query, or a Lucene query string —
and a search may combine several. See
{doc}`the Index reference </reference/sync-index>` for every document operation.

### Vectors

A vector carries its coordinates in either or both of two representations:

| Component | Type | Description |
|---|---|---|
| `id` | `str` | Unique identifier within a namespace |
| `values` | `list[float]` | Dense coordinates, one per dimension of the field |
| `sparse_values` | `SparseValues` | Non-zero dimensions only, as parallel `indices` and `values` |
| `metadata` | `dict[str, Any]` | Your own key-value pairs to filter on |

At least one of `values` and `sparse_values` must be populated — a vector with no
coordinates cannot be scored against anything. Which ones you populate is what makes a
vector dense, sparse, or hybrid:

- **Dense** — `values` only. The usual output of an embedding model; finds records by
  meaning.
- **Sparse** — `sparse_values` only. How term-based scoring such as BM25 is expressed;
  finds records by exact term. `values` stays empty.
- **Hybrid** — both, on the same record, so it is reachable by meaning and by term. The
  index has to declare a dense field and a sparse field for this to be accepted.

```python
from pinecone import Pinecone, SparseValues, Vector

dense = Vector(id="article-101", values=[0.12, 0.34, 0.56])
print(dense.sparse_values)  # None — nothing sparse on this record

hybrid = Vector(
    id="article-102",
    values=[0.12, 0.34, 0.56],
    sparse_values=SparseValues(indices=[10, 42], values=[0.4, 0.9]),
    metadata={"topic": "science", "published": 2024},
)

pc = Pinecone()
with pc.index(host="my-index-abc123.svc.pinecone.io") as index:
    results = index.query(
        vector=hybrid.values,
        sparse_vector=hybrid.sparse_values,
        top_k=10,
        filter={"topic": "science"},
    )
    for match in results.matches:
        print(match.id, match.score)
```

See [Upsert and query vectors](../how-to/vectors/upsert-and-query.md).

### Integrated inference

An index created with `pc.indexes.create_for_model()` embeds text for you. The `field_map`
names the record field to embed, and the same model embeds your queries at read time. In
the returned index that field shows up in the schema as a `semantic_text` field —
`create_for_model` is the only way to get one, since `semantic_text` cannot be declared in
a `schema=` you pass to `create()`.

```python
from pinecone import Pinecone

pc = Pinecone()
pc.indexes.create_for_model(
    name="semantic-search",
    cloud="aws",
    region="us-east-1",
    embed={"model": "multilingual-e5-large", "field_map": {"text": "chunk_text"}},
)

index = pc.index(name="semantic-search")
index.upsert_records(
    namespace="articles-en",
    records=[
        {"_id": "article-1", "chunk_text": "Quantum computing advances"},
        {"_id": "article-2", "chunk_text": "New discoveries in marine biology"},
    ],
)
```

Read it back with `index.search(namespace=..., inputs={"text": ...}, top_k=...)`, which
embeds the query string server-side. See
[Integrated records](../how-to/integrated-records.md).

For embedding and reranking as standalone operations, against text that is not going into
an index, use `pc.inference` — see [Embeddings](../how-to/inference/embeddings.md) and
[Reranking](../how-to/inference/reranking.md).


## Control plane vs data plane

Operations fall into two groups, served by different hosts.

The **control plane** manages index lifecycle: create, list, describe, configure, and
delete indexes, plus collections, backups, and restore jobs. Control-plane calls go to
`api.pinecone.io` and are made through the `Pinecone` client and its sub-clients.

The **data plane** reads and writes the records inside one index. Data-plane calls go to
that index's own host and are made through an `Index` (or `AsyncIndex`, or `GrpcIndex`).
Both planes authenticate with the same API key.

`pc.index()` is the bridge. Pass a `host` and it is used as-is. Pass a `name` and the host
is resolved with one describe request, then cached on the client, so a later call for the
same name costs nothing.

```python
from pinecone import Pinecone

pc = Pinecone()

# Control plane: what indexes exist, and where does this one live?
desc = pc.indexes.describe("movie-recommendations")
print(desc.host)

# Data plane: talk to that index directly
with pc.index(host=desc.host) as index:
    print(index.describe_index_stats().total_vector_count)
```

`Index` holds its own HTTP connection pool, separate from the one the `Pinecone` client
uses. Closing the client does not close index clients, so close each one — the `with`
block above does it for you.


## Sub-clients

`Pinecone` exposes control-plane operations as sub-clients rather than a flat list of
methods. Each is a lazily created property, so nothing is constructed until you touch it.

| Sub-client | Operations |
|---|---|
| `pc.indexes` | Create, list, describe, configure, delete indexes |
| `pc.collections` | Create, list, describe, delete collections |
| `pc.backups` | Create, list, describe, delete backups |
| `pc.backup_schedules` | Manage recurring backup schedules |
| `pc.restore_jobs` | Track restores of a backup into a new index |
| `pc.inference` | Embed text, rerank results |
| `pc.assistants` | Manage Pinecone Assistants |

```python
from pinecone import Pinecone

pc = Pinecone()

for index_model in pc.indexes.list():
    print(index_model.name, index_model.status.state)

if pc.indexes.exists("movie-recommendations"):
    pc.indexes.delete("movie-recommendations", timeout=-1)
```

`pc.indexes.delete()` blocks until the index is gone, polling until then. `timeout=-1`
returns as soon as the delete is accepted; a positive `timeout` raises
`PineconeTimeoutError` if the index outlives it.

`AsyncPinecone` carries the same sub-clients with the same names. Note that the
list-shaped methods on them are not coroutines — see
[Sync vs Async Clients](sync-vs-async.md).


## Collections and backups

Both are point-in-time snapshots of an index's data, held outside the index. Which one
applies depends on the index's deployment type, and they differ in what you can do with
the result.

| | Collection | Backup |
|---|---|---|
| Source index | Pod-based | Serverless or BYOC |
| Sub-client | `pc.collections` | `pc.backups` |
| Restorable | No | Yes, with `pc.create_index_from_backup()` |

```python
from pinecone import Pinecone

pc = Pinecone()

for collection in pc.collections.list():
    print(collection.name, collection.status)

for backup in pc.backups.list(limit=10):
    print(backup.backup_id, backup.source_index_name, backup.status)
```

See [Collections](../how-to/collections.md) and
[Backups and restore](../how-to/indexes/backups-and-restore.md).


## Where to go next

- [Quickstart](../getting-started/quickstart.md) — create an index and search it end to end.
- [Sync vs Async Clients](sync-vs-async.md) — which client pair to reach for.
- [Error handling](error-handling.md) — the exception hierarchy every method shares.
- [Pagination](pagination.md) — how the list-shaped methods page.
- [v10 migration guide](../migration/v10-migration.md) — the field-by-field mapping from 9.x.
