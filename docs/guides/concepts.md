# How Pinecone Works

A vector database stores numerical representations of data, called vectors or embeddings,
and retrieves the entries most similar to a query vector. Unlike a relational database
that matches rows by exact field values, a vector database uses approximate nearest-neighbor
algorithms to rank results by geometric closeness in high-dimensional space.


## Indexes

An index holds the vectors you store and query. Its schema declares the fields it
searches on, most commonly a single dense vector field with two required properties:

- **Dimension**: the length of vectors stored in that field. Every vector you upsert must
  have exactly this many values.
- **Metric**: the similarity function used when ranking query results: `cosine`,
  `euclidean`, or `dotproduct`.

Pinecone offers two index types:

| | Serverless | Pod-based |
|---|---|---|
| Capacity | Scales automatically | Fixed by pod type and count |
| Billing | Pay per operation | Pay per pod-hour |
| Use case | Variable or unpredictable workloads | Predictable, high-QPS workloads |

Create a serverless index:

```python
from pinecone import Pinecone, SchemaBuilder

pc = Pinecone()
schema = (
    SchemaBuilder()
    .add_dense_vector_field("embedding", dimension=1536, metric="cosine")
    .build()
)
pc.indexes.create(
    name="movie-recommendations",
    schema=schema,
    deployment={"deployment_type": "managed", "cloud": "aws", "region": "us-east-1"},
)
```


## Namespaces

A namespace is a logical partition within an index. Vectors in different namespaces are
completely isolated: upserts, queries, fetches, and deletes in one namespace never touch
another.

Common uses for namespaces include separating data by tenant, language, or environment
without creating separate indexes:

```python
index = pc.index(host="my-index-abc123.svc.pinecone.io")

# Each customer's vectors are isolated in their own namespace
index.upsert(vectors=[("doc-1", [0.1, 0.2, ...])], namespace="customer-acme")
index.upsert(vectors=[("doc-1", [0.4, 0.5, ...])], namespace="customer-globex")

results = index.query(vector=[0.1, 0.2, ...], top_k=5, namespace="customer-acme")
```

The empty string `""` is the default namespace. All operations that omit a namespace
target it implicitly.


## Vectors

A vector has four components:

| Component | Type | Required | Description |
|---|---|---|---|
| `id` | `str` | Yes | Unique identifier within a namespace |
| `values` | `list[float]` | Yes | Dense embedding values |
| `sparse_values` | `SparseValues` | No | Sparse representation for hybrid search |
| `metadata` | `dict[str, Any]` | No | Key-value data for filtering |

Upsert vectors by passing tuples or `Vector` objects:

```python
from pinecone import Vector

# Minimal tuple form
index.upsert(vectors=[("article-42", [0.012, -0.087, 0.153, ...])])

# Full object form with metadata
index.upsert(vectors=[
    Vector(
        id="article-42",
        values=[0.012, -0.087, 0.153, ...],
        metadata={"topic": "science", "published": 2024},
    ),
])
```

Query for similar vectors:

```python
results = index.query(
    vector=[0.012, -0.087, 0.153, ...],
    top_k=10,
    filter={"topic": "science"},
)
for match in results.matches:
    print(match.id, match.score)
```


## Records (Integrated Indexes)

Integrated indexes store text or structured data alongside each vector. Pinecone generates
the embeddings server-side using a hosted model. You upsert text records; the index
handles embedding automatically.

```python
from pinecone import Pinecone

pc = Pinecone()
pc.indexes.create_for_model(
    name="article-search",
    cloud="aws",
    region="us-east-1",
    embed={
        "model": "multilingual-e5-large",
        "field_map": {"text": "text"},
    },
)

index = pc.index(name="article-search")
index.upsert_records(
    namespace="articles-en",
    records=[
        {"id": "article-1", "text": "Quantum computing advances in 2024"},
        {"id": "article-2", "text": "New discoveries in marine biology"},
    ],
)

results = index.search(
    namespace="articles-en",
    inputs={"text": "latest physics research"},
    top_k=5,
)
```


## Control Plane vs Data Plane

Operations fall into two categories:

The **control plane** handles index lifecycle management: create, list, describe,
configure, and delete indexes, plus collections and backups. Control-plane calls go
through `api.pinecone.io`, accessed via the `Pinecone` client.

The **data plane** handles vector operations: upsert, query, fetch, update, delete, and
list vectors. Data-plane calls go directly to an index's host URL, accessed via the
`Index` (or `AsyncIndex`) client.

```python
from pinecone import Pinecone

pc = Pinecone()

# Control plane: describe an index to get its host
desc = pc.indexes.describe("movie-recommendations")

# Data plane: connect directly to the index
index = pc.index(host=desc.host)
index.upsert(vectors=[("movie-42", [0.1, 0.2, ...])])
```


## Namespace Pattern in the SDK

The `Pinecone` client exposes related operations as namespace objects rather than a flat
list of methods. This keeps the client's top-level API small and groups related
functionality together:

| Namespace | Operations |
|---|---|
| `pc.indexes` | Create, list, describe, configure, delete indexes |
| `pc.collections` | Create, list, describe, delete collections |
| `pc.inference` | Embed text, rerank results |
| `pc.assistants` | Manage Pinecone Assistants |

```python
pc = Pinecone()

# List all indexes
for index_model in pc.indexes.list():
    print(index_model.name, index_model.status.ready)

# Describe one index
desc = pc.indexes.describe("movie-recommendations")

# Delete an index
pc.indexes.delete("movie-recommendations")
```
