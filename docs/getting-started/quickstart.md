# Quickstart

Get from install to your first similarity search in five minutes.

## 1. Initialize the client

```python
from pinecone import Pinecone

# Option A: read the API key from the PINECONE_API_KEY environment variable
pc = Pinecone()

# Option B: pass it explicitly
pc = Pinecone(api_key="your-api-key")
```

With neither, construction raises `PineconeValueError` rather than failing later on
the first request. See [Authentication](authentication.md).

## 2. Create an index

An index's fields are declared as a schema. This one has a single dense vector field,
`embedding`:

```python
pc.indexes.create(
    name="quickstart",
    schema={"fields": {"embedding": {"type": "dense_vector", "dimension": 3, "metric": "cosine"}}},
    deployment={"deployment_type": "managed", "cloud": "aws", "region": "us-east-1"},
)
```

`create` blocks until the index is ready, so there is nothing to poll before the next
step. Pass `timeout=-1` to return as soon as the request is accepted, or a positive
number of seconds to raise {exc}`~pinecone.errors.exceptions.PineconeTimeoutError`
instead of waiting indefinitely. If you did return early, wait for readiness yourself:

```python
import time

while not pc.indexes.describe("quickstart").status.ready:
    time.sleep(1)
```

Declaring a schema makes this a **document index**: you read and write it through
`index.documents`, and each record is a JSON document whose fields you named yourself.
The vector methods on {class}`~pinecone.index.Index` —
{meth}`~pinecone.index.Index.upsert` and {meth}`~pinecone.index.Index.query` — belong
to the older vector indexes created with top-level `dimension` and `metric`, and the
server rejects them on a document index.
See [Choosing an interface](#choosing-an-interface) below.

## 3. Get an Index client

```python
index = pc.index("quickstart")
```

The client the control plane hands back is scoped to one index and talks to the data
plane, which is a different host.

## 4. Upsert documents

Each document needs an `_id`. Every other key is either a field you declared in the
schema — here, `embedding` — or arbitrary metadata:

```python
index.documents.upsert(
    namespace="movies",
    documents=[
        {"_id": "movie-001", "embedding": [0.1, 0.2, 0.3], "title": "Arrival"},
        {"_id": "movie-002", "embedding": [0.4, 0.5, 0.6], "title": "Interstellar"},
        {"_id": "movie-003", "embedding": [0.7, 0.8, 0.9], "title": "Dune"},
    ],
)
```

Upserts apply asynchronously, so a document may not be visible to the next search
immediately.

## 5. Search

Rank documents with a `score_by` clause naming the field to compare against.
`DenseVectorQuery` scores by cosine similarity on the `embedding` field, because that
is the metric the field was declared with:

```python
from pinecone import DenseVectorQuery

results = index.documents.search(
    namespace="movies",
    top_k=3,
    score_by=[DenseVectorQuery(field="embedding", values=[0.1, 0.2, 0.3])],
    include_fields=["title"],
)
for match in results.matches:
    print(match.id, match.score)
```

Omitting `include_fields` returns only `_id` and `_score`; pass `["*"]` for every field.

## 6. Clean up

```python
pc.indexes.delete("quickstart")
```

`delete` blocks until the index is gone, the same way `create` blocks until it is
ready. Pass `timeout=-1` to return as soon as the request is accepted.

(choosing-an-interface)=
## Choosing an interface

Three data-plane interfaces exist, and the way the index was created decides which one
applies:

| You created the index with | Use | Entry point |
| --- | --- | --- |
| `schema={"fields": {...}}` naming your own vector field | documents | `index.documents` |
| top-level `dimension=` and `metric=` | vectors | `index.upsert`, `index.query` |
| `create_for_model(...)`, embedding server-side | records | `index.upsert_records`, `index.search` |

Calling the vector methods on a document index fails with an
{exc}`~pinecone.errors.exceptions.ApiError` whose message names the documents
endpoint to use instead.

New indexes should declare a schema. `dimension=`, `metric=`, `vector_type=`, and
`spec=` are still accepted by {meth}`~pinecone.client.indexes.Indexes.create`, but
they are deprecated sugar that the SDK translates into a single-field `schema=` and a
`deployment=`; see the [migration guide](../migration/v10-migration.md) for moving
between them.

## Complete example

```python
from pinecone import DenseVectorQuery, Pinecone

pc = Pinecone()  # reads PINECONE_API_KEY

pc.indexes.create(
    name="quickstart",
    schema={"fields": {"embedding": {"type": "dense_vector", "dimension": 3, "metric": "cosine"}}},
    deployment={"deployment_type": "managed", "cloud": "aws", "region": "us-east-1"},
)

index = pc.index("quickstart")

index.documents.upsert(
    namespace="movies",
    documents=[
        {"_id": "movie-001", "embedding": [0.1, 0.2, 0.3], "title": "Arrival"},
        {"_id": "movie-002", "embedding": [0.4, 0.5, 0.6], "title": "Interstellar"},
        {"_id": "movie-003", "embedding": [0.7, 0.8, 0.9], "title": "Dune"},
    ],
)

results = index.documents.search(
    namespace="movies",
    top_k=3,
    score_by=[DenseVectorQuery(field="embedding", values=[0.1, 0.2, 0.3])],
    include_fields=["title"],
)
for match in results.matches:
    print(match.id, match.score)

pc.indexes.delete("quickstart")
```
