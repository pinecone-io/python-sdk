# Pinecone Python SDK

The Pinecone Python SDK provides access to the Pinecone vector database. Use `Pinecone` for control-plane operations (creating and managing indexes) and `Index` for data-plane operations (upserting and querying vectors). The `pc.index("name")` method bridges the two.

## Quick Start

```python
import os
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# Create a serverless index
pc.indexes.create(
    name="movie-recommendations",
    dimension=1536,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)

# Get a handle to the index (data-plane operations)
index = pc.index("movie-recommendations")

# Upsert vectors
index.upsert(vectors=[
    ("movie-42", [0.012, -0.087, 0.153]),  # 1536-dim vector
    ("movie-43", [0.045, 0.021, -0.064]),  # 1536-dim vector
])

# Query by vector similarity
results = index.query(vector=[0.012, -0.087, 0.153], top_k=5)  # 1536-dim vector
for match in results.matches:
    print(match.id, match.score)
```

## Key Classes

| Class | Import | Purpose |
|---|---|---|
| `Pinecone` | `from pinecone import Pinecone` | Sync client for control-plane operations (indexes, collections, backups) |
| `AsyncPinecone` | `from pinecone import AsyncPinecone` | Async variant of `Pinecone` for use with `asyncio` |
| `Index` | Obtained via `pc.index("name")` | Sync client for data-plane operations (upsert, query, delete, fetch) |
| `AsyncIndex` | Obtained via `async_pc.index("name")` | Async variant of `Index` |
| `Admin` | `from pinecone import Admin` | Organization and project management via OAuth2 credentials |

## Control Plane vs Data Plane

`Pinecone` manages indexes, collections, and backups. It talks to the Pinecone control-plane API. `Index` performs vector operations (upsert, query, fetch, delete) against a specific index. It talks to the data-plane API hosted on the index's own endpoint. Call `pc.index("name")` to get an `Index` handle from a `Pinecone` client. The two use different hosts and authentication scopes.

## Common Workflows

### Store and retrieve vectors

```python
from pinecone import Pinecone, Vector

pc = Pinecone(api_key="your-api-key")
index = pc.index("article-search")

index.upsert(vectors=[
    Vector(id="article-101", values=[0.012, -0.087, 0.153],  # 1536-dim vector
           metadata={"topic": "science", "year": 2024}),
])

results = index.query(
    vector=[0.012, -0.087, 0.153], top_k=10,  # 1536-dim vector
    filter={"topic": "science"}, namespace="articles-en",
)
```

Use `query()` for raw vector search on standard indexes. Use `search()` for text or vector search on indexes with integrated inference (server-side embeddings).

### Data loading methods

| Method | Use when | Batching |
|--------|----------|----------|
| `index.upsert(vectors=[...])` | You have pre-computed vectors (<1000 per call) | Manual — all vectors in one request |
| `index.upsert_from_dataframe(df)` | You have a pandas DataFrame of vectors | Automatic — batches of 500 (configurable) |
| `index.upsert_records(records=[...])` | Your index uses integrated inference (server-side embedding) | Manual — all records in one request |
| `index.start_import(uri="s3://...")` | Millions of vectors in cloud storage (Parquet) | Server-side — fully async |

For datasets larger than ~1000 vectors, use `upsert_from_dataframe()` or `start_import()`. Do not pass more than ~1000 vectors to a single `upsert()` call.

### Semantic search with integrated embeddings

```python
from pinecone import Pinecone, IntegratedSpec, EmbedConfig, EmbedModel

pc = Pinecone(api_key="your-api-key")
pc.indexes.create(
    name="product-catalog",
    spec=IntegratedSpec(cloud="aws", region="us-east-1",
        embed=EmbedConfig(model=EmbedModel.Multilingual_E5_Large,
                          field_map={"text": "description"})),
)
index = pc.index("product-catalog")
index.upsert_records(namespace="products", records=[
    {"id": "prod-001", "description": "Lightweight running shoes", "category": "footwear"},
])
results = index.search(
    namespace="products",
    top_k=5,
    inputs={"text": "comfortable shoes for trail running"},
)
for hit in results.result.hits:
    print(hit.id, hit.score)
```

### Generate embeddings and rerank

```python
pc = Pinecone(api_key="your-api-key")

# Embed text
response = pc.inference.embed(
    model="multilingual-e5-large",
    inputs=["How do I reset my password?"],
    parameters={"input_type": "query"},
)

# Rerank documents by relevance
result = pc.inference.rerank(
    model="pinecone-rerank-v0", query="best budget laptop", top_n=2,
    documents=["Affordable laptops under $500", "Premium gaming desktops"],
)
```

### Other data-plane operations

```python
pc = Pinecone(api_key="your-api-key")
index = pc.index("article-search")

# Fetch vectors by ID and inspect their values and metadata
result = index.fetch(ids=["movie-42", "movie-87"])
print(result.vectors["movie-42"].values)
print(result.vectors["movie-42"].metadata)

# Delete specific vectors or an entire namespace
index.delete(ids=["movie-42"])
index.delete(delete_all=True, namespace="old-data")

# Check vector counts and which namespaces exist
stats = index.describe_index_stats()
print(stats.total_vector_count)
print(stats.namespaces)
```

### Metadata filtering

Filter vectors by metadata fields using the operators below. Filters work on both `query(filter=...)` and `search(filter=...)`.

| Operator | Description |
|----------|-------------|
| `$eq` / `$ne` | Equal / not equal |
| `$gt` / `$gte` / `$lt` / `$lte` | Numeric comparison |
| `$in` / `$nin` | Set membership / exclusion |
| `$and` / `$or` | Logical combinators |

```python
# Range filter
results = index.query(vector=[...], top_k=10, filter={"year": {"$gte": 2020, "$lte": 2024}})

# Set membership
results = index.query(vector=[...], top_k=10, filter={"category": {"$in": ["science", "tech"]}})

# Combined condition
results = index.query(vector=[...], top_k=10,
    filter={"$and": [{"year": {"$gte": 2020}}, {"category": {"$in": ["science"]}}]})
```

### Backups and collections

```python
pc = Pinecone(api_key="your-api-key")

# Create a backup of an index
backup = pc.backups.create(index_name="my-index", name="pre-migration")

# Restore the backup as a new index
pc.create_index_from_backup(backup_id=backup.backup_id, name="my-index-restored")

# Collections (snapshots for pod-based indexes)
pc.collections.create(name="snapshot", source="my-index")
collection = pc.collections.describe("snapshot")
```

### Organization and project management (Admin API)

The `Admin` client uses OAuth2 credentials (not API keys) for organization-level operations.

```python
from pinecone import Admin, Pinecone, ServerlessSpec

admin = Admin(client_id="my-client-id", client_secret="my-client-secret")
# Or set PINECONE_CLIENT_ID and PINECONE_CLIENT_SECRET env vars

# List organizations and projects
for org in admin.organizations.list():
    print(org.name, org.id)

# Create a project and API key
project = admin.projects.create(name="my-project")
key = admin.api_keys.create(project_id=project.id, name="my-key")

# Bridge to Pinecone for data operations
pc = Pinecone(api_key=key.value)
pc.indexes.create(name="my-index", dimension=1536, metric="cosine",
                  spec=ServerlessSpec(cloud="aws", region="us-east-1"))
```

OAuth credentials are created in the Pinecone console under Organization Settings → Service Accounts.

## Async Usage

```python
from pinecone import AsyncPinecone

async with AsyncPinecone(api_key="your-api-key") as pc:
    desc = await pc.indexes.describe("my-index")
    index = pc.index(host=desc.host)
    async with index:
        results = await index.query(vector=[0.012, -0.087, 0.153], top_k=5)
        for match in results.matches:
            print(match.id, match.score)
```

```python
# Async with integrated inference
async with AsyncPinecone(api_key="your-api-key") as pc:
    desc = await pc.indexes.describe("my-index")
    index = pc.index(host=desc.host)
    async with index:
        await index.upsert_records(
            namespace="articles",
            records=[
                {"_id": "doc1", "text": "Vector databases enable similarity search."},
                {"_id": "doc2", "text": "RAG combines search with LLMs."},
            ],
        )
        results = await index.search(
            namespace="articles",
            top_k=5,
            inputs={"text": "how does vector search work?"},
        )
        for hit in results.result.hits:
            print(hit.id, hit.score)
```

**Note:** `AsyncPinecone.index(name=...)` is a coroutine — use `await pc.index(name="my-index")`. On cache miss it performs a non-blocking `await pc.indexes.describe(name)` to resolve the host automatically, matching sync `Pinecone.index(name=...)` behavior.

## Error Handling

All SDK exceptions inherit from `PineconeError`:

```
PineconeError
├── ApiError              # HTTP error from the API (has .status_code, .body)
│   ├── NotFoundError     # 404
│   ├── UnauthorizedError # 401
│   ├── ForbiddenError    # 403
│   ├── ConflictError     # 409
│   └── ServiceError      # 5xx
├── PineconeValueError    # Invalid argument (also a ValueError)
├── PineconeTypeError     # Wrong type (also a TypeError)
├── PineconeTimeoutError  # Request timed out
├── PineconeConnectionError # Network connectivity failure
└── ResponseParsingError  # Unexpected response format
```

Catch specific exceptions (`NotFoundError`, `UnauthorizedError`, etc.) or the base `ApiError` for HTTP errors. `ApiError` exposes `.status_code` and `.body` attributes.

```python
from pinecone import Pinecone, ApiError, PineconeError

pc = Pinecone()
index = pc.index("my-index")
try:
    index.upsert(vectors=[("id1", [0.1, 0.2, 0.3])])
except ApiError as e:
    print(e.status_code, e.body)  # HTTP error — check status code
except PineconeError as e:
    print(e)  # Validation, timeout, or connection error
```

**Retry behavior (HTTP):** All HTTP methods (GET, HEAD, POST, PUT, PATCH, DELETE) are automatically retried on transient failures: connection errors (`httpx.TransportError`), 408 Request Timeout, 429 Too Many Requests, and 5xx (500, 502, 503, 504). Pinecone's data-plane writes are idempotent at the server (upsert overwrites by ID, delete-by-ID is idempotent, update-by-ID is idempotent), so retrying upsert/query/fetch/delete/update on transient errors is safe. Backoff is floored full jitter: `uniform(0.1 * base, base)` where `base = min(backoff_factor**attempt, max_wait)`. Configure via `RetryConfig`.

**Retry behavior (gRPC):** gRPC retries on UNAVAILABLE, RESOURCE_EXHAUSTED (rate limit), and ABORTED (concurrency conflict). DEADLINE_EXCEEDED is not retried — set a longer client timeout instead. All three default retryable codes are safe for Pinecone data-plane operations (upsert, query, fetch, delete-by-id, update), which are idempotent. Backoff uses full-jitter exponential: `uniform(0, min(max_backoff, initial_backoff * multiplier^attempt))`. The set of retryable codes is configurable via `RetryConfig.retryable_codes`.

## Response Objects

Access patterns for the most common response types:

```python
# QueryResponse — from index.query()
results = index.query(vector=[0.012, -0.087, 0.153], top_k=5)  # 1536-dim vector
for match in results.matches:
    print(match.id, match.score)    # id and similarity score
    print(match.values)             # vector values (if include_values=True)
    print(match.metadata)           # metadata dict (if include_metadata=True)

# SearchRecordsResponse — from index.search() with integrated embeddings
results = index.search(namespace="products", top_k=5, inputs={"text": "..."})
for hit in results.result.hits:
    print(hit.id, hit.score)        # id and similarity score
    print(hit.fields)               # record fields dict

# EmbeddingsList — from pc.inference.embed()
embeddings = pc.inference.embed(model="multilingual-e5-large", inputs=["text"])
for embedding in embeddings:
    print(embedding.values)         # list of floats
```

## Common Mistakes

- **Calling `pc.upsert()` instead of `index.upsert()`** — upsert, query, fetch, and delete are on the `Index` object, not on `Pinecone`. Use `index = pc.index("name")` first.
- **Not waiting for index readiness** — a freshly created index is not immediately ready. Use `pc.indexes.describe("name")` and check `status.ready` before upserting. By default, `pc.indexes.create()` polls until the index is ready. Pass `timeout=-1` to return immediately without waiting.
- **Forgetting the namespace** — vectors in different namespaces are isolated. If you upsert to `namespace="articles-en"` but query without specifying a namespace, you query the default (`""`) namespace and get no results.
- **Mismatched vector dimensions** — the vector length in upsert and query must match the index's `dimension`. The API returns an error if they differ.
- **Using `from pinecone import Index` directly** — `Index` requires a host URL. Use `pc.index("name")` to resolve the host automatically.
