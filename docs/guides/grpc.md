# Using the gRPC Client

The SDK includes a `GrpcIndex` client that routes data-plane operations through a gRPC
transport backed by a native Rust extension (`pinecone._grpc`). For bulk upsert and
high-throughput workloads, gRPC typically delivers better performance than the default
REST client because it uses binary serialization and HTTP/2 multiplexing.

gRPC transport is included in the base `pinecone` package — no extra install required.

## Creating a GrpcIndex

You can obtain a `GrpcIndex` in two ways.

**Via `Pinecone.index()` with `grpc=True`** (recommended — resolves the host automatically):

```python
from pinecone import Pinecone

pc = Pinecone()
index = pc.index(name="product-search", grpc=True)
```

**Directly** (when you already know the host):

```python
from pinecone.grpc import GrpcIndex

index = GrpcIndex(
    host="product-search-abc123.svc.pinecone.io",
    api_key="YOUR_API_KEY",  # or set PINECONE_API_KEY env var
)
```

`GrpcIndex` is a context manager; always close it when finished:

```python
with GrpcIndex(host="product-search-abc123.svc.pinecone.io") as index:
    index.upsert(vectors=[("product-42", [0.1, 0.2, ...])])
```

## Basic Operations

`GrpcIndex` exposes the same interface as the HTTP `Index`:

```python
# Upsert
response = index.upsert(
    vectors=[
        ("product-42", [0.1, 0.2, ...]),
        ("product-99", [0.3, 0.4, ...]),
    ],
    namespace="catalog",
)
print(response.upserted_count)

# Query
results = index.query(
    top_k=10,
    vector=[0.1, 0.2, ...],
    namespace="catalog",
)
for match in results.matches:
    print(match.id, match.score)
```

## Async (Non-Blocking) Operations with PineconeFuture

Every data-plane method has an `_async` variant that returns a
{class}`~pinecone.grpc.future.PineconeFuture` immediately without blocking:

```python
from concurrent.futures import as_completed

futures = [
    index.upsert_async(vectors=[("product-42", [0.1, 0.2, ...])]),
    index.upsert_async(vectors=[("product-99", [0.3, 0.4, ...])]),
]

# Collect results as they complete
for future in as_completed(futures):
    result = future.result()  # blocks up to the default 5-second timeout
    print(result.upserted_count)
```

### PineconeFuture reference

| Method | Description |
|--------|-------------|
| `future.result(timeout=5.0)` | Block until the result is ready; raises `PineconeTimeoutError` if the timeout elapses |
| `future.done()` | `True` if the operation has completed (or been cancelled) |
| `future.cancel()` | Attempt to cancel the operation |

Pass `timeout=None` to `result()` to block indefinitely:

```python
result = future.result(timeout=None)
```

`PineconeFuture` is compatible with `concurrent.futures.as_completed()` and
`concurrent.futures.wait()`, so it integrates naturally with thread-pool patterns.

## Bulk Upsert from a DataFrame

For large-scale ingestion, `upsert_from_dataframe()` splits a pandas `DataFrame` into
batches and submits them via `upsert_async()`:

```python
import pandas as pd
from pinecone.grpc import GrpcIndex

df = pd.DataFrame([
    {"id": "product-42", "values": [0.1, 0.2, ...]},
    {"id": "product-99", "values": [0.3, 0.4, ...]},
])

with GrpcIndex(host="product-search-abc123.svc.pinecone.io") as index:
    response = index.upsert_from_dataframe(df, namespace="catalog", batch_size=500)
    print(response.upserted_count)
```

For large or slow ingests, pass `timeout` to raise the server-side deadline
applied to each batch (it bounds each batch, not the whole DataFrame). `None`,
the default, uses the client's configured request timeout (20s unless you set
`GrpcIndex(..., timeout=...)`):

```python
with GrpcIndex(host="product-search-abc123.svc.pinecone.io") as index:
    response = index.upsert_from_dataframe(
        df, namespace="catalog", batch_size=200, timeout=120.0
    )
    print(response.upserted_count)
```

## When to Prefer gRPC

| Scenario | Recommendation |
|----------|---------------|
| Bulk upsert (thousands of vectors) | gRPC — lower per-call overhead |
| High-throughput query loops | gRPC with `*_async()` |
| Async Python frameworks (FastAPI, asyncio) | Use `AsyncIndex` instead — `GrpcIndex` does not support `async/await` |
| Simple scripts and CLI tools | Either works; HTTP `Index` has no extra dependency |

## Limitations

- `GrpcIndex` is **sync only**. It does not support Python's `async/await`. For async
  concurrency, use `PineconeFuture` or switch to `AsyncIndex`.
- The Rust-backed gRPC channel (`pinecone._grpc`) is a platform-specific native extension
  bundled in the base package. If installation fails on your platform, check the package's
  supported-platform list for available wheels.
- `upsert_records` and `search` on `GrpcIndex` are routed over REST (the Pinecone gRPC
  API does not expose those endpoints).
