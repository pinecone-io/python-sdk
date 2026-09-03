# Using the gRPC Client

The SDK includes a `GrpcIndex` client that routes data-plane operations through a gRPC
transport backed by a native Rust extension (`pinecone._grpc`). For bulk upsert and
high-throughput workloads, gRPC typically delivers better performance than the default
REST client because it uses binary serialization and HTTP/2 multiplexing.

gRPC transport is included in the base `pinecone` package. There is no `grpc` extra to
install, and no `grpcio` dependency to add.

## Creating a GrpcIndex

You can obtain a `GrpcIndex` in two ways.

**Via `Pinecone.index()` with `grpc=True`** (recommended, since it resolves the host automatically):

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

Every argument is keyword-only, and a missing API key raises before the channel is
built. `GrpcIndex` is the canonical name; `GRPCIndex` is a legacy capitalisation alias
for the same class.

`GrpcIndex` is a context manager; always close it when finished:

```python
with GrpcIndex(host="product-search-abc123.svc.pinecone.io") as index:
    index.upsert(vectors=[("product-42", [0.012, -0.087, 0.153])])
```

## Endpoint scheme

The channel dials `https`. A data plane reached over something else — a plaintext
gateway, an egress proxy fronting a private endpoint, or a local simulator — needs
`grpc_scheme` set, on the client or on the index:

```python
pc = Pinecone(grpc_scheme="http")
index = pc.index(name="product-search", grpc=True)

index = GrpcIndex(host="http://10.0.0.7:50051", grpc_scheme="http")
```

`PINECONE_GRPC_SCHEME` sets the same thing through the environment; the keyword
argument wins over it.

The scheme decides whether the wire carries TLS, and `secure` supplies the material
for the handshake, so `grpc_scheme="http"` is plaintext whatever `secure` says.
`grpc_scheme="https"` with `secure=False` cannot connect and is refused when the
index is built. Leaving `grpc_scheme` unset takes the scheme from `secure`.

## Basic Operations

`GrpcIndex` carries the same data-plane methods as the HTTP `Index` — with one
exception, `documents`, covered under [Limitations](#limitations):

```python
# Upsert
response = index.upsert(
    vectors=[
        ("product-42", [0.012, -0.087, 0.153]),
        ("product-99", [0.045, 0.021, -0.064]),
    ],
    namespace="catalog",
)
print(response.upserted_count)

# Query
results = index.query(
    top_k=10,
    vector=[0.012, -0.087, 0.153],
    namespace="catalog",
)
for match in results.matches:
    print(match.id, match.score)
```

The vector literals here are three floats so they fit on the page; a real one carries as
many values as the index's vector field declares.

Not every method actually travels over gRPC. `upsert_records`, `search`, `search_records`
and the bulk-import methods (`start_import`, `describe_import`, `cancel_import`,
`list_imports`, `list_imports_paginated`) go out over REST from the same client, because
the Pinecone gRPC API exposes no equivalent. They work; they just do not get the
transport's benefit.

## Retries and Timeouts

gRPC and REST share the retry shape; [Retries and Resilience](retries.md#transport-differences)
owns the comparison. What follows is only where gRPC diverges.

**What it retries.** Exactly three gRPC status codes: `UNAVAILABLE`,
`RESOURCE_EXHAUSTED` and `ABORTED`. The defaults differ from REST's —
`max_retries=5`, `backoff_factor=0.1` — and a `retry_config` left unset on
`Pinecone()` is *not* inherited here, precisely so those defaults survive. Pass one
explicitly, to `Pinecone()` or straight to `GrpcIndex()`, and it applies.

**`DEADLINE_EXCEEDED` is not one of the three, and `max_retries` is the wrong knob for
it.** Raising retries is the intuitive response and it does nothing: a call whose
deadline expires raises `PineconeTimeoutError` after a single attempt, having never
been retried. The knob is a longer deadline.

```python
from pinecone import RetryConfig
from pinecone.grpc import GrpcIndex

host = "product-search-abc123.svc.pinecone.io"

# Does nothing for a deadline — the call was never retried.
index = GrpcIndex(host=host, retry_config=RetryConfig(max_retries=20))

# This. `timeout` is the per-attempt deadline; the index-level default is 20 seconds.
index = GrpcIndex(host=host, timeout=120.0)
```

A per-call `timeout=` does not replace the index-level one — both apply, and the
shorter fires. For a bulk method, `total_timeout` bounds the whole job rather than one
attempt.

**Backoff is decorrelated jitter**, `uniform(backoff_factor, previous_delay * 3)`
capped at `max_wait` — not an exponential curve. The first retry's window is seeded at
ten times `backoff_factor` rather than at `backoff_factor` itself: a backend restart
returns `UNAVAILABLE` to every client at once, and a narrow first window is where a
thundering herd forms.

**A server pushback hint wins over the computed backoff.** When a response carries
`grpc-retry-pushback-ms` (or `retry-after`), that value is used instead, clamped to
`max_wait` and smeared so concurrent clients do not all wake together. A *negative*
value is the server saying do not retry at all, and the call fails immediately.

**The retry budget is per channel**, not per process — one `GrpcIndex` holds one
budget. Sustained retrying drains it, after which retryable failures fail fast instead
of retrying, so a struggling backend is not hammered by a client that is only ever
retrying.

**Two things you cannot configure.** `RetryConfig.retryable_status_codes` carries HTTP
statuses and is deliberately not forwarded to this transport; the gRPC code set is
fixed and not reachable from Python. And a `retry_config` given to `GrpcIndex` reaches
only the gRPC channel — the REST client it keeps alongside for `upsert_records`,
`search` and the import methods stays on REST's defaults, so `max_retries` does not
cover everything on the object.

## Async (Non-Blocking) Operations with PineconeFuture

`upsert`, `query`, `fetch`, `delete`, `update`, and `query_namespaces` each have an
`_async` variant that hands the call to a background thread and returns a
`PineconeFuture` immediately:

```python
from concurrent.futures import as_completed

futures = [
    index.upsert_async(vectors=[("product-42", [0.012, -0.087, 0.153])]),
    index.upsert_async(vectors=[("product-99", [0.045, 0.021, -0.064])]),
]

# Collect results as they complete
for future in as_completed(futures):
    result = future.result()  # blocks up to the default 5-second timeout
    print(result.upserted_count)
```

Issuing several at once is the reason to prefer these over the blocking methods: the
requests overlap instead of queueing. Nothing is cancelled if you never collect a
future — the request still reaches the server.

**This is threads, not `await`.** A `PineconeFuture` is not awaitable, and the function
holding it does not need to be `async`. There is no async gRPC client: `AsyncPinecone.index()`
takes no `grpc` argument, and gRPC has no asyncio twin at all. Code already running under
asyncio should use `AsyncIndex` over REST, whose methods are coroutines and yield to the
event loop rather than parking a worker thread — see
[Sync vs Async](sync-vs-async.md).

### PineconeFuture reference

| Method | Description |
|--------|-------------|
| `future.result(timeout=5.0)` | Block until the result is ready; raises `PineconeTimeoutError` if the timeout elapses |
| `future.exception(timeout=5.0)` | The exception the call raised, or `None`; same timeout behaviour |
| `future.done()` | `True` if the operation has completed (or been cancelled) |
| `future.running()` | `True` while the operation is executing |
| `future.cancel()` | Attempt to cancel the operation; `False` once it is running or done |
| `future.cancelled()` | `True` if the cancellation took effect |
| `future.add_done_callback(fn)` | Call `fn(future)` when the operation finishes |

Pass `timeout=None` to `result()` to block indefinitely:

```python
result = future.result(timeout=None)
```

`PineconeFuture` subclasses `concurrent.futures.Future` and works with
`concurrent.futures.as_completed()` and `concurrent.futures.wait()`, so it integrates
naturally with thread-pool patterns. `from pinecone.grpc import PineconeGrpcFuture` also
resolves to it, under the pre-rewrite name.

## Bulk Upsert from a DataFrame

For large-scale ingestion, `upsert_from_dataframe()` splits a pandas `DataFrame` into
batches and submits them concurrently from a thread pool. pandas is not a dependency
of the SDK. Install it yourself (`pip install pandas`) before using this method:

```python
import pandas as pd
from pinecone.grpc import GrpcIndex

df = pd.DataFrame([
    {"id": "product-42", "values": [0.012, -0.087, 0.153]},
    {"id": "product-99", "values": [0.045, 0.021, -0.064]},
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

A batch that fails does not abort the ingest: the returned `UpsertResponse` reports what
landed and what did not. See
[gRPC `upsert_from_dataframe` reports partial failures](../migration/v10-grpc-partial-failures.md)
for the response fields and a bounded retry loop. `total_timeout` bounds the whole
ingest, as opposed to `timeout`, which bounds one attempt of one batch.

## When to Prefer gRPC

| Scenario | Recommendation |
|----------|---------------|
| Bulk upsert (thousands of vectors) | gRPC, lower per-call overhead |
| High-throughput query loops | gRPC with `*_async()` |
| Async Python frameworks (FastAPI, asyncio) | Use `AsyncIndex` instead. `GrpcIndex` does not support `async`/`await` |
| Schema-based indexes, via `index.documents` | HTTP `Index` — `GrpcIndex` has no `documents` namespace |
| Simple scripts and CLI tools | Either works; HTTP `Index` needs no compiled extension |

## Limitations

- **No `documents` namespace.** `index.documents`, the entry point for document
  operations on a schema-based index, exists only on the HTTP `Index`. It is the one
  data-plane method set `GrpcIndex` does not carry.
- **Sync only.** `GrpcIndex` does not support `async`/`await`, and no async gRPC client
  exists. For overlapping requests use the `*_async()` methods and `PineconeFuture`; for
  asyncio, use `AsyncIndex`.
- **No `async_req=True` or `pool_threads`.** Those are legacy shims on the HTTP `Index`.
  `pc.index(..., grpc=True)` ignores `pool_threads`; the `*_async()` methods are the
  gRPC equivalent.
- **`upsert_records`, `search`, `search_records` and the bulk-import methods travel over
  REST**, because the Pinecone gRPC API does not expose those endpoints.
- **Retries are less configurable than on REST.** The retried code set is fixed, and a
  `retry_config` reaches only the gRPC channel — see
  [Retries and Timeouts](#retries-and-timeouts).
- **The native extension is platform-specific.** `pinecone._grpc` is a compiled Rust
  extension bundled in the base package. If installation fails on your platform, check
  the package's supported-platform list for available wheels.
