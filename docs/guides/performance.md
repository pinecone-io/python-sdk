# Performance

The SDK is designed for low overhead. This page describes the key design choices and
the patterns that keep your application fast.

## Transport

The SDK ships three client variants, with two different application-protocol
stacks underneath:

- `Index` and `AsyncIndex` (REST): [httpx](https://www.python-httpx.org/) over
  HTTP/1.1 with connection keepalive.
- `GrpcIndex`: a native (Rust-backed) gRPC channel over HTTP/2 with binary
  protobuf framing.

This protocol gap is part of why gRPC has a measurable throughput edge on
bulk upsert workloads; see [When to Use gRPC](#when-to-use-grpc). For the
REST clients, parallel batched upsert (below) is what you reach for to drive
concurrency.

## Connection Pooling

Both `Pinecone` and `Index` maintain a persistent `httpx.Client` (or `httpx.AsyncClient`
for the async variants). Creating a new client for every request wastes time on TLS
handshakes and connection setup.

**Reuse the same `Index` instance** across calls rather than constructing a new one each
time:

```python
# Good: one client, many calls
from pinecone import Pinecone

pc = Pinecone()
desc = pc.indexes.describe("product-search")
index = pc.index(host=desc.host)

for batch in batches:
    index.upsert(vectors=batch)

# Bad: a new HTTP client for every upsert
for batch in batches:
    index = pc.index(host=desc.host)  # new client every time
    index.upsert(vectors=batch)
```

Use the context manager protocol to ensure connections are released when you are done:

```python
with pc.index(host=desc.host) as index:
    index.upsert(vectors=large_batch)
```

## Fast Serialization with msgspec and orjson

Response models are `msgspec.Struct` instances. `msgspec` uses zero-copy deserialization
and avoids Python object allocation overhead that Pydantic-based models incur. Request
bodies are serialized with `orjson`, which is typically 5–10× faster than the standard
library `json` module.

These libraries are always active; no configuration is needed.

## Cold Import Cost

The SDK uses lazy imports to keep `import pinecone` cheap. Top-level SDK
symbols (`Pinecone`, `AsyncPinecone`, etc.) are available as soon as you import
`pinecone`, but heavy modules, such as the gRPC channel, pandas (for
`upsert_from_dataframe`), and tqdm (for progress bars), are only loaded when you
actually use them.

If your application is latency-sensitive at startup, avoid importing `pinecone` in
module-level code that runs before it is needed:

```python
# Fine: deferred to first use
def get_index() -> Index:
    from pinecone import Pinecone
    pc = Pinecone()
    return pc.index(host="...")
```

## Batching Large Upserts

For datasets larger than a single request payload, pass `batch_size` to
`Index.upsert()`. The SDK splits the input into batches and sends them in parallel:
sync via a cached `ThreadPoolExecutor`, async via an `asyncio.Semaphore`. HTTP-level
retries happen automatically per batch.

```python
response = index.upsert(
    vectors=large_list,    # any length
    batch_size=100,        # vectors per request
    max_concurrency=8,     # parallel in-flight requests (default 8, range 1–64)
)
print(response.upserted_count)         # successful items
print(response.failed_item_count)      # 0 if everything succeeded
```

`AsyncIndex.upsert()` accepts the same `batch_size` and `max_concurrency` kwargs.
`Index.upsert_from_dataframe()` accepts `batch_size`, `max_concurrency`, and `total_timeout`,
with the same meanings on all three clients (REST sync, asyncio, and gRPC).
`Index.upsert_records()` does **not** accept `batch_size` or `max_concurrency`; it
sends a single NDJSON request per call, so chunk the record list yourself and call
`upsert_records()` once per chunk.

When `batch_size` is set, `upsert()` returns an `UpsertResponse` with partial-failure
information instead of raising on the first failed batch; see
[Handling partial failures](../how-to/vectors/upsert-and-query.md#handling-partial-failures).

For a walkthrough of what each of these knobs controls, beyond its effect on
throughput, see [How Bulk Ingest Behaves](bulk-ingest.md).

### How much faster is parallel batching?

Measured on 10k vectors / 1536-d / batch=100 against an aws-us-east-1 serverless
index ([Methodology](#methodology)); wall time, p50:

| Client | `max_concurrency` | REST sync | REST async | gRPC |
|---|---:|---:|---:|---:|
| v8 | sequential (baseline) | 112 s | 67 s | 34 s |
| v9 | 1 | 31.5 s | 32.7 s | 35.0 s |
| v9 | 4 | 9.6 s | 10.2 s | 10.0 s |
| v9 | 8 (default) | **5.7 s** | **5.9 s** | **5.7 s** |
| v9 | 16 | 5.0 s | 6.6 s | 4.0 s |
| v9 | 32 | 4.4 s | 5.0 s | 2.7 s |

The v8 row is the published `pinecone==8.x` client running its sequential
`batch_size=` loop; the v9 rows are this client using native parallel batched
upsert. The headline win for the typical caller, v8 REST sync sequential vs v9
REST sync, is ~12× at `max_concurrency=4` and ~20× at the default `8`.
Async REST shows a
similar shape with a smaller multiplier because v8 async sequential was
already faster than v8 sync sequential. gRPC is faster than REST at high
concurrency; see [When to Use gRPC](#when-to-use-grpc).

The `max_concurrency=1` row isn't a setting you'd reach for in practice: at
`c=1` you've opted out of the main reason to pass `batch_size=` in the first
place. But it's a useful diagnostic. It isolates how much of the v8 → v9
speedup comes from non-parallelism improvements in the client (request
building, serialization, response decoding, retry layer) versus the explicit
fan-out parallel batching adds on top. For REST sync, ~3.6× of the 11.7×
win at `max_concurrency=4` comes from those raw client improvements alone; the
remaining ~3.3× is parallelism. For gRPC, almost the entire win comes from
parallelism: v8 gRPC was already efficient at the request level.

### Tuning `max_concurrency`

The default is a flat `8` on every method and transport, identical on your laptop and
on your production hosts, so throughput is reproducible across machines. Push higher
only when you have a reason and can measure the result on your workload:

| `max_concurrency` | When to use it |
|---:|---|
| `1` | Strict per-second quota, or you want sequential semantics for ordering |
| `8` *(default)* | General use, including large bulk loads; no tuning required |
| `16–32` | Diminishing returns; the cluster (not the SDK) is usually the bottleneck above ~16 |
| `>32` | Rarely worth it for a single client; consider sharding the work across multiple clients instead |

Throughput saturates around `c≈16` for most workloads because cluster-side
ingress capacity becomes the bottleneck, not the SDK. If you do need to push past
that ceiling, run multiple `Index` instances from separate processes rather than
raising `max_concurrency` further on one client.

For multi-million-vector loads from cloud storage, prefer `index.start_import()`
over batched upsert; it avoids per-batch HTTP overhead entirely.

## Query Latency

Queries don't benefit from parallel batching the way bulk upserts do: each
query is a single round trip. But the v9 client decodes responses
substantially faster than v8 on REST for any query that returns more than a
trivial payload. The wins come from `msgspec.Struct` response models and
`orjson` for JSON decoding (see [Fast Serialization](#fast-serialization-with-msgspec-and-orjson)),
neither of which the v8 client uses.

Measured on the same 1536-d serverless index ([Methodology](#methodology)),
median latency:

| Client | Scenario | REST sync | REST async | gRPC |
|---|---|---:|---:|---:|
| v8 | `query_k10` | 35.8 ms | 34.3 ms | 32.5 ms |
| v9 | `query_k10` | 33.4 ms | 35.2 ms | 31.1 ms |
| v8 | `query_k100` + values + metadata | 800 ms | 708 ms | 133 ms |
| v9 | `query_k100` + values + metadata | **279 ms** | **260 ms** | 120 ms |
| v8 | `query_k1000` + values | 7.01 s | 7.12 s | 534 ms |
| v9 | `query_k1000` + values | **2.18 s** | **2.16 s** | 493 ms |

Two patterns stand out:

- **The REST win scales with response size.** Small `top_k=10` queries are
  near-parity (~1.05×); `top_k=100` with values + metadata is ~2.8× sync /
  ~2.7× async; `top_k=1000` with values is ~3.2× sync / ~3.3× async. The
  bottleneck on v8 REST queries was decoding large JSON payloads: exactly
  the failure mode `msgspec` + `orjson` were chosen to fix.
- **gRPC is at parity throughout** (~1.05–1.13× across scenarios). gRPC
  responses are protobuf, so they bypass the JSON decoding path entirely;
  there's no msgspec/orjson dividend to collect. If you're already on gRPC
  for queries, upgrading doesn't change much on the query side. If you're on
  REST and run heavy queries, the upgrade is a substantial win.

Filter complexity (`eq`, `in_50`, `nested`) on `top_k=100` adds modest extra
wins on REST (~1.15–1.22×) and stays at parity on gRPC. Filter overhead is
small relative to network and decoding.

## Async Concurrency

Pick the async client (`AsyncPinecone` / `AsyncIndex`) when your code is already
inside an `async def`, most often because you're conforming to the interface of
an async web framework like **FastAPI**, **Starlette**, or **Litestar**, where
request handlers are coroutines. In that setting, calling a blocking sync method
either stalls the event loop (degrading throughput for every concurrent request)
or forces you to offload to a thread; the async client lets you `await` Pinecone
calls inline without either workaround.

The async client is also natural when you want concurrent reads and writes that
should overlap (multiple queries in flight, or a query running while an upsert
finishes), though sync code can achieve the same with threads.

For **pure bulk upsert**, prefer native batched upsert over a hand-rolled
`asyncio.gather`: same parallelism, less code, automatic retries, and partial-failure
reporting:

```python
# Preferred
from pinecone import AsyncPinecone

async with AsyncPinecone() as pc:
    index = await pc.index(host="product-search-abc123.svc.pinecone.io")
    async with index:
        response = await index.upsert(
            vectors=large_list,
            batch_size=100,
            max_concurrency=8,
        )
```

For **mixed workloads** (concurrent upserts and queries, or query fan-out across
many namespaces), `asyncio.gather` over `AsyncIndex` calls is still the natural
pattern:

```python
async with AsyncPinecone() as pc:
    index = await pc.index(host="product-search-abc123.svc.pinecone.io")
    async with index:
        results = await asyncio.gather(
            index.upsert(vectors=writes_batch, batch_size=100),
            index.query(vector=q1, top_k=10),
            index.query(vector=q2, top_k=10),
        )
```

Sync vs async at high concurrency: with native batched upsert at `max_concurrency=32`,
sync (~4.4 s) edges out async (~5.0 s) on the 10k-vector benchmark: the cached
`ThreadPoolExecutor` is competitive with `asyncio.Semaphore` once cluster-side
ingress dominates. Pick the client that matches your application style; throughput
is similar at the saturation point.

## When to Use gRPC

`pinecone.grpc.GrpcIndex` accepts the same `batch_size=` and `max_concurrency=`
kwargs as the REST `Index`, so the call site looks identical. The wire-level
differences are HTTP/2 framing (vs HTTP/1.1 + keepalive on REST) and binary
protobuf encoding (vs JSON). The gRPC channel ships with the package; no
separate install step.

Reading off the [throughput table above](#how-much-faster-is-parallel-batching),
a few things about gRPC stand out:

- **Even sequential, v8 gRPC was ~3× faster than v8 REST sync** (34 s vs 112 s).
  HTTP/2 multiplexing and protobuf encoding buy a lot before any parallelism
  enters the picture, and that gap is structural to the protocols, not
  something parallel batching alone closes.
- **At the default `max_concurrency=8`, the three transports are essentially
  tied** (~5.7–5.9 s). For typical workloads, the choice is about API style,
  not throughput.
- **gRPC pulls ahead as concurrency rises**: at `max_concurrency=32`, gRPC
  finishes the same work 1.5–1.9× faster than REST.
- **`max_concurrency=1` doesn't help gRPC**: v8 gRPC was already pipelining
  requests over its HTTP/2 channel, so the new code path's win at gRPC comes
  from explicit fan-out at higher concurrency, not from the new partial-success
  machinery.

Pick gRPC when:

- You're doing **sustained bulk upserts at `max_concurrency` ≥ 16**: gRPC
  finishes the same work 1.5–1.9× faster than REST at that concurrency.
- You want the **lowest absolute write latency floor** on a single client
  (~2.7 s for 10k vectors at `c=32` on the reference workload).

Stay on REST when:

- You're at default settings or low concurrency: there is no measurable
  throughput benefit at `max_concurrency` ≤ 8.
- You need async: `GrpcIndex` is sync-only; for async workloads use
  `AsyncIndex` over REST.

```python
from pinecone import Pinecone

pc = Pinecone()
with pc.index(name="product-search", grpc=True) as index:
    response = index.upsert(
        vectors=large_list,
        batch_size=100,
        max_concurrency=16,
    )
```

## Summary

| Technique | Where it helps |
|-----------|---------------|
| HTTP keepalive (REST) / HTTP/2 (gRPC) | Reused TCP connections, lower per-call setup cost |
| Reuse `Index` instance | Eliminate per-call TLS/connection overhead |
| msgspec structs | Response deserialization, faster than Pydantic |
| orjson | Request serialization, faster than stdlib `json` |
| Lazy imports | Reduce cold-start time |
| `Index.upsert(batch_size=…, max_concurrency=…)` | Bulk upsert, typical 10–25× over a sequential loop |
| `AsyncIndex` + `asyncio.gather()` | Mixed concurrent read/write workloads |
| `GrpcIndex` (sync only) | Sustained bulk upserts at `max_concurrency` ≥ 16, ~1.5–1.9× over REST |
| `index.start_import()` | Multi-million-vector loads from cloud storage |

## Methodology

The numbers in this guide come from a controlled benchmark: 10,000 random
1536-dimensional vectors, `batch_size=100`, single client, fresh namespace per
run, against an aws-us-east-1 serverless index. The client ran on a GCP
`n2-standard-2` VM (2 vCPU, 8 GB) in `us-central1-a` running Ubuntu 24.04, so
every request crosses GCP → AWS; RTT and inter-cloud bandwidth are real
factors in the absolute numbers. The "v8 sequential" rows use
`pinecone==8.1.2` from PyPI (sequential `batch_size=` loop, fail-fast on
first batch error). The `max_concurrency=N` rows use this version of the SDK
with native parallel batched upsert.

Iteration counts vary by scenario. Batched-upsert cells use n=3 measured
iterations after 1 warmup: each iteration writes 10k vectors, so increasing
n trades wall time for precision. That table is best read as a directional
guide: the large speedup factors (≥3×) are well above run-to-run noise, but
small differences between adjacent rows in the same column should not be
over-interpreted. Query cells use n=25 (n=10 for `query_k1000_values`); the
query numbers are statistically firm. We plan to re-run the batched-upsert
sweep at higher iteration counts too; this page will be refreshed at that
time.

Your numbers will vary with client region, RTT, vector dimension, batch size,
payload metadata, and concurrent traffic from other clients. When in doubt,
measure on your own workload.
