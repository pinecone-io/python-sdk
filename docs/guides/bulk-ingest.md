# How Bulk Ingest Behaves

This page explains what actually happens when you hand the SDK a large amount
of data — `upsert(vectors=..., batch_size=...)` with a million vectors, or
`upsert_from_dataframe()` with a large DataFrame — and what the knobs really
control. For benchmark numbers and transport comparisons, see
[Performance](performance.md); for per-request retry configuration, see
[Retries](retries.md); for DataFrame-specific recipes, see
[Reliable Large Ingests with upsert_from_dataframe](../how-to/vectors/upsert-from-dataframe.md).

Everything here applies identically to the REST sync client, the asyncio
client, and the gRPC client, and to `documents.upsert()` as well as
`upsert()`. Bulk behavior is one shared engine underneath all of them.

## The four stations

A batched upsert flows through four stages:

```text
your vectors ──► chunker ──► admission gate ──► send batch ──► scorekeeper
                (batch_size)  (concurrency,      (one request   (counts, errors,
                              backpressure)       + retries)     final report)
```

- The **chunker** splits your input into batches of `batch_size` items.
- The **admission gate** decides how many batches may be in flight at once.
  This is where backpressure lives.
- Each admitted batch becomes **one request**, with its own automatic retries.
- The **scorekeeper** aggregates everything into the response you get back —
  including per-batch failures, which are reported, not raised.

## Why your ingest slows down sometimes — and why that's good

The gate maintains a per-host concurrency limit that adapts to what the
backend can absorb *right now*:

- When the server signals overload (HTTP 429, or a gRPC "slow down"), the
  limit **drops sharply**. If the server says *"retry after N seconds"*, the
  gate holds all new batches until that moment passes — no request sneaks
  out early.
- When batches succeed steadily, the limit **recovers gradually**, one step
  at a time.

Drop fast, recover slowly: the same rule TCP uses for network congestion.
A slowdown mid-ingest almost always means the SDK heard the backend asking
for room and gave it. The ingest finishes sooner than it would by hammering
a server that is already shedding load — and other clients of the same
index stay healthy too.

Two consequences worth knowing:

- The limit is **per backend host, shared process-wide**. Two `Index`
  handles in one process pointed at the same index share one gate, so they
  cannot accidentally double-team the backend.
- Retries are also **budgeted per host**, as a token bucket: every
  retryable failure spends from it and every success slowly refills it, and
  retries are suppressed once the bucket falls below half. That caps
  steady-state retry overhead near a tenth of successful traffic, and makes
  retries switch themselves off during a full outage rather than doubling
  load at the worst moment.

## The knobs

### `batch_size`

How many items go in one request. Defaults: 500 rows for
`upsert_from_dataframe`, unbatched for `upsert` unless you pass it.

Bigger batches mean fewer requests but heavier ones: more per-request
payload, longer per-request time, and a larger unit of failure (a batch
fails or succeeds as a whole). The SDK only requires `batch_size >= 1`; the
ceiling is the server's request size limit, which a batch of unusually
large vectors reaches sooner than the item count suggests. Lower it when
your vectors are large, raise it when they are tiny.

### `max_concurrency`

The most batches *this call* will have in flight at once. Default **8**,
range 1–64, identical on every method and transport.

This is your cap, not a target: the adaptive gate can hold effective
concurrency below it whenever the backend is pushing back. Raising it
raises the best case, not the worst case — and it also raises the worst-case
retry burst you commit to during an outage. See
[Tuning max_concurrency](performance.md#tuning-max_concurrency) for a
decision table.

### `total_timeout`

A deadline in seconds for the **whole call**, as opposed to `timeout`,
which bounds a single attempt of a single batch. Default `None` — no
wall-clock bound.

When the deadline expires, the SDK stops *submitting* batches. Batches
already in flight are allowed to finish and are never cancelled — so a
`total_timeout` never tears down work the server may already be applying.
Everything unsent is reported back to you as failed items you can retry.

There is deliberately no default deadline: a slow-but-progressing ingest is
healthy and should be allowed to finish. The failing case is bounded by the
stall detector instead (below), which watches *progress*, not the clock.
Set `total_timeout` when your job has a real external deadline (a batch
window, a cron slot), and size it generously.

### `timeout`

Per-attempt, per-batch request timeout. A batch that times out is retried
like any other transient failure, so with default retry settings one batch
may take up to ~4 × `timeout` plus backoff before it is reported failed.
Raise it for very large batches on slow networks; it is not a lever for
bounding the overall ingest — that's `total_timeout`.

## When the backend is actually down: the stall detector

If the gate has already dropped to its floor and several consecutive
batches then fail completely — every retry exhausted, zero successes — the
SDK concludes the backend is unavailable and **abandons the remainder of
the call loudly** instead of grinding through every batch of a large ingest
against a dead host:

- The returned response carries the unsent work in `failed_items`, and the
  abandonment errors say `backend appears unavailable`.
- A warning is logged with how many batches were abandoned.
- For a short cool-down (~30 seconds), further bulk calls to the same host
  fail fast the same way. After it, the SDK probes again cautiously and
  recovers on the first success.

What your job should do about it: treat it like any partial failure —
inspect the response, wait or alert, and retry the failed items later
(pattern below). One success anywhere resets the detector completely; it
never trips on a backend that is slow but working.

## Partial failures: the reporting contract

Batched calls **do not raise** on per-batch failures. The response tells
you what happened:

- `upserted_count` — items the server accepted.
- `failed_item_count` and `failed_items` — exactly what did not land, ready
  to feed back into a retry call.
- `errors` — one entry per failed batch. Each carries the items, the error,
  a `retryable` hint, and a `disposition` telling you *how* it failed:
  - `rejected` — the attempt completed with an error, after retries. The
    write may still have landed: a response can be lost after the server
    applied it.
  - `unsent` — never attempted because `total_timeout` expired first.
  - `abandoned` — never attempted because the stall detector fired.

  Dispositions are an open set. Branch on the values you handle and let an
  unfamiliar one fall through to your default; do not match exhaustively.

(`upsert_from_dataframe` accepts `on_error="raise"` if you'd rather the
lowest-indexed batch failure be re-raised after all batches settle; the
partial result rides on the exception's `response` attribute.)

## Retrying safely

Upserts are idempotent by vector id — re-sending an item that already
landed simply overwrites it with identical data. So retrying `failed_items`
is always safe. What is *not* safe is retrying in an unbounded loop:

```python
# DON'T: an unhealthy backend keeps this spinning forever
while response.failed_items:
    response = index.upsert(vectors=response.failed_items, batch_size=200)
```

Bound the attempts, filter on the `retryable` hint, and back off between
rounds:

```python
import time

from pinecone import Pinecone

pc = Pinecone()
index = pc.index(name="product-search")

response = index.upsert(vectors=vectors, batch_size=200)

for attempt in range(3):
    if not response.has_errors:
        break
    retryable = [
        item
        for err in response.errors
        if err.retryable
        for item in err.items
    ]
    if not retryable:
        break
    time.sleep(2**attempt)
    response = index.upsert(vectors=retryable, batch_size=200)
```

A batch marked `retryable=False` was rejected for a reason a retry cannot
fix (malformed data, dimension mismatch, auth) — re-sending it just burns
your retry budget. Log it and move on.

## What you don't have to manage

- **Thundering herds.** Retry delays are jittered and server `Retry-After`
  hints are smeared, so a fleet of clients told to come back at the same
  moment disperses instead of re-colliding.
- **Cross-client coordination in one process.** The gate and the retry
  budget are shared per host process-wide; concurrent bulk calls
  self-organize instead of competing.
- **Transport differences.** REST sync, asyncio, and gRPC run the same
  engine, the same knobs, the same reporting contract. Choose a transport
  for its own merits ([Sync vs Async](sync-vs-async.md),
  [gRPC](grpc.md)), not for bulk semantics.
