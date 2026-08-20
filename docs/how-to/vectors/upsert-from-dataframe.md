# Reliable Large Ingests with `upsert_from_dataframe`

`upsert_from_dataframe()` takes a pandas DataFrame of any size, batches it,
sends the batches in parallel with adaptive backpressure, and reports exactly
what landed and what didn't. This page is the operational recipe book: how to
size the knobs, bound the job in time, and handle partial failure — so a
million-row ingest is something you schedule, not something you babysit.

The signature and behavior are identical on all three clients — REST sync,
asyncio, and gRPC. Examples below use gRPC; swap `grpc=True` out (or `await`
in) freely. For the underlying mechanics — the admission gate, retry budget,
and stall detector — see [How Bulk Ingest Behaves](../../guides/bulk-ingest.md).

```{contents}
:local:
:depth: 2
```

## When to use it

| Your situation | Reach for |
|---|---|
| Vectors in a DataFrame, up to a few million rows | `upsert_from_dataframe()` — this page |
| Tens of millions of rows already in cloud storage | [`start_import()`](bulk-import.md) — server-side bulk load |
| Raw text, server-side embedding | `upsert_records()` |

## The DataFrame contract

Required columns: `id` (string) and `values` (list of floats, matching the
index dimension). Optional columns: `sparse_values` and `metadata`. A row may
omit an optional cell entirely — `NaN` cells are treated as absent, not sent
as nulls:

```python
import pandas as pd

df = pd.DataFrame(
    [
        {"id": "a-1", "values": [0.1, 0.2], "metadata": {"lang": "en"}},
        {"id": "a-2", "values": [0.3, 0.4]},   # no metadata: fine
    ]
)
```

## Minimal usage

```python
from pinecone import Pinecone

pc = Pinecone(api_key="...")
index = pc.index("articles", grpc=True)

response = index.upsert_from_dataframe(df)
print(response.upserted_count)
```

Defaults: 500 rows per batch, 8 batches in flight, per-request timeout from
the client, no overall deadline, partial failures collected on the response
rather than raised.

## The knobs, and how to size them

```python
response = index.upsert_from_dataframe(
    df,
    batch_size=500,        # rows per request
    max_concurrency=8,     # batches in flight at once (1-64)
    timeout=30,            # seconds per ATTEMPT of one batch
    total_timeout=1800,    # seconds for the WHOLE job
    on_error="collect",    # or "raise"
)
```

### `batch_size` — rows per request

Start at the default 500. Lower it (100–200) when rows are heavy —
high-dimension vectors or large metadata — so a single request stays
comfortably inside per-request limits and each retry re-sends less. Raise it
(up to ~1000) for small vectors when you want fewer round trips. A batch
fails or succeeds as a unit, so `batch_size` is also your unit of retry.

### `max_concurrency` — parallelism ceiling

The default 8 is flat across every machine and transport, and it is a
*ceiling*, not a target: the SDK's per-host adaptive limit backs off
automatically when the server signals pressure, then recovers. Raise it only
with measurement (see [Tuning max_concurrency](../../guides/performance.md#tuning-max_concurrency));
lower it to 1–2 when the ingest shares an index with latency-sensitive
query traffic.

### `timeout` — one attempt of one batch

This bounds a single request attempt. A batch that times out is retried like
any transient failure, so the worst case for one batch is roughly
`(max_retries + 1) × timeout` plus backoff. Raise `timeout` when batches are
large or the network is slow; configure attempts on the client:

```python
from pinecone import Pinecone, RetryConfig

pc = Pinecone(api_key="...", retry_config=RetryConfig(max_retries=2))
```

Don't use `timeout` to bound the overall job — that's `total_timeout`'s job,
and the error message you'd eventually see says exactly that.

### `total_timeout` — the whole job's deadline

The knob to set whenever the ingest runs inside a real window (a cron slot, a
deploy step, a batch pipeline stage). Semantics designed for safe re-runs:

- On expiry the SDK **stops submitting** new batches.
- Batches already in flight **finish and are never cancelled** — no torn
  writes.
- Everything unsent comes back in `response.failed_items` with
  `disposition="unsent"`, ready to retry.

There is deliberately no default: a slow-but-progressing ingest is healthy.
But note that `total_timeout` is the only *guaranteed* wall-clock bound — the
automatic stall detector (below) covers the common backend-overload case, not
every conceivable outage — so set it whenever "must finish by" is a real
requirement.

## Recipe: the nightly job

Bounded in time, partial results persisted, retry deferred to the next run:

```python
response = index.upsert_from_dataframe(
    df,
    batch_size=500,
    total_timeout=25 * 60,          # leave headroom inside a 30-min slot
)

if response.has_errors:
    leftovers = pd.DataFrame(response.failed_items)
    leftovers.to_parquet("retry-tomorrow.parquet")
    log.warning(
        "ingest finished %d/%d; %d rows deferred",
        response.upserted_count,
        response.total_item_count,
        response.failed_item_count,
    )
```

Re-running with the leftover rows is always safe: upserts are idempotent by
vector id, so a row that actually landed just before a timeout is simply
overwritten with identical data on the retry.

## Recipe: retry within the same run

Bound the attempts, filter on the `retryable` hint, back off between rounds —
never `while response.failed_items:`, which spins forever against an
unhealthy backend:

```python
import time

response = index.upsert_from_dataframe(df, total_timeout=1800)

for attempt in range(3):
    if not response.has_errors:
        break
    retryable = [
        item
        for err in response.errors
        if err.retryable          # skip rejections a retry cannot fix
        for item in err.items
    ]
    if not retryable:
        break
    time.sleep(2**attempt)
    response = index.upsert_from_dataframe(
        pd.DataFrame(retryable), total_timeout=600
    )
```

A batch with `retryable=False` was rejected for a structural reason —
malformed values, dimension mismatch, auth — and re-sending it only burns
time. Log it and move on.

## Recipe: fail loudly instead

Pipelines that prefer an exception over inspection pass `on_error="raise"`:
after every batch settles, the lowest-indexed failure is re-raised with the
partial result attached.

```python
from pinecone import PineconeTimeoutError

try:
    index.upsert_from_dataframe(df, total_timeout=1800, on_error="raise")
except PineconeTimeoutError as exc:
    partial = exc.response
    save_for_retry(partial.failed_items)
    raise
```

## Recipe: asyncio

Identical signature on the async client:

```python
from pinecone import PineconeAsyncio

async with PineconeAsyncio(api_key="...") as pc:
    index = pc.index("articles")
    response = await index.upsert_from_dataframe(
        df, batch_size=500, total_timeout=1800
    )
```

## Reading the response

| Field | Meaning |
|---|---|
| `upserted_count` | Rows the server accepted |
| `total_item_count` / `failed_item_count` | Submitted vs. not landed |
| `has_errors` | `True` when any batch failed |
| `failed_items` | The exact rows that did not land — feed them back in |
| `errors` | One `BatchError` per failed batch: `items`, `error`, `retryable`, `disposition` |

`disposition` tells you *how* a batch failed: `rejected` (the server or
transport refused it after retries), `unsent` (`total_timeout` expired
first), or `abandoned` (the stall detector gave up on an unresponsive
backend).

## What you get without configuring anything

Three protections are always on; you'll see their fingerprints in logs and
responses rather than in your code:

- **Adaptive concurrency.** A mid-ingest slowdown usually means the server
  asked for room (429 / Retry-After) and the SDK complied. The job finishes
  sooner than it would by pushing through.
- **A per-host retry budget.** During a partial outage, retries are
  rationed so the SDK never doubles the load on a backend that is already
  struggling.
- **The stall detector.** If the backend stops making progress entirely, the
  ingest aborts loudly in minutes — `disposition="abandoned"`, errors reading
  `backend appears unavailable` — instead of consuming your whole
  `total_timeout` achieving nothing. It re-probes cautiously after a short
  cool-down; treat the abandonment like any partial failure and retry later.

## Checklist for a large production ingest

- [ ] `id` and `values` columns present; optional cells absent, not null-ish
      placeholders
- [ ] `batch_size` sized to your row weight (default 500; lower for heavy rows)
- [ ] `total_timeout` set if the job has a real window
- [ ] `on_error` chosen: `"collect"` + persist `failed_items`, or `"raise"`
- [ ] Retry path bounded and filtered on `err.retryable`
- [ ] `max_concurrency` left at 8 unless you've measured a reason not to
