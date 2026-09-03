# Reliable Large Ingests with `upsert_from_dataframe`

{meth}`~pinecone.index.Index.upsert_from_dataframe` takes a pandas DataFrame, batches it, sends
the batches in parallel with adaptive backpressure, and reports exactly what landed and
what didn't. This page is the operational recipe book: how to size the knobs, bound the
job in time, and handle partial failure — so a large ingest is something you schedule,
not something you babysit.

The signature is the same on all three clients — REST sync, asyncio, and gRPC — so the
examples below carry over unchanged apart from the `await`. For the underlying mechanics
— the admission gate, retry budget, and stall detector — see [How Bulk Ingest
Behaves](../../guides/bulk-ingest.md).

:::{important}
`pandas` is not a dependency of this SDK, and is never installed as an extra. This is
the only method that needs it, so it is imported when you call the method: if the import
fails you get a `RuntimeError` naming `pip install pandas` rather than an import error at
`from pinecone import Pinecone`. Install pandas in your own environment.

If you would rather not take the dependency, {meth}`~pinecone.index.Index.upsert` does the same
parallel batching from a list of vectors — see
[Large datasets](upsert-and-query.md#large-datasets).
:::

```{contents}
:local:
:depth: 2
```

## When to use it

| Your situation | Reach for |
|---|---|
| Vectors in a pandas DataFrame | `upsert_from_dataframe()` — this page |
| Vectors in a list, and no pandas | [`upsert(batch_size=...)`](upsert-and-query.md#large-datasets) |
| Far more rows than you want to stream, already in cloud storage | [`start_import()`](bulk-import.md) — server-side bulk load |
| Raw text, embedded server-side | `upsert_records()` — needs an index with integrated inference |

## The DataFrame contract

Required columns: `id` (string) and `values` (list of floats, matching the index
dimension). Optional columns: `sparse_values` and `metadata`; a column outside those four
is rejected rather than ignored, so move your own fields into `metadata`.

A row may omit an optional cell entirely. Building a frame from row dicts where only some
rows carry `metadata` leaves `NaN` in the rest, pandas having no other way to fill the
gap, and those cells count as absent rather than being sent as nulls — without that, a
`NaN` would reach validation and surface as `metadata must be a dict, got float`:

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

pc = Pinecone(api_key="your-api-key")
index = pc.index("articles")

response = index.upsert_from_dataframe(df, namespace="articles-en")
print(response.upserted_count)
```

Defaults: 500 rows per batch, 8 batches in flight, per-request timeout from the client,
no overall deadline, partial failures collected on the response rather than raised. Only
`df` is positional; `namespace`, `batch_size`, `show_progress` and `timeout` can be
passed either way, and `max_concurrency`, `total_timeout` and `on_error` are keyword-only.

Omitting `namespace` writes to the default namespace `""`; see
{doc}`/how-to/vectors/namespaces`.

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

Start at the default 500. Lower it (100–200) when rows are heavy — high-dimension
vectors or large metadata — so a single request stays comfortably inside the server's
caps on vector count and encoded size, and each retry re-sends less. Raise it for small
vectors when you want fewer round trips; the SDK only requires a positive integer, and it
is the server that decides when a request is too big, so raise it by measurement rather
than by guess. A batch fails or succeeds as a unit, so `batch_size` is also your unit of
retry.

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
`(max_retries + 1) × timeout` plus backoff. `max_retries` defaults to 3 — four attempts
in all. Raise `timeout` when batches are large or the network is slow; change the number
of attempts on the client:

```python
from pinecone import Pinecone, RetryConfig

pc = Pinecone(api_key="your-api-key", retry_config=RetryConfig(max_retries=2))
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

`failed_items` comes back as a flat list of dicts keyed `id` / `values` /
`sparse_values` / `metadata`, which is exactly what `pd.DataFrame(...)` needs to rebuild
a frame you can feed straight back in.

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
    print(
        f"ingest finished {response.upserted_count}/{response.total_item_count}; "
        f"{response.failed_item_count} rows deferred"
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

The exception you catch is whichever one the batch actually raised —
{exc}`~pinecone.errors.exceptions.PineconeTimeoutError` for a batch that exhausted its
retries on timeout, an {exc}`~pinecone.errors.exceptions.ApiError` subclass for a
rejection — and the partial `UpsertResponse` is attached to it as `.response`:

```python
import pandas as pd
from pinecone import PineconeTimeoutError

try:
    index.upsert_from_dataframe(df, total_timeout=1800, on_error="raise")
except PineconeTimeoutError as exc:
    pd.DataFrame(exc.response.failed_items).to_parquet("retry.parquet")
    raise
```

## Recipe: asyncio

Identical signature on {class}`~pinecone.async_client.pinecone.AsyncPinecone`. Note that its `index()` is a
coroutine, unlike the sync client's — awaiting it is what keeps the host lookup off the
event loop:

```python
from pinecone import AsyncPinecone

async with AsyncPinecone(api_key="your-api-key") as pc:
    index = await pc.index("articles")
    response = await index.upsert_from_dataframe(
        df, namespace="articles-en", batch_size=500, total_timeout=1800
    )
```

## Reading the response

| Field | Meaning |
|---|---|
| `upserted_count` | Rows the server accepted |
| `total_item_count` / `failed_item_count` | Submitted vs. not landed |
| `has_errors` | `True` when any batch failed |
| `failed_items` | The exact rows that did not land — feed them back in |
| `errors` | One `BatchError` per failed batch: `batch_index`, `items`, `error`, `error_message`, `retryable`, `disposition` |

`disposition` tells you *how* a batch failed: `rejected` (the server or
transport refused it after retries), `unsent` (`total_timeout` expired
first), or `abandoned` (the stall detector gave up on an unresponsive
backend).

## What you get without configuring anything

Three protections are always on — adaptive concurrency, a per-host retry budget, and a
stall detector — and you see their fingerprints in the response rather than in your code.
The one worth recognising here is the stall detector: when the backend stops making
progress entirely, the ingest gives up in minutes rather than burning your whole
`total_timeout`, and the batches it dropped arrive with `disposition="abandoned"` and an
error reading `backend appears unavailable`. Treat that like any other partial failure
and retry later.

[How Bulk Ingest Behaves](../../guides/bulk-ingest.md#what-you-dont-have-to-manage)
explains all three.

## Checklist for a large production ingest

- [ ] `pandas` installed in the environment that runs the job
- [ ] `id` and `values` columns present, and no columns beyond the four accepted ones
- [ ] optional cells absent, not null-ish placeholders
- [ ] `batch_size` sized to your row weight (default 500; lower for heavy rows)
- [ ] `total_timeout` set if the job has a real window
- [ ] `on_error` chosen: `"collect"` + persist `failed_items`, or `"raise"`
- [ ] Retry path bounded and filtered on `err.retryable`
- [ ] `max_concurrency` left at 8 unless you've measured a reason not to
