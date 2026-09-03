# 10.0: gRPC `upsert_from_dataframe` reports partial failures instead of raising

In 9.1 and earlier, `GrpcIndex.upsert_from_dataframe` raised as soon as any batch
failed. From 10.0 it aggregates, matching `upsert(batch_size=...)` and the REST
transport, which has behaved this way since v9.0.0.

## What changed

```python
import pandas as pd

# 9.1 and earlier — gRPC
try:
    index.upsert_from_dataframe(df)
except Exception:
    # No way to tell how much landed. Re-run the whole frame.
    ...

# 10.0 — gRPC and REST
response = index.upsert_from_dataframe(df)
if response.failed_item_count:
    for error in response.errors:
        print(error.error_message)
    index.upsert_from_dataframe(pd.DataFrame(response.failed_items))
```

### Retrying failures safely

Do **not** wrap that retry in an unbounded loop:

```python
# Don't do this. During an outage it re-sends the whole frame back to back,
# forever, with no pause between rounds.
while response.failed_items:
    response = index.upsert_from_dataframe(pd.DataFrame(response.failed_items))
```

Failures that come back are usually failures because the backend is unwell, and
retrying them immediately is the thing most likely to keep it unwell. Bound the
attempts and back off between them:

```python
import time

import pandas as pd

remaining = df
for attempt in range(5):
    response = index.upsert_from_dataframe(remaining)
    retryable = [
        item
        for err in response.errors
        if err.retryable
        for item in err.items
    ]
    if not retryable:
        break
    remaining = pd.DataFrame(retryable)
    time.sleep(2**attempt)
else:
    raise RuntimeError(f"{len(remaining)} rows still failing after 5 attempts")
```

Filter on `err.retryable` before re-sending: a batch rejected for a
deterministic reason — a validation error, a 400 — fails identically on every
attempt, and re-sending it just burns your attempt budget on rows that can
never land. `err.disposition` says *how* each batch failed (`"rejected"`,
`"unsent"`, `"abandoned"` — an open set; don't match exhaustively), and
`"abandoned"` means the SDK concluded the backend was down and stopped
sending: wait before retrying rather than looping immediately.

The SDK already backs off *within* a call — per-attempt retries with jitter, a
retry budget, and an adaptive concurrency limiter that halves in-flight requests
when the backend pushes back. What it cannot do is stop a caller re-entering it
in a tight loop, which is why the attempt cap above is yours to set.

Note that when `total_timeout` expires, batches that were never sent also land in
`failed_items`. A `total_timeout` plus an unbounded retry loop is therefore a
fixed-period retry storm.

`UpsertResponse` already carried `upserted_count`, `failed_item_count`, `errors`
and `failed_items`, so there is no new model to learn — the gRPC transport simply
starts populating them.

## If you depended on the raise

Pass `on_error="raise"`:

```python
index.upsert_from_dataframe(df, on_error="raise")
```

That re-raises the lowest-indexed batch failure, which is what 9.1 did, with two
improvements:

- every batch settles before the exception propagates, so nothing is left running
  server-side;
- the partial result is attached to the exception, so the count is no longer lost:

```python
try:
    index.upsert_from_dataframe(df, on_error="raise")
except Exception as exc:
    print(exc.response.upserted_count)
    retry_these = exc.response.failed_items
```

`on_error` is a posture choice, not a migration shim — fail-fast on ingest is
legitimate, and it is not going away. It is available on both transports.

## Why this is a break we chose to take

Both behaviors were already public and depended on: REST has aggregated since
v9.0.0, and gRPC raised in every release before this one. Partial failure either
raises or it does not, so consistency and no-breaking-change were in genuine
conflict.

What a caller loses is the exception — code that never inspects the response no
longer hears about a partial failure, which is what the one-time warning below
covers. The blast radius is smaller than it looks: the old raise discarded the
partial count, so no caller relying on it could tell what had landed. A careful
gRPC caller was already re-running the whole frame, which is safe either way
because upserts are idempotent by vector ID.

## Finding out at runtime

The first time a gRPC ingest hits a partial failure without an explicit
`on_error`, the SDK warns once per process, naming `response.errors` and
`on_error="raise"`. Passing `on_error` explicitly — either value — silences it.
REST does not warn: its behavior did not change.

## Related

- `total_timeout` follows `on_error` the same way: under `"collect"` an expired
  deadline logs a warning and returns the partial response, under `"raise"` it
  raises `PineconeTimeoutError` carrying the same partial result on its
  `response` attribute. [Bulk ingest](../guides/bulk-ingest.md) covers the
  deadline itself.
- [Using the gRPC Client](../guides/grpc.md) covers the transport this change
  applies to, including what `GrpcIndex` does and does not carry.
