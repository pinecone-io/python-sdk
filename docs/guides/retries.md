# Retries and Resilience

The SDK retries failed requests automatically using multiple layers of resilience: per-request
backoff with decorrelated jitter and adaptive concurrency for bulk operations. This page covers
everything you need to know to tune that behavior or to decide when to hand retry control back to
your orchestrator.

For the exceptions the SDK raises when retries are exhausted, see {doc}`/guides/error-handling`.

---

## Defaults at a Glance

Out of the box — no configuration needed:

| What | Default behavior |
|------|------------------|
| Max retries (after initial attempt) | 3 for REST (4 total), 5 for gRPC (6 total) |
| Retryable HTTP status codes | 408, 429, 500, 502, 503, 504 |
| Retryable gRPC status codes | UNAVAILABLE, RESOURCE\_EXHAUSTED, ABORTED |
| Backoff algorithm | Decorrelated jitter — random walk bounded by `backoff_factor` floor and `max_wait` cap |
| Adaptive concurrency (bulk paths) | Self-tunes downward on throttling; `max_concurrency` is a ceiling, not a constant |

---

## Configuring Retries

Pass a `RetryConfig` to the `Pinecone` constructor to customize retry behavior for all
REST requests made by that client:

```python
from pinecone import Pinecone, RetryConfig

pc = Pinecone(
    retry_config=RetryConfig(
        max_retries=5,
        backoff_factor=0.5,
        max_wait=60.0,
        retryable_status_codes=frozenset({429, 500, 503}),
    )
)
```

### `RetryConfig` fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_retries` | `int` | `3` | Number of retry attempts *after* the initial attempt. Total attempts = `max_retries + 1`. |
| `backoff_factor` | `float` | `0.25` | Minimum delay floor in seconds (lower bound of decorrelated jitter). See [Jitter strategy](#jitter-strategy) for the full formula. |
| `max_wait` | `float` | `60.0` | Maximum delay cap in seconds. The jitter algorithm never waits longer than this between retries. |
| `retryable_status_codes` | `frozenset[int]` | `{408, 429, 500, 502, 503, 504}` | HTTP status codes that trigger a retry. The SDK retries on these codes and raises on all others. |

**`RetryConfig` applies to REST only.** The gRPC transport (Rust-backed) uses its own fixed retry
policy with 5 retries by default. See [Transport differences](#transport-differences).

### Disabling retries

To disable retries entirely, set `max_retries=0`:

```python
pc = Pinecone(retry_config=RetryConfig(max_retries=0))
```

With `max_retries=0`, the SDK makes exactly one attempt and raises immediately on any error.

### Handling rate limits without retrying

By default, 429 responses are retried automatically. To receive `RateLimitError` immediately
instead (for example, so your orchestrator can handle the retry), exclude 429 from the
retryable set:

```python
from pinecone import Pinecone, RetryConfig
from pinecone.errors import RateLimitError

pc = Pinecone(
    retry_config=RetryConfig(
        retryable_status_codes=frozenset({408, 500, 502, 503, 504}),  # no 429
    )
)

try:
    index.upsert(vectors=[...])
except RateLimitError:
    time.sleep(30.0)
    index.upsert(vectors=[...])
```

### Migration note: `backoff_factor` semantic change (v8 → v9)

In v8 and earlier, `backoff_factor` was an exponential multiplier. In v9, it became the
**minimum delay floor in seconds** — the lower bound of the decorrelated jitter window. The
default also changed from `2.0` to `0.25`. If you pinned `backoff_factor=2.0` in v8, the
new equivalent that produces a similar mean first-retry delay is `backoff_factor=0.5`; if
you want to restore the old default behavior (which caused ~4× longer delays than v9), pass
`backoff_factor=2.0` explicitly. Most users should use the v9 default or leave it unset.

---

## Jitter Strategy

Jitter spreads retries across time so that concurrent clients with the same retry budget
don't collide on the server at the same moment.

### Decorrelated jitter (backoff path)

When no server hint is present, the SDK uses decorrelated jitter:

```
delay = uniform(backoff_factor, max(backoff_factor, prev_delay * 3))
delay = min(delay, max_wait)
```

Starting from `prev_delay = backoff_factor`, each retry delay is drawn uniformly from
`[backoff_factor, prev_delay × 3]`, capped at `max_wait`. Because the next window's upper
bound grows with the previous delay, the sequence performs a random walk that diverges
naturally without a hard exponential schedule — neighboring clients are unlikely to pick
the same delay even when they start at the same time.

**Concrete example with defaults** (`backoff_factor=0.25`, `max_wait=60.0`):

| Attempt | Window (seconds) | Typical delay |
|---------|-----------------|---------------|
| 1st retry | [0.25, 0.75] | ~0.5 s |
| 2nd retry | [0.25, ~1.5] | ~0.9 s |
| 3rd retry | [0.25, ~4.5] | ~2.4 s |

---

## Adaptive Concurrency for Bulk Operations

When you run bulk upserts or other parallel operations, the SDK observes throttling
signals and automatically reduces the number of concurrent in-flight requests. When
throttling subsides, concurrency recovers.

### How it works

Each `Pinecone` client maintains a per-host concurrency limiter. On every retryable
response (429, 503, or equivalent gRPC code), the limiter halves the effective
concurrency floor for that host. After a streak of consecutive successful requests, it
recovers by one slot. The algorithm is AIMD (Additive Increase, Multiplicative Decrease)
— the same control loop used by TCP congestion control.

**You don't configure this directly.** The `max_concurrency` parameter you pass to
`upsert()` is a *ceiling* — the SDK self-tunes between 1 and that ceiling based on what
the server can absorb.

### Example

```python
from pinecone import Pinecone

pc = Pinecone()
index = pc.index(host="product-search-abc123.svc.pinecone.io")

# max_concurrency=8 is the ceiling.
# If the index throttles during the run, the SDK will automatically
# reduce effective concurrency (e.g. to 4, then 2) and recover as
# throttling subsides. No code changes required.
response = index.upsert(
    vectors=large_list,
    batch_size=200,
    max_concurrency=8,
)
print(response.upserted_count)
```

### Limiter scope

One limiter per index host per `Pinecone` client. If you create two `Pinecone` clients and
both target the same index, they each maintain an independent limiter — there is no
cross-client coordination (see [Multi-process and serverless workloads](#multi-process-and-serverless-workloads)).

---

## Transport Differences

The retry plan goal is parity across REST and gRPC. The remaining differences are small:

| Aspect | REST (`Index`, `AsyncIndex`) | gRPC (`GrpcIndex`) |
|--------|------------------------------|---------------------|
| Default `max_retries` | 3 (4 total attempts) | 5 (6 total attempts) |
| Configured via | `RetryConfig` passed to `Pinecone()` | Fixed in transport (not user-configurable) |
| Retryable codes | `{408, 429, 500, 502, 503, 504}` | UNAVAILABLE, RESOURCE\_EXHAUSTED, ABORTED |
| Jitter algorithm | Decorrelated jitter (Python) | Decorrelated jitter (Rust) |
| Async support | Yes (`AsyncIndex`) | No — gRPC transport is sync-only |
| Adaptive concurrency | Yes (REST + gRPC share the same per-host limiter registry) | Yes |

**gRPC retry is not configurable via `RetryConfig`.** If you need to tune gRPC retry
behavior, construct `GrpcIndex` directly (rather than through `Pinecone.index(grpc=True)`)
and pass `max_retries` explicitly.

---

## Multi-Process and Serverless Workloads

### What the SDK cannot do

The SDK's retry and adaptive concurrency machinery is per-process. If your workload fans
out across multiple Lambda invocations, Cloud Run instances, or Kubernetes pods, each
process runs its own independent retry loop. There is no shared state, no cross-process
coordination, and no distributed rate-limit awareness.

The per-client adaptive-limiter registry is capped at 256 hosts with LRU eviction; long-running
services that rotate through more than 256 distinct hosts will see infrequently-used hosts'
adaptive state reset on next use, which is harmless.

This means:

- N simultaneously throttled invocations each independently back off and retry. Without
  coordination, they can collide again at the end of the retry window.
- The adaptive concurrency limiter starts from scratch for each new process instance (e.g.
  a fresh Lambda cold start). It cannot inherit a reduced limit that another invocation
  learned from throttling.

### Recommended pattern for fan-out workloads

Let your orchestrator handle retries at the job level, and keep the SDK's retry window
narrow:

```python
from pinecone import Pinecone, RetryConfig
from pinecone.errors import RateLimitError

# Set max_retries=0 or 1: one attempt (or one fast retry), then raise.
# Let the SQS visibility timeout / Cloud Tasks retry / Step Functions catch
# handle the outer retry loop.
pc = Pinecone(retry_config=RetryConfig(max_retries=1))
index = pc.index(host="product-search-abc123.svc.pinecone.io")

try:
    response = index.upsert(vectors=batch, batch_size=100, max_concurrency=4)
except RateLimitError as exc:
    # Re-raise so the orchestrator sees a task failure and schedules a retry
    # after the visibility timeout expires.
    raise
```

### Why jitter still helps across processes

Even without coordination, the SDK's decorrelated jitter provides statistical relief. If N
independent Lambda invocations are all throttled at once, they don't all retry at the same
instant — each draws its own delay, spreading the retries across a window. The larger N is,
the more this matters.

### Summary: when to trust the SDK vs. the orchestrator

| Scenario | Recommended approach |
|----------|----------------------|
| Single-process bulk upsert | Use defaults — SDK handles everything |
| Long-running worker (persistent process) | Use defaults — adaptive limiter learns and recovers |
| Lambda / Cloud Functions / Cloud Run (stateless) | `max_retries=1`, catch `RateLimitError`, re-raise for orchestrator retry |
| Fan-out across many pods (e.g. Kubernetes Job) | Same as stateless — set low `max_retries`, rely on orchestrator |
| Strict per-invocation SLA (must not block) | `max_retries=0`, `retryable_status_codes=frozenset()` — raise immediately |

---

## Observability

The SDK emits namespaced log records with consistent `key=value` fields in each message,
so you can diagnose retry storms and throttling pressure without adding instrumentation
yourself.

### Log namespaces

| Logger | Events |
|--------|--------|
| `pinecone._internal.http_client` | Throttled HTTP response received; retry delay computed |
| `pinecone._internal.adaptive` | AIMD concurrency limit transitions |

### INFO messages

An INFO-level record is emitted the **first time** a given host rate-limits a client
instance:

```
Rate limited by host=<host>. Adaptive concurrency will reduce in-flight requests.
See https://docs.pinecone.io/python/retries for details.
```

This fires once per host per `Pinecone` / `AsyncPinecone` object, so it surfaces in your
logs without flooding them on repeated throttling.

### DEBUG messages

Enable DEBUG-level logging on the two namespaces above to see granular retry events:

```python
import logging
logging.getLogger("pinecone._internal.http_client").setLevel(logging.DEBUG)
logging.getLogger("pinecone._internal.adaptive").setLevel(logging.DEBUG)
```

**Throttle record** (emitted once per retry attempt that receives a retryable response):

```
Throttled response: status=429 host=my-index.svc.pinecone.io attempt=1/4 delay=0.531s
```

Fields: `status` (HTTP status code), `host`, `attempt` (N of total attempts),
`delay` (computed wait in seconds).

**AIMD limit decrease** (emitted when the adaptive limiter reduces concurrency):

```
AIMD limiter decreased: before=8 after=4 ceiling=8
```

**AIMD limit increase** (emitted when the limiter recovers a concurrency slot):

```
AIMD limiter increased: now=5 ceiling=8
```

Increase records only fire on actual transitions — not on every successful request —
so the volume is proportional to recovery events, not request throughput.

---

## See Also

- {doc}`/guides/error-handling` — Exception hierarchy and how to catch specific errors
- {doc}`/guides/performance` — Bulk upsert patterns, `max_concurrency` tuning, and transport selection
- {doc}`/guides/sync-vs-async` — When to use the async client and how to manage concurrency with `asyncio`
