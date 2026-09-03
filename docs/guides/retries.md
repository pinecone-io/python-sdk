# Retries and Resilience

The SDK retries failed requests automatically. Three mechanisms stack on top of each other: a
per-request retry loop with decorrelated jitter, a per-host retry budget that suppresses retries
during an outage, and an adaptive concurrency gate that throttles bulk operations down when a host
pushes back.

Most of that runs on fixed policy. The one configurable piece, `RetryConfig`, reaches the control
plane and gRPC but not data-plane REST. Read [Where Retry Configuration Applies](#where-retry-configuration-applies)
before you reach for it.

For the exceptions the SDK raises when retries are exhausted, see {doc}`/guides/error-handling`.

---

## Defaults at a Glance

Out of the box, no configuration needed:

| What | Default behavior |
|------|------------------|
| Max retries (after initial attempt) | 3 for REST (4 total attempts), 5 for gRPC (6 total attempts) |
| HTTP methods retried | All of them — GET, HEAD, POST, PUT, PATCH, DELETE alike |
| Retryable HTTP status codes | 408, 429, 500, 502, 503, 504 |
| Retried transport failures | Any `httpx.TransportError`: connect, read, write, and pool timeouts, and connection resets |
| Retryable gRPC status codes | UNAVAILABLE, RESOURCE\_EXHAUSTED, ABORTED. DEADLINE\_EXCEEDED is **not** retried |
| Backoff algorithm | Decorrelated jitter: random walk floored at `backoff_factor` and capped at `max_wait` |
| `Retry-After` header | Honored when it parses as a number of seconds, capped at `max_wait`, plus a random smear. The HTTP-date form is not parsed |
| Retry budget | Per host, process-global. Retries are suppressed while a host is failing steadily |
| Adaptive concurrency (bulk paths) | Self-tunes downward on throttling. `max_concurrency` is a ceiling, not a constant |

`max_retries` counts retries, not attempts. The default of `3` means up to **four**
requests reach the network, and `max_retries=0` means exactly one.

### Writes are retried too

Most HTTP clients retry only the methods the spec calls idempotent, and refuse to
retry a POST. This one retries every method, because Pinecone's data-plane writes
are idempotent at the server: upsert overwrites by record ID, and delete-by-ID and
update-by-ID both converge on the same state however many times they land. A retried
upsert that turns out to have succeeded the first time writes the same bytes again.

The consequence to hold on to is that a retryable failure never tells you the write
did *not* land — a response can be lost after the server applied it. That is
harmless for a data-plane write and it is not harmless everywhere: control-plane
deletes are not idempotent, so see
[Retry it, or give up](error-handling.md#retry-it-or-give-up) before you wrap one in
a retry loop of your own.

---

## Where Retry Configuration Applies

`RetryConfig` is a constructor argument on `Pinecone` and `AsyncPinecone`. It does not reach every
request those clients make:

| Requests | Governed by |
|----------|-------------|
| Control plane (`pc.indexes`, `pc.collections`, `pc.backups`, `pc.inference`, `pc.assistants`) | The `retry_config` you passed to `Pinecone()` / `AsyncPinecone()` |
| gRPC data plane (`pc.index(grpc=True)`, `GrpcIndex`) | The same `retry_config`, but only when you set it explicitly. See [Transport differences](#transport-differences) |
| REST data plane (`Index`, `AsyncIndex`, including everything from `pc.index()`) | Built-in defaults only. `retry_config` is not forwarded, and neither class accepts one |

So this reduces the control plane to a single attempt and leaves the upsert at four:

```python
from pinecone import Pinecone, RetryConfig

pc = Pinecone(retry_config=RetryConfig(max_retries=0))

pc.indexes.list()                            # 1 attempt
idx = pc.index(name="product-search")
idx.upsert(vectors=[("doc-1", [0.1, 0.2])])  # still 4 attempts, the built-in default
```

`Index(retry_config=...)` and `AsyncIndex(retry_config=...)` raise `TypeError`. Threading the
parameter through to the data plane is tracked as
[#159](https://github.com/pinecone-io/python-sdk-internal/issues/159); until it lands, the knobs
that do change data-plane behavior are in
[Tuning Data-Plane REST](#tuning-data-plane-rest).

The adaptive concurrency gate is not affected by any of this. It is fed from the transport itself,
so bulk operations self-tune against throttling on every path, configured or not.

---

## Configuring Retries

Pass a `RetryConfig` to the `Pinecone` constructor to customize control-plane retry behavior:

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
| `max_wait` | `float` | `60.0` | Maximum delay cap in seconds. Also caps how long a server's `Retry-After` can hold a request. |
| `retryable_status_codes` | `frozenset[int]` | `{408, 429, 500, 502, 503, 504}` | HTTP status codes that trigger a retry. The SDK retries on these codes and raises on all others. |

Those four are the whole configurable surface. `RetryConfig` carries a fifth field,
`on_throttle`, which is how the SDK wires its own adaptive-concurrency callback: the client
overwrites whatever you pass there, so it is not a hook you can use.

Transport-level failures are retried regardless of `retryable_status_codes`, which only classifies
responses that arrived. A connection reset or a read timeout never produces a status code, and the
retry loop treats every `httpx.TransportError` as retryable.

There is no method filter to configure, and no way to add one. See
[Writes are retried too](#writes-are-retried-too).

### Disabling retries

To disable retries entirely, set `max_retries=0`:

```python
from pinecone import Pinecone, RetryConfig

pc = Pinecone(retry_config=RetryConfig(max_retries=0))
```

With `max_retries=0`, control-plane calls make exactly one attempt and raise immediately on any
error. Data-plane REST calls keep their four attempts.

### Handling rate limits without retrying

By default, 429 responses are retried automatically. To receive `RateLimitError` immediately from
control-plane calls instead, exclude 429 from the retryable set:

```python
import time

from pinecone import Pinecone, RetryConfig
from pinecone.errors import RateLimitError

pc = Pinecone(
    retry_config=RetryConfig(
        retryable_status_codes=frozenset({408, 500, 502, 503, 504}),  # no 429
    )
)

schema = {"fields": {"embedding": {"type": "dense_vector", "dimension": 1536, "metric": "cosine"}}}

try:
    pc.indexes.create(name="product-search", schema=schema)
except RateLimitError as exc:
    time.sleep(exc.retry_after or 30.0)
```

`RateLimitError.retry_after` carries the server's own `Retry-After` in seconds, or `None`
when the header was absent or was an HTTP-date — the SDK does not parse that form.

Data-plane `RateLimitError` still arrives only after the built-in four attempts are spent.

### Migration note: `backoff_factor` semantic change (v8 → v9)

The default is `0.25` in both v8 and v9; what changed is what the number means. In v8 it
was an exponential multiplier fed to urllib3's `Retry`. In v9 it is the floor of a
decorrelated-jitter window, so no substitute value reproduces v8's curve exactly. If your
code depended on that timing, measure against the formula in
[Jitter strategy](#jitter-strategy) rather than assuming a 1:1 mapping. Otherwise leave it
unset.

---

## Tuning Data-Plane REST

The retry policy on `Index` and `AsyncIndex` is fixed, but three parameters still change how long a
data-plane call runs and how hard it pushes:

| Parameter | Where | Effect |
|-----------|-------|--------|
| `timeout` | `Pinecone(timeout=...)`, `Index(timeout=...)`, or per call | Deadline for one HTTP attempt. Each retry gets its own. |
| `total_timeout` | Bulk calls with `batch_size` set | Wall-clock deadline for the whole operation. Batches left unsent come back in `errors`. |
| `max_concurrency` | Bulk calls, 1 to 64, default 8 | Ceiling on in-flight batches for that call. |

`total_timeout` is the closest thing to "stop retrying and hand control back to me" on the data
plane, because it bounds the operation rather than any single attempt:

```python
response = idx.upsert(
    vectors=batch,
    batch_size=100,
    max_concurrency=4,
    total_timeout=30.0,
)
if response.has_errors:
    # Every item from every failed batch, flattened into one list.
    retry_later(response.failed_items)
```

The deadline does not raise. Batches it cut off arrive as `BatchError` entries in
`response.errors`, each with `disposition="unsent"` and a `PineconeTimeoutError` as its
`error`; `response.failed_items` flattens their contents back into one list. Two other
dispositions can appear alongside them: `"abandoned"` for batches dropped by
[stall detection](#stall-detection), and `"rejected"` for a batch whose attempt actually
completed with an error.

Filter on `err.retryable` before re-sending anything. It is `False` for the deterministic
failures — validation errors and 4xx rejections — that would fail identically on every
attempt, which is what keeps a poison batch from spinning forever:

```python
again = [
    item
    for err in response.errors
    if err.retryable
    for item in err.items
]
```

A non-batched call (no `batch_size`) has no `total_timeout`. Bound it with `timeout` and accept
that the worst case is four attempts plus backoff.

---

## Jitter Strategy

Jitter spreads retries across time so that concurrent clients with the same retry budget
don't collide on the server at the same moment.

### Decorrelated jitter (backoff path)

When no server hint is present, the SDK uses decorrelated jitter:

```
upper = min(max_wait, prev_delay * 3)
delay = uniform(backoff_factor, max(backoff_factor, upper))
```

Starting from `prev_delay = backoff_factor`, each retry delay is drawn uniformly from
`[backoff_factor, prev_delay × 3]`. `max_wait` clamps the window's upper bound before the draw
rather than clamping the drawn value, so a capped window still spreads its clients across the whole
range instead of piling them all on the cap. Because the next window's upper bound grows with the
previous delay, the sequence performs a random walk that diverges naturally without a hard
exponential schedule.

**Concrete example with defaults** (`backoff_factor=0.25`, `max_wait=60.0`):

| Attempt | Window (seconds) | Typical delay |
|---------|-----------------|---------------|
| 1st retry | [0.25, 0.75] | ~0.5 s |
| 2nd retry | [0.25, ~1.5] | ~0.9 s |
| 3rd retry | [0.25, ~2.6] | ~1.4 s |

### Retry-After (hinted path)

When a retryable response carries a `Retry-After` header the SDK follows it instead of drawing from
the backoff window. The header is read as a number of seconds; the HTTP-date form is not parsed, and
an unparseable or negative value falls back to decorrelated jitter. An accepted value is capped at
`max_wait`, then a random smear of up to half its length is added, so a fleet told to come back in
10 seconds arrives spread across 10 to 15 seconds rather than in one burst.

The delay the SDK actually waited seeds the next window either way, so one `Retry-After` hint widens
the subsequent backoff draws too.

---

## Retry Budget

Retrying harder is the wrong response to a host that is failing every request, so each host has a
token bucket that spends down as failures accumulate.

Every retryable failure costs one token out of 100; every 2xx earns 0.1 back. Retries are suppressed
while the bucket sits at or below half. First attempts never consult the budget, so a struggling host
still gets your traffic. It just stops getting the retry multiplier on top.

Against a host returning 429 to everything, calls start at the full four attempts and collapse to
one within about a dozen calls. Steady-state retry overhead stays near 10% of successful traffic.

The bucket is keyed by bare hostname and shared process-wide, so two `Pinecone` clients pointed at one
host draw on one ledger. It resets in a forked child.

gRPC runs the same policy with the same numbers, on a bucket held per channel rather than in a
process-wide registry. Two `GrpcIndex` objects against one host therefore keep separate ledgers,
where two `Index` objects share one.

---

## Adaptive Concurrency for Bulk Operations

When you run bulk upserts or other parallel operations, the SDK observes throttling
signals and automatically reduces the number of concurrent in-flight requests. When
throttling subsides, concurrency recovers.

### How it works

Every bulk call passes through a per-host admission gate. The gate's limit starts at 64 and moves by
AIMD (Additive Increase, Multiplicative Decrease), the same control loop TCP congestion control uses:

- **Decrease.** A retryable response halves the limit, measured from the number of requests actually
  in flight rather than from the limit itself. One decrease per in-flight epoch, so a single batch
  burning six retries cannot halve the limit six times while its siblings succeed.
- **Increase.** After a limit-sized streak of successful batches the limit goes up by one, but only
  if the gate was actually limit-bound during that streak. Idle headroom does not accumulate.
- **Pushback.** A `Retry-After` on the throttling response blocks admission until it elapses.

**You don't configure this directly.** The `max_concurrency` you pass to `upsert()` is a per-call
semaphore layered on top: your call never has more than its own bound outstanding, and global
admission works out to `min(max_concurrency, gate limit)`.

### Example

```python
from pinecone import Pinecone

pc = Pinecone()
index = pc.index(host="product-search-abc123.svc.pinecone.io")

# max_concurrency=8 bounds this call. If the index throttles mid-run, the gate
# halves the host-wide limit from what is in flight (8, then 4, then 2) and
# recovers as throttling subsides.
response = index.upsert(
    vectors=large_list,
    batch_size=200,
    max_concurrency=8,
)
print(response.upserted_count)
```

### Stall detection

At the floor, four consecutive all-failed batches mean the backend is not coming back on this
attempt. The gate flips to stalled and refuses admission for 30 seconds. Waiting callers are woken
so they can abandon their remainder instead of queueing against a dead host, and the abandoned
batches arrive in `errors` with `disposition="abandoned"` and a message saying they were never sent.
`response.failed_items` is what to re-submit once the host recovers.

The stall is a cool-down, not a terminal state. When it elapses the gate probes again from the floor.

### Gate scope

One gate per host, process-global. Two `Pinecone` clients in one process targeting the same index
share it, and so do concurrent bulk calls from different threads or event loops. The limit describes
a backend cell rather than a client object, so per-client gates would let each instance run the full
limit and would throw away everything learned whenever a client was recreated. Sharing can only
under-load a host, never over-load one.

The registry holds up to 1024 gates. At the cap it evicts a quiescent gate, never one with live
in-flight counts. A forked child starts with an empty registry.

The gate's own counters — how many throttle signals it heard, where the limit ended up, whether
it stalled — are not on any response object a public method hands back. `index.upsert()` returns
an `UpsertResponse`, which carries the per-batch outcome and nothing about the gate. To see what
the gate did, read the DEBUG records in [Observability](#observability).

---

## Transport Differences

REST and gRPC share the same retry shape. The differences are in the defaults, in what
`retryable_status_codes` means, and in which of them `retry_config` can reach:

| Aspect | REST control plane | REST data plane (`Index`, `AsyncIndex`) | gRPC (`GrpcIndex`) |
|--------|--------------------|------------------------------------------|---------------------|
| Default `max_retries` | 3 (4 total attempts) | 3 (4 total attempts) | 5 (6 total attempts) |
| Default `backoff_factor` | 0.25 | 0.25 | 0.1 |
| Configured via | `retry_config` passed to `Pinecone()` | Not configurable ([#159](https://github.com/pinecone-io/python-sdk-internal/issues/159)) | `retry_config` on `Pinecone()` when set explicitly, or passed directly to `GrpcIndex()` |
| Retryable codes | `{408, 429, 500, 502, 503, 504}` | Same, fixed | UNAVAILABLE, RESOURCE\_EXHAUSTED, ABORTED (fixed); `retryable_status_codes` is ignored |
| Jitter algorithm | Decorrelated jitter (Python) | Decorrelated jitter (Python) | Decorrelated jitter (Rust) |
| Server hint honored | `Retry-After` | `Retry-After` | `grpc-retry-pushback-ms`, then `retry-after` |
| Async support | Yes (`AsyncPinecone`) | Yes (`AsyncIndex`) | No (gRPC transport is sync-only) |
| Adaptive concurrency | n/a | Yes, shared process-global gate | Yes, same gate |

**A `retry_config` you pass to `Pinecone()` is only forwarded to gRPC if you set it
explicitly.** gRPC's defaults differ from REST's, so leaving `retry_config` unset keeps
`GrpcIndex` on its own numbers rather than inheriting REST's. To configure gRPC independently of
the control-plane client, construct `GrpcIndex` directly and pass `retry_config` to it.

Three gRPC behaviors have no REST counterpart:

- **DEADLINE_EXCEEDED is not retried.** A gRPC call that runs out of time raises rather than
  going round again, and `max_retries` is not the knob for it — raise `timeout` instead. The
  three codes that *are* retried compound with `timeout`, so under the default
  `max_retries=5` a batch that keeps failing on them can run to six attempts plus backoff.
- **The first retry draws from a wider window.** gRPC seeds the jitter window at ten times
  `backoff_factor` rather than at `backoff_factor`, so the first draw with the defaults spans
  roughly 0.1 to 3 seconds instead of 0.1 to 0.3. A backend restart returns UNAVAILABLE to
  every client at once and carries no hint, so the narrow window is exactly where a
  thundering herd would form.
- **A negative pushback means stop.** `grpc-retry-pushback-ms: -1` is the server refusing the
  retry, and the transport fails fast instead of backing off.

The set of retryable gRPC codes is fixed from Python. `RetryConfig.retryable_status_codes`
carries HTTP status codes and is not translated onto this transport.

---

## Multi-Process and Serverless Workloads

### What the SDK cannot do

The gate and the retry budget are process-global, not machine- or fleet-global. If your workload
fans out across multiple Lambda invocations, Cloud Run instances, or Kubernetes pods, each process
runs its own retry loop and its own gate. There is no shared state across processes, no
cross-process coordination, and no distributed rate-limit awareness.

This means:

- N simultaneously throttled invocations each independently back off and retry. Without
  coordination, they can collide again at the end of the retry window.
- The gate starts from a limit of 64 in each new process instance, for example a fresh Lambda cold
  start. It cannot inherit a reduced limit that another invocation learned from throttling.
- The retry budget starts full in each new process, so a host that a sibling invocation already
  found to be down still costs this one its full four attempts.

### Recommended pattern for fan-out workloads

Let your orchestrator handle retries at the job level, and keep each invocation's window narrow with
`total_timeout` rather than with `retry_config`, which does not reach data-plane REST:

```python
from pinecone import Pinecone

pc = Pinecone()
index = pc.index(host="product-search-abc123.svc.pinecone.io")

# Bound the operation, not the individual attempt. Let the SQS visibility
# timeout / Cloud Tasks retry / Step Functions catch handle the outer loop.
response = index.upsert(
    vectors=batch,
    batch_size=100,
    max_concurrency=4,
    total_timeout=20.0,
)
if response.has_errors:
    # Fail the task so the orchestrator reschedules it after the visibility
    # timeout, and carry the unsent items forward.
    raise RuntimeError(f"{response.failed_item_count} items unsent")
```

For control-plane work in the same handler, `RetryConfig(max_retries=1)` on the client does what you
would expect.

### Why jitter still helps across processes

Even without coordination, the SDK's decorrelated jitter provides statistical relief. If N
independent Lambda invocations are all throttled at once, they don't all retry at the same
instant; each draws its own delay, spreading the retries across a window. The larger N is,
the more this matters. A `Retry-After` hint gets the same treatment, smeared across up to 1.5×
the requested wait rather than snapping the whole fleet to one instant.

### Summary: when to trust the SDK vs. the orchestrator

| Scenario | Recommended approach |
|----------|----------------------|
| Single-process bulk upsert | Use defaults; SDK handles everything |
| Long-running worker (persistent process) | Use defaults; the gate learns and recovers, and the retry budget refills |
| Lambda / Cloud Functions / Cloud Run (stateless) | Set `total_timeout` on bulk calls, re-raise on `has_errors` for orchestrator retry |
| Fan-out across many pods (e.g. Kubernetes Job) | Same as stateless, plus a lower `max_concurrency` per pod |
| Strict per-invocation SLA (must not block) | Small `total_timeout`, or `timeout` on non-batched calls; hand `failed_items` back to the orchestrator |

---

## Observability

The SDK emits namespaced log records with consistent `key=value` fields in each message,
so you can diagnose retry storms and throttling pressure without adding instrumentation
yourself.

### Log namespaces

| Logger | Level | Events | Fires for |
|--------|-------|--------|-----------|
| `pinecone._internal.http_client` | DEBUG | Throttled response received, with the delay computed; connection error retried | Every REST request, control plane and data plane |
| `pinecone._internal.adaptive` | INFO, DEBUG | First throttle from a host (INFO); each concurrency-limit reduction (DEBUG) | Control-plane REST and gRPC only |
| `pinecone._internal.bulk.engine`, `.async_engine` | WARNING | Batches abandoned because the host looked unavailable | Bulk calls on any transport |

### INFO messages

An INFO-level record is emitted the **first time** a given host rate-limits a client
instance:

```
Rate limited by host=<host>. Adaptive concurrency will reduce in-flight requests.
See https://docs.pinecone.io/python/retries for details.
```

It fires once per host per `Pinecone` / `AsyncPinecone` object, so it surfaces in your
logs without flooding them on repeated throttling. Data-plane REST hosts do not produce it, because
the callback behind it rides on the `retry_config` that `Index` and `AsyncIndex` never receive
([#159](https://github.com/pinecone-io/python-sdk-internal/issues/159)). The DEBUG records below
are how you see data-plane throttling.

### DEBUG messages

Enable DEBUG-level logging on `pinecone._internal.http_client` to see granular retry events:

```python
import logging
logging.getLogger("pinecone._internal.http_client").setLevel(logging.DEBUG)
```

**Throttle record** (emitted once per retry attempt that receives a retryable response):

```
Throttled response: status=429 host=my-index.svc.pinecone.io attempt=1/4 delay=0.531s retry_after=absent
```

Fields: `status` (HTTP status code), `host`, `attempt` (N of total attempts),
`delay` (computed wait in seconds), `retry_after` (the response's `Retry-After` header
value, or `absent`).

**Connection error record** (emitted when a transport failure is retried):

```
Connection error on attempt 1/4, retrying: All connection attempts failed
```

Neither record is emitted for the final attempt, since nothing is retried after it. A request that
uses all four attempts logs three records and then raises.

**Limit reduction record**, on `pinecone._internal.adaptive`:

```
AIMD limiter decreased: before=8 after=4 ceiling=8
```

A bulk call whose logs end on a small `after` was being pushed back on, and `max_concurrency`
was not the thing bounding it.

---

## See Also

- {doc}`/guides/error-handling`: Exception hierarchy and how to catch specific errors
- {doc}`/guides/performance`: Bulk upsert patterns, `max_concurrency` tuning, and transport selection
- {doc}`/guides/sync-vs-async`: When to use the async client and how to manage concurrency with `asyncio`
