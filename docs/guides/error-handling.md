# Error Handling

Every exception the SDK defines derives from `PineconeError`, so `except PineconeError`
is the one blanket catch, and it is the right catch for anything that reaches the
network.

It is not a catch for calling the SDK wrongly. A misspelled or unsupported keyword
argument raises a plain built-in `TypeError` naming the argument, not a
`PineconeTypeError`:

```
TypeError: Index() got unexpected keyword arguments: ['retry_config']
```

`AsyncPinecone(proxy_headers=...)` likewise raises `NotImplementedError` — the sync
`Pinecone` accepts that argument and the async client does not. Neither is a
`PineconeError`. Both are programming mistakes that fail on the first call, so they
belong in your tests rather than in an `except` block.

Every class below is importable from the package root or from `pinecone.errors`;
the two paths reach the same class.

```python
from pinecone import NotFoundError
from pinecone.errors import NotFoundError  # same class
```

## Exception Hierarchy

```
PineconeError
├── ApiError                      # the server answered with an HTTP error status
│   ├── UnauthorizedError         # 401
│   ├── PaymentRequiredError      # 402
│   ├── ForbiddenError            # 403
│   ├── NotFoundError             # 404
│   ├── ConflictError             # 409
│   ├── FailedPreconditionError   # 412
│   ├── RateLimitError            # 429
│   └── ServiceError              # 5xx
├── PineconeValueError            # also a built-in ValueError
├── PineconeTypeError             # also a built-in TypeError
├── PineconeConnectionError       # the request never got an answer
├── PineconeTimeoutError          # also a built-in TimeoutError
├── ResponseParsingError          # the answer arrived and could not be decoded
├── IndexInitFailedError          # a polled index entered InitializationFailed
└── IndexTerminatedError          # a polled index entered Terminating or Disabled
```

Three of these multiply inherit from a Python built-in, which changes what your
existing handlers catch:

| Class | Also inherits | Consequence |
|-------|---------------|-------------|
| `PineconeValueError` | `ValueError` | An existing `except ValueError` already catches SDK validation failures |
| `PineconeTypeError` | `TypeError` | An existing `except TypeError` already catches them, alongside bare `TypeError`s |
| `PineconeTimeoutError` | `TimeoutError` | An existing `except TimeoutError` already catches SDK timeouts |

The inheritance runs one way only. `except PineconeError` does not catch a plain
`ValueError` or `TimeoutError` raised by something else in your stack.

A status the SDK does not have a subclass for — a `400` from a request the server
rejected, for instance — raises `ApiError` itself. Catch `ApiError` if you want
every server-side failure; catch a subclass to single one out.

Full API reference, including each class's attributes: {doc}`/reference/exceptions`.

## Who Raises What

The SDK validates arguments before it opens a socket. Anything it can decide
locally, it decides locally, so a bad namespace name or an empty ID list costs no
round trip:

| Raised by | Class | Cause |
|-----------|-------|-------|
| The SDK, before the request | `PineconeValueError` | An argument's value is not one the API accepts: an empty name, a namespace outside the allowed charset, a count outside the allowed range, or two mutually exclusive selectors passed together |
| The SDK, before the request | `PineconeTypeError` | An argument's type is wrong: a non-string vector ID, non-dict metadata, or a request body that cannot be JSON-encoded |
| The transport | `PineconeConnectionError` | DNS failure, connection refused, TLS failure, a reset mid-stream — the request never got an answer |
| The transport | `PineconeTimeoutError` | A connect, read, write, or pool deadline elapsed |
| The server | `ApiError` and its subclasses | The request reached Pinecone and came back with an error status |
| The SDK, after the response | `ResponseParsingError` | The response arrived but did not decode into the expected shape |

`PineconeConnectionError` and `PineconeTimeoutError` can come out of any method
that talks to Pinecone. No method's own documentation repeats them, because the
answer is the same everywhere: the call did not complete, and whether the write
landed is unknown.

## Catching Specific Errors

```python
from pinecone import Pinecone
from pinecone.errors import (
    NotFoundError,
    ConflictError,
    UnauthorizedError,
    ForbiddenError,
    PaymentRequiredError,
    FailedPreconditionError,
    RateLimitError,
    ServiceError,
    PineconeConnectionError,
    PineconeTimeoutError,
)

pc = Pinecone()

try:
    pc.indexes.describe("nonexistent-index")
except NotFoundError:
    print("Index does not exist")
except ConflictError:
    print("Operation conflicts with current state")
except UnauthorizedError:
    print("Invalid or missing API key")
except ForbiddenError:
    print("API key lacks permission for this operation")
except PaymentRequiredError as exc:
    print(f"Billing blocks this operation: {exc.message}")
except FailedPreconditionError as exc:
    print(f"Resource is not in a state that permits this: {exc.message}")
except RateLimitError:
    print("Rate limited; backing off before retrying")
except ServiceError as exc:
    print(f"Server error {exc.status_code}: {exc.message}")
except PineconeConnectionError:
    print("Network error; check your connection")
except PineconeTimeoutError:
    print("Request timed out")
```

Order matters: `except ApiError` before any of these would swallow all of them.

## ApiError Attributes

`ApiError` and every subclass carry structured context:

| Attribute | Type | Description |
|-----------|------|-------------|
| `status_code` | `int` | The HTTP status the server returned |
| `message` | `str` | The server's own error message, already unwrapped from the response body |
| `body` | `dict \| None` | The parsed JSON response body, or `None` when the body was not JSON |
| `reason` | `str \| None` | HTTP reason phrase |
| `headers` | `dict \| None` | Response headers |
| `error_code` | `str \| None` | The server's machine-readable error code, when the body carries one |
| `request_id` | `str \| None` | Request ID, for correlating with Pinecone support |

`str(exc)` renders the ones that matter: `[404 NOT_FOUND] message (request_id: ...)`.

```python
from pinecone.errors import ApiError

try:
    pc.indexes.describe("my-index")
except ApiError as exc:
    print(exc.status_code)
    print(exc.message)
    print(exc.request_id)
```

Reading `exc.body["error"]["message"]` yourself is redundant — that is where
`exc.message` comes from. Reach for `body` only for fields the SDK does not lift
out.

`RateLimitError` adds one attribute of its own: `retry_after`, the response's
`Retry-After` header as a number of seconds. It is `None` when the header was
absent, negative, or an HTTP-date, which the SDK does not parse.

## Retry It, or Give Up

Whether an exception is worth a second attempt is a property of the class, not of
the call site:

| Class | Retry by hand? |
|-------|----------------|
| `PineconeConnectionError`, `PineconeTimeoutError` | Yes, once the SDK's own retries are spent — but note the write may have landed |
| `RateLimitError` | Yes, after `retry_after` if it is set |
| `ServiceError` | Yes |
| `FailedPreconditionError` | Yes, once you satisfy the precondition — `message` names what is in the way |
| `ConflictError` | Only after resolving the conflict; a bare retry answers 409 again |
| `NotFoundError` | No, with one exception — see [When 404 does not mean "not found"](#when-404-does-not-mean-not-found) |
| `UnauthorizedError`, `ForbiddenError` | No. Fix the key or its permissions |
| `PaymentRequiredError` | No. An organization owner has to resolve the billing state |
| `PineconeValueError`, `PineconeTypeError` | No. The request was never sent; fix the argument |
| `ResponseParsingError` | No |

The SDK already retries the transient half of that table for you, several times,
before you ever see the exception. Reaching one of them means the built-in retries
were exhausted, so an immediate re-call is unlikely to help — see
{doc}`/guides/retries` for what it already tried and how long it waited.

Retrying a **delete** deserves its own note. Control-plane deletes are not
idempotent in the "second call also succeeds" sense: `pc.indexes.delete()` and
`admin.role_bindings.delete()` both answer 404 once the resource is gone. So if you
wrap a delete in your own retry loop, treat `NotFoundError` as success rather than
as failure.

## ConflictError when creating an index

If you call `pc.indexes.create()` and an index with that name already exists, the
server answers 409 and the SDK raises `ConflictError`. The idiomatic fix is to
guard the create call with `pc.indexes.exists()`:

```python
from pinecone import Pinecone

pc = Pinecone()

if not pc.indexes.exists("my-index"):
    pc.indexes.create(
        name="my-index",
        schema={"fields": {"embedding": {"type": "dense_vector", "dimension": 1536, "metric": "cosine"}}},
        deployment={"deployment_type": "managed", "cloud": "aws", "region": "us-east-1"},
    )
```

`exists()` returns `True` if an index with that name is present, `False` otherwise.

If you genuinely cannot check first — concurrent callers race between the check and
the create — catch `ConflictError` and treat it as a no-op:

```python
from pinecone import Pinecone
from pinecone.errors import ConflictError

pc = Pinecone()

try:
    pc.indexes.create(
        name="my-index",
        schema={"fields": {"embedding": {"type": "dense_vector", "dimension": 1536, "metric": "cosine"}}},
        deployment={"deployment_type": "managed", "cloud": "aws", "region": "us-east-1"},
    )
except ConflictError:
    pass  # index already exists, nothing to do
```

## When 404 does not mean "not found"

`pc.restore_jobs.describe()` is the exception to the rule that a 404 identifies a
missing resource. Every failure to read the restore-job store answers 404,
including a transient one, so `NotFoundError` from that call means "could not
produce this job" rather than "this job does not exist".

Control flow keyed on it is therefore unsafe: giving up, deleting local state, or
reporting the restore as gone can each be the wrong call on what was really a
temporary store failure. Do not match on the message text either — a job whose
target index was deleted also answers 404, with a different message than a
genuinely missing job produces.

```python
from pinecone.errors import NotFoundError

try:
    job = pc.restore_jobs.describe(job_id="rj-abc123")
except NotFoundError:
    # Possibly transient. Retry before concluding the job is gone.
    ...
```

## Retries

The SDK retries transient failures automatically, on every HTTP method, before
raising anything. Bulk operations additionally self-tune their concurrency down
when a host pushes back.

What is retried, how long the backoff waits, what is configurable, and what to do
in a multi-process fan-out: {doc}`/guides/retries`.

## Timeouts

The default request timeout is 30 seconds. Pass `timeout` to the `Pinecone`
constructor to change the client-wide default:

```python
pc = Pinecone(timeout=10.0)
```

Many methods also accept a per-request `timeout` that overrides the client default
for that call, and the bulk methods accept a `total_timeout` that bounds the whole
operation instead of one attempt. The distinction matters: `timeout` is the deadline
for a single HTTP attempt and each retry gets a fresh one, so a call can outlive its
`timeout` several times over. {doc}`/guides/retries` covers `total_timeout`.

A `timeout` that elapses raises `PineconeTimeoutError`, which also inherits from
Python's built-in `TimeoutError`:

```python
except TimeoutError:
    # Catches PineconeTimeoutError as well
    ...
```

`PineconeTimeoutError` carries a `response` attribute, which is `None` for an
ordinary request timeout. On the bulk paths that raise instead of returning —
`upsert_from_dataframe(on_error="raise")`, and the gRPC `upsert` — it holds the
partial result, so `response.failed_items` tells you what to re-send rather than
leaving you to resubmit everything.
