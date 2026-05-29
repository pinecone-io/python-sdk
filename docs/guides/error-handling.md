# Error Handling

All exceptions raised by the SDK are subclasses of `PineconeError`, so a single
`except PineconeError` block catches everything the SDK can raise. More specific
subclasses let you handle individual failure modes differently.

## Exception Hierarchy

```
PineconeError (base)
├── ApiError                    # Server returned an HTTP error response
│   ├── NotFoundError           # 404
│   ├── ConflictError           # 409
│   ├── UnauthorizedError       # 401
│   ├── ForbiddenError          # 403
│   ├── RateLimitError          # 429
│   └── ServiceError            # 5xx
├── PineconeConnectionError     # Network-level failure (DNS, refused, transport)
├── PineconeTimeoutError        # Operation exceeded its timeout
├── PineconeValueError          # Invalid value passed to the SDK
└── PineconeTypeError           # Wrong type passed to the SDK
```

Full reference: {doc}`/reference/exceptions`.

## Catching Specific Errors

```python
from pinecone import Pinecone
from pinecone.errors import (
    NotFoundError,
    ConflictError,
    UnauthorizedError,
    ForbiddenError,
    RateLimitError,
    ServiceError,
    PineconeConnectionError,
    PineconeTimeoutError,
)

pc = Pinecone()

try:
    pc.indexes.delete("nonexistent-index")
except NotFoundError:
    print("Index does not exist")
except ConflictError:
    print("Operation conflicts with current state")
except UnauthorizedError:
    print("Invalid or missing API key")
except ForbiddenError:
    print("API key lacks permission for this operation")
except RateLimitError as exc:
    # exc.retry_after is the parsed Retry-After header in seconds, or None
    print(f"Rate limited; retry after {exc.retry_after}s")
except ServiceError as exc:
    print(f"Server error {exc.status_code}: {exc.message}")
except PineconeConnectionError:
    print("Network error — check your connection")
except PineconeTimeoutError:
    print("Request timed out")
```

## ApiError Attributes

`ApiError` (and its subclasses) carry structured context:

| Attribute | Type | Description |
|-----------|------|-------------|
| `status_code` | `int` | HTTP status code returned by the server |
| `message` | `str` | Human-readable error description |
| `body` | `dict \| None` | Parsed response body, if available |
| `reason` | `str \| None` | HTTP reason phrase |
| `headers` | `dict \| None` | Response headers |

```python
from pinecone.errors import ApiError

try:
    pc.indexes.describe("my-index")
except ApiError as exc:
    print(exc.status_code)  # e.g. 404
    print(exc.message)
    if exc.body:
        print(exc.body.get("error", {}).get("message"))
```

## ConflictError when creating an index

If you call `pc.indexes.create()` and an index with that name already exists, the server
returns a 409 and the SDK raises `ConflictError`.  The idiomatic fix is to guard the
create call with `pc.indexes.exists()`:

```python
from pinecone import Pinecone

pc = Pinecone()

if not pc.indexes.exists("my-index"):
    pc.indexes.create(
        name="my-index",
        spec={"serverless": {"cloud": "aws", "region": "us-east-1"}},
        dimension=1536,
        metric="cosine",
    )
```

`exists()` returns `True` if an index with that name is present, `False` otherwise.

If you genuinely cannot check first (e.g. concurrent callers), catch `ConflictError`
and treat it as a no-op:

```python
from pinecone import Pinecone
from pinecone.errors import ConflictError

pc = Pinecone()

try:
    pc.indexes.create(
        name="my-index",
        spec={"serverless": {"cloud": "aws", "region": "us-east-1"}},
        dimension=1536,
        metric="cosine",
    )
except ConflictError:
    pass  # index already exists, nothing to do
```

## Retries

The SDK retries failed requests automatically. The default `RetryConfig` retries up to
**3 attempts total** (1 initial + 2 retries) with exponential backoff for status codes
`408`, `429`, `500`, `502`, `503`, and `504`.

Customize retry behavior by passing a `RetryConfig` to `Pinecone()`:

```python
from pinecone import Pinecone, RetryConfig

pc = Pinecone(
    retry_config=RetryConfig(
        max_retries=5,
        backoff_factor=1.0,
        max_wait=30.0,
        retryable_status_codes=frozenset({429, 500, 503}),
    )
)
```

To disable retries entirely, set `max_retries=1`:

```python
pc = Pinecone(retry_config=RetryConfig(max_retries=1))
```

A 429 response triggers a retry by default (with the `Retry-After` header honored if
present). If retries are exhausted, the SDK raises `RateLimitError` with the parsed
`Retry-After` value on `exc.retry_after`. To handle rate limits without relying on
retries, set `retryable_status_codes=frozenset()` excluding 429 and catch
`RateLimitError` directly.

## Timeouts

The default request timeout is **30 seconds**. Pass `timeout` to the `Pinecone`
constructor to change the client-wide default:

```python
pc = Pinecone(timeout=10.0)
```

Many methods also accept a per-request `timeout` keyword argument that overrides the
client default for that call:

```python
# Wait up to 60 seconds for this specific upsert
index.upsert(vectors=[...], timeout=60.0)
```

When a request times out, the SDK raises `PineconeTimeoutError`, which also inherits
from Python's built-in `TimeoutError`:

```python
except TimeoutError:
    # Catches PineconeTimeoutError as well
    ...
```
