"""httpx-based HTTP client for sync and async operations."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import random
import socket
import ssl
import sys
import threading
import time
from collections import OrderedDict
from collections.abc import AsyncGenerator, Generator, Mapping, Sequence
from enum import Enum
from typing import Any, NoReturn
from urllib.parse import urlsplit

import httpx
import orjson

from pinecone import __version__
from pinecone._internal.config import PineconeConfig, RetryConfig
from pinecone._internal.constants import API_VERSION_HEADER, DEFAULT_BASE_URL
from pinecone._internal.user_agent import build_user_agent
from pinecone.errors.exceptions import (
    ApiError,
    ConflictError,
    FailedPreconditionError,
    ForbiddenError,
    NotFoundError,
    PaymentRequiredError,
    PineconeConnectionError,
    PineconeTimeoutError,
    PineconeTypeError,
    RateLimitError,
    ServiceError,
    UnauthorizedError,
)

# Patchable seams for the retry backoff sleeps. Unit tests no-op these two
# names; binding them here keeps the patch scoped to this transport instead
# of mutating time.sleep / asyncio.sleep for the whole process (issue #45).
_retry_sleep = time.sleep
_async_retry_sleep = asyncio.sleep

logger = logging.getLogger(__name__)


def _build_socket_options() -> list[tuple[int, int, int]]:
    """Build platform-specific TCP socket options.

    Enables TCP keep-alive and disables Nagle's algorithm on all platforms.
    Adds platform-specific keep-alive tuning on Linux and macOS.
    """
    opts: list[tuple[int, int, int]] = [
        (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),
        (socket.IPPROTO_TCP, socket.TCP_NODELAY, 1),
    ]
    if sys.platform == "linux":
        opts.append((socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 300))
        opts.append((socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 60))
        opts.append((socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 4))
    elif sys.platform == "darwin":
        opts.append((socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 60))
    return opts


def _build_ssl_verify(config: PineconeConfig) -> ssl.SSLContext | bool:
    """Resolve the caller's SSL settings into an httpx ``verify`` value.

    ``ssl_ca_certs`` is turned into a context here rather than handed to httpx
    as a path, because httpx deprecated ``verify=<str>`` in 0.28 and drops the
    form entirely in 1.0. Raises ``FileNotFoundError`` if the bundle does not
    exist, since a CA path that cannot be loaded means the caller does not have
    the trust they asked for.
    """
    ca_certs = config.ssl_ca_certs
    if not ca_certs:
        return config.ssl_verify
    if os.path.isdir(ca_certs):
        return ssl.create_default_context(capath=ca_certs)
    return ssl.create_default_context(cafile=ca_certs)


def _build_proxy(config: PineconeConfig) -> httpx.Proxy | str | None:
    """Resolve the caller's proxy settings into an httpx ``proxy`` value.

    Returned for use on our own transport, not on ``httpx.Client``/
    ``httpx.AsyncClient``: handing it to the Client instead makes httpx mount a
    second, separate transport for proxied requests that carries none of
    ``_RetryTransport``'s retries and none of ``_build_socket_options()``'s
    socket tuning, since that mount is never the transport we built.
    """
    if not config.proxy_url:
        return None
    if config.proxy_headers:
        return httpx.Proxy(url=config.proxy_url, headers=config.proxy_headers)
    return config.proxy_url


def _default_pool_size() -> int:
    """Return the default connection pool size: 5x CPU count with a floor of 20."""
    return max(5 * (os.cpu_count() or 1), 20)


# Asymmetric on purpose: orjson accepts a signed 64-bit minimum through an
# unsigned 64-bit maximum, and rejects either side of that.
_JSON_INT_MIN = -(2**63)
_JSON_INT_MAX = 2**64 - 1

# An out-of-range integer can carry thousands of digits; a diagnostic does not
# need all of them.
_VALUE_REPR_MAX_LEN = 60

# orjson gives up around 255 levels of nesting, so a body deeper than this
# failed for its depth, not for a value the walk could name.
_ENCODE_WALK_MAX_DEPTH = 256


def _clip_repr(value: Any) -> str:
    try:
        text = repr(value)
    except Exception:
        return "<unrenderable>"
    if len(text) > _VALUE_REPR_MAX_LEN:
        return text[:_VALUE_REPR_MAX_LEN] + "..."
    return text


def _describe_int(value: int) -> str:
    """Render *value* for a diagnostic without tripping over its own size.

    ``str()`` on an integer wider than ``sys.get_int_max_str_digits()`` raises,
    so an absurd value would otherwise turn its own error message into a second
    error. ``bit_length`` has no such limit.
    """
    try:
        text = str(value)
    except ValueError:
        return f"<{value.bit_length()}-bit integer>"
    if len(text) > _VALUE_REPR_MAX_LEN:
        return f"{text[:_VALUE_REPR_MAX_LEN]}... ({len(text)} digits)"
    return text


def _locate_unencodable(body: Any, path_prefix: str = "") -> tuple[str, str] | None:
    """Find the value in *body* that orjson refused, as ``(path, reason)``.

    Depth-first in document order, so the reported path is the first offender a
    reader would find by eye. Iterative rather than recursive: a body deep
    enough to trip orjson's nesting limit would also overflow a recursive walk,
    turning one diagnostic into a ``RecursionError``. Returns ``None`` when the
    cause cannot be pinned to a single value (nesting depth, or an orjson
    rejection this walk does not model).

    *path_prefix* locates *body* within the request the caller made, for
    encoders that serialize one piece at a time: an NDJSON line is its own
    ``orjson.dumps`` call, but only ``records[2].x`` is actionable.

    Only ever called after ``orjson.dumps`` has already failed, so its cost
    stays off the success path entirely.
    """
    stack: list[tuple[Any, str, int]] = [(body, path_prefix, 0)]
    while stack:
        value, path, depth = stack.pop()
        if depth > _ENCODE_WALK_MAX_DEPTH:
            return None
        where = path or "<body>"
        # bool is an int subclass and always encodes, so it must be excluded
        # before the range check.
        if value is None or isinstance(value, (bool, str, float)):
            continue
        if isinstance(value, int):
            if not _JSON_INT_MIN <= value <= _JSON_INT_MAX:
                return where, (
                    f"integer {_describe_int(value)} is outside the range JSON encoding "
                    f"supports ({_JSON_INT_MIN} to {_JSON_INT_MAX}). Send it as a "
                    f"string, or use a value within that range"
                )
            continue
        if isinstance(value, dict):
            items = list(value.items())
            for key, _ in items:
                if not isinstance(key, str):
                    return where, (
                        f"dict key {_clip_repr(key)} is not a string. JSON object keys "
                        f"must be strings"
                    )
            for key, item in reversed(items):
                child = f"{path}.{key}" if path else str(key)
                stack.append((item, child, depth + 1))
            continue
        if isinstance(value, (list, tuple)):
            stack.extend(
                (value[index], f"{path}[{index}]", depth + 1)
                for index in reversed(range(len(value)))
            )
            continue
        return where, (
            f"value of type {type(value).__name__} is not JSON-serializable. Convert it "
            f"to a str, int, float, bool, list, dict, or None first"
        )
    return None


def _encode_error(body: Any, exc: TypeError, path_prefix: str = "") -> PineconeTypeError:
    """Build the Pinecone-typed replacement for an orjson encode ``TypeError``.

    ``PineconeTypeError`` subclasses both ``PineconeError`` and the built-in
    ``TypeError``, so this swap is strictly additive: callers who were catching
    the bare ``TypeError`` still catch it, and ``except PineconeError`` now
    works too.
    """
    located = _locate_unencodable(body, path_prefix)
    if located is None:
        return PineconeTypeError(f"Request body cannot be JSON-encoded: {exc}", path_prefix or None)
    path, reason = located
    return PineconeTypeError(f"Request body cannot be JSON-encoded: {reason} ({exc})", path)


def _encode_json(body: Any, path_prefix: str = "") -> bytes:
    """Serialize *body* to JSON bytes using orjson (2-3x faster than stdlib json).

    orjson reports an unencodable body as a bare ``TypeError`` that names
    neither the offending field nor the limit it broke — "Integer exceeds
    64-bit range" and nothing else. Locate the value and re-raise it as a
    Pinecone error carrying the path, so the caller learns which document key
    or metadata field to fix (issue #187).
    """
    try:
        return orjson.dumps(body)
    except TypeError as exc:
        raise _encode_error(body, exc, path_prefix) from exc


def _encode_ndjson(records: Sequence[Mapping[str, Any]]) -> bytes:
    """Serialize *records* as an NDJSON request body, one JSON object per line.

    Each record is its own encode call, so a rejected value is reported against
    the index the caller passed it at — ``records[2].metadata.n`` rather than an
    offset into a concatenated blob (issue #196).
    """
    return b"".join(
        _encode_json(record, f"records[{index}]") + b"\n" for index, record in enumerate(records)
    )


def _prepare_json_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """If *kwargs* contains ``json=``, replace it with ``content=`` + Content-Type header.

    This bypasses httpx's default stdlib ``json.dumps`` in favour of orjson.
    """
    if "json" in kwargs:
        data = kwargs.pop("json")
        kwargs["content"] = _encode_json(data)
        headers: dict[str, str] = kwargs.pop("headers", {})
        headers["Content-Type"] = "application/json"
        kwargs["headers"] = headers
    return kwargs


def _resolve_query_value(value: Any) -> Any:
    """Return the wire form of one query-parameter value.

    ``Enum`` members stand for their ``.value``; every other value is passed
    through untouched so httpx keeps encoding ints and bools as it always has.
    """
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (list, tuple)):
        return [_resolve_query_value(item) for item in value]
    return value


def _prepare_params(kwargs: dict[str, Any]) -> dict[str, Any]:
    """If *kwargs* contains ``params=``, replace enum members with their values.

    httpx encodes every query value with ``str()``, and a ``(str, Enum)`` member
    stringifies to ``"VectorType.DENSE"`` rather than ``"dense"`` — so a member
    that reaches this boundary unresolved goes on the wire mangled. Bodies are
    not exposed to this: orjson and msgspec both serialize a member by its value.
    Normalizing here rather than at each call site covers every query parameter
    on every surface, including ones added later (issue #371).

    An already-encoded ``params`` — a query string, or an ``httpx.QueryParams``,
    which stringifies its values on construction — is left alone. There is no
    member left in it to resolve, and rebuilding one as a dict would drop
    repeated keys.
    """
    params = kwargs.get("params")
    if params is None or isinstance(params, (str, bytes, httpx.QueryParams)):
        return kwargs
    if isinstance(params, Mapping):
        kwargs["params"] = {key: _resolve_query_value(value) for key, value in params.items()}
    elif isinstance(params, (list, tuple)):
        kwargs["params"] = [(key, _resolve_query_value(value)) for key, value in params]
    return kwargs


_DOT_SEGMENT_WIRE_FORMS = {".": "%2E", "..": "%2E%2E"}


def _split_off_query_and_fragment(url: str) -> tuple[str, str]:
    """Split *url* into its path and the query/fragment suffix that follows it.

    Dot-segment normalization applies only to the path, so the suffix must be
    handed back unexamined.
    """
    end = len(url)
    for delimiter in "?#":
        found = url.find(delimiter)
        if found != -1:
            end = min(end, found)
    return url[:end], url[end:]


def _prepare_path(path: str) -> str:
    """Percent-encode any ``.`` or ``..`` segment of *path* so it survives to the server.

    ``quote(value, safe="")`` — the encoding the path parameters in this client
    apply — deliberately leaves ``.`` alone, because ``.`` is an RFC 3986
    unreserved character. But httpx normalizes a URL on construction, and that
    normalization includes RFC 3986 ``remove_dot_segments``: a segment of ``.``
    is dropped and a segment of ``..`` removes the segment before it. So a
    caller-supplied value of ``.`` or ``..`` never reaches the server. It
    silently rewrites the request to address a different endpoint, which for the
    collapsing verbs is a defined route rather than a 404.

    ``%2E`` is the wire form of a literal ``.`` segment: httpx does not decode
    percent-escapes before removing dot segments, so an encoded segment survives
    normalization and the server decodes it back to the caller's value.

    Only a segment that is *entirely* ``.`` or ``..`` is rewritten, so a value
    that merely contains dots (``a..b``, ``my.index``) is returned untouched and
    an already-encoded segment is never double-encoded.

    The path must arrive as a ``str``; anything else is returned unchanged. An
    ``httpx.URL`` cannot be repaired here because it normalizes on construction,
    so its dot segments are gone before this boundary sees it.
    """
    if not isinstance(path, str) or "." not in path:
        return path
    head, suffix = _split_off_query_and_fragment(path)
    segments = head.split("/")
    if not any(segment in _DOT_SEGMENT_WIRE_FORMS for segment in segments):
        return path
    encoded = "/".join(_DOT_SEGMENT_WIRE_FORMS.get(segment, segment) for segment in segments)
    return encoded + suffix


def _build_headers(config: PineconeConfig, api_version: str) -> dict[str, str]:
    headers: dict[str, str] = {
        API_VERSION_HEADER: api_version,
        "User-Agent": build_user_agent(__version__, config.source_tag or None),
    }
    if config.api_key:
        headers["Api-Key"] = config.api_key
    if config.additional_headers:
        headers.update(config.additional_headers)
    return headers


_SENSITIVE_HEADERS = frozenset({"api-key", "authorization", "proxy-authorization"})


def _redact_headers(headers: dict[str, str]) -> dict[str, str]:
    """Return a copy of *headers* with sensitive values replaced by ``***``."""
    return {k: "***" if k.lower() in _SENSITIVE_HEADERS else v for k, v in headers.items()}


def _log_curl(
    method: str,
    url: str,
    headers: dict[str, str],
    body: bytes | None = None,
) -> None:
    """Log a curl-equivalent command for debugging when PINECONE_DEBUG_CURL is set."""
    if not os.environ.get("PINECONE_DEBUG_CURL"):
        return
    safe_headers = _redact_headers(headers)
    parts = [f"curl -X {method} '{url}'"]
    for key, value in safe_headers.items():
        parts.append(f"-H '{key}: {value}'")
    if body is not None:
        parts.append(f"-d '{body.decode('utf-8', errors='replace')}'")
    curl_cmd = " ".join(parts)
    logger.debug("curl equivalent:\n%s", curl_cmd)


def _compute_backoff(config: RetryConfig, attempt: int, prev_delay: float | None) -> float:
    """Decorrelated jitter: uniform(base, prev*3), capped at max_wait."""
    base_delay = config.backoff_factor
    if prev_delay is None:
        prev_delay = base_delay
    upper = min(config.max_wait, prev_delay * 3.0)
    return random.uniform(base_delay, max(base_delay, upper))


def _compute_retry_after_delay(
    config: RetryConfig,
    response: httpx.Response,
    attempt: int,
    prev_delay: float | None,
) -> float:
    retry_after = response.headers.get("retry-after")
    if retry_after is not None:
        try:
            ra = float(retry_after)
            if ra >= 0:
                ra = min(ra, config.max_wait)  # cap: prevents unbounded delays
                smear = random.uniform(0.0, ra * 0.5)
                return ra + smear
        except (ValueError, TypeError):
            pass
    return _compute_backoff(config, attempt, prev_delay)


def _notify_throttle(
    config: RetryConfig, request: httpx.Request, response: httpx.Response | None = None
) -> None:
    """Feed the bulk admission gate, then any user hook.

    The gate feed lives here — at the one place that owns retryable-status
    detection for every REST transport, sync and async — so no client
    construction path can forget to wire it (the #60 lesson). A Retry-After
    header rides along as the gate's pushback hold.
    """
    host = request.url.host
    pushback: float | None = None
    if response is not None:
        retry_after = response.headers.get("retry-after")
        if retry_after is not None:
            try:
                pushback = max(0.0, min(float(retry_after), config.max_wait))
            except (ValueError, TypeError):
                pushback = None
    try:
        from pinecone._internal.bulk import get_registry

        get_registry().report_throttled(host, pushback)
    except Exception as exc:
        logger.debug("gate throttle report raised, ignoring: %s", exc)
    cb = config.on_throttle
    if cb is None:
        return
    try:
        cb(host)
    except Exception as exc:
        logger.debug("on_throttle callback raised, ignoring: %s", exc)


_BUDGET_MAX_TOKENS = 100.0
_BUDGET_TOKEN_RATIO = 0.1
_MAX_BUDGETS = 1024


class _RetryBudget:
    """gRFC A6 retry throttling: a per-host token bucket bounding the retry
    multiplier during partial outages (issue #76, the REST half of #55).

    Every retryable failure spends 1.0 token; every 2xx earns back 0.1;
    retries are suppressed while the bucket is at or below half. Steady-state
    retry overhead is thus capped near 10% of successful traffic, and retries
    self-disable during a total outage instead of doubling load at the worst
    moment. First attempts never consult the budget — only retries are gated.
    Composes with the adaptive gate rather than duplicating it: the gate
    bounds concurrency, the budget bounds attempts per request.
    """

    __slots__ = ("_lock", "_max_tokens", "_tokens")

    def __init__(self, max_tokens: float = _BUDGET_MAX_TOKENS) -> None:
        self._lock = threading.Lock()
        self._max_tokens = max_tokens
        self._tokens = max_tokens

    def record_success(self) -> None:
        with self._lock:
            self._tokens = min(self._max_tokens, self._tokens + _BUDGET_TOKEN_RATIO)

    def record_failure(self) -> None:
        with self._lock:
            self._tokens = max(0.0, self._tokens - 1.0)

    def allows_retry(self) -> bool:
        with self._lock:
            return self._tokens > self._max_tokens / 2.0

    def tokens(self) -> float:
        with self._lock:
            return self._tokens


class _BudgetRegistry:
    """Process-global for the same reason the gate registry is: budget state
    is a statement about a backend host, not about a client object — two
    clients in one process hammering one host must share one ledger.

    Keys are normalized by the gate registry's ``host_key`` (one
    normalization function in the SDK — the #60 lesson). Eviction at the cap
    is plain LRU: unlike gates, budgets hold no in-flight counts, and a
    re-created budget starts full, which errs toward allowing retries for a
    host we have not seen among the last 1024.
    """

    __slots__ = ("_budgets", "_lock", "_max_tokens")

    def __init__(self, max_tokens: float = _BUDGET_MAX_TOKENS) -> None:
        self._lock = threading.Lock()
        self._max_tokens = max_tokens
        self._budgets: OrderedDict[str, _RetryBudget] = OrderedDict()

    def get(self, host: str) -> _RetryBudget:
        from pinecone._internal.bulk.registry import host_key

        key = host_key(host or "")
        with self._lock:
            budget = self._budgets.get(key)
            if budget is None:
                budget = _RetryBudget(self._max_tokens)
                if len(self._budgets) >= _MAX_BUDGETS:
                    self._budgets.popitem(last=False)
                self._budgets[key] = budget
            else:
                self._budgets.move_to_end(key)
            return budget

    def _reset(self) -> None:
        with self._lock:
            self._budgets.clear()

    def _reset_unlocked(self) -> None:
        """Fork-child reset: inherited lock state is undefined, so rebuild
        without trying to take it (mirrors the gate registry)."""
        self._lock = threading.Lock()
        self._budgets = OrderedDict()


_budget_registry = _BudgetRegistry()


def get_budget_registry() -> _BudgetRegistry:
    return _budget_registry


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_budget_registry._reset_unlocked)


class _RetryTransport(httpx.BaseTransport):
    """Sync transport wrapper that retries on transient server errors."""

    def __init__(
        self,
        *,
        transport: httpx.HTTPTransport,
        retry_config: RetryConfig | None = None,
        budget_registry: _BudgetRegistry | None = None,
    ) -> None:
        self._transport = transport
        self._config = retry_config or RetryConfig()
        self._budgets = budget_registry if budget_registry is not None else get_budget_registry()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        budget = self._budgets.get(request.url.host)
        last_exc: httpx.TransportError | None = None
        prev_delay: float | None = None
        for attempt in range(self._config.max_retries + 1):
            try:
                response = self._transport.handle_request(request)
            except httpx.TransportError as exc:
                last_exc = exc
                budget.record_failure()
                if attempt < self._config.max_retries and budget.allows_retry():
                    logger.debug(
                        "Connection error on attempt %d/%d, retrying: %s",
                        attempt + 1,
                        self._config.max_retries + 1,
                        exc,
                    )
                    delay = _compute_backoff(self._config, attempt, prev_delay)
                    prev_delay = delay
                    _retry_sleep(delay)
                    continue
                break
            last_exc = None
            if response.status_code not in self._config.retryable_status_codes:
                if response.is_success:
                    budget.record_success()
                return response
            budget.record_failure()
            _notify_throttle(self._config, request, response)
            if attempt < self._config.max_retries and budget.allows_retry():
                response.close()
                delay = _compute_retry_after_delay(self._config, response, attempt, prev_delay)
                prev_delay = delay
                logger.debug(
                    "Throttled response: status=%d host=%s attempt=%d/%d"
                    " delay=%.3fs retry_after=%s",
                    response.status_code,
                    request.url.host,
                    attempt + 1,
                    self._config.max_retries + 1,
                    delay,
                    response.headers.get("retry-after", "absent"),
                )
                _retry_sleep(delay)
            else:
                return response
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("max_retries must be non-negative")

    def close(self) -> None:
        self._transport.close()


class _AsyncRetryTransport(httpx.AsyncBaseTransport):
    """Async transport wrapper that retries on transient server errors."""

    def __init__(
        self,
        *,
        transport: httpx.AsyncHTTPTransport,
        retry_config: RetryConfig | None = None,
        budget_registry: _BudgetRegistry | None = None,
    ) -> None:
        self._transport = transport
        self._config = retry_config or RetryConfig()
        self._budgets = budget_registry if budget_registry is not None else get_budget_registry()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        budget = self._budgets.get(request.url.host)
        last_exc: httpx.TransportError | None = None
        prev_delay: float | None = None
        for attempt in range(self._config.max_retries + 1):
            try:
                response = await self._transport.handle_async_request(request)
            except httpx.TransportError as exc:
                last_exc = exc
                budget.record_failure()
                if attempt < self._config.max_retries and budget.allows_retry():
                    logger.debug(
                        "Connection error on attempt %d/%d, retrying: %s",
                        attempt + 1,
                        self._config.max_retries + 1,
                        exc,
                    )
                    delay = _compute_backoff(self._config, attempt, prev_delay)
                    prev_delay = delay
                    await _async_retry_sleep(delay)
                    continue
                break
            last_exc = None
            if response.status_code not in self._config.retryable_status_codes:
                if response.is_success:
                    budget.record_success()
                return response
            budget.record_failure()
            _notify_throttle(self._config, request, response)
            if attempt < self._config.max_retries and budget.allows_retry():
                await response.aclose()
                delay = _compute_retry_after_delay(self._config, response, attempt, prev_delay)
                prev_delay = delay
                logger.debug(
                    "Throttled response: status=%d host=%s attempt=%d/%d"
                    " delay=%.3fs retry_after=%s",
                    response.status_code,
                    request.url.host,
                    attempt + 1,
                    self._config.max_retries + 1,
                    delay,
                    response.headers.get("retry-after", "absent"),
                )
                await _async_retry_sleep(delay)
            else:
                return response
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("max_retries must be non-negative")

    async def aclose(self) -> None:
        await self._transport.aclose()


def _release_response_refs(response: httpx.Response) -> None:
    """Break internal httpx reference cycles so responses can be collected by refcount.

    httpx wraps every response stream in BoundSyncStream which holds
    ``._response`` pointing back to the Response, creating the cycle
    ``Response.stream → BoundSyncStream._response → Response``.  After the
    body has been read the stream is no longer needed, so we null the
    back-reference to allow immediate collection without waiting for the
    cyclic GC.
    """
    stream = getattr(response, "stream", None)
    if stream is not None and hasattr(stream, "_response"):
        object.__setattr__(stream, "_response", None)


_TEXT_BODY_MAX_LEN = 500


def _extract_message_and_error_code(body: Any, response: httpx.Response) -> tuple[str, str | None]:
    try:
        if isinstance(body, dict):
            error_obj = body.get("error")
            if isinstance(error_obj, dict):
                msg = error_obj.get("message")
                code = error_obj.get("code")
                if isinstance(msg, str) and msg:
                    return msg, (code if isinstance(code, str) else None)
            for key in ("message", "detail", "description"):
                val = body.get(key)
                if isinstance(val, str) and val:
                    return val, None

        raw = response.text.strip()
        if raw:
            if len(raw) > _TEXT_BODY_MAX_LEN:
                raw = raw[:_TEXT_BODY_MAX_LEN] + "... (truncated)"
            return raw, None

        reason = response.reason_phrase
        if reason:
            return reason, None

        return "", None
    except Exception:
        return "", None


def _extract_request_id(headers: httpx.Headers | dict[str, str]) -> str | None:
    try:
        for name in ("x-pinecone-request-id", "x-request-id"):
            val = headers.get(name)
            if isinstance(val, str) and val:
                return val
        return None
    except Exception:
        return None


def _raise_for_status(response: httpx.Response) -> None:
    if response.is_success:
        return

    body: Any = None
    try:
        body = response.json()
    except Exception:
        body = None

    message, error_code = _extract_message_and_error_code(body, response)
    request_id = _extract_request_id(response.headers)

    status = response.status_code
    reason = response.reason_phrase
    headers = dict(response.headers)
    if status == 401:
        raise UnauthorizedError(
            message=message,
            status_code=status,
            body=body,
            reason=reason,
            headers=headers,
            error_code=error_code,
            request_id=request_id,
        )
    if status == 403:
        raise ForbiddenError(
            message=message,
            status_code=status,
            body=body,
            reason=reason,
            headers=headers,
            error_code=error_code,
            request_id=request_id,
        )
    if status == 402:
        raise PaymentRequiredError(
            message=message,
            status_code=status,
            body=body,
            reason=reason,
            headers=headers,
            error_code=error_code,
            request_id=request_id,
        )
    if status == 404:
        raise NotFoundError(
            message=message,
            status_code=status,
            body=body,
            reason=reason,
            headers=headers,
            error_code=error_code,
            request_id=request_id,
        )
    if status == 412:
        raise FailedPreconditionError(
            message=message,
            status_code=status,
            body=body,
            reason=reason,
            headers=headers,
            error_code=error_code,
            request_id=request_id,
        )
    if status == 429:
        retry_after_raw = response.headers.get("retry-after")
        retry_after: float | None
        if retry_after_raw is None:
            retry_after = None
        else:
            try:
                parsed = float(retry_after_raw)
                # reject negative values; NaN is also filtered since nan >= 0 is False
                retry_after = parsed if parsed >= 0 else None
            except (ValueError, TypeError):
                retry_after = None
        raise RateLimitError(
            message=message,
            status_code=status,
            body=body,
            reason=reason,
            headers=headers,
            error_code=error_code,
            request_id=request_id,
            retry_after=retry_after,
        )
    if status == 409:
        raise ConflictError(
            message=message,
            status_code=status,
            body=body,
            reason=reason,
            headers=headers,
            error_code=error_code,
            request_id=request_id,
        )
    if 500 <= status <= 599:
        raise ServiceError(
            message=message,
            status_code=status,
            body=body,
            reason=reason,
            headers=headers,
            error_code=error_code,
            request_id=request_id,
        )
    raise ApiError(
        message=message,
        status_code=status,
        body=body,
        reason=reason,
        headers=headers,
        error_code=error_code,
        request_id=request_id,
    )


def _raise_transport_error(exc: httpx.TransportError) -> NoReturn:
    """Translate an httpx transport failure into the SDK's error type.

    The async transport reports timeouts with an empty message
    (``ReadTimeout('')``) where the sync transport says ``'timed out'``, so
    fall back to naming the httpx fault rather than raising an error whose
    ``str()`` is empty.
    """
    if isinstance(exc, httpx.TimeoutException):
        message = str(exc) or f"Request timed out ({type(exc).__name__})"
        raise PineconeTimeoutError(message) from exc
    message = str(exc) or f"Connection failed ({type(exc).__name__})"
    raise PineconeConnectionError(message) from exc


class HTTPClient:
    """Synchronous HTTP client wrapping httpx."""

    def __init__(self, config: PineconeConfig, api_version: str) -> None:
        self._config = config
        self._headers = _build_headers(config, api_version)
        verify = _build_ssl_verify(config)
        pool_size = (
            config.connection_pool_maxsize
            if config.connection_pool_maxsize > 0
            else _default_pool_size()
        )
        limits = httpx.Limits(
            max_connections=pool_size,
            max_keepalive_connections=pool_size // 2,
        )
        # httpx.Client discards its own verify= when an explicit transport is
        # supplied, so the transport must carry it. The proxy goes on the same
        # transport rather than on the Client: a proxy passed to the Client
        # makes httpx mount a second transport for proxied requests, and that
        # mount — not this one — is what Client._transport_for_url returns, so
        # _RetryTransport and its socket options would never be reached.
        transport = _RetryTransport(
            transport=httpx.HTTPTransport(
                verify=verify,
                http2=False,
                limits=limits,
                proxy=_build_proxy(config),
                socket_options=_build_socket_options(),
            ),
            retry_config=config.retry_config,
        )
        self._client = httpx.Client(
            base_url=config.host or DEFAULT_BASE_URL,
            headers=self._headers,
            timeout=config.timeout,
            transport=transport,
        )
        # Pre-built per-request constants for the POST hot path. Lets us
        # bypass httpx.Client.build_request's URL/header/cookie/queryparam
        # merge cost on every call. The base-URL string is rstripped of
        # the trailing '/' httpx adds so concatenation with leading-'/'
        # paths yields the same URL as ``Client._merge_url(path)`` for
        # base URLs with or without a path component.
        self._post_default_headers: dict[str, str] = {
            **self._headers,
            "Content-Type": "application/json",
        }
        self._default_timeout_extensions: dict[str, Any] = {
            "timeout": httpx.Timeout(config.timeout).as_dict()
        }
        self._base_url_str: str = str(self._client.base_url).rstrip("/")
        # Pre-tokenized Headers carrying the Host injection that
        # httpx.Request._prepare would normally add. Lets the POST fast
        # path use stream= (which skips encode_request/_prepare/read)
        # without losing the Host header that HTTP/1.1 servers require.
        # Cloning via httpx.Headers(other) is a fast list copy of the
        # internal _list (~0.5 µs) vs ~3.5 µs to rebuild from a dict.
        # Derive Host from the config (a plain str) rather than
        # _client.base_url so the value stays a real str even when
        # httpx.Client is mocked in unit tests.
        post_headers_with_host: dict[str, str] = dict(self._post_default_headers)
        host_netloc = urlsplit(config.host or DEFAULT_BASE_URL).netloc
        if host_netloc:
            post_headers_with_host["Host"] = host_netloc
        self._post_default_headers_obj: httpx.Headers = httpx.Headers(post_headers_with_host)
        # Cache of parsed httpx.URL by path. ~17 µs/call savings on hit
        # since URL parsing dominates Request construction. Capped to
        # prevent unbounded growth in long-lived clients with many
        # distinct paths (e.g. per-namespace operations).
        self._url_cache: dict[str, httpx.URL] = {}

    def _build_url(self, path: str) -> str:
        return f"{self._client.base_url}{path}"

    def get(
        self, path: str, timeout: float | httpx.Timeout | None = None, **kwargs: Any
    ) -> httpx.Response:
        path = _prepare_path(path)
        _log_curl("GET", self._build_url(path), dict(self._headers))
        effective_timeout = timeout if timeout is not None else self._config.timeout
        try:
            response = self._client.get(path, timeout=effective_timeout, **_prepare_params(kwargs))
        except httpx.TransportError as exc:
            _raise_transport_error(exc)
        _raise_for_status(response)
        _release_response_refs(response)
        return response

    def post(
        self, path: str, timeout: float | httpx.Timeout | None = None, **kwargs: Any
    ) -> httpx.Response:
        path = _prepare_path(path)
        # Fast path: callers only pass json= (and rarely timeout=). When
        # nothing else is in kwargs we can construct the httpx.Request
        # directly and skip Client.build_request's _merge_url/_merge_headers/
        # cookies/queryparams/timeout work — ~40 µs of GIL-held work per call.
        # When uncommon kwargs (params, files, content, headers) are passed,
        # fall back to build_request so its full machinery handles them.
        if kwargs.keys() <= {"json"}:
            content_bytes: bytes | None = _encode_json(kwargs["json"]) if "json" in kwargs else None
            if os.environ.get("PINECONE_DEBUG_CURL"):
                _log_curl(
                    "POST",
                    self._build_url(path),
                    self._post_default_headers,
                    body=content_bytes,
                )
            if timeout is None:
                extensions = self._default_timeout_extensions
            else:
                extensions = {"timeout": httpx.Timeout(timeout).as_dict()}
            try:
                url = self._url_cache.get(path)
                if url is None:
                    url = httpx.URL(f"{self._base_url_str}{path}")
                    if len(self._url_cache) < 256:
                        self._url_cache[path] = url
                # Clone the pre-tokenized Headers (fast list copy) and
                # add Content-Length per call — required because stream=
                # bypasses Request._prepare which would otherwise inject
                # it. Host is already baked into the cached Headers.
                req_headers = httpx.Headers(self._post_default_headers_obj)
                body_bytes = content_bytes if content_bytes is not None else b""
                req_headers["Content-Length"] = str(len(body_bytes))
                # stream= bypasses encode_request/_prepare/read inside
                # Request.__init__ — saves ~16 µs/call. We then restore
                # _content directly so request.content keeps working for
                # any caller that inspects the body.
                request = httpx.Request(
                    "POST",
                    url,
                    stream=httpx.ByteStream(body_bytes),
                    headers=req_headers,
                    extensions=extensions,
                )
                request._content = body_bytes
                # Use the live transport reference (tests may swap _client).
                response = self._client._transport.handle_request(request)
                response.request = request
                try:
                    response.read()
                except BaseException:
                    response.close()
                    raise
            except httpx.TransportError as exc:
                _raise_transport_error(exc)
            _raise_for_status(response)
            _release_response_refs(response)
            return response

        # Slow path: caller passed params=, files=, headers=, content=, etc.
        kwargs = _prepare_params(_prepare_json_kwargs(kwargs))
        body = kwargs.get("content") if isinstance(kwargs.get("content"), bytes) else None
        merged_headers = {**self._headers, **kwargs.get("headers", {})}
        _log_curl("POST", self._build_url(path), merged_headers, body=body)
        effective_timeout = timeout if timeout is not None else self._config.timeout
        try:
            request = self._client.build_request("POST", path, timeout=effective_timeout, **kwargs)
            response = self._client._transport.handle_request(request)
            response.request = request
            try:
                response.read()
            except BaseException:
                response.close()
                raise
        except httpx.TransportError as exc:
            _raise_transport_error(exc)
        _raise_for_status(response)
        _release_response_refs(response)
        return response

    def put(
        self, path: str, timeout: float | httpx.Timeout | None = None, **kwargs: Any
    ) -> httpx.Response:
        path = _prepare_path(path)
        kwargs = _prepare_json_kwargs(kwargs)
        body = kwargs.get("content") if isinstance(kwargs.get("content"), bytes) else None
        merged_headers = {**self._headers, **kwargs.get("headers", {})}
        _log_curl("PUT", self._build_url(path), merged_headers, body=body)
        effective_timeout = timeout if timeout is not None else self._config.timeout
        try:
            response = self._client.put(path, timeout=effective_timeout, **_prepare_params(kwargs))
        except httpx.TransportError as exc:
            _raise_transport_error(exc)
        _raise_for_status(response)
        _release_response_refs(response)
        return response

    def patch(
        self, path: str, timeout: float | httpx.Timeout | None = None, **kwargs: Any
    ) -> httpx.Response:
        path = _prepare_path(path)
        kwargs = _prepare_json_kwargs(kwargs)
        body = kwargs.get("content") if isinstance(kwargs.get("content"), bytes) else None
        merged_headers = {**self._headers, **kwargs.get("headers", {})}
        _log_curl("PATCH", self._build_url(path), merged_headers, body=body)
        effective_timeout = timeout if timeout is not None else self._config.timeout
        try:
            response = self._client.patch(
                path, timeout=effective_timeout, **_prepare_params(kwargs)
            )
        except httpx.TransportError as exc:
            _raise_transport_error(exc)
        _raise_for_status(response)
        _release_response_refs(response)
        return response

    def delete(
        self, path: str, timeout: float | httpx.Timeout | None = None, **kwargs: Any
    ) -> httpx.Response:
        path = _prepare_path(path)
        _log_curl("DELETE", self._build_url(path), dict(self._headers))
        effective_timeout = timeout if timeout is not None else self._config.timeout
        try:
            response = self._client.delete(
                path, timeout=effective_timeout, **_prepare_params(kwargs)
            )
        except httpx.TransportError as exc:
            _raise_transport_error(exc)
        _raise_for_status(response)
        _release_response_refs(response)
        return response

    @contextlib.contextmanager
    def stream(
        self,
        method: str,
        path: str,
        *,
        content: bytes | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> Generator[httpx.Response, None, None]:
        """Stream an HTTP response, wrapping transport errors as Pinecone exceptions.

        Opens a streaming request and yields the :class:`httpx.Response`.  If the
        server returns an error status, the response body is read and
        :func:`_raise_for_status` raises the appropriate exception before yielding.
        Transport-layer errors (timeouts, connection failures) raised either at
        connection time or during response iteration are caught and re-raised as
        :exc:`PineconeTimeoutError` or :exc:`PineconeConnectionError`.
        """
        path = _prepare_path(path)
        effective_timeout = timeout if timeout is not None else self._config.timeout
        try:
            with self._client.stream(
                method,
                path,
                content=content,
                headers=headers,
                timeout=effective_timeout,
            ) as response:
                if not response.is_success:
                    response.read()
                _raise_for_status(response)
                yield response
        except httpx.TransportError as exc:
            _raise_transport_error(exc)

    def close(self) -> None:
        self._client.close()


class AsyncHTTPClient:
    """Asynchronous HTTP client wrapping httpx.

    The underlying ``httpx.AsyncClient`` is created lazily on the first
    async method call rather than in ``__init__``.  This allows the
    client to be instantiated in a synchronous context (e.g. module
    scope) and used later inside an async event loop.
    """

    def __init__(self, config: PineconeConfig, api_version: str) -> None:
        self._config = config
        self._headers = _build_headers(config, api_version)
        self._client: httpx.AsyncClient | None = None

    def _ensure_client(self) -> httpx.AsyncClient:
        """Return the underlying client, creating it on first use."""
        if self._client is None:
            verify = _build_ssl_verify(self._config)
            pool_size = (
                self._config.connection_pool_maxsize
                if self._config.connection_pool_maxsize > 0
                else _default_pool_size()
            )
            limits = httpx.Limits(
                max_connections=pool_size,
                max_keepalive_connections=pool_size // 2,
            )
            transport = _AsyncRetryTransport(
                transport=httpx.AsyncHTTPTransport(
                    verify=verify,
                    http2=False,
                    limits=limits,
                    proxy=_build_proxy(self._config),
                    socket_options=_build_socket_options(),
                ),
                retry_config=self._config.retry_config,
            )
            self._client = httpx.AsyncClient(
                base_url=self._config.host or DEFAULT_BASE_URL,
                headers=self._headers,
                timeout=self._config.timeout,
                transport=transport,
            )
        return self._client

    async def get(
        self, path: str, timeout: float | httpx.Timeout | None = None, **kwargs: Any
    ) -> httpx.Response:
        path = _prepare_path(path)
        effective_timeout = timeout if timeout is not None else self._config.timeout
        try:
            response = await self._ensure_client().get(
                path, timeout=effective_timeout, **_prepare_params(kwargs)
            )
        except httpx.TransportError as exc:
            _raise_transport_error(exc)
        _raise_for_status(response)
        _release_response_refs(response)
        return response

    async def post(
        self, path: str, timeout: float | httpx.Timeout | None = None, **kwargs: Any
    ) -> httpx.Response:
        path = _prepare_path(path)
        effective_timeout = timeout if timeout is not None else self._config.timeout
        try:
            response = await self._ensure_client().post(
                path, timeout=effective_timeout, **_prepare_params(_prepare_json_kwargs(kwargs))
            )
        except httpx.TransportError as exc:
            _raise_transport_error(exc)
        _raise_for_status(response)
        _release_response_refs(response)
        return response

    async def put(
        self, path: str, timeout: float | httpx.Timeout | None = None, **kwargs: Any
    ) -> httpx.Response:
        path = _prepare_path(path)
        effective_timeout = timeout if timeout is not None else self._config.timeout
        try:
            response = await self._ensure_client().put(
                path, timeout=effective_timeout, **_prepare_params(_prepare_json_kwargs(kwargs))
            )
        except httpx.TransportError as exc:
            _raise_transport_error(exc)
        _raise_for_status(response)
        _release_response_refs(response)
        return response

    async def patch(
        self, path: str, timeout: float | httpx.Timeout | None = None, **kwargs: Any
    ) -> httpx.Response:
        path = _prepare_path(path)
        effective_timeout = timeout if timeout is not None else self._config.timeout
        try:
            response = await self._ensure_client().patch(
                path, timeout=effective_timeout, **_prepare_params(_prepare_json_kwargs(kwargs))
            )
        except httpx.TransportError as exc:
            _raise_transport_error(exc)
        _raise_for_status(response)
        _release_response_refs(response)
        return response

    async def delete(
        self, path: str, timeout: float | httpx.Timeout | None = None, **kwargs: Any
    ) -> httpx.Response:
        path = _prepare_path(path)
        effective_timeout = timeout if timeout is not None else self._config.timeout
        try:
            response = await self._ensure_client().delete(
                path, timeout=effective_timeout, **_prepare_params(kwargs)
            )
        except httpx.TransportError as exc:
            _raise_transport_error(exc)
        _raise_for_status(response)
        _release_response_refs(response)
        return response

    @contextlib.asynccontextmanager
    async def stream(
        self,
        method: str,
        path: str,
        *,
        content: bytes | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> AsyncGenerator[httpx.Response, None]:
        """Stream an async HTTP response, wrapping transport errors as Pinecone exceptions.

        Opens a streaming request and yields the :class:`httpx.Response`.  If the
        server returns an error status, the response body is read and
        :func:`_raise_for_status` raises the appropriate exception before yielding.
        Transport-layer errors (timeouts, connection failures) raised either at
        connection time or during response iteration are caught and re-raised as
        :exc:`PineconeTimeoutError` or :exc:`PineconeConnectionError`.
        """
        path = _prepare_path(path)
        effective_timeout = timeout if timeout is not None else self._config.timeout
        try:
            async with self._ensure_client().stream(
                method,
                path,
                content=content,
                headers=headers,
                timeout=effective_timeout,
            ) as response:
                if not response.is_success:
                    await response.aread()
                _raise_for_status(response)
                yield response
        except httpx.TransportError as exc:
            _raise_transport_error(exc)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
