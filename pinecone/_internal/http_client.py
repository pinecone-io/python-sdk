"""httpx-based HTTP client for sync and async operations."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import random
import socket
import sys
import time
from collections.abc import AsyncGenerator, Generator
from typing import Any
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
    ForbiddenError,
    NotFoundError,
    PineconeConnectionError,
    PineconeTimeoutError,
    RateLimitError,
    ServiceError,
    UnauthorizedError,
)

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


def _default_pool_size() -> int:
    """Return the default connection pool size: 5x CPU count with a floor of 20."""
    return max(5 * (os.cpu_count() or 1), 20)


def _encode_json(body: Any) -> bytes:
    """Serialize *body* to JSON bytes using orjson (2-3x faster than stdlib json)."""
    return orjson.dumps(body)


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


def _notify_throttle(config: RetryConfig, request: httpx.Request) -> None:
    cb = config.on_throttle
    if cb is None:
        return
    try:
        cb(request.url.host)
    except Exception as exc:
        logger.debug("on_throttle callback raised, ignoring: %s", exc)


class _RetryTransport(httpx.BaseTransport):
    """Sync transport wrapper that retries on transient server errors."""

    def __init__(
        self,
        *,
        transport: httpx.HTTPTransport,
        retry_config: RetryConfig | None = None,
    ) -> None:
        self._transport = transport
        self._config = retry_config or RetryConfig()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        last_exc: httpx.TransportError | None = None
        prev_delay: float | None = None
        for attempt in range(self._config.max_retries + 1):
            try:
                response = self._transport.handle_request(request)
            except httpx.TransportError as exc:
                last_exc = exc
                if attempt < self._config.max_retries:
                    logger.debug(
                        "Connection error on attempt %d/%d, retrying: %s",
                        attempt + 1,
                        self._config.max_retries + 1,
                        exc,
                    )
                    delay = _compute_backoff(self._config, attempt, prev_delay)
                    prev_delay = delay
                    time.sleep(delay)
                continue
            last_exc = None
            if response.status_code not in self._config.retryable_status_codes:
                return response
            _notify_throttle(self._config, request)
            if attempt < self._config.max_retries:
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
                time.sleep(delay)
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
    ) -> None:
        self._transport = transport
        self._config = retry_config or RetryConfig()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        last_exc: httpx.TransportError | None = None
        prev_delay: float | None = None
        for attempt in range(self._config.max_retries + 1):
            try:
                response = await self._transport.handle_async_request(request)
            except httpx.TransportError as exc:
                last_exc = exc
                if attempt < self._config.max_retries:
                    logger.debug(
                        "Connection error on attempt %d/%d, retrying: %s",
                        attempt + 1,
                        self._config.max_retries + 1,
                        exc,
                    )
                    delay = _compute_backoff(self._config, attempt, prev_delay)
                    prev_delay = delay
                    await asyncio.sleep(delay)
                continue
            last_exc = None
            if response.status_code not in self._config.retryable_status_codes:
                return response
            _notify_throttle(self._config, request)
            if attempt < self._config.max_retries:
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
                await asyncio.sleep(delay)
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


class HTTPClient:
    """Synchronous HTTP client wrapping httpx."""

    def __init__(self, config: PineconeConfig, api_version: str) -> None:
        self._config = config
        self._headers = _build_headers(config, api_version)
        verify: str | bool = config.ssl_ca_certs or config.ssl_verify
        pool_size = (
            config.connection_pool_maxsize
            if config.connection_pool_maxsize > 0
            else _default_pool_size()
        )
        limits = httpx.Limits(
            max_connections=pool_size,
            max_keepalive_connections=pool_size // 2,
        )
        transport = _RetryTransport(
            transport=httpx.HTTPTransport(
                http2=False, limits=limits, socket_options=_build_socket_options()
            ),
            retry_config=config.retry_config,
        )
        proxy: httpx.Proxy | str | None = None
        if config.proxy_url:
            if config.proxy_headers:
                proxy = httpx.Proxy(url=config.proxy_url, headers=config.proxy_headers)
            else:
                proxy = config.proxy_url
        self._client = httpx.Client(
            base_url=config.host or DEFAULT_BASE_URL,
            headers=self._headers,
            timeout=config.timeout,
            transport=transport,
            proxy=proxy,
            verify=verify,
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
        _log_curl("GET", self._build_url(path), dict(self._headers))
        effective_timeout = timeout if timeout is not None else self._config.timeout
        try:
            response = self._client.get(path, timeout=effective_timeout, **kwargs)
        except httpx.TimeoutException as exc:
            raise PineconeTimeoutError(str(exc)) from exc
        except httpx.TransportError as exc:
            raise PineconeConnectionError(str(exc)) from exc
        _raise_for_status(response)
        _release_response_refs(response)
        return response

    def post(
        self, path: str, timeout: float | httpx.Timeout | None = None, **kwargs: Any
    ) -> httpx.Response:
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
            except httpx.TimeoutException as exc:
                raise PineconeTimeoutError(str(exc)) from exc
            except httpx.TransportError as exc:
                raise PineconeConnectionError(str(exc)) from exc
            _raise_for_status(response)
            _release_response_refs(response)
            return response

        # Slow path: caller passed params=, files=, headers=, content=, etc.
        kwargs = _prepare_json_kwargs(kwargs)
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
        except httpx.TimeoutException as exc:
            raise PineconeTimeoutError(str(exc)) from exc
        except httpx.TransportError as exc:
            raise PineconeConnectionError(str(exc)) from exc
        _raise_for_status(response)
        _release_response_refs(response)
        return response

    def put(
        self, path: str, timeout: float | httpx.Timeout | None = None, **kwargs: Any
    ) -> httpx.Response:
        kwargs = _prepare_json_kwargs(kwargs)
        body = kwargs.get("content") if isinstance(kwargs.get("content"), bytes) else None
        merged_headers = {**self._headers, **kwargs.get("headers", {})}
        _log_curl("PUT", self._build_url(path), merged_headers, body=body)
        effective_timeout = timeout if timeout is not None else self._config.timeout
        try:
            response = self._client.put(path, timeout=effective_timeout, **kwargs)
        except httpx.TimeoutException as exc:
            raise PineconeTimeoutError(str(exc)) from exc
        except httpx.TransportError as exc:
            raise PineconeConnectionError(str(exc)) from exc
        _raise_for_status(response)
        _release_response_refs(response)
        return response

    def patch(
        self, path: str, timeout: float | httpx.Timeout | None = None, **kwargs: Any
    ) -> httpx.Response:
        kwargs = _prepare_json_kwargs(kwargs)
        body = kwargs.get("content") if isinstance(kwargs.get("content"), bytes) else None
        merged_headers = {**self._headers, **kwargs.get("headers", {})}
        _log_curl("PATCH", self._build_url(path), merged_headers, body=body)
        effective_timeout = timeout if timeout is not None else self._config.timeout
        try:
            response = self._client.patch(path, timeout=effective_timeout, **kwargs)
        except httpx.TimeoutException as exc:
            raise PineconeTimeoutError(str(exc)) from exc
        except httpx.TransportError as exc:
            raise PineconeConnectionError(str(exc)) from exc
        _raise_for_status(response)
        _release_response_refs(response)
        return response

    def delete(
        self, path: str, timeout: float | httpx.Timeout | None = None, **kwargs: Any
    ) -> httpx.Response:
        _log_curl("DELETE", self._build_url(path), dict(self._headers))
        effective_timeout = timeout if timeout is not None else self._config.timeout
        try:
            response = self._client.delete(path, timeout=effective_timeout, **kwargs)
        except httpx.TimeoutException as exc:
            raise PineconeTimeoutError(str(exc)) from exc
        except httpx.TransportError as exc:
            raise PineconeConnectionError(str(exc)) from exc
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
        except httpx.TimeoutException as exc:
            raise PineconeTimeoutError(str(exc)) from exc
        except httpx.TransportError as exc:
            raise PineconeConnectionError(str(exc)) from exc

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
            verify: str | bool = self._config.ssl_ca_certs or self._config.ssl_verify
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
                    http2=False, limits=limits, socket_options=_build_socket_options()
                ),
                retry_config=self._config.retry_config,
            )
            proxy: httpx.Proxy | str | None = None
            if self._config.proxy_url:
                if self._config.proxy_headers:
                    proxy = httpx.Proxy(
                        url=self._config.proxy_url, headers=self._config.proxy_headers
                    )
                else:
                    proxy = self._config.proxy_url
            self._client = httpx.AsyncClient(
                base_url=self._config.host or DEFAULT_BASE_URL,
                headers=self._headers,
                timeout=self._config.timeout,
                transport=transport,
                proxy=proxy,
                verify=verify,
            )
        return self._client

    async def get(
        self, path: str, timeout: float | httpx.Timeout | None = None, **kwargs: Any
    ) -> httpx.Response:
        effective_timeout = timeout if timeout is not None else self._config.timeout
        try:
            response = await self._ensure_client().get(path, timeout=effective_timeout, **kwargs)
        except httpx.TimeoutException as exc:
            raise PineconeTimeoutError(str(exc)) from exc
        except httpx.TransportError as exc:
            raise PineconeConnectionError(str(exc)) from exc
        _raise_for_status(response)
        _release_response_refs(response)
        return response

    async def post(
        self, path: str, timeout: float | httpx.Timeout | None = None, **kwargs: Any
    ) -> httpx.Response:
        effective_timeout = timeout if timeout is not None else self._config.timeout
        try:
            response = await self._ensure_client().post(
                path, timeout=effective_timeout, **_prepare_json_kwargs(kwargs)
            )
        except httpx.TimeoutException as exc:
            raise PineconeTimeoutError(str(exc)) from exc
        except httpx.TransportError as exc:
            raise PineconeConnectionError(str(exc)) from exc
        _raise_for_status(response)
        _release_response_refs(response)
        return response

    async def put(
        self, path: str, timeout: float | httpx.Timeout | None = None, **kwargs: Any
    ) -> httpx.Response:
        effective_timeout = timeout if timeout is not None else self._config.timeout
        try:
            response = await self._ensure_client().put(
                path, timeout=effective_timeout, **_prepare_json_kwargs(kwargs)
            )
        except httpx.TimeoutException as exc:
            raise PineconeTimeoutError(str(exc)) from exc
        except httpx.TransportError as exc:
            raise PineconeConnectionError(str(exc)) from exc
        _raise_for_status(response)
        _release_response_refs(response)
        return response

    async def patch(
        self, path: str, timeout: float | httpx.Timeout | None = None, **kwargs: Any
    ) -> httpx.Response:
        effective_timeout = timeout if timeout is not None else self._config.timeout
        try:
            response = await self._ensure_client().patch(
                path, timeout=effective_timeout, **_prepare_json_kwargs(kwargs)
            )
        except httpx.TimeoutException as exc:
            raise PineconeTimeoutError(str(exc)) from exc
        except httpx.TransportError as exc:
            raise PineconeConnectionError(str(exc)) from exc
        _raise_for_status(response)
        _release_response_refs(response)
        return response

    async def delete(
        self, path: str, timeout: float | httpx.Timeout | None = None, **kwargs: Any
    ) -> httpx.Response:
        effective_timeout = timeout if timeout is not None else self._config.timeout
        try:
            response = await self._ensure_client().delete(path, timeout=effective_timeout, **kwargs)
        except httpx.TimeoutException as exc:
            raise PineconeTimeoutError(str(exc)) from exc
        except httpx.TransportError as exc:
            raise PineconeConnectionError(str(exc)) from exc
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
        except httpx.TimeoutException as exc:
            raise PineconeTimeoutError(str(exc)) from exc
        except httpx.TransportError as exc:
            raise PineconeConnectionError(str(exc)) from exc

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
