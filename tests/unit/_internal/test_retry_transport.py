"""Tests for _RetryTransport retry behavior including connection errors."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from pinecone._internal.config import RetryConfig
from pinecone._internal.http_client import (
    _AsyncRetryTransport,
    _compute_backoff,
    _compute_retry_after_delay,
    _notify_throttle,
    _RetryTransport,
)


def _transport(max_retries: int = 3) -> tuple[_RetryTransport, MagicMock]:
    inner = MagicMock(spec=httpx.BaseTransport)
    cfg = RetryConfig(max_retries=max_retries, backoff_factor=0.001, max_wait=0.01)
    return _RetryTransport(transport=inner, retry_config=cfg), inner  # type: ignore[arg-type]


def _req() -> httpx.Request:
    return httpx.Request("POST", "https://example.com/test")


def test_connection_error_is_retried_and_succeeds() -> None:
    rt, inner = _transport(max_retries=3)
    inner.handle_request.side_effect = [
        httpx.RemoteProtocolError("peer closed connection"),
        httpx.Response(200),
    ]
    result = rt.handle_request(_req())
    assert result.status_code == 200
    assert inner.handle_request.call_count == 2


def test_connection_error_exhausts_retries_and_raises() -> None:
    # max_retries=1 → 2 total attempts (initial + 1 retry)
    rt, inner = _transport(max_retries=1)
    inner.handle_request.side_effect = [
        httpx.RemoteProtocolError("peer closed connection"),
        httpx.RemoteProtocolError("peer closed connection"),
    ]
    with pytest.raises(httpx.RemoteProtocolError):
        rt.handle_request(_req())


def test_retryable_status_code_still_retried() -> None:
    rt, inner = _transport(max_retries=3)
    inner.handle_request.side_effect = [httpx.Response(503), httpx.Response(200)]
    result = rt.handle_request(_req())
    assert result.status_code == 200


def test_non_retryable_status_returns_immediately() -> None:
    rt, inner = _transport(max_retries=3)
    inner.handle_request.return_value = httpx.Response(400)
    result = rt.handle_request(_req())
    assert result.status_code == 400
    assert inner.handle_request.call_count == 1


# --- async variants ---


def _async_transport(max_retries: int = 3) -> tuple[_AsyncRetryTransport, AsyncMock]:
    inner = AsyncMock(spec=httpx.AsyncBaseTransport)
    cfg = RetryConfig(max_retries=max_retries, backoff_factor=0.001, max_wait=0.01)
    return _AsyncRetryTransport(transport=inner, retry_config=cfg), inner  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_async_connection_error_is_retried_and_succeeds() -> None:
    rt, inner = _async_transport(max_retries=3)
    inner.handle_async_request.side_effect = [
        httpx.RemoteProtocolError("peer closed connection"),
        httpx.Response(200),
    ]
    result = await rt.handle_async_request(_req())
    assert result.status_code == 200
    assert inner.handle_async_request.call_count == 2


@pytest.mark.asyncio
async def test_async_connection_error_exhausts_retries_and_raises() -> None:
    # max_retries=1 → 2 total attempts (initial + 1 retry)
    rt, inner = _async_transport(max_retries=1)
    inner.handle_async_request.side_effect = [
        httpx.RemoteProtocolError("peer closed connection"),
        httpx.RemoteProtocolError("peer closed connection"),
    ]
    with pytest.raises(httpx.RemoteProtocolError):
        await rt.handle_async_request(_req())


@pytest.mark.asyncio
async def test_async_retryable_status_code_still_retried() -> None:
    rt, inner = _async_transport(max_retries=3)
    inner.handle_async_request.side_effect = [httpx.Response(503), httpx.Response(200)]
    result = await rt.handle_async_request(_req())
    assert result.status_code == 200


@pytest.mark.asyncio
async def test_async_non_retryable_status_returns_immediately() -> None:
    rt, inner = _async_transport(max_retries=3)
    inner.handle_async_request.return_value = httpx.Response(400)
    result = await rt.handle_async_request(_req())
    assert result.status_code == 400
    assert inner.handle_async_request.call_count == 1


# --- POST-specific named tests (sync) ---


def test_post_upsert_retried_on_transport_error() -> None:
    rt, inner = _transport(max_retries=3)
    inner.handle_request.side_effect = [
        httpx.ConnectError("boom"),
        httpx.Response(200),
    ]
    result = rt.handle_request(httpx.Request("POST", "https://example.com/vectors/upsert"))
    assert result.status_code == 200
    assert inner.handle_request.call_count == 2


def test_post_query_retried_on_503() -> None:
    rt, inner = _transport(max_retries=3)
    inner.handle_request.side_effect = [httpx.Response(503), httpx.Response(200)]
    result = rt.handle_request(httpx.Request("POST", "https://example.com/query"))
    assert result.status_code == 200
    assert inner.handle_request.call_count == 2


def test_post_retried_on_408() -> None:
    rt, inner = _transport(max_retries=3)
    inner.handle_request.side_effect = [httpx.Response(408), httpx.Response(200)]
    result = rt.handle_request(httpx.Request("POST", "https://example.com/query"))
    assert result.status_code == 200
    assert inner.handle_request.call_count == 2


def test_post_retried_on_429_with_retry_after() -> None:
    rt, inner = _transport(max_retries=3)
    inner.handle_request.side_effect = [
        httpx.Response(429, headers={"Retry-After": "0"}),
        httpx.Response(200),
    ]
    with patch("pinecone._internal.http_client.time.sleep") as mock_sleep:
        result = rt.handle_request(httpx.Request("POST", "https://example.com/query"))
    assert result.status_code == 200
    assert inner.handle_request.call_count == 2
    mock_sleep.assert_called_once_with(0.0)


def test_post_not_retried_on_400() -> None:
    rt, inner = _transport(max_retries=3)
    inner.handle_request.return_value = httpx.Response(400)
    result = rt.handle_request(httpx.Request("POST", "https://example.com/query"))
    assert result.status_code == 400
    assert inner.handle_request.call_count == 1


def test_post_exhausts_retries_then_returns_503() -> None:
    # max_retries=2 → 3 total attempts (initial + 2 retries)
    rt, inner = _transport(max_retries=2)
    inner.handle_request.side_effect = [
        httpx.Response(503),
        httpx.Response(503),
        httpx.Response(503),
    ]
    result = rt.handle_request(httpx.Request("POST", "https://example.com/query"))
    assert result.status_code == 503
    assert inner.handle_request.call_count == 3


def test_post_exhausts_retries_then_raises_transport_error() -> None:
    # max_retries=2 → 3 total attempts (initial + 2 retries)
    rt, inner = _transport(max_retries=2)
    inner.handle_request.side_effect = [
        httpx.ConnectError("fail"),
        httpx.ConnectError("fail"),
        httpx.ConnectError("fail"),
    ]
    with pytest.raises(httpx.ConnectError):
        rt.handle_request(httpx.Request("POST", "https://example.com/query"))
    assert inner.handle_request.call_count == 3


# --- POST-specific named tests (async) ---


@pytest.mark.asyncio
async def test_async_post_upsert_retried_on_transport_error() -> None:
    rt, inner = _async_transport(max_retries=3)
    inner.handle_async_request.side_effect = [
        httpx.ConnectError("boom"),
        httpx.Response(200),
    ]
    result = await rt.handle_async_request(
        httpx.Request("POST", "https://example.com/vectors/upsert")
    )
    assert result.status_code == 200
    assert inner.handle_async_request.call_count == 2


@pytest.mark.asyncio
async def test_async_post_query_retried_on_503() -> None:
    rt, inner = _async_transport(max_retries=3)
    inner.handle_async_request.side_effect = [httpx.Response(503), httpx.Response(200)]
    result = await rt.handle_async_request(httpx.Request("POST", "https://example.com/query"))
    assert result.status_code == 200
    assert inner.handle_async_request.call_count == 2


@pytest.mark.asyncio
async def test_async_post_retried_on_408() -> None:
    rt, inner = _async_transport(max_retries=3)
    inner.handle_async_request.side_effect = [httpx.Response(408), httpx.Response(200)]
    result = await rt.handle_async_request(httpx.Request("POST", "https://example.com/query"))
    assert result.status_code == 200
    assert inner.handle_async_request.call_count == 2


# --- module-level helper function tests ---


def _cfg(backoff_factor: float = 0.5, max_wait: float = 60.0) -> RetryConfig:
    return RetryConfig(backoff_factor=backoff_factor, max_wait=max_wait)


def test_compute_backoff_first_attempt_stays_at_base() -> None:
    cfg = _cfg(backoff_factor=0.5, max_wait=60.0)
    result = _compute_backoff(cfg, 0, None)
    assert 0.5 <= result <= 1.5  # uniform(base, max(base, min(max_wait, base*3)))


def test_compute_backoff_grows_with_prev_delay() -> None:
    cfg = _cfg(backoff_factor=0.5, max_wait=60.0)
    result = _compute_backoff(cfg, 1, 2.0)
    # upper = min(60, 2.0 * 3) = 6.0; result in [0.5, 6.0]
    assert 0.5 <= result <= 6.0


def test_compute_backoff_capped_at_max_wait() -> None:
    cfg = _cfg(backoff_factor=1.0, max_wait=2.0)
    result = _compute_backoff(cfg, 5, 100.0)
    # upper = min(2.0, 300.0) = 2.0; result in [1.0, 2.0]
    assert 1.0 <= result <= 2.0


def test_compute_retry_after_uses_header_value() -> None:
    cfg = _cfg()
    response = httpx.Response(429, headers={"Retry-After": "5"})
    result = _compute_retry_after_delay(cfg, response, 0, None)
    # Should be between 5 (no smear) and 7.5 (5 + 5*0.5)
    assert 5.0 <= result <= 7.5


def test_compute_retry_after_ignores_invalid_header() -> None:
    cfg = _cfg(backoff_factor=0.5, max_wait=60.0)
    response = httpx.Response(429, headers={"Retry-After": "not-a-number"})
    result = _compute_retry_after_delay(cfg, response, 0, None)
    # Falls back to _compute_backoff
    assert result >= 0.5


def test_compute_retry_after_no_header_falls_back_to_backoff() -> None:
    cfg = _cfg(backoff_factor=0.5, max_wait=60.0)
    response = httpx.Response(503)
    result = _compute_retry_after_delay(cfg, response, 0, None)
    assert result >= 0.5


def test_notify_throttle_calls_callback() -> None:
    calls: list[str] = []
    cfg = RetryConfig(on_throttle=lambda host: calls.append(host))
    request = httpx.Request("POST", "https://example.com/vectors/upsert")
    _notify_throttle(cfg, request)
    assert calls == ["example.com"]


def test_notify_throttle_no_callback_is_noop() -> None:
    cfg = RetryConfig(on_throttle=None)
    request = httpx.Request("POST", "https://example.com/query")
    _notify_throttle(cfg, request)  # should not raise


def test_notify_throttle_swallows_callback_exception() -> None:
    def bad_cb(host: str) -> None:
        raise RuntimeError("oops")

    cfg = RetryConfig(on_throttle=bad_cb)
    request = httpx.Request("POST", "https://example.com/query")
    _notify_throttle(cfg, request)  # should not raise


def test_sync_transport_calls_module_level_compute_backoff() -> None:
    """Verify sync transport uses module-level _compute_backoff (not an instance method)."""
    rt, inner = _transport(max_retries=1)
    inner.handle_request.side_effect = [
        httpx.ConnectError("boom"),
        httpx.Response(200),
    ]
    with patch("pinecone._internal.http_client._compute_backoff", return_value=0.001) as mock_cb:
        rt.handle_request(_req())
    mock_cb.assert_called_once()
    args = mock_cb.call_args[0]
    assert isinstance(args[0], RetryConfig)


def test_async_transport_calls_module_level_compute_backoff() -> None:
    """Verify async transport uses module-level _compute_backoff (not an instance method)."""
    import asyncio

    rt, inner = _async_transport(max_retries=1)
    inner.handle_async_request.side_effect = [
        httpx.ConnectError("boom"),
        httpx.Response(200),
    ]

    with patch("pinecone._internal.http_client._compute_backoff", return_value=0.001) as mock_cb:
        asyncio.get_event_loop().run_until_complete(rt.handle_async_request(_req()))
    mock_cb.assert_called_once()
    args = mock_cb.call_args[0]
    assert isinstance(args[0], RetryConfig)
