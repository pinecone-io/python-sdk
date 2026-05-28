"""Tests for _RetryTransport retry behavior including connection errors."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from pinecone._internal.config import RetryConfig
from pinecone._internal.http_client import _AsyncRetryTransport, _RetryTransport


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
