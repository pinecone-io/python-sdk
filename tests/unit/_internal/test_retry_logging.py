"""Tests for throttle-event and AIMD-transition log lines."""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from pinecone._internal.adaptive import _AdaptiveLimiter, _AdaptiveLimiterRegistry
from pinecone._internal.config import RetryConfig
from pinecone._internal.http_client import _AsyncRetryTransport, _RetryTransport


def _transport(max_retries: int = 1) -> tuple[_RetryTransport, MagicMock]:
    inner = MagicMock(spec=httpx.BaseTransport)
    cfg = RetryConfig(max_retries=max_retries, backoff_factor=0.001, max_wait=0.01)
    return _RetryTransport(transport=inner, retry_config=cfg), inner  # type: ignore[arg-type]


def _async_transport(max_retries: int = 1) -> tuple[_AsyncRetryTransport, AsyncMock]:
    inner = AsyncMock(spec=httpx.AsyncBaseTransport)
    cfg = RetryConfig(max_retries=max_retries, backoff_factor=0.001, max_wait=0.01)
    return _AsyncRetryTransport(transport=inner, retry_config=cfg), inner  # type: ignore[arg-type]


def _req() -> httpx.Request:
    return httpx.Request("POST", "https://example.com/test")


def test_throttle_response_logs_debug_with_fields(caplog: pytest.LogCaptureFixture) -> None:
    rt, inner = _transport(max_retries=1)
    inner.handle_request.side_effect = [
        httpx.Response(429),
        httpx.Response(200),
    ]
    with (
        caplog.at_level(logging.DEBUG, logger="pinecone._internal.http_client"),
        patch("pinecone._internal.http_client.time.sleep"),
    ):
        rt.handle_request(_req())

    throttle_records = [r for r in caplog.records if "Throttled response" in r.getMessage()]
    assert len(throttle_records) == 1
    msg = throttle_records[0].getMessage()
    assert throttle_records[0].levelname == "DEBUG"
    assert "status=429" in msg
    assert "host=example.com" in msg


def test_aimd_decrease_logs_debug(caplog: pytest.LogCaptureFixture) -> None:
    lim = _AdaptiveLimiter(ceiling=8)
    with caplog.at_level(logging.DEBUG, logger="pinecone._internal.adaptive"):
        lim.report_throttled()

    records = [r for r in caplog.records if "AIMD limiter decreased" in r.getMessage()]
    assert len(records) == 1
    assert "before=8 after=4" in records[0].getMessage()


def test_aimd_increase_logs_debug_only_on_transition(caplog: pytest.LogCaptureFixture) -> None:
    lim = _AdaptiveLimiter(ceiling=8)
    lim.report_throttled()  # 8 → 4; limit is now 4, streak is 0
    # Need exactly 4 successful calls to cross the threshold (streak reaches limit)
    with caplog.at_level(logging.DEBUG, logger="pinecone._internal.adaptive"):
        for _ in range(4):
            lim.report_success()

    increase_records = [r for r in caplog.records if "AIMD limiter increased" in r.getMessage()]
    assert len(increase_records) == 1


def test_aimd_no_log_when_at_ceiling(caplog: pytest.LogCaptureFixture) -> None:
    lim = _AdaptiveLimiter(ceiling=8)
    # No throttle; limit starts at ceiling (8)
    with caplog.at_level(logging.DEBUG, logger="pinecone._internal.adaptive"):
        for _ in range(100):
            lim.report_success()

    increase_records = [r for r in caplog.records if "AIMD limiter increased" in r.getMessage()]
    assert len(increase_records) == 0


def test_first_throttle_per_host_logs_info(caplog: pytest.LogCaptureFixture) -> None:
    reg = _AdaptiveLimiterRegistry()
    reg.get("api-1.pinecone.io", 8)

    with caplog.at_level(logging.INFO, logger="pinecone._internal.adaptive"):
        reg.report_throttled("api-1.pinecone.io")
        reg.report_throttled("api-1.pinecone.io")

    info_records = [
        r
        for r in caplog.records
        if r.levelname == "INFO" and "Rate limited by host=api-1.pinecone.io" in r.getMessage()
    ]
    assert len(info_records) == 1


@pytest.mark.asyncio
async def test_async_throttle_response_logs_debug(caplog: pytest.LogCaptureFixture) -> None:
    rt, inner = _async_transport(max_retries=1)
    inner.handle_async_request.side_effect = [
        httpx.Response(429),
        httpx.Response(200),
    ]
    with (
        caplog.at_level(logging.DEBUG, logger="pinecone._internal.http_client"),
        patch("pinecone._internal.http_client.asyncio.sleep"),
    ):
        await rt.handle_async_request(_req())

    throttle_records = [r for r in caplog.records if "Throttled response" in r.getMessage()]
    assert len(throttle_records) == 1
    msg = throttle_records[0].getMessage()
    assert throttle_records[0].levelname == "DEBUG"
    assert "status=429" in msg
    assert "host=example.com" in msg
