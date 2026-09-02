"""Self-tests for the storm fixture in _storm_fixture.py.

The fixture module's leading underscore keeps pytest's default
``python_files`` patterns from collecting it, so these lived there unrun.
They stay out of the fixture module for that reason: the four consumers
import it by name, and a helper that is also a test module invites both
double collection and import-order surprises.

What they pin is the fixture itself, not the retry transports: that
``_FaultInjectionTransport`` really returns 429s inside the throttle window
and 200s after it, that recorded timestamps disperse, and that
``request_amplification`` sees the retries. The transports' own behavior is
pinned by test_retry_storm_sync.py and test_retry_storm_async.py.
"""

from __future__ import annotations

import httpx
import pytest

from pinecone._internal.config import RetryConfig
from pinecone._internal.http_client import _AsyncRetryTransport, _RetryTransport
from tests.unit._internal._storm_fixture import StormConfig, StormScenario


@pytest.fixture
def suppress_retry_sleep() -> bool:
    """Keep real retry backoff: these tests need retries to actually advance
    the clock past the throttle window. See tests/unit/conftest.py."""
    return False


def _make_sync_retry_transport(
    scenario: StormScenario,
    max_retries: int = 5,
    backoff_factor: float = 0.01,
    max_wait: float = 0.5,
) -> _RetryTransport:
    cfg = RetryConfig(max_retries=max_retries, backoff_factor=backoff_factor, max_wait=max_wait)
    return _RetryTransport(transport=scenario.sync_transport, retry_config=cfg)  # type: ignore[arg-type]


def _make_async_retry_transport(
    scenario: StormScenario,
    max_retries: int = 5,
    backoff_factor: float = 0.01,
    max_wait: float = 0.5,
) -> _AsyncRetryTransport:
    cfg = RetryConfig(max_retries=max_retries, backoff_factor=backoff_factor, max_wait=max_wait)
    return _AsyncRetryTransport(transport=scenario.async_transport, retry_config=cfg)  # type: ignore[arg-type]


def test_storm_fixture_records_429s_during_window() -> None:
    config = StormConfig(n_clients=5, throttle_window_seconds=0.1, retry_after_seconds=0.04)
    scenario = StormScenario(config)
    rt = _make_sync_retry_transport(scenario)
    req = httpx.Request("GET", "https://example.com/test")

    scenario.run_sync(lambda: rt.handle_request(req))

    records = scenario.sync_transport.records
    outcomes = {r.outcome for r in records}
    assert "429" in outcomes, "expected some 429s during the throttle window"
    assert "200" in outcomes, "expected 200s after retries succeed"


def test_storm_fixture_dispersion_width_nonzero() -> None:
    config = StormConfig(n_clients=5, throttle_window_seconds=0.1, retry_after_seconds=0.04)
    scenario = StormScenario(config)
    rt = _make_sync_retry_transport(scenario)
    req = httpx.Request("GET", "https://example.com/test")

    scenario.run_sync(lambda: rt.handle_request(req))

    assert scenario.dispersion_width() > 0, "clients should not all arrive at the exact same moment"


def test_storm_fixture_request_amplification_above_one_under_throttle() -> None:
    config = StormConfig(n_clients=5, throttle_window_seconds=0.1, retry_after_seconds=0.04)
    scenario = StormScenario(config)
    rt = _make_sync_retry_transport(scenario)
    req = httpx.Request("GET", "https://example.com/test")

    scenario.run_sync(lambda: rt.handle_request(req))

    assert scenario.request_amplification() > 1.0, "retries should have fired"


@pytest.mark.asyncio
async def test_storm_fixture_async_records_429s_during_window() -> None:
    config = StormConfig(n_clients=5, throttle_window_seconds=0.1, retry_after_seconds=0.04)
    scenario = StormScenario(config)
    rt = _make_async_retry_transport(scenario)
    req = httpx.Request("GET", "https://example.com/test")

    await scenario.run_async(lambda: rt.handle_async_request(req))

    records = scenario.async_transport.records
    outcomes = {r.outcome for r in records}
    assert "429" in outcomes, "expected some 429s during the throttle window"
    assert "200" in outcomes, "expected 200s after retries succeed"


@pytest.mark.asyncio
async def test_storm_fixture_async_dispersion_width_nonzero() -> None:
    config = StormConfig(n_clients=5, throttle_window_seconds=0.1, retry_after_seconds=0.04)
    scenario = StormScenario(config)
    rt = _make_async_retry_transport(scenario)
    req = httpx.Request("GET", "https://example.com/test")

    await scenario.run_async(lambda: rt.handle_async_request(req))

    assert scenario.dispersion_width() > 0, "clients should not all arrive at the exact same moment"


@pytest.mark.asyncio
async def test_storm_fixture_async_request_amplification_above_one_under_throttle() -> None:
    config = StormConfig(n_clients=5, throttle_window_seconds=0.1, retry_after_seconds=0.04)
    scenario = StormScenario(config)
    rt = _make_async_retry_transport(scenario)
    req = httpx.Request("GET", "https://example.com/test")

    await scenario.run_async(lambda: rt.handle_async_request(req))

    assert scenario.request_amplification() > 1.0, "retries should have fired"
