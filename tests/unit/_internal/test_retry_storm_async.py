"""Storm simulation tests for REST async transport (thundering-herd dispersal).

Verifies that N concurrent async clients receiving identical Retry-After: 0.5 headers
disperse in time rather than re-colliding, and that request amplification stays bounded.
"""

from __future__ import annotations

import json
import pathlib

import httpx
import pytest

from pinecone._internal.config import RetryConfig
from pinecone._internal.http_client import _AsyncRetryTransport
from tests.unit._internal._storm_fixture import (
    StormConfig,
    StormScenario,
    _AsyncFaultInjectionTransport,
)

pytestmark = pytest.mark.asyncio

_PARITY_METRICS_PATH = pathlib.Path(__file__).parent / "_storm_parity_metrics_async.json"


# Override the unit-test conftest's autouse sleep-suppressor. Storm tests
# need real asyncio.sleep so retries actually advance the clock past the
# throttle window. Defining the same-named fixture here takes precedence
# over the conftest.py version for tests in this module.
@pytest.fixture(autouse=True)
def _no_retry_sleep() -> None:
    pass


def _make_async_transport(
    scenario: StormScenario,
    max_retries: int = 5,
    backoff_factor: float = 0.01,
    max_wait: float = 2.0,
) -> _AsyncRetryTransport:
    cfg = RetryConfig(max_retries=max_retries, backoff_factor=backoff_factor, max_wait=max_wait)
    return _AsyncRetryTransport(transport=scenario.async_transport, retry_config=cfg)  # type: ignore[arg-type]


async def test_async_thundering_herd_disperses_with_retry_after_smear() -> None:
    config = StormConfig(
        n_clients=50,
        throttle_window_seconds=1.0,
        retry_after_seconds=0.5,
        seed=0xC0FFEE,
    )
    scenario = StormScenario(config)
    transport = _make_async_transport(scenario)

    async def make_request() -> httpx.Response:
        return await transport.handle_async_request(
            httpx.Request("POST", "https://api.example.com/upsert")
        )

    await scenario.run_async(make_request)
    assert not scenario.exceptions, f"unexpected exceptions: {scenario.exceptions}"

    width = scenario.dispersion_width(only_successes=True)
    assert 0.10 <= width <= 0.70, f"async dispersion width {width:.3f}s outside [0.10, 0.70]"

    first = scenario.first_success_after_window()
    start = scenario.async_transport.start_time
    first_relative: float | None = (first - start) if first is not None else None
    assert first_relative is not None, "no successful responses observed after throttle window"
    lower = config.throttle_window_seconds
    upper = config.throttle_window_seconds + config.retry_after_seconds * 1.5 + 0.1
    assert lower <= first_relative <= upper, (
        f"first success at {first_relative:.3f}s, expected in [{lower:.2f}, {upper:.2f}]"
    )


async def test_async_request_amplification_bounded_under_throttle() -> None:
    config = StormConfig(
        n_clients=50,
        throttle_window_seconds=1.0,
        retry_after_seconds=0.5,
        seed=0xC0FFEE,
    )
    scenario = StormScenario(config)
    transport = _make_async_transport(scenario)

    await scenario.run_async(
        lambda: transport.handle_async_request(
            httpx.Request("POST", "https://api.example.com/upsert")
        )
    )

    amp = scenario.request_amplification()
    assert 1.5 <= amp <= 3.0, f"amplification {amp:.3f} outside [1.5, 3.0]"


async def test_async_n_200_clients_still_disperses() -> None:
    """Identical bounds hold for N=200 — catches N-dependent regressions in the jitter math."""
    config = StormConfig(
        n_clients=200,
        throttle_window_seconds=1.0,
        retry_after_seconds=0.5,
        seed=0xC0FFEE,
    )
    scenario = StormScenario(config)
    transport = _make_async_transport(scenario)

    await scenario.run_async(
        lambda: transport.handle_async_request(
            httpx.Request("POST", "https://api.example.com/upsert")
        )
    )
    assert not scenario.exceptions, f"unexpected exceptions: {scenario.exceptions}"

    width = scenario.dispersion_width(only_successes=True)
    assert 0.10 <= width <= 0.70, f"dispersion width {width:.3f}s outside [0.10, 0.70]"


async def test_async_no_throttle_no_amplification() -> None:
    config = StormConfig(
        n_clients=50,
        throttle_window_seconds=0.0,
        retry_after_seconds=0.5,
        seed=0xC0FFEE,
    )
    scenario = StormScenario(config)
    transport = _make_async_transport(scenario)

    await scenario.run_async(
        lambda: transport.handle_async_request(
            httpx.Request("POST", "https://api.example.com/upsert")
        )
    )

    assert scenario.request_amplification() == pytest.approx(1.0)


async def test_async_retry_after_smear_upper_bound_respected() -> None:
    config = StormConfig(
        n_clients=5,
        throttle_window_seconds=0.1,
        retry_after_seconds=2.0,
        seed=0xC0FFEE,
    )
    scenario = StormScenario(config)
    transport = _make_async_transport(scenario, max_retries=3, max_wait=5.0)

    await scenario.run_async(
        lambda: transport.handle_async_request(
            httpx.Request("POST", "https://api.example.com/upsert")
        )
    )

    ft: _AsyncFaultInjectionTransport = scenario.async_transport
    window_end = ft.start_time + config.throttle_window_seconds
    upper_bound = window_end + config.retry_after_seconds * 1.5 + 0.05
    success_records = [r for r in ft.records if r.outcome == "200"]
    assert success_records, "expected some successful responses"
    latest_success = max(r.timestamp for r in success_records)
    assert latest_success <= upper_bound, (
        f"latest success at {latest_success - window_end:.3f}s after window end, "
        f"expected <= {config.retry_after_seconds * 1.5:.3f}s"
    )


async def test_async_parity_metric_recorded() -> None:
    """Runs the canonical scenario and writes metrics to disk for cross-transport comparison.

    DX-0168 (bulk upsert) writes a sibling file; test_storm_parity.py reads all three
    and asserts they are within 2x (dispersion) and 1.5x (amplification) of each other.
    """
    config = StormConfig(
        n_clients=50,
        throttle_window_seconds=1.0,
        retry_after_seconds=0.5,
        seed=0xC0FFEE,
    )
    scenario = StormScenario(config)
    transport = _make_async_transport(scenario)

    await scenario.run_async(
        lambda: transport.handle_async_request(
            httpx.Request("POST", "https://api.example.com/upsert")
        )
    )

    first = scenario.first_success_after_window()
    start = scenario.async_transport.start_time
    first_relative: float | None = (first - start) if first is not None else None

    metrics: dict[str, object] = {
        "transport": "async",
        "n_clients": config.n_clients,
        "dispersion_width": scenario.dispersion_width(only_successes=True),
        "first_success_relative": first_relative,
        "request_amplification": scenario.request_amplification(),
    }
    _PARITY_METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    assert _PARITY_METRICS_PATH.exists()
