"""Storm simulation tests for REST sync transport (thundering-herd dispersal).

Verifies that N concurrent sync clients receiving identical Retry-After: 0.5 headers
disperse in time rather than re-colliding, and that request amplification stays bounded.

These tests pin the jitter/dispersal math under a full-outage window — precisely the
scenario where the gRFC A6 retry budget correctly suppresses retries — so transports
here get an effectively unlimited budget. test_retry_budget.py pins suppression itself.
"""

from __future__ import annotations

import httpx
import pytest

from pinecone._internal.config import RetryConfig
from pinecone._internal.http_client import _BudgetRegistry, _RetryTransport
from tests.unit._internal._storm_fixture import (
    StormConfig,
    StormScenario,
    _FaultInjectionTransport,
)


# Override the unit-test conftest's autouse sleep-suppressor. Storm tests
# need real time.sleep so retries actually advance the clock past the
# throttle window. Defining the same-named fixture here takes precedence
# over the conftest.py version for tests in this module.
@pytest.fixture(autouse=True)
def _no_retry_sleep() -> None:
    pass


def _make_transport(
    scenario: StormScenario,
    max_retries: int = 5,
    backoff_factor: float = 0.01,
    max_wait: float = 2.0,
) -> _RetryTransport:
    cfg = RetryConfig(max_retries=max_retries, backoff_factor=backoff_factor, max_wait=max_wait)
    return _RetryTransport(
        transport=scenario.sync_transport,  # type: ignore[arg-type]
        retry_config=cfg,
        budget_registry=_BudgetRegistry(max_tokens=1e9),
    )


def test_thundering_herd_disperses_with_retry_after_smear() -> None:
    config = StormConfig(
        n_clients=50,
        throttle_window_seconds=1.0,
        retry_after_seconds=0.5,
        seed=0xC0FFEE,
    )
    scenario = StormScenario(config)
    transport = _make_transport(scenario)

    def make_request() -> httpx.Response:
        return transport.handle_request(httpx.Request("POST", "https://api.example.com/upsert"))

    scenario.run_sync(make_request)
    assert not scenario.exceptions, f"unexpected exceptions: {scenario.exceptions}"

    # Dispersion: not all clients retry at the exact same moment.
    # With retry_after=0.5 and smear=uniform(0, 0.25), each client retries at uniform(0.5, 0.75)
    # twice before succeeding, so success times span ~[1.0, 1.5]s — max spread ≈ 0.5s.
    width = scenario.dispersion_width(only_successes=True)
    assert 0.10 <= width <= 0.70, f"dispersion width {width:.3f}s outside [0.10, 0.70]"

    # First success lands after the window opens and within retry_after * 1.5 of window end.
    # (Two delays of uniform(0.5, 0.75) can sum to as little as 1.0s, so the first success
    # can be very close to window_end; the upper bound ensures smear doesn't over-delay.)
    first = scenario.first_success_after_window()
    start = scenario.sync_transport.start_time
    first_relative: float | None = (first - start) if first is not None else None
    assert first_relative is not None, "no successful responses observed after throttle window"
    lower = config.throttle_window_seconds
    upper = config.throttle_window_seconds + config.retry_after_seconds * 1.5 + 0.1
    assert lower <= first_relative <= upper, (
        f"first success at {first_relative:.3f}s, expected in [{lower:.2f}, {upper:.2f}]"
    )


def test_request_amplification_bounded_under_throttle() -> None:
    config = StormConfig(
        n_clients=50,
        throttle_window_seconds=1.0,
        retry_after_seconds=0.5,
        seed=0xC0FFEE,
    )
    scenario = StormScenario(config)
    transport = _make_transport(scenario)

    scenario.run_sync(
        lambda: transport.handle_request(httpx.Request("POST", "https://api.example.com/upsert"))
    )

    amp = scenario.request_amplification()
    # Each client fires 1 initial + 1-2 retries on average; > 3 means jitter is failing to space requests out.
    assert 1.5 <= amp <= 3.0, f"amplification {amp:.3f} outside [1.5, 3.0]"


def test_n_200_clients_still_disperses() -> None:
    """Identical bounds hold for N=200 — catches N-dependent regressions in the jitter math."""
    config = StormConfig(
        n_clients=200,
        throttle_window_seconds=1.0,
        retry_after_seconds=0.5,
        seed=0xC0FFEE,
    )
    scenario = StormScenario(config)
    transport = _make_transport(scenario)

    scenario.run_sync(
        lambda: transport.handle_request(httpx.Request("POST", "https://api.example.com/upsert"))
    )
    assert not scenario.exceptions, f"unexpected exceptions: {scenario.exceptions}"

    width = scenario.dispersion_width(only_successes=True)
    assert 0.10 <= width <= 0.70, f"dispersion width {width:.3f}s outside [0.10, 0.70]"


def test_no_throttle_no_amplification() -> None:
    config = StormConfig(
        n_clients=50,
        throttle_window_seconds=0.0,
        retry_after_seconds=0.5,
        seed=0xC0FFEE,
    )
    scenario = StormScenario(config)
    transport = _make_transport(scenario)

    scenario.run_sync(
        lambda: transport.handle_request(httpx.Request("POST", "https://api.example.com/upsert"))
    )

    assert scenario.request_amplification() == pytest.approx(1.0)


# Alone among this module's tests, this one sets retry_after_seconds=2.0, and the
# smear can stretch a Retry-After by 1.5x — so real sleeps put a ~3.15s ceiling
# (0.1s window + 2.0 * 1.5 + jitter) under it, which the assertion below is itself
# checking. Measured 2.97s with 0.1% of it CPU: wall clock, not compute, so it
# does not amplify on a slower runner. 20s is ~6.7x, the ratio #306 used on
# test_storm_parity.py (4.27s -> 30s). Scoped, not module-level: the rest of the
# module runs at 1.41-1.48s and keeps the 5s default.
@pytest.mark.timeout(20)
def test_retry_after_smear_upper_bound_respected() -> None:
    config = StormConfig(
        n_clients=5,
        throttle_window_seconds=0.1,
        retry_after_seconds=2.0,
        seed=0xC0FFEE,
    )
    scenario = StormScenario(config)
    transport = _make_transport(scenario, max_retries=3, max_wait=5.0)

    scenario.run_sync(
        lambda: transport.handle_request(httpx.Request("POST", "https://api.example.com/upsert"))
    )

    ft: _FaultInjectionTransport = scenario.sync_transport
    window_end = ft.start_time + config.throttle_window_seconds
    # Smear can extend retry_after by up to 50%; add a small epsilon for scheduling jitter.
    upper_bound = window_end + config.retry_after_seconds * 1.5 + 0.05
    success_records = [r for r in ft.records if r.outcome == "200"]
    assert success_records, "expected some successful responses"
    latest_success = max(r.timestamp for r in success_records)
    assert latest_success <= upper_bound, (
        f"latest success at {latest_success - window_end:.3f}s after window end, "
        f"expected <= {config.retry_after_seconds * 1.5:.3f}s"
    )
