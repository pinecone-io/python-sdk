"""Cross-transport parity checks for storm simulation metrics.

Compares dispersion_width and request_amplification across sync (DX-0166),
async (DX-0167), and gRPC (DX-0168). All three canonical scenarios are run here,
in this session, so the comparison is always between metrics produced by the
same run — see tests/unit/_internal/_storm_parity_scenarios.py.
"""

from __future__ import annotations

import asyncio

import pytest

from tests.unit._internal._storm_fixture import RequestRecord, StormConfig, StormScenario
from tests.unit._internal._storm_parity_scenarios import (
    METRIC_KEYS,
    StormMetrics,
    async_parity_metrics,
    grpc_parity_metrics,
    sync_parity_metrics,
)

# The module-scoped `metrics` fixture sleeps for real (see `_no_retry_sleep`),
# measured at 4.27s, and lands in the setup phase of whichever test runs first.
# Under the global `timeout = 5` that is a coin flip, not a guard. Module-level
# so test ordering cannot decide who pays it.
pytestmark = pytest.mark.timeout(30)


# Override the unit-test conftest's autouse sleep-suppressor. Storm scenarios
# need real sleeps so retries actually advance the clock past the throttle
# window. Defining the same-named fixture here takes precedence.
@pytest.fixture(autouse=True)
def _no_retry_sleep() -> None:
    pass


@pytest.fixture(scope="module")
def metrics() -> dict[str, StormMetrics]:
    return {
        "sync": sync_parity_metrics(),
        "async": asyncio.run(async_parity_metrics()),
        "grpc": grpc_parity_metrics(),
    }


@pytest.mark.parametrize("transport", ["sync", "async", "grpc"])
def test_metrics_complete_and_finite(transport: str, metrics: dict[str, StormMetrics]) -> None:
    m = metrics[transport]
    assert set(m) == set(METRIC_KEYS)
    assert m["transport"] == transport
    assert m["n_clients"] == 50
    assert float(m["dispersion_width"]) > 0.0  # type: ignore[arg-type]
    assert m["first_success_relative"] is not None, "no success observed after throttle window"
    assert float(m["first_success_relative"]) > 0.0  # type: ignore[arg-type]
    assert float(m["request_amplification"]) >= 1.0  # type: ignore[arg-type]


def test_dispersion_widths_within_2x(metrics: dict[str, StormMetrics]) -> None:
    sync_w = float(metrics["sync"]["dispersion_width"])  # type: ignore[arg-type]
    async_w = float(metrics["async"]["dispersion_width"])  # type: ignore[arg-type]
    grpc_w = float(metrics["grpc"]["dispersion_width"])  # type: ignore[arg-type]

    assert async_w <= sync_w * 2.0, f"async dispersion {async_w:.3f} > 2x sync {sync_w:.3f}"
    assert sync_w <= async_w * 2.0, f"sync dispersion {sync_w:.3f} > 2x async {async_w:.3f}"
    assert grpc_w <= sync_w * 2.0, f"gRPC dispersion {grpc_w:.3f} > 2x sync {sync_w:.3f}"
    assert sync_w <= grpc_w * 2.0, f"sync dispersion {sync_w:.3f} > 2x gRPC {grpc_w:.3f}"


def test_amplifications_within_1_5x(metrics: dict[str, StormMetrics]) -> None:
    sync_a = float(metrics["sync"]["request_amplification"])  # type: ignore[arg-type]
    async_a = float(metrics["async"]["request_amplification"])  # type: ignore[arg-type]
    grpc_a = float(metrics["grpc"]["request_amplification"])  # type: ignore[arg-type]

    assert async_a <= sync_a * 1.5, f"async amp {async_a:.3f} > 1.5x sync {sync_a:.3f}"
    assert sync_a <= async_a * 1.5, f"sync amp {sync_a:.3f} > 1.5x async {async_a:.3f}"
    assert grpc_a <= sync_a * 1.5, f"gRPC amp {grpc_a:.3f} > 1.5x sync {sync_a:.3f}"
    assert sync_a <= grpc_a * 1.5, f"sync amp {sync_a:.3f} > 1.5x gRPC {grpc_a:.3f}"


def test_first_success_after_window_measures_the_window_from_the_run_that_happened() -> None:
    """A 200 from inside the real window must not be read as the first one after it.

    ``first_success_relative`` above is only as good as the window origin
    behind it. Measuring from ``sync_transport.start_time`` — which each
    transport stamps in its own ``__init__`` — ends an async run's window early
    by the construction-to-run gap, staged here as 0.3s, and admits the 0.85s
    success below.
    """
    config = StormConfig(n_clients=1, throttle_window_seconds=1.0)
    scenario = StormScenario(config)
    async_start = scenario.sync_transport.start_time + 0.3
    scenario.async_transport.start_time = async_start
    scenario._active_start_time = async_start

    inside_window = async_start + 0.85
    after_window = async_start + 1.5
    scenario._records = [
        RequestRecord(timestamp=inside_window, host="example.com", attempt_index=0, outcome="200"),
        RequestRecord(timestamp=after_window, host="example.com", attempt_index=1, outcome="200"),
    ]

    assert scenario.first_success_after_window() == after_window
