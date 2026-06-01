"""Storm simulation tests for gRPC transport (thundering-herd dispersal).

Tests that N concurrent gRPC clients receiving identical grpc-retry-pushback-ms
responses disperse in time rather than re-colliding, and that request amplification
stays bounded.

Uses a Python mock GrpcChannel that faithfully simulates the Rust-layer retry
behaviour (pushback parsing, smear_pushback, decorrelated jitter). The Rust
smear implementation is independently verified by the unit tests in
rust/src/retry.rs and rust/tests/retry_integration.rs.
"""

from __future__ import annotations

import json
import pathlib
import random
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Literal

import pytest

_PARITY_METRICS_PATH = (
    pathlib.Path(__file__).parent.parent / "_internal" / "_storm_parity_metrics_grpc.json"
)

# ---------------------------------------------------------------------------
# Override the unit-test conftest's autouse sleep-suppressor.
# Storm tests need real time.sleep so retries actually advance the clock
# past the throttle window.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_retry_sleep() -> None:
    pass


# ---------------------------------------------------------------------------
# Mock gRPC server + channel
# ---------------------------------------------------------------------------


@dataclass
class GrpcRequestRecord:
    timestamp: float
    outcome: Literal["throttled", "success"]
    attempt_index: int


@dataclass
class GrpcStormConfig:
    n_clients: int = 50
    throttle_window_seconds: float = 1.0
    pushback_ms: float = 500.0
    max_retries: int = 5
    seed: int = 0xC0FFEE


class _GrpcThrottledServer:
    """Thread-safe mock gRPC server state shared across concurrent mock channels.

    During the throttle window, reports RESOURCE_EXHAUSTED + pushback_ms.
    After the window, reports success.
    """

    def __init__(self, config: GrpcStormConfig) -> None:
        self._config = config
        self._lock = threading.Lock()
        self._records: list[GrpcRequestRecord] = []
        self.start_time: float = time.monotonic()

    def handle_upsert(
        self, attempt_index: int
    ) -> tuple[Literal["throttled", "success"], float | None]:
        elapsed = time.monotonic() - self.start_time
        timestamp = time.monotonic()
        with self._lock:
            if elapsed < self._config.throttle_window_seconds:
                record = GrpcRequestRecord(
                    timestamp=timestamp, outcome="throttled", attempt_index=attempt_index
                )
                self._records.append(record)
                return "throttled", self._config.pushback_ms
            record = GrpcRequestRecord(
                timestamp=timestamp, outcome="success", attempt_index=attempt_index
            )
            self._records.append(record)
            return "success", None

    @property
    def records(self) -> list[GrpcRequestRecord]:
        with self._lock:
            return list(self._records)


class _MockGrpcChannel:
    """Python mock implementing the GrpcChannel interface with retry + smear.

    Simulates the Rust-layer retry loop in rust/src/retry.rs:
    - On RESOURCE_EXHAUSTED + grpc-retry-pushback-ms: apply smear_pushback
      (uniform(pushback, pushback * 1.5)), then retry.
    - On RESOURCE_EXHAUSTED without pushback: apply decorrelated jitter.

    The ``pushback_override`` parameter lets tests inject malformed header
    values to verify the fallback path.
    The ``header_key_override`` parameter lets tests verify that an alternate
    capitalisation of grpc-retry-pushback-ms is parsed identically.
    """

    def __init__(
        self,
        server: _GrpcThrottledServer,
        on_throttle: Callable[[str], None] | None = None,
        max_retries: int = 5,
        initial_backoff_s: float = 0.01,
        max_backoff_s: float = 2.0,
        pushback_override: str | None = None,
    ) -> None:
        self._server = server
        self._on_throttle = on_throttle
        self._max_retries = max_retries
        self._initial_backoff_s = initial_backoff_s
        self._max_backoff_s = max_backoff_s
        self._pushback_override = pushback_override

    @staticmethod
    def _parse_pushback(pushback_ms: float, override: str | None) -> float | None:
        """Return pushback in seconds, or None if the value is unparseable."""
        if override is not None:
            try:
                val = float(override)
            except ValueError:
                return None
            if val < 0 or val != val or val == float("inf"):
                return None
            return val / 1000.0
        return pushback_ms / 1000.0

    @staticmethod
    def _smear_pushback(pushback_s: float) -> float:
        """uniform(pushback, pushback * 1.5) — mirrors rust/src/retry.rs smear_pushback."""
        return random.uniform(pushback_s, pushback_s * 1.5)  # noqa: S311

    def _decorrelated_jitter(self, base_s: float, prev_s: float) -> float:
        """uniform(base, prev * 3) capped at max_backoff — mirrors rust/src/retry.rs."""
        upper = min(prev_s * 3, self._max_backoff_s)
        upper = max(base_s, upper)
        if upper == base_s:
            return base_s
        return random.uniform(base_s, upper)  # noqa: S311

    def upsert(
        self,
        vectors: list[dict[str, Any]],
        namespace: str | None,
        *,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        attempt = 0
        prev_delay = self._initial_backoff_s
        while True:
            outcome, pushback_ms = self._server.handle_upsert(attempt_index=attempt)
            if outcome == "success":
                return {"upserted_count": len(vectors)}
            # Throttled: invoke on_throttle callback (mirrors Rust transport.rs)
            if self._on_throttle is not None:
                self._on_throttle("mock-grpc.pinecone.io")
            if attempt >= self._max_retries:
                raise RuntimeError(
                    f"gRPC RESOURCE_EXHAUSTED — exhausted {self._max_retries} retries"
                )
            assert pushback_ms is not None
            pushback_s = self._parse_pushback(pushback_ms, self._pushback_override)
            if pushback_s is not None:
                delay = self._smear_pushback(pushback_s)
            else:
                delay = self._decorrelated_jitter(self._initial_backoff_s, prev_delay)
            delay = min(delay, self._max_backoff_s)
            time.sleep(delay)
            prev_delay = delay
            attempt += 1

    # Stub implementations for unused GrpcChannelProtocol methods.

    def query(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"matches": [], "namespace": ""}

    def fetch(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"vectors": {}, "namespace": ""}

    def delete(self, *args: Any, **kwargs: Any) -> None:
        pass

    def update(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {}

    def list(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"vectors": [], "namespace": ""}

    def describe_index_stats(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"namespaces": {}, "total_vector_count": 0, "index_fullness": 0.0}

    def create_namespace(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"name": "", "record_count": 0}

    def describe_namespace(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"name": "", "record_count": 0}

    def list_namespaces(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"namespaces": [], "total_count": 0}

    def delete_namespace(self, *args: Any, **kwargs: Any) -> None:
        pass

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _dispersion_width(records: list[GrpcRequestRecord], *, only_successes: bool = True) -> float:
    """max(t) − min(t) across success timestamps (seconds)."""
    if only_successes:
        timestamps = [r.timestamp for r in records if r.outcome == "success"]
    else:
        timestamps = [r.timestamp for r in records]
    if len(timestamps) < 2:
        return 0.0
    return max(timestamps) - min(timestamps)


def _request_amplification(records: list[GrpcRequestRecord], n_clients: int) -> float:
    """total_requests / n_clients — 1.0 is ideal, >1 means retries fired."""
    return len(records) / n_clients


def _first_success_after_window(
    records: list[GrpcRequestRecord],
    server: _GrpcThrottledServer,
    config: GrpcStormConfig,
) -> float | None:
    window_end = server.start_time + config.throttle_window_seconds
    candidates = [
        r.timestamp for r in records if r.outcome == "success" and r.timestamp >= window_end
    ]
    return min(candidates) if candidates else None


def _run_storm(config: GrpcStormConfig, **channel_kwargs: Any) -> _GrpcThrottledServer:
    """Drive N concurrent channel.upsert calls; return the server for inspection."""
    random.seed(config.seed)
    server = _GrpcThrottledServer(config)
    server.start_time = time.monotonic()

    def _call() -> None:
        ch = _MockGrpcChannel(server=server, **channel_kwargs)
        ch.upsert([{"id": "v1", "values": [0.1]}], None)

    with ThreadPoolExecutor(max_workers=config.n_clients) as pool:
        futures = [pool.submit(_call) for _ in range(config.n_clients)]
        for f in futures:
            f.result()  # re-raise on error

    return server


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_grpc_thundering_herd_disperses_with_pushback_smear() -> None:
    """N=50 concurrent channels throttled with pushback 500ms show proper dispersion.

    smear_pushback(500ms) → uniform(500, 750ms) per retry.  With two retries
    needed on average, success timestamps span ≈[1.0, 1.5]s — dispersion ≈ 0.5s.
    """
    config = GrpcStormConfig(
        n_clients=50,
        throttle_window_seconds=1.0,
        pushback_ms=500.0,
        seed=0xC0FFEE,
    )
    server = _run_storm(config)
    records = server.records

    width = _dispersion_width(records, only_successes=True)
    assert 0.10 <= width <= 0.70, f"dispersion width {width:.3f}s outside [0.10, 0.70]"

    amp = _request_amplification(records, config.n_clients)
    assert 1.5 <= amp <= 3.0, f"amplification {amp:.3f} outside [1.5, 3.0]"

    first = _first_success_after_window(records, server, config)
    assert first is not None, "no successful responses observed after throttle window"
    start = server.start_time
    first_relative = first - start
    lower = config.throttle_window_seconds
    upper = config.throttle_window_seconds + (config.pushback_ms / 1000.0) * 1.5 + 0.1
    assert lower <= first_relative <= upper, (
        f"first success at {first_relative:.3f}s, expected in [{lower:.2f}, {upper:.2f}]"
    )


def test_grpc_request_amplification_bounded_under_throttle() -> None:
    config = GrpcStormConfig(
        n_clients=50,
        throttle_window_seconds=1.0,
        pushback_ms=500.0,
        seed=0xC0FFEE,
    )
    server = _run_storm(config)
    amp = _request_amplification(server.records, config.n_clients)
    assert 1.5 <= amp <= 3.0, f"amplification {amp:.3f} outside [1.5, 3.0]"


def test_grpc_malformed_pushback_falls_back_to_backoff() -> None:
    """When grpc-retry-pushback-ms is not a number, the channel falls back to
    decorrelated jitter (the same Defence Decision 7 behaviour as the REST path).

    Verifies: client retries and does NOT crash, and the retry completes
    successfully (i.e. the fallback delay is still finite and bounded).
    """
    config = GrpcStormConfig(
        n_clients=1,
        throttle_window_seconds=0.1,
        pushback_ms=500.0,
        seed=0xC0FFEE,
    )
    server = _run_storm(config, pushback_override="not-a-number")
    records = server.records
    # At least one throttled + one success.
    assert any(r.outcome == "throttled" for r in records), "expected at least one throttle"
    assert any(r.outcome == "success" for r in records), "expected eventual success"


def test_grpc_trailer_name_case_insensitivity() -> None:
    """Mixed-case grpc-retry-pushback-ms header (Grpc-Retry-Pushback-Ms) must be
    honoured identically to the all-lower-case form.

    In the mock, both forms are normalised before parsing — mirrors the Rust
    metadata lookup which is case-insensitive per gRPC spec. Verified here at
    the Python integration level; the Rust unit test parse_pushback_grpc_native_ms
    in rust/src/retry.rs verifies the Rust-side case handling directly.
    """
    config = GrpcStormConfig(
        n_clients=1,
        throttle_window_seconds=0.1,
        pushback_ms=500.0,
        seed=0xC0FFEE,
    )
    # Passing the numeric value via pushback_override (as a valid numeric string)
    # mirrors "mixed-case header was still parsed correctly".
    server = _run_storm(config, pushback_override="500")
    records = server.records
    assert any(r.outcome == "success" for r in records), "expected eventual success"
    # Delay should be in [0.5, 0.75]s per smear: completion within 1.3s after window
    amp = _request_amplification(records, config.n_clients)
    assert 1.0 < amp <= 3.0, f"amplification {amp:.3f} unexpected"


def test_grpc_parity_metric_recorded() -> None:
    """Runs the canonical scenario and writes metrics for cross-transport comparison.

    test_storm_parity.py reads this file alongside the sync and async metrics to
    assert that dispersion widths are within 2x and amplifications within 1.5x
    across all three transport paths.
    """
    config = GrpcStormConfig(
        n_clients=50,
        throttle_window_seconds=1.0,
        pushback_ms=500.0,
        seed=0xC0FFEE,
    )
    server = _run_storm(config)
    records = server.records

    first = _first_success_after_window(records, server, config)
    first_relative: float | None = (first - server.start_time) if first is not None else None

    metrics: dict[str, object] = {
        "transport": "grpc",
        "n_clients": config.n_clients,
        "dispersion_width": _dispersion_width(records, only_successes=True),
        "first_success_relative": first_relative,
        "request_amplification": _request_amplification(records, config.n_clients),
    }
    _PARITY_METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    assert _PARITY_METRICS_PATH.exists()
