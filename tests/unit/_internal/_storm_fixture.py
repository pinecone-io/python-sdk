"""Shared fault-injection transport + storm scenario fixture for retry validation tests.

Consumed by tests/unit/_internal/test_retry_storm_sync.py,
test_retry_storm_async.py, and test_bulk_upsert_storm.py.

Note: StormScenario.__init__ calls random.seed(config.seed) so that jitter math in
_compute_backoff / _compute_retry_after_delay is reproducible across runs.
"""

from __future__ import annotations

import asyncio
import random
import threading
import time
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Literal

import httpx
import pytest

from pinecone._internal.config import RetryConfig
from pinecone._internal.http_client import _AsyncRetryTransport, _RetryTransport


@dataclass
class RequestRecord:
    timestamp: float
    host: str
    attempt_index: int
    outcome: Literal["429", "200", "503"]


@dataclass
class StormConfig:
    n_clients: int = 50
    throttle_window_seconds: float = 1.0
    retry_after_seconds: float = 0.5
    post_window_capacity_rps: float = 1000.0
    seed: int = 0xC0FFEE


@dataclass
class QuotaConfig:
    """Config for in-flight-quota-based fault injection.

    The transport throttles when concurrent in-flight requests exceed
    ``max_concurrent_requests``, returning 429 + Retry-After immediately.
    Allowed requests sleep for ``request_delay_seconds`` then return 200
    with ``success_content`` as the response body.
    """

    max_concurrent_requests: int
    retry_after_seconds: float = 0.1
    request_delay_seconds: float = 0.01
    success_content: bytes = field(default=b"{}")


class _FaultInjectionTransport(httpx.BaseTransport):
    """Sync httpx transport — storm-window mode or in-flight-quota mode."""

    def __init__(self, config: StormConfig | QuotaConfig) -> None:
        self._config = config
        self._lock = threading.Lock()
        self._records: list[RequestRecord] = []
        self._attempt_counters: dict[tuple[str, int], int] = {}
        self.start_time: float = time.monotonic()
        # Quota-mode state
        self._in_flight: int = 0
        self._peak_in_flight: int = 0

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        if isinstance(self._config, QuotaConfig):
            return self._handle_quota_request(request)
        return self._handle_storm_request(request)

    def _handle_quota_request(self, request: httpx.Request) -> httpx.Response:
        cfg: QuotaConfig = self._config  # type: ignore[assignment]
        host = request.url.host
        with self._lock:
            if self._in_flight >= cfg.max_concurrent_requests:
                record = RequestRecord(
                    timestamp=time.monotonic(), host=host, attempt_index=0, outcome="429"
                )
                self._records.append(record)
                return httpx.Response(
                    429,
                    headers={"Retry-After": str(cfg.retry_after_seconds)},
                    request=request,
                )
            self._in_flight += 1
            if self._in_flight > self._peak_in_flight:
                self._peak_in_flight = self._in_flight

        time.sleep(cfg.request_delay_seconds)

        with self._lock:
            self._in_flight -= 1
            record = RequestRecord(
                timestamp=time.monotonic(), host=host, attempt_index=0, outcome="200"
            )
            self._records.append(record)

        return httpx.Response(200, content=cfg.success_content, request=request)

    def _handle_storm_request(self, request: httpx.Request) -> httpx.Response:
        elapsed = time.monotonic() - self.start_time
        host = request.url.host
        ident = threading.get_ident()
        key = (host, ident)
        storm_cfg: StormConfig = self._config  # type: ignore[assignment]
        with self._lock:
            idx = self._attempt_counters.get(key, 0)
            self._attempt_counters[key] = idx + 1
            if elapsed < storm_cfg.throttle_window_seconds:
                outcome: Literal["429", "200", "503"] = "429"
                record = RequestRecord(
                    timestamp=time.monotonic(),
                    host=host,
                    attempt_index=idx,
                    outcome=outcome,
                )
                self._records.append(record)
                return httpx.Response(
                    429,
                    headers={"Retry-After": str(storm_cfg.retry_after_seconds)},
                    request=request,
                )
            outcome = "200"
            record = RequestRecord(
                timestamp=time.monotonic(),
                host=host,
                attempt_index=idx,
                outcome=outcome,
            )
            self._records.append(record)
            return httpx.Response(200, request=request)

    @property
    def records(self) -> list[RequestRecord]:
        with self._lock:
            return list(self._records)

    @property
    def peak_in_flight(self) -> int:
        """Peak concurrent in-flight count (quota mode only)."""
        with self._lock:
            return self._peak_in_flight


class _AsyncFaultInjectionTransport(httpx.AsyncBaseTransport):
    """Async counterpart of _FaultInjectionTransport — storm-window or quota mode."""

    def __init__(self, config: StormConfig | QuotaConfig) -> None:
        self._config = config
        self._lock = asyncio.Lock()
        self._records: list[RequestRecord] = []
        self._attempt_counters: dict[tuple[str, int], int] = {}
        self.start_time: float = time.monotonic()
        # Quota-mode state
        self._in_flight: int = 0
        self._peak_in_flight: int = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if isinstance(self._config, QuotaConfig):
            return await self._handle_quota_request(request)
        return await self._handle_storm_request(request)

    async def _handle_quota_request(self, request: httpx.Request) -> httpx.Response:
        cfg: QuotaConfig = self._config  # type: ignore[assignment]
        host = request.url.host
        async with self._lock:
            if self._in_flight >= cfg.max_concurrent_requests:
                record = RequestRecord(
                    timestamp=time.monotonic(), host=host, attempt_index=0, outcome="429"
                )
                self._records.append(record)
                return httpx.Response(
                    429,
                    headers={"Retry-After": str(cfg.retry_after_seconds)},
                    request=request,
                )
            self._in_flight += 1
            if self._in_flight > self._peak_in_flight:
                self._peak_in_flight = self._in_flight

        await asyncio.sleep(cfg.request_delay_seconds)

        async with self._lock:
            self._in_flight -= 1
            record = RequestRecord(
                timestamp=time.monotonic(), host=host, attempt_index=0, outcome="200"
            )
            self._records.append(record)

        return httpx.Response(200, content=cfg.success_content, request=request)

    async def _handle_storm_request(self, request: httpx.Request) -> httpx.Response:
        elapsed = time.monotonic() - self.start_time
        host = request.url.host
        task = asyncio.current_task()
        ident = id(task) if task is not None else 0
        key = (host, ident)
        async with self._lock:
            idx = self._attempt_counters.get(key, 0)
            self._attempt_counters[key] = idx + 1
            storm_cfg: StormConfig = self._config  # type: ignore[assignment]
            if elapsed < storm_cfg.throttle_window_seconds:
                outcome: Literal["429", "200", "503"] = "429"
                record = RequestRecord(
                    timestamp=time.monotonic(),
                    host=host,
                    attempt_index=idx,
                    outcome=outcome,
                )
                self._records.append(record)
                return httpx.Response(
                    429,
                    headers={"Retry-After": str(storm_cfg.retry_after_seconds)},
                    request=request,
                )
            outcome = "200"
            record = RequestRecord(
                timestamp=time.monotonic(),
                host=host,
                attempt_index=idx,
                outcome=outcome,
            )
            self._records.append(record)
            return httpx.Response(200, request=request)

    @property
    def records(self) -> list[RequestRecord]:
        return list(self._records)

    @property
    def peak_in_flight(self) -> int:
        """Peak concurrent in-flight count (quota mode only)."""
        return self._peak_in_flight


class StormScenario:
    """Drives N concurrent clients through `make_request` and records aggregate behavior."""

    def __init__(self, config: StormConfig) -> None:
        random.seed(config.seed)
        self._config = config
        self.sync_transport = _FaultInjectionTransport(config)
        self.async_transport = _AsyncFaultInjectionTransport(config)
        self.exceptions: list[BaseException] = []
        self._records: list[RequestRecord] = []
        self._active_start_time: float = 0.0

    def run_sync(self, make_request: Callable[[], httpx.Response]) -> None:
        """Spawn config.n_clients threads, each calling make_request() once."""
        now = time.monotonic()
        self.sync_transport.start_time = now
        self._active_start_time = now
        with ThreadPoolExecutor(max_workers=self._config.n_clients) as pool:
            futures = [pool.submit(make_request) for _ in range(self._config.n_clients)]
            for f in futures:
                try:
                    f.result()
                except BaseException as e:
                    self.exceptions.append(e)
        self._records = self.sync_transport.records

    async def run_async(self, make_request: Callable[[], Awaitable[httpx.Response]]) -> None:
        """asyncio.gather config.n_clients coroutines, each awaiting make_request()."""
        now = time.monotonic()
        self.async_transport.start_time = now
        self._active_start_time = now

        async def _call() -> httpx.Response:
            return await make_request()

        results: list[httpx.Response | BaseException] = await asyncio.gather(
            *[_call() for _ in range(self._config.n_clients)],
            return_exceptions=True,
        )
        for r in results:
            if isinstance(r, BaseException):
                self.exceptions.append(r)
        self._records = self.async_transport.records

    def dispersion_width(self, *, only_successes: bool = True) -> float:
        """Return max(t) - min(t) across recorded request timestamps (seconds)."""
        if only_successes:
            timestamps = [r.timestamp for r in self._records if r.outcome == "200"]
        else:
            timestamps = [r.timestamp for r in self._records]
        if len(timestamps) < 2:
            return 0.0
        return max(timestamps) - min(timestamps)

    def first_success_after_window(self) -> float | None:
        """Return earliest 200 timestamp at or after throttle window end, or None."""
        window_end = self.sync_transport.start_time + self._config.throttle_window_seconds
        candidates = [
            r.timestamp for r in self._records if r.outcome == "200" and r.timestamp >= window_end
        ]
        return min(candidates) if candidates else None

    def request_amplification(self) -> float:
        """Return total_requests / n_clients. 1.0 = ideal, > 1.0 = retries fired."""
        return len(self._records) / self._config.n_clients

    def converged_concurrency(self, window_seconds: float = 0.1) -> int:
        """Max concurrent in-flight count observed over any window_seconds-wide bucket."""
        if not self._records:
            return 0
        timestamps = sorted(r.timestamp for r in self._records)
        max_count = 0
        for t in timestamps:
            count = sum(1 for ts in timestamps if t <= ts < t + window_seconds)
            max_count = max(max_count, count)
        return max_count


# ---------------------------------------------------------------------------
# Self-tests — collected by pytest when running on this module directly
# ---------------------------------------------------------------------------


# Override the unit-test conftest's autouse sleep-suppressor.  Storm tests
# need real time.sleep so retries actually advance the clock past the
# throttle window.  Defining a fixture with the same name here takes
# precedence over the conftest.py version for tests in this module.
@pytest.fixture(autouse=True)
def _no_retry_sleep() -> None:
    pass


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
