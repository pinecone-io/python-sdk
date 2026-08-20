"""Canonical storm scenarios behind the cross-transport parity check.

Each ``*_parity_metrics`` function runs the identical canonical storm — 50
concurrent clients, a 1.0s throttle window, 500ms Retry-After / pushback, seed
0xC0FFEE — over one transport and returns its metrics as a plain dict.

These replace a JSON-file handoff between test modules, which had two defects:
the artifacts were tracked, so every unit-test run dirtied the checkout, and the
gRPC producer collected *after* the parity consumer, so the gRPC arm of the
comparison always read the previous run's value. See "Cross-transport storm
parity" in README.md.
"""

from __future__ import annotations

import httpx

from pinecone._internal.config import RetryConfig
from pinecone._internal.http_client import _AsyncRetryTransport, _RetryTransport
from tests.unit._internal._storm_fixture import StormConfig, StormScenario

StormMetrics = dict[str, object]

METRIC_KEYS = (
    "transport",
    "n_clients",
    "dispersion_width",
    "first_success_relative",
    "request_amplification",
)


def canonical_config() -> StormConfig:
    return StormConfig(
        n_clients=50,
        throttle_window_seconds=1.0,
        retry_after_seconds=0.5,
        seed=0xC0FFEE,
    )


def _retry_config() -> RetryConfig:
    return RetryConfig(max_retries=5, backoff_factor=0.01, max_wait=2.0)


def sync_parity_metrics() -> StormMetrics:
    config = canonical_config()
    scenario = StormScenario(config)
    transport = _RetryTransport(  # type: ignore[arg-type]
        transport=scenario.sync_transport, retry_config=_retry_config()
    )

    scenario.run_sync(
        lambda: transport.handle_request(httpx.Request("POST", "https://api.example.com/upsert"))
    )

    first = scenario.first_success_after_window()
    start = scenario.sync_transport.start_time
    return {
        "transport": "sync",
        "n_clients": config.n_clients,
        "dispersion_width": scenario.dispersion_width(only_successes=True),
        "first_success_relative": (first - start) if first is not None else None,
        "request_amplification": scenario.request_amplification(),
    }


async def async_parity_metrics() -> StormMetrics:
    config = canonical_config()
    scenario = StormScenario(config)
    transport = _AsyncRetryTransport(  # type: ignore[arg-type]
        transport=scenario.async_transport, retry_config=_retry_config()
    )

    await scenario.run_async(
        lambda: transport.handle_async_request(
            httpx.Request("POST", "https://api.example.com/upsert")
        )
    )

    first = scenario.first_success_after_window()
    start = scenario.async_transport.start_time
    return {
        "transport": "async",
        "n_clients": config.n_clients,
        "dispersion_width": scenario.dispersion_width(only_successes=True),
        "first_success_relative": (first - start) if first is not None else None,
        "request_amplification": scenario.request_amplification(),
    }


def grpc_parity_metrics() -> StormMetrics:
    # Deferred: keeps this module free of a collection-time dependency on
    # another test module, which pytest also imports under the same name.
    from tests.unit.grpc.test_retry_storm_grpc import (
        GrpcStormConfig,
        _dispersion_width,
        _first_success_after_window,
        _request_amplification,
        _run_storm,
    )

    config = GrpcStormConfig(
        n_clients=50,
        throttle_window_seconds=1.0,
        pushback_ms=500.0,
        seed=0xC0FFEE,
    )
    server = _run_storm(config)
    records = server.records

    first = _first_success_after_window(records, server, config)
    return {
        "transport": "grpc",
        "n_clients": config.n_clients,
        "dispersion_width": _dispersion_width(records, only_successes=True),
        "first_success_relative": (first - server.start_time) if first is not None else None,
        "request_amplification": _request_amplification(records, config.n_clients),
    }
