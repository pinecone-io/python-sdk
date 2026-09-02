"""Regression tests for bulk upsert under quota-starved backends.

Verifies that upsert_in_batches does not reproduce the "self-reinforcing 429 storm"
failure mode documented in pipeline-state/retry-resilience-plan.md.

The quota-based fault-injection transport enforces a hard in-flight limit.
Assertions cover:
  - Operation completes successfully despite sustained throttling
  - Request amplification stays below 2.5x (not a storm)
  - Peak in-flight count never exceeds the backend quota
  - async query_namespaces fan-out stays bounded under quota

The AIMD ramp-down/ramp-up cycle used to be asserted here through the
pre-gate ``async_batch_execute``, the only bulk path that consulted this
limiter registry. That engine is gone; the cycle is now covered by
``tests/unit/_internal/bulk/test_core_aimd_contract.py`` at the arithmetic
level and by ``test_engine.test_engine_drives_aimd_recovery_end_to_end``
through a real engine.
"""

from __future__ import annotations

from typing import Any

import pytest

from pinecone._internal.adaptive import _AdaptiveLimiterRegistry
from pinecone._internal.config import RetryConfig
from pinecone._internal.http_client import _AsyncRetryTransport, _RetryTransport
from pinecone.async_client.async_index import AsyncIndex
from pinecone.index import Index
from tests.unit._internal._storm_fixture import (
    QuotaConfig,
    _AsyncFaultInjectionTransport,
    _FaultInjectionTransport,
)

INDEX_HOST = "test-index-abc1234.svc.us-east1-gcp.pinecone.io"

_UPSERT_BODY = b'{"upsertedCount":0}'
_QUERY_BODY = b"{}"


# Storm tests need real time.sleep; override the conftest sleep-suppressor.
@pytest.fixture(autouse=True)
def _no_retry_sleep() -> None:
    pass


def _make_vectors(n: int) -> list[dict[str, Any]]:
    return [{"id": f"v{i}", "values": [float(i), float(i + 1)]} for i in range(n)]


def _make_sync_index_with_quota(
    quota_config: QuotaConfig,
) -> tuple[Index, _FaultInjectionTransport]:
    """Create a sync Index wired to a quota fault-injection transport."""
    transport = _FaultInjectionTransport(quota_config)
    retry_cfg = RetryConfig(max_retries=6, backoff_factor=0.0, max_wait=2.0)
    retry_transport = _RetryTransport(transport=transport, retry_config=retry_cfg)  # type: ignore[arg-type]
    index = Index(host=INDEX_HOST, api_key="test-key")
    index._http._client._transport = retry_transport
    return index, transport


def _make_async_index_with_quota(
    quota_config: QuotaConfig,
    registry: _AdaptiveLimiterRegistry | None = None,
) -> tuple[AsyncIndex, _AsyncFaultInjectionTransport]:
    """Create an AsyncIndex wired to an async quota fault-injection transport."""
    transport = _AsyncFaultInjectionTransport(quota_config)
    on_throttle = registry.report_throttled if registry is not None else None
    retry_cfg = RetryConfig(
        max_retries=6, backoff_factor=0.0, max_wait=2.0, on_throttle=on_throttle
    )
    retry_transport = _AsyncRetryTransport(transport=transport, retry_config=retry_cfg)  # type: ignore[arg-type]
    async_index = AsyncIndex(
        host=INDEX_HOST,
        api_key="test-key",
        _limiter_registry=registry,
    )
    # Force lazy client creation then inject transport.
    http_client = async_index._http._ensure_client()
    http_client._transport = retry_transport
    return async_index, transport


# ---------------------------------------------------------------------------
# Test 1: sync upsert converges under quota starvation
# ---------------------------------------------------------------------------


def test_sync_bulk_upsert_under_quota_starvation_converges() -> None:
    """Sync upsert with max_concurrency=32 against a quota-4 backend completes without storm."""
    quota = 4
    n_vectors = 200
    batch_size = 10  # 20 batches

    quota_cfg = QuotaConfig(
        max_concurrent_requests=quota,
        retry_after_seconds=0.05,
        request_delay_seconds=0.005,
        success_content=_UPSERT_BODY,
    )
    index, transport = _make_sync_index_with_quota(quota_cfg)

    result = index.upsert(
        vectors=_make_vectors(n_vectors),
        batch_size=batch_size,
        max_concurrency=32,
        show_progress=False,
    )

    assert result.upserted_count == n_vectors, "all vectors should be counted as upserted"
    assert not result.has_errors, f"expected no batch errors; got {result.errors}"

    records = transport.records
    n_batches = n_vectors // batch_size
    total_requests = len(records)
    amplification = total_requests / n_batches
    assert amplification <= 2.5, (
        f"amplification {amplification:.2f}x exceeded 2.5x — storm may have reproduced"
    )

    assert transport.peak_in_flight <= quota, (
        f"peak in-flight {transport.peak_in_flight} exceeded quota {quota}"
    )


# ---------------------------------------------------------------------------
# Test 2: async upsert converges under quota starvation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_bulk_upsert_under_quota_starvation_converges() -> None:
    """Async upsert with max_concurrency=32 against a quota-4 backend completes without storm."""
    quota = 4
    n_vectors = 200
    batch_size = 10  # 20 batches

    quota_cfg = QuotaConfig(
        max_concurrent_requests=quota,
        retry_after_seconds=0.05,
        request_delay_seconds=0.005,
        success_content=_UPSERT_BODY,
    )
    async_index, transport = _make_async_index_with_quota(quota_cfg)

    result = await async_index.upsert(
        vectors=_make_vectors(n_vectors),
        batch_size=batch_size,
        max_concurrency=32,
        show_progress=False,
    )

    assert result.upserted_count == n_vectors, "all vectors should be counted as upserted"
    assert not result.has_errors, f"expected no batch errors; got {result.errors}"

    records = transport.records
    n_batches = n_vectors // batch_size
    total_requests = len(records)
    amplification = total_requests / n_batches
    assert amplification <= 2.5, (
        f"amplification {amplification:.2f}x exceeded 2.5x — storm may have reproduced"
    )

    assert transport.peak_in_flight <= quota, (
        f"peak in-flight {transport.peak_in_flight} exceeded quota {quota}"
    )


# ---------------------------------------------------------------------------
# Test 3: anti-amplification — no storm under quota starvation
# ---------------------------------------------------------------------------


def test_no_storm_under_quota_starvation() -> None:
    """Total recorded requests must stay below 2.5x batch count.

    If AIMD or retry jitter fails, the thundering-herd effect would push
    amplification to 10x+. This test catches that regression loud.
    """
    quota = 4
    n_vectors = 500
    batch_size = 10  # 50 batches

    quota_cfg = QuotaConfig(
        max_concurrent_requests=quota,
        retry_after_seconds=0.05,
        request_delay_seconds=0.005,
        success_content=_UPSERT_BODY,
    )
    index, transport = _make_sync_index_with_quota(quota_cfg)

    result = index.upsert(
        vectors=_make_vectors(n_vectors),
        batch_size=batch_size,
        max_concurrency=32,
        show_progress=False,
    )

    assert not result.has_errors, f"expected no batch errors; got {result.errors}"

    n_batches = n_vectors // batch_size
    total_requests = len(transport.records)
    assert total_requests < 2.5 * n_batches, (
        f"total requests {total_requests} >= 2.5 * n_batches ({2.5 * n_batches:.0f}); "
        f"amplification {total_requests / n_batches:.2f}x suggests storm behavior"
    )


# ---------------------------------------------------------------------------
# Test 5: query_namespaces bounded fan-out under quota
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_namespaces_bounded_fan_out_under_quota() -> None:
    """async query_namespaces stays bounded even against a quota-4 transport.

    DX-0161 fixed query_namespaces to use the AIMD-gated limiter (ceiling=10).
    This test drives 50 namespaces against a quota=4 transport and asserts:
    - The operation completes (all namespaces queried)
    - Peak in-flight never exceeds the internal ceiling of 10
    """
    quota = 4
    n_namespaces = 50
    internal_ceiling = 10  # hardcoded in AsyncIndex.query_namespaces

    registry = _AdaptiveLimiterRegistry()
    quota_cfg = QuotaConfig(
        max_concurrent_requests=quota,
        retry_after_seconds=0.05,
        request_delay_seconds=0.005,
        success_content=_QUERY_BODY,
    )
    async_index, transport = _make_async_index_with_quota(quota_cfg, registry=registry)

    namespaces = [f"ns-{i}" for i in range(n_namespaces)]
    query_results = await async_index.query_namespaces(
        vector=[0.1, 0.2],
        namespaces=namespaces,
        metric="cosine",
        top_k=5,
    )

    # All namespaces should have been queried (matches may be empty, but no error).
    assert query_results is not None

    # Peak in-flight at transport level must not exceed the AIMD ceiling.
    assert transport.peak_in_flight <= internal_ceiling, (
        f"peak in-flight {transport.peak_in_flight} exceeded internal ceiling {internal_ceiling}"
    )
