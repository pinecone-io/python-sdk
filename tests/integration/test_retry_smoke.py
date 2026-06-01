"""Opt-in smoke test that exercises retry behavior against the real Pinecone API.

Gated behind PINECONE_RETRY_SMOKE=1. Creates a serverless index, drives a
high-concurrency upsert until the API rate-limits, and asserts the operation
completes successfully. Costs < $1 per run.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Generator

import pytest

from pinecone import AsyncPinecone, Pinecone
from pinecone.models.indexes.specs import ServerlessSpec
from tests.integration.conftest import ensure_index_deleted, unique_name

pytestmark = pytest.mark.skipif(
    os.environ.get("PINECONE_RETRY_SMOKE") != "1",
    reason="Set PINECONE_RETRY_SMOKE=1 to run live retry smoke tests.",
)

# Expected wall-clock for 100K vectors at batch_size=100, max_concurrency=64:
#   Best case (no throttling): ~30s
#   With throttling (this test): 60-180s, dominated by AIMD ramp-up and
#     server-side quota recovery windows.
# The 180s bound is intentionally generous; tighten on stable infra.


@pytest.fixture(scope="module")
def smoke_index_rest(api_key: str) -> Generator[str, None, None]:
    """Module-scoped serverless index (dim=1536) for REST smoke tests."""
    pc = Pinecone(api_key=api_key)
    name = unique_name("smoke-rest")
    pc.indexes.create(
        name=name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        timeout=300,
    )
    try:
        yield name
    finally:
        ensure_index_deleted(pc, name)


@pytest.fixture(scope="module")
def smoke_index_async(api_key: str) -> Generator[str, None, None]:
    """Module-scoped serverless index (dim=1536) for async smoke tests."""
    pc = Pinecone(api_key=api_key)
    name = unique_name("smoke-async")
    pc.indexes.create(
        name=name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        timeout=300,
    )
    try:
        yield name
    finally:
        ensure_index_deleted(pc, name)


@pytest.fixture(scope="module")
def smoke_index_grpc(api_key: str) -> Generator[str, None, None]:
    """Module-scoped serverless index (dim=1536) for gRPC smoke tests."""
    pc = Pinecone(api_key=api_key)
    name = unique_name("smoke-grpc")
    pc.indexes.create(
        name=name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        timeout=300,
    )
    try:
        yield name
    finally:
        ensure_index_deleted(pc, name)


@pytest.mark.integration
def test_high_concurrency_upsert_under_real_throttling(
    client: Pinecone,
    smoke_index_rest: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Sync REST: 100K vectors at max_concurrency=64 triggers rate limiting and recovers."""
    caplog.set_level(logging.DEBUG, logger="pinecone._internal.http_client")
    caplog.set_level(logging.DEBUG, logger="pinecone._internal.adaptive")

    index = client.index(name=smoke_index_rest)
    vectors: list[tuple[str, list[float]]] = [(f"id-{i}", [0.1] * 1536) for i in range(100_000)]

    start = time.monotonic()
    index.upsert(vectors=vectors, batch_size=100, max_concurrency=64, show_progress=False)
    elapsed = time.monotonic() - start

    # Operation must complete successfully.
    assert index.describe_index_stats().total_vector_count >= 100_000

    # Must complete in reasonable wall-clock time (3 minutes; serverless quota
    # recovery + AIMD ramp; exact bound is empirical — adjust on first run).
    assert elapsed < 180, f"upsert took {elapsed:.1f}s, expected < 180s"

    # We must have actually hit the rate limiter (otherwise the test isn't
    # exercising the retry path).
    throttled = [r for r in caplog.records if "Throttled response" in r.getMessage()]
    assert len(throttled) > 0, (
        "test did not trigger rate limiting; increase concurrency or batch count"
    )

    # AIMD must have actually engaged.
    decreased = [r for r in caplog.records if "AIMD limiter decreased" in r.getMessage()]
    assert len(decreased) > 0, (
        "AIMD limiter never decreased; throttling did not trigger adaptive concurrency"
    )


@pytest.mark.integration
@pytest.mark.anyio
async def test_async_high_concurrency_upsert_under_real_throttling(
    async_client: AsyncPinecone,
    smoke_index_async: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Async REST: 100K vectors at max_concurrency=64 triggers rate limiting and recovers."""
    caplog.set_level(logging.DEBUG, logger="pinecone._internal.http_client")
    caplog.set_level(logging.DEBUG, logger="pinecone._internal.adaptive")

    index = await async_client.index(name=smoke_index_async)
    vectors: list[tuple[str, list[float]]] = [(f"id-{i}", [0.1] * 1536) for i in range(100_000)]

    start = time.monotonic()
    await index.upsert(vectors=vectors, batch_size=100, max_concurrency=64, show_progress=False)
    elapsed = time.monotonic() - start

    # Operation must complete successfully.
    stats = await index.describe_index_stats()
    assert stats.total_vector_count >= 100_000

    # Must complete in reasonable wall-clock time (3 minutes; serverless quota
    # recovery + AIMD ramp; exact bound is empirical — adjust on first run).
    assert elapsed < 180, f"upsert took {elapsed:.1f}s, expected < 180s"

    # We must have actually hit the rate limiter (otherwise the test isn't
    # exercising the retry path).
    throttled = [r for r in caplog.records if "Throttled response" in r.getMessage()]
    assert len(throttled) > 0, (
        "test did not trigger rate limiting; increase concurrency or batch count"
    )

    # AIMD must have actually engaged.
    decreased = [r for r in caplog.records if "AIMD limiter decreased" in r.getMessage()]
    assert len(decreased) > 0, (
        "AIMD limiter never decreased; throttling did not trigger adaptive concurrency"
    )


@pytest.mark.integration
def test_grpc_high_concurrency_upsert_under_real_throttling(
    client: Pinecone,
    smoke_index_grpc: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """gRPC: 100K vectors at max_concurrency=64 triggers rate limiting via the shared AIMD limiter.

    Phase C (DX-0160) wired gRPC to the same _AdaptiveLimiterRegistry as REST, so
    the AIMD decrease assertion validates end-to-end gRPC throttle propagation.
    """
    caplog.set_level(logging.DEBUG, logger="pinecone._internal.http_client")
    caplog.set_level(logging.DEBUG, logger="pinecone._internal.adaptive")

    index = client.index(name=smoke_index_grpc, grpc=True)
    vectors: list[tuple[str, list[float]]] = [(f"id-{i}", [0.1] * 1536) for i in range(100_000)]

    start = time.monotonic()
    index.upsert(vectors=vectors, batch_size=100, max_concurrency=64, show_progress=False)
    elapsed = time.monotonic() - start

    # Operation must complete successfully.
    assert index.describe_index_stats().total_vector_count >= 100_000

    # Must complete in reasonable wall-clock time (3 minutes; serverless quota
    # recovery + AIMD ramp; exact bound is empirical — adjust on first run).
    assert elapsed < 180, f"upsert took {elapsed:.1f}s, expected < 180s"

    # We must have actually hit the rate limiter (otherwise the test isn't
    # exercising the retry path).
    throttled = [r for r in caplog.records if "Throttled response" in r.getMessage()]
    assert len(throttled) > 0, (
        "test did not trigger rate limiting; increase concurrency or batch count"
    )

    # AIMD must have actually engaged (Phase C wired gRPC to the same limiter via DX-0160).
    decreased = [r for r in caplog.records if "AIMD limiter decreased" in r.getMessage()]
    assert len(decreased) > 0, (
        "AIMD limiter never decreased; throttling did not trigger adaptive concurrency"
    )
