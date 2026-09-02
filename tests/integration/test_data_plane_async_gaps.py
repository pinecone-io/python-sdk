"""Async data-plane gap coverage: batched upsert + admission control.

Mirrors the sync coverage in ``test_data_plane_gaps.py`` for the async
client. Specifically covers the async ``Index.upsert(batch_size=...,
max_concurrency=...)`` and ``AsyncIndex.upsert_from_dataframe(...)`` code
paths, which run per-batch requests concurrently under an asyncio concurrency
limit (``bulk_execute_async``). These paths were previously untested for the
async client.

Uses ``@pytest.mark.anyio`` (not ``@pytest.mark.asyncio``): pytest-anyio owns
the event loop. Creates real serverless indexes; fixtures clean up.
"""

from __future__ import annotations

import uuid
from collections.abc import Generator

import pytest

from pinecone import AsyncIndex, AsyncPinecone, Pinecone
from pinecone.errors import PineconeValueError
from pinecone.models.indexes.specs import ServerlessSpec
from pinecone.models.vectors.responses import UpsertResponse
from tests.integration.conftest import (
    async_poll_until,
    ensure_index_deleted,
    unique_name,
)


def _make_vectors(n: int, prefix: str):
    return [
        {
            "id": f"{prefix}-{i:03d}",
            "values": [0.5 + 0.01 * i, 0.7 - 0.01 * i],
            "metadata": {"i": i},
        }
        for i in range(n)
    ]


@pytest.fixture(scope="module")
def shared_index_dim2(api_key: str) -> Generator[str, None, None]:
    """Shared serverless index (dim=2, cosine) reused across all tests in this module."""
    sync_pc = Pinecone(api_key=api_key)
    name = unique_name("idx-async-gaps-dim2")
    sync_pc.indexes.create(
        name=name,
        dimension=2,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        timeout=300,
    )
    try:
        yield name
    finally:
        ensure_index_deleted(sync_pc, name)


# ---------------------------------------------------------------------------
# Batched async upsert via batch_size + max_concurrency (multi-batch fan-out)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_async_batched_upsert_fanout_reports_counts(
    async_client: AsyncPinecone, shared_index_dim2: str
) -> None:
    """async upsert with batch_size < n splits into concurrent batches and reports
    correct aggregate counts (the async admission-controlled fan-out path)."""
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    idx = await async_client.index(name=shared_index_dim2)
    assert isinstance(idx, AsyncIndex)
    n = 23
    vectors = _make_vectors(n, prefix="fan")

    resp = await idx.upsert(
        vectors=vectors,
        namespace=ns,
        batch_size=10,
        max_concurrency=3,
        show_progress=False,
    )

    assert isinstance(resp, UpsertResponse)
    assert resp.upserted_count == n
    assert resp.total_item_count == n
    assert resp.failed_item_count == 0
    assert resp.total_batch_count == 3  # ceil(23 / 10)
    assert resp.successful_batch_count == 3
    assert resp.failed_batch_count == 0
    assert resp.errors == []

    got = await async_poll_until(
        query_fn=lambda: idx.fetch(ids=[v["id"] for v in vectors], namespace=ns),
        check_fn=lambda r: len(r.vectors) == n,
        timeout=120,
        description="all 23 batched vectors fetchable",
    )
    assert len(got.vectors) == n


@pytest.mark.integration
@pytest.mark.anyio
async def test_async_batched_upsert_single_batch_reports(
    async_client: AsyncPinecone, shared_index_dim2: str
) -> None:
    """batch_size >= n produces exactly one batch and correct counts."""
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    idx = await async_client.index(name=shared_index_dim2)
    vectors = _make_vectors(7, prefix="one")

    resp = await idx.upsert(
        vectors=vectors,
        namespace=ns,
        batch_size=100,  # larger than n -> single batch
        max_concurrency=2,
        show_progress=False,
    )

    assert resp.upserted_count == 7
    assert resp.total_batch_count == 1
    assert resp.successful_batch_count == 1
    assert resp.failed_batch_count == 0

    await async_poll_until(
        query_fn=lambda: idx.fetch(ids=[v["id"] for v in vectors], namespace=ns),
        check_fn=lambda r: len(r.vectors) == 7,
        timeout=120,
        description="single-batch vectors fetchable",
    )


# ---------------------------------------------------------------------------
# async upsert_from_dataframe with batch_size + max_concurrency
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_async_upsert_from_dataframe_batched(
    async_client: AsyncPinecone, shared_index_dim2: str
) -> None:
    pandas = pytest.importorskip("pandas")
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    idx = await async_client.index(name=shared_index_dim2)

    data = [
        {"id": f"df-{i:03d}", "values": [0.5 + 0.01 * i, 0.7 - 0.01 * i], "metadata": {"i": i}}
        for i in range(13)
    ]
    df = pandas.DataFrame(data)

    resp = await idx.upsert_from_dataframe(
        df,
        namespace=ns,
        batch_size=5,
        max_concurrency=2,
        show_progress=False,
    )

    assert resp.upserted_count == 13
    assert resp.total_batch_count == 3  # ceil(13 / 5)
    assert resp.failed_batch_count == 0

    ids = [r["id"] for r in data]
    await async_poll_until(
        query_fn=lambda: idx.fetch(ids=ids, namespace=ns),
        check_fn=lambda r: len(r.vectors) == 13,
        timeout=120,
        description="all 13 dataframe-upserted vectors fetchable",
    )


# ---------------------------------------------------------------------------
# Validation of batch params on the async path
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_size": 0},
        {"batch_size": -3},
        {"batch_size": 5, "max_concurrency": 0},
        {"batch_size": 5, "max_concurrency": 65},
    ],
    ids=["batch0", "batchneg", "conc0", "conc65"],
)
async def test_async_batched_upsert_rejects_bad_params(
    async_client: AsyncPinecone, shared_index_dim2: str, kwargs: dict
) -> None:
    idx = await async_client.index(name=shared_index_dim2)
    with pytest.raises(PineconeValueError):
        await idx.upsert(
            vectors=_make_vectors(3, "bad"), namespace="ns-bad", show_progress=False, **kwargs
        )


@pytest.mark.integration
@pytest.mark.anyio
async def test_async_batched_upsert_on_error_collect_path(
    async_client: AsyncPinecone, shared_index_dim2: str
) -> None:
    """Default on_error for batched async upsert is collect; nothing raises on a clean run."""
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    idx = await async_client.index(name=shared_index_dim2)
    vectors = _make_vectors(6, prefix="collect")
    resp = await idx.upsert(
        vectors=vectors,
        namespace=ns,
        batch_size=2,
        max_concurrency=1,
        show_progress=False,
    )
    assert resp.upserted_count == 6
    assert not resp.has_errors
