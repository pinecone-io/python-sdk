"""Integration tests for sync data-plane batching/retry code paths.

Area tag: batching-concurrency.

Coverage gap: test_data_plane.py and test_data_plane_extended.py exercise
single-batch REST/gRPC upsert, query, fetch, delete, update, but never the
batched parallel code path (``Index.upsert(batch_size=..., max_concurrency=...)``,
``GrpcIndex.upsert(batch_size=...)``, ``upsert_from_dataframe``) that runs through
``pinecone._internal.batch.batch_execute`` / ``bulk_execute_sync``. Those paths
are the ones with the new ThreadPoolExecutor + adaptive-limiter + total_timeout
logic. These tests drive them against real serverless indexes on both transports.
"""

from __future__ import annotations

import uuid
from collections.abc import Generator

import pytest

from pinecone import Pinecone
from pinecone.errors import PineconeValueError
from pinecone.models.indexes.specs import ServerlessSpec
from tests.integration.conftest import (
    ensure_index_deleted,
    poll_until,
    unique_name,
)


@pytest.fixture(scope="module")
def shared_index_dim2(client: Pinecone) -> Generator[str, None, None]:
    """Shared serverless index (dim=2, cosine) reused across all tests in this module."""
    name = unique_name("idx-gap-shared")
    client.indexes.create(
        name=name,
        dimension=2,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        timeout=300,
    )
    try:
        yield name
    finally:
        ensure_index_deleted(client, name)


def _make_vectors(n: int, prefix: str) -> list[dict]:
    # Offset so no vector is all-zero (backend rejects zero vectors).
    return [
        {
            "id": f"{prefix}-{i:03d}",
            "values": [0.5 + 0.01 * i, 0.7 - 0.01 * i],
            "metadata": {"i": i},
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Batched upsert via batch_size + max_concurrency (multi-batch fan-out)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize("grpc", [False, True], ids=["rest", "grpc"])
def test_batched_upsert_fanout_reports_counts(
    client: Pinecone, shared_index_dim2: str, grpc: bool
) -> None:
    """Batched upsert with batch_size < n splits into several parallel batches and
    reports correct aggregate counts on both transports."""
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    index = client.index(name=shared_index_dim2, grpc=grpc)
    n = 23
    vectors = _make_vectors(n, prefix="fan")

    resp = index.upsert(
        vectors=vectors,
        namespace=ns,
        batch_size=10,
        max_concurrency=3,
        show_progress=False,
    )

    # Aggregate counters across the concurrent batches.
    assert resp.upserted_count == n
    assert resp.total_item_count == n
    assert resp.failed_item_count == 0
    assert resp.total_batch_count == 3  # ceil(23 / 10)
    assert resp.successful_batch_count == 3
    assert resp.failed_batch_count == 0
    assert resp.errors == []

    # Verify everything actually landed server-side.
    got = poll_until(
        query_fn=lambda: index.fetch(ids=[v["id"] for v in vectors], namespace=ns),
        check_fn=lambda r: len(r.vectors) == n,
        timeout=120,
        description="all 23 batched vectors fetchable",
    )
    assert len(got.vectors) == n


@pytest.mark.integration
@pytest.mark.parametrize("grpc", [False, True], ids=["rest", "grpc"])
def test_batched_upsert_single_batch_reports(
    client: Pinecone, shared_index_dim2: str, grpc: bool
) -> None:
    """batch_size >= n produces exactly one batch and correct counts."""
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    index = client.index(name=shared_index_dim2, grpc=grpc)
    vectors = _make_vectors(7, prefix="one")

    resp = index.upsert(
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

    poll_until(
        query_fn=lambda: index.fetch(ids=[v["id"] for v in vectors], namespace=ns),
        check_fn=lambda r: len(r.vectors) == 7,
        timeout=120,
        description="6 batched vectors fetchable",
    )


# ---------------------------------------------------------------------------
# upsert_from_dataframe with batch_size + max_concurrency
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize("grpc", [False, True], ids=["rest", "grpc"])
def test_upsert_from_dataframe_batched(
    client: Pinecone, shared_index_dim2: str, grpc: bool
) -> None:
    pandas = pytest.importorskip("pandas")
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    index = client.index(name=shared_index_dim2, grpc=grpc)

    data = [
        {"id": f"df-{i:03d}", "values": [0.5 + 0.01 * i, 0.7 - 0.01 * i], "metadata": {"i": i}}
        for i in range(13)
    ]
    df = pandas.DataFrame(data)

    resp = index.upsert_from_dataframe(
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
    poll_until(
        query_fn=lambda: index.fetch(ids=ids, namespace=ns),
        check_fn=lambda r: len(r.vectors) == 13,
        timeout=120,
        description="all 13 dataframe-upserted vectors fetchable",
    )


# ---------------------------------------------------------------------------
# Validation of batch params (both transports)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize("grpc", [False, True], ids=["rest", "grpc"])
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
def test_batched_upsert_rejects_bad_params(
    client: Pinecone, shared_index_dim2: str, grpc: bool, kwargs: dict
) -> None:
    index = client.index(name=shared_index_dim2, grpc=grpc)
    with pytest.raises(PineconeValueError):
        index.upsert(
            vectors=_make_vectors(3, "bad"), namespace="ns-bad", show_progress=False, **kwargs
        )


@pytest.mark.integration
def test_batched_upsert_on_error_raise_path(client: Pinecone, shared_index_dim2: str) -> None:
    """Default on_error for batched upsert is collect; nothing should raise on a clean run."""
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    index = client.index(name=shared_index_dim2)
    vectors = _make_vectors(6, prefix="collect")
    resp = index.upsert(
        vectors=vectors,
        namespace=ns,
        batch_size=2,
        max_concurrency=1,
        show_progress=False,
    )
    assert resp.upserted_count == 6
    assert not resp.has_errors
