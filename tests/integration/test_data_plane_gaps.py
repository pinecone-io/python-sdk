"""Integration tests for sync data-plane batching/retry code paths.

Area tag: batching-concurrency.

Coverage gap: test_data_plane.py and test_data_plane_extended.py exercise
single-batch REST/gRPC upsert, query, fetch, delete, update, but never the
batched parallel code path (``Index.upsert(batch_size=..., max_concurrency=...)``,
``GrpcIndex.upsert(batch_size=...)``, ``upsert_from_dataframe``) that runs through
``pinecone._internal.batch.batch_execute`` / ``bulk_execute_sync``. Those paths
are the ones with the new ThreadPoolExecutor + adaptive-limiter + total_timeout
logic. These tests drive them against a real index on both transports.

The shared index comes from :func:`legacy_index_factory`, not from
``pc.indexes.create``: 2026-07 has no way to create an index the vectors API
will serve, and every write here is a vectors-API call. See
:mod:`tests.integration.legacy_index` for the sanctioned pattern. The fixture
calls ``assert_serves_vectors_api`` once, because a document-schema index
refuses writes while leaving ``fetch`` succeeding-but-empty, which would make
every count assertion below pass against data that was never there.
"""

from __future__ import annotations

import uuid

import pytest

from pinecone import Pinecone
from pinecone.errors import PineconeValueError
from tests.integration.conftest import (
    LegacyIndexFactory,
    poll_until,
)
from tests.integration.legacy_index import assert_serves_vectors_api


@pytest.fixture(scope="module")
def shared_index_dim2(client: Pinecone, legacy_index_factory: LegacyIndexFactory) -> str:
    """Shared legacy index (dim=2, cosine) reused across all tests in this module."""
    index = legacy_index_factory(dimension=2)
    assert_serves_vectors_api(client, index)
    return index.name


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
