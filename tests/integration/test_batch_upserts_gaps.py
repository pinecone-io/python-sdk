"""Integration tests for real-API batch upserts and upsert_from_dataframe gaps.

Phase 3 area tags: udf-batch-size, udf-concurrency, udf-total-timeout,
udf-on-error, udf-overwrite, udf-metadata, udf-sparse, udf-namespace,
udf-grpc.

Covers the admission-gate rework on main: upsert_from_dataframe with
batch_size / max_concurrency / total_timeout / on_error, verifying counts
through describe_index_stats and fetch on a real serverless index, across
the sync REST and gRPC transports.

The pandas-gated unit/smoke coverage exercises parameter validation and
batch partitioning with a Mock backend; this file drives the real API.
"""

from __future__ import annotations

import math

import pytest

pd = pytest.importorskip("pandas", reason="pandas required for upsert_from_dataframe")

from pinecone import Pinecone, PineconeTimeoutError, PineconeValueError  # noqa: E402
from pinecone.models.indexes.specs import ServerlessSpec  # noqa: E402
from tests.integration.conftest import cleanup_resource, poll_until, unique_name  # noqa: E402

DIM = 8


def _frame(n_rows: int) -> pd.DataFrame:
    """Build a DataFrame of dense vectors."""
    rows = [
        {"id": f"udf-{i}", "values": [0.1 * (i % 10) + j * 0.001 for j in range(DIM)]}
        for i in range(n_rows)
    ]
    return pd.DataFrame(rows)


def _frame_with_sparse(n: int) -> pd.DataFrame:
    """DataFrame with ``sparse_values`` and ``metadata`` columns."""
    rows = [
        {
            "id": f"udf-sparse-{i}",
            "values": [0.1 + i * 0.01 + j * 0.001 for j in range(DIM)],
            "sparse_values": {"indices": [0, 3], "values": [0.9, 0.8]},
            "metadata": {"idx": i},
        }
        for i in range(n)
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# upsert_from_dataframe — sync REST
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_upsert_from_dataframe_rest_batch_size(client: Pinecone) -> None:
    """Multiple batches of a DataFrame, verified via stats."""
    name = unique_name("ef-udf-bs")
    try:
        client.indexes.create(
            name=name,
            dimension=DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )
        index = client.index(name=name)

        resp = index.upsert_from_dataframe(_frame(100), batch_size=20, show_progress=False)
        assert resp.upserted_count == 100, f"expected 100, got {resp.upserted_count}"
        assert resp.failed_item_count == 0

        stats = poll_until(
            query_fn=lambda: index.describe_index_stats(),
            check_fn=lambda r: r.total_vector_count >= 100,
            timeout=120,
            description=f"udf 100 rows visible in stats ({name})",
        )
        assert stats.total_vector_count == 100
    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


@pytest.mark.integration
def test_udf_max_concurrency_accepted(client: Pinecone) -> None:
    """Non-default max_concurrency path round-trips end-to-end."""
    name = unique_name("ef-udf-mc")
    try:
        client.indexes.create(
            name=name,
            dimension=DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )
        index = client.index(name=name)

        resp = index.upsert_from_dataframe(
            _frame(40), batch_size=5, max_concurrency=4, show_progress=False
        )
        assert resp.upserted_count == 40
        stats = poll_until(
            query_fn=lambda: index.describe_index_stats(),
            check_fn=lambda r: r.total_vector_count >= 40,
            timeout=120,
            description=f"udf max_concurrency 40 rows in stats ({name})",
        )
        assert stats.total_vector_count == 40
    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


@pytest.mark.integration
def test_udf_metadata_and_sparse_roundtrip(client: Pinecone) -> None:
    """udf with metadata AND sparse_values round-trips via fetch."""
    name = unique_name("ef-udf-ms")
    try:
        client.indexes.create(
            name=name,
            dimension=DIM,
            metric="dotproduct",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )
        index = client.index(name=name)

        df = _frame_with_sparse(5)
        resp = index.upsert_from_dataframe(df, namespace="sparse-ns", show_progress=False)
        assert resp.upserted_count == 5

        fetched = poll_until(
            query_fn=lambda: index.fetch(
                ids=[f"udf-sparse-{i}" for i in range(5)], namespace="sparse-ns"
            ),
            check_fn=lambda r: len(r.vectors) == 5,
            timeout=120,
            description="udf sparse vectors fetchable",
        )
        for i in range(5):
            v = fetched.vectors[f"udf-sparse-{i}"]
            assert v.sparse_values is not None, f"sparse_values missing for row {i}"
            assert v.sparse_values.indices == [0, 3]
            assert v.metadata is not None and v.metadata.get("idx") == i
            assert len(v.values) == DIM
    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


@pytest.mark.integration
def test_udf_overwrite_duplicates_last_write_wins(client: Pinecone) -> None:
    """Duplicate IDs across udf batches: last row wins; dup ids dedupe in stats."""
    name = unique_name("ef-udf-ow")
    try:
        client.indexes.create(
            name=name,
            dimension=DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )
        index = client.index(name=name)

        rows: list[dict] = [{"id": "dup", "values": [0.1 * j for j in range(DIM)]}]
        rows += [
            {"id": f"ow-{i}", "values": [0.01 * i + j * 0.001 for j in range(DIM)]}
            for i in range(49)
        ]
        # overwrite 'dup' last with a distinct signature
        rows.append({"id": "dup", "values": [0.99] * DIM})
        df = pd.DataFrame(rows)

        resp = index.upsert_from_dataframe(df, batch_size=10, show_progress=False)
        # 51 rows processed (50 unique ids; 'dup' appears twice — the first row and
        # the trailing overwrite row are each upsert operations, last-write-wins).
        assert resp.upserted_count == 51, f"expected 51 rows upserted, got {resp.upserted_count}"

        fetched = poll_until(
            query_fn=lambda: index.fetch(ids=["dup"]),
            check_fn=lambda r: "dup" in r.vectors,
            timeout=120,
            description=f"dup id fetchable ({name})",
        )
        dup = fetched.vectors["dup"]
        assert all(math.isclose(x, 0.99, rel_tol=1e-5) for x in dup.values), (
            f"expected last-write [0.99]*{DIM}, got {dup.values[:4]}..."
        )

        stats = poll_until(
            query_fn=lambda: index.describe_index_stats(),
            check_fn=lambda r: r.total_vector_count >= 50,
            timeout=120,
            description=f"50 unique rows in stats ({name})",
        )
        assert stats.total_vector_count == 50
    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# upsert_from_dataframe — total_timeout + on_error
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_udf_total_timeout_expires_reports_abandoned(client: Pinecone) -> None:
    """total_timeout=0 abandons all batches; surfaced as failed_items.

    Verifies the whole-ingest deadline is honored: with on_error='collect'
    (default) no exception propagates and abandoned rows are reported in
    failed_items / failed_item_count rather than silently dropped.
    """
    name = unique_name("ef-udf-tt")
    try:
        client.indexes.create(
            name=name,
            dimension=DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )
        index = client.index(name=name)

        resp = index.upsert_from_dataframe(
            _frame(12), batch_size=2, total_timeout=0, show_progress=False
        )
        assert resp.upserted_count == 0, f"expected 0 upserted under 0s, got {resp.upserted_count}"
        assert resp.failed_item_count == 12, (
            f"expected 12 failed_items under 0s deadline, got {resp.failed_item_count}"
        )
        assert resp.has_errors
        tt_errors = [e for e in resp.errors if "total_timeout" in e.error_message]
        assert len(tt_errors) >= 1, "expected an abandoned-batch error mentioning total_timeout"
    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


@pytest.mark.integration
def test_udf_on_error_raise_aggregates(client: Pinecone) -> None:
    """on_error='raise' re-raises the lowest-indexed failure with .response."""
    name = unique_name("ef-udf-raise")
    try:
        client.indexes.create(
            name=name,
            dimension=DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )
        index = client.index(name=name)

        with pytest.raises(PineconeTimeoutError) as exc_info:
            index.upsert_from_dataframe(
                _frame(5), batch_size=5, total_timeout=0, show_progress=False, on_error="raise"
            )
        exc = exc_info.value
        assert "total_timeout" in str(exc)
        resp = getattr(exc, "response", None)
        assert resp is not None, "expected partial UpsertResponse attached to raised error"
        assert resp.failed_item_count == 5
    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# upsert_from_dataframe — gRPC
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_udf_grpc_batch(client: Pinecone) -> None:
    """udf through a gRPC index handle, verified via stats."""
    name = unique_name("ef-udf-grpc")
    try:
        client.indexes.create(
            name=name,
            dimension=DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )
        index = client.index(name=name, grpc=True)

        resp = index.upsert_from_dataframe(_frame(60), batch_size=20, show_progress=False)
        assert resp.upserted_count == 60

        stats = poll_until(
            query_fn=lambda: index.describe_index_stats(),
            check_fn=lambda r: r.total_vector_count >= 60,
            timeout=120,
            description=f"udf grpc 60 rows in stats ({name})",
        )
        assert stats.total_vector_count == 60
    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# upsert_from_dataframe — input validation against a dummy host
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_udf_validation_no_api_call(client: Pinecone) -> None:
    """Bad df / batch_size / max_concurrency raise PineconeValueError pre-call."""
    index = client.index(host="https://dummy.example.com")
    with pytest.raises(PineconeValueError):
        index.upsert_from_dataframe("not-a-dataframe")  # type: ignore[arg-type]
    with pytest.raises(PineconeValueError):
        index.upsert_from_dataframe(_frame(2), batch_size=0)
    with pytest.raises(PineconeValueError):
        index.upsert_from_dataframe(_frame(2), max_concurrency=0)
    with pytest.raises(PineconeValueError):
        index.upsert_from_dataframe(_frame(2), max_concurrency=65)
