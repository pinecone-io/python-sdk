"""Integration tests for real-API batch upserts and upsert_from_dataframe gaps.

Phase 3 area tags: udf-batch-size, udf-concurrency, udf-total-timeout,
udf-on-error, udf-overwrite, udf-metadata, udf-sparse, udf-namespace,
udf-grpc.

Covers the admission-gate rework: upsert_from_dataframe with batch_size /
max_concurrency / total_timeout / on_error, verifying counts through
describe_index_stats and fetch, across the sync REST and gRPC transports.

The pandas-gated unit/smoke coverage exercises parameter validation and
batch partitioning with a Mock backend; this file drives the real API.

The shared indexes come from :func:`legacy_index_factory`, not from
``pc.indexes.create``: 2026-07 has no way to create an index the vectors API
will serve, and every write here is a vectors-API call. See
:mod:`tests.integration.legacy_index` for the sanctioned pattern. Each
fixture calls ``assert_serves_vectors_api`` once, because a document-schema
index refuses writes while leaving ``fetch`` and ``describe_index_stats``
succeeding-but-empty, which would make every count assertion below pass
against data that was never there.

Because the indexes are shared per shape for the whole session, every test
writes into its own namespace and asserts on
``describe_index_stats().namespaces[ns]`` rather than on
``total_vector_count``, which other tests also contribute to.
"""

from __future__ import annotations

import math
import uuid
from typing import Any

import pytest

pd = pytest.importorskip("pandas", reason="pandas required for upsert_from_dataframe")

from pinecone import Pinecone, PineconeTimeoutError, PineconeValueError  # noqa: E402
from tests.integration.conftest import LegacyIndexFactory, poll_until  # noqa: E402
from tests.integration.legacy_index import assert_serves_vectors_api  # noqa: E402

DIM = 8


@pytest.fixture(scope="module")
def shared_index_dim8(client: Pinecone, legacy_index_factory: LegacyIndexFactory) -> str:
    """Shared legacy index (dim=8, cosine) for the dense udf arms."""
    index = legacy_index_factory(dimension=DIM)
    assert_serves_vectors_api(client, index)
    return index.name


@pytest.fixture(scope="module")
def shared_index_dim8_dotproduct(client: Pinecone, legacy_index_factory: LegacyIndexFactory) -> str:
    """Shared legacy index (dim=8, dotproduct) for the sparse-values arm."""
    index = legacy_index_factory(dimension=DIM, metric="dotproduct")
    assert_serves_vectors_api(client, index)
    return index.name


def _ns(tag: str) -> str:
    return f"udf-{tag}-{uuid.uuid4().hex[:8]}"


def _assert_fully_upserted(resp: Any, expected: int) -> None:
    """Assert every row landed, naming the per-batch errors when they did not.

    ``upsert_from_dataframe`` collects batch failures rather than raising, so a
    bare ``assert resp.upserted_count == N`` reports ``0 == 60`` and discards
    the only evidence of why — which is exactly what a partial failure against
    the real backend leaves behind.
    """
    detail = "; ".join(
        f"batch {err.batch_index}: {type(err.error).__name__}: {err.error_message}"
        for err in resp.errors
    )
    assert resp.upserted_count == expected, (
        f"expected {expected} rows upserted, got {resp.upserted_count} "
        f"({resp.failed_item_count} failed) — {detail or 'no errors reported'}"
    )
    assert resp.failed_item_count == 0, detail


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
def test_upsert_from_dataframe_rest_batch_size(client: Pinecone, shared_index_dim8: str) -> None:
    """Multiple batches of a DataFrame, verified via stats."""
    ns = _ns("bs")
    index = client.index(name=shared_index_dim8)

    resp = index.upsert_from_dataframe(
        _frame(100), namespace=ns, batch_size=20, show_progress=False
    )
    _assert_fully_upserted(resp, 100)

    stats = poll_until(
        query_fn=lambda: index.describe_index_stats(),
        check_fn=lambda r: ns in r.namespaces and r.namespaces[ns].vector_count >= 100,
        timeout=120,
        description=f"udf 100 rows visible in stats ({ns})",
    )
    assert stats.namespaces[ns].vector_count == 100


@pytest.mark.integration
def test_udf_max_concurrency_accepted(client: Pinecone, shared_index_dim8: str) -> None:
    """Non-default max_concurrency path round-trips end-to-end."""
    ns = _ns("mc")
    index = client.index(name=shared_index_dim8)

    resp = index.upsert_from_dataframe(
        _frame(40), namespace=ns, batch_size=5, max_concurrency=4, show_progress=False
    )
    _assert_fully_upserted(resp, 40)
    stats = poll_until(
        query_fn=lambda: index.describe_index_stats(),
        check_fn=lambda r: ns in r.namespaces and r.namespaces[ns].vector_count >= 40,
        timeout=120,
        description=f"udf max_concurrency 40 rows in stats ({ns})",
    )
    assert stats.namespaces[ns].vector_count == 40


@pytest.mark.integration
def test_udf_metadata_and_sparse_roundtrip(
    client: Pinecone, shared_index_dim8_dotproduct: str
) -> None:
    """udf with metadata AND sparse_values round-trips via fetch."""
    ns = _ns("sparse")
    index = client.index(name=shared_index_dim8_dotproduct)

    df = _frame_with_sparse(5)
    resp = index.upsert_from_dataframe(df, namespace=ns, show_progress=False)
    _assert_fully_upserted(resp, 5)

    fetched = poll_until(
        query_fn=lambda: index.fetch(ids=[f"udf-sparse-{i}" for i in range(5)], namespace=ns),
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


@pytest.mark.integration
def test_udf_overwrite_duplicates_last_write_wins(client: Pinecone, shared_index_dim8: str) -> None:
    """Duplicate IDs across udf batches: last row wins; dup ids dedupe in stats.

    ``max_concurrency=1`` is load-bearing rather than tuning. "Last row wins" is
    a claim about the order batches reach the server, and at the default
    ``max_concurrency=8`` all six batches of this frame are in flight at once,
    so the batch carrying the first ``dup`` row can settle *after* the one
    carrying the trailing row and the earlier value is what survives.
    Serializing submission is what makes the trailing row genuinely last; the
    guarantee under test is the server's overwrite, not the client's scheduler.

    The fetch then polls on the *value*: the id is already present from the
    first write, so a presence check can return while the overwrite is still
    working its way to the read path.
    """
    ns = _ns("ow")
    index = client.index(name=shared_index_dim8)

    rows: list[dict] = [{"id": "dup", "values": [0.1 * j for j in range(DIM)]}]
    rows += [
        {"id": f"ow-{i}", "values": [0.01 * i + j * 0.001 for j in range(DIM)]} for i in range(49)
    ]
    # overwrite 'dup' last with a distinct signature
    rows.append({"id": "dup", "values": [0.99] * DIM})
    df = pd.DataFrame(rows)

    resp = index.upsert_from_dataframe(
        df, namespace=ns, batch_size=10, max_concurrency=1, show_progress=False
    )
    # 51 rows processed (50 unique ids; 'dup' appears twice — the first row and
    # the trailing overwrite row are each upsert operations, last-write-wins).
    _assert_fully_upserted(resp, 51)

    def _carries_last_write(response: Any) -> bool:
        vector = response.vectors.get("dup")
        return vector is not None and all(
            math.isclose(x, 0.99, rel_tol=1e-5) for x in vector.values
        )

    fetched = poll_until(
        query_fn=lambda: index.fetch(ids=["dup"], namespace=ns),
        check_fn=_carries_last_write,
        timeout=120,
        description=f"dup id carries the last write ({ns})",
    )
    dup = fetched.vectors["dup"]
    assert all(math.isclose(x, 0.99, rel_tol=1e-5) for x in dup.values), (
        f"expected last-write [0.99]*{DIM}, got {dup.values[:4]}..."
    )

    stats = poll_until(
        query_fn=lambda: index.describe_index_stats(),
        check_fn=lambda r: ns in r.namespaces and r.namespaces[ns].vector_count >= 50,
        timeout=120,
        description=f"50 unique rows in stats ({ns})",
    )
    assert stats.namespaces[ns].vector_count == 50


# ---------------------------------------------------------------------------
# upsert_from_dataframe — total_timeout + on_error
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_udf_total_timeout_expires_reports_abandoned(
    client: Pinecone, shared_index_dim8: str
) -> None:
    """total_timeout=0 abandons all batches; surfaced as failed_items.

    Verifies the whole-ingest deadline is honored: with on_error='collect'
    (default) no exception propagates and abandoned rows are reported in
    failed_items / failed_item_count rather than silently dropped.
    """
    index = client.index(name=shared_index_dim8)

    resp = index.upsert_from_dataframe(
        _frame(12), namespace=_ns("tt"), batch_size=2, total_timeout=0, show_progress=False
    )
    assert resp.upserted_count == 0, f"expected 0 upserted under 0s, got {resp.upserted_count}"
    assert resp.failed_item_count == 12, (
        f"expected 12 failed_items under 0s deadline, got {resp.failed_item_count}"
    )
    assert resp.has_errors
    tt_errors = [e for e in resp.errors if "total_timeout" in e.error_message]
    assert len(tt_errors) >= 1, "expected an abandoned-batch error mentioning total_timeout"


@pytest.mark.integration
def test_udf_on_error_raise_aggregates(client: Pinecone, shared_index_dim8: str) -> None:
    """on_error='raise' re-raises the lowest-indexed failure with .response."""
    index = client.index(name=shared_index_dim8)

    with pytest.raises(PineconeTimeoutError) as exc_info:
        index.upsert_from_dataframe(
            _frame(5),
            namespace=_ns("raise"),
            batch_size=5,
            total_timeout=0,
            show_progress=False,
            on_error="raise",
        )
    exc = exc_info.value
    assert "total_timeout" in str(exc)
    resp = getattr(exc, "response", None)
    assert resp is not None, "expected partial UpsertResponse attached to raised error"
    assert resp.failed_item_count == 5


# ---------------------------------------------------------------------------
# upsert_from_dataframe — gRPC
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_udf_grpc_batch(client: Pinecone, shared_index_dim8: str) -> None:
    """udf through a gRPC index handle, verified via stats."""
    ns = _ns("grpc")
    index = client.index(name=shared_index_dim8, grpc=True)

    resp = index.upsert_from_dataframe(_frame(60), namespace=ns, batch_size=20, show_progress=False)
    _assert_fully_upserted(resp, 60)

    stats = poll_until(
        query_fn=lambda: index.describe_index_stats(),
        check_fn=lambda r: ns in r.namespaces and r.namespaces[ns].vector_count >= 60,
        timeout=120,
        description=f"udf grpc 60 rows in stats ({ns})",
    )
    assert stats.namespaces[ns].vector_count == 60


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
