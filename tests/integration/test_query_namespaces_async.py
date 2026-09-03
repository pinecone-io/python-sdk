"""Integration tests for advanced query_namespaces operations (async REST).

Phase 3 Tier 5: query-namespaces-filter, query-namespaces-many.
ET-019: query-namespaces-dedup.
DX-0075: query-namespaces-sparse (async).

The shared indexes come from :func:`legacy_index_factory`, not from
``pc.indexes.create``: 2026-07 has no way to create an index the vectors API
will serve, and ``query_namespaces`` is a vectors-API call. See
:mod:`tests.integration.legacy_index` for the sanctioned pattern. Each
shared-index fixture calls ``assert_serves_vectors_api`` once, because a
document-schema index would not fail this module loudly -- writes are refused
but ``query`` / ``fetch`` succeed and return **empty** (#322).

This module shares its session-scoped indexes with the sync twin
(test_query_namespaces.py), so every namespace literal here carries an
"a" suffix distinct from the sync module's, even where the two modules
exercise the same shape.
"""
# area tags covered: query-namespaces-filter, query-namespaces-many, query-namespaces-dedup, query-namespaces-sparse

from __future__ import annotations

import time

import pytest

from pinecone import AsyncPinecone, Pinecone
from pinecone.errors.exceptions import ValidationError
from pinecone.models.vectors.query_aggregator import QueryNamespacesResults
from pinecone.models.vectors.vector import ScoredVector
from tests.integration.conftest import LegacyIndexFactory, async_poll_until
from tests.integration.legacy_index import assert_serves_vectors_api

# ---------------------------------------------------------------------------
# Module-scoped shared indexes
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def shared_index_dim2(client: Pinecone, legacy_index_factory: LegacyIndexFactory) -> str:
    index = legacy_index_factory(dimension=2)
    assert_serves_vectors_api(client, index)
    return index.name


@pytest.fixture(scope="module")
def shared_index_dim3(client: Pinecone, legacy_index_factory: LegacyIndexFactory) -> str:
    index = legacy_index_factory(dimension=3)
    assert_serves_vectors_api(client, index)
    return index.name


@pytest.fixture(scope="module")
def shared_index_dim2_euclidean(client: Pinecone, legacy_index_factory: LegacyIndexFactory) -> str:
    index = legacy_index_factory(dimension=2, metric="euclidean")
    assert_serves_vectors_api(client, index)
    return index.name


@pytest.fixture(scope="module")
def shared_index_dim2_dotproduct(client: Pinecone, legacy_index_factory: LegacyIndexFactory) -> str:
    index = legacy_index_factory(dimension=2, metric="dotproduct")
    assert_serves_vectors_api(client, index)
    return index.name


@pytest.fixture(scope="module")
def shared_index_sparse(client: Pinecone, legacy_index_factory: LegacyIndexFactory) -> str:
    index = legacy_index_factory(vector_type="sparse", metric="dotproduct")
    assert_serves_vectors_api(client, index)
    return index.name


# ---------------------------------------------------------------------------
# query-namespaces-filter — REST async
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_query_namespaces_filter_rest_async(
    async_client: AsyncPinecone, shared_index_dim2: str
) -> None:
    """query_namespaces() with filter applies it per-namespace and returns metadata (REST async)."""
    idx = await async_client.index(name=shared_index_dim2)

    # Upsert comedy + drama vectors into two namespaces
    await idx.upsert(
        vectors=[
            {"id": "qnfa-ns1-com1", "values": [0.1, 0.2], "metadata": {"genre": "comedy"}},
            {"id": "qnfa-ns1-dra1", "values": [0.9, 0.8], "metadata": {"genre": "drama"}},
        ],
        namespace="qnfa-ns1",
    )
    await idx.upsert(
        vectors=[
            {"id": "qnfa-ns2-com1", "values": [0.2, 0.3], "metadata": {"genre": "comedy"}},
            {"id": "qnfa-ns2-dra1", "values": [0.8, 0.7], "metadata": {"genre": "drama"}},
        ],
        namespace="qnfa-ns2",
    )

    # Wait for all vectors in both namespaces to be queryable
    await async_poll_until(
        query_fn=lambda: idx.query(vector=[0.1, 0.2], top_k=10, namespace="qnfa-ns1"),
        check_fn=lambda r: len(r.matches) >= 2,
        timeout=120,
        description="ns1 vectors queryable (async) before query_namespaces_filter",
    )
    await async_poll_until(
        query_fn=lambda: idx.query(vector=[0.1, 0.2], top_k=10, namespace="qnfa-ns2"),
        check_fn=lambda r: len(r.matches) >= 2,
        timeout=120,
        description="ns2 vectors queryable (async) before query_namespaces_filter",
    )

    # Call query_namespaces with comedy filter and include_metadata=True
    results = await idx.query_namespaces(
        vector=[0.1, 0.2],
        namespaces=["qnfa-ns1", "qnfa-ns2"],
        metric="cosine",
        top_k=10,
        filter={"genre": {"$eq": "comedy"}},
        include_metadata=True,
    )

    # Verify result type and structure
    assert isinstance(results, QueryNamespacesResults)
    assert isinstance(results.matches, list)
    assert len(results.matches) >= 1

    # Each match is a ScoredVector
    for match in results.matches:
        assert isinstance(match, ScoredVector)
        assert isinstance(match.id, str)
        assert isinstance(match.score, float)

    # Filter must have been applied: only comedy vectors should appear
    match_ids = {m.id for m in results.matches}
    comedy_ids = {"qnfa-ns1-com1", "qnfa-ns2-com1"}
    drama_ids = {"qnfa-ns1-dra1", "qnfa-ns2-dra1"}
    assert len(match_ids & comedy_ids) >= 1
    assert match_ids.isdisjoint(drama_ids), (
        f"Drama vectors leaked through filter: {match_ids & drama_ids}"
    )

    # Metadata must be present on matches (include_metadata=True)
    for match in results.matches:
        assert match.metadata is not None
        assert "genre" in match.metadata
        assert match.metadata["genre"] == "comedy"

    # Scores should be in descending order (cosine)
    scores = [m.score for m in results.matches]
    assert scores == sorted(scores, reverse=True)

    # Per-namespace usage should be populated
    assert isinstance(results.ns_usage, dict)
    assert "qnfa-ns1" in results.ns_usage
    assert "qnfa-ns2" in results.ns_usage

    # Total usage
    assert results.usage is not None
    assert isinstance(results.usage.read_units, int)
    assert results.usage.read_units >= 2


# ---------------------------------------------------------------------------
# query-namespaces-dedup — REST async
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_query_namespaces_dedup_rest_async(
    async_client: AsyncPinecone, shared_index_dim2: str
) -> None:
    """query_namespaces() deduplicates repeated namespaces: no vector appears twice, ns_usage has one key per unique namespace (REST async).

    Verifies unified-vec-0034: duplicate entries in the namespaces list are
    removed before fan-out, so each namespace is queried exactly once.
    """
    idx = await async_client.index(name=shared_index_dim2)

    # Upsert distinct vectors into two namespaces
    await idx.upsert(
        vectors=[
            {"id": "qnda-ns1-v1", "values": [0.1, 0.9]},
            {"id": "qnda-ns1-v2", "values": [0.9, 0.1]},
        ],
        namespace="qnda-ns1",
    )
    await idx.upsert(
        vectors=[
            {"id": "qnda-ns2-v1", "values": [0.5, 0.5]},
            {"id": "qnda-ns2-v2", "values": [0.6, 0.4]},
        ],
        namespace="qnda-ns2",
    )

    # Wait for both namespaces to be queryable
    await async_poll_until(
        query_fn=lambda: idx.query(vector=[0.5, 0.5], top_k=10, namespace="qnda-ns1"),
        check_fn=lambda r: len(r.matches) >= 2,
        timeout=120,
        description="qnda-ns1 vectors queryable (async) before dedup test",
    )
    await async_poll_until(
        query_fn=lambda: idx.query(vector=[0.5, 0.5], top_k=10, namespace="qnda-ns2"),
        check_fn=lambda r: len(r.matches) >= 2,
        timeout=120,
        description="qnda-ns2 vectors queryable (async) before dedup test",
    )

    # Query with a duplicated namespaces list: ns1 appears twice
    results = await idx.query_namespaces(
        vector=[0.5, 0.5],
        namespaces=["qnda-ns1", "qnda-ns2", "qnda-ns1"],
        metric="cosine",
        top_k=10,
    )

    assert isinstance(results, QueryNamespacesResults)
    assert isinstance(results.matches, list)

    # Each match is a ScoredVector
    for match in results.matches:
        assert isinstance(match, ScoredVector)
        assert isinstance(match.id, str)
        assert isinstance(match.score, float)

    # Dedup: no vector ID should appear more than once in results
    result_ids = [m.id for m in results.matches]
    assert len(result_ids) == len(set(result_ids)), (
        f"Duplicate vector IDs in results (ns1 was queried twice): {result_ids}"
    )

    # ns_usage must have exactly 2 keys — the deduplicated set
    assert isinstance(results.ns_usage, dict)
    assert set(results.ns_usage.keys()) == {"qnda-ns1", "qnda-ns2"}, (
        f"Expected ns_usage keys {{'qnda-ns1','qnda-ns2'}}, got {set(results.ns_usage.keys())}"
    )

    # Scores must be in descending order
    scores = [m.score for m in results.matches]
    assert scores == sorted(scores, reverse=True), (
        f"Matches not sorted by descending score: {scores}"
    )


# ---------------------------------------------------------------------------
# query-namespaces-many — REST async
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_query_namespaces_many_rest_async(
    async_client: AsyncPinecone, shared_index_dim2: str
) -> None:
    """query_namespaces() across 5+ namespaces merges and sorts results; ns_usage has entry per namespace (REST async)."""
    idx = await async_client.index(name=shared_index_dim2)

    # Upsert 2 vectors into each of 5 namespaces
    namespaces = [f"qnm-ns-{i}" for i in range(5)]
    for i, ns in enumerate(namespaces):
        base = float(i) / 5.0
        await idx.upsert(
            vectors=[
                {"id": f"{ns}-v1", "values": [base, 1.0 - base]},
                {"id": f"{ns}-v2", "values": [1.0 - base, base]},
            ],
            namespace=ns,
        )

    # Wait for each namespace to have both vectors queryable
    for ns in namespaces:
        await async_poll_until(
            query_fn=lambda ns=ns: idx.query(vector=[0.5, 0.5], top_k=10, namespace=ns),
            check_fn=lambda r: len(r.matches) >= 2,
            timeout=120,
            description=f"{ns} vectors queryable (async) before query_namespaces_many",
        )

    # Query across all 5 namespaces at once
    results = await idx.query_namespaces(
        vector=[0.5, 0.5],
        namespaces=namespaces,
        metric="cosine",
        top_k=5,
    )

    # Verify result type and structure
    assert isinstance(results, QueryNamespacesResults)
    assert isinstance(results.matches, list)
    assert len(results.matches) >= 1

    # Each match must be a ScoredVector
    for match in results.matches:
        assert isinstance(match, ScoredVector)
        assert isinstance(match.id, str)
        assert isinstance(match.score, float)

    # Results must be sorted by descending score (merged across namespaces)
    scores = [m.score for m in results.matches]
    assert scores == sorted(scores, reverse=True), (
        f"Matches not sorted by descending score: {scores}"
    )

    # ns_usage must contain an entry for every queried namespace
    assert isinstance(results.ns_usage, dict)
    for ns in namespaces:
        assert ns in results.ns_usage, (
            f"Expected ns_usage entry for {ns!r}, got keys: {list(results.ns_usage.keys())}"
        )

    # Total usage must be present and reflect work across all namespaces
    assert results.usage is not None
    assert isinstance(results.usage.read_units, int)
    assert results.usage.read_units >= len(namespaces)


# ---------------------------------------------------------------------------
# query-namespaces-default-top-k — REST async
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_query_namespaces_default_top_k_rest_async(
    async_client: AsyncPinecone, shared_index_dim2: str
) -> None:
    """query_namespaces() defaults top_k to 10 when not specified (REST async).

    Verifies claim unified-vec-0028: Cross-namespace query defaults to returning
    the top 10 results when top_k is not specified.

    Strategy: upsert 7 vectors into two namespaces (14 total > 10 default), then
    call query_namespaces without top_k and assert that at most 10 matches are
    returned, confirming the default is applied.
    """
    idx = await async_client.index(name=shared_index_dim2)

    # Upsert 7 vectors into each of 2 namespaces = 14 total (exceeds default top_k=10)
    ns_a_vectors = [
        {"id": f"qtka-ns-a-{i}", "values": [float(i) / 7, 1.0 - float(i) / 7]} for i in range(7)
    ]
    ns_b_vectors = [
        {"id": f"qtka-ns-b-{i}", "values": [float(i) / 14, 1.0 - float(i) / 14]} for i in range(7)
    ]
    await idx.upsert(vectors=ns_a_vectors, namespace="qtka-ns-a")
    await idx.upsert(vectors=ns_b_vectors, namespace="qtka-ns-b")

    # Wait for all 7 vectors in each namespace to become queryable
    await async_poll_until(
        query_fn=lambda: idx.query(vector=[0.5, 0.5], top_k=10, namespace="qtka-ns-a"),
        check_fn=lambda r: len(r.matches) >= 7,
        timeout=120,
        description="all 7 qtka-ns-a vectors queryable before default-top-k test",
    )
    await async_poll_until(
        query_fn=lambda: idx.query(vector=[0.5, 0.5], top_k=10, namespace="qtka-ns-b"),
        check_fn=lambda r: len(r.matches) >= 7,
        timeout=120,
        description="all 7 qtka-ns-b vectors queryable before default-top-k test",
    )

    # Query without top_k — should use default of 10
    results = await idx.query_namespaces(
        vector=[0.5, 0.5],
        namespaces=["qtka-ns-a", "qtka-ns-b"],
        metric="cosine",
    )

    assert isinstance(results, QueryNamespacesResults)
    assert isinstance(results.matches, list)
    # Key assertion: default top_k caps results at 10 even though 14 vectors exist
    assert len(results.matches) <= 10, (
        f"Expected at most 10 matches (default top_k), got {len(results.matches)}"
    )
    assert len(results.matches) > 0, "Expected at least one match"

    # Results must be sorted by descending score
    scores = [m.score for m in results.matches]
    assert scores == sorted(scores, reverse=True), (
        f"Matches not sorted by descending score: {scores}"
    )

    # Each match is a ScoredVector
    for match in results.matches:
        assert isinstance(match, ScoredVector)
        assert isinstance(match.id, str)
        assert isinstance(match.score, float)


# ---------------------------------------------------------------------------
# query-namespaces-euclidean — ascending score ordering — REST async
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_query_namespaces_euclidean_scores_ascending_rest_async(
    async_client: AsyncPinecone, shared_index_dim2_euclidean: str
) -> None:
    """query_namespaces() with euclidean metric returns matches sorted ascending by score (REST async).

    For euclidean, lower scores indicate smaller distance (closer vectors) and should
    rank first — the opposite of cosine/dotproduct where higher scores rank first.

    Verifies claim unified-vec-0036: "Multi-namespace query results are aggregated using
    a heap-based algorithm; for cosine/dotproduct, higher scores rank first; for
    euclidean, lower scores rank first."

    Strategy:
    - Create an index with euclidean metric.
    - Upsert three vectors at known distances from the query vector [0.0, 0.0]:
        ns1: "euca-close" at [0.1, 0.0] (euclidean dist ≈ 0.1 — closest)
        ns1: "euca-far"   at [0.9, 0.0] (euclidean dist ≈ 0.9 — farthest)
        ns2: "euca-mid"   at [0.4, 0.0] (euclidean dist ≈ 0.4 — middle)
    - Query with vector [0.0, 0.0] and metric="euclidean".
    - Assert scores are non-decreasing (ascending), confirming lower scores rank first.
    """
    idx = await async_client.index(name=shared_index_dim2_euclidean)

    await idx.upsert(
        vectors=[
            {"id": "euca-close", "values": [0.1, 0.0]},
            {"id": "euca-far", "values": [0.9, 0.0]},
        ],
        namespace="euca-ns1",
    )
    await idx.upsert(
        vectors=[
            {"id": "euca-mid", "values": [0.4, 0.0]},
        ],
        namespace="euca-ns2",
    )

    # Wait for all 3 vectors to be queryable across both namespaces
    await async_poll_until(
        query_fn=lambda: idx.query(vector=[0.0, 0.0], top_k=10, namespace="euca-ns1"),
        check_fn=lambda r: len(r.matches) >= 2,
        timeout=120,
        description="euca-ns1 vectors queryable before euclidean sort test",
    )
    await async_poll_until(
        query_fn=lambda: idx.query(vector=[0.0, 0.0], top_k=10, namespace="euca-ns2"),
        check_fn=lambda r: len(r.matches) >= 1,
        timeout=120,
        description="euca-ns2 vector queryable before euclidean sort test",
    )

    results = await idx.query_namespaces(
        vector=[0.0, 0.0],
        namespaces=["euca-ns1", "euca-ns2"],
        metric="euclidean",
        top_k=5,
    )

    assert isinstance(results, QueryNamespacesResults)
    assert isinstance(results.matches, list)
    assert len(results.matches) == 3, (
        f"Expected 3 matches (all vectors), got {len(results.matches)}"
    )

    # unified-vec-0036: for euclidean, scores must be sorted ascending
    scores = [m.score for m in results.matches]
    assert scores == sorted(scores), (
        f"For euclidean metric, scores must be ascending (lower = closer); got: {scores}"
    )

    # All scores must be non-negative (euclidean distance is always >= 0)
    for score in scores:
        assert score >= 0.0, f"Euclidean distance score must be non-negative, got {score}"

    # The closest vector (euca-close at [0.1, 0.0]) must rank first
    assert results.matches[0].id == "euca-close", (
        f"Expected 'euca-close' (closest to [0,0]) to rank first; "
        f"got {results.matches[0].id} (score={results.matches[0].score:.4f})"
    )

    # The farthest vector (euca-far at [0.9, 0.0]) must rank last
    assert results.matches[-1].id == "euca-far", (
        f"Expected 'euca-far' (farthest from [0,0]) to rank last; "
        f"got {results.matches[-1].id} (score={results.matches[-1].score:.4f})"
    )

    # Verify ScoredVector structure for all matches
    for match in results.matches:
        assert isinstance(match, ScoredVector)
        assert isinstance(match.id, str)
        assert isinstance(match.score, float)


# ---------------------------------------------------------------------------
# query-namespaces include_values — REST async
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_query_namespaces_include_values_rest_async(
    async_client: AsyncPinecone, shared_index_dim3: str
) -> None:
    """query_namespaces(include_values=True) returns vector values on each match;
    omitting include_values leaves match.values as None (REST async).

    Verifies:
    - unified-vec-0023: Query results do not include vector values unless explicitly
      requested — tested via the multi-namespace fan-out path.
    - unified-vec-0016: Can query multiple namespaces and return a merged result set
      with all optional fields populated when requested.
    """
    idx = await async_client.index(name=shared_index_dim3)

    # Upsert 2 vectors into each of 2 namespaces with known values
    await idx.upsert(
        vectors=[
            {"id": "iva-ns1-v1", "values": [0.1, 0.2, 0.3]},
            {"id": "iva-ns1-v2", "values": [0.4, 0.5, 0.6]},
        ],
        namespace="iva-ns1",
    )
    await idx.upsert(
        vectors=[
            {"id": "iva-ns2-v1", "values": [0.7, 0.8, 0.9]},
            {"id": "iva-ns2-v2", "values": [0.2, 0.3, 0.4]},
        ],
        namespace="iva-ns2",
    )

    # Wait for all vectors to be queryable in both namespaces
    await async_poll_until(
        query_fn=lambda: idx.query(vector=[0.1, 0.2, 0.3], top_k=10, namespace="iva-ns1"),
        check_fn=lambda r: len(r.matches) >= 2,
        timeout=120,
        description="iva-ns1 vectors queryable before include_values test",
    )
    await async_poll_until(
        query_fn=lambda: idx.query(vector=[0.1, 0.2, 0.3], top_k=10, namespace="iva-ns2"),
        check_fn=lambda r: len(r.matches) >= 2,
        timeout=120,
        description="iva-ns2 vectors queryable before include_values test",
    )

    # --- Part 1: include_values=True → values present on every match ---
    results_with_values = await idx.query_namespaces(
        vector=[0.1, 0.2, 0.3],
        namespaces=["iva-ns1", "iva-ns2"],
        metric="cosine",
        top_k=4,
        include_values=True,
    )

    assert isinstance(results_with_values, QueryNamespacesResults)
    assert len(results_with_values.matches) >= 1, (
        "Expected at least one match when include_values=True"
    )

    for match in results_with_values.matches:
        assert isinstance(match, ScoredVector)
        # values must be a non-empty list of floats when include_values=True
        assert match.values is not None, (
            f"match.values must not be None when include_values=True (id={match.id!r})"
        )
        assert isinstance(match.values, list), (
            f"match.values must be a list, got {type(match.values)} (id={match.id!r})"
        )
        assert len(match.values) == 3, (
            f"match.values length must equal index dimension 3, "
            f"got {len(match.values)} (id={match.id!r})"
        )
        assert all(isinstance(v, float) for v in match.values), (
            f"match.values elements must be floats (id={match.id!r}): {match.values}"
        )
        # metadata was not requested — must be None
        assert match.metadata is None, (
            f"match.metadata must be None when include_metadata not set (id={match.id!r})"
        )

    # --- Part 2: include_values omitted (default False) → values absent ---
    results_no_values = await idx.query_namespaces(
        vector=[0.1, 0.2, 0.3],
        namespaces=["iva-ns1", "iva-ns2"],
        metric="cosine",
        top_k=4,
    )

    assert isinstance(results_no_values, QueryNamespacesResults)
    assert len(results_no_values.matches) >= 1, (
        "Expected at least one match when include_values not set"
    )

    for match in results_no_values.matches:
        assert isinstance(match, ScoredVector)
        # values must be empty list when include_values is not requested
        # (ScoredVector defaults values to [] — not None — when the API omits the field)
        assert match.values == [], (
            f"match.values must be empty [] when include_values not requested (id={match.id!r}), "
            f"got {match.values!r}"
        )


# ---------------------------------------------------------------------------
# query-namespaces-sparse — REST async
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_query_namespaces_sparse_rest_async(
    async_client: AsyncPinecone, shared_index_sparse: str
) -> None:
    """query_namespaces() with sparse_vector on a sparse dotproduct index returns merged results (REST async).

    Verifies that a sparse-only index can be queried across multiple namespaces
    using only sparse_vector (no dense vector), with results merged and sorted
    by dotproduct score (descending) and per-namespace usage populated.
    """
    idx = await async_client.index(name=shared_index_sparse)

    # Upsert sparse-only vectors into two namespaces
    await idx.upsert(
        vectors=[
            {
                "id": "qnsa-ns1-v1",
                "sparse_values": {"indices": [0, 1, 2], "values": [0.5, 0.8, 0.3]},
            },
            {
                "id": "qnsa-ns1-v2",
                "sparse_values": {"indices": [1, 3, 5], "values": [0.2, 0.7, 0.4]},
            },
        ],
        namespace="qnsa-ns1",
    )
    await idx.upsert(
        vectors=[
            {
                "id": "qnsa-ns2-v1",
                "sparse_values": {"indices": [0, 2, 4], "values": [0.6, 0.3, 0.9]},
            },
            {
                "id": "qnsa-ns2-v2",
                "sparse_values": {"indices": [1, 2, 3], "values": [0.4, 0.5, 0.6]},
            },
        ],
        namespace="qnsa-ns2",
    )

    # Wait until sparse vectors are fetchable in both namespaces
    await async_poll_until(
        query_fn=lambda: idx.fetch(ids=["qnsa-ns1-v1", "qnsa-ns1-v2"], namespace="qnsa-ns1"),
        check_fn=lambda r: len(r.vectors) == 2,
        timeout=120,
        description="qnsa-ns1 sparse vectors fetchable before query_namespaces_sparse_async",
    )
    await async_poll_until(
        query_fn=lambda: idx.fetch(ids=["qnsa-ns2-v1", "qnsa-ns2-v2"], namespace="qnsa-ns2"),
        check_fn=lambda r: len(r.vectors) == 2,
        timeout=120,
        description="qnsa-ns2 sparse vectors fetchable before query_namespaces_sparse_async",
    )

    # Sparse-only query: pass sparse_vector, not vector
    results = await idx.query_namespaces(
        namespaces=["qnsa-ns1", "qnsa-ns2"],
        sparse_vector={"indices": [0, 1, 2], "values": [0.1, 0.2, 0.3]},
        metric="dotproduct",
        top_k=5,
    )

    assert isinstance(results, QueryNamespacesResults)
    assert isinstance(results.matches, list)
    assert len(results.matches) >= 1

    for match in results.matches:
        assert isinstance(match, ScoredVector)
        assert isinstance(match.id, str)
        assert isinstance(match.score, float)

    # Scores sorted descending for dotproduct
    scores = [m.score for m in results.matches]
    assert scores == sorted(scores, reverse=True), (
        f"Expected scores sorted descending for dotproduct, got: {scores}"
    )

    # Per-namespace usage
    assert isinstance(results.ns_usage, dict)
    assert "qnsa-ns1" in results.ns_usage
    assert "qnsa-ns2" in results.ns_usage
    for ns_usage_val in results.ns_usage.values():
        assert ns_usage_val.read_units >= 0

    # Total usage
    assert results.usage is not None
    assert isinstance(results.usage.read_units, int)
    assert results.usage.read_units >= 2


# ---------------------------------------------------------------------------
# query-namespaces-parallel — REST async
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_query_namespaces_parallel_faster_than_serial_rest_async(
    async_client: AsyncPinecone, shared_index_dim2: str
) -> None:
    """query_namespaces() across 10 namespaces executes queries concurrently (async).

    Verifies claim unified-vec-0035: individual per-namespace queries are fanned
    out via asyncio.gather, so wall-clock time is substantially less than a
    sequential baseline.
    """
    idx = await async_client.index(name=shared_index_dim2)

    namespaces = [f"qnpa-ns-{i}" for i in range(10)]

    for ns in namespaces:
        await idx.upsert(
            vectors=[
                {"id": f"{ns}-v{j}", "values": [float(j) / 5, 1.0 - float(j) / 5]} for j in range(5)
            ],
            namespace=ns,
        )

    # Wait for each namespace to have all 5 vectors queryable
    for ns in namespaces:
        await async_poll_until(
            query_fn=lambda ns=ns: idx.query(vector=[0.5, 0.5], top_k=10, namespace=ns),
            check_fn=lambda r: len(r.matches) >= 5,
            timeout=120,
            description=f"{ns} vectors queryable (async) before parallel test",
        )

    # Warm up the HTTP connection pool before measuring. The first
    # parallel fan-out has to open ~N concurrent TCP+TLS connections
    # (http2=False, so each query needs its own connection); without
    # this warmup, parallel-vs-serial comparisons capture handshake
    # overhead rather than query overlap.
    await idx.query_namespaces(vector=[0.5, 0.5], namespaces=namespaces, metric="cosine", top_k=5)

    # Take the best (min) of multiple samples to reject CI noise — a
    # single tail-latency outlier on one parallel query inflates
    # max(times), so any cleaner sample is a more honest signal.
    samples = 3
    serial_elapsed = float("inf")
    parallel_elapsed = float("inf")
    results: QueryNamespacesResults | None = None
    for _ in range(samples):
        serial_start = time.monotonic()
        for ns in namespaces:
            await idx.query(vector=[0.5, 0.5], top_k=5, namespace=ns)
        serial_elapsed = min(serial_elapsed, time.monotonic() - serial_start)

        parallel_start = time.monotonic()
        results = await idx.query_namespaces(
            vector=[0.5, 0.5],
            namespaces=namespaces,
            metric="cosine",
            top_k=5,
        )
        parallel_elapsed = min(parallel_elapsed, time.monotonic() - parallel_start)

    # Correctness assertions
    assert results is not None
    assert isinstance(results, QueryNamespacesResults)
    assert isinstance(results.matches, list)
    assert 1 <= len(results.matches) <= 5
    for match in results.matches:
        assert isinstance(match, ScoredVector)
        assert isinstance(match.id, str)
        assert isinstance(match.score, float)
    scores = [m.score for m in results.matches]
    assert scores == sorted(scores, reverse=True)
    for ns in namespaces:
        assert ns in results.ns_usage

    # Skip if backend is too fast to distinguish parallel from serial
    if serial_elapsed < 0.1:
        pytest.skip(f"serial baseline too fast to be meaningful: {serial_elapsed:.3f}s")

    # Parallelism assertion: parallel must be substantially faster.
    # Threshold is generous (0.75) to absorb residual CI variance —
    # even a small parallel benefit proves fan-out is happening.
    assert parallel_elapsed < serial_elapsed * 0.75, (
        f"query_namespaces must fan out queries in parallel. "
        f"serial={serial_elapsed:.3f}s parallel={parallel_elapsed:.3f}s "
        f"ratio={parallel_elapsed / serial_elapsed:.2f} (expected < 0.75)"
    )


# ---------------------------------------------------------------------------
# query-namespaces-validation-errors — REST async
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
@pytest.mark.timeout(600)
async def test_query_namespaces_validation_errors_rest_async(
    async_client: AsyncPinecone, shared_index_dim2: str
) -> None:
    """query_namespaces() raises ValidationError for invalid argument combinations (REST async).

    Verifies the validation branches at pinecone/async_client/async_index.py:495-504:
    - empty namespaces list
    - neither vector nor sparse_vector provided
    - invalid metric value
    - empty vector list (falsy [] treated as missing vector)
    """
    idx = await async_client.index(name=shared_index_dim2)

    # Case 1 — empty namespaces list
    with pytest.raises(ValidationError) as excinfo:
        await idx.query_namespaces(
            vector=[0.1, 0.2],
            namespaces=[],
            metric="cosine",
            top_k=5,
        )
    assert "namespaces" in str(excinfo.value).lower()

    # Case 2 — neither vector nor sparse_vector provided
    with pytest.raises(ValidationError) as excinfo:
        await idx.query_namespaces(
            namespaces=["qnea-ns1"],
            metric="cosine",
            top_k=5,
        )
    msg = str(excinfo.value).lower()
    assert "vector" in msg and "sparse_vector" in msg

    # Case 3 — invalid metric value
    with pytest.raises(ValidationError) as excinfo:
        await idx.query_namespaces(
            vector=[0.1, 0.2],
            namespaces=["qnea-ns1"],
            metric="manhattan",
            top_k=5,
        )
    assert "metric" in str(excinfo.value).lower()
    assert "manhattan" in str(excinfo.value)

    # Case 4 — empty vector list (falsy [] treated as missing vector)
    with pytest.raises(ValidationError):
        await idx.query_namespaces(
            vector=[],
            namespaces=["qnea-ns1"],
            metric="cosine",
            top_k=5,
        )


# ---------------------------------------------------------------------------
# query-namespaces-tie-breaking — REST async
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_query_namespaces_tie_breaking_rest_async(
    async_client: AsyncPinecone, shared_index_dim2: str
) -> None:
    """query_namespaces() preserves deterministic order for tied scores (REST async).

    Verifies unified-vec-0037: when multiple matches share the same score, the
    heap-based aggregator preserves a deterministic order (insertion order).
    """
    idx = await async_client.index(name=shared_index_dim2)

    # Upsert identical vectors into two namespaces so every match scores 1.0
    await idx.upsert(
        vectors=[{"id": f"qnta-ns1-v{j}", "values": [1.0, 0.0]} for j in range(3)],
        namespace="qnta-ns1",
    )
    await idx.upsert(
        vectors=[{"id": f"qnta-ns2-v{j}", "values": [1.0, 0.0]} for j in range(3)],
        namespace="qnta-ns2",
    )

    # Wait for each namespace to have all 3 vectors queryable
    for ns in ("qnta-ns1", "qnta-ns2"):
        await async_poll_until(
            query_fn=lambda ns=ns: idx.query(vector=[1.0, 0.0], top_k=10, namespace=ns),
            check_fn=lambda r: len(r.matches) >= 3,
            timeout=120,
            description=f"{ns} vectors queryable (async) before tie-breaking test",
        )

    # Query twice with the same namespaces order
    results_a = await idx.query_namespaces(
        vector=[1.0, 0.0],
        namespaces=["qnta-ns1", "qnta-ns2"],
        metric="cosine",
        top_k=6,
    )
    results_b = await idx.query_namespaces(
        vector=[1.0, 0.0],
        namespaces=["qnta-ns1", "qnta-ns2"],
        metric="cosine",
        top_k=6,
    )

    assert isinstance(results_a, QueryNamespacesResults)
    assert isinstance(results_b, QueryNamespacesResults)
    assert len(results_a.matches) == 6
    assert len(results_b.matches) == 6

    # All scores must be (approximately) 1.0
    for m in results_a.matches:
        assert m.score == pytest.approx(1.0, abs=1e-4), (
            f"Expected score ~1.0 for {m.id}, got {m.score}"
        )

    # Deterministic: same order on repeated calls with the same namespace order
    assert [m.id for m in results_a.matches] == [m.id for m in results_b.matches], (
        "query_namespaces() returned different match order on repeated calls with identical "
        f"input — non-deterministic tie-breaking detected (async).\n"
        f"  call 1: {[m.id for m in results_a.matches]}\n"
        f"  call 2: {[m.id for m in results_b.matches]}"
    )


# ---------------------------------------------------------------------------
# query-namespaces-large-top-k-merge — REST async
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_query_namespaces_large_top_k_merge_rest_async(
    async_client: AsyncPinecone, shared_index_dim2: str
) -> None:
    """query_namespaces() merges large per-namespace result sets into a correct top-k (REST async).

    Verifies unified-vec-0036: heap-based aggregation yields the global top-k
    when each namespace contributes many matches.
    """
    idx = await async_client.index(name=shared_index_dim2)

    # Upsert 50 distinct vectors per namespace with interpolated values so scores differ
    ns1_vectors = [
        {"id": f"qnla-ns1-v{j}", "values": [float(j) / 50, 1.0 - float(j) / 50]} for j in range(50)
    ]
    ns2_vectors = [
        {"id": f"qnla-ns2-v{j}", "values": [float(j) / 100, 1.0 - float(j) / 100]}
        for j in range(50)
    ]
    await idx.upsert(vectors=ns1_vectors, namespace="qnla-ns1")
    await idx.upsert(vectors=ns2_vectors, namespace="qnla-ns2")

    # Wait for all vectors to be queryable (use top_k=50 when polling)
    for ns in ("qnla-ns1", "qnla-ns2"):
        await async_poll_until(
            query_fn=lambda ns=ns: idx.query(vector=[0.5, 0.5], top_k=50, namespace=ns),
            check_fn=lambda r: len(r.matches) >= 50,
            timeout=180,
            description=f"{ns} 50 vectors queryable (async) before large-top-k test",
        )

    results = await idx.query_namespaces(
        vector=[0.5, 0.5],
        namespaces=["qnla-ns1", "qnla-ns2"],
        metric="cosine",
        top_k=25,
    )

    assert isinstance(results, QueryNamespacesResults)
    assert len(results.matches) == 25

    # Scores must be in descending order
    scores = [m.score for m in results.matches]
    assert scores == sorted(scores, reverse=True), (
        f"Matches not sorted by descending score: {scores}"
    )

    # No duplicate IDs
    assert len({m.id for m in results.matches}) == 25, (
        "Duplicate match IDs in large-top-k results (async)"
    )

    # Both namespaces must appear in ns_usage
    assert set(results.ns_usage.keys()) == {"qnla-ns1", "qnla-ns2"}, (
        f"Expected ns_usage keys {{'qnla-ns1', 'qnla-ns2'}}, got {set(results.ns_usage.keys())}"
    )

    # Cosine with non-negative vectors cannot yield negative scores
    for m in results.matches:
        assert m.score >= 0.0, f"Unexpected negative cosine score {m.score} for {m.id}"


# ---------------------------------------------------------------------------
# query-namespaces-dense-dotproduct — descending score ordering — REST async
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_query_namespaces_dense_dotproduct_scores_descending_rest_async(
    async_client: AsyncPinecone, shared_index_dim2_dotproduct: str
) -> None:
    """query_namespaces() with dense dotproduct metric returns matches sorted descending (REST async).

    Verifies unified-vec-0036 for the dense+dotproduct combination: higher dot
    products rank first in the merged result set.

    Strategy:
    - Dense index, dimension 2, metric="dotproduct".
    - Query vector: [1.0, 0.0].
    - Upsert three vectors at known dot products from the query:
        ns1: "ddpa-high"  at [1.0, 0.0]  -> dotproduct 1.0
        ns1: "ddpa-low"   at [0.1, 0.0]  -> dotproduct 0.1
        ns2: "ddpa-mid"   at [0.5, 0.0]  -> dotproduct 0.5
    - Assert scores are non-increasing (descending), and that "ddpa-high" ranks
      first while "ddpa-low" ranks last.
    """
    idx = await async_client.index(name=shared_index_dim2_dotproduct)

    await idx.upsert(
        vectors=[
            {"id": "ddpa-high", "values": [1.0, 0.0]},
            {"id": "ddpa-low", "values": [0.1, 0.0]},
        ],
        namespace="ddpa-ns1",
    )
    await idx.upsert(
        vectors=[{"id": "ddpa-mid", "values": [0.5, 0.0]}],
        namespace="ddpa-ns2",
    )

    await async_poll_until(
        query_fn=lambda: idx.query(vector=[1.0, 0.0], top_k=10, namespace="ddpa-ns1"),
        check_fn=lambda r: len(r.matches) >= 2,
        timeout=120,
        description="ddpa-ns1 vectors queryable before dotproduct sort test",
    )
    await async_poll_until(
        query_fn=lambda: idx.query(vector=[1.0, 0.0], top_k=10, namespace="ddpa-ns2"),
        check_fn=lambda r: len(r.matches) >= 1,
        timeout=120,
        description="ddpa-ns2 vector queryable before dotproduct sort test",
    )

    results = await idx.query_namespaces(
        vector=[1.0, 0.0],
        namespaces=["ddpa-ns1", "ddpa-ns2"],
        metric="dotproduct",
        top_k=5,
    )

    assert isinstance(results, QueryNamespacesResults)
    assert len(results.matches) == 3

    scores = [m.score for m in results.matches]
    assert scores == sorted(scores, reverse=True), f"Dotproduct scores must be descending: {scores}"

    assert results.matches[0].id == "ddpa-high"
    assert results.matches[-1].id == "ddpa-low"

    for m in results.matches:
        assert isinstance(m, ScoredVector)
        assert isinstance(m.id, str)
        assert isinstance(m.score, float)

    assert results.ns_usage.keys() == {"ddpa-ns1", "ddpa-ns2"}
