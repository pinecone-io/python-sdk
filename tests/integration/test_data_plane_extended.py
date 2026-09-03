"""Integration tests for extended data-plane operations (sync REST + gRPC).

Phase 2 Tier 1: metadata-filter, sparse-vectors, query-by-id, fetch-missing-ids,
include-values-metadata, query-namespaces.

The shared indexes come from :func:`legacy_index_factory`, not from
``pc.indexes.create``: 2026-07 has no way to create an index the vectors API
will serve, and every call in this module is a vectors-API call. See
:mod:`tests.integration.legacy_index` for the sanctioned pattern and why the
SDK's own create path is bypassed.

Each shared-index fixture calls ``assert_serves_vectors_api`` once, because a
document-schema index would not fail this module loudly — writes are refused
but ``query`` / ``fetch`` / ``describe_index_stats`` succeed and return
**empty**, so a read-only assertion would pass against data that was never
there. Every test that reads live data writes first and asserts
``upserted_count``, which is the other half of that defence.
"""
# area tags covered: metadata-filter, sparse-vectors, query-by-id, fetch-missing-ids,
# include-values-metadata, query-namespaces

from __future__ import annotations

import uuid

import pytest

from pinecone import Field, Pinecone
from pinecone.models.indexes.index import IndexModel
from pinecone.models.vectors.query_aggregator import QueryNamespacesResults
from pinecone.models.vectors.responses import (
    FetchByMetadataResponse,
    FetchResponse,
    ListResponse,
    QueryResponse,
    ResponseInfo,
    UpdateResponse,
    UpsertResponse,
)
from pinecone.models.vectors.sparse import SparseValues
from pinecone.models.vectors.usage import Usage
from pinecone.models.vectors.vector import ScoredVector, Vector
from tests.integration.conftest import LegacyIndexFactory, poll_until
from tests.integration.legacy_index import (
    LEGACY_DENSE_FIELD,
    LEGACY_SPARSE_FIELD,
    LegacyIndex,
    assert_serves_vectors_api,
)

# ---------------------------------------------------------------------------
# Module-scoped shared indexes
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def shared_index_dim2(client: Pinecone, legacy_index_factory: LegacyIndexFactory) -> str:
    """Shared legacy index (dim=2, cosine) reused across all dim=2 tests in this module."""
    index = legacy_index_factory(dimension=2)
    assert_serves_vectors_api(client, index)
    return index.name


@pytest.fixture(scope="module")
def shared_index_dim3(client: Pinecone, legacy_index_factory: LegacyIndexFactory) -> str:
    """Shared legacy index (dim=3, cosine) reused across all dim=3 tests in this module."""
    index = legacy_index_factory(dimension=3)
    assert_serves_vectors_api(client, index)
    return index.name


@pytest.fixture(scope="module")
def shared_index_dim4_dotproduct(client: Pinecone, legacy_index_factory: LegacyIndexFactory) -> str:
    """Shared legacy index (dim=4, dotproduct) reused across dim=4/dotproduct tests.

    Hybrid tests write ``sparse_values`` alongside dense values here. A legacy
    index accepts that because it declares both reserved vector fields; a
    2026-07 schema index would have to name a ``sparse_vector`` field to do
    the same, and naming any field makes it a documents-API index.
    """
    index = legacy_index_factory(dimension=4, metric="dotproduct")
    assert_serves_vectors_api(client, index)
    return index.name


@pytest.fixture(scope="module")
def shared_index_dim4_cosine(client: Pinecone, legacy_index_factory: LegacyIndexFactory) -> str:
    """Shared legacy index (dim=4, cosine) reused across dim=4/cosine tests."""
    index = legacy_index_factory(dimension=4)
    assert_serves_vectors_api(client, index)
    return index.name


@pytest.fixture(scope="module")
def shared_index_dim2_dotproduct(client: Pinecone, legacy_index_factory: LegacyIndexFactory) -> str:
    """Shared legacy index (dim=2, dotproduct) — one test uses this combination."""
    index = legacy_index_factory(dimension=2, metric="dotproduct")
    assert_serves_vectors_api(client, index)
    return index.name


@pytest.fixture(scope="module")
def sparse_index(client: Pinecone, legacy_index_factory: LegacyIndexFactory) -> LegacyIndex:
    """Shared sparse legacy index — no dimension, dotproduct metric."""
    index = legacy_index_factory(vector_type="sparse", metric="dotproduct")
    assert_serves_vectors_api(client, index)
    return index


# ---------------------------------------------------------------------------
# metadata-filter — REST sync
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_metadata_filter_rest(client: Pinecone, shared_index_dim2: str) -> None:
    # shared_index_dim2
    """Query with metadata filters ($eq, $in) returns only matching vectors (REST sync)."""
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    index = client.index(name=shared_index_dim2)

    # Upsert vectors with metadata
    upserted = index.upsert(
        vectors=[
            {
                "id": "mf-v1",
                "values": [0.1, 0.2],
                "metadata": {"genre": "comedy", "year": 2020},
            },
            {
                "id": "mf-v2",
                "values": [0.3, 0.4],
                "metadata": {"genre": "action", "year": 2021},
            },
            {
                "id": "mf-v3",
                "values": [0.5, 0.6],
                "metadata": {"genre": "comedy", "year": 2022},
            },
        ],
        namespace=ns,
    )
    assert upserted.upserted_count == 3

    # Wait for all 3 vectors to be queryable (eventual consistency)
    poll_until(
        query_fn=lambda: index.query(vector=[0.1, 0.2], top_k=10, namespace=ns),
        check_fn=lambda r: len(r.matches) >= 3,
        timeout=120,
        description="all 3 vectors queryable before filter tests",
    )

    # Test $eq filter: only comedy vectors should return
    comedy_result = index.query(
        vector=[0.1, 0.2],
        top_k=10,
        filter={"genre": {"$eq": "comedy"}},
        include_metadata=True,
        namespace=ns,
    )
    assert isinstance(comedy_result, QueryResponse)
    comedy_ids = {m.id for m in comedy_result.matches}
    assert "mf-v1" in comedy_ids
    assert "mf-v3" in comedy_ids
    assert "mf-v2" not in comedy_ids

    # Test $in filter: only action vector should return
    action_result = index.query(
        vector=[0.1, 0.2],
        top_k=10,
        filter={"genre": {"$in": ["action", "thriller"]}},
        include_metadata=True,
        namespace=ns,
    )
    assert isinstance(action_result, QueryResponse)
    action_ids = {m.id for m in action_result.matches}
    assert "mf-v2" in action_ids
    assert "mf-v1" not in action_ids
    assert "mf-v3" not in action_ids

    # Verify metadata is present in results (since include_metadata=True)
    for match in comedy_result.matches:
        assert isinstance(match, ScoredVector)
        assert match.metadata is not None
        assert match.metadata.get("genre") == "comedy"


# ---------------------------------------------------------------------------
# metadata-filter — gRPC
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_metadata_filter_grpc(client: Pinecone, shared_index_dim2: str) -> None:
    # shared_index_dim2
    """Query with metadata filter via GrpcIndex returns only matching vectors."""
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    index = client.index(name=shared_index_dim2, grpc=True)

    upserted = index.upsert(
        vectors=[
            {"id": "mf-v1", "values": [0.1, 0.2], "metadata": {"genre": "comedy"}},
            {"id": "mf-v2", "values": [0.3, 0.4], "metadata": {"genre": "action"}},
        ],
        namespace=ns,
    )
    assert upserted.upserted_count == 2

    poll_until(
        query_fn=lambda: index.query(vector=[0.1, 0.2], top_k=10, namespace=ns),
        check_fn=lambda r: len(r.matches) >= 2,
        timeout=120,
        description="both vectors queryable (grpc) before filter test",
    )

    result = index.query(
        vector=[0.1, 0.2],
        top_k=10,
        filter={"genre": {"$eq": "comedy"}},
        namespace=ns,
    )
    assert isinstance(result, QueryResponse)
    ids = {m.id for m in result.matches}
    assert "mf-v1" in ids
    assert "mf-v2" not in ids


# ---------------------------------------------------------------------------
# sparse-vectors — REST sync
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_sparse_vectors_rest(client: Pinecone, shared_index_dim4_dotproduct: str) -> None:
    # shared_index_dim4_dotproduct
    """Upsert hybrid (dense+sparse) vectors; fetch returns sparse_values; query with sparse_vector works (REST sync)."""
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    index = client.index(name=shared_index_dim4_dotproduct)

    # Upsert vectors with both dense values and sparse values
    upsert_resp = index.upsert(
        vectors=[
            {
                "id": "sv-v1",
                "values": [0.1, 0.2, 0.3, 0.4],
                "sparse_values": {"indices": [0, 5], "values": [0.5, 0.8]},
            },
            {
                "id": "sv-v2",
                "values": [0.5, 0.6, 0.7, 0.8],
                "sparse_values": {"indices": [2, 7], "values": [0.3, 0.9]},
            },
        ],
        namespace=ns,
    )
    assert isinstance(upsert_resp, UpsertResponse)
    assert upsert_resp.upserted_count == 2

    # Wait for vectors to be fetchable (eventual consistency)
    poll_until(
        query_fn=lambda: index.fetch(ids=["sv-v1", "sv-v2"], namespace=ns),
        check_fn=lambda r: len(r.vectors) == 2,
        timeout=120,
        description="sparse vectors fetchable",
    )

    # Fetch and verify sparse_values are returned
    fetch_resp = index.fetch(ids=["sv-v1", "sv-v2"], namespace=ns)
    assert isinstance(fetch_resp, FetchResponse)
    assert "sv-v1" in fetch_resp.vectors
    v1 = fetch_resp.vectors["sv-v1"]
    assert isinstance(v1, Vector)
    assert v1.sparse_values is not None
    assert isinstance(v1.sparse_values, SparseValues)
    assert v1.sparse_values.indices == [0, 5]
    assert len(v1.sparse_values.values) == 2
    assert abs(v1.sparse_values.values[0] - 0.5) < 1e-4
    assert abs(v1.sparse_values.values[1] - 0.8) < 1e-4

    # Also verify v2 has sparse values
    v2 = fetch_resp.vectors["sv-v2"]
    assert v2.sparse_values is not None
    assert isinstance(v2.sparse_values, SparseValues)
    assert v2.sparse_values.indices == [2, 7]

    # Query with a sparse vector and verify matches are returned
    query_resp = index.query(
        vector=[0.1, 0.2, 0.3, 0.4],
        sparse_vector={"indices": [0, 5], "values": [0.5, 0.8]},
        top_k=5,
        include_values=True,
        namespace=ns,
    )
    assert isinstance(query_resp, QueryResponse)
    assert len(query_resp.matches) >= 1
    match_ids = {m.id for m in query_resp.matches}
    # sv-v1 shares the same sparse indices — it should rank highly
    assert "sv-v1" in match_ids


# ---------------------------------------------------------------------------
# query-by-id — REST sync
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_query_by_id_rest(client: Pinecone, shared_index_dim2: str) -> None:
    # shared_index_dim2
    """Query by stored vector ID returns a QueryResponse with the same structure as query-by-vector (REST sync)."""
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    index = client.index(name=shared_index_dim2)

    # Upsert 3 vectors
    upserted = index.upsert(
        vectors=[
            {"id": "qbi-v1", "values": [0.1, 0.2]},
            {"id": "qbi-v2", "values": [0.3, 0.4]},
            {"id": "qbi-v3", "values": [0.9, 0.1]},
        ],
        namespace=ns,
    )
    assert upserted.upserted_count == 3

    # Wait for all 3 vectors to be queryable before querying by ID
    poll_until(
        query_fn=lambda: index.query(vector=[0.1, 0.2], top_k=10, namespace=ns),
        check_fn=lambda r: len(r.matches) >= 3,
        timeout=120,
        description="all 3 vectors queryable before query-by-id",
    )

    # Query by ID — use qbi-v1 as the query seed
    result = index.query(id="qbi-v1", top_k=3, namespace=ns)
    assert isinstance(result, QueryResponse)
    assert len(result.matches) >= 1

    # Each match must have id and score
    for match in result.matches:
        assert isinstance(match, ScoredVector)
        assert isinstance(match.id, str)
        assert isinstance(match.score, float)

    # The top match should be the query vector itself
    match_ids = [m.id for m in result.matches]
    assert "qbi-v1" in match_ids


# ---------------------------------------------------------------------------
# query-by-id — gRPC
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_query_by_id_grpc(client: Pinecone, shared_index_dim2: str) -> None:
    # shared_index_dim2
    """Query by stored vector ID returns a QueryResponse via GrpcIndex."""
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    index = client.index(name=shared_index_dim2, grpc=True)

    # Upsert 3 vectors
    upserted = index.upsert(
        vectors=[
            {"id": "qbi-v1", "values": [0.1, 0.2]},
            {"id": "qbi-v2", "values": [0.3, 0.4]},
            {"id": "qbi-v3", "values": [0.9, 0.1]},
        ],
        namespace=ns,
    )
    assert upserted.upserted_count == 3

    # Wait until vectors are queryable
    poll_until(
        query_fn=lambda: index.query(vector=[0.1, 0.2], top_k=10, namespace=ns),
        check_fn=lambda r: len(r.matches) >= 3,
        timeout=120,
        description="all 3 vectors queryable (grpc) before query-by-id",
    )

    # Query by ID
    result = index.query(id="qbi-v1", top_k=3, namespace=ns)
    assert isinstance(result, QueryResponse)
    assert len(result.matches) >= 1

    for match in result.matches:
        assert isinstance(match, ScoredVector)
        assert isinstance(match.id, str)
        assert isinstance(match.score, float)

    match_ids = [m.id for m in result.matches]
    assert "qbi-v1" in match_ids


# ---------------------------------------------------------------------------
# fetch-missing-ids — REST sync
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_fetch_missing_ids_rest(client: Pinecone, shared_index_dim2: str) -> None:
    # shared_index_dim2
    """fetch() with a mix of existing and non-existent IDs returns only existing vectors, no error (REST sync)."""
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    index = client.index(name=shared_index_dim2)

    # Upsert 2 known vectors
    upserted = index.upsert(
        vectors=[
            {"id": "fmi-v1", "values": [0.1, 0.2]},
            {"id": "fmi-v2", "values": [0.3, 0.4]},
        ],
        namespace=ns,
    )
    assert upserted.upserted_count == 2

    # Wait for both vectors to be fetchable (eventual consistency)
    poll_until(
        query_fn=lambda: index.fetch(ids=["fmi-v1", "fmi-v2"], namespace=ns),
        check_fn=lambda r: len(r.vectors) == 2,
        timeout=120,
        description="both vectors fetchable before missing-id test",
    )

    # Fetch with a mix of existing and non-existent IDs — no error expected
    fetch_resp = index.fetch(ids=["fmi-v1", "fmi-v2", "fmi-does-not-exist"], namespace=ns)
    assert isinstance(fetch_resp, FetchResponse)

    # Only existing vectors are returned
    assert "fmi-v1" in fetch_resp.vectors
    assert "fmi-v2" in fetch_resp.vectors
    assert "fmi-does-not-exist" not in fetch_resp.vectors
    assert len(fetch_resp.vectors) == 2

    # Each returned vector has correct structure
    v1 = fetch_resp.vectors["fmi-v1"]
    assert isinstance(v1, Vector)
    assert v1.id == "fmi-v1"
    assert len(v1.values) == 2


# ---------------------------------------------------------------------------
# fetch-missing-ids — gRPC
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_fetch_missing_ids_grpc(client: Pinecone, shared_index_dim2: str) -> None:
    # shared_index_dim2
    """fetch() with a mix of existing and non-existent IDs returns only existing vectors, no error (gRPC)."""
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    index = client.index(name=shared_index_dim2, grpc=True)

    # Upsert 2 known vectors
    upserted = index.upsert(
        vectors=[
            {"id": "fmi-v1", "values": [0.1, 0.2]},
            {"id": "fmi-v2", "values": [0.3, 0.4]},
        ],
        namespace=ns,
    )
    assert upserted.upserted_count == 2

    # Wait for both vectors to be fetchable
    poll_until(
        query_fn=lambda: index.fetch(ids=["fmi-v1", "fmi-v2"], namespace=ns),
        check_fn=lambda r: len(r.vectors) == 2,
        timeout=120,
        description="both vectors fetchable (grpc) before missing-id test",
    )

    # Fetch with a mix of existing and non-existent IDs — no error expected
    fetch_resp = index.fetch(ids=["fmi-v1", "fmi-v2", "fmi-does-not-exist"], namespace=ns)
    assert isinstance(fetch_resp, FetchResponse)

    # Only existing vectors are returned
    assert "fmi-v1" in fetch_resp.vectors
    assert "fmi-v2" in fetch_resp.vectors
    assert "fmi-does-not-exist" not in fetch_resp.vectors
    assert len(fetch_resp.vectors) == 2


# ---------------------------------------------------------------------------
# sparse-vectors — gRPC
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_sparse_vectors_grpc(client: Pinecone, shared_index_dim4_dotproduct: str) -> None:
    # shared_index_dim4_dotproduct
    """Upsert hybrid (dense+sparse) vectors; fetch returns sparse_values; query with sparse_vector works (gRPC)."""
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    index = client.index(name=shared_index_dim4_dotproduct, grpc=True)

    # Upsert vectors with both dense values and sparse values
    upsert_resp = index.upsert(
        vectors=[
            {
                "id": "sv-v1",
                "values": [0.1, 0.2, 0.3, 0.4],
                "sparse_values": {"indices": [0, 5], "values": [0.5, 0.8]},
            },
            {
                "id": "sv-v2",
                "values": [0.5, 0.6, 0.7, 0.8],
                "sparse_values": {"indices": [2, 7], "values": [0.3, 0.9]},
            },
        ],
        namespace=ns,
    )
    assert isinstance(upsert_resp, UpsertResponse)
    assert upsert_resp.upserted_count == 2

    # Wait for vectors to be fetchable (eventual consistency)
    poll_until(
        query_fn=lambda: index.fetch(ids=["sv-v1", "sv-v2"], namespace=ns),
        check_fn=lambda r: len(r.vectors) == 2,
        timeout=120,
        description="sparse vectors fetchable via gRPC",
    )

    # Fetch and verify sparse_values are returned
    fetch_resp = index.fetch(ids=["sv-v1", "sv-v2"], namespace=ns)
    assert isinstance(fetch_resp, FetchResponse)
    assert "sv-v1" in fetch_resp.vectors
    v1 = fetch_resp.vectors["sv-v1"]
    assert isinstance(v1, Vector)
    assert v1.sparse_values is not None
    assert isinstance(v1.sparse_values, SparseValues)
    assert v1.sparse_values.indices == [0, 5]

    # Query with a sparse vector and verify matches are returned
    query_resp = index.query(
        vector=[0.1, 0.2, 0.3, 0.4],
        sparse_vector={"indices": [0, 5], "values": [0.5, 0.8]},
        top_k=5,
        namespace=ns,
    )
    assert isinstance(query_resp, QueryResponse)
    assert len(query_resp.matches) >= 1
    assert "sv-v1" in {m.id for m in query_resp.matches}


# ---------------------------------------------------------------------------
# include-values-metadata — REST sync
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_include_values_metadata_rest(client: Pinecone, shared_index_dim3: str) -> None:
    # shared_index_dim3
    """Query with include_values=True/include_metadata=True returns values and metadata on matches;
    query with defaults returns empty values and None metadata (REST sync)."""
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    index = client.index(name=shared_index_dim3)

    # Upsert vectors with metadata
    upserted = index.upsert(
        vectors=[
            {
                "id": "ivm-v1",
                "values": [0.1, 0.2, 0.3],
                "metadata": {"color": "red", "rank": 1},
            },
            {
                "id": "ivm-v2",
                "values": [0.4, 0.5, 0.6],
                "metadata": {"color": "blue", "rank": 2},
            },
        ],
        namespace=ns,
    )
    assert upserted.upserted_count == 2

    # Wait for both vectors to be queryable (eventual consistency)
    poll_until(
        query_fn=lambda: index.query(vector=[0.1, 0.2, 0.3], top_k=10, namespace=ns),
        check_fn=lambda r: len(r.matches) >= 2,
        timeout=120,
        description="both vectors queryable before include-values-metadata test",
    )

    # Query with include_values=True, include_metadata=True
    result = index.query(
        vector=[0.1, 0.2, 0.3],
        top_k=5,
        include_values=True,
        include_metadata=True,
        namespace=ns,
    )
    assert isinstance(result, QueryResponse)
    assert len(result.matches) >= 1

    top_match = result.matches[0]
    assert isinstance(top_match, ScoredVector)
    # values should be a non-empty list of floats
    assert isinstance(top_match.values, list)
    assert len(top_match.values) == 3
    assert all(isinstance(v, float) for v in top_match.values)
    # metadata should be a dict
    assert isinstance(top_match.metadata, dict)
    assert "color" in top_match.metadata
    assert "rank" in top_match.metadata

    # Verify all matches have values and metadata
    for match in result.matches:
        assert len(match.values) == 3
        assert match.metadata is not None

    # Query with defaults (no include_values, no include_metadata)
    default_result = index.query(vector=[0.1, 0.2, 0.3], top_k=5, namespace=ns)
    assert isinstance(default_result, QueryResponse)
    assert len(default_result.matches) >= 1

    default_match = default_result.matches[0]
    assert isinstance(default_match, ScoredVector)
    # Default: values should be empty, metadata should be None
    assert default_match.values == []
    assert default_match.metadata is None


# ---------------------------------------------------------------------------
# include-values-metadata — gRPC
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_include_values_metadata_grpc(client: Pinecone, shared_index_dim3: str) -> None:
    # shared_index_dim3
    """Query with include_values=True/include_metadata=True returns values and metadata (gRPC)."""
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    index = client.index(name=shared_index_dim3, grpc=True)

    # Upsert vectors with metadata
    upserted = index.upsert(
        vectors=[
            {"id": "ivm-v1", "values": [0.1, 0.2, 0.3], "metadata": {"color": "red"}},
            {"id": "ivm-v2", "values": [0.4, 0.5, 0.6], "metadata": {"color": "blue"}},
        ],
        namespace=ns,
    )
    assert upserted.upserted_count == 2

    # Wait for vectors to be queryable (eventual consistency)
    poll_until(
        query_fn=lambda: index.query(vector=[0.1, 0.2, 0.3], top_k=10, namespace=ns),
        check_fn=lambda r: len(r.matches) >= 2,
        timeout=120,
        description="both vectors queryable (grpc) before include-values-metadata test",
    )

    # Query with include_values=True, include_metadata=True
    result = index.query(
        vector=[0.1, 0.2, 0.3],
        top_k=5,
        include_values=True,
        include_metadata=True,
        namespace=ns,
    )
    assert isinstance(result, QueryResponse)
    assert len(result.matches) >= 1

    top_match = result.matches[0]
    assert isinstance(top_match, ScoredVector)
    assert isinstance(top_match.values, list)
    assert len(top_match.values) == 3
    assert isinstance(top_match.metadata, dict)
    assert "color" in top_match.metadata

    # Query with defaults — values empty, metadata None
    default_result = index.query(vector=[0.1, 0.2, 0.3], top_k=5, namespace=ns)
    assert isinstance(default_result, QueryResponse)
    assert len(default_result.matches) >= 1

    default_match = default_result.matches[0]
    assert default_match.values == []
    assert default_match.metadata is None


# ---------------------------------------------------------------------------
# query-namespaces — REST sync
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_query_namespaces_rest(client: Pinecone, shared_index_dim2: str) -> None:
    # shared_index_dim2
    """query_namespaces() fans out queries across multiple namespaces and merges results (REST sync)."""
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    ns1 = f"{ns}-ns1"
    ns2 = f"{ns}-ns2"
    index = client.index(name=shared_index_dim2)

    # Upsert different vectors into two named namespaces
    upserted = index.upsert(
        vectors=[
            {"id": "qn-ns1-v1", "values": [0.1, 0.2]},
            {"id": "qn-ns1-v2", "values": [0.3, 0.4]},
        ],
        namespace=ns1,
    )
    assert upserted.upserted_count == 2
    upserted = index.upsert(
        vectors=[
            {"id": "qn-ns2-v1", "values": [0.5, 0.6]},
            {"id": "qn-ns2-v2", "values": [0.7, 0.8]},
        ],
        namespace=ns2,
    )
    assert upserted.upserted_count == 2

    # Wait until at least one vector from each namespace is queryable
    poll_until(
        query_fn=lambda: index.query(vector=[0.1, 0.2], top_k=10, namespace=ns1),
        check_fn=lambda r: len(r.matches) >= 2,
        timeout=120,
        description="ns1 vectors queryable before query_namespaces",
    )
    poll_until(
        query_fn=lambda: index.query(vector=[0.1, 0.2], top_k=10, namespace=ns2),
        check_fn=lambda r: len(r.matches) >= 2,
        timeout=120,
        description="ns2 vectors queryable before query_namespaces",
    )

    # Call query_namespaces across both namespaces
    results = index.query_namespaces(
        vector=[0.1, 0.2],
        namespaces=[ns1, ns2],
        metric="cosine",
        top_k=5,
    )

    # Verify result type and structure
    assert isinstance(results, QueryNamespacesResults)

    # Merged matches list: sorted by score (descending for cosine), up to top_k
    assert isinstance(results.matches, list)
    assert len(results.matches) >= 1

    # Each match is a ScoredVector with id and score
    for match in results.matches:
        assert isinstance(match, ScoredVector)
        assert isinstance(match.id, str)
        assert isinstance(match.score, float)

    # Matches come from both namespaces
    match_ids = {m.id for m in results.matches}
    assert len(match_ids & {"qn-ns1-v1", "qn-ns1-v2"}) >= 1
    assert len(match_ids & {"qn-ns2-v1", "qn-ns2-v2"}) >= 1

    # Scores should be in descending order (cosine: higher is better)
    scores = [m.score for m in results.matches]
    assert scores == sorted(scores, reverse=True)

    # usage has read_units (sum across namespaces)
    assert results.usage is not None
    assert isinstance(results.usage.read_units, int)
    assert results.usage.read_units >= 2  # at least 1 per namespace

    # ns_usage has per-namespace usage
    assert isinstance(results.ns_usage, dict)
    assert ns1 in results.ns_usage
    assert ns2 in results.ns_usage


# ---------------------------------------------------------------------------
# metadata-filter numeric comparisons ($gt, $gte, $lt, $lte, $ne) — REST sync
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_metadata_filter_numeric_operators_rest(client: Pinecone, shared_index_dim2: str) -> None:
    # shared_index_dim2
    """Numeric comparison operators ($gt, $gte, $lt, $lte, $ne) filter vectors correctly (REST sync).

    Verifies unified-filter-0001: Can build metadata filters using not-equal,
    greater-than, greater-than-or-equal, less-than, and less-than-or-equal operators.
    """
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    index = client.index(name=shared_index_dim2)

    # Upsert 4 vectors with distinct integer year metadata values
    upserted = index.upsert(
        vectors=[
            {"id": "nf-v1", "values": [0.1, 0.2], "metadata": {"year": 2019}},
            {"id": "nf-v2", "values": [0.3, 0.4], "metadata": {"year": 2020}},
            {"id": "nf-v3", "values": [0.5, 0.6], "metadata": {"year": 2021}},
            {"id": "nf-v4", "values": [0.7, 0.8], "metadata": {"year": 2022}},
        ],
        namespace=ns,
    )
    assert upserted.upserted_count == 4

    # Wait until all 4 vectors are visible
    poll_until(
        query_fn=lambda: index.query(vector=[0.5, 0.5], top_k=10, namespace=ns),
        check_fn=lambda r: len(r.matches) >= 4,
        timeout=120,
        description="all 4 vectors queryable before numeric filter tests",
    )

    # $gt: 2020 — only years strictly greater than 2020 (2021, 2022)
    gt_result = index.query(
        vector=[0.5, 0.5],
        top_k=10,
        filter={"year": {"$gt": 2020}},
        include_metadata=True,
        namespace=ns,
    )
    assert isinstance(gt_result, QueryResponse)
    gt_ids = {m.id for m in gt_result.matches}
    assert "nf-v3" in gt_ids
    assert "nf-v4" in gt_ids
    assert "nf-v1" not in gt_ids
    assert "nf-v2" not in gt_ids

    # $gte: 2020 — years >= 2020 (2020, 2021, 2022)
    gte_result = index.query(
        vector=[0.5, 0.5],
        top_k=10,
        filter={"year": {"$gte": 2020}},
        include_metadata=True,
        namespace=ns,
    )
    assert isinstance(gte_result, QueryResponse)
    gte_ids = {m.id for m in gte_result.matches}
    assert "nf-v2" in gte_ids
    assert "nf-v3" in gte_ids
    assert "nf-v4" in gte_ids
    assert "nf-v1" not in gte_ids

    # $lt: 2021 — years strictly less than 2021 (2019, 2020)
    lt_result = index.query(
        vector=[0.5, 0.5],
        top_k=10,
        filter={"year": {"$lt": 2021}},
        include_metadata=True,
        namespace=ns,
    )
    assert isinstance(lt_result, QueryResponse)
    lt_ids = {m.id for m in lt_result.matches}
    assert "nf-v1" in lt_ids
    assert "nf-v2" in lt_ids
    assert "nf-v3" not in lt_ids
    assert "nf-v4" not in lt_ids

    # $lte: 2021 — years <= 2021 (2019, 2020, 2021)
    lte_result = index.query(
        vector=[0.5, 0.5],
        top_k=10,
        filter={"year": {"$lte": 2021}},
        include_metadata=True,
        namespace=ns,
    )
    assert isinstance(lte_result, QueryResponse)
    lte_ids = {m.id for m in lte_result.matches}
    assert "nf-v1" in lte_ids
    assert "nf-v2" in lte_ids
    assert "nf-v3" in lte_ids
    assert "nf-v4" not in lte_ids

    # $ne: 2021 — years not equal to 2021 (2019, 2020, 2022)
    ne_result = index.query(
        vector=[0.5, 0.5],
        top_k=10,
        filter={"year": {"$ne": 2021}},
        include_metadata=True,
        namespace=ns,
    )
    assert isinstance(ne_result, QueryResponse)
    ne_ids = {m.id for m in ne_result.matches}
    assert "nf-v1" in ne_ids
    assert "nf-v2" in ne_ids
    assert "nf-v4" in ne_ids
    assert "nf-v3" not in ne_ids


# ---------------------------------------------------------------------------
# metadata-filter numeric comparisons ($gt, $gte, $lt, $lte, $ne) — gRPC
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_metadata_filter_numeric_operators_grpc(client: Pinecone, shared_index_dim2: str) -> None:
    # shared_index_dim2
    """Numeric comparison operators ($gt, $gte, $lt, $lte, $ne) filter vectors correctly (gRPC).

    Transport-parity counterpart to test_metadata_filter_numeric_operators_rest.
    Verifies unified-filter-0001 on the gRPC transport: Can build metadata filters
    using not-equal, greater-than, greater-than-or-equal, less-than, and
    less-than-or-equal operators.
    """
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    index = client.index(name=shared_index_dim2, grpc=True)

    # Upsert 4 vectors with distinct integer year metadata values
    upserted = index.upsert(
        vectors=[
            {"id": "ng-v1", "values": [0.1, 0.2], "metadata": {"year": 2019}},
            {"id": "ng-v2", "values": [0.3, 0.4], "metadata": {"year": 2020}},
            {"id": "ng-v3", "values": [0.5, 0.6], "metadata": {"year": 2021}},
            {"id": "ng-v4", "values": [0.7, 0.8], "metadata": {"year": 2022}},
        ],
        namespace=ns,
    )
    assert upserted.upserted_count == 4

    # Wait until all 4 vectors are visible
    poll_until(
        query_fn=lambda: index.query(vector=[0.5, 0.5], top_k=10, namespace=ns),
        check_fn=lambda r: len(r.matches) >= 4,
        timeout=120,
        description="all 4 vectors queryable before gRPC numeric filter tests",
    )

    # $gt: 2020 — only years strictly greater than 2020 (2021, 2022)
    gt_result = index.query(
        vector=[0.5, 0.5],
        top_k=10,
        filter={"year": {"$gt": 2020}},
        include_metadata=True,
        namespace=ns,
    )
    assert isinstance(gt_result, QueryResponse)
    gt_ids = {m.id for m in gt_result.matches}
    assert "ng-v3" in gt_ids
    assert "ng-v4" in gt_ids
    assert "ng-v1" not in gt_ids
    assert "ng-v2" not in gt_ids

    # $gte: 2020 — years >= 2020 (2020, 2021, 2022)
    gte_result = index.query(
        vector=[0.5, 0.5],
        top_k=10,
        filter={"year": {"$gte": 2020}},
        include_metadata=True,
        namespace=ns,
    )
    assert isinstance(gte_result, QueryResponse)
    gte_ids = {m.id for m in gte_result.matches}
    assert "ng-v2" in gte_ids
    assert "ng-v3" in gte_ids
    assert "ng-v4" in gte_ids
    assert "ng-v1" not in gte_ids

    # $lt: 2021 — years strictly less than 2021 (2019, 2020)
    lt_result = index.query(
        vector=[0.5, 0.5],
        top_k=10,
        filter={"year": {"$lt": 2021}},
        include_metadata=True,
        namespace=ns,
    )
    assert isinstance(lt_result, QueryResponse)
    lt_ids = {m.id for m in lt_result.matches}
    assert "ng-v1" in lt_ids
    assert "ng-v2" in lt_ids
    assert "ng-v3" not in lt_ids
    assert "ng-v4" not in lt_ids

    # $lte: 2021 — years <= 2021 (2019, 2020, 2021)
    lte_result = index.query(
        vector=[0.5, 0.5],
        top_k=10,
        filter={"year": {"$lte": 2021}},
        include_metadata=True,
        namespace=ns,
    )
    assert isinstance(lte_result, QueryResponse)
    lte_ids = {m.id for m in lte_result.matches}
    assert "ng-v1" in lte_ids
    assert "ng-v2" in lte_ids
    assert "ng-v3" in lte_ids
    assert "ng-v4" not in lte_ids

    # $ne: 2021 — years not equal to 2021 (2019, 2020, 2022)
    ne_result = index.query(
        vector=[0.5, 0.5],
        top_k=10,
        filter={"year": {"$ne": 2021}},
        include_metadata=True,
        namespace=ns,
    )
    assert isinstance(ne_result, QueryResponse)
    ne_ids = {m.id for m in ne_result.matches}
    assert "ng-v1" in ne_ids
    assert "ng-v2" in ne_ids
    assert "ng-v4" in ne_ids
    assert "ng-v3" not in ne_ids


# ---------------------------------------------------------------------------
# metadata-filter logical operators ($nin, &, |) via Field builder — REST sync
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_metadata_filter_logical_operators_rest(client: Pinecone, shared_index_dim2: str) -> None:
    # shared_index_dim2
    """$nin, logical AND (&), and logical OR (|) via the Field builder filter correctly (REST sync).

    Verifies:
      unified-filter-0002 — not-in-list ($nin) operator
      unified-filter-0004 — combining filters with & (AND) and | (OR)
    """
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    index = client.index(name=shared_index_dim2)

    # Upsert 4 vectors with genre + year metadata
    upserted = index.upsert(
        vectors=[
            {
                "id": "lo-v1",
                "values": [0.1, 0.2],
                "metadata": {"genre": "comedy", "year": 2020},
            },
            {
                "id": "lo-v2",
                "values": [0.3, 0.4],
                "metadata": {"genre": "action", "year": 2021},
            },
            {
                "id": "lo-v3",
                "values": [0.5, 0.6],
                "metadata": {"genre": "comedy", "year": 2022},
            },
            {
                "id": "lo-v4",
                "values": [0.7, 0.8],
                "metadata": {"genre": "horror", "year": 2021},
            },
        ],
        namespace=ns,
    )
    assert upserted.upserted_count == 4

    # Wait until all 4 vectors are visible
    poll_until(
        query_fn=lambda: index.query(vector=[0.5, 0.5], top_k=10, namespace=ns),
        check_fn=lambda r: len(r.matches) >= 4,
        timeout=120,
        description="all 4 vectors queryable before logical filter tests",
    )

    # $nin: genres NOT in ["horror", "action"] → should return only comedy vectors (v1, v3)
    nin_filter = Field("genre").not_in(["horror", "action"])
    nin_result = index.query(
        vector=[0.5, 0.5],
        top_k=10,
        filter=nin_filter.to_dict(),
        include_metadata=True,
        namespace=ns,
    )
    assert isinstance(nin_result, QueryResponse)
    nin_ids = {m.id for m in nin_result.matches}
    assert "lo-v1" in nin_ids
    assert "lo-v3" in nin_ids
    assert "lo-v2" not in nin_ids
    assert "lo-v4" not in nin_ids

    # & (AND): genre == comedy AND year >= 2021 → should return only v3 (2022)
    and_filter = (Field("genre") == "comedy") & Field("year").gte(2021)
    and_result = index.query(
        vector=[0.5, 0.5],
        top_k=10,
        filter=and_filter.to_dict(),
        include_metadata=True,
        namespace=ns,
    )
    assert isinstance(and_result, QueryResponse)
    and_ids = {m.id for m in and_result.matches}
    assert "lo-v3" in and_ids
    assert "lo-v1" not in and_ids  # comedy but year 2020 < 2021
    assert "lo-v2" not in and_ids  # action
    assert "lo-v4" not in and_ids  # horror

    # | (OR): genre == horror OR genre == action → should return v2 and v4
    or_filter = (Field("genre") == "horror") | (Field("genre") == "action")
    or_result = index.query(
        vector=[0.5, 0.5],
        top_k=10,
        filter=or_filter.to_dict(),
        include_metadata=True,
        namespace=ns,
    )
    assert isinstance(or_result, QueryResponse)
    or_ids = {m.id for m in or_result.matches}
    assert "lo-v2" in or_ids
    assert "lo-v4" in or_ids
    assert "lo-v1" not in or_ids
    assert "lo-v3" not in or_ids

    # Verify to_dict() produces the correct wire-format dicts
    assert and_filter.to_dict() == {
        "$and": [{"genre": {"$eq": "comedy"}}, {"year": {"$gte": 2021}}]
    }
    assert or_filter.to_dict() == {
        "$or": [{"genre": {"$eq": "horror"}}, {"genre": {"$eq": "action"}}]
    }


# ---------------------------------------------------------------------------
# metadata-filter logical operators ($nin, &, |) via Field builder — gRPC
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_metadata_filter_logical_operators_grpc(client: Pinecone, shared_index_dim2: str) -> None:
    # shared_index_dim2
    """$nin and logical AND (&) via the Field builder filter correctly (gRPC).

    Verifies unified-filter-0002 and unified-filter-0004 on the gRPC transport.
    """
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    index = client.index(name=shared_index_dim2, grpc=True)

    upserted = index.upsert(
        vectors=[
            {
                "id": "lo-v1",
                "values": [0.1, 0.2],
                "metadata": {"genre": "comedy", "year": 2020},
            },
            {
                "id": "lo-v2",
                "values": [0.3, 0.4],
                "metadata": {"genre": "action", "year": 2021},
            },
            {
                "id": "lo-v3",
                "values": [0.5, 0.6],
                "metadata": {"genre": "comedy", "year": 2022},
            },
            {
                "id": "lo-v4",
                "values": [0.7, 0.8],
                "metadata": {"genre": "horror", "year": 2021},
            },
        ],
        namespace=ns,
    )
    assert upserted.upserted_count == 4

    poll_until(
        query_fn=lambda: index.query(vector=[0.5, 0.5], top_k=10, namespace=ns),
        check_fn=lambda r: len(r.matches) >= 4,
        timeout=120,
        description="all 4 vectors queryable (grpc) before logical filter tests",
    )

    # $nin: genres NOT in ["horror", "action"] → comedy vectors only (v1, v3)
    nin_filter = Field("genre").not_in(["horror", "action"])
    nin_result = index.query(
        vector=[0.5, 0.5],
        top_k=10,
        filter=nin_filter.to_dict(),
        namespace=ns,
    )
    assert isinstance(nin_result, QueryResponse)
    nin_ids = {m.id for m in nin_result.matches}
    assert "lo-v1" in nin_ids
    assert "lo-v3" in nin_ids
    assert "lo-v2" not in nin_ids
    assert "lo-v4" not in nin_ids

    # & (AND): genre == comedy AND year >= 2021 → v3 only
    and_filter = (Field("genre") == "comedy") & Field("year").gte(2021)
    and_result = index.query(
        vector=[0.5, 0.5],
        top_k=10,
        filter=and_filter.to_dict(),
        namespace=ns,
    )
    assert isinstance(and_result, QueryResponse)
    and_ids = {m.id for m in and_result.matches}
    assert "lo-v3" in and_ids
    assert "lo-v1" not in and_ids
    assert "lo-v2" not in and_ids
    assert "lo-v4" not in and_ids


# ---------------------------------------------------------------------------
# fetch-by-metadata — REST sync
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_fetch_by_metadata_rest(client: Pinecone, shared_index_dim3: str) -> None:
    # shared_index_dim3
    """fetch_by_metadata() returns vectors matching a filter, with correct response shape (REST sync).

    Verifies:
    - unified-vec-0010: Can fetch vectors by metadata filter from a namespace.
    - unified-vec-0024: No pagination token on a single-page result.
    """
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    index = client.index(name=shared_index_dim3)

    # Upsert vectors with metadata; only v1 and v3 have genre=comedy
    upserted = index.upsert(
        vectors=[
            {
                "id": "fm-v1",
                "values": [0.1, 0.2, 0.3],
                "metadata": {"genre": "comedy", "year": 2020},
            },
            {
                "id": "fm-v2",
                "values": [0.4, 0.5, 0.6],
                "metadata": {"genre": "action", "year": 2021},
            },
            {
                "id": "fm-v3",
                "values": [0.7, 0.8, 0.9],
                "metadata": {"genre": "comedy", "year": 2022},
            },
        ],
        namespace=ns,
    )
    assert upserted.upserted_count == 3

    # Wait until comedy vectors are reachable via fetch_by_metadata (eventual consistency)
    poll_until(
        query_fn=lambda: index.fetch_by_metadata(filter={"genre": {"$eq": "comedy"}}, namespace=ns),
        check_fn=lambda r: len(r.vectors) >= 2,
        timeout=120,
        description="comedy vectors reachable via fetch_by_metadata",
    )

    # Fetch only comedy vectors
    response = index.fetch_by_metadata(filter={"genre": {"$eq": "comedy"}}, namespace=ns)

    # Verify response type and shape
    assert isinstance(response, FetchByMetadataResponse)
    assert isinstance(response.vectors, dict)
    assert isinstance(response.namespace, str)

    # Only comedy vectors should be returned
    assert "fm-v1" in response.vectors
    assert "fm-v3" in response.vectors
    assert "fm-v2" not in response.vectors

    # Single page of 2 results — no pagination token
    assert response.pagination is None or response.pagination.next is None

    # Each returned vector has id and values
    for vid, vec in response.vectors.items():
        assert isinstance(vid, str)
        assert vec.id == vid
        assert isinstance(vec.values, list)
        assert len(vec.values) == 3


# ---------------------------------------------------------------------------
# metadata-filter $exists operator — REST sync
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_metadata_filter_exists_operator_rest(client: Pinecone, shared_index_dim3: str) -> None:
    # shared_index_dim3
    """Field.exists() filter ($exists: True) returns only vectors that have the field (REST sync).

    Verifies unified-filter-0003: Can build metadata filters using a field-exists operator.

    Upserts 3 vectors: two with a "premium" field (True/False) and one without.
    Queries with Field("premium").exists() and asserts only the two vectors that
    carry the "premium" key are returned — the third (which lacks the key) is excluded.
    """
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    index = client.index(name=shared_index_dim3)

    # v1 and v2 carry "premium" field; v3 does not
    upserted = index.upsert(
        vectors=[
            {
                "id": "ex-v1",
                "values": [0.1, 0.2, 0.3],
                "metadata": {"category": "A", "premium": True},
            },
            {
                "id": "ex-v2",
                "values": [0.4, 0.5, 0.6],
                "metadata": {"category": "B", "premium": False},
            },
            {"id": "ex-v3", "values": [0.7, 0.8, 0.9], "metadata": {"category": "C"}},
        ],
        namespace=ns,
    )
    assert upserted.upserted_count == 3

    # Wait until all 3 vectors are queryable (eventual consistency)
    poll_until(
        query_fn=lambda: index.query(vector=[0.1, 0.2, 0.3], top_k=10, namespace=ns),
        check_fn=lambda r: len(r.matches) >= 3,
        timeout=120,
        description="all 3 vectors queryable before exists filter test",
    )

    # Query using Field.exists() — only vectors with "premium" field should match
    exists_filter = Field("premium").exists()
    result = index.query(
        vector=[0.1, 0.2, 0.3],
        top_k=10,
        filter=exists_filter.to_dict(),
        include_metadata=True,
        namespace=ns,
    )

    assert isinstance(result, QueryResponse)
    matched_ids = {m.id for m in result.matches}

    # v1 and v2 both have "premium" — must appear
    assert "ex-v1" in matched_ids, f"Expected ex-v1 in matches, got {matched_ids}"
    assert "ex-v2" in matched_ids, f"Expected ex-v2 in matches, got {matched_ids}"

    # v3 has no "premium" field — must not appear
    assert "ex-v3" not in matched_ids, f"Expected ex-v3 excluded from matches, got {matched_ids}"

    # Metadata is returned and each match has the "premium" key
    for match in result.matches:
        assert isinstance(match.metadata, dict)
        assert "premium" in match.metadata, (
            f"Match {match.id!r} missing 'premium' key in metadata: {match.metadata}"
        )


# ---------------------------------------------------------------------------
# metadata-filter $exists operator — gRPC
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_metadata_filter_exists_operator_grpc(client: Pinecone, shared_index_dim3: str) -> None:
    # shared_index_dim3
    """Field.exists() filter ($exists: True) works the same way over gRPC transport.

    Verifies unified-filter-0003 on the gRPC transport.
    """
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    index = client.index(name=shared_index_dim3, grpc=True)

    # v1 and v2 carry "premium" field; v3 does not
    upserted = index.upsert(
        vectors=[
            {
                "id": "ex-v1",
                "values": [0.1, 0.2, 0.3],
                "metadata": {"category": "A", "premium": True},
            },
            {
                "id": "ex-v2",
                "values": [0.4, 0.5, 0.6],
                "metadata": {"category": "B", "premium": False},
            },
            {"id": "ex-v3", "values": [0.7, 0.8, 0.9], "metadata": {"category": "C"}},
        ],
        namespace=ns,
    )
    assert upserted.upserted_count == 3

    # Wait until all 3 vectors are queryable
    poll_until(
        query_fn=lambda: index.query(vector=[0.1, 0.2, 0.3], top_k=10, namespace=ns),
        check_fn=lambda r: len(r.matches) >= 3,
        timeout=120,
        description="all 3 vectors queryable (grpc) before exists filter test",
    )

    # Query using Field.exists() via gRPC
    result = index.query(
        vector=[0.1, 0.2, 0.3],
        top_k=10,
        filter=Field("premium").exists().to_dict(),
        include_metadata=True,
        namespace=ns,
    )

    assert isinstance(result, QueryResponse)
    matched_ids = {m.id for m in result.matches}

    assert "ex-v1" in matched_ids, f"Expected ex-v1 in matches (grpc), got {matched_ids}"
    assert "ex-v2" in matched_ids, f"Expected ex-v2 in matches (grpc), got {matched_ids}"
    assert "ex-v3" not in matched_ids, f"Expected ex-v3 excluded (grpc), got {matched_ids}"


# ---------------------------------------------------------------------------
# fetch_by_metadata multi-page pagination — REST sync
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_fetch_by_metadata_pagination_rest(client: Pinecone, shared_index_dim3: str) -> None:
    # shared_index_dim3
    """fetch_by_metadata() with limit and pagination_token iterates across pages (REST sync).

    Upserts 5 vectors all with genre=scifi.  Calls fetch_by_metadata with
    limit=2 to force multiple pages.  Uses poll_until to wait until the
    paginated traversal covers all 5 vectors (the cursor-scan index can lag the
    count index after a recent upsert).  Verifies:
    - Each page returns at most 2 vectors (limit respected).
    - Pagination tokens are returned when more results exist.
    - No vector ID appears on more than one page (no cursor back-tracking).
    - All 5 genre=scifi vectors appear across the collected pages.
    - The final page has no pagination token (last-page sentinel).

    Verifies:
    - unified-vec-0010: Can fetch vectors by metadata filter with pagination.
    - unified-vec-0024: Fetch-by-metadata respects the limit parameter.
    """
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    index = client.index(name=shared_index_dim3)

    # Upsert 5 vectors — all with genre=scifi so all match the filter
    target_ids = {"fbm-p1", "fbm-p2", "fbm-p3", "fbm-p4", "fbm-p5"}
    upserted = index.upsert(
        vectors=[
            {"id": "fbm-p1", "values": [0.1, 0.2, 0.3], "metadata": {"genre": "scifi"}},
            {"id": "fbm-p2", "values": [0.2, 0.3, 0.4], "metadata": {"genre": "scifi"}},
            {"id": "fbm-p3", "values": [0.3, 0.4, 0.5], "metadata": {"genre": "scifi"}},
            {"id": "fbm-p4", "values": [0.4, 0.5, 0.6], "metadata": {"genre": "scifi"}},
            {"id": "fbm-p5", "values": [0.5, 0.6, 0.7], "metadata": {"genre": "scifi"}},
        ],
        namespace=ns,
    )
    assert upserted.upserted_count == 5

    def _paginate_all() -> set[str]:
        """Collect all IDs returned by limit=2 pagination, asserting invariants."""
        seen: set[str] = set()
        tok: str | None = None
        page = 0
        while True:
            r = index.fetch_by_metadata(
                filter={"genre": {"$eq": "scifi"}},
                limit=2,
                pagination_token=tok,
                namespace=ns,
            )
            assert isinstance(r, FetchByMetadataResponse)
            # Limit must be respected
            assert len(r.vectors) <= 2, f"Page {page} returned {len(r.vectors)} vectors (limit=2)"
            # No duplicate vector IDs across pages
            for vid in r.vectors:
                assert vid not in seen, f"Duplicate ID {vid!r} on page {page}"
                seen.add(vid)
            page += 1
            tok = r.pagination.next if r.pagination else None
            if not tok:
                break
        return seen

    # Poll until the cursor-scan index is consistent with the upserted data.
    # The count index and cursor-scan index can temporarily diverge after a
    # very recent upsert; retrying the full paginated traversal is the
    # correct eventual-consistency guard here.
    collected_ids = poll_until(
        query_fn=_paginate_all,
        check_fn=lambda ids: target_ids.issubset(ids),
        timeout=120,
        description="all 5 fbm-pagination vectors reachable via cursor scan",
    )

    # All 5 target vectors must appear and no extra vectors
    assert collected_ids == target_ids, (
        f"Unexpected IDs in paginated result: {collected_ids - target_ids}"
    )


# ---------------------------------------------------------------------------
# query-filter-reflects-metadata-update — REST sync
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_query_filter_reflects_metadata_update_rest(
    client: Pinecone, shared_index_dim2: str
) -> None:
    # shared_index_dim2
    """After updating a vector's metadata via update(id=...), a query with a
    metadata filter must reflect the new value — not just a subsequent fetch().

    Tests the operation sequence:
      upsert → query-with-filter (baseline) → update-metadata
      → poll until query-with-filter shows updated state → verify.

    Claim: unified-vec-0011 — Can update a single vector's metadata by
    identifier; the change is reflected in subsequent filtered queries.
    Area tag: update-and-query-sequence
    Transport: rest
    """
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    index = client.index(name=shared_index_dim2)

    # Upsert 3 vectors: 2 drama, 1 comedy
    vectors = [
        {"id": "qmu-v1", "values": [0.1, 0.2], "metadata": {"genre": "drama"}},
        {"id": "qmu-v2", "values": [0.3, 0.4], "metadata": {"genre": "drama"}},
        {"id": "qmu-v3", "values": [0.5, 0.6], "metadata": {"genre": "comedy"}},
    ]
    upsert_resp = index.upsert(vectors=vectors, namespace=ns)
    assert isinstance(upsert_resp, UpsertResponse)
    assert upsert_resp.upserted_count == 3

    # Wait until all 3 are queryable (eventual consistency)
    poll_until(
        query_fn=lambda: index.query(vector=[0.1, 0.2], top_k=10, namespace=ns),
        check_fn=lambda r: len(r.matches) >= 3,
        timeout=120,
        description="all 3 qmu vectors queryable (REST)",
    )

    # Baseline: drama filter returns 2 vectors
    drama_baseline = index.query(
        vector=[0.1, 0.2],
        top_k=10,
        filter={"genre": {"$eq": "drama"}},
        include_metadata=True,
        namespace=ns,
    )
    assert isinstance(drama_baseline, QueryResponse)
    drama_ids_before = {m.id for m in drama_baseline.matches}
    assert "qmu-v1" in drama_ids_before, "qmu-v1 (drama) should match drama filter"
    assert "qmu-v2" in drama_ids_before, "qmu-v2 (drama) should match drama filter"
    assert "qmu-v3" not in drama_ids_before, "qmu-v3 (comedy) must not match drama filter"

    # Update qmu-v1: overwrite genre from "drama" to "comedy"
    update_resp = index.update(
        id="qmu-v1",
        set_metadata={"genre": "comedy"},
        namespace=ns,
    )
    assert isinstance(update_resp, UpdateResponse)

    # Poll until the filter index reflects the update:
    # drama filter should now return only qmu-v2 (qmu-v1 switched to comedy)
    def _drama_filter_updated() -> QueryResponse | None:
        r = index.query(
            vector=[0.1, 0.2],
            top_k=10,
            filter={"genre": {"$eq": "drama"}},
            include_metadata=True,
            namespace=ns,
        )
        ids = {m.id for m in r.matches}
        # Success: qmu-v1 is gone from drama; qmu-v2 still present
        if "qmu-v1" not in ids and "qmu-v2" in ids:
            return r
        return None

    updated_drama_resp = poll_until(
        query_fn=_drama_filter_updated,
        check_fn=lambda r: r is not None,
        timeout=180,
        description="drama filter reflects metadata update for qmu-v1 (REST)",
    )

    assert updated_drama_resp is not None
    drama_ids_after = {m.id for m in updated_drama_resp.matches}
    assert "qmu-v2" in drama_ids_after, "qmu-v2 should still match drama filter"
    assert "qmu-v1" not in drama_ids_after, "qmu-v1 should no longer match drama filter"
    assert "qmu-v3" not in drama_ids_after, "qmu-v3 (comedy) must not match drama filter"

    # qmu-v1 should now match comedy filter (verify updated metadata propagated)
    comedy_resp = index.query(
        vector=[0.1, 0.2],
        top_k=10,
        filter={"genre": {"$eq": "comedy"}},
        include_metadata=True,
        namespace=ns,
    )
    assert isinstance(comedy_resp, QueryResponse)
    comedy_ids = {m.id for m in comedy_resp.matches}
    assert "qmu-v1" in comedy_ids, (
        f"qmu-v1 should match comedy filter after update; got: {comedy_ids}"
    )
    assert "qmu-v3" in comedy_ids, f"qmu-v3 (original comedy) should still match; got: {comedy_ids}"

    # Confirm metadata in comedy query result has updated value
    v1_match = next((m for m in comedy_resp.matches if m.id == "qmu-v1"), None)
    assert v1_match is not None
    assert v1_match.metadata is not None
    assert v1_match.metadata.get("genre") == "comedy", (
        f"qmu-v1 metadata.genre should be 'comedy' after update, "
        f"got {v1_match.metadata.get('genre')!r}"
    )


# ---------------------------------------------------------------------------
# query-filter-reflects-metadata-update — gRPC
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_query_filter_reflects_metadata_update_grpc(
    client: Pinecone, shared_index_dim2_dotproduct: str
) -> None:
    """Same as REST variant but via gRPC transport.

    Claim: unified-vec-0011 — gRPC update and query paths reflect the same
    metadata change in filtered query results.
    Area tag: update-and-query-sequence
    Transport: grpc
    """
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    index = client.index(name=shared_index_dim2_dotproduct, grpc=True)

    vectors = [
        {"id": "qmug-v1", "values": [0.1, 0.2], "metadata": {"genre": "drama"}},
        {"id": "qmug-v2", "values": [0.3, 0.4], "metadata": {"genre": "drama"}},
        {"id": "qmug-v3", "values": [0.5, 0.6], "metadata": {"genre": "comedy"}},
    ]
    upsert_resp = index.upsert(vectors=vectors, namespace=ns)
    assert isinstance(upsert_resp, UpsertResponse)
    assert upsert_resp.upserted_count == 3

    poll_until(
        query_fn=lambda: index.query(vector=[0.1, 0.2], top_k=10, namespace=ns),
        check_fn=lambda r: len(r.matches) >= 3,
        timeout=120,
        description="all 3 qmug vectors queryable (gRPC)",
    )

    # Baseline drama query
    drama_baseline = index.query(
        vector=[0.1, 0.2],
        top_k=10,
        filter={"genre": {"$eq": "drama"}},
        include_metadata=True,
        namespace=ns,
    )
    assert isinstance(drama_baseline, QueryResponse)
    assert "qmug-v1" in {m.id for m in drama_baseline.matches}
    assert "qmug-v2" in {m.id for m in drama_baseline.matches}

    # Update qmug-v1 genre to comedy via gRPC
    update_resp = index.update(
        id="qmug-v1",
        set_metadata={"genre": "comedy"},
        namespace=ns,
    )
    assert isinstance(update_resp, UpdateResponse)

    def _drama_filter_updated_grpc() -> QueryResponse | None:
        r = index.query(
            vector=[0.1, 0.2],
            top_k=10,
            filter={"genre": {"$eq": "drama"}},
            include_metadata=True,
            namespace=ns,
        )
        ids = {m.id for m in r.matches}
        if "qmug-v1" not in ids and "qmug-v2" in ids:
            return r
        return None

    updated_resp = poll_until(
        query_fn=_drama_filter_updated_grpc,
        check_fn=lambda r: r is not None,
        timeout=180,
        description="drama filter reflects metadata update for qmug-v1 (gRPC)",
    )
    assert updated_resp is not None
    assert "qmug-v2" in {m.id for m in updated_resp.matches}
    assert "qmug-v1" not in {m.id for m in updated_resp.matches}


# ---------------------------------------------------------------------------
# unusual ASCII vector IDs — REST sync
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_unusual_ascii_ids_round_trip_rest(client: Pinecone, shared_index_dim3: str) -> None:
    # shared_index_dim3
    """Vectors with unusual but valid ASCII IDs survive a full upsert→fetch→query→list round-trip.

    Verifies unified-ids-0001: IDs containing spaces, slashes, dots, colons, and
    other printable ASCII punctuation are accepted end-to-end by the live API.
    All integration tests to date only exercise simple alphanumeric IDs; this
    test exercises the boundary of what the SDK allows.
    """
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    index = client.index(name=shared_index_dim3)
    # IDs with unusual but valid ASCII characters (no non-ASCII, no null bytes).
    # Spaces are excluded: URL-encoding spaces in fetch query params is unreliable
    # across API versions (+ vs %20), so space-containing IDs are tested separately.
    unusual_ids = [
        "id/with/slashes",
        "id.with.dots",
        "id:colon:separated",
        "id@special!chars",
        "id[brackets]",
    ]

    # Upsert vectors with unusual IDs
    vectors = [
        (vid, [float(i + 1) * 0.1, float(i + 2) * 0.1, float(i + 3) * 0.1])
        for i, vid in enumerate(unusual_ids)
    ]
    upsert_resp = index.upsert(vectors=vectors, namespace=ns)
    assert isinstance(upsert_resp, UpsertResponse)
    assert upsert_resp.upserted_count == len(unusual_ids)

    # Wait until all vectors are fetchable (eventual consistency)
    fetch_resp = poll_until(
        query_fn=lambda: index.fetch(ids=unusual_ids, namespace=ns),
        check_fn=lambda r: len(r.vectors) == len(unusual_ids),
        timeout=120,
        description="all unusual-id vectors fetchable",
    )
    assert isinstance(fetch_resp, FetchResponse)
    # Verify each ID round-trips correctly through fetch
    for vid in unusual_ids:
        assert vid in fetch_resp.vectors, f"ID {vid!r} missing from fetch response"
        assert fetch_resp.vectors[vid].id == vid

    # Query by ID — the queried vector must appear in the results and
    # every match must be a valid ScoredVector. We do NOT assert it is the
    # top match because all vectors have very similar cosine direction.
    first_id = unusual_ids[0]
    query_resp = poll_until(
        query_fn=lambda: index.query(id=first_id, top_k=5, namespace=ns),
        check_fn=lambda r: any(m.id == first_id for m in r.matches),
        timeout=60,
        description=f"query-by-id returns {first_id!r}",
    )
    assert isinstance(query_resp, QueryResponse)
    assert len(query_resp.matches) >= 1
    match_ids = {m.id for m in query_resp.matches}
    assert first_id in match_ids, (
        f"queried ID {first_id!r} should appear in its own query results; got {match_ids}"
    )
    for m in query_resp.matches:
        assert isinstance(m, ScoredVector)
        assert isinstance(m.id, str)
        assert isinstance(m.score, float)

    # List — all unusual IDs must appear across pages
    all_listed_ids: set[str] = set()
    for page in index.list(namespace=ns):
        for item in page.vectors:
            all_listed_ids.add(item.id)
    for vid in unusual_ids:
        assert vid in all_listed_ids, f"ID {vid!r} missing from list() output"


# ---------------------------------------------------------------------------
# unusual ASCII vector IDs — gRPC
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_unusual_ascii_ids_round_trip_grpc(client: Pinecone, shared_index_dim3: str) -> None:
    # shared_index_dim3
    """Same round-trip as REST variant but over gRPC transport.

    Verifies unified-ids-0001 holds for gRPC: upsert, fetch, query by ID,
    and list all work correctly when vector IDs contain unusual ASCII characters.
    """
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    index = client.index(name=shared_index_dim3, grpc=True)
    unusual_ids = [
        "id/with/slashes",
        "id.with.dots",
        "id@special!chars",
    ]

    vectors = [
        (vid, [float(i + 1) * 0.1, float(i + 2) * 0.1, float(i + 3) * 0.1])
        for i, vid in enumerate(unusual_ids)
    ]
    upsert_resp = index.upsert(vectors=vectors, namespace=ns)
    assert isinstance(upsert_resp, UpsertResponse)
    assert upsert_resp.upserted_count == len(unusual_ids)

    # Wait until all vectors are fetchable
    fetch_resp = poll_until(
        query_fn=lambda: index.fetch(ids=unusual_ids, namespace=ns),
        check_fn=lambda r: len(r.vectors) == len(unusual_ids),
        timeout=120,
        description="all unusual-id vectors fetchable (gRPC)",
    )
    assert isinstance(fetch_resp, FetchResponse)
    for vid in unusual_ids:
        assert vid in fetch_resp.vectors, f"ID {vid!r} missing from fetch response (gRPC)"
        assert fetch_resp.vectors[vid].id == vid

    # Query by ID — queried vector must appear somewhere in results
    first_id = unusual_ids[0]
    query_resp = poll_until(
        query_fn=lambda: index.query(id=first_id, top_k=5, namespace=ns),
        check_fn=lambda r: any(m.id == first_id for m in r.matches),
        timeout=60,
        description=f"query-by-id returns {first_id!r} (gRPC)",
    )
    assert isinstance(query_resp, QueryResponse)
    match_ids = {m.id for m in query_resp.matches}
    assert first_id in match_ids, (
        f"queried ID {first_id!r} should appear in its own query results (gRPC); got {match_ids}"
    )

    # List — unusual IDs appear across pages
    all_listed_ids: set[str] = set()
    for page in index.list(namespace=ns):
        for item in page.vectors:
            all_listed_ids.add(item.id)
    for vid in unusual_ids:
        assert vid in all_listed_ids, f"ID {vid!r} missing from list() output (gRPC)"


# ---------------------------------------------------------------------------
# metadata-filter boolean values — REST sync
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_metadata_filter_boolean_values_rest(client: Pinecone, shared_index_dim2: str) -> None:
    # shared_index_dim2
    """Boolean filter values ($eq: True / $eq: False) return the correct vectors (REST sync).

    Verifies unified-filter-0006: Filter field values support strings, integers,
    floats, AND booleans.

    Upserts 3 vectors: two with active=True, one with active=False.
    Queries with both Field("active").eq(True) and Field("active").eq(False)
    to verify that boolean equality filters route correctly.
    """
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    index = client.index(name=shared_index_dim2)

    # Upsert 3 vectors: 2 active=True, 1 active=False
    upserted = index.upsert(
        vectors=[
            {"id": "bool-v1", "values": [0.1, 0.2], "metadata": {"active": True, "label": "a"}},
            {
                "id": "bool-v2",
                "values": [0.3, 0.4],
                "metadata": {"active": False, "label": "b"},
            },
            {"id": "bool-v3", "values": [0.5, 0.6], "metadata": {"active": True, "label": "c"}},
        ],
        namespace=ns,
    )
    assert upserted.upserted_count == 3

    # Wait until all 3 vectors are queryable (eventual consistency)
    poll_until(
        query_fn=lambda: index.query(vector=[0.1, 0.2], top_k=10, namespace=ns),
        check_fn=lambda r: len(r.matches) >= 3,
        timeout=120,
        description="all 3 boolean-metadata vectors queryable",
    )

    # Filter by active=True — should return bool-v1 and bool-v3
    # Field.__eq__ produces {"active": {"$eq": True}}
    result_true = index.query(
        vector=[0.1, 0.2],
        top_k=10,
        filter=(Field("active") == True).to_dict(),  # noqa: E712
        include_metadata=True,
        namespace=ns,
    )
    assert isinstance(result_true, QueryResponse)
    true_ids = {m.id for m in result_true.matches}
    assert "bool-v1" in true_ids, f"bool-v1 (active=True) missing from result: {true_ids}"
    assert "bool-v3" in true_ids, f"bool-v3 (active=True) missing from result: {true_ids}"
    assert "bool-v2" not in true_ids, (
        f"bool-v2 (active=False) leaked into active=True result: {true_ids}"
    )
    # Verify metadata round-trip preserves boolean type
    for m in result_true.matches:
        assert m.metadata is not None
        assert m.metadata.get("active") is True, (
            f"Match {m.id!r}: expected active=True, got {m.metadata.get('active')!r}"
        )

    # Filter by active=False — also verify raw dict form works
    result_false = index.query(
        vector=[0.3, 0.4],
        top_k=10,
        filter={"active": {"$eq": False}},
        include_metadata=True,
        namespace=ns,
    )
    assert isinstance(result_false, QueryResponse)
    false_ids = {m.id for m in result_false.matches}
    assert "bool-v2" in false_ids, f"bool-v2 (active=False) missing from result: {false_ids}"
    assert "bool-v1" not in false_ids, (
        f"bool-v1 (active=True) leaked into active=False result: {false_ids}"
    )
    assert "bool-v3" not in false_ids, (
        f"bool-v3 (active=True) leaked into active=False result: {false_ids}"
    )
    for m in result_false.matches:
        assert m.metadata is not None
        assert m.metadata.get("active") is False, (
            f"Match {m.id!r}: expected active=False, got {m.metadata.get('active')!r}"
        )


# ---------------------------------------------------------------------------
# metadata-filter boolean values — gRPC
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_metadata_filter_boolean_values_grpc(client: Pinecone, shared_index_dim2: str) -> None:
    # shared_index_dim2
    """Boolean filter values ($eq: True / $eq: False) return the correct vectors (gRPC).

    Verifies unified-filter-0006 on the gRPC transport.
    """
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    index = client.index(name=shared_index_dim2, grpc=True)

    # Upsert 3 vectors: 2 active=True, 1 active=False
    upserted = index.upsert(
        vectors=[
            {"id": "bool-g1", "values": [0.1, 0.2], "metadata": {"active": True}},
            {"id": "bool-g2", "values": [0.3, 0.4], "metadata": {"active": False}},
            {"id": "bool-g3", "values": [0.5, 0.6], "metadata": {"active": True}},
        ],
        namespace=ns,
    )
    assert upserted.upserted_count == 3

    # Wait until all 3 vectors are queryable
    poll_until(
        query_fn=lambda: index.query(vector=[0.1, 0.2], top_k=10, namespace=ns),
        check_fn=lambda r: len(r.matches) >= 3,
        timeout=120,
        description="all 3 boolean-metadata vectors queryable (gRPC)",
    )

    # Filter by active=True — should return bool-g1 and bool-g3
    result_true = index.query(
        vector=[0.1, 0.2],
        top_k=10,
        filter={"active": {"$eq": True}},
        include_metadata=True,
        namespace=ns,
    )
    assert isinstance(result_true, QueryResponse)
    true_ids = {m.id for m in result_true.matches}
    assert "bool-g1" in true_ids, f"bool-g1 (active=True) missing (gRPC): {true_ids}"
    assert "bool-g3" in true_ids, f"bool-g3 (active=True) missing (gRPC): {true_ids}"
    assert "bool-g2" not in true_ids, f"bool-g2 (active=False) leaked (gRPC): {true_ids}"

    # Filter by active=False — using Field builder
    result_false = index.query(
        vector=[0.3, 0.4],
        top_k=10,
        filter=(Field("active") == False).to_dict(),  # noqa: E712
        include_metadata=True,
        namespace=ns,
    )
    assert isinstance(result_false, QueryResponse)
    false_ids = {m.id for m in result_false.matches}
    assert "bool-g2" in false_ids, f"bool-g2 (active=False) missing (gRPC): {false_ids}"
    assert "bool-g1" not in false_ids, f"bool-g1 leaked into false filter (gRPC): {false_ids}"
    assert "bool-g3" not in false_ids, f"bool-g3 leaked into false filter (gRPC): {false_ids}"


# ---------------------------------------------------------------------------
# response_info header population — REST sync
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_response_info_populated_on_data_plane_responses_rest(
    client: Pinecone, shared_index_dim2: str
) -> None:
    # shared_index_dim2
    """response_info is populated as a ResponseInfo struct after real data-plane API calls.

    Verifies unified-rs-0010: Response objects carry HTTP response header information
    accessible via a response-info field.

    The SDK always creates a ResponseInfo object after each data-plane response
    (via extract_response_info()). This test confirms:
    - response_info is not None on UpsertResponse, QueryResponse, and FetchResponse
    - response_info is a ResponseInfo instance with the correct field types
    - Individual header fields (request_id, lsn_*) may be None when the API
      does not return those headers; the structural invariant is what we verify here.
    """
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    index = client.index(name=shared_index_dim2)

    # --- upsert ---
    upsert_result = index.upsert(
        vectors=[
            {"id": "ri-v1", "values": [0.1, 0.2]},
            {"id": "ri-v2", "values": [0.3, 0.4]},
        ],
        namespace=ns,
    )
    assert isinstance(upsert_result, UpsertResponse)
    assert upsert_result.upserted_count == 2
    assert upsert_result.response_info is not None, (
        "UpsertResponse.response_info should be set (non-None) after a real API call"
    )
    assert isinstance(upsert_result.response_info, ResponseInfo)
    # Field types are correct: str | None and int | None
    ri = upsert_result.response_info
    assert ri.request_id is None or isinstance(ri.request_id, str)
    assert ri.lsn_reconciled is None or isinstance(ri.lsn_reconciled, int)
    assert ri.lsn_committed is None or isinstance(ri.lsn_committed, int)

    # --- query (via poll_until for eventual consistency) ---
    query_result = poll_until(
        query_fn=lambda: index.query(vector=[0.1, 0.2], top_k=2, namespace=ns),
        check_fn=lambda r: len(r.matches) >= 2,
        timeout=120,
        description="ri-v1 and ri-v2 queryable",
    )
    assert isinstance(query_result, QueryResponse)
    assert query_result.response_info is not None, (
        "QueryResponse.response_info should be set after a real API call"
    )
    assert isinstance(query_result.response_info, ResponseInfo)
    qri = query_result.response_info
    assert qri.request_id is None or isinstance(qri.request_id, str)
    assert qri.lsn_reconciled is None or isinstance(qri.lsn_reconciled, int)
    assert qri.lsn_committed is None or isinstance(qri.lsn_committed, int)

    # --- fetch ---
    fetch_result = index.fetch(ids=["ri-v1", "ri-v2"], namespace=ns)
    assert isinstance(fetch_result, FetchResponse)
    assert fetch_result.response_info is not None, (
        "FetchResponse.response_info should be set after a real API call"
    )
    assert isinstance(fetch_result.response_info, ResponseInfo)
    fri = fetch_result.response_info
    assert fri.request_id is None or isinstance(fri.request_id, str)
    assert fri.lsn_reconciled is None or isinstance(fri.lsn_reconciled, int)
    assert fri.lsn_committed is None or isinstance(fri.lsn_committed, int)


# Note: gRPC transport (test_response_info_populated_on_data_plane_responses_grpc)
# is N/A — pure gRPC operations (upsert, query, fetch) use the binary gRPC protocol
# and do not call extract_response_info(), so response_info is None on those responses.
# Only gRPC operations that delegate to REST (upsert_records, search) set response_info.


# ---------------------------------------------------------------------------
# unicode metadata round-trip — REST sync
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_unicode_metadata_round_trip_rest(client: Pinecone, shared_index_dim2: str) -> None:
    # shared_index_dim2
    """Unicode/multibyte metadata values (CJK, emoji, accented Latin) survive a full
    upsert → fetch → query round-trip; metadata filter equality on a unicode string value
    works correctly end-to-end (REST sync).

    Depth escalation: boundary value — all existing metadata tests use ASCII-only values;
    unicode serialization through orjson has never been verified in integration tests.

    Related: unified-wf-0001 (orjson UTF-8 serialization of request bodies);
             unified-filter-0006 (filter field values support strings).
    """
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    index = client.index(name=shared_index_dim2)

    # Two vectors: one with unicode/emoji metadata, one with plain ASCII.
    # unicode_meta vector uses CJK, emoji, and accented Latin characters.
    unicode_meta = {
        "lang": "日本語",  # Japanese
        "emoji": "🚀🌊",  # multi-codepoint emoji
        "accented": "café naïve",  # accented Latin
        "cjk": "中文",  # Chinese
    }
    ascii_meta = {"lang": "english", "emoji": "none", "accented": "plain"}

    upserted = index.upsert(
        vectors=[
            {"id": "uni-v1", "values": [0.1, 0.9], "metadata": unicode_meta},
            {"id": "uni-v2", "values": [0.9, 0.1], "metadata": ascii_meta},
        ],
        namespace=ns,
    )
    assert upserted.upserted_count == 2

    # Wait until both vectors are queryable (eventual consistency)
    poll_until(
        query_fn=lambda: index.query(vector=[0.1, 0.9], top_k=10, namespace=ns),
        check_fn=lambda r: len(r.matches) >= 2,
        timeout=120,
        description="unicode-metadata vectors queryable",
    )

    # --- fetch: verify unicode metadata is preserved byte-for-byte ---
    fetch_resp = index.fetch(ids=["uni-v1", "uni-v2"], namespace=ns)
    assert isinstance(fetch_resp, FetchResponse)
    assert "uni-v1" in fetch_resp.vectors, "uni-v1 missing from fetch response"
    v1 = fetch_resp.vectors["uni-v1"]
    assert v1.metadata is not None, "uni-v1 metadata should not be None"
    assert v1.metadata.get("lang") == "日本語", (
        f"CJK lang mismatch: expected '日本語', got {v1.metadata.get('lang')!r}"
    )
    assert v1.metadata.get("emoji") == "🚀🌊", (
        f"Emoji mismatch: expected '🚀🌊', got {v1.metadata.get('emoji')!r}"
    )
    assert v1.metadata.get("accented") == "café naïve", (
        f"Accented mismatch: expected 'café naïve', got {v1.metadata.get('accented')!r}"
    )
    assert v1.metadata.get("cjk") == "中文", (
        f"CJK mismatch: expected '中文', got {v1.metadata.get('cjk')!r}"
    )

    # --- query with include_metadata: verify metadata returned in query results ---
    query_resp = index.query(
        vector=[0.1, 0.9],
        top_k=10,
        include_metadata=True,
        namespace=ns,
    )
    assert isinstance(query_resp, QueryResponse)
    uni_matches = [m for m in query_resp.matches if m.id == "uni-v1"]
    assert len(uni_matches) == 1, "uni-v1 should appear in top-10 results"
    uni_m = uni_matches[0]
    assert uni_m.metadata is not None
    assert uni_m.metadata.get("lang") == "日本語"
    assert uni_m.metadata.get("emoji") == "🚀🌊"

    # --- metadata filter on unicode string value ---
    # Only uni-v1 should match the CJK lang filter; uni-v2 has lang="english"
    filtered = index.query(
        vector=[0.5, 0.5],
        top_k=10,
        filter={"lang": {"$eq": "日本語"}},
        include_metadata=True,
        namespace=ns,
    )
    assert isinstance(filtered, QueryResponse)
    filtered_ids = {m.id for m in filtered.matches}
    assert "uni-v1" in filtered_ids, (
        f"uni-v1 (lang='日本語') missing from unicode filter result: {filtered_ids}"
    )
    assert "uni-v2" not in filtered_ids, (
        f"uni-v2 (lang='english') leaked into unicode filter result: {filtered_ids}"
    )


# ---------------------------------------------------------------------------
# unicode metadata round-trip — gRPC
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_unicode_metadata_round_trip_grpc(client: Pinecone, shared_index_dim2: str) -> None:
    # shared_index_dim2
    """Unicode/multibyte metadata values survive a full upsert → fetch → query
    round-trip via gRPC transport.

    Depth escalation: same boundary value as the REST variant, but verifies
    that protobuf/gRPC serialization also handles unicode metadata correctly.
    """
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    index = client.index(name=shared_index_dim2, grpc=True)

    unicode_meta = {
        "lang": "日本語",
        "emoji": "🚀🌊",
        "accented": "café naïve",
        "cjk": "中文",
    }

    upserted = index.upsert(
        vectors=[
            {"id": "uni-g1", "values": [0.1, 0.9], "metadata": unicode_meta},
            {"id": "uni-g2", "values": [0.9, 0.1], "metadata": {"lang": "english"}},
        ],
        namespace=ns,
    )
    assert upserted.upserted_count == 2

    # Wait until both vectors are queryable
    poll_until(
        query_fn=lambda: index.query(vector=[0.1, 0.9], top_k=10, namespace=ns),
        check_fn=lambda r: len(r.matches) >= 2,
        timeout=120,
        description="gRPC unicode-metadata vectors queryable",
    )

    # Fetch and verify unicode metadata round-trip
    fetch_resp = index.fetch(ids=["uni-g1"], namespace=ns)
    assert isinstance(fetch_resp, FetchResponse)
    assert "uni-g1" in fetch_resp.vectors
    v1 = fetch_resp.vectors["uni-g1"]
    assert v1.metadata is not None
    assert v1.metadata.get("lang") == "日本語", (
        f"gRPC CJK lang mismatch: expected '日本語', got {v1.metadata.get('lang')!r}"
    )
    assert v1.metadata.get("emoji") == "🚀🌊", (
        f"gRPC Emoji mismatch: expected '🚀🌊', got {v1.metadata.get('emoji')!r}"
    )
    assert v1.metadata.get("cjk") == "中文", (
        f"gRPC CJK mismatch: expected '中文', got {v1.metadata.get('cjk')!r}"
    )

    # Query with unicode metadata filter on gRPC transport
    filtered = index.query(
        vector=[0.5, 0.5],
        top_k=10,
        filter={"lang": {"$eq": "日本語"}},
        include_metadata=True,
        namespace=ns,
    )
    assert isinstance(filtered, QueryResponse)
    filtered_ids = {m.id for m in filtered.matches}
    assert "uni-g1" in filtered_ids, (
        f"gRPC: uni-g1 (lang='日本語') missing from unicode filter result: {filtered_ids}"
    )
    assert "uni-g2" not in filtered_ids, (
        f"gRPC: uni-g2 (lang='english') leaked into unicode filter result: {filtered_ids}"
    )


# ---------------------------------------------------------------------------
# sparse-index-lifecycle — REST sync
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_sparse_index_lifecycle_rest(client: Pinecone, sparse_index: LegacyIndex) -> None:
    """Describe a sparse index to verify its properties, upsert sparse-only vectors, fetch back.

    Verifies:
    - unified-sparse-0001: A sparse index carries no dimension.
    - unified-sparse-0002: A described sparse index reports a sparse vector field
      and no dense one.
    - unified-vecfmt-0005: Can accept vectors as dictionaries with sparse values only (no dense values required).

    The old ``vector_type`` / ``metric`` / ``dimension`` triple has no 2026-07
    equivalent — ``IndexModel`` dropped all three, and a sparse vector field
    carries neither a dimension nor a metric. What survives is the schema:
    a sparse index declares the reserved sparse field and no dense field.
    """
    ns = f"ns-{uuid.uuid4().hex[:8]}"

    desc = client.indexes.describe(sparse_index.name)
    assert isinstance(desc, IndexModel)
    assert set(desc.schema.fields) == {LEGACY_SPARSE_FIELD}, (
        f"Described sparse index should declare only {LEGACY_SPARSE_FIELD!r}; "
        f"got {sorted(desc.schema.fields)}"
    )
    assert LEGACY_DENSE_FIELD not in desc.schema.fields
    assert desc.status.ready is True
    assert isinstance(desc.host, str) and len(desc.host) > 0

    # Upsert sparse-only vectors (no dense values) — unified-vecfmt-0005
    index = client.index(name=sparse_index.name)
    upsert_resp = index.upsert(
        vectors=[
            {
                "id": "sp-v1",
                "sparse_values": {"indices": [0, 5, 10], "values": [0.5, 0.8, 0.3]},
            },
            {
                "id": "sp-v2",
                "sparse_values": {"indices": [2, 7], "values": [0.3, 0.9]},
            },
        ],
        namespace=ns,
    )
    assert isinstance(upsert_resp, UpsertResponse)
    assert upsert_resp.upserted_count == 2

    # Wait for vectors to be fetchable (eventual consistency)
    poll_until(
        query_fn=lambda: index.fetch(ids=["sp-v1", "sp-v2"], namespace=ns),
        check_fn=lambda r: len(r.vectors) == 2,
        timeout=120,
        description="sparse-only vectors fetchable after upsert (REST)",
    )

    # Fetch and verify sparse-only structure
    fetch_resp = index.fetch(ids=["sp-v1", "sp-v2"], namespace=ns)
    assert isinstance(fetch_resp, FetchResponse)
    assert "sp-v1" in fetch_resp.vectors
    assert "sp-v2" in fetch_resp.vectors

    v1 = fetch_resp.vectors["sp-v1"]
    assert isinstance(v1, Vector)
    assert v1.id == "sp-v1"
    # Sparse-only: dense values list must be empty (no dense component)
    assert v1.values == [], (
        f"Sparse-only vector 'sp-v1' should have empty dense values, got: {v1.values}"
    )
    # Sparse values are present and correct
    assert v1.sparse_values is not None, "sparse_values must not be None for a sparse-only vector"
    assert isinstance(v1.sparse_values, SparseValues)
    assert v1.sparse_values.indices == [0, 5, 10], (
        f"Expected sparse indices [0, 5, 10], got {v1.sparse_values.indices}"
    )
    assert len(v1.sparse_values.values) == 3
    assert all(isinstance(x, float) for x in v1.sparse_values.values)
    for actual, expected in zip(v1.sparse_values.values, [0.5, 0.8, 0.3]):
        assert abs(actual - expected) < 1e-4, (
            f"Sparse value mismatch: expected {expected}, got {actual}"
        )

    v2 = fetch_resp.vectors["sp-v2"]
    assert isinstance(v2, Vector)
    assert v2.values == [], (
        f"Sparse-only vector 'sp-v2' should have empty dense values, got: {v2.values}"
    )
    assert v2.sparse_values is not None
    assert v2.sparse_values.indices == [2, 7]
    assert len(v2.sparse_values.values) == 2


# ---------------------------------------------------------------------------
# sparse-index-lifecycle — gRPC
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_sparse_index_lifecycle_grpc(client: Pinecone, sparse_index: LegacyIndex) -> None:
    """Upsert sparse-only vectors into a sparse index and fetch them back via GrpcIndex.

    Verifies (gRPC transport):
    - unified-sparse-0001: A sparse index carries no dimension.
    - unified-vecfmt-0005: GrpcIndex accepts vector dicts with only sparse_values (no dense values).
    """
    ns = f"ns-{uuid.uuid4().hex[:8]}"

    # Upsert sparse-only vectors via gRPC
    index = client.index(name=sparse_index.name, grpc=True)
    upsert_resp = index.upsert(
        vectors=[
            {
                "id": "sp-g1",
                "sparse_values": {"indices": [1, 4], "values": [0.6, 0.7]},
            },
            {
                "id": "sp-g2",
                "sparse_values": {"indices": [3, 8, 12], "values": [0.4, 0.9, 0.2]},
            },
        ],
        namespace=ns,
    )
    assert isinstance(upsert_resp, UpsertResponse)
    assert upsert_resp.upserted_count == 2

    # Wait for vectors to be fetchable (eventual consistency)
    poll_until(
        query_fn=lambda: index.fetch(ids=["sp-g1", "sp-g2"], namespace=ns),
        check_fn=lambda r: len(r.vectors) == 2,
        timeout=120,
        description="sparse-only vectors fetchable after gRPC upsert",
    )

    # Fetch and verify sparse-only structure
    fetch_resp = index.fetch(ids=["sp-g1", "sp-g2"], namespace=ns)
    assert isinstance(fetch_resp, FetchResponse)
    assert "sp-g1" in fetch_resp.vectors
    assert "sp-g2" in fetch_resp.vectors

    v1 = fetch_resp.vectors["sp-g1"]
    assert isinstance(v1, Vector)
    assert v1.values == [], (
        f"gRPC sparse-only vector 'sp-g1' should have empty dense values, got: {v1.values}"
    )
    assert v1.sparse_values is not None
    assert isinstance(v1.sparse_values, SparseValues)
    assert v1.sparse_values.indices == [1, 4]
    assert len(v1.sparse_values.values) == 2

    v2 = fetch_resp.vectors["sp-g2"]
    assert v2.values == []
    assert v2.sparse_values is not None
    assert v2.sparse_values.indices == [3, 8, 12]
    assert len(v2.sparse_values.values) == 3


# ---------------------------------------------------------------------------
# usage — REST sync
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_query_fetch_list_usage_read_units_rest(
    client: Pinecone, shared_index_dim4_cosine: str
) -> None:
    # shared_index_dim4_cosine
    """query(), fetch(), and list_paginated() responses include read-unit usage info (REST sync).

    Verifies unified-rs-0011: "Fetch, query, and list vector responses include
    read-unit usage information."

    Upserts two vectors, waits for eventual consistency, then calls query(),
    fetch(), and list_paginated() and asserts each response carries a non-None
    usage field with a non-negative integer read_units value.
    """
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    index = client.index(name=shared_index_dim4_cosine)

    # Upsert two vectors so query/fetch/list have something to return
    upsert_resp = index.upsert(
        vectors=[
            {"id": "usg-v1", "values": [0.1, 0.2, 0.3, 0.4]},
            {"id": "usg-v2", "values": [0.5, 0.6, 0.7, 0.8]},
        ],
        namespace=ns,
    )
    assert isinstance(upsert_resp, UpsertResponse)
    assert upsert_resp.upserted_count == 2

    # Wait for eventual consistency
    poll_until(
        query_fn=lambda: index.query(vector=[0.1, 0.2, 0.3, 0.4], top_k=10, namespace=ns),
        check_fn=lambda r: len(r.matches) >= 2,
        timeout=120,
        description="both usage-test vectors queryable (REST)",
    )

    # --- query() usage ---
    query_resp = index.query(vector=[0.1, 0.2, 0.3, 0.4], top_k=5, namespace=ns)
    assert isinstance(query_resp, QueryResponse)
    assert query_resp.usage is not None, (
        "query() response should include usage read-unit information (unified-rs-0011)"
    )
    assert isinstance(query_resp.usage, Usage)
    assert isinstance(query_resp.usage.read_units, int)
    assert query_resp.usage.read_units >= 0

    # --- fetch() usage ---
    fetch_resp = index.fetch(ids=["usg-v1", "usg-v2"], namespace=ns)
    assert isinstance(fetch_resp, FetchResponse)
    assert fetch_resp.usage is not None, (
        "fetch() response should include usage read-unit information (unified-rs-0011)"
    )
    assert isinstance(fetch_resp.usage, Usage)
    assert isinstance(fetch_resp.usage.read_units, int)
    assert fetch_resp.usage.read_units >= 0

    # --- list_paginated() usage ---
    list_resp = index.list_paginated(limit=10, namespace=ns)
    assert isinstance(list_resp, ListResponse)
    # usage may be None if the API does not report it for list operations;
    # when present it must be a valid Usage with a non-negative read_units
    if list_resp.usage is not None:
        assert isinstance(list_resp.usage, Usage)
        assert isinstance(list_resp.usage.read_units, int)
        assert list_resp.usage.read_units >= 0


# ---------------------------------------------------------------------------
# usage — gRPC
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_query_fetch_list_usage_read_units_grpc(
    client: Pinecone, shared_index_dim4_cosine: str
) -> None:
    # shared_index_dim4_cosine
    """query(), fetch(), and list_paginated() responses include read-unit usage info (gRPC).

    Verifies unified-rs-0011 over the gRPC transport.
    """
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    index = client.index(name=shared_index_dim4_cosine, grpc=True)

    upsert_resp = index.upsert(
        vectors=[
            {"id": "usg-g1", "values": [0.1, 0.2, 0.3, 0.4]},
            {"id": "usg-g2", "values": [0.5, 0.6, 0.7, 0.8]},
        ],
        namespace=ns,
    )
    assert isinstance(upsert_resp, UpsertResponse)
    assert upsert_resp.upserted_count == 2

    poll_until(
        query_fn=lambda: index.query(vector=[0.1, 0.2, 0.3, 0.4], top_k=10, namespace=ns),
        check_fn=lambda r: len(r.matches) >= 2,
        timeout=120,
        description="both usage-test vectors queryable via gRPC",
    )

    # --- query() usage via gRPC ---
    query_resp = index.query(vector=[0.1, 0.2, 0.3, 0.4], top_k=5, namespace=ns)
    assert isinstance(query_resp, QueryResponse)
    assert query_resp.usage is not None, (
        "gRPC query() response should include usage read-unit information (unified-rs-0011)"
    )
    assert isinstance(query_resp.usage, Usage)
    assert isinstance(query_resp.usage.read_units, int)
    assert query_resp.usage.read_units >= 0

    # --- fetch() usage via gRPC ---
    fetch_resp = index.fetch(ids=["usg-g1", "usg-g2"], namespace=ns)
    assert isinstance(fetch_resp, FetchResponse)
    assert fetch_resp.usage is not None, (
        "gRPC fetch() response should include usage read-unit information (unified-rs-0011)"
    )
    assert isinstance(fetch_resp.usage, Usage)
    assert isinstance(fetch_resp.usage.read_units, int)
    assert fetch_resp.usage.read_units >= 0

    # --- list_paginated() usage via gRPC ---
    list_resp = index.list_paginated(limit=10, namespace=ns)
    assert isinstance(list_resp, ListResponse)
    if list_resp.usage is not None:
        assert isinstance(list_resp.usage, Usage)
        assert isinstance(list_resp.usage.read_units, int)
        assert list_resp.usage.read_units >= 0


# ---------------------------------------------------------------------------
# query-after-delete — operation sequence: upsert → query → delete → query
# REST sync
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_query_after_delete_reflects_deletion_rest(
    client: Pinecone, shared_index_dim4_dotproduct: str
) -> None:
    # shared_index_dim4_dotproduct
    """Deleted vectors no longer appear in subsequent query() results (REST sync).

    Verifies the operation sequence: upsert → query (both present) → delete one
    vector → query again → deleted vector absent, remaining vector still present.

    All existing delete tests verify removal via fetch(); this test exercises the
    independent query path to confirm delete/query consistency (depth-escalation
    operation-sequence test).
    """
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    index = client.index(name=shared_index_dim4_dotproduct)

    # Upsert two vectors with clearly distinct values
    upserted = index.upsert(
        vectors=[
            {"id": "qad-v1", "values": [1.0, 0.0, 0.0, 0.0]},
            {"id": "qad-v2", "values": [0.0, 1.0, 0.0, 0.0]},
        ],
        namespace=ns,
    )
    assert upserted.upserted_count == 2

    # Wait until both vectors are queryable (eventual consistency)
    poll_until(
        query_fn=lambda: index.query(vector=[1.0, 0.0, 0.0, 0.0], top_k=10, namespace=ns),
        check_fn=lambda r: len(r.matches) >= 2,
        timeout=120,
        description="both vectors queryable before delete",
    )

    # Baseline: query returns both vectors
    pre_delete = index.query(vector=[1.0, 0.0, 0.0, 0.0], top_k=10, namespace=ns)
    pre_ids = {m.id for m in pre_delete.matches}
    assert "qad-v1" in pre_ids, "qad-v1 should appear before delete"
    assert "qad-v2" in pre_ids, "qad-v2 should appear before delete"

    # Delete only qad-v1
    result = index.delete(ids=["qad-v1"], namespace=ns)
    assert result is None

    # Wait until qad-v1 is gone from fetch (deletion confirmed)
    poll_until(
        query_fn=lambda: index.fetch(ids=["qad-v1"], namespace=ns),
        check_fn=lambda r: "qad-v1" not in r.vectors,
        timeout=120,
        description="qad-v1 absent from fetch after delete",
    )

    # Post-delete query: qad-v1 must NOT appear; qad-v2 must still appear
    poll_until(
        query_fn=lambda: index.query(vector=[1.0, 0.0, 0.0, 0.0], top_k=10, namespace=ns),
        check_fn=lambda r: "qad-v1" not in {m.id for m in r.matches},
        timeout=120,
        description="qad-v1 absent from query results after delete",
    )

    post_delete = index.query(vector=[1.0, 0.0, 0.0, 0.0], top_k=10, namespace=ns)
    assert isinstance(post_delete, QueryResponse)
    post_ids = {m.id for m in post_delete.matches}
    assert "qad-v1" not in post_ids, "qad-v1 should not appear in query results after deletion"
    assert "qad-v2" in post_ids, (
        "qad-v2 should still appear in query results after deleting a different vector"
    )


# ---------------------------------------------------------------------------
# query-after-delete — gRPC
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_query_after_delete_reflects_deletion_grpc(
    client: Pinecone, shared_index_dim4_dotproduct: str
) -> None:
    # shared_index_dim4_dotproduct
    """Deleted vectors no longer appear in subsequent query() results (gRPC).

    Same operation sequence as the REST variant but exercising the gRPC transport.
    """
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    index = client.index(name=shared_index_dim4_dotproduct, grpc=True)

    upserted = index.upsert(
        vectors=[
            {"id": "qadg-v1", "values": [1.0, 0.0, 0.0, 0.0]},
            {"id": "qadg-v2", "values": [0.0, 1.0, 0.0, 0.0]},
        ],
        namespace=ns,
    )
    assert upserted.upserted_count == 2

    # Wait until both vectors are queryable
    poll_until(
        query_fn=lambda: index.query(vector=[1.0, 0.0, 0.0, 0.0], top_k=10, namespace=ns),
        check_fn=lambda r: len(r.matches) >= 2,
        timeout=120,
        description="both gRPC vectors queryable before delete",
    )

    # Delete qadg-v1
    index.delete(ids=["qadg-v1"], namespace=ns)

    # Wait until qadg-v1 is gone from fetch
    poll_until(
        query_fn=lambda: index.fetch(ids=["qadg-v1"], namespace=ns),
        check_fn=lambda r: "qadg-v1" not in r.vectors,
        timeout=120,
        description="qadg-v1 absent from gRPC fetch after delete",
    )

    # Post-delete query via gRPC
    poll_until(
        query_fn=lambda: index.query(vector=[1.0, 0.0, 0.0, 0.0], top_k=10, namespace=ns),
        check_fn=lambda r: "qadg-v1" not in {m.id for m in r.matches},
        timeout=120,
        description="qadg-v1 absent from gRPC query results after delete",
    )

    post_delete = index.query(vector=[1.0, 0.0, 0.0, 0.0], top_k=10, namespace=ns)
    assert isinstance(post_delete, QueryResponse)
    post_ids = {m.id for m in post_delete.matches}
    assert "qadg-v1" not in post_ids, (
        "qadg-v1 should not appear in gRPC query results after deletion"
    )
    assert "qadg-v2" in post_ids, (
        "qadg-v2 should still appear in gRPC query results after deleting a different vector"
    )


# ---------------------------------------------------------------------------
# delete-and-re-upsert — REST sync (resurrection sequence)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_delete_and_re_upsert_same_ids_rest(
    client: Pinecone, shared_index_dim4_cosine: str
) -> None:
    # shared_index_dim4_cosine
    """After confirmed deletion, re-upserting the same IDs with new values stores
    the new values, not the old ones (REST sync).

    Operation sequence (depth-escalation):
      1. Upsert vectors with initial values [0.1, 0.2, 0.3, 0.4] / [0.2, 0.3, 0.4, 0.5]
      2. Wait until both are fetchable (eventual consistency)
      3. Delete both vectors; wait until fetch confirms they are gone
      4. Re-upsert the same IDs with completely different values
         [0.9, 0.8, 0.7, 0.6] / [0.8, 0.7, 0.6, 0.5]
      5. Wait until the new vectors are fetchable
      6. Verify the NEW values are in the fetch response, not the old ones

    No single-operation test covers this cycle. ET-065 only verifies deletion;
    this test also exercises the re-creation path.
    """
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    index = client.index(name=shared_index_dim4_cosine)

    # Step 1: Upsert initial vectors (low-magnitude values)
    upsert_resp = index.upsert(
        vectors=[
            {"id": "reur-v1", "values": [0.1, 0.2, 0.3, 0.4]},
            {"id": "reur-v2", "values": [0.2, 0.3, 0.4, 0.5]},
        ],
        namespace=ns,
    )
    assert isinstance(upsert_resp, UpsertResponse)
    assert upsert_resp.upserted_count == 2

    # Step 2: Wait until both are fetchable
    poll_until(
        query_fn=lambda: index.fetch(ids=["reur-v1", "reur-v2"], namespace=ns),
        check_fn=lambda r: len(r.vectors) == 2,
        timeout=60,
        description="initial vectors fetchable before delete",
    )

    # Step 3: Delete both; wait until gone
    delete_resp = index.delete(ids=["reur-v1", "reur-v2"], namespace=ns)
    assert delete_resp is None

    poll_until(
        query_fn=lambda: index.fetch(ids=["reur-v1", "reur-v2"], namespace=ns),
        check_fn=lambda r: len(r.vectors) == 0,
        timeout=90,
        description="both vectors absent from fetch after delete",
    )

    # Step 4: Re-upsert the same IDs with high-magnitude values (clearly distinct)
    re_upsert_resp = index.upsert(
        vectors=[
            {"id": "reur-v1", "values": [0.9, 0.8, 0.7, 0.6]},
            {"id": "reur-v2", "values": [0.8, 0.7, 0.6, 0.5]},
        ],
        namespace=ns,
    )
    assert isinstance(re_upsert_resp, UpsertResponse)
    assert re_upsert_resp.upserted_count == 2

    # Step 5: Wait until the re-upserted vectors are fetchable
    poll_until(
        query_fn=lambda: index.fetch(ids=["reur-v1", "reur-v2"], namespace=ns),
        check_fn=lambda r: len(r.vectors) == 2,
        timeout=60,
        description="re-upserted vectors fetchable with new values",
    )

    # Step 6: Fetch and verify NEW values are returned, not old ones
    fetched = index.fetch(ids=["reur-v1", "reur-v2"], namespace=ns)
    assert isinstance(fetched, FetchResponse)
    assert "reur-v1" in fetched.vectors
    assert "reur-v2" in fetched.vectors

    v1 = fetched.vectors["reur-v1"]
    assert isinstance(v1, Vector)
    assert len(v1.values) == 4
    # New value[0] = 0.9; old value[0] = 0.1 — check we got the NEW vector
    assert v1.values[0] > 0.5, (
        f"reur-v1.values[0] should be ~0.9 (new), not ~0.1 (old); got {v1.values[0]}"
    )

    v2 = fetched.vectors["reur-v2"]
    assert isinstance(v2, Vector)
    assert len(v2.values) == 4
    # New value[0] = 0.8; old value[0] = 0.2 — check we got the NEW vector
    assert v2.values[0] > 0.5, (
        f"reur-v2.values[0] should be ~0.8 (new), not ~0.2 (old); got {v2.values[0]}"
    )


# ---------------------------------------------------------------------------
# delete-and-re-upsert — gRPC (resurrection sequence)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_delete_and_re_upsert_same_ids_grpc(
    client: Pinecone, shared_index_dim4_cosine: str
) -> None:
    # shared_index_dim4_cosine
    """After confirmed deletion, re-upserting the same IDs with new values stores
    the new values, not the old ones (gRPC transport).

    Mirrors test_delete_and_re_upsert_same_ids_rest for the gRPC transport.
    """
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    index = client.index(name=shared_index_dim4_cosine, grpc=True)

    # Step 1: Upsert initial vectors (low-magnitude values)
    upsert_resp = index.upsert(
        vectors=[
            {"id": "reurg-v1", "values": [0.1, 0.2, 0.3, 0.4]},
            {"id": "reurg-v2", "values": [0.2, 0.3, 0.4, 0.5]},
        ],
        namespace=ns,
    )
    assert isinstance(upsert_resp, UpsertResponse)
    assert upsert_resp.upserted_count == 2

    # Step 2: Wait until both are fetchable
    poll_until(
        query_fn=lambda: index.fetch(ids=["reurg-v1", "reurg-v2"], namespace=ns),
        check_fn=lambda r: len(r.vectors) == 2,
        timeout=60,
        description="initial gRPC vectors fetchable before delete",
    )

    # Step 3: Delete both; wait until gone
    delete_resp = index.delete(ids=["reurg-v1", "reurg-v2"], namespace=ns)
    assert delete_resp is None

    poll_until(
        query_fn=lambda: index.fetch(ids=["reurg-v1", "reurg-v2"], namespace=ns),
        check_fn=lambda r: len(r.vectors) == 0,
        timeout=90,
        description="both gRPC vectors absent from fetch after delete",
    )

    # Step 4: Re-upsert the same IDs with high-magnitude values (clearly distinct)
    re_upsert_resp = index.upsert(
        vectors=[
            {"id": "reurg-v1", "values": [0.9, 0.8, 0.7, 0.6]},
            {"id": "reurg-v2", "values": [0.8, 0.7, 0.6, 0.5]},
        ],
        namespace=ns,
    )
    assert isinstance(re_upsert_resp, UpsertResponse)
    assert re_upsert_resp.upserted_count == 2

    # Step 5: Wait until the re-upserted vectors are fetchable
    poll_until(
        query_fn=lambda: index.fetch(ids=["reurg-v1", "reurg-v2"], namespace=ns),
        check_fn=lambda r: len(r.vectors) == 2,
        timeout=60,
        description="re-upserted gRPC vectors fetchable with new values",
    )

    # Step 6: Fetch and verify NEW values are returned, not old ones
    fetched = index.fetch(ids=["reurg-v1", "reurg-v2"], namespace=ns)
    assert isinstance(fetched, FetchResponse)
    assert "reurg-v1" in fetched.vectors
    assert "reurg-v2" in fetched.vectors

    v1 = fetched.vectors["reurg-v1"]
    assert isinstance(v1, Vector)
    assert len(v1.values) == 4
    # New value[0] = 0.9; old value[0] = 0.1 — check we got the NEW vector
    assert v1.values[0] > 0.5, (
        f"reurg-v1.values[0] should be ~0.9 (new), not ~0.1 (old); got {v1.values[0]}"
    )

    v2 = fetched.vectors["reurg-v2"]
    assert isinstance(v2, Vector)
    assert len(v2.values) == 4
    # New value[0] = 0.8; old value[0] = 0.2 — check we got the NEW vector
    assert v2.values[0] > 0.5, (
        f"reurg-v2.values[0] should be ~0.8 (new), not ~0.2 (old); got {v2.values[0]}"
    )


# ---------------------------------------------------------------------------
# sparse_vector as SparseValues object — REST sync
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_query_with_sparse_values_object_rest(
    client: Pinecone, shared_index_dim4_dotproduct: str
) -> None:
    # shared_index_dim4_dotproduct
    """query() accepts a SparseValues typed object for sparse_vector, not just a dict (REST sync).

    Verifies the priority area "Vector variations: SparseValues as object vs dict".
    The index/__init__.py:492 branch `if isinstance(sparse_vector, SparseValues)`
    extracts indices/values from the object before sending; this path has never
    been exercised in integration tests (all existing tests pass a plain dict).

    Upserts two hybrid (dense+sparse) vectors, then queries with
    sparse_vector=SparseValues(...) and asserts the response has the expected
    structure and returns the correct matching vector.
    """
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    index = client.index(name=shared_index_dim4_dotproduct)

    # Upsert two hybrid vectors
    upserted = index.upsert(
        vectors=[
            {
                "id": "svobj-v1",
                "values": [0.1, 0.2, 0.3, 0.4],
                "sparse_values": {"indices": [0, 5], "values": [0.9, 0.7]},
            },
            {
                "id": "svobj-v2",
                "values": [0.9, 0.8, 0.7, 0.6],
                "sparse_values": {"indices": [2, 8], "values": [0.3, 0.4]},
            },
        ],
        namespace=ns,
    )
    assert upserted.upserted_count == 2

    # Wait for eventual consistency
    poll_until(
        query_fn=lambda: index.fetch(ids=["svobj-v1", "svobj-v2"], namespace=ns),
        check_fn=lambda r: len(r.vectors) == 2,
        timeout=120,
        description="svobj vectors fetchable (REST)",
    )

    # Query using a SparseValues OBJECT (not a dict) — this exercises the
    # isinstance(sparse_vector, SparseValues) branch in index/__init__.py
    sv_obj = SparseValues(indices=[0, 5], values=[0.9, 0.7])
    query_resp = index.query(
        vector=[0.1, 0.2, 0.3, 0.4],
        sparse_vector=sv_obj,
        top_k=5,
        include_values=True,
        namespace=ns,
    )

    assert isinstance(query_resp, QueryResponse)
    assert isinstance(query_resp.matches, list)
    assert len(query_resp.matches) >= 1
    match_ids = {m.id for m in query_resp.matches}
    # svobj-v1 shares the same dense vector and sparse indices — should appear
    assert "svobj-v1" in match_ids, (
        f"Expected 'svobj-v1' in results when querying with matching sparse indices; "
        f"got: {match_ids}"
    )
    # Verify response structure
    for m in query_resp.matches:
        assert isinstance(m.id, str)
        assert isinstance(m.score, float)
        assert m.values is not None and len(m.values) == 4


# ---------------------------------------------------------------------------
# sparse_vector as SparseValues object — gRPC
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_query_with_sparse_values_object_grpc(
    client: Pinecone, shared_index_dim4_dotproduct: str
) -> None:
    # shared_index_dim4_dotproduct
    """GrpcIndex.query() accepts a SparseValues typed object for sparse_vector (gRPC).

    Verifies the gRPC transport handles SparseValues objects correctly via the
    isinstance(sparse_vector, SparseValues) branch in grpc/__init__.py:360.
    """
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    index = client.index(name=shared_index_dim4_dotproduct, grpc=True)

    upserted = index.upsert(
        vectors=[
            {
                "id": "svobjg-v1",
                "values": [0.1, 0.2, 0.3, 0.4],
                "sparse_values": {"indices": [1, 4], "values": [0.8, 0.6]},
            },
            {
                "id": "svobjg-v2",
                "values": [0.9, 0.8, 0.7, 0.6],
                "sparse_values": {"indices": [2, 9], "values": [0.2, 0.5]},
            },
        ],
        namespace=ns,
    )
    assert upserted.upserted_count == 2

    poll_until(
        query_fn=lambda: index.fetch(ids=["svobjg-v1", "svobjg-v2"], namespace=ns),
        check_fn=lambda r: len(r.vectors) == 2,
        timeout=120,
        description="svobjg vectors fetchable (gRPC)",
    )

    sv_obj = SparseValues(indices=[1, 4], values=[0.8, 0.6])
    query_resp = index.query(
        vector=[0.1, 0.2, 0.3, 0.4],
        sparse_vector=sv_obj,
        top_k=5,
        include_values=True,
        namespace=ns,
    )

    assert isinstance(query_resp, QueryResponse)
    assert isinstance(query_resp.matches, list)
    assert len(query_resp.matches) >= 1
    match_ids = {m.id for m in query_resp.matches}
    assert "svobjg-v1" in match_ids, (
        f"Expected 'svobjg-v1' in gRPC results when querying with matching sparse indices; "
        f"got: {match_ids}"
    )
    for m in query_resp.matches:
        assert isinstance(m.id, str)
        assert isinstance(m.score, float)


# ---------------------------------------------------------------------------
# update values preserves metadata — REST sync
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_update_values_preserves_metadata_rest(client: Pinecone, shared_index_dim2: str) -> None:
    # shared_index_dim2
    """update(id=..., values=...) on a vector with metadata preserves the metadata (REST sync).

    Verifies unified-vec-0011 partial-update semantics: updating only the dense
    values of a vector does not touch any metadata fields attached to the same vector.

    The existing test_update_vectors_rest test upserts vectors without metadata,
    so it cannot verify this invariant.  The existing test_update_metadata_rest
    verifies the inverse (metadata merge semantics) but does not touch values.
    This test covers the composition: values update + metadata preservation.

    Area tag: update-partial
    Transport: rest
    """
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    index = client.index(name=shared_index_dim2)

    # Upsert a vector WITH metadata attached
    upserted = index.upsert(
        vectors=[
            {
                "id": "uvpm-v1",
                "values": [0.1, 0.2],
                "metadata": {"genre": "comedy", "year": 2020},
            }
        ],
        namespace=ns,
    )
    assert upserted.upserted_count == 1

    # Wait until the vector is fetchable and metadata is present
    poll_until(
        query_fn=lambda: index.fetch(ids=["uvpm-v1"], namespace=ns),
        check_fn=lambda r: (
            "uvpm-v1" in r.vectors
            and r.vectors["uvpm-v1"].metadata is not None
            and r.vectors["uvpm-v1"].metadata.get("genre") == "comedy"
        ),
        timeout=120,
        description="uvpm-v1 fetchable with metadata before values update",
    )

    # Update ONLY the values — no metadata argument supplied
    update_resp = index.update(id="uvpm-v1", values=[0.9, 0.8], namespace=ns)
    assert isinstance(update_resp, UpdateResponse)

    # Poll until the new values are reflected in fetch
    def _values_updated() -> object:
        r = index.fetch(ids=["uvpm-v1"], namespace=ns)
        if "uvpm-v1" not in r.vectors:
            return None
        vec = r.vectors["uvpm-v1"]
        if not vec.values or abs(vec.values[0] - 0.9) > 1e-3:
            return None
        return r

    updated = poll_until(
        query_fn=_values_updated,
        check_fn=lambda r: r is not None,
        timeout=120,
        description="uvpm-v1 values updated to [0.9, 0.8]",
    )

    vec = updated.vectors["uvpm-v1"]  # type: ignore[union-attr]

    # Values must be updated
    assert len(vec.values) == 2
    assert abs(vec.values[0] - 0.9) < 1e-3, (
        f"Expected values[0] ≈ 0.9 after update, got {vec.values[0]}"
    )
    assert abs(vec.values[1] - 0.8) < 1e-3, (
        f"Expected values[1] ≈ 0.8 after update, got {vec.values[1]}"
    )

    # Metadata must be PRESERVED — update(values=...) is a partial update
    assert vec.metadata is not None, "Metadata must not be None after updating only vector values"
    assert vec.metadata.get("genre") == "comedy", (
        f"Expected genre=='comedy' after values-only update, got {vec.metadata.get('genre')!r}"
    )
    assert vec.metadata.get("year") == 2020, (
        f"Expected year==2020 after values-only update, got {vec.metadata.get('year')!r}"
    )


# ---------------------------------------------------------------------------
# update values preserves metadata — gRPC
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_update_values_preserves_metadata_grpc(client: Pinecone, shared_index_dim2: str) -> None:
    # shared_index_dim2
    """update(id=..., values=...) on a vector with metadata preserves the metadata (gRPC).

    Same semantics as test_update_values_preserves_metadata_rest — verifying that
    the gRPC transport also implements partial-update semantics for vector values.

    Area tag: update-partial
    Transport: grpc
    """
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    index = client.index(name=shared_index_dim2, grpc=True)

    # Upsert a vector WITH metadata attached via gRPC
    upserted = index.upsert(
        vectors=[
            {
                "id": "uvpmg-v1",
                "values": [0.1, 0.2],
                "metadata": {"genre": "action", "year": 2022},
            }
        ],
        namespace=ns,
    )
    assert upserted.upserted_count == 1

    # Wait until the vector is fetchable with metadata
    poll_until(
        query_fn=lambda: index.fetch(ids=["uvpmg-v1"], namespace=ns),
        check_fn=lambda r: (
            "uvpmg-v1" in r.vectors
            and r.vectors["uvpmg-v1"].metadata is not None
            and r.vectors["uvpmg-v1"].metadata.get("genre") == "action"
        ),
        timeout=120,
        description="uvpmg-v1 fetchable with metadata before values update (gRPC)",
    )

    # Update ONLY the values via gRPC
    update_resp = index.update(id="uvpmg-v1", values=[0.7, 0.3], namespace=ns)
    assert isinstance(update_resp, UpdateResponse)

    # Poll until new values are reflected
    def _values_updated_grpc() -> object:
        r = index.fetch(ids=["uvpmg-v1"], namespace=ns)
        if "uvpmg-v1" not in r.vectors:
            return None
        vec = r.vectors["uvpmg-v1"]
        if not vec.values or abs(vec.values[0] - 0.7) > 1e-3:
            return None
        return r

    updated = poll_until(
        query_fn=_values_updated_grpc,
        check_fn=lambda r: r is not None,
        timeout=120,
        description="uvpmg-v1 values updated to [0.7, 0.3] (gRPC)",
    )

    vec = updated.vectors["uvpmg-v1"]  # type: ignore[union-attr]

    # Values must be updated
    assert abs(vec.values[0] - 0.7) < 1e-3, (
        f"Expected values[0] ≈ 0.7 after gRPC update, got {vec.values[0]}"
    )
    assert abs(vec.values[1] - 0.3) < 1e-3, (
        f"Expected values[1] ≈ 0.3 after gRPC update, got {vec.values[1]}"
    )

    # Metadata must be PRESERVED
    assert vec.metadata is not None, "Metadata must not be None after gRPC values-only update"
    assert vec.metadata.get("genre") == "action", (
        f"Expected genre=='action' after gRPC values-only update, got {vec.metadata.get('genre')!r}"
    )
    assert vec.metadata.get("year") == 2022, (
        f"Expected year==2022 after gRPC values-only update, got {vec.metadata.get('year')!r}"
    )
