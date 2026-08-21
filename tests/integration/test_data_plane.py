"""Integration tests for data-plane vector operations (sync REST + gRPC).

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

from __future__ import annotations

import uuid
from concurrent.futures import as_completed
from typing import Any

import httpx
import orjson
import pytest
import respx

from pinecone import GrpcIndex, Index, Pinecone
from pinecone.errors import ApiError, ConflictError, NotFoundError, PineconeValueError
from pinecone.errors.exceptions import ValidationError
from pinecone.grpc.future import PineconeFuture
from pinecone.models.namespaces.models import ListNamespacesResponse, NamespaceDescription
from pinecone.models.vectors.responses import (
    DescribeIndexStatsResponse,
    FetchResponse,
    ListItem,
    ListResponse,
    NamespaceSummary,
    QueryResponse,
    UpdateResponse,
    UpsertResponse,
)
from pinecone.models.vectors.vector import ScoredVector, Vector
from tests.integration.conftest import (
    LegacyIndexFactory,
    cleanup_resource,
    poll_until,
)
from tests.integration.legacy_index import assert_serves_vectors_api

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


# ---------------------------------------------------------------------------
# delete-vectors — REST sync
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_delete_vectors_rest(client: Pinecone, shared_index_dim2: str) -> None:
    # shared_index_dim2
    """Delete vectors by IDs via REST Index and verify they are gone."""
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    index = client.index(name=shared_index_dim2)

    upserted = index.upsert(
        vectors=[
            {"id": "del-v1", "values": [0.1, 0.2]},
            {"id": "del-v2", "values": [0.3, 0.4]},
            {"id": "del-v3", "values": [0.5, 0.6]},
        ],
        namespace=ns,
    )
    assert upserted.upserted_count == 3

    # Wait for all 3 vectors to be fetchable (eventual consistency)
    poll_until(
        query_fn=lambda: index.fetch(ids=["del-v1", "del-v2", "del-v3"], namespace=ns),
        check_fn=lambda r: len(r.vectors) == 3,
        timeout=120,
        description="all 3 vectors fetchable before delete",
    )

    # Delete just v1 and v2 by IDs
    result = index.delete(ids=["del-v1", "del-v2"], namespace=ns)
    assert result is None  # delete returns None on success

    # Wait until deleted vectors are gone (eventual consistency)
    poll_until(
        query_fn=lambda: index.fetch(ids=["del-v1", "del-v2"], namespace=ns),
        check_fn=lambda r: len(r.vectors) == 0,
        timeout=120,
        description="deleted vectors gone after delete",
    )

    # Verify v3 is still present
    remaining = index.fetch(ids=["del-v3"], namespace=ns)
    assert isinstance(remaining, FetchResponse)
    assert "del-v3" in remaining.vectors
    assert "del-v1" not in remaining.vectors
    assert "del-v2" not in remaining.vectors


# ---------------------------------------------------------------------------
# delete-vectors — gRPC
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_delete_vectors_grpc(client: Pinecone, shared_index_dim2: str) -> None:
    # shared_index_dim2
    """Delete vectors by IDs via GrpcIndex and verify they are gone."""
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    index = client.index(name=shared_index_dim2, grpc=True)

    upserted = index.upsert(
        vectors=[
            {"id": "del-v1", "values": [0.1, 0.2]},
            {"id": "del-v2", "values": [0.3, 0.4]},
        ],
        namespace=ns,
    )
    assert upserted.upserted_count == 2

    index.delete(ids=["del-v1"], namespace=ns)

    remaining = index.fetch(ids=["del-v1", "del-v2"], namespace=ns)
    assert "del-v1" not in remaining.vectors
    assert "del-v2" in remaining.vectors


# ---------------------------------------------------------------------------
# upsert — REST sync
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_upsert_vectors_rest(client: Pinecone, shared_index_dim2: str) -> None:
    # shared_index_dim2
    """Upsert vectors via REST Index and verify upserted_count."""
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    index = client.index(name=shared_index_dim2)

    result = index.upsert(
        vectors=[
            {"id": "v1", "values": [0.1, 0.2]},
            {"id": "v2", "values": [0.3, 0.4]},
            {"id": "v3", "values": [0.5, 0.6]},
        ],
        namespace=ns,
    )

    assert isinstance(result, UpsertResponse)
    assert result.upserted_count == 3


# ---------------------------------------------------------------------------
# query — REST sync
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_query_by_vector_rest(client: Pinecone, shared_index_dim2: str) -> None:
    # shared_index_dim2
    """Query by vector via REST Index and verify matches structure."""
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    index = client.index(name=shared_index_dim2)

    upserted = index.upsert(
        vectors=[
            {"id": "v1", "values": [0.1, 0.2]},
            {"id": "v2", "values": [0.3, 0.4]},
            {"id": "v3", "values": [0.5, 0.6]},
        ],
        namespace=ns,
    )
    assert upserted.upserted_count == 3

    # Wait for all 3 vectors to be queryable (eventual consistency)
    poll_until(
        query_fn=lambda: index.query(vector=[0.1, 0.2], top_k=3, namespace=ns),
        check_fn=lambda r: len(r.matches) == 3,
        timeout=120,
        description="all 3 vectors queryable after upsert",
    )

    result = index.query(vector=[0.1, 0.2], top_k=2, namespace=ns)

    assert isinstance(result, QueryResponse)
    assert len(result.matches) == 2
    for match in result.matches:
        assert isinstance(match, ScoredVector)
        assert isinstance(match.id, str)
        assert isinstance(match.score, float)
    # Keyed by id, never by position: the server returns matches unsorted by
    # score (#368), so asserting an order would encode a server bug as our
    # contract and break when it is fixed. The set and the argmax are checked
    # instead, which is what "the right neighbours, scored right" means.
    by_match = {m.id: m for m in result.matches}
    assert "v1" in by_match, f"Self-match v1 should be in top-2 results, got ids={set(by_match)}"
    assert by_match["v1"].score == pytest.approx(1.0, abs=1e-2)
    assert max(by_match, key=lambda i: by_match[i].score) == "v1"


# ---------------------------------------------------------------------------
# upsert — gRPC
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_upsert_vectors_grpc(client: Pinecone, shared_index_dim2: str) -> None:
    # shared_index_dim2
    """Upsert vectors via GrpcIndex and verify upserted_count."""
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    index = client.index(name=shared_index_dim2, grpc=True)

    result = index.upsert(
        vectors=[
            {"id": "v1", "values": [0.1, 0.2]},
            {"id": "v2", "values": [0.3, 0.4]},
            {"id": "v3", "values": [0.5, 0.6]},
        ],
        namespace=ns,
    )

    assert isinstance(result, UpsertResponse)
    assert result.upserted_count == 3


# ---------------------------------------------------------------------------
# query — gRPC
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_query_by_vector_grpc(client: Pinecone, shared_index_dim2: str) -> None:
    # shared_index_dim2
    """Query by vector via GrpcIndex and verify matches structure."""
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    index = client.index(name=shared_index_dim2, grpc=True)

    upserted = index.upsert(
        vectors=[
            {"id": "v1", "values": [0.1, 0.2]},
            {"id": "v2", "values": [0.3, 0.4]},
        ],
        namespace=ns,
    )
    assert upserted.upserted_count == 2

    poll_until(
        query_fn=lambda: index.query(vector=[0.1, 0.2], top_k=1, namespace=ns),
        check_fn=lambda r: len(r.matches) >= 1,
        timeout=120,
        description="vectors queryable after upsert",
    )

    result = index.query(vector=[0.1, 0.2], top_k=1, namespace=ns)

    assert isinstance(result, QueryResponse)
    # Keyed by id, never by position: the server returns matches unsorted by
    # score (#368), so asserting an order would encode a server bug as our
    # contract and break when it is fixed.
    assert len(result.matches) == 1
    assert {m.id for m in result.matches} == {"v1"}


# ---------------------------------------------------------------------------
# fetch — REST sync
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_fetch_vectors_rest(client: Pinecone, shared_index_dim2: str) -> None:
    # shared_index_dim2
    """Fetch vectors by ID via REST Index and verify returned vector data."""
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    index = client.index(name=shared_index_dim2)

    upserted = index.upsert(
        vectors=[
            {"id": "v1", "values": [0.1, 0.2]},
            {"id": "v2", "values": [0.3, 0.4]},
            {"id": "v3", "values": [0.5, 0.6]},
        ],
        namespace=ns,
    )
    assert upserted.upserted_count == 3

    # Wait for vectors to be fetchable (eventual consistency)
    poll_until(
        query_fn=lambda: index.fetch(ids=["v1", "v2", "v3"], namespace=ns),
        check_fn=lambda r: len(r.vectors) == 3,
        timeout=120,
        description="all 3 vectors fetchable after upsert",
    )

    result = index.fetch(ids=["v1", "v2"], namespace=ns)

    assert isinstance(result, FetchResponse)
    assert "v1" in result.vectors
    assert "v2" in result.vectors
    assert "v3" not in result.vectors

    v1 = result.vectors["v1"]
    assert isinstance(v1, Vector)
    assert v1.id == "v1"
    assert len(v1.values) == 2
    assert abs(v1.values[0] - 0.1) < 1e-5
    assert abs(v1.values[1] - 0.2) < 1e-5

    assert isinstance(result.namespace, str)


# ---------------------------------------------------------------------------
# fetch — gRPC
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_fetch_vectors_grpc(client: Pinecone, shared_index_dim2: str) -> None:
    # shared_index_dim2
    """Fetch vectors by ID via GrpcIndex and verify returned vector data."""
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    index = client.index(name=shared_index_dim2, grpc=True)

    upserted = index.upsert(
        vectors=[
            {"id": "v1", "values": [0.1, 0.2]},
            {"id": "v2", "values": [0.3, 0.4]},
        ],
        namespace=ns,
    )
    assert upserted.upserted_count == 2

    poll_until(
        query_fn=lambda: index.fetch(ids=["v1", "v2"], namespace=ns),
        check_fn=lambda r: len(r.vectors) == 2,
        timeout=120,
        description="vectors fetchable after upsert",
    )

    result = index.fetch(ids=["v1", "v2"], namespace=ns)

    assert isinstance(result, FetchResponse)
    assert "v1" in result.vectors
    assert "v2" in result.vectors
    assert isinstance(result.vectors["v1"], Vector)


# ---------------------------------------------------------------------------
# list-vectors — REST sync
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_list_vectors_rest(client: Pinecone, shared_index_dim2: str) -> None:
    # shared_index_dim2
    """List vectors via REST Index and verify pagination structure and IDs."""
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    index = client.index(name=shared_index_dim2)

    upserted = index.upsert(
        vectors=[
            {"id": "lst-v1", "values": [0.1, 0.2]},
            {"id": "lst-v2", "values": [0.3, 0.4]},
            {"id": "lst-v3", "values": [0.5, 0.6]},
        ],
        namespace=ns,
    )
    assert upserted.upserted_count == 3

    # Wait for all 3 vectors to appear in list results (eventual consistency)
    def _collect_ids() -> list[str]:
        return [
            item.id
            for page in index.list(prefix="lst-", namespace=ns)
            for item in page.vectors
            if item.id is not None
        ]

    poll_until(
        query_fn=_collect_ids,
        check_fn=lambda ids: len(ids) >= 3,
        timeout=120,
        description="all 3 vectors listable after upsert",
    )

    # Collect all pages and verify structure
    pages: list[ListResponse] = list(index.list(prefix="lst-", namespace=ns))

    assert len(pages) >= 1, "expected at least one page"
    all_ids: list[str] = []
    for page in pages:
        assert isinstance(page, ListResponse)
        assert isinstance(page.namespace, str)
        for item in page.vectors:
            assert isinstance(item, ListItem)
            assert isinstance(item.id, str)
            all_ids.append(item.id)

    assert "lst-v1" in all_ids
    assert "lst-v2" in all_ids
    assert "lst-v3" in all_ids


# ---------------------------------------------------------------------------
# list-vectors — gRPC
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_list_vectors_grpc(client: Pinecone, shared_index_dim2: str) -> None:
    # shared_index_dim2
    """List vectors via GrpcIndex and verify structure."""
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    index = client.index(name=shared_index_dim2, grpc=True)

    upserted = index.upsert(
        vectors=[
            {"id": "lst-v1", "values": [0.1, 0.2]},
            {"id": "lst-v2", "values": [0.3, 0.4]},
        ],
        namespace=ns,
    )
    assert upserted.upserted_count == 2

    poll_until(
        query_fn=lambda: index.fetch(ids=["lst-v1", "lst-v2"], namespace=ns),
        check_fn=lambda r: len(r.vectors) == 2,
        timeout=120,
        description="vectors fetchable before list",
    )

    all_ids: list[str] = []
    for page in index.list(prefix="lst-", namespace=ns):
        assert isinstance(page, ListResponse)
        for item in page.vectors:
            assert isinstance(item, ListItem)
            if item.id is not None:
                all_ids.append(item.id)

    assert "lst-v1" in all_ids
    assert "lst-v2" in all_ids


# ---------------------------------------------------------------------------
# update-vectors — REST sync
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_update_vectors_rest(client: Pinecone, shared_index_dim2: str) -> None:
    # shared_index_dim2
    """Update a vector's values via REST Index and verify the change is reflected."""
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    index = client.index(name=shared_index_dim2)

    upserted = index.upsert(
        vectors=[
            {"id": "upd-v1", "values": [0.1, 0.2]},
            {"id": "upd-v2", "values": [0.3, 0.4]},
        ],
        namespace=ns,
    )
    assert upserted.upserted_count == 2

    # Wait for vectors to be fetchable (eventual consistency)
    poll_until(
        query_fn=lambda: index.fetch(ids=["upd-v1", "upd-v2"], namespace=ns),
        check_fn=lambda r: len(r.vectors) == 2,
        timeout=120,
        description="vectors fetchable before update",
    )

    # Update upd-v1 with new values
    result = index.update(id="upd-v1", values=[0.9, 0.8], namespace=ns)

    assert isinstance(result, UpdateResponse)
    # The update API returns {} on success; matched_records may be None
    assert result.matched_records is None or isinstance(result.matched_records, int)

    # Poll until the updated values are reflected
    poll_until(
        query_fn=lambda: index.fetch(ids=["upd-v1"], namespace=ns),
        check_fn=lambda r: (
            "upd-v1" in r.vectors
            and len(r.vectors["upd-v1"].values) == 2
            and abs(r.vectors["upd-v1"].values[0] - 0.9) < 1e-4
        ),
        timeout=120,
        description="updated values reflected in fetch",
    )

    # Verify upd-v2 was not modified
    check = index.fetch(ids=["upd-v2"], namespace=ns)
    assert abs(check.vectors["upd-v2"].values[0] - 0.3) < 1e-4


# ---------------------------------------------------------------------------
# update-vectors — gRPC
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_update_vectors_grpc(client: Pinecone, shared_index_dim2: str) -> None:
    # shared_index_dim2
    """Update a vector's values via GrpcIndex and verify the change is reflected."""
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    index = client.index(name=shared_index_dim2, grpc=True)

    upserted = index.upsert(
        vectors=[
            {"id": "upd-v1", "values": [0.1, 0.2]},
        ],
        namespace=ns,
    )
    assert upserted.upserted_count == 1

    poll_until(
        query_fn=lambda: index.fetch(ids=["upd-v1"], namespace=ns),
        check_fn=lambda r: "upd-v1" in r.vectors,
        timeout=120,
        description="vector fetchable before update",
    )

    result = index.update(id="upd-v1", values=[0.9, 0.8], namespace=ns)
    assert isinstance(result, UpdateResponse)

    poll_until(
        query_fn=lambda: index.fetch(ids=["upd-v1"], namespace=ns),
        check_fn=lambda r: (
            "upd-v1" in r.vectors and abs(r.vectors["upd-v1"].values[0] - 0.9) < 1e-4
        ),
        timeout=120,
        description="updated values reflected in fetch",
    )


# ---------------------------------------------------------------------------
# describe-stats — REST sync
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_describe_index_stats_rest(client: Pinecone, shared_index_dim3: str) -> None:
    # shared_index_dim3
    """Call describe_index_stats() via REST Index and verify the response structure."""
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    index = client.index(name=shared_index_dim3)

    # Upsert a few vectors so stats are non-trivial
    upserted = index.upsert(
        vectors=[
            {"id": "st-v1", "values": [0.1, 0.2, 0.3]},
            {"id": "st-v2", "values": [0.4, 0.5, 0.6]},
        ],
        namespace=ns,
    )
    assert upserted.upserted_count == 2

    # Wait until our namespace appears in stats (eventual consistency)
    poll_until(
        query_fn=lambda: index.describe_index_stats(),
        check_fn=lambda r: ns in r.namespaces and r.namespaces[ns].vector_count >= 1,
        timeout=120,
        description="test namespace counted in stats after upsert",
    )

    stats = index.describe_index_stats()

    assert isinstance(stats, DescribeIndexStatsResponse)
    assert stats.dimension == 3
    assert stats.total_vector_count >= 1
    assert isinstance(stats.index_fullness, float)
    assert 0.0 <= stats.index_fullness <= 1.0
    assert isinstance(stats.namespaces, dict)
    assert len(stats.namespaces) >= 1
    # Verify at least one namespace entry has the expected structure
    for ns_name, ns_summary in stats.namespaces.items():
        assert isinstance(ns_name, str)
        assert isinstance(ns_summary, NamespaceSummary)
        assert isinstance(ns_summary.vector_count, int)
        assert ns_summary.vector_count >= 0
    # Total across namespaces should match total_vector_count
    ns_total = sum(ns_summary.vector_count for ns_summary in stats.namespaces.values())
    assert ns_total == stats.total_vector_count


# ---------------------------------------------------------------------------
# describe-stats — gRPC
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_describe_index_stats_grpc(client: Pinecone, shared_index_dim3: str) -> None:
    # shared_index_dim3
    """Call describe_index_stats() via GrpcIndex and verify response structure."""
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    index = client.index(name=shared_index_dim3, grpc=True)

    upserted = index.upsert(
        vectors=[
            {"id": "st-v1", "values": [0.1, 0.2, 0.3]},
        ],
        namespace=ns,
    )
    assert upserted.upserted_count == 1

    poll_until(
        query_fn=lambda: index.describe_index_stats(),
        check_fn=lambda r: r.total_vector_count >= 1,
        timeout=120,
        description="at least 1 vector counted in stats after upsert",
    )

    stats = index.describe_index_stats()

    assert isinstance(stats, DescribeIndexStatsResponse)
    assert stats.dimension == 3
    assert isinstance(stats.total_vector_count, int)
    assert isinstance(stats.index_fullness, float)


# ---------------------------------------------------------------------------
# namespaces — REST sync
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_namespaces_rest(client: Pinecone, shared_index_dim2: str) -> None:
    # shared_index_dim2
    """Upsert to named namespace, query within it, verify namespace isolation."""
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    named_ns = f"{ns}-alpha"
    def_ns = f"{ns}-def"
    index = client.index(name=shared_index_dim2)

    # Upsert vectors into named namespace
    ns_result = index.upsert(
        vectors=[
            {"id": "ns-v1", "values": [0.1, 0.2]},
            {"id": "ns-v2", "values": [0.3, 0.4]},
        ],
        namespace=named_ns,
    )
    assert isinstance(ns_result, UpsertResponse)
    assert ns_result.upserted_count == 2

    # Upsert different vectors into the per-test default namespace
    def_result = index.upsert(
        vectors=[
            {"id": "def-v1", "values": [0.9, 0.8]},
        ],
        namespace=def_ns,
    )
    assert isinstance(def_result, UpsertResponse)
    assert def_result.upserted_count == 1

    # Wait until ns-alpha vectors are queryable in the named namespace
    poll_until(
        query_fn=lambda: index.query(vector=[0.1, 0.2], top_k=10, namespace=named_ns),
        check_fn=lambda r: len(r.matches) >= 2,
        timeout=120,
        description="named namespace vectors queryable",
    )

    # Query in the named namespace
    ns_query = index.query(vector=[0.1, 0.2], top_k=10, namespace=named_ns)
    assert isinstance(ns_query, QueryResponse)
    assert ns_query.namespace == named_ns
    ns_ids = {m.id for m in ns_query.matches}
    assert "ns-v1" in ns_ids
    assert "ns-v2" in ns_ids
    # Default namespace vectors must NOT appear in named-namespace query
    assert "def-v1" not in ns_ids

    # Verify stats shows the named namespace with exactly 2 vectors
    poll_until(
        query_fn=lambda: index.describe_index_stats(),
        check_fn=lambda s: named_ns in s.namespaces and s.namespaces[named_ns].vector_count == 2,
        timeout=120,
        description="stats reflect 2 vectors in named namespace",
    )
    stats = index.describe_index_stats()
    assert isinstance(stats.namespaces, dict)
    assert named_ns in stats.namespaces
    assert stats.namespaces[named_ns].vector_count == 2


# ---------------------------------------------------------------------------
# namespaces — gRPC
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_namespaces_grpc(client: Pinecone, shared_index_dim2: str) -> None:
    # shared_index_dim2
    """Upsert to named namespace via GrpcIndex and query within it."""
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    named_ns = f"{ns}-alpha"
    index = client.index(name=shared_index_dim2, grpc=True)

    upserted = index.upsert(
        vectors=[
            {"id": "ns-v1", "values": [0.1, 0.2]},
            {"id": "ns-v2", "values": [0.3, 0.4]},
        ],
        namespace=named_ns,
    )
    assert upserted.upserted_count == 2

    poll_until(
        query_fn=lambda: index.query(vector=[0.1, 0.2], top_k=10, namespace=named_ns),
        check_fn=lambda r: len(r.matches) >= 1,
        timeout=120,
        description="named namespace vectors queryable",
    )

    result = index.query(vector=[0.1, 0.2], top_k=10, namespace=named_ns)
    assert isinstance(result, QueryResponse)
    assert result.namespace == named_ns
    ids = {m.id for m in result.matches}
    assert "ns-v1" in ids or "ns-v2" in ids


# ---------------------------------------------------------------------------
# list_paginated — REST sync
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_list_paginated_returns_single_page_rest(client: Pinecone, shared_index_dim2: str) -> None:
    # shared_index_dim2
    """list_paginated() returns one page with correct structure; no token on last page."""
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    index = client.index(name=shared_index_dim2)

    # Upsert 4 vectors with a shared prefix so we can filter and verify limit
    upserted = index.upsert(
        vectors=[
            {"id": "pg-v1", "values": [0.1, 0.2]},
            {"id": "pg-v2", "values": [0.3, 0.4]},
            {"id": "pg-v3", "values": [0.5, 0.6]},
            {"id": "pg-v4", "values": [0.7, 0.8]},
        ],
        namespace=ns,
    )
    assert upserted.upserted_count == 4

    # Wait until all 4 vectors appear in list results
    poll_until(
        query_fn=lambda: index.list_paginated(prefix="pg-", limit=100, namespace=ns),
        check_fn=lambda r: len(r.vectors) >= 4,
        timeout=120,
        description="all 4 vectors listable after upsert",
    )

    # 1. list_paginated() returns a ListResponse (not a generator)
    page = index.list_paginated(prefix="pg-", limit=100, namespace=ns)
    assert isinstance(page, ListResponse)

    # 2. Vector list contains ListItem objects with string IDs
    assert isinstance(page.vectors, list)
    assert len(page.vectors) >= 4
    for item in page.vectors:
        assert isinstance(item, ListItem)
        assert isinstance(item.id, str)

    # 3. All 4 IDs are present
    ids = {item.id for item in page.vectors}
    assert {"pg-v1", "pg-v2", "pg-v3", "pg-v4"} <= ids

    # 4. Namespace field is a string
    assert isinstance(page.namespace, str)

    # 5. No pagination token when the page contains all results (last page)
    #    The claim unified-vec-0056 says: no token when this is the final page.
    assert page.pagination is None or page.pagination.next is None

    # 6. list_paginated with limit=2 returns at most 2 items (limit is respected)
    limited_page = index.list_paginated(prefix="pg-", limit=2, namespace=ns)
    assert isinstance(limited_page, ListResponse)
    assert len(limited_page.vectors) <= 2


# ---------------------------------------------------------------------------
# list_paginated — gRPC
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_list_paginated_returns_single_page_grpc(client: Pinecone, shared_index_dim2: str) -> None:
    # shared_index_dim2
    """list_paginated() via GrpcIndex returns one page with correct structure."""
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    index = client.index(name=shared_index_dim2, grpc=True)

    upserted = index.upsert(
        vectors=[
            {"id": "pg-v1", "values": [0.1, 0.2]},
            {"id": "pg-v2", "values": [0.3, 0.4]},
            {"id": "pg-v3", "values": [0.5, 0.6]},
        ],
        namespace=ns,
    )
    assert upserted.upserted_count == 3

    # Wait until all 3 vectors are listable
    poll_until(
        query_fn=lambda: index.list_paginated(prefix="pg-", limit=100, namespace=ns),
        check_fn=lambda r: len(r.vectors) >= 3,
        timeout=120,
        description="all 3 vectors listable after upsert",
    )

    page = index.list_paginated(prefix="pg-", limit=100, namespace=ns)
    assert isinstance(page, ListResponse)
    assert isinstance(page.vectors, list)
    assert len(page.vectors) >= 3
    for item in page.vectors:
        assert isinstance(item, ListItem)
        assert isinstance(item.id, str)
    ids = {item.id for item in page.vectors}
    assert {"pg-v1", "pg-v2", "pg-v3"} <= ids
    # No pagination token on the final page
    assert page.pagination is None or page.pagination.next is None


# ---------------------------------------------------------------------------
# describe-stats with filter — serverless rejects (REST sync)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_describe_index_stats_filter_unsupported_on_serverless_rest(
    client: Pinecone, shared_index_dim3: str
) -> None:
    # shared_index_dim3
    """Verify describe_index_stats(filter=...) raises ApiError(400) on a serverless index."""
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    index = client.index(name=shared_index_dim3)

    # Upsert one vector so the index has some content
    upserted = index.upsert(vectors=[{"id": "fs-v1", "values": [0.1, 0.2, 0.3]}], namespace=ns)
    assert upserted.upserted_count == 1

    # The filter parameter is not supported on serverless/starter indexes —
    # the API returns 400 and the SDK should surface it as ApiError.
    with pytest.raises(ApiError) as exc_info:
        index.describe_index_stats(filter={"tag": {"$eq": "a"}})

    assert exc_info.value.status_code == 400


# ---------------------------------------------------------------------------
# describe-stats with filter — serverless rejects (gRPC)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_describe_index_stats_filter_unsupported_on_serverless_grpc(
    client: Pinecone, shared_index_dim3: str
) -> None:
    # shared_index_dim3
    """Verify describe_index_stats(filter=...) raises an error on a serverless index via gRPC."""
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    index = client.index(name=shared_index_dim3, grpc=True)

    upserted = index.upsert(vectors=[{"id": "fg-v1", "values": [0.1, 0.2, 0.3]}], namespace=ns)
    assert upserted.upserted_count == 1

    # The filter parameter is not supported on serverless/starter indexes via gRPC either.
    with pytest.raises(Exception):
        index.describe_index_stats(filter={"tag": {"$eq": "a"}})


# ---------------------------------------------------------------------------
# namespace CRUD — REST sync
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_namespace_crud_lifecycle_rest(client: Pinecone, shared_index_dim2: str) -> None:
    # shared_index_dim2
    """create_namespace / describe_namespace / list_namespaces_paginated / delete_namespace round-trip.

    Verifies claims:
    - unified-ns-0001: Can create a named namespace.
    - unified-ns-0002: Creation returns name and record_count == 0.
    - unified-ns-0003: Can describe a namespace by name.
    - unified-ns-0004: Can delete a namespace by name.
    - unified-ns-0005: Can list all namespaces with optional prefix filtering.
    - unified-ns-0008: Namespace list response omits pagination token on the final page.

    ``size_bytes`` is asserted as a non-negative int rather than an exact value:
    the backend documents it as approximate (see #198), so a fresh namespace may
    report 0 whether or not size tracking has caught up.
    """
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    ns_name = f"{ns}-alpha"
    index = client.index(name=shared_index_dim2)

    # 1. Create namespace — returns NamespaceDescription with record_count == 0
    created = index.create_namespace(name=ns_name)
    assert isinstance(created, NamespaceDescription)
    assert created.name == ns_name
    assert created.record_count == 0  # unified-ns-0002
    assert isinstance(created.size_bytes, int) and created.size_bytes >= 0

    # 2. Describe namespace — returns same details
    described = index.describe_namespace(name=ns_name)
    assert isinstance(described, NamespaceDescription)
    assert described.name == ns_name
    assert isinstance(described.record_count, int)
    assert isinstance(described.size_bytes, int) and described.size_bytes >= 0

    # 2b. Creating the reserved default namespace is refused client-side
    with pytest.raises(ValidationError, match="reserved and cannot be created"):
        index.create_namespace(name="__default__")

    # 2c. Describing a namespace that does not exist is a 404
    with pytest.raises(NotFoundError):
        index.describe_namespace(name=f"{ns}-absent")

    # 3. Namespace appears in list_namespaces_paginated with prefix match
    list_resp = index.list_namespaces_paginated(prefix=f"{ns}-", limit=100)
    assert isinstance(list_resp, ListNamespacesResponse)
    ns_names = [ns_item.name for ns_item in list_resp.namespaces]
    assert ns_name in ns_names

    # Each entry is a NamespaceDescription with a string name
    for ns_item in list_resp.namespaces:
        assert isinstance(ns_item, NamespaceDescription)
        assert isinstance(ns_item.name, str)
        assert isinstance(ns_item.record_count, int)

    # 4. Pagination token is absent on the final page (unified-ns-0008)
    assert list_resp.pagination is None or list_resp.pagination.next is None

    # 5. Delete namespace — returns None on success
    result = index.delete_namespace(name=ns_name)
    assert result is None  # unified-ns-0004

    # 6. After deletion, namespace no longer appears in listing (eventual consistency)
    post_delete = poll_until(
        query_fn=lambda: index.list_namespaces_paginated(prefix=f"{ns}-", limit=100),
        check_fn=lambda r: ns_name not in [ns_item.name for ns_item in r.namespaces],
        timeout=60,
        description=f"namespace {ns_name} removed from listing after delete",
    )
    assert isinstance(post_delete, ListNamespacesResponse)


# ---------------------------------------------------------------------------
# namespace CRUD — gRPC
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_namespace_crud_lifecycle_grpc(client: Pinecone, shared_index_dim2: str) -> None:
    # shared_index_dim2
    """GrpcIndex create_namespace / describe_namespace / list_namespaces_paginated / delete_namespace round-trip.

    Verifies claims:
    - unified-ns-0001: Can create a named namespace via gRPC.
    - unified-ns-0002: Creation returns name and record_count == 0.
    - unified-ns-0003: Can describe a namespace by name via gRPC.
    - unified-ns-0004: Can delete a namespace by name via gRPC; returns None.
    - unified-ns-0005: Can list namespaces via gRPC with optional prefix filtering.
    """
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    ns_name = f"{ns}-grpc"
    index: GrpcIndex = client.index(name=shared_index_dim2, grpc=True)

    # 1. Create namespace — returns NamespaceDescription with record_count == 0
    created = index.create_namespace(name=ns_name)
    assert isinstance(created, NamespaceDescription)
    assert created.name == ns_name
    assert created.record_count == 0  # unified-ns-0002

    # 2. Describe namespace — poll for eventual consistency after create
    described = poll_until(
        query_fn=lambda: index.describe_namespace(name=ns_name),
        check_fn=lambda r: isinstance(r, NamespaceDescription),
        timeout=60,
        description=f"namespace {ns_name} visible after create via gRPC",
    )
    assert isinstance(described, NamespaceDescription)
    assert described.name == ns_name
    assert isinstance(described.record_count, int)

    # 3. Namespace appears in list_namespaces_paginated with prefix match
    list_resp = poll_until(
        query_fn=lambda: index.list_namespaces_paginated(prefix=f"{ns}-", limit=100),
        check_fn=lambda r: ns_name in [ns_item.name for ns_item in r.namespaces],
        timeout=60,
        description=f"namespace {ns_name} visible in listing after create via gRPC",
    )
    assert isinstance(list_resp, ListNamespacesResponse)

    # 4. Delete namespace — returns None on success (unified-ns-0004)
    result = index.delete_namespace(name=ns_name)
    assert result is None

    # 5. After deletion, namespace no longer appears in listing (eventual consistency)
    post_delete = poll_until(
        query_fn=lambda: index.list_namespaces_paginated(prefix=f"{ns}-", limit=100),
        check_fn=lambda r: ns_name not in [ns_item.name for ns_item in r.namespaces],
        timeout=60,
        description=f"namespace {ns_name} removed from listing after delete via gRPC",
    )
    assert isinstance(post_delete, ListNamespacesResponse)


# ---------------------------------------------------------------------------
# list_namespaces generator — REST sync
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_list_namespaces_generator_rest(client: Pinecone, shared_index_dim2: str) -> None:
    # shared_index_dim2
    """list_namespaces() generator yields ListNamespacesResponse pages with NamespaceDescription
    items; generator follows pagination tokens automatically until exhausted.

    Verifies claims:
    - unified-ns-0007: The namespace list generator yields individual NamespaceDescription
      objects per page; the paginated method returns one page (not auto-paginating).
    - unified-ns-0005: Can list all namespaces in an index with optional prefix filtering.
    """
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    ns_a = f"{ns}-a"
    ns_b = f"{ns}-b"
    index = client.index(name=shared_index_dim2)

    # Upsert one vector into each namespace to create them implicitly
    upserted = index.upsert(vectors=[{"id": "lnsg-v1", "values": [0.1, 0.2]}], namespace=ns_a)
    assert upserted.upserted_count == 1
    upserted = index.upsert(vectors=[{"id": "lnsg-v2", "values": [0.3, 0.4]}], namespace=ns_b)
    assert upserted.upserted_count == 1

    # Poll until both namespaces appear in list_namespaces_paginated
    poll_until(
        query_fn=lambda: index.list_namespaces_paginated(prefix=f"{ns}-", limit=100),
        check_fn=lambda r: len(r.namespaces) >= 2,
        timeout=120,
        description="both test namespaces appear via list_namespaces_paginated",
    )

    # --- Exercise the generator ---
    pages = list(index.list_namespaces(prefix=f"{ns}-"))
    assert len(pages) >= 1, "list_namespaces() generator must yield at least one page"

    # Collect all namespace names across all yielded pages
    all_ns_names = [ns_item.name for page in pages for ns_item in page.namespaces]
    assert ns_a in all_ns_names, f"Expected {ns_a!r} in generator output; got {all_ns_names}"
    assert ns_b in all_ns_names, f"Expected {ns_b!r} in generator output; got {all_ns_names}"

    # Verify shape of every yielded page and its namespace descriptions
    for page in pages:
        assert isinstance(page, ListNamespacesResponse)
        assert len(page.namespaces) >= 1, "Each yielded page must contain at least one namespace"
        for ns_item in page.namespaces:
            assert isinstance(ns_item, NamespaceDescription)
            assert isinstance(ns_item.name, str) and ns_item.name.startswith(f"{ns}-")
            assert isinstance(ns_item.record_count, int) and ns_item.record_count >= 0


# ---------------------------------------------------------------------------
# list_namespaces generator — gRPC
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_list_namespaces_generator_grpc(client: Pinecone, shared_index_dim2: str) -> None:
    # shared_index_dim2
    """list_namespaces() generator on GrpcIndex yields ListNamespacesResponse pages;
    generator follows pagination tokens automatically until exhausted.

    Verifies claims:
    - unified-ns-0007 (gRPC): The namespace list generator yields ListNamespacesResponse
      pages; the paginated method returns one page (not auto-paginating).
    """
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    ns_a = f"{ns}-a"
    ns_b = f"{ns}-b"
    index = client.index(name=shared_index_dim2, grpc=True)

    # Upsert one vector into each namespace to create them implicitly
    upserted = index.upsert(vectors=[{"id": "lnsg-v1", "values": [0.1, 0.2]}], namespace=ns_a)
    assert upserted.upserted_count == 1
    upserted = index.upsert(vectors=[{"id": "lnsg-v2", "values": [0.3, 0.4]}], namespace=ns_b)
    assert upserted.upserted_count == 1

    # Poll until both namespaces appear
    poll_until(
        query_fn=lambda: index.list_namespaces_paginated(prefix=f"{ns}-", limit=100),
        check_fn=lambda r: len(r.namespaces) >= 2,
        timeout=120,
        description="both test namespaces appear via list_namespaces_paginated (grpc)",
    )

    # --- Exercise the generator ---
    pages = list(index.list_namespaces(prefix=f"{ns}-"))
    assert len(pages) >= 1, "list_namespaces() generator must yield at least one page"

    # Collect all namespace names across all yielded pages
    all_ns_names = [ns_item.name for page in pages for ns_item in page.namespaces]
    assert ns_a in all_ns_names, f"Expected {ns_a!r} in generator output; got {all_ns_names}"
    assert ns_b in all_ns_names, f"Expected {ns_b!r} in generator output; got {all_ns_names}"

    # Verify shape of every yielded page and its namespace descriptions
    for page in pages:
        assert isinstance(page, ListNamespacesResponse)
        assert len(page.namespaces) >= 1, "Each yielded page must contain at least one namespace"
        for ns_item in page.namespaces:
            assert isinstance(ns_item, NamespaceDescription)
            assert isinstance(ns_item.name, str) and ns_item.name.startswith(f"{ns}-")
            assert isinstance(ns_item.record_count, int) and ns_item.record_count >= 0


# ---------------------------------------------------------------------------
# list_paginated multi-page — REST sync
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_list_paginated_multi_page_rest(client: Pinecone, shared_index_dim2: str) -> None:
    # shared_index_dim2
    """list_paginated() with limit=2 returns a token when more pages exist; following the token
    reaches the next page; the final page has no token.

    Verifies claims:
    - unified-vec-0030: paginated list method returns a single page (caller must follow the token
      explicitly — it does not auto-paginate)
    - unified-vec-0056: list-paginated returns no pagination token when the current page is the last
    - unified-pag-0002: vector listing supports cursor-based pagination via a single-page method
    """
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    index = client.index(name=shared_index_dim2)

    # Upsert 4 vectors with a shared prefix
    upserted = index.upsert(
        vectors=[
            {"id": "mp-v1", "values": [0.1, 0.2]},
            {"id": "mp-v2", "values": [0.3, 0.4]},
            {"id": "mp-v3", "values": [0.5, 0.6]},
            {"id": "mp-v4", "values": [0.7, 0.8]},
        ],
        namespace=ns,
    )
    assert upserted.upserted_count == 4

    # Wait until all 4 vectors appear in list results
    poll_until(
        query_fn=lambda: index.list_paginated(prefix="mp-", limit=100, namespace=ns),
        check_fn=lambda r: len(r.vectors) >= 4,
        timeout=120,
        description="all 4 vectors listable after upsert",
    )

    # Traverse all pages manually using limit=2 (forces at least 2 pages)
    all_ids: list[str] = []
    token: str | None = None
    pages_seen = 0

    while True:
        page = index.list_paginated(prefix="mp-", limit=2, pagination_token=token, namespace=ns)
        assert isinstance(page, ListResponse)
        assert isinstance(page.vectors, list)
        assert len(page.vectors) <= 2  # limit is respected each page

        for item in page.vectors:
            assert isinstance(item, ListItem)
            assert isinstance(item.id, str)
            all_ids.append(item.id)

        pages_seen += 1

        # Extract next token (if any)
        if page.pagination is not None and page.pagination.next is not None:
            # More pages exist — token should be a non-empty string
            assert isinstance(page.pagination.next, str)
            assert len(page.pagination.next) > 0
            token = page.pagination.next
        else:
            # Final page — no token (unified-vec-0056)
            break

    # Collected all 4 IDs across pages
    assert {"mp-v1", "mp-v2", "mp-v3", "mp-v4"} <= set(all_ids)
    # Required at least 2 pages (limit=2, 4 vectors → ≥2 pages)
    assert pages_seen >= 2


# ---------------------------------------------------------------------------
# list-paginated — multi-page (gRPC)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_list_paginated_multi_page_grpc(client: Pinecone, shared_index_dim2: str) -> None:
    # shared_index_dim2
    """list_paginated() via GrpcIndex with limit=2 spans multiple pages; pagination tokens
    are returned when more pages exist; the final page has no token.

    Verifies claims (gRPC transport parity with REST):
    - unified-vec-0030: paginated list returns a single page; caller must follow the
      token explicitly — it does not auto-paginate
    - unified-vec-0056: list-paginated returns no pagination token when the current
      page is the last one
    - unified-pag-0002: vector listing supports cursor-based pagination via a
      single-page method

    This test is the gRPC counterpart of test_list_paginated_multi_page_rest.
    The single-page gRPC path is covered by test_list_paginated_returns_single_page_grpc;
    this test covers the multi-page pagination token flow on gRPC.
    """
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    index = client.index(name=shared_index_dim2, grpc=True)

    # Upsert 4 vectors with a shared prefix so limit=2 forces ≥ 2 pages
    upserted = index.upsert(
        vectors=[
            {"id": "gmp-v1", "values": [0.1, 0.2]},
            {"id": "gmp-v2", "values": [0.3, 0.4]},
            {"id": "gmp-v3", "values": [0.5, 0.6]},
            {"id": "gmp-v4", "values": [0.7, 0.8]},
        ],
        namespace=ns,
    )
    assert upserted.upserted_count == 4

    # Wait until all 4 vectors appear in list results (eventual consistency)
    poll_until(
        query_fn=lambda: index.list_paginated(prefix="gmp-", limit=100, namespace=ns),
        check_fn=lambda r: len(r.vectors) >= 4,
        timeout=120,
        description="all 4 gmp- vectors listable after upsert (gRPC)",
    )

    # Traverse all pages manually using limit=2 — forces at least 2 pages
    all_ids: list[str] = []
    token: str | None = None
    pages_seen = 0

    while True:
        page = index.list_paginated(prefix="gmp-", limit=2, pagination_token=token, namespace=ns)
        assert isinstance(page, ListResponse)
        assert isinstance(page.vectors, list)
        # Limit must be respected on every page
        assert len(page.vectors) <= 2, (
            f"Page {pages_seen} returned {len(page.vectors)} vectors (limit=2)"
        )

        for item in page.vectors:
            assert isinstance(item, ListItem)
            assert isinstance(item.id, str)
        all_ids.extend(item.id for item in page.vectors)

        pages_seen += 1

        if page.pagination is not None and page.pagination.next is not None:
            # More pages available — token must be a non-empty string
            assert isinstance(page.pagination.next, str)
            assert len(page.pagination.next) > 0
            token = page.pagination.next
        else:
            # Final page — pagination token must be absent (unified-vec-0056)
            break

    # All 4 vectors must be collected across the pages
    assert {"gmp-v1", "gmp-v2", "gmp-v3", "gmp-v4"} <= set(all_ids)
    # Must have traversed at least 2 pages (4 vectors ÷ limit=2)
    assert pages_seen >= 2


# ---------------------------------------------------------------------------
# delete-nonexistent-ids — REST sync + gRPC
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_delete_nonexistent_ids_returns_none_rest(client: Pinecone, shared_index_dim2: str) -> None:
    # shared_index_dim2
    """Delete with IDs that don't exist in the namespace returns None without error.

    Verifies unified-vec-0032: "Deleting vectors does not raise an error when
    the specified IDs do not exist."

    Two sub-cases:
    1. IDs never upserted — tested after namespace is established (fresh namespace
       raises 404; the claim applies within an existing namespace).
    2. ID that was upserted, deleted, then deleted again (idempotency).
    """
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    index = client.index(name=shared_index_dim2)

    # Establish the namespace by upserting a sentinel vector
    upserted = index.upsert(vectors=[{"id": "dn-v1", "values": [0.1, 0.9]}], namespace=ns)
    assert upserted.upserted_count == 1
    poll_until(
        query_fn=lambda: index.fetch(ids=["dn-v1"], namespace=ns),
        check_fn=lambda r: len(r.vectors) == 1,
        timeout=120,
        description="dn-v1 fetchable (namespace established)",
    )

    # Sub-case 1: delete IDs that were never upserted — should return None, not raise
    result = index.delete(ids=["never-existed-x", "never-existed-y"], namespace=ns)
    assert result is None

    # Sub-case 2: delete dn-v1 (exists), then delete it again (already gone)
    first = index.delete(ids=["dn-v1"], namespace=ns)
    assert first is None

    # Second delete — vector is gone; must still return None (idempotency)
    second = index.delete(ids=["dn-v1"], namespace=ns)
    assert second is None


@pytest.mark.integration
def test_delete_nonexistent_ids_returns_none_grpc(client: Pinecone, shared_index_dim2: str) -> None:
    # shared_index_dim2
    """Same as REST variant but over gRPC transport (unified-vec-0032)."""
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    index = client.index(name=shared_index_dim2, grpc=True)

    # Establish namespace by upserting a sentinel vector
    upserted = index.upsert(vectors=[{"id": "dn-g1", "values": [0.2, 0.8]}], namespace=ns)
    assert upserted.upserted_count == 1
    poll_until(
        query_fn=lambda: index.fetch(ids=["dn-g1"], namespace=ns),
        check_fn=lambda r: len(r.vectors) == 1,
        timeout=120,
        description="dn-g1 fetchable (namespace established for gRPC test)",
    )

    # Sub-case 1: delete IDs that were never upserted
    result = index.delete(ids=["never-existed-grpc-x", "never-existed-grpc-y"], namespace=ns)
    assert result is None

    # Sub-case 2: delete dn-g1 (exists), then delete again (already gone)
    first = index.delete(ids=["dn-g1"], namespace=ns)
    assert first is None

    second = index.delete(ids=["dn-g1"], namespace=ns)
    assert second is None


# ---------------------------------------------------------------------------
# context-manager — REST sync + gRPC
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_index_context_manager_rest(client: Pinecone, shared_index_dim2: str) -> None:
    # shared_index_dim2
    """REST Index supports the context manager protocol.

    Verifies unified-async-0002 (sync equivalent): the REST index client
    implements __enter__ / __exit__ for automatic resource cleanup.

    - 'with idx as rest_idx:' makes operations available inside the block
    - __enter__ returns the index object itself (not a copy)
    - describe_index_stats() works normally inside the context
    - After the with-block exits (__exit__ calls close()), calling close()
      again must not raise (idempotent resource release)

    Area tag: context-manager
    Transport: rest
    """
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    rest_idx = client.index(name=shared_index_dim2)
    with rest_idx as idx:
        # __enter__ must return the same object
        assert idx is rest_idx, "__enter__ must return self"
        assert isinstance(idx, Index)
        # Write first: describe_index_stats succeeds and reports zero on an
        # index that refuses vector writes, so a bare shape assertion would
        # hold against data that never landed (#322).
        upserted = idx.upsert(vectors=[{"id": "cm-v1", "values": [0.1, 0.2]}], namespace=ns)
        assert upserted.upserted_count == 1
        stats = poll_until(
            query_fn=lambda: idx.describe_index_stats(),
            check_fn=lambda s: ns in s.namespaces and s.namespaces[ns].vector_count == 1,
            timeout=120,
            description="context-manager namespace visible in stats",
        )
        assert isinstance(stats, DescribeIndexStatsResponse)
    # After __exit__ called close(), calling close() again must not raise
    rest_idx.close()


@pytest.mark.integration
def test_index_context_manager_grpc(client: Pinecone, shared_index_dim2: str) -> None:
    # shared_index_dim2
    """GrpcIndex supports the context manager protocol.

    Verifies unified-async-0002 (gRPC equivalent): GrpcIndex implements
    __enter__ / __exit__ so it can be used as a context manager.

    Area tag: context-manager
    Transport: grpc
    """
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    grpc_idx = client.index(name=shared_index_dim2, grpc=True)
    with grpc_idx as gidx:
        assert gidx is grpc_idx, "__enter__ must return self"
        assert isinstance(gidx, GrpcIndex)
        upserted = gidx.upsert(vectors=[{"id": "cmg-v1", "values": [0.1, 0.2]}], namespace=ns)
        assert upserted.upserted_count == 1
        stats = poll_until(
            query_fn=lambda: gidx.describe_index_stats(),
            check_fn=lambda s: ns in s.namespaces and s.namespaces[ns].vector_count == 1,
            timeout=120,
            description="context-manager namespace visible in stats (gRPC)",
        )
        assert isinstance(stats, DescribeIndexStatsResponse)
    grpc_idx.close()


# ---------------------------------------------------------------------------
# grpc-async-futures — gRPC
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_grpc_async_futures_upsert_query_fetch_grpc(
    client: Pinecone, shared_index_dim3: str
) -> None:
    # shared_index_dim3
    """GrpcIndex.*_async() methods return PineconeFuture objects that resolve to
    the correct response types against a live API.

    Verifies unified-grpc-0003: "Can execute gRPC operations asynchronously
    using futures."

    Test sequence:
      1. Submit two parallel upsert_async() futures and collect results via
         as_completed(); each future resolves to a UpsertResponse.
      2. Poll until vectors are queryable (eventual consistency).
      3. Submit a query_async() future and resolve it; verify QueryResponse
         has matches with correct structure.
      4. Submit a fetch_async() future and resolve it; verify FetchResponse
         contains the requested IDs.

    Area tag: grpc-async-futures
    Transport: grpc
    Claim: unified-grpc-0003
    """
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    grpc_idx = client.index(name=shared_index_dim3, grpc=True)

    # --- 1. upsert_async() — two parallel futures ---
    batch_a = [
        {"id": "fut-a1", "values": [0.1, 0.2, 0.3]},
        {"id": "fut-a2", "values": [0.4, 0.5, 0.6]},
    ]
    batch_b = [
        {"id": "fut-b1", "values": [0.7, 0.8, 0.9]},
        {"id": "fut-b2", "values": [0.9, 0.8, 0.7]},
    ]
    fut_a: PineconeFuture[UpsertResponse] = grpc_idx.upsert_async(vectors=batch_a, namespace=ns)
    fut_b: PineconeFuture[UpsertResponse] = grpc_idx.upsert_async(vectors=batch_b, namespace=ns)

    # Collect results via as_completed; use a generous timeout for live API
    upsert_futures = [fut_a, fut_b]
    for fut in as_completed(upsert_futures, timeout=60):
        result = fut.result(timeout=30.0)
        assert isinstance(result, UpsertResponse), f"Expected UpsertResponse, got {type(result)}"
        assert result.upserted_count == 2

    # --- 2. Poll until vectors are queryable (eventual consistency) ---
    poll_until(
        query_fn=lambda: grpc_idx.fetch(ids=["fut-a1", "fut-a2", "fut-b1", "fut-b2"], namespace=ns),
        check_fn=lambda r: len(r.vectors) >= 4,
        timeout=120,
        description="all 4 futures-upserted vectors indexed",
    )

    # --- 3. query_async() — submit future, resolve, verify QueryResponse ---
    qfut: PineconeFuture[QueryResponse] = grpc_idx.query_async(
        vector=[0.1, 0.2, 0.3],
        top_k=4,
        namespace=ns,
    )
    q_result = qfut.result(timeout=30.0)
    assert isinstance(q_result, QueryResponse), (
        f"Expected QueryResponse from query_async(), got {type(q_result)}"
    )
    assert isinstance(q_result.matches, list)
    assert len(q_result.matches) >= 1, "query_async() must return at least one match"
    # Verify match structure; scores may vary due to ANN approximation
    upserted_ids = {"fut-a1", "fut-a2", "fut-b1", "fut-b2"}
    for m in q_result.matches:
        assert isinstance(m.id, str)
        assert isinstance(m.score, float)
        assert m.id in upserted_ids, f"Unexpected match ID {m.id!r}"

    # --- 4. fetch_async() — submit future, resolve, verify FetchResponse ---
    ffut: PineconeFuture[FetchResponse] = grpc_idx.fetch_async(
        ids=["fut-a1", "fut-b1"],
        namespace=ns,
    )
    f_result = ffut.result(timeout=30.0)
    assert isinstance(f_result, FetchResponse), (
        f"Expected FetchResponse from fetch_async(), got {type(f_result)}"
    )
    assert isinstance(f_result.vectors, dict)
    assert "fut-a1" in f_result.vectors, "fut-a1 must be present in fetch_async() result"
    assert "fut-b1" in f_result.vectors, "fut-b1 must be present in fetch_async() result"


# ---------------------------------------------------------------------------
# fetch-nonexistent — REST sync + gRPC
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_fetch_nonexistent_ids_returns_empty_vectors_rest(
    client: Pinecone, shared_index_dim3: str
) -> None:
    # shared_index_dim3
    """Fetching IDs that were never upserted returns an empty vectors map, not an error.

    Verifies unified-vec-0053: "Fetching IDs that do not exist returns an empty
    vectors map rather than an error."

    Area tag: fetch-nonexistent
    Transport: rest
    """
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    idx = client.index(name=shared_index_dim3)

    # Upsert one real vector to establish the namespace
    upserted = idx.upsert(vectors=[{"id": "real-v1", "values": [0.1, 0.2, 0.3]}], namespace=ns)
    assert upserted.upserted_count == 1
    poll_until(
        query_fn=lambda: idx.fetch(ids=["real-v1"], namespace=ns),
        check_fn=lambda r: "real-v1" in r.vectors,
        timeout=120,
        description="real-v1 fetchable after upsert",
    )

    # Fetch IDs that were never upserted — should return empty dict, not raise
    result = idx.fetch(ids=["never-upserted-aaa", "never-upserted-bbb"], namespace=ns)

    assert isinstance(result, FetchResponse)
    assert isinstance(result.vectors, dict)
    # Non-existent IDs are simply absent — no error raised
    assert "never-upserted-aaa" not in result.vectors
    assert "never-upserted-bbb" not in result.vectors


@pytest.mark.integration
def test_fetch_nonexistent_ids_returns_empty_vectors_grpc(
    client: Pinecone, shared_index_dim3: str
) -> None:
    # shared_index_dim3
    """Fetching IDs that were never upserted returns an empty vectors map via gRPC.

    Verifies unified-vec-0053 on the gRPC transport.

    Area tag: fetch-nonexistent
    Transport: grpc
    """
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    idx = client.index(name=shared_index_dim3, grpc=True)

    # Upsert one real vector to establish the namespace
    upserted = idx.upsert(vectors=[{"id": "real-v1", "values": [0.1, 0.2, 0.3]}], namespace=ns)
    assert upserted.upserted_count == 1
    poll_until(
        query_fn=lambda: idx.fetch(ids=["real-v1"], namespace=ns),
        check_fn=lambda r: "real-v1" in r.vectors,
        timeout=120,
        description="real-v1 fetchable after upsert (gRPC)",
    )

    # Fetch IDs that were never upserted — should return empty dict, not raise
    result = idx.fetch(ids=["never-upserted-aaa", "never-upserted-bbb"], namespace=ns)

    assert isinstance(result, FetchResponse)
    assert isinstance(result.vectors, dict)
    assert "never-upserted-aaa" not in result.vectors
    assert "never-upserted-bbb" not in result.vectors


# ---------------------------------------------------------------------------
# grpc-delete-async — gRPC only
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_grpc_delete_async_future_resolves_to_none(
    client: Pinecone, shared_index_dim3: str
) -> None:
    # shared_index_dim3
    """GrpcIndex.delete_async() returns a PineconeFuture[None] that resolves to None.

    Verifies unified-grpc-0003: "Can execute gRPC operations asynchronously
    using futures." — specifically the delete_async() variant not covered by
    ET-078 (which covered upsert_async, query_async, fetch_async).

    Test sequence:
    1. Upsert 3 vectors via sync gRPC upsert().
    2. Poll until all 3 vectors are fetchable.
    3. delete_async(ids=[...]) for 2 of the 3 → future.result() must be None.
    4. Poll until the 2 deleted vectors are absent from fetch().
    5. Verify the 3rd vector is still present (partial delete correctness).
    6. delete_async(delete_all=True) → future.result() must be None.
    7. Poll until namespace is empty.

    Area tag: grpc-delete-async
    Transport: grpc
    Claim: unified-grpc-0003
    """
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    grpc_idx = client.index(name=shared_index_dim3, grpc=True)

    # --- 1. Upsert 3 vectors via sync gRPC upsert() ---
    vectors = [
        {"id": "da-v1", "values": [0.1, 0.2, 0.3]},
        {"id": "da-v2", "values": [0.4, 0.5, 0.6]},
        {"id": "da-v3", "values": [0.7, 0.8, 0.9]},
    ]
    upserted = grpc_idx.upsert(vectors=vectors, namespace=ns)
    assert upserted.upserted_count == len(vectors)

    # --- 2. Poll until all 3 vectors are fetchable ---
    poll_until(
        query_fn=lambda: grpc_idx.fetch(ids=["da-v1", "da-v2", "da-v3"], namespace=ns),
        check_fn=lambda r: len(r.vectors) == 3,
        timeout=120,
        description="all 3 vectors fetchable after upsert (gRPC delete_async test)",
    )

    # --- 3. delete_async(ids=[...]) for 2 of the 3 vectors ---
    del_future: PineconeFuture[None] = grpc_idx.delete_async(
        ids=["da-v1", "da-v2"],
        namespace=ns,
    )
    # future.result() must resolve to None (delete returns no content)
    del_result = del_future.result(timeout=30.0)
    assert del_result is None, f"delete_async().result() must be None, got {del_result!r}"

    # --- 4. Poll until the 2 deleted vectors are absent ---
    poll_until(
        query_fn=lambda: grpc_idx.fetch(ids=["da-v1", "da-v2"], namespace=ns),
        check_fn=lambda r: len(r.vectors) == 0,
        timeout=120,
        description="da-v1 and da-v2 absent after delete_async()",
    )

    # --- 5. Verify da-v3 is still present (partial delete) ---
    remaining = grpc_idx.fetch(ids=["da-v3"], namespace=ns)
    assert isinstance(remaining, FetchResponse)
    assert "da-v3" in remaining.vectors, "da-v3 should survive partial delete_async()"

    # --- 6. delete_async(delete_all=True) clears the namespace ---
    all_del_future: PineconeFuture[None] = grpc_idx.delete_async(
        delete_all=True,
        namespace=ns,
    )
    all_del_result = all_del_future.result(timeout=30.0)
    assert all_del_result is None, (
        f"delete_async(delete_all=True).result() must be None, got {all_del_result!r}"
    )

    # --- 7. Poll until namespace is empty ---
    poll_until(
        query_fn=lambda: grpc_idx.fetch(ids=["da-v3"], namespace=ns),
        check_fn=lambda r: len(r.vectors) == 0,
        timeout=120,
        description="namespace empty after delete_async(delete_all=True)",
    )


# ---------------------------------------------------------------------------
# namespace creation error paths — REST sync
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_create_namespace_error_paths_rest(client: Pinecone, shared_index_dim2: str) -> None:
    # shared_index_dim2
    """create_namespace() rejects invalid names client-side and raises ConflictError for duplicates.

    Verifies claims:
    - unified-ns-0010: Namespace creation is rejected when the name breaks the
      ASCII / no-NUL / 1-512 rule, or names the reserved ``__default__``.
    - unified-ns-0012: Creating a namespace that already exists raises a ConflictError (HTTP 409).

    No integration test covered these error paths; only unit tests (test-ns-0009) exercised them.

    A whitespace-only name is deliberately absent: the rule is ASCII/no-NUL/1-512,
    not "non-blank", so ``"   "`` is a legal name the server accepts. Asserting a
    rejection here would both fail and leave a stray namespace on the shared index.
    """
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    ns_name = f"{ns}-alpha"
    index = client.index(name=shared_index_dim2)

    # unified-ns-0010: each of these is refused client-side, so no API call is made
    with pytest.raises(PineconeValueError):
        index.create_namespace(name="")

    with pytest.raises(PineconeValueError):
        index.create_namespace(name="naïve")

    with pytest.raises(PineconeValueError):
        index.create_namespace(name="a" * 513)

    with pytest.raises(PineconeValueError):
        index.create_namespace(name="__default__")

    # Precondition for ns-0012: create the namespace successfully
    created = index.create_namespace(name=ns_name)
    assert isinstance(created, NamespaceDescription)
    assert created.name == ns_name

    try:
        # unified-ns-0012: creating the same namespace again raises ConflictError (409)
        with pytest.raises(ConflictError) as exc_info:
            index.create_namespace(name=ns_name)
        assert exc_info.value.status_code == 409
    finally:
        cleanup_resource(lambda: index.delete_namespace(name=ns_name), ns_name, "namespace")


# ---------------------------------------------------------------------------
# namespace creation with schema — REST sync
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_create_namespace_with_schema_rest(client: Pinecone, shared_index_dim2: str) -> None:
    # shared_index_dim2
    """create_namespace() accepts an optional schema dict and creates the namespace successfully.

    Verifies:
    - unified-ns-0001: Can create a named namespace, optionally providing a schema configuration.

    The existing test_namespace_crud_lifecycle_rest only calls create_namespace(name=...) without
    a schema parameter. This test exercises the schema= code path in the SDK: when schema is not
    None, the SDK adds body["schema"] = schema to the POST /namespaces request. No integration test
    previously exercised this path.
    """
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    ns_name = f"{ns}-alpha"
    index = client.index(name=shared_index_dim2)

    # Create a namespace with a schema specifying that "genre" is filterable
    created = index.create_namespace(
        name=ns_name,
        schema={"fields": {"genre": {"filterable": True}}},
    )

    try:
        # Verify response type and structure
        assert isinstance(created, NamespaceDescription)
        assert created.name == ns_name
        assert created.record_count == 0  # new namespace has no vectors

        # describe_namespace returns the namespace as accessible (schema was accepted)
        # Poll for eventual consistency — backend may not expose the namespace immediately after create
        described = poll_until(
            query_fn=lambda: index.describe_namespace(name=ns_name),
            check_fn=lambda r: isinstance(r, NamespaceDescription),
            timeout=60,
            description=f"namespace {ns_name} visible after create",
        )
        assert isinstance(described, NamespaceDescription)
        assert described.name == ns_name
        assert isinstance(described.record_count, int)
        assert described.record_count == 0
    finally:
        cleanup_resource(lambda: index.delete_namespace(name=ns_name), ns_name, "namespace")


# ---------------------------------------------------------------------------
# describe_namespace record_count after upsert — REST sync
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_describe_namespace_record_count_updates_after_upsert_rest(
    client: Pinecone, shared_index_dim2: str
) -> None:
    # shared_index_dim2
    """describe_namespace().record_count reflects the vector count after upsert (REST sync).

    Verifies claim unified-ns-0003: "Can describe a namespace by name, returning its
    record count and schema." The record_count must accurately track the number of
    vectors in the namespace — not just be 0 at creation time.

    Operation sequence tested:
    1. Create namespace → verify record_count == 0
    2. Upsert 4 vectors into the namespace
    3. Poll describe_namespace() until record_count > 0 (eventual consistency)
    4. Verify record_count == 4

    No existing test verifies the state transition from 0 to N for record_count.
    """
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    ns_name = f"{ns}-alpha"
    index = client.index(name=shared_index_dim2)

    # 1. Create namespace explicitly — record_count starts at 0
    created = index.create_namespace(name=ns_name)
    assert isinstance(created, NamespaceDescription)
    assert created.name == ns_name
    assert created.record_count == 0, (
        f"Freshly created namespace should have record_count == 0, got {created.record_count}"
    )

    try:
        # 2. Upsert 4 vectors into the namespace
        upsert_resp = index.upsert(
            vectors=[
                {"id": "rcnt-v1", "values": [0.1, 0.2]},
                {"id": "rcnt-v2", "values": [0.3, 0.4]},
                {"id": "rcnt-v3", "values": [0.5, 0.6]},
                {"id": "rcnt-v4", "values": [0.7, 0.8]},
            ],
            namespace=ns_name,
        )
        assert isinstance(upsert_resp, UpsertResponse)
        assert upsert_resp.upserted_count == 4

        # 3. Poll until describe_namespace reports at least 4 vectors (eventual consistency)
        final = poll_until(
            query_fn=lambda: index.describe_namespace(name=ns_name),
            check_fn=lambda r: r.record_count >= 4,
            timeout=120,
            description="describe_namespace record_count reaches 4 after upsert",
        )
        assert isinstance(final, NamespaceDescription)

        # 4. Verify the record_count equals the number of vectors upserted
        assert final.record_count == 4, (
            f"Expected record_count == 4 after upserting 4 vectors, got {final.record_count}"
        )
        assert final.name == ns_name
    finally:
        cleanup_resource(lambda: index.delete_namespace(name=ns_name), ns_name, "namespace")


# ---------------------------------------------------------------------------
# list_namespaces multi-page pagination — REST sync
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_list_namespaces_multi_page_pagination_rest(
    client: Pinecone, shared_index_dim2: str
) -> None:
    # shared_index_dim2
    """list_namespaces_paginated() with limit=1 forces multi-page results; intermediate
    pages carry a non-None pagination token; the final page has no token.

    Verifies claims:
    - unified-ns-0008: Namespace list response omits the pagination token on the final page.
    - unified-ns-0005: Can list all namespaces with optional prefix filtering and pagination.
    """
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    prefix = f"{ns}-"
    ns_names = [f"{ns}-a", f"{ns}-b", f"{ns}-c"]
    index = client.index(name=shared_index_dim2)

    # Upsert one vector per namespace to create them implicitly
    for i, ns_label in enumerate(ns_names):
        upserted = index.upsert(
            vectors=[{"id": f"mpns-v{i}", "values": [0.1 * (i + 1), 0.2 * (i + 1)]}],
            namespace=ns_label,
        )
        assert upserted.upserted_count == 1

    # Wait until all 3 namespaces appear (eventual consistency)
    poll_until(
        query_fn=lambda: index.list_namespaces_paginated(prefix=prefix, limit=100),
        check_fn=lambda r: len(r.namespaces) >= 3,
        timeout=120,
        description="all 3 test namespaces visible via list_namespaces_paginated",
    )

    # Traverse pages manually with limit=1 (forces ≥3 pages for 3 namespaces)
    collected_names: list[str] = []
    token: str | None = None
    pages_seen = 0

    while True:
        page = index.list_namespaces_paginated(prefix=prefix, limit=1, pagination_token=token)
        assert isinstance(page, ListNamespacesResponse)
        assert len(page.namespaces) <= 1  # limit is respected per page

        for ns_item in page.namespaces:
            assert isinstance(ns_item, NamespaceDescription)
            assert isinstance(ns_item.name, str)
            assert isinstance(ns_item.record_count, int)
            collected_names.append(ns_item.name)

        pages_seen += 1

        if page.pagination is not None and page.pagination.next is not None:
            # Intermediate page — token must be a non-empty string
            assert isinstance(page.pagination.next, str)
            assert len(page.pagination.next) > 0
            token = page.pagination.next
        else:
            # Final page — no token (unified-ns-0008)
            break

    # All 3 namespaces found across pages
    for ns_label in ns_names:
        assert ns_label in collected_names, (
            f"Expected {ns_label!r} in paginated results; got {collected_names}"
        )
    # At least 3 pages (one per namespace with limit=1)
    assert pages_seen >= 3, f"Expected >=3 pages with limit=1 and 3 namespaces; saw {pages_seen}"


# ---------------------------------------------------------------------------
# Legacy async_req=True opt-in (sync REST)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_upsert_async_req_rest(client_pool: Pinecone, shared_index_dim2: str) -> None:
    # shared_index_dim2
    """async_req=True on upsert returns ApplyResult; .get() yields UpsertResponse."""
    from multiprocessing.pool import ApplyResult

    ns = f"ns-{uuid.uuid4().hex[:8]}"
    with client_pool.index(name=shared_index_dim2) as index:
        result: Any = index.upsert(  # type: ignore[call-arg]
            vectors=[
                {"id": "asy-v1", "values": [0.1, 0.2]},
                {"id": "asy-v2", "values": [0.3, 0.4]},
            ],
            namespace=ns,
            async_req=True,
        )
        assert isinstance(result, ApplyResult)
        resolved = result.get(timeout=60)
        assert isinstance(resolved, UpsertResponse)
        assert resolved.upserted_count == 2


@pytest.mark.integration
def test_query_async_req_rest(client_pool: Pinecone, shared_index_dim2: str) -> None:
    # shared_index_dim2
    """async_req=True on query returns ApplyResult; .get() yields QueryResponse."""
    from multiprocessing.pool import ApplyResult

    ns = f"ns-{uuid.uuid4().hex[:8]}"
    with client_pool.index(name=shared_index_dim2) as index:
        upserted = index.upsert(
            vectors=[
                {"id": "qry-v1", "values": [0.1, 0.2]},
                {"id": "qry-v2", "values": [0.3, 0.4]},
            ],
            namespace=ns,
        )
        assert upserted.upserted_count == 2
        poll_until(
            query_fn=lambda: index.fetch(ids=["qry-v1", "qry-v2"], namespace=ns),
            check_fn=lambda r: len(r.vectors) == 2,
            timeout=120,
            description="2 vectors fetchable before async query",
        )
        result: Any = index.query(top_k=2, vector=[0.1, 0.2], namespace=ns, async_req=True)  # type: ignore[call-arg]
        assert isinstance(result, ApplyResult)
        resolved = result.get(timeout=60)
        assert isinstance(resolved, QueryResponse)
        assert len(resolved.matches) == 2


@pytest.mark.integration
def test_describe_index_stats_async_req_rest(client_pool: Pinecone, shared_index_dim2: str) -> None:
    # shared_index_dim2
    """async_req=True on describe_index_stats returns ApplyResult."""
    from multiprocessing.pool import ApplyResult

    ns = f"ns-{uuid.uuid4().hex[:8]}"
    with client_pool.index(name=shared_index_dim2) as index:
        # Write first: stats succeed and report zero on an index that refuses
        # vector writes, so the shape assertions below would hold against data
        # that never landed (#322).
        upserted = index.upsert(vectors=[{"id": "ar-v1", "values": [0.1, 0.2]}], namespace=ns)
        assert upserted.upserted_count == 1
        poll_until(
            query_fn=lambda: index.describe_index_stats(),
            check_fn=lambda s: ns in s.namespaces and s.namespaces[ns].vector_count == 1,
            timeout=120,
            description="async_req namespace visible in stats",
        )

        result: Any = index.describe_index_stats(async_req=True)  # type: ignore[call-arg]
        assert isinstance(result, ApplyResult)
        resolved = result.get(timeout=60)
        assert isinstance(resolved, DescribeIndexStatsResponse)
        assert resolved.dimension == 2
        assert resolved.namespaces[ns].vector_count == 1


@pytest.mark.integration
def test_list_paginated_async_req_rest(client_pool: Pinecone, shared_index_dim2: str) -> None:
    # shared_index_dim2
    """async_req=True on list_paginated returns ApplyResult."""
    from multiprocessing.pool import ApplyResult

    ns = f"ns-{uuid.uuid4().hex[:8]}"
    with client_pool.index(name=shared_index_dim2) as index:
        upserted = index.upsert(
            vectors=[
                {"id": "lst-v1", "values": [0.1, 0.2]},
                {"id": "lst-v2", "values": [0.3, 0.4]},
                {"id": "lst-v3", "values": [0.5, 0.6]},
            ],
            namespace=ns,
        )
        assert upserted.upserted_count == 3
        poll_until(
            query_fn=lambda: index.fetch(ids=["lst-v1", "lst-v2", "lst-v3"], namespace=ns),
            check_fn=lambda r: len(r.vectors) == 3,
            timeout=120,
            description="3 vectors fetchable before async list",
        )
        result: Any = index.list_paginated(namespace=ns, async_req=True)  # type: ignore[call-arg]
        assert isinstance(result, ApplyResult)
        resolved = result.get(timeout=60)
        assert isinstance(resolved, ListResponse)
        assert len(resolved.vectors) == 3


@pytest.mark.integration
def test_async_req_concurrent_fanout_rest(client_pool: Pinecone, shared_index_dim2: str) -> None:
    # shared_index_dim2
    """Multiple in-flight async_req=True calls all resolve successfully."""
    from multiprocessing.pool import ApplyResult

    ns = f"ns-{uuid.uuid4().hex[:8]}"
    with client_pool.index(name=shared_index_dim2) as index:
        upserted = index.upsert(
            vectors=[
                {"id": f"con-v{i}", "values": [(i + 1) * 0.1, (i + 1) * 0.2]} for i in range(8)
            ],
            namespace=ns,
        )
        assert upserted.upserted_count == 8
        poll_until(
            query_fn=lambda: index.fetch(ids=[f"con-v{i}" for i in range(8)], namespace=ns),
            check_fn=lambda r: len(r.vectors) == 8,
            timeout=120,
            description="8 vectors fetchable before concurrent fan-out",
        )

        # Fan out 4 simultaneous queries. With pool_threads=4, all
        # four should be in flight at once.
        results: list[Any] = [
            index.query(
                top_k=2, vector=[(i + 1) * 0.1, (i + 1) * 0.2], namespace=ns, async_req=True
            )  # type: ignore[call-arg]
            for i in range(4)
        ]
        for r in results:
            assert isinstance(r, ApplyResult)
        resolved: list[Any] = [r.get(timeout=60) for r in results]
        assert all(isinstance(q, QueryResponse) for q in resolved)
        assert all(len(q.matches) > 0 for q in resolved)


# ---------------------------------------------------------------------------
# search with dense vector — wire format (mock HTTP)
# ---------------------------------------------------------------------------

_SEARCH_HOST = "dense-vec-test.svc.pinecone.io"
_SEARCH_URL = f"https://{_SEARCH_HOST}/records/namespaces/vec-ns/search"
_SEARCH_MOCK_RESPONSE: dict[str, object] = {
    "result": {
        "hits": [
            {"_id": "r1", "_score": 0.91},
            {"_id": "r2", "_score": 0.75},
        ]
    },
    "usage": {"read_units": 3},
}


@respx.mock
def test_search_with_dense_vector() -> None:
    """search(vector=...) sends vector as {"values": [...]} object, not a bare array.

    Verifies SYNC-0098: the wire payload for a dense-vector query must be
    {"query": {"vector": {"values": [...]}, ...}} so serde on the backend can
    deserialize RecordsVectorQuery correctly.
    """
    route = respx.post(_SEARCH_URL).mock(
        return_value=httpx.Response(200, json=_SEARCH_MOCK_RESPONSE),
    )
    idx = Index(host=_SEARCH_HOST, api_key="test-key")
    response = idx.search(namespace="vec-ns", top_k=3, vector=[0.1, 0.2, 0.3])

    body = orjson.loads(route.calls.last.request.content)
    assert body["query"]["vector"] == {"values": [0.1, 0.2, 0.3]}, (
        f"Expected vector as object with 'values' key; got {body['query']['vector']!r}"
    )
    assert isinstance(response.result.hits, list)
    assert response.usage.read_units >= 0


# ---------------------------------------------------------------------------
# search with sparse/hybrid vector — wire format (mock HTTP)
# ---------------------------------------------------------------------------


@respx.mock
def test_search_with_sparse_vector() -> None:
    """search(vector=dict) passes the dict through as-is for sparse/hybrid queries.

    Verifies AGT-0046: a Mapping passed as vector is forwarded verbatim so that
    sparse_indices and sparse_values reach the backend without wrapping.
    """
    route = respx.post(_SEARCH_URL).mock(
        return_value=httpx.Response(200, json=_SEARCH_MOCK_RESPONSE),
    )
    idx = Index(host=_SEARCH_HOST, api_key="test-key")
    response = idx.search(
        namespace="vec-ns",
        top_k=3,
        vector={"sparse_indices": [10, 20], "sparse_values": [0.5, 0.3]},
    )

    body = orjson.loads(route.calls.last.request.content)
    assert body["query"]["vector"] == {"sparse_indices": [10, 20], "sparse_values": [0.5, 0.3]}
    assert isinstance(response.result.hits, list)
    assert response.usage.read_units >= 0


@respx.mock
def test_search_with_hybrid_vector() -> None:
    """search(vector=dict) with both dense values and sparse indices is passed through."""
    route = respx.post(_SEARCH_URL).mock(
        return_value=httpx.Response(200, json=_SEARCH_MOCK_RESPONSE),
    )
    idx = Index(host=_SEARCH_HOST, api_key="test-key")
    response = idx.search(
        namespace="vec-ns",
        top_k=3,
        vector={
            "values": [0.1, 0.2, 0.3],
            "sparse_indices": [10, 20],
            "sparse_values": [0.5, 0.3],
        },
    )

    body = orjson.loads(route.calls.last.request.content)
    assert body["query"]["vector"] == {
        "values": [0.1, 0.2, 0.3],
        "sparse_indices": [10, 20],
        "sparse_values": [0.5, 0.3],
    }
    assert isinstance(response.result.hits, list)


# ---------------------------------------------------------------------------
# async search with sparse/hybrid vector — wire format (mock HTTP)
# ---------------------------------------------------------------------------


@respx.mock
@pytest.mark.anyio
async def test_async_search_with_sparse_vector() -> None:
    """AsyncIndex.search(vector=dict) passes the dict through as-is for sparse queries."""
    from pinecone.async_client.async_index import AsyncIndex

    route = respx.post(_SEARCH_URL).mock(
        return_value=httpx.Response(200, json=_SEARCH_MOCK_RESPONSE),
    )
    idx = AsyncIndex(host=_SEARCH_HOST, api_key="test-key")
    response = await idx.search(
        namespace="vec-ns",
        top_k=3,
        vector={"sparse_indices": [10, 20], "sparse_values": [0.5, 0.3]},
    )

    body = orjson.loads(route.calls.last.request.content)
    assert body["query"]["vector"] == {"sparse_indices": [10, 20], "sparse_values": [0.5, 0.3]}
    assert isinstance(response.result.hits, list)
    assert response.usage.read_units >= 0


# ---------------------------------------------------------------------------
# upsert_records — client-side ID validation
# ---------------------------------------------------------------------------


def test_upsert_records_id_must_be_string() -> None:
    """upsert_records raises ValidationError when '_id' is not a string."""
    idx = Index(host="my-index.svc.pinecone.io", api_key="test-key")
    with pytest.raises(ValidationError, match="'_id' must be a string"):
        idx.upsert_records(namespace="ns", records=[{"_id": 123, "text": "hello"}])


@respx.mock
def test_upsert_records_both_id_fields_strips_id() -> None:
    """When both '_id' and 'id' are present, 'id' is dropped and '_id' is used."""
    import json as _json

    upsert_url = "https://my-index.svc.pinecone.io/records/namespaces/ns/upsert"
    route = respx.post(upsert_url).mock(return_value=httpx.Response(201))
    idx = Index(host="my-index.svc.pinecone.io", api_key="test-key")
    idx.upsert_records(namespace="ns", records=[{"_id": "wins", "id": "loses", "text": "hello"}])
    body = route.calls.last.request.content.decode("utf-8")
    parsed = _json.loads(body.strip())
    assert parsed["_id"] == "wins"
    assert "id" not in parsed


# ---------------------------------------------------------------------------
# start_import — error_mode default behavior (mock HTTP)
# ---------------------------------------------------------------------------

_IMPORTS_HOST = "test-index-abc1234.svc.us-east1-gcp.pinecone.io"
_IMPORTS_URL = f"https://{_IMPORTS_HOST}/bulk/imports"


# ---------------------------------------------------------------------------
# fetch_by_metadata — client-side limit validation
# ---------------------------------------------------------------------------


def test_fetch_by_metadata_limit_validation() -> None:
    """fetch_by_metadata raises when limit=0 (minimum is 1 per OAS spec)."""
    idx = Index(host="my-index.svc.pinecone.io", api_key="test-key")
    with pytest.raises((PineconeValueError, ValidationError), match="limit"):
        idx.fetch_by_metadata(filter={"a": "b"}, limit=0)


def test_fetch_by_metadata_limit_validation_negative() -> None:
    """fetch_by_metadata raises when limit is negative."""
    idx = Index(host="my-index.svc.pinecone.io", api_key="test-key")
    with pytest.raises((PineconeValueError, ValidationError), match="limit"):
        idx.fetch_by_metadata(filter={"a": "b"}, limit=-5)


# ---------------------------------------------------------------------------
# start_import — error_mode default behavior (mock HTTP)
# ---------------------------------------------------------------------------


@respx.mock
def test_start_import_error_mode_default() -> None:
    """Calling start_import without error_mode omits errorMode from request body."""
    from pinecone.models.imports.model import StartImportResponse

    route = respx.post(_IMPORTS_URL).mock(
        return_value=httpx.Response(200, json={"id": "import-default"}),
    )
    idx = Index(host=_IMPORTS_HOST, api_key="test-key")
    result = idx.start_import(uri="s3://my-bucket/vectors/")

    assert isinstance(result, StartImportResponse)
    assert result.id == "import-default"

    body = orjson.loads(route.calls.last.request.content)
    assert body["uri"] == "s3://my-bucket/vectors/"
    assert "errorMode" not in body


@respx.mock
def test_start_import_error_mode_abort_in_body() -> None:
    """Calling start_import(error_mode='abort') sends errorMode.onError='abort'."""
    route = respx.post(_IMPORTS_URL).mock(
        return_value=httpx.Response(200, json={"id": "import-abort"}),
    )
    idx = Index(host=_IMPORTS_HOST, api_key="test-key")
    idx.start_import(uri="s3://my-bucket/vectors/", error_mode="abort")

    body = orjson.loads(route.calls.last.request.content)
    assert body["errorMode"] == {"onError": "abort"}


@respx.mock
def test_start_import_error_mode_continue_in_body() -> None:
    """Calling start_import(error_mode='continue') sends errorMode.onError='continue'."""
    route = respx.post(_IMPORTS_URL).mock(
        return_value=httpx.Response(200, json={"id": "import-continue"}),
    )
    idx = Index(host=_IMPORTS_HOST, api_key="test-key")
    idx.start_import(uri="s3://my-bucket/vectors/", error_mode="continue")

    body = orjson.loads(route.calls.last.request.content)
    assert body["errorMode"] == {"onError": "continue"}
