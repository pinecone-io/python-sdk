"""Integration tests for data-plane vector operations (async REST).

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

import httpx
import orjson
import pytest
import respx

from pinecone import AsyncIndex, AsyncPinecone, Pinecone
from pinecone.errors import ApiError, ConflictError, PineconeValueError
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
    async_cleanup_resource,
    async_poll_until,
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
# delete-vectors — REST async
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_delete_vectors_rest_async(
    async_client: AsyncPinecone, shared_index_dim2: str
) -> None:
    # shared_index_dim2
    """Delete vectors by IDs via AsyncIndex (REST) and verify they are gone."""
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    idx = await async_client.index(name=shared_index_dim2)

    upserted = await idx.upsert(
        vectors=[
            {"id": "del-v1", "values": [0.1, 0.2]},
            {"id": "del-v2", "values": [0.3, 0.4]},
            {"id": "del-v3", "values": [0.5, 0.6]},
        ],
        namespace=ns,
    )
    assert upserted.upserted_count == 3

    # Wait for all 3 vectors to be fetchable (eventual consistency)
    await async_poll_until(
        query_fn=lambda: idx.fetch(ids=["del-v1", "del-v2", "del-v3"], namespace=ns),
        check_fn=lambda r: len(r.vectors) == 3,
        timeout=120,
        description="all 3 vectors fetchable before delete",
    )

    # Delete just v1 and v2 by IDs — delete() returns None on success
    await idx.delete(ids=["del-v1", "del-v2"], namespace=ns)

    # Wait until deleted vectors are gone (eventual consistency)
    await async_poll_until(
        query_fn=lambda: idx.fetch(ids=["del-v1", "del-v2"], namespace=ns),
        check_fn=lambda r: len(r.vectors) == 0,
        timeout=120,
        description="deleted vectors gone after delete",
    )

    # Verify v3 is still present
    remaining = await idx.fetch(ids=["del-v3"], namespace=ns)
    assert isinstance(remaining, FetchResponse)
    assert "del-v3" in remaining.vectors
    assert "del-v1" not in remaining.vectors
    assert "del-v2" not in remaining.vectors


# ---------------------------------------------------------------------------
# upsert — REST async
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_upsert_vectors_rest_async(
    async_client: AsyncPinecone, shared_index_dim2: str
) -> None:
    # shared_index_dim2
    """Upsert vectors via AsyncIndex (REST) and verify upserted_count."""
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    idx = await async_client.index(name=shared_index_dim2)

    result = await idx.upsert(
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
# query — REST async
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_query_by_vector_rest_async(
    async_client: AsyncPinecone, shared_index_dim2: str
) -> None:
    # shared_index_dim2
    """Query by vector via AsyncIndex (REST) and verify matches structure."""
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    idx = await async_client.index(name=shared_index_dim2)

    upserted = await idx.upsert(
        vectors=[
            {"id": "v1", "values": [0.1, 0.2]},
            {"id": "v2", "values": [0.3, 0.4]},
            {"id": "v3", "values": [0.5, 0.6]},
        ],
        namespace=ns,
    )
    assert upserted.upserted_count == 3

    # Wait for all 3 vectors to be queryable (eventual consistency)
    await async_poll_until(
        query_fn=lambda: idx.query(vector=[0.1, 0.2], top_k=3, namespace=ns),
        check_fn=lambda r: len(r.matches) == 3,
        timeout=120,
        description="all 3 vectors queryable after upsert",
    )

    result = await idx.query(vector=[0.1, 0.2], top_k=2, namespace=ns)

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
# fetch — REST async
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_fetch_vectors_rest_async(
    async_client: AsyncPinecone, shared_index_dim2: str
) -> None:
    # shared_index_dim2
    """Fetch vectors by ID via AsyncIndex (REST) and verify returned vector data."""
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    idx = await async_client.index(name=shared_index_dim2)

    upserted = await idx.upsert(
        vectors=[
            {"id": "v1", "values": [0.1, 0.2]},
            {"id": "v2", "values": [0.3, 0.4]},
            {"id": "v3", "values": [0.5, 0.6]},
        ],
        namespace=ns,
    )
    assert upserted.upserted_count == 3

    # Wait for vectors to be fetchable (eventual consistency)
    await async_poll_until(
        query_fn=lambda: idx.fetch(ids=["v1", "v2", "v3"], namespace=ns),
        check_fn=lambda r: len(r.vectors) == 3,
        timeout=120,
        description="all 3 vectors fetchable after upsert",
    )

    result = await idx.fetch(ids=["v1", "v2"], namespace=ns)

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
# list-vectors — REST async
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_list_vectors_rest_async(async_client: AsyncPinecone, shared_index_dim2: str) -> None:
    # shared_index_dim2
    """List vectors via AsyncIndex (REST) and verify pagination structure and IDs."""
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    idx = await async_client.index(name=shared_index_dim2)

    upserted = await idx.upsert(
        vectors=[
            {"id": "lst-v1", "values": [0.1, 0.2]},
            {"id": "lst-v2", "values": [0.3, 0.4]},
            {"id": "lst-v3", "values": [0.5, 0.6]},
        ],
        namespace=ns,
    )
    assert upserted.upserted_count == 3

    # Wait for all 3 vectors to appear in list results (eventual consistency)
    async def _collect_ids() -> list[str]:
        return [
            item.id
            async for page in idx.list(prefix="lst-", namespace=ns)
            for item in page.vectors
            if item.id is not None
        ]

    await async_poll_until(
        query_fn=_collect_ids,
        check_fn=lambda ids: len(ids) >= 3,
        timeout=120,
        description="all 3 vectors listable after upsert",
    )

    # Collect all pages and verify structure
    pages: list[ListResponse] = [page async for page in idx.list(prefix="lst-", namespace=ns)]

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
# update-vectors — REST async
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_update_vectors_rest_async(
    async_client: AsyncPinecone, shared_index_dim2: str
) -> None:
    # shared_index_dim2
    """Update a vector's values via AsyncIndex (REST) and verify the change is reflected."""
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    idx = await async_client.index(name=shared_index_dim2)

    upserted = await idx.upsert(
        vectors=[
            {"id": "upd-v1", "values": [0.1, 0.2]},
            {"id": "upd-v2", "values": [0.3, 0.4]},
        ],
        namespace=ns,
    )
    assert upserted.upserted_count == 2

    # Wait for vectors to be fetchable (eventual consistency)
    await async_poll_until(
        query_fn=lambda: idx.fetch(ids=["upd-v1", "upd-v2"], namespace=ns),
        check_fn=lambda r: len(r.vectors) == 2,
        timeout=120,
        description="vectors fetchable before update",
    )

    # Update upd-v1 with new values
    result = await idx.update(id="upd-v1", values=[0.9, 0.8], namespace=ns)

    assert isinstance(result, UpdateResponse)
    # The update API returns {} on success; matched_records may be None
    assert result.matched_records is None or isinstance(result.matched_records, int)

    # Poll until the updated values are reflected
    await async_poll_until(
        query_fn=lambda: idx.fetch(ids=["upd-v1"], namespace=ns),
        check_fn=lambda r: (
            "upd-v1" in r.vectors
            and len(r.vectors["upd-v1"].values) == 2
            and abs(r.vectors["upd-v1"].values[0] - 0.9) < 1e-4
        ),
        timeout=120,
        description="updated values reflected in fetch",
    )

    # Verify upd-v2 was not modified
    check = await idx.fetch(ids=["upd-v2"], namespace=ns)
    assert abs(check.vectors["upd-v2"].values[0] - 0.3) < 1e-4


# ---------------------------------------------------------------------------
# describe-stats — REST async
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_describe_index_stats_rest_async(
    async_client: AsyncPinecone, shared_index_dim3: str
) -> None:
    # shared_index_dim3
    """Call describe_index_stats() via AsyncIndex (REST) and verify response structure."""
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    idx = await async_client.index(name=shared_index_dim3)

    # Upsert a few vectors so stats are non-trivial
    upserted = await idx.upsert(
        vectors=[
            {"id": "st-v1", "values": [0.1, 0.2, 0.3]},
            {"id": "st-v2", "values": [0.4, 0.5, 0.6]},
        ],
        namespace=ns,
    )
    assert upserted.upserted_count == 2

    # Wait until our namespace appears in stats (eventual consistency)
    await async_poll_until(
        query_fn=lambda: idx.describe_index_stats(),
        check_fn=lambda r: ns in r.namespaces and r.namespaces[ns].vector_count >= 1,
        timeout=120,
        description="test namespace counted in stats after upsert",
    )

    stats = await idx.describe_index_stats()

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
# namespaces — REST async
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_namespaces_rest_async(async_client: AsyncPinecone, shared_index_dim2: str) -> None:
    # shared_index_dim2
    """Upsert to named namespace via AsyncIndex (REST) and query within it."""
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    named_ns = f"{ns}-alpha"
    def_ns = f"{ns}-def"
    idx = await async_client.index(name=shared_index_dim2)

    # Upsert vectors into named namespace
    ns_result = await idx.upsert(
        vectors=[
            {"id": "ns-v1", "values": [0.1, 0.2]},
            {"id": "ns-v2", "values": [0.3, 0.4]},
        ],
        namespace=named_ns,
    )
    assert isinstance(ns_result, UpsertResponse)
    assert ns_result.upserted_count == 2

    # Upsert different vectors into the per-test default namespace
    def_result = await idx.upsert(
        vectors=[
            {"id": "def-v1", "values": [0.9, 0.8]},
        ],
        namespace=def_ns,
    )
    assert isinstance(def_result, UpsertResponse)
    assert def_result.upserted_count == 1

    # Wait until named namespace vectors are queryable
    await async_poll_until(
        query_fn=lambda: idx.query(vector=[0.1, 0.2], top_k=10, namespace=named_ns),
        check_fn=lambda r: len(r.matches) >= 2,
        timeout=120,
        description="named namespace vectors queryable",
    )

    # Query in the named namespace
    ns_query = await idx.query(vector=[0.1, 0.2], top_k=10, namespace=named_ns)
    assert isinstance(ns_query, QueryResponse)
    assert ns_query.namespace == named_ns
    ns_ids = {m.id for m in ns_query.matches}
    assert "ns-v1" in ns_ids
    assert "ns-v2" in ns_ids
    # Default namespace vectors must NOT appear in named-namespace query
    assert "def-v1" not in ns_ids

    # Verify stats shows the named namespace with exactly 2 vectors
    await async_poll_until(
        query_fn=lambda: idx.describe_index_stats(),
        check_fn=lambda s: named_ns in s.namespaces and s.namespaces[named_ns].vector_count == 2,
        timeout=120,
        description="stats reflect 2 vectors in named namespace",
    )
    stats = await idx.describe_index_stats()
    assert isinstance(stats.namespaces, dict)
    assert named_ns in stats.namespaces
    assert stats.namespaces[named_ns].vector_count == 2


# ---------------------------------------------------------------------------
# list_paginated — REST async
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_list_paginated_returns_single_page_rest_async(
    async_client: AsyncPinecone,
    shared_index_dim2: str,
) -> None:
    # shared_index_dim2
    """list_paginated() via AsyncIndex returns one page; limit is respected; no token on last page."""
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    idx = await async_client.index(name=shared_index_dim2)

    # Upsert 4 vectors with a shared prefix
    upserted = await idx.upsert(
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
    await async_poll_until(
        query_fn=lambda: idx.list_paginated(prefix="pg-", limit=100, namespace=ns),
        check_fn=lambda r: len(r.vectors) >= 4,
        timeout=120,
        description="all 4 vectors listable after upsert",
    )

    # 1. list_paginated() returns a ListResponse (not an async generator)
    page = await idx.list_paginated(prefix="pg-", limit=100, namespace=ns)
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

    # 5. No pagination token on the final page (unified-vec-0056)
    assert page.pagination is None or page.pagination.next is None

    # 6. limit=2 returns at most 2 items
    limited_page = await idx.list_paginated(prefix="pg-", limit=2, namespace=ns)
    assert isinstance(limited_page, ListResponse)
    assert len(limited_page.vectors) <= 2


# ---------------------------------------------------------------------------
# describe-stats with filter — serverless rejects (REST async)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_describe_index_stats_filter_unsupported_on_serverless_rest_async(
    async_client: AsyncPinecone,
    shared_index_dim3: str,
) -> None:
    # shared_index_dim3
    """Verify describe_index_stats(filter=...) raises ApiError(400) on a serverless index (async)."""
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    idx = await async_client.index(name=shared_index_dim3)

    # Upsert one vector so the index has some content
    upserted = await idx.upsert(vectors=[{"id": "fa-v1", "values": [0.1, 0.2, 0.3]}], namespace=ns)
    assert upserted.upserted_count == 1

    # The filter parameter is not supported on serverless/starter indexes —
    # the API returns 400 and the SDK should surface it as ApiError.
    with pytest.raises(ApiError) as exc_info:
        await idx.describe_index_stats(filter={"tag": {"$eq": "a"}})

    assert exc_info.value.status_code == 400


# ---------------------------------------------------------------------------
# namespace CRUD — REST async
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_namespace_crud_lifecycle_rest_async(
    async_client: AsyncPinecone, shared_index_dim2: str
) -> None:
    # shared_index_dim2
    """Async create_namespace / describe_namespace / list_namespaces_paginated / delete_namespace.

    Verifies claims:
    - unified-ns-0001: Can create a named namespace.
    - unified-ns-0002: Creation returns name and record_count == 0.
    - unified-ns-0003: Can describe a namespace by name.
    - unified-ns-0004: Can delete a namespace by name.
    - unified-ns-0005: Can list all namespaces with optional prefix filtering.
    - unified-ns-0008: Namespace list response omits pagination token on the final page.
    """
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    ns_name = f"{ns}-crud-ns-beta"
    idx = await async_client.index(name=shared_index_dim2)
    try:
        # 1. Create namespace — returns NamespaceDescription with record_count == 0
        created = await idx.create_namespace(name=ns_name)
        assert isinstance(created, NamespaceDescription)
        assert created.name == ns_name
        assert created.record_count == 0  # unified-ns-0002

        # 2. Describe namespace — poll for eventual consistency after create
        described = await async_poll_until(
            query_fn=lambda: idx.describe_namespace(name=ns_name),
            check_fn=lambda r: isinstance(r, NamespaceDescription),
            timeout=60,
            description=f"namespace {ns_name} visible after create",
        )
        assert isinstance(described, NamespaceDescription)
        assert described.name == ns_name
        assert isinstance(described.record_count, int)

        # 3. Namespace appears in list_namespaces_paginated with prefix match
        list_resp = await idx.list_namespaces_paginated(prefix=f"{ns}-crud-ns-", limit=100)
        assert isinstance(list_resp, ListNamespacesResponse)
        listed_names = [n.name for n in list_resp.namespaces]
        assert ns_name in listed_names

        # Each entry is a NamespaceDescription with string name and int record_count
        for n in list_resp.namespaces:
            assert isinstance(n, NamespaceDescription)
            assert isinstance(n.name, str)
            assert isinstance(n.record_count, int)

        # 4. Pagination token absent on the final page (unified-ns-0008)
        assert list_resp.pagination is None or list_resp.pagination.next is None

        # 5. Delete namespace — returns None on success (unified-ns-0004)
        await idx.delete_namespace(name=ns_name)

        # 6. After deletion, namespace no longer appears in listing
        post_delete = await idx.list_namespaces_paginated(prefix=f"{ns}-crud-ns-", limit=100)
        assert isinstance(post_delete, ListNamespacesResponse)
        post_names = [n.name for n in post_delete.namespaces]
        assert ns_name not in post_names
    finally:
        await async_cleanup_resource(
            lambda: idx.delete_namespace(name=ns_name), ns_name, "namespace"
        )


# ---------------------------------------------------------------------------
# list_namespaces generator — REST async
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_list_namespaces_generator_rest_async(
    async_client: AsyncPinecone, shared_index_dim2: str
) -> None:
    # shared_index_dim2
    """list_namespaces() async generator yields ListNamespacesResponse pages with
    NamespaceDescription items; generator follows pagination tokens automatically.

    Verifies claims:
    - unified-ns-0007: The namespace list generator yields NamespaceDescription objects per page.
    - unified-ns-0005: Can list all namespaces with optional prefix filtering and pagination.
    """
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    ns_a = f"{ns}-lnsgen-ns-a"
    ns_b = f"{ns}-lnsgen-ns-b"
    idx = await async_client.index(name=shared_index_dim2)

    # Upsert one vector into each namespace to create them implicitly
    upserted = await idx.upsert(vectors=[{"id": "lnsg-v1", "values": [0.1, 0.2]}], namespace=ns_a)
    assert upserted.upserted_count == 1
    upserted = await idx.upsert(vectors=[{"id": "lnsg-v2", "values": [0.3, 0.4]}], namespace=ns_b)
    assert upserted.upserted_count == 1

    # Poll until both namespaces appear in list_namespaces_paginated
    await async_poll_until(
        query_fn=lambda: idx.list_namespaces_paginated(prefix=f"{ns}-lnsgen-ns-", limit=100),
        check_fn=lambda r: len(r.namespaces) >= 2,
        timeout=120,
        description="both lnsgen-ns-* namespaces appear via list_namespaces_paginated",
    )

    # --- Exercise the async generator ---
    pages: list[ListNamespacesResponse] = []
    async for page in idx.list_namespaces(prefix=f"{ns}-lnsgen-ns-"):
        pages.append(page)

    assert len(pages) >= 1, "list_namespaces() async generator must yield at least one page"

    # Collect all namespace names across all yielded pages
    all_ns_names = [ns_entry.name for page in pages for ns_entry in page.namespaces]
    assert ns_a in all_ns_names, f"Expected {ns_a!r} in generator output; got {all_ns_names}"
    assert ns_b in all_ns_names, f"Expected {ns_b!r} in generator output; got {all_ns_names}"

    # Verify shape of every yielded page and its namespace descriptions
    for page in pages:
        assert isinstance(page, ListNamespacesResponse)
        assert len(page.namespaces) >= 1, "Each yielded page must contain at least one namespace"
        for ns_entry in page.namespaces:
            assert isinstance(ns_entry, NamespaceDescription)
            assert isinstance(ns_entry.name, str) and ns_entry.name.startswith(f"{ns}-lnsgen-ns-")
            assert isinstance(ns_entry.record_count, int) and ns_entry.record_count >= 0


# ---------------------------------------------------------------------------
# list_paginated multi-page — REST async
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_list_paginated_multi_page_rest_async(
    async_client: AsyncPinecone, shared_index_dim2: str
) -> None:
    # shared_index_dim2
    """list_paginated() with limit=2 returns a token when more pages exist; following the token
    reaches the next page; the final page has no token (async variant).

    Verifies claims:
    - unified-vec-0030: paginated list method returns a single page (caller must follow token)
    - unified-vec-0056: list-paginated returns no pagination token on the final page
    - unified-pag-0002: vector listing supports cursor-based pagination via single-page method
    """
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    idx = await async_client.index(name=shared_index_dim2)

    # Upsert 4 vectors with a shared prefix
    upserted = await idx.upsert(
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
    async def _list_all() -> ListResponse:
        return await idx.list_paginated(prefix="mp-", limit=100, namespace=ns)

    await async_poll_until(
        query_fn=_list_all,
        check_fn=lambda r: len(r.vectors) >= 4,
        timeout=120,
        description="all 4 vectors listable after upsert",
    )

    # Traverse all pages manually using limit=2 (forces at least 2 pages)
    all_ids: list[str] = []
    token: str | None = None
    pages_seen = 0

    while True:
        page = await idx.list_paginated(prefix="mp-", limit=2, pagination_token=token, namespace=ns)
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
# delete-nonexistent-ids — REST async
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_delete_nonexistent_ids_returns_none_async(
    async_client: AsyncPinecone, shared_index_dim2: str
) -> None:
    # shared_index_dim2
    """Delete with IDs that were never upserted (or already deleted) returns None.

    Verifies unified-vec-0032: "Deleting vectors does not raise an error when
    the specified IDs do not exist."
    """
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    idx = await async_client.index(name=shared_index_dim2)

    # Establish namespace by upserting a sentinel vector first
    upserted = await idx.upsert(vectors=[{"id": "dn-a1", "values": [0.3, 0.7]}], namespace=ns)
    assert upserted.upserted_count == 1
    await async_poll_until(
        query_fn=lambda: idx.fetch(ids=["dn-a1"], namespace=ns),
        check_fn=lambda r: len(r.vectors) == 1,
        timeout=120,
        description="dn-a1 fetchable (namespace established for async test)",
    )

    # Sub-case 1: delete IDs that were never upserted (namespace now exists) — no error raised
    await idx.delete(ids=["never-existed-async-x", "never-existed-async-y"], namespace=ns)

    # Sub-case 2: delete dn-a1 (exists), then delete again (already gone — idempotency)
    await idx.delete(ids=["dn-a1"], namespace=ns)
    await idx.delete(ids=["dn-a1"], namespace=ns)


# ---------------------------------------------------------------------------
# context-manager — REST async
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_index_context_manager_async(
    async_client: AsyncPinecone, shared_index_dim2: str
) -> None:
    # shared_index_dim2
    """AsyncIndex supports the async context manager protocol.

    Verifies unified-async-0002: the async index client implements __aenter__
    and __aexit__ for automatic resource cleanup.

    - 'async with idx as ai:' allows operations inside the block
    - __aenter__ returns the index object itself (not a copy)
    - describe_index_stats() works normally inside the context
    - After the with-block exits (__aexit__ calls close()), calling close()
      again must not raise (idempotent resource release)

    Area tag: context-manager
    Transport: rest-async
    """
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    async_index = await async_client.index(name=shared_index_dim2)
    async with async_index as ai:
        # __aenter__ must return the same object
        assert ai is async_index, "__aenter__ must return self"
        assert isinstance(ai, AsyncIndex)
        # Write first: describe_index_stats succeeds and reports zero on an
        # index that refuses vector writes, so a bare shape assertion would
        # hold against data that never landed (#322).
        upserted = await ai.upsert(vectors=[{"id": "cma-v1", "values": [0.1, 0.2]}], namespace=ns)
        assert upserted.upserted_count == 1
        stats = await async_poll_until(
            query_fn=lambda: ai.describe_index_stats(),
            check_fn=lambda s: ns in s.namespaces and s.namespaces[ns].vector_count == 1,
            timeout=120,
            description="context-manager namespace visible in stats (async)",
        )
        assert isinstance(stats, DescribeIndexStatsResponse)
    # After __aexit__ called close(), calling close() again must not raise
    await async_index.close()


# ---------------------------------------------------------------------------
# fetch-nonexistent — REST async
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_fetch_nonexistent_ids_returns_empty_vectors_async(
    async_client: AsyncPinecone,
    shared_index_dim3: str,
) -> None:
    # shared_index_dim3
    """Fetching IDs that were never upserted returns an empty vectors map, not an error.

    Verifies unified-vec-0053: "Fetching IDs that do not exist returns an empty
    vectors map rather than an error."

    Area tag: fetch-nonexistent
    Transport: rest-async
    """
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    idx = await async_client.index(name=shared_index_dim3)

    # Upsert one real vector to establish the namespace
    upserted = await idx.upsert(
        vectors=[{"id": "real-v1", "values": [0.1, 0.2, 0.3]}], namespace=ns
    )
    assert upserted.upserted_count == 1
    await async_poll_until(
        query_fn=lambda: idx.fetch(ids=["real-v1"], namespace=ns),
        check_fn=lambda r: "real-v1" in r.vectors,
        timeout=120,
        description="real-v1 fetchable after upsert (async)",
    )

    # Fetch IDs that were never upserted — should return empty dict, not raise
    result = await idx.fetch(ids=["never-upserted-aaa", "never-upserted-bbb"], namespace=ns)

    assert isinstance(result, FetchResponse)
    assert isinstance(result.vectors, dict)
    # Non-existent IDs are simply absent — no error raised
    assert "never-upserted-aaa" not in result.vectors
    assert "never-upserted-bbb" not in result.vectors


# ---------------------------------------------------------------------------
# namespace creation error paths — REST async
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_create_namespace_error_paths_async(
    async_client: AsyncPinecone, shared_index_dim2: str
) -> None:
    # shared_index_dim2
    """create_namespace() rejects invalid names client-side and raises ConflictError for duplicates.

    Verifies claims:
    - unified-ns-0010: Namespace creation is rejected when the name breaks the
      ASCII / no-NUL / 1-512 rule, or names the reserved ``__default__``.
    - unified-ns-0012: Creating a namespace that already exists raises a ConflictError (HTTP 409).

    Async transport parity for test_create_namespace_error_paths_rest.

    A whitespace-only name is deliberately absent: the rule is ASCII/no-NUL/1-512,
    not "non-blank", so ``"   "`` is a legal name the server accepts. Asserting a
    rejection here would both fail and leave a stray namespace on the shared index.
    """
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    ns_name = f"{ns}-cnep-ns-beta"
    idx = await async_client.index(name=shared_index_dim2)
    try:
        # unified-ns-0010: each of these is refused client-side, so no API call is made
        with pytest.raises(PineconeValueError):
            await idx.create_namespace(name="")

        with pytest.raises(PineconeValueError):
            await idx.create_namespace(name="naïve")

        with pytest.raises(PineconeValueError):
            await idx.create_namespace(name="a" * 513)

        with pytest.raises(PineconeValueError):
            await idx.create_namespace(name="__default__")

        # Precondition for ns-0012: create the namespace successfully
        created = await idx.create_namespace(name=ns_name)
        assert isinstance(created, NamespaceDescription)
        assert created.name == ns_name

        # unified-ns-0012: creating the same namespace again raises ConflictError (409)
        with pytest.raises(ConflictError) as exc_info:
            await idx.create_namespace(name=ns_name)
        assert exc_info.value.status_code == 409

    finally:
        await async_cleanup_resource(
            lambda: idx.delete_namespace(name=ns_name), ns_name, "namespace"
        )


# ---------------------------------------------------------------------------
# namespace creation with schema — REST async
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_create_namespace_with_schema_async(
    async_client: AsyncPinecone, shared_index_dim2: str
) -> None:
    # shared_index_dim2
    """create_namespace() accepts an optional schema dict and creates the namespace successfully (async).

    Verifies:
    - unified-ns-0001: Can create a named namespace, optionally providing a schema configuration.

    Async transport parity for test_create_namespace_with_schema_rest.
    The schema= path in the SDK sends body["schema"] = schema in the POST /namespaces request.
    """
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    ns_name = f"{ns}-schema-ns-beta"
    idx = await async_client.index(name=shared_index_dim2)

    try:
        # Create a namespace with a schema specifying that "genre" is filterable
        created = await idx.create_namespace(
            name=ns_name,
            schema={"fields": {"genre": {"filterable": True}}},
        )

        # Verify response type and structure
        assert isinstance(created, NamespaceDescription)
        assert created.name == ns_name
        assert created.record_count == 0  # new namespace has no vectors

        # describe_namespace returns the namespace as accessible (schema was accepted)
        # Poll for eventual consistency — backend may not expose the namespace immediately after create
        described = await async_poll_until(
            query_fn=lambda: idx.describe_namespace(name=ns_name),
            check_fn=lambda r: isinstance(r, NamespaceDescription),
            timeout=60,
            description=f"namespace {ns_name} visible after create",
        )
        assert isinstance(described, NamespaceDescription)
        assert described.name == ns_name
        assert isinstance(described.record_count, int)
        assert described.record_count == 0

    finally:
        await async_cleanup_resource(
            lambda: idx.delete_namespace(name=ns_name), ns_name, "namespace"
        )


# ---------------------------------------------------------------------------
# describe_namespace record_count after upsert — REST async
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_describe_namespace_record_count_updates_after_upsert_async(
    async_client: AsyncPinecone,
    shared_index_dim2: str,
) -> None:
    # shared_index_dim2
    """describe_namespace().record_count reflects the vector count after upsert (REST async).

    Verifies claim unified-ns-0003: "Can describe a namespace by name, returning its
    record count and schema." The record_count must accurately track the number of
    vectors in the namespace — not just be 0 at creation time.

    Operation sequence tested:
    1. Create namespace → verify record_count == 0
    2. Upsert 4 vectors into the namespace
    3. Poll describe_namespace() until record_count > 0 (eventual consistency)
    4. Verify record_count == 4

    Async transport parity for test_describe_namespace_record_count_updates_after_upsert_rest.
    """
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    ns_name = f"{ns}-rcnt-ns-beta"
    idx = await async_client.index(name=shared_index_dim2)
    try:
        # 1. Create namespace explicitly — record_count starts at 0
        created = await idx.create_namespace(name=ns_name)
        assert isinstance(created, NamespaceDescription)
        assert created.name == ns_name
        assert created.record_count == 0, (
            f"Freshly created namespace should have record_count == 0, got {created.record_count}"
        )

        # 2. Upsert 4 vectors into the namespace
        upsert_resp = await idx.upsert(
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
        final = await async_poll_until(
            query_fn=lambda: idx.describe_namespace(name=ns_name),
            check_fn=lambda r: r.record_count >= 4,
            timeout=120,
            description="describe_namespace record_count reaches 4 after upsert (async)",
        )
        assert isinstance(final, NamespaceDescription)

        # 4. Verify the record_count equals the number of vectors upserted
        assert final.record_count == 4, (
            f"Expected record_count == 4 after upserting 4 vectors, got {final.record_count}"
        )
        assert final.name == ns_name

    finally:
        await async_cleanup_resource(
            lambda: idx.delete_namespace(name=ns_name), ns_name, "namespace"
        )


# ---------------------------------------------------------------------------
# list_namespaces multi-page pagination — REST async
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_list_namespaces_multi_page_pagination_async(
    async_client: AsyncPinecone, shared_index_dim2: str
) -> None:
    # shared_index_dim2
    """list_namespaces_paginated() with limit=1 forces multi-page results; intermediate
    pages carry a non-None pagination token; the final page has no token (async variant).

    Verifies claims:
    - unified-ns-0008: Namespace list response omits the pagination token on the final page.
    - unified-ns-0005: Can list all namespaces with optional prefix filtering and pagination.
    """
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    prefix = f"{ns}-mpns-"
    ns_names = [f"{ns}-mpns-a", f"{ns}-mpns-b", f"{ns}-mpns-c"]
    idx = await async_client.index(name=shared_index_dim2)

    # Upsert one vector per namespace to create them implicitly
    for i, ns_item in enumerate(ns_names):
        upserted = await idx.upsert(
            vectors=[{"id": f"mpns-v{i}", "values": [0.1 * (i + 1), 0.2 * (i + 1)]}],
            namespace=ns_item,
        )
        assert upserted.upserted_count == 1

    # Wait until all 3 namespaces appear (eventual consistency)
    await async_poll_until(
        query_fn=lambda: idx.list_namespaces_paginated(prefix=prefix, limit=100),
        check_fn=lambda r: len(r.namespaces) >= 3,
        timeout=120,
        description="all 3 mpns-* namespaces visible via list_namespaces_paginated (async)",
    )

    # Traverse pages manually with limit=1 (forces >=3 pages for 3 namespaces)
    collected_names: list[str] = []
    token: str | None = None
    pages_seen = 0

    while True:
        page = await idx.list_namespaces_paginated(prefix=prefix, limit=1, pagination_token=token)
        assert isinstance(page, ListNamespacesResponse)
        assert len(page.namespaces) <= 1  # limit is respected per page

        for ns_entry in page.namespaces:
            assert isinstance(ns_entry, NamespaceDescription)
            assert isinstance(ns_entry.name, str)
            assert isinstance(ns_entry.record_count, int)
            collected_names.append(ns_entry.name)

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
    for ns_item in ns_names:
        assert ns_item in collected_names, (
            f"Expected {ns_item!r} in paginated results; got {collected_names}"
        )
    # At least 3 pages (one per namespace with limit=1)
    assert pages_seen >= 3, f"Expected >=3 pages with limit=1 and 3 namespaces; saw {pages_seen}"


# ---------------------------------------------------------------------------
# search with dense vector — wire format (mock HTTP)
# ---------------------------------------------------------------------------

_SEARCH_HOST = "dense-vec-async-test.svc.pinecone.io"
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
@pytest.mark.anyio
async def test_search_with_dense_vector() -> None:
    """AsyncIndex.search(vector=...) sends vector as {"values": [...]} object, not a bare array.

    Verifies SYNC-0098: the async code path sends the same corrected wire format
    as the sync path.
    """
    route = respx.post(_SEARCH_URL).mock(
        return_value=httpx.Response(200, json=_SEARCH_MOCK_RESPONSE),
    )
    idx = AsyncIndex(host=_SEARCH_HOST, api_key="test-key")
    try:
        response = await idx.search(namespace="vec-ns", top_k=3, vector=[0.1, 0.2, 0.3])
    finally:
        await idx.close()

    body = orjson.loads(route.calls.last.request.content)
    assert body["query"]["vector"] == {"values": [0.1, 0.2, 0.3]}, (
        f"Expected vector as object with 'values' key; got {body['query']['vector']!r}"
    )
    assert isinstance(response.result.hits, list)
    assert response.usage.read_units >= 0


# ---------------------------------------------------------------------------
# fetch_by_metadata — client-side limit validation
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_fetch_by_metadata_limit_validation() -> None:
    """fetch_by_metadata raises when limit=0 (minimum is 1 per OAS spec)."""
    idx = AsyncIndex(host="my-index.svc.pinecone.io", api_key="test-key")
    with pytest.raises((PineconeValueError, ValueError), match="limit"):
        await idx.fetch_by_metadata(filter={"a": "b"}, limit=0)
    await idx.close()


@pytest.mark.anyio
async def test_fetch_by_metadata_limit_validation_negative() -> None:
    """fetch_by_metadata raises when limit is negative."""
    idx = AsyncIndex(host="my-index.svc.pinecone.io", api_key="test-key")
    with pytest.raises((PineconeValueError, ValueError), match="limit"):
        await idx.fetch_by_metadata(filter={"a": "b"}, limit=-5)
    await idx.close()


# ---------------------------------------------------------------------------
# start_import — error_mode default behavior (mock HTTP)
# ---------------------------------------------------------------------------

_IMPORTS_HOST = "test-index-abc1234.svc.us-east1-gcp.pinecone.io"
_IMPORTS_URL = f"https://{_IMPORTS_HOST}/bulk/imports"


@respx.mock
@pytest.mark.anyio
async def test_async_start_import_error_mode_default() -> None:
    """Calling start_import without error_mode omits errorMode from request body."""
    from pinecone.models.imports.model import StartImportResponse

    route = respx.post(_IMPORTS_URL).mock(
        return_value=httpx.Response(200, json={"id": "import-default"}),
    )
    idx = AsyncIndex(host=_IMPORTS_HOST, api_key="test-key")
    try:
        result = await idx.start_import(uri="s3://my-bucket/vectors/")
    finally:
        await idx.close()

    assert isinstance(result, StartImportResponse)
    assert result.id == "import-default"

    body = orjson.loads(route.calls.last.request.content)
    assert body["uri"] == "s3://my-bucket/vectors/"
    assert "errorMode" not in body


@respx.mock
@pytest.mark.anyio
async def test_async_start_import_error_mode_abort_in_body() -> None:
    """Calling start_import(error_mode='abort') sends errorMode.onError='abort'."""
    route = respx.post(_IMPORTS_URL).mock(
        return_value=httpx.Response(200, json={"id": "import-abort"}),
    )
    idx = AsyncIndex(host=_IMPORTS_HOST, api_key="test-key")
    try:
        await idx.start_import(uri="s3://my-bucket/vectors/", error_mode="abort")
    finally:
        await idx.close()

    body = orjson.loads(route.calls.last.request.content)
    assert body["errorMode"] == {"onError": "abort"}


@respx.mock
@pytest.mark.anyio
async def test_async_start_import_error_mode_continue_in_body() -> None:
    """Calling start_import(error_mode='continue') sends errorMode.onError='continue'."""
    route = respx.post(_IMPORTS_URL).mock(
        return_value=httpx.Response(200, json={"id": "import-continue"}),
    )
    idx = AsyncIndex(host=_IMPORTS_HOST, api_key="test-key")
    try:
        await idx.start_import(uri="s3://my-bucket/vectors/", error_mode="continue")
    finally:
        await idx.close()

    body = orjson.loads(route.calls.last.request.content)
    assert body["errorMode"] == {"onError": "continue"}
