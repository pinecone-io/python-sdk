"""Integration tests for deep data-ingestion scenarios (async REST).

Phase 3 area tags: upsert-formats, upsert-batch, upsert-overwrite,
upsert-records, upsert-records-batch, update-metadata, update-sparse,
update-by-filter, delete-by-filter, delete-all-namespace

The dense-vector arms get their indexes from :func:`legacy_index_factory`,
not from ``pc.indexes.create``: 2026-07 has no way to create an index the
vectors API will serve. See :mod:`tests.integration.legacy_index` for the
sanctioned pattern. The integrated-embedding (``upsert_records`` /
``search``) arms are unaffected by the #322 gate -- a semantic-text-only
schema is served by the Records API -- so they keep creating their own
index at 2026-07 via ``client.indexes.create_for_model``.
"""

from __future__ import annotations

import math
import uuid

import pytest
import pytest_asyncio  # noqa: F401

from pinecone import AsyncPinecone, Pinecone, PineconeValueError, Vector
from pinecone.models.indexes.specs import ServerlessSpec
from pinecone.models.vectors.responses import (
    DescribeIndexStatsResponse,
    FetchResponse,
    UpdateResponse,
    UpsertRecordsResponse,
    UpsertResponse,
)
from pinecone.models.vectors.search import Hit, SearchRecordsResponse
from pinecone.models.vectors.sparse import SparseValues
from tests.integration.conftest import (
    LegacyIndexFactory,
    async_cleanup_resource,
    async_poll_until,
    unique_name,
)
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
def shared_index_dim4_dotproduct(client: Pinecone, legacy_index_factory: LegacyIndexFactory) -> str:
    index = legacy_index_factory(dimension=4, metric="dotproduct")
    assert_serves_vectors_api(client, index)
    return index.name


# ---------------------------------------------------------------------------
# upsert-formats — REST async
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_upsert_formats_async(
    async_client: AsyncPinecone, shared_index_dim4_dotproduct: str
) -> None:
    """Upsert using all accepted input formats in a single call via async REST.

    Formats under test:
    1. Vector object with dense values and metadata
    2. (id, values) tuple
    3. (id, values, metadata) tuple
    4. dict with id, values, sparse_values, and metadata
    """
    index = await async_client.index(name=shared_index_dim4_dotproduct)
    ns = f"ns-{uuid.uuid4().hex[:8]}"

    # Format 1: Vector object
    vec1 = Vector(id="fmt-v1", values=[0.1, 0.2, 0.3, 0.4], metadata={"fmt": "object", "n": 1})
    # Format 2: (id, values) tuple
    vec2 = ("fmt-v2", [0.2, 0.3, 0.4, 0.5])
    # Format 3: (id, values, metadata) tuple
    vec3 = ("fmt-v3", [0.3, 0.4, 0.5, 0.6], {"fmt": "tuple3", "n": 3})
    # Format 4: dict with sparse_values and metadata
    vec4 = {
        "id": "fmt-v4",
        "values": [0.4, 0.5, 0.6, 0.7],
        "sparse_values": {"indices": [0, 2], "values": [0.9, 0.8]},
        "metadata": {"fmt": "dict", "n": 4},
    }

    result = await index.upsert(namespace=ns, vectors=[vec1, vec2, vec3, vec4])
    assert isinstance(result, UpsertResponse)
    assert result.upserted_count == 4

    # Wait for eventual consistency — all 4 vectors must be fetchable
    fetched = await async_poll_until(
        query_fn=lambda: index.fetch(namespace=ns, ids=["fmt-v1", "fmt-v2", "fmt-v3", "fmt-v4"]),
        check_fn=lambda r: len(r.vectors) == 4,
        timeout=120,
        description="all 4 upserted vectors fetchable (async)",
    )

    assert isinstance(fetched, FetchResponse)

    # Verify Format 1: Vector object with metadata
    v1 = fetched.vectors["fmt-v1"]
    assert v1.id == "fmt-v1"
    assert len(v1.values) == 4
    assert all(math.isclose(a, b, rel_tol=1e-5) for a, b in zip(v1.values, [0.1, 0.2, 0.3, 0.4]))
    assert v1.metadata is not None
    assert v1.metadata.get("fmt") == "object"
    assert v1.metadata.get("n") == 1

    # Verify Format 2: (id, values) tuple — no metadata
    v2 = fetched.vectors["fmt-v2"]
    assert v2.id == "fmt-v2"
    assert len(v2.values) == 4
    assert all(math.isclose(a, b, rel_tol=1e-5) for a, b in zip(v2.values, [0.2, 0.3, 0.4, 0.5]))

    # Verify Format 3: (id, values, metadata) tuple
    v3 = fetched.vectors["fmt-v3"]
    assert v3.id == "fmt-v3"
    assert len(v3.values) == 4
    assert all(math.isclose(a, b, rel_tol=1e-5) for a, b in zip(v3.values, [0.3, 0.4, 0.5, 0.6]))
    assert v3.metadata is not None
    assert v3.metadata.get("fmt") == "tuple3"
    assert v3.metadata.get("n") == 3

    # Verify Format 4: dict with sparse_values and metadata
    v4 = fetched.vectors["fmt-v4"]
    assert v4.id == "fmt-v4"
    assert len(v4.values) == 4
    assert all(math.isclose(a, b, rel_tol=1e-5) for a, b in zip(v4.values, [0.4, 0.5, 0.6, 0.7]))
    assert v4.sparse_values is not None
    assert isinstance(v4.sparse_values, SparseValues)
    assert v4.sparse_values.indices == [0, 2]
    assert all(
        math.isclose(a, b, rel_tol=1e-5) for a, b in zip(v4.sparse_values.values, [0.9, 0.8])
    )
    assert v4.metadata is not None
    assert v4.metadata.get("fmt") == "dict"
    assert v4.metadata.get("n") == 4


# ---------------------------------------------------------------------------
# upsert-batch — REST async
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_upsert_batch_async(async_client: AsyncPinecone, shared_index_dim2: str) -> None:
    """Upsert 200 vectors in a single call via async REST.

    Verifies:
    - upserted_count == 200
    - describe_index_stats() reports total_vector_count >= 200 after consistency
    """
    index = await async_client.index(name=shared_index_dim2)
    ns = f"ns-{uuid.uuid4().hex[:8]}"

    vectors = [
        {"id": f"batch-{i}", "values": [float(i) / 200, 1.0 - float(i) / 200]} for i in range(200)
    ]

    result = await index.upsert(namespace=ns, vectors=vectors)
    assert isinstance(result, UpsertResponse)
    assert result.upserted_count == 200

    # Poll until all 200 vectors are registered in stats
    stats = await async_poll_until(
        query_fn=lambda: index.describe_index_stats(),
        check_fn=lambda r: r.total_vector_count >= 200,
        timeout=120,
        description="total_vector_count >= 200 in stats (async)",
    )
    assert stats.total_vector_count >= 200


# ---------------------------------------------------------------------------
# upsert-overwrite — REST async
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_upsert_overwrite_async(async_client: AsyncPinecone, shared_index_dim2: str) -> None:
    """Second upsert of the same ID fully replaces values AND metadata (async REST).

    Verifies:
    - Initial upsert stores values [0.1, 0.2] and metadata {"v": 1, "original": "yes"}
    - Second upsert of same ID with values [0.9, 0.8] and metadata {"v": 2, "new_key": "hello"}
      completely replaces the first write — old metadata keys are gone
    """
    index = await async_client.index(name=shared_index_dim2)
    ns = f"ns-{uuid.uuid4().hex[:8]}"

    # First write
    await index.upsert(
        namespace=ns,
        vectors=[{"id": "ow-1", "values": [0.1, 0.2], "metadata": {"v": 1, "original": "yes"}}],
    )

    # Wait for first write to be visible
    await async_poll_until(
        query_fn=lambda: index.fetch(namespace=ns, ids=["ow-1"]),
        check_fn=lambda r: "ow-1" in r.vectors,
        timeout=120,
        description="first upsert of ow-1 fetchable (async)",
    )

    # Verify first write values before overwriting
    fetched_before = await index.fetch(namespace=ns, ids=["ow-1"])
    v_before = fetched_before.vectors["ow-1"]
    assert all(math.isclose(a, b, rel_tol=1e-5) for a, b in zip(v_before.values, [0.1, 0.2]))
    assert v_before.metadata is not None
    assert v_before.metadata.get("v") == 1
    assert v_before.metadata.get("original") == "yes"

    # Second write — overwrite same ID
    await index.upsert(
        namespace=ns,
        vectors=[{"id": "ow-1", "values": [0.9, 0.8], "metadata": {"v": 2, "new_key": "hello"}}],
    )

    # Wait for second write to propagate — poll until values change
    async def _second_write_visible() -> object:
        r = await index.fetch(namespace=ns, ids=["ow-1"])
        if "ow-1" not in r.vectors:
            return None
        v = r.vectors["ow-1"]
        if not math.isclose(v.values[0], 0.9, rel_tol=1e-5):
            return None
        return r

    fetched_after = await async_poll_until(
        query_fn=_second_write_visible,
        check_fn=lambda r: r is not None,
        timeout=120,
        description="second upsert of ow-1 propagated (async, values[0] ~ 0.9)",
    )

    v_after = fetched_after.vectors["ow-1"]  # type: ignore[union-attr]
    assert v_after.id == "ow-1"
    # Values completely replaced
    assert all(math.isclose(a, b, rel_tol=1e-5) for a, b in zip(v_after.values, [0.9, 0.8])), (
        f"expected [0.9, 0.8] but got {v_after.values}"
    )
    # Metadata completely replaced — new keys present
    assert v_after.metadata is not None
    assert v_after.metadata.get("v") == 2
    assert v_after.metadata.get("new_key") == "hello"
    # Old metadata key gone
    assert "original" not in v_after.metadata, (
        f"old key 'original' should not persist after overwrite; got metadata={v_after.metadata}"
    )


# ---------------------------------------------------------------------------
# upsert-records-batch — REST async
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_upsert_records_batch_async(async_client: AsyncPinecone) -> None:
    """Upsert 50 records in one call to an integrated-inference index via async REST.

    Verifies:
    - upsert_records() returns UpsertRecordsResponse with record_count == 50
    - Records become searchable via search(inputs={"text": ...})
    - Hit structure has id (str) and score (float)
    """
    name = unique_name("idx")
    namespace = "urb-ns"
    try:
        await async_client.indexes.create_for_model(
            name=name,
            cloud="aws",
            region="us-east-1",
            embed={"model": "multilingual-e5-large", "field_map": {"text": "text"}},
        )

        # Wait for the index to be ready via async polling
        await async_poll_until(
            query_fn=lambda: async_client.indexes.describe(name),
            check_fn=lambda r: r.status.ready,
            timeout=300,
            interval=5,
            description=f"integrated index {name!r} ready",
        )

        # Populate host cache and get async index handle
        desc = await async_client.indexes.describe(name)
        index = await async_client.index(host=desc.host)

        records = [
            {
                "_id": f"urb-{i}",
                "text": f"Record number {i}: vector database similarity search use case {i}.",
            }
            for i in range(50)
        ]
        response = await index.upsert_records(records=records, namespace=namespace)
        assert isinstance(response, UpsertRecordsResponse)
        assert response.record_count == 50

        # Poll until at least some records are searchable (eventual consistency)
        search_resp = await async_poll_until(
            query_fn=lambda: index.search(
                namespace=namespace,
                top_k=10,
                inputs={"text": "vector database similarity search"},
            ),
            check_fn=lambda r: len(r.result.hits) > 0,
            timeout=120,
            description="batch upserted records searchable via async REST",
        )

        assert isinstance(search_resp, SearchRecordsResponse)
        assert len(search_resp.result.hits) > 0
        first_hit = search_resp.result.hits[0]
        assert isinstance(first_hit, Hit)
        assert isinstance(first_hit.id, str)
        assert isinstance(first_hit.score, float)
        assert first_hit.id.startswith("urb-")

    finally:
        await async_cleanup_resource(lambda: async_client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# upsert-records — REST async
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_upsert_records_async(async_client: AsyncPinecone) -> None:
    """Upsert records into an integrated-inference index via async REST.

    Verifies:
    - upsert_records() returns UpsertRecordsResponse with record_count == N
    - Uploaded records become searchable via search(inputs={"text": ...})
    - Hit structure has id (str) and score (float)
    """
    name = unique_name("idx")
    namespace = "urec-ns"
    try:
        await async_client.indexes.create_for_model(
            name=name,
            cloud="aws",
            region="us-east-1",
            embed={"model": "multilingual-e5-large", "field_map": {"text": "text"}},
        )

        # Wait for the index to be ready via async polling
        await async_poll_until(
            query_fn=lambda: async_client.indexes.describe(name),
            check_fn=lambda r: r.status.ready,
            timeout=300,
            interval=5,
            description=f"integrated index {name!r} ready",
        )

        # Populate host cache and get async index handle
        desc = await async_client.indexes.describe(name)
        index = await async_client.index(host=desc.host)

        records = [
            {"_id": "urec-1", "text": "Vector databases enable fast similarity search."},
            {"_id": "urec-2", "text": "RAG combines retrieval with language model generation."},
            {"_id": "urec-3", "text": "Embeddings are dense vector representations of data."},
        ]
        response = await index.upsert_records(records=records, namespace=namespace)
        assert isinstance(response, UpsertRecordsResponse)
        assert response.record_count == 3

        # Poll until records are searchable (eventual consistency)
        search_resp = await async_poll_until(
            query_fn=lambda: index.search(
                namespace=namespace,
                top_k=5,
                inputs={"text": "similarity search with embeddings"},
            ),
            check_fn=lambda r: len(r.result.hits) > 0,
            timeout=120,
            description="upserted records searchable via async REST",
        )

        assert isinstance(search_resp, SearchRecordsResponse)
        assert len(search_resp.result.hits) > 0
        first_hit = search_resp.result.hits[0]
        assert isinstance(first_hit, Hit)
        assert isinstance(first_hit.id, str)
        assert isinstance(first_hit.score, float)
        assert first_hit.id.startswith("urec-")

    finally:
        await async_cleanup_resource(lambda: async_client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# update-metadata — REST async
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_update_metadata_async(async_client: AsyncPinecone, shared_index_dim2: str) -> None:
    """index.update(id=..., set_metadata=...) merges metadata, not replaces (async REST).

    Verifies:
    - After update(set_metadata={"color": "blue"}), fetch returns color == "blue"
    - The existing key "size" == 5 is preserved (merge semantics)
    - update() returns an UpdateResponse
    """
    index = await async_client.index(name=shared_index_dim2)
    ns = f"ns-{uuid.uuid4().hex[:8]}"

    # Upsert a vector with two metadata fields
    await index.upsert(
        namespace=ns,
        vectors=[
            {
                "id": "um-v1",
                "values": [0.1, 0.2],
                "metadata": {"color": "red", "size": 5},
            }
        ],
    )

    # Wait for vector to be fetchable
    await async_poll_until(
        query_fn=lambda: index.fetch(namespace=ns, ids=["um-v1"]),
        check_fn=lambda r: "um-v1" in r.vectors,
        timeout=120,
        description="um-v1 fetchable before update (async)",
    )

    # Update only the "color" field — "size" should survive (merge semantics)
    update_resp = await index.update(namespace=ns, id="um-v1", set_metadata={"color": "blue"})
    assert isinstance(update_resp, UpdateResponse)

    # Poll until the metadata change propagates
    async def _color_updated() -> object:
        r = await index.fetch(namespace=ns, ids=["um-v1"])
        if "um-v1" not in r.vectors:
            return None
        meta = r.vectors["um-v1"].metadata
        if meta is None or meta.get("color") != "blue":
            return None
        return r

    fetched = await async_poll_until(
        query_fn=_color_updated,
        check_fn=lambda r: r is not None,
        timeout=120,
        description="um-v1 color updated to blue (async)",
    )

    v = fetched.vectors["um-v1"]  # type: ignore[union-attr]
    assert v.metadata is not None
    # Updated field
    assert v.metadata.get("color") == "blue", (
        f"expected color='blue', got {v.metadata.get('color')!r}"
    )
    # Preserved field — merge semantics (NOT replaced)
    assert v.metadata.get("size") == 5, (
        f"expected size=5 to be preserved but got {v.metadata.get('size')!r}"
    )


# ---------------------------------------------------------------------------
# update-sparse — REST async
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_update_sparse_async(
    async_client: AsyncPinecone, shared_index_dim4_dotproduct: str
) -> None:
    """index.update(id=..., sparse_values=...) replaces sparse component while preserving dense values (async REST).

    Verifies:
    - Upsert hybrid vector with sparse_values {"indices": [0, 3], "values": [0.5, 0.8]}
    - Update with new sparse_values {"indices": [1, 2], "values": [0.9, 0.7]}
    - Fetch and verify new sparse indices/values present
    - Dense values [0.1, 0.2, 0.3, 0.4] are unchanged
    - update() returns an UpdateResponse
    """
    index = await async_client.index(name=shared_index_dim4_dotproduct)
    ns = f"ns-{uuid.uuid4().hex[:8]}"

    # Upsert hybrid vector with initial sparse values
    await index.upsert(
        namespace=ns,
        vectors=[
            {
                "id": "us-v1",
                "values": [0.1, 0.2, 0.3, 0.4],
                "sparse_values": {"indices": [0, 3], "values": [0.5, 0.8]},
            }
        ],
    )

    # Wait for vector to be fetchable
    await async_poll_until(
        query_fn=lambda: index.fetch(namespace=ns, ids=["us-v1"]),
        check_fn=lambda r: "us-v1" in r.vectors,
        timeout=120,
        description="us-v1 fetchable before sparse update (async)",
    )

    # Update only the sparse values — dense values should be preserved
    update_resp = await index.update(
        namespace=ns,
        id="us-v1",
        sparse_values={"indices": [1, 2], "values": [0.9, 0.7]},
    )
    assert isinstance(update_resp, UpdateResponse)

    # Poll until the sparse values change propagates
    async def _sparse_updated() -> object:
        r = await index.fetch(namespace=ns, ids=["us-v1"])
        if "us-v1" not in r.vectors:
            return None
        v = r.vectors["us-v1"]
        if v.sparse_values is None or v.sparse_values.indices != [1, 2]:
            return None
        return r

    fetched = await async_poll_until(
        query_fn=_sparse_updated,
        check_fn=lambda r: r is not None,
        timeout=120,
        description="us-v1 sparse values updated to indices=[1, 2] (async)",
    )

    v = fetched.vectors["us-v1"]  # type: ignore[union-attr]
    # New sparse values present
    assert v.sparse_values is not None, "sparse_values should be present after update (async)"
    assert isinstance(v.sparse_values, SparseValues)
    assert v.sparse_values.indices == [1, 2], (
        f"expected sparse indices [1, 2], got {v.sparse_values.indices}"
    )
    assert all(
        math.isclose(a, b, rel_tol=1e-5) for a, b in zip(v.sparse_values.values, [0.9, 0.7])
    ), f"expected sparse values [0.9, 0.7], got {v.sparse_values.values}"
    # Dense values preserved
    assert len(v.values) == 4, f"expected 4 dense values, got {len(v.values)}"
    assert all(math.isclose(a, b, rel_tol=1e-5) for a, b in zip(v.values, [0.1, 0.2, 0.3, 0.4])), (
        f"expected dense values [0.1, 0.2, 0.3, 0.4], got {v.values}"
    )


# ---------------------------------------------------------------------------
# update-by-filter — REST async
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_update_by_filter_async(async_client: AsyncPinecone, shared_index_dim2: str) -> None:
    """Filter-based bulk metadata update via async REST.

    Upsert 5 vectors: 3 with genre=drama, 2 with genre=comedy.
    First test dry_run=True — verify it returns a matched_records count
    without mutating any vectors. Then apply the filter-based update and
    confirm only the 3 drama vectors received reviewed=True.
    """
    index = await async_client.index(name=shared_index_dim2)
    ns = f"ns-{uuid.uuid4().hex[:8]}"

    # Upsert 3 drama and 2 comedy vectors
    vectors = [
        {"id": "ubf-d1", "values": [0.1, 0.2], "metadata": {"genre": "drama"}},
        {"id": "ubf-d2", "values": [0.2, 0.3], "metadata": {"genre": "drama"}},
        {"id": "ubf-d3", "values": [0.3, 0.4], "metadata": {"genre": "drama"}},
        {"id": "ubf-c1", "values": [0.5, 0.6], "metadata": {"genre": "comedy"}},
        {"id": "ubf-c2", "values": [0.6, 0.7], "metadata": {"genre": "comedy"}},
    ]
    result = await index.upsert(namespace=ns, vectors=vectors)
    assert isinstance(result, UpsertResponse)
    assert result.upserted_count == 5

    all_ids = ["ubf-d1", "ubf-d2", "ubf-d3", "ubf-c1", "ubf-c2"]

    # Wait for all 5 vectors to be fetchable (eventual consistency)
    await async_poll_until(
        query_fn=lambda: index.fetch(namespace=ns, ids=all_ids),
        check_fn=lambda r: len(r.vectors) == 5,
        timeout=120,
        description="all 5 update-by-filter vectors fetchable (async)",
    )

    # Dry-run first — should return matched_records count without mutating
    dry_resp = await index.update(
        namespace=ns,
        filter={"genre": {"$eq": "drama"}},
        set_metadata={"reviewed": True},
        dry_run=True,
    )
    assert isinstance(dry_resp, UpdateResponse)
    # matched_records may be None if not yet indexed, otherwise should be >= 0
    if dry_resp.matched_records is not None:
        assert dry_resp.matched_records >= 0, (
            f"dry_run matched_records should be non-negative, got {dry_resp.matched_records}"
        )

    # Verify dry_run did NOT mutate — drama vectors should NOT have reviewed=True yet
    fetched_after_dry = await index.fetch(namespace=ns, ids=all_ids)
    for vid in ["ubf-d1", "ubf-d2", "ubf-d3"]:
        v = fetched_after_dry.vectors.get(vid)
        if v is not None and v.metadata is not None:
            assert v.metadata.get("reviewed") is None, (
                f"dry_run should not have mutated {vid}: got reviewed={v.metadata.get('reviewed')!r}"
            )

    # Now apply the real filter-based update
    update_resp = await index.update(
        namespace=ns,
        filter={"genre": {"$eq": "drama"}},
        set_metadata={"reviewed": True},
    )
    assert isinstance(update_resp, UpdateResponse)

    # Poll until the 3 drama vectors all have reviewed=True
    async def _all_drama_reviewed_async() -> object:
        r = await index.fetch(namespace=ns, ids=all_ids)
        if len(r.vectors) < 5:
            return None
        for vid in ["ubf-d1", "ubf-d2", "ubf-d3"]:
            v = r.vectors.get(vid)
            if v is None or v.metadata is None or v.metadata.get("reviewed") is not True:
                return None
        return r

    fetched = await async_poll_until(
        query_fn=_all_drama_reviewed_async,
        check_fn=lambda r: r is not None,
        timeout=180,
        description="all 3 drama vectors have reviewed=True after filter-update (async)",
    )

    # Verify drama vectors have reviewed=True
    for vid in ["ubf-d1", "ubf-d2", "ubf-d3"]:
        v = fetched.vectors[vid]  # type: ignore[union-attr]
        assert v.metadata is not None, f"{vid} should have metadata"
        assert v.metadata.get("reviewed") is True, (
            f"{vid} should have reviewed=True, got {v.metadata.get('reviewed')!r}"
        )
        assert v.metadata.get("genre") == "drama", (
            f"{vid} should still have genre=drama, got {v.metadata.get('genre')!r}"
        )

    # Verify comedy vectors were NOT touched
    for vid in ["ubf-c1", "ubf-c2"]:
        v = fetched.vectors[vid]  # type: ignore[union-attr]
        assert v.metadata is not None, f"{vid} should have metadata"
        assert v.metadata.get("reviewed") is None, (
            f"{vid} (comedy) should NOT have reviewed, got {v.metadata.get('reviewed')!r}"
        )
        assert v.metadata.get("genre") == "comedy", (
            f"{vid} should still have genre=comedy, got {v.metadata.get('genre')!r}"
        )


# ---------------------------------------------------------------------------
# delete-by-filter — REST async
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_delete_by_filter_async(async_client: AsyncPinecone, shared_index_dim2: str) -> None:
    """index.delete(filter=...) removes only vectors matching the filter (async REST).

    Upserts 5 vectors: 2 with status="obsolete", 3 with status="active".
    Calls delete(filter={"status": {"$eq": "obsolete"}}).
    Polls until the 2 obsolete vectors are absent from fetch.
    Verifies the 3 active vectors remain intact.
    """
    index = await async_client.index(name=shared_index_dim2)
    ns = f"ns-{uuid.uuid4().hex[:8]}"

    # 2 obsolete + 3 active vectors
    vectors = [
        {"id": "dbf-o1", "values": [0.1, 0.2], "metadata": {"status": "obsolete"}},
        {"id": "dbf-o2", "values": [0.2, 0.3], "metadata": {"status": "obsolete"}},
        {"id": "dbf-a1", "values": [0.5, 0.6], "metadata": {"status": "active"}},
        {"id": "dbf-a2", "values": [0.6, 0.7], "metadata": {"status": "active"}},
        {"id": "dbf-a3", "values": [0.7, 0.8], "metadata": {"status": "active"}},
    ]
    result = await index.upsert(namespace=ns, vectors=vectors)
    assert isinstance(result, UpsertResponse)
    assert result.upserted_count == 5

    all_ids = ["dbf-o1", "dbf-o2", "dbf-a1", "dbf-a2", "dbf-a3"]
    obsolete_ids = ["dbf-o1", "dbf-o2"]
    active_ids = ["dbf-a1", "dbf-a2", "dbf-a3"]

    # Wait for all 5 vectors to be fetchable before deleting
    await async_poll_until(
        query_fn=lambda: index.fetch(namespace=ns, ids=all_ids),
        check_fn=lambda r: len(r.vectors) == 5,
        timeout=120,
        description="all 5 delete-by-filter vectors fetchable (async)",
    )

    # Delete only the obsolete vectors via metadata filter
    await index.delete(namespace=ns, filter={"status": {"$eq": "obsolete"}})

    # Poll until both obsolete vectors disappear from fetch
    await async_poll_until(
        query_fn=lambda: index.fetch(namespace=ns, ids=obsolete_ids),
        check_fn=lambda r: len(r.vectors) == 0,
        timeout=120,
        description="obsolete vectors deleted by filter (async)",
    )

    # Verify active vectors still present
    active_fetch = await index.fetch(namespace=ns, ids=active_ids)
    assert isinstance(active_fetch, FetchResponse)
    for vid in active_ids:
        assert vid in active_fetch.vectors, (
            f"active vector {vid!r} should remain after filter-delete (async)"
        )
        v = active_fetch.vectors[vid]
        assert v.metadata is not None
        assert v.metadata.get("status") == "active", (
            f"{vid} should still have status='active', got {v.metadata.get('status')!r}"
        )

    # Confirm obsolete vectors are truly gone
    obsolete_fetch = await index.fetch(namespace=ns, ids=obsolete_ids)
    assert len(obsolete_fetch.vectors) == 0, (
        f"obsolete vectors should be deleted but found: {list(obsolete_fetch.vectors.keys())}"
    )


# ---------------------------------------------------------------------------
# delete-all-namespace — REST async
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_delete_all_namespace_async(
    async_client: AsyncPinecone, shared_index_dim2: str
) -> None:
    """index.delete(delete_all=True, namespace=...) deletes all vectors in a named
    namespace (async REST) while leaving other namespaces untouched.

    Upserts 3 vectors into a cleanup namespace and 2 vectors into a second,
    untouched namespace (this test's own — not the index's true default
    namespace, which other tests in this module also share).
    Calls delete(delete_all=True, namespace=<cleanup namespace>).
    Polls describe_index_stats() until the cleanup namespace is absent or has
    vector_count==0.
    Verifies the other namespace still has 2 vectors.
    """
    index = await async_client.index(name=shared_index_dim2)
    ns = f"dan-cleanup-{uuid.uuid4().hex[:8]}"
    other_ns = f"dan-other-{uuid.uuid4().hex[:8]}"
    other_ids = ["dan-def-1", "dan-def-2"]
    ns_ids = ["dan-ns-1", "dan-ns-2", "dan-ns-3"]

    # Upsert into the cleanup namespace
    ns_vectors = [
        {"id": "dan-ns-1", "values": [0.1, 0.2]},
        {"id": "dan-ns-2", "values": [0.3, 0.4]},
        {"id": "dan-ns-3", "values": [0.5, 0.6]},
    ]
    result = await index.upsert(vectors=ns_vectors, namespace=ns)
    assert isinstance(result, UpsertResponse)
    assert result.upserted_count == 3

    # Upsert into the other namespace
    other_vectors = [
        {"id": "dan-def-1", "values": [0.7, 0.8]},
        {"id": "dan-def-2", "values": [0.9, 0.1]},
    ]
    result2 = await index.upsert(vectors=other_vectors, namespace=other_ns)
    assert isinstance(result2, UpsertResponse)
    assert result2.upserted_count == 2

    # Wait for both namespaces to be indexed in stats before deleting
    await async_poll_until(
        query_fn=lambda: index.describe_index_stats(),
        check_fn=lambda r: (
            ns in r.namespaces
            and r.namespaces[ns].vector_count >= 3
            and other_ns in r.namespaces
            and r.namespaces[other_ns].vector_count >= 2
        ),
        timeout=120,
        description="both namespaces appear in stats before delete-all (async)",
    )

    # Delete all vectors in the cleanup namespace
    await index.delete(delete_all=True, namespace=ns)

    # Poll until the cleanup namespace is gone or empty
    await async_poll_until(
        query_fn=lambda: index.describe_index_stats(),
        check_fn=lambda r: ns not in r.namespaces or r.namespaces[ns].vector_count == 0,
        timeout=120,
        description="cleanup namespace empty after delete_all=True (async)",
    )

    # Verify cleanup-namespace vectors are gone from fetch
    ns_fetch = await index.fetch(ids=ns_ids, namespace=ns)
    assert isinstance(ns_fetch, FetchResponse)
    assert len(ns_fetch.vectors) == 0, (
        f"cleanup-namespace vectors should be gone but found: {list(ns_fetch.vectors.keys())} (async)"
    )

    # Verify the other namespace is unaffected
    other_fetch = await index.fetch(ids=other_ids, namespace=other_ns)
    assert isinstance(other_fetch, FetchResponse)
    for vid in other_ids:
        assert vid in other_fetch.vectors, (
            f"other-namespace vector {vid!r} should survive delete_all on a different namespace (async)"
        )

    # Verify stats: cleanup namespace is absent or has 0 vectors; other namespace still has 2
    stats = await index.describe_index_stats()
    assert isinstance(stats, DescribeIndexStatsResponse)
    if ns in stats.namespaces:
        assert stats.namespaces[ns].vector_count == 0, (
            f"cleanup namespace should be empty but has {stats.namespaces[ns].vector_count} vectors (async)"
        )
    assert stats.namespaces[other_ns].vector_count == 2, (
        f"other namespace should still hold 2 vectors, got "
        f"{stats.namespaces[other_ns].vector_count} (async)"
    )


# ---------------------------------------------------------------------------
# upsert-records input validation — REST async
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_upsert_records_validation_async(async_client: AsyncPinecone) -> None:
    """upsert_records raises PineconeValueError before any API call for invalid inputs.

    Verifies claims:
    - unified-vec-0049: Record upsert requires a non-empty records list.
    - unified-vec-0048: Each record must contain an '_id' or 'id' identifier field.

    All validation is client-side; no real index is created. The AsyncIndex is
    constructed with a dummy host so that validation fires before any HTTP call.
    """
    index = await async_client.index(host="https://dummy.example.com")

    # unified-vec-0049: empty records list raises PineconeValueError
    with pytest.raises(PineconeValueError):
        await index.upsert_records(records=[], namespace="test-ns")

    # unified-vec-0048: record missing both '_id' and 'id' raises PineconeValueError
    with pytest.raises(PineconeValueError):
        await index.upsert_records(
            records=[{"text": "no identifier field here"}],
            namespace="test-ns",
        )

    # namespace must be a non-empty string — whitespace-only is rejected
    with pytest.raises(PineconeValueError):
        await index.upsert_records(
            records=[{"_id": "v1", "text": "hello"}],
            namespace="",
        )

    with pytest.raises(PineconeValueError):
        await index.upsert_records(
            records=[{"_id": "v1", "text": "hello"}],
            namespace="   ",
        )


# ---------------------------------------------------------------------------
# delete() mode validation — REST async
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_delete_mode_validation_async(async_client: AsyncPinecone) -> None:
    """delete() raises PineconeValueError for conflicting or absent mode selection (async).

    Verifies claim unified-vec-0041: The delete operation accepts exactly one of:
    a list of identifiers, a delete-all flag, or a metadata filter; combining modes
    is not allowed. Passing no mode at all is also rejected.

    All validation is client-side; no real index is created. The AsyncIndex is
    constructed with a dummy host so that validation fires before any HTTP request.
    """
    index = await async_client.index(host="fake-index.svc.pinecone.io")

    # No mode at all raises PineconeValueError
    with pytest.raises(PineconeValueError):
        await index.delete()

    # ids + filter combined raises PineconeValueError
    with pytest.raises(PineconeValueError):
        await index.delete(ids=["v1"], filter={"category": {"$eq": "test"}})

    # ids + delete_all combined raises PineconeValueError
    with pytest.raises(PineconeValueError):
        await index.delete(ids=["v1"], delete_all=True)

    # delete_all + filter combined raises PineconeValueError
    with pytest.raises(PineconeValueError):
        await index.delete(delete_all=True, filter={"category": {"$eq": "test"}})

    # All three combined raises PineconeValueError
    with pytest.raises(PineconeValueError):
        await index.delete(ids=["v1"], delete_all=True, filter={"category": {"$eq": "test"}})


# ---------------------------------------------------------------------------
# upsert_from_dataframe — async REST (gated on pandas)
# ---------------------------------------------------------------------------


pd_udf = pytest.importorskip("pandas", reason="pandas required for upsert_from_dataframe")


@pytest.mark.integration
@pytest.mark.anyio
async def test_upsert_from_dataframe_async(async_client: AsyncPinecone) -> None:
    """AsyncIndex.upsert_from_dataframe works end-to-end over async REST.

    ``upsert_from_dataframe`` carries one signature across all three
    transports, so the async client batches and upserts a DataFrame just as
    the sync and gRPC clients do (#525).

    The index comes from ``indexes.create``'s legacy ``dimension=``/``spec=``
    sugar, which synthesizes a schema of exactly the reserved ``_values`` /
    ``_sparse_values`` fields — the shape the vectors API serves (#500, #504).
    """
    name = unique_name("idx")
    try:
        await async_client.indexes.create(
            name=name,
            dimension=4,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )
        desc = await async_client.indexes.describe(name)
        index = await async_client.index(host=desc.host)

        df = pd_udf.DataFrame(
            [
                {"id": f"audf-{i}", "values": [0.1 + i * 0.01 + j * 0.001 for j in range(4)]}
                for i in range(20)
            ]
        )
        resp = await index.upsert_from_dataframe(df, batch_size=5, show_progress=False)
        assert resp.upserted_count == 20
        assert resp.failed_item_count == 0

        stats = await async_poll_until(
            query_fn=lambda: index.describe_index_stats(),
            check_fn=lambda r: r.total_vector_count >= 20,
            timeout=120,
            description=f"async udf 20 rows visible in stats ({name})",
        )
        assert stats.total_vector_count == 20
    finally:
        await async_cleanup_resource(lambda: async_client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# upsert_records "id" field normalization and "_id" precedence — REST async
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_upsert_records_id_field_normalization_async(async_client: AsyncPinecone) -> None:
    """upsert_records normalizes "id" key → "_id" before sending to the API (async).

    Verifies unified-bp-0007: A record submitted for upsert must contain either '_id'
    or 'id'; 'id' is normalized to '_id' when '_id' is absent, and '_id' takes
    precedence when both keys are present (stripping the extra 'id' key).

    Mirrors test_upsert_records_id_field_normalization_rest for the async transport.
    """
    name = unique_name("idx")
    namespace = "id-norm-ns"
    try:
        await async_client.indexes.create_for_model(
            name=name,
            cloud="aws",
            region="us-east-1",
            embed={"model": "multilingual-e5-large", "field_map": {"text": "text"}},
        )

        # Wait for the index to be ready via async polling
        await async_poll_until(
            query_fn=lambda: async_client.indexes.describe(name),
            check_fn=lambda r: r.status.ready,
            timeout=300,
            interval=5,
            description=f"integrated index {name!r} ready",
        )

        desc = await async_client.indexes.describe(name)
        index = await async_client.index(host=desc.host)

        # Three normalization cases:
        # 1. Only "id" key — SDK must rename it to "_id" before sending
        # 2. Only "_id" key — no change needed
        # 3. Both "_id" and "id" — "_id" wins; "id" must be stripped before sending
        records = [
            {"id": "id-field-record", "text": "The id field is normalized to _id before sending."},
            {"_id": "underscore-id-record", "text": "Standard _id key for comparison."},
            {"_id": "underscore-id-wins", "id": "plain-id-loses", "text": "both keys test"},
        ]
        response = await index.upsert_records(records=records, namespace=namespace)
        assert isinstance(response, UpsertRecordsResponse)
        assert response.record_count == 3

        # Poll until all records appear in search results
        search_resp = await async_poll_until(
            query_fn=lambda: index.search(
                namespace=namespace,
                top_k=5,
                inputs={"text": "id normalization field"},
            ),
            check_fn=lambda r: len(r.result.hits) >= 3,
            timeout=120,
            description="all upserted records searchable (id normalization test, async)",
        )

        hit_ids = {hit.id for hit in search_resp.result.hits}
        assert "id-field-record" in hit_ids, (
            f"Expected 'id-field-record' in hit IDs (id key normalised to _id) but got: {hit_ids}"
        )
        assert "underscore-id-record" in hit_ids, (
            f"Expected 'underscore-id-record' in hit IDs but got: {hit_ids}"
        )
        assert "underscore-id-wins" in hit_ids, (
            f"Expected 'underscore-id-wins' in hit IDs (_id takes precedence) but got: {hit_ids}"
        )
        assert "plain-id-loses" not in hit_ids, (
            f"Expected 'plain-id-loses' NOT in hit IDs (id stripped when _id present) but got: {hit_ids}"
        )

    finally:
        await async_cleanup_resource(lambda: async_client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# upsert-duplicate-ids — REST async
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_upsert_duplicate_ids_in_batch_async(
    async_client: AsyncPinecone, shared_index_dim2: str
) -> None:
    """Duplicate vector IDs within a single upsert batch: last entry wins (async REST).

    Sends a batch of 3 vectors where "dup-v1" appears twice with different
    values and metadata.  Verifies:
    - The upsert call succeeds (no error)
    - After eventual consistency, exactly one record exists for "dup-v1"
    - The last submitted values for "dup-v1" are the ones persisted (last-write-wins)
    - The non-duplicate "dup-v2" is also stored correctly

    Depth-escalation boundary test.
    """
    index = await async_client.index(name=shared_index_dim2)
    ns = f"ns-{uuid.uuid4().hex[:8]}"

    first_values = [0.1, 0.2]
    last_values = [0.9, 0.8]
    vectors = [
        {"id": "dup-v1", "values": first_values, "metadata": {"version": 1}},
        {"id": "dup-v2", "values": [0.5, 0.5]},
        {"id": "dup-v1", "values": last_values, "metadata": {"version": 2}},
    ]

    result = await index.upsert(namespace=ns, vectors=vectors)
    assert isinstance(result, UpsertResponse)
    assert result.upserted_count >= 1, (
        f"upserted_count should be at least 1, got {result.upserted_count}"
    )

    await async_poll_until(
        query_fn=lambda: index.fetch(namespace=ns, ids=["dup-v1", "dup-v2"]),
        check_fn=lambda r: len(r.vectors) == 2,
        timeout=120,
        description="both dup-v1 and dup-v2 fetchable after duplicate batch upsert (async)",
    )

    fetched = await index.fetch(namespace=ns, ids=["dup-v1", "dup-v2"])
    assert isinstance(fetched, FetchResponse)
    assert len(fetched.vectors) == 2, (
        f"Expected exactly 2 unique vectors, got {len(fetched.vectors)}: "
        f"{list(fetched.vectors.keys())}"
    )
    assert "dup-v1" in fetched.vectors
    assert "dup-v2" in fetched.vectors

    v1 = fetched.vectors["dup-v1"]
    assert v1.values is not None
    assert len(v1.values) == 2
    assert (
        abs(v1.values[0] - last_values[0]) < 1e-4 and abs(v1.values[1] - last_values[1]) < 1e-4
    ), f"async: dup-v1 should have last_values {last_values!r} (last-write-wins), got {v1.values!r}"

    v2 = fetched.vectors["dup-v2"]
    assert v2.values is not None
    assert abs(v2.values[0] - 0.5) < 1e-4 and abs(v2.values[1] - 0.5) < 1e-4, (
        f"async: dup-v2 should have values [0.5, 0.5], got {v2.values!r}"
    )
