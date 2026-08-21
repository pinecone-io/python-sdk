"""Integration tests for collection lifecycle (async REST).

Phase 4 area tag: collection-lifecycle
Transport: rest-async

NOTE: Collections can only be created from pod-based indexes (not serverless).
Tests use a p1.x1 pod index in us-east-1-aws.
"""

from __future__ import annotations

import pytest

from pinecone import AsyncPinecone
from pinecone.errors.exceptions import ApiError
from pinecone.models.collections.list import CollectionList
from pinecone.models.collections.model import CollectionModel
from pinecone.models.indexes.specs import PodSpec
from tests.integration.conftest import async_cleanup_resource, async_poll_until, unique_name

# ---------------------------------------------------------------------------
# collection-lifecycle — REST async
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_collection_lifecycle_async(async_client: AsyncPinecone) -> None:
    """Full collection CRUD lifecycle via async REST: create a pod index,
    seed vectors, create a collection, verify CollectionModel fields, list,
    describe, then delete both the collection and the source index.

    Area tag: collection-lifecycle
    Transport: rest-async

    NOTE: Collections require a pod-based index. Serverless indexes do not
    support collection creation (HTTP 400: "Cannot create collections from
    serverless indexes").
    """
    index_name = unique_name("idx")
    col_name = unique_name("col")

    try:
        # 1. Create a small pod-based source index
        # Collections only work from pod-based indexes.
        await async_client.indexes.create(
            name=index_name,
            dimension=2,
            metric="cosine",
            spec=PodSpec(environment="us-east-1-aws", pod_type="p1.x1"),
            timeout=300,
        )

        # 2. Seed vectors — get the index host from describe() for async
        desc = await async_client.indexes.describe(index_name)
        index = await async_client.index(host=desc.host)
        upsert_result = await index.upsert(
            vectors=[
                {"id": "col-v1", "values": [0.1, 0.9]},
                {"id": "col-v2", "values": [0.5, 0.5]},
                {"id": "col-v3", "values": [0.9, 0.1]},
            ]
        )
        assert upsert_result.upserted_count == 3

        # Wait for vectors to be indexed before snapshotting
        await async_poll_until(
            query_fn=lambda: index.describe_index_stats(),
            check_fn=lambda s: s.total_vector_count >= 3,
            timeout=120,
            description="3 vectors indexed before collection creation",
        )

        # 3. Create the collection (snapshot of the index)
        col = await async_client.collections.create(name=col_name, source=index_name)
        assert isinstance(col, CollectionModel)
        assert col.name == col_name
        assert col.status in ("Initializing", "Ready")

        # 4. Poll until the collection is Ready (can take several minutes)
        ready_col = await async_poll_until(
            query_fn=lambda: async_client.collections.describe(col_name),
            check_fn=lambda c: c.status == "Ready",
            timeout=600,
            interval=10,
            description="collection Ready",
        )
        assert isinstance(ready_col, CollectionModel)
        assert ready_col.name == col_name
        assert ready_col.status == "Ready"

        # 5. Verify key CollectionModel fields
        assert ready_col.dimension == 2
        assert isinstance(ready_col.vector_count, int)
        assert ready_col.vector_count >= 3
        assert isinstance(ready_col.size, int)
        assert ready_col.size > 0
        assert ready_col.environment != ""

        # 6. list() — verify the collection appears
        col_list = await async_client.collections.list()
        assert isinstance(col_list, CollectionList)
        assert col_name in col_list.names()

        # 7. describe() returns the same CollectionModel
        desc_col = await async_client.collections.describe(col_name)
        assert isinstance(desc_col, CollectionModel)
        assert desc_col.name == col_name
        assert desc_col.dimension == 2
        assert isinstance(desc_col.vector_count, int)

    finally:
        # Clean up collection first, then the source index
        await async_cleanup_resource(
            lambda: async_client.collections.delete(col_name),
            col_name,
            "collection",
        )
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(index_name),
            index_name,
            "index",
        )


# ---------------------------------------------------------------------------
# collection-from-serverless — REST async
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_collection_from_serverless_raises_error_async(async_client: AsyncPinecone) -> None:
    """Creating a collection from a serverless index raises ApiError(400) — async REST.

    Collections are a pod-index-only feature (unified-col-0008). The Pinecone
    API returns HTTP 400 when the caller attempts to snapshot a serverless
    index. This test verifies the async client surfaces that error as ApiError
    with status_code=400.

    Area tag: collection-lifecycle
    Transport: rest-async
    Claim: unified-col-0008
    """
    index_name = unique_name("idx")
    col_name = unique_name("col")

    try:
        # Create a plain 2026-07 serverless index. This test does not touch
        # the vectors API (it only checks that collections.create rejects a
        # non-pod source), so it needs no legacy-index workaround.
        await async_client.indexes.create(
            name=index_name,
            schema={
                "fields": {
                    "embedding": {"type": "dense_vector", "dimension": 2, "metric": "cosine"}
                }
            },
            deployment={"deployment_type": "managed", "cloud": "aws", "region": "us-east-1"},
            timeout=120,
        )

        # Attempt to create a collection from the serverless index.
        # The API must reject this with HTTP 400.
        with pytest.raises(ApiError) as exc_info:
            await async_client.collections.create(name=col_name, source=index_name)

        err = exc_info.value
        assert err.status_code == 400, (
            f"Expected HTTP 400 for serverless collection source, got {err.status_code}"
        )

    finally:
        # The collection was never created (error was raised), so only clean
        # up the source index.
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(index_name),
            index_name,
            "index",
        )
