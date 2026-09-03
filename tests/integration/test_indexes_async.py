"""Integration tests for index CRUD operations (async / REST async)."""

from __future__ import annotations

from typing import Any

import pytest

from pinecone import AsyncIndex, AsyncPinecone
from pinecone.errors import ForbiddenError, PineconeValueError
from pinecone.models.indexes.deployment import ManagedDeployment
from pinecone.models.indexes.index import (
    IndexModel,
    IndexSpec,
    IndexStatus,
    ServerlessSpecInfo,
)
from pinecone.models.indexes.schema import (
    DenseVectorField,
    IndexSchema,
    SemanticTextField,
)
from tests.integration.conftest import async_cleanup_resource, unique_name

_DENSE_SCHEMA: dict[str, Any] = {
    "fields": {"embedding": {"type": "dense_vector", "dimension": 2, "metric": "cosine"}}
}
_DOTPRODUCT_SCHEMA: dict[str, Any] = {
    "fields": {"embedding": {"type": "dense_vector", "dimension": 4, "metric": "dotproduct"}}
}
_MANAGED_AWS: dict[str, Any] = {
    "deployment_type": "managed",
    "cloud": "aws",
    "region": "us-east-1",
}

# ---------------------------------------------------------------------------
# list-indexes
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_list_indexes_returns_async_paginator(async_client: AsyncPinecone) -> None:
    """async pc.indexes.list() yields IndexModel instances with distinct non-empty names.

    2026-07 replaced IndexList (len()/.names()) with an AsyncPaginator that is
    no longer a coroutine — it must be iterated with ``async for``, not awaited.
    """
    items = [idx async for idx in async_client.indexes.list()]

    for item in items:
        assert isinstance(item, IndexModel)

    names = [idx.name for idx in items]
    assert len(names) == len(items)
    for name in names:
        assert isinstance(name, str)
        assert len(name) > 0
    assert len(set(names)) == len(names)


# ---------------------------------------------------------------------------
# create-index
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_create_serverless_index_becomes_ready(async_client: AsyncPinecone) -> None:
    """Create a serverless index asynchronously, wait for ready state, verify fields, then delete."""
    name = unique_name("idx")
    try:
        model = await async_client.indexes.create(
            name=name,
            schema=_DENSE_SCHEMA,
            deployment=_MANAGED_AWS,
            timeout=300,
        )

        assert model.name == name
        assert model.status.ready is True
        assert model.status.state == "Ready"

        embedding = model.schema.fields["embedding"]
        assert isinstance(embedding, DenseVectorField)
        assert embedding.dimension == 2
        assert embedding.metric == "cosine"

        assert isinstance(model.deployment, ManagedDeployment)
        assert model.deployment.cloud == "aws"
        assert model.deployment.region == "us-east-1"

        assert model.deletion_protection == "disabled"
        assert isinstance(model.host, str)
        assert len(model.host) > 0
    finally:
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(name),
            name,
            "index",
        )


@pytest.mark.integration
@pytest.mark.anyio
@pytest.mark.timeout(400)
async def test_create_integrated_dense_index_becomes_ready_async(
    async_client: AsyncPinecone,
) -> None:
    """Create an integrated dense index via create_for_model, verify fields, then delete.

    2026-07 moved integrated creation off ``create(spec=IntegratedSpec(...))``
    onto ``create_for_model`` (new on the async namespace), and the embedding
    configuration comes back as a SemanticTextField named after the field_map
    text entry rather than as ``model.embed``.
    """
    name = unique_name("int")
    try:
        model = await async_client.indexes.create_for_model(
            name=name,
            cloud="aws",
            region="us-east-1",
            embed={
                "model": "llama-text-embed-v2",
                "field_map": {"text": "chunk_text"},
                "metric": "cosine",
            },
            timeout=300,
        )

        described = await async_client.indexes.describe(name)
        for result in (model, described):
            assert result.name == name
            assert result.status.ready is True
            assert result.status.state == "Ready"
            chunk_text = result.schema.fields["chunk_text"]
            assert isinstance(chunk_text, SemanticTextField)
            assert chunk_text.model == "llama-text-embed-v2"
            assert chunk_text.metric == "cosine"
    finally:
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(name),
            name,
            "index",
        )


# ---------------------------------------------------------------------------
# describe-index
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_describe_index_returns_full_model(async_client: AsyncPinecone) -> None:
    """Create a serverless index asynchronously, describe it, verify all IndexModel fields."""
    name = unique_name("idx")
    try:
        await async_client.indexes.create(
            name=name,
            schema=_DOTPRODUCT_SCHEMA,
            deployment=_MANAGED_AWS,
            timeout=300,
        )

        desc = await async_client.indexes.describe(name)

        assert isinstance(desc, IndexModel)
        assert desc.name == name
        assert desc.deletion_protection == "disabled"

        assert isinstance(desc.status, IndexStatus)
        assert desc.status.ready is True
        assert isinstance(desc.status.state, str)
        assert len(desc.status.state) > 0

        assert isinstance(desc.schema, IndexSchema)
        embedding = desc.schema.fields["embedding"]
        assert isinstance(embedding, DenseVectorField)
        assert embedding.dimension == 4
        assert embedding.metric == "dotproduct"

        assert isinstance(desc.deployment, ManagedDeployment)
        assert desc.deployment.cloud == "aws"
        assert desc.deployment.region == "us-east-1"

        assert isinstance(desc.host, str)
        assert len(desc.host) > 0

        assert desc.dimension == 4
        assert desc.metric == "dotproduct"
        assert desc.vector_type == "dense"

        assert isinstance(desc.spec, IndexSpec)
        assert isinstance(desc.spec.serverless, ServerlessSpecInfo)
        assert desc.spec.serverless.cloud == "aws"
        assert desc.spec.serverless.region == "us-east-1"
        assert desc.spec.pod is None
        assert desc.spec.byoc is None

        assert desc.embed is None

        with pytest.raises(AttributeError, match="was removed in the 2026-07"):
            _ = desc.created_at
    finally:
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(name),
            name,
            "index",
        )


# ---------------------------------------------------------------------------
# index-handle
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_index_handle_rest_async(async_client: AsyncPinecone) -> None:
    """async pc.index(name=...) returns an AsyncIndex with the correct host.

    AsyncPinecone.index() requires the host to be cached via a prior
    describe call. We call describe() first, which populates the host cache,
    then pc.index(name=name) should succeed.
    """
    name = unique_name("idx")
    idx = None
    try:
        await async_client.indexes.create(
            name=name,
            schema=_DENSE_SCHEMA,
            deployment=_MANAGED_AWS,
            timeout=300,
        )

        # describe() caches the host in AsyncPinecone's host cache
        desc = await async_client.indexes.describe(name)
        expected_host = desc.host

        # Get an AsyncIndex handle by name (uses cached host)
        idx = await async_client.index(name=name)

        assert isinstance(idx, AsyncIndex)
        assert isinstance(idx.host, str)
        assert len(idx.host) > 0
        # AsyncIndex normalizes host by prepending 'https://', so the raw describe
        # host (bare hostname) will appear within idx.host
        assert expected_host in idx.host
    finally:
        if idx is not None:
            await idx.close()
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(name),
            name,
            "index",
        )


# ---------------------------------------------------------------------------
# index-tags
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_create_index_with_tags(async_client: AsyncPinecone) -> None:
    """Create a serverless index with tags asynchronously and verify they are returned by describe."""
    name = unique_name("idx")
    tags = {"env": "integration-test", "version": "1"}
    try:
        model = await async_client.indexes.create(
            name=name,
            schema=_DENSE_SCHEMA,
            deployment=_MANAGED_AWS,
            tags=tags,
            timeout=300,
        )

        # Tags should be present on the create response
        assert model.tags is not None
        assert model.tags.get("env") == "integration-test"
        assert model.tags.get("version") == "1"

        # Tags should also be present on describe
        desc = await async_client.indexes.describe(name)
        assert desc.tags is not None
        assert desc.tags.get("env") == "integration-test"
        assert desc.tags.get("version") == "1"
    finally:
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(name),
            name,
            "index",
        )


# ---------------------------------------------------------------------------
# index-exists
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_index_exists_returns_correct_bool(async_client: AsyncPinecone) -> None:
    """async indexes.exists() returns False before creation, True after, and False after deletion."""
    name = unique_name("idx")

    # Before creation: non-existent name → False
    assert await async_client.indexes.exists(name) is False

    try:
        await async_client.indexes.create(
            name=name,
            schema=_DENSE_SCHEMA,
            deployment=_MANAGED_AWS,
            timeout=300,
        )

        # After creation: existing index → True
        assert await async_client.indexes.exists(name) is True

        # Delete the index and wait for it to disappear
        await async_client.indexes.delete(name, timeout=120)

        # After deletion: name no longer exists → False
        assert await async_client.indexes.exists(name) is False
    finally:
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(name),
            name,
            "index",
        )


@pytest.mark.integration
@pytest.mark.anyio
async def test_index_exists_with_empty_name_raises(async_client: AsyncPinecone) -> None:
    """An empty name raises before any network call.

    2026-07 changed the async lane from returning False to raising
    PineconeValueError, bringing it in line with the sync client.
    """
    with pytest.raises(PineconeValueError):
        await async_client.indexes.exists("")
    with pytest.raises(PineconeValueError):
        await async_client.has_index("")


# ---------------------------------------------------------------------------
# configure-index
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_configure_index_updates_tags(async_client: AsyncPinecone) -> None:
    """async configure() merges tags — add new tags, update existing, remove via empty string."""
    name = unique_name("idx")
    try:
        await async_client.indexes.create(
            name=name,
            schema=_DENSE_SCHEMA,
            deployment=_MANAGED_AWS,
            tags={"env": "integration-test", "version": "1", "to-remove": "yes"},
            timeout=300,
        )

        # Add a new tag and update an existing tag
        await async_client.indexes.configure(
            name,
            tags={"version": "2", "new-key": "new-val"},
        )

        desc = await async_client.indexes.describe(name)
        assert desc.tags is not None
        assert desc.tags.get("env") == "integration-test"  # untouched
        assert desc.tags.get("version") == "2"  # updated
        assert desc.tags.get("new-key") == "new-val"  # added
        assert desc.tags.get("to-remove") == "yes"  # not yet removed

        # Remove a tag by setting its value to ""
        await async_client.indexes.configure(
            name,
            tags={"to-remove": ""},
        )

        desc2 = await async_client.indexes.describe(name)
        assert desc2.tags is not None
        assert "to-remove" not in desc2.tags or desc2.tags.get("to-remove") == ""
        assert desc2.tags.get("version") == "2"  # preserved from previous configure
    finally:
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(name),
            name,
            "index",
        )


@pytest.mark.integration
@pytest.mark.anyio
async def test_configure_deletion_protection_toggle_async(async_client: AsyncPinecone) -> None:
    """async configure() can enable/disable deletion protection; delete raises ForbiddenError when enabled."""
    name = unique_name("idx")
    try:
        await async_client.indexes.create(
            name=name,
            schema=_DENSE_SCHEMA,
            deployment=_MANAGED_AWS,
            timeout=300,
        )

        # Enable deletion protection
        await async_client.indexes.configure(name, deletion_protection="enabled")

        desc = await async_client.indexes.describe(name)
        assert desc.deletion_protection == "enabled"

        # Attempting to delete a protected index must raise ForbiddenError (HTTP 403)
        with pytest.raises(ForbiddenError) as exc_info:
            await async_client.indexes.delete(name)
        assert exc_info.value.status_code == 403

        # Disable deletion protection so the index can be cleaned up
        await async_client.indexes.configure(name, deletion_protection="disabled")

        desc2 = await async_client.indexes.describe(name)
        assert desc2.deletion_protection == "disabled"
    finally:
        # Ensure protection is off before deletion (in case test failed mid-way)
        try:
            await async_client.indexes.configure(name, deletion_protection="disabled")
        except Exception:
            pass
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(name),
            name,
            "index",
        )


# ---------------------------------------------------------------------------
# delete with timeout=-1 (no-wait deletion) — REST async
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_delete_index_timeout_minus1_returns_immediately_async(
    async_client: AsyncPinecone,
) -> None:
    """async indexes.delete(name, timeout=-1) returns None immediately without polling.

    Verifies claims:
    - unified-index-0057: deletion with timeout=-1 returns immediately without polling
    - unified-rs-0002: index deletion returns no response body (None)
    """
    import asyncio

    from pinecone.errors import NotFoundError

    name = unique_name("idx")
    deleted = False
    try:
        await async_client.indexes.create(
            name=name,
            schema=_DENSE_SCHEMA,
            deployment=_MANAGED_AWS,
            timeout=300,
        )

        # delete with timeout=-1 must return None immediately (no polling)
        result = await async_client.indexes.delete(name, timeout=-1)
        deleted = True
        assert result is None  # unified-rs-0002: returns no response body

        # Poll until the index is gone (verify the deletion was actually triggered)
        gone = False
        for _ in range(24):  # up to 120 seconds (24 * 5s)
            try:
                await async_client.indexes.describe(name)
                await asyncio.sleep(5)
            except NotFoundError:
                gone = True
                break
        assert gone, f"Index '{name}' still exists 120s after async delete(timeout=-1)"
    finally:
        if not deleted:
            await async_cleanup_resource(
                lambda: async_client.indexes.delete(name),
                name,
                "index",
            )


# ---------------------------------------------------------------------------
# configure-index returns the updated model and preserves unspecified fields
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_configure_returns_model_and_preserves_deletion_protection_async(
    async_client: AsyncPinecone,
) -> None:
    """async configure() returns the updated IndexModel; omitted fields stay unchanged.

    2026-07 changed configure() from returning None (unified-index-0029) to
    returning the updated IndexModel, so the response itself is now the
    assertion surface for unified-index-0022 (fields omitted from the PATCH
    body keep their current value).
    """
    name = unique_name("idx")
    try:
        await async_client.indexes.create(
            name=name,
            schema=_DENSE_SCHEMA,
            deployment=_MANAGED_AWS,
            timeout=300,
        )

        result1 = await async_client.indexes.configure(name, deletion_protection="enabled")
        assert isinstance(result1, IndexModel)
        assert result1.deletion_protection == "enabled"

        desc1 = await async_client.indexes.describe(name)
        assert desc1.deletion_protection == "enabled"

        result2 = await async_client.indexes.configure(name, tags={"test-key": "test-val"})
        assert isinstance(result2, IndexModel)
        assert result2.deletion_protection == "enabled", (
            "deletion_protection must be preserved when configure() is called without it "
            "(unified-index-0022)"
        )

        desc2 = await async_client.indexes.describe(name)
        assert desc2.deletion_protection == "enabled"
        assert desc2.tags is not None
        assert desc2.tags.get("test-key") == "test-val"

    finally:
        # Ensure deletion protection is disabled before attempting to delete
        try:
            await async_client.indexes.configure(name, deletion_protection="disabled")
        except Exception:
            pass
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(name),
            name,
            "index",
        )


# ---------------------------------------------------------------------------
# async index factory requires prior describe; delete clears cache
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_async_index_factory_auto_resolves_on_cache_miss_rest_async(
    async_client: AsyncPinecone,
) -> None:
    """AsyncPinecone.index(name) auto-resolves the host via describe() on cache miss.

    Verifies claims:
    - unified-index-0020: Deleting an index removes that index's cached host URL.
    - unified-index-0024: Both sync and async index clients auto-resolve via
      describe() on cache miss; there is no asymmetry between the two.

    Sequence:
    1. Create index.
    2. Pop the host cache entry to simulate a cold-cache scenario.
    3. Call await async_client.index(name) — cache miss → describe is called →
       AsyncIndex returned; cache is repopulated.
    4. Delete the index (clears cache, polls until gone).
    5. Call await async_client.index(name) after deletion — describe returns 404
       → NotFoundError raised.
    """
    from pinecone.errors import NotFoundError

    name = unique_name("idx")
    deleted = False
    try:
        await async_client.indexes.create(
            name=name,
            schema=_DENSE_SCHEMA,
            deployment=_MANAGED_AWS,
            timeout=300,
        )
        # create() with timeout polling already populated the cache via describe.
        # Clear it to simulate a cold-cache scenario for the factory test.
        async_client._host_cache.pop(name, None)
        assert name not in async_client._host_cache

        # Step 3: cache miss → auto-resolve via describe → AsyncIndex returned
        idx = await async_client.index(name=name)
        assert isinstance(idx, AsyncIndex)
        assert name in async_client._host_cache, (
            "Host must be cached after auto-resolve on cache miss (unified-index-0024)"
        )

        # Step 4: delete clears cache immediately and polls until gone
        await async_client.indexes.delete(name)
        deleted = True

        assert name not in async_client._host_cache, (
            "Host cache must be cleared after delete() (unified-index-0020)"
        )

        # Step 5: cache miss after delete → describe returns 404 → NotFoundError
        with pytest.raises(NotFoundError):
            await async_client.index(name=name)

    finally:
        if not deleted:
            await async_cleanup_resource(
                lambda: async_client.indexes.delete(name),
                name,
                "index",
            )


# ---------------------------------------------------------------------------
# IndexModel bracket access — unified-index-0026
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
@pytest.mark.timeout(300)
async def test_index_model_bracket_access_on_real_describe_async(
    async_client: AsyncPinecone,
) -> None:
    """Async variant: IndexModel supports bracket access on a real describe() response.

    Verifies unified-index-0026 (bracket access) on the async transport path.

    Area tag: index-model-bracket-access
    Transport: rest-async
    """
    index_name = unique_name("idx")
    try:
        await async_client.indexes.create(
            name=index_name,
            schema=_DENSE_SCHEMA,
            deployment=_MANAGED_AWS,
            timeout=-1,
        )

        model = await async_client.indexes.describe(index_name)
        assert isinstance(model, IndexModel)

        for field in ("name", "host", "deletion_protection", "schema", "deployment", "status"):
            assert model[field] == getattr(model, field), (
                f"model[{field!r}] must equal model.{field}"
            )
            assert field in model, f"'{field}' must be in IndexModel"

        assert model["name"] == index_name, "Bracket 'name' must match the created index name"
        assert model["deletion_protection"] == "disabled", (
            "Bracket 'deletion_protection' must be 'disabled'"
        )
        assert model["schema"].fields["embedding"].metric == "cosine"

        assert "nonexistent_field_xyz" not in model, "Non-existent key must NOT be in IndexModel"

        for legacy in ("dimension", "metric", "vector_type", "spec", "embed"):
            assert legacy in model, f"Legacy accessor {legacy!r} must be in IndexModel"
            assert model[legacy] == getattr(model, legacy), (
                f"model[{legacy!r}] must equal model.{legacy}"
            )
        assert model["embed"] is None, "An index with no semantic text field has embed=None"

        assert "created_at" not in model, "'created_at' must not be in IndexModel"
        with pytest.raises(KeyError, match="was removed in the 2026-07"):
            _ = model["created_at"]

        with pytest.raises(KeyError):
            _ = model["nonexistent_field_xyz"]

    finally:
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(index_name),
            index_name,
            "index",
        )
