"""Integration tests for index CRUD operations (sync / REST + gRPC)."""

from __future__ import annotations

import contextlib
from typing import Any

import pytest

from pinecone import GrpcIndex, Index, Pinecone
from pinecone.errors import ForbiddenError, NotFoundError, PineconeValueError
from pinecone.models.indexes.deployment import ManagedDeployment
from pinecone.models.indexes.index import IndexModel, IndexStatus
from pinecone.models.indexes.schema import (
    DenseVectorField,
    IndexSchema,
    SemanticTextField,
    SparseVectorField,
)
from tests.integration.conftest import cleanup_resource, unique_name

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
def test_list_indexes_returns_paginator(client: Pinecone) -> None:
    """pc.indexes.list() yields IndexModel instances with distinct non-empty names.

    2026-07 replaced IndexList (which had len() and .names()) with a Paginator,
    so the name list is built by comprehension.
    """
    items = list(client.indexes.list())

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
def test_create_serverless_index_becomes_ready(client: Pinecone) -> None:
    """Create a serverless index, wait for ready state, verify fields, then delete."""
    name = unique_name("idx")
    try:
        model = client.indexes.create(
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
        cleanup_resource(
            lambda: client.indexes.delete(name),
            name,
            "index",
        )


# ---------------------------------------------------------------------------
# create-index — integrated (model-backed) dense
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(400)
def test_create_integrated_dense_index_becomes_ready(client: Pinecone) -> None:
    """Create an integrated dense index via create_for_model, verify fields, then delete.

    2026-07 moved integrated creation off ``create(spec=IntegratedSpec(...))``
    onto ``create_for_model``, and the embedding configuration comes back as a
    SemanticTextField named after the field_map text entry rather than as
    ``model.embed``.
    """
    name = unique_name("int")
    try:
        model = client.indexes.create_for_model(
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

        for result in (model, client.indexes.describe(name)):
            assert result.name == name
            assert result.status.ready is True
            assert result.status.state == "Ready"
            chunk_text = result.schema.fields["chunk_text"]
            assert isinstance(chunk_text, SemanticTextField)
            assert chunk_text.model == "llama-text-embed-v2"
            assert chunk_text.metric == "cosine"
    finally:
        cleanup_resource(
            lambda: client.indexes.delete(name),
            name,
            "index",
        )


# ---------------------------------------------------------------------------
# create-index — sparse vector field
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(400)
def test_create_sparse_index_becomes_ready(client: Pinecone) -> None:
    """Create an index with a sparse_vector schema field, verify fields, then delete.

    2026-07 replaced the ``vector_type`` discriminator with the schema field
    type: a SparseVectorField is what makes an index sparse, and it carries no
    dimension.
    """
    name = unique_name("spr")
    try:
        model = client.indexes.create(
            name=name,
            schema={"fields": {"sparse_embedding": {"type": "sparse_vector"}}},
            deployment=_MANAGED_AWS,
            timeout=300,
        )

        for result in (model, client.indexes.describe(name)):
            assert result.name == name
            assert result.status.ready is True
            assert result.status.state == "Ready"
            sparse = result.schema.fields["sparse_embedding"]
            assert isinstance(sparse, SparseVectorField)
            assert not any(
                isinstance(field, DenseVectorField) for field in result.schema.fields.values()
            )
    finally:
        cleanup_resource(
            lambda: client.indexes.delete(name),
            name,
            "index",
        )


# ---------------------------------------------------------------------------
# describe-index
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_describe_index_returns_full_model(client: Pinecone) -> None:
    """Create a serverless index, describe it, verify all IndexModel fields."""
    name = unique_name("idx")
    try:
        client.indexes.create(
            name=name,
            schema=_DOTPRODUCT_SCHEMA,
            deployment=_MANAGED_AWS,
            timeout=300,
        )

        desc = client.indexes.describe(name)

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

        for removed in ("spec", "embed", "created_at"):
            with pytest.raises(AttributeError, match="was removed in the 2026-07"):
                getattr(desc, removed)
    finally:
        cleanup_resource(
            lambda: client.indexes.delete(name),
            name,
            "index",
        )


# ---------------------------------------------------------------------------
# index-handle
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_index_handle_rest(client: Pinecone) -> None:
    """pc.index(name=...) returns a REST Index with the correct host."""
    name = unique_name("idx")
    try:
        client.indexes.create(
            name=name,
            schema=_DENSE_SCHEMA,
            deployment=_MANAGED_AWS,
            timeout=300,
        )

        # Get the expected host from describe
        desc = client.indexes.describe(name)
        expected_host = desc.host

        # Get an Index handle by name — triggers a describe call internally
        idx = client.index(name=name)

        assert isinstance(idx, Index)
        assert isinstance(idx.host, str)
        assert len(idx.host) > 0
        # Index normalizes host by prepending 'https://', so the raw describe
        # host (bare hostname) will be a suffix of idx.host
        assert expected_host in idx.host
    finally:
        cleanup_resource(
            lambda: client.indexes.delete(name),
            name,
            "index",
        )


@pytest.mark.integration
def test_index_handle_grpc(client: Pinecone) -> None:
    """pc.index(name=..., grpc=True) returns a GrpcIndex with the correct host."""
    name = unique_name("idx")
    try:
        client.indexes.create(
            name=name,
            schema=_DENSE_SCHEMA,
            deployment=_MANAGED_AWS,
            timeout=300,
        )

        # Get the expected host from describe
        desc = client.indexes.describe(name)
        expected_host = desc.host

        # Get a GrpcIndex handle by name
        idx = client.index(name=name, grpc=True)

        assert isinstance(idx, GrpcIndex)
        assert isinstance(idx.host, str)
        assert len(idx.host) > 0
        # GrpcIndex normalizes host similarly; bare hostname should appear in idx.host
        assert expected_host in idx.host
    finally:
        cleanup_resource(
            lambda: client.indexes.delete(name),
            name,
            "index",
        )


# ---------------------------------------------------------------------------
# index-tags
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_create_index_with_tags(client: Pinecone) -> None:
    """Create a serverless index with tags and verify they are returned by describe."""
    name = unique_name("idx")
    tags = {"env": "integration-test", "version": "1"}
    try:
        model = client.indexes.create(
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
        desc = client.indexes.describe(name)
        assert desc.tags is not None
        assert desc.tags.get("env") == "integration-test"
        assert desc.tags.get("version") == "1"
    finally:
        cleanup_resource(
            lambda: client.indexes.delete(name),
            name,
            "index",
        )


# ---------------------------------------------------------------------------
# index-exists
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_index_exists_returns_correct_bool(client: Pinecone) -> None:
    """indexes.exists() returns False before creation, True after, and False after deletion."""
    name = unique_name("idx")

    # Before creation: non-existent name → False
    assert client.indexes.exists(name) is False

    try:
        client.indexes.create(
            name=name,
            schema=_DENSE_SCHEMA,
            deployment=_MANAGED_AWS,
            timeout=300,
        )

        # After creation: existing index → True
        assert client.indexes.exists(name) is True

        # Delete the index and wait for it to disappear
        client.indexes.delete(name, timeout=120)

        # After deletion: name no longer exists → False
        assert client.indexes.exists(name) is False
    finally:
        cleanup_resource(
            lambda: client.indexes.delete(name),
            name,
            "index",
        )


@pytest.mark.integration
def test_index_exists_with_empty_name_raises(client: Pinecone) -> None:
    """An empty name raises before any network call.

    2026-07 changed this from returning False to raising PineconeValueError.
    """
    with pytest.raises(PineconeValueError):
        client.indexes.exists("")
    with pytest.raises(PineconeValueError):
        client.has_index("")


# ---------------------------------------------------------------------------
# configure-index
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_configure_index_updates_tags(client: Pinecone) -> None:
    """configure() merges tags — add new tags, update existing tags, remove tags via empty string."""
    name = unique_name("idx")
    try:
        client.indexes.create(
            name=name,
            schema=_DENSE_SCHEMA,
            deployment=_MANAGED_AWS,
            tags={"env": "integration-test", "version": "1", "to-remove": "yes"},
            timeout=300,
        )

        # Add a new tag and update an existing tag
        client.indexes.configure(
            name,
            tags={"version": "2", "new-key": "new-val"},
        )

        desc = client.indexes.describe(name)
        assert desc.tags is not None
        assert desc.tags.get("env") == "integration-test"  # untouched
        assert desc.tags.get("version") == "2"  # updated
        assert desc.tags.get("new-key") == "new-val"  # added
        assert desc.tags.get("to-remove") == "yes"  # not yet removed

        # Remove a tag by setting its value to ""
        client.indexes.configure(
            name,
            tags={"to-remove": ""},
        )

        desc2 = client.indexes.describe(name)
        assert desc2.tags is not None
        assert "to-remove" not in desc2.tags or desc2.tags.get("to-remove") == ""
        assert desc2.tags.get("version") == "2"  # preserved from previous configure
    finally:
        cleanup_resource(
            lambda: client.indexes.delete(name),
            name,
            "index",
        )


@pytest.mark.integration
def test_configure_deletion_protection_toggle_rest(client: Pinecone) -> None:
    """configure() can enable and disable deletion protection; delete raises ForbiddenError when enabled."""
    name = unique_name("idx")
    try:
        client.indexes.create(
            name=name,
            schema=_DENSE_SCHEMA,
            deployment=_MANAGED_AWS,
            timeout=300,
        )

        # Enable deletion protection
        client.indexes.configure(name, deletion_protection="enabled")

        desc = client.indexes.describe(name)
        assert desc.deletion_protection == "enabled"

        # Attempting to delete a protected index must raise ForbiddenError (HTTP 403)
        with pytest.raises(ForbiddenError) as exc_info:
            client.indexes.delete(name)
        assert exc_info.value.status_code == 403

        # Disable deletion protection so the index can be cleaned up
        client.indexes.configure(name, deletion_protection="disabled")

        desc2 = client.indexes.describe(name)
        assert desc2.deletion_protection == "disabled"
    finally:
        # Ensure protection is off before deletion (in case test failed mid-way)
        with contextlib.suppress(Exception):
            client.indexes.configure(name, deletion_protection="disabled")
        cleanup_resource(
            lambda: client.indexes.delete(name),
            name,
            "index",
        )


# ---------------------------------------------------------------------------
# delete with timeout=-1 (no-wait deletion) — REST sync
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_delete_index_timeout_minus1_returns_immediately(client: Pinecone) -> None:
    """indexes.delete(name, timeout=-1) returns None immediately without polling.

    Verifies claims:
    - unified-index-0057: deletion with timeout=-1 returns immediately without polling
    - unified-rs-0002: index deletion returns no response body (None)
    """
    from pinecone.errors import NotFoundError

    name = unique_name("idx")
    deleted = False
    try:
        client.indexes.create(
            name=name,
            schema=_DENSE_SCHEMA,
            deployment=_MANAGED_AWS,
            timeout=300,
        )

        # delete with timeout=-1 must return None immediately (no polling)
        result = client.indexes.delete(name, timeout=-1)
        deleted = True
        assert result is None  # unified-rs-0002: returns no response body

        # The index may still exist briefly (we didn't wait) — verify it eventually
        # disappears by polling the describe endpoint until NotFoundError
        import time

        start = time.monotonic()
        gone = False
        while time.monotonic() - start < 120:
            try:
                client.indexes.describe(name)
                time.sleep(5)
            except NotFoundError:
                gone = True
                break
        assert gone, f"Index '{name}' still exists 120s after delete(timeout=-1)"
    finally:
        if not deleted:
            cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# configure-index returns the updated model and preserves unspecified fields
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_configure_returns_model_and_preserves_deletion_protection(client: Pinecone) -> None:
    """configure() returns the updated IndexModel; omitting a field leaves it unchanged.

    2026-07 changed configure() from returning None (unified-index-0029) to
    returning the updated IndexModel, so the response itself is now the
    assertion surface for unified-index-0022 (fields omitted from the PATCH
    body keep their current value).
    """
    name = unique_name("idx")
    try:
        client.indexes.create(
            name=name,
            schema=_DENSE_SCHEMA,
            deployment=_MANAGED_AWS,
            timeout=300,
        )

        result1 = client.indexes.configure(name, deletion_protection="enabled")
        assert isinstance(result1, IndexModel)
        assert result1.deletion_protection == "enabled"

        desc1 = client.indexes.describe(name)
        assert desc1.deletion_protection == "enabled"

        result2 = client.indexes.configure(name, tags={"test-key": "test-val"})
        assert isinstance(result2, IndexModel)
        assert result2.deletion_protection == "enabled", (
            "deletion_protection must be preserved when configure() is called without it "
            "(unified-index-0022)"
        )

        desc2 = client.indexes.describe(name)
        assert desc2.deletion_protection == "enabled"
        assert desc2.tags is not None
        assert desc2.tags.get("test-key") == "test-val"

    finally:
        # Ensure deletion protection is disabled before attempting to delete
        with contextlib.suppress(Exception):
            client.indexes.configure(name, deletion_protection="disabled")
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# host-cache invalidation after delete
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_delete_index_clears_host_cache_rest(client: Pinecone) -> None:
    """Deleting an index clears the cached host URL; pc.index(name) then raises NotFoundError.

    Verifies claims:
    - unified-index-0020: Deleting an index removes that index's cached host URL.

    Sequence:
    1. Create index (populates nothing in cache yet).
    2. Call pc.index(name) — triggers describe + caches the resolved host.
    3. Verify the cache entry now exists.
    4. Delete the index (default timeout — polls until fully gone, clears cache).
    5. Verify cache entry was removed.
    6. Call pc.index(name) again — cache miss → fresh describe → NotFoundError.
    """
    name = unique_name("idx")
    deleted = False
    try:
        client.indexes.create(
            name=name,
            schema=_DENSE_SCHEMA,
            deployment=_MANAGED_AWS,
            timeout=300,
        )

        # Step 2: resolve host via name — this populates the cache
        idx = client.index(name=name)
        assert isinstance(idx, Index)

        # Step 3: host should now be cached
        assert name in client._host_cache, (
            "Host must be cached after pc.index(name=name) (unified-index-0019)"
        )

        # Step 4: delete clears cache immediately then polls until gone
        client.indexes.delete(name)
        deleted = True

        # Step 5: cache entry must be gone
        assert name not in client._host_cache, (
            "Host cache must be cleared after delete() (unified-index-0020)"
        )

        # Step 6: cache miss → auto-describe → NotFoundError (index is gone)
        with pytest.raises(NotFoundError):
            client.index(name=name)

    finally:
        if not deleted:
            cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# IndexModel bracket access — unified-index-0026
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_index_model_bracket_access_on_real_describe(client: Pinecone) -> None:
    """IndexModel supports bracket access and containment check on a real describe() response.

    unified-index-0026: "The describe-index response supports both attribute and
    bracket access."

    This test verifies that the string-key bracket syntax (model['name']) and the
    'in' operator work correctly on a real API-deserialized IndexModel, and that
    accessing a non-existent key raises KeyError.

    Index creation uses timeout=-1 so the test does not wait for the index to be
    ready — describe() returns a valid IndexModel even in Initializing state.

    Area tag: index-model-bracket-access
    Transport: rest
    """
    index_name = unique_name("idx")
    try:
        client.indexes.create(
            name=index_name,
            schema=_DENSE_SCHEMA,
            deployment=_MANAGED_AWS,
            timeout=-1,
        )

        model = client.indexes.describe(index_name)
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
        for removed in ("dimension", "metric", "vector_type", "spec", "embed"):
            assert removed not in model, f"Removed field {removed!r} must not be in IndexModel"
            with pytest.raises(KeyError):
                _ = model[removed]

        with pytest.raises(KeyError):
            _ = model["nonexistent_field_xyz"]

    finally:
        cleanup_resource(
            lambda: client.indexes.delete(index_name),
            index_name,
            "index",
        )


# ---------------------------------------------------------------------------
# back-compat shim — Pinecone.create_index_for_model()
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(400)
def test_create_index_for_model_shim_creates_index(client: Pinecone) -> None:
    """create_index_for_model() creates an index matching the indexes.create_for_model path.

    Verifies:
    - Passing embed as a plain dict is normalised correctly.
    - The returned IndexModel surfaces the embedding as a SemanticTextField named
      after the field_map text entry, matching test_create_integrated_dense_index.
    """
    name = unique_name("shim")
    try:
        model = client.create_index_for_model(
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

        assert model.name == name
        assert model.status.ready is True
        chunk_text = model.schema.fields["chunk_text"]
        assert isinstance(chunk_text, SemanticTextField)
        assert chunk_text.model == "llama-text-embed-v2"
    finally:
        cleanup_resource(
            lambda: client.indexes.delete(name),
            name,
            "index",
        )
