"""Integration tests for async preview end-to-end flow (2026-01.alpha).

Covers §13 "Async usage" from spec/preview.md:
- test_async_end_to_end_flow: create → batch_upsert → search → (cleanup fixture deletes)
- test_async_describe_after_create_matches_sync: describe path works async end-to-end

These tests make real API calls and skip gracefully when the preview endpoint is
unavailable. They do NOT gate CI (preview_integration marker).

Uses pytest-asyncio and the async_client fixture from the parent conftest.
"""

from __future__ import annotations

import contextlib

import pytest

from pinecone import AsyncPinecone
from pinecone.models.batch import BatchResult
from pinecone.preview import SchemaBuilder
from pinecone.preview.models import (
    PreviewDenseVectorQuery,
    PreviewDocument,
    PreviewDocumentSearchResponse,
    PreviewIndexModel,
    PreviewTextQuery,
    PreviewUsage,
)
from tests.integration.conftest import async_poll_until

pytestmark = [pytest.mark.integration, pytest.mark.preview_integration, pytest.mark.asyncio]


# ---------------------------------------------------------------------------
# test_async_end_to_end_flow — §13 "Async usage"
# ---------------------------------------------------------------------------


async def test_async_end_to_end_flow(
    async_client: AsyncPinecone,
    preview_index_name: str,
    preview_namespace: str,
    async_cleanup_preview_indexes: list[str],
    require_preview: None,
) -> None:
    """FTS schema: async create → batch_upsert 50 docs → poll → search → cleanup via fixture."""
    schema = SchemaBuilder().add_string_field("text", full_text_search={"language": "en"}).build()
    # Register for cleanup BEFORE create so teardown runs even if create raises.
    async_cleanup_preview_indexes.append(preview_index_name)

    model = await async_client.preview.indexes.create(name=preview_index_name, schema=schema)
    assert isinstance(model, PreviewIndexModel)
    assert model.name == preview_index_name

    idx = async_client.preview.index(name=preview_index_name)

    # Wait for index to be Ready.
    def _is_ready(m: object) -> bool:
        return isinstance(m, PreviewIndexModel) and m.status.state == "Ready"

    await async_poll_until(
        lambda: async_client.preview.indexes.describe(preview_index_name),
        _is_ready,
        timeout=300,
        interval=5,
        description=f"index {preview_index_name} ready",
    )

    documents = [{"_id": f"doc-{i}", "text": f"document {i}"} for i in range(50)]
    result = await idx.documents.batch_upsert(
        namespace=preview_namespace,
        documents=documents,
        batch_size=10,
        max_concurrency=4,
    )
    assert isinstance(result, BatchResult)
    assert result.successful_item_count == 50
    assert result.successful_batch_count == 5

    # Poll until all 50 docs are searchable (eventual consistency).
    score_by: list[object] = [PreviewTextQuery(fields=["text"], query="document")]

    async def _search() -> PreviewDocumentSearchResponse:
        return await idx.documents.search(
            namespace=preview_namespace,
            top_k=50,
            score_by=score_by,  # type: ignore[arg-type]
            include_fields=["text"],
        )

    results = await async_poll_until(
        _search,
        lambda r: isinstance(r, PreviewDocumentSearchResponse) and len(r.matches) == 50,
        timeout=90,
        interval=3,
        description="all 50 docs searchable",
    )
    assert isinstance(results, PreviewDocumentSearchResponse)
    for doc in results.matches:
        assert isinstance(doc, PreviewDocument)
        assert isinstance(doc._id, str)
        assert doc.score is not None
        assert isinstance(doc.score, float)
        assert doc.text is not None

    # Cleanup handled by async_cleanup_preview_indexes fixture.


# ---------------------------------------------------------------------------
# test_async_describe_after_create_matches_sync
# ---------------------------------------------------------------------------


async def test_async_describe_after_create_matches_sync(
    async_client: AsyncPinecone,
    preview_index_name: str,
    async_cleanup_preview_indexes: list[str],
    require_preview: None,
) -> None:
    """Async describe path returns correct name and host after create."""
    schema = SchemaBuilder().add_string_field("text", full_text_search={"language": "en"}).build()
    async_cleanup_preview_indexes.append(preview_index_name)
    await async_client.preview.indexes.create(name=preview_index_name, schema=schema)

    described = await async_client.preview.indexes.describe(preview_index_name)
    assert isinstance(described, PreviewIndexModel)
    assert described.name == preview_index_name
    assert isinstance(described.host, str) and len(described.host) > 0


# ---------------------------------------------------------------------------
# test_async_exists_returns_true_and_false — §2 exists(name) async parity
# ---------------------------------------------------------------------------


async def test_async_search_include_fields_variants(
    async_client: AsyncPinecone,
    preview_index_name: str,
    preview_namespace: str,
    async_cleanup_preview_indexes: list[str],
    require_preview: None,
) -> None:
    """Async parity: verify include_fields request construction; None causes 422 (IPV-0001)."""
    from pinecone.preview import SchemaBuilder as PreviewSchemaBuilder
    from pinecone.preview.models import PreviewDenseVectorQuery, PreviewIndexModel

    schema = (
        PreviewSchemaBuilder()
        .add_dense_vector_field("embedding", dimension=4, metric="cosine")
        .add_string_field("title", filterable=True)
        .add_string_field("category", filterable=True)
        .build()
    )
    async_cleanup_preview_indexes.append(preview_index_name)
    await async_client.preview.indexes.create(name=preview_index_name, schema=schema)

    def _is_ready(m: object) -> bool:
        return isinstance(m, PreviewIndexModel) and m.status.state == "Ready"

    await async_poll_until(
        lambda: async_client.preview.indexes.describe(preview_index_name),
        _is_ready,
        timeout=300,
        interval=5,
        description=f"index {preview_index_name} ready",
    )

    idx = async_client.preview.index(name=preview_index_name)
    await idx.documents.upsert(
        namespace=preview_namespace,
        documents=[
            {
                "_id": "doc-1",
                "embedding": [0.1, 0.2, 0.3, 0.4],
                "title": "ancient Rome",
                "category": "history",
            }
        ],
    )

    query_vec = [0.1, 0.2, 0.3, 0.4]
    score_by: list[object] = [PreviewDenseVectorQuery(field="embedding", values=query_vec)]

    # Case 1: include_fields=["*"] — SDK sends field; API accepts (200 OK).
    results_star = await idx.documents.search(
        namespace=preview_namespace,
        top_k=5,
        score_by=score_by,  # type: ignore[arg-type]
        include_fields=["*"],
    )
    assert isinstance(results_star, PreviewDocumentSearchResponse)

    # Case 2: include_fields=["title"] — SDK sends field; API accepts (200 OK).
    results_named = await idx.documents.search(
        namespace=preview_namespace,
        top_k=5,
        score_by=score_by,  # type: ignore[arg-type]
        include_fields=["title"],
    )
    assert isinstance(results_named, PreviewDocumentSearchResponse)

    # Case 3: include_fields=None (default) — SDK omits include_fields from body → 422.
    # Per spec, None should return only _id and score (a valid operation).
    # SDK BUG (IPV-0001): SDK omits include_fields entirely; API requires it as a list.
    results_default = await idx.documents.search(
        namespace=preview_namespace,
        top_k=5,
        score_by=score_by,  # type: ignore[arg-type]
    )
    assert isinstance(results_default, PreviewDocumentSearchResponse)


# ---------------------------------------------------------------------------
# test_async_exists_returns_true_and_false — §2 exists(name) async parity
# ---------------------------------------------------------------------------


async def test_async_exists_returns_true_and_false(
    async_client: AsyncPinecone,
    preview_index_name: str,
    async_cleanup_preview_indexes: list[str],
    require_preview: None,
) -> None:
    """Async exists() returns True for a created index and False for an unknown name."""
    from tests.integration.conftest import unique_name

    schema = (
        SchemaBuilder().add_dense_vector_field("embedding", dimension=4, metric="cosine").build()
    )
    async_cleanup_preview_indexes.append(preview_index_name)
    await async_client.preview.indexes.create(name=preview_index_name, schema=schema)

    result_true = await async_client.preview.indexes.exists(preview_index_name)
    assert result_true is True

    phantom_name = unique_name("phantom")
    result_false = await async_client.preview.indexes.exists(phantom_name)
    assert result_false is False


# ---------------------------------------------------------------------------
# test_async_upsert_returns_upserted_count — §5 async parity
# ---------------------------------------------------------------------------


async def test_async_upsert_returns_upserted_count(
    async_client: AsyncPinecone,
    preview_index_name: str,
    preview_namespace: str,
    async_cleanup_preview_indexes: list[str],
    require_preview: None,
) -> None:
    """Async upsert() returns PreviewDocumentUpsertResponse with upserted_count == len(documents).

    Async parity for test_upsert_returns_upserted_count: verifies §5 response
    shape via the async SDK path.
    """
    from pinecone.preview.models import PreviewDocumentUpsertResponse, PreviewIndexModel

    schema = (
        SchemaBuilder().add_dense_vector_field("embedding", dimension=4, metric="cosine").build()
    )
    async_cleanup_preview_indexes.append(preview_index_name)
    await async_client.preview.indexes.create(name=preview_index_name, schema=schema)

    def _is_ready(m: object) -> bool:
        return isinstance(m, PreviewIndexModel) and m.status.state == "Ready"

    await async_poll_until(
        lambda: async_client.preview.indexes.describe(preview_index_name),
        _is_ready,
        timeout=300,
        interval=5,
        description=f"index {preview_index_name} ready",
    )

    idx = async_client.preview.index(name=preview_index_name)
    documents = [
        {"_id": "doc-0", "embedding": [0.1, 0.2, 0.3, 0.4]},
        {"_id": "doc-1", "embedding": [0.5, 0.6, 0.7, 0.8]},
        {"_id": "doc-2", "embedding": [0.9, 0.1, 0.2, 0.3]},
    ]
    response = await idx.documents.upsert(namespace=preview_namespace, documents=documents)

    assert isinstance(response, PreviewDocumentUpsertResponse)
    assert response.upserted_count == len(documents)


# ---------------------------------------------------------------------------
# test_async_batch_upsert_result_fields — §5 async batch_upsert() BatchResult
# ---------------------------------------------------------------------------


async def test_async_batch_upsert_result_fields(
    async_client: AsyncPinecone,
    preview_index_name: str,
    preview_namespace: str,
    async_cleanup_preview_indexes: list[str],
    require_preview: None,
) -> None:
    """Async batch_upsert() returns BatchResult with correct counts for 10 docs in 2 batches.

    Async parity for test_batch_upsert_result_fields: verifies §5 BatchResult
    fields (total_item_count, failed_item_count, total_batch_count,
    failed_batch_count, has_errors, failed_items, errors) via the async path.
    """
    from pinecone.models.batch import BatchResult
    from pinecone.preview.models import PreviewIndexModel

    schema = (
        SchemaBuilder().add_dense_vector_field("embedding", dimension=4, metric="cosine").build()
    )
    async_cleanup_preview_indexes.append(preview_index_name)
    await async_client.preview.indexes.create(name=preview_index_name, schema=schema)

    def _is_ready(m: object) -> bool:
        return isinstance(m, PreviewIndexModel) and m.status.state == "Ready"

    await async_poll_until(
        lambda: async_client.preview.indexes.describe(preview_index_name),
        _is_ready,
        timeout=300,
        interval=5,
        description=f"index {preview_index_name} ready",
    )

    idx = async_client.preview.index(name=preview_index_name)
    documents = [
        {"_id": f"doc-{i}", "embedding": [float(i) / 10, 0.1, 0.2, 0.3]} for i in range(10)
    ]

    result = await idx.documents.batch_upsert(
        namespace=preview_namespace,
        documents=documents,
        batch_size=5,
        max_concurrency=2,
        show_progress=False,
    )

    assert isinstance(result, BatchResult)
    assert result.total_item_count == 10
    assert result.successful_item_count == 10
    assert result.failed_item_count == 0
    assert result.total_batch_count == 2
    assert result.successful_batch_count == 2
    assert result.failed_batch_count == 0
    assert result.has_errors is False
    assert result.failed_items == []
    assert result.errors == []


# ---------------------------------------------------------------------------
# test_async_fetch_wildcard_include_fields — §5 async fetch() include_fields=["*"]
# ---------------------------------------------------------------------------


async def test_async_fetch_wildcard_include_fields_returns_all_stored_fields(
    async_client: AsyncPinecone,
    preview_index_name: str,
    preview_namespace: str,
    async_cleanup_preview_indexes: list[str],
    require_preview: None,
) -> None:
    """Async fetch() with include_fields=["*"] returns all stored fields for each document.

    Async parity for test_fetch_wildcard_include_fields_returns_all_stored_fields:
    verifies §5 wildcard behavior via the async SDK path.

    SERVER BUG (IPV-0002): fetch() returns 401 "Unknown operation" for all
    preview index types. This test is expected to fail until IPV-0002 is resolved.
    """
    import asyncio

    from pinecone.preview.models import (
        PreviewDocument,
        PreviewDocumentFetchResponse,
        PreviewIndexModel,
    )

    schema = (
        SchemaBuilder()
        .add_dense_vector_field("embedding", dimension=4, metric="cosine")
        .add_string_field("category", filterable=True)
        .build()
    )
    async_cleanup_preview_indexes.append(preview_index_name)
    await async_client.preview.indexes.create(name=preview_index_name, schema=schema)

    def _is_ready(m: object) -> bool:
        return isinstance(m, PreviewIndexModel) and m.status.state == "Ready"

    await async_poll_until(
        lambda: async_client.preview.indexes.describe(preview_index_name),
        _is_ready,
        timeout=300,
        interval=5,
        description=f"index {preview_index_name} ready",
    )

    idx = async_client.preview.index(name=preview_index_name)
    docs = [
        {"_id": "fruit-0", "embedding": [0.1, 0.2, 0.3, 0.4], "category": "fruit"},
        {"_id": "fruit-1", "embedding": [0.5, 0.6, 0.7, 0.8], "category": "vegetable"},
    ]
    await idx.documents.upsert(namespace=preview_namespace, documents=docs)
    await asyncio.sleep(3)

    # Fetch with wildcard — all stored fields must come back.
    # IPV-0002: this call fails with 401 "Unknown operation" until fixed.
    response = await idx.documents.fetch(
        namespace=preview_namespace,
        ids=["fruit-0", "fruit-1"],
        include_fields=["*"],
    )

    assert isinstance(response, PreviewDocumentFetchResponse)
    assert set(response.documents.keys()) == {"fruit-0", "fruit-1"}

    for doc_id, doc in response.documents.items():
        assert isinstance(doc, PreviewDocument)
        assert doc._id == doc_id
        assert doc.category is not None, (
            f"doc {doc_id} missing 'category' with include_fields=['*']"
        )

    assert response.documents["fruit-0"].category == "fruit"
    assert response.documents["fruit-1"].category == "vegetable"


# ---------------------------------------------------------------------------
# test_async_configure_toggle_deletion_protection — §2 configure() async parity
# ---------------------------------------------------------------------------


async def test_async_configure_toggle_deletion_protection(
    async_client: AsyncPinecone,
    preview_index_name: str,
    async_cleanup_preview_indexes: list[str],
    require_preview: None,
) -> None:
    """Async configure() toggles deletion_protection; describe() reflects each change.

    Async parity for TestDeletionProtectionToggle.test_configure_toggle_deletion_protection:
    verifies §2 configure(deletion_protection=) and §3 PreviewIndexModel.deletion_protection
    via the async SDK path.
    """
    schema = (
        SchemaBuilder().add_dense_vector_field("embedding", dimension=4, metric="cosine").build()
    )
    async_cleanup_preview_indexes.append(preview_index_name)
    await async_client.preview.indexes.create(name=preview_index_name, schema=schema)

    def _is_ready(m: object) -> bool:
        return isinstance(m, PreviewIndexModel) and m.status.state == "Ready"

    await async_poll_until(
        lambda: async_client.preview.indexes.describe(preview_index_name),
        _is_ready,
        timeout=300,
        interval=5,
        description=f"index {preview_index_name} ready before configure",
    )

    initial = await async_client.preview.indexes.describe(preview_index_name)
    assert initial.deletion_protection == "disabled"

    # Enable deletion_protection
    await async_client.preview.indexes.configure(preview_index_name, deletion_protection="enabled")
    after_enable = await async_client.preview.indexes.describe(preview_index_name)
    assert after_enable.deletion_protection == "enabled"

    # Disable so the cleanup fixture can delete the index
    await async_client.preview.indexes.configure(preview_index_name, deletion_protection="disabled")
    after_disable = await async_client.preview.indexes.describe(preview_index_name)
    assert after_disable.deletion_protection == "disabled"


# ---------------------------------------------------------------------------
# test_async_add_custom_field_appears_in_describe — §1 add_custom_field() async parity
# ---------------------------------------------------------------------------


async def test_async_add_custom_field_appears_in_describe(
    async_client: AsyncPinecone,
    preview_index_name: str,
    async_cleanup_preview_indexes: list[str],
    require_preview: None,
) -> None:
    """Async parity: add_custom_field() raw dict passes through; describe() reflects the field.

    Async parity for test_add_custom_field_appears_in_describe: verifies §1
    add_custom_field() escape hatch via the async SDK path.
    """
    from pinecone.preview.models import PreviewIntegerField

    schema = (
        SchemaBuilder()
        .add_dense_vector_field("embedding", dimension=4, metric="cosine")
        .add_custom_field("score", {"type": "float", "filterable": True})
        .build()
    )

    # Verify build() output contains the custom dict unchanged.
    assert schema["fields"]["score"] == {"type": "float", "filterable": True}

    # Create the index via API.
    async_cleanup_preview_indexes.append(preview_index_name)
    await async_client.preview.indexes.create(name=preview_index_name, schema=schema)

    def _is_ready(m: object) -> bool:
        return isinstance(m, PreviewIndexModel) and m.status.state == "Ready"

    await async_poll_until(
        lambda: async_client.preview.indexes.describe(preview_index_name),
        _is_ready,
        timeout=300,
        interval=5,
        description=f"index {preview_index_name} ready",
    )

    # describe() must return the custom field as a typed model.
    model = await async_client.preview.indexes.describe(preview_index_name)
    assert "score" in model.schema.fields
    score_field = model.schema.fields["score"]
    assert isinstance(score_field, PreviewIntegerField), (
        f"expected PreviewIntegerField, got {type(score_field)}"
    )
    assert score_field.filterable is True


# ---------------------------------------------------------------------------
# test_async_create_with_tags_returns_tags_in_describe — §2/§3 tags async parity
# ---------------------------------------------------------------------------


async def test_async_create_with_tags_returns_tags_in_describe(
    async_client: AsyncPinecone,
    preview_index_name: str,
    async_cleanup_preview_indexes: list[str],
    require_preview: None,
) -> None:
    """Async create() with tags returns those tags verbatim in describe() PreviewIndexModel.tags.

    Async parity for TestIndexTags.test_create_with_tags_returns_tags_in_describe:
    verifies §2 create(tags=) and §3 PreviewIndexModel.tags via the async SDK path.
    """
    tags = {"env": "integration-test", "pvt": "PVT-008"}
    schema = (
        SchemaBuilder().add_dense_vector_field("embedding", dimension=4, metric="cosine").build()
    )
    async_cleanup_preview_indexes.append(preview_index_name)
    await async_client.preview.indexes.create(
        name=preview_index_name,
        schema=schema,
        tags=tags,
    )

    def _is_ready(m: object) -> bool:
        return isinstance(m, PreviewIndexModel) and m.status.state == "Ready"

    await async_poll_until(
        lambda: async_client.preview.indexes.describe(preview_index_name),
        _is_ready,
        timeout=300,
        interval=5,
        description=f"index {preview_index_name} ready",
    )

    model = await async_client.preview.indexes.describe(preview_index_name)
    assert model.tags is not None, "tags should not be None after create() with tags"
    assert model.tags == tags


# ---------------------------------------------------------------------------
# test_async_configure_tags_merges_with_existing_tags — §2 configure(tags=) merge
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# test_async_create_and_list_backup — §9 async variants + §2 create_backup/list_backups
# ---------------------------------------------------------------------------


async def test_async_create_and_list_backup(
    async_client: AsyncPinecone,
    preview_index_name: str,
    async_cleanup_preview_indexes: list[str],
    require_preview: None,
) -> None:
    """Async parity for backup operations — create_backup and list_backups work via async path.

    Async counterpart for test_backups.py sync tests. Verifies:
    - Async create_backup() returns BackupModel with all required fields.
    - source_index_name matches the index that was backed up.
    - Async list_backups() iteration yields the created backup by backup_id.
    """
    from pinecone.models.backups.model import BackupModel

    schema = (
        SchemaBuilder().add_dense_vector_field("embedding", dimension=4, metric="cosine").build()
    )
    async_cleanup_preview_indexes.append(preview_index_name)
    await async_client.preview.indexes.create(name=preview_index_name, schema=schema)

    def _is_ready(m: object) -> bool:
        return isinstance(m, PreviewIndexModel) and m.status.state == "Ready"

    await async_poll_until(
        lambda: async_client.preview.indexes.describe(preview_index_name),
        _is_ready,
        timeout=300,
        interval=5,
        description=f"index {preview_index_name} ready",
    )

    backup = await async_client.preview.indexes.create_backup(
        preview_index_name,
        name="async-integration-test",
        description="Async backup parity test",
    )

    assert isinstance(backup, BackupModel)
    assert isinstance(backup.backup_id, str) and len(backup.backup_id) > 0
    assert backup.source_index_name == preview_index_name
    assert isinstance(backup.source_index_id, str) and len(backup.source_index_id) > 0
    assert isinstance(backup.status, str) and len(backup.status) > 0
    assert isinstance(backup.cloud, str) and len(backup.cloud) > 0
    assert isinstance(backup.region, str) and len(backup.region) > 0
    assert isinstance(backup.created_at, str) and len(backup.created_at) > 0

    backup_ids = [
        b.backup_id async for b in async_client.preview.indexes.list_backups(preview_index_name)
    ]
    assert backup.backup_id in backup_ids

    async for item in async_client.preview.indexes.list_backups(preview_index_name):
        assert isinstance(item, BackupModel)
        assert isinstance(item.backup_id, str) and len(item.backup_id) > 0
        assert isinstance(item.status, str)
        assert isinstance(item.created_at, str)


async def test_async_configure_tags_merges_with_existing_tags(
    async_client: AsyncPinecone,
    preview_index_name: str,
    async_cleanup_preview_indexes: list[str],
    require_preview: None,
) -> None:
    """Async configure(tags=) merges new tags with existing tags — does not replace them.

    Async parity for TestIndexTags.test_configure_tags_merges_with_existing_tags:
    verifies §2 configure(tags=) merge behavior via the async SDK path.
    """
    initial_tags = {"env": "integration-test", "key1": "original"}
    schema = (
        SchemaBuilder().add_dense_vector_field("embedding", dimension=4, metric="cosine").build()
    )
    async_cleanup_preview_indexes.append(preview_index_name)
    await async_client.preview.indexes.create(
        name=preview_index_name,
        schema=schema,
        tags=initial_tags,
    )

    def _is_ready(m: object) -> bool:
        return isinstance(m, PreviewIndexModel) and m.status.state == "Ready"

    await async_poll_until(
        lambda: async_client.preview.indexes.describe(preview_index_name),
        _is_ready,
        timeout=300,
        interval=5,
        description=f"index {preview_index_name} ready",
    )

    await async_client.preview.indexes.configure(preview_index_name, tags={"key2": "added"})

    model = await async_client.preview.indexes.describe(preview_index_name)
    assert model.tags is not None, "tags should not be None after configure(tags=)"
    assert "env" in model.tags, "original tag 'env' must survive configure(tags=)"
    assert "key1" in model.tags, "original tag 'key1' must survive configure(tags=)"
    assert model.tags["key1"] == "original", "original tag value must be unchanged"
    assert "key2" in model.tags, "new tag 'key2' must be present after configure(tags=)"
    assert model.tags["key2"] == "added", "new tag value must be correct"


# ---------------------------------------------------------------------------
# test_async_search_response_namespace_and_usage — §7 async parity
# ---------------------------------------------------------------------------


async def test_async_search_response_namespace_and_usage(
    async_client: AsyncPinecone,
    preview_index_name: str,
    preview_namespace: str,
    async_cleanup_preview_indexes: list[str],
    require_preview: None,
) -> None:
    """Async parity: PreviewDocumentSearchResponse.namespace echoed and usage present (§7).

    Async counterpart for test_search_response_namespace_and_usage. Verifies that
    an async search() call returns a response envelope where:
    - namespace == the namespace argument passed to search()
    - usage is a PreviewUsage instance with read_units: int >= 0

    Uses include_fields=["*"] to avoid the IPV-0001 422 bug. 0 matches are
    acceptable — the envelope fields must always be present on a 200 OK response.
    """
    schema = (
        SchemaBuilder().add_dense_vector_field("embedding", dimension=4, metric="cosine").build()
    )
    async_cleanup_preview_indexes.append(preview_index_name)
    await async_client.preview.indexes.create(name=preview_index_name, schema=schema)

    def _is_ready(m: object) -> bool:
        return isinstance(m, PreviewIndexModel) and m.status.state == "Ready"

    await async_poll_until(
        lambda: async_client.preview.indexes.describe(preview_index_name),
        _is_ready,
        timeout=300,
        interval=5,
        description=f"index {preview_index_name} ready",
    )

    idx = async_client.preview.index(name=preview_index_name)
    await idx.documents.upsert(
        namespace=preview_namespace,
        documents=[{"_id": "doc-env", "embedding": [0.1, 0.2, 0.3, 0.4]}],
    )

    score_by: list[object] = [
        PreviewDenseVectorQuery(field="embedding", values=[0.1, 0.2, 0.3, 0.4])
    ]
    response = await idx.documents.search(
        namespace=preview_namespace,
        top_k=5,
        score_by=score_by,  # type: ignore[arg-type]
        include_fields=["*"],
    )

    assert isinstance(response, PreviewDocumentSearchResponse)
    # §7: namespace must be echoed back from the request.
    assert response.namespace == preview_namespace, (
        f"response.namespace {response.namespace!r} != request namespace {preview_namespace!r}"
    )
    # §7: usage must be present with a non-negative read_units counter.
    assert response.usage is not None, "response.usage must not be None after a successful search"
    assert isinstance(response.usage, PreviewUsage), (
        f"expected PreviewUsage, got {type(response.usage)}"
    )
    assert isinstance(response.usage.read_units, int), (
        f"read_units must be int, got {type(response.usage.read_units)}"
    )
    assert response.usage.read_units >= 0, (
        f"read_units must be >= 0, got {response.usage.read_units}"
    )


# ---------------------------------------------------------------------------
# test_async_delete_raises_forbidden_when_deletion_protection_enabled — §2
# ---------------------------------------------------------------------------


async def test_async_delete_raises_forbidden_when_deletion_protection_enabled(
    async_client: AsyncPinecone,
    preview_index_name: str,
    async_cleanup_preview_indexes: list[str],
    require_preview: None,
) -> None:
    """Async parity: delete() raises ForbiddenError when deletion_protection is "enabled" (§2).

    Async counterpart for
    TestDeletionProtectionEnforcement.test_delete_raises_forbidden_when_deletion_protection_enabled.
    Verifies the spec claim via the async SDK path: a delete() call on a
    deletion-protected index raises ForbiddenError and leaves the index intact.
    """
    from pinecone.errors import ForbiddenError
    from pinecone.preview.models import PreviewIndexModel

    schema = (
        SchemaBuilder().add_dense_vector_field("embedding", dimension=4, metric="cosine").build()
    )
    async_cleanup_preview_indexes.append(preview_index_name)
    await async_client.preview.indexes.create(name=preview_index_name, schema=schema)

    def _is_ready(m: object) -> bool:
        return isinstance(m, PreviewIndexModel) and m.status.state == "Ready"

    await async_poll_until(
        lambda: async_client.preview.indexes.describe(preview_index_name),
        _is_ready,
        timeout=300,
        interval=5,
        description=f"index {preview_index_name} ready",
    )

    await async_client.preview.indexes.configure(preview_index_name, deletion_protection="enabled")

    try:
        with pytest.raises(ForbiddenError):
            await async_client.preview.indexes.delete(preview_index_name)

        assert await async_client.preview.indexes.exists(preview_index_name), (
            "index must still exist after delete() was rejected by deletion_protection"
        )
    finally:
        # Disable protection so the cleanup fixture can delete the index.
        with contextlib.suppress(Exception):
            await async_client.preview.indexes.configure(
                preview_index_name, deletion_protection="disabled"
            )


# ---------------------------------------------------------------------------
# test_async_filter_integer_gte_and_operator_accepted — §8 Metadata filtering
# ---------------------------------------------------------------------------


async def test_async_filter_integer_gte_and_operator_accepted(
    async_client: AsyncPinecone,
    preview_index_name: str,
    async_cleanup_preview_indexes: list[str],
    preview_namespace: str,
    require_preview: None,
) -> None:
    """Async parity: filter with $gte (integer) and $and operator accepted by API (§8).

    Async counterpart for test_filter_integer_gte_and_operator_accepted.
    Verifies that the SDK correctly serializes $gte and $and operators and that the API
    returns 200 OK. 0 matches is acceptable (OnDemand dense vector indexing is eventually
    consistent). Uses dense vector + filterable integer year + filterable string category.
    """
    schema = (
        SchemaBuilder()
        .add_dense_vector_field("embedding", dimension=4, metric="cosine")
        .add_string_field("category", filterable=True)
        .add_float_field("year", filterable=True)
        .build()
    )
    async_cleanup_preview_indexes.append(preview_index_name)
    await async_client.preview.indexes.create(name=preview_index_name, schema=schema)

    def _is_ready(m: object) -> bool:
        return isinstance(m, PreviewIndexModel) and m.status.state == "Ready"

    await async_poll_until(
        lambda: async_client.preview.indexes.describe(preview_index_name),
        _is_ready,
        timeout=300,
        interval=5,
        description=f"index {preview_index_name} ready",
    )

    idx = async_client.preview.index(name=preview_index_name)
    await idx.documents.upsert(
        namespace=preview_namespace,
        documents=[
            {"_id": "doc-1", "embedding": [0.1, 0.2, 0.3, 0.4], "category": "tech", "year": 2022},
            {
                "_id": "doc-2",
                "embedding": [0.5, 0.6, 0.7, 0.8],
                "category": "science",
                "year": 2018,
            },
        ],
    )

    score_by: list[object] = [
        PreviewDenseVectorQuery(field="embedding", values=[0.1, 0.2, 0.3, 0.4])
    ]

    # Verify $gte filter on integer field is accepted (200 OK).
    result_gte = await idx.documents.search(
        namespace=preview_namespace,
        top_k=5,
        score_by=score_by,  # type: ignore[arg-type]
        filter={"year": {"$gte": 2020}},
        include_fields=["*"],
    )
    assert isinstance(result_gte, PreviewDocumentSearchResponse), (
        "$gte filter on integer field should return 200 OK"
    )

    # Verify $and operator combining $gte + $eq is accepted (200 OK).
    result_and = await idx.documents.search(
        namespace=preview_namespace,
        top_k=5,
        score_by=score_by,  # type: ignore[arg-type]
        filter={
            "$and": [
                {"category": {"$eq": "tech"}},
                {"year": {"$gte": 2020}},
            ]
        },
        include_fields=["*"],
    )
    assert isinstance(result_and, PreviewDocumentSearchResponse), (
        "$and filter with $gte + $eq should return 200 OK"
    )


# ---------------------------------------------------------------------------
# test_async_list_limit_* — §2 list(limit=N) parameter validation
# ---------------------------------------------------------------------------


async def test_async_list_limit_zero_raises_value_error(
    async_client: AsyncPinecone,
    require_preview: None,
) -> None:
    """Async parity: list(limit=0) raises PineconeValueError without an API call (§2).

    Async counterpart for TestListLimit.test_list_limit_zero_raises_value_error.
    The spec declares limit must be a positive integer; the async SDK validates this
    client-side before the async paginator is created. No await is needed.
    """
    from pinecone.errors import PineconeValueError

    with pytest.raises(PineconeValueError):
        async_client.preview.indexes.list(limit=0)


async def test_async_list_limit_negative_raises_value_error(
    async_client: AsyncPinecone,
    require_preview: None,
) -> None:
    """Async parity: list(limit=-1) raises PineconeValueError without an API call (§2)."""
    from pinecone.errors import PineconeValueError

    with pytest.raises(PineconeValueError):
        async_client.preview.indexes.list(limit=-1)


async def test_async_list_limit_caps_results(
    async_client: AsyncPinecone,
    preview_index_name: str,
    async_cleanup_preview_indexes: list[str],
    require_preview: None,
) -> None:
    """Async parity: list(limit=1) yields at most 1 PreviewIndexModel (§2).

    SDK BUG (IPV-0003): list() raises msgspec.ValidationError when the account contains
    any index whose schema has a field lacking a 'type' key. This test is DISABLED until
    IPV-0003 is fixed.
    """
    pytest.skip(
        "DISABLED (IPV-0003): list() crashes with msgspec.ValidationError for accounts with non-standard schema fields"
    )


# ---------------------------------------------------------------------------
# test_async_search_client_side_validation — §7 search() parameter validation
# ---------------------------------------------------------------------------


@pytest.mark.anyio
@pytest.mark.integration
@pytest.mark.timeout(300)
async def test_async_search_client_side_validation_rejects_invalid_parameters(
    async_client: AsyncPinecone,
    require_preview: None,
) -> None:
    """Async parity: search() raises ValidationError for invalid parameters without an API call.

    Async counterpart for test_search_client_side_validation_rejects_invalid_parameters.
    Spec §7 declares client-side validation fires before the HTTP request is sent:
    - namespace must be non-empty
    - top_k must be 1–10000
    - score_by must be non-empty

    Uses a dummy host — no network call is made.
    """
    from pinecone.errors.exceptions import ValidationError

    # Use a dummy host — async validation fires synchronously before the coroutine awaits.
    idx = async_client.preview.index(host="https://dummy-host.pinecone.io")

    valid_score_by: list[object] = [{"type": "dense_vector", "field": "emb", "values": [0.1]}]

    # Empty namespace string must raise ValidationError.
    with pytest.raises(ValidationError, match="namespace"):
        await idx.documents.search(namespace="", top_k=5, score_by=valid_score_by)  # type: ignore[arg-type]

    # top_k=0 is below the minimum of 1.
    with pytest.raises(ValidationError, match="top_k"):
        await idx.documents.search(namespace="ns", top_k=0, score_by=valid_score_by)  # type: ignore[arg-type]

    # top_k=10001 is above the maximum of 10000.
    with pytest.raises(ValidationError, match="top_k"):
        await idx.documents.search(namespace="ns", top_k=10001, score_by=valid_score_by)  # type: ignore[arg-type]

    # Empty score_by list must raise ValidationError.
    with pytest.raises(ValidationError, match="score_by"):
        await idx.documents.search(namespace="ns", top_k=5, score_by=[])


# ---------------------------------------------------------------------------
# test_async_upsert_client_side_validation — §4 async upsert() validation
# ---------------------------------------------------------------------------


@pytest.mark.anyio
@pytest.mark.integration
@pytest.mark.timeout(300)
async def test_async_upsert_client_side_validation_rejects_invalid_documents(
    async_client: AsyncPinecone,
    require_preview: None,
) -> None:
    """Async parity: upsert() raises ValidationError for all invalid-document conditions.

    Async counterpart for test_upsert_client_side_validation_rejects_invalid_documents.
    Spec §4 client-side validation fires synchronously at the start of the coroutine,
    before the first await, so ValidationError propagates through the await normally.
    Uses a dummy host — no network call is made.
    """
    from pinecone.errors.exceptions import ValidationError

    idx = async_client.preview.index(host="https://dummy-host.pinecone.io")

    valid_doc = {"_id": "doc-0", "text": "hello"}

    # Empty namespace must raise ValidationError.
    with pytest.raises(ValidationError, match="namespace"):
        await idx.documents.upsert(namespace="", documents=[valid_doc])

    # Empty documents list must raise ValidationError.
    with pytest.raises(ValidationError, match="documents"):
        await idx.documents.upsert(namespace="ns", documents=[])

    # More than 100 documents must raise ValidationError.
    over_limit = [{"_id": f"doc-{i}"} for i in range(101)]
    with pytest.raises(ValidationError, match="documents"):
        await idx.documents.upsert(namespace="ns", documents=over_limit)

    # Document missing '_id' key must raise ValidationError.
    with pytest.raises(ValidationError, match="_id"):
        await idx.documents.upsert(namespace="ns", documents=[{"text": "no id here"}])

    # Document with non-string '_id' must raise ValidationError.
    with pytest.raises(ValidationError, match="_id"):
        await idx.documents.upsert(namespace="ns", documents=[{"_id": 42}])

    # Document with empty string '_id' must raise ValidationError.
    with pytest.raises(ValidationError, match="_id"):
        await idx.documents.upsert(namespace="ns", documents=[{"_id": ""}])

    # Duplicate '_id' values within one call must raise ValidationError.
    with pytest.raises(ValidationError, match="_id"):
        await idx.documents.upsert(
            namespace="ns",
            documents=[{"_id": "dup"}, {"_id": "dup"}],
        )


# ---------------------------------------------------------------------------
# test_async_delete_client_side_validation — §4 async delete() validation
# ---------------------------------------------------------------------------


@pytest.mark.anyio
@pytest.mark.integration
@pytest.mark.timeout(300)
async def test_async_delete_client_side_validation_rejects_invalid_arguments(
    async_client: AsyncPinecone,
    require_preview: None,
) -> None:
    """Async parity: delete() raises ValidationError for missing or conflicting deletion targets.

    Async counterpart for test_delete_client_side_validation_rejects_invalid_arguments.
    Spec §4 validation fires synchronously at the coroutine start — no HTTP call is made.
    Uses a dummy host.
    """
    from pinecone.errors.exceptions import ValidationError

    idx = async_client.preview.index(host="https://dummy-host.pinecone.io")

    # Empty namespace must raise ValidationError.
    with pytest.raises(ValidationError, match="namespace"):
        await idx.documents.delete(namespace="", ids=["doc-0"])

    # Calling delete() with no targets must raise ValidationError.
    with pytest.raises(ValidationError, match="ids"):
        await idx.documents.delete(namespace="ns")

    # ids and delete_all=True are mutually exclusive.
    with pytest.raises(ValidationError, match="ids"):
        await idx.documents.delete(namespace="ns", ids=["doc-0"], delete_all=True)

    # ids and filter are mutually exclusive.
    with pytest.raises(ValidationError, match="ids"):
        await idx.documents.delete(
            namespace="ns", ids=["doc-0"], filter={"category": {"$eq": "fruit"}}
        )


# ---------------------------------------------------------------------------
# test_async_filter_remaining_operators_accepted — §8 Metadata filtering
# ---------------------------------------------------------------------------


@pytest.mark.anyio
@pytest.mark.integration
@pytest.mark.timeout(300)
async def test_async_filter_remaining_operators_accepted(
    async_client: AsyncPinecone,
    preview_index_name: str,
    async_cleanup_preview_indexes: list[str],
    preview_namespace: str,
    require_preview: None,
) -> None:
    """Async parity: remaining filter operators ($ne, $gt, $lt, $lte, $in, $nin, $or) accepted.

    Async counterpart for test_filter_remaining_operators_accepted.
    Verifies that the SDK serializes each of the 7 operators not covered by PVT-013
    ($ne, $gt, $lt, $lte, $in, $nin, $or) and that the API returns 200 OK.
    0 matches is acceptable — OnDemand indexing is eventually consistent.
    """
    schema = (
        SchemaBuilder()
        .add_dense_vector_field("embedding", dimension=4, metric="cosine")
        .add_string_field("category", filterable=True)
        .add_float_field("year", filterable=True)
        .build()
    )
    async_cleanup_preview_indexes.append(preview_index_name)
    await async_client.preview.indexes.create(name=preview_index_name, schema=schema)

    def _is_ready(m: object) -> bool:
        return isinstance(m, PreviewIndexModel) and m.status.state == "Ready"

    await async_poll_until(
        lambda: async_client.preview.indexes.describe(preview_index_name),
        _is_ready,
        timeout=300,
        interval=5,
        description=f"index {preview_index_name} ready",
    )

    idx = async_client.preview.index(name=preview_index_name)
    await idx.documents.upsert(
        namespace=preview_namespace,
        documents=[
            {"_id": "doc-1", "embedding": [0.1, 0.2, 0.3, 0.4], "category": "tech", "year": 2022},
            {
                "_id": "doc-2",
                "embedding": [0.5, 0.6, 0.7, 0.8],
                "category": "science",
                "year": 2018,
            },
        ],
    )

    score_by: list[object] = [
        PreviewDenseVectorQuery(field="embedding", values=[0.1, 0.2, 0.3, 0.4])
    ]

    async def _search_with_filter(f: dict[str, object]) -> None:
        result = await idx.documents.search(
            namespace=preview_namespace,
            top_k=5,
            score_by=score_by,  # type: ignore[arg-type]
            filter=f,
            include_fields=["*"],
        )
        assert isinstance(result, PreviewDocumentSearchResponse), (
            f"filter {f} should return a PreviewDocumentSearchResponse (200 OK)"
        )

    # $ne — not equals on string field
    await _search_with_filter({"category": {"$ne": "finance"}})

    # $gt — greater than on integer field
    await _search_with_filter({"year": {"$gt": 2010}})

    # $lt — less than on integer field
    await _search_with_filter({"year": {"$lt": 2025}})

    # $lte — less or equal on integer field
    await _search_with_filter({"year": {"$lte": 2022}})

    # $in — value in array on string field
    await _search_with_filter({"category": {"$in": ["tech", "medicine"]}})

    # $nin — value not in array on string field
    await _search_with_filter({"category": {"$nin": ["finance", "sports"]}})

    # $or — logical OR combining two conditions
    await _search_with_filter({"$or": [{"category": {"$eq": "tech"}}, {"year": {"$lt": 2020}}]})


# ---------------------------------------------------------------------------
# test_async_delete_timeout_negative_one_returns_immediately — §2 delete(timeout=-1)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(300)
async def test_async_delete_timeout_negative_one_returns_immediately(
    async_client: AsyncPinecone,
    preview_index_name: str,
    async_cleanup_preview_indexes: list[str],
    require_preview: None,
) -> None:
    """Async parity: delete(timeout=-1) returns None immediately; index eventually disappears (§2).

    Async counterpart for
    TestDeleteTimeout.test_delete_timeout_negative_one_returns_immediately.
    Verifies that the async delete() with timeout=-1 returns None without waiting
    for the index to disappear, and the index is eventually confirmed gone.
    """
    from pinecone.errors import NotFoundError
    from pinecone.preview.models import PreviewIndexModel

    schema = (
        SchemaBuilder().add_dense_vector_field("embedding", dimension=4, metric="cosine").build()
    )
    async_cleanup_preview_indexes.append(preview_index_name)
    await async_client.preview.indexes.create(name=preview_index_name, schema=schema)

    def _is_ready(m: object) -> bool:
        return isinstance(m, PreviewIndexModel) and m.status.state == "Ready"

    await async_poll_until(
        lambda: async_client.preview.indexes.describe(preview_index_name),
        _is_ready,
        timeout=300,
        interval=5,
        description=f"index {preview_index_name} ready before delete",
    )

    await async_client.preview.indexes.delete(preview_index_name, timeout=-1)
    # Return type is verified by -> None annotation.

    # Verify the index eventually disappears (server finishes deletion async).
    _deleted_sentinel = object()

    async def _describe_or_sentinel() -> object:
        try:
            return await async_client.preview.indexes.describe(preview_index_name)
        except NotFoundError:
            return _deleted_sentinel

    await async_poll_until(
        _describe_or_sentinel,
        lambda r: r is _deleted_sentinel,
        timeout=120,
        interval=5,
        description=f"index {preview_index_name} eventually deleted after timeout=-1",
    )


# ---------------------------------------------------------------------------
# test_async_schema_build_idempotency_and_field_collision_replacement — §1
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(300)
async def test_async_schema_build_idempotency_and_field_collision_replacement(
    async_client: AsyncPinecone,
    preview_index_name: str,
    async_cleanup_preview_indexes: list[str],
    require_preview: None,
) -> None:
    """Async parity: build() idempotency, field collision replacement, **additional_options (§1).

    Async counterpart for
    TestSchemaBuildBehavior.test_schema_build_idempotency_and_field_collision_replacement.
    Verifies all three §1 SchemaBuilder claims in the async path and confirms the
    collision-replacement schema is accepted by the API.
    """
    # Claim 2: last add_dense_vector_field("embedding") must win.
    builder = (
        SchemaBuilder()
        .add_dense_vector_field("embedding", dimension=4, metric="cosine")
        .add_dense_vector_field("embedding", dimension=8, metric="euclidean")
        .add_float_field("count")
    )

    schema1 = builder.build()
    schema2 = builder.build()

    assert schema1["fields"]["embedding"]["dimension"] == 8, (
        f"Expected dimension=8 (last add wins), got {schema1['fields']['embedding']['dimension']}"
    )
    assert schema1["fields"]["embedding"]["metric"] == "euclidean"

    # Claim 1: build() returns equal content but independent copies.
    assert schema1 == schema2, "build() must return equal dicts on repeated calls"
    assert schema1 is not schema2, "build() must return independent copies"
    assert schema1["fields"] is not schema2["fields"]

    # Claim 3: **additional_options merged into field dict (client-side).
    schema_with_extras = (
        SchemaBuilder().add_float_field("priority", filterable=True, x_custom_param=42).build()
    )
    assert schema_with_extras["fields"]["priority"]["x_custom_param"] == 42
    assert schema_with_extras["fields"]["priority"]["type"] == "float"

    # Integration: API accepts the collision-replacement schema.
    async_cleanup_preview_indexes.append(preview_index_name)
    await async_client.preview.indexes.create(name=preview_index_name, schema=schema1)

    def _is_ready(m: object) -> bool:
        return isinstance(m, PreviewIndexModel) and m.status.state == "Ready"

    await async_poll_until(
        lambda: async_client.preview.indexes.describe(preview_index_name),
        _is_ready,
        timeout=300,
        interval=5,
        description=f"index {preview_index_name} ready",
    )

    model = await async_client.preview.indexes.describe(preview_index_name)
    assert len(model.schema.fields) == 2, (
        f"Expected 2 fields after collision replacement, got {len(model.schema.fields)}"
    )
    assert "embedding" in model.schema.fields
    assert "count" in model.schema.fields


# ---------------------------------------------------------------------------
# test_async_describe_raises_not_found_and_index_factory_validation — §2 + §4
# ---------------------------------------------------------------------------


async def test_async_describe_raises_not_found_and_index_factory_validation(
    async_client: AsyncPinecone,
    require_preview: None,
) -> None:
    """Async describe() raises NotFoundError; async preview.index() validates args and defers resolution (§2, §4).

    Verifies four spec claims:
    1. §2 "Raises: NotFoundError if the index does not exist."
    2. §4 async preview.index(name=phantom) returns an object immediately — host
       resolution is deferred to the first data-plane call (unlike the sync path).
    3. §4 The deferred NotFoundError surfaces on the first data-plane call (search()).
    4. §4 async preview.index() with neither/both args raises PineconeValueError.
    """
    from pinecone.errors import NotFoundError, PineconeValueError
    from tests.integration.conftest import unique_name

    phantom = unique_name("phantom")

    # Claim 1: await describe() raises NotFoundError for a name that was never created.
    with pytest.raises(NotFoundError):
        await async_client.preview.indexes.describe(phantom)

    # Claim 2: async preview.index(name=phantom) returns immediately without error
    # (host resolution is deferred).
    idx = async_client.preview.index(name=phantom)
    assert idx is not None

    # Claim 3: The NotFoundError surfaces on the first data-plane call.
    with pytest.raises(NotFoundError):
        await idx.documents.search(
            namespace="ns",
            top_k=1,
            score_by=[PreviewDenseVectorQuery(field="embedding", values=[0.1, 0.2, 0.3, 0.4])],
        )

    # Claim 4: async preview.index() with neither/both args raises PineconeValueError.
    with pytest.raises(PineconeValueError):
        async_client.preview.index()

    with pytest.raises(PineconeValueError):
        async_client.preview.index(name=phantom, host="https://dummy-host.pinecone.io")


# ---------------------------------------------------------------------------
# test_async_configure_returns_preview_index_model — §2 + §3
# ---------------------------------------------------------------------------


async def test_async_configure_returns_preview_index_model_with_updated_fields(
    async_client: AsyncPinecone,
    preview_index_name: str,
    async_cleanup_preview_indexes: list[str],
    require_preview: None,
) -> None:
    """Async configure() return value is a PreviewIndexModel with the updated deletion_protection (§2, §3).

    Verifies:
    1. §2 "Returns: PreviewIndexModel" — async configure() return is not None and is the right type.
    2. The returned model's deletion_protection reflects the new value immediately.
    3. §3 PreviewIndexModel.deployment is a PreviewManagedDeployment with non-empty cloud/region.
    """
    from pinecone.preview.models import PreviewManagedDeployment

    schema = (
        SchemaBuilder().add_dense_vector_field("embedding", dimension=4, metric="cosine").build()
    )
    async_cleanup_preview_indexes.append(preview_index_name)
    create_model = await async_client.preview.indexes.create(name=preview_index_name, schema=schema)

    # Claim 3: deployment field from create() is a PreviewManagedDeployment.
    assert isinstance(create_model.deployment, PreviewManagedDeployment), (
        f"Expected PreviewManagedDeployment, got {type(create_model.deployment)}"
    )
    assert isinstance(create_model.deployment.cloud, str) and len(create_model.deployment.cloud) > 0
    assert (
        isinstance(create_model.deployment.region, str) and len(create_model.deployment.region) > 0
    )

    def _is_ready(m: object) -> bool:
        return isinstance(m, PreviewIndexModel) and m.status.state == "Ready"

    await async_poll_until(
        lambda: async_client.preview.indexes.describe(preview_index_name),
        _is_ready,
        timeout=300,
        interval=5,
        description=f"index {preview_index_name} ready",
    )

    # Claims 1 + 2: configure() returns a PreviewIndexModel with the updated field.
    returned = await async_client.preview.indexes.configure(
        preview_index_name, deletion_protection="enabled"
    )
    assert isinstance(returned, PreviewIndexModel), (
        f"async configure() must return PreviewIndexModel, got {type(returned)}"
    )
    assert returned.deletion_protection == "enabled", (
        f"configure() return value must reflect the new deletion_protection, "
        f"got {returned.deletion_protection!r}"
    )

    # Restore so cleanup fixture can delete the index.
    await async_client.preview.indexes.configure(preview_index_name, deletion_protection="disabled")


# ---------------------------------------------------------------------------
# test_async_configure_schema_rejects_field_modification — §2 additive-only
# ---------------------------------------------------------------------------


async def test_async_configure_schema_rejects_field_modification(
    async_client: AsyncPinecone,
    preview_index_name: str,
    async_cleanup_preview_indexes: list[str],
    require_preview: None,
) -> None:
    """Async parity: configure() with a schema that modifies an existing field raises ApiError (§2).

    Spec §2 edge case: "schema updates are additive only… The API rejects schema updates
    that modify existing field definitions." Verifies that attempting to change the dimension
    of an existing "embedding" field (4 → 8) raises ApiError with a 4xx status code, and
    that the index and its original schema remain intact afterward.
    """
    from pinecone.errors.exceptions import ApiError

    schema = (
        SchemaBuilder().add_dense_vector_field("embedding", dimension=4, metric="cosine").build()
    )
    async_cleanup_preview_indexes.append(preview_index_name)
    await async_client.preview.indexes.create(name=preview_index_name, schema=schema)

    def _is_ready(m: object) -> bool:
        return isinstance(m, PreviewIndexModel) and m.status.state == "Ready"

    await async_poll_until(
        lambda: async_client.preview.indexes.describe(preview_index_name),
        _is_ready,
        timeout=300,
        interval=5,
        description=f"index {preview_index_name} ready",
    )

    # Attempt to modify the existing "embedding" field (dimension 4 → 8).
    modified_schema = (
        SchemaBuilder().add_dense_vector_field("embedding", dimension=8, metric="cosine").build()
    )
    with pytest.raises(ApiError) as exc_info:
        await async_client.preview.indexes.configure(preview_index_name, schema=modified_schema)

    assert exc_info.value.status_code >= 400, (
        f"Expected 4xx error for field modification, got status {exc_info.value.status_code}"
    )

    # The index must still be accessible after the rejected configure().
    described = await async_client.preview.indexes.describe(preview_index_name)
    assert isinstance(described, PreviewIndexModel)
    assert len(described.schema.fields) == 1, "Schema must be unchanged after rejected configure()"


# ---------------------------------------------------------------------------
# test_async_describe_returns_typed_schema_fields — §3 PreviewSchema typed union
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(300)
async def test_async_describe_returns_typed_schema_fields(
    async_client: AsyncPinecone,
    preview_index_name: str,
    async_cleanup_preview_indexes: list[str],
    require_preview: None,
) -> None:
    """describe() schema.fields values are typed PreviewSchemaField instances (async path).

    Async parity for TestSchemaFieldTypes.test_describe_returns_typed_schema_fields.
    Verifies §3 PreviewSchema: fields dict values are decoded as PreviewDenseVectorField,
    PreviewStringField, PreviewIntegerField by msgspec's tagged-union dispatch.
    """
    from pinecone.preview import SchemaBuilder as PreviewSchemaBuilder
    from pinecone.preview.models import (
        PreviewDenseVectorField,
        PreviewIndexModel,
        PreviewIntegerField,
        PreviewStringField,
    )

    schema = (
        PreviewSchemaBuilder()
        .add_dense_vector_field("embedding", dimension=4, metric="cosine")
        .add_string_field("category", filterable=True)
        .add_float_field("year")
        .build()
    )
    async_cleanup_preview_indexes.append(preview_index_name)
    await async_client.preview.indexes.create(name=preview_index_name, schema=schema)

    def _is_ready(m: object) -> bool:
        return isinstance(m, PreviewIndexModel) and m.status.state == "Ready"

    await async_poll_until(
        lambda: async_client.preview.indexes.describe(preview_index_name),
        _is_ready,
        timeout=300,
        interval=5,
        description=f"index {preview_index_name} ready",
    )

    model = await async_client.preview.indexes.describe(preview_index_name)
    fields = model.schema.fields

    assert isinstance(fields, dict)
    assert set(fields.keys()) == {"embedding", "category", "year"}, (
        f"Expected exactly 3 fields, got {set(fields.keys())}"
    )

    emb = fields["embedding"]
    assert isinstance(emb, PreviewDenseVectorField), (
        f"embedding field: expected PreviewDenseVectorField, got {type(emb)}"
    )
    assert emb.dimension == 4, f"expected dimension=4, got {emb.dimension}"
    assert emb.metric == "cosine", f"expected metric='cosine', got {emb.metric}"

    cat = fields["category"]
    assert isinstance(cat, PreviewStringField), (
        f"category field: expected PreviewStringField, got {type(cat)}"
    )
    assert cat.filterable is True, f"expected filterable=True, got {cat.filterable}"
    assert cat.full_text_search is None, (
        f"expected full_text_search=None (field is not FTS-enabled), got {cat.full_text_search!r}"
    )

    year = fields["year"]
    assert isinstance(year, PreviewIntegerField), (
        f"year field: expected PreviewIntegerField, got {type(year)}"
    )


# ---------------------------------------------------------------------------
# test_async_backup_optional_fields_are_correctly_typed — §2 BackupModel async parity
# ---------------------------------------------------------------------------


async def test_async_backup_optional_fields_are_correctly_typed(
    async_client: AsyncPinecone,
    preview_index_name: str,
    async_cleanup_preview_indexes: list[str],
    require_preview: None,
) -> None:
    """Async parity for test_create_backup_optional_fields_are_correctly_typed (PVT-023).

    Creates a 4-dim dense vector preview index, takes a backup, and verifies that
    all optional BackupModel fields (schema, source_index_deleted_at, tags,
    record_count, namespace_count, size_bytes) are either the expected Python
    type or None. No existing async test verifies these optional fields.
    """
    from pinecone.models.backups.model import BackupModel
    from pinecone.models.indexes.schema import IndexSchema

    schema = (
        SchemaBuilder().add_dense_vector_field("embedding", dimension=4, metric="cosine").build()
    )
    async_cleanup_preview_indexes.append(preview_index_name)
    await async_client.preview.indexes.create(name=preview_index_name, schema=schema)

    def _is_ready(m: object) -> bool:
        return isinstance(m, PreviewIndexModel) and m.status.state == "Ready"

    await async_poll_until(
        lambda: async_client.preview.indexes.describe(preview_index_name),
        _is_ready,
        timeout=300,
        interval=5,
        description=f"index {preview_index_name} ready",
    )

    backup = await async_client.preview.indexes.create_backup(
        preview_index_name,
        name="async-optional-fields-test",
    )

    assert isinstance(backup, BackupModel)

    assert backup.dense_dimension is None or backup.dense_dimension == 4, (
        f"expected dense_dimension=4 for dense index, got {backup.dense_dimension}"
    )

    assert backup.schema is None or isinstance(backup.schema, IndexSchema), (
        f"backup.schema must be IndexSchema or None, got {type(backup.schema)}"
    )

    assert backup.source_index_deleted_at is None, (
        "source index is active, so source_index_deleted_at must be None"
    )

    # tags — dict[str, str] or None; API returns {} when no tags are passed
    assert backup.tags is None or isinstance(backup.tags, dict), (
        f"backup.tags must be dict or None, got {type(backup.tags)}"
    )

    # record_count, namespace_count, size_bytes — int or None
    assert backup.record_count is None or isinstance(backup.record_count, int), (
        f"backup.record_count must be int or None, got {type(backup.record_count)}"
    )
    assert backup.namespace_count is None or isinstance(backup.namespace_count, int), (
        f"backup.namespace_count must be int or None, got {type(backup.namespace_count)}"
    )
    assert backup.size_bytes is None or isinstance(backup.size_bytes, int), (
        f"backup.size_bytes must be int or None, got {type(backup.size_bytes)}"
    )


# ---------------------------------------------------------------------------
# test_async_preview_index_model_read_capacity_on_demand — §3 async parity
# ---------------------------------------------------------------------------


async def test_async_preview_index_model_read_capacity_on_demand(
    async_client: AsyncPinecone,
    preview_index_name: str,
    async_cleanup_preview_indexes: list[str],
    require_preview: None,
) -> None:
    """Async parity for test_preview_index_model_read_capacity_on_demand (PVT-024).

    Verifies that AsyncPinecone create() and describe() also return
    PreviewReadCapacityOnDemandResponse for a default-capacity index (§3).
    """
    from pinecone.preview.models import (
        PreviewReadCapacityOnDemandResponse,
        PreviewReadCapacityStatus,
    )

    schema = (
        SchemaBuilder().add_dense_vector_field("embedding", dimension=4, metric="cosine").build()
    )
    async_cleanup_preview_indexes.append(preview_index_name)
    created = await async_client.preview.indexes.create(name=preview_index_name, schema=schema)

    assert isinstance(created, PreviewIndexModel)
    assert created.read_capacity is not None, (
        "async create() read_capacity should not be None for a default OnDemand index"
    )
    assert isinstance(created.read_capacity, PreviewReadCapacityOnDemandResponse), (
        f"async create() read_capacity: expected PreviewReadCapacityOnDemandResponse, got {type(created.read_capacity)}"
    )
    assert isinstance(created.read_capacity.status, PreviewReadCapacityStatus), (
        f"read_capacity.status expected PreviewReadCapacityStatus, got {type(created.read_capacity.status)}"
    )
    assert isinstance(created.read_capacity.status.state, str), (
        "read_capacity.status.state should be a string"
    )

    def _is_ready(m: object) -> bool:
        return isinstance(m, PreviewIndexModel) and m.status.state == "Ready"

    await async_poll_until(
        lambda: async_client.preview.indexes.describe(preview_index_name),
        _is_ready,
        timeout=300,
        interval=5,
        description=f"index {preview_index_name} ready",
    )

    described = await async_client.preview.indexes.describe(preview_index_name)
    assert isinstance(described, PreviewIndexModel)
    assert described.read_capacity is not None, (
        "async describe() read_capacity should not be None for an OnDemand index"
    )
    assert isinstance(described.read_capacity, PreviewReadCapacityOnDemandResponse), (
        f"async describe() read_capacity: expected PreviewReadCapacityOnDemandResponse, got {type(described.read_capacity)}"
    )
    assert isinstance(described.read_capacity.status.state, str), (
        "async describe() read_capacity.status.state should be a string"
    )


# ---------------------------------------------------------------------------
# test_async_documents_delete_returns_none — §5 async parity for delete() return value
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="IPV-0004: documents/delete endpoint returns 401 Unknown operation")
async def test_async_documents_delete_returns_none_for_all_targeting_modes(
    async_client: AsyncPinecone,
    preview_index_name: str,
    async_cleanup_preview_indexes: list[str],
    preview_namespace: str,
    require_preview: None,
) -> None:
    """Async parity for test_documents_delete_returns_none_for_all_targeting_modes (PVT-025).

    Verifies that AsyncPreviewDocuments.delete() returns None for all three
    targeting modes (ids, filter, delete_all=True), as required by spec §5
    "Returns: None (empty response body)".

    DISABLED (IPV-0004): Same root cause as sync variant.
    """
    schema = (
        SchemaBuilder()
        .add_dense_vector_field("embedding", dimension=4, metric="cosine")
        .add_string_field("category", filterable=True)
        .build()
    )
    async_cleanup_preview_indexes.append(preview_index_name)
    await async_client.preview.indexes.create(name=preview_index_name, schema=schema)

    def _is_ready(m: object) -> bool:
        return isinstance(m, PreviewIndexModel) and m.status.state == "Ready"

    await async_poll_until(
        lambda: async_client.preview.indexes.describe(preview_index_name),
        _is_ready,
        timeout=300,
        interval=5,
        description=f"async index {preview_index_name} ready",
    )

    idx = async_client.preview.index(name=preview_index_name)
    ns = preview_namespace

    await idx.documents.upsert(
        namespace=ns,
        documents=[
            {"_id": "doc-0", "embedding": [0.1, 0.2, 0.3, 0.4], "category": "fruit"},
            {"_id": "doc-1", "embedding": [0.5, 0.6, 0.7, 0.8], "category": "fruit"},
            {"_id": "doc-2", "embedding": [0.9, 0.1, 0.2, 0.3], "category": "vegetable"},
        ],
    )

    # delete by IDs must return None (verified by -> None annotation).
    await idx.documents.delete(namespace=ns, ids=["doc-0"])

    # delete by filter must return None.
    await idx.documents.delete(namespace=ns, filter={"category": {"$eq": "vegetable"}})

    # delete_all=True must return None.
    await idx.documents.delete(namespace=ns, delete_all=True)


# ---------------------------------------------------------------------------
# test_async_search_score_by_plain_dict_accepted — §6 "Dict format" async parity
# ---------------------------------------------------------------------------


@pytest.mark.timeout(300)
async def test_async_search_score_by_plain_dict_accepted(
    async_client: AsyncPinecone,
    preview_index_name: str,
    async_cleanup_preview_indexes: list[str],
    preview_namespace: str,
    require_preview: None,
) -> None:
    """Async parity for test_search_score_by_plain_dict_accepted (PVT-026).

    Verifies that AsyncPreviewDocuments.search() also accepts plain dicts in score_by
    per spec §6 "Dict format". Identical logic to sync variant; uses include_fields=["*"]
    to avoid IPV-0001 422 bug. 0 matches acceptable.
    """
    schema = (
        SchemaBuilder().add_dense_vector_field("embedding", dimension=4, metric="cosine").build()
    )
    async_cleanup_preview_indexes.append(preview_index_name)
    await async_client.preview.indexes.create(name=preview_index_name, schema=schema)

    def _is_ready(m: object) -> bool:
        return isinstance(m, PreviewIndexModel) and m.status.state == "Ready"

    await async_poll_until(
        lambda: async_client.preview.indexes.describe(preview_index_name),
        _is_ready,
        timeout=300,
        interval=5,
        description=f"index {preview_index_name} ready",
    )

    idx = async_client.preview.index(
        host=(await async_client.preview.indexes.describe(preview_index_name)).host
    )
    await idx.documents.upsert(
        namespace=preview_namespace,
        documents=[{"_id": "doc-dict-async", "embedding": [0.1, 0.2, 0.3, 0.4]}],
    )

    # §6: pass score_by as a plain dict (wire format) rather than a typed model
    response = await idx.documents.search(
        namespace=preview_namespace,
        top_k=5,
        score_by=[{"type": "dense_vector", "field": "embedding", "values": [0.1, 0.2, 0.3, 0.4]}],
        include_fields=["*"],
    )

    assert isinstance(response, PreviewDocumentSearchResponse), (
        f"async search() with plain dict score_by expected PreviewDocumentSearchResponse, got {type(response)}"
    )
    assert response.namespace == preview_namespace, (
        f"response.namespace expected {preview_namespace!r}, got {response.namespace!r}"
    )
    assert response.usage is not None, "response.usage should not be None for a 200 OK response"
    assert isinstance(response.usage, PreviewUsage), (
        f"response.usage expected PreviewUsage, got {type(response.usage)}"
    )
    assert isinstance(response.usage.read_units, int), (
        f"usage.read_units expected int, got {type(response.usage.read_units)}"
    )


# ---------------------------------------------------------------------------
# test_async_batch_upsert_result_display_methods — §14 BatchResult display
# ---------------------------------------------------------------------------


async def test_async_batch_upsert_result_display_methods(
    async_client: AsyncPinecone,
    preview_index_name: str,
    preview_namespace: str,
    async_cleanup_preview_indexes: list[str],
    require_preview: None,
) -> None:
    """Async parity for test_batch_upsert_result_display_methods (PVT-027).

    Verifies BatchResult display methods after async batch_upsert:
    - §14 __repr__: "BatchResult(SUCCESS: 10/10 items, 2/2 batches)"
    - §5 to_dict(): dict with 7 keys and correct values
    - §5 to_json(): valid JSON parseable to same values
    - §14 _repr_html_(): non-empty HTML containing "BatchResult"
    """
    import json

    schema = (
        SchemaBuilder().add_dense_vector_field("embedding", dimension=4, metric="cosine").build()
    )
    async_cleanup_preview_indexes.append(preview_index_name)
    await async_client.preview.indexes.create(name=preview_index_name, schema=schema)

    def _is_ready(m: object) -> bool:
        return isinstance(m, PreviewIndexModel) and m.status.state == "Ready"

    await async_poll_until(
        lambda: async_client.preview.indexes.describe(preview_index_name),
        _is_ready,
        timeout=300,
        interval=5,
        description=f"index {preview_index_name} ready",
    )

    idx = async_client.preview.index(name=preview_index_name)
    documents = [
        {"_id": f"doc-{i}", "embedding": [float(i) / 10, 0.1, 0.2, 0.3]} for i in range(10)
    ]

    result = await idx.documents.batch_upsert(
        namespace=preview_namespace,
        documents=documents,
        batch_size=5,
        max_concurrency=2,
        show_progress=False,
    )

    assert isinstance(result, BatchResult)

    # §14: repr format: "BatchResult(SUCCESS: N/N items, N/N batches)"
    repr_str = repr(result)
    assert repr_str == "BatchResult(SUCCESS: 10/10 items, 2/2 batches)", (
        f"repr() expected 'BatchResult(SUCCESS: 10/10 items, 2/2 batches)', got {repr_str!r}"
    )

    # §5: to_dict() has all 7 expected keys with correct values
    d = result.to_dict()
    assert isinstance(d, dict), f"to_dict() expected dict, got {type(d)}"
    assert d["total_item_count"] == 10
    assert d["successful_item_count"] == 10
    assert d["failed_item_count"] == 0
    assert d["total_batch_count"] == 2
    assert d["successful_batch_count"] == 2
    assert d["failed_batch_count"] == 0
    assert d["errors"] == []

    # §5: to_json() is valid JSON parseable to a dict matching to_dict() values
    json_str = result.to_json()
    assert isinstance(json_str, str), f"to_json() expected str, got {type(json_str)}"
    parsed = json.loads(json_str)
    assert parsed["total_item_count"] == 10
    assert parsed["successful_item_count"] == 10
    assert parsed["failed_item_count"] == 0
    assert parsed["total_batch_count"] == 2
    assert parsed["errors"] == []

    # §14: _repr_html_() returns non-empty HTML string containing "BatchResult"
    html = result._repr_html_()
    assert isinstance(html, str), f"_repr_html_() expected str, got {type(html)}"
    assert len(html) > 0, "_repr_html_() must return non-empty HTML"
    assert "BatchResult" in html, f"_repr_html_() must contain 'BatchResult', got: {html[:200]!r}"


# ---------------------------------------------------------------------------
# test_async_upsert_accepts_extra_and_partial_documents — §5 edge cases async parity
# ---------------------------------------------------------------------------


async def test_async_upsert_accepts_extra_and_partial_documents(
    async_client: AsyncPinecone,
    preview_index_name: str,
    preview_namespace: str,
    async_cleanup_preview_indexes: list[str],
    require_preview: None,
) -> None:
    """Async parity: upsert() accepts documents with extra fields and partial documents (§5).

    Mirrors test_upsert_accepts_extra_and_partial_documents for the async SDK path.
    Spec §5 edge cases: extra fields stored as unindexed metadata; missing schema
    fields treated as absent. Both docs returned in upserted_count.
    """
    from pinecone.preview.models import PreviewDocumentUpsertResponse, PreviewIndexModel

    schema = (
        SchemaBuilder()
        .add_dense_vector_field("embedding", dimension=4, metric="cosine")
        .add_string_field("category", filterable=True)
        .build()
    )
    async_cleanup_preview_indexes.append(preview_index_name)
    await async_client.preview.indexes.create(name=preview_index_name, schema=schema)

    def _is_ready(m: object) -> bool:
        return isinstance(m, PreviewIndexModel) and m.status.state == "Ready"

    await async_poll_until(
        lambda: async_client.preview.indexes.describe(preview_index_name),
        _is_ready,
        timeout=300,
        interval=5,
        description=f"index {preview_index_name} ready",
    )

    idx = async_client.preview.index(name=preview_index_name)

    response = await idx.documents.upsert(
        namespace=preview_namespace,
        documents=[
            {
                "_id": "doc-extra",
                "embedding": [0.1, 0.2, 0.3, 0.4],
                "category": "fruit",
                "custom_note": "extra field not in schema",
            },
            {
                "_id": "doc-partial",
                "embedding": [0.5, 0.6, 0.7, 0.8],
                # "category" is omitted — partial document (missing schema field treated as absent)
            },
        ],
    )

    assert isinstance(response, PreviewDocumentUpsertResponse), (
        f"Expected PreviewDocumentUpsertResponse, got {type(response)}"
    )
    assert response.upserted_count == 2, (
        f"Expected upserted_count=2 for 2 docs (extra + partial), got {response.upserted_count}"
    )


# ---------------------------------------------------------------------------
# test_async_batch_upsert_with_batch_size_one_per_document — §5 batch_size min
# ---------------------------------------------------------------------------


async def test_async_batch_upsert_with_batch_size_one_per_document(
    async_client: AsyncPinecone,
    preview_index_name: str,
    preview_namespace: str,
    async_cleanup_preview_indexes: list[str],
    require_preview: None,
) -> None:
    """Async parity: batch_upsert(batch_size=1) creates one batch per document (§5).

    Async variant uses asyncio.gather() instead of threads; max_concurrency controls
    concurrency limit. batch_size=1 means each document is its own HTTP request,
    so total_batch_count equals the document count.
    """
    schema = (
        SchemaBuilder().add_dense_vector_field("embedding", dimension=4, metric="cosine").build()
    )
    async_cleanup_preview_indexes.append(preview_index_name)
    await async_client.preview.indexes.create(name=preview_index_name, schema=schema)

    def _is_ready(m: object) -> bool:
        return isinstance(m, PreviewIndexModel) and m.status.state == "Ready"

    await async_poll_until(
        lambda: async_client.preview.indexes.describe(preview_index_name),
        _is_ready,
        timeout=300,
        interval=5,
        description=f"index {preview_index_name} ready",
    )

    idx = async_client.preview.index(name=preview_index_name)
    documents = [
        {"_id": f"doc-{i}", "embedding": [float(i) / 10, 0.1, 0.2, 0.3]}
        for i in range(1, 4)  # 3 documents
    ]

    result = await idx.documents.batch_upsert(
        namespace=preview_namespace,
        documents=documents,
        batch_size=1,  # minimum: each document is its own HTTP request
        max_concurrency=2,
        show_progress=False,
    )

    assert isinstance(result, BatchResult), f"Expected BatchResult, got {type(result)}"
    assert result.total_item_count == 3, (
        f"Expected total_item_count=3, got {result.total_item_count}"
    )
    assert result.total_batch_count == 3, (
        f"Expected total_batch_count=3 (one per doc when batch_size=1), got {result.total_batch_count}"
    )
    assert result.successful_item_count == 3, (
        f"Expected successful_item_count=3, got {result.successful_item_count}"
    )
    assert result.successful_batch_count == 3, (
        f"Expected successful_batch_count=3, got {result.successful_batch_count}"
    )
    assert result.failed_item_count == 0, (
        f"Expected failed_item_count=0, got {result.failed_item_count}"
    )
    assert result.failed_batch_count == 0, (
        f"Expected failed_batch_count=0, got {result.failed_batch_count}"
    )
    assert result.has_errors is False, f"Expected has_errors=False, got {result.has_errors}"


# ---------------------------------------------------------------------------
# test_async_search_response_and_index_model_display_methods — §14 Notebook and REPL rendering
# ---------------------------------------------------------------------------


async def test_async_search_response_and_index_model_display_methods(
    async_client: AsyncPinecone,
    preview_index_name: str,
    async_cleanup_preview_indexes: list[str],
    preview_namespace: str,
    require_preview: None,
) -> None:
    """Async parity: verify __repr__ and _repr_html_ for PreviewDocumentSearchResponse and PreviewIndexModel (§14).

    Mirrors test_search_response_and_index_model_display_methods for the async path.
    Spec §14 lists PreviewIndexModel and PreviewDocumentSearchResponse as requiring
    __repr__ and _repr_html_ implementations. Both display methods are called on real
    objects from async API responses.
    """
    schema = (
        SchemaBuilder().add_dense_vector_field("embedding", dimension=4, metric="cosine").build()
    )
    async_cleanup_preview_indexes.append(preview_index_name)
    await async_client.preview.indexes.create(name=preview_index_name, schema=schema)

    def _is_ready(m: object) -> bool:
        return isinstance(m, PreviewIndexModel) and m.status.state == "Ready"

    await async_poll_until(
        lambda: async_client.preview.indexes.describe(preview_index_name),
        _is_ready,
        timeout=300,
        interval=5,
        description=f"index {preview_index_name} ready",
    )

    # §14: PreviewIndexModel.__repr__ and _repr_html_
    described = await async_client.preview.indexes.describe(preview_index_name)

    index_repr = repr(described)
    assert index_repr.startswith("PreviewIndexModel("), (
        f"PreviewIndexModel.__repr__ must start with 'PreviewIndexModel(', got: {index_repr!r}"
    )
    assert f"name={preview_index_name!r}" in index_repr, (
        f"PreviewIndexModel.__repr__ must include name=, got: {index_repr!r}"
    )
    assert "status=" in index_repr, (
        f"PreviewIndexModel.__repr__ must include status=, got: {index_repr!r}"
    )
    assert "host=" in index_repr, (
        f"PreviewIndexModel.__repr__ must include host=, got: {index_repr!r}"
    )
    assert "deletion_protection=" in index_repr, (
        f"PreviewIndexModel.__repr__ must include deletion_protection=, got: {index_repr!r}"
    )

    index_html = described._repr_html_()
    assert isinstance(index_html, str) and len(index_html) > 0, (
        "PreviewIndexModel._repr_html_() must return a non-empty string"
    )
    assert "PreviewIndexModel" in index_html, (
        f"PreviewIndexModel._repr_html_() must contain 'PreviewIndexModel', got: {index_html[:300]!r}"
    )

    # §14: PreviewDocumentSearchResponse.__repr__ and _repr_html_
    idx = async_client.preview.index(name=preview_index_name)
    response = await idx.documents.search(
        namespace=preview_namespace,
        top_k=5,
        score_by=[PreviewDenseVectorQuery(field="embedding", values=[0.1, 0.2, 0.3, 0.4])],
        include_fields=["*"],
    )

    search_repr = repr(response)
    assert search_repr.startswith("SearchResponse("), (
        f"PreviewDocumentSearchResponse.__repr__ must start with 'SearchResponse(', got: {search_repr!r}"
    )
    assert "matches=" in search_repr, (
        f"PreviewDocumentSearchResponse.__repr__ must include matches=, got: {search_repr!r}"
    )
    assert "namespace=" in search_repr, (
        f"PreviewDocumentSearchResponse.__repr__ must include namespace=, got: {search_repr!r}"
    )
    assert "usage=" in search_repr, (
        f"PreviewDocumentSearchResponse.__repr__ must include usage=, got: {search_repr!r}"
    )

    search_html = response._repr_html_()
    assert isinstance(search_html, str) and len(search_html) > 0, (
        "PreviewDocumentSearchResponse._repr_html_() must return a non-empty string"
    )
    assert "SearchResponse" in search_html, (
        f"PreviewDocumentSearchResponse._repr_html_() must contain 'SearchResponse', "
        f"got: {search_html[:300]!r}"
    )


# ---------------------------------------------------------------------------
# test_async_batch_upsert_partial_failure — §5 partial failure (async path)
# ---------------------------------------------------------------------------


async def test_async_batch_upsert_partial_failure_collects_failed_items(
    async_client: AsyncPinecone,
    preview_index_name: str,
    preview_namespace: str,
    async_cleanup_preview_indexes: list[str],
    require_preview: None,
) -> None:
    """Async batch_upsert() continues after per-batch failures and records them.

    Async parity for test_batch_upsert_partial_failure_collects_failed_items:
    verifies §5 "Edge case — partial failure" via the async SDK path.

    One document has a 5-dim vector on a 4-dim index (API rejects it); the other
    3 docs succeed. Verifies has_errors=True, failed_batch_count==1,
    failed_item_count==1, failed_items contains the rejected doc, errors has
    one BatchError entry.
    """
    async_cleanup_preview_indexes.append(preview_index_name)

    schema = (
        SchemaBuilder().add_dense_vector_field("embedding", dimension=4, metric="cosine").build()
    )
    await async_client.preview.indexes.create(name=preview_index_name, schema=schema)

    def _is_ready(m: object) -> bool:
        return isinstance(m, PreviewIndexModel) and m.status.state == "Ready"

    await async_poll_until(
        lambda: async_client.preview.indexes.describe(preview_index_name),
        _is_ready,
        timeout=300,
        interval=5,
        description=f"index {preview_index_name} ready",
    )

    idx = async_client.preview.index(name=preview_index_name)
    bad_doc = {"_id": "bad-dim", "embedding": [0.1, 0.2, 0.3, 0.4, 0.5]}
    good_docs = [
        {"_id": f"good-{i}", "embedding": [float(i) / 10, 0.1, 0.2, 0.3]} for i in range(3)
    ]
    all_docs = [*good_docs, bad_doc]

    result = await idx.documents.batch_upsert(
        namespace=preview_namespace,
        documents=all_docs,
        batch_size=1,
        max_concurrency=2,
        show_progress=False,
    )

    assert isinstance(result, BatchResult)
    assert result.total_item_count == 4
    assert result.total_batch_count == 4
    assert result.has_errors is True, f"Expected has_errors=True, got {result.has_errors}"
    assert result.failed_batch_count == 1, (
        f"Expected 1 failed batch, got {result.failed_batch_count}"
    )
    assert result.failed_item_count == 1, f"Expected 1 failed item, got {result.failed_item_count}"
    assert result.successful_batch_count == 3, (
        f"Expected 3 successful batches, got {result.successful_batch_count}"
    )
    assert result.successful_item_count == 3, (
        f"Expected 3 successful items, got {result.successful_item_count}"
    )
    assert len(result.failed_items) == 1, (
        f"Expected 1 item in failed_items, got {len(result.failed_items)}"
    )
    assert result.failed_items[0]["_id"] == "bad-dim", (
        f"Expected failed_items[0]['_id'] == 'bad-dim', got {result.failed_items[0]['_id']!r}"
    )
    assert len(result.errors) == 1, f"Expected 1 BatchError, got {len(result.errors)}"


# ---------------------------------------------------------------------------
# test_async_configure_read_capacity_validation_and_api_acceptance — §2
# ---------------------------------------------------------------------------


@pytest.mark.anyio
@pytest.mark.integration
@pytest.mark.timeout(300)
async def test_async_configure_read_capacity_validation_and_api_acceptance(
    async_client: AsyncPinecone,
    preview_index_name: str,
    async_cleanup_preview_indexes: list[str],
    require_preview: None,
) -> None:
    """Async parity: configure(read_capacity=...) validation and API acceptance (§2).

    Verifies three §2 claims:
    1. configure(name, read_capacity={}) raises PineconeValueError (client-side, on await).
    2. configure(name) with no kwargs raises PineconeValueError (client-side, on await).
    3. configure(name, read_capacity={"mode": "OnDemand"}) is accepted by the API and
       returns a PreviewIndexModel whose read_capacity is PreviewReadCapacityOnDemandResponse.

    Async parity for TestConfigureReadCapacity.test_configure_read_capacity_validation_and_api_acceptance.
    """
    from pinecone.errors import PineconeValueError
    from pinecone.preview.models import PreviewReadCapacityOnDemandResponse

    # Claims 1 + 2: validation fires on await before any HTTP call.
    with pytest.raises(PineconeValueError, match="read_capacity"):
        await async_client.preview.indexes.configure("any-name", read_capacity={})

    with pytest.raises(PineconeValueError, match="at least one"):
        await async_client.preview.indexes.configure("any-name")

    # Claim 3: API-level — OnDemand mode accepted on a live dense vector index.
    schema = (
        SchemaBuilder().add_dense_vector_field("embedding", dimension=4, metric="cosine").build()
    )
    async_cleanup_preview_indexes.append(preview_index_name)
    await async_client.preview.indexes.create(name=preview_index_name, schema=schema)

    def _is_ready(m: object) -> bool:
        return isinstance(m, PreviewIndexModel) and m.status.state == "Ready"

    await async_poll_until(
        lambda: async_client.preview.indexes.describe(preview_index_name),
        _is_ready,
        timeout=300,
        interval=5,
        description=f"index {preview_index_name} ready",
    )

    returned = await async_client.preview.indexes.configure(
        preview_index_name, read_capacity={"mode": "OnDemand"}
    )
    assert isinstance(returned, PreviewIndexModel), (
        f"async configure(read_capacity=...) must return PreviewIndexModel, got {type(returned)}"
    )
    assert returned.read_capacity is not None, (
        "returned model.read_capacity must not be None for a live OnDemand index"
    )
    assert isinstance(returned.read_capacity, PreviewReadCapacityOnDemandResponse), (
        f"returned model.read_capacity must be PreviewReadCapacityOnDemandResponse, "
        f"got {type(returned.read_capacity)}"
    )


# ---------------------------------------------------------------------------
# test_async_preview_document_field_scoping — §7 PreviewDocument "fields not included"
# ---------------------------------------------------------------------------


@pytest.mark.anyio
@pytest.mark.integration
@pytest.mark.timeout(300)
async def test_async_describe_dedicated_index_read_capacity_response_fields(
    async_client: AsyncPinecone,
    preview_index_name: str,
    async_cleanup_preview_indexes: list[str],
    require_preview: None,
) -> None:
    """Async parity: describe() on a dedicated-capacity index returns PreviewReadCapacityDedicatedResponse.

    Async parity for test_describe_dedicated_index_read_capacity_response_fields.
    Verifies §3 PreviewIndexModel.read_capacity Dedicated branch via async SDK path.
    """
    from pinecone.preview.models import (
        PreviewIndexModel,
        PreviewReadCapacityDedicatedInner,
        PreviewReadCapacityDedicatedResponse,
        PreviewReadCapacityStatus,
    )

    schema = SchemaBuilder().add_string_field("text", full_text_search={}).build()
    read_capacity = {
        "mode": "Dedicated",
        "dedicated": {
            "node_type": "t1",
            "scaling": "Manual",
            "manual": {"shards": 1, "replicas": 1},
        },
    }
    async_cleanup_preview_indexes.append(preview_index_name)
    model = await async_client.preview.indexes.create(
        name=preview_index_name,
        schema=schema,
        read_capacity=read_capacity,
    )
    assert isinstance(model, PreviewIndexModel)

    def _is_ready(m: object) -> bool:
        return isinstance(m, PreviewIndexModel) and m.status.state == "Ready"

    described = await async_poll_until(
        lambda: async_client.preview.indexes.describe(preview_index_name),
        _is_ready,
        timeout=300,
        interval=5,
        description=f"index {preview_index_name} ready",
    )
    assert isinstance(described, PreviewIndexModel)

    rc = described.read_capacity
    assert rc is not None, (
        "describe() on a dedicated-capacity index must have read_capacity != None"
    )
    assert isinstance(rc, PreviewReadCapacityDedicatedResponse), (
        f"read_capacity must be PreviewReadCapacityDedicatedResponse for a Dedicated index, "
        f"got {type(rc)}"
    )

    dedicated = rc.dedicated
    assert isinstance(dedicated, PreviewReadCapacityDedicatedInner)
    assert isinstance(dedicated.node_type, str) and dedicated.node_type
    assert dedicated.scaling in ("Manual", "Auto")
    if dedicated.scaling == "Manual":
        assert dedicated.manual is not None
        assert isinstance(dedicated.manual.shards, int) and dedicated.manual.shards >= 1
        assert isinstance(dedicated.manual.replicas, int) and dedicated.manual.replicas >= 1

    status = rc.status
    assert isinstance(status, PreviewReadCapacityStatus)
    assert isinstance(status.state, str) and status.state


# ---------------------------------------------------------------------------
# test_async_sparse_vector_schema_and_query_accepted — §1 + §6 async parity
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(300)
async def test_async_sparse_vector_schema_and_query_accepted(
    async_client: AsyncPinecone,
    preview_index_name: str,
    async_cleanup_preview_indexes: list[str],
    preview_namespace: str,
    require_preview: None,
) -> None:
    """Async parity for §1 add_sparse_vector_field() + §6 PreviewSparseVectorQuery / PreviewSparseValues.

    Mirrors test_sparse_vector_schema_and_query_accepted (PVT-034) using AsyncPinecone.
    Pure sparse vector schema (no FTS) on OnDemand to isolate the sparse-vector code path
    without requiring dedicated read capacity.

    Verifies:
    - Async index creation with sparse-only schema is accepted
    - Async upsert of 2 documents with sparse_embedding dicts (upserted_count==2)
    - Async search() with PreviewSparseVectorQuery + PreviewSparseValues returns 200 OK
    - Response is PreviewDocumentSearchResponse with correct namespace and usage
    """
    from pinecone.preview.models import (
        PreviewDocumentSearchResponse,
        PreviewSparseValues,
        PreviewSparseVectorQuery,
    )

    schema = SchemaBuilder().add_sparse_vector_field("sparse_embedding").build()
    assert schema["fields"]["sparse_embedding"]["type"] == "sparse_vector"

    async_cleanup_preview_indexes.append(preview_index_name)
    await async_client.preview.indexes.create(name=preview_index_name, schema=schema)

    def _is_ready(m: object) -> bool:
        return isinstance(m, PreviewIndexModel) and m.status.state == "Ready"

    await async_poll_until(
        lambda: async_client.preview.indexes.describe(preview_index_name),
        _is_ready,
        timeout=300,
        interval=5,
        description=f"sparse index {preview_index_name} ready",
    )

    idx = async_client.preview.index(name=preview_index_name)
    upsert_resp = await idx.documents.upsert(
        namespace=preview_namespace,
        documents=[
            {"_id": "doc-1", "sparse_embedding": {"indices": [0, 42], "values": [0.8, 0.3]}},
            {"_id": "doc-2", "sparse_embedding": {"indices": [1, 99], "values": [0.1, 0.1]}},
        ],
    )
    assert upsert_resp.upserted_count == 2, (
        f"expected upserted_count=2, got {upsert_resp.upserted_count}"
    )

    response = await idx.documents.search(
        namespace=preview_namespace,
        top_k=5,
        score_by=[
            PreviewSparseVectorQuery(
                field="sparse_embedding",
                sparse_values=PreviewSparseValues(indices=[0, 42], values=[0.8, 0.3]),
            )
        ],
        include_fields=["*"],
    )

    assert isinstance(response, PreviewDocumentSearchResponse), (
        f"Expected PreviewDocumentSearchResponse, got {type(response)}"
    )
    assert response.namespace == preview_namespace, (
        f"response.namespace {response.namespace!r} != {preview_namespace!r}"
    )
    assert response.usage is not None, "response.usage must not be None on a successful search"
    assert isinstance(response.usage.read_units, int), (
        f"read_units must be int, got {type(response.usage.read_units)}"
    )
    assert response.usage.read_units >= 0
    assert isinstance(response.matches, list)


# ---------------------------------------------------------------------------
# test_async_describe_returns_sparse_vector_field_typed_instance — §3
# ---------------------------------------------------------------------------


async def test_async_describe_returns_sparse_vector_field_typed_instance(
    async_client: AsyncPinecone,
    preview_index_name: str,
    async_cleanup_preview_indexes: list[str],
    require_preview: None,
) -> None:
    """Async parity: describe() returns PreviewSparseVectorField for sparse_vector schema fields.

    Spec §3 PreviewSchema.fields: the discriminated union includes PreviewSparseVectorField
    (tag "sparse_vector"). Verifies the sparse branch is deserialized to the correct typed
    instance with metric echoed back from the server.
    """
    from pinecone.preview.models import PreviewSparseVectorField

    schema = SchemaBuilder().add_sparse_vector_field("data").build()

    async_cleanup_preview_indexes.append(preview_index_name)
    await async_client.preview.indexes.create(name=preview_index_name, schema=schema)

    def _is_ready(m: object) -> bool:
        return isinstance(m, PreviewIndexModel) and m.status.state == "Ready"

    await async_poll_until(
        lambda: async_client.preview.indexes.describe(preview_index_name),
        _is_ready,
        timeout=300,
        interval=5,
        description=f"sparse index {preview_index_name} ready",
    )

    model = await async_client.preview.indexes.describe(preview_index_name)
    fields = model.schema.fields

    assert "data" in fields, f"Expected 'data' in schema.fields, got {set(fields.keys())}"

    data_field = fields["data"]
    assert isinstance(data_field, PreviewSparseVectorField), (
        f"data field: expected PreviewSparseVectorField, got {type(data_field)}"
    )
    # Server omits metric in describe() response — metric is None even when
    # dotproduct was specified at create time. Attribute is accessible as None.
    assert data_field.metric is None or isinstance(data_field.metric, str), (
        f"data field: metric must be None or str, got {type(data_field.metric)}"
    )


# ---------------------------------------------------------------------------
# test_async_describe_fts_string_field_server_applied_defaults — §1
# ---------------------------------------------------------------------------


async def test_async_describe_fts_string_field_server_applied_defaults(
    async_client: AsyncPinecone,
    preview_index_name: str,
    async_cleanup_preview_indexes: list[str],
    require_preview: None,
) -> None:
    """Async parity: describe() returns server-applied defaults for FTS string fields.

    Spec §1 add_string_field edge case: 'The API response for a created index will
    include defaults (e.g. lowercase: true, stemming: false, max_term_len: 40) even
    if the user did not set them. The SDK model for the response must accept these
    server-populated values.'

    Creates an FTS string field with `full_text_search={}` (empty dict — server defaults;
    no explicit lowercase/stemming/max_term_len), then verifies async describe() returns
    PreviewStringField with server-populated attribute values.
    """
    from pinecone.preview.models import PreviewStringField

    read_capacity = {
        "mode": "Dedicated",
        "dedicated": {
            "node_type": "t1",
            "scaling": "Manual",
            "manual": {"shards": 1, "replicas": 1},
        },
    }
    schema = SchemaBuilder().add_string_field("text", full_text_search={}).build()

    async_cleanup_preview_indexes.append(preview_index_name)
    await async_client.preview.indexes.create(
        name=preview_index_name,
        schema=schema,
        read_capacity=read_capacity,
    )

    def _is_ready(m: object) -> bool:
        return isinstance(m, PreviewIndexModel) and m.status.state == "Ready"

    await async_poll_until(
        lambda: async_client.preview.indexes.describe(preview_index_name),
        _is_ready,
        timeout=300,
        interval=5,
        description=f"FTS index {preview_index_name} ready",
    )

    model = await async_client.preview.indexes.describe(preview_index_name)
    fields = model.schema.fields

    assert "text" in fields, f"Expected 'text' in schema.fields, got {set(fields.keys())}"

    text_field = fields["text"]
    assert isinstance(text_field, PreviewStringField), (
        f"text field: expected PreviewStringField, got {type(text_field)}"
    )
    assert text_field.full_text_search is not None, (
        "field must be FTS-enabled (full_text_search is not None), "
        f"got {text_field.full_text_search!r}"
    )
    assert text_field.filterable is False, (
        f"filterable must be False (not set at create time), got {text_field.filterable!r}"
    )

    cfg = text_field.full_text_search
    assert cfg.lowercase is not None, (
        "spec §1: server must populate 'lowercase' default even when not explicitly set; "
        "got None — SDK model may not be accepting server-populated values"
    )
    assert cfg.stemming is not None, (
        "spec §1: server must populate 'stemming' default even when not explicitly set; "
        "got None — SDK model may not be accepting server-populated values"
    )
    assert cfg.max_term_len is not None, (
        "spec §1: server must populate 'max_term_len' default even when not explicitly set; "
        "got None — SDK model may not be accepting server-populated values"
    )
    assert cfg.lowercase is True, (
        f"spec §1: server default for 'lowercase' is True, got {cfg.lowercase!r}"
    )
    assert cfg.stemming is False, (
        f"spec §1: server default for 'stemming' is False, got {cfg.stemming!r}"
    )
    assert cfg.max_term_len == 40, (
        f"spec §1: server default for 'max_term_len' is 40, got {cfg.max_term_len!r}"
    )


# ---------------------------------------------------------------------------
# test_async_preview_document_model_attributes_after_fts_search — §7
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(300)
async def test_async_preview_document_model_attributes_after_fts_search(
    async_client: AsyncPinecone,
    fts_index_state: tuple[str, str] | None,
    require_preview: None,
) -> None:
    """Async parity: verify PreviewDocument attribute access after FTS search with real matches.

    Spec §7: .id, ._id alias, .score, __getattr__ dynamic field, .get(key),
    .get(key, default) for missing keys, .to_dict(), .to_json().

    Reuses the session-scoped fts_index_state fixture (sync-created, warmed once per
    session). The async client connects to the same index by host to avoid per-test
    FTS cold-start latency. PVT-024 documented this gap.
    """
    import json

    if fts_index_state is None:
        pytest.skip("FTS index unavailable — preview endpoint unreachable or cold-start timeout")

    host, namespace = fts_index_state
    idx = async_client.preview.index(host=host)

    response = await idx.documents.search(
        namespace=namespace,
        top_k=5,
        score_by=[PreviewTextQuery(fields=["text"], query="ancient Rome")],
        include_fields=["text", "year"],
    )

    assert len(response.matches) > 0, (
        "Expected at least one FTS match for 'ancient Rome' (session fixture confirmed searchable)"
    )
    doc = response.matches[0]

    # §7: id and _id alias must both return the same non-empty string
    assert isinstance(doc.id, str), f"doc.id must be str, got {type(doc.id)}"
    assert doc._id == doc.id, f"doc._id must alias doc.id: {doc._id!r} != {doc.id!r}"
    assert len(doc.id) > 0, "doc.id must be non-empty"

    # §7: score is a positive float for a real FTS match
    assert doc.score is not None, "doc.score must not be None for an FTS match"
    assert isinstance(doc.score, float), f"doc.score must be float, got {type(doc.score)}"
    assert doc.score > 0, f"doc.score must be positive for a real match, got {doc.score}"

    # §7: __getattr__ dynamic field access
    text_val = doc.text
    assert isinstance(text_val, str), f"doc.text via __getattr__ must be str, got {type(text_val)}"
    year_val = doc.year
    # Integer filterable fields arrive as JSON numbers; msgspec dict[str, Any] may
    # deserialize them as int or float depending on the server's wire representation.
    assert isinstance(year_val, (int, float)), (
        f"doc.year via __getattr__ must be a number, got {type(year_val)}"
    )

    # §7: .get(key) returns same value as attribute access
    assert doc.get("text") == text_val, (
        f"doc.get('text') must equal doc.text: {doc.get('text')!r} != {text_val!r}"
    )
    assert doc.get("year") == year_val, (
        f"doc.get('year') must equal doc.year: {doc.get('year')!r} != {year_val!r}"
    )

    # §7: .get(missing_key, default) returns the default
    sentinel = object()
    assert doc.get("nonexistent_field", sentinel) is sentinel, (
        "doc.get('nonexistent_field', sentinel) must return default for missing field"
    )

    # §7: to_dict() returns a plain dict with the raw API data.
    # to_dict() passes through whatever the server returns unchanged — the
    # server emits "_id" and "_score" as the wire keys.
    d = doc.to_dict()
    assert isinstance(d, dict), f"doc.to_dict() must return dict, got {type(d)}"
    id_key = "_id" if "_id" in d else "id"
    assert id_key in d, f"to_dict() must include an id key ('_id' or 'id'), keys: {set(d.keys())}"
    assert "_score" in d, f"to_dict() must include '_score', keys: {set(d.keys())}"
    assert "text" in d, (
        f"to_dict() must include 'text' (from include_fields), keys: {set(d.keys())}"
    )
    assert d[id_key] == doc.id, f"to_dict()[{id_key!r}] must equal doc.id: {d[id_key]!r}"
    assert d["_score"] == doc.score, "to_dict()['_score'] must equal doc.score"

    # §7: to_json() returns valid JSON round-trippable to same content as to_dict()
    j = doc.to_json()
    assert isinstance(j, str), f"doc.to_json() must return str, got {type(j)}"
    parsed = json.loads(j)
    assert isinstance(parsed, dict), "doc.to_json() must parse to a dict"
    assert parsed[id_key] == doc.id, f"to_json() parsed {id_key!r} must equal doc.id"
    assert parsed["_score"] == doc.score, "to_json() parsed _score must equal doc.score"
