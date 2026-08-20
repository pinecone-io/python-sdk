"""Integration tests for preview backup lifecycle (create, list).

Tests run against Pinecone API version 2026-01.alpha. Only ``create_backup``
and ``list_backups`` (scoped to an index) are in this API version's endpoint
set — standalone describe, delete, and restore operations are NOT tested here.

Tests skip gracefully when the preview endpoint is unavailable. They do NOT
gate CI (preview_integration mark).
"""

from __future__ import annotations

import pytest

from pinecone import Pinecone
from pinecone.models.backups.model import BackupModel
from pinecone.models.indexes.schema import IndexSchema
from pinecone.preview import SchemaBuilder as PreviewSchemaBuilder
from tests.integration.conftest import poll_until

pytestmark = [pytest.mark.integration, pytest.mark.preview_integration]


def _simple_fts_schema() -> dict:  # type: ignore[type-arg]
    return PreviewSchemaBuilder().add_string_field("text", full_text_search={}).build()


def _is_ready(m: object) -> bool:
    from pinecone.preview.models import PreviewIndexModel

    return isinstance(m, PreviewIndexModel) and m.status.state == "Ready"


@pytest.fixture
def ready_preview_index(
    client: Pinecone,
    preview_index_name: str,
    cleanup_preview_indexes: list[str],
    require_preview: None,
) -> str:
    """Create a minimal FTS preview index, wait until Ready, return its name."""
    cleanup_preview_indexes.append(preview_index_name)
    client.preview.indexes.create(
        name=preview_index_name,
        schema=_simple_fts_schema(),
    )
    poll_until(
        lambda: client.preview.indexes.describe(preview_index_name),
        _is_ready,
        timeout=300,
        interval=5,
        description=f"preview index {preview_index_name} ready",
    )
    return preview_index_name


def test_create_backup_returns_initializing_backup(
    client: Pinecone,
    ready_preview_index: str,
) -> None:
    backup = client.preview.indexes.create_backup(
        ready_preview_index,
        name="before-migration",
        description="Snapshot before schema change",
    )
    assert isinstance(backup, BackupModel)
    assert isinstance(backup.backup_id, str) and len(backup.backup_id) > 0
    assert isinstance(backup.status, str) and len(backup.status) > 0


def test_list_backups_includes_created_backup(
    client: Pinecone,
    ready_preview_index: str,
) -> None:
    backup = client.preview.indexes.create_backup(
        ready_preview_index,
        name="integration-list-test",
    )
    backup_ids = [b.backup_id for b in client.preview.indexes.list_backups(ready_preview_index)]
    assert backup.backup_id in backup_ids
    for item in client.preview.indexes.list_backups(ready_preview_index):
        assert isinstance(item, BackupModel)
        assert isinstance(item.backup_id, str) and len(item.backup_id) > 0
        assert isinstance(item.status, str)
        assert isinstance(item.created_at, str)


def test_list_backups_empty_for_new_index(
    client: Pinecone,
    ready_preview_index: str,
) -> None:
    items = list(client.preview.indexes.list_backups(ready_preview_index))
    assert len(items) == 0


def test_list_backups_server_side_limit(
    client: Pinecone,
    ready_preview_index: str,
) -> None:
    items = list(client.preview.indexes.list_backups(ready_preview_index, limit=2))
    assert len(items) <= 2


# ---------------------------------------------------------------------------
# Dense-vector fixture — avoids FTS dedicated-read-capacity requirement
# ---------------------------------------------------------------------------


@pytest.fixture
def ready_dense_preview_index(
    client: Pinecone,
    preview_index_name: str,
    cleanup_preview_indexes: list[str],
    require_preview: None,
) -> str:
    """Create a dense-vector preview index, wait until Ready, return its name."""
    cleanup_preview_indexes.append(preview_index_name)
    client.preview.indexes.create(
        name=preview_index_name,
        schema=PreviewSchemaBuilder()
        .add_dense_vector_field("embedding", dimension=4, metric="cosine")
        .build(),
    )
    poll_until(
        lambda: client.preview.indexes.describe(preview_index_name),
        _is_ready,
        timeout=300,
        interval=5,
        description=f"preview index {preview_index_name} ready",
    )
    return preview_index_name


def test_create_backup_all_required_fields_present(
    client: Pinecone,
    ready_dense_preview_index: str,
) -> None:
    """create_backup() response contains all required BackupModel fields.

    Existing tests verify backup_id and status only. This test checks the full
    set of required fields: source_index_name, source_index_id, cloud, region,
    created_at, name, and description. Uses a dense-vector index (no FTS).
    """
    backup = client.preview.indexes.create_backup(
        ready_dense_preview_index,
        name="all-fields-test",
        description="Verifying all required backup fields",
    )
    assert isinstance(backup, BackupModel)
    assert isinstance(backup.backup_id, str) and len(backup.backup_id) > 0
    assert backup.source_index_name == ready_dense_preview_index
    assert isinstance(backup.source_index_id, str) and len(backup.source_index_id) > 0
    assert isinstance(backup.status, str) and len(backup.status) > 0
    assert isinstance(backup.cloud, str) and len(backup.cloud) > 0
    assert isinstance(backup.region, str) and len(backup.region) > 0
    assert isinstance(backup.created_at, str) and len(backup.created_at) > 0
    assert backup.name == "all-fields-test"
    assert backup.description == "Verifying all required backup fields"


# ---------------------------------------------------------------------------
# test_create_backup_optional_fields_are_correctly_typed — §2 BackupModel optional fields
# ---------------------------------------------------------------------------


def test_describe_backup_returns_preview_backup_model(
    client: Pinecone,
    ready_dense_preview_index: str,
) -> None:
    """describe_backup() returns a BackupModel with expected fields."""
    backup = client.preview.indexes.create_backup(
        ready_dense_preview_index,
        name="describe-test",
    )
    described = client.preview.indexes.describe_backup(backup.backup_id)
    assert isinstance(described, BackupModel)
    assert described.backup_id == backup.backup_id
    assert isinstance(described.status, str) and len(described.status) > 0
    assert described.source_index_name == ready_dense_preview_index


def test_create_backup_optional_fields_are_correctly_typed(
    client: Pinecone,
    ready_dense_preview_index: str,
) -> None:
    """Optional BackupModel fields have correct Python types (int, dict, or None).

    PVT-010 (test_create_backup_all_required_fields_present) verified required fields:
    backup_id, source_index_name, source_index_id, status, cloud, region, created_at,
    name, description.

    This test covers the optional fields declared in §2 BackupModel:
    - dense_dimension: int | None — 4 for a 4-dim dense vector index when the
      server returns a typed schema
    - schema: IndexSchema | None
    - source_index_deleted_at: str | None — None while the source index is active
    - tags: dict | None      — None when no tags passed to create_backup()
    - record_count: int | None
    - namespace_count: int | None
    - size_bytes: int | None

    Verifies the SDK's msgspec deserialization does not raise for any of these fields,
    and that the values are the expected Python types (not strings, lists, etc.).
    """
    backup = client.preview.indexes.create_backup(
        ready_dense_preview_index,
        name="optional-fields-test",
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

    # tags — dict[str, Any] or None; API returns {} when no tags are passed
    assert backup.tags is None or isinstance(backup.tags, dict), (
        f"backup.tags must be dict or None, got {type(backup.tags)}"
    )

    # record_count — int or None; fresh backup may be 0 or None before Ready
    assert backup.record_count is None or isinstance(backup.record_count, int), (
        f"backup.record_count must be int or None, got {type(backup.record_count)}"
    )

    # namespace_count — int or None
    assert backup.namespace_count is None or isinstance(backup.namespace_count, int), (
        f"backup.namespace_count must be int or None, got {type(backup.namespace_count)}"
    )

    # size_bytes — int or None; may be 0 or None while backup is still Initializing
    assert backup.size_bytes is None or isinstance(backup.size_bytes, int), (
        f"backup.size_bytes must be int or None, got {type(backup.size_bytes)}"
    )
