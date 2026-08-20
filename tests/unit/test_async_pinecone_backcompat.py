"""Tests for AsyncPinecone deprecated flat-method delegates (backcompat API)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from pinecone.async_client.pinecone import AsyncPinecone
from pinecone.inference.models.index_embed import IndexEmbed
from pinecone.models.enums import CloudProvider
from pinecone.models.indexes.specs import EmbedConfig, ServerlessSpec

SCHEMA = {"fields": {"embedding": {"type": "dense_vector", "dimension": 4, "metric": "cosine"}}}


def _make_async_pc_with_mock_indexes() -> tuple[AsyncPinecone, MagicMock]:
    pc = AsyncPinecone(api_key="test-key")
    mock_indexes = MagicMock()
    mock_indexes.create = AsyncMock(return_value=MagicMock())
    mock_indexes.create_for_model = AsyncMock(return_value=MagicMock())
    mock_indexes.describe = AsyncMock(return_value=MagicMock())
    mock_indexes.list = MagicMock(return_value=MagicMock())
    mock_indexes.exists = AsyncMock(return_value=True)
    mock_indexes.configure = AsyncMock(return_value=MagicMock())
    mock_indexes.delete = AsyncMock(return_value=None)
    pc._indexes = mock_indexes
    return pc, mock_indexes


def _make_async_pc_with_mock_collections() -> tuple[AsyncPinecone, MagicMock]:
    pc = AsyncPinecone(api_key="test-key")
    mock_collections = MagicMock()
    mock_collections.create = AsyncMock(return_value=MagicMock())
    mock_collections.list = AsyncMock(return_value=MagicMock())
    mock_collections.describe = AsyncMock(return_value=MagicMock())
    mock_collections.delete = AsyncMock(return_value=None)
    pc._collections = mock_collections
    return pc, mock_collections


def _make_async_pc_with_mock_backups() -> tuple[AsyncPinecone, MagicMock]:
    pc = AsyncPinecone(api_key="test-key")
    mock_backups = MagicMock()
    mock_backups.create = AsyncMock(return_value=MagicMock())
    mock_backups.list = AsyncMock(return_value=MagicMock())
    mock_backups.describe = AsyncMock(return_value=MagicMock())
    mock_backups.delete = AsyncMock(return_value=None)
    pc._backups = mock_backups
    return pc, mock_backups


def _make_async_pc_with_mock_restore_jobs() -> tuple[AsyncPinecone, MagicMock]:
    pc = AsyncPinecone(api_key="test-key")
    mock_restore_jobs = MagicMock()
    mock_restore_jobs.list = AsyncMock(return_value=MagicMock())
    mock_restore_jobs.describe = AsyncMock(return_value=MagicMock())
    pc._restore_jobs = mock_restore_jobs
    return pc, mock_restore_jobs


# ---------------------------------------------------------------------------
# create_index delegate
# ---------------------------------------------------------------------------


async def test_async_create_index_delegate_forwards() -> None:
    pc, mock_indexes = _make_async_pc_with_mock_indexes()
    await pc.create_index(
        name="x",
        schema=SCHEMA,
        deployment={"deployment_type": "managed", "cloud": "aws", "region": "us-east-1"},
    )
    mock_indexes.create.assert_called_once()
    _, kwargs = mock_indexes.create.call_args
    assert kwargs["name"] == "x"
    assert kwargs["schema"] == SCHEMA
    assert kwargs["deployment"] == {
        "deployment_type": "managed",
        "cloud": "aws",
        "region": "us-east-1",
    }
    assert kwargs["deletion_protection"] is None


async def test_async_create_index_delegate_forwards_all_new_kwargs() -> None:
    pc, mock_indexes = _make_async_pc_with_mock_indexes()
    await pc.create_index(
        name="x",
        schema=SCHEMA,
        read_capacity={"mode": "OnDemand"},
        deletion_protection="enabled",
        tags={"env": "prod"},
        cmek_id="key-1",
        timeout=-1,
    )
    _, kwargs = mock_indexes.create.call_args
    assert kwargs["read_capacity"] == {"mode": "OnDemand"}
    assert kwargs["cmek_id"] == "key-1"
    assert kwargs["timeout"] == -1


async def test_async_create_index_delegate_forwards_legacy_kwargs_for_interception() -> None:
    """Legacy kwargs pass through so AsyncIndexes.create raises the guided error."""
    pc, mock_indexes = _make_async_pc_with_mock_indexes()
    await pc.create_index(
        name="x",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        dimension=4,
    )
    _, kwargs = mock_indexes.create.call_args
    assert kwargs["dimension"] == 4
    assert isinstance(kwargs["spec"], ServerlessSpec)


# ---------------------------------------------------------------------------
# create_index_for_model delegate
# ---------------------------------------------------------------------------


async def test_async_create_index_for_model_delegate_forwards_index_embed() -> None:
    pc, mock_indexes = _make_async_pc_with_mock_indexes()
    index_embed = IndexEmbed(
        model="multilingual-e5-large",
        field_map={"text": "my_field"},
    )
    await pc.create_index_for_model(
        name="my-index",
        cloud=CloudProvider.AWS,
        region="us-east-1",
        embed=index_embed,
    )
    _, kwargs = mock_indexes.create_for_model.call_args
    assert kwargs["cloud"] == "aws"
    assert kwargs["region"] == "us-east-1"
    assert kwargs["embed"] is index_embed


async def test_async_create_index_for_model_delegate_forwards_embed_config() -> None:
    pc, mock_indexes = _make_async_pc_with_mock_indexes()
    embed_config = EmbedConfig(
        model="multilingual-e5-large",
        field_map={"text": "my_field"},
    )
    await pc.create_index_for_model(
        name="my-index",
        cloud=CloudProvider.AWS,
        region="us-east-1",
        embed=embed_config,
    )
    _, kwargs = mock_indexes.create_for_model.call_args
    assert kwargs["embed"] is embed_config


async def test_async_create_index_for_model_delegate_drops_disabled_deletion_protection() -> None:
    pc, mock_indexes = _make_async_pc_with_mock_indexes()
    await pc.create_index_for_model(
        name="my-index",
        cloud=CloudProvider.AWS,
        region="us-east-1",
        embed={"model": "m", "field_map": {"text": "a"}},
        deletion_protection="disabled",
    )
    _, kwargs = mock_indexes.create_for_model.call_args
    assert kwargs["deletion_protection"] is None


async def test_async_create_index_for_model_delegate_forwards_schema() -> None:
    pc, mock_indexes = _make_async_pc_with_mock_indexes()
    await pc.create_index_for_model(
        name="my-index",
        cloud="aws",
        region="us-east-1",
        embed={"model": "multilingual-e5-large", "field_map": {"text": "body"}},
        schema={"body": {"filterable": True}},
    )
    _, kwargs = mock_indexes.create_for_model.call_args
    assert kwargs["schema"] == {"body": {"filterable": True}}


async def test_async_create_index_for_model_delegate_forwards_read_capacity() -> None:
    pc, mock_indexes = _make_async_pc_with_mock_indexes()
    await pc.create_index_for_model(
        name="my-index",
        cloud="aws",
        region="us-east-1",
        embed={"model": "multilingual-e5-large", "field_map": {"text": "body"}},
        read_capacity={"mode": "OnDemand"},
    )
    _, kwargs = mock_indexes.create_for_model.call_args
    assert kwargs["read_capacity"] == {"mode": "OnDemand"}


async def test_async_create_index_for_model_delegate_schema_none_by_default() -> None:
    pc, mock_indexes = _make_async_pc_with_mock_indexes()
    await pc.create_index_for_model(
        name="my-index",
        cloud="aws",
        region="us-east-1",
        embed={"model": "multilingual-e5-large", "field_map": {"text": "body"}},
    )
    _, kwargs = mock_indexes.create_for_model.call_args
    assert kwargs["schema"] is None
    assert kwargs["read_capacity"] is None


# ---------------------------------------------------------------------------
# describe_index / list_indexes delegates
# ---------------------------------------------------------------------------


async def test_async_describe_index_delegate_forwards() -> None:
    pc, mock_indexes = _make_async_pc_with_mock_indexes()
    await pc.describe_index("my-index")
    mock_indexes.describe.assert_called_once_with("my-index")


def test_async_list_indexes_delegate_forwards() -> None:
    """list_indexes() is no longer a coroutine: it returns the paginator directly."""
    pc, mock_indexes = _make_async_pc_with_mock_indexes()
    pc.list_indexes()
    mock_indexes.list.assert_called_once()


# ---------------------------------------------------------------------------
# configure_index delegate
# ---------------------------------------------------------------------------


async def test_async_configure_index_delegate_forwards() -> None:
    pc, mock_indexes = _make_async_pc_with_mock_indexes()
    await pc.configure_index("my-index", deletion_protection="enabled")
    mock_indexes.configure.assert_called_once()
    _, kwargs = mock_indexes.configure.call_args
    assert kwargs["deletion_protection"] == "enabled"


async def test_async_configure_index_delegate_forwards_all_kwargs() -> None:
    pc, mock_indexes = _make_async_pc_with_mock_indexes()
    await pc.configure_index(
        "my-index",
        deployment={"replicas": 3, "pod_type": "p2.x2"},
        deletion_protection="enabled",
        tags={"env": "prod"},
        read_capacity={"mode": "OnDemand"},
    )
    mock_indexes.configure.assert_called_once_with(
        "my-index",
        deployment={"replicas": 3, "pod_type": "p2.x2"},
        schema=None,
        read_capacity={"mode": "OnDemand"},
        deletion_protection="enabled",
        tags={"env": "prod"},
    )


async def test_async_configure_index_delegate_forwards_legacy_kwargs_for_interception() -> None:
    pc, mock_indexes = _make_async_pc_with_mock_indexes()
    await pc.configure_index(
        "my-index",
        serverless_read_capacity={"mode": "OnDemand"},
    )
    mock_indexes.configure.assert_called_once_with(
        "my-index",
        deployment=None,
        schema=None,
        read_capacity=None,
        deletion_protection=None,
        tags=None,
        serverless_read_capacity={"mode": "OnDemand"},
    )


# ---------------------------------------------------------------------------
# Collection delegates
# ---------------------------------------------------------------------------


async def test_async_create_collection_delegate_forwards() -> None:
    pc, mock_collections = _make_async_pc_with_mock_collections()
    await pc.create_collection(name="my-coll", source="my-index")
    mock_collections.create.assert_called_once_with(name="my-coll", source="my-index")


async def test_async_list_collections_delegate_forwards() -> None:
    pc, mock_collections = _make_async_pc_with_mock_collections()
    await pc.list_collections()
    mock_collections.list.assert_called_once()


async def test_async_describe_collection_delegate_forwards() -> None:
    pc, mock_collections = _make_async_pc_with_mock_collections()
    await pc.describe_collection("my-coll")
    mock_collections.describe.assert_called_once_with("my-coll")


# ---------------------------------------------------------------------------
# Backup delegates
# ---------------------------------------------------------------------------


async def test_async_create_backup_delegate_forwards() -> None:
    pc, mock_backups = _make_async_pc_with_mock_backups()
    await pc.create_backup(index_name="my-index", backup_name="my-backup")
    mock_backups.create.assert_called_once_with(
        index_name="my-index", name="my-backup", description=None
    )


async def test_async_list_backups_delegate_forwards() -> None:
    pc, mock_backups = _make_async_pc_with_mock_backups()
    await pc.list_backups(index_name="my-index")
    mock_backups.list.assert_called_once_with(
        index_name="my-index", limit=None, pagination_token=None, include_deleted=None
    )


async def test_async_list_backups_delegate_forwards_include_deleted() -> None:
    pc, mock_backups = _make_async_pc_with_mock_backups()
    await pc.list_backups(index_name="my-index", include_deleted=True)
    mock_backups.list.assert_called_once_with(
        index_name="my-index", limit=None, pagination_token=None, include_deleted=True
    )


async def test_async_describe_backup_delegate_forwards() -> None:
    pc, mock_backups = _make_async_pc_with_mock_backups()
    await pc.describe_backup(backup_id="bkp-123")
    mock_backups.describe.assert_called_once_with(backup_id="bkp-123")


# ---------------------------------------------------------------------------
# Restore job delegates
# ---------------------------------------------------------------------------


async def test_async_list_restore_jobs_delegate_forwards() -> None:
    pc, mock_restore_jobs = _make_async_pc_with_mock_restore_jobs()
    await pc.list_restore_jobs()
    mock_restore_jobs.list.assert_called_once_with(limit=None, pagination_token=None)


async def test_async_describe_restore_job_delegate_forwards() -> None:
    pc, mock_restore_jobs = _make_async_pc_with_mock_restore_jobs()
    await pc.describe_restore_job(job_id="job-456")
    mock_restore_jobs.describe.assert_called_once_with(job_id="job-456")


# ---------------------------------------------------------------------------
# IndexAsyncio factory delegate
# ---------------------------------------------------------------------------


def test_async_index_asyncio_delegate_returns_async_index() -> None:
    from pinecone.async_client.async_index import AsyncIndex

    pc = AsyncPinecone(api_key="test-key")
    idx = pc.IndexAsyncio(host="my-index.svc.pinecone.io")
    assert isinstance(idx, AsyncIndex)


# ---------------------------------------------------------------------------
# __repr__ masking
# ---------------------------------------------------------------------------


def test_async_pinecone_repr_masks_full_api_key() -> None:
    pc = AsyncPinecone(api_key="pcsk_secret_12345")
    result = repr(pc)
    assert "pcsk_secret_12345" not in result
    assert "...2345" in result
    assert "host=" in result
    assert "AsyncPinecone" in result


def test_async_pinecone_repr_masks_short_api_key() -> None:
    pc = AsyncPinecone(api_key="ab")
    result = repr(pc)
    assert "api_key='***'" in result
    assert "api_key='ab'" not in result


def test_async_pinecone_repr_exactly_four_char_key_shows_last_four() -> None:
    pc = AsyncPinecone(api_key="wxyz")
    result = repr(pc)
    assert "...wxyz" in result


# ---------------------------------------------------------------------------
# has_index / delete_index delegates
# ---------------------------------------------------------------------------


async def test_async_has_index_delegate_forwards() -> None:
    pc, mock_indexes = _make_async_pc_with_mock_indexes()
    result = await pc.has_index("my-index")
    assert result is True
    mock_indexes.exists.assert_awaited_once_with("my-index")


async def test_async_delete_index_delegate_forwards() -> None:
    pc, mock_indexes = _make_async_pc_with_mock_indexes()
    await pc.delete_index("my-index", timeout=30)
    mock_indexes.delete.assert_awaited_once_with("my-index", timeout=30)

    mock_indexes.delete.reset_mock()
    await pc.delete_index("my-index")
    mock_indexes.delete.assert_awaited_once_with("my-index", timeout=None)


# ---------------------------------------------------------------------------
# delete_collection delegate
# ---------------------------------------------------------------------------


async def test_async_delete_collection_delegate_forwards() -> None:
    pc, mock_collections = _make_async_pc_with_mock_collections()
    await pc.delete_collection("my-coll")
    mock_collections.delete.assert_awaited_once_with("my-coll")


# ---------------------------------------------------------------------------
# delete_backup delegate
# ---------------------------------------------------------------------------


async def test_async_delete_backup_delegate_forwards() -> None:
    pc, mock_backups = _make_async_pc_with_mock_backups()
    await pc.delete_backup(backup_id="bkp-123")
    mock_backups.delete.assert_awaited_once_with(backup_id="bkp-123")
