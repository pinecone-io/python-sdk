"""Tests for AsyncPinecone deprecated flat-method delegates (backcompat API)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from pinecone.async_client.pinecone import AsyncPinecone
from pinecone.inference.models.index_embed import IndexEmbed
from pinecone.models.enums import CloudProvider
from pinecone.models.indexes.specs import EmbedConfig, IntegratedSpec, ServerlessSpec


def _make_async_pc_with_mock_indexes() -> tuple[AsyncPinecone, MagicMock]:
    pc = AsyncPinecone(api_key="test-key")
    mock_indexes = MagicMock()
    mock_indexes.create = AsyncMock(return_value=MagicMock())
    mock_indexes.describe = AsyncMock(return_value=MagicMock())
    mock_indexes.list = AsyncMock(return_value=MagicMock())
    mock_indexes.exists = AsyncMock(return_value=True)
    mock_indexes.configure = AsyncMock(return_value=None)
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
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        dimension=4,
    )
    mock_indexes.create.assert_called_once()
    _, kwargs = mock_indexes.create.call_args
    assert kwargs["metric"] == "cosine"
    assert kwargs["vector_type"] == "dense"
    assert kwargs["deletion_protection"] == "disabled"


async def test_async_create_index_delegate_with_none_deletion_protection_defaults_disabled() -> (
    None
):
    pc, mock_indexes = _make_async_pc_with_mock_indexes()
    await pc.create_index(
        name="x",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        deletion_protection=None,
    )
    _, kwargs = mock_indexes.create.call_args
    assert kwargs["deletion_protection"] == "disabled"


async def test_async_create_index_delegate_with_explicit_metric_forwards_verbatim() -> None:
    pc, mock_indexes = _make_async_pc_with_mock_indexes()
    await pc.create_index(
        name="x",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        metric="euclidean",
        vector_type="sparse",
    )
    _, kwargs = mock_indexes.create.call_args
    assert kwargs["metric"] == "euclidean"
    assert kwargs["vector_type"] == "sparse"


async def test_async_create_index_delegate_forwards_schema() -> None:
    pc, mock_indexes = _make_async_pc_with_mock_indexes()
    await pc.create_index(
        name="x",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        dimension=1536,
        schema={"text_field": {"type": "str"}},
    )
    _, kwargs = mock_indexes.create.call_args
    assert kwargs["schema"] == {"text_field": {"type": "str"}}


async def test_async_create_index_delegate_schema_none_by_default() -> None:
    pc, mock_indexes = _make_async_pc_with_mock_indexes()
    await pc.create_index(
        name="x",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        dimension=4,
    )
    _, kwargs = mock_indexes.create.call_args
    assert kwargs["schema"] is None


# ---------------------------------------------------------------------------
# create_index_for_model delegate
# ---------------------------------------------------------------------------


async def test_async_create_index_for_model_delegate_with_index_embed_converts() -> None:
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
    _, kwargs = mock_indexes.create.call_args
    spec = kwargs["spec"]
    assert isinstance(spec, IntegratedSpec)
    assert isinstance(spec.embed, EmbedConfig)
    assert spec.embed.model == "multilingual-e5-large"
    assert spec.embed.field_map == {"text": "my_field"}
    assert spec.cloud == "aws"


async def test_async_create_index_for_model_delegate_with_embed_config_passes_through() -> None:
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
    _, kwargs = mock_indexes.create.call_args
    spec = kwargs["spec"]
    assert isinstance(spec, IntegratedSpec)
    assert spec.embed is embed_config
    assert spec.cloud == "aws"


async def test_async_create_index_for_model_delegate_with_dict_constructs_embed_config() -> None:
    pc, mock_indexes = _make_async_pc_with_mock_indexes()
    await pc.create_index_for_model(
        name="my-index",
        cloud=CloudProvider.AWS,
        region="us-east-1",
        embed={"model": "m", "field_map": {"text": "a"}},
    )
    _, kwargs = mock_indexes.create.call_args
    spec = kwargs["spec"]
    assert isinstance(spec, IntegratedSpec)
    assert isinstance(spec.embed, EmbedConfig)
    assert spec.embed.model == "m"
    assert spec.embed.field_map == {"text": "a"}
    assert spec.cloud == "aws"


async def test_async_create_index_for_model_delegate_forwards_schema() -> None:
    pc, mock_indexes = _make_async_pc_with_mock_indexes()
    await pc.create_index_for_model(
        name="my-index",
        cloud="aws",
        region="us-east-1",
        embed={"model": "multilingual-e5-large", "field_map": {"text": "body"}},
        schema={"body": {"type": "str"}},
    )
    _, kwargs = mock_indexes.create.call_args
    assert kwargs["schema"] == {"body": {"type": "str"}}


async def test_async_create_index_for_model_delegate_forwards_read_capacity() -> None:
    pc, mock_indexes = _make_async_pc_with_mock_indexes()
    await pc.create_index_for_model(
        name="my-index",
        cloud="aws",
        region="us-east-1",
        embed={"model": "multilingual-e5-large", "field_map": {"text": "body"}},
        read_capacity={"mode": "OnDemand"},
    )
    _, kwargs = mock_indexes.create.call_args
    assert kwargs["read_capacity"] == {"mode": "OnDemand"}


async def test_async_create_index_for_model_delegate_schema_none_by_default() -> None:
    pc, mock_indexes = _make_async_pc_with_mock_indexes()
    await pc.create_index_for_model(
        name="my-index",
        cloud="aws",
        region="us-east-1",
        embed={"model": "multilingual-e5-large", "field_map": {"text": "body"}},
    )
    _, kwargs = mock_indexes.create.call_args
    assert kwargs["schema"] is None
    assert kwargs["read_capacity"] is None


# ---------------------------------------------------------------------------
# describe_index / list_indexes delegates
# ---------------------------------------------------------------------------


async def test_async_describe_index_delegate_forwards() -> None:
    pc, mock_indexes = _make_async_pc_with_mock_indexes()
    await pc.describe_index("my-index")
    mock_indexes.describe.assert_called_once_with("my-index")


async def test_async_list_indexes_delegate_forwards() -> None:
    pc, mock_indexes = _make_async_pc_with_mock_indexes()
    await pc.list_indexes()
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
        index_name="my-index", limit=None, pagination_token=None
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


def test_async_index_asyncio_delegate_forwards_explicit_ssl_verify() -> None:
    from unittest.mock import patch

    pc = AsyncPinecone(api_key="test-key", ssl_verify=True)
    with patch("pinecone.async_client.async_index.AsyncIndex") as mock_async_index:
        pc.IndexAsyncio(host="my-index.svc.pinecone.io", ssl_verify=False)
    _, kwargs = mock_async_index.call_args
    assert kwargs["ssl_verify"] is False


def test_async_index_asyncio_delegate_rejects_unknown_kwargs() -> None:
    pc = AsyncPinecone(api_key="test-key")
    with pytest.raises(TypeError, match="unexpected keyword arguments"):
        pc.IndexAsyncio(host="my-index.svc.pinecone.io", bogus=True)


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
