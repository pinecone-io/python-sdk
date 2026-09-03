"""Tests for Pinecone.__repr__ masking and deprecated delegate methods."""

from __future__ import annotations

import warnings
from unittest.mock import MagicMock, patch

import pytest

from pinecone import Pinecone
from pinecone.inference.models.index_embed import IndexEmbed
from pinecone.models.enums import CloudProvider
from pinecone.models.indexes.specs import EmbedConfig, ServerlessSpec


class TestPineconeRepr:
    def test_pinecone_repr_masks_full_api_key(self) -> None:
        pc = Pinecone(api_key="pcsk_secret_12345")
        result = repr(pc)
        assert "pcsk_secret_12345" not in result
        assert "...2345" in result
        assert "host=" in result

    def test_pinecone_repr_masks_short_api_key(self) -> None:
        pc = Pinecone(api_key="ab")
        result = repr(pc)
        assert "api_key='***'" in result
        assert "ab" not in result

    def test_pinecone_repr_exactly_four_char_key_shows_last_four(self) -> None:
        pc = Pinecone(api_key="wxyz")
        result = repr(pc)
        assert "...wxyz" in result


def _make_pc_with_mock_indexes() -> tuple[Pinecone, MagicMock]:
    pc = Pinecone(api_key="test-key")
    mock_indexes = MagicMock()
    mock_indexes.create.return_value = MagicMock()
    pc._indexes = mock_indexes
    return pc, mock_indexes


def test_pinecone_create_index_delegate_forwards() -> None:
    pc, mock_indexes = _make_pc_with_mock_indexes()
    spec = ServerlessSpec(cloud="aws", region="us-east-1")
    pc.create_index(
        name="x",
        spec=spec,
        dimension=4,
        metric="cosine",
        vector_type="dense",
        deletion_protection="enabled",
        tags={"env": "prod"},
        timeout=-1,
    )
    mock_indexes.create.assert_called_once_with(
        name="x",
        spec=spec,
        dimension=4,
        metric="cosine",
        vector_type="dense",
        deletion_protection="enabled",
        tags={"env": "prod"},
        timeout=-1,
    )


def test_pinecone_create_index_delegate_defaults() -> None:
    pc, mock_indexes = _make_pc_with_mock_indexes()
    pc.create_index(name="x")
    mock_indexes.create.assert_called_once_with(
        name="x",
        spec=None,
        dimension=None,
        metric=None,
        vector_type=None,
        deletion_protection=None,
        tags=None,
        timeout=None,
    )


def test_pinecone_create_index_delegate_rejects_2026_07_only_kwargs() -> None:
    pc, _ = _make_pc_with_mock_indexes()
    with pytest.raises(TypeError):
        pc.create_index(name="x", schema={"fields": {}})
    with pytest.raises(TypeError):
        pc.create_index(name="x", deployment={"deployment_type": "managed"})
    with pytest.raises(TypeError):
        pc.create_index(name="x", cmek_id="key-1")


def test_pinecone_create_index_for_model_delegate_forwards_index_embed() -> None:
    pc, mock_indexes = _make_pc_with_mock_indexes()
    index_embed = IndexEmbed(
        model="multilingual-e5-large",
        field_map={"text": "my_field"},
    )
    pc.create_index_for_model(
        name="my-index",
        cloud=CloudProvider.AWS,
        region="us-east-1",
        embed=index_embed,
    )
    _, kwargs = mock_indexes.create_for_model.call_args
    assert kwargs["cloud"] is CloudProvider.AWS
    assert kwargs["region"] == "us-east-1"
    assert kwargs["embed"] is index_embed


def test_pinecone_create_index_for_model_delegate_forwards_embed_config() -> None:
    pc, mock_indexes = _make_pc_with_mock_indexes()
    embed_config = EmbedConfig(
        model="multilingual-e5-large",
        field_map={"text": "my_field"},
    )
    pc.create_index_for_model(
        name="my-index",
        cloud=CloudProvider.AWS,
        region="us-east-1",
        embed=embed_config,
    )
    _, kwargs = mock_indexes.create_for_model.call_args
    assert kwargs["embed"] is embed_config


def test_pinecone_create_index_for_model_delegate_defaults_deletion_protection_to_disabled() -> (
    None
):
    pc, mock_indexes = _make_pc_with_mock_indexes()
    pc.create_index_for_model(
        name="my-index",
        cloud=CloudProvider.AWS,
        region="us-east-1",
        embed={"model": "m", "field_map": {"text": "a"}},
    )
    _, kwargs = mock_indexes.create_for_model.call_args
    assert kwargs["deletion_protection"] == "disabled"


def test_pinecone_create_index_for_model_delegate_forwards_explicit_disabled() -> None:
    """An explicit deletion_protection='disabled' must not be dropped."""
    pc, mock_indexes = _make_pc_with_mock_indexes()
    pc.create_index_for_model(
        name="my-index",
        cloud=CloudProvider.AWS,
        region="us-east-1",
        embed={"model": "m", "field_map": {"text": "a"}},
        deletion_protection="disabled",
    )
    _, kwargs = mock_indexes.create_for_model.call_args
    assert kwargs["deletion_protection"] == "disabled"


# ---------------------------------------------------------------------------
# Helpers for backcompat delegate tests
# ---------------------------------------------------------------------------


def _make_pc_with_mock_indexes_delegates() -> tuple[Pinecone, MagicMock]:
    pc = Pinecone(api_key="test-key")
    mock_indexes = MagicMock()
    pc._indexes = mock_indexes
    return pc, mock_indexes


def _make_pc_with_mock_collections() -> tuple[Pinecone, MagicMock]:
    pc = Pinecone(api_key="test-key")
    mock_collections = MagicMock()
    pc._collections = mock_collections
    return pc, mock_collections


def _make_pc_with_mock_backups() -> tuple[Pinecone, MagicMock]:
    pc = Pinecone(api_key="test-key")
    mock_backups = MagicMock()
    pc._backups = mock_backups
    return pc, mock_backups


def _make_pc_with_mock_restore_jobs() -> tuple[Pinecone, MagicMock]:
    pc = Pinecone(api_key="test-key")
    mock_restore_jobs = MagicMock()
    pc._restore_jobs = mock_restore_jobs
    return pc, mock_restore_jobs


# ---------------------------------------------------------------------------
# describe_index delegate
# ---------------------------------------------------------------------------


class TestDescribeIndex:
    def test_forwards(self) -> None:
        pc, mock_indexes = _make_pc_with_mock_indexes_delegates()
        pc.describe_index("my-index")
        mock_indexes.describe.assert_called_once_with("my-index")


# ---------------------------------------------------------------------------
# list_indexes delegate
# ---------------------------------------------------------------------------


class TestListIndexes:
    def test_forwards_and_wraps_in_index_list(self) -> None:
        from pinecone.models.indexes.list import IndexList

        pc, mock_indexes = _make_pc_with_mock_indexes_delegates()
        mock_paginator = MagicMock()
        mock_paginator.to_list.return_value = ["index-a", "index-b"]
        mock_indexes.list.return_value = mock_paginator

        result = pc.list_indexes()

        mock_indexes.list.assert_called_once()
        mock_paginator.to_list.assert_called_once()
        assert isinstance(result, IndexList)
        assert list(result) == ["index-a", "index-b"]

    def test_names_returns_list_of_str(self) -> None:
        from pinecone.models.indexes.index import IndexModel

        pc, mock_indexes = _make_pc_with_mock_indexes_delegates()
        idx_a = MagicMock(spec=IndexModel)
        idx_a.name = "index-a"
        idx_b = MagicMock(spec=IndexModel)
        idx_b.name = "index-b"
        mock_paginator = MagicMock()
        mock_paginator.to_list.return_value = [idx_a, idx_b]
        mock_indexes.list.return_value = mock_paginator

        names = pc.list_indexes().names()

        assert names == ["index-a", "index-b"]
        assert all(isinstance(n, str) for n in names)


# ---------------------------------------------------------------------------
# has_index delegate
# ---------------------------------------------------------------------------


class TestHasIndex:
    def test_forwards(self) -> None:
        pc, mock_indexes = _make_pc_with_mock_indexes_delegates()
        mock_indexes.exists.return_value = True
        result = pc.has_index("my-index")
        mock_indexes.exists.assert_called_once_with("my-index")
        assert result is True


# ---------------------------------------------------------------------------
# configure_index delegate
# ---------------------------------------------------------------------------


class TestConfigureIndex:
    def test_minimal_forwards(self) -> None:
        pc, mock_indexes = _make_pc_with_mock_indexes_delegates()
        pc.configure_index("my-index", deletion_protection="enabled")
        mock_indexes.configure.assert_called_once_with(
            "my-index",
            replicas=None,
            pod_type=None,
            deletion_protection="enabled",
            tags=None,
            embed=None,
            read_capacity=None,
            serverless_read_capacity=None,
        )

    def test_all_kwargs_forwarded(self) -> None:
        pc, mock_indexes = _make_pc_with_mock_indexes_delegates()
        pc.configure_index(
            "my-index",
            replicas=3,
            pod_type="p2.x2",
            deletion_protection="enabled",
            tags={"env": "prod"},
        )
        mock_indexes.configure.assert_called_once_with(
            "my-index",
            replicas=3,
            pod_type="p2.x2",
            deletion_protection="enabled",
            tags={"env": "prod"},
            embed=None,
            read_capacity=None,
            serverless_read_capacity=None,
        )

    def test_serverless_read_capacity_forwarded(self) -> None:
        pc, mock_indexes = _make_pc_with_mock_indexes_delegates()
        pc.configure_index(
            "my-index",
            serverless_read_capacity={"mode": "OnDemand"},
        )
        mock_indexes.configure.assert_called_once_with(
            "my-index",
            replicas=None,
            pod_type=None,
            deletion_protection=None,
            tags=None,
            embed=None,
            read_capacity=None,
            serverless_read_capacity={"mode": "OnDemand"},
        )

    def test_does_not_accept_2026_07_only_kwargs(self) -> None:
        pc, mock_indexes = _make_pc_with_mock_indexes_delegates()
        with pytest.raises(TypeError, match=r"pc\.indexes\.configure"):
            pc.configure_index("my-index", deployment={"replicas": 3})
        with pytest.raises(TypeError, match=r"pc\.indexes\.configure"):
            pc.configure_index("my-index", schema={"fields": {}})
        mock_indexes.configure.assert_not_called()

    def test_embed_reaches_configure_for_guided_error(self) -> None:
        from pinecone.errors.exceptions import PineconeTypeError

        pc = Pinecone(api_key="test-key")
        with pytest.raises(PineconeTypeError, match="no longer accepts embed="):
            pc.configure_index("my-index", embed={"model": "multilingual-e5-large"})


# ---------------------------------------------------------------------------
# delete_index delegate
# ---------------------------------------------------------------------------


class TestDeleteIndex:
    def test_without_timeout_forwards(self) -> None:
        pc, mock_indexes = _make_pc_with_mock_indexes_delegates()
        pc.delete_index("my-index")
        mock_indexes.delete.assert_called_once_with("my-index", timeout=None)

    def test_with_timeout_forwards_timeout(self) -> None:
        pc, mock_indexes = _make_pc_with_mock_indexes_delegates()
        pc.delete_index("my-index", timeout=30)
        mock_indexes.delete.assert_called_once_with("my-index", timeout=30)


# ---------------------------------------------------------------------------
# create_collection delegate
# ---------------------------------------------------------------------------


class TestCreateCollection:
    def test_forwards(self) -> None:
        pc, mock_collections = _make_pc_with_mock_collections()
        pc.create_collection(name="my-coll", source="my-index")
        mock_collections.create.assert_called_once_with(name="my-coll", source="my-index")


# ---------------------------------------------------------------------------
# list_collections delegate
# ---------------------------------------------------------------------------


class TestListCollections:
    def test_forwards(self) -> None:
        pc, mock_collections = _make_pc_with_mock_collections()
        pc.list_collections()
        mock_collections.list.assert_called_once()


# ---------------------------------------------------------------------------
# describe_collection delegate
# ---------------------------------------------------------------------------


class TestDescribeCollection:
    def test_forwards(self) -> None:
        pc, mock_collections = _make_pc_with_mock_collections()
        pc.describe_collection("my-coll")
        mock_collections.describe.assert_called_once_with("my-coll")


# ---------------------------------------------------------------------------
# delete_collection delegate
# ---------------------------------------------------------------------------


class TestDeleteCollection:
    def test_forwards(self) -> None:
        pc, mock_collections = _make_pc_with_mock_collections()
        pc.delete_collection("my-coll")
        mock_collections.delete.assert_called_once_with("my-coll")


# ---------------------------------------------------------------------------
# create_backup delegate
# ---------------------------------------------------------------------------


class TestCreateBackup:
    def test_forwards(self) -> None:
        pc, mock_backups = _make_pc_with_mock_backups()
        pc.create_backup(index_name="my-index", backup_name="my-backup")
        mock_backups.create.assert_called_once_with(
            index_name="my-index", name="my-backup", description=None
        )


# ---------------------------------------------------------------------------
# list_backups delegate
# ---------------------------------------------------------------------------


class TestListBackups:
    def test_forwards(self) -> None:
        pc, mock_backups = _make_pc_with_mock_backups()
        pc.list_backups(index_name="my-index")
        mock_backups.list.assert_called_once_with(
            index_name="my-index", limit=None, pagination_token=None, include_deleted=None
        )

    def test_limit_none_forwarded_as_none(self) -> None:
        pc, mock_backups = _make_pc_with_mock_backups()
        pc.list_backups(limit=None)
        mock_backups.list.assert_called_once_with(
            index_name=None, limit=None, pagination_token=None, include_deleted=None
        )

    def test_explicit_limit_forwarded(self) -> None:
        pc, mock_backups = _make_pc_with_mock_backups()
        pc.list_backups(limit=25)
        mock_backups.list.assert_called_once_with(
            index_name=None, limit=25, pagination_token=None, include_deleted=None
        )

    def test_include_deleted_forwarded(self) -> None:
        pc, mock_backups = _make_pc_with_mock_backups()
        pc.list_backups(index_name="my-index", include_deleted=True)
        mock_backups.list.assert_called_once_with(
            index_name="my-index", limit=None, pagination_token=None, include_deleted=True
        )


# ---------------------------------------------------------------------------
# describe_backup delegate
# ---------------------------------------------------------------------------


class TestDescribeBackup:
    def test_forwards(self) -> None:
        pc, mock_backups = _make_pc_with_mock_backups()
        pc.describe_backup(backup_id="bkp-123")
        mock_backups.describe.assert_called_once_with(backup_id="bkp-123")


# ---------------------------------------------------------------------------
# delete_backup delegate
# ---------------------------------------------------------------------------


class TestDeleteBackup:
    def test_forwards(self) -> None:
        pc, mock_backups = _make_pc_with_mock_backups()
        pc.delete_backup(backup_id="bkp-123")
        mock_backups.delete.assert_called_once_with(backup_id="bkp-123")


# ---------------------------------------------------------------------------
# list_restore_jobs delegate
# ---------------------------------------------------------------------------


class TestListRestoreJobs:
    def test_forwards(self) -> None:
        pc, mock_restore_jobs = _make_pc_with_mock_restore_jobs()
        pc.list_restore_jobs()
        mock_restore_jobs.list.assert_called_once_with(limit=None, pagination_token=None)

    def test_limit_none_forwarded_as_none(self) -> None:
        pc, mock_restore_jobs = _make_pc_with_mock_restore_jobs()
        pc.list_restore_jobs(limit=None)
        mock_restore_jobs.list.assert_called_once_with(limit=None, pagination_token=None)

    def test_explicit_limit_forwarded(self) -> None:
        pc, mock_restore_jobs = _make_pc_with_mock_restore_jobs()
        pc.list_restore_jobs(limit=25)
        mock_restore_jobs.list.assert_called_once_with(limit=25, pagination_token=None)


# ---------------------------------------------------------------------------
# describe_restore_job delegate
# ---------------------------------------------------------------------------


class TestDescribeRestoreJob:
    def test_forwards(self) -> None:
        pc, mock_restore_jobs = _make_pc_with_mock_restore_jobs()
        pc.describe_restore_job(job_id="job-456")
        mock_restore_jobs.describe.assert_called_once_with(job_id="job-456")


# ---------------------------------------------------------------------------
# Index() factory delegate
# ---------------------------------------------------------------------------


class TestPineconeIndexDelegate:
    def test_forwards(self) -> None:
        pc = Pinecone(api_key="test-key")
        mock_index = MagicMock()
        pc.index = mock_index  # type: ignore[method-assign]
        pc.Index(name="x", host="h")
        mock_index.assert_called_once_with(name="x", host="h", pool_threads=None)

    def test_forwards_pool_threads(self) -> None:
        pc = Pinecone(api_key="test-key")
        mock_index = MagicMock()
        pc.index = mock_index  # type: ignore[method-assign]
        pc.Index(name="x", host="h", pool_threads=20)  # type: ignore[call-arg]
        mock_index.assert_called_once_with(name="x", host="h", pool_threads=20)

    def test_rejects_unknown_kwargs(self) -> None:
        pc = Pinecone(api_key="test-key")
        with pytest.raises(TypeError, match="unexpected keyword arguments"):
            pc.Index(name="x", host="h", bogus=True)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# IndexAsyncio() factory delegate
# ---------------------------------------------------------------------------


class TestPineconeIndexAsyncioDelegate:
    def test_constructs_async_index(self) -> None:
        pc = Pinecone(api_key="test-key")
        with patch("pinecone.async_client.async_index.AsyncIndex") as mock_async_index:
            pc.IndexAsyncio(host="my-index.svc.pinecone.io")
        mock_async_index.assert_called_once()
        _, kwargs = mock_async_index.call_args
        assert kwargs["host"] == "my-index.svc.pinecone.io"


# ---------------------------------------------------------------------------
# Lazy namespace property first-access and caching
# ---------------------------------------------------------------------------


class TestLazyNamespaces:
    def test_collections_lazy_instantiation(self) -> None:
        from pinecone.client.collections import Collections

        pc = Pinecone(api_key="test-key")
        assert pc._collections is None
        result = pc.collections
        assert isinstance(result, Collections)
        assert pc._collections is result

    def test_collections_cached_on_second_access(self) -> None:
        pc = Pinecone(api_key="test-key")
        first_access = pc.collections
        assert pc.collections is first_access

    def test_backups_lazy_instantiation(self) -> None:
        from pinecone.client.backups import Backups

        pc = Pinecone(api_key="test-key")
        assert pc._backups is None
        result = pc.backups
        assert isinstance(result, Backups)
        assert pc._backups is result

    def test_backups_cached_on_second_access(self) -> None:
        pc = Pinecone(api_key="test-key")
        first_access = pc.backups
        assert pc.backups is first_access

    def test_restore_jobs_lazy_instantiation(self) -> None:
        from pinecone.client.restore_jobs import RestoreJobs

        pc = Pinecone(api_key="test-key")
        assert pc._restore_jobs is None
        result = pc.restore_jobs
        assert isinstance(result, RestoreJobs)
        assert pc._restore_jobs is result

    def test_restore_jobs_cached_on_second_access(self) -> None:
        pc = Pinecone(api_key="test-key")
        first_access = pc.restore_jobs
        assert pc.restore_jobs is first_access


# ---------------------------------------------------------------------------
# pool_threads= backcompat shim (BCG-020)
# ---------------------------------------------------------------------------


class TestPoolThreadsBackcompat:
    def test_pool_threads_kwarg_accepted_silently(self) -> None:
        pc = Pinecone(api_key="x", pool_threads=4)
        assert pc is not None

    def test_pool_threads_kwarg_stored_as_attribute(self) -> None:
        pc = Pinecone(api_key="x", pool_threads=4)
        assert pc._legacy_pool_threads == 4  # type: ignore[attr-defined]

    def test_pool_threads_kwarg_does_not_warn(self) -> None:
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            Pinecone(api_key="x", pool_threads=4)
        assert len(record) == 0

    def test_unknown_kwarg_still_rejected(self) -> None:
        with pytest.raises(TypeError, match="unexpected keyword arguments"):
            Pinecone(api_key="x", bogus_kwarg=True)
