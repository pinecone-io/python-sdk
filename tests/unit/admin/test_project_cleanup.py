"""Unit tests for Projects._cleanup_project_resources()."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pinecone._internal.config import PineconeConfig
from pinecone._internal.constants import ADMIN_API_VERSION
from pinecone._internal.http_client import HTTPClient
from pinecone.admin.projects import Projects
from pinecone.errors.exceptions import NotFoundError

BASE_URL = "https://api.test.pinecone.io"


@pytest.fixture
def http_client() -> HTTPClient:
    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    return HTTPClient(config, ADMIN_API_VERSION)


@pytest.fixture
def projects(http_client: HTTPClient) -> Projects:
    return Projects(http=http_client)


def _make_index(name: str) -> MagicMock:
    idx = MagicMock()
    idx.name = name
    return idx


def _make_collection(name: str) -> MagicMock:
    col = MagicMock()
    col.name = name
    return col


def _make_backup(backup_id: str) -> MagicMock:
    bk = MagicMock()
    bk.backup_id = backup_id
    return bk


def _make_assistant(name: str) -> MagicMock:
    a = MagicMock()
    a.name = name
    return a


def _empty_project(mock_pc: MagicMock) -> None:
    """Give every resource namespace an explicit empty list.

    A bare MagicMock iterates empty, so without this a loop the cleanup
    forgot to run and a loop over nothing look identical.
    """
    mock_pc.indexes.list.return_value = []
    mock_pc.collections.list.return_value = []
    mock_pc.assistants.list.return_value = []
    mock_pc.backups.list.return_value = []


@patch("pinecone._client.Pinecone")
def test_cleanup_deletes_indexes(mock_pinecone_cls: MagicMock, projects: Projects) -> None:
    """Verify delete is called for each index returned by list."""
    mock_pc = MagicMock()
    mock_pinecone_cls.return_value = mock_pc

    _empty_project(mock_pc)
    mock_pc.indexes.list.return_value = [_make_index("index-1"), _make_index("index-2")]

    projects._cleanup_project_resources(api_key="test-project-key")

    mock_pinecone_cls.assert_called_once_with(api_key="test-project-key")
    assert mock_pc.indexes.delete.call_count == 2
    mock_pc.indexes.delete.assert_any_call("index-1")
    mock_pc.indexes.delete.assert_any_call("index-2")
    mock_pc.close.assert_called_once()


@patch("pinecone._client.Pinecone")
def test_cleanup_deletes_backups(mock_pinecone_cls: MagicMock, projects: Projects) -> None:
    """Verify delete is called for each backup returned by list."""
    mock_pc = MagicMock()
    mock_pinecone_cls.return_value = mock_pc

    _empty_project(mock_pc)
    mock_pc.backups.list.return_value = [_make_backup("bk-001"), _make_backup("bk-002")]

    projects._cleanup_project_resources(api_key="test-project-key")

    assert mock_pc.backups.delete.call_count == 2
    mock_pc.backups.delete.assert_any_call(backup_id="bk-001")
    mock_pc.backups.delete.assert_any_call(backup_id="bk-002")
    mock_pc.close.assert_called_once()


@patch("pinecone._client.Pinecone")
def test_cleanup_deletes_collections(mock_pinecone_cls: MagicMock, projects: Projects) -> None:
    """Verify delete is called for each collection returned by list."""
    mock_pc = MagicMock()
    mock_pinecone_cls.return_value = mock_pc

    _empty_project(mock_pc)
    mock_pc.collections.list.return_value = [_make_collection("col-a"), _make_collection("col-b")]

    projects._cleanup_project_resources(api_key="test-project-key")

    assert mock_pc.collections.delete.call_count == 2
    mock_pc.collections.delete.assert_any_call("col-a")
    mock_pc.collections.delete.assert_any_call("col-b")
    mock_pc.close.assert_called_once()


@patch("pinecone._client.Pinecone")
def test_cleanup_deletes_assistants(mock_pinecone_cls: MagicMock, projects: Projects) -> None:
    """Verify delete is called for each assistant returned by list."""
    mock_pc = MagicMock()
    mock_pinecone_cls.return_value = mock_pc

    _empty_project(mock_pc)
    mock_pc.assistants.list.return_value = [
        _make_assistant("assistant-a"),
        _make_assistant("assistant-b"),
    ]

    projects._cleanup_project_resources(api_key="test-project-key")

    assert mock_pc.assistants.delete.call_count == 2
    mock_pc.assistants.delete.assert_any_call(name="assistant-a")
    mock_pc.assistants.delete.assert_any_call(name="assistant-b")
    mock_pc.close.assert_called_once()


@patch("pinecone._client.Pinecone")
def test_cleanup_waits_for_each_assistant_to_disappear(
    mock_pinecone_cls: MagicMock, projects: Projects
) -> None:
    """The delete must not opt out of polling.

    An assistant that has only been asked to terminate still counts against
    the project, so passing ``timeout=-1`` here would reintroduce the 412
    this cleanup exists to avoid.
    """
    mock_pc = MagicMock()
    mock_pinecone_cls.return_value = mock_pc

    _empty_project(mock_pc)
    mock_pc.assistants.list.return_value = [_make_assistant("assistant-a")]

    projects._cleanup_project_resources(api_key="test-project-key")

    _, kwargs = mock_pc.assistants.delete.call_args
    assert kwargs.get("timeout") is None


@patch("pinecone._client.Pinecone")
def test_cleanup_follows_server_blocker_order(
    mock_pinecone_cls: MagicMock, projects: Projects
) -> None:
    """Resources are cleared in the order the project delete checks them."""
    mock_pc = MagicMock()
    mock_pinecone_cls.return_value = mock_pc

    _empty_project(mock_pc)
    mock_pc.indexes.list.return_value = [_make_index("index-1")]
    mock_pc.collections.list.return_value = [_make_collection("col-a")]
    mock_pc.assistants.list.return_value = [_make_assistant("assistant-a")]
    mock_pc.backups.list.return_value = [_make_backup("bk-001")]

    order: list[str] = []
    mock_pc.indexes.delete.side_effect = lambda *a, **kw: order.append("index")
    mock_pc.collections.delete.side_effect = lambda *a, **kw: order.append("collection")
    mock_pc.assistants.delete.side_effect = lambda *a, **kw: order.append("assistant")
    mock_pc.backups.delete.side_effect = lambda *a, **kw: order.append("backup")

    projects._cleanup_project_resources(api_key="test-project-key")

    assert order == ["index", "collection", "assistant", "backup"]


@patch("pinecone._client.Pinecone")
def test_cleanup_ignores_not_found(mock_pinecone_cls: MagicMock, projects: Projects) -> None:
    """Verify NotFoundError during deletion is swallowed, not propagated."""
    mock_pc = MagicMock()
    mock_pinecone_cls.return_value = mock_pc

    mock_pc.indexes.list.return_value = [_make_index("index-gone")]
    mock_pc.collections.list.return_value = [_make_collection("col-gone")]
    mock_pc.assistants.list.return_value = [_make_assistant("assistant-gone")]
    mock_pc.backups.list.return_value = [_make_backup("bk-gone")]

    mock_pc.indexes.delete.side_effect = NotFoundError(message="not found")
    mock_pc.collections.delete.side_effect = NotFoundError(message="not found")
    mock_pc.assistants.delete.side_effect = NotFoundError(message="not found")
    mock_pc.backups.delete.side_effect = NotFoundError(message="not found")

    # Should not raise
    projects._cleanup_project_resources(api_key="test-project-key")

    mock_pc.indexes.delete.assert_called_once_with("index-gone")
    mock_pc.collections.delete.assert_called_once_with("col-gone")
    mock_pc.assistants.delete.assert_called_once_with(name="assistant-gone")
    mock_pc.backups.delete.assert_called_once_with(backup_id="bk-gone")
    mock_pc.close.assert_called_once()


@patch("pinecone._client.Pinecone")
def test_cleanup_empty_project(mock_pinecone_cls: MagicMock, projects: Projects) -> None:
    """Verify cleanup completes successfully when project has no resources."""
    mock_pc = MagicMock()
    mock_pinecone_cls.return_value = mock_pc

    _empty_project(mock_pc)

    projects._cleanup_project_resources(api_key="test-project-key")

    mock_pc.indexes.delete.assert_not_called()
    mock_pc.collections.delete.assert_not_called()
    mock_pc.assistants.delete.assert_not_called()
    mock_pc.backups.delete.assert_not_called()
    mock_pc.close.assert_called_once()


@patch("pinecone._client.Pinecone")
def test_cleanup_closes_client_on_error(mock_pinecone_cls: MagicMock, projects: Projects) -> None:
    """Verify the Pinecone client is closed even if an unexpected error occurs."""
    mock_pc = MagicMock()
    mock_pinecone_cls.return_value = mock_pc

    mock_pc.indexes.list.side_effect = RuntimeError("unexpected")

    with pytest.raises(RuntimeError, match="unexpected"):
        projects._cleanup_project_resources(api_key="test-project-key")

    mock_pc.close.assert_called_once()
