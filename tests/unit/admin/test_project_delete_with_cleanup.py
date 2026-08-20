"""Unit tests for Projects.delete_with_cleanup()."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pinecone._internal.config import PineconeConfig
from pinecone._internal.constants import ADMIN_API_VERSION
from pinecone._internal.http_client import HTTPClient
from pinecone.admin.projects import Projects
from pinecone.errors.exceptions import ApiError, ForbiddenError, PineconeError

BASE_URL = "https://api.test.pinecone.io"


def _make_temp_key(key_id: str = "tmpkey-001", value: str = "pcsk_secret") -> MagicMock:
    """Create a mock APIKeyWithSecret."""
    key_model = MagicMock()
    key_model.id = key_id

    temp_key = MagicMock()
    temp_key.key = key_model
    temp_key.value = value
    return temp_key


@pytest.fixture
def http_client() -> HTTPClient:
    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    return HTTPClient(config, ADMIN_API_VERSION)


@pytest.fixture
def mock_admin() -> MagicMock:
    admin = MagicMock()
    admin.api_keys = MagicMock()
    return admin


@pytest.fixture
def projects(http_client: HTTPClient, mock_admin: MagicMock) -> Projects:
    return Projects(http=http_client, admin=mock_admin)


def test_delete_with_cleanup_happy_path(projects: Projects, mock_admin: MagicMock) -> None:
    """Verify the full happy path: create temp key, cleanup, delete key, delete project."""
    temp_key = _make_temp_key()
    mock_admin.api_keys.create.return_value = temp_key

    with (
        patch.object(projects, "_cleanup_project_resources") as mock_cleanup,
        patch.object(projects, "delete") as mock_delete,
    ):
        projects.delete_with_cleanup(project_id="proj-123")

        # Temp key created with correct params
        mock_admin.api_keys.create.assert_called_once_with(
            project_id="proj-123",
            name="_cleanup_temp_key",
            roles=["ProjectEditor"],
        )

        # Cleanup called once with the secret value
        mock_cleanup.assert_called_once_with(api_key="pcsk_secret")

        # Temp key deleted in finally block
        mock_admin.api_keys.delete.assert_called_once_with(api_key_id="tmpkey-001")

        # Project deleted after cleanup
        mock_delete.assert_called_once_with(project_id="proj-123")


def test_delete_with_cleanup_retries_on_failure(projects: Projects, mock_admin: MagicMock) -> None:
    """Verify cleanup retries on failure and succeeds on third attempt."""
    temp_key = _make_temp_key()
    mock_admin.api_keys.create.return_value = temp_key

    call_count = 0

    def cleanup_side_effect(*, api_key: str) -> None:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise RuntimeError("transient failure")

    with (
        patch.object(
            projects, "_cleanup_project_resources", side_effect=cleanup_side_effect
        ) as mock_cleanup,
        patch.object(projects, "delete") as mock_delete,
        patch("pinecone.admin.projects.time.sleep") as mock_sleep,
    ):
        projects.delete_with_cleanup(project_id="proj-123", retry_delay=0.0)

        # Cleanup called 3 times (failed twice, succeeded on third)
        assert mock_cleanup.call_count == 3

        # Sleep called between retries (2 times for 2 failures)
        assert mock_sleep.call_count == 2

        # Temp key still deleted
        mock_admin.api_keys.delete.assert_called_once_with(api_key_id="tmpkey-001")

        # Project deleted after successful cleanup
        mock_delete.assert_called_once_with(project_id="proj-123")


def test_delete_with_cleanup_cleans_up_temp_key_on_failure(
    projects: Projects, mock_admin: MagicMock
) -> None:
    """Verify temp key is deleted even when all cleanup attempts fail."""
    temp_key = _make_temp_key()
    mock_admin.api_keys.create.return_value = temp_key

    with (
        patch.object(
            projects, "_cleanup_project_resources", side_effect=RuntimeError("permanent failure")
        ) as mock_cleanup,
        patch.object(projects, "delete") as mock_delete,
        patch("pinecone.admin.projects.time.sleep"),
    ):
        with pytest.raises(RuntimeError, match="permanent failure"):
            projects.delete_with_cleanup(project_id="proj-123", max_attempts=3, retry_delay=0.0)

        # All attempts made
        assert mock_cleanup.call_count == 3

        # Temp key STILL deleted (finally block)
        mock_admin.api_keys.delete.assert_called_once_with(api_key_id="tmpkey-001")

        # Project NOT deleted since cleanup failed
        mock_delete.assert_not_called()


def test_delete_with_cleanup_no_admin_raises(http_client: HTTPClient) -> None:
    """Verify PineconeError raised when Projects has no admin back-reference."""
    projects = Projects(http=http_client)

    with pytest.raises(PineconeError, match="delete_with_cleanup requires an Admin"):
        projects.delete_with_cleanup(project_id="proj-123")


def test_delete_with_cleanup_key_deletion_failure_does_not_block_project_delete(
    projects: Projects, mock_admin: MagicMock
) -> None:
    """Verify project deletion still proceeds when temp key deletion fails after successful cleanup."""
    temp_key = _make_temp_key()
    mock_admin.api_keys.create.return_value = temp_key
    mock_admin.api_keys.delete.side_effect = ApiError("server error", status_code=500)

    with (
        patch.object(projects, "_cleanup_project_resources"),
        patch.object(projects, "delete") as mock_delete,
    ):
        # Should not raise — key deletion error is swallowed
        projects.delete_with_cleanup(project_id="proj-123")

        # Project deletion still called despite key deletion failure
        mock_delete.assert_called_once_with(project_id="proj-123")


def test_delete_with_cleanup_original_error_preserved_when_key_deletion_also_fails(
    projects: Projects, mock_admin: MagicMock
) -> None:
    """Verify the original cleanup error propagates when both cleanup and key deletion fail."""
    temp_key = _make_temp_key()
    mock_admin.api_keys.create.return_value = temp_key
    mock_admin.api_keys.delete.side_effect = ApiError("key delete error", status_code=500)

    original_error = RuntimeError("original cleanup failure")

    with (
        patch.object(projects, "_cleanup_project_resources", side_effect=original_error),
        patch.object(projects, "delete") as mock_delete,
        patch("pinecone.admin.projects.time.sleep"),
    ):
        with pytest.raises(RuntimeError, match="original cleanup failure"):
            projects.delete_with_cleanup(project_id="proj-123", max_attempts=1)

        # Project NOT deleted since cleanup failed
        mock_delete.assert_not_called()


def test_delete_with_cleanup_quota_full_raises_actionable_error(
    projects: Projects, mock_admin: MagicMock
) -> None:
    """A quota-full project gets a 403 naming the quota, and nothing is touched."""
    mock_admin.api_keys.create.side_effect = ForbiddenError(
        "You have reached the maximum of 5 API keys allowed for this project.",
        status_code=403,
        body={"error": {"code": "PERMISSION_DENIED"}},
        error_code="PERMISSION_DENIED",
        request_id="req-quota-1",
    )

    with (
        patch.object(projects, "_cleanup_project_resources") as mock_cleanup,
        patch.object(projects, "delete") as mock_delete,
        patch("pinecone.admin.projects.time.sleep") as mock_sleep,
    ):
        with pytest.raises(ForbiddenError) as excinfo:
            projects.delete_with_cleanup(project_id="proj-123")

        mock_cleanup.assert_not_called()
        mock_admin.api_keys.delete.assert_not_called()
        mock_delete.assert_not_called()
        mock_sleep.assert_not_called()

    err = excinfo.value
    message = err.message

    assert "delete_with_cleanup" in message
    assert "project_id='proj-123'" in message
    assert "API-key quota" in message
    assert "maximum of 5 API keys" in message
    assert "Nothing was deleted" in message
    assert "admin.api_keys.list(project_id='proj-123')" in message
    assert "admin.api_keys.delete(api_key_id=...)" in message

    assert err.status_code == 403
    assert err.error_code == "PERMISSION_DENIED"
    assert err.request_id == "req-quota-1"
    assert err.body == {"error": {"code": "PERMISSION_DENIED"}}
    assert isinstance(err.__cause__, ForbiddenError)


def test_delete_with_cleanup_non_403_key_create_failure_propagates_unchanged(
    projects: Projects, mock_admin: MagicMock
) -> None:
    """Only 403s get the quota treatment; other key-create failures pass through."""
    original = ApiError("boom", status_code=500)
    mock_admin.api_keys.create.side_effect = original

    with (
        patch.object(projects, "_cleanup_project_resources") as mock_cleanup,
        patch.object(projects, "delete") as mock_delete,
    ):
        with pytest.raises(ApiError) as excinfo:
            projects.delete_with_cleanup(project_id="proj-123")

        assert excinfo.value is original
        mock_cleanup.assert_not_called()
        mock_delete.assert_not_called()


def test_delete_with_cleanup_call_order(projects: Projects, mock_admin: MagicMock) -> None:
    """Verify operations happen in the correct order."""
    temp_key = _make_temp_key()
    mock_admin.api_keys.create.return_value = temp_key

    order: list[str] = []

    def track_cleanup(*, api_key: str) -> None:
        order.append("cleanup")

    def track_delete_key(*, api_key_id: str) -> None:
        order.append("delete_key")

    def track_delete_project(*, project_id: str) -> None:
        order.append("delete_project")

    mock_admin.api_keys.delete.side_effect = track_delete_key

    with (
        patch.object(projects, "_cleanup_project_resources", side_effect=track_cleanup),
        patch.object(projects, "delete", side_effect=track_delete_project),
    ):
        projects.delete_with_cleanup(project_id="proj-123")

    assert order == ["cleanup", "delete_key", "delete_project"]
