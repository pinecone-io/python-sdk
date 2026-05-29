"""Unit tests for AsyncRestoreJobs namespace and AsyncPinecone.create_index_from_backup."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, patch

import httpx
import orjson
import pytest
import respx

from pinecone._internal.config import PineconeConfig
from pinecone._internal.constants import CONTROL_PLANE_API_VERSION
from pinecone._internal.http_client import AsyncHTTPClient
from pinecone.async_client.pinecone import AsyncPinecone
from pinecone.async_client.restore_jobs import AsyncRestoreJobs
from pinecone.errors.exceptions import ValidationError
from pinecone.models.backups.list import RestoreJobList
from pinecone.models.backups.model import CreateIndexFromBackupResponse, RestoreJobModel
from pinecone.models.indexes.index import IndexModel
from tests.factories import make_index_response, make_restore_job_response

BASE_URL = "https://api.test.pinecone.io"
DEFAULT_BASE_URL = "https://api.pinecone.io"


@pytest.fixture
async def async_http_client() -> AsyncGenerator[AsyncHTTPClient]:
    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    client = AsyncHTTPClient(config, CONTROL_PLANE_API_VERSION)
    yield client
    await client.close()


@pytest.fixture
def async_restore_jobs(async_http_client: AsyncHTTPClient) -> AsyncRestoreJobs:
    return AsyncRestoreJobs(http=async_http_client)


@pytest.fixture
def pc() -> AsyncPinecone:
    return AsyncPinecone(api_key="test-key")


# ---------------------------------------------------------------------------
# list()
# ---------------------------------------------------------------------------


@respx.mock
async def test_async_list_restore_jobs(async_restore_jobs: AsyncRestoreJobs) -> None:
    respx.get(f"{BASE_URL}/restore-jobs").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    make_restore_job_response(),
                    make_restore_job_response(restore_job_id="rj-second"),
                ],
            },
        ),
    )

    result = await async_restore_jobs.list()

    assert isinstance(result, RestoreJobList)
    assert len(result) == 2


@respx.mock
async def test_async_list_restore_jobs_no_limit_param_when_default(
    async_restore_jobs: AsyncRestoreJobs,
) -> None:
    route = respx.get(f"{BASE_URL}/restore-jobs").mock(
        return_value=httpx.Response(200, json={"data": []}),
    )

    await async_restore_jobs.list()

    request = route.calls[0].request
    assert "limit" not in request.url.params


@respx.mock
async def test_async_list_restore_jobs_with_pagination(
    async_restore_jobs: AsyncRestoreJobs,
) -> None:
    route = respx.get(f"{BASE_URL}/restore-jobs").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [make_restore_job_response()],
                "pagination": {"next": "token-next"},
            },
        ),
    )

    result = await async_restore_jobs.list(limit=5, pagination_token="token-xyz")

    assert isinstance(result, RestoreJobList)
    assert len(result) == 1

    # Verify query params
    request = route.calls[0].request
    assert request.url.params["limit"] == "5"
    assert request.url.params["paginationToken"] == "token-xyz"

    # Verify pagination token is extracted
    assert result.pagination is not None
    assert result.pagination.next == "token-next"


# ---------------------------------------------------------------------------
# describe()
# ---------------------------------------------------------------------------


@respx.mock
async def test_async_describe_restore_job(async_restore_jobs: AsyncRestoreJobs) -> None:
    job_id = "rj-670e8400-e29b-41d4-a716-446655440001"
    respx.get(f"{BASE_URL}/restore-jobs/{job_id}").mock(
        return_value=httpx.Response(
            200,
            json=make_restore_job_response(completed_at="2025-02-04T12:15:00Z"),
        ),
    )

    result = await async_restore_jobs.describe(job_id=job_id)

    assert isinstance(result, RestoreJobModel)
    assert result.restore_job_id == job_id
    assert result.completed_at == "2025-02-04T12:15:00Z"


async def test_async_describe_empty_id_raises(async_restore_jobs: AsyncRestoreJobs) -> None:
    with pytest.raises(ValidationError) as exc_info:
        await async_restore_jobs.describe(job_id="")

    assert "job_id" in str(exc_info.value)
    assert "non-empty" in str(exc_info.value)


@respx.mock
async def test_async_restore_job_created_at_optional(async_restore_jobs: AsyncRestoreJobs) -> None:
    """list() must not crash when the backend omits or nulls created_at."""
    payload = make_restore_job_response()
    del payload["created_at"]
    respx.get(f"{BASE_URL}/restore-jobs").mock(
        return_value=httpx.Response(200, json={"data": [payload]}),
    )

    result = await async_restore_jobs.list()

    assert len(result) == 1
    job = result[0]
    assert job.created_at is None or isinstance(job.created_at, str)


@respx.mock
async def test_async_restore_job_created_at_null(async_restore_jobs: AsyncRestoreJobs) -> None:
    """list() decodes created_at: null as None without raising DecodeError."""
    payload = make_restore_job_response(created_at=None)
    respx.get(f"{BASE_URL}/restore-jobs").mock(
        return_value=httpx.Response(200, json={"data": [payload]}),
    )

    result = await async_restore_jobs.list()

    assert result[0].created_at is None


# ---------------------------------------------------------------------------
# create_index_from_backup — basic success
# ---------------------------------------------------------------------------


@respx.mock
async def test_async_create_index_from_backup_basic(pc: AsyncPinecone) -> None:
    """POST creates the index, then polling returns a ready IndexModel."""
    respx.post(f"{DEFAULT_BASE_URL}/backups/bk-123/create-index").mock(
        return_value=httpx.Response(
            202,
            json={"restore_job_id": "rj-1", "index_id": "idx-1"},
        ),
    )
    respx.get(f"{DEFAULT_BASE_URL}/indexes/restored-index").mock(
        return_value=httpx.Response(200, json=make_index_response(name="restored-index")),
    )

    result = await pc.create_index_from_backup(name="restored-index", backup_id="bk-123")

    assert isinstance(result, IndexModel)
    assert result.name == "restored-index"


# ---------------------------------------------------------------------------
# create_index_from_backup — tags and deletion protection
# ---------------------------------------------------------------------------


@respx.mock
async def test_async_create_index_from_backup_with_tags_and_protection(pc: AsyncPinecone) -> None:
    """Tags and deletion_protection appear in the request body."""
    route = respx.post(f"{DEFAULT_BASE_URL}/backups/bk-456/create-index").mock(
        return_value=httpx.Response(
            202,
            json={"restore_job_id": "rj-2", "index_id": "idx-2"},
        ),
    )
    respx.get(f"{DEFAULT_BASE_URL}/indexes/my-restored").mock(
        return_value=httpx.Response(200, json=make_index_response(name="my-restored")),
    )

    await pc.create_index_from_backup(
        name="my-restored",
        backup_id="bk-456",
        deletion_protection="enabled",
        tags={"env": "prod"},
    )

    request = route.calls[0].request
    import orjson

    body = orjson.loads(request.content)
    assert body["name"] == "my-restored"
    assert body["deletion_protection"] == "enabled"
    assert body["tags"] == {"env": "prod"}


# ---------------------------------------------------------------------------
# create_index_from_backup — no-poll (timeout=-1)
# ---------------------------------------------------------------------------


@respx.mock
async def test_async_create_index_from_backup_no_poll(pc: AsyncPinecone) -> None:
    """When timeout=-1, returns CreateIndexFromBackupResponse immediately without polling."""
    respx.post(f"{DEFAULT_BASE_URL}/backups/bk-789/create-index").mock(
        return_value=httpx.Response(
            202,
            json={"restore_job_id": "rj-3", "index_id": "idx-3"},
        ),
    )

    result = await pc.create_index_from_backup(name="quick-restore", backup_id="bk-789", timeout=-1)

    assert isinstance(result, CreateIndexFromBackupResponse)
    assert result.restore_job_id == "rj-3"
    assert result.index_id == "idx-3"


# ---------------------------------------------------------------------------
# create_index_from_backup — polling
# ---------------------------------------------------------------------------


@patch("pinecone._internal.indexes_helpers.asyncio.sleep", new_callable=AsyncMock)
@respx.mock
async def test_async_create_index_from_backup_polls_until_ready(
    mock_sleep: object, pc: AsyncPinecone
) -> None:
    """Describe is called multiple times until the index becomes ready."""
    respx.post(f"{DEFAULT_BASE_URL}/backups/bk-poll/create-index").mock(
        return_value=httpx.Response(
            202,
            json={"restore_job_id": "rj-4", "index_id": "idx-4"},
        ),
    )
    not_ready = make_index_response(
        name="poll-index",
        status={"ready": False, "state": "Initializing"},
    )
    ready = make_index_response(
        name="poll-index",
        status={"ready": True, "state": "Ready"},
    )
    describe_route = respx.get(f"{DEFAULT_BASE_URL}/indexes/poll-index").mock(
        side_effect=[
            httpx.Response(200, json=not_ready),
            httpx.Response(200, json=ready),
        ]
    )

    result = await pc.create_index_from_backup(name="poll-index", backup_id="bk-poll", timeout=60)

    assert isinstance(result, IndexModel)
    assert result.name == "poll-index"
    # Describe should be called at least 2 times (first not-ready, then ready)
    assert describe_route.call_count >= 2


# ---------------------------------------------------------------------------
# create_index_from_backup — validation errors
# ---------------------------------------------------------------------------


async def test_async_create_index_from_backup_empty_name_raises(pc: AsyncPinecone) -> None:
    with pytest.raises(ValidationError) as exc_info:
        await pc.create_index_from_backup(name="", backup_id="bk-123")

    assert "name" in str(exc_info.value)
    assert "non-empty" in str(exc_info.value)


async def test_async_create_index_from_backup_empty_backup_id_raises(pc: AsyncPinecone) -> None:
    with pytest.raises(ValidationError) as exc_info:
        await pc.create_index_from_backup(name="my-index", backup_id="")

    assert "backup_id" in str(exc_info.value)
    assert "non-empty" in str(exc_info.value)


# ---------------------------------------------------------------------------
# AsyncPinecone.restore_jobs property
# ---------------------------------------------------------------------------


def test_async_pinecone_restore_jobs_property() -> None:
    pc = AsyncPinecone(api_key="test-key")
    rj = pc.restore_jobs
    assert isinstance(rj, AsyncRestoreJobs)
    # Verify lazy caching — same instance returned
    assert pc.restore_jobs is rj


# ---------------------------------------------------------------------------
# repr()
# ---------------------------------------------------------------------------


def test_async_restore_jobs_repr() -> None:
    from unittest.mock import MagicMock

    restore_jobs = AsyncRestoreJobs(http=MagicMock())
    assert repr(restore_jobs) == "AsyncRestoreJobs()"
