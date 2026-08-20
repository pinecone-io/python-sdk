"""Query composition for every offset-token listing (#252).

The control-plane pagination token is a base64 ``{limit, offset}`` pair. A
``limit`` sent alongside a token overrides the token's limit while keeping its
offset, so the next page starts where the *old* page size put the boundary and
then runs for the *new* length -- rows are skipped or served twice
(``svc-global-apis/src/http/pagination.rs:47-51`` @ ``f6fd0a4019``).

So: one assertion per operation, sync and async, that a request carrying
``paginationToken`` carries no ``limit`` -- and its mirror, that a request
without a token still carries the caller's ``limit``, so the fix did not
simply stop honouring page sizes.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

import httpx
import pytest
import respx

from pinecone._internal.config import PineconeConfig
from pinecone._internal.constants import CONTROL_PLANE_API_VERSION
from pinecone._internal.http_client import AsyncHTTPClient, HTTPClient
from pinecone.async_client.backups import AsyncBackups
from pinecone.async_client.indexes import AsyncIndexes
from pinecone.async_client.restore_jobs import AsyncRestoreJobs
from pinecone.client.backup_schedules import BackupSchedules
from pinecone.client.backups import Backups
from pinecone.client.indexes import Indexes
from pinecone.client.restore_jobs import RestoreJobs
from pinecone.preview.async_indexes import AsyncPreviewIndexes
from pinecone.preview.indexes import PreviewIndexes

BASE_URL = "https://api.test.pinecone.io"
SCHEDULE_ID = "e88f7273-42aa-47e9-af73-593827136867"

_BACKUP: dict[str, Any] = {
    "backup_id": "bkp_123abc",
    "source_index_name": "my-index",
    "source_index_id": "idx_456",
    "name": "backup_2025_03_15",
    "status": "Ready",
    "cloud": "aws",
    "region": "us-east-1",
    "record_count": 1,
    "namespace_count": 1,
    "size_bytes": 1,
    "created_at": "2025-03-15T10:30:00Z",
}

_RESTORE_JOB: dict[str, Any] = {
    "restore_job_id": "rj_123",
    "backup_id": "bkp_123abc",
    "target_index_name": "restored",
    "target_index_id": "idx_789",
    "status": "Completed",
    "created_at": "2025-03-15T10:30:00Z",
    "percent_complete": 100.0,
}

_SCHEDULE: dict[str, Any] = {
    "schedule_id": SCHEDULE_ID,
    "name": "daily-compliance-backup",
    "index_id": "8cbf7ba6-4135-438e-a3c3-4a89a3298905",
    "project_id": "71ce31ea-75f7-45d6-a147-ef67f661a1b0",
    "schedule_type": "time-based",
    "frequency": "daily",
    "retention_expire_after_days": 90,
    "enabled": True,
    "next_scheduled_run": "2026-04-03T06:00:00+00:00",
    "created_at": "2026-04-02T18:22:56.712605+00:00",
}


def _http() -> HTTPClient:
    return HTTPClient(PineconeConfig(api_key="test-key", host=BASE_URL), CONTROL_PLANE_API_VERSION)


@pytest.fixture
async def async_http() -> AsyncGenerator[AsyncHTTPClient]:
    client = AsyncHTTPClient(
        PineconeConfig(api_key="test-key", host=BASE_URL), CONTROL_PLANE_API_VERSION
    )
    yield client
    await client.close()


def _two_pages(row: dict[str, Any]) -> list[httpx.Response]:
    return [
        httpx.Response(200, json={"data": [row], "pagination": {"next": "tok-2"}}),
        httpx.Response(200, json={"data": [row], "pagination": None}),
    ]


class TestSinglePageListings:
    """Operations that hand the caller one page and its token."""

    @respx.mock
    def test_project_backups_list(self) -> None:
        route = respx.get(f"{BASE_URL}/backups").mock(
            return_value=httpx.Response(200, json={"data": [_BACKUP]})
        )

        Backups(http=_http()).list(limit=5, pagination_token="tok")

        params = route.calls.last.request.url.params
        assert params["paginationToken"] == "tok"
        assert "limit" not in params

    @respx.mock
    def test_index_backups_list_keeps_include_deleted(self) -> None:
        route = respx.get(f"{BASE_URL}/indexes/my-index/backups").mock(
            return_value=httpx.Response(200, json={"data": [_BACKUP]})
        )

        Backups(http=_http()).list(
            index_name="my-index", limit=5, pagination_token="tok", include_deleted=True
        )

        params = route.calls.last.request.url.params
        assert params["paginationToken"] == "tok"
        assert params["include_deleted"] == "true"
        assert "limit" not in params

    @respx.mock
    def test_restore_jobs_list(self) -> None:
        route = respx.get(f"{BASE_URL}/restore-jobs").mock(
            return_value=httpx.Response(200, json={"data": [_RESTORE_JOB]})
        )

        RestoreJobs(http=_http()).list(limit=5, pagination_token="tok")

        params = route.calls.last.request.url.params
        assert params["paginationToken"] == "tok"
        assert "limit" not in params

    @respx.mock
    def test_schedules_list(self) -> None:
        route = respx.get(f"{BASE_URL}/indexes/my-index/backup-schedules").mock(
            return_value=httpx.Response(200, json={"data": [_SCHEDULE]})
        )

        BackupSchedules(http=_http()).list(index_name="my-index", limit=5, pagination_token="tok")

        params = route.calls.last.request.url.params
        assert params["paginationToken"] == "tok"
        assert "limit" not in params

    @respx.mock
    def test_schedule_history(self) -> None:
        route = respx.get(f"{BASE_URL}/backup-schedules/{SCHEDULE_ID}/history").mock(
            return_value=httpx.Response(200, json={"data": []})
        )

        BackupSchedules(http=_http()).history(
            schedule_id=SCHEDULE_ID, limit=5, pagination_token="tok"
        )

        params = route.calls.last.request.url.params
        assert params["paginationToken"] == "tok"
        assert "limit" not in params

    @respx.mock
    async def test_async_project_backups_list(self, async_http: AsyncHTTPClient) -> None:
        route = respx.get(f"{BASE_URL}/backups").mock(
            return_value=httpx.Response(200, json={"data": [_BACKUP]})
        )

        await AsyncBackups(http=async_http).list(limit=5, pagination_token="tok")

        params = route.calls.last.request.url.params
        assert params["paginationToken"] == "tok"
        assert "limit" not in params

    @respx.mock
    async def test_async_index_backups_list_keeps_include_deleted(
        self, async_http: AsyncHTTPClient
    ) -> None:
        route = respx.get(f"{BASE_URL}/indexes/my-index/backups").mock(
            return_value=httpx.Response(200, json={"data": [_BACKUP]})
        )

        await AsyncBackups(http=async_http).list(
            index_name="my-index", limit=5, pagination_token="tok", include_deleted=True
        )

        params = route.calls.last.request.url.params
        assert params["paginationToken"] == "tok"
        assert params["include_deleted"] == "true"
        assert "limit" not in params

    @respx.mock
    async def test_async_restore_jobs_list(self, async_http: AsyncHTTPClient) -> None:
        route = respx.get(f"{BASE_URL}/restore-jobs").mock(
            return_value=httpx.Response(200, json={"data": [_RESTORE_JOB]})
        )

        await AsyncRestoreJobs(http=async_http).list(limit=5, pagination_token="tok")

        params = route.calls.last.request.url.params
        assert params["paginationToken"] == "tok"
        assert "limit" not in params


class TestLimitStillTravelsWithoutAToken:
    """The mirror: the fix must not have stopped honouring page sizes."""

    @respx.mock
    def test_project_backups_list(self) -> None:
        route = respx.get(f"{BASE_URL}/backups").mock(
            return_value=httpx.Response(200, json={"data": [_BACKUP]})
        )

        Backups(http=_http()).list(limit=5)

        assert route.calls.last.request.url.params["limit"] == "5"

    @respx.mock
    def test_restore_jobs_list(self) -> None:
        route = respx.get(f"{BASE_URL}/restore-jobs").mock(
            return_value=httpx.Response(200, json={"data": [_RESTORE_JOB]})
        )

        RestoreJobs(http=_http()).list(limit=5)

        assert route.calls.last.request.url.params["limit"] == "5"

    @respx.mock
    async def test_async_project_backups_list(self, async_http: AsyncHTTPClient) -> None:
        route = respx.get(f"{BASE_URL}/backups").mock(
            return_value=httpx.Response(200, json={"data": [_BACKUP]})
        )

        await AsyncBackups(http=async_http).list(limit=5)

        assert route.calls.last.request.url.params["limit"] == "5"

    @respx.mock
    async def test_async_restore_jobs_list(self, async_http: AsyncHTTPClient) -> None:
        route = respx.get(f"{BASE_URL}/restore-jobs").mock(
            return_value=httpx.Response(200, json={"data": [_RESTORE_JOB]})
        )

        await AsyncRestoreJobs(http=async_http).list(limit=5)

        assert route.calls.last.request.url.params["limit"] == "5"


class TestPaginators:
    """Auto-paginating surfaces: page one carries the limit, page two the token.

    The client-side cap still applies, so ``limit`` keeps its documented
    meaning even on the pages that no longer send it.
    """

    @respx.mock
    def test_indexes_list_backups(self) -> None:
        route = respx.get(f"{BASE_URL}/indexes/my-index/backups").mock(
            side_effect=_two_pages(_BACKUP)
        )

        list(Indexes(http=_http()).list_backups("my-index", limit=5))

        assert route.call_count == 2
        assert route.calls[0].request.url.params["limit"] == "5"
        assert route.calls[1].request.url.params["paginationToken"] == "tok-2"
        assert "limit" not in route.calls[1].request.url.params

    @respx.mock
    def test_indexes_list_backups_resumed_from_a_token(self) -> None:
        route = respx.get(f"{BASE_URL}/indexes/my-index/backups").mock(
            return_value=httpx.Response(200, json={"data": [_BACKUP], "pagination": None})
        )

        items = list(
            Indexes(http=_http()).list_backups("my-index", limit=5, pagination_token="resume-me")
        )

        params = route.calls[0].request.url.params
        assert params["paginationToken"] == "resume-me"
        assert "limit" not in params
        assert len(items) == 1

    @respx.mock
    def test_iter_schedules(self) -> None:
        route = respx.get(f"{BASE_URL}/indexes/my-index/backup-schedules").mock(
            side_effect=_two_pages(_SCHEDULE)
        )

        list(BackupSchedules(http=_http()).iter_schedules(index_name="my-index", limit=5))

        assert route.call_count == 2
        assert route.calls[0].request.url.params["limit"] == "5"
        assert route.calls[1].request.url.params["paginationToken"] == "tok-2"
        assert "limit" not in route.calls[1].request.url.params

    @respx.mock
    def test_iter_history(self) -> None:
        history_row = {**_BACKUP, "scheduled_execution_at": "2026-04-03T06:00:00+00:00"}
        route = respx.get(f"{BASE_URL}/backup-schedules/{SCHEDULE_ID}/history").mock(
            side_effect=_two_pages(history_row)
        )

        list(BackupSchedules(http=_http()).iter_history(schedule_id=SCHEDULE_ID, limit=5))

        assert route.call_count == 2
        assert route.calls[0].request.url.params["limit"] == "5"
        assert route.calls[1].request.url.params["paginationToken"] == "tok-2"
        assert "limit" not in route.calls[1].request.url.params

    @respx.mock
    def test_preview_indexes_list_backups(self) -> None:
        route = respx.get(f"{BASE_URL}/indexes/my-index/backups").mock(
            side_effect=_two_pages(_BACKUP)
        )
        indexes = PreviewIndexes(config=PineconeConfig(api_key="test-key", host=BASE_URL))

        list(indexes.list_backups("my-index", limit=5))

        assert route.call_count == 2
        assert route.calls[0].request.url.params["limit"] == "5"
        assert route.calls[1].request.url.params["paginationToken"] == "tok-2"
        assert "limit" not in route.calls[1].request.url.params

    @respx.mock
    async def test_async_indexes_list_backups(self, async_http: AsyncHTTPClient) -> None:
        route = respx.get(f"{BASE_URL}/indexes/my-index/backups").mock(
            side_effect=_two_pages(_BACKUP)
        )

        paginator = AsyncIndexes(http=async_http).list_backups("my-index", limit=5)
        await paginator.to_list()

        assert route.call_count == 2
        assert route.calls[0].request.url.params["limit"] == "5"
        assert route.calls[1].request.url.params["paginationToken"] == "tok-2"
        assert "limit" not in route.calls[1].request.url.params

    @respx.mock
    async def test_async_preview_indexes_list_backups(self) -> None:
        route = respx.get(f"{BASE_URL}/indexes/my-index/backups").mock(
            side_effect=_two_pages(_BACKUP)
        )
        indexes = AsyncPreviewIndexes(config=PineconeConfig(api_key="test-key", host=BASE_URL))

        await indexes.list_backups("my-index", limit=5).to_list()

        assert route.call_count == 2
        assert route.calls[0].request.url.params["limit"] == "5"
        assert route.calls[1].request.url.params["paginationToken"] == "tok-2"
        assert "limit" not in route.calls[1].request.url.params
