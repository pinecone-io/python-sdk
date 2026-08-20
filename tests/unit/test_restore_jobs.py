"""Unit tests for RestoreJobs namespace — list and describe operations."""

from __future__ import annotations

import httpx
import pytest
import respx

from pinecone._internal.config import PineconeConfig
from pinecone._internal.constants import CONTROL_PLANE_API_VERSION
from pinecone._internal.http_client import HTTPClient
from pinecone.client.restore_jobs import RestoreJobs
from pinecone.errors.exceptions import ValidationError
from pinecone.models.backups.list import RestoreJobList
from pinecone.models.backups.model import RestoreJobModel
from tests.factories import make_restore_job_response

BASE_URL = "https://api.test.pinecone.io"


@pytest.fixture
def http_client() -> HTTPClient:
    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    return HTTPClient(config, CONTROL_PLANE_API_VERSION)


@pytest.fixture
def restore_jobs(http_client: HTTPClient) -> RestoreJobs:
    return RestoreJobs(http=http_client)


# ---------------------------------------------------------------------------
# list()
# ---------------------------------------------------------------------------


@respx.mock
def test_list_restore_jobs(restore_jobs: RestoreJobs) -> None:
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

    result = restore_jobs.list()

    assert isinstance(result, RestoreJobList)
    assert len(result) == 2


@respx.mock
def test_list_restore_jobs_no_limit_param_when_default(restore_jobs: RestoreJobs) -> None:
    route = respx.get(f"{BASE_URL}/restore-jobs").mock(
        return_value=httpx.Response(200, json={"data": []}),
    )

    restore_jobs.list()

    request = route.calls[0].request
    assert "limit" not in request.url.params


@respx.mock
def test_list_restore_jobs_with_pagination(restore_jobs: RestoreJobs) -> None:
    route = respx.get(f"{BASE_URL}/restore-jobs").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [make_restore_job_response()],
                "pagination": {"next": "token-next"},
            },
        ),
    )

    result = restore_jobs.list(limit=5, pagination_token="token-xyz")

    assert isinstance(result, RestoreJobList)
    assert len(result) == 1

    request = route.calls[0].request
    assert request.url.params["paginationToken"] == "token-xyz"
    assert "limit" not in request.url.params

    # Verify pagination token is extracted
    assert result.pagination is not None
    assert result.pagination.next == "token-next"


# ---------------------------------------------------------------------------
# describe()
# ---------------------------------------------------------------------------


@respx.mock
def test_describe_restore_job(restore_jobs: RestoreJobs) -> None:
    job_id = "rj-670e8400-e29b-41d4-a716-446655440001"
    respx.get(f"{BASE_URL}/restore-jobs/{job_id}").mock(
        return_value=httpx.Response(
            200,
            json=make_restore_job_response(completed_at="2025-02-04T12:15:00Z"),
        ),
    )

    result = restore_jobs.describe(job_id=job_id)

    assert isinstance(result, RestoreJobModel)
    assert result.restore_job_id == job_id
    assert result.completed_at == "2025-02-04T12:15:00Z"


def test_describe_empty_id_raises(restore_jobs: RestoreJobs) -> None:
    with pytest.raises(ValidationError) as exc_info:
        restore_jobs.describe(job_id="")

    assert "job_id" in str(exc_info.value)
    assert "non-empty" in str(exc_info.value)


@respx.mock
def test_restore_job_created_at_optional(restore_jobs: RestoreJobs) -> None:
    """list() must not crash when the backend omits or nulls created_at (backend nullable field)."""
    payload = make_restore_job_response()
    del payload["created_at"]
    respx.get(f"{BASE_URL}/restore-jobs").mock(
        return_value=httpx.Response(200, json={"data": [payload]}),
    )

    result = restore_jobs.list()

    assert len(result) == 1
    job = result[0]
    assert job.created_at is None or isinstance(job.created_at, str)


@respx.mock
def test_restore_job_created_at_null(restore_jobs: RestoreJobs) -> None:
    """list() decodes created_at: null as None without raising DecodeError."""
    payload = make_restore_job_response(created_at=None)
    respx.get(f"{BASE_URL}/restore-jobs").mock(
        return_value=httpx.Response(200, json={"data": [payload]}),
    )

    result = restore_jobs.list()

    assert result[0].created_at is None


# ---------------------------------------------------------------------------
# Negative: no update or cancel methods (claim unified-bak-0019)
# ---------------------------------------------------------------------------


def test_no_update_or_cancel_methods(restore_jobs: RestoreJobs) -> None:
    assert not hasattr(restore_jobs, "update")
    assert not hasattr(restore_jobs, "cancel")
    assert not hasattr(restore_jobs, "delete")
