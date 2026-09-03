"""2026-07 conformance for the two db_control restore-job operations.

Deferred by #117 until #112 flipped ``CONTROL_PLANE_API_VERSION`` to
``2026-07``, for the same reason as the collection claims: a
``version-bump-only`` claim made under the old constant certifies nothing.

``RestoreJobModel`` is byte-identical to ``2025-10`` and both operations
differ only in the ``X-Pinecone-Api-Version`` default. ``RestoreJobList`` is
the one schema that did change shape: ``2025-10`` referenced the shared
``PaginationResponse`` component, and ``2026-07`` inlines it and marks it
``nullable: true``. That is a documentation change catching up with the
backend — ``base/restore_jobs.rs:47-50`` @ pinecone-db ``cbee5a67fe``
declares ``pagination: Option<PaginationResponse>`` with no
``skip_serializing_if``, so the final page has always gone out as a literal
``"pagination": null`` — but it means ``null`` is now a spec-legal value the
gate can hold the decode to, which is why it gets a claim of its own below.

Absent-key and explicit-``null`` are two different wire shapes and are
covered separately: the backend emits the second, a spec-conformant server
may emit either (``pagination`` is not in ``required``), and both must land
on ``pagination=None``.

The module-level fixtures are named for reuse by #118's async claims.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import httpx
import pytest
import respx

from pinecone import Pinecone
from pinecone._internal.adapters.restore_jobs_adapter import _RestoreJobListEnvelope
from pinecone._internal.constants import DEFAULT_BASE_URL
from pinecone.models.backups.model import RestoreJobModel
from tests.unit.conformance import api_op

BASE_URL = DEFAULT_BASE_URL
JOB_ID = "rj-conformance-123"

RESTORE_JOB: dict[str, Any] = {
    "restore_job_id": JOB_ID,
    "backup_id": "bkp-conformance-456",
    "target_index_name": "conformance-index",
    "target_index_id": "idx-conformance-789",
    "status": "Completed",
    "created_at": "2026-07-15T09:00:00Z",
    "completed_at": "2026-07-15T09:12:00Z",
    "percent_complete": 100.0,
}

RESTORE_JOB_OPTIONALS = ["completed_at", "percent_complete"]

RESTORE_JOB_LIST: dict[str, Any] = {"data": [RESTORE_JOB], "pagination": {"next": "page-2"}}

RESTORE_JOB_LIST_FINAL_PAGE: dict[str, Any] = {"data": [RESTORE_JOB], "pagination": None}

RESTORE_JOB_LIST_NO_PAGINATION_KEY: dict[str, Any] = {"data": [RESTORE_JOB]}


@pytest.fixture
def pc() -> Iterator[Pinecone]:
    client = Pinecone(api_key="conformance-key")
    yield client
    client.close()


def _conforms(
    claim: Any,
    route: respx.Route,
    model: type,
    payload: dict[str, Any],
    optional_absent: list[str],
) -> None:
    request = route.calls.last.request
    claim.assert_request(request)
    claim.assert_api_version(request)
    claim.assert_roundtrip(model, payload, optional_absent=optional_absent)


@api_op("db_control:list_restore_jobs")
def test_list_restore_jobs(claim: Any, pc: Pinecone, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.get(f"{BASE_URL}/restore-jobs").mock(
        return_value=httpx.Response(200, json=RESTORE_JOB_LIST)
    )
    result = pc.restore_jobs.list(limit=5)
    assert result[0].restore_job_id == JOB_ID
    assert result.pagination is not None
    assert result.pagination.next == "page-2"
    assert route.calls.last.request.url.params["limit"] == "5"
    _conforms(claim, route, _RestoreJobListEnvelope, RESTORE_JOB_LIST, ["pagination"])


@api_op("db_control:list_restore_jobs")
def test_list_restore_jobs_final_page_null_pagination(
    claim: Any, pc: Pinecone, respx_mock: respx.MockRouter
) -> None:
    """The wire shape the backend actually sends on the last page."""
    route = respx_mock.get(f"{BASE_URL}/restore-jobs").mock(
        return_value=httpx.Response(200, json=RESTORE_JOB_LIST_FINAL_PAGE)
    )
    result = pc.restore_jobs.list(pagination_token="page-2")
    assert result[0].restore_job_id == JOB_ID
    assert result.pagination is None
    assert result.to_dict() == {"data": [RESTORE_JOB]}
    assert route.calls.last.request.url.params["paginationToken"] == "page-2"
    _conforms(claim, route, _RestoreJobListEnvelope, RESTORE_JOB_LIST_FINAL_PAGE, ["pagination"])


@api_op("db_control:list_restore_jobs")
def test_list_restore_jobs_pagination_key_absent(
    claim: Any, pc: Pinecone, respx_mock: respx.MockRouter
) -> None:
    """The other spec-legal single-page shape: ``pagination`` is not required.

    ``optional_absent`` names ``data`` rather than ``pagination`` because the
    recorder only accepts fields the payload actually carries — here the
    absence under test *is* the payload, so it is asserted directly on the
    decoded wrapper instead.
    """
    route = respx_mock.get(f"{BASE_URL}/restore-jobs").mock(
        return_value=httpx.Response(200, json=RESTORE_JOB_LIST_NO_PAGINATION_KEY)
    )
    result = pc.restore_jobs.list()
    assert result[0].restore_job_id == JOB_ID
    assert result.pagination is None
    _conforms(claim, route, _RestoreJobListEnvelope, RESTORE_JOB_LIST_NO_PAGINATION_KEY, ["data"])


@api_op("db_control:describe_restore_job")
def test_describe_restore_job(claim: Any, pc: Pinecone, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.get(f"{BASE_URL}/restore-jobs/{JOB_ID}").mock(
        return_value=httpx.Response(200, json=RESTORE_JOB)
    )
    result = pc.restore_jobs.describe(job_id=JOB_ID)
    assert result.restore_job_id == JOB_ID
    assert result.percent_complete == 100.0
    _conforms(claim, route, RestoreJobModel, RESTORE_JOB, RESTORE_JOB_OPTIONALS)
