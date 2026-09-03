"""2026-07 conformance for the asyncio transport of the restore-job operations (#118).

``list_restore_jobs`` and ``describe_restore_job`` are ``version-bump-only``
for 2026-07: diffing ``db_control_2026-04.oas.yaml`` against
``db_control_2026-07.oas.yaml`` over both path items and the
``RestoreJobList`` / ``RestoreJobModel`` component schemas leaves exactly one
difference — the ``X-Pinecone-Api-Version`` parameter default going
``2026-04`` -> ``2026-07``.

The sync variants live in ``test_db_control_restore_jobs_2026_07.py`` (#117);
both lanes may claim the same operation (see README, "Additional rules"), so
this file adds no operation ids of its own to the coverage numerator. It adds
that ``AsyncRestoreJobs`` — whose header is built by ``AsyncHTTPClient``, not
by the sync client — is held to 2026-07 and to the spec endpoints.

``TestNullablePaginationEnvelope`` covers the one shape 2026-07 states
explicitly and the older ``$ref``-to-``PaginationResponse`` form did not: the
envelope is inlined and ``nullable: true``, documented as "``null`` (or
absent) on the final page of results". Both wire shapes therefore have to
decode to ``pagination=None``, and a decoder that only tolerated the absent
key would pass a suite that never sent the explicit ``null``.

Fixtures are restated rather than imported from the sync module because #117
and #118 land as independent PRs; sync/async drift is ruled out by
``tests/unit/test_async_collections_restore_jobs_parity.py`` instead.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
import respx

from pinecone._internal.adapters.restore_jobs_adapter import _RestoreJobListEnvelope
from pinecone._internal.constants import DEFAULT_BASE_URL
from pinecone.async_client.pinecone import AsyncPinecone
from pinecone.models.backups.model import RestoreJobModel
from tests.unit.conformance import api_op

BASE_URL = DEFAULT_BASE_URL
JOB_ID = "670e8400-e29b-41d4-a716-446655440001"

RESTORE_JOB: dict[str, Any] = {
    "restore_job_id": JOB_ID,
    "backup_id": "670e8400-e29b-41d4-a716-446655440000",
    "target_index_name": "conformance-index",
    "target_index_id": "idx_456",
    "status": "Completed",
    "created_at": "2026-07-15T10:30:00Z",
    "completed_at": "2026-07-15T10:35:00Z",
    "percent_complete": 100.0,
}

RESTORE_JOB_OPTIONALS = ["completed_at", "percent_complete"]

RESTORE_JOB_LIST: dict[str, Any] = {
    "data": [RESTORE_JOB],
    "pagination": {"next": "dXNlcl9pZD11c2VyXzE="},
}


@pytest.fixture
async def async_pc() -> AsyncIterator[AsyncPinecone]:
    client = AsyncPinecone(api_key="conformance-key")
    yield client
    await client.close()


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
async def test_async_list_restore_jobs(
    claim: Any, async_pc: AsyncPinecone, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.get(f"{BASE_URL}/restore-jobs").mock(
        return_value=httpx.Response(200, json=RESTORE_JOB_LIST)
    )
    result = await async_pc.restore_jobs.list(limit=5)
    assert [job.restore_job_id for job in result] == [JOB_ID]
    assert result.pagination is not None
    assert result.pagination.next == "dXNlcl9pZD11c2VyXzE="
    assert dict(route.calls.last.request.url.params) == {"limit": "5"}
    _conforms(claim, route, _RestoreJobListEnvelope, RESTORE_JOB_LIST, ["pagination"])


@api_op("db_control:describe_restore_job")
async def test_async_describe_restore_job(
    claim: Any, async_pc: AsyncPinecone, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.get(f"{BASE_URL}/restore-jobs/{JOB_ID}").mock(
        return_value=httpx.Response(200, json=RESTORE_JOB)
    )
    result = await async_pc.restore_jobs.describe(job_id=JOB_ID)
    assert result.status == "Completed"
    _conforms(claim, route, RestoreJobModel, RESTORE_JOB, RESTORE_JOB_OPTIONALS)


class TestNullablePaginationEnvelope:
    """Both final-page wire shapes 2026-07 allows must decode to ``pagination=None``."""

    @pytest.mark.parametrize(
        ("payload", "shape"),
        [
            ({"data": [RESTORE_JOB], "pagination": None}, "explicit-null"),
            ({"data": [RESTORE_JOB]}, "absent-key"),
        ],
        ids=["explicit-null", "absent-key"],
    )
    async def test_final_page_decodes_to_none(
        self,
        payload: dict[str, Any],
        shape: str,
        async_pc: AsyncPinecone,
        respx_mock: respx.MockRouter,
    ) -> None:
        respx_mock.get(f"{BASE_URL}/restore-jobs").mock(
            return_value=httpx.Response(200, json=payload)
        )
        result = await async_pc.restore_jobs.list()
        assert result.pagination is None, shape
        assert len(result) == 1

    async def test_empty_final_page(
        self, async_pc: AsyncPinecone, respx_mock: respx.MockRouter
    ) -> None:
        respx_mock.get(f"{BASE_URL}/restore-jobs").mock(
            return_value=httpx.Response(200, json={"data": []})
        )
        result = await async_pc.restore_jobs.list()
        assert len(result) == 0
        assert result.pagination is None
