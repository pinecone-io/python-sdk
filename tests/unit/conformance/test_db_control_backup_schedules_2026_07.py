"""2026-07 conformance for the six db_control backup-schedule operations.

These claims were deliberately deferred by #115 (PR #255) until #112 flipped
``CONTROL_PLANE_API_VERSION`` to ``2026-07``. They are claimed through the
sync :class:`Pinecone` only: no async backup-schedules namespace exists yet
(#116 owns it), and that ticket can add the async twins the way #114/#133
did for backups and indexes.

``list_backup_schedule_history`` gets a **spec-shaped fixture, not a #224
divergence entry** — the per-op decision #115 deferred to #112, decided the
same way as the backup read ops (see
``test_db_control_backups_2026_07.py``): the gate certifies spec
conformance; the backend's legacy-shaped history rows (served from its
shared backup handler, nulling spec-required fields) are pinned by #115's
``TestBackendShapedHistoryDecode`` doctrine suite in
``tests/unit/client/test_backup_schedules.py``; and the legacy shape has no
2026-07 component schema a divergence entry could reference. #224 stays
open tracking the backend gap.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import httpx
import pytest
import respx

from pinecone import Pinecone
from pinecone._internal.adapters.backup_schedules_adapter import (
    _BackupScheduleHistoryEnvelope,
    _BackupScheduleListEnvelope,
)
from pinecone._internal.constants import DEFAULT_BASE_URL
from pinecone.models.backups.schedules import BackupScheduleModel
from tests.unit.conformance import api_op

BASE_URL = DEFAULT_BASE_URL
INDEX_NAME = "conformance-index"
SCHEDULE_ID = "sched-conformance-123"

SCHEDULE: dict[str, Any] = {
    "schedule_id": SCHEDULE_ID,
    "name": "nightly",
    "index_id": "idx-conformance-456",
    "project_id": "proj-conformance-789",
    "schedule_type": "time-based",
    "frequency": "daily",
    "retention_expire_after_days": 30,
    "enabled": True,
    "next_scheduled_run": "2026-07-16T09:00:00Z",
    "created_at": "2026-07-15T09:00:00Z",
}

SCHEDULE_OPTIONALS = ["next_scheduled_run"]

SCHEDULE_LIST: dict[str, Any] = {"data": [SCHEDULE], "pagination": {"next": "page-2"}}

HISTORY_ITEM: dict[str, Any] = {
    "backup_id": "bkp-conformance-123",
    "source_index_id": "idx-conformance-456",
    "source_index_name": INDEX_NAME,
    "name": "nightly-20260716T090000Z",
    "status": "Ready",
    "cloud": "aws",
    "region": "us-east-1",
    "record_count": 120,
    "namespace_count": 3,
    "size_bytes": 10000000,
    "created_at": "2026-07-16T09:00:00Z",
}

HISTORY_LIST: dict[str, Any] = {"data": [HISTORY_ITEM], "pagination": {"next": "page-2"}}


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


@api_op("db_control:create_backup_schedule")
def test_create_backup_schedule(claim: Any, pc: Pinecone, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.post(f"{BASE_URL}/indexes/{INDEX_NAME}/backup-schedules").mock(
        return_value=httpx.Response(201, json=SCHEDULE)
    )
    result = pc.backup_schedules.create(
        index_name=INDEX_NAME, name="nightly", frequency="daily", retention_days=30
    )
    assert result.schedule_id == SCHEDULE_ID
    _conforms(claim, route, BackupScheduleModel, SCHEDULE, SCHEDULE_OPTIONALS)


@api_op("db_control:describe_backup_schedule")
def test_describe_backup_schedule(claim: Any, pc: Pinecone, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.get(f"{BASE_URL}/backup-schedules/{SCHEDULE_ID}").mock(
        return_value=httpx.Response(200, json=SCHEDULE)
    )
    assert pc.backup_schedules.describe(schedule_id=SCHEDULE_ID).schedule_id == SCHEDULE_ID
    _conforms(claim, route, BackupScheduleModel, SCHEDULE, SCHEDULE_OPTIONALS)


@api_op("db_control:update_backup_schedule")
def test_update_backup_schedule(claim: Any, pc: Pinecone, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.patch(f"{BASE_URL}/backup-schedules/{SCHEDULE_ID}").mock(
        return_value=httpx.Response(200, json=SCHEDULE)
    )
    result = pc.backup_schedules.update(schedule_id=SCHEDULE_ID, retention_days=30)
    assert result.schedule_id == SCHEDULE_ID
    _conforms(claim, route, BackupScheduleModel, SCHEDULE, SCHEDULE_OPTIONALS)


@api_op("db_control:delete_backup_schedule")
def test_delete_backup_schedule(claim: Any, pc: Pinecone, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.delete(f"{BASE_URL}/backup-schedules/{SCHEDULE_ID}").mock(
        return_value=httpx.Response(204)
    )
    returned = pc.backup_schedules.delete(schedule_id=SCHEDULE_ID)
    request = route.calls.last.request
    claim.assert_request(request)
    claim.assert_api_version(request)
    claim.assert_no_response_body(returned)


@api_op("db_control:list_backup_schedules")
def test_list_backup_schedules(claim: Any, pc: Pinecone, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.get(f"{BASE_URL}/indexes/{INDEX_NAME}/backup-schedules").mock(
        return_value=httpx.Response(200, json=SCHEDULE_LIST)
    )
    assert pc.backup_schedules.list(index_name=INDEX_NAME)[0].schedule_id == SCHEDULE_ID
    _conforms(claim, route, _BackupScheduleListEnvelope, SCHEDULE_LIST, ["pagination"])


@api_op("db_control:list_backup_schedule_history")
def test_list_backup_schedule_history(
    claim: Any, pc: Pinecone, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.get(f"{BASE_URL}/backup-schedules/{SCHEDULE_ID}/history").mock(
        return_value=httpx.Response(200, json=HISTORY_LIST)
    )
    result = pc.backup_schedules.history(schedule_id=SCHEDULE_ID)
    assert result[0].backup_id == "bkp-conformance-123"
    _conforms(claim, route, _BackupScheduleHistoryEnvelope, HISTORY_LIST, ["pagination"])
