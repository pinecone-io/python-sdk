"""2026-07 conformance for the six db_control backup operations.

These claims were deliberately deferred by #113 (sync, PR #235) and #114
(async, PR #240) until #112 flipped ``CONTROL_PLANE_API_VERSION`` to
``2026-07``. Every operation is claimed twice — once through
:class:`Pinecone`, once through :class:`AsyncPinecone` — because the header
has to appear on the wire for both transports, and both read it from the
same ``CONTROL_PLANE_API_VERSION``.

The backup read ops (``describe_backup``, ``list_index_backups``,
``list_project_backups``) get **spec-shaped fixtures, not #224 divergence
entries** — the per-op decision #113/#114 deferred to #112. Rationale: the
gate certifies spec conformance, and a spec-shaped ``BackupModel`` is a
payload a spec-conformant server could send; the SDK's tolerance for the
live backend's legacy shape (``dimension`` on the wire, legacy metadata
schema) is pinned by the doctrine suites in
``tests/unit/test_backups_2026_07.py`` / ``test_async_backups_2026_07.py``
(``TestBackendShapedDecode``); and a divergence entry must name an
*alternative component schema from the OAS*, which the legacy backup shape
does not have in 2026-07 — the exception mechanism cannot express it
without inventing a schema. #224 stays open tracking the backend gap.

The ``list_index_backups`` claims exercise ``include_deleted=True`` and a
``source_index_deleted_at``-stamped row, pinning the client half of the
liveness contract #112's added AC covers (the query parameter reaches the
wire under the 2026-07 header; the v202604 handler the 2026-07 backend
routes to honors it — pinecone-db v202607/mod.rs:14 →
v202604/backups.rs:259-282 @ f6fd0a4019).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx
import pytest
import respx

from pinecone import Pinecone
from pinecone._internal.adapters.backups_adapter import _BackupListEnvelope
from pinecone._internal.constants import DEFAULT_BASE_URL
from pinecone.async_client.pinecone import AsyncPinecone
from pinecone.models.backups.model import BackupModel, CreateIndexFromBackupResponse
from tests.unit.conformance import api_op

BASE_URL = DEFAULT_BASE_URL
INDEX_NAME = "conformance-index"
BACKUP_ID = "bkp-conformance-123"

BACKUP: dict[str, Any] = {
    "backup_id": BACKUP_ID,
    "source_index_name": INDEX_NAME,
    "source_index_id": "idx-conformance-456",
    "status": "Ready",
    "cloud": "aws",
    "region": "us-east-1",
    "name": "nightly",
    "description": "conformance fixture",
    "schema": {
        "fields": {"embedding": {"type": "dense_vector", "dimension": 1024, "metric": "cosine"}}
    },
    "record_count": 120,
    "namespace_count": 3,
    "size_bytes": 10000000,
    "tags": {"env": "conformance"},
    "created_at": "2026-07-15T09:00:00Z",
}

BACKUP_DELETED_SOURCE: dict[str, Any] = {
    **BACKUP,
    "source_index_deleted_at": "2026-07-16T00:00:00Z",
}

BACKUP_OPTIONALS = ["schema", "tags"]

BACKUP_LIST: dict[str, Any] = {"data": [BACKUP], "pagination": {"next": "page-2"}}
BACKUP_LIST_DELETED: dict[str, Any] = {
    "data": [BACKUP_DELETED_SOURCE],
    "pagination": {"next": "page-2"},
}

RESTORE: dict[str, Any] = {
    "restore_job_id": "job-conformance-789",
    "index_id": "idx-restored-321",
}


@pytest.fixture
def pc() -> Iterator[Pinecone]:
    client = Pinecone(api_key="conformance-key")
    yield client
    client.close()


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


def _conforms_bodyless(claim: Any, route: respx.Route, returned: Any) -> None:
    request = route.calls.last.request
    claim.assert_request(request)
    claim.assert_api_version(request)
    claim.assert_no_response_body(returned)


@api_op("db_control:create_backup")
def test_create_backup(claim: Any, pc: Pinecone, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.post(f"{BASE_URL}/indexes/{INDEX_NAME}/backups").mock(
        return_value=httpx.Response(201, json=BACKUP)
    )
    result = pc.backups.create(index_name=INDEX_NAME, name="nightly")
    assert result.backup_id == BACKUP_ID
    _conforms(claim, route, BackupModel, BACKUP, BACKUP_OPTIONALS)


@api_op("db_control:create_backup")
async def test_async_create_backup(
    claim: Any, async_pc: AsyncPinecone, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.post(f"{BASE_URL}/indexes/{INDEX_NAME}/backups").mock(
        return_value=httpx.Response(201, json=BACKUP)
    )
    result = await async_pc.backups.create(index_name=INDEX_NAME, name="nightly")
    assert result.backup_id == BACKUP_ID
    _conforms(claim, route, BackupModel, BACKUP, BACKUP_OPTIONALS)


@api_op("db_control:describe_backup")
def test_describe_backup(claim: Any, pc: Pinecone, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.get(f"{BASE_URL}/backups/{BACKUP_ID}").mock(
        return_value=httpx.Response(200, json=BACKUP)
    )
    assert pc.backups.describe(backup_id=BACKUP_ID).backup_id == BACKUP_ID
    _conforms(claim, route, BackupModel, BACKUP, BACKUP_OPTIONALS)


@api_op("db_control:describe_backup")
async def test_async_describe_backup(
    claim: Any, async_pc: AsyncPinecone, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.get(f"{BASE_URL}/backups/{BACKUP_ID}").mock(
        return_value=httpx.Response(200, json=BACKUP)
    )
    result = await async_pc.backups.describe(backup_id=BACKUP_ID)
    assert result.backup_id == BACKUP_ID
    _conforms(claim, route, BackupModel, BACKUP, BACKUP_OPTIONALS)


@api_op("db_control:list_index_backups")
def test_list_index_backups_include_deleted(
    claim: Any, pc: Pinecone, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.get(f"{BASE_URL}/indexes/{INDEX_NAME}/backups").mock(
        return_value=httpx.Response(200, json=BACKUP_LIST_DELETED)
    )
    result = pc.backups.list(index_name=INDEX_NAME, include_deleted=True)
    assert result[0].source_index_deleted_at == "2026-07-16T00:00:00Z"
    request = route.calls.last.request
    assert request.url.params["include_deleted"] == "true"
    _conforms(claim, route, _BackupListEnvelope, BACKUP_LIST_DELETED, ["pagination"])


@api_op("db_control:list_index_backups")
async def test_async_list_index_backups_include_deleted(
    claim: Any, async_pc: AsyncPinecone, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.get(f"{BASE_URL}/indexes/{INDEX_NAME}/backups").mock(
        return_value=httpx.Response(200, json=BACKUP_LIST_DELETED)
    )
    result = await async_pc.backups.list(index_name=INDEX_NAME, include_deleted=True)
    assert result[0].source_index_deleted_at == "2026-07-16T00:00:00Z"
    request = route.calls.last.request
    assert request.url.params["include_deleted"] == "true"
    _conforms(claim, route, _BackupListEnvelope, BACKUP_LIST_DELETED, ["pagination"])


@api_op("db_control:list_project_backups")
def test_list_project_backups(claim: Any, pc: Pinecone, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.get(f"{BASE_URL}/backups").mock(
        return_value=httpx.Response(200, json=BACKUP_LIST)
    )
    assert pc.backups.list()[0].backup_id == BACKUP_ID
    _conforms(claim, route, _BackupListEnvelope, BACKUP_LIST, ["pagination"])


@api_op("db_control:list_project_backups")
async def test_async_list_project_backups(
    claim: Any, async_pc: AsyncPinecone, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.get(f"{BASE_URL}/backups").mock(
        return_value=httpx.Response(200, json=BACKUP_LIST)
    )
    result = await async_pc.backups.list()
    assert result[0].backup_id == BACKUP_ID
    _conforms(claim, route, _BackupListEnvelope, BACKUP_LIST, ["pagination"])


@api_op("db_control:delete_backup")
def test_delete_backup(claim: Any, pc: Pinecone, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.delete(f"{BASE_URL}/backups/{BACKUP_ID}").mock(
        return_value=httpx.Response(202)
    )
    _conforms_bodyless(claim, route, pc.backups.delete(backup_id=BACKUP_ID))


@api_op("db_control:delete_backup")
async def test_async_delete_backup(
    claim: Any, async_pc: AsyncPinecone, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.delete(f"{BASE_URL}/backups/{BACKUP_ID}").mock(
        return_value=httpx.Response(202)
    )
    returned = await async_pc.backups.delete(backup_id=BACKUP_ID)
    _conforms_bodyless(claim, route, returned)


@api_op("db_control:create_index_from_backup_operation")
def test_create_index_from_backup(claim: Any, pc: Pinecone, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.post(f"{BASE_URL}/backups/{BACKUP_ID}/create-index").mock(
        return_value=httpx.Response(202, json=RESTORE)
    )
    result = pc.create_index_from_backup(name="restored-index", backup_id=BACKUP_ID, timeout=-1)
    assert result.restore_job_id == "job-conformance-789"
    _conforms(claim, route, CreateIndexFromBackupResponse, RESTORE, [])


@api_op("db_control:create_index_from_backup_operation")
async def test_async_create_index_from_backup(
    claim: Any, async_pc: AsyncPinecone, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.post(f"{BASE_URL}/backups/{BACKUP_ID}/create-index").mock(
        return_value=httpx.Response(202, json=RESTORE)
    )
    result = await async_pc.create_index_from_backup(
        name="restored-index", backup_id=BACKUP_ID, timeout=-1
    )
    assert result.restore_job_id == "job-conformance-789"
    _conforms(claim, route, CreateIndexFromBackupResponse, RESTORE, [])
