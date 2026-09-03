"""Unit tests for the 2026-07 backup endpoints on ``AsyncPinecone.backups`` (#114).

The asyncio twin of ``tests/unit/test_backups_2026_07.py``. Payloads named
``SPEC_*`` are copied from
``apis/src/release/db/control/resources/indexes/ListBackups.yaml`` and
``backups/DescribeBackup.yaml`` @ apis 5f808858, so a spec change breaks a
test here. Payloads named ``BACKEND_*`` are the shape the v202607 router
actually returns today, because it delegates ``/backups`` to the v202604
handler (legacy metadata schema, ``dimension`` on the wire, ``Failed``
remapped to ``InitializationFailed``) — pinecone-db
``svc-global-apis/src/control_plane/http/handler/global/v202604/backups.rs``
@ f6fd0a40. Both must decode: see #224.

The fixtures are byte-identical to the sync module's on purpose — the two
lanes share ``BackupsAdapter`` and ``_internal/backups_helpers.py``, so a
decode or query-string regression that only shows up on one transport is
exactly what these pin.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

import httpx
import orjson
import pytest
import respx

from pinecone._internal.config import PineconeConfig
from pinecone._internal.constants import CONTROL_PLANE_API_VERSION
from pinecone._internal.http_client import AsyncHTTPClient
from pinecone.async_client.backups import AsyncBackups
from pinecone.async_client.pinecone import AsyncPinecone
from pinecone.errors.exceptions import ApiError, NotFoundError, PineconeValueError
from pinecone.models.backups.model import CreateIndexFromBackupResponse
from pinecone.models.indexes.index import IndexModel
from pinecone.models.indexes.schema import DenseVectorField, LegacyMetadataField
from tests.factories import make_index_response

BASE_URL = "https://api.test.pinecone.io"

SPEC_ACTIVE_BACKUP: dict[str, Any] = {
    "backup_id": "bkp_123abc",
    "source_index_name": "my-index",
    "source_index_id": "idx_456",
    "name": "backup_2025_03_15",
    "description": "Monthly backup of production index",
    "status": "Ready",
    "cloud": "aws",
    "region": "us-east-1",
    "schema": {
        "fields": {"embedding": {"type": "dense_vector", "dimension": 1536, "metric": "cosine"}}
    },
    "record_count": 120000,
    "namespace_count": 3,
    "size_bytes": 10000000,
    "tags": {"environment": "production", "type": "monthly"},
    "created_at": "2025-03-15T10:30:00Z",
}

SPEC_DELETED_SOURCE_BACKUP: dict[str, Any] = {
    **SPEC_ACTIVE_BACKUP,
    "backup_id": "bkp_oldidx",
    "source_index_id": "idx_legacy",
    "name": "backup_before_delete",
    "description": "Backup from a deleted index that used the same name",
    "created_at": "2025-03-01T09:00:00Z",
    "source_index_deleted_at": "2025-03-05T12:00:00Z",
}

BACKEND_BACKUP: dict[str, Any] = {
    "backup_id": "bkp_123abc",
    "source_index_name": "my-index",
    "source_index_id": "idx_456",
    "name": "backup_2025_03_15",
    "status": "InitializationFailed",
    "cloud": "aws",
    "region": "us-east-1",
    "dimension": 1536,
    "schema": {"fields": {"genre": {"filterable": True}}},
    "record_count": 120000,
    "namespace_count": 3,
    "size_bytes": 10000000,
    "tags": None,
    "created_at": "2025-03-15T10:30:00Z",
}


@pytest.fixture
async def backups() -> AsyncGenerator[AsyncBackups]:
    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    http = AsyncHTTPClient(config, CONTROL_PLANE_API_VERSION)
    yield AsyncBackups(http=http)
    await http.close()


class TestIncludeDeletedQueryParam:
    @respx.mock
    async def test_true_sends_lowercase_true(self, backups: AsyncBackups) -> None:
        route = respx.get(f"{BASE_URL}/indexes/my-index/backups").mock(
            return_value=httpx.Response(200, json={"data": [SPEC_DELETED_SOURCE_BACKUP]})
        )

        await backups.list(index_name="my-index", include_deleted=True)

        assert route.calls.last.request.url.params["include_deleted"] == "true"

    @respx.mock
    async def test_false_sends_lowercase_false(self, backups: AsyncBackups) -> None:
        route = respx.get(f"{BASE_URL}/indexes/my-index/backups").mock(
            return_value=httpx.Response(200, json={"data": []})
        )

        await backups.list(index_name="my-index", include_deleted=False)

        assert route.calls.last.request.url.params["include_deleted"] == "false"

    @respx.mock
    async def test_omitted_leaves_param_absent_not_false(self, backups: AsyncBackups) -> None:
        route = respx.get(f"{BASE_URL}/indexes/my-index/backups").mock(
            return_value=httpx.Response(200, json={"data": []})
        )

        await backups.list(index_name="my-index")

        assert "include_deleted" not in route.calls.last.request.url.params

    async def test_rejected_on_the_project_wide_listing(self, backups: AsyncBackups) -> None:
        with pytest.raises(PineconeValueError) as exc:
            await backups.list(include_deleted=True)

        assert "include_deleted" in str(exc.value)
        assert "index_name" in str(exc.value)

    @respx.mock
    async def test_rejected_before_any_request_is_issued(self, backups: AsyncBackups) -> None:
        """The coroutine must reject client-side, not after awaiting a round trip."""
        respx.get(f"{BASE_URL}/backups").mock(return_value=httpx.Response(200, json={"data": []}))

        with pytest.raises(PineconeValueError):
            await backups.list(include_deleted=False)

        assert len(respx.calls) == 0

    @respx.mock
    async def test_deleted_source_backup_round_trips(self, backups: AsyncBackups) -> None:
        respx.get(f"{BASE_URL}/indexes/my-index/backups").mock(
            return_value=httpx.Response(
                200, json={"data": [SPEC_ACTIVE_BACKUP, SPEC_DELETED_SOURCE_BACKUP]}
            )
        )

        result = await backups.list(index_name="my-index", include_deleted=True)

        assert [b.source_index_deleted_at for b in result] == [None, "2025-03-05T12:00:00Z"]
        assert result[1].to_dict()["source_index_deleted_at"] == "2025-03-05T12:00:00Z"

    @respx.mock
    async def test_404_surfaces_rather_than_becoming_an_empty_list(
        self, backups: AsyncBackups
    ) -> None:
        respx.get(f"{BASE_URL}/indexes/gone/backups").mock(
            return_value=httpx.Response(
                404, json={"error": {"code": "NOT_FOUND", "message": "Index gone not found"}}
            )
        )

        with pytest.raises(NotFoundError):
            await backups.list(index_name="gone")


class TestSpecShapedDecode:
    @respx.mock
    async def test_describe_decodes_typed_schema(self, backups: AsyncBackups) -> None:
        respx.get(f"{BASE_URL}/backups/bkp_123abc").mock(
            return_value=httpx.Response(200, json=SPEC_ACTIVE_BACKUP)
        )

        backup = await backups.describe(backup_id="bkp_123abc")

        assert backup.schema is not None
        field = backup.schema.fields["embedding"]
        assert isinstance(field, DenseVectorField)
        assert field.dimension == 1536
        assert backup.dense_dimension == 1536

    @respx.mock
    async def test_removed_fields_raise_a_guided_attribute_error(
        self, backups: AsyncBackups
    ) -> None:
        respx.get(f"{BASE_URL}/backups/bkp_123abc").mock(
            return_value=httpx.Response(200, json=SPEC_ACTIVE_BACKUP)
        )

        backup = await backups.describe(backup_id="bkp_123abc")

        for removed in ("dimension", "metric"):
            with pytest.raises(AttributeError) as exc:
                getattr(backup, removed)
            assert "schema.fields" in str(exc.value)

    @respx.mock
    async def test_project_listing_terminates_on_null_pagination(
        self, backups: AsyncBackups
    ) -> None:
        respx.get(f"{BASE_URL}/backups").mock(
            return_value=httpx.Response(
                200, json={"data": [SPEC_ACTIVE_BACKUP], "pagination": None}
            )
        )

        assert (await backups.list()).pagination is None

    @respx.mock
    async def test_get_is_an_alias_for_describe(self, backups: AsyncBackups) -> None:
        route = respx.get(f"{BASE_URL}/backups/bkp_123abc").mock(
            return_value=httpx.Response(200, json=SPEC_ACTIVE_BACKUP)
        )

        assert (await backups.get(backup_id="bkp_123abc")).backup_id == "bkp_123abc"
        assert route.calls.last.request.url.path == "/backups/bkp_123abc"


class TestBackendShapedDecode:
    """#224 doctrine: the surface is spec-shaped, the decode stays backend-tolerant."""

    @respx.mock
    async def test_legacy_wire_shape_still_decodes(self, backups: AsyncBackups) -> None:
        respx.get(f"{BASE_URL}/backups/bkp_123abc").mock(
            return_value=httpx.Response(200, json=BACKEND_BACKUP)
        )

        backup = await backups.describe(backup_id="bkp_123abc")

        assert backup.schema is not None
        assert isinstance(backup.schema.fields["genre"], LegacyMetadataField)
        assert backup.tags is None

    @respx.mock
    async def test_degraded_dense_dimension_is_none_not_an_error(
        self, backups: AsyncBackups
    ) -> None:
        respx.get(f"{BASE_URL}/backups/bkp_123abc").mock(
            return_value=httpx.Response(200, json=BACKEND_BACKUP)
        )

        assert (await backups.describe(backup_id="bkp_123abc")).dense_dimension is None

    @respx.mock
    async def test_initialization_failed_status_decodes_verbatim(
        self, backups: AsyncBackups
    ) -> None:
        respx.get(f"{BASE_URL}/backups/bkp_123abc").mock(
            return_value=httpx.Response(200, json=BACKEND_BACKUP)
        )

        assert (await backups.describe(backup_id="bkp_123abc")).status == "InitializationFailed"

    @respx.mock
    async def test_wire_dimension_is_not_resurrected_as_an_attribute(
        self, backups: AsyncBackups
    ) -> None:
        respx.get(f"{BASE_URL}/backups/bkp_123abc").mock(
            return_value=httpx.Response(200, json=BACKEND_BACKUP)
        )

        backup = await backups.describe(backup_id="bkp_123abc")

        with pytest.raises(AttributeError):
            backup.dimension
        assert "dimension" not in backup.to_dict()

    @respx.mock
    async def test_legacy_listing_decodes_through_the_index_scoped_path(
        self, backups: AsyncBackups
    ) -> None:
        respx.get(f"{BASE_URL}/indexes/my-index/backups").mock(
            return_value=httpx.Response(200, json={"data": [BACKEND_BACKUP]})
        )

        result = await backups.list(index_name="my-index", include_deleted=True)

        assert len(result) == 1
        assert result[0].source_index_deleted_at is None


class TestMethodAndPath:
    @respx.mock
    async def test_create_backup(self, backups: AsyncBackups) -> None:
        route = respx.post(f"{BASE_URL}/indexes/my-index/backups").mock(
            return_value=httpx.Response(201, json=SPEC_ACTIVE_BACKUP)
        )

        await backups.create(index_name="my-index")

        request = route.calls.last.request
        assert (request.method, request.url.path) == ("POST", "/indexes/my-index/backups")

    @respx.mock
    async def test_create_backup_tolerates_the_backends_200(self, backups: AsyncBackups) -> None:
        respx.post(f"{BASE_URL}/indexes/my-index/backups").mock(
            return_value=httpx.Response(200, json=SPEC_ACTIVE_BACKUP)
        )

        assert (await backups.create(index_name="my-index")).backup_id == "bkp_123abc"

    @respx.mock
    async def test_list_index_backups(self, backups: AsyncBackups) -> None:
        route = respx.get(f"{BASE_URL}/indexes/my-index/backups").mock(
            return_value=httpx.Response(200, json={"data": []})
        )

        await backups.list(index_name="my-index")

        request = route.calls.last.request
        assert (request.method, request.url.path) == ("GET", "/indexes/my-index/backups")

    @respx.mock
    async def test_list_project_backups(self, backups: AsyncBackups) -> None:
        route = respx.get(f"{BASE_URL}/backups").mock(
            return_value=httpx.Response(200, json={"data": []})
        )

        await backups.list()

        request = route.calls.last.request
        assert (request.method, request.url.path) == ("GET", "/backups")

    @respx.mock
    async def test_describe_backup(self, backups: AsyncBackups) -> None:
        route = respx.get(f"{BASE_URL}/backups/bkp_123abc").mock(
            return_value=httpx.Response(200, json=SPEC_ACTIVE_BACKUP)
        )

        await backups.describe(backup_id="bkp_123abc")

        request = route.calls.last.request
        assert (request.method, request.url.path) == ("GET", "/backups/bkp_123abc")

    @respx.mock
    async def test_delete_backup_is_bodyless(self, backups: AsyncBackups) -> None:
        route = respx.delete(f"{BASE_URL}/backups/bkp_123abc").mock(
            return_value=httpx.Response(202)
        )

        assert await backups.delete(backup_id="bkp_123abc") is None

        request = route.calls.last.request
        assert (request.method, request.url.path) == ("DELETE", "/backups/bkp_123abc")

    @respx.mock
    async def test_delete_backup_surfaces_the_pending_restore_precondition(
        self, backups: AsyncBackups
    ) -> None:
        respx.delete(f"{BASE_URL}/backups/bkp_123abc").mock(
            return_value=httpx.Response(
                412,
                json={
                    "error": {
                        "code": "FAILED_PRECONDITION",
                        "message": (
                            "Unable to delete backup. There are pending restore jobs "
                            "for this backup: ['670e8400-e29b-41d4-a716-446655440000']"
                        ),
                    },
                    "status": 412,
                },
            )
        )

        with pytest.raises(ApiError) as exc:
            await backups.delete(backup_id="bkp_123abc")

        assert "pending restore jobs" in str(exc.value)


class TestApiVersionHeaderComesFromTheSdk:
    @respx.mock
    async def test_every_backup_request_carries_the_control_plane_constant(
        self, backups: AsyncBackups
    ) -> None:
        respx.get(f"{BASE_URL}/backups").mock(return_value=httpx.Response(200, json={"data": []}))
        respx.get(f"{BASE_URL}/indexes/my-index/backups").mock(
            return_value=httpx.Response(200, json={"data": []})
        )
        respx.post(f"{BASE_URL}/indexes/my-index/backups").mock(
            return_value=httpx.Response(201, json=SPEC_ACTIVE_BACKUP)
        )
        respx.get(f"{BASE_URL}/backups/bkp_123abc").mock(
            return_value=httpx.Response(200, json=SPEC_ACTIVE_BACKUP)
        )
        respx.delete(f"{BASE_URL}/backups/bkp_123abc").mock(return_value=httpx.Response(202))

        await backups.list()
        await backups.list(index_name="my-index", include_deleted=True)
        await backups.create(index_name="my-index")
        await backups.describe(backup_id="bkp_123abc")
        await backups.delete(backup_id="bkp_123abc")

        assert len(respx.calls) == 5
        for call in respx.calls:
            assert call.request.headers["X-Pinecone-Api-Version"] == CONTROL_PLANE_API_VERSION


class TestAsyncCreateIndexFromBackup:
    """AC 3: the ``timeout=-1`` immediate-return contract, on the 2026-07 models.

    ``read_capacity`` is *not* asserted here — adding the kwarg means editing
    ``pinecone/async_client/pinecone.py``, which #133 holds. See the PR body.
    """

    @respx.mock
    async def test_timeout_minus_one_returns_the_response_model_without_polling(self) -> None:
        restore = respx.post(f"{BASE_URL}/backups/bkp_123abc/create-index").mock(
            return_value=httpx.Response(202, json={"restore_job_id": "rj-1", "index_id": "idx-1"})
        )
        describe = respx.get(f"{BASE_URL}/indexes/restored").mock(
            return_value=httpx.Response(200, json=make_index_response(name="restored"))
        )

        async with AsyncPinecone(api_key="test-key", host=BASE_URL) as pc:
            result = await pc.create_index_from_backup(
                name="restored", backup_id="bkp_123abc", timeout=-1
            )

        assert isinstance(result, CreateIndexFromBackupResponse)
        assert (result.restore_job_id, result.index_id) == ("rj-1", "idx-1")
        assert restore.called
        assert not describe.called, "timeout=-1 must not poll for readiness"

    @respx.mock
    async def test_default_timeout_polls_and_returns_the_index_model(self) -> None:
        respx.post(f"{BASE_URL}/backups/bkp_123abc/create-index").mock(
            return_value=httpx.Response(202, json={"restore_job_id": "rj-1", "index_id": "idx-1"})
        )
        describe = respx.get(f"{BASE_URL}/indexes/restored").mock(
            return_value=httpx.Response(200, json=make_index_response(name="restored"))
        )

        async with AsyncPinecone(api_key="test-key", host=BASE_URL) as pc:
            result = await pc.create_index_from_backup(name="restored", backup_id="bkp_123abc")

        assert isinstance(result, IndexModel)
        assert result.name == "restored"
        assert describe.called, "the default timeout must poll describe until ready"

    @respx.mock
    async def test_body_matches_the_sync_lane_for_the_shared_kwargs(self) -> None:
        """The async body must stay byte-identical to the sync lane's."""
        route = respx.post(f"{BASE_URL}/backups/bkp_123abc/create-index").mock(
            return_value=httpx.Response(202, json={"restore_job_id": "rj-1", "index_id": "idx-1"})
        )

        async with AsyncPinecone(api_key="test-key", host=BASE_URL) as pc:
            await pc.create_index_from_backup(
                name="restored",
                backup_id="bkp_123abc",
                deletion_protection="enabled",
                tags={"env": "prod"},
                timeout=-1,
            )

        assert orjson.loads(route.calls.last.request.content) == {
            "name": "restored",
            "deletion_protection": "enabled",
            "tags": {"env": "prod"},
        }
