"""Unit tests for ``pc.backup_schedules`` — the six 2026-07 schedule endpoints.

No ``@api_op`` conformance claims are added here, following the #131/#113
precedent: ``claim.assert_api_version`` requires ``2026-07`` on the wire while
``CONTROL_PLANE_API_VERSION`` is still ``2025-10`` until #112, so a db_control
claim would be a red test the coverage gate refuses to count. The three claim
categories are covered without the hardcoded version instead —
:class:`TestMethodAndPath`, :class:`TestApiVersionHeaderComesFromTheSdk`, and
:class:`TestSpecShapedDecode`.

Response and request fixtures are copied from the **source** spec files
(``apis`` @ 5f808858), not the build: the built OAS leaves its date-time
example scalars unquoted, so a YAML loader materialises them as ``datetime``
objects and renders them back with a space separator instead of RFC 3339.
"""

from __future__ import annotations

import json
import re
from typing import Any

import httpx
import pytest
import respx

from pinecone._internal.backups_helpers import SCHEDULED_BACKUPS_PLAN_HINT
from pinecone._internal.config import PineconeConfig
from pinecone._internal.constants import CONTROL_PLANE_API_VERSION
from pinecone._internal.http_client import HTTPClient
from pinecone.client.backup_schedules import BackupSchedules
from pinecone.errors.exceptions import (
    ConflictError,
    ForbiddenError,
    NotFoundError,
    PineconeValueError,
    ResponseParsingError,
)
from pinecone.models.backups.list import BackupScheduleHistoryList, BackupScheduleList
from pinecone.models.backups.schedules import BackupScheduleHistoryItem, BackupScheduleModel
from pinecone.models.indexes.schema import DenseVectorField, LegacyMetadataField
from pinecone.models.pagination import Paginator

BASE_URL = "https://api.test.pinecone.io"

SCHEDULE_ID = "e88f7273-42aa-47e9-af73-593827136867"

SPEC_SCHEDULE: dict[str, Any] = {
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

SPEC_UPDATED_SCHEDULE: dict[str, Any] = {
    "schedule_id": SCHEDULE_ID,
    "name": "daily-compliance-backup",
    "index_id": "8cbf7ba6-4135-438e-a3c3-4a89a3298905",
    "project_id": "71ce31ea-75f7-45d6-a147-ef67f661a1b0",
    "schedule_type": "time-based",
    "frequency": "weekly",
    "retention_expire_after_days": 30,
    "enabled": False,
    "created_at": "2026-04-02T18:22:56.712605+00:00",
}

SPEC_CREATE_BODY: dict[str, Any] = {
    "name": "daily-compliance-backup",
    "schedule": {"type": "time-based", "frequency": "daily"},
    "retention": {"expire_after_days": 90},
}

SPEC_UPDATE_BODY: dict[str, Any] = {
    "frequency": "weekly",
    "retention": {"expire_after_days": 30},
    "enabled": False,
}

_TYPED_SCHEMA: dict[str, Any] = {
    "fields": {"embedding": {"type": "dense_vector", "dimension": 1536, "metric": "cosine"}}
}

SPEC_HISTORY_READY: dict[str, Any] = {
    "backup_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "source_index_id": "8cbf7ba6-4135-438e-a3c3-4a89a3298905",
    "source_index_name": "my-index",
    "name": "daily-compliance-backup-20260403T060000Z",
    "status": "Ready",
    "cloud": "aws",
    "region": "us-east-1",
    "schema": _TYPED_SCHEMA,
    "record_count": 500000,
    "namespace_count": 1,
    "size_bytes": 104857600,
    "created_at": "2026-04-03T06:00:00+00:00",
}

SPEC_HISTORY_SCHEDULED: dict[str, Any] = {
    "backup_id": "b2c3d4e5-f6a7-8901-bcde-f12345678901",
    "source_index_id": "8cbf7ba6-4135-438e-a3c3-4a89a3298905",
    "source_index_name": "my-index",
    "name": "daily-compliance-backup-20260404T060000Z",
    "status": "Scheduled",
    "cloud": "aws",
    "region": "us-east-1",
    "schema": _TYPED_SCHEMA,
    "record_count": 0,
    "namespace_count": 0,
    "size_bytes": 0,
    "created_at": "2026-04-03T06:00:01+00:00",
    "scheduled_execution_at": "2026-04-04T06:00:00+00:00",
}

PLAN_GATE_403: dict[str, Any] = {
    "error": {
        "code": "PERMISSION_DENIED",
        "message": "Scheduled backups are not available for your plan",
    },
    "status": 403,
}

BACKEND_PLAN_GATE_403: dict[str, Any] = {
    "error": {
        "code": "PERMISSION_DENIED",
        "message": (
            "Your organization's plan does not include backups. Upgrade to "
            "Standard or Enterprise to enable this feature."
        ),
    },
    "status": 403,
}

ALREADY_ENABLED_409: dict[str, Any] = {
    "error": {
        "code": "ALREADY_EXISTS",
        "message": (
            "This index already has an enabled backup schedule. Disable or delete it first."
        ),
    },
    "status": 409,
}


def _error(code: str, message: str, status: int) -> dict[str, Any]:
    return {"error": {"code": code, "message": message}, "status": status}


@pytest.fixture
def schedules() -> BackupSchedules:
    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    return BackupSchedules(http=HTTPClient(config, CONTROL_PLANE_API_VERSION))


class TestMethodAndPath:
    """Each operation reaches the method and path the manifest records for it."""

    @respx.mock
    def test_create_posts_to_the_index_scoped_path(self, schedules: BackupSchedules) -> None:
        route = respx.post(f"{BASE_URL}/indexes/my-index/backup-schedules").mock(
            return_value=httpx.Response(201, json=SPEC_SCHEDULE)
        )

        schedules.create(
            index_name="my-index",
            name="daily-compliance-backup",
            frequency="daily",
            retention_days=90,
        )

        request = route.calls.last.request
        assert request.method == "POST"
        assert request.url.path == "/indexes/my-index/backup-schedules"

    @respx.mock
    def test_list_gets_the_index_scoped_path(self, schedules: BackupSchedules) -> None:
        route = respx.get(f"{BASE_URL}/indexes/my-index/backup-schedules").mock(
            return_value=httpx.Response(200, json={"data": [SPEC_SCHEDULE]})
        )

        schedules.list(index_name="my-index")

        request = route.calls.last.request
        assert request.method == "GET"
        assert request.url.path == "/indexes/my-index/backup-schedules"

    @respx.mock
    def test_describe_gets_the_schedule_path(self, schedules: BackupSchedules) -> None:
        route = respx.get(f"{BASE_URL}/backup-schedules/{SCHEDULE_ID}").mock(
            return_value=httpx.Response(200, json=SPEC_SCHEDULE)
        )

        schedules.describe(schedule_id=SCHEDULE_ID)

        request = route.calls.last.request
        assert request.method == "GET"
        assert request.url.path == f"/backup-schedules/{SCHEDULE_ID}"

    @respx.mock
    def test_update_patches_the_schedule_path(self, schedules: BackupSchedules) -> None:
        route = respx.patch(f"{BASE_URL}/backup-schedules/{SCHEDULE_ID}").mock(
            return_value=httpx.Response(200, json=SPEC_UPDATED_SCHEDULE)
        )

        schedules.update(schedule_id=SCHEDULE_ID, enabled=False)

        request = route.calls.last.request
        assert request.method == "PATCH"
        assert request.url.path == f"/backup-schedules/{SCHEDULE_ID}"

    @respx.mock
    def test_delete_deletes_the_schedule_path(self, schedules: BackupSchedules) -> None:
        route = respx.delete(f"{BASE_URL}/backup-schedules/{SCHEDULE_ID}").mock(
            return_value=httpx.Response(204)
        )

        schedules.delete(schedule_id=SCHEDULE_ID)

        request = route.calls.last.request
        assert request.method == "DELETE"
        assert request.url.path == f"/backup-schedules/{SCHEDULE_ID}"

    @respx.mock
    def test_history_gets_the_history_path(self, schedules: BackupSchedules) -> None:
        route = respx.get(f"{BASE_URL}/backup-schedules/{SCHEDULE_ID}/history").mock(
            return_value=httpx.Response(200, json={"data": [SPEC_HISTORY_READY]})
        )

        schedules.history(schedule_id=SCHEDULE_ID)

        request = route.calls.last.request
        assert request.method == "GET"
        assert request.url.path == f"/backup-schedules/{SCHEDULE_ID}/history"


class TestApiVersionHeaderComesFromTheSdk:
    """Every schedule request carries the SDK's own control-plane version."""

    @respx.mock
    def test_all_six_operations_send_the_sdk_constant(self, schedules: BackupSchedules) -> None:
        create = respx.post(f"{BASE_URL}/indexes/my-index/backup-schedules").mock(
            return_value=httpx.Response(201, json=SPEC_SCHEDULE)
        )
        listing = respx.get(f"{BASE_URL}/indexes/my-index/backup-schedules").mock(
            return_value=httpx.Response(200, json={"data": [SPEC_SCHEDULE]})
        )
        describe = respx.get(f"{BASE_URL}/backup-schedules/{SCHEDULE_ID}").mock(
            return_value=httpx.Response(200, json=SPEC_SCHEDULE)
        )
        update = respx.patch(f"{BASE_URL}/backup-schedules/{SCHEDULE_ID}").mock(
            return_value=httpx.Response(200, json=SPEC_UPDATED_SCHEDULE)
        )
        delete = respx.delete(f"{BASE_URL}/backup-schedules/{SCHEDULE_ID}").mock(
            return_value=httpx.Response(204)
        )
        history = respx.get(f"{BASE_URL}/backup-schedules/{SCHEDULE_ID}/history").mock(
            return_value=httpx.Response(200, json={"data": [SPEC_HISTORY_READY]})
        )

        schedules.create(
            index_name="my-index",
            name="daily-compliance-backup",
            frequency="daily",
            retention_days=90,
        )
        schedules.list(index_name="my-index")
        schedules.describe(schedule_id=SCHEDULE_ID)
        schedules.update(schedule_id=SCHEDULE_ID, enabled=False)
        schedules.delete(schedule_id=SCHEDULE_ID)
        schedules.history(schedule_id=SCHEDULE_ID)

        for route in (create, listing, describe, update, delete, history):
            header = route.calls.last.request.headers["X-Pinecone-Api-Version"]
            assert header == CONTROL_PLANE_API_VERSION


class TestSpecShapedDecode:
    """Every schedule example in the spec decodes through the endpoint path."""

    @respx.mock
    def test_create_decodes_the_created_example(self, schedules: BackupSchedules) -> None:
        respx.post(f"{BASE_URL}/indexes/my-index/backup-schedules").mock(
            return_value=httpx.Response(201, json=SPEC_SCHEDULE)
        )

        result = schedules.create(
            index_name="my-index",
            name="daily-compliance-backup",
            frequency="daily",
            retention_days=90,
        )

        assert isinstance(result, BackupScheduleModel)
        assert result.schedule_id == SCHEDULE_ID
        assert result.schedule_type == "time-based"
        assert result.frequency == "daily"
        assert result.retention_expire_after_days == 90
        assert result.enabled is True
        assert result.next_scheduled_run is not None

    @respx.mock
    def test_list_decodes_the_schedules_example(self, schedules: BackupSchedules) -> None:
        respx.get(f"{BASE_URL}/indexes/my-index/backup-schedules").mock(
            return_value=httpx.Response(200, json={"data": [SPEC_SCHEDULE]})
        )

        result = schedules.list(index_name="my-index")

        assert isinstance(result, BackupScheduleList)
        assert len(result) == 1
        assert result.names() == ["daily-compliance-backup"]
        assert [s.schedule_id for s in result.enabled_schedules()] == [SCHEDULE_ID]
        assert result.pagination is None

    @respx.mock
    def test_list_decodes_the_empty_example(self, schedules: BackupSchedules) -> None:
        respx.get(f"{BASE_URL}/indexes/my-index/backup-schedules").mock(
            return_value=httpx.Response(200, json={"data": []})
        )

        result = schedules.list(index_name="my-index")

        assert len(result) == 0
        assert result.names() == []
        assert result.enabled_schedules() == []
        assert result.pagination is None

    @respx.mock
    def test_describe_decodes_the_schedule_example(self, schedules: BackupSchedules) -> None:
        respx.get(f"{BASE_URL}/backup-schedules/{SCHEDULE_ID}").mock(
            return_value=httpx.Response(200, json=SPEC_SCHEDULE)
        )

        result = schedules.describe(schedule_id=SCHEDULE_ID)

        assert result.name == "daily-compliance-backup"
        assert result.index_id == "8cbf7ba6-4135-438e-a3c3-4a89a3298905"
        assert result.project_id == "71ce31ea-75f7-45d6-a147-ef67f661a1b0"

    @respx.mock
    def test_update_decodes_the_updated_example_with_next_run_absent(
        self, schedules: BackupSchedules
    ) -> None:
        """The spec's ``updated`` example omits ``next_scheduled_run`` entirely.

        The schema marks it required-but-nullable, so the example contradicts
        the schema. The model tolerates both spellings, and this pins the
        spec's own bytes rather than a corrected version of them.
        """
        respx.patch(f"{BASE_URL}/backup-schedules/{SCHEDULE_ID}").mock(
            return_value=httpx.Response(200, json=SPEC_UPDATED_SCHEDULE)
        )

        result = schedules.update(
            schedule_id=SCHEDULE_ID, frequency="weekly", retention_days=30, enabled=False
        )

        assert result.frequency == "weekly"
        assert result.retention_expire_after_days == 30
        assert result.enabled is False
        assert result.next_scheduled_run is None

    @respx.mock
    def test_update_decodes_an_explicit_null_next_run(self, schedules: BackupSchedules) -> None:
        """What the backend and the simulator actually send when disabled."""
        payload = {**SPEC_UPDATED_SCHEDULE, "next_scheduled_run": None}
        respx.patch(f"{BASE_URL}/backup-schedules/{SCHEDULE_ID}").mock(
            return_value=httpx.Response(200, json=payload)
        )

        result = schedules.update(schedule_id=SCHEDULE_ID, enabled=False)

        assert result.next_scheduled_run is None

    @respx.mock
    def test_history_decodes_the_ready_example(self, schedules: BackupSchedules) -> None:
        respx.get(f"{BASE_URL}/backup-schedules/{SCHEDULE_ID}/history").mock(
            return_value=httpx.Response(200, json={"data": [SPEC_HISTORY_READY]})
        )

        result = schedules.history(schedule_id=SCHEDULE_ID)

        assert isinstance(result, BackupScheduleHistoryList)
        row = result[0]
        assert isinstance(row, BackupScheduleHistoryItem)
        assert row.status == "Ready"
        assert row.is_scheduled is False
        assert row.scheduled_execution_at is None
        assert row.record_count == 500000
        assert row.namespace_count == 1
        assert row.size_bytes == 104857600
        assert row.schema is not None
        assert isinstance(row.schema.fields["embedding"], DenseVectorField)
        assert result.scheduled() == []

    @respx.mock
    def test_history_decodes_the_scheduled_row_example(self, schedules: BackupSchedules) -> None:
        """The spec's ``Scheduled`` row decodes, including ``scheduled_execution_at``.

        Unreachable against today's backend — see
        :class:`TestBackendShapedHistoryDecode` — but the spec declares it, so
        the SDK decodes it.
        """
        respx.get(f"{BASE_URL}/backup-schedules/{SCHEDULE_ID}/history").mock(
            return_value=httpx.Response(200, json={"data": [SPEC_HISTORY_SCHEDULED]})
        )

        result = schedules.history(schedule_id=SCHEDULE_ID)

        row = result[0]
        assert row.status == "Scheduled"
        assert row.is_scheduled is True
        assert row.scheduled_execution_at is not None
        assert row.scheduled_execution_at.isoformat() == "2026-04-04T06:00:00+00:00"
        assert row.record_count == 0
        assert [r.backup_id for r in result.scheduled()] == [row.backup_id]

    @respx.mock
    def test_history_decodes_a_mixed_page(self, schedules: BackupSchedules) -> None:
        respx.get(f"{BASE_URL}/backup-schedules/{SCHEDULE_ID}/history").mock(
            return_value=httpx.Response(
                200, json={"data": [SPEC_HISTORY_READY, SPEC_HISTORY_SCHEDULED]}
            )
        )

        result = schedules.history(schedule_id=SCHEDULE_ID)

        assert len(result) == 2
        assert [r.status for r in result] == ["Ready", "Scheduled"]
        assert len(result.scheduled()) == 1
        assert json.dumps(result.to_dict())


class TestCreateRequestBody:
    @respx.mock
    def test_body_matches_the_spec_example(self, schedules: BackupSchedules) -> None:
        route = respx.post(f"{BASE_URL}/indexes/my-index/backup-schedules").mock(
            return_value=httpx.Response(201, json=SPEC_SCHEDULE)
        )

        schedules.create(
            index_name="my-index",
            name="daily-compliance-backup",
            frequency="daily",
            retention_days=90,
        )

        assert json.loads(route.calls.last.request.content) == SPEC_CREATE_BODY

    @respx.mock
    def test_schedule_type_is_filled_in_by_the_sdk(self, schedules: BackupSchedules) -> None:
        route = respx.post(f"{BASE_URL}/indexes/my-index/backup-schedules").mock(
            return_value=httpx.Response(201, json=SPEC_SCHEDULE)
        )

        schedules.create(index_name="my-index", name="weekly", frequency="weekly", retention_days=7)

        body = json.loads(route.calls.last.request.content)
        assert body["schedule"] == {"type": "time-based", "frequency": "weekly"}
        assert body["retention"] == {"expire_after_days": 7}

    @respx.mock
    def test_tolerates_the_backends_200_where_the_spec_says_201(
        self, schedules: BackupSchedules
    ) -> None:
        """The backend answers 200; the spec documents 201. Both must work.

        The SDK does not assert success status codes, so this pins that the
        tolerance is real rather than incidental.
        """
        respx.post(f"{BASE_URL}/indexes/my-index/backup-schedules").mock(
            return_value=httpx.Response(200, json=SPEC_SCHEDULE)
        )

        result = schedules.create(
            index_name="my-index",
            name="daily-compliance-backup",
            frequency="daily",
            retention_days=90,
        )

        assert result.schedule_id == SCHEDULE_ID

    @respx.mock
    def test_a_long_schedule_name_is_sent_rather_than_rejected(
        self, schedules: BackupSchedules
    ) -> None:
        """A name over 28 chars is a documented footgun, not a client-side error.

        Each run names its backup ``"{name}-YYYYMMDDTHHMMSSZ"`` -- a fixed
        17-character suffix against a 45-character resource name limit, so a
        schedule name longer than 28 produces backup names the backup
        endpoints would themselves reject. Nothing server-side validates it
        either, so enforcing it here would reject names the API accepts. The
        caveat lives in the docstring; this pins that the value still reaches
        the wire untouched, so a future decision to enforce is deliberate.
        """
        long_name = "a" * 29
        route = respx.post(f"{BASE_URL}/indexes/my-index/backup-schedules").mock(
            return_value=httpx.Response(201, json=SPEC_SCHEDULE)
        )

        schedules.create(
            index_name="my-index", name=long_name, frequency="daily", retention_days=90
        )

        assert json.loads(route.calls.last.request.content)["name"] == long_name
        assert len(f"{long_name}-20260403T060000Z") > 45


class TestUpdateRequestBody:
    @respx.mock
    def test_body_matches_the_spec_example(self, schedules: BackupSchedules) -> None:
        route = respx.patch(f"{BASE_URL}/backup-schedules/{SCHEDULE_ID}").mock(
            return_value=httpx.Response(200, json=SPEC_UPDATED_SCHEDULE)
        )

        schedules.update(
            schedule_id=SCHEDULE_ID, frequency="weekly", retention_days=30, enabled=False
        )

        assert json.loads(route.calls.last.request.content) == SPEC_UPDATE_BODY

    @pytest.mark.parametrize(
        ("kwargs", "expected"),
        [
            ({"enabled": False}, {"enabled": False}),
            ({"enabled": True}, {"enabled": True}),
            ({"frequency": "monthly"}, {"frequency": "monthly"}),
            ({"retention_days": 14}, {"retention": {"expire_after_days": 14}}),
            ({"frequency": "daily", "enabled": True}, {"frequency": "daily", "enabled": True}),
        ],
    )
    @respx.mock
    def test_omits_every_unset_field(
        self, schedules: BackupSchedules, kwargs: dict[str, Any], expected: dict[str, Any]
    ) -> None:
        route = respx.patch(f"{BASE_URL}/backup-schedules/{SCHEDULE_ID}").mock(
            return_value=httpx.Response(200, json=SPEC_SCHEDULE)
        )

        schedules.update(schedule_id=SCHEDULE_ID, **kwargs)

        assert json.loads(route.calls.last.request.content) == expected

    @respx.mock
    def test_no_fields_sends_an_empty_object(self, schedules: BackupSchedules) -> None:
        route = respx.patch(f"{BASE_URL}/backup-schedules/{SCHEDULE_ID}").mock(
            return_value=httpx.Response(200, json=SPEC_SCHEDULE)
        )

        result = schedules.update(schedule_id=SCHEDULE_ID)

        assert json.loads(route.calls.last.request.content) == {}
        assert result.schedule_id == SCHEDULE_ID


class TestDelete:
    @respx.mock
    def test_returns_none_on_204_without_parsing_a_body(self, schedules: BackupSchedules) -> None:
        respx.delete(f"{BASE_URL}/backup-schedules/{SCHEDULE_ID}").mock(
            return_value=httpx.Response(204)
        )

        assert schedules.delete(schedule_id=SCHEDULE_ID) is None

    @respx.mock
    def test_ignores_a_body_the_spec_does_not_declare(self, schedules: BackupSchedules) -> None:
        """A 204 with an unexpected body must not become a parse error."""
        respx.delete(f"{BASE_URL}/backup-schedules/{SCHEDULE_ID}").mock(
            return_value=httpx.Response(204, content=b"not json")
        )

        assert schedules.delete(schedule_id=SCHEDULE_ID) is None

    @respx.mock
    def test_second_delete_404s(self, schedules: BackupSchedules) -> None:
        respx.delete(f"{BASE_URL}/backup-schedules/{SCHEDULE_ID}").mock(
            return_value=httpx.Response(
                404, json=_error("NOT_FOUND", "Backup schedule not found.", 404)
            )
        )

        with pytest.raises(NotFoundError) as exc:
            schedules.delete(schedule_id=SCHEDULE_ID)

        assert "Backup schedule not found." in str(exc.value)


class TestQueryParams:
    @respx.mock
    def test_list_omits_both_params_when_unset(self, schedules: BackupSchedules) -> None:
        route = respx.get(f"{BASE_URL}/indexes/my-index/backup-schedules").mock(
            return_value=httpx.Response(200, json={"data": []})
        )

        schedules.list(index_name="my-index")

        assert route.calls.last.request.url.params == httpx.QueryParams()

    @respx.mock
    def test_list_sends_camel_cased_token_and_drops_the_limit(
        self, schedules: BackupSchedules
    ) -> None:
        route = respx.get(f"{BASE_URL}/indexes/my-index/backup-schedules").mock(
            return_value=httpx.Response(200, json={"data": []})
        )

        schedules.list(index_name="my-index", limit=5, pagination_token="tok-1")

        params = route.calls.last.request.url.params
        assert params["paginationToken"] == "tok-1"
        assert "limit" not in params

    @respx.mock
    def test_list_sends_the_limit_when_there_is_no_token(self, schedules: BackupSchedules) -> None:
        route = respx.get(f"{BASE_URL}/indexes/my-index/backup-schedules").mock(
            return_value=httpx.Response(200, json={"data": []})
        )

        schedules.list(index_name="my-index", limit=5)

        assert route.calls.last.request.url.params["limit"] == "5"

    @respx.mock
    def test_history_sends_camel_cased_token_and_drops_the_limit(
        self, schedules: BackupSchedules
    ) -> None:
        route = respx.get(f"{BASE_URL}/backup-schedules/{SCHEDULE_ID}/history").mock(
            return_value=httpx.Response(200, json={"data": []})
        )

        schedules.history(schedule_id=SCHEDULE_ID, limit=3, pagination_token="tok-2")

        params = route.calls.last.request.url.params
        assert params["paginationToken"] == "tok-2"
        assert "limit" not in params

    @respx.mock
    def test_history_sends_the_limit_when_there_is_no_token(
        self, schedules: BackupSchedules
    ) -> None:
        route = respx.get(f"{BASE_URL}/backup-schedules/{SCHEDULE_ID}/history").mock(
            return_value=httpx.Response(200, json={"data": []})
        )

        schedules.history(schedule_id=SCHEDULE_ID, limit=3)

        assert route.calls.last.request.url.params["limit"] == "3"

    @respx.mock
    def test_list_carries_the_pagination_envelope(self, schedules: BackupSchedules) -> None:
        respx.get(f"{BASE_URL}/indexes/my-index/backup-schedules").mock(
            return_value=httpx.Response(
                200, json={"data": [SPEC_SCHEDULE], "pagination": {"next": "tok-next"}}
            )
        )

        result = schedules.list(index_name="my-index")

        assert result.pagination is not None
        assert result.pagination.next == "tok-next"

    @respx.mock
    def test_a_null_pagination_envelope_reads_as_final(self, schedules: BackupSchedules) -> None:
        respx.get(f"{BASE_URL}/indexes/my-index/backup-schedules").mock(
            return_value=httpx.Response(200, json={"data": [SPEC_SCHEDULE], "pagination": None})
        )

        result = schedules.list(index_name="my-index")

        assert result.pagination is None


class TestClientSideValidation:
    @pytest.mark.parametrize("bad", ["", "   "])
    def test_create_rejects_an_empty_index_name(self, schedules: BackupSchedules, bad: str) -> None:
        with pytest.raises(PineconeValueError, match="index_name"):
            schedules.create(index_name=bad, name="s", frequency="daily", retention_days=1)

    @pytest.mark.parametrize("bad", ["", "   "])
    def test_create_rejects_an_empty_schedule_name(
        self, schedules: BackupSchedules, bad: str
    ) -> None:
        with pytest.raises(PineconeValueError, match="name"):
            schedules.create(index_name="my-index", name=bad, frequency="daily", retention_days=1)

    @pytest.mark.parametrize("bad", ["hourly", "0 * * * *", "DAILY", "", "yearly"])
    def test_create_rejects_an_unsupported_frequency_naming_all_three(
        self, schedules: BackupSchedules, bad: str
    ) -> None:
        with pytest.raises(PineconeValueError) as exc:
            schedules.create(index_name="my-index", name="s", frequency=bad, retention_days=1)

        message = str(exc.value)
        assert "daily | weekly | monthly" in message
        assert repr(bad) in message

    def test_a_cron_string_says_cron_is_not_supported(self, schedules: BackupSchedules) -> None:
        with pytest.raises(PineconeValueError, match="cron expressions are not supported"):
            schedules.create(
                index_name="my-index", name="s", frequency="0 6 * * *", retention_days=1
            )

    @pytest.mark.parametrize("bad", [0, -1, -365])
    def test_create_rejects_retention_below_one(self, schedules: BackupSchedules, bad: int) -> None:
        with pytest.raises(PineconeValueError) as exc:
            schedules.create(index_name="my-index", name="s", frequency="daily", retention_days=bad)

        assert "max_backup_retention_days" in str(exc.value)
        assert f"got {bad}" in str(exc.value)

    def test_update_rejects_an_unsupported_frequency(self, schedules: BackupSchedules) -> None:
        with pytest.raises(PineconeValueError, match="daily \\| weekly \\| monthly"):
            schedules.update(schedule_id=SCHEDULE_ID, frequency="hourly")

    def test_update_rejects_retention_below_one(self, schedules: BackupSchedules) -> None:
        with pytest.raises(PineconeValueError, match="got 0"):
            schedules.update(schedule_id=SCHEDULE_ID, retention_days=0)

    @pytest.mark.parametrize(
        "call",
        [
            lambda s: s.describe(schedule_id=""),
            lambda s: s.get(schedule_id=""),
            lambda s: s.delete(schedule_id=""),
            lambda s: s.update(schedule_id=""),
            lambda s: s.history(schedule_id=""),
            lambda s: s.iter_history(schedule_id=""),
            lambda s: s.list(index_name=""),
            lambda s: s.iter_schedules(index_name=""),
        ],
    )
    def test_empty_path_parameters_raise(self, schedules: BackupSchedules, call: Any) -> None:
        with pytest.raises(PineconeValueError):
            call(schedules)

    @pytest.mark.parametrize("bad", [0, -1])
    def test_non_positive_limits_raise(self, schedules: BackupSchedules, bad: int) -> None:
        with pytest.raises(PineconeValueError, match="limit"):
            schedules.list(index_name="my-index", limit=bad)
        with pytest.raises(PineconeValueError, match="limit"):
            schedules.history(schedule_id=SCHEDULE_ID, limit=bad)
        with pytest.raises(PineconeValueError, match="limit"):
            schedules.iter_schedules(index_name="my-index", limit=bad)
        with pytest.raises(PineconeValueError, match="limit"):
            schedules.iter_history(schedule_id=SCHEDULE_ID, limit=bad)

    @respx.mock
    def test_validation_happens_before_any_http(self, schedules: BackupSchedules) -> None:
        route = respx.post(f"{BASE_URL}/indexes/my-index/backup-schedules").mock(
            return_value=httpx.Response(201, json=SPEC_SCHEDULE)
        )

        with pytest.raises(PineconeValueError):
            schedules.create(index_name="my-index", name="s", frequency="hourly", retention_days=90)

        assert not route.called

    def test_a_client_side_error_is_also_a_plain_value_error(
        self, schedules: BackupSchedules
    ) -> None:
        """``except ValueError`` still catches it, as it did on the raw model."""
        with pytest.raises(ValueError):
            schedules.create(index_name="my-index", name="s", frequency="hourly", retention_days=1)


class TestPlanGate:
    @respx.mock
    def test_the_specs_403_gains_the_upgrade_hint(self, schedules: BackupSchedules) -> None:
        respx.post(f"{BASE_URL}/indexes/my-index/backup-schedules").mock(
            return_value=httpx.Response(403, json=PLAN_GATE_403)
        )

        with pytest.raises(ForbiddenError) as exc:
            schedules.create(index_name="my-index", name="s", frequency="daily", retention_days=90)

        message = exc.value.message
        assert message.startswith("Scheduled backups are not available for your plan")
        assert SCHEDULED_BACKUPS_PLAN_HINT in message
        assert exc.value.error_code == "PERMISSION_DENIED"
        assert exc.value.status_code == 403

    @respx.mock
    def test_the_backends_403_keeps_its_own_wording_as_the_prefix(
        self, schedules: BackupSchedules
    ) -> None:
        respx.post(f"{BASE_URL}/indexes/my-index/backup-schedules").mock(
            return_value=httpx.Response(403, json=BACKEND_PLAN_GATE_403)
        )

        with pytest.raises(ForbiddenError) as exc:
            schedules.create(index_name="my-index", name="s", frequency="daily", retention_days=90)

        assert exc.value.message.startswith("Your organization's plan does not include backups.")
        assert SCHEDULED_BACKUPS_PLAN_HINT in exc.value.message

    @respx.mock
    def test_a_403_that_is_not_the_plan_gate_is_left_alone(
        self, schedules: BackupSchedules
    ) -> None:
        payload = _error(
            "PERMISSION_DENIED", "API key does not have write access to this project.", 403
        )
        respx.get(f"{BASE_URL}/backup-schedules/{SCHEDULE_ID}").mock(
            return_value=httpx.Response(403, json=payload)
        )

        with pytest.raises(ForbiddenError) as exc:
            schedules.describe(schedule_id=SCHEDULE_ID)

        assert exc.value.message == "API key does not have write access to this project."
        assert SCHEDULED_BACKUPS_PLAN_HINT not in exc.value.message

    @respx.mock
    def test_the_hint_reaches_the_listing_paths_too(self, schedules: BackupSchedules) -> None:
        respx.get(f"{BASE_URL}/backup-schedules/{SCHEDULE_ID}/history").mock(
            return_value=httpx.Response(403, json=PLAN_GATE_403)
        )

        with pytest.raises(ForbiddenError) as exc:
            schedules.history(schedule_id=SCHEDULE_ID)

        assert SCHEDULED_BACKUPS_PLAN_HINT in exc.value.message

    @respx.mock
    def test_the_hint_reaches_the_paginators_too(self, schedules: BackupSchedules) -> None:
        respx.get(f"{BASE_URL}/indexes/my-index/backup-schedules").mock(
            return_value=httpx.Response(403, json=PLAN_GATE_403)
        )

        with pytest.raises(ForbiddenError) as exc:
            list(schedules.iter_schedules(index_name="my-index"))

        assert SCHEDULED_BACKUPS_PLAN_HINT in exc.value.message


class TestServerErrorsSurfaceIntact:
    @respx.mock
    def test_create_409_keeps_the_backends_message(self, schedules: BackupSchedules) -> None:
        respx.post(f"{BASE_URL}/indexes/my-index/backup-schedules").mock(
            return_value=httpx.Response(409, json=ALREADY_ENABLED_409)
        )

        with pytest.raises(ConflictError) as exc:
            schedules.create(index_name="my-index", name="s", frequency="daily", retention_days=90)

        assert exc.value.message == (
            "This index already has an enabled backup schedule. Disable or delete it first."
        )
        assert exc.value.error_code == "ALREADY_EXISTS"

    @respx.mock
    def test_re_enable_409_keeps_its_distinct_wording(self, schedules: BackupSchedules) -> None:
        """The PATCH path words its 409 differently from the create path."""
        message = (
            "This index already has an enabled backup schedule. "
            "Disable it first before re-enabling this one."
        )
        respx.patch(f"{BASE_URL}/backup-schedules/{SCHEDULE_ID}").mock(
            return_value=httpx.Response(409, json=_error("ALREADY_EXISTS", message, 409))
        )

        with pytest.raises(ConflictError) as exc:
            schedules.update(schedule_id=SCHEDULE_ID, enabled=True)

        assert exc.value.message == message

    @respx.mock
    def test_create_404_for_a_missing_index(self, schedules: BackupSchedules) -> None:
        respx.post(f"{BASE_URL}/indexes/nope/backup-schedules").mock(
            return_value=httpx.Response(404, json=_error("NOT_FOUND", "Index nope not found.", 404))
        )

        with pytest.raises(NotFoundError, match=re.escape("Index nope not found.")):
            schedules.create(index_name="nope", name="s", frequency="daily", retention_days=90)

    @respx.mock
    def test_create_400_for_a_pod_index(self, schedules: BackupSchedules) -> None:
        respx.post(f"{BASE_URL}/indexes/pod-index/backup-schedules").mock(
            return_value=httpx.Response(
                400,
                json=_error("INVALID_ARGUMENT", "Pod indexes do not support backup schedules", 400),
            )
        )

        with pytest.raises(Exception, match="Pod indexes do not support backup schedules"):
            schedules.create(index_name="pod-index", name="s", frequency="daily", retention_days=90)

    @respx.mock
    def test_create_400_for_retention_above_the_projects_maximum(
        self, schedules: BackupSchedules
    ) -> None:
        """The SDK does not know the upper bound, so the server enforces it."""
        message = "retention.expire_after_days must be between 1 and 365, got 400."
        route = respx.post(f"{BASE_URL}/indexes/my-index/backup-schedules").mock(
            return_value=httpx.Response(400, json=_error("INVALID_ARGUMENT", message, 400))
        )

        with pytest.raises(Exception, match="must be between 1 and 365"):
            schedules.create(index_name="my-index", name="s", frequency="daily", retention_days=400)

        assert json.loads(route.calls.last.request.content)["retention"] == {
            "expire_after_days": 400
        }

    @respx.mock
    def test_describe_404(self, schedules: BackupSchedules) -> None:
        respx.get(f"{BASE_URL}/backup-schedules/{SCHEDULE_ID}").mock(
            return_value=httpx.Response(
                404, json=_error("NOT_FOUND", "Backup schedule not found.", 404)
            )
        )

        with pytest.raises(NotFoundError, match=re.escape("Backup schedule not found.")):
            schedules.describe(schedule_id=SCHEDULE_ID)


class TestPaginators:
    @respx.mock
    def test_iter_schedules_walks_every_page(self, schedules: BackupSchedules) -> None:
        first = {
            "data": [{**SPEC_SCHEDULE, "schedule_id": "s-1"}],
            "pagination": {"next": "tok-2"},
        }
        second = {"data": [{**SPEC_SCHEDULE, "schedule_id": "s-2"}], "pagination": None}
        route = respx.get(f"{BASE_URL}/indexes/my-index/backup-schedules").mock(
            side_effect=[httpx.Response(200, json=first), httpx.Response(200, json=second)]
        )

        result = schedules.iter_schedules(index_name="my-index")

        assert isinstance(result, Paginator)
        assert [s.schedule_id for s in result] == ["s-1", "s-2"]
        assert route.call_count == 2
        assert route.calls[1].request.url.params["paginationToken"] == "tok-2"

    @respx.mock
    def test_iter_history_walks_every_page(self, schedules: BackupSchedules) -> None:
        first = {
            "data": [{**SPEC_HISTORY_READY, "backup_id": "b-1"}],
            "pagination": {"next": "tok-2"},
        }
        second = {"data": [{**SPEC_HISTORY_SCHEDULED, "backup_id": "b-2"}]}
        route = respx.get(f"{BASE_URL}/backup-schedules/{SCHEDULE_ID}/history").mock(
            side_effect=[httpx.Response(200, json=first), httpx.Response(200, json=second)]
        )

        rows = list(schedules.iter_history(schedule_id=SCHEDULE_ID))

        assert [r.backup_id for r in rows] == ["b-1", "b-2"]
        assert [r.status for r in rows] == ["Ready", "Scheduled"]
        assert route.call_count == 2

    @respx.mock
    def test_an_absent_pagination_key_terminates_iteration(
        self, schedules: BackupSchedules
    ) -> None:
        route = respx.get(f"{BASE_URL}/indexes/my-index/backup-schedules").mock(
            return_value=httpx.Response(200, json={"data": [SPEC_SCHEDULE]})
        )

        assert len(list(schedules.iter_schedules(index_name="my-index"))) == 1
        assert route.call_count == 1

    @respx.mock
    def test_limit_caps_the_yield_and_is_forwarded(self, schedules: BackupSchedules) -> None:
        page = {
            "data": [
                {**SPEC_SCHEDULE, "schedule_id": "s-1"},
                {**SPEC_SCHEDULE, "schedule_id": "s-2"},
            ],
            "pagination": {"next": "tok-2"},
        }
        route = respx.get(f"{BASE_URL}/indexes/my-index/backup-schedules").mock(
            return_value=httpx.Response(200, json=page)
        )

        result = list(schedules.iter_schedules(index_name="my-index", limit=1))

        assert [s.schedule_id for s in result] == ["s-1"]
        assert route.calls[0].request.url.params["limit"] == "1"

    @respx.mock
    def test_iterators_start_from_a_supplied_token(self, schedules: BackupSchedules) -> None:
        route = respx.get(f"{BASE_URL}/backup-schedules/{SCHEDULE_ID}/history").mock(
            return_value=httpx.Response(200, json={"data": [], "pagination": None})
        )

        list(schedules.iter_history(schedule_id=SCHEDULE_ID, pagination_token="resume-me"))

        assert route.calls[0].request.url.params["paginationToken"] == "resume-me"


class TestBackendShapedHistoryDecode:
    """Doctrine pins (#224): spec-shaped surface, backend-tolerant decode.

    ``v202607`` nests ``/backup-schedules`` on ``v202604::backup_schedules``,
    whose history handler answers with rows of ``BackupResponse`` rather than
    ``BackupScheduleHistoryItem``. These pin what that really means for
    callers today, so a backend change breaks a test instead of a user.
    """

    def _backend_row(self, **overrides: Any) -> dict[str, Any]:
        row: dict[str, Any] = {
            "backup_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "source_index_id": "8cbf7ba6-4135-438e-a3c3-4a89a3298905",
            "source_index_name": "my-index",
            "name": None,
            "status": "Initializing",
            "cloud": "aws",
            "region": "us-east-1",
            "description": None,
            "schema": {"fields": {"genre": {"filterable": True}}},
            "record_count": None,
            "namespace_count": None,
            "size_bytes": None,
            "dimension": 1536,
            "tags": None,
            "created_at": "2026-04-03T06:00:00+00:00",
        }
        row.update(overrides)
        return row

    @respx.mock
    def test_the_backends_row_decodes_rather_than_raising(self, schedules: BackupSchedules) -> None:
        respx.get(f"{BASE_URL}/backup-schedules/{SCHEDULE_ID}/history").mock(
            return_value=httpx.Response(200, json={"data": [self._backend_row()]})
        )

        row = schedules.history(schedule_id=SCHEDULE_ID)[0]

        assert row.name is None
        assert row.record_count is None
        assert row.namespace_count is None
        assert row.size_bytes is None

    @respx.mock
    def test_the_legacy_metadata_schema_decodes_through_the_retry(
        self, schedules: BackupSchedules
    ) -> None:
        """Without ``decode_backups_envelope`` this raises ResponseParsingError."""
        respx.get(f"{BASE_URL}/backup-schedules/{SCHEDULE_ID}/history").mock(
            return_value=httpx.Response(200, json={"data": [self._backend_row()]})
        )

        row = schedules.history(schedule_id=SCHEDULE_ID)[0]

        assert row.schema is not None
        assert isinstance(row.schema.fields["genre"], LegacyMetadataField)

    @respx.mock
    def test_the_undocumented_dimension_key_is_not_resurrected(
        self, schedules: BackupSchedules
    ) -> None:
        respx.get(f"{BASE_URL}/backup-schedules/{SCHEDULE_ID}/history").mock(
            return_value=httpx.Response(200, json={"data": [self._backend_row()]})
        )

        row = schedules.history(schedule_id=SCHEDULE_ID)[0]

        with pytest.raises(AttributeError):
            _ = row.dimension
        assert "dimension" not in row.to_dict()

    @respx.mock
    def test_scheduled_execution_at_is_none_against_todays_backend(
        self, schedules: BackupSchedules
    ) -> None:
        """The backend response has no such field, and never says ``Scheduled``.

        The status remap on the delegated handler produces
        ``Initializing`` / ``Ready`` / ``InitializationFailed`` only, so the
        spec's ``Scheduled`` row is unreachable server-side today. The field
        stays typed and tested; it just returns ``None`` until the backend
        graduates.
        """
        respx.get(f"{BASE_URL}/backup-schedules/{SCHEDULE_ID}/history").mock(
            return_value=httpx.Response(200, json={"data": [self._backend_row()]})
        )

        row = schedules.history(schedule_id=SCHEDULE_ID)[0]

        assert row.scheduled_execution_at is None
        assert row.is_scheduled is False

    @respx.mock
    def test_initialization_failed_decodes_verbatim(self, schedules: BackupSchedules) -> None:
        respx.get(f"{BASE_URL}/backup-schedules/{SCHEDULE_ID}/history").mock(
            return_value=httpx.Response(
                200, json={"data": [self._backend_row(status="InitializationFailed")]}
            )
        )

        assert schedules.history(schedule_id=SCHEDULE_ID)[0].status == "InitializationFailed"

    @respx.mock
    def test_a_schedule_listing_needs_no_legacy_retry(self, schedules: BackupSchedules) -> None:
        """Schedules carry no schema, so a malformed one is a real error."""
        respx.get(f"{BASE_URL}/indexes/my-index/backup-schedules").mock(
            return_value=httpx.Response(200, json={"data": [{"schedule_id": "s-1"}]})
        )

        with pytest.raises(ResponseParsingError):
            schedules.list(index_name="my-index")


class TestSurface:
    def test_repr(self, schedules: BackupSchedules) -> None:
        assert repr(schedules) == "BackupSchedules()"

    @respx.mock
    def test_get_is_an_alias_for_describe(self, schedules: BackupSchedules) -> None:
        route = respx.get(f"{BASE_URL}/backup-schedules/{SCHEDULE_ID}").mock(
            return_value=httpx.Response(200, json=SPEC_SCHEDULE)
        )

        described = schedules.describe(schedule_id=SCHEDULE_ID)
        gotten = schedules.get(schedule_id=SCHEDULE_ID)

        assert described == gotten
        assert route.call_count == 2

    def test_every_method_is_keyword_only(self, schedules: BackupSchedules) -> None:
        with pytest.raises(TypeError):
            schedules.describe(SCHEDULE_ID)  # type: ignore[misc]
        with pytest.raises(TypeError):
            schedules.list("my-index")  # type: ignore[misc]

    def test_the_namespace_is_reachable_and_cached_on_the_client(self) -> None:
        from pinecone import Pinecone

        pc = Pinecone(api_key="test-key", host=BASE_URL)

        assert isinstance(pc.backup_schedules, BackupSchedules)
        assert pc.backup_schedules is pc.backup_schedules
