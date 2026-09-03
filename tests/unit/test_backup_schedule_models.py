"""Tests for backup schedule models (2026-07 API).

Payloads marked "spec example" are copied verbatim from
``apis/src/release/db/control/resources`` (``indexes/CreateBackupSchedule.yaml``,
``indexes/ListBackupSchedules.yaml``, ``backup-schedules/DescribeBackupSchedule.yaml``,
``backup-schedules/UpdateBackupSchedule.yaml``, and
``backup-schedules/ListBackupScheduleHistory.yaml``) so a spec change breaks a
test here.

Payloads marked "backend example" are shaped after what pinecone-db actually
returns, which for schedule *history* is its shared backup handler
(``v202604/backups.rs`` ``BackupResponse``) rather than the spec's
``BackupScheduleHistoryItem``. See the divergence recorded on issue #224.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import msgspec
import pytest
from msgspec import Struct

from pinecone._internal.adapters.backups_adapter import decode_backups_envelope
from pinecone.errors.exceptions import ResponseParsingError
from pinecone.models.backups.list import BackupScheduleHistoryList, BackupScheduleList
from pinecone.models.backups.schedules import (
    BACKUP_SCHEDULE_FREQUENCIES,
    SCHEDULE_TYPE_TIME_BASED,
    BackupScheduleHistoryItem,
    BackupScheduleModel,
    CreateBackupScheduleRequest,
    UpdateBackupScheduleRequest,
)
from pinecone.models.indexes.schema import DenseVectorField, LegacyMetadataField
from pinecone.models.vectors.responses import Pagination

SPEC_CREATE_REQUEST: dict[str, Any] = {
    "name": "daily-compliance-backup",
    "schedule": {"type": "time-based", "frequency": "daily"},
    "retention": {"expire_after_days": 90},
}

SPEC_UPDATE_REQUEST: dict[str, Any] = {
    "frequency": "weekly",
    "retention": {"expire_after_days": 30},
    "enabled": False,
}

SPEC_CREATED_SCHEDULE: dict[str, Any] = {
    "schedule_id": "e88f7273-42aa-47e9-af73-593827136867",
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

SPEC_DESCRIBED_SCHEDULE: dict[str, Any] = dict(SPEC_CREATED_SCHEDULE)

SPEC_UPDATED_SCHEDULE: dict[str, Any] = {
    "schedule_id": "e88f7273-42aa-47e9-af73-593827136867",
    "name": "daily-compliance-backup",
    "index_id": "8cbf7ba6-4135-438e-a3c3-4a89a3298905",
    "project_id": "71ce31ea-75f7-45d6-a147-ef67f661a1b0",
    "schedule_type": "time-based",
    "frequency": "weekly",
    "retention_expire_after_days": 30,
    "enabled": False,
    "created_at": "2026-04-02T18:22:56.712605+00:00",
}

SPEC_SCHEDULE_LIST: dict[str, Any] = {"data": [dict(SPEC_CREATED_SCHEDULE)]}

SPEC_EMPTY_SCHEDULE_LIST: dict[str, Any] = {"data": []}

_SPEC_HISTORY_SCHEMA: dict[str, Any] = {
    "fields": {"embedding": {"type": "dense_vector", "dimension": 1536, "metric": "cosine"}}
}

SPEC_HISTORY_READY: dict[str, Any] = {
    "data": [
        {
            "backup_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "source_index_id": "8cbf7ba6-4135-438e-a3c3-4a89a3298905",
            "source_index_name": "my-index",
            "name": "daily-compliance-backup-20260403T060000Z",
            "status": "Ready",
            "cloud": "aws",
            "region": "us-east-1",
            "schema": _SPEC_HISTORY_SCHEMA,
            "record_count": 500000,
            "namespace_count": 1,
            "size_bytes": 104857600,
            "created_at": "2026-04-03T06:00:00+00:00",
        }
    ]
}

SPEC_HISTORY_SCHEDULED: dict[str, Any] = {
    "data": [
        {
            "backup_id": "b2c3d4e5-f6a7-8901-bcde-f12345678901",
            "source_index_id": "8cbf7ba6-4135-438e-a3c3-4a89a3298905",
            "source_index_name": "my-index",
            "name": "daily-compliance-backup-20260404T060000Z",
            "status": "Scheduled",
            "cloud": "aws",
            "region": "us-east-1",
            "schema": _SPEC_HISTORY_SCHEMA,
            "record_count": 0,
            "namespace_count": 0,
            "size_bytes": 0,
            "created_at": "2026-04-03T06:00:01+00:00",
            "scheduled_execution_at": "2026-04-04T06:00:00+00:00",
        }
    ]
}

BACKEND_DISABLED_SCHEDULE: dict[str, Any] = {
    "schedule_id": "e88f7273-42aa-47e9-af73-593827136867",
    "name": "daily-compliance-backup",
    "index_id": "8cbf7ba6-4135-438e-a3c3-4a89a3298905",
    "project_id": "71ce31ea-75f7-45d6-a147-ef67f661a1b0",
    "schedule_type": "time-based",
    "frequency": "weekly",
    "retention_expire_after_days": 30,
    "enabled": False,
    "next_scheduled_run": None,
    "created_at": "2026-04-02T18:22:56.712605+00:00",
}

BACKEND_HISTORY_ROW: dict[str, Any] = {
    "backup_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "source_index_id": "8cbf7ba6-4135-438e-a3c3-4a89a3298905",
    "source_index_name": "my-index",
    "tags": None,
    "name": None,
    "description": None,
    "status": "InitializationFailed",
    "cloud": "aws",
    "region": "us-east-1",
    "dimension": 1536,
    "schema": {"fields": {"genre": {"filterable": True}}},
    "record_count": None,
    "namespace_count": None,
    "size_bytes": None,
    "created_at": "2026-04-03T06:00:00+00:00",
}


class _ScheduleListEnvelope(Struct, kw_only=True):
    data: list[BackupScheduleModel] = []
    pagination: Pagination | None = None


class _HistoryEnvelope(Struct, kw_only=True):
    data: list[BackupScheduleHistoryItem] = []
    pagination: Pagination | None = None


def _decode_schedule(payload: dict[str, Any]) -> BackupScheduleModel:
    return msgspec.json.decode(msgspec.json.encode(payload), type=BackupScheduleModel)


def _decode_history(payload: dict[str, Any]) -> _HistoryEnvelope:
    return decode_backups_envelope(msgspec.json.encode(payload), _HistoryEnvelope)


def _make_schedule(**overrides: object) -> BackupScheduleModel:
    defaults: dict[str, object] = {
        "schedule_id": "sched-1",
        "name": "daily-compliance-backup",
        "index_id": "idx-1",
        "project_id": "proj-1",
        "schedule_type": "time-based",
        "frequency": "daily",
        "retention_expire_after_days": 90,
        "enabled": True,
        "created_at": datetime(2026, 4, 2, 18, 22, 56, tzinfo=timezone.utc),
        "next_scheduled_run": datetime(2026, 4, 3, 6, tzinfo=timezone.utc),
    }
    defaults.update(overrides)
    return BackupScheduleModel(**defaults)  # type: ignore[arg-type]


class TestSpecScheduleExamples:
    def test_created_schedule_decodes(self) -> None:
        schedule = _decode_schedule(SPEC_CREATED_SCHEDULE)
        assert schedule.schedule_id == "e88f7273-42aa-47e9-af73-593827136867"
        assert schedule.name == "daily-compliance-backup"
        assert schedule.index_id == "8cbf7ba6-4135-438e-a3c3-4a89a3298905"
        assert schedule.project_id == "71ce31ea-75f7-45d6-a147-ef67f661a1b0"
        assert schedule.schedule_type == SCHEDULE_TYPE_TIME_BASED
        assert schedule.frequency == "daily"
        assert schedule.retention_expire_after_days == 90
        assert schedule.enabled is True
        assert schedule.next_scheduled_run == datetime(2026, 4, 3, 6, tzinfo=timezone.utc)
        assert schedule.created_at == datetime(2026, 4, 2, 18, 22, 56, 712605, tzinfo=timezone.utc)

    def test_described_schedule_decodes(self) -> None:
        assert _decode_schedule(SPEC_DESCRIBED_SCHEDULE).enabled is True

    def test_updated_schedule_decodes_with_next_run_absent(self) -> None:
        assert "next_scheduled_run" not in SPEC_UPDATED_SCHEDULE
        schedule = _decode_schedule(SPEC_UPDATED_SCHEDULE)
        assert schedule.enabled is False
        assert schedule.next_scheduled_run is None
        assert schedule.frequency == "weekly"
        assert schedule.retention_expire_after_days == 30

    def test_disabled_schedule_decodes_with_next_run_null(self) -> None:
        schedule = _decode_schedule(BACKEND_DISABLED_SCHEDULE)
        assert schedule.enabled is False
        assert schedule.next_scheduled_run is None

    def test_schedule_list_decodes(self) -> None:
        envelope = msgspec.json.decode(
            msgspec.json.encode(SPEC_SCHEDULE_LIST), type=_ScheduleListEnvelope
        )
        listing = BackupScheduleList(envelope.data, pagination=envelope.pagination)
        assert len(listing) == 1
        assert listing.names() == ["daily-compliance-backup"]
        assert listing.pagination is None

    def test_empty_schedule_list_decodes(self) -> None:
        envelope = msgspec.json.decode(
            msgspec.json.encode(SPEC_EMPTY_SCHEDULE_LIST), type=_ScheduleListEnvelope
        )
        listing = BackupScheduleList(envelope.data)
        assert len(listing) == 0
        assert listing.names() == []
        assert listing.enabled_schedules() == []
        assert listing.to_dict() == {"data": []}

    def test_missing_required_field_raises(self) -> None:
        payload = dict(SPEC_CREATED_SCHEDULE)
        del payload["schedule_id"]
        with pytest.raises(msgspec.ValidationError):
            _decode_schedule(payload)


class TestNextScheduledRunSemantics:
    def test_none_when_disabled(self) -> None:
        schedule = _make_schedule(enabled=False, next_scheduled_run=None)
        assert schedule.enabled is False
        assert schedule.next_scheduled_run is None

    def test_datetime_is_comparable(self) -> None:
        schedule = _make_schedule()
        assert schedule.next_scheduled_run is not None
        assert schedule.next_scheduled_run > schedule.created_at

    def test_naive_timestamp_still_decodes(self) -> None:
        payload = dict(SPEC_CREATED_SCHEDULE)
        payload["next_scheduled_run"] = "2026-04-03T06:00:00"
        assert _decode_schedule(payload).next_scheduled_run == datetime(2026, 4, 3, 6)

    def test_zulu_timestamp_decodes(self) -> None:
        payload = dict(SPEC_CREATED_SCHEDULE)
        payload["next_scheduled_run"] = "2026-04-03T06:00:00Z"
        assert _decode_schedule(payload).next_scheduled_run == datetime(
            2026, 4, 3, 6, tzinfo=timezone.utc
        )


class TestScheduleModelSurface:
    def test_to_dict_renders_rfc3339_strings(self) -> None:
        result = _make_schedule().to_dict()
        assert result["created_at"] == "2026-04-02T18:22:56Z"
        assert result["next_scheduled_run"] == "2026-04-03T06:00:00Z"
        assert msgspec.json.encode(result)

    def test_to_dict_keeps_none_next_run(self) -> None:
        result = _make_schedule(enabled=False, next_scheduled_run=None).to_dict()
        assert result["next_scheduled_run"] is None

    def test_to_dict_covers_every_field(self) -> None:
        schedule = _make_schedule()
        assert set(schedule.to_dict()) == set(schedule.__struct_fields__)

    def test_bracket_access(self) -> None:
        schedule = _make_schedule()
        assert schedule["frequency"] == "daily"
        with pytest.raises(KeyError):
            schedule["nope"]

    def test_contains(self) -> None:
        schedule = _make_schedule()
        assert "enabled" in schedule
        assert "cron" not in schedule

    def test_unknown_attribute_raises(self) -> None:
        with pytest.raises(AttributeError, match="has no attribute 'cron'"):
            _make_schedule().cron

    def test_repr_names_the_toggle_fields(self) -> None:
        text = repr(_make_schedule())
        assert "BackupScheduleModel(" in text
        assert "frequency='daily'" in text
        assert "enabled=True" in text

    def test_repr_html_mentions_disabled_next_run(self) -> None:
        html = _make_schedule(enabled=False, next_scheduled_run=None)._repr_html_()
        assert "none (schedule disabled)" in html


class TestSpecHistoryExamples:
    def test_ready_row_decodes(self) -> None:
        item = _decode_history(SPEC_HISTORY_READY).data[0]
        assert item.backup_id == "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        assert item.source_index_id == "8cbf7ba6-4135-438e-a3c3-4a89a3298905"
        assert item.source_index_name == "my-index"
        assert item.name == "daily-compliance-backup-20260403T060000Z"
        assert item.status == "Ready"
        assert item.cloud == "aws"
        assert item.region == "us-east-1"
        assert item.record_count == 500000
        assert item.namespace_count == 1
        assert item.size_bytes == 104857600
        assert item.created_at == datetime(2026, 4, 3, 6, tzinfo=timezone.utc)
        assert item.scheduled_execution_at is None
        assert item.is_scheduled is False
        assert item.description is None
        assert item.tags is None

    def test_ready_row_reuses_typed_index_schema(self) -> None:
        item = _decode_history(SPEC_HISTORY_READY).data[0]
        assert item.schema is not None
        field = item.schema.fields["embedding"]
        assert isinstance(field, DenseVectorField)
        assert field.dimension == 1536
        assert field.metric == "cosine"

    def test_scheduled_row_decodes_with_execution_time(self) -> None:
        item = _decode_history(SPEC_HISTORY_SCHEDULED).data[0]
        assert item.status == "Scheduled"
        assert item.is_scheduled is True
        assert item.scheduled_execution_at == datetime(2026, 4, 4, 6, tzinfo=timezone.utc)
        assert item.record_count == 0
        assert item.size_bytes == 0

    def test_history_list_wrapper_filters_scheduled_rows(self) -> None:
        payload = {"data": SPEC_HISTORY_READY["data"] + SPEC_HISTORY_SCHEDULED["data"]}
        envelope = _decode_history(payload)
        listing = BackupScheduleHistoryList(envelope.data, pagination=envelope.pagination)
        assert len(listing) == 2
        assert [item.status for item in listing.scheduled()] == ["Scheduled"]
        assert msgspec.json.encode(listing.to_dict())

    def test_empty_history_decodes(self) -> None:
        envelope = _decode_history({"data": [], "pagination": None})
        listing = BackupScheduleHistoryList(envelope.data, pagination=envelope.pagination)
        assert len(listing) == 0
        assert listing.to_dict() == {"data": []}

    def test_paginated_history_keeps_the_token(self) -> None:
        payload = {"data": SPEC_HISTORY_READY["data"], "pagination": {"next": "dXNlcl9pZD0x"}}
        envelope = _decode_history(payload)
        listing = BackupScheduleHistoryList(envelope.data, pagination=envelope.pagination)
        assert listing.to_dict()["pagination"] == {"next": "dXNlcl9pZD0x"}

    def test_history_row_optional_fields_absent(self) -> None:
        minimal = {
            "backup_id": "bkp-1",
            "source_index_id": "idx-1",
            "source_index_name": "my-index",
            "status": "Ready",
            "cloud": "aws",
            "region": "us-east-1",
            "created_at": "2026-04-03T06:00:00+00:00",
        }
        item = _decode_history({"data": [minimal]}).data[0]
        assert item.name is None
        assert item.description is None
        assert item.schema is None
        assert item.record_count is None
        assert item.namespace_count is None
        assert item.size_bytes is None
        assert item.tags is None
        assert item.scheduled_execution_at is None

    def test_history_row_missing_required_field_raises(self) -> None:
        payload = dict(SPEC_HISTORY_READY["data"][0])
        del payload["status"]
        with pytest.raises(ResponseParsingError, match="missing required field `status`"):
            _decode_history({"data": [payload]})


class TestBackendHistoryDivergence:
    def test_backend_shaped_row_decodes(self) -> None:
        item = _decode_history({"data": [BACKEND_HISTORY_ROW]}).data[0]
        assert item.name is None
        assert item.record_count is None
        assert item.namespace_count is None
        assert item.size_bytes is None
        assert item.status == "InitializationFailed"
        assert item.scheduled_execution_at is None

    def test_backend_legacy_schema_decodes_as_legacy_field(self) -> None:
        item = _decode_history({"data": [BACKEND_HISTORY_ROW]}).data[0]
        assert item.schema is not None
        field = item.schema.fields["genre"]
        assert isinstance(field, LegacyMetadataField)
        assert field.filterable is True

    def test_legacy_schema_to_dict_drops_the_internal_tag(self) -> None:
        item = _decode_history({"data": [BACKEND_HISTORY_ROW]}).data[0]
        assert item.to_dict()["schema"] == {"fields": {"genre": {"filterable": True}}}

    def test_extra_backend_only_keys_are_ignored(self) -> None:
        item = _decode_history({"data": [BACKEND_HISTORY_ROW]}).data[0]
        assert "dimension" not in item
        with pytest.raises(AttributeError):
            item.dimension


class TestHistoryItemSurface:
    def test_to_dict_covers_every_field(self) -> None:
        item = _decode_history(SPEC_HISTORY_SCHEDULED).data[0]
        result = item.to_dict()
        assert set(result) == set(item.__struct_fields__)
        assert result["scheduled_execution_at"] == "2026-04-04T06:00:00Z"
        assert result["created_at"] == "2026-04-03T06:00:01Z"
        assert msgspec.json.encode(result)

    def test_bracket_access_and_contains(self) -> None:
        item = _decode_history(SPEC_HISTORY_READY).data[0]
        assert item["status"] == "Ready"
        assert "size_bytes" in item
        with pytest.raises(KeyError):
            item["nope"]

    def test_repr_shows_scheduled_execution_time(self) -> None:
        text = repr(_decode_history(SPEC_HISTORY_SCHEDULED).data[0])
        assert "BackupScheduleHistoryItem(" in text
        assert "scheduled_execution_at='2026-04-04T06:00:00Z'" in text

    def test_repr_html_renders(self) -> None:
        html = _decode_history(SPEC_HISTORY_READY).data[0]._repr_html_()
        assert "BackupScheduleHistoryItem" in html
        assert "my-index" in html


class TestCreateBackupScheduleRequest:
    def test_flat_kwargs_build_the_spec_request_body(self) -> None:
        request = CreateBackupScheduleRequest(
            name="daily-compliance-backup", frequency="daily", retention_days=90
        )
        assert request.to_wire() == SPEC_CREATE_REQUEST

    @pytest.mark.parametrize("frequency", BACKUP_SCHEDULE_FREQUENCIES)
    def test_every_supported_frequency_is_accepted(self, frequency: str) -> None:
        wire = CreateBackupScheduleRequest(
            name="s", frequency=frequency, retention_days=1
        ).to_wire()
        assert wire["schedule"] == {"type": SCHEDULE_TYPE_TIME_BASED, "frequency": frequency}

    def test_wire_body_is_json_encodable(self) -> None:
        request = CreateBackupScheduleRequest(name="s", frequency="monthly", retention_days=7)
        assert msgspec.json.decode(msgspec.json.encode(request.to_wire())) == request.to_wire()

    def test_invalid_frequency_lists_the_allowed_values(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            CreateBackupScheduleRequest(name="s", frequency="hourly", retention_days=90)
        message = str(excinfo.value)
        assert "'hourly'" in message
        for allowed in BACKUP_SCHEDULE_FREQUENCIES:
            assert allowed in message

    def test_cron_string_is_rejected_as_a_frequency(self) -> None:
        with pytest.raises(ValueError, match="cron expressions are not supported"):
            CreateBackupScheduleRequest(name="s", frequency="0 6 * * *", retention_days=90)

    @pytest.mark.parametrize("retention_days", [0, -1, -365])
    def test_retention_below_one_is_rejected(self, retention_days: int) -> None:
        with pytest.raises(ValueError) as excinfo:
            CreateBackupScheduleRequest(name="s", frequency="daily", retention_days=retention_days)
        message = str(excinfo.value)
        assert "must be between 1 and" in message
        assert "max_backup_retention_days" in message
        assert f"got {retention_days}." in message

    def test_upper_bound_is_left_to_the_server(self) -> None:
        request = CreateBackupScheduleRequest(name="s", frequency="daily", retention_days=100_000)
        assert request.to_wire()["retention"] == {"expire_after_days": 100_000}


class TestUpdateBackupScheduleRequest:
    def test_flat_kwargs_build_the_spec_request_body(self) -> None:
        request = UpdateBackupScheduleRequest(frequency="weekly", retention_days=30, enabled=False)
        assert request.to_wire() == SPEC_UPDATE_REQUEST

    def test_unset_fields_stay_off_the_wire(self) -> None:
        assert UpdateBackupScheduleRequest(enabled=True).to_wire() == {"enabled": True}
        assert UpdateBackupScheduleRequest(retention_days=7).to_wire() == {
            "retention": {"expire_after_days": 7}
        }
        assert UpdateBackupScheduleRequest(frequency="monthly").to_wire() == {
            "frequency": "monthly"
        }

    def test_empty_patch_encodes_to_an_empty_body(self) -> None:
        assert UpdateBackupScheduleRequest().to_wire() == {}

    def test_disable_is_distinct_from_unset(self) -> None:
        assert UpdateBackupScheduleRequest(enabled=False).to_wire() == {"enabled": False}

    def test_invalid_frequency_lists_the_allowed_values(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            UpdateBackupScheduleRequest(frequency="yearly")
        for allowed in BACKUP_SCHEDULE_FREQUENCIES:
            assert allowed in str(excinfo.value)

    def test_retention_below_one_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be between 1 and"):
            UpdateBackupScheduleRequest(retention_days=0)


class TestSyncAsyncParity:
    def test_models_reachable_from_the_backups_package(self) -> None:
        from pinecone.models import backups as backups_package

        for name in (
            "BackupScheduleModel",
            "BackupScheduleHistoryItem",
            "BackupScheduleList",
            "BackupScheduleHistoryList",
            "CreateBackupScheduleRequest",
            "UpdateBackupScheduleRequest",
        ):
            assert getattr(backups_package, name) is not None

    def test_exported_from_pinecone_models(self) -> None:
        import pinecone.models as models_package

        for name in (
            "BackupScheduleModel",
            "BackupScheduleHistoryItem",
            "BackupScheduleList",
            "BackupScheduleHistoryList",
            "CreateBackupScheduleRequest",
            "UpdateBackupScheduleRequest",
        ):
            assert name in models_package.__all__
            assert getattr(models_package, name) is not None
