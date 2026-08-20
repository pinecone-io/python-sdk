"""Property-based tests for backup schedule request/response round trips (2026-07).

Pins the loop a caller actually goes through: build a request from flat
``(frequency, expire_after_days, enabled)`` values, encode it, let a
spec-shaped server echo it back, and decode. Nothing in that loop may lose
or mangle a value.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import msgspec
from hypothesis import given
from hypothesis import strategies as st

from pinecone.models.backups.schedules import (
    BACKUP_SCHEDULE_FREQUENCIES,
    SCHEDULE_TYPE_TIME_BASED,
    BackupScheduleModel,
    CreateBackupScheduleRequest,
    UpdateBackupScheduleRequest,
)

_frequency = st.sampled_from(BACKUP_SCHEDULE_FREQUENCIES)
_retention_days = st.integers(min_value=1, max_value=3650)
_name = st.text(min_size=1, max_size=45).filter(lambda s: s.strip() == s and s.strip() != "")


def _server_echo(
    *, name: str, frequency: str, retention_days: int, enabled: bool
) -> dict[str, Any]:
    """Build the response a spec-conformant server returns for this schedule."""
    created_at = datetime(2026, 4, 2, 18, 22, 56, 712605, tzinfo=timezone.utc)
    next_run = created_at + timedelta(days=1) if enabled else None
    return {
        "schedule_id": "e88f7273-42aa-47e9-af73-593827136867",
        "name": name,
        "index_id": "8cbf7ba6-4135-438e-a3c3-4a89a3298905",
        "project_id": "71ce31ea-75f7-45d6-a147-ef67f661a1b0",
        "schedule_type": SCHEDULE_TYPE_TIME_BASED,
        "frequency": frequency,
        "retention_expire_after_days": retention_days,
        "enabled": enabled,
        "next_scheduled_run": None if next_run is None else next_run.isoformat(),
        "created_at": created_at.isoformat(),
    }


@given(name=_name, frequency=_frequency, retention_days=_retention_days, enabled=st.booleans())
def test_request_encode_response_decode_loses_nothing(
    name: str, frequency: str, retention_days: int, enabled: bool
) -> None:
    request = CreateBackupScheduleRequest(
        name=name, frequency=frequency, retention_days=retention_days
    )
    body = msgspec.json.decode(msgspec.json.encode(request.to_wire()))
    assert body["name"] == name
    assert body["schedule"] == {"type": SCHEDULE_TYPE_TIME_BASED, "frequency": frequency}
    assert body["retention"] == {"expire_after_days": retention_days}

    payload = _server_echo(
        name=body["name"],
        frequency=body["schedule"]["frequency"],
        retention_days=body["retention"]["expire_after_days"],
        enabled=enabled,
    )
    schedule = msgspec.json.decode(msgspec.json.encode(payload), type=BackupScheduleModel)

    assert schedule.name == name
    assert schedule.frequency == frequency
    assert schedule.retention_expire_after_days == retention_days
    assert schedule.enabled is enabled
    assert (schedule.next_scheduled_run is None) is (not enabled)


@given(name=_name, frequency=_frequency, retention_days=_retention_days, enabled=st.booleans())
def test_decode_to_dict_re_decode_is_stable(
    name: str, frequency: str, retention_days: int, enabled: bool
) -> None:
    payload = _server_echo(
        name=name, frequency=frequency, retention_days=retention_days, enabled=enabled
    )
    schedule = msgspec.json.decode(msgspec.json.encode(payload), type=BackupScheduleModel)
    round_tripped = msgspec.json.decode(
        msgspec.json.encode(schedule.to_dict()), type=BackupScheduleModel
    )
    assert round_tripped == schedule


@given(
    frequency=st.one_of(st.none(), _frequency),
    retention_days=st.one_of(st.none(), _retention_days),
    enabled=st.one_of(st.none(), st.booleans()),
)
def test_update_request_emits_exactly_the_fields_that_were_set(
    frequency: str | None, retention_days: int | None, enabled: bool | None
) -> None:
    request = UpdateBackupScheduleRequest(
        frequency=frequency, retention_days=retention_days, enabled=enabled
    )
    body = msgspec.json.decode(msgspec.json.encode(request.to_wire()))

    expected_keys = {
        key
        for key, value in (
            ("frequency", frequency),
            ("retention", retention_days),
            ("enabled", enabled),
        )
        if value is not None
    }
    assert set(body) == expected_keys

    if frequency is not None:
        assert body["frequency"] == frequency
    if retention_days is not None:
        assert body["retention"] == {"expire_after_days": retention_days}
    if enabled is not None:
        assert body["enabled"] is enabled
