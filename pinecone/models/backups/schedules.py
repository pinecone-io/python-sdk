"""Backup schedule models (2026-07 API).

A backup schedule attaches an automatic, time-based backup cadence to a
single index, at one of three cadences (``daily``, ``weekly``,
``monthly``). **There is no cron support**: the run time is chosen
server-side and surfaced through
:attr:`BackupScheduleModel.next_scheduled_run`; there is no way to
express an arbitrary cron expression or a caller-chosen timezone.

The only schedule type the SDK sends is ``"time-based"``. The request
models here therefore take **flat** keyword arguments and fill the type in
themselves (see :meth:`CreateBackupScheduleRequest.to_wire`): callers write
``frequency="daily", retention_days=90`` instead of assembling
``{"schedule": {"type": ..., "frequency": ...}, "retention": {...}}``.

Timestamps on these models are ``datetime`` objects rather than the
``str`` used by :class:`~pinecone.models.backups.model.BackupModel`.
Schedule timestamps exist to be *compared* -- "when does this next run",
"is this snapshot past its retention window" -- and that is arithmetic, not
display. :meth:`BackupScheduleModel.to_dict` renders them back to RFC 3339
strings so dict output stays JSON-serialisable.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import msgspec
from msgspec import Struct

from pinecone.models._display import render_table
from pinecone.models.indexes.schema import IndexSchema, _strip_untyped_tags

__all__ = [
    "BackupScheduleHistoryItem",
    "BackupScheduleModel",
    "CreateBackupScheduleRequest",
    "UpdateBackupScheduleRequest",
]

#: The only schedule type the SDK sends. Filled in by the request models.
SCHEDULE_TYPE_TIME_BASED = "time-based"

#: Cadences accepted by ``frequency``. There is no cron alternative.
BACKUP_SCHEDULE_FREQUENCIES: tuple[str, ...] = ("daily", "weekly", "monthly")


def _validate_frequency(frequency: str) -> None:
    """Reject a cadence the API does not accept, naming the ones it does.

    Raises:
        ValueError: If *frequency* is not one of ``daily``, ``weekly``, or
            ``monthly``. The message lists all three, because the only
            alternative the API offers is another one of them -- there is
            no cron expression to fall back to.
    """
    if frequency not in BACKUP_SCHEDULE_FREQUENCIES:
        allowed = " | ".join(BACKUP_SCHEDULE_FREQUENCIES)
        raise ValueError(
            f"Invalid frequency {frequency!r}: expected one of {allowed}. "
            "Backup schedules are time-based only; cron expressions are not supported."
        )


def _validate_retention_days(retention_days: int) -> None:
    """Reject a retention window the API will certainly reject.

    Only the lower bound (1) is checked here, because it is the one the SDK
    knows. The upper bound is a per-project setting, so a too-large value is
    rejected server-side rather than here.

    Raises:
        ValueError: If *retention_days* is less than 1.
    """
    if retention_days < 1:
        raise ValueError(
            "retention_days must be between 1 and your project's "
            f"max_backup_retention_days, got {retention_days}."
        )


def _rfc3339(value: datetime | None) -> str | None:
    if value is None:
        return None
    rendered: str = msgspec.to_builtins(value)
    return rendered


class BackupScheduleModel(Struct, kw_only=True):
    """One recurring backup cadence attached to an index.

    Returned by every :class:`~pinecone.client.backup_schedules.BackupSchedules`
    method that reads or writes a schedule; not constructed directly.

    Attributes:
        schedule_id: Unique identifier for the schedule. Used as the path
            parameter for describe / update / delete / history calls.
        name: User-defined name for the schedule. Backups it produces are
            named ``"{name}-{run timestamp}"``.
        index_id: Identifier of the index this schedule backs up. This is
            the index *id*, not its name -- schedules are created against a
            name but reported against the id, so a deleted-and-recreated
            index does not inherit the old schedule.
        project_id: Project containing the schedule, always the same
            project as the source index.
        schedule_type: Schedule category. ``"time-based"`` for any schedule
            created through this SDK, which always sends that value; the
            server does not constrain the field, so a schedule created by
            another client can report something else.
        frequency: Cadence, one of ``"daily"``, ``"weekly"``, ``"monthly"``.
        retention_expire_after_days: Days each backup produced by this
            schedule is retained. (The create/update request models spell
            the same value ``retention_days``, mirroring the request body's
            ``retention.expire_after_days``.)
        enabled: Whether the schedule is active. A disabled schedule does
            not run and is not deleted.
        next_scheduled_run: When the next backup is planned, or ``None``.
            ``None`` **iff** ``enabled`` is ``False``: disabling clears the
            pending run, and re-enabling recomputes it from the moment of
            the update, so a disable/re-enable cycle shifts the cadence
            rather than resuming the old slot.
        created_at: When the schedule was created.

    Note:
        Only one *enabled* schedule may exist per index. Creating a second
        one raises :exc:`ConflictError`, telling you to disable or delete the
        first; re-enabling a disabled schedule while another is enabled fails
        the same way.
    """

    schedule_id: str
    name: str
    index_id: str
    project_id: str
    schedule_type: str
    frequency: str
    retention_expire_after_days: int
    enabled: bool
    created_at: datetime
    next_scheduled_run: datetime | None = None

    def __getattr__(self, name: str) -> Any:
        """Raise AttributeError for attributes this model does not define."""
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def __getitem__(self, key: str) -> Any:
        """Support bracket access (e.g. ``schedule['schedule_id']``)."""
        if key not in self.__struct_fields__:
            raise KeyError(key)
        return getattr(self, key)

    def __contains__(self, key: object) -> bool:
        """Support the ``in`` operator (e.g. ``'enabled' in schedule``)."""
        return key in self.__struct_fields__

    def to_dict(self) -> dict[str, Any]:
        """Return a dict representation of this schedule.

        Returns:
            Dictionary with all fields, including ``next_scheduled_run``
            when it is ``None``. Timestamps are rendered back to RFC 3339
            strings (normalised to UTC ``Z`` form), so the result is
            JSON-serialisable.

        Examples:
            >>> from datetime import datetime, timezone
            >>> from pinecone.models.backups.schedules import BackupScheduleModel
            >>> schedule = BackupScheduleModel(
            ...     schedule_id="sched-1",
            ...     name="daily-compliance-backup",
            ...     index_id="idx-1",
            ...     project_id="proj-1",
            ...     schedule_type="time-based",
            ...     frequency="daily",
            ...     retention_expire_after_days=90,
            ...     enabled=False,
            ...     created_at=datetime(2026, 4, 2, 18, 22, 56, tzinfo=timezone.utc),
            ... )
            >>> schedule.to_dict()["created_at"]
            '2026-04-02T18:22:56Z'
            >>> schedule.to_dict()["next_scheduled_run"] is None
            True
        """
        return {
            "schedule_id": self.schedule_id,
            "name": self.name,
            "index_id": self.index_id,
            "project_id": self.project_id,
            "schedule_type": self.schedule_type,
            "frequency": self.frequency,
            "retention_expire_after_days": self.retention_expire_after_days,
            "enabled": self.enabled,
            "next_scheduled_run": _rfc3339(self.next_scheduled_run),
            "created_at": _rfc3339(self.created_at),
        }

    def __repr__(self) -> str:
        return (
            f"BackupScheduleModel(schedule_id={self.schedule_id!r}, name={self.name!r}, "
            f"frequency={self.frequency!r}, enabled={self.enabled!r}, "
            f"retention_expire_after_days={self.retention_expire_after_days!r}, "
            f"next_scheduled_run={_rfc3339(self.next_scheduled_run)!r})"
        )

    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        """Pretty-printer support for IPython."""
        if cycle:
            p.text("BackupScheduleModel(...)")
            return

        p.text("BackupScheduleModel(")
        with p.group(2, "", ")"):
            p.breakable()
            p.text(f"schedule_id={self.schedule_id!r},")
            p.breakable()
            p.text(f"name={self.name!r},")
            p.breakable()
            p.text(f"index_id={self.index_id!r},")
            p.breakable()
            p.text(f"project_id={self.project_id!r},")
            p.breakable()
            p.text(f"schedule_type={self.schedule_type!r},")
            p.breakable()
            p.text(f"frequency={self.frequency!r},")
            p.breakable()
            p.text(f"retention_expire_after_days={self.retention_expire_after_days!r},")
            p.breakable()
            p.text(f"enabled={self.enabled!r},")
            p.breakable()
            p.text(f"next_scheduled_run={_rfc3339(self.next_scheduled_run)!r},")
            p.breakable()
            p.text(f"created_at={_rfc3339(self.created_at)!r}")

    def _repr_html_(self) -> str:
        """Jupyter notebook HTML representation."""
        rows: list[tuple[str, str | int]] = [
            ("Schedule ID:", self.schedule_id),
            ("Name:", self.name),
            ("Index ID:", self.index_id),
            ("Project ID:", self.project_id),
            ("Type:", self.schedule_type),
            ("Frequency:", self.frequency),
            ("Retention (days):", self.retention_expire_after_days),
            ("Enabled:", str(self.enabled)),
            (
                "Next run:",
                _rfc3339(self.next_scheduled_run) or "none (schedule disabled)",
            ),
            ("Created:", _rfc3339(self.created_at) or "unknown"),
        ]
        return render_table("BackupScheduleModel", rows)


class BackupScheduleHistoryItem(Struct, kw_only=True):
    """One backup produced, or planned, by a schedule.

    Returned by :meth:`~pinecone.client.backup_schedules.BackupSchedules.history`
    and its iterator twin; not constructed directly. History rows describe
    backup *snapshots*, not the schedule itself, and a row appears as soon as
    a run is planned, so the list mixes runs that have not happened yet with
    ones that have.

    Attributes:
        backup_id: Unique identifier for the backup snapshot.
        source_index_id: Identifier of the index that was backed up.
        source_index_name: Name of the index that was backed up.
        status: Lifecycle status of the snapshot -- ``"Scheduled"``
            (planned, not yet started), ``"Initializing"``, ``"Ready"``, or
            ``"InitializationFailed"``. Left as a plain ``str`` so a value
            the SDK has not seen before still decodes.
        cloud: Cloud provider where the snapshot is stored.
        region: Cloud region where the snapshot is stored.
        created_at: When the backup *record* was created -- which for a
            ``Scheduled`` row is when the run was planned, not when data
            was captured.
        scheduled_execution_at: When the run is planned to happen. Present
            when ``status`` is ``"Scheduled"``; ``None`` once the run has
            started, and ``None`` on servers that do not report it.
        name: Name of the snapshot, generated as
            ``"{schedule name}-{run timestamp}"``.
        description: Description of the snapshot, or ``None``.
        schema: Schema captured from the source index, or ``None`` when the
            server reports none. Metadata-only schemas from older indexes
            decode to
            :class:`~pinecone.models.indexes.schema.LegacyMetadataField`
            entries.
        record_count: Records in the snapshot. ``0`` for a ``Scheduled``
            row -- nothing has been captured yet.
        namespace_count: Namespaces in the snapshot.
        size_bytes: Approximate stored size of the snapshot, in bytes.
        tags: Tags carried over from the source index, or ``None`` (the API
            sends ``null`` rather than ``{}`` when there are none).

    Note:
        ``name``, ``record_count``, ``namespace_count`` and ``size_bytes``
        can each be absent from a history row even though the API documents
        them as required, so they are typed as optional here and can come
        back ``None``. Guard on them rather than assuming a value.
    """

    backup_id: str
    source_index_id: str
    source_index_name: str
    status: str
    cloud: str
    region: str
    created_at: datetime
    scheduled_execution_at: datetime | None = None
    name: str | None = None
    description: str | None = None
    schema: IndexSchema | None = None
    record_count: int | None = None
    namespace_count: int | None = None
    size_bytes: int | None = None
    tags: dict[str, Any] | None = None

    @property
    def is_scheduled(self) -> bool:
        """Whether this row is a planned run that has not started yet."""
        return self.status == "Scheduled"

    def __getattr__(self, name: str) -> Any:
        """Raise AttributeError for attributes this model does not define."""
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def __getitem__(self, key: str) -> Any:
        """Support bracket access (e.g. ``item['backup_id']``)."""
        if key not in self.__struct_fields__:
            raise KeyError(key)
        return getattr(self, key)

    def __contains__(self, key: object) -> bool:
        """Support the ``in`` operator (e.g. ``'status' in item``)."""
        return key in self.__struct_fields__

    def to_dict(self) -> dict[str, Any]:
        """Return a dict representation of this history row.

        Returns:
            Dictionary with all fields, including optional ones that are
            ``None``. Timestamps are rendered back to RFC 3339 strings
            (normalised to UTC ``Z`` form), ``schema`` becomes a plain dict
            with the SDK's internal untyped-field tag stripped, and the
            result is JSON-serialisable.
        """
        return {
            "backup_id": self.backup_id,
            "source_index_id": self.source_index_id,
            "source_index_name": self.source_index_name,
            "status": self.status,
            "cloud": self.cloud,
            "region": self.region,
            "created_at": _rfc3339(self.created_at),
            "scheduled_execution_at": _rfc3339(self.scheduled_execution_at),
            "name": self.name,
            "description": self.description,
            "schema": None
            if self.schema is None
            else _strip_untyped_tags(msgspec.to_builtins(self.schema)),
            "record_count": self.record_count,
            "namespace_count": self.namespace_count,
            "size_bytes": self.size_bytes,
            "tags": None if self.tags is None else dict(self.tags),
        }

    def __repr__(self) -> str:
        parts = [
            f"backup_id={self.backup_id!r}",
            f"status={self.status!r}",
            f"source_index_name={self.source_index_name!r}",
            f"created_at={_rfc3339(self.created_at)!r}",
        ]
        if self.name is not None:
            parts.append(f"name={self.name!r}")
        if self.scheduled_execution_at is not None:
            parts.append(f"scheduled_execution_at={_rfc3339(self.scheduled_execution_at)!r}")
        return f"BackupScheduleHistoryItem({', '.join(parts)})"

    def _repr_html_(self) -> str:
        """Jupyter notebook HTML representation."""
        rows: list[tuple[str, str | int]] = [
            ("Backup ID:", self.backup_id),
            ("Source Index:", self.source_index_name),
            ("Source Index ID:", self.source_index_id),
            ("Status:", self.status),
            ("Cloud:", self.cloud),
            ("Region:", self.region),
            ("Created:", _rfc3339(self.created_at) or "unknown"),
        ]
        if self.scheduled_execution_at is not None:
            rows.append(("Scheduled for:", _rfc3339(self.scheduled_execution_at) or ""))
        if self.name is not None:
            rows.append(("Name:", self.name))
        if self.description is not None:
            rows.append(("Description:", self.description))
        if self.schema is not None:
            rows.append(("Schema fields:", len(self.schema.fields)))
        if self.record_count is not None:
            rows.append(("Records:", self.record_count))
        if self.namespace_count is not None:
            rows.append(("Namespaces:", self.namespace_count))
        if self.size_bytes is not None:
            rows.append(("Size:", f"{self.size_bytes} bytes"))
        if self.tags:
            rows.append(("Tags:", ", ".join(f"{k}={v}" for k, v in self.tags.items())))
        return render_table("BackupScheduleHistoryItem", rows)


class CreateBackupScheduleRequest(Struct, kw_only=True):
    """Request model for creating a backup schedule.

    Takes flat keyword arguments and builds the nested request body in
    :meth:`to_wire`, filling in ``schedule.type`` rather than making every
    caller repeat the one value the SDK sends.

    Attributes:
        name: Name for the schedule (required). Produced backups are named
            ``"{name}-{run timestamp}"``.
        frequency: Cadence (required), one of ``"daily"``, ``"weekly"``,
            ``"monthly"``. Validated on construction.
        retention_days: Days to retain each backup this schedule produces
            (required). Must be at least 1, which is checked here; the
            maximum is a per-project setting enforced server-side.
            Serialised as ``retention.expire_after_days``.

    Raises:
        ValueError: If *frequency* is not a supported cadence, or
            *retention_days* is less than 1.

    Examples:
        >>> from pinecone.models.backups.schedules import CreateBackupScheduleRequest
        >>> request = CreateBackupScheduleRequest(
        ...     name="daily-compliance-backup", frequency="daily", retention_days=90
        ... )
        >>> request.to_wire() == {
        ...     "name": "daily-compliance-backup",
        ...     "schedule": {"type": "time-based", "frequency": "daily"},
        ...     "retention": {"expire_after_days": 90},
        ... }
        True
    """

    name: str
    frequency: str
    retention_days: int

    def __post_init__(self) -> None:
        _validate_frequency(self.frequency)
        _validate_retention_days(self.retention_days)

    def to_wire(self) -> dict[str, Any]:
        """Return the nested JSON body the create-schedule endpoint expects.

        This is the encoding entry point for this model: the flat fields do
        not match the wire shape, so encode ``to_wire()`` rather than the
        struct itself.
        """
        return {
            "name": self.name,
            "schedule": {"type": SCHEDULE_TYPE_TIME_BASED, "frequency": self.frequency},
            "retention": {"expire_after_days": self.retention_days},
        }


class UpdateBackupScheduleRequest(Struct, kw_only=True):
    """Request model for updating an existing backup schedule.

    Every field is optional; omitted fields are left unchanged. Like
    :class:`CreateBackupScheduleRequest`, this takes flat keyword arguments
    and builds the nested body in :meth:`to_wire`, which emits only the
    fields you set. A request with nothing set encodes to ``{}`` and is a
    no-op server-side.

    The schedule's ``name`` cannot be changed, and neither can the index it
    is attached to -- the API exposes no field for either.

    Attributes:
        frequency: New cadence, one of ``"daily"``, ``"weekly"``,
            ``"monthly"``, or ``None`` to leave it unchanged.
        retention_days: New retention window in days, or ``None`` to leave
            it unchanged. Must be at least 1; serialised as
            ``retention.expire_after_days``. Changing it also re-times the
            pending deletions of backups this schedule already produced.
        enabled: ``False`` to disable the schedule (clearing its
            ``next_scheduled_run``), ``True`` to re-enable it, or ``None``
            to leave it unchanged. Re-enabling **enqueues a new backup**
            and recomputes the next run from now, so it is not a free
            toggle; it also raises :exc:`ConflictError` if another schedule
            on the same index is already enabled.

    Raises:
        ValueError: If *frequency* is set to an unsupported cadence, or
            *retention_days* is set to less than 1.

    Examples:
        >>> from pinecone.models.backups.schedules import UpdateBackupScheduleRequest
        >>> UpdateBackupScheduleRequest(enabled=False).to_wire()
        {'enabled': False}
        >>> UpdateBackupScheduleRequest(frequency="weekly", retention_days=30).to_wire() == {
        ...     "frequency": "weekly",
        ...     "retention": {"expire_after_days": 30},
        ... }
        True
    """

    frequency: str | None = None
    retention_days: int | None = None
    enabled: bool | None = None

    def __post_init__(self) -> None:
        if self.frequency is not None:
            _validate_frequency(self.frequency)
        if self.retention_days is not None:
            _validate_retention_days(self.retention_days)

    def to_wire(self) -> dict[str, Any]:
        """Return the sparse nested JSON body the update-schedule endpoint expects.

        Only the fields you set appear, so unset fields are left unchanged
        server-side rather than being reset to a default.
        """
        body: dict[str, Any] = {}
        if self.frequency is not None:
            body["frequency"] = self.frequency
        if self.retention_days is not None:
            body["retention"] = {"expire_after_days": self.retention_days}
        if self.enabled is not None:
            body["enabled"] = self.enabled
        return body
