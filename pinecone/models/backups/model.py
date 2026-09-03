"""Backup and restore models (2026-07 API)."""

from __future__ import annotations

from typing import Any

import msgspec
from msgspec import Struct

from pinecone.models._display import render_table
from pinecone.models._mixin import StructDictMixin
from pinecone.models.indexes.schema import (
    DenseVectorField,
    IndexSchema,
    _strip_untyped_tags,
)

__all__ = [
    "BackupModel",
    "CreateIndexFromBackupRequest",
    "CreateIndexFromBackupResponse",
    "RestoreJobModel",
]

_REMOVED_FIELD_HINTS: dict[str, str] = {
    "dimension": (
        "read it from schema.fields instead, e.g. "
        "backup.schema.fields['<field-name>'].dimension on the DenseVectorField, "
        "or use the backup.dense_dimension convenience property"
    ),
    "metric": (
        "read it from schema.fields instead, e.g. "
        "backup.schema.fields['<field-name>'].metric on the vector field"
    ),
}


class BackupModel(Struct, kw_only=True):
    """One stored, point-in-time snapshot of an index.

    Returned by :meth:`~pinecone.client.backups.Backups.create`,
    :meth:`~pinecone.client.backups.Backups.describe` and the backup
    listings; not constructed directly.

    Attributes:
        backup_id: Unique identifier for the backup.
        source_index_name: Name of the index that was backed up.
        source_index_id: Unique identifier of the source index.
        status: Current status of the backup — ``"Initializing"``,
            ``"Ready"``, or ``"Failed"``.
        cloud: Cloud provider where the backup is stored.
        region: Region where the backup is stored.
        source_index_deleted_at: Timestamp at which the source index was
            deleted, or ``None`` while the source index is still active. An
            index-scoped listing only surfaces these rows when it is passed
            ``include_deleted=True``.
        name: User-provided name for the backup.
        description: User-provided description for the backup.
        schema: Schema captured from the source index, or ``None`` when the
            server returns no schema (e.g. schedule-produced backups of an
            index that declared none). Legacy metadata-only schemas decode
            to :class:`~pinecone.models.indexes.schema.LegacyMetadataField`
            entries.
        record_count: Number of records in the backup.
        namespace_count: Number of namespaces in the backup.
        size_bytes: Size of the backup in bytes.
        tags: User-defined key-value tags, or ``None`` when the source index
            had none (the API returns ``"tags": null`` rather than ``{}``).
        created_at: Timestamp when the backup was created.
    """

    backup_id: str
    source_index_name: str
    source_index_id: str
    status: str
    cloud: str
    region: str
    source_index_deleted_at: str | None = None
    name: str | None = None
    description: str | None = None
    schema: IndexSchema | None = None
    record_count: int | None = None
    namespace_count: int | None = None
    size_bytes: int | None = None
    tags: dict[str, Any] | None = None
    created_at: str | None = None

    @property
    def dense_dimension(self) -> int | None:
        """Dimension of the backup's single dense vector field, if there is one.

        Returns ``None`` when the schema is absent, declares no
        ``dense_vector`` field, or declares more than one — in which case
        read the dimension off the field you want via
        :attr:`schema`\\ ``.fields['<field-name>'].dimension``.
        """
        if self.schema is None:
            return None
        dims = [f.dimension for f in self.schema.fields.values() if isinstance(f, DenseVectorField)]
        return dims[0] if len(dims) == 1 else None

    def __getattr__(self, name: str) -> Any:
        if name in _REMOVED_FIELD_HINTS:
            raise AttributeError(
                f"BackupModel.{name} was removed in the 2026-07 Pinecone API: "
                f"{_REMOVED_FIELD_HINTS[name]}. "
                "See https://sdk.pinecone.io/python/migration/v10-migration.html."
            )
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def __getitem__(self, key: str) -> Any:
        """Support bracket access (e.g. backup['backup_id'])."""
        if key not in self.__struct_fields__:
            raise KeyError(key)
        return getattr(self, key)

    def __contains__(self, key: object) -> bool:
        """Support ``in`` operator (e.g. ``'backup_id' in backup``)."""
        return key in self.__struct_fields__

    def __dir__(self) -> list[str]:
        attrs = set(super().__dir__())
        public = {name for name in attrs if not name.startswith("_")}
        return sorted(public)

    def to_dict(self) -> dict[str, Any]:
        """Return a dict representation of this backup model.

        Returns:
            Dictionary with all fields, including optional ones that are
            ``None`` (e.g. ``name``, ``description``, ``record_count``,
            ``source_index_deleted_at``). ``schema`` becomes a plain dict;
            legacy untyped schema fields are emitted without a ``type`` key,
            matching the wire format.

        Examples:
            >>> from pinecone.models.backups.model import BackupModel
            >>> backup = BackupModel(
            ...     backup_id="bkp-1",
            ...     source_index_name="my-index",
            ...     source_index_id="idx-abc",
            ...     status="Ready",
            ...     cloud="aws",
            ...     region="us-east-1",
            ...     name="weekly-backup",
            ... )
            >>> d = backup.to_dict()
            >>> d["backup_id"]
            'bkp-1'
            >>> d["name"]
            'weekly-backup'
            >>> d["description"] is None
            True
            >>> d["source_index_deleted_at"] is None
            True
        """
        return {f: _to_builtins_stripped(getattr(self, f)) for f in self.__struct_fields__}

    def __repr__(self) -> str:
        parts = [
            f"backup_id={self.backup_id!r}",
            f"status={self.status!r}",
            f"source_index_name={self.source_index_name!r}",
            f"created_at={self.created_at!r}",
        ]
        if self.name is not None:
            parts.append(f"name={self.name!r}")
        if self.source_index_deleted_at is not None:
            parts.append(f"source_index_deleted_at={self.source_index_deleted_at!r}")
        if self.schema is not None:
            parts.append(f"schema_fields={len(self.schema.fields)}")
        return f"BackupModel({', '.join(parts)})"

    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        """Pretty-printer support for IPython."""
        if cycle:
            p.text("BackupModel(...)")
            return

        p.text("BackupModel(")
        with p.group(2, "", ")"):
            p.breakable()
            p.text(f"backup_id={self.backup_id!r},")
            p.breakable()
            p.text(f"source_index_name={self.source_index_name!r},")
            p.breakable()
            p.text(f"source_index_id={self.source_index_id!r},")
            p.breakable()
            p.text(f"status={self.status!r},")
            p.breakable()
            p.text(f"cloud={self.cloud!r},")
            p.breakable()
            p.text(f"region={self.region!r},")
            p.breakable()
            p.text(f"created_at={self.created_at!r}")

            if self.source_index_deleted_at is not None:
                p.breakable()
                p.text(f"source_index_deleted_at={self.source_index_deleted_at!r}")
            if self.name is not None:
                p.breakable()
                p.text(f"name={self.name!r}")
            if self.description is not None:
                p.breakable()
                p.text(f"description={self.description!r}")
            if self.record_count is not None:
                p.breakable()
                p.text(f"record_count={self.record_count}")
            if self.namespace_count is not None:
                p.breakable()
                p.text(f"namespace_count={self.namespace_count}")
            if self.size_bytes is not None:
                p.breakable()
                p.text(f"size_bytes={self.size_bytes}")
            if self.tags:
                p.breakable()
                p.text(f"tags={self.tags!r}")
            if self.schema is not None:
                p.breakable()
                p.text(f"schema=IndexSchema(fields={len(self.schema.fields)} fields)")

    def _repr_html_(self) -> str:
        """Jupyter notebook HTML representation."""
        rows: list[tuple[str, str | int]] = [
            ("Backup ID:", self.backup_id),
            ("Source Index:", self.source_index_name),
            ("Source Index ID:", self.source_index_id),
            ("Status:", self.status),
            ("Cloud:", self.cloud),
            ("Region:", self.region),
            ("Created:", self.created_at if self.created_at is not None else "unknown"),
        ]

        if self.source_index_deleted_at is not None:
            rows.append(("Source Index Deleted:", self.source_index_deleted_at))
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
            tags_str = ", ".join(f"{k}={v}" for k, v in self.tags.items())
            rows.append(("Tags:", tags_str))

        return render_table("BackupModel", rows)


def _to_builtins_stripped(value: Any) -> Any:
    if isinstance(value, Struct):
        return _strip_untyped_tags(msgspec.to_builtins(value))
    if isinstance(value, dict):
        return dict(value)
    return value


class RestoreJobModel(Struct, kw_only=True):
    """One attempt at turning a backup back into an index.

    Returned by :meth:`~pinecone.client.restore_jobs.RestoreJobs.describe` and
    :meth:`~pinecone.client.restore_jobs.RestoreJobs.list`; not constructed
    directly.

    Attributes:
        restore_job_id: Unique identifier for the restore job.
        backup_id: Identifier of the backup being restored.
        target_index_name: Name of the index being restored to.
        target_index_id: Unique identifier of the target index.
        status: ``"Pending"``, ``"Completed"``, ``"Failed"``, or
            ``"Cancelled"``. There is no in-progress value: a restore that is
            actively running reports ``"Pending"``.
        created_at: Timestamp when the restore job was created, or ``None`` if the
            backend has not yet assigned a creation timestamp.
        completed_at: Timestamp when the restore job completed, or ``None``
            until then.
        percent_complete: ``100`` once ``status`` is ``"Completed"``, and
            ``None`` at every other point — it reports completion rather than
            progress, so it cannot drive a progress bar.
    """

    restore_job_id: str
    backup_id: str
    target_index_name: str
    target_index_id: str
    status: str
    created_at: str | None = None
    completed_at: str | None = None
    percent_complete: float | None = None

    def __getattr__(self, name: str) -> Any:
        """Raise AttributeError for unknown attributes (legacy dict-style delegation)."""
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def __getitem__(self, key: str) -> Any:
        """Support bracket access (e.g. job['restore_job_id'])."""
        if key not in self.__struct_fields__:
            raise KeyError(key)
        return getattr(self, key)

    def __contains__(self, key: object) -> bool:
        """Support ``in`` operator (e.g. ``'restore_job_id' in job``)."""
        return key in self.__struct_fields__

    def to_dict(self) -> dict[str, Any]:
        """Return a dict representation of this restore job model.

        Returns:
            Dictionary with all fields, including optional ones that are ``None``
            (``completed_at`` and ``percent_complete``). Values are not
            recursively converted.

        Examples:
            >>> from pinecone.models.backups.model import RestoreJobModel
            >>> job = RestoreJobModel(
            ...     restore_job_id="rj-1",
            ...     backup_id="bkp-1",
            ...     target_index_name="my-index",
            ...     target_index_id="idx-abc",
            ...     status="Pending",
            ...     created_at="2024-01-01T00:00:00Z",
            ... )
            >>> d = job.to_dict()
            >>> d["restore_job_id"]
            'rj-1'
            >>> d["completed_at"] is None
            True
        """
        return {f: getattr(self, f) for f in self.__struct_fields__}


class CreateIndexFromBackupRequest(Struct, kw_only=True, omit_defaults=True):
    """Request model for creating an index from a backup.

    Optionals you leave unset stay off the wire, so a request built with only
    ``name`` serialises to ``{"name": ...}`` and the server applies its own
    defaults: on-demand read capacity, deletion protection disabled, and the
    backup's own tags.

    Attributes:
        name: Name for the restored index (required). Subject to the same
            naming rules as a new index, which the server rather than the
            client enforces on this path.
        tags: Optional key-value tags for the restored index. When omitted,
            the server copies the backup's tags.
        deletion_protection: Optional deletion protection setting
            (``"enabled"`` or ``"disabled"``).
        read_capacity: Optional read capacity configuration, letting the
            restore land directly on dedicated read nodes instead of
            defaulting to on-demand capacity.
    """

    name: str
    tags: dict[str, str] | None = None
    deletion_protection: str | None = None
    read_capacity: dict[str, Any] | None = None


class CreateIndexFromBackupResponse(StructDictMixin, Struct, kw_only=True):
    """Response model for creating an index from a backup.

    Attributes:
        restore_job_id: Identifier of the restore job created.
        index_id: Identifier of the new index being created.
    """

    restore_job_id: str
    index_id: str
