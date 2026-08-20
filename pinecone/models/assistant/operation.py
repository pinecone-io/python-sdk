"""Assistant operation response model."""

from __future__ import annotations

from typing import Any

from msgspec import Struct

from pinecone.models._display import HtmlBuilder, safe_display, truncate_text
from pinecone.models.assistant._mixin import StructDictMixin

__all__ = ["OperationModel"]


class OperationModel(
    StructDictMixin,
    Struct,
    kw_only=True,
    rename={"operation_id": "id", "created_at": "created_on", "error": "error_message"},
):
    """Response model for a long-running assistant operation.

    Returned by the file endpoints that start an operation (``POST /files/{assistant_name}``,
    ``PUT /files/{assistant_name}/{file_id}``, ``DELETE /files/{assistant_name}/{file_id}``)
    and by the operations endpoints (``GET /operations/{assistant_name}/{operation_id}``,
    ``GET /operations/{assistant_name}``).

    The API uses ``id``, ``created_on`` and ``error_message``; the rename mapping presents
    them as ``operation_id``, ``created_at`` and ``error`` in Python for clarity. Every
    other attribute carries its wire name.

    Every field except ``operation_id`` and ``status`` is optional so that the smaller
    body shipped by the ``2026-04`` upsert path still decodes. The server omits
    ``completed_on``, ``error_message`` and ``ingestion_units`` while they do not apply;
    a spec-conformant server may instead send them as ``null``. Both decode to ``None``.

    Attributes:
        operation_id: Unique identifier for the operation (JSON field: ``id``).
        status: Current status of the operation: ``"Processing"`` while it is in
            progress, ``"Completed"`` when it finished successfully, ``"Failed"``
            when it did not (see :attr:`error`).
        operation_type: The kind of action this operation represents —
            ``"upload_file"``, ``"upsert_file"``, ``"update_file_metadata"`` or
            ``"delete_file"`` — or ``None`` when the server did not report one.
        file_id: Identifier of the file being operated on, or ``None``.
        created_at: ISO 8601 timestamp when the operation was created, or ``None``
            (JSON field: ``created_on``).
        completed_on: ISO 8601 timestamp when the operation completed or failed, or
            ``None`` while ``status`` is ``"Processing"``.
        percent_complete: Progress of the operation as a percentage from 0 to 100,
            or ``None`` when the server did not report progress.
        error: Error message if the operation failed, or ``None``
            (JSON field: ``error_message``). Goes stale across a retry: the
            backend writes this column with ``COALESCE``, so it is never
            cleared once set — a retried operation that is back to
            ``"Processing"``, or that eventually succeeds, still carries the
            earlier attempt's text. Read it only when ``status`` is
            ``"Failed"``.
        ingestion_units: Ingestion units consumed by this operation, reported once a
            file ingestion operation has completed, or ``None``.
    """

    operation_id: str
    status: str
    operation_type: str | None = None
    file_id: str | None = None
    created_at: str | None = None
    completed_on: str | None = None
    percent_complete: int | None = None
    error: str | None = None
    ingestion_units: float | None = None

    @safe_display
    def __repr__(self) -> str:
        parts = [f"operation_id={self.operation_id!r}", f"status={self.status!r}"]
        if self.operation_type is not None:
            parts.append(f"operation_type={self.operation_type!r}")
        if self.file_id is not None:
            parts.append(f"file_id={self.file_id!r}")
        if self.percent_complete is not None:
            parts.append(f"percent_complete={self.percent_complete!r}")
        if self.error is not None:
            parts.append(f"error={truncate_text(self.error, 80)!r}")
        return f"OperationModel({', '.join(parts)})"

    @safe_display
    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        if cycle:
            p.text("OperationModel(...)")
            return
        with p.group(2, "OperationModel(", ")"):
            p.breakable()
            p.text(f"operation_id={self.operation_id!r},")
            p.breakable()
            p.text(f"status={self.status!r},")
            if self.operation_type is not None:
                p.breakable()
                p.text(f"operation_type={self.operation_type!r},")
            if self.file_id is not None:
                p.breakable()
                p.text(f"file_id={self.file_id!r},")
            if self.percent_complete is not None:
                p.breakable()
                p.text(f"percent_complete={self.percent_complete!r},")
            if self.created_at is not None:
                p.breakable()
                p.text(f"created_at={self.created_at!r},")
            if self.completed_on is not None:
                p.breakable()
                p.text(f"completed_on={self.completed_on!r},")
            if self.ingestion_units is not None:
                p.breakable()
                p.text(f"ingestion_units={self.ingestion_units!r},")
            if self.error is not None:
                p.breakable()
                p.text(f"error={truncate_text(self.error, 80)!r},")

    @safe_display
    def _repr_html_(self) -> str:
        builder = HtmlBuilder("OperationModel")
        builder.row("Operation ID:", self.operation_id)
        builder.row("Status:", self.status)
        if self.operation_type is not None:
            builder.row("Type:", self.operation_type)
        if self.file_id is not None:
            builder.row("File ID:", self.file_id)
        if self.percent_complete is not None:
            builder.row("Progress:", f"{self.percent_complete}%")
        if self.created_at is not None:
            builder.row("Created:", self.created_at)
        if self.completed_on is not None:
            builder.row("Completed:", self.completed_on)
        if self.ingestion_units is not None:
            builder.row("Ingestion units:", self.ingestion_units)
        if self.error is not None:
            builder.section("Error", [("Message", truncate_text(self.error, 200))], theme="error")
        return builder.build()
