"""Assistant file response model."""

from __future__ import annotations

from typing import Any

from msgspec import Struct

from pinecone.models._display import HtmlBuilder, abbreviate_dict, safe_display, truncate_text
from pinecone.models.assistant._mixin import StructDictMixin

_REMOVED_FIELD_HINTS: dict[str, str] = {
    "percent_done": (
        "processing progress is tracked by the operations API instead — call "
        "describe_operation() and read OperationModel.status"
    ),
    "error_message": (
        "processing failure detail is reported by the operations API instead — "
        "call describe_operation() and read OperationModel.error"
    ),
}


class AssistantFileModel(
    StructDictMixin,
    Struct,
    kw_only=True,
    rename={"content_hash": "crc32c_hash"},
):
    """Response model for a file attached to a Pinecone assistant.

    Attributes:
        name: The name of the file.
        id: Unique identifier for the file. On ``2026-07`` this may be a
            user-provided identifier, so it is not guaranteed to be a UUID.
        metadata: Optional metadata dictionary associated with the file,
            or ``None`` if not set.
        created_on: ISO 8601 timestamp when the file was created, or ``None``.
        updated_on: ISO 8601 timestamp when the file was last updated, or ``None``.
        status: Current status of the file (e.g. ``"Processing"``,
            ``"Available"``, ``"Deleting"``, ``"ProcessingFailed"``),
            or ``None``.
        size: Size of the file in bytes, or ``None``.
        multimodal: Whether the file was processed as multimodal, or ``None``.
        signed_url: A temporary signed URL for downloading the file, or ``None``
            when not requested or unavailable.
        content_hash: Hash of the file content (wire key ``crc32c_hash``), or
            ``None`` when not available.  Legacy callers can also access this
            value via the :attr:`crc32c_hash` property alias.

    ``percent_done`` and ``error_message`` were removed in the ``2026-07``
    API; accessing them raises an :class:`AttributeError` naming
    ``describe_operation`` as the replacement.
    """

    name: str
    id: str
    metadata: dict[str, object] | None = None
    created_on: str | None = None
    updated_on: str | None = None
    status: str | None = None
    size: int | None = None
    multimodal: bool | None = None
    signed_url: str | None = None
    content_hash: str | None = None

    @property
    def crc32c_hash(self) -> str | None:
        """Backwards-compatibility alias for :attr:`content_hash`."""
        return self.content_hash

    def __getattr__(self, name: str) -> Any:
        if name in _REMOVED_FIELD_HINTS:
            raise AttributeError(
                f"AssistantFileModel.{name} was removed in the 2026-07 Pinecone API: "
                f"{_REMOVED_FIELD_HINTS[name]}. "
                "See https://sdk.pinecone.io/python/migration/v10-migration.html."
            )
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    @safe_display
    def __repr__(self) -> str:
        parts = [f"name={self.name!r}", f"id={self.id!r}"]
        if self.status is not None:
            parts.append(f"status={self.status!r}")
        if self.size is not None:
            parts.append(f"size={self.size!r}")
        return f"AssistantFileModel({', '.join(parts)})"

    @safe_display
    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        if cycle:
            p.text("AssistantFileModel(...)")
            return
        with p.group(2, "AssistantFileModel(", ")"):
            p.breakable()
            p.text(f"name={self.name!r},")
            p.breakable()
            p.text(f"id={self.id!r},")
            if self.status is not None:
                p.breakable()
                p.text(f"status={self.status!r},")
            if self.size is not None:
                p.breakable()
                p.text(f"size={self.size!r},")
            if self.multimodal is not None:
                p.breakable()
                p.text(f"multimodal={self.multimodal!r},")
            if self.signed_url is not None:
                p.breakable()
                p.text(f"signed_url={truncate_text(self.signed_url, 80)!r},")
            if self.content_hash is not None:
                p.breakable()
                p.text(f"content_hash={self.content_hash!r},")
            if self.metadata is not None:
                p.breakable()
                p.text(f"metadata={abbreviate_dict(self.metadata)},")
            if self.created_on is not None:
                p.breakable()
                p.text(f"created_on={self.created_on!r},")
            if self.updated_on is not None:
                p.breakable()
                p.text(f"updated_on={self.updated_on!r},")

    @safe_display
    def _repr_html_(self) -> str:
        builder = HtmlBuilder("AssistantFileModel")
        builder.row("Name:", self.name)
        builder.row("ID:", self.id)
        if self.status is not None:
            builder.row("Status:", self.status)
        if self.size is not None:
            builder.row("Size:", self.size)
        if self.multimodal is not None:
            builder.row("Multimodal:", self.multimodal)
        if self.signed_url is not None:
            builder.row("Signed URL:", truncate_text(self.signed_url, max_chars=80))
        if self.content_hash is not None:
            builder.row("Content Hash:", self.content_hash)
        if self.metadata is not None:
            builder.row("Metadata:", abbreviate_dict(self.metadata))
        if self.created_on is not None:
            builder.row("Created:", self.created_on)
        if self.updated_on is not None:
            builder.row("Updated:", self.updated_on)
        if self.status is not None and "Failed" in self.status:
            builder.section(
                "Error",
                [("Detail:", "call describe_operation() for the failure reason")],
                theme="error",
            )
        return builder.build()
