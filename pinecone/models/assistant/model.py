"""Assistant response model."""

from __future__ import annotations

from typing import Any

from msgspec import Struct

from pinecone._internal.config import normalize_host
from pinecone.models._display import HtmlBuilder, abbreviate_dict, safe_display, truncate_text
from pinecone.models.assistant._legacy_methods import AssistantModelLegacyMethodsMixin
from pinecone.models.assistant._mixin import StructDictMixin


class AssistantModel(
    AssistantModelLegacyMethodsMixin,
    StructDictMixin,
    Struct,
    dict=True,
    kw_only=True,
):
    """Response model for a Pinecone assistant.

    Attributes:
        name: The name of the assistant.
        status: Current status of the assistant (e.g. ``"Initializing"``,
            ``"Ready"``, ``"Terminating"``, ``"Terminated"``,
            ``"InitializationFailed"``).
        created_at: ISO 8601 timestamp when the assistant was created, or
            ``None`` if not returned by the API.
        updated_at: ISO 8601 timestamp when the assistant was last updated, or
            ``None`` if not returned by the API.
        metadata: Optional metadata dictionary associated with the assistant,
            or ``None`` if not set.
        instructions: Optional description or directive for the assistant
            to apply to all responses, or ``None`` if not set.
        host: The host where the assistant is deployed, or ``None`` if
            not yet available.
        region: The region the assistant is deployed in (``"us"`` or
            ``"eu"``), or ``None`` if not returned by the API.
    """

    name: str
    status: str
    metadata: dict[str, Any] | None = None
    instructions: str | None = None
    host: str | None = None
    region: str | None = None
    created_at: str | None = None
    updated_at: str | None = None

    def __post_init__(self) -> None:
        """Normalize host to always include https:// scheme when present."""
        if self.host is not None:
            self.host = normalize_host(self.host)

    @safe_display
    def __repr__(self) -> str:
        parts = [f"name={self.name!r}", f"status={self.status!r}"]
        if self.region is not None:
            parts.append(f"region={self.region!r}")
        if self.host is not None:
            parts.append(f"host={self.host!r}")
        if self.instructions is not None:
            parts.append(f"instructions={truncate_text(self.instructions, 40)!r}")
        if self.metadata is not None:
            parts.append(f"metadata=<{len(self.metadata)} keys>")
        return f"AssistantModel({', '.join(parts)})"

    @safe_display
    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        if cycle:
            p.text("AssistantModel(...)")
            return
        with p.group(2, "AssistantModel(", ")"):
            p.breakable()
            p.text(f"name={self.name!r},")
            p.breakable()
            p.text(f"status={self.status!r},")
            if self.region is not None:
                p.breakable()
                p.text(f"region={self.region!r},")
            if self.host is not None:
                p.breakable()
                p.text(f"host={self.host!r},")
            if self.instructions is not None:
                p.breakable()
                p.text(f"instructions={truncate_text(self.instructions, 40)!r},")
            if self.metadata is not None:
                p.breakable()
                p.text(f"metadata=<{len(self.metadata)} keys>,")
            if self.created_at is not None:
                p.breakable()
                p.text(f"created_at={self.created_at!r},")
            if self.updated_at is not None:
                p.breakable()
                p.text(f"updated_at={self.updated_at!r},")

    @safe_display
    def _repr_html_(self) -> str:
        builder = HtmlBuilder("AssistantModel")
        builder.row("Name:", self.name)
        builder.row("Status:", self.status)
        if self.region is not None:
            builder.row("Region:", self.region)
        if self.host is not None:
            builder.row("Host:", self.host)
        if self.instructions is not None:
            builder.row("Instructions:", truncate_text(self.instructions, 80))
        if self.metadata is not None:
            builder.row("Metadata:", abbreviate_dict(self.metadata))
        if self.created_at is not None:
            builder.row("Created:", self.created_at)
        if self.updated_at is not None:
            builder.row("Updated:", self.updated_at)
        return builder.build()
