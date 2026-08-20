"""Context options for assistant chat and context operations."""

from __future__ import annotations

from typing import Any

from msgspec import Struct

from pinecone.models._display import HtmlBuilder, safe_display
from pinecone.models.assistant._mixin import StructDictMixin


class ContextOptions(StructDictMixin, Struct, kw_only=True):
    """Options controlling how context is retrieved for assistant operations.

    All fields are optional and default to ``None``, letting the server
    apply its own defaults.

    Attributes:
        top_k: Maximum number of context snippets to retrieve. The backend
            accepts 1-64; ``0`` and values above 64 are each rejected with a
            400 (the spec documents a default of 16).
        snippet_size: Target size (in tokens) for each context snippet. The
            backend accepts 512-8192; anything outside that range is rejected
            with a 400 (the spec documents a default of 2048).
        multimodal: Whether to include multimodal (image) content in
            retrieved context.
        include_binary_content: Whether to include binary file content
            in retrieved context.
    """

    top_k: int | None = None
    snippet_size: int | None = None
    multimodal: bool | None = None
    include_binary_content: bool | None = None

    @safe_display
    def __repr__(self) -> str:
        fields: list[str] = []
        if self.top_k is not None:
            fields.append(f"top_k={self.top_k!r}")
        if self.snippet_size is not None:
            fields.append(f"snippet_size={self.snippet_size!r}")
        if self.multimodal is not None:
            fields.append(f"multimodal={self.multimodal!r}")
        if self.include_binary_content is not None:
            fields.append(f"include_binary_content={self.include_binary_content!r}")
        if not fields:
            return "ContextOptions(<default>)"
        return f"ContextOptions({', '.join(fields)})"

    @safe_display
    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        if cycle:
            p.text("ContextOptions(...)")
            return
        fields: list[str] = []
        if self.top_k is not None:
            fields.append(f"top_k={self.top_k!r}")
        if self.snippet_size is not None:
            fields.append(f"snippet_size={self.snippet_size!r}")
        if self.multimodal is not None:
            fields.append(f"multimodal={self.multimodal!r}")
        if self.include_binary_content is not None:
            fields.append(f"include_binary_content={self.include_binary_content!r}")
        if not fields:
            p.text("ContextOptions(<default>)")
            return
        with p.group(2, "ContextOptions(", ")"):
            for i, field in enumerate(fields):
                if i > 0:
                    p.text(",")
                    p.breakable()
                p.text(field)

    @safe_display
    def _repr_html_(self) -> str:
        builder = HtmlBuilder("ContextOptions")
        all_none = (
            self.top_k is None
            and self.snippet_size is None
            and self.multimodal is None
            and self.include_binary_content is None
        )
        if all_none:
            builder.row("(server defaults)", "")
        else:
            if self.top_k is not None:
                builder.row("Top K:", self.top_k)
            if self.snippet_size is not None:
                builder.row("Snippet Size:", self.snippet_size)
            if self.multimodal is not None:
                builder.row("Multimodal:", self.multimodal)
            if self.include_binary_content is not None:
                builder.row("Include Binary Content:", self.include_binary_content)
        return builder.build()

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ContextOptions:
        """Construct a ``ContextOptions`` from a plain dict representation."""
        return cls(
            top_k=d.get("top_k"),
            snippet_size=d.get("snippet_size"),
            multimodal=d.get("multimodal"),
            include_binary_content=d.get("include_binary_content"),
        )
