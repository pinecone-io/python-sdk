"""Message model for the Assistant API."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from msgspec import Struct

from pinecone.models._display import HtmlBuilder, safe_display, truncate_text
from pinecone.models.assistant._mixin import StructDictMixin


class Message(StructDictMixin, Struct, kw_only=True):
    """A message to send to an assistant.

    Attributes:
        content: The text content of the message. Must not be blank — the
            backend trims before checking, so ``""`` and a whitespace-only
            string alike come back 400 ``"Message content cannot be empty"``.
        role: The role of the message author. Defaults to ``"user"``. The
            backend accepts only the exact strings ``"user"`` and
            ``"assistant"``, compared case-sensitively: ``"User"`` is rejected
            with 400 ``"Role 'User' is not valid"``, and ``""`` with 400
            ``"Role cannot be empty"``. Neither field is validated
            client-side.
    """

    content: str
    role: str = "user"

    @safe_display
    def __repr__(self) -> str:
        truncated = truncate_text(self.content, max_chars=80)
        return f"Message(role={self.role!r}, content={truncated!r})"

    @safe_display
    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        truncated = truncate_text(self.content, max_chars=200)
        p.text(f"Message(role={self.role!r}, content={truncated!r})")

    @safe_display
    def _repr_html_(self) -> str:
        return (
            HtmlBuilder("Message")
            .row("Role", self.role)
            .row("Content", truncate_text(self.content, max_chars=500))
            .build()
        )

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Message:
        """Create a ``Message`` from a dictionary.

        Extracts ``"content"`` and ``"role"`` keys, defaulting role to
        ``"user"`` when not present.

        Args:
            d: A dictionary with at least a ``"content"`` key.

        Returns:
            A new ``Message`` instance.
        """
        return cls(content=d["content"], role=d.get("role", "user"))
