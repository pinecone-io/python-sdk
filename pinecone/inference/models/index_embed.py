"""Backwards-compatibility shim — IndexEmbed model for integrated indexes.

Defines the class that used to live at
:mod:`pinecone.inference.models.index_embed` in earlier releases, so that code
written against it keeps working.

:meta private:
"""

from __future__ import annotations

import dataclasses
from typing import Any

__all__ = ["IndexEmbed"]


@dataclasses.dataclass(frozen=True)
class IndexEmbed:
    """Which model embeds an integrated index, and which field it embeds.

    Accepted as the ``embed`` argument of
    :meth:`~pinecone.Pinecone.create_index_for_model`, alongside a plain dict
    and ``EmbedConfig``. Kept here, and importable from ``pinecone``, for code
    written against earlier releases.

    :meta private:
    """

    model: str
    field_map: dict[str, Any]
    metric: str | None = None
    read_parameters: dict[str, Any] = dataclasses.field(default_factory=dict)
    write_parameters: dict[str, Any] = dataclasses.field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return the instance's field values as a plain dictionary."""
        return self.__dict__
