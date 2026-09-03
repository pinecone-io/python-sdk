"""The sequence wrapper the legacy index listing returns."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from pinecone.models.indexes.index import IndexModel


class IndexList:
    """The indexes that the legacy ``pc.list_indexes()`` hands back.

    A thin sequence of :class:`~pinecone.models.indexes.index.IndexModel`:
    iterate it, subscript it, take its ``len()``, or call :meth:`names`. Not
    constructed directly.

    New code should call :meth:`pc.indexes.list()
    <pinecone.client.indexes.Indexes.list>`, which returns a
    :class:`~pinecone.models.pagination.Paginator` instead — the same
    iteration, without materialising every index up front.

    Examples:
        >>> for idx in pc.list_indexes():
        ...     print(idx.name, idx.status.ready)
    """

    def __init__(self, indexes: list[IndexModel]) -> None:
        self._indexes = indexes

    @property
    def indexes(self) -> list[IndexModel]:
        """The underlying list, when you need a real ``list`` to hold on to."""
        return self._indexes

    def __iter__(self) -> Iterator[IndexModel]:
        return iter(self._indexes)

    def __len__(self) -> int:
        return len(self._indexes)

    def __getitem__(self, index: int) -> IndexModel:
        return self._indexes[index]

    def to_dict(self) -> dict[str, Any]:
        """Return the listing as nested plain dicts, for logging or JSON.

        Returns:
            A dict with a single ``"data"`` key holding one entry per index,
            each the output of
            :meth:`IndexModel.to_dict
            <pinecone.models.indexes.index.IndexModel.to_dict>`.

        Examples:
            >>> list(pc.list_indexes().to_dict())
            ['data']
        """
        return {"data": [i.to_dict() for i in self._indexes]}

    def names(self) -> list[str]:
        """Just the index names, in the order the server returned them."""
        return [idx.name for idx in self._indexes]

    def __repr__(self) -> str:
        summaries = ", ".join(
            f"<name={idx.name!r}, ready={idx.status.ready}>" for idx in self._indexes
        )
        return f"IndexList([{summaries}])"
