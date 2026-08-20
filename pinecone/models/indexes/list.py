"""IndexList wrapper for index listing responses."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from pinecone.models.indexes.index import IndexModel


class IndexList:
    """Wrapper around a list of IndexModel with convenience methods."""

    def __init__(self, indexes: list[IndexModel]) -> None:
        self._indexes = indexes

    @property
    def indexes(self) -> list[IndexModel]:
        """Return the list of indexes."""
        return self._indexes

    def __iter__(self) -> Iterator[IndexModel]:
        return iter(self._indexes)

    def __len__(self) -> int:
        return len(self._indexes)

    def __getitem__(self, index: int) -> IndexModel:
        return self._indexes[index]

    def to_dict(self) -> dict[str, Any]:
        """Return the list as a serializable dict.

        Returns:
            dict[str, Any]: A dict with a ``"data"`` key containing a list of
            index dicts, each produced by :meth:`IndexModel.to_dict`.

        Examples:
            >>> from pinecone import Pinecone
            >>> pc = Pinecone(api_key="your-api-key")
            >>> indexes = pc.list_indexes()
            >>> indexes.to_dict()  # doctest: +SKIP
            {'data': [{'name': 'movie-recommendations', ...}, {'name': 'product-search', ...}]}
        """
        return {"data": [i.to_dict() for i in self._indexes]}

    def names(self) -> list[str]:
        """Return a list of index names."""
        return [idx.name for idx in self._indexes]

    def __repr__(self) -> str:
        summaries = ", ".join(
            f"<name={idx.name!r}, ready={idx.status.ready}>" for idx in self._indexes
        )
        return f"IndexList([{summaries}])"
