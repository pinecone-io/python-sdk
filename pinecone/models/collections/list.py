"""CollectionList wrapper for collection listing responses."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from pinecone.models.collections.model import CollectionModel


class CollectionList:
    """The collections in a project, as returned by
    :meth:`Collections.list() <pinecone.client.collections.Collections.list>`.

    Iterating yields :class:`~pinecone.models.collections.model.CollectionModel`
    objects. ``len()`` and integer indexing work too, and every collection in the
    project is present — there is no pagination to follow.

    Examples:
        >>> collections = pc.collections.list()
        >>> len(collections)
        2
        >>> collections[0].name
        'movie-embeddings-snapshot'
    """

    def __init__(self, collections: list[CollectionModel]) -> None:
        self._collections = collections

    def __iter__(self) -> Iterator[CollectionModel]:
        return iter(self._collections)

    def __len__(self) -> int:
        return len(self._collections)

    def __getitem__(self, index: int) -> CollectionModel:
        return self._collections[index]

    def to_dict(self) -> dict[str, Any]:
        """Return the list as a serializable dict.

        Returns:
            dict[str, Any]: A dict with a ``"data"`` key containing a list of collection
            dicts, each produced by :meth:`CollectionModel.to_dict()
            <pinecone.models.collections.model.CollectionModel.to_dict>`.

        Examples:
            >>> collections = pc.collections.list()
            >>> payload = collections.to_dict()
            >>> [c["name"] for c in payload["data"]]
            ['movie-embeddings-snapshot', 'product-catalog-snapshot']
            >>> sorted(payload["data"][0])
            ['dimension', 'environment', 'name', 'size', 'status', 'vector_count']
        """
        return {"data": [c.to_dict() for c in self._collections]}

    def names(self) -> list[str]:
        """Return just the collection names, in the order the API returned them.

        Examples:
            >>> pc.collections.list().names()
            ['movie-embeddings-snapshot', 'product-catalog-snapshot']
        """
        return [c.name for c in self._collections]

    def __repr__(self) -> str:
        summaries = ", ".join(f"<name={c.name!r}, status={c.status!r}>" for c in self._collections)
        return f"CollectionList([{summaries}])"
