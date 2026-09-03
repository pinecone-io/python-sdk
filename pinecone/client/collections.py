"""Collections namespace — create, list, describe, and delete operations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from urllib.parse import quote

from pinecone._internal.adapters.collections_adapter import CollectionsAdapter
from pinecone._internal.validation import require_non_empty, require_valid_resource_name
from pinecone.models.collections.list import CollectionList
from pinecone.models.collections.model import CollectionModel

if TYPE_CHECKING:
    from pinecone._internal.http_client import HTTPClient

logger = logging.getLogger(__name__)


class Collections:
    """Control-plane operations for Pinecone collections.

    Provides methods to create, list, describe, and delete collections.

    Args:
        http (HTTPClient): HTTP client for making API requests.

    Examples:

        .. code-block:: python

            from pinecone import Pinecone

            pc = Pinecone(api_key="your-api-key")
            for col in pc.collections.list():
                print(col.name, col.status)
    """

    def __init__(self, http: HTTPClient) -> None:
        self._http = http
        self._adapter = CollectionsAdapter()

    def __repr__(self) -> str:
        """Return developer-friendly representation."""
        return "Collections()"

    def create(self, *, name: str, source: str) -> CollectionModel:
        """Create a collection from an existing pod-based index.

        A collection is a static copy of an index's vector data, held outside
        the index as a snapshot of its contents at the moment it was taken.
        Only a pod-based index can be used as a source, and it must already
        be ready. The call returns as soon as creation starts — it does not
        wait for the collection to become ready.

        .. note::
           The 2026-07 API has no path from a collection back to an index.
           :meth:`Pinecone.indexes.create` rejects ``source_collection`` with
           a :exc:`PineconeTypeError` in both spellings — as a top-level
           keyword argument, and nested in a
           :class:`~pinecone.models.indexes.specs.PodSpec` passed to the
           deprecated ``spec=`` argument. The backend rejects the request
           too, with ``400 Creating an index from collection or backup is
           not yet supported``. Use
           :meth:`Pinecone.create_index_from_backup` for the supported
           restore path.

        Args:
            name (str): Name for the new collection. 1-45 characters,
                lowercase alphanumeric and hyphens only, and can't start or
                end with a hyphen (e.g. ``"movie-embeddings-snapshot"``).
            source (str): Name of the pod-based index to copy.

        Returns:
            A CollectionModel describing the created collection.

        Raises:
            PineconeValueError: If *name* or *source* is empty, or *name*
                doesn't meet the naming rules above.
            NotFoundError: If *source* does not name an index in this
                project.

        Examples:
            The collection is still being built when the call returns, so its
            status is ``"Initializing"`` rather than ``"Ready"``:

            >>> col = pc.collections.create(
            ...     name="movie-embeddings-snapshot", source="movie-recommendations"
            ... )
            >>> col.status
            'Initializing'

            There is no ``timeout=`` argument to wait on. Poll
            :meth:`describe` until the status leaves ``"Initializing"``, then
            read ``col.status`` to see where it settled:

            >>> import time
            >>> while col.status == "Initializing":
            ...     time.sleep(5)
            ...     col = pc.collections.describe(col.name)
            >>> col.status
            'Ready'
        """
        require_valid_resource_name("name", name)
        require_non_empty("source", source)
        logger.info("Creating collection %r from source %r", name, source)
        response = self._http.post("/collections", json={"name": name, "source": source})
        result = self._adapter.to_collection(response.content)
        logger.debug("Created collection %r", name)
        return result

    def list(self) -> CollectionList:
        """List every collection in the project.

        There's no filtering, sorting, or pagination — all collections come
        back at once.

        Returns:
            A CollectionList supporting iteration, len(), index access,
            and a names() convenience method.

        Examples:
            >>> collections = pc.collections.list()
            >>> collections.names()
            ['movie-embeddings-snapshot', 'product-catalog-snapshot']
            >>> for col in collections:
            ...     print(col.name, col.status)
            movie-embeddings-snapshot Ready
            product-catalog-snapshot Initializing
        """
        logger.info("Listing collections")
        response = self._http.get("/collections")
        result = self._adapter.to_collection_list(response.content)
        logger.debug("Listed %d collections", len(result))
        return result

    def describe(self, name: str) -> CollectionModel:
        """Get details about a collection.

        Args:
            name (str): Name of the collection to describe.

        Returns:
            A CollectionModel with the collection's name, status, size,
            dimension, vector_count, and environment.

        Raises:
            PineconeValueError: If *name* is empty.
            NotFoundError: If the collection does not exist.

        Examples:
            ``size`` is how much space the snapshot occupies, in bytes — not
            the dimension of its vectors. It, ``dimension``, and
            ``vector_count`` are ``None`` until the collection finishes
            initializing:

            >>> desc = pc.collections.describe("movie-embeddings-snapshot")
            >>> print(desc.status, desc.dimension, desc.vector_count, desc.size)
            Ready 1024 99 3126700
        """
        require_non_empty("name", name)
        logger.info("Describing collection %r", name)
        response = self._http.get(f"/collections/{quote(name, safe='')}")
        result = self._adapter.to_collection(response.content)
        logger.debug("Described collection %r", name)
        return result

    def delete(self, name: str) -> None:
        """Delete a collection permanently.

        Deletion is asynchronous: the call returns as soon as the request is
        accepted, and the collection can still show up in :meth:`list` for a
        short time afterwards. The source index can't be deleted until the
        collection is really gone.

        Args:
            name (str): Name of the collection to delete.

        Raises:
            PineconeValueError: If *name* is empty.
            NotFoundError: If the collection does not exist.

        Examples:
            >>> pc.collections.delete("movie-embeddings-snapshot")
        """
        require_non_empty("name", name)
        logger.info("Deleting collection %r", name)
        self._http.delete(f"/collections/{quote(name, safe='')}")
        logger.debug("Deleted collection %r", name)
