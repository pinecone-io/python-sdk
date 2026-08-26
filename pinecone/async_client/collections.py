"""Async Collections namespace — create, list, describe, and delete operations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from urllib.parse import quote

from pinecone._internal.adapters.collections_adapter import CollectionsAdapter
from pinecone._internal.validation import require_non_empty, require_valid_resource_name
from pinecone.models.collections.list import CollectionList
from pinecone.models.collections.model import CollectionModel

if TYPE_CHECKING:
    from pinecone._internal.http_client import AsyncHTTPClient

logger = logging.getLogger(__name__)


class AsyncCollections:
    """Async control-plane operations for Pinecone collections.

    Provides methods to create, list, describe, and delete collections.

    Args:
        http (AsyncHTTPClient): Async HTTP client for making API requests.

    Examples:

        .. code-block:: python

            from pinecone import AsyncPinecone

            async with AsyncPinecone(api_key="your-api-key") as pc:
                for col in await pc.collections.list():
                    print(col.name)
    """

    def __init__(self, http: AsyncHTTPClient) -> None:
        self._http = http
        self._adapter = CollectionsAdapter()

    def __repr__(self) -> str:
        """Return developer-friendly representation."""
        return "AsyncCollections()"

    async def create(self, *, name: str, source: str) -> CollectionModel:
        """Create a collection from an existing pod-based index.

        A collection is a static copy of an index's vector data. Create one
        to preserve an index's contents, then later pass its name as
        ``source_collection`` when creating a new index to restore the data.
        Only a pod-based index can be used as a source, and it must already
        be ready. The call returns as soon as creation starts — it does not
        wait for the collection to become ready.

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

            .. code-block:: python

                col = await pc.collections.create(name="my-collection", source="my-index")
                print(col.status)
        """
        require_valid_resource_name("name", name)
        require_non_empty("source", source)
        logger.info("Creating collection %r from source %r", name, source)
        response = await self._http.post("/collections", json={"name": name, "source": source})
        result = self._adapter.to_collection(response.content)
        logger.debug("Created collection %r", name)
        return result

    async def list(self) -> CollectionList:
        """List every collection in the project.

        There's no filtering, sorting, or pagination — all collections come
        back at once.

        Returns:
            A CollectionList supporting iteration, len(), index access,
            and a names() convenience method.

        Examples:

            .. code-block:: python

                collections = await pc.collections.list()
                print(collections.names())
                for col in collections:
                    print(col.name, col.status)
        """
        logger.info("Listing collections")
        response = await self._http.get("/collections")
        result = self._adapter.to_collection_list(response.content)
        logger.debug("Listed %d collections", len(result))
        return result

    async def describe(self, name: str) -> CollectionModel:
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

            .. code-block:: python

                desc = await pc.collections.describe("my-collection")
                print(desc.size)
        """
        require_non_empty("name", name)
        logger.info("Describing collection %r", name)
        response = await self._http.get(f"/collections/{quote(name, safe='')}")
        result = self._adapter.to_collection(response.content)
        logger.debug("Described collection %r", name)
        return result

    async def delete(self, name: str) -> None:
        """Delete a collection permanently.

        Args:
            name (str): Name of the collection to delete.

        Raises:
            PineconeValueError: If *name* is empty.
            NotFoundError: If the collection does not exist.

        Examples:

            .. code-block:: python

                await pc.collections.delete("my-collection")
        """
        require_non_empty("name", name)
        logger.info("Deleting collection %r", name)
        await self._http.delete(f"/collections/{quote(name, safe='')}")
        logger.debug("Deleted collection %r", name)
