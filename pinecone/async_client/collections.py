"""Async Collections namespace — create, list, describe, and delete operations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

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
        """Create a collection from an existing index.

        Returns immediately after the API call without polling for
        readiness.

        Collections are a **pod-only** feature. The source index must be a pod
        index and must already be ``Ready``; see the note below for the exact
        checks and the order the server applies them in.

        Args:
            name (str): Name for the new collection.
            source (str): Name of the source index.

        Returns:
            A CollectionModel describing the created collection.

        Raises:
            PineconeValueError: If *name* is empty, longer than 45 characters,
                contains characters outside ``[a-z0-9-]``, or starts/ends with
                a hyphen. Also raised if *source* is empty.
            NotFoundError: If *source* does not name an index in this project.
            ApiError: If the API returns another error response (e.g. the
                ``400``\\ s in the note below, authentication failure, or a
                server error).

        .. note::
           The server checks the source index in this order and stops at the
           first failure:

           1. **Existence.** An unknown *source* is ``404``.
           2. **Readiness.** A source index that is not yet ``Ready`` is
              ``400`` "Source index is not ready". This is checked *before*
              the capacity-mode check, so a serverless index that is still
              initialising reports a readiness failure rather than naming the
              real problem.
           3. **Capacity mode.** Only pod indexes can be collected. A
              serverless *or* BYOC source is ``400`` "Cannot create
              collections from serverless indexes" — the message says
              "serverless" for a BYOC source too.

           A per-project collection quota is enforced after these checks.

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
        """List all collections in the project.

        Returns all collections in a single response without filtering,
        sorting, or pagination.

        Returns:
            A CollectionList supporting iteration, len(), index access,
            and a names() convenience method.

        Raises:
            ApiError: If the API returns an error response (e.g. authentication
                failure or server error).

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
        """Get detailed information about a named collection.

        Args:
            name (str): The name of the collection to describe.

        Returns:
            A CollectionModel with name, status, size, dimension,
            vector_count, and environment.

        Raises:
            PineconeValueError: If *name* is empty.
            NotFoundError: If the collection does not exist.
            ApiError: If the API returns another error response.

        Examples:

            .. code-block:: python

                desc = await pc.collections.describe("my-collection")
                print(desc.size)
        """
        require_non_empty("name", name)
        logger.info("Describing collection %r", name)
        response = await self._http.get(f"/collections/{name}")
        result = self._adapter.to_collection(response.content)
        logger.debug("Described collection %r", name)
        return result

    async def delete(self, name: str) -> None:
        """Delete a collection by name.

        Args:
            name (str): The name of the collection to delete.

        Raises:
            PineconeValueError: If *name* is empty.
            NotFoundError: If the collection does not exist.
            ApiError: If the API returns another error response.

        Examples:

            .. code-block:: python

                await pc.collections.delete("my-collection")
        """
        require_non_empty("name", name)
        logger.info("Deleting collection %r", name)
        await self._http.delete(f"/collections/{name}")
        logger.debug("Deleted collection %r", name)
