"""Async Backups namespace — create, list, describe, and delete operations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from pinecone._internal.adapters.backups_adapter import BackupsAdapter
from pinecone._internal.backups_helpers import (
    backup_list_params,
    require_index_scope_for_include_deleted,
)
from pinecone._internal.validation import require_non_empty
from pinecone.models.backups.list import BackupList
from pinecone.models.backups.model import BackupModel

if TYPE_CHECKING:
    from pinecone._internal.http_client import AsyncHTTPClient

logger = logging.getLogger(__name__)


class AsyncBackups:
    """Async control-plane operations for Pinecone backups.

    Provides methods to create, list, describe, and delete backups.

    Args:
        http (AsyncHTTPClient): Async HTTP client for making API requests.

    Examples:

        .. code-block:: python

            from pinecone import AsyncPinecone

            async with AsyncPinecone(api_key="your-api-key") as pc:
                first_page = await pc.backups.list(limit=100)
                for backup in first_page:
                    print(backup.backup_id)
    """

    def __init__(self, http: AsyncHTTPClient) -> None:
        self._http = http
        self._adapter = BackupsAdapter()

    def __repr__(self) -> str:
        """Return developer-friendly representation."""
        return "AsyncBackups()"

    async def create(
        self,
        *,
        index_name: str,
        name: str | None = None,
        description: str | None = None,
    ) -> BackupModel:
        """Create a backup of an existing index.

        A backup is a stored, point-in-time snapshot of an index's data and
        schema. Restore one into a new index with
        :meth:`AsyncPinecone.create_index_from_backup`. Only serverless and
        BYOC indexes can be backed up.

        Args:
            index_name (str): Name of the index to back up.
            name (str | None): Name for the backup, e.g. ``"daily-20240115"``.
                When omitted, the backup has no name and is identified only
                by its ``backup_id``.
            description (str | None): Description for the backup.

        Returns:
            A :class:`BackupModel` describing the new backup. The call
            returns once the backup is initiated; check its ``status`` via
            :meth:`describe` to see when it's ready.

        Raises:
            :exc:`PineconeValueError`: If *index_name* is empty.
            :exc:`ForbiddenError`: If the organization's plan does not
                include backups.
            :exc:`NotFoundError`: If *index_name* does not resolve to an
                index in this project.
            :exc:`ApiError`: If the API returns another error response, for
                example because *index_name* names a pod-based index.

        Examples:
            Creating a backup is asynchronous. The call returns as soon as
            the backup is initiated, so the model it hands back reports
            ``"Initializing"`` rather than ``"Ready"``. Poll :meth:`describe`
            until the status leaves ``"Initializing"``: a backup that fails
            settles on ``"Failed"``, so waiting for ``"Ready"`` specifically
            would never return.

            .. code-block:: python

                import asyncio

                from pinecone import AsyncPinecone

                async with AsyncPinecone(api_key="your-api-key") as pc:
                    backup = await pc.backups.create(index_name="product-search")
                    print(backup.backup_id, backup.status)

                    while backup.status == "Initializing":
                        await asyncio.sleep(10)
                        backup = await pc.backups.describe(backup_id=backup.backup_id)
                    print(backup.status)

            Give the backup a name and description so a later listing
            identifies it by more than its server-assigned ``backup_id``:

            .. code-block:: python

                async with AsyncPinecone(api_key="your-api-key") as pc:
                    backup = await pc.backups.create(
                        index_name="product-search",
                        name="daily-20240115",
                        description="Scheduled daily backup before reindexing",
                    )
        """
        require_non_empty("index_name", index_name)
        body: dict[str, Any] = {}
        if name is not None:
            body["name"] = name
        if description is not None:
            body["description"] = description
        logger.info("Creating backup for index %r", index_name)
        response = await self._http.post(
            f"/indexes/{quote(index_name, safe='')}/backups", json=body
        )
        result = self._adapter.to_backup(response.content)
        logger.debug("Created backup %r", result.backup_id)
        return result

    async def list(
        self,
        *,
        index_name: str | None = None,
        limit: int | None = None,
        pagination_token: str | None = None,
        include_deleted: bool | None = None,
    ) -> BackupList:
        """List backups.

        When *index_name* is given, lists backups of that index only.
        Otherwise lists every backup in the project.

        .. versionchanged:: 10.0
           Added *include_deleted*. :class:`BackupModel` now carries
           :attr:`~pinecone.models.backups.model.BackupModel.source_index_deleted_at`
           instead of ``dimension``/``metric``.

        .. note::
           If every index that ever used *index_name* has since been
           deleted, listing without *include_deleted* raises
           :exc:`NotFoundError` rather than returning an empty list. Pass
           ``include_deleted=True`` to see backups of deleted indexes too.

           Because paging walks a live result set rather than a fixed
           snapshot, backups created or deleted between requests can shift
           later pages. De-duplicate by ``backup_id`` rather than relying on
           page order, and stop once ``pagination`` is ``None``.

        Args:
            index_name (str | None): Index name to scope the listing to, or
                ``None`` for every backup in the project.
            limit (int | None): Maximum number of results per page. When
                ``None``, the parameter is omitted and the server applies its
                own default. Omitted too when *pagination_token* is given: the
                token already carries the page size it was minted with, and a
                different one sent alongside it would skip or repeat rows.
            pagination_token (str | None): Offset token naming the next page,
                taken from ``BackupList.pagination.next``. Takes precedence
                over *limit* — see above.
            include_deleted (bool | None): When ``True``, include backups of
                every index that has ever used *index_name*, deleted ones
                included. When ``None`` (the default) the parameter is
                omitted entirely and the server's default (``false``)
                applies. Only valid together with *index_name*.

        Returns:
            A :class:`BackupList` supporting iteration, len(), and index access.
            ``BackupList.pagination`` is ``None`` on the final page.

        Raises:
            :exc:`PineconeValueError`: If *include_deleted* is given without
                *index_name*.
            :exc:`NotFoundError`: If *index_name* does not resolve to an
                active index and *include_deleted* is not ``True``.
            :exc:`ApiError`: If the API returns another error response.

        Examples:
            One call returns one page. Iterating the result walks that page
            and stops — it does not follow ``pagination`` on your behalf:

            .. code-block:: python

                from pinecone import AsyncPinecone

                async with AsyncPinecone(api_key="your-api-key") as pc:
                    page = await pc.backups.list(limit=100)
                    for backup in page:
                        print(backup.backup_id, backup.name)

            Walk the rest by driving the token yourself, consuming each page
            before asking for the next one:

            .. code-block:: python

                async with AsyncPinecone(api_key="your-api-key") as pc:
                    page = await pc.backups.list(limit=100)
                    backups = list(page)
                    while page.pagination and page.pagination.next:
                        page = await pc.backups.list(
                            pagination_token=page.pagination.next
                        )
                        backups.extend(page)

                print(len(backups))

            Passing *index_name* scopes the listing to one index.
            :meth:`~pinecone.async_client.indexes.AsyncIndexes.list_backups`
            covers the same ground with a paginator that walks every page for
            you:

            .. code-block:: python

                async with AsyncPinecone(api_key="your-api-key") as pc:
                    for backup in await pc.backups.list(index_name="product-search"):
                        print(backup.name, backup.status)

            Backups outlive the index they were taken from, but an
            index-scoped listing resolves *index_name* against the active
            indexes first. Pass ``include_deleted=True`` to reach the backups
            of an index you have already torn down:

            .. code-block:: python

                async with AsyncPinecone(api_key="your-api-key") as pc:
                    orphaned = await pc.backups.list(
                        index_name="legacy-catalog",
                        include_deleted=True,
                    )
                    print([b.backup_id for b in orphaned if b.source_index_deleted_at])
        """
        require_index_scope_for_include_deleted(index_name, include_deleted)
        params: dict[str, Any] = backup_list_params(
            limit=limit,
            pagination_token=pagination_token,
            include_deleted=include_deleted,
        )

        if index_name is not None:
            path = f"/indexes/{quote(str(index_name), safe='')}/backups"
        else:
            path = "/backups"

        logger.info("Listing backups (path=%s)", path)
        response = await self._http.get(path, params=params)
        result = self._adapter.to_backup_list(response.content)
        logger.debug("Listed %d backups", len(result))
        return result

    async def describe(self, *, backup_id: str) -> BackupModel:
        """Get detailed information about a backup.

        Args:
            backup_id (str): The identifier of the backup to describe.

        Returns:
            A :class:`BackupModel` with full backup details.

        Raises:
            :exc:`PineconeValueError`: If *backup_id* is empty.
            :exc:`NotFoundError`: If the backup does not exist.
            :exc:`ApiError`: If the API returns another error response.

        Examples:
            .. code-block:: python

                from pinecone import AsyncPinecone

                async with AsyncPinecone(api_key="your-api-key") as pc:
                    backup = await pc.backups.describe(backup_id="bk-abc123")
                    print(backup.status, backup.source_index_name)
        """
        require_non_empty("backup_id", backup_id)
        logger.info("Describing backup %r", backup_id)
        response = await self._http.get(f"/backups/{quote(backup_id, safe='')}")
        result = self._adapter.to_backup(response.content)
        logger.debug("Described backup %r", backup_id)
        return result

    async def get(self, *, backup_id: str) -> BackupModel:
        """Get detailed information about a backup (alias for :meth:`describe`).

        Args:
            backup_id (str): The identifier of the backup.

        Returns:
            A :class:`BackupModel` with full backup details.

        Raises:
            :exc:`PineconeValueError`: If *backup_id* is empty.
            :exc:`NotFoundError`: If the backup does not exist.
            :exc:`ApiError`: If the API returns another error response.

        Examples:
            .. code-block:: python

                from pinecone import AsyncPinecone

                async with AsyncPinecone(api_key="your-api-key") as pc:
                    backup = await pc.backups.get(backup_id="bk-abc123")
                    print(backup.status, backup.source_index_name)
        """
        return await self.describe(backup_id=backup_id)

    async def delete(self, *, backup_id: str) -> None:
        """Delete a backup.

        Args:
            backup_id (str): The identifier of the backup to delete.

        Raises:
            :exc:`PineconeValueError`: If *backup_id* is empty.
            :exc:`NotFoundError`: If the backup does not exist.
            :exc:`ApiError`: If the API returns another error response.

        Examples:
            Deleting a backup discards the snapshot only. The index it was
            taken from is untouched, and other backups of that index remain:

            .. code-block:: python

                from pinecone import AsyncPinecone

                async with AsyncPinecone(api_key="your-api-key") as pc:
                    await pc.backups.delete(backup_id="bk-abc123")
        """
        require_non_empty("backup_id", backup_id)
        logger.info("Deleting backup %r", backup_id)
        await self._http.delete(f"/backups/{quote(backup_id, safe='')}")
        logger.debug("Deleted backup %r", backup_id)
