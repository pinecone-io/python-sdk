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
    """Stored, point-in-time snapshots of a serverless or BYOC index.

    A backup captures an index's records and schema so that a new index can
    be created from it later with
    :meth:`~pinecone.AsyncPinecone.create_index_from_backup`. Backups are
    identified by a ``backup_id`` of their own and outlive the index they
    were taken from. Reached as ``pc.backups``; not constructed directly.

    Backups are the snapshot mechanism for serverless and BYOC indexes.
    :class:`~pinecone.async_client.collections.AsyncCollections` is the
    pod-based equivalent, and the two do not interchange: a pod-based index
    is snapshotted into a collection, a serverless or BYOC index into a
    backup.

    Examples:

        .. code-block:: python

            from pinecone import AsyncPinecone

            async with AsyncPinecone(api_key="your-api-key") as pc:
                page = await pc.backups.list(limit=100)
                print([b.backup_id for b in page])

    .. seealso::
       - :meth:`~pinecone.async_client.indexes.AsyncIndexes.list_backups` —
         the index-scoped listing, which walks every page for you.
       - :doc:`/guides/error-handling` — the exceptions any of these methods
         can raise, and which ones are worth retrying.
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

        Only serverless and BYOC indexes can be backed up. The call returns as
        soon as the snapshot is initiated, not when it is ready.

        Args:
            index_name (str): Name of the index to back up.
            name (str | None): Name for the backup, e.g. ``"daily-20240115"``.
                When omitted, the backup has no name and is identified only
                by its ``backup_id``.
            description (str | None): Description for the backup.

        Returns:
            A :class:`BackupModel` describing the new backup. The call
            returns once the backup is initiated, so ``status`` is
            ``"Initializing"`` rather than ``"Ready"``; poll
            :meth:`describe` to follow it.

        Raises:
            :exc:`PineconeValueError`: If *index_name* is empty.
            :exc:`ForbiddenError`: If the organization's plan does not
                include backups.
            :exc:`NotFoundError`: If *index_name* does not resolve to an
                index in this project.
            :exc:`ApiError`: If *index_name* names a pod-based index, which
                is snapshotted into a collection rather than a backup.

        Examples:
            Poll :meth:`describe` until the status *leaves*
            ``"Initializing"``: a backup that fails settles on ``"Failed"``,
            so waiting for ``"Ready"`` specifically would never return.

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

        .. seealso::
           - :meth:`~pinecone.async_client.backup_schedules.AsyncBackupSchedules.create`
             — a recurring cadence instead of this one-off snapshot.
           - :meth:`~pinecone.AsyncPinecone.create_index_from_backup` —
             restoring a backup into a new index.
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
        """List one page of backups.

        When *index_name* is given, lists backups of that index only.
        Otherwise lists every backup in the project. One call returns one
        page: iterating the result walks that page and stops rather than
        following ``pagination`` on your behalf. Drive the token yourself to
        walk the rest — see :doc:`/guides/pagination`.

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
            ``BackupList.pagination`` is ``None`` on the final page. Paging
            walks a live result set rather than a fixed snapshot, so
            de-duplicate by ``backup_id`` rather than relying on page order.

        Raises:
            :exc:`PineconeValueError`: If *include_deleted* is given without
                *index_name*.
            :exc:`NotFoundError`: If *index_name* does not resolve to an
                active index and *include_deleted* is not ``True``.

        Examples:
            Passing *index_name* scopes the listing to one index:

            .. code-block:: python

                from pinecone import AsyncPinecone

                async with AsyncPinecone(api_key="your-api-key") as pc:
                    for backup in await pc.backups.list(index_name="product-search"):
                        print(backup.name, backup.status)

            Walk the project-wide listing by driving the token yourself,
            consuming each page before asking for the next one:

            .. code-block:: python

                async with AsyncPinecone(api_key="your-api-key") as pc:
                    page = await pc.backups.list(limit=100)
                    backups = list(page)
                    while page.pagination and page.pagination.next:
                        page = await pc.backups.list(
                            pagination_token=page.pagination.next
                        )
                        backups.extend(page)
                    print([b.backup_id for b in backups])

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

        .. note::
           If every index that ever used *index_name* has since been
           deleted, listing without *include_deleted* raises
           :exc:`NotFoundError` rather than returning an empty list. Pass
           ``include_deleted=True`` to see backups of deleted indexes too.

        .. seealso::
           :meth:`~pinecone.async_client.indexes.AsyncIndexes.list_backups` —
           the same index-scoped listing as a paginator that walks every page,
           instead of one page plus a token.

        .. versionchanged:: 10.0
           Added *include_deleted*. :class:`BackupModel` now carries
           :attr:`~pinecone.models.backups.model.BackupModel.source_index_deleted_at`
           instead of ``dimension``/``metric``.
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
        """Get the current state of one backup.

        Args:
            backup_id (str): The identifier of the backup to describe.

        Returns:
            A :class:`BackupModel` whose ``status`` is ``"Initializing"``,
            ``"Ready"``, or ``"Failed"``, alongside the ``source_index_name``
            it was taken from, the captured ``schema``, and the
            ``record_count`` and ``size_bytes`` of the snapshot.

        Raises:
            :exc:`PineconeValueError`: If *backup_id* is empty.
            :exc:`NotFoundError`: If the backup does not exist.

        Examples:
            .. code-block:: python

                from pinecone import AsyncPinecone

                async with AsyncPinecone(api_key="your-api-key") as pc:
                    backup = await pc.backups.describe(backup_id="bk-abc123")
                    print(backup.status, backup.source_index_name)

        .. seealso::
           :meth:`~pinecone.async_client.indexes.AsyncIndexes.describe_backup`
           — the same call reached from the ``indexes`` namespace, taking the
           backup id positionally.
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
            A :class:`BackupModel` whose ``status`` is ``"Initializing"``,
            ``"Ready"``, or ``"Failed"``, alongside the ``source_index_name``
            it was taken from, the captured ``schema``, and the
            ``record_count`` and ``size_bytes`` of the snapshot.

        Raises:
            :exc:`PineconeValueError`: If *backup_id* is empty.
            :exc:`NotFoundError`: If the backup does not exist.

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
