"""Async Backups namespace — create, list, describe, and delete operations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

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
                for backup in await pc.backups.list():
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

        Args:
            index_name (str): Name of the index to back up.
            name (str | None): Optional name for the backup.
            description (str | None): Description for the backup. When ``None``
                (the default), no description is sent and the backend stores ``None``.

        Returns:
            A :class:`BackupModel` describing the created backup.

        Raises:
            :exc:`PineconeValueError`: If *index_name* is empty.
            :exc:`ApiError`: If the API returns an error response.

        Examples:

            .. code-block:: python

                # Create a backup of an index
                from pinecone import AsyncPinecone
                async with AsyncPinecone(api_key="your-api-key") as pc:
                    backup = await pc.backups.create(
                        index_name="product-search",
                    )
                    print(backup.backup_id)

            .. code-block:: python

                # Create a backup with a name and description
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
        response = await self._http.post(f"/indexes/{index_name}/backups", json=body)
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

        When *index_name* is provided, lists backups for that index only
        (``list_index_backups``). Otherwise lists every backup in the project
        (``list_project_backups``).

        .. versionchanged:: 10.0
           Added *include_deleted*. Backups now carry
           :attr:`~pinecone.models.backups.model.BackupModel.source_index_deleted_at`
           instead of ``dimension``/``metric`` — see
           ``docs/migration/v10-2026-07-backup-models.md``.

        .. important::
           A 404 from the index-scoped listing does not necessarily mean
           "no such index ever existed". With *include_deleted* omitted or
           ``False``, *index_name* must resolve to an **active** index: if
           every index that used the name has been deleted, the API answers
           404 rather than an empty list. Retrying with
           ``include_deleted=True`` returns those backups; a 404 there means
           the name was never used in this project.

        Args:
            index_name (str | None): Index name to scope the listing to, or
                ``None`` for every backup in the project.
            limit (int | None): Maximum number of results per page, 1-100.
                When ``None``, the parameter is omitted and the server
                applies its own default. **Ignored when
                *pagination_token* is given**: the token already carries the
                page size it was minted with, and sending a different one
                alongside it would skip or repeat rows.
            pagination_token (str | None): Token for cursor-based pagination.
                Takes precedence over *limit* — see above.
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
            :exc:`NotFoundError`: If *index_name* does not resolve — see the
                404 semantics above.
            :exc:`ApiError`: If the API returns another error response.

        Examples:

            .. code-block:: python

                # List all backups in the project
                from pinecone import AsyncPinecone
                async with AsyncPinecone(api_key="your-api-key") as pc:
                    for backup in await pc.backups.list():
                        print(backup.backup_id, backup.name)

            .. code-block:: python

                # List backups for a specific index
                async with AsyncPinecone(api_key="your-api-key") as pc:
                    for backup in await pc.backups.list(
                        index_name="product-search",
                    ):
                        print(backup.name)

            .. code-block:: python

                # Recover backups of an index that has since been deleted
                async with AsyncPinecone(api_key="your-api-key") as pc:
                    orphaned = await pc.backups.list(
                        index_name="product-search",
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
            path = f"/indexes/{index_name}/backups"
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
                    backup = await pc.backups.describe(
                        backup_id="bk-daily-20240115",
                    )
                    print(backup.status)
        """
        require_non_empty("backup_id", backup_id)
        logger.info("Describing backup %r", backup_id)
        response = await self._http.get(f"/backups/{backup_id}")
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
                    backup = await pc.backups.get(
                        backup_id="bk-daily-20240115",
                    )
                    print(backup.status)
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
            .. code-block:: python

                from pinecone import AsyncPinecone
                async with AsyncPinecone(api_key="your-api-key") as pc:
                    await pc.backups.delete(backup_id="bk-daily-20240115")
        """
        require_non_empty("backup_id", backup_id)
        logger.info("Deleting backup %r", backup_id)
        await self._http.delete(f"/backups/{backup_id}")
        logger.debug("Deleted backup %r", backup_id)
