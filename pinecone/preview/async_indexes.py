"""Async preview indexes namespace — control-plane operations backed by 2026-01.alpha."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

import msgspec

from pinecone._internal.backups_helpers import backup_list_params
from pinecone._internal.constants import DEFAULT_BASE_URL
from pinecone._internal.validation import require_non_empty, require_positive
from pinecone.errors.exceptions import NotFoundError, PineconeTimeoutError, PineconeValueError
from pinecone.models.backups.model import BackupModel
from pinecone.models.pagination import AsyncPaginator, Page
from pinecone.preview._internal.adapters.backups import (
    PreviewDescribeBackupAdapter,
    PreviewListBackupsAdapter,
)
from pinecone.preview._internal.adapters.indexes import (
    PreviewConfigureIndexAdapter,
    PreviewCreateIndexAdapter,
    PreviewDescribeIndexAdapter,
    PreviewListIndexesAdapter,
)
from pinecone.preview._internal.constants import INDEXES_API_VERSION
from pinecone.preview._internal.validation import validate_tags
from pinecone.preview.models.indexes import PreviewIndexModel
from pinecone.preview.models.requests import PreviewConfigureIndexRequest, PreviewCreateIndexRequest

if TYPE_CHECKING:
    from pinecone._internal.config import PineconeConfig

logger = logging.getLogger(__name__)

_POLL_INTERVAL_SECONDS = 5

__all__ = ["AsyncPreviewIndexes"]


class AsyncPreviewIndexes:
    """Async control-plane operations for preview indexes (``2026-01.alpha``).

    .. admonition:: Preview
       :class: warning

       Uses Pinecone API version ``2026-01.alpha``.
       Preview surface is not covered by SemVer — signatures and behavior
       may change in any minor SDK release. Pin your SDK version when
       relying on preview features.

    Access via ``pc.preview.indexes``.

    Args:
        config: SDK configuration used to construct an async HTTP client targeting
            the preview API version.

    Examples:
        .. code-block:: python

            pc = AsyncPinecone(api_key="...")
            index = await pc.preview.indexes.create(
                schema={"fields": {"embedding": {"type": "dense_vector", "dimension": 1536}}},
                name="my-preview-index",
            )
    """

    def __init__(self, config: PineconeConfig) -> None:
        from pinecone._internal.config import PineconeConfig as _PineconeConfig
        from pinecone._internal.http_client import AsyncHTTPClient as _AsyncHTTPClient

        self._config = config
        cp_host = (config.host or DEFAULT_BASE_URL).rstrip("/")
        cp_config = _PineconeConfig(
            api_key=config.api_key,
            host=cp_host,
            timeout=config.timeout,
            additional_headers=config.additional_headers,
            source_tag=config.source_tag or "",
            proxy_url=config.proxy_url or "",
            proxy_headers=config.proxy_headers,
            ssl_ca_certs=config.ssl_ca_certs,
            ssl_verify=config.ssl_verify,
            connection_pool_maxsize=config.connection_pool_maxsize,
            retry_config=config.retry_config,
        )
        self._http = _AsyncHTTPClient(cp_config, INDEXES_API_VERSION)

    async def close(self) -> None:
        """Close the underlying HTTP client. Idempotent.

        .. admonition:: Preview
           :class: warning

           Uses Pinecone API version ``2026-01.alpha``.
           Preview surface is not covered by SemVer — signatures and behavior
           may change in any minor SDK release. Pin your SDK version when
           relying on preview features.
        """
        await self._http.close()

    def __repr__(self) -> str:
        return "AsyncPreviewIndexes()"

    async def create(
        self,
        *,
        schema: dict[str, Any],
        name: str | None = None,
        deployment: dict[str, Any] | None = None,
        read_capacity: dict[str, Any] | None = None,
        deletion_protection: str | None = None,
        tags: dict[str, str] | None = None,
        source_collection: str | None = None,
        source_backup_id: str | None = None,
        cmek_id: str | None = None,
    ) -> PreviewIndexModel:
        """Create a new preview index.

        .. admonition:: Preview
           :class: warning

           Uses Pinecone API version ``2026-01.alpha``.
           Preview surface is not covered by SemVer — signatures and behavior
           may change in any minor SDK release. Pin your SDK version when
           relying on preview features.

        Args:
            schema: Index schema definition. A dict with a ``"fields"`` key
                mapping field names to their typed field definitions. Example::

                    {
                        "fields": {
                            "embedding": {"type": "dense_vector", "dimension": 1536},
                            "title": {"type": "string", "full_text_search": {}},
                        }
                    }

            name: Optional name for the index. If omitted, the API assigns one.
            deployment: Optional deployment configuration dict. Must include a
                ``"deployment_type"`` key (e.g. ``"managed"``, ``"pod"``,
                ``"byoc"``). When omitted the API uses its default deployment.
            read_capacity: Optional read capacity configuration dict. Must
                include a ``"mode"`` key (``"OnDemand"`` or ``"Dedicated"``).
                When omitted the API uses on-demand mode.
            deletion_protection: Optional deletion-protection setting —
                ``"enabled"`` or ``"disabled"``. Defaults to ``"disabled"``
                if not provided.
            tags: Optional key-value tags for the index. Keys must be at most
                80 characters and contain only alphanumeric characters,
                underscores, or hyphens. Values must be at most 120 printable
                ASCII characters. A maximum of 20 tags is allowed.
            source_collection: Optional name of an existing collection to
                create the index from.
            source_backup_id: Optional ID of an existing backup to create
                the index from.
            cmek_id: Optional Customer-Managed Encryption Key ID. Valid for
                managed and BYOC indexes; returns 400 for pod indexes.

        Returns:
            :class:`PreviewIndexModel` describing the newly created index. The
            returned model's ``status.ready`` may be ``False`` — use
            ``describe()`` to poll until the index is ready.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`: If a tag key
                exceeds 80 characters, contains invalid characters, if a tag
                value exceeds 120 characters or contains non-ASCII/non-printable
                characters, or if more than 20 tags are provided.
            :exc:`~pinecone.errors.exceptions.ApiError`: If the API returns an
                error response.

        Examples:
            .. code-block:: python

                index = await pc.preview.indexes.create(
                    schema={"fields": {"embedding": {"type": "dense_vector", "dimension": 1536}}},
                    name="my-preview-index",
                )
        """
        validate_tags(tags)

        req = PreviewCreateIndexRequest(
            schema=schema,
            name=name,
            deployment=deployment,
            read_capacity=read_capacity,
            deletion_protection=deletion_protection,
            tags=tags,
            source_collection=source_collection,
            source_backup_id=source_backup_id,
            cmek_id=cmek_id,
        )

        logger.info("Creating preview index name=%r", name)
        response = await self._http.post(
            "/indexes",
            content=PreviewCreateIndexAdapter.to_request(req),
            headers={"Content-Type": "application/json"},
        )
        return PreviewCreateIndexAdapter.from_response(response.content)

    async def configure(
        self,
        name: str,
        *,
        schema: dict[str, Any] | None = None,
        deletion_protection: str | None = None,
        tags: dict[str, str] | None = None,
        read_capacity: dict[str, Any] | None = None,
        deployment: dict[str, Any] | None = None,
    ) -> PreviewIndexModel:
        """Update configuration of an existing preview index.

        .. admonition:: Preview
           :class: warning

           Uses Pinecone API version ``2026-01.alpha``.
           Preview surface is not covered by SemVer — signatures and behavior
           may change in any minor SDK release. Pin your SDK version when
           relying on preview features.

        Only the fields you provide are updated; omitted parameters are left
        unchanged on the server.

        Args:
            name: Name of the preview index to configure.
            schema: Updated schema definition.  Pass a dict with a
                ``"fields"`` key mapping field names to typed field
                definitions.  The server only allows additive changes — it
                rejects requests that remove or modify existing fields.
                Example::

                    await pc.preview.indexes.configure(
                        "my-index",
                        schema={"fields": {
                            "summary": {"type": "semantic_text", "model": "multilingual-e5-large"},
                        }},
                    )

            deletion_protection: ``"enabled"`` to prevent accidental deletion;
                ``"disabled"`` to allow it.
            tags: Replacement set of key-value tags for the index.  Keys must
                be at most 80 characters; values at most 120 characters.
            read_capacity: Updated read capacity configuration dict.  Must
                include a ``"mode"`` key (``"OnDemand"`` or ``"Dedicated"``).
            deployment: Updated pod deployment configuration dict. For pod
                indexes only. May include ``"replicas"`` (int) and/or
                ``"pod_type"`` (str).

        Returns:
            :class:`PreviewIndexModel` reflecting the updated index state.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`: If *name*
                is empty; if all kwargs are ``None``; if *schema*, *tags*,
                *read_capacity*, or *deployment* is an empty dict; or if a
                tag key/value exceeds the length limit.
            :exc:`~pinecone.errors.exceptions.ApiError`: If the API returns
                an error response.

        Examples:

            .. code-block:: python

                async def main():
                    await pc.preview.indexes.configure(
                        "my-index",
                        schema={"fields": {
                            "summary": {
                                "type": "semantic_text",
                                "model": "multilingual-e5-large",
                            },
                        }},
                    )

            Update read capacity to dedicated::

                async def main():
                    await pc.preview.indexes.configure(
                        "my-index",
                        read_capacity={"mode": "Dedicated", "node_type": "b1",
                                       "scaling": "Manual",
                                       "manual": {"shards": 2, "replicas": 1}},
                    )

            Update tags::

                async def main():
                    await pc.preview.indexes.configure(
                        "my-index",
                        tags={"env": "prod", "team": "search"},
                    )

            Enable deletion protection::

                async def main():
                    await pc.preview.indexes.configure("my-index", deletion_protection="enabled")
        """
        require_non_empty("name", name)

        if schema is not None and not schema:
            raise PineconeValueError("schema cannot be an empty dict")
        if tags is not None and not tags:
            raise PineconeValueError("tags cannot be an empty dict")
        if read_capacity is not None and not read_capacity:
            raise PineconeValueError("read_capacity cannot be an empty dict")
        if deployment is not None and not deployment:
            raise PineconeValueError("deployment cannot be an empty dict")

        if schema is not None:
            fields = schema.get("fields", {})
            for field_name, field_def in fields.items():
                if field_def.get("type") != "semantic_text":
                    raise PineconeValueError(
                        f"Schema field {field_name!r} has type {field_def.get('type')!r}; "
                        "configure() only supports adding 'semantic_text' fields."
                    )

        if (
            schema is None
            and deletion_protection is None
            and tags is None
            and read_capacity is None
            and deployment is None
        ):
            raise PineconeValueError(
                "at least one configuration parameter must be provided: "
                "schema, deletion_protection, tags, read_capacity, or deployment"
            )

        validate_tags(tags)

        req = PreviewConfigureIndexRequest(
            schema=schema,
            read_capacity=read_capacity,
            deletion_protection=deletion_protection,
            tags=tags,
            deployment=deployment,
        )

        provided = [
            k
            for k, v in {
                "schema": schema,
                "deletion_protection": deletion_protection,
                "tags": tags,
                "read_capacity": read_capacity,
                "deployment": deployment,
            }.items()
            if v is not None
        ]
        logger.info("Configuring preview index name=%r params=%r", name, provided)

        response = await self._http.patch(
            f"/indexes/{name}",
            content=PreviewConfigureIndexAdapter.to_request(req),
            headers={"Content-Type": "application/json"},
        )
        return PreviewConfigureIndexAdapter.from_response(response.content)

    async def describe(self, name: str) -> PreviewIndexModel:
        """Get detailed information about a named preview index.

        .. admonition:: Preview
           :class: warning

           Uses Pinecone API version ``2026-01.alpha``.
           Preview surface is not covered by SemVer — signatures and behavior
           may change in any minor SDK release. Pin your SDK version when
           relying on preview features.

        Args:
            name: Name of the preview index to describe.

        Returns:
            :class:`PreviewIndexModel` with name, host, schema, deployment,
            read_capacity, status, deletion_protection, and tags.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`: If *name* is empty.
            :exc:`~pinecone.errors.exceptions.NotFoundError`: If the index does not exist.
            :exc:`~pinecone.errors.exceptions.ApiError`: If the API returns another error response.

        Examples:

            .. code-block:: python

                desc = await pc.preview.indexes.describe("product-search-preview")
                print(desc.host)
        """
        require_non_empty("name", name)
        logger.info("Describing preview index name=%r", name)
        response = await self._http.get(f"/indexes/{name}")
        return PreviewDescribeIndexAdapter.from_response(response.content)

    def list(
        self,
        *,
        limit: int | None = None,
        pagination_token: str | None = None,
    ) -> AsyncPaginator[PreviewIndexModel]:
        """List all preview indexes.

        .. admonition:: Preview
           :class: warning

           Uses Pinecone API version ``2026-01.alpha``.
           Preview surface is not covered by SemVer — signatures and behavior
           may change in any minor SDK release. Pin your SDK version when
           relying on preview features.

        The 2026-01.alpha server returns all indexes in a single page. The
        returned :class:`~pinecone.models.pagination.AsyncPaginator` always
        yields exactly one page and then terminates, but the paginator interface
        is used for consistency and forward compatibility.

        Args:
            limit: Maximum number of items to yield across all pages. Must be
                a positive integer. ``None`` yields all items.
            pagination_token: Token to resume pagination from a previous call.
                ``None`` starts from the beginning.

        Returns:
            :class:`~pinecone.models.pagination.AsyncPaginator` over
            :class:`~pinecone.preview.models.indexes.PreviewIndexModel`
            instances.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`: If *limit*
                is zero or negative.
            :exc:`~pinecone.errors.exceptions.ApiError`: If the API returns an
                error response.

        Examples:
            .. code-block:: python

                async for index in pc.preview.indexes.list():
                    print(index.name)

            Collect all into a list::

                all_indexes = await pc.preview.indexes.list().to_list()
        """
        if limit is not None:
            require_positive("limit", limit)

        async def fetch_page(token: str | None) -> Page[PreviewIndexModel]:
            response = await self._http.get("/indexes")
            items = PreviewListIndexesAdapter.from_response(response.content)
            return Page(items=items, pagination_token=None)

        return AsyncPaginator(fetch_page=fetch_page, initial_token=pagination_token, limit=limit)

    async def exists(self, name: str) -> bool:
        """Check whether a named preview index exists.

        .. admonition:: Preview
           :class: warning

           Uses Pinecone API version ``2026-01.alpha``.
           Preview surface is not covered by SemVer — signatures and behavior
           may change in any minor SDK release. Pin your SDK version when
           relying on preview features.

        Uses :meth:`describe` internally; catches :exc:`NotFoundError` and
        returns ``False``.

        Args:
            name: Name of the preview index to check.

        Returns:
            ``True`` if the index exists, ``False`` if it does not.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`: If *name* is empty.
            :exc:`~pinecone.errors.exceptions.ApiError`: If the API returns an error other than 404.

        Examples:

            .. code-block:: python

                if await pc.preview.indexes.exists("product-search-preview"):
                    print("Index found")
        """
        require_non_empty("name", name)
        try:
            await self.describe(name)
            return True
        except NotFoundError:
            return False

    async def delete(self, name: str, *, timeout: int | None = None) -> None:
        """Delete a named preview index.

        .. admonition:: Preview
           :class: warning

           Uses Pinecone API version ``2026-01.alpha``.
           Preview surface is not covered by SemVer — signatures and behavior
           may change in any minor SDK release. Pin your SDK version when
           relying on preview features.

        Sends a DELETE request and, depending on *timeout*, polls until the
        index disappears.

        Args:
            name: Name of the preview index to delete.
            timeout: Controls post-delete polling behaviour.

                - ``None`` (default): poll :meth:`describe` every 5 seconds
                  with no upper bound until the index is gone.
                - ``-1``: return immediately after the DELETE response without
                  polling.
                - Positive integer: poll until the index is gone or *timeout*
                  seconds have elapsed. Raises :exc:`PineconeTimeoutError` if
                  the deadline is reached before the index disappears.

        Returns:
            ``None``

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`: If *name* is empty.
            :exc:`~pinecone.errors.exceptions.NotFoundError`: If the index does not exist.
            :exc:`~pinecone.errors.exceptions.ForbiddenError`: If deletion protection is enabled.
            :exc:`~pinecone.errors.exceptions.PineconeTimeoutError`: If *timeout* seconds elapse
                before the index disappears.
            :exc:`~pinecone.errors.exceptions.ApiError`: If the API returns another error response.

        Examples:

            .. code-block:: python

                await pc.preview.indexes.delete("product-search-preview")

                # Delete and wait up to 60 seconds
                await pc.preview.indexes.delete("product-search-preview", timeout=60)

                # Delete without polling
                await pc.preview.indexes.delete("product-search-preview", timeout=-1)
        """
        require_non_empty("name", name)
        logger.info("Deleting preview index name=%r", name)
        await self._http.delete(f"/indexes/{name}")

        if timeout == -1:
            return

        start = time.monotonic()
        while True:
            try:
                await self.describe(name)
            except NotFoundError:
                return
            if timeout is not None:
                elapsed = time.monotonic() - start
                if elapsed >= timeout:
                    raise PineconeTimeoutError(f"Index {name!r} still exists after {timeout}s")
            await asyncio.sleep(_POLL_INTERVAL_SECONDS)

    async def create_backup(
        self,
        index_name: str,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> BackupModel:
        """Create a backup of a preview index.

        .. admonition:: Preview
           :class: warning

           Uses Pinecone API version ``2026-01.alpha``.
           Preview surface is not covered by SemVer — signatures and behavior
           may change in any minor SDK release. Pin your SDK version when
           relying on preview features.

        Args:
            index_name: Name of the index to back up.
            name: Optional user-defined name for the backup.
            description: Optional description providing context for the backup.

        Returns:
            :class:`~pinecone.models.backups.model.BackupModel`
            describing the newly created backup. The ``status`` field will
            typically be ``"Initializing"`` immediately after creation. Poll
            ``describe_backup()`` until ``status == "Ready"`` before using the
            backup::

                backup = await pc.preview.indexes.create_backup("my-index", name="nightly")
                # Poll until ready
                import asyncio
                while backup.status != "Ready":
                    await asyncio.sleep(5)
                    backup = await pc.preview.indexes.describe_backup(backup.backup_id)

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`: If
                *index_name* is empty.
            :exc:`~pinecone.errors.exceptions.ApiError`: If the API returns an
                error response.

        Examples:
            .. code-block:: python

                await pc.preview.indexes.create_backup("my-index")

                await pc.preview.indexes.create_backup(
                    "my-index", name="nightly", description="Daily backup"
                )
        """
        require_non_empty("index_name", index_name)

        body: dict[str, str] = {}
        if name is not None:
            body["name"] = name
        if description is not None:
            body["description"] = description
        content = msgspec.json.encode(body)

        logger.info("Creating backup for preview index index_name=%r", index_name)
        response = await self._http.post(
            f"/indexes/{index_name}/backups",
            content=content,
            headers={"Content-Type": "application/json"},
        )
        return PreviewDescribeBackupAdapter.from_response(response.content)

    def list_backups(
        self,
        index_name: str,
        *,
        limit: int | None = None,
        pagination_token: str | None = None,
    ) -> AsyncPaginator[BackupModel]:
        """List backups for a preview index.

        .. admonition:: Preview
           :class: warning

           Uses Pinecone API version ``2026-01.alpha``.
           Preview surface is not covered by SemVer — signatures and behavior
           may change in any minor SDK release. Pin your SDK version when
           relying on preview features.

        Args:
            index_name: Name of the index whose backups to list.
            limit: Maximum number of backups to yield across all pages. Must be
                a positive integer. ``None`` yields all backups. It also sets
                the requested page size, but only on a request that carries no
                pagination token: every later page is sized by the token,
                which already encodes it.
            pagination_token: Token to resume pagination from a previous call.
                ``None`` starts from the beginning. *limit* still caps the
                total yield, but it is not sent alongside a token — see above.

        Returns:
            :class:`~pinecone.models.pagination.AsyncPaginator` over
            :class:`~pinecone.models.backups.model.BackupModel`
            instances.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`: If
                *index_name* is empty or *limit* is zero or negative.
            :exc:`~pinecone.errors.exceptions.ApiError`: If the API returns an
                error response.

        Examples:
            .. code-block:: python

                async for backup in pc.preview.indexes.list_backups("my-index"):
                    print(backup.backup_id, backup.status)

            Access page-level metadata::

                paginator = pc.preview.indexes.list_backups("my-index")
                async for page in paginator.pages():
                    print(f"Got {len(page.items)} backups")
                    for backup in page.items:
                        print(backup.backup_id)
        """
        require_non_empty("index_name", index_name)
        if limit is not None:
            require_positive("limit", limit)

        async def fetch_page(token: str | None) -> Page[BackupModel]:
            params = backup_list_params(limit=limit, pagination_token=token)
            response = await self._http.get(f"/indexes/{index_name}/backups", params=params)
            items, next_token = PreviewListBackupsAdapter.from_response(response.content)
            return Page(items=items, pagination_token=next_token)

        return AsyncPaginator(fetch_page=fetch_page, initial_token=pagination_token, limit=limit)

    async def describe_backup(self, backup_id: str) -> BackupModel:
        """Describe a backup by its ID.

        .. admonition:: Preview
           :class: warning

           Uses Pinecone API version ``2026-01.alpha``.
           Preview surface is not covered by SemVer — signatures and behavior
           may change in any minor SDK release. Pin your SDK version when
           relying on preview features.

        Args:
            backup_id: The unique identifier of the backup to describe.

        Returns:
            :class:`~pinecone.models.backups.model.BackupModel`
            with the current state of the backup.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`: If
                *backup_id* is empty.
            :exc:`~pinecone.errors.exceptions.ApiError`: If the API returns an
                error response.

        Examples:
            .. code-block:: python

                backup = await pc.preview.indexes.create_backup("my-index", name="nightly")
                # Poll until ready
                import asyncio
                while backup.status != "Ready":
                    await asyncio.sleep(5)
                    backup = await pc.preview.indexes.describe_backup(backup.backup_id)
        """
        require_non_empty("backup_id", backup_id)
        logger.info("Describing backup backup_id=%r", backup_id)
        response = await self._http.get(f"/backups/{backup_id}")
        return PreviewDescribeBackupAdapter.from_response(response.content)
