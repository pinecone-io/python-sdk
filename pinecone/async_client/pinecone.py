"""Asynchronous Pinecone client."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from pinecone._internal.adaptive import _AdaptiveLimiterRegistry
from pinecone._internal.config import PineconeConfig, RetryConfig
from pinecone._internal.constants import CONTROL_PLANE_API_VERSION, DEFAULT_BASE_URL
from pinecone._internal.indexes_helpers import IndexKwargs, async_poll_index_until_ready
from pinecone._internal.keyword_only import keyword_only_methods
from pinecone._internal.validation import require_non_empty
from pinecone.errors.exceptions import ValidationError

if TYPE_CHECKING:
    from pinecone.async_client.assistants import AsyncAssistants
    from pinecone.async_client.async_index import AsyncIndex
    from pinecone.async_client.backup_schedules import AsyncBackupSchedules
    from pinecone.async_client.backups import AsyncBackups
    from pinecone.async_client.collections import AsyncCollections
    from pinecone.async_client.indexes import AsyncIndexes
    from pinecone.async_client.inference import AsyncInference
    from pinecone.async_client.restore_jobs import AsyncRestoreJobs
    from pinecone.client._assistant_namespace_proxy import _AsyncAssistantNamespaceProxy
    from pinecone.inference.models.index_embed import IndexEmbed
    from pinecone.models.backups.list import BackupList, RestoreJobList
    from pinecone.models.backups.model import (
        BackupModel,
        CreateIndexFromBackupResponse,
        RestoreJobModel,
    )
    from pinecone.models.collections.list import CollectionList
    from pinecone.models.collections.model import CollectionModel
    from pinecone.models.enums import (
        AwsRegion,
        AzureRegion,
        CloudProvider,
        DeletionProtection,
        GcpRegion,
    )
    from pinecone.models.indexes.index import IndexModel
    from pinecone.models.indexes.specs import EmbedConfig
    from pinecone.models.pagination import AsyncPaginator


@keyword_only_methods
class AsyncPinecone:
    """Asynchronous Pinecone client for control-plane operations.

    Args:
        api_key (str | None): Pinecone API key. Falls back to ``PINECONE_API_KEY`` env var.
        host (str | None): Control-plane API host. Falls back to ``PINECONE_CONTROLLER_HOST``
            env var, then defaults to ``https://api.pinecone.io``.
        additional_headers (Mapping[str, str] | None): Extra headers included in every request.
        source_tag (str | None): Tag appended to the User-Agent string for request attribution.
        proxy_url (str | None): HTTP proxy URL for outgoing requests.
        proxy_headers (Mapping[str, str] | None): Not yet supported. Raises
            ``NotImplementedError`` if provided.
        ssl_ca_certs (str | None): Path to a CA certificate bundle for SSL verification.
        ssl_verify (bool): Whether to verify SSL certificates. Defaults to ``True``.
        timeout (float): Request timeout in seconds. Defaults to ``30.0``.
        connection_pool_maxsize (int): Maximum number of connections to keep in the
            pool. ``0`` (default) uses httpx defaults.
        retry_config (RetryConfig | None): Custom retry configuration. When ``None``
            (default), uses built-in defaults (5 attempts, exponential backoff, retries
            on 500/502/503/504 for GET/HEAD).

    Raises:
        :exc:`PineconeValueError`: If no API key can be resolved from arguments or
            environment variables.
        :exc:`FileNotFoundError`: If ``ssl_ca_certs`` names a path that does not
            exist, so a mistyped path cannot leave you silently verifying against
            the default trust store instead. The connection pool is built lazily,
            so this is raised on the first request rather than at construction. A
            bundle that exists but cannot be parsed as a certificate raises
            :exc:`ssl.SSLError` at the same point.

    Examples:

        .. code-block:: python

            from pinecone import AsyncPinecone

            async with AsyncPinecone(api_key="your-api-key") as pc:
                index = await pc.index(name="my-index")
                async with index:
                    results = await index.query(
                        vector=[0.012, -0.087, 0.153, ...],  # 1536-dim embedding
                        top_k=10,
                    )

    .. note:: **Differences from sync Pinecone**

        1. **index() is a coroutine.** Unlike the sync ``Pinecone`` client,
           ``AsyncPinecone.index()`` must be awaited: ``idx = await pc.index(name="my-index")``.
           On cache miss it performs a non-blocking describe call to resolve
           the host — no manual two-step dance needed.

        2. **upsert_from_dataframe() is not supported.** ``AsyncIndex``
           raises ``NotImplementedError`` for this method. Use batched
           ``upsert()`` calls instead.

        3. **No grpc parameter on index().** Async gRPC transport is not
           yet available, so the ``grpc`` option accepted by the sync
           client is absent here.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        host: str | None = None,
        additional_headers: Mapping[str, str] | None = None,
        source_tag: str | None = None,
        proxy_url: str | None = None,
        proxy_headers: Mapping[str, str] | None = None,
        ssl_ca_certs: str | None = None,
        ssl_verify: bool = True,
        timeout: float = 30.0,
        connection_pool_maxsize: int = 0,
        retry_config: RetryConfig | None = None,
    ) -> None:
        if proxy_headers:
            raise NotImplementedError("proxy_headers is not yet supported for the async client")

        self._limiter_registry = _AdaptiveLimiterRegistry()
        augmented_retry_config = replace(
            retry_config or RetryConfig(),
            on_throttle=self._limiter_registry.report_throttled,
        )
        config = PineconeConfig(
            api_key=api_key or "",
            host=host or "",
            timeout=timeout,
            additional_headers=dict(additional_headers or {}),
            source_tag=source_tag or "",
            proxy_url=proxy_url or "",
            ssl_ca_certs=ssl_ca_certs,
            ssl_verify=ssl_verify,
            connection_pool_maxsize=connection_pool_maxsize,
            retry_config=augmented_retry_config,
        )

        if not config.api_key:
            raise ValidationError(
                "No API key provided. Pass api_key='...' or set the "
                "PINECONE_API_KEY environment variable."
            )

        # Apply default host if none resolved
        resolved_host = config.host or DEFAULT_BASE_URL
        if resolved_host != config.host:
            config = replace(config, host=resolved_host)

        self._config = config

        from pinecone._internal.http_client import AsyncHTTPClient

        self._http = AsyncHTTPClient(config, CONTROL_PLANE_API_VERSION)
        self._indexes: AsyncIndexes | None = None
        self._collections: AsyncCollections | None = None
        self._assistants: AsyncAssistants | None = None
        self._backups: AsyncBackups | None = None
        self._backup_schedules: AsyncBackupSchedules | None = None
        self._restore_jobs: AsyncRestoreJobs | None = None
        self._inference: AsyncInference | None = None
        self._host_cache: dict[str, str] = {}

    def __repr__(self) -> str:
        masked = f"...{self._config.api_key[-4:]}" if len(self._config.api_key) >= 4 else "***"
        return f"AsyncPinecone(api_key='{masked}', host='{self._config.host}')"

    @property
    def indexes(self) -> AsyncIndexes:
        """Access the AsyncIndexes namespace for control-plane index operations.

        Lazily imported and instantiated on first access.

        Returns:
            :class:`AsyncIndexes` namespace instance.

        Examples:

            .. code-block:: python

                async with AsyncPinecone(api_key="your-api-key") as pc:
                    async for idx in pc.indexes.list():
                        print(idx.name)
        """
        if self._indexes is None:
            from pinecone.async_client.indexes import AsyncIndexes as _AsyncIndexes

            self._indexes = _AsyncIndexes(http=self._http, host_cache=self._host_cache)
        return self._indexes

    @property
    def collections(self) -> AsyncCollections:
        """Access the AsyncCollections namespace for control-plane collection operations.

        Lazily imported and instantiated on first access.

        Returns:
            :class:`AsyncCollections` namespace instance.

        Examples:

            .. code-block:: python

                async with AsyncPinecone(api_key="your-api-key") as pc:
                    for col in await pc.collections.list():
                        print(col.name)
        """
        if self._collections is None:
            from pinecone.async_client.collections import AsyncCollections as _AsyncCollections

            self._collections = _AsyncCollections(http=self._http)
        return self._collections

    @property
    def assistants(self) -> AsyncAssistants:
        """Access the AsyncAssistants namespace for assistant operations.

        Lazily imported and instantiated on first access.

        Returns:
            :class:`AsyncAssistants` namespace instance.

        Examples:

            .. code-block:: python

                async with AsyncPinecone(api_key="your-api-key") as pc:
                    assistants = await pc.assistants.list()
        """
        if self._assistants is None:
            from pinecone.async_client.assistants import AsyncAssistants as _AsyncAssistants

            self._assistants = _AsyncAssistants(config=self._config)
        return self._assistants

    @property
    def assistant(self) -> _AsyncAssistantNamespaceProxy:
        """Convenience alias for :attr:`AsyncPinecone.assistants`.

        Returns a proxy that supports both namespace-style access
        (``pc.assistant.create_assistant(...)``) and the convenience call
        form (``await pc.assistant("my-name")`` — shortcut for
        ``await pc.assistants.describe(name="my-name")``).

        The canonical entry point is :attr:`AsyncPinecone.assistants`;
        this alias is provided for ergonomic singular-form access and is
        not deprecated.
        """
        from pinecone.client._assistant_namespace_proxy import _AsyncAssistantNamespaceProxy

        return _AsyncAssistantNamespaceProxy(self.assistants)

    @property
    def backups(self) -> AsyncBackups:
        """Access the AsyncBackups namespace for control-plane backup operations.

        Lazily imported and instantiated on first access.

        Returns:
            :class:`AsyncBackups` namespace instance.

        Examples:

            .. code-block:: python

                async with AsyncPinecone(api_key="your-api-key") as pc:
                    for backup in await pc.backups.list():
                        print(backup.backup_id)
        """
        if self._backups is None:
            from pinecone.async_client.backups import AsyncBackups as _AsyncBackups

            self._backups = _AsyncBackups(http=self._http)
        return self._backups

    @property
    def backup_schedules(self) -> AsyncBackupSchedules:
        """Access the AsyncBackupSchedules namespace for automatic, recurring backups.

        Lazily imported and instantiated on first access.

        Returns:
            :class:`AsyncBackupSchedules` namespace instance.

        Examples:

            .. code-block:: python

                async with AsyncPinecone(api_key="your-api-key") as pc:
                    schedules = await pc.backup_schedules.list(index_name="my-index")
        """
        if self._backup_schedules is None:
            from pinecone.async_client.backup_schedules import (
                AsyncBackupSchedules as _AsyncBackupSchedules,
            )

            self._backup_schedules = _AsyncBackupSchedules(http=self._http)
        return self._backup_schedules

    @property
    def restore_jobs(self) -> AsyncRestoreJobs:
        """Access the AsyncRestoreJobs namespace for restore job operations.

        Lazily imported and instantiated on first access.

        Returns:
            :class:`AsyncRestoreJobs` namespace instance.

        Examples:

            .. code-block:: python

                async with AsyncPinecone(api_key="your-api-key") as pc:
                    for job in await pc.restore_jobs.list():
                        print(job.restore_job_id)
        """
        if self._restore_jobs is None:
            from pinecone.async_client.restore_jobs import AsyncRestoreJobs as _AsyncRestoreJobs

            self._restore_jobs = _AsyncRestoreJobs(http=self._http)
        return self._restore_jobs

    @property
    def inference(self) -> AsyncInference:
        """Access the AsyncInference namespace for embed and rerank operations.

        Lazily imported and instantiated on first access.

        Returns:
            :class:`AsyncInference` namespace instance.

        Examples:

            .. code-block:: python

                async with AsyncPinecone(api_key="your-api-key") as pc:
                    embeddings = await pc.inference.embed(
                        model="multilingual-e5-large",
                        inputs=["Hello, world!"],
                    )
        """
        if self._inference is None:
            from pinecone.async_client.inference import AsyncInference as _AsyncInference

            self._inference = _AsyncInference(config=self._config)
        return self._inference

    async def create_index_from_backup(
        self,
        *,
        name: str,
        backup_id: str,
        deletion_protection: DeletionProtection | str | None = None,
        tags: Mapping[str, str] | None = None,
        read_capacity: dict[str, Any] | None = None,
        timeout: int | None = None,
    ) -> CreateIndexFromBackupResponse | IndexModel:
        """Create a new index by restoring from a backup.

        Sends a POST to ``/backups/{backup_id}/create-index`` and then
        polls until the index is ready (unless *timeout* is ``-1``).

        This is the only supported way to restore a backup:
        :meth:`AsyncPinecone.create_index` rejects ``source_backup_id=`` with
        a message pointing here.

        .. versionchanged:: 10.0
           Added *read_capacity*, so a restore can land straight onto
           dedicated read nodes instead of defaulting to on-demand capacity.

        Args:
            name (str): Name for the new index.
            backup_id (str): Identifier of the backup to restore from.  Obtain this
                from :meth:`AsyncPinecone.backups.create` or :meth:`AsyncPinecone.backups.list`.
            deletion_protection (DeletionProtection | str | None): ``"enabled"`` or
                ``"disabled"``. Defaults to ``"disabled"`` server-side when omitted.
            tags (Mapping[str, str] | None): Optional key-value tags for the new index.
                When omitted, the server copies the backup's own tags.
            read_capacity (dict[str, Any] | None): Optional read capacity for the
                restored index — ``{"mode": "OnDemand"}`` or ``{"mode": "Dedicated",
                "dedicated": {"node_type": ..., "scaling": "Manual",
                "manual": {"shards": ..., "replicas": ...}}}``. Omitted entirely
                when ``None``, leaving the server's on-demand default in place.
                Serverless backups only; the server rejects a dedicated
                configuration too small for the backup.
            timeout (int | None): Seconds to wait for readiness. ``None`` (default)
                blocks up to 300 s. ``-1`` returns a :class:`CreateIndexFromBackupResponse`
                immediately (contains ``restore_job_id`` and ``index_id``) without polling.

        Returns:
            A :class:`CreateIndexFromBackupResponse` when *timeout* is ``-1`` (contains
            ``restore_job_id`` and ``index_id``), or an :class:`IndexModel` describing
            the restored index once it is ready.

        Raises:
            :exc:`PineconeValueError`: If *name* or *backup_id* is empty, or
                *read_capacity* is an empty dict.
            :exc:`PineconeTimeoutError`: If the index is not ready within the timeout.
            :exc:`IndexInitFailedError`: If the index enters ``InitializationFailed`` state.
            :exc:`IndexTerminatedError`: If the index enters ``Terminating`` or ``Disabled`` state.
            :exc:`ApiError`: If the API returns an error response — including 404
                for an unknown backup, 409 when an index of that name exists, and
                a failed precondition when the backup is not yet complete.

        Examples:

            .. code-block:: python

                # Restore an index from a backup
                from pinecone import AsyncPinecone
                async with AsyncPinecone(api_key="your-api-key") as pc:
                    index = await pc.create_index_from_backup(
                        name="product-search-restored",
                        backup_id="bk-daily-20240115",
                    )

            .. code-block:: python

                # Restore without waiting (returns restore_job_id)
                async with AsyncPinecone(api_key="your-api-key") as pc:
                    result = await pc.create_index_from_backup(
                        name="product-search-restored",
                        backup_id="bk-daily-20240115",
                        timeout=-1,
                    )
                    print(result.restore_job_id)

            .. code-block:: python

                # Restore directly onto dedicated read nodes
                async with AsyncPinecone(api_key="your-api-key") as pc:
                    index = await pc.create_index_from_backup(
                        name="restored-drn-index",
                        backup_id="bk-daily-20240115",
                        read_capacity={
                            "mode": "Dedicated",
                            "dedicated": {
                                "node_type": "t1",
                                "scaling": "Manual",
                                "manual": {"shards": 2, "replicas": 2},
                            },
                        },
                    )
        """
        require_non_empty("name", name)
        require_non_empty("backup_id", backup_id)
        if read_capacity is not None and not read_capacity:
            raise ValidationError("read_capacity cannot be an empty dict")

        dp_val: str | None = None
        if deletion_protection is not None:
            dp_val = (
                deletion_protection.value
                if hasattr(deletion_protection, "value")
                else deletion_protection
            )

        from pinecone._internal.adapters.backups_adapter import BackupsAdapter
        from pinecone.models.backups.model import CreateIndexFromBackupRequest

        request = CreateIndexFromBackupRequest(
            name=name,
            tags=dict(tags) if tags is not None else None,
            deletion_protection=dp_val,
            read_capacity=read_capacity,
        )
        response = await self._http.post(
            f"/backups/{quote(backup_id, safe='')}/create-index",
            content=BackupsAdapter.to_create_index_from_backup_request(request),
            headers={"Content-Type": "application/json"},
        )
        create_response = BackupsAdapter.to_create_index_from_backup_response(response.content)

        if timeout == -1:
            return create_response

        effective_timeout = timeout if timeout is not None else 300
        return await async_poll_index_until_ready(self.indexes.describe, name, effective_timeout)

    @property
    def config(self) -> PineconeConfig:
        """The resolved configuration for this client.

        Returns:
            :class:`~pinecone._internal.config.PineconeConfig` containing the
            resolved API key, host, timeout, and connection settings.
        """
        return self._config

    # ---- Backcompat flat-method delegates (:meta private:) ----

    async def create_index(
        self,
        *,
        schema: dict[str, Any] | None = None,
        name: str | None = None,
        deployment: dict[str, Any] | None = None,
        read_capacity: dict[str, Any] | None = None,
        deletion_protection: DeletionProtection | str | None = None,
        tags: Mapping[str, str] | None = None,
        cmek_id: str | None = None,
        timeout: int | None = None,
        **legacy_kwargs: Any,
    ) -> IndexModel:
        """Backwards-compatibility shim for :meth:`AsyncPinecone.indexes.create`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New code
        should use ``await pc.indexes.create(...)`` instead of
        ``await pc.create_index(...)``.

        .. versionchanged:: 10.0
           Mirrors the 2026-07 schema-based signature of
           :meth:`AsyncPinecone.indexes.create`. Legacy 2025-10 keyword
           arguments (``spec``, ``dimension``, ``metric``, ``vector_type``,
           ...) raise a :exc:`~pinecone.errors.exceptions.PineconeTypeError`
           whose message shows the equivalent 2026-07 call.

        :meta private:
        """
        return await self.indexes.create(
            schema=schema,
            name=name,
            deployment=deployment,
            read_capacity=read_capacity,
            deletion_protection=deletion_protection,
            tags=tags,
            cmek_id=cmek_id,
            timeout=timeout,
            **legacy_kwargs,
        )

    async def create_index_for_model(
        self,
        name: str,
        cloud: CloudProvider | str,
        region: AwsRegion | GcpRegion | AzureRegion | str,
        embed: IndexEmbed | EmbedConfig | dict[str, Any],
        tags: Mapping[str, str] | None = None,
        deletion_protection: DeletionProtection | str | None = None,
        read_capacity: dict[str, Any] | None = None,
        schema: dict[str, Any] | None = None,
        timeout: int | None = None,
    ) -> IndexModel:
        """Backwards-compatibility shim for :meth:`AsyncPinecone.indexes.create_for_model`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New
        code should use ``await pc.indexes.create_for_model(...)`` instead of
        ``await pc.create_index_for_model(...)``.

        :meta private:
        """
        deletion_protection_str: str | None = None
        if deletion_protection is not None:
            resolved = (
                deletion_protection.value
                if hasattr(deletion_protection, "value")
                else deletion_protection
            )
            deletion_protection_str = None if resolved == "disabled" else resolved
        return await self.indexes.create_for_model(
            name=name,
            cloud=cloud.value if hasattr(cloud, "value") else str(cloud),
            region=region.value if hasattr(region, "value") else str(region),
            embed=embed,
            deletion_protection=deletion_protection_str,
            tags=tags,
            schema=schema,
            read_capacity=read_capacity,
            timeout=timeout,
        )

    async def describe_index(self, name: str) -> IndexModel:
        """Backwards-compatibility shim for :meth:`AsyncPinecone.indexes.describe`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New code
        should use ``await pc.indexes.describe(...)`` instead of
        ``await pc.describe_index(...)``.

        :meta private:
        """
        return await self.indexes.describe(name)

    def list_indexes(self) -> AsyncPaginator[IndexModel]:
        """Backwards-compatibility shim for :meth:`AsyncPinecone.indexes.list`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New
        code should use ``pc.indexes.list()`` instead of ``pc.list_indexes()``.

        .. versionchanged:: 10.0
           Returns an :class:`~pinecone.models.pagination.AsyncPaginator`
           instead of an ``IndexList`` and is no longer a coroutine; replace
           ``(await pc.list_indexes()).names()`` with
           ``[idx.name async for idx in pc.list_indexes()]``.

        :meta private:
        """
        return self.indexes.list()

    async def has_index(self, name: str) -> bool:
        """Backwards-compatibility shim for :meth:`AsyncPinecone.indexes.exists`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New code
        should use ``await pc.indexes.exists(...)`` instead of
        ``await pc.has_index(...)``.

        :meta private:
        """
        return await self.indexes.exists(name)

    async def configure_index(
        self,
        name: str,
        *,
        deployment: dict[str, Any] | None = None,
        schema: dict[str, Any] | None = None,
        read_capacity: dict[str, Any] | None = None,
        deletion_protection: DeletionProtection | str | None = None,
        tags: Mapping[str, str] | None = None,
        **legacy_kwargs: Any,
    ) -> IndexModel:
        """Backwards-compatibility shim for :meth:`AsyncPinecone.indexes.configure`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New code
        should use ``await pc.indexes.configure(...)`` instead of
        ``await pc.configure_index(...)``.

        .. versionchanged:: 10.0
           Mirrors the 2026-07 signature of
           :meth:`AsyncPinecone.indexes.configure` (pod scaling nests under
           ``deployment=``; ``embed=`` and ``serverless_read_capacity=`` were
           removed) and returns the updated :class:`IndexModel` instead of
           ``None``. Legacy keyword arguments raise a
           :exc:`~pinecone.errors.exceptions.PineconeTypeError` whose message
           shows the equivalent 2026-07 call.

        :meta private:
        """
        return await self.indexes.configure(
            name,
            deployment=deployment,
            schema=schema,
            read_capacity=read_capacity,
            deletion_protection=deletion_protection,
            tags=tags,
            **legacy_kwargs,
        )

    async def delete_index(self, name: str, timeout: int | None = None) -> None:
        """Backwards-compatibility shim for :meth:`AsyncPinecone.indexes.delete`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New code
        should use ``await pc.indexes.delete(...)`` instead of
        ``await pc.delete_index(...)``.

        :meta private:
        """
        await self.indexes.delete(name, timeout=timeout)

    async def create_collection(self, name: str, source: str) -> CollectionModel:
        """Backwards-compatibility shim for :meth:`AsyncPinecone.collections.create`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New code
        should use ``await pc.collections.create(...)`` instead of
        ``await pc.create_collection(...)``.

        :meta private:
        """
        return await self.collections.create(name=name, source=source)

    async def list_collections(self) -> CollectionList:
        """Backwards-compatibility shim for :meth:`AsyncPinecone.collections.list`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New code
        should use ``await pc.collections.list()`` instead of
        ``await pc.list_collections()``.

        :meta private:
        """
        return await self.collections.list()

    async def describe_collection(self, name: str) -> CollectionModel:
        """Backwards-compatibility shim for :meth:`AsyncPinecone.collections.describe`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New code
        should use ``await pc.collections.describe(...)`` instead of
        ``await pc.describe_collection(...)``.

        :meta private:
        """
        return await self.collections.describe(name)

    async def delete_collection(self, name: str) -> None:
        """Backwards-compatibility shim for :meth:`AsyncPinecone.collections.delete`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New code
        should use ``await pc.collections.delete(...)`` instead of
        ``await pc.delete_collection(...)``.

        :meta private:
        """
        await self.collections.delete(name)

    async def create_backup(
        self,
        *,
        index_name: str,
        backup_name: str | None = None,
        description: str | None = None,
    ) -> BackupModel:
        """Backwards-compatibility shim for :meth:`AsyncPinecone.backups.create`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New code
        should use ``await pc.backups.create(...)`` instead of
        ``await pc.create_backup(...)``.

        :meta private:
        """
        return await self.backups.create(
            index_name=index_name,
            name=backup_name,
            description=description,
        )

    async def list_backups(
        self,
        *,
        index_name: str | None = None,
        limit: int | None = None,
        pagination_token: str | None = None,
        include_deleted: bool | None = None,
    ) -> BackupList:
        """Backwards-compatibility shim for :meth:`AsyncPinecone.backups.list`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New code
        should use ``await pc.backups.list(...)`` instead of
        ``await pc.list_backups(...)``.

        :meta private:
        """
        return await self.backups.list(
            index_name=index_name,
            limit=limit,
            pagination_token=pagination_token,
            include_deleted=include_deleted,
        )

    async def describe_backup(self, *, backup_id: str) -> BackupModel:
        """Backwards-compatibility shim for :meth:`AsyncPinecone.backups.describe`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New code
        should use ``await pc.backups.describe(...)`` instead of
        ``await pc.describe_backup(...)``.

        :meta private:
        """
        return await self.backups.describe(backup_id=backup_id)

    async def delete_backup(self, *, backup_id: str) -> None:
        """Backwards-compatibility shim for :meth:`AsyncPinecone.backups.delete`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New code
        should use ``await pc.backups.delete(...)`` instead of
        ``await pc.delete_backup(...)``.

        :meta private:
        """
        await self.backups.delete(backup_id=backup_id)

    async def list_restore_jobs(
        self,
        *,
        limit: int | None = None,
        pagination_token: str | None = None,
    ) -> RestoreJobList:
        """Backwards-compatibility shim for :meth:`AsyncPinecone.restore_jobs.list`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New code
        should use ``await pc.restore_jobs.list(...)`` instead of
        ``await pc.list_restore_jobs(...)``.

        :meta private:
        """
        return await self.restore_jobs.list(
            limit=limit,
            pagination_token=pagination_token,
        )

    async def describe_restore_job(self, *, job_id: str) -> RestoreJobModel:
        """Backwards-compatibility shim for :meth:`AsyncPinecone.restore_jobs.describe`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New code
        should use ``await pc.restore_jobs.describe(...)`` instead of
        ``await pc.describe_restore_job(...)``.

        :meta private:
        """
        return await self.restore_jobs.describe(job_id=job_id)

    def IndexAsyncio(self, host: str, **kwargs: Any) -> AsyncIndex:  # noqa: N802
        """Backwards-compatibility shim for :meth:`AsyncPinecone.index`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New
        code should use ``pc.index(host=...)`` (where ``pc`` is an
        :class:`AsyncPinecone` instance) instead of ``pc.IndexAsyncio(...)``.

        :meta private:
        """
        from pinecone.async_client.async_index import AsyncIndex as _AsyncIndex

        return _AsyncIndex(
            host=host,
            api_key=self._config.api_key,
            additional_headers=dict(self._config.additional_headers),
            timeout=self._config.timeout,
            proxy_url=self._config.proxy_url,
            proxy_headers=dict(self._config.proxy_headers),
            ssl_ca_certs=self._config.ssl_ca_certs,
            ssl_verify=self._config.ssl_verify,
            source_tag=self._config.source_tag,
            connection_pool_maxsize=self._config.connection_pool_maxsize,
            _limiter_registry=self._limiter_registry,
        )

    def _build_index_kwargs(self, host: str) -> IndexKwargs:
        """Return the kwargs dict for constructing an AsyncIndex."""
        return IndexKwargs(
            host=host,
            api_key=self._config.api_key,
            additional_headers=dict(self._config.additional_headers),
            timeout=self._config.timeout,
            proxy_url=self._config.proxy_url,
            proxy_headers=dict(self._config.proxy_headers),
            ssl_ca_certs=self._config.ssl_ca_certs,
            ssl_verify=self._config.ssl_verify,
            source_tag=self._config.source_tag,
            connection_pool_maxsize=self._config.connection_pool_maxsize,
        )

    async def _resolve_index_host(self, *, name: str, host: str) -> str:
        """Resolve the data plane host from explicit host, cache, or describe call.

        Async parallel of ``Pinecone._resolve_index_host``. Performs a
        non-blocking describe-index lookup when *name* is given and the host is
        not yet cached.

        Args:
            name: Index name (triggers describe if not cached).
            host: Direct host URL (returned as-is if provided).

        Returns:
            The resolved host string.

        Raises:
            ValidationError: If neither *name* nor *host* is provided.
        """
        if host:
            return host

        if name:
            cached_host = self._host_cache.get(name)
            if cached_host:
                return cached_host

            desc = await self.indexes.describe(name)
            if desc.host is None:
                raise ValidationError(
                    f"Index {name!r} does not yet have a host assigned — "
                    "the index may still be initializing. "
                    "Wait until the index status is 'Ready' before connecting."
                )
            self._host_cache[name] = desc.host
            return desc.host

        raise ValidationError("Either name or host must be provided to create an Index client.")

    async def index(
        self,
        name: str = "",
        *,
        host: str = "",
    ) -> AsyncIndex:
        """Create an async data plane client targeting a specific index.

        Can target by host URL directly (skips the describe call) or by
        index name (triggers an async describe-index lookup to resolve the host
        on cache miss).

        .. seealso::
           Use ``pc.indexes`` for control-plane operations (create, list,
           describe, delete, configure).

        Args:
            name (str): Name of the index. Triggers an async describe call to
                resolve host on cache miss.
            host (str): Direct host URL of the index. Skips the describe call.

        Returns:
            An async :class:`AsyncIndex` data plane client.

        Raises:
            :exc:`ValidationError`: If neither *name* nor *host* is provided.
            :exc:`NotFoundError`: If *name* is given but no such index exists.

        Examples:

            .. code-block:: python

                async with AsyncPinecone(api_key="...") as pc:
                    idx = await pc.index(host="my-index-abc123.svc.pinecone.io")
                    # or
                    idx = await pc.index(name="my-index")  # triggers describe on cache miss

        .. warning::
           The returned :class:`AsyncIndex` manages its own HTTP client.
           Always use ``async with index:`` or call ``await index.close()``
           when done — closing the parent ``AsyncPinecone`` does not close
           index clients.
        """
        from pinecone.async_client.async_index import AsyncIndex as _AsyncIndex

        resolved_host = await self._resolve_index_host(name=name, host=host)
        return _AsyncIndex(
            **self._build_index_kwargs(resolved_host),
            _limiter_registry=self._limiter_registry,
        )

    async def close(self) -> None:
        """Close all open HTTP connections.

        Closes the main control-plane client and any namespace clients (inference,
        assistants) that were initialized during this session.

        Prefer the async context manager form (``async with AsyncPinecone(...) as pc:``)
        which calls :meth:`close` automatically on exit.

        Examples:
            Close the client explicitly after use:

            >>> import asyncio
            >>> from pinecone import AsyncPinecone
            >>> async def example():
            ...     client = AsyncPinecone(api_key="your-api-key")
            ...     await client.close()
            >>> asyncio.run(example())

            Use AsyncPinecone as a context manager (``close`` is called automatically):

            >>> async def example():
            ...     async with AsyncPinecone(api_key="your-api-key") as pc:
            ...         _ = await pc.indexes.list().to_list()
            >>> asyncio.run(example())
        """
        await self._http.close()
        if self._assistants is not None:
            await self._assistants.close()
        if self._inference is not None:
            await self._inference.close()

    async def __aenter__(self) -> AsyncPinecone:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()
