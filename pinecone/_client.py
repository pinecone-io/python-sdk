"""Synchronous Pinecone client."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

from pinecone._internal.adaptive import _AdaptiveLimiterRegistry
from pinecone._internal.config import PineconeConfig, RetryConfig
from pinecone._internal.constants import CONTROL_PLANE_API_VERSION, DEFAULT_BASE_URL
from pinecone._internal.indexes_helpers import _LegacyIndexKwargs, poll_index_until_ready
from pinecone._internal.validation import require_non_empty
from pinecone.errors.exceptions import ValidationError

if TYPE_CHECKING:
    from pinecone.client._assistant_namespace_proxy import _AssistantNamespaceProxy
    from pinecone.client.assistants import Assistants
    from pinecone.client.backups import Backups
    from pinecone.client.collections import Collections
    from pinecone.client.indexes import Indexes
    from pinecone.client.inference import Inference
    from pinecone.client.restore_jobs import RestoreJobs
    from pinecone.grpc import GrpcIndex
    from pinecone.index import Index
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
        Metric,
        VectorType,
    )
    from pinecone.models.indexes.index import IndexModel
    from pinecone.models.indexes.list import IndexList
    from pinecone.models.indexes.specs import (
        ByocSpec,
        EmbedConfig,
        IntegratedSpec,
        PodSpec,
        ServerlessSpec,
    )
    from pinecone.preview import Preview


class Pinecone:
    """Synchronous Pinecone client for control-plane operations.

    Args:
        api_key (str | None): Pinecone API key. Falls back to ``PINECONE_API_KEY`` env var.
        host (str | None): Control-plane API host. Falls back to ``PINECONE_CONTROLLER_HOST``
            env var, then defaults to ``https://api.pinecone.io``.
        additional_headers (Mapping[str, str] | None): Extra headers included in every request.
        source_tag (str | None): Tag appended to the User-Agent string for request attribution.
        proxy_url (str | None): HTTP proxy URL for outgoing requests.
        proxy_headers (Mapping[str, str] | None): Custom headers for proxy authentication.
        ssl_ca_certs (str | None): Path to a CA certificate bundle for SSL verification.
        ssl_verify (bool): Whether to verify SSL certificates. Defaults to ``True``.
        timeout (float): Request timeout in seconds. Defaults to ``30.0``.
        connection_pool_maxsize (int): Maximum number of connections to keep in the
            pool. ``0`` (default) uses httpx defaults.
        retry_config (RetryConfig | None): Custom retry configuration. When ``None``
            (default), uses built-in defaults (5 attempts, exponential backoff, retries
            on 500/502/503/504 for GET/HEAD).
        pool_threads (int | None): Opt-in for the legacy ``async_req=True`` execution
            model on data-plane methods. When set, indexes created via
            :meth:`index` accept ``async_req=True`` on ``upsert``, ``query``,
            ``describe_index_stats``, and ``list_paginated``. **For new code, prefer**
            :class:`~pinecone.async_client.AsyncPinecone` **or**
            :class:`concurrent.futures.ThreadPoolExecutor`. This kwarg exists for
            backcompat with pre-rewrite callers.

    Raises:
        :exc:`PineconeValueError`: If no API key can be resolved from arguments or
            environment variables.

    Examples:

        .. code-block:: python

            from pinecone import Pinecone

            pc = Pinecone(api_key="your-api-key")  # or set PINECONE_API_KEY env var

            # Control plane: manage indexes
            indexes = pc.indexes.list()

            # Data plane: operate on vectors
            index = pc.index("my-index")
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
        **kwargs: Any,
    ) -> None:
        legacy_pool_threads = kwargs.pop("pool_threads", None)
        if kwargs:
            raise TypeError(f"Pinecone() got unexpected keyword arguments: {sorted(kwargs)!r}")
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
            proxy_headers=dict(proxy_headers or {}),
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

        from pinecone._internal.http_client import HTTPClient

        self._http = HTTPClient(config, CONTROL_PLANE_API_VERSION)
        self._indexes: Indexes | None = None
        self._collections: Collections | None = None
        self._backups: Backups | None = None
        self._restore_jobs: RestoreJobs | None = None
        self._inference: Inference | None = None
        self._assistants: Assistants | None = None
        self._host_cache: dict[str, str] = {}
        self._preview: Preview | None = None
        self._legacy_pool_threads: int | None = legacy_pool_threads

    def __repr__(self) -> str:
        masked = f"...{self._config.api_key[-4:]}" if len(self._config.api_key) >= 4 else "***"
        return f"Pinecone(api_key='{masked}', host='{self._config.host}')"

    @property
    def indexes(self) -> Indexes:
        """Access the Indexes namespace for control-plane index operations.

        Lazily imported and instantiated on first access.

        Returns:
            :class:`Indexes` namespace instance.

        Examples:

            >>> names = [idx.name for idx in pc.indexes.list()]  # doctest: +SKIP
        """
        if self._indexes is None:
            from pinecone.client.indexes import Indexes as _Indexes

            self._indexes = _Indexes(http=self._http, host_cache=self._host_cache)
        return self._indexes

    @property
    def collections(self) -> Collections:
        """Access the Collections namespace for control-plane collection operations.

        Lazily imported and instantiated on first access.

        Returns:
            :class:`Collections` namespace instance.

        Examples:

            >>> names = [col.name for col in pc.collections.list()]  # doctest: +SKIP
        """
        if self._collections is None:
            from pinecone.client.collections import Collections as _Collections

            self._collections = _Collections(http=self._http)
        return self._collections

    @property
    def backups(self) -> Backups:
        """Access the Backups namespace for control-plane backup operations.

        Lazily imported and instantiated on first access.

        Returns:
            :class:`Backups` namespace instance.

        Examples:

            >>> ids = [backup.backup_id for backup in pc.backups.list()]  # doctest: +SKIP
        """
        if self._backups is None:
            from pinecone.client.backups import Backups as _Backups

            self._backups = _Backups(http=self._http)
        return self._backups

    @property
    def restore_jobs(self) -> RestoreJobs:
        """Access the RestoreJobs namespace for restore job operations.

        Lazily imported and instantiated on first access.

        Returns:
            :class:`RestoreJobs` namespace instance.

        Examples:

            >>> ids = [job.restore_job_id for job in pc.restore_jobs.list()]  # doctest: +SKIP
        """
        if self._restore_jobs is None:
            from pinecone.client.restore_jobs import RestoreJobs as _RestoreJobs

            self._restore_jobs = _RestoreJobs(http=self._http)
        return self._restore_jobs

    @property
    def inference(self) -> Inference:
        """Access the Inference namespace for embed and rerank operations.

        Lazily imported and instantiated on first access.

        Returns:
            :class:`Inference` namespace instance.

        Examples:

            >>> embeddings = pc.inference.embed(  # doctest: +SKIP
            ...     model="multilingual-e5-large",
            ...     inputs=["Solar panels reduce energy costs and lower carbon emissions."],
            ... )
        """
        if self._inference is None:
            from pinecone.client.inference import Inference as _Inference

            self._inference = _Inference(config=self._config)
        return self._inference

    @property
    def assistants(self) -> Assistants:
        """Access the Assistants namespace for assistant operations.

        Lazily imported and instantiated on first access.

        Returns:
            :class:`Assistants` namespace instance.

        Examples:

            >>> names = [assistant.name for assistant in pc.assistants.list()]  # doctest: +SKIP
        """
        if self._assistants is None:
            from pinecone.client.assistants import Assistants as _Assistants

            self._assistants = _Assistants(config=self._config)
        return self._assistants

    @property
    def assistant(self) -> _AssistantNamespaceProxy:
        """Return a callable proxy alias for :attr:`Pinecone.assistants`.

        Returns a proxy that supports both namespace-style access
        (``pc.assistant.create_assistant(...)``) and the convenience call
        form (``pc.assistant("my-name")`` — shortcut for
        ``pc.assistants.describe(name="my-name")``).

        The canonical entry point is :attr:`Pinecone.assistants`; this
        alias is provided for ergonomic singular-form access and is not
        deprecated.

        Returns:
            Proxy supporting namespace-style access (``pc.assistant.create_assistant(...)``)
            and the shorthand call form (``pc.assistant("my-name")``).

        Examples:

            >>> bot = pc.assistant("acme-support-bot")  # doctest: +SKIP
            >>> pc.assistant.create_assistant(  # doctest: +SKIP
            ...     name="support-bot",
            ...     instructions="Help users with billing questions.",
            ... )
        """
        from pinecone.client._assistant_namespace_proxy import _AssistantNamespaceProxy

        return _AssistantNamespaceProxy(self.assistants)

    @property
    def preview(self) -> Preview:
        """Access the Preview namespace for pre-release API features.

        Lazily imported and instantiated on first access. Preview surface is
        not covered by SemVer — signatures and behavior may change in any
        minor SDK release.

        Returns:
            :class:`~pinecone.preview.Preview` namespace instance.

        Examples:

            >>> info = pc.preview.indexes.describe("articles-en-preview")  # doctest: +SKIP
        """
        if self._preview is None:
            from pinecone.preview import Preview as _Preview

            self._preview = _Preview(http=self._http, config=self._config)
        return self._preview

    def index(
        self,
        name: str = "",
        *,
        host: str = "",
        grpc: bool = False,
        pool_threads: int | None = None,
    ) -> Index | GrpcIndex:
        """Create a data plane client targeting a specific index.

        Can target by host URL directly (skips the describe call) or by
        index name (triggers a describe-index lookup to resolve the host).

        .. seealso::
           Use ``pc.indexes`` for control-plane operations (create, list,
           describe, delete, configure). To create an index from a backup,
           use :meth:`Pinecone.create_index_from_backup`.

        Args:
            name (str): Name of the index. Triggers a describe call to resolve host.
            host (str): Direct host URL of the index. Skips the describe call.
            grpc (bool): If ``True``, return a :class:`~pinecone.grpc.GrpcIndex`
                that routes data-plane operations over gRPC instead of HTTP.
                Defaults to ``False``.
            pool_threads (int | None): Maximum number of threads in the connection pool
                used by the underlying HTTP client. Pass ``None`` to use the client-level
                default set at :class:`Pinecone` construction time. Has no effect when
                ``grpc=True``. Defaults to ``None``.

        Returns:
            A sync :class:`Index` (HTTP) or :class:`~pinecone.grpc.GrpcIndex`
            (gRPC) data plane client.

        Raises:
            :exc:`PineconeValueError`: If neither ``name`` nor ``host`` is provided.
            :exc:`~pinecone.errors.NotFoundError`: If ``name`` is given but no index with that
                name exists.

        Examples:

            .. code-block:: python

                from pinecone import Pinecone

                pc = Pinecone(api_key="your-api-key")
                idx = pc.index(host="product-search-abc123.svc.pinecone.io")
                # or resolve the host by name
                idx = pc.index(name="product-search")
                # gRPC transport for high-throughput upserts
                idx = pc.index(name="product-search", grpc=True)
        """
        resolved_host = self._resolve_index_host(name=name, host=host)

        if grpc:
            from pinecone.grpc import GrpcIndex as _GrpcIndex

            return _GrpcIndex(
                host=resolved_host,
                api_key=self._config.api_key,
                source_tag=self._config.source_tag or None,
                on_throttle=self._limiter_registry.report_throttled,
            )

        from pinecone.index import Index as _Index

        return _Index(**self._build_index_kwargs(resolved_host, pool_threads=pool_threads))

    def _build_index_kwargs(
        self,
        host: str,
        *,
        pool_threads: int | None = None,
    ) -> _LegacyIndexKwargs:
        """Return the kwargs dict for constructing an Index."""
        kwargs: _LegacyIndexKwargs = _LegacyIndexKwargs(
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
        effective = pool_threads if pool_threads is not None else self._legacy_pool_threads
        if effective is not None:
            kwargs["pool_threads"] = effective
        return kwargs

    def _resolve_index_host(self, *, name: str, host: str) -> str:
        """Resolve the data plane host from explicit host, cache, or describe call.

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

            desc = self.indexes.describe(name)
            if desc.host is None:
                raise ValidationError(
                    f"Index {name!r} does not yet have a host assigned — "
                    "the index may still be initializing. "
                    "Wait until the index status is 'Ready' before connecting."
                )
            self._host_cache[name] = desc.host
            return desc.host

        raise ValidationError("Either name or host must be provided to create an Index client.")

    def create_index_from_backup(
        self,
        *,
        name: str,
        backup_id: str,
        deletion_protection: DeletionProtection | str | None = None,
        tags: Mapping[str, str] | None = None,
        timeout: int | None = None,
    ) -> CreateIndexFromBackupResponse | IndexModel:
        """Create a new index by restoring from a backup.

        Sends a POST to ``/backups/{backup_id}/create-index`` and then
        polls until the index is ready (unless *timeout* is ``-1``).

        Args:
            name (str): Name for the new index.
            backup_id (str): Identifier of the backup to restore from.  Obtain this
                from :meth:`Pinecone.backups.create` or :meth:`Pinecone.backups.list`.
            deletion_protection (DeletionProtection | str | None): ``"enabled"`` or
                ``"disabled"``. Defaults to ``"disabled"`` server-side when omitted.
            tags (Mapping[str, str] | None): Optional key-value tags for the new index.
            timeout (int | None): Seconds to wait for readiness. ``None`` (default)
                blocks up to 300 s. ``-1`` returns a :class:`CreateIndexFromBackupResponse`
                immediately (contains ``restore_job_id`` and ``index_id``) without polling.

        Returns:
            A :class:`CreateIndexFromBackupResponse` when *timeout* is ``-1`` (contains
            ``restore_job_id`` and ``index_id``), or an :class:`IndexModel` describing
            the restored index once it is ready.

        Raises:
            :exc:`PineconeValueError`: If *name* or *backup_id* is empty.
            :exc:`PineconeTimeoutError`: If the index is not ready within the timeout.
            :exc:`IndexInitFailedError`: If the index enters ``InitializationFailed`` state.
            :exc:`IndexTerminatedError`: If the index enters ``Terminating`` or ``Disabled`` state.
            :exc:`ApiError`: If the API returns an error response.

        Examples:
            >>> from pinecone import Pinecone
            >>> pc = Pinecone(api_key="your-api-key")
            >>> index = pc.create_index_from_backup(  # doctest: +SKIP
            ...     name="product-search-restored",
            ...     backup_id="bk-daily-20240115",
            ... )

            >>> result = pc.create_index_from_backup(  # doctest: +SKIP
            ...     name="product-search-restored",
            ...     backup_id="bk-daily-20240115",
            ...     timeout=-1,
            ... )
            >>> print(result.restore_job_id)  # doctest: +SKIP
        """
        require_non_empty("name", name)
        require_non_empty("backup_id", backup_id)

        body: dict[str, Any] = {"name": name}
        if deletion_protection is not None:
            dp_val = (
                deletion_protection.value
                if hasattr(deletion_protection, "value")
                else deletion_protection
            )
            body["deletion_protection"] = dp_val
        if tags is not None:
            body["tags"] = tags

        from pinecone._internal.adapters.backups_adapter import BackupsAdapter

        response = self._http.post(f"/backups/{backup_id}/create-index", json=body)
        create_response = BackupsAdapter.to_create_index_from_backup_response(response.content)

        if timeout == -1:
            return create_response

        effective_timeout = timeout if timeout is not None else 300
        return poll_index_until_ready(self.indexes.describe, name, effective_timeout)

    @property
    def config(self) -> PineconeConfig:
        """Return the resolved configuration for this client.

        Returns:
            :class:`~pinecone._internal.config.PineconeConfig` containing the
            resolved API key, host, timeout, and connection settings.

        Examples:

            >>> cfg = pc.config
            >>> cfg.timeout
            30.0
        """
        return self._config

    # ---- Backcompat flat-method delegates (:meta private:) ----

    def create_index(
        self,
        name: str,
        spec: ServerlessSpec | PodSpec | ByocSpec | IntegratedSpec | dict[str, Any],
        dimension: int | None = None,
        metric: Metric | str | None = "cosine",
        timeout: int | None = None,
        deletion_protection: DeletionProtection | str | None = "disabled",
        vector_type: VectorType | str = "dense",
        tags: Mapping[str, str] | None = None,
        schema: dict[str, Any] | None = None,
    ) -> IndexModel:
        """Backwards-compatibility shim for :meth:`Pinecone.indexes.create`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New code
        should use ``pc.indexes.create()`` instead of ``pc.create_index()``.

        :meta private:
        """
        resolved_dp = deletion_protection if deletion_protection is not None else "disabled"
        return self.indexes.create(
            name=name,
            spec=spec,
            dimension=dimension,
            metric=metric if metric is not None else "cosine",
            vector_type=vector_type,
            deletion_protection=resolved_dp,
            tags=tags,
            schema=schema,
            timeout=timeout,
        )

    def create_index_for_model(
        self,
        name: str,
        cloud: CloudProvider | str,
        region: AwsRegion | GcpRegion | AzureRegion | str,
        embed: IndexEmbed | EmbedConfig | dict[str, Any],
        tags: Mapping[str, str] | None = None,
        deletion_protection: DeletionProtection | str | None = "disabled",
        read_capacity: dict[str, Any] | None = None,
        schema: dict[str, Any] | None = None,
        timeout: int | None = None,
    ) -> IndexModel:
        """Backwards-compatibility shim for :meth:`Pinecone.indexes.create`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New
        code should use ``pc.indexes.create()`` with an ``IntegratedSpec``
        (``IntegratedSpec(cloud=..., region=..., embed=EmbedConfig(...))``)
        instead of ``pc.create_index_for_model()``.

        :meta private:
        """
        from pinecone.inference.models.index_embed import IndexEmbed as _IndexEmbed
        from pinecone.models.indexes.specs import EmbedConfig as _EmbedConfig
        from pinecone.models.indexes.specs import IntegratedSpec as _IntegratedSpec

        if isinstance(embed, _IndexEmbed):
            embed_config: EmbedConfig = _EmbedConfig(
                model=embed.model,
                field_map={k: str(v) for k, v in embed.field_map.items()},
                metric=embed.metric,
                read_parameters=embed.read_parameters or None,
                write_parameters=embed.write_parameters or None,
            )
        elif isinstance(embed, _EmbedConfig):
            embed_config = embed
        else:
            embed_config = _EmbedConfig(**embed)

        cloud_str = cloud.value if hasattr(cloud, "value") else str(cloud)
        region_str = region.value if hasattr(region, "value") else str(region)
        spec = _IntegratedSpec(cloud=cloud_str, region=region_str, embed=embed_config)
        resolved_dp = deletion_protection if deletion_protection is not None else "disabled"
        return self.indexes.create(
            name=name,
            spec=spec,
            tags=tags,
            deletion_protection=resolved_dp,
            schema=schema,
            read_capacity=read_capacity,
            timeout=timeout,
        )

    def describe_index(self, name: str) -> IndexModel:
        """Backwards-compatibility shim for :meth:`Pinecone.indexes.describe`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New code
        should use ``pc.indexes.describe()`` instead of ``pc.describe_index()``.

        :meta private:
        """
        return self.indexes.describe(name)

    def list_indexes(self) -> IndexList:
        """Backwards-compatibility shim for :meth:`Pinecone.indexes.list`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New
        code should use ``pc.indexes.list()`` instead of ``pc.list_indexes()``.

        :meta private:
        """
        return self.indexes.list()

    def has_index(self, name: str) -> bool:
        """Backwards-compatibility shim for :meth:`Pinecone.indexes.exists`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New code
        should use ``pc.indexes.exists()`` instead of ``pc.has_index()``.

        :meta private:
        """
        return self.indexes.exists(name)

    def configure_index(
        self,
        name: str,
        replicas: int | None = None,
        pod_type: str | None = None,
        deletion_protection: DeletionProtection | str | None = None,
        tags: Mapping[str, str] | None = None,
        embed: dict[str, Any] | None = None,
        read_capacity: dict[str, Any] | None = None,
        serverless_read_capacity: dict[str, Any] | None = None,
    ) -> None:
        """Backwards-compatibility shim for :meth:`Pinecone.indexes.configure`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New code
        should use ``pc.indexes.configure()`` instead of ``pc.configure_index()``.

        Args:
            name (str): Name of the index to configure.
            replicas (int | None): Number of replicas. Only applies to pod-based indexes.
                Defaults to ``None`` (no change).
            pod_type (str | None): Pod type (e.g. ``"p1.x1"``). Only applies to pod-based
                indexes. Defaults to ``None`` (no change).
            deletion_protection (DeletionProtection | str | None): Whether to enable deletion
                protection (``"enabled"`` or ``"disabled"``). Defaults to ``None`` (no change).
            tags (Mapping[str, str] | None): Key-value tags to apply to the index. Defaults to
                ``None`` (no change).
            embed (dict[str, Any] | None): Integrated inference embedding configuration.
                Defaults to ``None`` (no change).
            read_capacity (dict[str, Any] | None): Read capacity settings for the index.
                Defaults to ``None`` (no change).
            serverless_read_capacity (dict[str, Any] | None): Serverless read capacity settings.
                Defaults to ``None`` (no change).

        :meta private:
        """
        self.indexes.configure(
            name=name,
            replicas=replicas,
            pod_type=pod_type,
            deletion_protection=deletion_protection,
            tags=tags,
            embed=embed,
            read_capacity=read_capacity,
            serverless_read_capacity=serverless_read_capacity,
        )

    def delete_index(self, name: str, timeout: int | None = None) -> None:
        """Backwards-compatibility shim for :meth:`Pinecone.indexes.delete`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New code
        should use ``pc.indexes.delete()`` instead of ``pc.delete_index()``.

        :meta private:
        """
        self.indexes.delete(name, timeout=timeout)

    def create_collection(self, name: str, source: str) -> CollectionModel:
        """Backwards-compatibility shim for :meth:`Pinecone.collections.create`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New code
        should use ``pc.collections.create()`` instead of ``pc.create_collection()``.

        :meta private:
        """
        return self.collections.create(name=name, source=source)

    def list_collections(self) -> CollectionList:
        """Backwards-compatibility shim for :meth:`Pinecone.collections.list`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New code
        should use ``pc.collections.list()`` instead of ``pc.list_collections()``.

        :meta private:
        """
        return self.collections.list()

    def describe_collection(self, name: str) -> CollectionModel:
        """Backwards-compatibility shim for :meth:`Pinecone.collections.describe`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New code
        should use ``pc.collections.describe()`` instead of ``pc.describe_collection()``.

        :meta private:
        """
        return self.collections.describe(name)

    def delete_collection(self, name: str) -> None:
        """Backwards-compatibility shim for :meth:`Pinecone.collections.delete`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New code
        should use ``pc.collections.delete()`` instead of ``pc.delete_collection()``.

        :meta private:
        """
        self.collections.delete(name)

    def create_backup(
        self,
        *,
        index_name: str,
        backup_name: str | None = None,
        description: str | None = None,
    ) -> BackupModel:
        """Backwards-compatibility shim for :meth:`Pinecone.backups.create`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New code
        should use ``pc.backups.create()`` instead of ``pc.create_backup()``.

        :meta private:
        """
        return self.backups.create(
            index_name=index_name,
            name=backup_name,
            description=description,
        )

    def list_backups(
        self,
        *,
        index_name: str | None = None,
        limit: int | None = None,
        pagination_token: str | None = None,
    ) -> BackupList:
        """Backwards-compatibility shim for :meth:`Pinecone.backups.list`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New code
        should use ``pc.backups.list()`` instead of ``pc.list_backups()``.

        :meta private:
        """
        return self.backups.list(
            index_name=index_name,
            limit=limit,
            pagination_token=pagination_token,
        )

    def describe_backup(self, *, backup_id: str) -> BackupModel:
        """Backwards-compatibility shim for :meth:`Pinecone.backups.describe`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New code
        should use ``pc.backups.describe()`` instead of ``pc.describe_backup()``.

        :meta private:
        """
        return self.backups.describe(backup_id=backup_id)

    def delete_backup(self, *, backup_id: str) -> None:
        """Backwards-compatibility shim for :meth:`Pinecone.backups.delete`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New code
        should use ``pc.backups.delete()`` instead of ``pc.delete_backup()``.

        :meta private:
        """
        self.backups.delete(backup_id=backup_id)

    def list_restore_jobs(
        self,
        *,
        limit: int | None = None,
        pagination_token: str | None = None,
    ) -> RestoreJobList:
        """Backwards-compatibility shim for :meth:`Pinecone.restore_jobs.list`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New code
        should use ``pc.restore_jobs.list()`` instead of ``pc.list_restore_jobs()``.

        :meta private:
        """
        return self.restore_jobs.list(
            limit=limit,
            pagination_token=pagination_token,
        )

    def describe_restore_job(self, *, job_id: str) -> RestoreJobModel:
        """Backwards-compatibility shim for :meth:`Pinecone.restore_jobs.describe`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New code
        should use ``pc.restore_jobs.describe()`` instead of ``pc.describe_restore_job()``.

        :meta private:
        """
        return self.restore_jobs.describe(job_id=job_id)

    def Index(self, name: str = "", host: str = "", **kwargs: Any) -> Index | GrpcIndex:  # noqa: N802
        """Backwards-compatibility shim for :meth:`Pinecone.index`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New code
        should use ``pc.index(name=..., host=...)`` instead of ``pc.Index(...)``.
        Accepts a legacy ``pool_threads=`` kwarg and forwards it to size the
        ``async_req=True`` thread pool; other unknown kwargs raise ``TypeError``.

        :meta private:
        """
        pool_threads = kwargs.pop("pool_threads", None)
        if kwargs:
            raise TypeError(
                f"Pinecone.Index() got unexpected keyword arguments: {sorted(kwargs)!r}"
            )
        from pinecone.index import Index as _Index

        return cast(_Index, self.index(name=name, host=host, pool_threads=pool_threads))

    def IndexAsyncio(self, host: str, **kwargs: Any) -> Any:  # noqa: N802
        """Backwards-compatibility shim that returns an :class:`AsyncIndex`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New
        code should construct an :class:`AsyncPinecone` and call ``.index(host=...)``
        on it (or instantiate :class:`AsyncIndex` directly) instead of
        ``Pinecone.IndexAsyncio(...)``.

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
        )

    def close(self) -> None:
        """Close all open HTTP connections.

        Closes the main control-plane client and any namespace clients (inference,
        assistants, preview) that were initialized during this session.

        Prefer the context manager form (``with Pinecone(...) as pc:``) which calls
        :meth:`close` automatically on exit.

        Examples:
            Close the client explicitly after use:

            >>> from pinecone import Pinecone
            >>> client = Pinecone(api_key="your-api-key")
            >>> client.close()

            Use Pinecone as a context manager (``close`` is called automatically):

            >>> with Pinecone(api_key="your-api-key") as pinecone_client:
            ...     _ = pinecone_client.indexes.list()
        """
        self._http.close()
        if self._inference is not None:
            self._inference.close()
        if self._assistants is not None:
            self._assistants.close()
        if self._preview is not None:
            self._preview.close()

    def __enter__(self) -> Pinecone:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
