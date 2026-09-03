"""Synchronous Pinecone client."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Literal, NoReturn, overload
from urllib.parse import quote

from pinecone._internal.adaptive import _AdaptiveLimiterRegistry
from pinecone._internal.config import PineconeConfig, RetryConfig
from pinecone._internal.constants import CONTROL_PLANE_API_VERSION, DEFAULT_BASE_URL
from pinecone._internal.index_migration import (
    MIGRATION_GUIDE,
    reject_new_only_configure_kwargs,
    reject_new_only_create_kwargs,
)
from pinecone._internal.indexes_helpers import _LegacyIndexKwargs, poll_index_until_ready
from pinecone._internal.keyword_only import keyword_only_methods
from pinecone._internal.validation import require_non_empty
from pinecone.errors.exceptions import ValidationError
from pinecone.models.indexes.list import IndexList

if TYPE_CHECKING:
    from pinecone.client._assistant_namespace_proxy import _AssistantNamespaceProxy
    from pinecone.client.assistants import Assistants
    from pinecone.client.backup_schedules import BackupSchedules
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
        PodType,
        VectorType,
    )
    from pinecone.models.indexes.index import IndexModel
    from pinecone.models.indexes.specs import EmbedConfig


#: Client attributes retired at 10.0.0, each mapped to what replaced it.
_REMOVED_CLIENT_ATTRIBUTES: dict[str, str] = {
    "preview": (
        "the 2026-01.alpha preview surface graduated onto the client itself, so "
        "pc.preview.indexes is now pc.indexes, pc.preview.index(...) is now "
        "pc.index(...), and pc.preview.close() is now pc.close()"
    )
}


def _removed_client_attribute_message(owner: str, name: str) -> str:
    """Build the guided message for a client attribute retired at 10.0.0.

    Shared by :class:`Pinecone` and
    :class:`~pinecone.async_client.AsyncPinecone` so both clients explain a
    removal in the same words.
    """
    return (
        f"{owner}.{name} was removed at 10.0.0: {_REMOVED_CLIENT_ATTRIBUTES[name]}. "
        f"See {MIGRATION_GUIDE}."
    )


@keyword_only_methods
class Pinecone:
    """Entry point to Pinecone's control plane, over blocking HTTP.

    One client carries your API key, resolved host, and connection pool, so
    construct it once and reuse it. Its namespace properties — :attr:`indexes`,
    :attr:`collections`, :attr:`backups`, :attr:`backup_schedules`,
    :attr:`restore_jobs`, :attr:`inference`, and :attr:`assistants` — create and
    inspect those resources. Vectors are read and written on the data plane
    instead, through the separate client :meth:`index` hands back.

    :class:`~pinecone.async_client.AsyncPinecone` covers the same surface for
    ``asyncio`` code, and :doc:`/guides/sync-vs-async` compares the two; the one
    shape difference is :meth:`index`, which is a plain call here and a coroutine
    there. Any call can raise the connection, timeout, and API errors catalogued
    in :doc:`/guides/error-handling`, and every ``Raises:`` section below names
    only what is specific to that method. :doc:`/guides/retries` covers what the
    client retries on your behalf and what *retry_config* changes.

    Args:
        api_key (str | None): Your Pinecone API key. ``None`` (default) reads
            ``PINECONE_API_KEY`` from the environment, which is how most
            deployments supply it.
        host (str | None): Control-plane host, e.g. ``"https://api.pinecone.io"``.
            A value with no scheme is read as ``https``. ``None`` (default) reads
            ``PINECONE_CONTROLLER_HOST``, then falls back to the public API. Point
            it elsewhere for a gateway, a private endpoint, or a local simulator.
        additional_headers (Mapping[str, str] | None): Headers added to every
            control-plane request, e.g. ``{"X-Request-Source": "nightly-reindex"}``.
            When omitted, the client reads ``PINECONE_ADDITIONAL_HEADERS`` as a JSON
            object instead.
        source_tag (str | None): Attribution tag appended to the User-Agent, e.g.
            ``"acme-search-service"``. Lowercased, spaces become underscores, and
            anything outside ``a-z``, ``0-9``, ``_`` and ``:`` is dropped.
        proxy_url (str | None): Proxy for outgoing requests, e.g.
            ``"http://proxy.corp.internal:3128"``.
        proxy_headers (Mapping[str, str] | None): Headers sent to the proxy itself,
            for a proxy that authenticates.
        ssl_ca_certs (str | None): Path to a CA bundle file, or to a directory of
            them, for a corporate root or a self-signed endpoint. It wins over
            ``ssl_verify=False``: pass both and verification stays on.
        ssl_verify (bool): Whether to verify the server's certificate. ``True``
            (default) is right everywhere but a throwaway test endpoint.
        grpc_scheme ("http" | "https" | None): URL scheme that :meth:`index` with
            ``grpc=True`` dials the data plane over. State it when the data plane is
            reached over something other than public TLS — a plaintext gateway, an
            egress proxy, a private endpoint, or a local simulator — rather than
            leaving the SDK to assume one. ``None`` (default) falls back to the
            ``PINECONE_GRPC_SCHEME`` env var, and then to ``https``. Has no effect on
            REST clients, which take the scheme from the host they are given.
        timeout (float): Deadline in seconds for a single HTTP attempt, not for the
            whole call — each retry gets its own. Defaults to ``30.0``.
        connection_pool_maxsize (int): Ceiling on connections held open to the
            control plane. ``0`` (default) leaves httpx's own ceiling in place;
            raise it for a process issuing many concurrent control-plane calls.
        retry_config (RetryConfig | None): Retry policy for control-plane requests,
            and for gRPC data-plane clients when you set it explicitly. ``None``
            (default) uses the built-in policy, which suits most callers; see
            :doc:`/guides/retries` for the defaults, for how to switch retries off,
            and for why it does not reach data-plane REST.
        pool_threads (int | None): Opt-in for the legacy ``async_req=True`` execution
            model on data-plane methods. When set, indexes created via
            :meth:`index` accept ``async_req=True`` on ``upsert``, ``query``,
            ``describe_index_stats``, and ``list_paginated``. **For new code, prefer**
            :class:`~pinecone.async_client.AsyncPinecone` **or**
            :class:`concurrent.futures.ThreadPoolExecutor`. This keyword is here for
            9.x callers.

    Raises:
        :exc:`PineconeValueError`: If no API key is given and ``PINECONE_API_KEY``
            is unset, since nothing would authenticate the first request.
        :exc:`FileNotFoundError`: If ``ssl_ca_certs`` names a path that does not
            exist, raised when the client is constructed, so a mistyped path
            cannot leave you silently verifying against the default trust store
            instead. A bundle that exists but cannot be parsed as a certificate
            raises :exc:`ssl.SSLError` instead.

    Examples:

        Construct once, then reach the control plane through the namespace
        properties. Leaving ``api_key`` off entirely reads it from
        ``PINECONE_API_KEY``:

        .. code-block:: python

            from pinecone import Pinecone

            pc = Pinecone(api_key="your-api-key")
            if not pc.indexes.exists("product-search"):
                pc.indexes.create(
                    name="product-search",
                    schema={"fields": {"embedding": {
                        "type": "dense_vector", "dimension": 1536, "metric": "cosine"}}},
                )

        :meth:`index` hands back a separate data-plane client scoped to one
        index, which is what reads and writes vectors. A query vector has to
        be as wide as the index's dense field — the three floats below stand
        in for a full 1536-dimensional embedding:

        .. code-block:: python

            idx = pc.index(name="product-search")
            results = idx.query(vector=[0.012, -0.087, 0.153], top_k=10)
            for match in results.matches:
                print(match.id, match.score)
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
        grpc_scheme: Literal["http", "https"] | None = None,
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
        # gRPC's own defaults differ from REST's (5 retries / 0.1s floor vs 3 / 0.25s),
        # so an unset retry_config must stay unset on that transport rather than
        # inheriting REST's numbers. An explicit one is honored on both.
        self._grpc_retry_config = augmented_retry_config if retry_config is not None else None
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
            grpc_scheme=grpc_scheme,
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
        self._backup_schedules: BackupSchedules | None = None
        self._restore_jobs: RestoreJobs | None = None
        self._inference: Inference | None = None
        self._assistants: Assistants | None = None
        self._host_cache: dict[str, str] = {}
        self._legacy_pool_threads: int | None = legacy_pool_threads

    def __repr__(self) -> str:
        """Return a debug-friendly representation with the API key masked.

        Only the last four characters of the API key are shown, so it is safe
        to include in logs, tracebacks, or ``repr()`` output.

        Returns:
            A string like ``"Pinecone(api_key='...ab12', host='https://api.pinecone.io')"``.
        """
        masked = f"...{self._config.api_key[-4:]}" if len(self._config.api_key) >= 4 else "***"
        return f"Pinecone(api_key='{masked}', host='{self._config.host}')"

    @property
    def indexes(self) -> Indexes:
        """Create, inspect, configure, and delete the project's indexes.

        Returns:
            The :class:`~pinecone.client.indexes.Indexes` namespace.

        Examples:

            >>> for index in pc.indexes.list():
            ...     print(index.name, index.status.state)
        """
        if self._indexes is None:
            from pinecone.client.indexes import Indexes as _Indexes

            self._indexes = _Indexes(http=self._http, host_cache=self._host_cache)
        return self._indexes

    @property
    def collections(self) -> Collections:
        """Create and inspect collections: static snapshots of a pod-based index.

        A collection is the pod-based snapshot format. The serverless equivalent
        is a backup, under :attr:`backups`.

        Returns:
            The :class:`~pinecone.client.collections.Collections` namespace.

        Examples:

            >>> for col in pc.collections.list():
            ...     print(col.name, col.status)
            movie-embeddings-snapshot Ready
            product-catalog-snapshot Initializing
        """
        if self._collections is None:
            from pinecone.client.collections import Collections as _Collections

            self._collections = _Collections(http=self._http)
        return self._collections

    @property
    def backups(self) -> Backups:
        """Create, inspect, and delete backups of a serverless index.

        Restoring one is not done from here: pass a ``backup_id`` to
        :meth:`create_index_from_backup`, which creates a new index from it. For
        pod-based indexes the snapshot format is a collection, under
        :attr:`collections`.

        Returns:
            The :class:`~pinecone.client.backups.Backups` namespace.

        Examples:

            >>> for backup in pc.backups.list(limit=100):
            ...     print(backup.backup_id, backup.source_index_name, backup.status)
            bk-abc123 product-search Ready
            bk-def456 product-search Ready
        """
        if self._backups is None:
            from pinecone.client.backups import Backups as _Backups

            self._backups = _Backups(http=self._http)
        return self._backups

    @property
    def backup_schedules(self) -> BackupSchedules:
        """Attach a recurring backup cadence to an index.

        A schedule gives an index a daily, weekly, or monthly cadence, so
        Pinecone creates each backup for you rather than you triggering one
        every time.

        Returns:
            The
            :class:`~pinecone.client.backup_schedules.BackupSchedules`
            namespace.

        Examples:

            >>> for schedule in pc.backup_schedules.list(index_name="product-search"):
            ...     print(schedule.name, schedule.frequency, schedule.enabled)
            compliance-snapshots daily True
        """
        if self._backup_schedules is None:
            from pinecone.client.backup_schedules import BackupSchedules as _BackupSchedules

            self._backup_schedules = _BackupSchedules(http=self._http)
        return self._backup_schedules

    @property
    def restore_jobs(self) -> RestoreJobs:
        """Track a restore that :meth:`create_index_from_backup` started.

        A restore job is the request itself, so it is what to follow when you
        passed ``timeout=-1`` and the target index does not exist yet.

        Returns:
            The :class:`~pinecone.client.restore_jobs.RestoreJobs` namespace.

        Examples:

            >>> for job in pc.restore_jobs.list(limit=10):
            ...     print(job.restore_job_id, job.target_index_name, job.status)
            rj-abc123 product-search-restored Completed
        """
        if self._restore_jobs is None:
            from pinecone.client.restore_jobs import RestoreJobs as _RestoreJobs

            self._restore_jobs = _RestoreJobs(http=self._http)
        return self._restore_jobs

    @property
    def inference(self) -> Inference:
        """Embed text or images, and rerank documents, on hosted models.

        Reach for this when you want vectors or relevance scores without
        hosting a model yourself.

        Returns:
            The :class:`~pinecone.client.inference.Inference` namespace.

        Examples:

            ``multilingual-e5-large`` is asymmetric, so ``input_type`` tells it
            which side of a search the text belongs to — ``"passage"`` for text
            you intend to store, ``"query"`` for text you intend to search with:

            >>> embeddings = pc.inference.embed(
            ...     model="multilingual-e5-large",
            ...     inputs=["Solar panels reduce energy costs and lower carbon emissions."],
            ...     parameters={"input_type": "passage"},
            ... )
            >>> len(embeddings.data)
            1
        """
        if self._inference is None:
            from pinecone.client.inference import Inference as _Inference

            self._inference = _Inference(config=self._config)
        return self._inference

    @property
    def assistants(self) -> Assistants:
        """Create and manage Pinecone Assistants.

        An assistant is a hosted, retrieval-augmented chat service: upload
        files to it and it answers questions grounded in their content, with no
        index or embedding pipeline of your own.

        Returns:
            The :class:`~pinecone.client.assistants.Assistants` namespace.

        Examples:

            >>> for assistant in pc.assistants.list():
            ...     print(assistant.name, assistant.status)
        """
        if self._assistants is None:
            from pinecone.client.assistants import Assistants as _Assistants

            self._assistants = _Assistants(config=self._config)
        return self._assistants

    @property
    def assistant(self) -> _AssistantNamespaceProxy:
        """Reach one assistant by name, or the whole namespace by attribute.

        :attr:`assistants` is the canonical namespace; this singular alias is
        not deprecated. It forwards every attribute there, and calling it with
        a name is shorthand for
        :meth:`~pinecone.client.assistants.Assistants.describe`.

        Returns:
            A proxy that behaves like the :class:`Assistants` namespace for
            attribute access (``pc.assistant.create(...)``) and, when called
            with a name, returns that assistant's details.

        Examples:

            Calling the proxy with a name is shorthand for
            :meth:`~pinecone.client.assistants.Assistants.describe`:

            >>> bot = pc.assistant("acme-support-bot")
            >>> bot.status
            'Ready'

            Every other attribute forwards to the plural namespace, so
            ``pc.assistant.create`` and ``pc.assistants.create`` are the same
            method reached two ways:

            >>> new_bot = pc.assistant.create(
            ...     name="acme-billing-bot",
            ...     instructions="Help users with billing questions.",
            ... )
            >>> new_bot.status
            'Ready'
        """
        from pinecone.client._assistant_namespace_proxy import _AssistantNamespaceProxy

        return _AssistantNamespaceProxy(self.assistants)

    @overload
    def index(
        self,
        name: str = ...,
        *,
        host: str = ...,
        grpc: Literal[False] = ...,
        pool_threads: int | None = ...,
    ) -> Index: ...

    @overload
    def index(
        self,
        name: str = ...,
        *,
        host: str = ...,
        grpc: Literal[True],
        pool_threads: int | None = ...,
    ) -> GrpcIndex: ...

    @overload
    def index(
        self,
        name: str = ...,
        *,
        host: str = ...,
        grpc: bool,
        pool_threads: int | None = ...,
    ) -> Index | GrpcIndex: ...

    def index(
        self,
        name: str = "",
        *,
        host: str = "",
        grpc: bool = False,
        pool_threads: int | None = None,
    ) -> Index | GrpcIndex:
        """Open a data-plane client for one index, to read and write vectors.

        A plain call, not a coroutine: it blocks while it resolves the host. An
        explicit *host* is used as-is, a *name* is served from this client's host
        cache, and a *name* that misses the cache costs one describe request. The
        async twin, :meth:`AsyncPinecone.index()
        <pinecone.async_client.pinecone.AsyncPinecone.index>`, is a coroutine you
        await, and cannot return a gRPC client.

        Args:
            name (str): Name of the index, e.g. ``"product-search"``. Costs one
                describe request the first time, then comes from the host cache.
            host (str): The index's host, e.g.
                ``"product-search-abc123.svc.pinecone.io"``. Pass it when you have
                it already and the describe request is skipped entirely.
            grpc (bool): Return a :class:`~pinecone.grpc.GrpcIndex` that carries
                data-plane operations over gRPC rather than HTTP. The scheme it
                dials comes from the ``grpc_scheme`` given to :class:`Pinecone`.
                Defaults to ``False``; see :doc:`/guides/grpc` for when it pays off.
            pool_threads (int | None): Size of the thread pool backing
                ``async_req=True`` calls on the returned index. ``None`` (default)
                uses the client-level ``pool_threads``. No effect when ``grpc=True``.

        Returns:
            An :class:`~pinecone.index.Index` over HTTP, or a
            :class:`~pinecone.grpc.GrpcIndex` when ``grpc=True``.

        Raises:
            :exc:`PineconeValueError`: If neither *name* nor *host* is given, or if
                *name* names an index that has no host yet — it is still
                initializing, so wait for its status to reach ``Ready``.
            :exc:`~pinecone.errors.NotFoundError`: If *name* names no index in this
                project.

        Examples:

            >>> idx = pc.index(name="product-search")

            Passing the host skips the lookup, which saves a round trip when you
            already know it — from :meth:`Indexes.describe
            <pinecone.client.indexes.Indexes.describe>`, or from your own config:

            >>> idx = pc.index(host="product-search-abc123.svc.pinecone.io")

            Either form accepts ``grpc=True`` for the gRPC transport:

            >>> idx = pc.index(name="product-search", grpc=True)

        .. seealso::
           :attr:`indexes` — the control-plane namespace, for creating, listing,
           describing, configuring, and deleting indexes rather than reading from
           one.
        """
        resolved_host = self._resolve_index_host(name=name, host=host)

        if grpc:
            from pinecone.grpc import GrpcIndex as _GrpcIndex

            return _GrpcIndex(
                host=resolved_host,
                api_key=self._config.api_key,
                grpc_scheme=self._config.grpc_scheme,
                source_tag=self._config.source_tag or None,
                retry_config=self._grpc_retry_config,
                proxy_url=self._config.proxy_url or None,
                on_throttle=self._limiter_registry.report_throttled,
                limiter_registry=self._limiter_registry,
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
        read_capacity: dict[str, Any] | None = None,
        timeout: int | None = None,
    ) -> CreateIndexFromBackupResponse | IndexModel:
        """Create a new index by restoring from a backup.

        Blocks and polls until the restored index is ready, unless *timeout* is
        ``-1``. This is the only supported way to restore a backup:
        :meth:`create_index` rejects ``source_backup_id=`` with a message
        pointing here.

        Args:
            name (str): Name for the new index.
            backup_id (str): Identifier of the backup to restore from. Obtain it from
                :meth:`Backups.create <pinecone.client.backups.Backups.create>` or
                :meth:`Backups.list <pinecone.client.backups.Backups.list>`.
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
                configuration too small to hold the backup.
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
            :exc:`NotFoundError`: If *backup_id* does not match an existing backup.
            :exc:`ConflictError`: If an index named *name* already exists.
            :exc:`ApiError`: If the API returns another error response, for
                example if the backup is not yet complete.

        Examples:

            The default form blocks until the restored index is ready and hands
            back the index itself, so the next call can use it:

            >>> index = pc.create_index_from_backup(
            ...     name="product-search-restored",
            ...     backup_id="bk-abc123",
            ... )
            >>> index.status.state
            'Ready'

            ``timeout=-1`` returns as soon as the restore is accepted. What
            comes back is a :class:`CreateIndexFromBackupResponse`, not an
            index — the index does not exist yet, so follow the restore through
            ``pc.restore_jobs`` rather than treating the return value as one:

            >>> result = pc.create_index_from_backup(
            ...     name="product-search-restored",
            ...     backup_id="bk-abc123",
            ...     timeout=-1,
            ... )
            >>> job = pc.restore_jobs.describe(job_id=result.restore_job_id)
            >>> job.status
            'Completed'

            A restore can land straight onto dedicated read nodes instead of
            the on-demand default:

            >>> index = pc.create_index_from_backup(
            ...     name="product-search-restored",
            ...     backup_id="bk-abc123",
            ...     read_capacity={
            ...         "mode": "Dedicated",
            ...         "dedicated": {
            ...             "node_type": "t1",
            ...             "scaling": "Manual",
            ...             "manual": {"shards": 2, "replicas": 2},
            ...         },
            ...     },
            ... )
            >>> index.status.state
            'Ready'

        .. versionchanged:: 10.0
           Added *read_capacity*, so a restore can land straight onto dedicated
           read nodes instead of defaulting to on-demand capacity.
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
        response = self._http.post(
            f"/backups/{quote(backup_id, safe='')}/create-index",
            content=BackupsAdapter.to_create_index_from_backup_request(request),
            headers={"Content-Type": "application/json"},
        )
        create_response = BackupsAdapter.to_create_index_from_backup_response(response.content)

        if timeout == -1:
            return create_response

        effective_timeout = timeout if timeout is not None else 300
        return poll_index_until_ready(self.indexes.describe, name, effective_timeout)

    @property
    def config(self) -> PineconeConfig:
        """Read back the settings this client resolved at construction.

        Returns:
            :class:`~pinecone._internal.config.PineconeConfig` carrying the
            resolved API key, host, timeout, and connection settings.

        Examples:

            The values are post-resolution, with defaults and environment
            variables folded in, so this is where to confirm which host a client
            is actually pointed at:

            >>> pc.config.host
            'https://api.pinecone.io'
            >>> pc.config.timeout
            30.0
        """
        return self._config

    # ---- Backcompat flat-method delegates (:meta private:) ----

    def create_index(
        self,
        name: str,
        spec: Any = None,
        dimension: int | None = None,
        metric: Metric | str | None = None,
        timeout: int | None = None,
        deletion_protection: DeletionProtection | str | None = None,
        vector_type: VectorType | str | None = None,
        tags: Mapping[str, str] | None = None,
        **legacy_kwargs: Any,
    ) -> IndexModel:
        """Backwards-compatibility shim for :meth:`Pinecone.indexes.create`.

        Preserved to ease migration from the legacy (9.x) Pinecone Python
        SDK, with the same legacy parameter list — including its positional
        order, so ``pc.create_index("movies", ServerlessSpec(...), 1536)``
        keeps working. New code should use
        ``pc.indexes.create()`` instead, which additionally accepts the current
        ``schema=``/``deployment=``/``read_capacity=``/``cmek_id=`` surface not
        available here. ``pods=``/``metadata_config=``/
        ``source_collection=``/``source_backup_id=`` reach
        :meth:`Pinecone.indexes.create`, which raises for them.

        Args:
            name (str): Name for the index — 1-45 characters, lowercase
                alphanumerics and hyphens.
            spec (Any): A :class:`~pinecone.models.indexes.specs.ServerlessSpec`,
                :class:`~pinecone.models.indexes.specs.PodSpec`,
                :class:`~pinecone.models.indexes.specs.ByocSpec`, or equivalent
                dict, translated into ``deployment=`` by
                :meth:`Pinecone.indexes.create`.
            dimension (int | None): Dense vector width, translated into a
                single-field ``schema=``.
            metric (Metric | str | None): Similarity metric — ``"cosine"``
                (default), ``"euclidean"``, or ``"dotproduct"``.
            timeout (int | None): Seconds to wait for the index to become ready.
                ``None`` (default) waits indefinitely; ``-1`` returns immediately.
            deletion_protection (DeletionProtection | str | None): ``"enabled"``
                or ``"disabled"`` (default).
            vector_type (VectorType | str | None): ``"dense"`` (default) or
                ``"sparse"``.
            tags (Mapping[str, str] | None): Optional key-value tags to attach.
            **legacy_kwargs: Legacy keywords such as ``pods=``,
                ``metadata_config=``, ``source_collection=``, and
                ``source_backup_id=``, forwarded to
                :meth:`Pinecone.indexes.create`, which rejects them.

        Returns:
            :class:`IndexModel` describing the created index.

        Raises:
            :exc:`PineconeTypeError`: If an unsupported legacy keyword is passed.
            :exc:`IndexInitFailedError`: If the index fails to initialize.
            :exc:`PineconeTimeoutError`: If the index isn't ready before
                *timeout* elapses.
            :exc:`ApiError`: If the API returns another error response.

        Examples:
            >>> pc.create_index(  # doctest: +SKIP
            ...     name="movie-recommendations", dimension=1536, metric="cosine",
            ... )

        :meta private:
        """
        reject_new_only_create_kwargs(legacy_kwargs)
        return self.indexes.create(
            name=name,
            spec=spec,
            dimension=dimension,
            metric=metric,
            vector_type=vector_type,
            deletion_protection=deletion_protection,
            tags=tags,
            timeout=timeout,
            **legacy_kwargs,
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
        """Backwards-compatibility shim for :meth:`Pinecone.indexes.create_for_model`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New
        code should use ``pc.indexes.create_for_model()`` instead of
        ``pc.create_index_for_model()``.

        Args:
            name (str): Name for the index — 1-45 characters, lowercase
                alphanumerics and hyphens.
            cloud (CloudProvider | str): Public cloud provider — ``"aws"``,
                ``"gcp"``, or ``"azure"``.
            region (AwsRegion | GcpRegion | AzureRegion | str): Cloud region,
                e.g. ``"us-east-1"``.
            embed (IndexEmbed | EmbedConfig | dict[str, Any]): Embedding
                configuration with required ``model`` and ``field_map``, and
                optional ``metric``, ``dimension``, ``read_parameters``,
                ``write_parameters``. The model cannot be changed after creation.
            tags (Mapping[str, str] | None): Optional key-value tags to attach.
            deletion_protection (DeletionProtection | str | None): ``"enabled"``
                or ``"disabled"`` (default).
            read_capacity (dict[str, Any] | None): Optional read capacity dict.
            schema (dict[str, Any] | None): Optional metadata schema for
                filterable metadata fields.
            timeout (int | None): Seconds to wait for the index to become ready.
                ``None`` (default) waits indefinitely; ``-1`` returns immediately.

        Returns:
            :class:`IndexModel` describing the created index.

        Raises:
            :exc:`PineconeValueError`: If *name*, *cloud*, *region*, or *embed*
                fail client-side validation.
            :exc:`ApiError`: If the API returns another error response.

        Examples:
            >>> pc.create_index_for_model(  # doctest: +SKIP
            ...     name="semantic-search",
            ...     cloud="aws",
            ...     region="us-east-1",
            ...     embed={"model": "multilingual-e5-large",
            ...            "field_map": {"text": "chunk_text"}},
            ... )

        :meta private:
        """
        return self.indexes.create_for_model(
            name=name,
            cloud=cloud,
            region=region,
            embed=embed,
            deletion_protection=deletion_protection,
            tags=tags,
            schema=schema,
            read_capacity=read_capacity,
            timeout=timeout,
        )

    def describe_index(self, name: str) -> IndexModel:
        """Backwards-compatibility shim for :meth:`Pinecone.indexes.describe`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New code
        should use ``pc.indexes.describe()`` instead of ``pc.describe_index()``.

        Args:
            name (str): The name of the index to describe.

        Returns:
            :class:`IndexModel` with name, host, schema, deployment,
            read_capacity, status, deletion_protection, and tags.

        Raises:
            :exc:`PineconeValueError`: If *name* is empty.
            :exc:`NotFoundError`: If the index does not exist.
            :exc:`ApiError`: If the API returns another error response.

        Examples:
            >>> desc = pc.describe_index("my-index")
            >>> desc.host  # doctest: +SKIP
            'https://my-index.svc.pinecone.io'

        :meta private:
        """
        return self.indexes.describe(name)

    def list_indexes(self) -> IndexList:
        """Backwards-compatibility shim for :meth:`Pinecone.indexes.list`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New
        code should use ``pc.indexes.list()`` instead of ``pc.list_indexes()``,
        which returns a :class:`~pinecone.models.pagination.Paginator`.

        Returns:
            :class:`IndexList` wrapping every index in the project.

        Raises:
            :exc:`ApiError`: If the API returns an error response.

        Examples:
            >>> for index in pc.list_indexes():  # doctest: +SKIP
            ...     print(index.name)

        :meta private:
        """
        return IndexList(self.indexes.list().to_list())

    def has_index(self, name: str) -> bool:
        """Backwards-compatibility shim for :meth:`Pinecone.indexes.exists`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New code
        should use ``pc.indexes.exists()`` instead of ``pc.has_index()``.

        Args:
            name (str): The name of the index to check.

        Returns:
            True if the index exists, False otherwise.

        Raises:
            :exc:`PineconeValueError`: If *name* is empty.
            :exc:`ApiError`: If the API returns an error response other than a not-found error.

        :meta private:
        """
        return self.indexes.exists(name)

    def configure_index(
        self,
        name: str,
        replicas: int | None = None,
        pod_type: PodType | str | None = None,
        deletion_protection: DeletionProtection | str | None = None,
        tags: Mapping[str, str] | None = None,
        embed: Any = None,
        read_capacity: dict[str, Any] | None = None,
        serverless_read_capacity: dict[str, Any] | None = None,
        **legacy_kwargs: Any,
    ) -> IndexModel:
        """Backwards-compatibility shim for :meth:`Pinecone.indexes.configure`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New
        code should use ``pc.indexes.configure()`` instead of
        ``pc.configure_index()``, which additionally accepts the current
        ``deployment=``/``schema=`` surface not available here. ``embed=``
        reaches :meth:`Pinecone.indexes.configure`, which raises for it: the
        convert-to-integrated flow it drove has no current equivalent.

        .. versionchanged:: 10.0
           Returns the updated :class:`IndexModel` where the 9.x method
           returned ``None``.

        Args:
            name (str): Name of the index to configure.
            replicas (int | None): Pod-based replica count.
            pod_type (PodType | str | None): Pod type, e.g. ``"p1.x2"``.
            deletion_protection (DeletionProtection | str | None):
                ``"enabled"`` or ``"disabled"``.
            tags (Mapping[str, str] | None): Tag updates, merged with
                existing tags on the server. Set a value to ``""`` to
                delete that key.
            embed (Any): Not supported by this shim; passing a value raises.
            read_capacity (dict[str, Any] | None): Updated read capacity,
                e.g. ``{"mode": "OnDemand"}``.
            serverless_read_capacity (dict[str, Any] | None): Alias for
                *read_capacity*.

        Returns:
            :class:`IndexModel` reflecting the updated index state.

        Raises:
            :exc:`PineconeTypeError`: If ``embed=`` is passed, or an
                argument outside this shim's signature is given.
            :exc:`PineconeValueError`: If *name* is empty or every other
                argument is left at its default.
            :exc:`NotFoundError`: If the index does not exist.
            :exc:`ApiError`: If the API returns another error response.

        :meta private:
        """
        reject_new_only_configure_kwargs(legacy_kwargs)
        return self.indexes.configure(
            name,
            replicas=replicas,
            pod_type=pod_type,
            deletion_protection=deletion_protection,
            tags=tags,
            embed=embed,
            read_capacity=read_capacity,
            serverless_read_capacity=serverless_read_capacity,
            **legacy_kwargs,
        )

    def delete_index(self, name: str, timeout: int | None = None) -> None:
        """Backwards-compatibility shim for :meth:`Pinecone.indexes.delete`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New code
        should use ``pc.indexes.delete()`` instead of ``pc.delete_index()``.

        Args:
            name (str): The name of the index to delete.
            timeout (int | None): Seconds to wait for the index to disappear.
                Use ``None`` (default) to poll indefinitely until the index
                is gone. Use a positive int to poll with a deadline. Use
                ``-1`` to return immediately without polling.

        Raises:
            :exc:`PineconeValueError`: If *name* is empty.
            :exc:`NotFoundError`: If the index does not exist.
            :exc:`ForbiddenError`: If deletion protection is enabled on the index.
            :exc:`PineconeTimeoutError`: If the index still exists after *timeout* seconds.
            :exc:`ApiError`: If the API returns another error response.

        :meta private:
        """
        self.indexes.delete(name, timeout=timeout)

    def create_collection(self, name: str, source: str) -> CollectionModel:
        """Backwards-compatibility shim for :meth:`Pinecone.collections.create`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New code
        should use ``pc.collections.create()`` instead of ``pc.create_collection()``.

        Args:
            name (str): Name for the new collection. 1-45 characters,
                lowercase alphanumeric and hyphens only, and can't start or
                end with a hyphen (e.g. ``"movie-embeddings-snapshot"``).
            source (str): Name of the pod-based index to copy.

        Returns:
            A :class:`CollectionModel` describing the created collection.

        Raises:
            :exc:`PineconeValueError`: If *name* or *source* is empty, or
                *name* doesn't meet the naming rules above.
            :exc:`NotFoundError`: If *source* does not name an index in
                this project.

        :meta private:
        """
        return self.collections.create(name=name, source=source)

    def list_collections(self) -> CollectionList:
        """Backwards-compatibility shim for :meth:`Pinecone.collections.list`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New code
        should use ``pc.collections.list()`` instead of ``pc.list_collections()``.

        Returns:
            A :class:`CollectionList` supporting iteration, len(), index access,
            and a names() convenience method.

        :meta private:
        """
        return self.collections.list()

    def describe_collection(self, name: str) -> CollectionModel:
        """Backwards-compatibility shim for :meth:`Pinecone.collections.describe`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New code
        should use ``pc.collections.describe()`` instead of ``pc.describe_collection()``.

        Args:
            name (str): Name of the collection to describe.

        Returns:
            A :class:`CollectionModel` with the collection's name, status,
            size, dimension, vector_count, and environment.

        Raises:
            :exc:`PineconeValueError`: If *name* is empty.
            :exc:`NotFoundError`: If the collection does not exist.

        :meta private:
        """
        return self.collections.describe(name)

    def delete_collection(self, name: str) -> None:
        """Backwards-compatibility shim for :meth:`Pinecone.collections.delete`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New code
        should use ``pc.collections.delete()`` instead of ``pc.delete_collection()``.

        Args:
            name (str): Name of the collection to delete.

        Raises:
            :exc:`PineconeValueError`: If *name* is empty.
            :exc:`NotFoundError`: If the collection does not exist.

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
        Note the keyword is *backup_name* here, where :meth:`Pinecone.backups.create`
        uses *name*.

        Args:
            index_name (str): Name of the index to back up.
            backup_name (str | None): Name for the backup, e.g.
                ``"daily-20240115"``. When omitted, the backup has no name
                and is identified only by its ``backup_id``.
            description (str | None): Description for the backup.

        Returns:
            A :class:`BackupModel` describing the new backup. The call
            returns once the backup is initiated; check its ``status`` via
            :meth:`Pinecone.describe_backup` to see when it's ready.

        Raises:
            :exc:`PineconeValueError`: If *index_name* is empty.
            :exc:`ForbiddenError`: If the organization's plan does not
                include backups.
            :exc:`NotFoundError`: If *index_name* does not resolve to an
                index in this project.
            :exc:`ApiError`: If the API returns another error response, for
                example because *index_name* names a pod-based index.

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
        include_deleted: bool | None = None,
    ) -> BackupList:
        """Backwards-compatibility shim for :meth:`Pinecone.backups.list`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New code
        should use ``pc.backups.list()`` instead of ``pc.list_backups()``.

        Args:
            index_name (str | None): Index name to scope the listing to, or
                ``None`` for every backup in the project.
            limit (int | None): Maximum number of results per page.
            pagination_token (str | None): Token naming the next page,
                taken from a previous :class:`BackupList`'s pagination info.
            include_deleted (bool | None): When ``True``, include backups of
                every index that has ever used *index_name*, deleted ones
                included. Only valid together with *index_name*.

        Returns:
            A :class:`BackupList` supporting iteration, len(), and index
            access.

        Raises:
            :exc:`PineconeValueError`: If *include_deleted* is given without
                *index_name*.
            :exc:`NotFoundError`: If *index_name* does not resolve to an
                active index and *include_deleted* is not ``True``.
            :exc:`ApiError`: If the API returns another error response.

        :meta private:
        """
        return self.backups.list(
            index_name=index_name,
            limit=limit,
            pagination_token=pagination_token,
            include_deleted=include_deleted,
        )

    def describe_backup(self, *, backup_id: str) -> BackupModel:
        """Backwards-compatibility shim for :meth:`Pinecone.backups.describe`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New code
        should use ``pc.backups.describe()`` instead of ``pc.describe_backup()``.

        Args:
            backup_id (str): The identifier of the backup to describe.

        Returns:
            A :class:`BackupModel` with full backup details.

        Raises:
            :exc:`PineconeValueError`: If *backup_id* is empty.
            :exc:`NotFoundError`: If the backup does not exist.
            :exc:`ApiError`: If the API returns another error response.

        :meta private:
        """
        return self.backups.describe(backup_id=backup_id)

    def delete_backup(self, *, backup_id: str) -> None:
        """Backwards-compatibility shim for :meth:`Pinecone.backups.delete`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New code
        should use ``pc.backups.delete()`` instead of ``pc.delete_backup()``.

        Args:
            backup_id (str): The identifier of the backup to delete.

        Raises:
            :exc:`PineconeValueError`: If *backup_id* is empty.
            :exc:`NotFoundError`: If the backup does not exist.
            :exc:`ApiError`: If the API returns another error response.

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
        Returns a single page and does not auto-fetch further pages; see
        :meth:`Pinecone.restore_jobs.list` for pagination details and a note
        on its known limitations.

        Args:
            limit (int | None): Maximum number of results per page.
            pagination_token (str | None): Token naming the next page, taken
                from a previous :class:`RestoreJobList`'s pagination info.

        Returns:
            A :class:`RestoreJobList` supporting iteration, len(), and index
            access.

        Raises:
            :exc:`ApiError`: If the API returns an error response.

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
        See :meth:`Pinecone.restore_jobs.describe` for important caveats about
        what :exc:`NotFoundError` from this call does and does not mean.

        Args:
            job_id (str): The identifier of the restore job to describe.

        Returns:
            A :class:`RestoreJobModel` with full restore job details.

        Raises:
            :exc:`PineconeValueError`: If *job_id* is empty.
            :exc:`NotFoundError`: If the API answers with a not-found
                response for this job.
            :exc:`ApiError`: If the API returns another error response.

        :meta private:
        """
        return self.restore_jobs.describe(job_id=job_id)

    def Index(self, name: str = "", host: str = "", **kwargs: Any) -> Index | GrpcIndex:  # noqa: N802
        """Backwards-compatibility shim for :meth:`Pinecone.index`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New code
        should use ``pc.index(name=..., host=...)`` instead of ``pc.Index(...)``.
        Always returns an HTTP :class:`Index`; use :meth:`Pinecone.index` with
        ``grpc=True`` for a gRPC client instead.

        Args:
            name (str): Name of the index. Triggers a describe call to resolve
                its host, same as in :meth:`Pinecone.index`.
            host (str): Direct host URL of the index. Skips the describe call.
            **kwargs: Accepts the legacy ``pool_threads`` keyword, sizing the
                connection pool used by the ``async_req=True`` thread pool. Any
                other keyword raises ``TypeError``.

        Returns:
            A sync :class:`Index` data-plane client.

        Raises:
            :exc:`TypeError`: If a keyword other than ``pool_threads`` is passed.
            :exc:`PineconeValueError`: If neither ``name`` nor ``host`` is provided.
            :exc:`~pinecone.errors.NotFoundError`: If ``name`` is given but no
                index with that name exists.

        Examples:
            >>> from pinecone import Pinecone
            >>> pc = Pinecone(api_key="your-api-key")
            >>> idx = pc.Index("product-search")  # doctest: +SKIP

        :meta private:
        """
        pool_threads = kwargs.pop("pool_threads", None)
        if kwargs:
            raise TypeError(
                f"Pinecone.Index() got unexpected keyword arguments: {sorted(kwargs)!r}"
            )
        return self.index(name=name, host=host, pool_threads=pool_threads)

    def IndexAsyncio(self, host: str, **kwargs: Any) -> Any:  # noqa: N802
        """Backwards-compatibility shim that returns an :class:`AsyncIndex`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New
        code should construct an :class:`AsyncPinecone` and call ``.index(host=...)``
        on it (or instantiate :class:`AsyncIndex` directly) instead of
        ``Pinecone.IndexAsyncio(...)``. A plain (non-``async def``) method: it
        builds the client from this :class:`Pinecone` client's resolved
        configuration without any I/O of its own.

        Args:
            host (str): Direct host URL of the index.
            **kwargs: Unused; accepted for signature compatibility with the
                legacy SDK.

        Returns:
            An :class:`AsyncIndex` data-plane client, configured with this
            client's API key, timeout, proxy, and TLS settings.

        Examples:
            >>> from pinecone import Pinecone
            >>> pc = Pinecone(api_key="your-api-key")
            >>> idx = pc.IndexAsyncio(host="my-index-abc123.svc.pinecone.io")  # doctest: +SKIP

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
        """Release this client's control-plane connections.

        Closes the control-plane connection pool, plus the :attr:`inference` and
        :attr:`assistants` pools if those namespaces were used. Index clients from
        :meth:`index` hold their own connections and are not closed here. Prefer
        the context manager form, ``with Pinecone(...) as pc:``, which calls this
        on the way out.

        Examples:
            The context manager form closes the client on the way out, on an
            exception as well as on a normal exit:

            >>> from pinecone import Pinecone
            >>> with Pinecone(api_key="your-api-key") as client:
            ...     for index in client.indexes.list():
            ...         print(index.name)

            Close it yourself when the client has to outlive a single block:

            >>> client = Pinecone(api_key="your-api-key")
            >>> try:
            ...     print(client.indexes.exists("product-search"))
            ... finally:
            ...     client.close()
            True
        """
        self._http.close()
        if self._inference is not None:
            self._inference.close()
        if self._assistants is not None:
            self._assistants.close()

    def __enter__(self) -> Pinecone:
        """Enter the context manager, returning this client.

        Returns:
            This :class:`Pinecone` instance.

        Examples:
            >>> with Pinecone(api_key="your-api-key") as client:
            ...     for index in client.indexes.list():
            ...         print(index.name)
        """
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit the context manager, closing the client.

        Calls :meth:`close` to release open HTTP connections.

        Examples:
            >>> with Pinecone(api_key="your-api-key") as client:
            ...     for index in client.indexes.list():
            ...         print(index.name)
        """
        self.close()

    if not TYPE_CHECKING:

        def __getattr__(self, name: str) -> NoReturn:
            """Explain a removed client attribute rather than failing bare.

            Reached only for names normal lookup did not find. ``preview`` gets
            a message naming its replacement; every other name raises Python's
            own wording, so ``hasattr`` and ``getattr`` defaults behave as
            usual. Hidden from type checkers on purpose: a visible
            ``__getattr__`` makes every attribute name valid, which would stop
            them reporting a misspelled one.

            Raises:
                AttributeError: Always.
            """
            if name in _REMOVED_CLIENT_ATTRIBUTES:
                raise AttributeError(_removed_client_attribute_message(type(self).__name__, name))
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
