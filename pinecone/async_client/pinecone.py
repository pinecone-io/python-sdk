"""Asynchronous Pinecone client."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Any, NoReturn
from urllib.parse import quote

from pinecone._internal.adaptive import _AdaptiveLimiterRegistry
from pinecone._internal.config import PineconeConfig, RetryConfig
from pinecone._internal.constants import CONTROL_PLANE_API_VERSION, DEFAULT_BASE_URL
from pinecone._internal.index_migration import (
    reject_new_only_configure_kwargs,
    reject_new_only_create_kwargs,
)
from pinecone._internal.indexes_helpers import IndexKwargs, async_poll_index_until_ready
from pinecone._internal.keyword_only import keyword_only_methods
from pinecone._internal.validation import require_non_empty
from pinecone.errors.exceptions import ValidationError
from pinecone.models.indexes.list import IndexList

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
        Metric,
        PodType,
        VectorType,
    )
    from pinecone.models.indexes.index import IndexModel
    from pinecone.models.indexes.specs import EmbedConfig


@keyword_only_methods
class AsyncPinecone:
    """Entry point to Pinecone's control plane, over ``asyncio``.

    One client carries your API key, resolved host, and connection pool, so build
    it once — inside ``async with``, which closes it for you — and reuse it. Its
    namespace properties — :attr:`indexes`, :attr:`collections`, :attr:`backups`,
    :attr:`backup_schedules`, :attr:`restore_jobs`, :attr:`inference`, and
    :attr:`assistants` — create and inspect those resources. Vectors are read and
    written on the data plane instead, through the separate client :meth:`index`
    hands back.

    :class:`~pinecone.Pinecone` is the blocking twin, and
    :doc:`/guides/sync-vs-async` compares the two. The one shape difference is
    :meth:`index`: here it is a coroutine you await, so the one describe request a
    cache-missing name costs does not block the event loop, and it takes no
    ``grpc=`` argument, the gRPC transport being sync-only. Any call can raise the
    connection, timeout, and API errors catalogued in
    :doc:`/guides/error-handling`, and every ``Raises:`` section below names only
    what is specific to that method. :doc:`/guides/retries` covers what the client
    retries on your behalf and what *retry_config* changes.

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
        proxy_headers (Mapping[str, str] | None): Not supported here: a non-empty
            mapping raises :exc:`NotImplementedError` at construction. Use
            :class:`~pinecone.Pinecone` when the proxy needs headers of its own.
        ssl_ca_certs (str | None): Path to a CA bundle file, or to a directory of
            them, for a corporate root or a self-signed endpoint. It wins over
            ``ssl_verify=False``: pass both and verification stays on.
        ssl_verify (bool): Whether to verify the server's certificate. ``True``
            (default) is right everywhere but a throwaway test endpoint.
        timeout (float): Deadline in seconds for a single HTTP attempt, not for the
            whole call — each retry gets its own. Defaults to ``30.0``.
        connection_pool_maxsize (int): Ceiling on connections held open to the
            control plane. ``0`` (default) leaves httpx's own ceiling in place;
            raise it for a process issuing many concurrent control-plane calls.
        retry_config (RetryConfig | None): Retry policy for control-plane requests.
            ``None`` (default) uses the built-in policy, which suits most callers;
            see :doc:`/guides/retries` for the defaults, for how to switch retries
            off, and for why it does not reach data-plane REST.

    Raises:
        :exc:`PineconeValueError`: If no API key is given and ``PINECONE_API_KEY``
            is unset, since nothing would authenticate the first request.
        :exc:`NotImplementedError`: If *proxy_headers* is non-empty.
        :exc:`FileNotFoundError`: If ``ssl_ca_certs`` names a path that does not
            exist, raised on the first request rather than when the client is
            constructed, so a mistyped path cannot leave you silently verifying
            against the default trust store instead. A bundle that exists but
            cannot be parsed as a certificate raises :exc:`ssl.SSLError` at the
            same point.

    Examples:

        Construct inside ``async with``, then reach the control plane through the
        namespace properties. Leaving ``api_key`` off entirely reads it from
        ``PINECONE_API_KEY``:

        .. code-block:: python

            from pinecone import AsyncPinecone

            async with AsyncPinecone(api_key="your-api-key") as pc:
                if not await pc.indexes.exists("product-search"):
                    await pc.indexes.create(
                        name="product-search",
                        schema={"fields": {"embedding": {
                            "type": "dense_vector", "dimension": 1536,
                            "metric": "cosine"}}},
                    )

        :meth:`index` hands back a separate data-plane client scoped to one index,
        which is what reads and writes vectors. It manages its own connections, so
        it gets its own ``async with`` block. A query vector has to be as wide as
        the index's dense field — the three floats below stand in for a full
        1536-dimensional embedding:

        .. code-block:: python

            async with AsyncPinecone(api_key="your-api-key") as pc:
                idx = await pc.index(name="product-search")
                async with idx:
                    results = await idx.query(
                        vector=[0.012, -0.087, 0.153],
                        top_k=10,
                    )
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
        """Return a debug-friendly representation with the API key masked.

        Only the last four characters of the API key are shown, so it is safe
        to include in logs, tracebacks, or ``repr()`` output.

        Returns:
            A string like ``"AsyncPinecone(api_key='...ab12', host='https://api.pinecone.io')"``.
        """
        masked = f"...{self._config.api_key[-4:]}" if len(self._config.api_key) >= 4 else "***"
        return f"AsyncPinecone(api_key='{masked}', host='{self._config.host}')"

    @property
    def indexes(self) -> AsyncIndexes:
        """Create, inspect, configure, and delete the project's indexes.

        Returns:
            The :class:`~pinecone.async_client.indexes.AsyncIndexes` namespace.

        Examples:

            .. code-block:: python

                async with AsyncPinecone(api_key="your-api-key") as pc:
                    async for index in pc.indexes.list():
                        print(index.name, index.status.state)
        """
        if self._indexes is None:
            from pinecone.async_client.indexes import AsyncIndexes as _AsyncIndexes

            self._indexes = _AsyncIndexes(http=self._http, host_cache=self._host_cache)
        return self._indexes

    @property
    def collections(self) -> AsyncCollections:
        """Create and inspect collections: static snapshots of a pod-based index.

        A collection is the pod-based snapshot format. The serverless equivalent
        is a backup, under :attr:`backups`.

        Returns:
            The
            :class:`~pinecone.async_client.collections.AsyncCollections`
            namespace.

        Examples:

            .. code-block:: python

                async with AsyncPinecone(api_key="your-api-key") as pc:
                    for col in await pc.collections.list():
                        print(col.name, col.status)
        """
        if self._collections is None:
            from pinecone.async_client.collections import AsyncCollections as _AsyncCollections

            self._collections = _AsyncCollections(http=self._http)
        return self._collections

    @property
    def assistants(self) -> AsyncAssistants:
        """Create and manage Pinecone Assistants.

        An assistant is a hosted, retrieval-augmented chat service: upload
        files to it and it answers questions grounded in their content, with no
        index or embedding pipeline of your own.

        Returns:
            The
            :class:`~pinecone.async_client.assistants.AsyncAssistants`
            namespace.

        Examples:

            .. code-block:: python

                async with AsyncPinecone(api_key="your-api-key") as pc:
                    async for assistant in pc.assistants.list():
                        print(assistant.name, assistant.status)
        """
        if self._assistants is None:
            from pinecone.async_client.assistants import AsyncAssistants as _AsyncAssistants

            self._assistants = _AsyncAssistants(config=self._config)
        return self._assistants

    @property
    def assistant(self) -> _AsyncAssistantNamespaceProxy:
        """Reach one assistant by name, or the whole namespace by attribute.

        :attr:`assistants` is the canonical namespace; this singular alias is
        not deprecated. It forwards every attribute there, and awaiting it with
        a name is shorthand for
        :meth:`~pinecone.async_client.assistants.AsyncAssistants.describe`.

        Returns:
            A proxy that behaves like the :class:`AsyncAssistants` namespace
            for attribute access (``pc.assistant.create(...)``) and, when
            awaited with a name, returns that assistant's details.

        Examples:

            Awaiting the proxy with a name is shorthand for
            :meth:`~pinecone.async_client.assistants.AsyncAssistants.describe`:

            .. code-block:: python

                async with AsyncPinecone(api_key="your-api-key") as pc:
                    bot = await pc.assistant("acme-support-bot")
                    print(bot.status, bot.instructions)

            Every other attribute forwards to the plural namespace, so
            ``pc.assistant.create`` and ``pc.assistants.create`` are the same
            method reached two ways:

            .. code-block:: python

                async with AsyncPinecone(api_key="your-api-key") as pc:
                    new_bot = await pc.assistant.create(
                        name="acme-billing-bot",
                        instructions="Help users with billing questions.",
                    )
                    print(new_bot.status)
        """
        from pinecone.client._assistant_namespace_proxy import _AsyncAssistantNamespaceProxy

        return _AsyncAssistantNamespaceProxy(self.assistants)

    @property
    def backups(self) -> AsyncBackups:
        """Create, inspect, and delete backups of a serverless index.

        Restoring one is not done from here: pass a ``backup_id`` to
        :meth:`create_index_from_backup`, which creates a new index from it. For
        pod-based indexes the snapshot format is a collection, under
        :attr:`collections`.

        Returns:
            The :class:`~pinecone.async_client.backups.AsyncBackups` namespace.

        Examples:

            .. code-block:: python

                async with AsyncPinecone(api_key="your-api-key") as pc:
                    for backup in await pc.backups.list(limit=100):
                        print(backup.backup_id, backup.source_index_name, backup.status)
        """
        if self._backups is None:
            from pinecone.async_client.backups import AsyncBackups as _AsyncBackups

            self._backups = _AsyncBackups(http=self._http)
        return self._backups

    @property
    def backup_schedules(self) -> AsyncBackupSchedules:
        """Attach a recurring backup cadence to an index.

        A schedule gives an index a daily, weekly, or monthly cadence, so
        Pinecone creates each backup for you rather than you triggering one
        every time.

        Returns:
            The
            :class:`~pinecone.async_client.backup_schedules.AsyncBackupSchedules`
            namespace.

        Examples:

            .. code-block:: python

                async with AsyncPinecone(api_key="your-api-key") as pc:
                    for schedule in await pc.backup_schedules.list(
                        index_name="product-search"
                    ):
                        print(schedule.name, schedule.frequency, schedule.enabled)
        """
        if self._backup_schedules is None:
            from pinecone.async_client.backup_schedules import (
                AsyncBackupSchedules as _AsyncBackupSchedules,
            )

            self._backup_schedules = _AsyncBackupSchedules(http=self._http)
        return self._backup_schedules

    @property
    def restore_jobs(self) -> AsyncRestoreJobs:
        """Track a restore that :meth:`create_index_from_backup` started.

        A restore job is the request itself, so it is what to follow when you
        passed ``timeout=-1`` and the target index does not exist yet.

        Returns:
            The
            :class:`~pinecone.async_client.restore_jobs.AsyncRestoreJobs`
            namespace.

        Examples:

            .. code-block:: python

                async with AsyncPinecone(api_key="your-api-key") as pc:
                    for job in await pc.restore_jobs.list(limit=10):
                        print(job.restore_job_id, job.target_index_name, job.status)
        """
        if self._restore_jobs is None:
            from pinecone.async_client.restore_jobs import AsyncRestoreJobs as _AsyncRestoreJobs

            self._restore_jobs = _AsyncRestoreJobs(http=self._http)
        return self._restore_jobs

    @property
    def inference(self) -> AsyncInference:
        """Embed text or images, and rerank documents, on hosted models.

        Reach for this when you want vectors or relevance scores without
        hosting a model yourself.

        Returns:
            The :class:`~pinecone.async_client.inference.AsyncInference`
            namespace.

        Examples:

            ``multilingual-e5-large`` is asymmetric, so ``input_type`` tells it
            which side of a search the text belongs to — ``"passage"`` for text
            you intend to store, ``"query"`` for text you intend to search with:

            .. code-block:: python

                async with AsyncPinecone(api_key="your-api-key") as pc:
                    embeddings = await pc.inference.embed(
                        model="multilingual-e5-large",
                        inputs=["Solar panels reduce energy costs and lower carbon emissions."],
                        parameters={"input_type": "passage"},
                    )
                    print(len(embeddings.data))
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

        Polls until the restored index is ready, unless *timeout* is ``-1``. This
        is the only supported way to restore a backup: :meth:`create_index`
        rejects ``source_backup_id=`` with a message pointing here.

        Args:
            name (str): Name for the new index.
            backup_id (str): Identifier of the backup to restore from. Obtain it from
                :meth:`AsyncBackups.create
                <pinecone.async_client.backups.AsyncBackups.create>` or
                :meth:`AsyncBackups.list
                <pinecone.async_client.backups.AsyncBackups.list>`.
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

            .. code-block:: python

                from pinecone import AsyncPinecone

                async with AsyncPinecone(api_key="your-api-key") as pc:
                    index = await pc.create_index_from_backup(
                        name="product-search-restored",
                        backup_id="bk-abc123",
                    )
                    print(index.status.state)

            ``timeout=-1`` returns as soon as the restore is accepted. What
            comes back is a :class:`CreateIndexFromBackupResponse`, not an
            index — the index does not exist yet, so follow the restore through
            ``pc.restore_jobs`` rather than treating the return value as one:

            .. code-block:: python

                async with AsyncPinecone(api_key="your-api-key") as pc:
                    result = await pc.create_index_from_backup(
                        name="product-search-restored",
                        backup_id="bk-abc123",
                        timeout=-1,
                    )
                    job = await pc.restore_jobs.describe(job_id=result.restore_job_id)
                    print(job.status)

            A restore can land straight onto dedicated read nodes instead of
            the on-demand default:

            .. code-block:: python

                async with AsyncPinecone(api_key="your-api-key") as pc:
                    index = await pc.create_index_from_backup(
                        name="product-search-restored",
                        backup_id="bk-abc123",
                        read_capacity={
                            "mode": "Dedicated",
                            "dedicated": {
                                "node_type": "t1",
                                "scaling": "Manual",
                                "manual": {"shards": 2, "replicas": 2},
                            },
                        },
                    )
                    print(index.status.state)

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
        """Read back the settings this client resolved at construction.

        Returns:
            :class:`~pinecone._internal.config.PineconeConfig` carrying the
            resolved API key, host, timeout, and connection settings.

        Examples:

            The values are post-resolution, with defaults and environment
            variables folded in, so this is where to confirm which host a client
            is actually pointed at:

            .. code-block:: python

                async with AsyncPinecone(api_key="your-api-key") as pc:
                    print(pc.config.host, pc.config.timeout)
        """
        return self._config

    # ---- Backcompat flat-method delegates (:meta private:) ----

    async def create_index(
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
        """Backwards-compatibility shim for :meth:`AsyncPinecone.indexes.create`.

        Preserved to ease migration from the legacy (9.x) Pinecone Python
        SDK, with the same legacy parameter list — including its positional
        order, so ``await pc.create_index("movies", ServerlessSpec(...), 1536)``
        keeps working. New code should use
        ``await pc.indexes.create(...)`` instead, which additionally accepts
        the current ``schema=``/``deployment=``/``read_capacity=``/
        ``cmek_id=`` surface not available here. ``pods=``/``metadata_config=``/
        ``source_collection=``/``source_backup_id=`` reach
        :meth:`AsyncPinecone.indexes.create`, which raises for them.

        Args:
            name (str): Name for the index — 1-45 characters, lowercase
                alphanumerics and hyphens.
            spec (Any): A :class:`~pinecone.models.indexes.specs.ServerlessSpec`,
                :class:`~pinecone.models.indexes.specs.PodSpec`,
                :class:`~pinecone.models.indexes.specs.ByocSpec`, or equivalent
                dict, translated into ``deployment=`` by
                :meth:`AsyncPinecone.indexes.create`.
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
                :meth:`AsyncPinecone.indexes.create`, which rejects them.

        Returns:
            :class:`IndexModel` describing the created index.

        Raises:
            :exc:`PineconeTypeError`: If an unsupported legacy keyword is passed.
            :exc:`IndexInitFailedError`: If the index fails to initialize.
            :exc:`PineconeTimeoutError`: If the index isn't ready before
                *timeout* elapses.
            :exc:`ApiError`: If the API returns another error response.

        Examples:
            >>> async with AsyncPinecone(api_key="your-api-key") as pc:  # doctest: +SKIP
            ...     index = await pc.create_index(
            ...         name="movie-recommendations", dimension=1536, metric="cosine",
            ...     )

        :meta private:
        """
        reject_new_only_create_kwargs(legacy_kwargs)
        return await self.indexes.create(
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

    async def create_index_for_model(
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
        """Backwards-compatibility shim for :meth:`AsyncPinecone.indexes.create_for_model`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New
        code should use ``await pc.indexes.create_for_model(...)`` instead of
        ``await pc.create_index_for_model(...)``.

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
            >>> async with AsyncPinecone(api_key="your-api-key") as pc:  # doctest: +SKIP
            ...     index = await pc.create_index_for_model(
            ...         name="semantic-search",
            ...         cloud="aws",
            ...         region="us-east-1",
            ...         embed={"model": "multilingual-e5-large",
            ...                "field_map": {"text": "chunk_text"}},
            ...     )

        :meta private:
        """
        return await self.indexes.create_for_model(
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

    async def describe_index(self, name: str) -> IndexModel:
        """Backwards-compatibility shim for :meth:`AsyncPinecone.indexes.describe`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New code
        should use ``await pc.indexes.describe(...)`` instead of
        ``await pc.describe_index(...)``.

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
            >>> desc = await pc.describe_index("my-index")  # doctest: +SKIP
            >>> desc.host  # doctest: +SKIP
            'https://my-index.svc.pinecone.io'

        :meta private:
        """
        return await self.indexes.describe(name)

    async def list_indexes(self) -> IndexList:
        """Backwards-compatibility shim for :meth:`AsyncPinecone.indexes.list`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New
        code should use ``pc.indexes.list()`` instead of
        ``await pc.list_indexes()``, which returns an
        :class:`~pinecone.models.pagination.AsyncPaginator`.

        Returns:
            :class:`IndexList` wrapping every index in the project.

        Raises:
            :exc:`ApiError`: If the API returns an error response.

        Examples:
            >>> for index in await pc.list_indexes():  # doctest: +SKIP
            ...     print(index.name)

        :meta private:
        """
        return IndexList(await self.indexes.list().to_list())

    async def has_index(self, name: str) -> bool:
        """Backwards-compatibility shim for :meth:`AsyncPinecone.indexes.exists`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New code
        should use ``await pc.indexes.exists(...)`` instead of
        ``await pc.has_index(...)``.

        Args:
            name (str): The name of the index to check.

        Returns:
            True if the index exists, False otherwise.

        Raises:
            :exc:`PineconeValueError`: If *name* is empty.
            :exc:`ApiError`: If the API returns an error response other than a not-found error.

        :meta private:
        """
        return await self.indexes.exists(name)

    async def configure_index(
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
        """Backwards-compatibility shim for :meth:`AsyncPinecone.indexes.configure`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New
        code should use ``await pc.indexes.configure(...)`` instead of
        ``await pc.configure_index(...)``, which additionally accepts the
        current ``deployment=``/``schema=`` surface not available here.
        ``embed=`` reaches :meth:`AsyncPinecone.indexes.configure`, which
        raises for it: the convert-to-integrated flow it drove has no current
        equivalent.

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
        return await self.indexes.configure(
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

    async def delete_index(self, name: str, timeout: int | None = None) -> None:
        """Backwards-compatibility shim for :meth:`AsyncPinecone.indexes.delete`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New code
        should use ``await pc.indexes.delete(...)`` instead of
        ``await pc.delete_index(...)``.

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
        await self.indexes.delete(name, timeout=timeout)

    async def create_collection(self, name: str, source: str) -> CollectionModel:
        """Backwards-compatibility shim for :meth:`AsyncPinecone.collections.create`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New code
        should use ``await pc.collections.create(...)`` instead of
        ``await pc.create_collection(...)``.

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
        return await self.collections.create(name=name, source=source)

    async def list_collections(self) -> CollectionList:
        """Backwards-compatibility shim for :meth:`AsyncPinecone.collections.list`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New code
        should use ``await pc.collections.list()`` instead of
        ``await pc.list_collections()``.

        Returns:
            A :class:`CollectionList` supporting iteration, len(), index access,
            and a names() convenience method.

        :meta private:
        """
        return await self.collections.list()

    async def describe_collection(self, name: str) -> CollectionModel:
        """Backwards-compatibility shim for :meth:`AsyncPinecone.collections.describe`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New code
        should use ``await pc.collections.describe(...)`` instead of
        ``await pc.describe_collection(...)``.

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
        return await self.collections.describe(name)

    async def delete_collection(self, name: str) -> None:
        """Backwards-compatibility shim for :meth:`AsyncPinecone.collections.delete`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New code
        should use ``await pc.collections.delete(...)`` instead of
        ``await pc.delete_collection(...)``.

        Args:
            name (str): Name of the collection to delete.

        Raises:
            :exc:`PineconeValueError`: If *name* is empty.
            :exc:`NotFoundError`: If the collection does not exist.

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
        ``await pc.create_backup(...)``. Note the keyword is *backup_name*
        here, where :meth:`AsyncPinecone.backups.create` uses *name*.

        Args:
            index_name (str): Name of the index to back up.
            backup_name (str | None): Name for the backup, e.g.
                ``"daily-20240115"``. When omitted, the backup has no name
                and is identified only by its ``backup_id``.
            description (str | None): Description for the backup.

        Returns:
            A :class:`BackupModel` describing the new backup. The call
            returns once the backup is initiated; check its ``status`` via
            :meth:`AsyncPinecone.describe_backup` to see when it's ready.

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
        return await self.backups.describe(backup_id=backup_id)

    async def delete_backup(self, *, backup_id: str) -> None:
        """Backwards-compatibility shim for :meth:`AsyncPinecone.backups.delete`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New code
        should use ``await pc.backups.delete(...)`` instead of
        ``await pc.delete_backup(...)``.

        Args:
            backup_id (str): The identifier of the backup to delete.

        Raises:
            :exc:`PineconeValueError`: If *backup_id* is empty.
            :exc:`NotFoundError`: If the backup does not exist.
            :exc:`ApiError`: If the API returns another error response.

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
        ``await pc.list_restore_jobs(...)``. Returns a single page and does
        not auto-fetch further pages; see :meth:`AsyncPinecone.restore_jobs.list`
        for pagination details and a note on its known limitations.

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
        return await self.restore_jobs.list(
            limit=limit,
            pagination_token=pagination_token,
        )

    async def describe_restore_job(self, *, job_id: str) -> RestoreJobModel:
        """Backwards-compatibility shim for :meth:`AsyncPinecone.restore_jobs.describe`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New code
        should use ``await pc.restore_jobs.describe(...)`` instead of
        ``await pc.describe_restore_job(...)``. See
        :meth:`AsyncPinecone.restore_jobs.describe` for important caveats about
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
        return await self.restore_jobs.describe(job_id=job_id)

    def IndexAsyncio(self, host: str, **kwargs: Any) -> AsyncIndex:  # noqa: N802
        """Backwards-compatibility shim for :meth:`AsyncPinecone.index`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New
        code should use ``pc.index(host=...)`` (where ``pc`` is an
        :class:`AsyncPinecone` instance) instead of ``pc.IndexAsyncio(...)``.
        Unlike :meth:`AsyncPinecone.index`, this is a plain (non-``async def``)
        method that only accepts a host, not a name — it builds the client
        directly, without a describe call.

        Args:
            host (str): Direct host URL of the index.
            **kwargs: Unused; accepted for signature compatibility with the
                legacy SDK.

        Returns:
            An :class:`AsyncIndex` data-plane client, configured with this
            client's API key, timeout, proxy, and TLS settings.

        Examples:
            >>> from pinecone import AsyncPinecone
            >>> pc = AsyncPinecone(api_key="your-api-key")
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
        """Open an async data-plane client for one index, to read and write vectors.

        A coroutine: awaiting it is what keeps the host lookup off the event loop.
        An explicit *host* is used as-is, a *name* is served from this client's
        host cache, and a *name* that misses the cache costs one describe request.
        The sync twin, :meth:`Pinecone.index() <pinecone.Pinecone.index>`, is a
        plain call, and is the only one of the two that can return a gRPC client.

        Args:
            name (str): Name of the index, e.g. ``"product-search"``. Costs one
                describe request the first time, then comes from the host cache.
            host (str): The index's host, e.g.
                ``"product-search-abc123.svc.pinecone.io"``. Pass it when you have
                it already and the describe request is skipped entirely.

        Returns:
            An :class:`~pinecone.async_client.async_index.AsyncIndex`.

        Raises:
            :exc:`PineconeValueError`: If neither *name* nor *host* is given, or if
                *name* names an index that has no host yet — it is still
                initializing, so wait for its status to reach ``Ready``.
            :exc:`~pinecone.errors.NotFoundError`: If *name* names no index in this
                project.

        Examples:

            .. code-block:: python

                async with AsyncPinecone(api_key="your-api-key") as pc:
                    idx = await pc.index(name="product-search")
                    async with idx:
                        print(await idx.describe_index_stats())

            Passing the host skips the lookup, which saves a round trip when you
            already know it — from :meth:`AsyncIndexes.describe
            <pinecone.async_client.indexes.AsyncIndexes.describe>`, or from your
            own config:

            .. code-block:: python

                async with AsyncPinecone(api_key="your-api-key") as pc:
                    idx = await pc.index(host="product-search-abc123.svc.pinecone.io")
                    async with idx:
                        print(await idx.describe_index_stats())

        .. warning::
           The returned index manages its own HTTP client. Always use
           ``async with idx:`` or call ``await idx.close()`` when done — closing
           the parent :class:`AsyncPinecone` does not close index clients.

        .. seealso::
           :attr:`indexes` — the control-plane namespace, for creating, listing,
           describing, configuring, and deleting indexes rather than reading from
           one.
        """
        from pinecone.async_client.async_index import AsyncIndex as _AsyncIndex

        resolved_host = await self._resolve_index_host(name=name, host=host)
        return _AsyncIndex(
            **self._build_index_kwargs(resolved_host),
            _limiter_registry=self._limiter_registry,
        )

    async def close(self) -> None:
        """Release this client's control-plane connections.

        Closes the control-plane connection pool, plus the :attr:`inference` and
        :attr:`assistants` pools if those namespaces were used. Index clients from
        :meth:`index` hold their own connections and are not closed here. Prefer
        the async context manager form, ``async with AsyncPinecone(...) as pc:``,
        which calls this on the way out.

        Examples:
            The async context manager form closes the client on the way out,
            on an exception as well as on a normal exit:

            .. code-block:: python

                from pinecone import AsyncPinecone

                async with AsyncPinecone(api_key="your-api-key") as pc:
                    async for index in pc.indexes.list():
                        print(index.name)

            Close it yourself when the client has to outlive a single block:

            .. code-block:: python

                pc = AsyncPinecone(api_key="your-api-key")
                try:
                    print(await pc.indexes.exists("product-search"))
                finally:
                    await pc.close()
        """
        await self._http.close()
        if self._assistants is not None:
            await self._assistants.close()
        if self._inference is not None:
            await self._inference.close()

    async def __aenter__(self) -> AsyncPinecone:
        """Enter the async context manager, returning this client.

        Returns:
            This :class:`AsyncPinecone` instance.

        Examples:

            .. code-block:: python

                async with AsyncPinecone(api_key="your-api-key") as pc:
                    async for index in pc.indexes.list():
                        print(index.name)
        """
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Exit the async context manager, closing the client.

        Calls :meth:`close` to release open HTTP connections.

        Examples:

            .. code-block:: python

                async with AsyncPinecone(api_key="your-api-key") as pc:
                    async for index in pc.indexes.list():
                        print(index.name)
        """
        await self.close()

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
            from pinecone._client import (
                _REMOVED_CLIENT_ATTRIBUTES,
                _removed_client_attribute_message,
            )

            if name in _REMOVED_CLIENT_ATTRIBUTES:
                raise AttributeError(_removed_client_attribute_message(type(self).__name__, name))
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
