"""Asynchronous data plane client for a Pinecone index."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import quote

if TYPE_CHECKING:
    import pandas as pd  # type: ignore[import-untyped]

    from pinecone.async_client.documents import AsyncDocuments

from pinecone._internal.adapters.imports_adapter import ImportsAdapter
from pinecone._internal.adapters.vectors_adapter import VectorsAdapter, extract_response_info
from pinecone._internal.adaptive import _AdaptiveLimiterRegistry
from pinecone._internal.batching import validate_batch_size
from pinecone._internal.bulk import bulk_execute_async
from pinecone._internal.config import PineconeConfig, check_api_key_is_header_safe
from pinecone._internal.constants import DATA_PLANE_API_VERSION, DEFAULT_MAX_CONCURRENCY
from pinecone._internal.data_plane_helpers import (
    _build_search_records_body,
    _limiter_host_key,
    _validate_host,
    _vector_to_dict,
    _warn_if_plaintext_public_host,
)
from pinecone._internal.dataframe import _resolve_on_error, extract_records
from pinecone._internal.keyword_only import keyword_only_methods
from pinecone._internal.validation import (
    DELETE_EMPTY_FILTER_MESSAGE,
    FETCH_BY_METADATA_EMPTY_FILTER_MESSAGE,
    QUERY_TOP_K_MAX,
    UPDATE_EMPTY_FILTER_MESSAGE,
    require_creatable_namespace_name,
    require_delete_selectors,
    require_in_range,
    require_non_empty_filter,
    require_query_selectors,
    require_update_selectors,
    require_valid_fetch_by_metadata_limit,
    require_valid_id_prefix,
    require_valid_list_limit,
    require_valid_namespace_limit,
    require_valid_namespace_name,
    require_valid_namespace_prefix,
    require_valid_namespace_schema,
    require_valid_vector_id,
    require_valid_vector_ids,
)
from pinecone._internal.vector_factory import VectorFactory
from pinecone.errors.exceptions import PineconeTypeError, PineconeValueError, ValidationError
from pinecone.models.imports.list import ImportList
from pinecone.models.imports.model import ImportModel, StartImportResponse
from pinecone.models.namespaces.models import ListNamespacesResponse, NamespaceDescription
from pinecone.models.response_info import ResponseInfo
from pinecone.models.vectors.query_aggregator import QueryNamespacesResults, QueryResultsAggregator
from pinecone.models.vectors.responses import (
    DescribeIndexStatsResponse,
    FetchByMetadataResponse,
    FetchResponse,
    ListResponse,
    QueryResponse,
    UpdateResponse,
    UpsertRecordsResponse,
    UpsertResponse,
)
from pinecone.models.vectors.search import (
    RerankConfig,
    SearchInputs,
    SearchQuery,
    SearchRecordsResponse,
)
from pinecone.models.vectors.sparse import SparseValues
from pinecone.models.vectors.vector import Vector

logger = logging.getLogger(__name__)


@keyword_only_methods
class AsyncIndex:
    """Asynchronous data plane client targeting a specific Pinecone index.

    An index's data plane is where its records live, and this is the client that
    reads and writes them. Reach one with ``await pc.index(name="article-search")``,
    or construct it here from a host URL when you already know the host and want to
    skip the describe-index lookup that resolving a name costs. Every method is an
    ``async def``; :class:`~pinecone.index.Index` is the blocking twin.

    Only errors specific to a single method are documented on that method. For the
    exception hierarchy every method shares, see
    `Error handling <https://sdk.pinecone.io/python/guides/error-handling.html>`__.

    Args:
        host (str): The index-specific data plane host URL.
            A scheme you include is kept as given; ``"http://"`` is sent in
            cleartext and emits a :class:`RuntimeWarning` unless the host is a
            loopback or RFC 1918 IP address, ``localhost``, or ``*.localhost``.
        api_key (str | None): Pinecone API key. Falls back to ``PINECONE_API_KEY`` env var.
        additional_headers (Mapping[str, str] | None): Extra headers included in every request.
            A name matching one of the SDK's own headers, in any capitalization,
            replaces it.
        timeout (float): Request timeout in seconds. Defaults to ``30.0``.
        proxy_url (str | None): HTTP proxy URL for outgoing requests.
        ssl_ca_certs (str | None): Path to a CA certificate bundle for SSL verification.
        ssl_verify (bool): Whether to verify SSL certificates. Defaults to ``True``.
        source_tag (str | None): Tag appended to the User-Agent string for request attribution.
        connection_pool_maxsize (int): Maximum number of connections to keep in the pool.
            ``0`` (default) uses httpx defaults.

    Raises:
        :exc:`PineconeValueError`: If no API key can be resolved or the host is invalid,
            or the key is not a valid HTTP header value (blank, or with a trailing newline),
            or ``proxy_url`` has a scheme other than ``http``, ``https``, ``socks5`` or
            ``socks5h``, no host, an invalid port, surrounding whitespace, or an
            unencoded ``#``, ``/`` or ``?`` in the username or password.
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
                async with await pc.index(name="article-search") as idx:
                    stats = await idx.describe_index_stats()
                    print(stats.total_vector_count)

    .. seealso::
       :class:`~pinecone.index.Index` — the same surface, blocking.
       `Sync vs async clients <https://sdk.pinecone.io/python/guides/sync-vs-async.html>`__
       — which lane to pick.
    """

    def __init__(
        self,
        *,
        host: str,
        api_key: str | None = None,
        additional_headers: Mapping[str, str] | None = None,
        timeout: float = 30.0,
        proxy_url: str | None = None,
        proxy_headers: Mapping[str, str] | None = None,
        ssl_ca_certs: str | None = None,
        ssl_verify: bool = True,
        source_tag: str | None = None,
        connection_pool_maxsize: int = 0,
        _limiter_registry: _AdaptiveLimiterRegistry | None = None,
    ) -> None:
        # Resolve API key: explicit arg > env var (check BEFORE host per unified-ord-0001)
        resolved_key = api_key or os.environ.get("PINECONE_API_KEY", "")
        if not resolved_key:
            raise ValidationError(
                "No API key provided. Pass api_key='...' or set the "
                "PINECONE_API_KEY environment variable."
            )

        # Validate and normalize host
        self._host = _validate_host(host)
        _warn_if_plaintext_public_host(self._host)

        config = PineconeConfig(
            api_key=resolved_key,
            host=self._host,
            timeout=timeout,
            additional_headers=dict(additional_headers or {}),
            proxy_url=proxy_url or "",
            proxy_headers=dict(proxy_headers or {}),
            ssl_ca_certs=ssl_ca_certs,
            ssl_verify=ssl_verify,
            source_tag=source_tag or "",
            connection_pool_maxsize=connection_pool_maxsize,
        )
        check_api_key_is_header_safe(config.api_key)
        self._config = config
        self._limiter_registry = _limiter_registry

        from pinecone._internal.http_client import AsyncHTTPClient

        self._http = AsyncHTTPClient(config, DATA_PLANE_API_VERSION)
        self._adapter = VectorsAdapter()
        self._imports_adapter = ImportsAdapter()
        self._documents: AsyncDocuments | None = None

        logger.info("AsyncIndex client created for host %s", self._host)

    @property
    def host(self) -> str:
        """The data plane host URL for this index."""
        return self._host

    @property
    def documents(self) -> AsyncDocuments:
        """Entry point for document operations on a schema-based index.

        A schema-based index stores JSON records instead of raw vectors. Reach the
        document operations — ``upsert``, ``search``, ``fetch`` and the rest — through
        here; the vector methods on this class (:meth:`upsert`, :meth:`query`) are for
        a vector-based index. The same instance is returned on every access, and
        reaching it does not await.

        Returns:
            :class:`~pinecone.async_client.documents.AsyncDocuments` namespace instance.

        Examples:

            >>> from pinecone import AsyncIndex
            >>> idx = AsyncIndex(host="article-search-abc123.svc.pinecone.io", api_key="...")
            >>> idx.documents
            AsyncDocuments()

        .. seealso::
           :class:`~pinecone.async_client.documents.AsyncDocuments` — every document
           operation, with an example each.
        """
        if self._documents is None:
            from pinecone.async_client.documents import AsyncDocuments as _AsyncDocuments

            self._documents = _AsyncDocuments(http=self._http, host=self._host)
        return self._documents

    async def upsert_records(
        self,
        *,
        records: list[dict[str, Any]],
        namespace: str,
        timeout: float | None = None,
    ) -> UpsertRecordsResponse:
        """Upsert text records into a legacy integrated index (legacy Records API).

        Serves legacy integrated indexes, created with
        :meth:`~pinecone.client.indexes.Indexes.create_for_model`: the server embeds
        each record's text with the index's model. On a document index use
        :attr:`documents` instead. Upserts apply asynchronously, so a record may not
        be searchable immediately.

        Args:
            records (list[dict[str, Any]]): Records to write, each with an ``_id``
                (or ``id``) string plus the fields to store, e.g.
                ``{"_id": "article-101", "text": "Vector databases for search."}``.
                The index embeds the field its embed configuration maps and stores
                the rest with the record. A record giving both ``_id`` and ``id``
                keeps ``_id``.
            namespace (str): Target namespace, e.g. ``"articles-en"``. Required: the legacy
                Records API has no implicit default namespace, so pass ``"__default__"`` to
                write to it.
            timeout (float | None): Per-request timeout in seconds, e.g. ``10.0``.
                Overrides the client-level default for this call only.

        Returns:
            :class:`~pinecone.models.vectors.responses.UpsertRecordsResponse` whose
            ``record_count`` is the number of records the client sent.

        Raises:
            :exc:`PineconeValueError`: If *namespace* is not a non-empty string,
                *records* is empty, or a record has no ``_id``/``id`` field or one
                that is not a string. Raised before any request is made.

        Examples:
            .. code-block:: python

                response = await idx.upsert_records(
                    namespace="articles-en",
                    records=[
                        {"_id": "article-101", "text": "Vector databases for search."},
                        {"_id": "article-102", "text": "RAG combines search with LLMs."},
                    ],
                )
                print(response.record_count)

        .. seealso::
           - :meth:`search`: the read side of a legacy integrated index.
           - :meth:`upsert`: write vectors you embedded yourself.
        """
        if not isinstance(namespace, str):
            raise ValidationError("namespace must be a string")
        if not namespace or not namespace.strip():
            raise ValidationError("namespace must be a non-empty string")
        if not records:
            raise ValidationError("records must be a non-empty list")

        for i, record in enumerate(records):
            if "_id" not in record and "id" not in record:
                raise ValidationError(f"Record at index {i} must contain an '_id' or 'id' field")

        from pinecone._internal.http_client import _encode_ndjson

        normalized: list[dict[str, Any]] = []
        for i, record in enumerate(records):
            r = dict(record)  # shallow copy
            if "_id" not in r and "id" in r:
                r["_id"] = r.pop("id")
            elif "_id" in r and "id" in r:
                del r["id"]  # _id wins; drop the redundant 'id' key
            resolved_id = r.get("_id")
            if not isinstance(resolved_id, str):
                got = type(resolved_id).__name__
                raise ValidationError(f"Record at index {i}: '_id' must be a string, got {got!r}")
            normalized.append(r)

        ndjson_body = _encode_ndjson(normalized)

        logger.info("Upserting %d records into namespace %r (NDJSON)", len(records), namespace)
        response = await self._http.post(
            f"/records/namespaces/{quote(namespace, safe='')}/upsert",
            timeout=timeout,
            content=ndjson_body,
            headers={"Content-Type": "application/x-ndjson"},
        )
        result = UpsertRecordsResponse(record_count=len(records))
        result.response_info = extract_response_info(response)
        return result

    async def upsert(
        self,
        *,
        vectors: Sequence[
            Vector
            | tuple[str, Sequence[float]]
            | tuple[str, Sequence[float], Mapping[str, Any]]
            | Mapping[str, Any]
        ],
        namespace: str = "",
        batch_size: int | None = None,
        show_progress: bool = True,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        timeout: float | None = None,
        total_timeout: float | None = None,
    ) -> UpsertResponse:
        """Upsert a batch of vectors into a namespace.

        Serves indexes created with ``dimension``/``metric`` and legacy integrated
        indexes; on a document index use :attr:`documents` instead. A vector whose ID
        already exists in the namespace is overwritten. Each request is capped on both
        vector count and encoded size, and wide vectors or large metadata usually hit
        the size cap first, so pass ``batch_size`` to split a long sequence into
        requests that stay under both.

        Args:
            vectors: Vectors to upsert. Each element can be a
                :class:`~pinecone.models.vectors.vector.Vector`, a tuple of
                ``(id, values)`` or ``(id, values, metadata)``, or a dict with ``id``,
                ``values``, and optional ``sparse_values`` / ``metadata`` keys.
            namespace (str): Target namespace, e.g. ``"articles-en"``. Defaults to the
                index's default namespace.
            batch_size (int | None): Split *vectors* into chunks of this size and send
                one request per chunk, e.g. ``100``. ``None`` (default) sends every vector
                in one request. Must be a positive integer.
            show_progress (bool): When ``True`` (default) and ``tqdm`` is installed,
                display a progress bar that advances as batches complete. No effect when
                ``batch_size`` is ``None``.
            max_concurrency (int): Most batch requests this call keeps in flight at once,
                e.g. ``16``. Defaults to ``8``. Only used when ``batch_size`` is set. A
                process-wide limit per host also applies and holds concurrency lower when
                the host pushes back; see
                `Bulk ingest <https://sdk.pinecone.io/python/guides/bulk-ingest.html>`__.
            timeout (float | None): Timeout in seconds for one attempt of one request,
                e.g. ``30.0``. A batch is retried on its own under the client's retry
                policy, so this does not bound the whole call. ``None`` (default) uses
                the client-level default.
            total_timeout (float | None): Deadline in seconds for the whole batched
                operation, e.g. ``600.0``. Only used when ``batch_size`` is set. On expiry
                no further batches are submitted, batches already in flight are awaited,
                and unsent batches are reported in ``failed_items``. ``None`` (default)
                means no deadline. Zero or a negative value raises
                :exc:`~pinecone.errors.exceptions.PineconeValueError`.

        Returns:
            :class:`~pinecone.models.vectors.responses.UpsertResponse` with
            ``upserted_count``. With ``batch_size`` set, a failed batch does not raise:
            ``has_errors``, ``errors`` and ``failed_items`` report what did not land, and
            ``failed_items`` can be passed straight back to :meth:`upsert`.
            ``response_info`` then carries the aggregate LSN of the successful batches,
            or ``None`` if no LSN headers came back.

        Raises:
            :exc:`PineconeTypeError`: If a vector element is not one of the forms listed
                above.
            :exc:`PineconeValueError`: If a vector element is malformed, *batch_size* is
                not a positive integer, or *max_concurrency* is out of range.
            :exc:`~pinecone.errors.exceptions.PayloadTooLargeError`: If ``batch_size`` is
                ``None`` and the request exceeds the server's cap on encoded size. Set a
                ``batch_size`` and retry.
            :exc:`ApiError`: If ``batch_size`` is ``None`` and the request exceeds the
                server's cap on vectors per request.

        Examples:
            All three vector forms are interchangeable within one call. The values
            below are truncated to three floats for the page; pass your index's full
            dimension.

            .. code-block:: python

                from pinecone import Vector

                response = await idx.upsert(
                    vectors=[
                        Vector(id="article-101", values=[0.012, -0.087, 0.153]),
                        ("article-102", [0.045, 0.021, -0.064]),
                        {"id": "article-103", "values": [0.091, -0.032, 0.178]},
                    ],
                    namespace="articles-en",
                )
                print(response.upserted_count)

            For a sequence too long for one request (``embeddings`` below being your
            whole list of vectors), set ``batch_size`` and resend what did not land:

            .. code-block:: python

                response = await idx.upsert(
                    vectors=embeddings,
                    namespace="articles-en",
                    batch_size=100,
                )
                if response.has_errors:
                    await idx.upsert(vectors=response.failed_items, namespace="articles-en")

        .. seealso::
           - :meth:`upsert_records`: text in, embedded server-side, for a legacy
             integrated index.
           - :meth:`upsert_from_dataframe`: the same write from a pandas DataFrame,
             batched for you.
           - :meth:`start_import`: millions of vectors from cloud storage, server-side
             and asynchronous.
        """
        if batch_size is None:
            return await self._upsert_one_batch(
                vectors=vectors, namespace=namespace, timeout=timeout
            )

        validate_batch_size(batch_size)
        require_in_range("max_concurrency", max_concurrency, 1, 64)

        built = [VectorFactory.build(v) for v in vectors]
        items: list[dict[str, Any]] = [_vector_to_dict(v) for v in built]

        async def _operation(chunk: list[dict[str, Any]]) -> UpsertResponse:
            return await self._upsert_dict_batch(items=chunk, namespace=namespace, timeout=timeout)

        batch_result = await bulk_execute_async(
            items=items,
            operation=_operation,
            batch_size=batch_size,
            max_concurrency=max_concurrency,
            show_progress=show_progress,
            desc="Upserting",
            host=self._host,
            total_timeout=total_timeout,
        )

        synth_headers: dict[str, str] = {}
        if batch_result.response_info is not None:
            if batch_result.response_info.lsn_reconciled is not None:
                synth_headers["x-pinecone-lsn-reconciled"] = str(
                    batch_result.response_info.lsn_reconciled
                )
            if batch_result.response_info.lsn_committed is not None:
                synth_headers["x-pinecone-lsn-committed"] = str(
                    batch_result.response_info.lsn_committed
                )
        synth_response_info = ResponseInfo(raw_headers=synth_headers) if synth_headers else None
        return UpsertResponse(
            upserted_count=batch_result.successful_item_count,
            response_info=synth_response_info,
            total_item_count=batch_result.total_item_count,
            failed_item_count=batch_result.failed_item_count,
            total_batch_count=batch_result.total_batch_count,
            successful_batch_count=batch_result.successful_batch_count,
            failed_batch_count=batch_result.failed_batch_count,
            errors=batch_result.errors,
        )

    async def _upsert_one_batch(
        self,
        *,
        vectors: Sequence[
            Vector
            | tuple[str, Sequence[float]]
            | tuple[str, Sequence[float], Mapping[str, Any]]
            | Mapping[str, Any]
        ],
        namespace: str,
        timeout: float | None,
    ) -> UpsertResponse:
        built = [VectorFactory.build(v) for v in vectors]
        body: dict[str, Any] = {
            "vectors": [_vector_to_dict(v) for v in built],
        }
        if namespace:
            body["namespace"] = namespace

        logger.info("Upserting %d vectors into namespace %r", len(built), namespace)
        response = await self._http.post("/vectors/upsert", timeout=timeout, json=body)
        result = self._adapter.to_upsert_response(response.content)
        result.response_info = extract_response_info(response)
        logger.debug("Upserted %d vectors", result.upserted_count)
        return result

    async def _upsert_dict_batch(
        self,
        *,
        items: list[dict[str, Any]],
        namespace: str,
        timeout: float | None,
    ) -> UpsertResponse:
        body: dict[str, Any] = {"vectors": items}
        if namespace:
            body["namespace"] = namespace
        response = await self._http.post("/vectors/upsert", timeout=timeout, json=body)
        result = self._adapter.to_upsert_response(response.content)
        result.response_info = extract_response_info(response)
        return result

    async def upsert_from_dataframe(
        self,
        df: pd.DataFrame,
        namespace: str | None = None,
        batch_size: int = 500,
        show_progress: bool = True,
        timeout: float | None = None,
        *,
        max_concurrency: int | None = None,
        total_timeout: float | None = None,
        on_error: Literal["raise", "collect"] | None = None,
    ) -> UpsertResponse:
        """Upsert vectors from a pandas DataFrame, in batches.

        Serves indexes created with ``dimension``/``metric`` and legacy integrated
        indexes; on a document index use :attr:`documents` instead. Each row becomes
        one vector, and the rows are sent through :meth:`upsert` with ``batch_size``
        set, so a batch that fails is reported on the response rather than raised
        unless you pass ``on_error="raise"``. ``pandas`` is not an SDK dependency;
        install it yourself.

        Args:
            df: A ``pandas.DataFrame`` with ``id`` and ``values`` columns and, optionally,
                ``sparse_values`` and ``metadata`` columns. A missing cell (``None`` or
                ``NaN``) in an optional column leaves that field off the vector.
            namespace (str | None): Target namespace, e.g. ``"articles-en"``. ``None``
                (default) targets the index's default namespace.
            batch_size (int): Rows per upsert request, e.g. ``100``. Defaults to ``500``.
                Lower it for wide vectors or large metadata.
            show_progress (bool): When ``True`` (default) and ``tqdm`` is installed,
                display a progress bar that advances as batches complete.
            timeout (float | None): Timeout in seconds for one attempt of one batch's
                request, e.g. ``30.0``. A batch is retried on its own under the client's
                retry policy, so this does not bound the whole call. ``None`` (default)
                uses the client-level default.
            max_concurrency (int | None): Most batch requests this call keeps in flight
                at once, e.g. ``16``. ``None`` (default) uses ``8``. A process-wide limit
                per host also applies and holds concurrency lower when the host pushes
                back; see
                `Bulk ingest <https://sdk.pinecone.io/python/guides/bulk-ingest.html>`__.
            total_timeout (float | None): Deadline in seconds for the whole ingest, e.g.
                ``600.0``. On expiry no further batches are submitted, batches already in
                flight are awaited, and unsent rows are reported in ``failed_items``.
                ``None`` (default) means no deadline. Zero or a negative value raises
                :exc:`~pinecone.errors.exceptions.PineconeValueError`.
            on_error (str | None): What to do when some batches fail. ``"collect"``
                (``None`` behaves the same) returns the response with ``errors`` and
                ``failed_items`` filled in. This holds when every batch fails too:
                nothing raises, so check ``response.has_errors``. ``"raise"`` raises the
                lowest-indexed batch failure once every batch has settled, with the
                partial response on the exception's ``response`` attribute.

        Returns:
            :class:`~pinecone.models.vectors.responses.UpsertResponse` with
            ``upserted_count`` totalled across the batches that landed, plus
            ``has_errors``, ``errors`` and ``failed_items`` for the ones that did not.
            ``failed_items`` can be passed straight back to :meth:`upsert`.

        Raises:
            :exc:`RuntimeError`: If ``pandas`` is not installed.
            :exc:`PineconeValueError`: If *df* is not a ``pandas.DataFrame`` or lacks an
                ``id`` or ``values`` column, *batch_size* is not a positive integer,
                *max_concurrency* is out of range, or *on_error* is not ``"collect"`` or
                ``"raise"``.
            :exc:`PineconeTimeoutError`: If ``on_error="raise"`` and a batch exhausted
                its retries on timeout.

        Examples:
            A ``metadata`` column is optional; where it is present its dict lands on the
            vector as written. The values below are truncated to three floats for the
            page; pass your index's full dimension.

            .. code-block:: python

                import pandas as pd

                df = pd.DataFrame([
                    {
                        "id": "article-101",
                        "values": [0.012, -0.087, 0.153],
                        "metadata": {"topic": "science", "year": 2024},
                    },
                    {
                        "id": "article-102",
                        "values": [0.045, 0.021, -0.064],
                        "metadata": {"topic": "technology", "year": 2024},
                    },
                ])
                response = await idx.upsert_from_dataframe(
                    df,
                    namespace="articles-en",
                    batch_size=100,
                )
                if response.has_errors:
                    await idx.upsert(vectors=response.failed_items, namespace="articles-en")

        .. seealso::
           - :meth:`upsert`: the same batched write from a list of vectors, with no
             ``pandas`` dependency.
           - :meth:`start_import`: millions of vectors from cloud storage, server-side
             and asynchronous.
        """
        try:
            import pandas as pd
        except ImportError:
            raise RuntimeError(
                "pandas is required for upsert_from_dataframe, and is not a "
                "dependency of this SDK — it is only needed by this one method. "
                "Install it in your own environment: pip install pandas"
            ) from None

        if not isinstance(df, pd.DataFrame):
            raise PineconeValueError("df must be a pandas DataFrame")

        records: list[dict[str, Any]] = extract_records(df)

        resolved_on_error = _resolve_on_error(on_error)

        response = await self.upsert(
            vectors=records,
            namespace=namespace or "",
            batch_size=batch_size,
            show_progress=show_progress,
            max_concurrency=(
                DEFAULT_MAX_CONCURRENCY if max_concurrency is None else max_concurrency
            ),
            timeout=timeout,
            total_timeout=total_timeout,
        )

        if resolved_on_error == "raise" and response.errors:
            error = min(response.errors, key=lambda err: err.batch_index).error
            error.response = response  # type: ignore[attr-defined]
            raise error

        return response

    async def query(
        self,
        *,
        top_k: int,
        vector: Sequence[float] | None = None,
        id: str | None = None,
        namespace: str = "",
        filter: Mapping[str, Any] | None = None,
        include_values: bool = False,
        include_metadata: bool = False,
        sparse_vector: SparseValues | Mapping[str, Any] | None = None,
        scan_factor: float | None = None,
        max_candidates: int | None = None,
        timeout: float | None = None,
    ) -> QueryResponse:
        """Query a namespace for the nearest neighbors of a vector you supply.

        Serves indexes created with ``dimension``/``metric`` and legacy integrated
        indexes; on a document index use :attr:`documents` instead. You supply the query vector;
        nothing is embedded for you. Pass a dense *vector*, a *sparse_vector*, both for
        a hybrid query, or the *id* of a stored vector to use as the query.

        Args:
            top_k (int): Number of results to return, e.g. ``5``. The server caps it.
            vector (list[float] | None): Dense query vector, at your index's
                dimension.
            id (str | None): ID of a stored vector to use as the query, e.g.
                ``"article-101"``. Cannot be combined with *vector* or
                *sparse_vector*.
            namespace (str): Namespace to query, e.g. ``"articles-en"``. Defaults
                to the index's default namespace.
            filter (dict[str, Any] | None): Metadata filter expression restricting
                which vectors are searched, e.g. ``{"year": {"$gte": 2020}}``.
            include_values (bool): Return each match's vector values. ``False``
                (default) keeps the response small.
            include_metadata (bool): Return each match's metadata. Set it when you
                need the fields you filtered on back in the result.
            sparse_vector (SparseValues | dict[str, Any] | None): Sparse query vector
                with indices and values, e.g.
                ``{"indices": [17, 42], "values": [0.4, 0.9]}``. Combine it with
                *vector* for a hybrid query on a dense index with the
                ``dotproduct`` metric.
            scan_factor (float | None): Recall/latency trade for dense indexes on
                dedicated read nodes: a multiplier on how much of the index is
                scanned. Above 1 scans more and favors recall; below 1 scans less
                and favors latency. Omit to let the server choose.
            max_candidates (int | None): Recall/latency trade for dense indexes on
                dedicated read nodes: caps how many candidates are reranked before
                *top_k* is taken. Must be at least *top_k*.
            timeout (float | None): Per-request timeout in seconds. Overrides the
                client-level default for this call only.

        Returns:
            :class:`~pinecone.models.vectors.responses.QueryResponse` with ``matches``
            (ordered from most to least similar, each carrying ``id`` and ``score``),
            ``namespace``, and ``usage``.

        Raises:
            :exc:`PineconeValueError`: If *top_k* is not positive or exceeds the
                server's maximum, if *id* is combined with *vector* or
                *sparse_vector*, if none of *vector*, *id*, or *sparse_vector* is
                given, or if *id* is not a legal vector ID. Raised before any
                request is sent.
            :exc:`ApiError`: If *scan_factor* or *max_candidates* is set on an
                index that does not run on dedicated read nodes, or on a sparse
                index.

        Examples:
            Query vectors are truncated to three floats on this page; pass your
            index's full dimension.

            .. code-block:: python

                response = await idx.query(
                    top_k=5,
                    vector=[0.012, -0.087, 0.153],
                    namespace="articles-en",
                )
                for match in response.matches:
                    print(match.id, match.score)

            A ``filter`` narrows the search before ranking, and ``include_metadata``
            returns the fields it selected on:

            .. code-block:: python

                response = await idx.query(
                    top_k=5,
                    vector=[0.012, -0.087, 0.153],
                    namespace="movies-en",
                    filter={"genre": "comedy", "year": {"$gte": 2020}},
                    include_metadata=True,
                )
                for match in response.matches:
                    print(match.id, match.score, match.metadata["genre"])

        .. seealso::
           - :meth:`search`: the same search on a legacy integrated index, where
             you pass text and the server embeds it.
           - :meth:`query_namespaces`: the same query fanned out over several
             namespaces, merged into one ranking.
        """
        require_in_range("top_k", top_k, 1, QUERY_TOP_K_MAX)
        require_query_selectors(vector=vector, id=id, sparse_vector=sparse_vector)
        if id is not None:
            require_valid_vector_id("id", id)

        body: dict[str, Any] = {
            "topK": top_k,
            "includeValues": include_values,
            "includeMetadata": include_metadata,
        }
        if namespace:
            body["namespace"] = namespace
        if vector is not None:
            body["vector"] = vector
        if id is not None:
            body["id"] = id
        if filter is not None:
            body["filter"] = filter
        if sparse_vector is not None:
            if isinstance(sparse_vector, SparseValues):
                body["sparseVector"] = {
                    "indices": sparse_vector.indices,
                    "values": sparse_vector.values,
                }
            else:
                body["sparseVector"] = sparse_vector
        if scan_factor is not None:
            body["scanFactor"] = scan_factor
        if max_candidates is not None:
            body["maxCandidates"] = max_candidates

        logger.info("Querying index with top_k=%d", top_k)
        response = await self._http.post("/query", timeout=timeout, json=body)
        result = self._adapter.to_query_response(response.content)
        result.response_info = extract_response_info(response)
        logger.debug("Query returned %d matches", len(result.matches))
        return result

    async def query_namespaces(
        self,
        *,
        vector: Sequence[float] | None = None,
        namespaces: Sequence[str],
        metric: str,
        top_k: int | None = None,
        filter: Mapping[str, Any] | None = None,
        include_values: bool = False,
        include_metadata: bool = False,
        sparse_vector: SparseValues | Mapping[str, Any] | None = None,
        scan_factor: float | None = None,
        max_candidates: int | None = None,
        timeout: float | None = None,
    ) -> QueryNamespacesResults:
        """Query several namespaces at once and merge them into one ranking.

        Serves indexes created with ``dimension``/``metric`` and legacy integrated
        indexes. Runs one :meth:`query` per namespace concurrently, then merges the
        results so you get the overall top-k rather than top-k per namespace. The merge
        ranks by *metric*, which the client does not read off the index, so pass the
        metric the index was created with.

        Args:
            vector (list[float] | None): Dense query vector. Omit it on a sparse
                index and pass *sparse_vector* instead.
            namespaces (list[str]): Namespaces to query, e.g.
                ``["articles-en", "articles-fr"]``. Must be non-empty; duplicates
                are removed while preserving order.
            metric (str): Metric the index was created with: ``"cosine"``,
                ``"euclidean"``, or ``"dotproduct"``. A wrong metric is not
                rejected; the merged ranking just comes out in the wrong order.
            top_k (int | None): Number of results to return after merging, e.g.
                ``10``. Defaults to 10. Each namespace is queried for this many.
            filter (dict[str, Any] | None): Metadata filter expression applied to
                every namespace, e.g. ``{"year": {"$gte": 2020}}``.
            include_values (bool): Return each match's vector values.
            include_metadata (bool): Return each match's metadata.
            sparse_vector (SparseValues | dict[str, Any] | None): Sparse query vector
                with indices and values. Required on a sparse index, where *vector*
                is omitted.
            scan_factor (float | None): Recall/latency trade for dense indexes on
                dedicated read nodes, applied to every namespace. See :meth:`query`.
            max_candidates (int | None): Candidate cap for dense indexes on dedicated
                read nodes, applied per namespace. Must be at least *top_k*.
            timeout (float | None): Per-request timeout in seconds, applied to each
                namespace's query rather than to the call as a whole.

        Returns:
            :class:`~pinecone.models.vectors.query_aggregator.QueryNamespacesResults`
            with the merged ``matches``, ``usage`` totalled over every namespace, and
            ``ns_usage`` keyed by namespace name. A match does not record which
            namespace produced it, so query one namespace at a time when you need
            that.

        Raises:
            :exc:`PineconeValueError`: If *namespaces* is empty, if both *vector*
                and *sparse_vector* are absent or empty, or if *metric* is not one
                of ``"cosine"``, ``"euclidean"``, or ``"dotproduct"``. Raised
                before any request is sent.
            :exc:`ApiError`: If any one namespace's query fails. The first failure
                propagates and the merged result is lost, so retry the whole call.

        Examples:
            The query vector is truncated to three floats on this page; pass your
            index's full dimension.

            .. code-block:: python

                results = await idx.query_namespaces(
                    vector=[0.012, -0.087, 0.153],
                    namespaces=["articles-en", "articles-fr", "articles-de"],
                    metric="cosine",
                    top_k=10,
                )
                for match in results.matches:
                    print(match.id, match.score)

            On a sparse index, pass *sparse_vector* instead and rank by
            ``"dotproduct"``:

            .. code-block:: python

                results = await idx.query_namespaces(
                    sparse_vector={"indices": [17, 42, 108], "values": [0.4, 0.9, 0.2]},
                    namespaces=["articles-en", "articles-fr"],
                    metric="dotproduct",
                    top_k=10,
                )

        .. seealso::
           :meth:`query`: one namespace, and the only form that takes an *id* as the
           query.
        """
        if not namespaces:
            raise ValidationError("namespaces must be a non-empty list")
        if not vector and not sparse_vector:
            raise ValidationError("at least one of 'vector' or 'sparse_vector' must be provided")

        valid_metrics = {"cosine", "euclidean", "dotproduct"}
        if metric not in valid_metrics:
            raise ValidationError(
                f"Invalid metric {metric!r}. Must be one of: {', '.join(sorted(valid_metrics))}"
            )

        namespaces = list(dict.fromkeys(namespaces))
        effective_top_k = top_k if top_k is not None else 10
        aggregator = QueryResultsAggregator(metric=metric, top_k=effective_top_k)

        query_kwargs: dict[str, Any] = {
            "top_k": effective_top_k,
            "filter": filter,
            "include_values": include_values,
            "include_metadata": include_metadata,
            "sparse_vector": sparse_vector,
            "scan_factor": scan_factor,
            "max_candidates": max_candidates,
            "timeout": timeout,
        }
        if vector is not None:
            query_kwargs["vector"] = vector

        async def _query_ns(ns: str) -> tuple[str, QueryResponse]:
            result = await self.query(namespace=ns, **query_kwargs)
            return (ns, result)

        # Limiter-bounded fan-out. Internal ceiling = 10 (hardcoded by design;
        # users hitting this limit can split their call across multiple invocations).
        _internal_concurrency_ceiling = 10
        limiter = (
            self._limiter_registry.get(_limiter_host_key(self._host), _internal_concurrency_ceiling)
            if self._limiter_registry is not None
            else None
        )

        if limiter is None:
            sem = asyncio.Semaphore(_internal_concurrency_ceiling)

            async def _gated_sem(idx: int, ns: str) -> tuple[int, str, QueryResponse]:
                async with sem:
                    ns_back, resp = await _query_ns(ns)
                    return idx, ns_back, resp

            indexed = await asyncio.gather(*[_gated_sem(i, ns) for i, ns in enumerate(namespaces)])
        else:
            inflight = 0
            inflight_lock = asyncio.Lock()

            async def _gated_lim(idx: int, ns: str) -> tuple[int, str, QueryResponse]:
                nonlocal inflight
                while True:
                    async with inflight_lock:
                        if inflight < limiter.current_limit():
                            inflight += 1
                            break
                    await asyncio.sleep(0.05)
                try:
                    ns_back, resp = await _query_ns(ns)
                    return idx, ns_back, resp
                finally:
                    async with inflight_lock:
                        inflight -= 1

            indexed = await asyncio.gather(*[_gated_lim(i, ns) for i, ns in enumerate(namespaces)])

        indexed.sort(key=lambda t: t[0])
        for _idx, ns, response in indexed:
            aggregator.add_results(ns, response)

        return aggregator.get_results()

    async def fetch(
        self,
        *,
        ids: Sequence[str],
        namespace: str = "",
        timeout: float | None = None,
    ) -> FetchResponse:
        """Fetch vectors by their IDs, exactly as stored.

        Serves indexes created with ``dimension``/``metric`` and legacy integrated
        indexes; on a document index use :attr:`documents` instead. A lookup, not a
        search: nothing is ranked and no score is returned. An ID that is not in the
        namespace is silently absent from the result, so compare the keys you got back
        against the ones you asked for.

        Args:
            ids (list[str]): Vector IDs to fetch, e.g. ``["article-101", "article-102"]``.
                Must be non-empty.
            namespace (str): Namespace to fetch from, e.g. ``"articles-en"``. Defaults
                to the index's default namespace.
            timeout (float | None): Per-request timeout in seconds. Overrides the
                client-level default for this call only.

        Returns:
            :class:`~pinecone.models.vectors.responses.FetchResponse` with ``vectors``,
            a map of ID to :class:`~pinecone.models.vectors.vector.Vector` holding values
            and metadata as stored, plus ``namespace`` and ``usage``. IDs the namespace
            does not hold are absent from the map rather than raising.

        Raises:
            :exc:`PineconeValueError`: If *ids* is empty or holds an ID that is empty,
                non-ASCII, contains a NUL, or is longer than the API allows. Raised
                before any request is sent.

        Examples:
            .. code-block:: python

                wanted = ["article-101", "article-102"]
                response = await idx.fetch(ids=wanted, namespace="articles-en")
                for vid, vec in response.vectors.items():
                    print(vid, vec.metadata)
                print("not in this namespace:", set(wanted) - set(response.vectors))

        .. seealso::
           - :meth:`fetch_by_metadata`: when you know the metadata you want rather
             than the IDs.
           - :meth:`query`: when you want the nearest vectors rather than named ones.
        """
        require_valid_vector_ids("ids", ids)

        params: dict[str, Any] = {"ids": ids}
        if namespace:
            params["namespace"] = namespace

        logger.info("Fetching %d vectors", len(ids))
        response = await self._http.get("/vectors/fetch", timeout=timeout, params=params)
        result = self._adapter.to_fetch_response(response.content)
        result.response_info = extract_response_info(response)
        logger.debug("Fetched %d vectors", len(result.vectors))
        return result

    async def fetch_by_metadata(
        self,
        *,
        filter: Mapping[str, Any],
        namespace: str = "",
        limit: int | None = None,
        pagination_token: str | None = None,
        timeout: float | None = None,
    ) -> FetchByMetadataResponse:
        """Fetch one page of the vectors whose metadata matches a filter.

        Serves indexes created with ``dimension``/``metric`` and legacy integrated
        indexes; on a document index use :attr:`documents` instead. A lookup, not a
        search: matches are not ranked and carry no score. One page is returned per
        call, so follow ``pagination.next`` to reach the rest.

        Args:
            filter (dict[str, Any]): Metadata filter expression, e.g.
                ``{"year": {"$gte": 2020}}``. Must carry at least one condition; an
                empty filter is rejected rather than treated as "match everything".
            namespace (str): Namespace to fetch from, e.g. ``"movies-en"``. Defaults to
                the index's default namespace.
            limit (int | None): Maximum number of vectors in this page, e.g. ``100``.
                Omit to let the server choose the page size.
            pagination_token (str | None): ``pagination.next`` from the previous
                response. ``None`` (default) fetches the first page.
            timeout (float | None): Per-request timeout in seconds. Overrides the
                client-level default for this call only.

        Returns:
            :class:`~pinecone.models.vectors.responses.FetchByMetadataResponse` with
            ``vectors`` as stored, plus ``namespace``, ``usage``, and ``pagination``,
            which holds the next page's token in ``next`` and is ``None`` on the last
            page.

        Raises:
            :exc:`PineconeValueError`: If *filter* is empty or *limit* is not a
                positive integer within the API's maximum page size. Raised before any
                request is sent.

        Examples:
            .. code-block:: python

                response = await idx.fetch_by_metadata(
                    filter={"genre": "comedy", "year": {"$gte": 2020}},
                    namespace="movies-en",
                )
                for vid, vec in response.vectors.items():
                    print(vid, vec.metadata)

        .. seealso::
           - :meth:`fetch`: when you already know the IDs.
           - :meth:`query`: when you want the nearest vectors to a query rather than
             every vector a filter admits.
           - `Pagination <https://sdk.pinecone.io/python/guides/pagination.html>`__:
             walking every page.
        """
        if limit is not None:
            require_valid_fetch_by_metadata_limit("limit", limit)
        require_non_empty_filter(
            "filter", filter, server_message=FETCH_BY_METADATA_EMPTY_FILTER_MESSAGE
        )
        body: dict[str, Any] = {"filter": filter}
        if namespace:
            body["namespace"] = namespace
        if limit is not None:
            body["limit"] = limit
        if pagination_token is not None:
            body["paginationToken"] = pagination_token

        logger.info("Fetching vectors by metadata")
        response = await self._http.post("/vectors/fetch_by_metadata", timeout=timeout, json=body)
        result = self._adapter.to_fetch_by_metadata_response(response.content)
        result.response_info = extract_response_info(response)
        return result

    async def delete(
        self,
        *,
        ids: Sequence[str] | None = None,
        delete_all: bool = False,
        filter: Mapping[str, Any] | None = None,
        namespace: str = "",
        timeout: float | None = None,
    ) -> None:
        """Delete vectors from a namespace by ID, by filter, or all of them.

        Serves indexes created with ``dimension``/``metric`` and legacy integrated
        indexes; on a document index use :attr:`documents` instead. Pass exactly one
        selector: *ids*, *filter*, or ``delete_all=True``. The delete is irreversible,
        and IDs the namespace does not hold are ignored rather than reported, so a
        successful call is not evidence anything was deleted.

        Args:
            ids (list[str] | None): Vector IDs to delete, e.g.
                ``["article-101", "article-102"]``.
            delete_all (bool): Delete every vector in *namespace*. The namespace itself
                survives; :meth:`delete_namespace` removes that too.
            filter (dict[str, Any] | None): Metadata filter expression selecting what
                to delete, e.g. ``{"status": {"$eq": "retracted"}}``. Must carry at
                least one condition. The client refuses *ids* alongside *filter*, because
                the server would apply the filter alone and delete everything it matches
                rather than the intersection; to delete the intersection, :meth:`query`
                with the filter first and delete the IDs you get back.
            namespace (str): Namespace to delete from, e.g. ``"articles-en"``. Defaults
                to the index's default namespace.
            timeout (float | None): Per-request timeout in seconds. Overrides the
                client-level default for this call only.

        Returns:
            None. A successful delete returns no payload.

        Raises:
            :exc:`PineconeValueError`: If zero or more than one selector is given, if
                *filter* is empty, or if an ID is not a legal vector ID. Raised before
                any request is sent.

        Examples:
            By ID:

            .. code-block:: python

                await idx.delete(
                    ids=["article-101", "article-102"],
                    namespace="articles-en",
                )

            By metadata filter, which deletes every vector the filter admits:

            .. code-block:: python

                await idx.delete(
                    filter={"status": {"$eq": "retracted"}},
                    namespace="articles-en",
                )

            Emptying a whole namespace cannot be undone:

            .. code-block:: python

                await idx.delete(delete_all=True, namespace="articles-staging")

        .. seealso::
           :meth:`delete_namespace`: removes the namespace along with everything in it,
           where ``delete_all=True`` empties it and leaves it in place.
        """
        require_delete_selectors(ids=ids, delete_all=delete_all, filter=filter)
        if ids is not None:
            require_valid_vector_ids("ids", ids)
        if filter is not None:
            require_non_empty_filter("filter", filter, server_message=DELETE_EMPTY_FILTER_MESSAGE)

        body: dict[str, Any] = {"namespace": namespace}
        if ids is not None:
            body["ids"] = ids
        if delete_all:
            body["deleteAll"] = True
        if filter is not None:
            body["filter"] = filter

        logger.info("Deleting vectors from namespace %r", namespace)
        await self._http.post("/vectors/delete", timeout=timeout, json=body)

    async def update(
        self,
        *,
        id: str | None = None,
        values: Sequence[float] | None = None,
        sparse_values: SparseValues | Mapping[str, Any] | None = None,
        set_metadata: Mapping[str, Any] | None = None,
        namespace: str = "",
        filter: Mapping[str, Any] | None = None,
        dry_run: bool = False,
        timeout: float | None = None,
    ) -> UpdateResponse:
        """Patch one vector by ID, or patch metadata across a filter.

        Serves indexes created with ``dimension``/``metric`` and legacy integrated
        indexes; on a document index use :attr:`documents` instead. A partial update:
        fields you do not mention keep the values they had, so
        ``set_metadata={"year": 2021}`` leaves every other metadata key in place. Pass
        exactly one selector, *id* or *filter*; a by-filter update is metadata-only,
        since *values* and *sparse_values* belong to one record. The write applies
        asynchronously, so a read straight afterwards can still see the old value.

        Args:
            id (str | None): ID of the one vector to patch, e.g. ``"article-101"``.
            values (list[float] | None): Replacement dense values, at your index's
                dimension. Only with *id*.
            sparse_values (SparseValues | dict[str, Any] | None): Replacement sparse
                vector, with ``indices`` and ``values``. Only with *id*.
            set_metadata (dict[str, Any] | None): Metadata keys to set or overwrite,
                e.g. ``{"year": 2021}``. Keys you omit are left as they are; this never
                clears a field.
            namespace (str): Namespace to target, e.g. ``"movies-en"``. Defaults to the
                index's default namespace.
            filter (dict[str, Any] | None): Metadata filter expression selecting which
                vectors to patch, e.g. ``{"genre": {"$eq": "drama"}}``. Must carry at
                least one condition.
            dry_run (bool): Report how many records the *filter* would touch without
                writing anything. Ignored for a by-ID update. Run it first when the
                filter is broader than you can check by eye.
            timeout (float | None): Per-request timeout in seconds. Overrides the
                client-level default for this call only.

        Returns:
            :class:`~pinecone.models.vectors.responses.UpdateResponse` whose
            ``matched_records`` counts the records the filter matched, which under
            ``dry_run`` are the records that would have been patched. It is ``None``
            for a by-ID update, which reports no count.

        Raises:
            :exc:`PineconeValueError`: If both or neither of *id* and *filter* are
                given, if *filter* is combined with *values* or *sparse_values*, if
                *filter* is empty, or if *id* is not a legal vector ID. Raised before
                any request is sent.

        Examples:
            Replacing one vector's values, truncated here to three floats:

            .. code-block:: python

                await idx.update(
                    id="article-101",
                    values=[0.012, -0.087, 0.153],
                    namespace="articles-en",
                )

            Patching metadata across a filter. ``dry_run`` reports the reach first,
            and the ``genre`` of every patched record survives untouched:

            .. code-block:: python

                preview = await idx.update(
                    filter={"genre": {"$eq": "drama"}},
                    set_metadata={"reviewed": True},
                    namespace="movies-en",
                    dry_run=True,
                )
                print(preview.matched_records)
                await idx.update(
                    filter={"genre": {"$eq": "drama"}},
                    set_metadata={"reviewed": True},
                    namespace="movies-en",
                )

        .. seealso::
           :meth:`upsert`: replaces a whole vector rather than patching it, and
           creates it if the ID is new.
        """
        require_update_selectors(id=id, filter=filter, values=values, sparse_values=sparse_values)
        if id is not None:
            require_valid_vector_id("id", id)
        if filter is not None:
            require_non_empty_filter("filter", filter, server_message=UPDATE_EMPTY_FILTER_MESSAGE)

        body: dict[str, Any] = {"namespace": namespace}
        if id is not None:
            body["id"] = id
        if values is not None:
            body["values"] = values
        if sparse_values is not None:
            if isinstance(sparse_values, SparseValues):
                body["sparseValues"] = {
                    "indices": sparse_values.indices,
                    "values": sparse_values.values,
                }
            else:
                body["sparseValues"] = sparse_values
        if set_metadata is not None:
            body["setMetadata"] = set_metadata
        if filter is not None:
            body["filter"] = filter
        if dry_run:
            body["dryRun"] = True

        logger.info("Updating vectors in namespace %r", namespace)
        response = await self._http.post("/vectors/update", timeout=timeout, json=body)
        result = self._adapter.to_update_response(response.content)
        result.response_info = extract_response_info(response)
        return result

    async def search(
        self,
        *,
        namespace: str,
        top_k: int | None = None,
        inputs: SearchInputs | Mapping[str, Any] | None = None,
        vector: Sequence[float] | Mapping[str, Any] | None = None,
        id: str | None = None,
        filter: Mapping[str, Any] | None = None,
        fields: Sequence[str] | None = None,
        rerank: RerankConfig | Mapping[str, Any] | None = None,
        match_terms: Mapping[str, Any] | None = None,
        query: SearchQuery | Mapping[str, Any] | None = None,
        timeout: float | None = None,
    ) -> SearchRecordsResponse:
        """Search records by text, vector, or ID, optionally reranking (legacy Records API).

        Give exactly one query source. Text in *inputs* needs a legacy integrated
        index, created with
        :meth:`~pinecone.client.indexes.Indexes.create_for_model`, whose model
        embeds it server-side. A *vector* or *id* query also works on an index
        created with ``dimension``/``metric``, though :meth:`query` is the usual
        call there. On a document index use :attr:`documents` instead.

        Args:
            namespace (str): Namespace to search, e.g. ``"articles-en"``. Required: pass
                ``"__default__"`` to search the default namespace.
            top_k (int): Number of results to return, at least 1, e.g. ``10``. Required
                unless *query* is given.
            inputs (SearchInputs | dict[str, Any] | None): Text for the index's model to
                embed, e.g. ``{"text": "trail running shoes"}``.
                :class:`~pinecone.models.vectors.search.SearchInputs` gives typed keys and
                IDE completion.
            vector (list[float] | dict[str, Any] | None): Query vector. A
                ``list[float]`` is a dense query; a dict with ``values``,
                ``sparse_indices`` and/or ``sparse_values`` is a sparse or hybrid one.
                :class:`~pinecone.models.vectors.search.SearchQueryVector` is the typed
                helper.
            id (str | None): ID of a stored record whose vector is the query, e.g.
                ``"article-101"``.
            filter (dict[str, Any] | None): Metadata filter, e.g.
                ``{"category": {"$eq": "footwear"}}``.
            fields (list[str] | None): Record fields to return in each hit, e.g.
                ``["title", "text"]``. ``None`` returns every stored field.
            rerank (RerankConfig | dict[str, Any] | None): Rerank the hits in the same call.
                Needs ``model`` and ``rank_fields``; ``top_n``, ``parameters`` and
                ``query`` are optional.
                :class:`~pinecone.models.vectors.search.RerankConfig` gives IDE completion.
            match_terms (dict[str, Any] | None): Require hits to contain query terms, e.g.
                ``{"strategy": "all", "terms": ["running", "shoes"]}``. Valid only with
                *inputs*, on a sparse index whose model supports it; the server names
                the supported model when it refuses.
            timeout (float | None): Per-request timeout in seconds, e.g. ``10.0``.
                Overrides the client-level default for this call only.
            query (SearchQuery | dict[str, Any] | None): The older nested form of this
                call: ``top_k`` plus one query source in one mapping. Pass the fields
                directly instead.

        Returns:
            :class:`~pinecone.models.vectors.search.SearchRecordsResponse` whose
            ``result.hits`` run from most to least relevant. Read each
            :class:`~pinecone.models.vectors.search.Hit` as ``hit.id``, ``hit.score``
            and ``hit.fields``. ``usage`` reports read units, plus embedding tokens
            and rerank units when those stages ran.

        Raises:
            :exc:`PineconeValueError`: If *namespace* is not a non-empty string, *top_k*
                is missing or below 1, no query source is given, a *vector* dict has an
                unsupported key, or *rerank* lacks ``model`` or ``rank_fields``. Raised
                before any request is made.
            :exc:`TypeError`: If ``query`` is combined with any of the flat keyword
                arguments it replaces (``top_k``, ``inputs``, ``vector``, ``id``,
                ``filter``, ``match_terms``), so pass one form or the other, not both;
                or if ``query`` is neither a :class:`SearchQuery` nor a mapping.

        Examples:
            Text in, embedded server-side. The hits nest one level down, under
            ``result``:

            .. code-block:: python

                response = await idx.search(
                    namespace="articles-en",
                    top_k=10,
                    inputs={"text": "benefits of vector databases for search"},
                    fields=["title", "text"],
                )
                for hit in response.result.hits:
                    print(hit.id, hit.score, hit.fields["title"])

            Reranking in the same call retrieves ``top_k`` hits and returns the
            ``top_n`` the reranker scores highest:

            .. code-block:: python

                response = await idx.search(
                    namespace="articles-en",
                    top_k=50,
                    inputs={"text": "benefits of vector databases"},
                    rerank={
                        "model": "bge-reranker-v2-m3",
                        "rank_fields": ["text"],
                        "top_n": 5,
                    },
                )

        .. seealso::
           - :meth:`query`: search with a vector you embedded yourself.
           - :meth:`Inference.rerank() <pinecone.client.inference.Inference.rerank>`:
             rerank results that came from somewhere other than this index.
        """
        if not isinstance(namespace, str):
            raise ValidationError("namespace must be a string")
        if not namespace or not namespace.strip():
            raise ValidationError("namespace must be a non-empty string")
        body = _build_search_records_body(
            method_name="AsyncIndex.search",
            top_k=top_k,
            inputs=inputs,
            vector=vector,
            id=id,
            filter=filter,
            fields=fields,
            rerank=rerank,
            match_terms=match_terms,
            query=query,
        )

        logger.info("Searching namespace %r with top_k=%d", namespace, body["query"]["top_k"])
        response = await self._http.post(
            f"/records/namespaces/{quote(namespace, safe='')}/search", timeout=timeout, json=body
        )
        result = self._adapter.to_search_response(response.content)
        result.response_info = extract_response_info(response)
        return result

    async def search_records(
        self,
        *,
        namespace: str,
        top_k: int | None = None,
        inputs: SearchInputs | Mapping[str, Any] | None = None,
        vector: Sequence[float] | Mapping[str, Any] | None = None,
        id: str | None = None,
        filter: Mapping[str, Any] | None = None,
        fields: Sequence[str] | None = None,
        rerank: RerankConfig | Mapping[str, Any] | None = None,
        match_terms: Mapping[str, Any] | None = None,
        query: SearchQuery | Mapping[str, Any] | None = None,
        timeout: float | None = None,
    ) -> SearchRecordsResponse:
        """Alias for :meth:`search` (legacy Records API), kept for callers written against 9.x.

        Every argument, return value and error is :meth:`search`'s. Call that one in
        new code.
        """
        return await self.search(
            namespace=namespace,
            top_k=top_k,
            inputs=inputs,
            vector=vector,
            id=id,
            filter=filter,
            fields=fields,
            rerank=rerank,
            match_terms=match_terms,
            query=query,
            timeout=timeout,
        )

    async def list_paginated(
        self,
        *,
        prefix: str | None = None,
        limit: int | None = None,
        pagination_token: str | None = None,
        namespace: str = "",
        timeout: float | None = None,
    ) -> ListResponse:
        """Fetch one page of vector IDs, holding the pagination token yourself.

        Serves indexes created with ``dimension``/``metric`` and legacy integrated indexes;
        on a document index use :attr:`documents` instead. Returns IDs only, without values
        or metadata. :meth:`list` walks the pages for you; use this one when you need to
        keep the token between calls.

        Args:
            prefix (str | None): Return only IDs starting with this prefix, e.g.
                ``"article-2024#"``. Must be ASCII, without a NUL character, and no
                longer than the API allows. The empty prefix matches everything.
            limit (int | None): Maximum number of IDs in this page, e.g. ``50``. Must be
                a positive integer within the API's maximum page size. ``None`` uses the
                server's default page size.
            pagination_token (str | None): ``pagination.next`` from the previous
                response. ``None`` (default) fetches the first page.
            namespace (str): Namespace to list from, e.g. ``"articles-en"``.
                Defaults to the default namespace.
            timeout (float | None): Per-request timeout in seconds. Overrides
                the client-level default for this call only.

        Returns:
            :class:`~pinecone.models.vectors.responses.ListResponse` whose ``vectors`` are
            :class:`~pinecone.models.vectors.responses.ListItem` entries carrying only an
            ``id``, plus ``namespace``, ``usage``, and ``pagination``. ``pagination.next``
            is the token for the following page; the last page carries no token.

        Raises:
            :exc:`PineconeValueError`: If *prefix* or *limit* is not legal. Raised
                before any request is sent.

        Examples:
            .. code-block:: python

                response = await idx.list_paginated(
                    prefix="article-2024#",
                    limit=50,
                    namespace="articles-en",
                )
                for item in response.vectors:
                    print(item.id)
                next_token = response.pagination.next if response.pagination else None

        .. seealso::
           - :meth:`list`: the same listing with the token handled for you.
           - :meth:`fetch`: the vectors behind those IDs.
           - `Pagination <https://sdk.pinecone.io/python/guides/pagination.html>`__:
             how the SDK pages generally.
        """
        if prefix is not None:
            require_valid_id_prefix("prefix", prefix)
        if limit is not None:
            require_valid_list_limit("limit", limit)

        params: dict[str, Any] = {"namespace": namespace}
        if prefix is not None:
            params["prefix"] = prefix
        if limit is not None:
            params["limit"] = limit
        if pagination_token is not None:
            params["paginationToken"] = pagination_token

        logger.info("Listing vectors in namespace %r", namespace)
        response = await self._http.get("/vectors/list", timeout=timeout, params=params)
        result = self._adapter.to_list_response(response.content)
        result.response_info = extract_response_info(response)
        return result

    async def list(
        self,
        *,
        prefix: str | None = None,
        limit: int | None = None,
        namespace: str = "",
        timeout: float | None = None,
    ) -> AsyncIterator[ListResponse]:
        """List vector IDs in a namespace, a page at a time.

        Serves indexes created with ``dimension``/``metric`` and legacy integrated indexes;
        on a document index use :attr:`documents` instead. Returns IDs only, without values
        or metadata. Yields one page per iteration and follows the pagination tokens itself.
        Nothing is requested until you iterate, so a bad *prefix* or *limit* is not reported
        until then either.

        Args:
            prefix (str | None): Return only IDs starting with this prefix, e.g.
                ``"article-2024#"``. Must be ASCII, without a NUL character, and no
                longer than the API allows. The empty prefix matches everything.
            limit (int | None): Maximum number of IDs per page, e.g. ``50``. Must be a
                positive integer within the API's maximum page size. ``None`` uses the
                server's default page size.
            namespace (str): Namespace to list from, e.g. ``"articles-en"``.
                Defaults to the default namespace.
            timeout (float | None): Per-request timeout in seconds, applied to
                each underlying page request. Overrides the client-level
                default for this call only.

        Yields:
            :class:`~pinecone.models.vectors.responses.ListResponse` per page, whose
            ``vectors`` carry the IDs. A page with no IDs is skipped rather than yielded.

        Raises:
            :exc:`PineconeValueError`: If *prefix* or *limit* is not legal. Raised on
                first iteration, not at the call.

        Examples:
            .. code-block:: python

                async for page in idx.list(prefix="article-2024#", namespace="articles-en"):
                    ids = [item.id for item in page.vectors]
                    fetched = await idx.fetch(ids=ids, namespace="articles-en")
                    for vid, vec in fetched.vectors.items():
                        print(vid, vec.metadata)

        .. seealso::
           - :meth:`list_paginated`: one page, with the token in your hands.
           - :meth:`fetch`: the vectors behind a page of IDs.
           - `Pagination <https://sdk.pinecone.io/python/guides/pagination.html>`__:
             how the SDK pages generally.
        """
        pagination_token: str | None = None
        while True:
            page = await self.list_paginated(
                prefix=prefix,
                limit=limit,
                pagination_token=pagination_token,
                namespace=namespace,
                timeout=timeout,
            )
            if page.vectors:
                yield page
            if page.pagination is not None and page.pagination.next is not None:
                pagination_token = page.pagination.next
            else:
                break

    async def describe_index_stats(
        self,
        *,
        filter: Mapping[str, Any] | None = None,
        timeout: float | None = None,
    ) -> DescribeIndexStatsResponse:
        """Report vector counts per namespace, dimension, and fullness for this index.

        Works on every kind of index, including document indexes. The counts lag writes, so
        a record just upserted may not be counted yet.

        Args:
            filter (dict[str, Any] | None): Accepted for API compatibility only. The
                server rejects a non-empty filter, so leave it unset; the statistics
                always describe the whole index.
            timeout (float | None): Per-request timeout in seconds. Overrides
                the client-level default for this call only.

        Returns:
            :class:`~pinecone.models.vectors.responses.DescribeIndexStatsResponse` with
            ``total_vector_count``, ``dimension``, ``index_fullness``, and ``namespaces``,
            which maps each namespace name to a
            :class:`~pinecone.models.vectors.responses.NamespaceSummary` carrying its
            ``vector_count``. The default namespace appears under ``"__default__"``.

        Raises:
            :exc:`ApiError`: If *filter* is non-empty.

        Examples:
            .. code-block:: python

                stats = await idx.describe_index_stats()
                print(stats.dimension, stats.index_fullness)
                for name, summary in stats.namespaces.items():
                    print(name, summary.vector_count)
        .. seealso::
           :meth:`list_namespaces`: per-namespace record counts alongside each
           namespace's schema and ``size_bytes``.
        """
        body: dict[str, Any] = {}
        if filter is not None:
            body["filter"] = filter

        logger.info("Describing index stats")
        response = await self._http.post("/describe_index_stats", timeout=timeout, json=body)
        result = self._adapter.to_stats_response(response.content)
        result.response_info = extract_response_info(response)
        return result

    async def create_namespace(
        self,
        *,
        name: str,
        schema: dict[str, Any] | None = None,
    ) -> NamespaceDescription:
        """Create a named namespace in the index.

        Optional: upserting into a name that does not exist yet creates the namespace.
        Create one explicitly when it must exist before any data lands in it, or to choose
        which metadata fields it indexes. Not idempotent: an existing name raises
        :exc:`~pinecone.errors.exceptions.ConflictError`.

        Args:
            name (str): Name for the new namespace, e.g. ``"movies-en"``. Must be
                non-empty ASCII without the NUL character, and no longer than the API's
                maximum name length. ``__default__`` is reserved and cannot be created.
                On an index created with ``dimension``/``metric`` it names the namespace
                requests address when they omit one, which always exists; on a document
                index, upserting documents into ``__default__`` creates it.
            schema (dict[str, Any] | None): Optional metadata-index configuration,
                ``{"fields": {<field>: {"filterable": True}}}``. Omitting it does not
                mean "index everything": the namespace inherits the index's own
                metadata-index configuration. Supply *schema* to index exactly the
                fields listed in this namespace. Each field must set ``filterable`` to
                ``True``; to leave a field unindexed, omit it from ``fields``.

        Returns:
            :class:`~pinecone.models.namespaces.models.NamespaceDescription` with the
            namespace ``name``, ``record_count``, ``schema``, ``indexed_fields`` and
            ``size_bytes``.

        Raises:
            :exc:`PineconeValueError`: If *name* violates the rules above, or *schema*
                is malformed. Raised before any request is sent.
            :exc:`ConflictError`: a namespace of that name already exists.

        Examples:
            .. code-block:: python

                ns = await idx.create_namespace(name="movies-en")
                print(ns.name, ns.record_count, ns.size_bytes)

            Naming the filterable fields up front overrides what the namespace would
            otherwise inherit from the index:

            .. code-block:: python

                ns = await idx.create_namespace(
                    name="movies-fr",
                    schema={"fields": {"genre": {"filterable": True}}},
                )
                print(ns.indexed_fields)

        .. seealso::
           - :meth:`describe_namespace`: the same description for a namespace that
             already exists.
           - :meth:`list_namespaces`: the namespaces the index already has.
        """
        require_creatable_namespace_name("name", name)

        body: dict[str, Any] = {"name": name}
        if schema is not None:
            require_valid_namespace_schema("schema", schema)
            body["schema"] = schema

        logger.info("Creating namespace %r", name)
        response = await self._http.post("/namespaces", json=body)
        return self._adapter.to_namespace_description(response.content)

    async def describe_namespace(
        self,
        *,
        name: str | None = None,
        **kwargs: str,
    ) -> NamespaceDescription:
        """Describe a namespace by name.

        Rate limited per index. To describe more than one namespace, use
        :meth:`list_namespaces`, which returns a whole page of descriptions per request.

        Args:
            name (str): Name of the namespace to describe, e.g. ``"movies-en"``. Must
                be non-empty ASCII without the NUL character, and no longer than the
                API's maximum name length. On an index created with
                ``dimension``/``metric``, pass ``__default__`` to describe the namespace
                requests address when they omit one. On a document index ``__default__``
                is an ordinary name, so describing it raises
                :exc:`~pinecone.errors.exceptions.NotFoundError` until documents are
                upserted into it.

        Returns:
            :class:`~pinecone.models.namespaces.models.NamespaceDescription` with the
            namespace ``name``, ``record_count``, ``schema``, ``indexed_fields`` and
            ``size_bytes``. ``size_bytes`` is approximate.

        Raises:
            :exc:`PineconeValueError`: If *name* violates the rules above. Raised
                before any request is sent.
            :exc:`TypeError`: If unexpected keyword arguments are passed.
            :exc:`NotFoundError`: no namespace of that name exists on the index.
            :exc:`RateLimitError`: this operation's per-index limit was exceeded.
                Use :meth:`list_namespaces` to describe many namespaces.

        Examples:
            .. code-block:: python

                ns = await idx.describe_namespace(name="movies-en")
                print(ns.name, ns.record_count, ns.size_bytes)

            On an index created with ``dimension``/``metric``, the namespace that
            unnamespaced requests address answers to ``__default__``:

            .. code-block:: python

                ns = await idx.describe_namespace(name="__default__")
                print(ns.record_count)

        .. seealso::
           :meth:`list_namespaces`: every namespace, a page of descriptions per
           request.
        """
        legacy_namespace: str | None = kwargs.pop("namespace", None)
        if kwargs:
            raise TypeError(
                f"describe_namespace() got unexpected keyword arguments: {sorted(kwargs)!r}"
            )
        if name is not None and legacy_namespace is not None:
            raise ValidationError("Provide either name= or namespace=, not both")
        effective: str = name if name is not None else (legacy_namespace or "")
        require_valid_namespace_name("name", effective)

        logger.info("Describing namespace %r", effective)
        response = await self._http.get(f"/namespaces/{quote(effective, safe='')}")
        return self._adapter.to_namespace_description(response.content)

    async def delete_namespace(
        self,
        *,
        name: str | None = None,
        timeout: float | None = None,
        **kwargs: str,
    ) -> None:
        """Delete a namespace and everything in it.

        Irreversible: every record in the namespace is deleted, and the namespace itself
        stops existing. To empty a namespace but keep it, use :meth:`delete` with
        ``delete_all=True``.

        Args:
            name (str): Name of the namespace to delete, e.g. ``"movies-deprecated"``.
                Must be non-empty ASCII without the NUL character, and no longer than
                the API's maximum name length.
            timeout (float | None): Per-request timeout in seconds. Overrides the
                client-level default for this call only.

        Returns:
            None. A successful delete returns no payload.

        Raises:
            :exc:`PineconeValueError`: If *name* violates the rules above. Raised
                before any request is sent.
            :exc:`TypeError`: If unexpected keyword arguments are passed.
            :exc:`NotFoundError`: no namespace of that name exists on the index.

        Examples:
            .. code-block:: python

                await idx.delete_namespace(name="movies-deprecated")

        .. seealso::
           :meth:`delete`: ``delete_all=True`` empties a namespace and leaves it in
           place.
        """
        legacy_namespace: str | None = kwargs.pop("namespace", None)
        if kwargs:
            raise TypeError(
                f"delete_namespace() got unexpected keyword arguments: {sorted(kwargs)!r}"
            )
        if name is not None and legacy_namespace is not None:
            raise ValidationError("Provide either name= or namespace=, not both")
        effective: str = name if name is not None else (legacy_namespace or "")
        require_valid_namespace_name("name", effective)

        logger.info("Deleting namespace %r", effective)
        await self._http.delete(f"/namespaces/{quote(effective, safe='')}", timeout=timeout)

    async def list_namespaces_paginated(
        self,
        *,
        prefix: str | None = None,
        limit: int | None = None,
        pagination_token: str | None = None,
    ) -> ListNamespacesResponse:
        """Fetch one page of namespace descriptions, holding the token yourself.

        :meth:`list_namespaces` walks the pages for you; reach for this one when you
        need to persist the token between calls or hand it to a caller of your own.

        Args:
            prefix (str | None): Return only namespaces whose names start with this
                prefix, e.g. ``"movies-"``. Same character rules as namespace names; the
                empty prefix matches every namespace.
            limit (int | None): Maximum number of namespaces in this page, e.g. ``50``.
                ``None`` (default) lets the server choose.
            pagination_token (str | None): ``pagination.next`` from the previous
                response. ``None`` (default) fetches the first page.

        Returns:
            :class:`~pinecone.models.namespaces.models.ListNamespacesResponse` with
            ``namespaces`` (each a
            :class:`~pinecone.models.namespaces.models.NamespaceDescription`),
            ``total_count`` across every page, and ``pagination``, whose ``next`` is
            ``None`` on the last page.

        Raises:
            :exc:`PineconeValueError`: If *prefix* or *limit* is out of range. Raised
                before any request is sent.

        Examples:
            .. code-block:: python

                response = await idx.list_namespaces_paginated(prefix="movies-", limit=10)
                for ns in response.namespaces:
                    print(ns.name, ns.record_count, ns.size_bytes)
                next_token = response.pagination.next if response.pagination else None

        .. seealso::
           - :meth:`list_namespaces`: the same listing with the token handled for
             you.
           - `Pagination <https://sdk.pinecone.io/python/guides/pagination.html>`__:
             how the SDK pages generally.
        """
        params: dict[str, Any] = {}
        if prefix is not None:
            require_valid_namespace_prefix("prefix", prefix)
            params["prefix"] = prefix
        if limit is not None:
            require_valid_namespace_limit("limit", limit)
            params["limit"] = limit
        if pagination_token is not None:
            params["paginationToken"] = pagination_token

        logger.info("Listing namespaces")
        response = await self._http.get("/namespaces", params=params)
        return self._adapter.to_list_namespaces_response(response.content)

    async def list_namespaces(
        self,
        *,
        prefix: str | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[ListNamespacesResponse]:
        """List every namespace, a page at a time.

        Yields one :class:`~pinecone.models.namespaces.models.ListNamespacesResponse`
        per page and follows the pagination tokens itself, so nothing is requested until
        you iterate. Each page describes many namespaces in one request, so prefer this
        over repeated :meth:`describe_namespace` calls, which are rate limited per index.

        Args:
            prefix (str | None): Return only namespaces whose names start with this
                prefix, e.g. ``"movies-"``. Same character rules as namespace names; the
                empty prefix matches every namespace.
            limit (int | None): Maximum number of namespaces per page, e.g. ``50``.
                ``None`` (default) lets the server choose.

        Yields:
            :class:`~pinecone.models.namespaces.models.ListNamespacesResponse` per page,
            each carrying ``namespaces`` of
            :class:`~pinecone.models.namespaces.models.NamespaceDescription`. A page
            with no namespaces is skipped rather than yielded.

        Raises:
            :exc:`PineconeValueError`: If *prefix* or *limit* is out of range. Raised
                on first iteration, not at the call.

        Examples:
            .. code-block:: python

                async for page in idx.list_namespaces(prefix="movies-"):
                    for ns in page.namespaces:
                        print(ns.name, ns.record_count, ns.size_bytes)

        .. seealso::
           - :meth:`list_namespaces_paginated`: one page, with the token in your
             hands.
           - :meth:`describe_namespace`: one namespace, when you know its name.
           - `Pagination <https://sdk.pinecone.io/python/guides/pagination.html>`__:
             how the SDK pages generally.
        """
        pagination_token: str | None = None
        while True:
            page = await self.list_namespaces_paginated(
                prefix=prefix,
                limit=limit,
                pagination_token=pagination_token,
            )
            if page.namespaces:
                yield page
            if page.pagination is not None and page.pagination.next is not None:
                pagination_token = page.pagination.next
            else:
                break

    def _validate_import_id(self, id: str | int) -> str:
        """Validate and normalize an import operation ID.

        Args:
            id: Import operation ID. If int, converted to str silently.

        Returns:
            The validated string ID.

        Raises:
            :exc:`PineconeTypeError`: If the ID is not a ``str`` or an ``int``.
            :exc:`PineconeValueError`: If the ID is empty or exceeds 1000 characters.
        """
        if not isinstance(id, (str, int)):
            raise PineconeTypeError(f"id must be a string or an integer, got {type(id).__name__}")
        str_id = str(id) if isinstance(id, int) else id
        if not str_id or len(str_id) > 1000:
            raise ValidationError(
                "import id must be between 1 and 1000 characters, "
                f"got {len(str_id) if str_id else 0}"
            )
        return str_id

    async def start_import(
        self,
        uri: str,
        *,
        error_mode: str | None = None,
        integration_id: str | None = None,
    ) -> StartImportResponse:
        """Start a server-side bulk import of vectors from cloud storage.

        Returns as soon as the server accepts the import, not when it finishes: poll
        :meth:`describe_import` for the outcome. Imports work on indexes created with
        ``dimension``/``metric`` and on document indexes; the server refuses them on an
        index with integrated embedding (a legacy integrated index, or a document index
        with an ``embed`` field).

        Args:
            uri (str): Directory prefix holding the Parquet files, not a single
                file. Three forms are accepted: ``s3://`` for Amazon S3,
                ``gs://`` for Google Cloud Storage, and an ``https://`` URL
                naming an Azure Blob Storage container. ``s3://`` additionally
                requires that the index itself be hosted on AWS.
            error_mode (str | None): How to handle a record the import cannot
                read. ``"continue"`` skips it and imports the rest; ``"abort"``
                ends the whole import at the first such record. Case-insensitive.
                Defaults to ``"abort"`` when omitted, so an unreadable record
                fails the import unless you opt into skipping.
            integration_id (str | None): ID of a storage integration in your
                project that grants Pinecone read access to a private bucket. Its
                cloud provider must match the scheme of *uri*.

        Returns:
            :class:`~pinecone.models.imports.model.StartImportResponse` with ``id``, the
            handle the other import methods take.

        Raises:
            :exc:`PineconeValueError`: If *error_mode* is supplied but is neither
                ``"continue"`` nor ``"abort"``. Raised before any HTTP request is
                made.
            :exc:`ApiError`: If the server rejects *uri* (for example, an
                ``s3://`` URI on an index not hosted on AWS) or the index uses
                integrated embedding.

        Examples:
            Starting an import and waiting it out is on you; three of the five
            statuses are terminal:

            .. code-block:: python

                import asyncio

                response = await idx.start_import(uri="s3://article-embeddings/2024/")
                import_op = await idx.describe_import(response.id)
                while import_op.status not in ("Completed", "Failed", "Cancelled"):
                    await asyncio.sleep(10)
                    import_op = await idx.describe_import(response.id)
                print(import_op.status, import_op.records_imported)

            ``error_mode="continue"`` finishes the import around records it cannot
            read, rather than stopping at the first one:

            .. code-block:: python

                response = await idx.start_import(
                    uri="s3://article-embeddings/2024/",
                    error_mode="continue",
                )

        .. note::
           *uri* must name a directory of Parquet files following Pinecone's import
           schema. See the
           `import guide <https://docs.pinecone.io/guides/data/understanding-imports>`_
           for that schema and the supported storage formats.

        .. seealso::
           - :meth:`describe_import`: progress and the terminal status.
           - :meth:`cancel_import`: stops an import that is still running.
           - :meth:`upsert`: writes vectors directly, for data too small to need an
             import.
        """
        if error_mode is not None:
            error_mode = error_mode.lower()
            if error_mode not in ("continue", "abort"):
                raise ValidationError(
                    f"error_mode must be 'continue' or 'abort', got {error_mode!r}"
                )

        body: dict[str, Any] = {"uri": uri}
        if error_mode is not None:
            body["errorMode"] = {"onError": error_mode}
        if integration_id is not None:
            body["integrationId"] = integration_id

        logger.info("Starting bulk import from %s", uri)
        response = await self._http.post("/bulk/imports", json=body)
        return self._imports_adapter.to_start_import_response(response.content)

    async def describe_import(self, id: str | int) -> ImportModel:
        """Describe a bulk import operation by ID.

        Args:
            id: The ``id`` :meth:`start_import` returned, e.g. ``"1"``. An
                ``int`` is accepted and converted to a string.

        Returns:
            :class:`~pinecone.models.imports.model.ImportModel` with ``status``
            (``"Pending"``, ``"InProgress"``, ``"Failed"``, ``"Completed"``, or
            ``"Cancelled"``; the last three are terminal), ``percent_complete``,
            ``records_imported``, ``uri``, and ``error`` when it failed.

        Raises:
            :exc:`PineconeTypeError`: If *id* is neither a ``str`` nor an ``int``.
            :exc:`PineconeValueError`: If *id* is empty or too long. Raised
                before any HTTP request is made.

        Examples:
            .. code-block:: python

                import_op = await idx.describe_import("1")
                print(import_op.status, import_op.percent_complete)
                if import_op.status == "Failed":
                    print(import_op.error)

        .. seealso::
           - :meth:`start_import`: starts an import and returns the ``id`` these
             methods take.
           - :meth:`list_imports`: every import on the index, when you do not have an
             ``id``.
        """
        str_id = self._validate_import_id(id)
        logger.info("Describing import %s", str_id)
        response = await self._http.get(f"/bulk/imports/{quote(str_id, safe='')}")
        return self._imports_adapter.to_import_model(response.content)

    async def cancel_import(self, id: str | int) -> None:
        """Cancel a bulk import operation by ID.

        Args:
            id: The ``id`` :meth:`start_import` returned, e.g. ``"1"``. An
                ``int`` is accepted and converted to a string.

        Returns:
            None. Poll :meth:`describe_import` to see the import reach
            ``"Cancelled"``.

        Raises:
            :exc:`PineconeTypeError`: If *id* is neither a ``str`` nor an ``int``.
            :exc:`PineconeValueError`: If *id* is empty or too long. Raised
                before any HTTP request is made.

        Examples:
            .. code-block:: python

                await idx.cancel_import("1")

        .. seealso::
           - :meth:`start_import`: starts an import and returns the ``id`` these
             methods take.
           - :meth:`list_imports`: every import on the index, when you do not have an
             ``id``.
        """
        str_id = self._validate_import_id(id)
        logger.info("Cancelling import %s", str_id)
        await self._http.delete(f"/bulk/imports/{quote(str_id, safe='')}")

    async def list_imports(
        self,
        *,
        limit: int | None = None,
        pagination_token: str | None = None,
    ) -> AsyncIterator[ImportModel]:
        """List every bulk import on this index, following pagination.

        Yields :class:`~pinecone.models.imports.model.ImportModel` objects rather
        than pages, and fetches the next page as you exhaust the current one, so
        nothing is requested until you iterate. See
        `Pagination <https://sdk.pinecone.io/python/guides/pagination.html>`__.

        Args:
            limit (int | None): Maximum number of imports per page, e.g. ``10``.
                Omit to let the server choose the page size.
            pagination_token (str | None): Token to resume from, when you are
                continuing a listing rather than starting one.

        Yields:
            :class:`~pinecone.models.imports.model.ImportModel` for each import
            operation.

        Raises:
            :exc:`ApiError`: If a page request fails part-way through the listing;
                the imports already yielded are still yours, the rest are not.

        Examples:
            .. code-block:: python

                async for imp in idx.list_imports():
                    print(imp.id, imp.status, imp.uri)

        .. seealso::
           - :meth:`list_imports_paginated`: one page at a time, with the token in
             your hands.
           - :meth:`describe_import`: one import, when you know its ``id``.
        """
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if pagination_token is not None:
            params["paginationToken"] = pagination_token

        while True:
            response = await self._http.get("/bulk/imports", params=params)
            import_list = self._imports_adapter.to_import_list(response.content)
            for item in import_list:
                yield item
            next_token = import_list.pagination.next if import_list.pagination else None
            if next_token is None:
                break
            params["paginationToken"] = next_token

    async def list_imports_paginated(
        self,
        *,
        limit: int | None = None,
        pagination_token: str | None = None,
    ) -> ImportList:
        """Fetch one page of bulk imports, holding the token yourself.

        :meth:`list_imports` walks the pages for you; use this one when you need to
        persist the token between calls. See
        `Pagination <https://sdk.pinecone.io/python/guides/pagination.html>`__.

        Args:
            limit (int | None): Maximum number of imports in this page, e.g.
                ``10``.
            pagination_token (str | None): ``pagination.next`` from the previous
                response. ``None`` (default) fetches the first page.

        Returns:
            :class:`~pinecone.models.imports.list.ImportList` you can iterate for this
            page's :class:`~pinecone.models.imports.model.ImportModel` objects, with
            ``pagination.next`` holding the token for the following page or ``None``
            on the last one.

        Examples:
            .. code-block:: python

                page = await idx.list_imports_paginated(limit=10)
                for imp in page:
                    print(imp.id, imp.status)
                next_token = page.pagination.next if page.pagination else None

        .. seealso::
           - :meth:`list_imports`: the same listing with the token handled for you.
        """
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if pagination_token is not None:
            params["paginationToken"] = pagination_token

        response = await self._http.get("/bulk/imports", params=params)
        return self._imports_adapter.to_import_list(response.content)

    async def close(self) -> None:
        """Close the underlying HTTP client and release its resources.

        Calls on a closed index fail. Prefer ``async with`` over calling this by
        hand, which closes the client even if the body raises.

        Returns:
            None.

        Examples:
            .. code-block:: python

                async with await pc.index(name="article-search") as idx:
                    await idx.upsert(
                        vectors=[("article-101", [0.012, -0.087, 0.153])],
                        namespace="articles-en",
                    )
        """
        await self._http.close()

    async def __aenter__(self) -> AsyncIndex:
        """Enter the async context manager, returning this index.

        Returns:
            This :class:`AsyncIndex` instance.

        Examples:
            .. code-block:: python

                async with await pc.index(name="article-search") as idx:
                    await idx.upsert(
                        vectors=[("article-101", [0.012, -0.087, 0.153])],
                        namespace="articles-en",
                    )
        """
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Exit the async context manager, calling :meth:`close`.

        Returns:
            None.
        """
        await self.close()

    def __repr__(self) -> str:
        """Return developer-friendly representation."""
        return f"AsyncIndex(host='{self._host}')"
