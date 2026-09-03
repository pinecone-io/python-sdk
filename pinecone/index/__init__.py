"""Synchronous data plane client for a Pinecone index."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import quote

if TYPE_CHECKING:
    import pandas as pd  # type: ignore[import-untyped]

    from pinecone.client.documents import Documents

from pinecone._internal.adapters.imports_adapter import ImportsAdapter
from pinecone._internal.adapters.vectors_adapter import VectorsAdapter, extract_response_info
from pinecone._internal.batching import validate_batch_size
from pinecone._internal.bulk import bulk_execute_sync
from pinecone._internal.config import PineconeConfig
from pinecone._internal.constants import DATA_PLANE_API_VERSION, DEFAULT_MAX_CONCURRENCY
from pinecone._internal.data_plane_helpers import (
    _build_search_records_body,
    _validate_host,
    _vector_to_dict,
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
from pinecone.errors.exceptions import PineconeValueError, ValidationError
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
class Index:
    """Synchronous data plane client targeting a specific Pinecone index.

    An index's data plane is where its records live, and this is the client that
    reads and writes them. Reach one with ``pc.index(name="article-search")``, or
    construct it here from a host URL when you already know the host and want to
    skip the describe-index lookup that resolving a name costs. Every call blocks;
    :class:`~pinecone.async_client.async_index.AsyncIndex` is the ``asyncio`` twin.

    Only errors specific to a single method are documented on that method. For the
    exception hierarchy every method shares, see :doc:`/guides/error-handling`.

    Args:
        host (str): The index-specific data plane host URL.
        api_key (str | None): Pinecone API key. Falls back to ``PINECONE_API_KEY`` env var.
        additional_headers (Mapping[str, str] | None): Extra headers included in every request.
        timeout (float): Request timeout in seconds. Defaults to ``30.0``.
        proxy_url (str | None): HTTP proxy URL for outgoing requests.
        ssl_ca_certs (str | None): Path to a CA certificate bundle for SSL verification.
        ssl_verify (bool): Whether to verify SSL certificates. Defaults to ``True``.
        source_tag (str | None): Tag appended to the User-Agent string for request attribution.
        connection_pool_maxsize (int): Maximum number of connections to keep in the pool.
            ``0`` (default) uses httpx defaults.
        pool_threads (int | None): Tune the thread pool used by the legacy
            ``async_req=True`` execution model on ``upsert``, ``query``,
            ``describe_index_stats``, and ``list_paginated``. Defaults to ``10``.
            The pool is lazy-constructed on first ``async_req=True`` call and shut
            down by :meth:`close`; ``multiprocessing.pool`` is not imported until
            then. **For new code, prefer**
            :class:`~pinecone.async_client.async_index.AsyncIndex` **or**
            :class:`concurrent.futures.ThreadPoolExecutor`. This kwarg exists for
            backcompat with pre-rewrite callers.

    Raises:
        :exc:`PineconeValueError`: If no API key can be resolved or the host is invalid.
        :exc:`FileNotFoundError`: If ``ssl_ca_certs`` names a path that does not
            exist, raised when the client is constructed, so a mistyped path
            cannot leave you silently verifying against the default trust store
            instead. A bundle that exists but cannot be parsed as a certificate
            raises :exc:`ssl.SSLError` instead.

    Examples:
        >>> from pinecone import Pinecone
        >>> pc = Pinecone(api_key="your-api-key")
        >>> idx = pc.index(name="article-search")
        >>> idx.describe_index_stats().total_vector_count
        0

    .. seealso::
       :class:`~pinecone.async_client.async_index.AsyncIndex` — the same surface on
       ``asyncio``. :class:`~pinecone.grpc.GrpcIndex` — the gRPC transport, for
       write-heavy ingest.
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
        **kwargs: Any,
    ) -> None:
        legacy_pool_threads = kwargs.pop("pool_threads", None)
        if kwargs:
            raise TypeError(f"Index() got unexpected keyword arguments: {sorted(kwargs)!r}")
        # Resolve API key: explicit arg > env var (check BEFORE host per unified-ord-0001)
        resolved_key = api_key or os.environ.get("PINECONE_API_KEY", "")
        if not resolved_key:
            raise ValidationError(
                "No API key provided. Pass api_key='...' or set the "
                "PINECONE_API_KEY environment variable."
            )

        # Validate and normalize host
        self._host = _validate_host(host)

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
        self._config = config

        from pinecone._internal.http_client import HTTPClient

        self._http = HTTPClient(config, DATA_PLANE_API_VERSION)
        self._adapter = VectorsAdapter()
        self._imports_adapter = ImportsAdapter()
        self._documents: Documents | None = None

        from pinecone._legacy.async_req import (
            _DEFAULT_POOL_THREADS,
            install_async_req_support,
        )

        install_async_req_support(
            self,
            legacy_pool_threads if legacy_pool_threads is not None else _DEFAULT_POOL_THREADS,
        )

        logger.info("Index client created for host %s", self._host)

    @property
    def host(self) -> str:
        """The data plane host URL for this index."""
        return self._host

    @property
    def documents(self) -> Documents:
        """Entry point for document operations on a schema-based index.

        A schema-based index stores JSON records instead of raw vectors. Reach the
        document operations — ``upsert``, ``search``, ``fetch`` and the rest — through
        here; the vector methods on this class (:meth:`upsert`, :meth:`query`) are for
        a vector-based index. The same instance is returned on every access.

        Returns:
            :class:`~pinecone.client.documents.Documents` namespace instance.

        Examples:

            >>> from pinecone import Pinecone
            >>> pc = Pinecone(api_key="your-api-key")
            >>> idx = pc.index(name="article-search")
            >>> response = idx.documents.upsert(
            ...     namespace="published",
            ...     documents=[{"_id": "article-101", "title": "Intro to vectors"}],
            ... )
            >>> response.upserted_count
            1

        .. seealso::
           :class:`~pinecone.client.documents.Documents` — every document operation,
           with an example each.
        """
        if self._documents is None:
            from pinecone.client.documents import Documents as _Documents

            self._documents = _Documents(http=self._http, host=self._host)
        return self._documents

    def upsert(
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

        If a vector with the same ID already exists in the namespace, it is
        overwritten.

        Each request is capped on both vector count and encoded payload size;
        wide vectors or large metadata tend to hit the size cap first. Pass
        ``batch_size`` to split a long sequence of vectors into requests that
        stay under both limits.

        Args:
            vectors: Sequence of vectors to upsert. Each element can be a
                :class:`Vector`, a tuple of ``(id, values)`` or
                ``(id, values, metadata)``, or a dict with ``id``, ``values``,
                and optional ``sparse_values`` / ``metadata`` keys.
            namespace (str): Target namespace, e.g. ``"articles-en"``. Defaults to
                the empty string, which addresses the index's default namespace.
            batch_size (int | None): Split *vectors* into chunks of this size and
                send one request per chunk, e.g. ``100``. ``None`` (default) sends
                every vector in one request. Must be a positive integer.
            show_progress (bool): When ``True`` (default) and ``tqdm`` is installed,
                display a progress bar that advances as batches complete. No effect
                when ``batch_size`` is ``None`` or ``tqdm`` is not installed.
            max_concurrency (int): Batch requests in flight at once, 1-64. Defaults
                to ``8``. Only used when ``batch_size`` is set.
            timeout (float | None): Per-request timeout in seconds. Overrides the
                client-level default for this call only.
            total_timeout (float | None): Deadline in seconds for the whole batched
                operation, as opposed to *timeout*, which bounds one attempt of one
                batch. On expiry no further batches are submitted; batches already
                in flight are awaited and never cancelled; unsent batches are
                reported in ``failed_items``. ``None`` (default) means no deadline.

        Returns:
            :class:`UpsertResponse` with ``upserted_count``. When ``batch_size``
            triggers multiple requests, ``response_info`` carries the aggregate LSN
            from all successful batches, or ``None`` if no LSN headers came back,
            and ``errors`` / ``failed_items`` name whatever did not land.

        Raises:
            :exc:`PineconeTypeError`: If a vector element is not one of the forms
                listed above.
            :exc:`PineconeValueError`: If a vector element is malformed, *batch_size*
                is not a positive integer, or *max_concurrency* falls outside 1-64.
            :exc:`ApiError`: If one request exceeds the server's cap on vectors per
                request or on encoded request size — lower ``batch_size`` and retry.

        Examples:
            All three vector forms are interchangeable within one call. The values
            below are truncated to three floats for the page; pass your index's full
            dimension.

            >>> from pinecone import Vector
            >>> idx = pc.index(name="article-search")
            >>> response = idx.upsert(
            ...     vectors=[
            ...         Vector(id="article-101", values=[0.012, -0.087, 0.153]),
            ...         ("article-102", [0.045, 0.021, -0.064]),
            ...         {"id": "article-103", "values": [0.091, -0.032, 0.178]},
            ...     ],
            ...     namespace="articles-en",
            ... )
            >>> response.upserted_count
            3

            For a sequence too long for one request — ``embeddings`` below being your
            whole list of vectors — set ``batch_size`` and check the response for
            batches that did not land:

            .. code-block:: python

                response = idx.upsert(
                    vectors=embeddings,
                    namespace="articles-en",
                    batch_size=100,
                )
                if response.has_errors:
                    idx.upsert(vectors=response.failed_items, namespace="articles-en")

        .. note::
           With ``batch_size`` set, batches are submitted in parallel through a
           ``ThreadPoolExecutor`` of ``max_concurrency`` workers, and **a partial
           failure does not raise** — ``response.has_errors``, ``response.errors``
           and ``response.failed_items`` report it, and ``failed_items`` can be
           passed straight back to ``upsert``. Each batch is retried on its own
           under the client's retry policy, timeouts included, so *timeout* bounds
           one attempt rather than the batch; see :doc:`/guides/retries`.

        .. seealso::
           - :meth:`upsert_records` — text in, embedded server-side, for an index
             with integrated inference.
           - :meth:`upsert_from_dataframe` — the same write from a pandas
             DataFrame, batched for you.
           - :meth:`start_import` — millions of vectors from cloud storage,
             server-side and asynchronous.
        """
        if batch_size is None:
            return self._upsert_one_batch(vectors=vectors, namespace=namespace, timeout=timeout)

        validate_batch_size(batch_size)
        require_in_range("max_concurrency", max_concurrency, 1, 64)

        built = [VectorFactory.build(v) for v in vectors]
        items: list[dict[str, Any]] = [_vector_to_dict(v) for v in built]

        def _operation(chunk: list[dict[str, Any]]) -> UpsertResponse:
            return self._upsert_dict_batch(items=chunk, namespace=namespace, timeout=timeout)

        batch_result = bulk_execute_sync(
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

    def _upsert_one_batch(
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
        body: dict[str, Any] = {"vectors": [_vector_to_dict(v) for v in built]}
        if namespace:
            body["namespace"] = namespace
        logger.info("Upserting %d vectors into namespace %r", len(built), namespace)
        response = self._http.post("/vectors/upsert", timeout=timeout, json=body)
        result = self._adapter.to_upsert_response(response.content)
        result.response_info = extract_response_info(response)
        logger.debug("Upserted %d vectors", result.upserted_count)
        return result

    def _upsert_dict_batch(
        self,
        *,
        items: list[dict[str, Any]],
        namespace: str,
        timeout: float | None,
    ) -> UpsertResponse:
        body: dict[str, Any] = {"vectors": items}
        if namespace:
            body["namespace"] = namespace
        response = self._http.post("/vectors/upsert", timeout=timeout, json=body)
        result = self._adapter.to_upsert_response(response.content)
        result.response_info = extract_response_info(response)
        return result

    def upsert_from_dataframe(
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
        """Upsert vectors from a pandas DataFrame.

        Convenience method that accepts a DataFrame with columns ``id``,
        ``values``, and optionally ``sparse_values`` and ``metadata``,
        batches the rows, and upserts them via :meth:`upsert`.

        Args:
            df: A ``pandas.DataFrame`` with at least ``id`` and ``values``
                columns. ``sparse_values`` and ``metadata`` columns are
                included when present and non-None.
            namespace: Target namespace, e.g. ``"articles-en"``. Defaults to the
                index's default namespace.
            batch_size: Number of rows per upsert batch. Defaults to 500.
            show_progress: If ``True`` (default) and ``tqdm`` is installed, display
                a progress bar that advances as batches complete. If ``tqdm`` is not
                installed, silently falls back to no progress bar.
            timeout: Client-side request timeout in seconds applied to *each
                batch's* upsert request — not to the DataFrame as a whole.
                ``None`` (default) uses the client-level default. Raise it to
                accommodate large or slow batches.
            max_concurrency: Number of batches in flight at once, range
                ``[1, 64]``. ``None`` (default) uses ``8`` — flat and
                identical across every transport. The host's adaptive limit
                still applies underneath.
            total_timeout: Deadline in seconds for the **whole ingest**, as
                opposed to *timeout*, which bounds a single attempt of a
                single batch. On expiry no further batches are submitted;
                batches already in flight are awaited and never cancelled;
                unsent batches are reported in ``failed_items``. ``None``
                (default) means no deadline.
            on_error: What to do when some batches fail. ``"collect"`` (the
                default) returns an :class:`UpsertResponse` carrying
                ``failed_item_count``, ``errors`` and ``failed_items``.
                ``"raise"`` re-raises the lowest-indexed batch failure once
                every batch has settled, with the partial result attached to
                the exception's ``response`` attribute.

        Returns:
            :class:`UpsertResponse` with ``upserted_count`` totalled across every
            batch that landed.

        Raises:
            :exc:`RuntimeError`: If ``pandas`` is not installed. It is not an SDK
                dependency; install it yourself with ``pip install pandas``.
            :exc:`PineconeValueError`: If *df* is not a ``pandas.DataFrame``,
                *batch_size* is not a positive integer, or *max_concurrency* falls
                outside 1-64.
            :exc:`PineconeTimeoutError`: If ``on_error="raise"`` and a batch
                exhausted its retries on timeout. Under the default
                ``on_error="collect"`` that same failure is reported on the
                returned response rather than raised.

        Examples:
            A ``metadata`` column is optional; where it is present its dict lands on
            the vector as written.

            .. code-block:: python

                import pandas as pd
                from pinecone import Pinecone

                pc = Pinecone(api_key="your-api-key")
                idx = pc.index(name="article-search")
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
                response = idx.upsert_from_dataframe(
                    df,
                    namespace="articles-en",
                    batch_size=100,
                )
                print(response.upserted_count)

        .. note::
           ``pandas`` is not an SDK dependency — this is the only method that needs
           it, so install it in your own environment.

        .. seealso::
           - :meth:`upsert` — the same write from a list of vectors, with the same
             ``batch_size`` and no ``pandas`` dependency.
           - :meth:`upsert_records` — text in, embedded server-side, for an index
             with integrated inference.
           - :meth:`start_import` — millions of vectors from cloud storage,
             server-side and asynchronous.
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

        ns = namespace or ""
        response = self.upsert(
            vectors=records,
            namespace=ns,
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

    def upsert_records(
        self,
        *,
        records: list[dict[str, Any]],
        namespace: str,
        timeout: float | None = None,
    ) -> UpsertRecordsResponse:
        """Upsert records for indexes with integrated inference.

        Records are sent as newline-delimited JSON (NDJSON). Embeddings are
        generated server-side.

        Args:
            records: Record dicts, each carrying an ``_id`` (or ``id``) plus the
                fields to store; the index's embedding model decides which field it
                embeds. A record giving both ``_id`` and ``id`` keeps ``_id`` and
                the client drops the ``id`` before sending.
            namespace (str): Target namespace, e.g. ``"articles-en"``. Required and
                non-empty — unlike :meth:`upsert`, the records API has no default
                namespace to fall back on.
            timeout (float | None): Per-request timeout in seconds. Overrides the
                client-level default for this call only.

        Returns:
            :class:`UpsertRecordsResponse` with ``record_count``, the number of
            records submitted.

        Raises:
            :exc:`PineconeValueError`: If *namespace* is not a non-empty string,
                *records* is empty, or a record has no ``_id``/``id`` field or one
                that is not a string. Raised before any HTTP request is made.

        Examples:
            >>> idx = pc.index(name="article-search")
            >>> response = idx.upsert_records(
            ...     namespace="articles-en",
            ...     records=[
            ...         {"_id": "article-101", "text": "Vector databases for search."},
            ...         {"_id": "article-102", "text": "RAG combines search with LLMs."},
            ...     ],
            ... )
            >>> response.record_count
            2

        .. seealso::
           - :meth:`search` — the read side of an integrated-inference index: text
             in, embedded server-side.
           - :meth:`upsert` — for an index where you embed the text yourself and
             send vectors.
           - :meth:`start_import` — millions of vectors from cloud storage,
             server-side and asynchronous.
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
        response = self._http.post(
            f"/records/namespaces/{quote(namespace, safe='')}/upsert",
            timeout=timeout,
            content=ndjson_body,
            headers={"Content-Type": "application/x-ndjson"},
        )
        result = UpsertRecordsResponse(record_count=len(records))
        result.response_info = extract_response_info(response)
        return result

    def query(
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
        """Query a namespace for the nearest neighbours of a vector you supply.

        You supply the query vector; nothing is embedded for you. At least one query
        selector is required: a dense *vector*, a *sparse_vector*, both together for
        a hybrid query, or the *id* of a vector the index already holds. An *id* is a
        reference to stored data, so it cannot be mixed with either vector form.

        Args:
            top_k (int): Number of results to return, 1-10000, e.g. ``5``.
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
                with indices and values. Can be combined with *vector* for a
                hybrid query on indexes that support both.
            scan_factor (float | None): Recall/latency trade for dedicated read
                node (DRN) indexes — a multiplier on how much of the index is
                scanned. Above 1 scans more and favours recall; below 1 scans
                less and favours latency. Omit to let the server choose.
            max_candidates (int | None): Recall/latency trade for dedicated read
                node (DRN) indexes — caps how many candidates are reranked before
                ``top_k`` is taken. Must be at least ``top_k``: a smaller value is
                rejected rather than clamped, since it could not fill the page.
            timeout (float | None): Per-request timeout in seconds. Overrides the
                client-level default for this call only.

        Returns:
            :class:`QueryResponse` with ``matches`` (ordered from most to least
            similar, each carrying ``id`` and ``score``), ``namespace``, and
            ``usage``.

        Raises:
            :exc:`PineconeValueError`: If *top_k* falls outside 1-10000, if *id* is
                combined with *vector* or *sparse_vector*, if none of *vector*,
                *id*, or *sparse_vector* is given, or if *id* is not a legal vector
                ID. Raised before any HTTP request is made.
            :exc:`ApiError`: If *scan_factor* or *max_candidates* is out of range,
                or the index is not a dense DRN index — both knobs are rejected on
                on-demand indexes and on sparse indexes.

        Examples:
            Query vectors are truncated to three floats on this page; pass your
            index's full dimension.

            >>> idx = pc.index(name="product-search")
            >>> response = idx.query(
            ...     top_k=2,
            ...     vector=[0.012, -0.087, 0.153],
            ...     namespace="electronics",
            ... )
            >>> [(match.id, match.score) for match in response.matches]
            [('prod-0', 0.9), ('prod-1', 0.8)]

            A match comes back stripped by default, and the two flags are not
            symmetric about it — an unrequested ``values`` is empty, an unrequested
            ``metadata`` is ``None``:

            >>> response.matches[0].values
            []
            >>> response.matches[0].metadata is None
            True

            Ask, and both arrive. A ``filter`` narrows the search before ranking, so
            ``include_metadata`` is how you see the fields you selected on:

            >>> response = idx.query(
            ...     top_k=2,
            ...     vector=[0.012, -0.087, 0.153],
            ...     namespace="electronics",
            ...     filter={"category": {"$eq": "audio"}},
            ...     include_metadata=True,
            ... )
            >>> response.matches[0].metadata["description"]
            'Noise-cancelling over-ear headphones 0'

        .. seealso::
           - :meth:`search` — the same search on an integrated-inference index:
             you pass text, the server embeds it.
           - :attr:`documents` — ``documents.search`` for a schema-based index,
             which stores JSON records rather than raw vectors.
           - :meth:`query_namespaces` — the same query fanned out over several
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
        response = self._http.post("/query", timeout=timeout, json=body)
        result = self._adapter.to_query_response(response.content)
        result.response_info = extract_response_info(response)
        logger.debug("Query returned %d matches", len(result.matches))
        return result

    def query_namespaces(
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

        One :meth:`query` per namespace, issued in parallel on a thread pool of up
        to 32 workers, then merged so the result is the overall top-k rather than
        top-k per namespace. Because the merge ranks by *metric*, you have to name
        the index's metric yourself — nothing here reads it off the index.

        Args:
            vector: Dense query vector values. Required for dense and hybrid
                indexes; omit for sparse-only indexes (use *sparse_vector* instead).
            namespaces: Namespaces to query, e.g.
                ``["articles-en", "articles-fr"]``. Must be non-empty; duplicates
                are removed while preserving order.
            metric: Distance metric the merge ranks by — ``"cosine"``,
                ``"euclidean"``, or ``"dotproduct"``. Pass the metric the index was
                created with, or the merged ranking will be wrong.
            top_k: Maximum number of results to return after merging, e.g. ``10``.
                Defaults to 10. Each namespace is queried for this many, so the
                merge chooses from ``top_k × len(namespaces)`` candidates.
            filter: Metadata filter expression applied to every namespace.
            include_values: Return each match's vector values.
            include_metadata: Return each match's metadata.
            sparse_vector: Sparse query vector with indices and values.
                Required for sparse-only indexes when *vector* is omitted.
            scan_factor: Recall/latency trade for dedicated read node (DRN)
                indexes — a multiplier on how much of the index is scanned.
                Above 1 scans more and favours recall; below 1 scans less and
                favours latency. Applied to every namespace queried.
            max_candidates: Recall/latency trade for dedicated read node (DRN)
                indexes — caps how many candidates are reranked before ``top_k``
                is taken, per namespace. Must be at least ``top_k``.
            timeout: Per-request timeout in seconds, applied to each namespace's
                query rather than to the fan-out as a whole.

        Returns:
            :class:`QueryNamespacesResults` with the merged ``matches``, ``usage``
            totalled over every namespace, and ``ns_usage`` keyed by namespace name.
            A match carries no record of which namespace produced it, so query one
            namespace at a time when you need that.

        Raises:
            :exc:`PineconeValueError`: If *namespaces* is empty, if both *vector*
                and *sparse_vector* are absent or empty, or if *metric* is not one
                of ``"cosine"``, ``"euclidean"``, or ``"dotproduct"``. Raised
                before any HTTP request is made.
            :exc:`ApiError`: If any one namespace's query fails; the first such
                failure propagates and the merged result is lost, so retry the
                whole call.

        Examples:
            The query vector is truncated to three floats on this page; pass your
            index's full dimension.

            .. code-block:: python

                results = idx.query_namespaces(
                    vector=[0.012, -0.087, 0.153],
                    namespaces=["articles-en", "articles-fr", "articles-de"],
                    metric="cosine",
                    top_k=10,
                )
                for match in results.matches:
                    print(match.id, match.score)

            On a sparse-only index, pass *sparse_vector* instead and rank by
            ``"dotproduct"``:

            .. code-block:: python

                results = idx.query_namespaces(
                    sparse_vector={"indices": [17, 42, 108], "values": [0.4, 0.9, 0.2]},
                    namespaces=["docs-en", "docs-fr"],
                    metric="dotproduct",
                    top_k=10,
                )

        .. seealso::
           :meth:`query` — one namespace, and the place every argument here is
           documented in full.
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

        # Submit every query before iterating results so queries run concurrently;
        # collect results in input namespace order so the aggregator's
        # insertion-order tie-break stays deterministic across runs.
        with ThreadPoolExecutor(max_workers=min(len(namespaces), 32)) as pool:
            futures = [pool.submit(self.query, namespace=ns, **query_kwargs) for ns in namespaces]
            for ns, future in zip(namespaces, futures, strict=True):
                aggregator.add_results(ns, future.result())

        return aggregator.get_results()

    def fetch(
        self,
        *,
        ids: Sequence[str],
        namespace: str = "",
        timeout: float | None = None,
    ) -> FetchResponse:
        """Fetch vectors by their IDs, exactly as stored.

        A lookup, not a search: nothing is ranked and no score is returned. An ID
        that is not in the namespace is silently absent from the result, so compare
        the keys you got back against the ones you asked for.

        Args:
            ids (list[str]): Vector IDs to fetch, e.g.
                ``["article-101", "article-102"]``. Must be non-empty, and every ID
                must be 1-512 ASCII characters without a NUL.
            namespace (str): Namespace to fetch from, e.g. ``"articles-en"``.
                Defaults to the index's default namespace.
            timeout (float | None): Per-request timeout in seconds. Overrides the
                client-level default for this call only.

        Returns:
            :class:`FetchResponse` with ``vectors``, a map of ID to
            :class:`Vector` holding values and metadata as stored, plus
            ``namespace`` and ``usage``. IDs the namespace does not hold are absent
            from the map rather than raising.

        Raises:
            :exc:`PineconeValueError`: If *ids* is empty or holds an ID that is not
                1-512 ASCII characters without a NUL. Raised before any HTTP
                request is made.

        Examples:
            .. code-block:: python

                wanted = ["article-101", "article-102"]
                response = idx.fetch(ids=wanted, namespace="articles-en")
                for vid, vec in response.vectors.items():
                    print(vid, vec.metadata)
                print("not in this namespace:", set(wanted) - set(response.vectors))

        .. seealso::
           - :meth:`fetch_by_metadata` — when you know the metadata you want rather
             than the IDs.
           - :meth:`query` — when you want the nearest vectors rather than named
             ones.
        """
        require_valid_vector_ids("ids", ids)

        params: dict[str, Any] = {"ids": ids}
        if namespace:
            params["namespace"] = namespace

        logger.info("Fetching %d vectors", len(ids))
        response = self._http.get("/vectors/fetch", timeout=timeout, params=params)
        result = self._adapter.to_fetch_response(response.content)
        result.response_info = extract_response_info(response)
        logger.debug("Fetched %d vectors", len(result.vectors))
        return result

    def fetch_by_metadata(
        self,
        *,
        filter: Mapping[str, Any],
        namespace: str = "",
        limit: int | None = None,
        pagination_token: str | None = None,
        timeout: float | None = None,
    ) -> FetchByMetadataResponse:
        """Fetch one page of the vectors whose metadata matches a filter.

        A lookup, not a search: matches are not ranked and carry no score. One page
        is returned per call, so follow ``pagination.next`` to reach the rest — see
        :doc:`/guides/pagination`.

        Args:
            filter (dict[str, Any]): Metadata filter expression, e.g.
                ``{"year": {"$gte": 2020}}``. Must carry at least one condition; an
                empty filter is rejected rather than treated as "match everything".
            namespace (str): Namespace to fetch from, e.g. ``"movies-en"``.
                Defaults to the index's default namespace.
            limit (int | None): Maximum number of vectors in this page, 1-10000.
                Omit to let the server choose the page size.
            pagination_token (str | None): ``pagination.next`` from the previous
                response. ``None`` (default) fetches the first page.
            timeout (float | None): Per-request timeout in seconds. Overrides the
                client-level default for this call only.

        Returns:
            :class:`FetchByMetadataResponse` with ``vectors`` as stored, plus
            ``namespace``, ``usage``, and ``pagination`` whose ``next`` is the token
            for the following page or ``None`` on the last one.

        Raises:
            :exc:`PineconeValueError`: If *filter* is empty or *limit* falls outside
                1-10000. Raised before any HTTP request is made.

        Examples:
            .. code-block:: python

                response = idx.fetch_by_metadata(
                    filter={"genre": "comedy", "year": {"$gte": 2020}},
                    namespace="movies-en",
                )
                for vid, vec in response.vectors.items():
                    print(vid, vec.metadata)

        .. seealso::
           - :meth:`fetch` — when you already know the IDs.
           - :meth:`query` — when you want the nearest vectors to a query rather
             than every vector a filter admits.
           - :doc:`/guides/pagination` — walking every page.
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
        response = self._http.post("/vectors/fetch_by_metadata", timeout=timeout, json=body)
        result = self._adapter.to_fetch_by_metadata_response(response.content)
        result.response_info = extract_response_info(response)
        return result

    def delete(
        self,
        *,
        ids: Sequence[str] | None = None,
        delete_all: bool = False,
        filter: Mapping[str, Any] | None = None,
        namespace: str = "",
        timeout: float | None = None,
    ) -> None:
        """Delete vectors from a namespace by ID, by filter, or all of them.

        Exactly one selector: *ids*, *filter*, or ``delete_all=True``. The delete is
        irreversible and IDs the namespace does not hold are ignored rather than
        reported, so a successful call is not evidence anything was deleted.

        Args:
            ids (list[str] | None): Vector IDs to delete, e.g.
                ``["article-101", "article-102"]``. Every ID must be 1-512 ASCII
                characters without a NUL.
            delete_all (bool): Delete every vector in *namespace*. The namespace
                itself survives; :meth:`delete_namespace` removes that too.
            filter (dict[str, Any] | None): Metadata filter expression selecting
                what to delete, e.g. ``{"status": {"$eq": "retracted"}}``. Must
                carry at least one condition, and cannot be combined with *ids* —
                see the note below.
            namespace (str): Namespace to delete from, e.g. ``"articles-en"``.
                Defaults to the index's default namespace.
            timeout (float | None): Per-request timeout in seconds. Overrides the
                client-level default for this call only.

        Returns:
            None — a successful delete returns no payload.

        Raises:
            :exc:`PineconeValueError`: If zero or more than one selector is given,
                if *filter* is empty, or if an ID is not legal. Raised before any
                HTTP request is made.
            :exc:`ApiError`: If a by-filter delete carries a text-match operator, or
                the index is a dedicated index scaled to zero replicas.

        Examples:
            By ID:

            .. code-block:: python

                idx.delete(ids=["article-101", "article-102"], namespace="articles-en")

            By metadata filter, which deletes every vector the filter admits:

            .. code-block:: python

                idx.delete(
                    filter={"status": {"$eq": "retracted"}},
                    namespace="articles-en",
                )

            Emptying a whole namespace is unbounded and cannot be undone:

            .. code-block:: python

                idx.delete(delete_all=True, namespace="articles-staging")

        .. note::
           Three things are true only of a by-filter delete. *ids* alongside
           *filter* is rejected here rather than sent, because the server lets the
           filter win and would delete everything it matches rather than the
           intersection — :meth:`query` with the filter first, then delete the IDs
           you got back. A text-match operator (``$match_phrase``, ``$match_all``,
           ``$match_any``) is rejected rather than ignored, because evaluated
           against metadata it matches everything and would widen the delete to
           every record the rest of the filter admits; text matching belongs in
           :meth:`search`. And a by-filter delete reads before it writes, so a
           dedicated index scaled to zero replicas refuses it — add replicas first.
           Deleting by ID or with ``delete_all`` is subject to none of this.

        .. seealso::
           :meth:`delete_namespace` — removes the namespace along with everything
           in it, where ``delete_all=True`` empties it and leaves it in place.
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
        self._http.post("/vectors/delete", timeout=timeout, json=body)

    def update(
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

        A partial update: fields you do not mention keep the values they had, so
        ``set_metadata={"year": 2021}`` leaves every other metadata key in place.
        Exactly one selector — *id* or *filter* — and a by-filter update is
        metadata-only, since *values* and *sparse_values* belong to one record. The
        write applies asynchronously, so a read straight afterwards can still see
        the old value.

        Args:
            id (str | None): ID of the one vector to patch, e.g. ``"article-101"``.
                Must be 1-512 ASCII characters without a NUL.
            values (list[float] | None): Replacement dense values, at your index's
                dimension. Only with *id*.
            sparse_values (SparseValues | dict[str, Any] | None): Replacement sparse
                vector, with ``indices`` and ``values`` keys. Only with *id*.
            set_metadata (dict[str, Any] | None): Metadata keys to set or overwrite,
                e.g. ``{"year": 2021}``. Keys you omit are left as they are; this
                never clears a field.
            namespace (str): Namespace to target, e.g. ``"movies-en"``. Defaults to
                the index's default namespace.
            filter (dict[str, Any] | None): Metadata filter expression selecting
                which vectors to patch, e.g. ``{"genre": {"$eq": "drama"}}``. Must
                carry at least one condition — see the note below.
            dry_run (bool): Report how many records the *filter* would touch without
                writing anything. Ignored for a by-ID update. Run it first when the
                filter is broader than you can check by eye.
            timeout (float | None): Per-request timeout in seconds. Overrides the
                client-level default for this call only.

        Returns:
            :class:`UpdateResponse` whose ``matched_records`` counts the records
            patched, or under ``dry_run`` the records that would have been. It is
            ``None`` when the server does not report a count, which a by-ID update
            does not.

        Raises:
            :exc:`PineconeValueError`: If both or neither of *id* and *filter* are
                given, if *filter* is combined with *values* or *sparse_values*, or
                if *filter* is empty. Raised before any HTTP request is made.
            :exc:`ApiError`: If a by-filter update carries a text-match operator, or
                the index is a dedicated index scaled to zero replicas.

        Examples:
            Replacing one vector's values, truncated here to three floats:

            .. code-block:: python

                idx.update(
                    id="article-101",
                    values=[0.012, -0.087, 0.153],
                    namespace="articles-en",
                )

            Patching metadata across a filter. ``dry_run`` reports the reach first,
            and the ``genre`` of every patched record survives untouched:

            .. code-block:: python

                preview = idx.update(
                    filter={"genre": {"$eq": "drama"}},
                    set_metadata={"reviewed": True},
                    namespace="movies-en",
                    dry_run=True,
                )
                print(preview.matched_records)
                idx.update(
                    filter={"genre": {"$eq": "drama"}},
                    set_metadata={"reviewed": True},
                    namespace="movies-en",
                )

        .. note::
           Two things are true only of a by-filter update. A text-match operator
           (``$match_phrase``, ``$match_all``, ``$match_any``) is rejected rather
           than ignored, because evaluated against metadata it matches everything
           and would widen the patch to every record the rest of the filter admits;
           text matching belongs in :meth:`search`. And a by-filter update reads
           before it writes, so a dedicated index scaled to zero replicas refuses it
           — add replicas first. Updating by ID is subject to neither.

        .. seealso::
           :meth:`upsert` — replaces a whole vector rather than patching it, and
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
        response = self._http.post("/vectors/update", timeout=timeout, json=body)
        result = self._adapter.to_update_response(response.content)
        result.response_info = extract_response_info(response)
        return result

    def describe_index_stats(
        self,
        *,
        filter: Mapping[str, Any] | None = None,
        timeout: float | None = None,
    ) -> DescribeIndexStatsResponse:
        """Report vector counts, dimension, and fullness for this index.

        The counts lag writes, so a vector just upserted may not be counted yet.

        Args:
            filter (dict[str, Any] | None): Metadata filter expression. Accepted
                for API compatibility, but a non-empty filter is rejected for
                every index type, so the call fails instead of returning
                filtered counts. Leave it unset: the statistics returned always
                describe the whole index.
            timeout (float | None): Per-request timeout in seconds. Overrides
                the client-level default for this call only.

        Returns:
            :class:`DescribeIndexStatsResponse` with ``total_vector_count``,
            ``dimension``, ``index_fullness``, and ``namespaces`` mapping each
            namespace name to a summary carrying its ``vector_count``. The default
            namespace appears in that mapping under the empty string.

        Raises:
            :exc:`ApiError`: If *filter* is non-empty. Every index type rejects it.

        Examples:

            >>> idx = pc.index(name="article-search")
            >>> idx.describe_index_stats().total_vector_count
            0

            Per-namespace counts come back on the same response:

            .. code-block:: python

                stats = idx.describe_index_stats()
                print(stats.dimension, stats.index_fullness)
                for name, summary in stats.namespaces.items():
                    print(name or "(default)", summary.vector_count)

        .. seealso::
           :meth:`list_namespaces` — namespace record counts alongside each
           namespace's schema and ``size_bytes``.
        """
        body: dict[str, Any] = {}
        if filter is not None:
            body["filter"] = filter

        logger.info("Describing index stats")
        response = self._http.post("/describe_index_stats", timeout=timeout, json=body)
        result = self._adapter.to_stats_response(response.content)
        result.response_info = extract_response_info(response)
        return result

    def search(
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
        """Search records by text, vector, or ID, optionally reranking the hits.

        Pass *inputs* and the index's own embedding model turns your text into the
        query vector server-side — that is what separates this from :meth:`query`,
        where you supply the vector. A *vector* or an *id* works here too, for the
        cases where you already have one.

        Args:
            namespace (str): Namespace to search, e.g. ``"articles-en"``. Required
                and non-empty.
            top_k (int): Number of results to return, at least 1, e.g. ``10``.
            inputs (SearchInputs | dict[str, Any] | None): Inputs for
                server-side embedding (e.g. ``{"text": "query text"}``).
                Use :class:`SearchInputs` for typed key validation and IDE
                autocompletion (e.g. ``SearchInputs(text="query text")``).
            vector (list[float] | dict[str, Any] | None): Query vector. Pass a
                ``list[float]`` for a dense-only query (wrapped automatically as
                ``{"values": [...]}``) or a dict for sparse/hybrid queries with
                keys ``values``, ``sparse_indices``, and/or ``sparse_values``
                (passed through as-is). See :class:`SearchQueryVector` for the
                typed helper.
            id (str | None): ID of an existing record to use as the query.
            filter (dict[str, Any] | None): Metadata filter expression.
            fields (list[str] | None): Field names to include in results.
                When ``None``, the server returns all available fields.
            rerank (RerankConfig | dict[str, Any] | None): Reranking
                configuration with ``model`` (required), ``rank_fields``
                (required), and optional ``top_n``, ``parameters``, ``query``
                keys. Use :class:`RerankConfig` for IDE autocompletion.
            match_terms (dict[str, Any] | None): Term-matching constraint for
                sparse search. Requires keys ``"strategy"`` (currently only
                ``"all"``) and ``"terms"`` (list of strings).
                Valid only on a text query — combined with ``vector`` or ``id``
                it is rejected — and only on a sparse index whose embedding model
                supports it; the server names the supported model when it
                refuses. ``None`` disables term matching.
            timeout (float | None): Per-request timeout in seconds. Overrides
                the client-level default for this call only.
            query (SearchQuery | dict[str, Any] | None): The pre-flattening form of
                this call — ``top_k`` plus one of ``inputs``, ``vector``, or ``id``,
                nested in one mapping. Pass the fields directly instead.

        Returns:
            :class:`SearchRecordsResponse` whose ``result.hits`` are ordered from
            most to least relevant. Read each :class:`Hit` as ``hit.id``,
            ``hit.score``, and ``hit.fields``; the hits nest one level down, under
            ``result``. ``usage`` breaks the cost out by stage.

        Raises:
            :exc:`PineconeValueError`: If ``namespace`` is not a non-empty string,
                ``top_k`` is below 1, or ``rerank`` is missing ``model`` or
                ``rank_fields``. Raised before any HTTP request is made.
            :exc:`TypeError`: If ``query`` is combined with any of the flat keyword
                arguments it replaces (``top_k``, ``inputs``, ``vector``, ``id``,
                ``filter``, ``match_terms``) — pass one form or the other, not
                both — or if ``query`` is neither a :class:`SearchQuery` nor a
                mapping.

        Examples:
            Text in, embedded server-side. The hits nest one level down, under
            ``result`` — forgetting that is the usual first stumble here:

            >>> idx = pc.index(name="article-search")
            >>> response = idx.search(
            ...     namespace="articles-en",
            ...     top_k=2,
            ...     inputs={"text": "benefits of vector databases for search"},
            ...     fields=["chunk_text"],
            ... )
            >>> [(hit.id, hit.score) for hit in response.result.hits]
            [('article-1', 0.92), ('article-2', 0.74)]
            >>> response.result.hits[0].fields["chunk_text"]
            'Vector databases accelerate AI search.'

            Reranking in the same call retrieves ``top_k`` and returns only the
            ``top_n`` the reranker likes best, so the hit count drops:

            >>> response = idx.search(
            ...     namespace="articles-en",
            ...     top_k=2,
            ...     inputs={"text": "benefits of vector databases"},
            ...     rerank={
            ...         "model": "bge-reranker-v2-m3",
            ...         "rank_fields": ["chunk_text"],
            ...         "top_n": 1,
            ...     },
            ... )
            >>> len(response.result.hits)
            1
            >>> response.usage.rerank_units
            1

        .. seealso::
           - :meth:`query` — for a vector-based index, where you supply the query
             vector and nothing is embedded for you.
           - :attr:`documents` — ``documents.search`` for a schema-based index,
             which stores JSON records and ranks with ``score_by`` clauses.
           - ``pc.inference.rerank`` — reranking on its own, for hits that came
             from somewhere other than this index.
        """
        if not isinstance(namespace, str):
            raise ValidationError("namespace must be a string")
        if not namespace or not namespace.strip():
            raise ValidationError("namespace must be a non-empty string")
        body = _build_search_records_body(
            method_name="Index.search",
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
        response = self._http.post(
            f"/records/namespaces/{quote(namespace, safe='')}/search", timeout=timeout, json=body
        )
        result = self._adapter.to_search_response(response.content)
        result.response_info = extract_response_info(response)
        return result

    def search_records(
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
        """Alias for :meth:`search`, kept for callers written against 9.x.

        Every argument, return value and error is :meth:`search`'s. Call that one in
        new code; nothing here differs.
        """
        return self.search(
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

    def create_namespace(
        self,
        *,
        name: str,
        schema: dict[str, Any] | None = None,
    ) -> NamespaceDescription:
        """Create a named namespace in the index.

        Args:
            name (str): Name for the new namespace, e.g. ``"movies-en"``. Must be
                ASCII, must not contain the NUL character, and must be 1-512
                characters long. ``__default__`` is reserved and cannot be created:
                it names the namespace requests address when they omit a namespace,
                so it always exists.
            schema (dict[str, Any] | None): Optional metadata-index configuration,
                ``{"fields": {<field>: {"filterable": True}}}``. Omitting it does
                not mean "index everything": the namespace inherits the index's
                own metadata-index configuration, so an index that restricts which
                fields are indexed passes that restriction on. Supply *schema* to
                override the inherited configuration for this namespace, indexing
                exactly the fields listed. ``filterable`` is required on each field
                and must be ``True`` — to leave a field unindexed, omit it from
                ``fields``.

        Returns:
            :class:`NamespaceDescription` with the namespace name, record count,
            schema, indexed fields, and ``size_bytes``.

        Raises:
            :exc:`PineconeValueError`: If *name* violates the rules above, or *schema*
                is malformed. Raised before any HTTP request is made.
            :exc:`ConflictError`: a namespace of that name already exists.

        Examples:
            .. code-block:: python

                ns = idx.create_namespace(name="movies-en")
                print(ns.name, ns.record_count, ns.size_bytes)

            Naming the filterable fields up front overrides what the namespace would
            otherwise inherit from the index:

            .. code-block:: python

                ns = idx.create_namespace(
                    name="movies-fr",
                    schema={"fields": {"genre": {"filterable": True}}},
                )
                print(ns.indexed_fields)

        .. seealso::
           - :meth:`describe_namespace` — the same description for a namespace that
             already exists.
           - :meth:`list_namespaces` — the namespaces the index already has.
        """
        require_creatable_namespace_name("name", name)

        body: dict[str, Any] = {"name": name}
        if schema is not None:
            require_valid_namespace_schema("schema", schema)
            body["schema"] = schema

        logger.info("Creating namespace %r", name)
        response = self._http.post("/namespaces", json=body)
        return self._adapter.to_namespace_description(response.content)

    def describe_namespace(
        self,
        *,
        name: str | None = None,
        **kwargs: str,
    ) -> NamespaceDescription:
        """Describe a namespace by name.

        This operation is rate limited per index, independently of the other
        namespace operations. Prefer :meth:`list_namespaces` when describing more
        than one namespace: it returns the same information for every namespace
        in a single request and is not subject to that limit.

        Args:
            name (str): Name of the namespace to describe. Must be ASCII, must not
                contain the NUL character, and must be 1-512 characters long. Pass
                ``__default__`` to describe the namespace that requests address
                when they omit a namespace.

        Returns:
            :class:`NamespaceDescription` with the namespace name, record count,
            schema, indexed fields, and ``size_bytes``. ``size_bytes`` is
            approximate: data written before size tracking reads as 0, and
            recently deleted data may still be counted; compaction converges the
            value.

        Raises:
            :exc:`PineconeValueError`: If *name* violates the rules above. Raised
                before any HTTP request is made.
            :exc:`TypeError`: If unexpected keyword arguments are passed.
            :exc:`NotFoundError`: no namespace of that name exists on the index.
            :exc:`RateLimitError`: this operation's per-index limit was
                exceeded. Use :meth:`list_namespaces` to describe many namespaces.

        Examples:
            .. code-block:: python

                ns = idx.describe_namespace(name="movies-en")
                print(ns.name, ns.record_count, ns.size_bytes)

            The namespace that unnamespaced requests address answers to
            ``__default__``:

            .. code-block:: python

                ns = idx.describe_namespace(name="__default__")
                print(ns.record_count)

        .. seealso::
           :meth:`list_namespaces` — every namespace at once, and the operation to
           reach for when you are describing more than one.
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
        response = self._http.get(f"/namespaces/{quote(effective, safe='')}")
        return self._adapter.to_namespace_description(response.content)

    def delete_namespace(
        self,
        *,
        name: str | None = None,
        timeout: float | None = None,
        **kwargs: str,
    ) -> None:
        """Delete a namespace and everything in it.

        Irreversible: every vector in the namespace goes with it, and the namespace
        itself stops existing. To empty a namespace but keep it, use
        :meth:`delete` with ``delete_all=True``.

        Args:
            name (str): Name of the namespace to delete, e.g.
                ``"movies-deprecated"``. Must be ASCII, must not contain the NUL
                character, and must be 1-512 characters long.
            timeout (float | None): Per-request timeout in seconds. Overrides
                the client-level default for this call only.

        Returns:
            None — a successful delete returns no payload.

        Raises:
            :exc:`PineconeValueError`: If *name* violates the rules above. Raised
                before any HTTP request is made.
            :exc:`TypeError`: If unexpected keyword arguments are passed.
            :exc:`NotFoundError`: no namespace of that name exists on the index.

        Examples:
            .. code-block:: python

                idx.delete_namespace(name="movies-deprecated")

        .. seealso::
           :meth:`delete` — ``delete_all=True`` empties a namespace and leaves it
           in place.
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
        self._http.delete(f"/namespaces/{quote(effective, safe='')}", timeout=timeout)

    def list_namespaces_paginated(
        self,
        *,
        prefix: str | None = None,
        limit: int | None = None,
        pagination_token: str | None = None,
    ) -> ListNamespacesResponse:
        """Fetch one page of namespace descriptions, holding the token yourself.

        :meth:`list_namespaces` walks the pages for you; reach for this one when you
        need to persist the token between calls or hand it to a caller of your own.
        See :doc:`/guides/pagination`.

        Args:
            prefix (str | None): Return only namespaces whose names start with this
                prefix, e.g. ``"movies-"``. Must be ASCII, must not contain the NUL
                character, and must be at most 512 characters. The empty prefix
                matches every namespace.
            limit (int | None): Maximum number of namespaces in this page, 1-100.
            pagination_token (str | None): ``pagination.next`` from the previous
                response. ``None`` (default) fetches the first page.

        Returns:
            :class:`ListNamespacesResponse` with ``namespaces``, each a
            :class:`NamespaceDescription` carrying its record count, schema,
            indexed fields and ``size_bytes``, plus a total count and
            ``pagination`` whose ``next`` is ``None`` on the last page.

        Raises:
            :exc:`PineconeValueError`: If *prefix* or *limit* violates the rules
                above. Raised before any HTTP request is made.

        Examples:
            .. code-block:: python

                response = idx.list_namespaces_paginated(prefix="movies-", limit=10)
                for ns in response.namespaces:
                    print(ns.name, ns.record_count, ns.size_bytes)
                next_token = response.pagination.next if response.pagination else None

        .. seealso::
           - :meth:`list_namespaces` — the same listing with the token handled for
             you.
           - :doc:`/guides/pagination` — how the SDK pages generally.
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
        response = self._http.get("/namespaces", params=params)
        return self._adapter.to_list_namespaces_response(response.content)

    def list_namespaces(
        self,
        *,
        prefix: str | None = None,
        limit: int | None = None,
    ) -> Iterator[ListNamespacesResponse]:
        """List every namespace, a page at a time.

        Yields one :class:`ListNamespacesResponse` per page and follows the
        pagination tokens itself, so nothing is requested until you iterate. A page
        describes every namespace it holds in one request, which makes this the
        operation to reach for over repeated :meth:`describe_namespace` calls —
        those are rate limited per index and this is not.

        Args:
            prefix (str | None): Return only namespaces whose names start with this
                prefix, e.g. ``"movies-"``. Must be ASCII, must not contain the NUL
                character, and must be at most 512 characters. The empty prefix
                matches every namespace.
            limit (int | None): Maximum number of namespaces per page, 1-100.

        Yields:
            :class:`ListNamespacesResponse` per page, each carrying ``namespaces``
            of :class:`NamespaceDescription` with record count, schema, indexed
            fields and ``size_bytes``. A page with no namespaces is skipped rather
            than yielded.

        Raises:
            :exc:`PineconeValueError`: If *prefix* or *limit* violates the rules
                above. Raised on first iteration, not at the call.

        Examples:
            .. code-block:: python

                for page in idx.list_namespaces(prefix="movies-"):
                    for ns in page.namespaces:
                        print(ns.name, ns.record_count, ns.size_bytes)

        .. seealso::
           - :meth:`list_namespaces_paginated` — one page, with the token in your
             hands.
           - :meth:`describe_namespace` — one namespace, when you know its name.
           - :doc:`/guides/pagination` — how the SDK pages generally.
        """
        pagination_token: str | None = None
        while True:
            page = self.list_namespaces_paginated(
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

    def list_paginated(
        self,
        *,
        prefix: str | None = None,
        limit: int | None = None,
        pagination_token: str | None = None,
        namespace: str = "",
        timeout: float | None = None,
    ) -> ListResponse:
        """Fetch one page of vector IDs, holding the token yourself.

        IDs only — no values and no metadata. :meth:`list` walks the pages for you;
        reach for this one when you need to persist the token between calls. See
        :doc:`/guides/pagination`.

        Args:
            prefix (str | None): Return only IDs starting with this prefix, e.g.
                ``"article-2024#"``. At most 512 ASCII characters without a NUL; the
                empty prefix matches everything.
            limit (int | None): Maximum number of IDs in this page, 1-100.
            pagination_token (str | None): ``pagination.next`` from the previous
                response. ``None`` (default) fetches the first page.
            namespace (str): Namespace to list from, e.g. ``"articles-en"``.
                Defaults to the index's default namespace.
            timeout (float | None): Per-request timeout in seconds. Overrides
                the client-level default for this call only.

        Returns:
            :class:`ListResponse` with ``vectors`` — each carrying an ``id`` and
            nothing else — plus ``namespace``, ``usage``, and ``pagination`` whose
            ``next`` is the token for the following page or ``None`` on the last one.

        Raises:
            :exc:`PineconeValueError`: If *prefix* is not legal or *limit* falls
                outside 1-100. Raised before any HTTP request is made.

        Examples:
            .. code-block:: python

                response = idx.list_paginated(
                    prefix="article-2024#",
                    limit=50,
                    namespace="articles-en",
                )
                for item in response.vectors:
                    print(item.id)
                next_token = response.pagination.next if response.pagination else None

        .. seealso::
           - :meth:`list` — the same listing with the token handled for you.
           - :meth:`fetch` — the vectors behind those IDs.
           - :doc:`/guides/pagination` — how the SDK pages generally.
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
        response = self._http.get("/vectors/list", timeout=timeout, params=params)
        result = self._adapter.to_list_response(response.content)
        result.response_info = extract_response_info(response)
        return result

    def list(
        self,
        *,
        prefix: str | None = None,
        limit: int | None = None,
        namespace: str = "",
        timeout: float | None = None,
    ) -> Iterator[ListResponse]:
        """List vector IDs in a namespace, a page at a time.

        IDs only — no values and no metadata. Yields one :class:`ListResponse` per
        page and follows the pagination tokens itself, so nothing is requested until
        you iterate, and a bad *prefix* or *limit* is not reported until then either.

        Args:
            prefix (str | None): Return only IDs starting with this prefix, e.g.
                ``"article-2024#"``. At most 512 ASCII characters without a NUL; the
                empty prefix matches everything.
            limit (int | None): Maximum number of IDs per page, 1-100.
            namespace (str): Namespace to list from, e.g. ``"articles-en"``.
                Defaults to the index's default namespace.
            timeout (float | None): Per-request timeout in seconds, applied to
                each underlying page request. Overrides the client-level
                default for this call only.

        Yields:
            :class:`ListResponse` per page, each carrying ``vectors`` of IDs. A page
            with no IDs is skipped rather than yielded.

        Raises:
            :exc:`PineconeValueError`: If *prefix* is not legal or *limit* falls
                outside 1-100. Raised on first iteration, not at the call.

        Examples:
            .. code-block:: python

                for page in idx.list(prefix="article-2024#", namespace="articles-en"):
                    ids = [item.id for item in page.vectors]
                    fetched = idx.fetch(ids=ids, namespace="articles-en")
                    for vid, vec in fetched.vectors.items():
                        print(vid, vec.metadata)

        .. seealso::
           - :meth:`list_paginated` — one page, with the token in your hands.
           - :meth:`fetch` — the vectors behind a page of IDs.
           - :doc:`/guides/pagination` — how the SDK pages generally.
        """
        pagination_token: str | None = None
        while True:
            page = self.list_paginated(
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

    def _validate_import_id(self, id: str | int) -> str:
        """Validate and normalize an import operation ID.

        Args:
            id: Import operation ID. If int, converted to str silently.

        Returns:
            The validated string ID.

        Raises:
            :exc:`PineconeValueError`: If the ID is empty or exceeds 1000 characters.
        """
        str_id = str(id) if isinstance(id, int) else id
        if not str_id or len(str_id) > 1000:
            raise ValidationError(
                "import id must be between 1 and 1000 characters, "
                f"got {len(str_id) if str_id else 0}"
            )
        return str_id

    def start_import(
        self,
        uri: str,
        *,
        error_mode: str | None = None,
        integration_id: str | None = None,
    ) -> StartImportResponse:
        """Start a server-side bulk import of vectors from cloud storage.

        Returns as soon as the import is accepted, not when it finishes: the work
        happens server-side, and :meth:`describe_import` is how you learn whether it
        completed. Nothing here polls for you.

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
            integration_id (str | None): Optional integration ID for the import.

        Returns:
            :class:`StartImportResponse` with ``id``, the handle every other import
            method takes.

        Raises:
            :exc:`PineconeValueError`: If *error_mode* is supplied but is neither
                ``"continue"`` nor ``"abort"``. Raised before any HTTP request is
                made.
            :exc:`ApiError`: If *uri* is empty or longer than the server accepts,
                uses an unsupported scheme, is an ``s3://`` URI on an index not
                hosted on AWS, or names an S3 directory bucket, which imports do
                not support.

        Examples:
            The call returns the moment the import is accepted:

            >>> idx = pc.index(name="article-search")
            >>> response = idx.start_import(uri="s3://acme-exports/articles/")

            Waiting it out is on you; three of the five statuses are terminal:

            .. code-block:: python

                import time

                import_op = idx.describe_import(response.id)
                while import_op.status not in ("Completed", "Failed", "Cancelled"):
                    time.sleep(10)
                    import_op = idx.describe_import(response.id)
                print(import_op.status, import_op.records_imported)

            ``error_mode="continue"`` finishes the import around records it cannot
            read, rather than stopping at the first one:

            >>> response = idx.start_import(
            ...     uri="s3://acme-exports/articles/",
            ...     error_mode="continue",
            ... )

        .. note::
           *uri* must name a directory of Parquet files following Pinecone's import
           schema. See the
           `import guide <https://docs.pinecone.io/guides/data/understanding-imports>`_
           for that schema and the supported storage formats.

        .. seealso::
           - :meth:`describe_import` — progress, and the terminal status.
           - :meth:`cancel_import` — stopping one that is still running.
           - :meth:`upsert` — the right tool below the millions of vectors an
             import is for.
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
        response = self._http.post("/bulk/imports", json=body)
        return self._imports_adapter.to_start_import_response(response.content)

    def describe_import(self, id: str | int) -> ImportModel:
        """Describe a bulk import operation by ID.

        Args:
            id: The ``id`` :meth:`start_import` returned, e.g. ``"import-123"``.
                An ``int`` is accepted and stringified. 1-1000 characters.

        Returns:
            :class:`ImportModel` with ``status`` — one of ``"Pending"``,
            ``"InProgress"``, ``"Failed"``, ``"Completed"``, ``"Cancelled"``, the
            last three terminal — plus ``percent_complete``, ``records_imported``,
            ``uri``, and ``error`` when it failed.

        Raises:
            :exc:`PineconeValueError`: If *id* is empty or over 1000 characters.
                Raised before any HTTP request is made.

        Examples:
            >>> idx = pc.index("article-search")
            >>> import_op = idx.describe_import("import-123")
            >>> print(import_op.status, import_op.percent_complete, import_op.records_imported)
            InProgress 50.0 12000
            >>> import_op.uri
            's3://acme-exports/articles/'

            Read ``error`` only once ``status`` is ``"Failed"``; it is ``None`` otherwise.

            .. code-block:: python

                if import_op.status == "Failed":
                    print(import_op.error)

        .. seealso::
           - :meth:`start_import` — starting one, and the ``id`` these methods take.
           - :meth:`list_imports` — every import on the index rather than one.
        """
        str_id = self._validate_import_id(id)
        logger.info("Describing import %s", str_id)
        response = self._http.get(f"/bulk/imports/{quote(str_id, safe='')}")
        return self._imports_adapter.to_import_model(response.content)

    def cancel_import(self, id: str | int) -> None:
        """Cancel a bulk import operation by ID.

        Args:
            id: The ``id`` :meth:`start_import` returned, e.g. ``"import-123"``.
                An ``int`` is accepted and stringified. 1-1000 characters.

        Returns:
            None — a successful cancellation returns no payload. Poll
            :meth:`describe_import` to see the operation reach ``"Cancelled"``.

        Raises:
            :exc:`PineconeValueError`: If *id* is empty or over 1000 characters.
                Raised before any HTTP request is made.

        Examples:
            >>> idx = pc.index(name="article-search")
            >>> idx.cancel_import("import-123")

        .. seealso::
           - :meth:`start_import` — starting one, and the ``id`` these methods take.
           - :meth:`list_imports` — every import on the index rather than one.
        """
        str_id = self._validate_import_id(id)
        logger.info("Cancelling import %s", str_id)
        self._http.delete(f"/bulk/imports/{quote(str_id, safe='')}")

    def list_imports(
        self,
        *,
        limit: int | None = None,
        pagination_token: str | None = None,
    ) -> Iterator[ImportModel]:
        """List every bulk import on this index, following pagination.

        Yields the :class:`ImportModel` objects themselves rather than pages, and
        fetches the next page as you exhaust the current one, so nothing is
        requested until you iterate. See :doc:`/guides/pagination`.

        Args:
            limit (int | None): Maximum number of imports per page, e.g. ``10``.
                Omit to let the server choose the page size.
            pagination_token (str | None): Token to resume from, when you are
                continuing a listing rather than starting one.

        Yields:
            :class:`ImportModel` per import operation, oldest page first.

        Raises:
            :exc:`ApiError`: If a page request fails part-way through the listing;
                the imports already yielded are still yours, the rest are not.
        Examples:
            >>> idx = pc.index(name="article-search")
            >>> for imp in idx.list_imports():
            ...     print(imp.id, imp.status, imp.uri)

        .. seealso::
           - :meth:`list_imports_paginated` — one page, with the token in your
             hands.
           - :meth:`describe_import` — one import, when you know its ``id``.
           - :doc:`/guides/pagination` — how the SDK pages generally.
        """
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if pagination_token is not None:
            params["paginationToken"] = pagination_token

        while True:
            response = self._http.get("/bulk/imports", params=params)
            import_list = self._imports_adapter.to_import_list(response.content)
            yield from import_list
            next_token = import_list.pagination.next if import_list.pagination else None
            if next_token is None:
                break
            params["paginationToken"] = next_token

    def list_imports_paginated(
        self,
        *,
        limit: int | None = None,
        pagination_token: str | None = None,
    ) -> ImportList:
        """Fetch one page of bulk imports, holding the token yourself.

        :meth:`list_imports` walks the pages for you; reach for this one when you
        need to persist the token between calls. See :doc:`/guides/pagination`.

        Args:
            limit (int | None): Maximum number of imports in this page, e.g.
                ``10``.
            pagination_token (str | None): ``pagination.next`` from the previous
                response. ``None`` (default) fetches the first page.

        Returns:
            :class:`ImportList` you can iterate for this page's
            :class:`ImportModel` objects, with ``pagination.next`` holding the
            token for the following page or ``None`` on the last one.
        Examples:
            >>> idx = pc.index(name="article-search")
            >>> page = idx.list_imports_paginated(limit=10)
            >>> [imp.id for imp in page]
            []

            ``pagination`` is absent on the last page, which is how you know to
            stop:

            >>> page.pagination is None
            True

        .. seealso::
           - :meth:`list_imports` — the same listing with the token handled for
             you.
           - :doc:`/guides/pagination` — how the SDK pages generally.
        """
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if pagination_token is not None:
            params["paginationToken"] = pagination_token

        response = self._http.get("/bulk/imports", params=params)
        return self._imports_adapter.to_import_list(response.content)

    def close(self) -> None:
        """Close the underlying HTTP client and release its resources.

        Calls on a closed index fail. Prefer ``with`` over calling this by hand,
        which closes the client even if the body raises.

        Returns:
            None.

        Examples:
            .. code-block:: python

                with pc.index(name="article-search") as idx:
                    idx.upsert(
                        vectors=[("article-101", [0.012, -0.087, 0.153])],
                        namespace="articles-en",
                    )
        """
        self._http.close()
        legacy_pool = getattr(self, "_legacy_async_pool", None)
        if legacy_pool is not None:
            legacy_pool.close()

    def __enter__(self) -> Index:
        """Enter the context manager, returning this index.

        Returns:
            This :class:`Index` instance.

        Examples:
            .. code-block:: python

                with pc.index(name="article-search") as idx:
                    idx.upsert(
                        vectors=[("article-101", [0.012, -0.087, 0.153])],
                        namespace="articles-en",
                    )
        """
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit the context manager, calling :meth:`close`.

        Returns:
            None.
        """
        self.close()

    def __repr__(self) -> str:
        """Return developer-friendly representation."""
        return f"Index(host='{self._host}')"
