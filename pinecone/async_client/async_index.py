"""Asynchronous data plane client for a Pinecone index."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd  # type: ignore[import-untyped]

from pinecone._internal.adapters.imports_adapter import ImportsAdapter
from pinecone._internal.adapters.vectors_adapter import VectorsAdapter, extract_response_info
from pinecone._internal.adaptive import _AdaptiveLimiterRegistry
from pinecone._internal.batch import async_batch_execute
from pinecone._internal.batching import validate_batch_size
from pinecone._internal.config import PineconeConfig
from pinecone._internal.constants import DATA_PLANE_API_VERSION
from pinecone._internal.data_plane_helpers import (
    _build_search_records_body,
    _validate_host,
    _vector_to_dict,
)
from pinecone._internal.validation import require_in_range, require_positive
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


class AsyncIndex:
    """Asynchronous data plane client targeting a specific Pinecone index.

    Can be constructed directly with a host URL, or via the
    :meth:`AsyncPinecone.index` factory method.

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

    Raises:
        :exc:`PineconeValueError`: If no API key can be resolved or the host is invalid.

    Examples:

        .. code-block:: python

            from pinecone import AsyncIndex

            async with AsyncIndex(host="my-index-abc123.svc.pinecone.io", api_key="...") as idx:
                print(idx.host)
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
        self._limiter_registry = _limiter_registry

        from pinecone._internal.http_client import AsyncHTTPClient

        self._http = AsyncHTTPClient(config, DATA_PLANE_API_VERSION)
        self._adapter = VectorsAdapter()
        self._imports_adapter = ImportsAdapter()

        logger.info("AsyncIndex client created for host %s", self._host)

    @property
    def host(self) -> str:
        """The data plane host URL for this index."""
        return self._host

    async def upsert_records(
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
            records: List of record dicts. Each must contain an ``_id`` or
                ``id`` field. Additional fields are passed through for
                server-side embedding.
            namespace (str): Target namespace (required). Unlike :meth:`upsert`,
                namespace has no default because the records API requires an
                explicit namespace (must be non-empty).

        Returns:
            :class:`UpsertRecordsResponse` with the count of records submitted.

        Raises:
            :exc:`PineconeValueError`: If namespace is not a string or is empty/whitespace,
                records is empty, or a record is missing an identifier field.
            :exc:`ApiError`: If the API returns an error response.
            :exc:`PineconeConnectionError`: If a network-level connection
                fails (DNS, refused, transport error).
            :exc:`PineconeTimeoutError`: If the request exceeds the configured timeout.

        Examples:

            .. code-block:: python

                response = await idx.upsert_records(
                    namespace="articles-en",
                    records=[
                        {
                            "_id": "article-101",
                            "text": "Vector databases enable similarity search.",
                        },
                        {"_id": "article-102", "text": "RAG combines search with LLMs."},
                    ],
                )
                print(response.record_count)

        .. seealso::
           - :meth:`upsert` — for indexes where you provide your own vectors
             (no server-side embedding).
           - :meth:`start_import` — for bulk loading millions of vectors
             from cloud storage (S3, GCS).
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

        import orjson

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

        ndjson_lines = [orjson.dumps(r).decode("utf-8") for r in normalized]
        ndjson_body = "\n".join(ndjson_lines) + "\n"

        logger.info("Upserting %d records into namespace %r (NDJSON)", len(records), namespace)
        response = await self._http.post(
            f"/records/namespaces/{namespace}/upsert",
            timeout=timeout,
            content=ndjson_body.encode("utf-8"),
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
        max_concurrency: int = 4,
        timeout: float | None = None,
    ) -> UpsertResponse:
        """Upsert a batch of vectors into a namespace.

        If a vector with the same ID already exists in the namespace, it is
        overwritten.

        Args:
            vectors: Sequence of vectors to upsert. Each element can be a
                ``Vector`` instance, a tuple of ``(id, values)`` or
                ``(id, values, metadata)``, or a dict with ``id``, ``values``,
                and optional ``sparse_values`` / ``metadata`` keys.
            namespace (str): Target namespace. Defaults to the default
                (empty-string) namespace.
            batch_size (int | None): Split *vectors* into chunks of this size
                and send one request per chunk. Default ``None`` sends a single
                request (current behaviour). Must be a positive integer if
                provided.
            show_progress (bool): When ``True`` and ``tqdm`` is installed,
                display a progress bar across batches. Has no effect when
                ``batch_size`` is ``None`` or ``tqdm`` is not installed.
                Defaults to ``True``.
            max_concurrency (int): Asyncio concurrency limit for concurrent batch
                requests (range 1–64, default 4). Only used when ``batch_size``
                is set.
            timeout (float | None): Per-request timeout in seconds. Overrides
                the client-level default for this call only.

        Returns:
            :class:`UpsertResponse` with the count of vectors upserted.
            When ``batch_size`` triggers multiple requests, ``response_info``
            carries the aggregate LSN from all successful batches (or ``None``
            if no LSN headers were returned).

        Raises:
            :exc:`PineconeTypeError`: If a vector element is not a recognized format.
            :exc:`PineconeValueError`: If a vector element is malformed.
            :exc:`PineconeValueError`: If *batch_size* is not a positive integer.
            :exc:`PineconeValueError`: If *max_concurrency* is outside [1, 64].
            :exc:`ApiError`: If the API returns an error response.
            :exc:`PineconeConnectionError`: If a network-level connection
                fails (DNS, refused, transport error).
            :exc:`PineconeTimeoutError`: If the request exceeds the configured timeout.

        Examples:

            .. code-block:: python

                from pinecone import Vector

                response = await idx.upsert(
                    vectors=[
                        Vector(
                            id="article-101",
                            values=[0.012, -0.087, 0.153],  # truncated; use your actual dimension
                        ),
                        ("article-102", [0.045, 0.021, -0.064]),  # truncated
                        {"id": "article-103", "values": [0.091, -0.032, 0.178]},  # truncated
                    ],
                    namespace="articles-en",
                )
                print(response.upserted_count)

                # Upsert 1000 vectors in batches of 100
                response = await idx.upsert(
                    vectors=large_vector_list,
                    batch_size=100,
                    show_progress=True,
                )
                print(response.upserted_count)

        .. note::
           When ``batch_size`` is set, batches are submitted **concurrently** via an
           ``asyncio.Semaphore`` of ``max_concurrency`` slots (default 4, range 1–64).
           Per-batch HTTP retries are handled by the client's configured
           ``RetryConfig``. **Partial failures do not raise** — per-batch errors are
           captured on the returned :class:`UpsertResponse` (see
           ``response.has_errors``, ``response.errors``, ``response.failed_items``).
           To retry only the failures, pass ``response.failed_items`` back to
           ``upsert(...)``.

        .. seealso::
           - :meth:`upsert_records` — for indexes with integrated inference
             (text in, server-side embedding).
           - :meth:`start_import` — for bulk loading millions of vectors
             from cloud storage (S3, GCS).
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

        batch_result = await async_batch_execute(
            items=items,
            operation=_operation,
            batch_size=batch_size,
            max_concurrency=max_concurrency,
            show_progress=show_progress,
            desc="Upserting",
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
    ) -> UpsertResponse:
        """Not supported for async clients.

        This method is a known limitation of the async client. Instead, batch your data
        and call upsert() in a loop. For very large datasets, use start_import() for
        bulk loading from cloud storage.

        Raises:
            :exc:`NotImplementedError`: Always.
            :exc:`PineconeValueError`: If *batch_size* is not a positive integer.
        """
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise PineconeValueError("batch_size must be a positive integer")

        raise NotImplementedError(
            "upsert_from_dataframe is not supported for async clients. "
            "Instead, batch your data and call upsert() in a loop. "
            "For very large datasets, use start_import() for bulk loading from cloud storage."
        )

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
        """Query a namespace for the nearest neighbors of a vector.

        Args:
            top_k (int): Number of results to return (must be >= 1).
            vector (list[float] | None): Dense query vector values.
            id (str | None): ID of a stored vector to use as the query.
            namespace (str): Namespace to query. Defaults to the default namespace.
            filter (dict[str, Any] | None): Metadata filter expression.
            include_values (bool): Whether to include vector values in results.
            include_metadata (bool): Whether to include metadata in results.
            sparse_vector (SparseValues | dict[str, Any] | None): Sparse query vector
                with indices and values.
            scan_factor (float | None): DRN optimization — adjusts how much of the
                index is scanned. Range 0.5–4.0. Only supported for dedicated read
                node indexes. None uses server default.
            max_candidates (int | None): DRN optimization — caps candidate vectors to
                rerank. Range 1–100000. Only supported for dedicated read node indexes.
                None uses server default.

        Returns:
            :class:`QueryResponse` with matches, namespace, and usage info.

        Raises:
            :exc:`PineconeValueError`: If top_k < 1, both vector and id are provided,
                or none of vector, id, or sparse_vector are provided.
            :exc:`ApiError`: If the API returns an error response.
            :exc:`PineconeConnectionError`: If a network-level connection
                fails (DNS, refused, transport error).
            :exc:`PineconeTimeoutError`: If the request exceeds the configured timeout.

        Examples:

            .. code-block:: python

                response = await idx.query(
                    top_k=10,
                    vector=[0.012, -0.087, 0.153],  # truncated; use your actual dimension
                )
                for match in response.matches:
                    print(match.id, match.score)

            Query with a metadata filter:

            .. code-block:: python

                response = await idx.query(
                    top_k=10,
                    vector=[0.012, -0.087, 0.153],
                    filter={"genre": "comedy", "year": {"$gte": 2020}},
                    namespace="movies-en",
                )
        """
        if top_k < 1:
            raise ValidationError(f"top_k must be a positive integer, got {top_k}")

        has_vector = vector is not None
        has_id = id is not None
        has_sparse = sparse_vector is not None
        if has_vector and has_id:
            raise ValidationError("Exactly one of vector or id must be provided, not both")
        if not has_vector and not has_id and not has_sparse:
            raise ValidationError("At least one of vector, id, or sparse_vector must be provided")

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
        """Query multiple namespaces concurrently and return merged top results.

        Fans out individual ``query()`` calls across all given namespaces
        using ``asyncio.gather``, then merges results via a heap-based
        aggregator that returns the overall top-k matches ranked by the
        specified metric.

        Args:
            vector: Dense query vector values. Required for dense and hybrid
                indexes; omit for sparse-only indexes (use *sparse_vector* instead).
            namespaces: Namespaces to query (must be non-empty). Duplicates
                are removed while preserving order.
            metric: Distance metric — ``"cosine"``, ``"euclidean"``, or
                ``"dotproduct"``.
            top_k: Maximum number of results to return. Defaults to 10.
            filter: Metadata filter expression applied to every namespace.
            include_values: Whether to include vector values in results.
            include_metadata: Whether to include metadata in results.
            sparse_vector: Sparse query vector with indices and values.
                Required for sparse-only indexes when *vector* is omitted.
            scan_factor: DRN performance tuning — controls how much of the
                index is scanned during a query. Higher values scan more
                data and may improve recall at the cost of latency.
            max_candidates: DRN performance tuning — maximum number of
                candidate vectors to consider during the search phase.

        Returns:
            :class:`QueryNamespacesResults` with the merged top-k matches, total
            usage, and per-namespace usage.

        Raises:
            :exc:`PineconeValueError`: If *namespaces* is empty, or if both
                *vector* and *sparse_vector* are absent/empty.
            :exc:`ValueError`: If *metric* is not a recognized value.
            :exc:`ApiError`: If any individual namespace query fails.
            :exc:`PineconeConnectionError`: If a network-level connection
                fails (DNS, refused, transport error).
            :exc:`PineconeTimeoutError`: If the request exceeds the configured timeout.

        Examples:

            .. code-block:: python

                # Dense query
                results = await idx.query_namespaces(
                    vector=[0.012, -0.087, 0.153],  # truncated; use your actual dimension
                    namespaces=["articles-en", "articles-fr", "articles-de"],
                    metric="cosine",
                    top_k=10,
                )

                # Sparse-only query (sparse index)
                results = await idx.query_namespaces(
                    sparse_vector={"indices": [0, 1, 2], "values": [0.1, 0.2, 0.3]},
                    namespaces=["docs-en", "docs-fr"],
                    metric="dotproduct",
                    top_k=10,
                )

                for match in results.matches:
                    print(match.id, match.score)
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
            self._limiter_registry.get(self._host, _internal_concurrency_ceiling)
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
        """Fetch vectors by their IDs from a namespace.

        Args:
            ids (list[str]): List of vector IDs to fetch (must be non-empty).
            namespace (str): Namespace to fetch from. Defaults to the default namespace.

        Returns:
            :class:`FetchResponse` with a map of vector IDs to Vector objects, namespace,
            and usage info. IDs that do not exist are omitted from the map rather
            than raising an error.

        Raises:
            :exc:`PineconeValueError`: If ids is empty.
            :exc:`ApiError`: If the API returns an error response.
            :exc:`PineconeConnectionError`: If a network-level connection
                fails (DNS, refused, transport error).
            :exc:`PineconeTimeoutError`: If the request exceeds the configured timeout.

        Examples:

            .. code-block:: python

                response = await idx.fetch(ids=["article-101", "article-102"])
                for vid, vec in response.vectors.items():
                    print(vid, vec.values)
        """
        if not ids:
            raise ValidationError("ids must be a non-empty list")

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
        """Fetch vectors matching a metadata filter expression.

        Returns vectors whose metadata satisfies the given filter, with
        pagination support. The server returns up to 100 vectors per page
        when no limit is specified.

        Args:
            filter: Metadata filter expression (required).
            namespace: Namespace to fetch from. Defaults to the default
                namespace.
            limit: Maximum number of vectors to return per page. When
                ``None``, the server default (100) is used.
            pagination_token: Token from a previous response to fetch the
                next page. When ``None``, fetches the first page.

        Returns:
            :class:`FetchByMetadataResponse` with matched vectors, namespace, usage,
            and pagination token for the next page (if any).

        Raises:
            :exc:`ApiError`: If the API returns an error response (e.g. authentication
                failure or server error).

        Examples:

            .. code-block:: python

                response = await idx.fetch_by_metadata(
                    filter={"genre": {"$eq": "comedy"}},
                    namespace="movies",
                )
                for vid, vec in response.vectors.items():
                    print(vid, vec.values)

                # Paginate through all results
                token = response.pagination.next if response.pagination else None
                while token:
                    response = await idx.fetch_by_metadata(
                        filter={"genre": {"$eq": "comedy"}},
                        namespace="movies",
                        pagination_token=token,
                    )
                    token = response.pagination.next if response.pagination else None
        """
        if limit is not None:
            require_positive("limit", limit)
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
        """Delete vectors from a namespace by ID, filter, or delete-all flag.

        Exactly one of ``ids``, ``delete_all``, or ``filter`` must be specified.
        Deleting IDs that do not exist does not raise an error.

        Args:
            ids (list[str] | None): List of vector IDs to delete.
            delete_all (bool): If True, delete all vectors in the namespace.
            filter (dict[str, Any] | None): Metadata filter expression selecting vectors to delete.
            namespace (str): Namespace to delete from. Defaults to the default namespace.

        Returns:
            None — a successful delete returns no payload.

        Raises:
            :exc:`PineconeValueError`: If zero or more than one deletion mode is specified.
            :exc:`ApiError`: If the API returns an error response.
            :exc:`PineconeConnectionError`: If a network-level connection
                fails (DNS, refused, transport error).
            :exc:`PineconeTimeoutError`: If the request exceeds the configured timeout.

        Examples:

            .. code-block:: python

                # Delete by IDs
                await idx.delete(ids=["article-101", "article-102"])

                # Delete all vectors in a namespace
                await idx.delete(delete_all=True, namespace="articles-deprecated")

                # Delete by metadata filter
                await idx.delete(filter={"category": {"$eq": "obsolete"}})
        """
        mode_count = sum([ids is not None, delete_all, filter is not None])
        if mode_count == 0:
            raise ValidationError("Must specify one of ids, delete_all, or filter")
        if mode_count > 1:
            raise ValidationError(
                "Cannot combine ids, delete_all, and filter — specify exactly one"
            )

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
        """Update vectors by ID or metadata filter.

        Exactly one of ``id`` or ``filter`` must be specified.

        Args:
            id (str | None): ID of the vector to update.
            values (list[float] | None): New dense vector values.
            sparse_values (SparseValues | dict[str, Any] | None): New sparse vector.
            set_metadata (dict[str, Any] | None): Metadata fields to set or overwrite.
            namespace (str): Namespace to target. Defaults to the default namespace.
            filter (dict[str, Any] | None): Metadata filter expression selecting vectors to update.
            dry_run (bool): If True, return the count of records that would be
                affected without applying changes.

        Returns:
            :class:`UpdateResponse` with matched_records count (when available).

        Raises:
            :exc:`PineconeValueError`: If both or neither of id and filter are provided.
            :exc:`ApiError`: If the API returns an error response.
            :exc:`PineconeConnectionError`: If a network-level connection
                fails (DNS, refused, transport error).
            :exc:`PineconeTimeoutError`: If the request exceeds the configured timeout.

        Examples:

            .. code-block:: python

                # Update by ID
                # truncated values; use your actual dimension
                await idx.update(id="article-101", values=[0.012, -0.087, 0.153])

                # Bulk-update metadata by filter
                await idx.update(
                    filter={"genre": {"$eq": "drama"}},
                    set_metadata={"year": 2020},
                )
        """
        has_id = id is not None
        has_filter = filter is not None
        if has_id and has_filter:
            raise ValidationError("Exactly one of id or filter must be provided, not both")
        if not has_id and not has_filter:
            raise ValidationError("Exactly one of id or filter must be provided, got neither")

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
        """Search records by text, vector, or ID with optional reranking.

        Searches a namespace using integrated inference (text inputs embedded
        server-side), a raw vector, or an existing record ID as the query.

        Args:
            namespace (str): Namespace to search in (required).
            top_k (int): Number of results to return (must be >= 1).
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
                ``"all"``) and ``"terms"`` (list of strings). Only supported
                for sparse indexes using ``pinecone-sparse-english-v0``.
                ``None`` disables term matching.
            query (dict[str, Any] | None): Legacy query body containing
                ``top_k`` plus one of ``inputs``, ``vector``, or ``id``. Prefer
                passing these fields directly.

        Returns:
            :class:`SearchRecordsResponse` with hits and usage statistics.

        Raises:
            :exc:`PineconeValueError`: If ``namespace`` is not a string, ``top_k < 1``,
                or ``rerank`` is missing required keys.
            :exc:`ApiError`: If the API returns an error response.
            :exc:`PineconeConnectionError`: If a network-level connection
                fails (DNS, refused, transport error).
            :exc:`PineconeTimeoutError`: If the request exceeds the configured timeout.

        Examples:

            .. code-block:: python

                response = await idx.search(
                    namespace="articles-en",
                    top_k=10,
                    inputs={"text": "benefits of vector databases for search"},
                )
                for hit in response.result.hits:
                    print(hit.id, hit.score)
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
            f"/records/namespaces/{namespace}/search", timeout=timeout, json=body
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
        """Alias for :meth:`search`.

        Prefer calling :meth:`search` directly — this alias exists for backwards compatibility."""
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
        """Fetch a single page of vector IDs from a namespace.

        Args:
            prefix (str | None): Return only IDs starting with this prefix.
            limit (int | None): Maximum number of IDs to return in this page.
            pagination_token (str | None): Token from a previous response to fetch the next page.
            namespace (str): Namespace to list from. Defaults to the default namespace.

        Returns:
            :class:`ListResponse` with vector IDs, pagination info, namespace, and usage.

        Raises:
            :exc:`PineconeValueError`: If inputs are invalid.
            :exc:`ApiError`: If the API returns an error response.

        Examples:

            .. code-block:: python

                response = await idx.list_paginated(prefix="doc1#", limit=50)
                for item in response.vectors:
                    print(item.id)
        """
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
        """List vector IDs in a namespace, automatically following pagination.

        Yields one ``ListResponse`` per page. The generator automatically
        follows pagination tokens until all pages have been retrieved.

        Args:
            prefix (str | None): Return only IDs starting with this prefix.
            limit (int | None): Maximum number of IDs to return per page.
            namespace (str): Namespace to list from. Defaults to the default namespace.

        Yields:
            :class:`ListResponse` for each page of results.

        Examples:

            .. code-block:: python

                async for page in idx.list(prefix="doc1#"):
                    for item in page.vectors:
                        print(item.id)
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
        """Return statistics for this index.

        Returns aggregate statistics including total vector count,
        per-namespace vector counts, dimension, and index fullness.

        Args:
            filter (dict[str, Any] | None): Metadata filter expression. When
                provided, only vectors matching the filter are counted.

        Returns:
            :class:`DescribeIndexStatsResponse` with namespace summaries, dimension,
            total vector count, and fullness metrics.

        Raises:
            :exc:`ApiError`: If the API returns an error response.
            :exc:`PineconeConnectionError`: If a network-level connection
                fails (DNS, refused, transport error).
            :exc:`PineconeTimeoutError`: If the request exceeds the configured timeout.

        Examples:

            .. code-block:: python

                stats = await idx.describe_index_stats()
                print(stats.total_vector_count, stats.dimension)

                # With filter — only count vectors matching the expression
                stats = await idx.describe_index_stats(
                    filter={"genre": {"$eq": "drama"}}
                )
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

        Args:
            name (str): Name for the new namespace (must be non-empty).
            schema (dict[str, Any] | None): Optional schema configuration
                with metadata field indexing settings.

        Returns:
            :class:`NamespaceDescription` with the namespace name and record count.

        Raises:
            :exc:`PineconeValueError`: If the name is not a string or is empty/whitespace.
            :exc:`ApiError`: If the API returns an error response (e.g. 409 conflict
                when namespace already exists).
            :exc:`PineconeConnectionError`: If a network-level connection
                fails (DNS, refused, transport error).
            :exc:`PineconeTimeoutError`: If the request exceeds the configured timeout.

        Examples:

            .. code-block:: python

                ns = await idx.create_namespace(name="my-ns")
                print(ns.name, ns.record_count)
        """
        if not isinstance(name, str):
            raise ValidationError("namespace name must be a string")
        if not name or not name.strip():
            raise ValidationError("namespace name must be a non-empty string")

        body: dict[str, Any] = {"name": name}
        if schema is not None:
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

        Args:
            name (str): Name of the namespace to describe.

        Returns:
            :class:`NamespaceDescription` with the namespace name, record count,
            and schema information.

        Raises:
            :exc:`PineconeValueError`: If the name is not a string or is empty/whitespace.
            :exc:`ApiError`: If the API returns an error response.
            :exc:`PineconeConnectionError`: If a network-level connection
                fails (DNS, refused, transport error).
            :exc:`PineconeTimeoutError`: If the request exceeds the configured timeout.

        Examples:

            .. code-block:: python

                ns = await idx.describe_namespace(name="my-ns")
                print(ns.name, ns.record_count)
        """
        legacy_namespace: str | None = kwargs.pop("namespace", None)
        if kwargs:
            raise TypeError(
                f"describe_namespace() got unexpected keyword arguments: {sorted(kwargs)!r}"
            )
        if name is not None and legacy_namespace is not None:
            raise ValidationError("Provide either name= or namespace=, not both")
        effective: str = name if name is not None else (legacy_namespace or "")
        if not isinstance(effective, str):
            raise ValidationError("namespace name must be a string")
        if not effective or not effective.strip():
            raise ValidationError("namespace name must be a non-empty string")

        logger.info("Describing namespace %r", effective)
        response = await self._http.get(f"/namespaces/{effective}")
        return self._adapter.to_namespace_description(response.content)

    async def delete_namespace(
        self,
        *,
        name: str | None = None,
        timeout: float | None = None,
        **kwargs: str,
    ) -> None:
        """Delete a namespace by name, removing all its vectors.

        Args:
            name (str): Name of the namespace to delete.

        Returns:
            None — a successful delete returns no payload.

        Raises:
            :exc:`PineconeValueError`: If the name is not a string or is empty/whitespace.
            :exc:`ApiError`: If the API returns an error response.
            :exc:`PineconeConnectionError`: If a network-level connection
                fails (DNS, refused, transport error).
            :exc:`PineconeTimeoutError`: If the request exceeds the configured timeout.

        Examples:

            .. code-block:: python

                await idx.delete_namespace(name="old-data")
        """
        legacy_namespace: str | None = kwargs.pop("namespace", None)
        if kwargs:
            raise TypeError(
                f"delete_namespace() got unexpected keyword arguments: {sorted(kwargs)!r}"
            )
        if name is not None and legacy_namespace is not None:
            raise ValidationError("Provide either name= or namespace=, not both")
        effective: str = name if name is not None else (legacy_namespace or "")
        if not isinstance(effective, str):
            raise ValidationError("namespace name must be a string")
        if not effective or not effective.strip():
            raise ValidationError("namespace name must be a non-empty string")

        logger.info("Deleting namespace %r", effective)
        await self._http.delete(f"/namespaces/{effective}", timeout=timeout)

    async def list_namespaces_paginated(
        self,
        *,
        prefix: str | None = None,
        limit: int | None = None,
        pagination_token: str | None = None,
    ) -> ListNamespacesResponse:
        """Fetch a single page of namespace descriptions.

        Args:
            prefix (str | None): Return only namespaces whose names start with this prefix.
            limit (int | None): Maximum number of namespaces to return in this page.
            pagination_token (str | None): Token from a previous response to fetch the next page.

        Returns:
            :class:`ListNamespacesResponse` with namespace descriptions, pagination info,
            and total count.

        Raises:
            :exc:`ApiError`: If the API returns an error response.

        Examples:

            .. code-block:: python

                response = await idx.list_namespaces_paginated(prefix="prod-", limit=10)
                for ns in response.namespaces:
                    print(ns.name, ns.record_count)
        """
        params: dict[str, Any] = {}
        if prefix is not None:
            params["prefix"] = prefix
        if limit is not None:
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
        """List namespaces, automatically following pagination.

        Yields one ``ListNamespacesResponse`` per page. The generator
        automatically follows pagination tokens until all pages have been
        retrieved.

        Args:
            prefix (str | None): Return only namespaces whose names start with this prefix.
            limit (int | None): Maximum number of namespaces to return per page.

        Yields:
            :class:`ListNamespacesResponse` for each page of results.

        Examples:

            .. code-block:: python

                async for page in idx.list_namespaces(prefix="prod-"):
                    for ns in page.namespaces:
                        print(ns.name, ns.record_count)
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
            :exc:`PineconeValueError`: If the ID is empty or exceeds 1000 characters.
        """
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
        """Start a bulk import operation from an external data source.

        Initiates an asynchronous bulk import of vectors from cloud storage
        into the index. The import runs server-side; use :meth:`describe_import`
        to poll for progress and completion.

        .. note::
           The import URI must point to a directory of Parquet files in cloud
           storage (``s3://`` or ``gs://``). Each Parquet file must follow the
           Pinecone-required schema. See
           `Pinecone import docs <https://docs.pinecone.io/guides/data/understanding-imports>`_
           for the required Parquet schema and supported storage formats.

        Args:
            uri (str): Source URI for the import data (e.g.
                ``"s3://my-bucket/vectors/"`` or ``"gs://my-bucket/vectors/"``).
            error_mode (str | None): How to handle errors during import. Must be
                ``"continue"`` or ``"abort"`` when supplied. Case-insensitive.
                Optional; when omitted the backend default (abort) applies.
            integration_id (str | None): Optional integration ID for the import.

        Returns:
            :class:`StartImportResponse` with the ID of the created import
            operation.

        Raises:
            :exc:`PineconeValueError`: If ``error_mode`` is supplied but not
                ``"continue"`` or ``"abort"``.
            :exc:`ApiError`: If the API returns an error response.
            :exc:`PineconeConnectionError`: If a network-level connection
                fails (DNS, refused, transport error).
            :exc:`PineconeTimeoutError`: If the request exceeds the configured timeout.

        Examples:

            .. code-block:: python

                import asyncio

                # Start an import and poll until complete
                response = await idx.start_import(uri="s3://my-bucket/vectors/")
                import_id = response.id

                import_op = await idx.describe_import(import_id)
                while import_op.status not in ("Completed", "Failed", "Cancelled"):
                    await asyncio.sleep(10)
                    import_op = await idx.describe_import(import_id)
                print(f"Status: {import_op.status}, records imported: {import_op.records_imported}")

            .. code-block:: python

                # Abort on first error instead of continuing
                response = await idx.start_import(
                    uri="s3://my-bucket/vectors/",
                    error_mode="abort",
                )

        .. seealso::
           - :meth:`upsert` — for upserting vectors directly in small
             batches (single request per call).
           - :meth:`upsert_records` — for indexes with integrated inference
             (text in, server-side embedding).
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
            id: Import operation ID. Integers are converted to strings silently.

        Returns:
            :class:`ImportModel` with the import operation details.

        Raises:
            :exc:`PineconeValueError`: If the ID is empty or exceeds 1000 characters.
            :exc:`ApiError`: If the API returns an error response.
            :exc:`PineconeConnectionError`: If a network-level connection
                fails (DNS, refused, transport error).
            :exc:`PineconeTimeoutError`: If the request exceeds the configured timeout.

        Examples:

            .. code-block:: python

                import_op = await idx.describe_import("import-123")
                print(import_op.status, import_op.percent_complete)
        """
        str_id = self._validate_import_id(id)
        logger.info("Describing import %s", str_id)
        response = await self._http.get(f"/bulk/imports/{str_id}")
        return self._imports_adapter.to_import_model(response.content)

    async def cancel_import(self, id: str | int) -> None:
        """Cancel a bulk import operation by ID.

        Args:
            id: Import operation ID. Integers are converted to strings silently.

        Returns:
            None — a successful cancellation returns no payload.

        Raises:
            :exc:`PineconeValueError`: If the ID is empty or exceeds 1000 characters.
            :exc:`ApiError`: If the API returns an error response.
            :exc:`PineconeConnectionError`: If a network-level connection
                fails (DNS, refused, transport error).
            :exc:`PineconeTimeoutError`: If the request exceeds the configured timeout.

        Examples:

            .. code-block:: python

                await idx.cancel_import("import-123")
        """
        str_id = self._validate_import_id(id)
        logger.info("Cancelling import %s", str_id)
        await self._http.delete(f"/bulk/imports/{str_id}")

    async def list_imports(
        self,
        *,
        limit: int | None = None,
        pagination_token: str | None = None,
    ) -> AsyncIterator[ImportModel]:
        """List bulk import operations, automatically following pagination.

        Yields individual :class:`ImportModel` objects, fetching additional
        pages transparently until all results have been returned.

        Args:
            limit (int | None): Maximum number of imports per page
                (max 100, server default 100).
            pagination_token (str | None): Token to resume pagination
                from a previous call.

        Yields:
            :class:`ImportModel` for each import operation.

        Raises:
            :exc:`ApiError`: If the API returns an error response.

        Examples:

            .. code-block:: python

                async for imp in idx.list_imports():
                    print(imp.id, imp.status)
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
        """Fetch a single page of bulk import operations.

        Returns an :class:`ImportList` for one page. The caller is responsible
        for managing the pagination token.

        Args:
            limit (int | None): Maximum number of imports to return in this page.
            pagination_token (str | None): Token from a previous response to
                fetch the next page.

        Returns:
            :class:`ImportList` with the import operations for the requested page.

        Raises:
            :exc:`ApiError`: If the API returns an error response.

        Examples:

            .. code-block:: python

                page = await idx.list_imports_paginated(limit=10)
                for imp in page:
                    print(imp.id, imp.status)
        """
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if pagination_token is not None:
            params["paginationToken"] = pagination_token

        response = await self._http.get("/bulk/imports", params=params)
        return self._imports_adapter.to_import_list(response.content)

    async def close(self) -> None:
        """Close the underlying HTTP client and release resources."""
        await self._http.close()

    async def __aenter__(self) -> AsyncIndex:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    def __repr__(self) -> str:
        """Return developer-friendly representation."""
        return f"AsyncIndex(host='{self._host}')"
