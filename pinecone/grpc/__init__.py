"""Synchronous gRPC data plane client for a Pinecone index."""

from __future__ import annotations

import builtins
import logging
import os
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd  # type: ignore[import-untyped]

from pinecone._internal.adapters.vectors_adapter import VectorsAdapter, extract_response_info
from pinecone._internal.batch import batch_execute
from pinecone._internal.batching import chunked, validate_batch_size, with_progress
from pinecone._internal.config import PineconeConfig
from pinecone._internal.constants import DATA_PLANE_API_VERSION
from pinecone._internal.data_plane_helpers import _validate_host
from pinecone._internal.validation import require_in_range
from pinecone._internal.vector_factory import VectorFactory
from pinecone.errors.exceptions import (
    PineconeValueError,
    ValidationError,
)
from pinecone.grpc._protocol import GrpcChannelProtocol
from pinecone.grpc.future import PineconeFuture
from pinecone.models.namespaces.models import (
    IndexedFields,
    ListNamespacesResponse,
    NamespaceDescription,
    NamespaceFieldConfig,
    NamespaceSchema,
)
from pinecone.models.vectors.responses import (
    DescribeIndexStatsResponse,
    FetchResponse,
    ListItem,
    ListResponse,
    NamespaceSummary,
    Pagination,
    QueryResponse,
    UpdateResponse,
    UpsertRecordsResponse,
    UpsertResponse,
)
from pinecone.models.vectors.search import RerankConfig, SearchInputs, SearchRecordsResponse
from pinecone.models.vectors.sparse import SparseValues
from pinecone.models.vectors.usage import Usage
from pinecone.models.vectors.vector import ScoredVector, Vector

logger = logging.getLogger(__name__)


def _build_grpc_endpoint(host: str, secure: bool) -> str:
    """Build a gRPC endpoint URL from a host string.

    Strips any existing scheme and applies the correct one for gRPC.
    """
    bare = host
    for prefix in ("https://", "http://"):
        if bare.startswith(prefix):
            bare = bare[len(prefix) :]
            break

    scheme = "https" if secure else "http"
    return f"{scheme}://{bare}"


def _vector_to_grpc_dict(v: Vector) -> dict[str, Any]:
    """Serialize a Vector to a dict matching GrpcChannel's expected input format."""
    d: dict[str, Any] = {"id": v.id, "values": v.values}
    if v.sparse_values is not None:
        d["sparse_values"] = {
            "indices": v.sparse_values.indices,
            "values": v.sparse_values.values,
        }
    if v.metadata is not None:
        d["metadata"] = v.metadata
    return d


def _dict_to_vector(vid: str, data: dict[str, Any]) -> Vector:
    """Convert a GrpcChannel vector dict to a Vector model."""
    sparse = None
    sv = data.get("sparse_values")
    if sv is not None:
        sparse = SparseValues(sv["indices"], sv["values"])
    return Vector(
        id=vid,
        values=data.get("values", []),
        sparse_values=sparse,
        metadata=data.get("metadata"),
    )


def _dict_to_scored_vector(data: dict[str, Any]) -> ScoredVector:
    """Convert a GrpcChannel scored vector dict to a ScoredVector model."""
    sparse = None
    sv = data.get("sparse_values")
    if sv is not None:
        sparse = SparseValues(sv["indices"], sv["values"])
    return ScoredVector(
        id=data["id"],
        score=data.get("score", 0.0),
        values=data.get("values", []),
        sparse_values=sparse,
        metadata=data.get("metadata"),
    )


def _dict_to_usage(data: dict[str, Any] | None) -> Usage | None:
    """Convert a usage dict to a Usage model, or None."""
    if data is None:
        return None
    return Usage(read_units=data.get("read_units", 0))


def _dict_to_namespace_description(data: dict[str, Any]) -> NamespaceDescription:
    """Convert a GrpcChannel namespace dict to a NamespaceDescription model.

    Shared by create_namespace, describe_namespace, and list_namespaces_paginated
    to convert the dict payload returned by the Rust-backed GrpcChannel into a
    typed NamespaceDescription, including optional schema and indexed_fields.
    """
    schema: NamespaceSchema | None = None
    raw_schema = data.get("schema")
    if raw_schema is not None:
        schema = NamespaceSchema(
            fields={
                k: NamespaceFieldConfig(filterable=v["filterable"])
                for k, v in raw_schema.get("fields", {}).items()
            }
        )

    indexed_fields: IndexedFields | None = None
    raw_indexed = data.get("indexed_fields")
    if raw_indexed is not None:
        indexed_fields = IndexedFields(fields=list(raw_indexed))

    return NamespaceDescription(
        name=data.get("name", ""),
        record_count=data.get("record_count", 0),
        schema=schema,
        indexed_fields=indexed_fields,
    )


class GrpcIndex:
    """Synchronous gRPC data plane client targeting a specific Pinecone index.

    Provides the same interface as :class:`~pinecone.index.Index` but routes
    data-plane operations through a gRPC transport (via the Rust-backed
    :class:`~pinecone._grpc.GrpcChannel`) instead of HTTP/REST.

    Args:
        host (str): The index-specific data plane host URL.
        api_key (str | None): Pinecone API key. Falls back to ``PINECONE_API_KEY`` env var.
        api_version (str): API version string. Defaults to the current data plane version.
        source_tag (str | None): Tag appended to the User-Agent string for request attribution.
        secure (bool): Whether to use TLS encryption. Defaults to ``True``.
        timeout (float): Request timeout in seconds. Defaults to ``20.0``.
        connect_timeout (float): Connection timeout in seconds. Defaults to ``1.0``.

    Raises:
        :exc:`ValidationError`: If no API key can be resolved or the host is invalid.

    Examples:

        .. code-block:: python

            from pinecone.grpc import GrpcIndex

            idx = GrpcIndex(host="movie-recs-abc123.svc.pinecone.io", api_key="...")
    """

    def __init__(
        self,
        *,
        host: str,
        api_key: str | None = None,
        api_version: str = DATA_PLANE_API_VERSION,
        source_tag: str | None = None,
        secure: bool = True,
        timeout: float = 20.0,
        connect_timeout: float = 1.0,
        on_throttle: Callable[[str], None] | None = None,
    ) -> None:
        # Resolve API key: explicit arg > env var
        resolved_key = api_key or os.environ.get("PINECONE_API_KEY", "")
        if not resolved_key:
            raise ValidationError(
                "No API key provided. Pass api_key='...' or set the "
                "PINECONE_API_KEY environment variable."
            )

        # Validate and normalize host
        self._host = _validate_host(host)
        self._source_tag = source_tag

        # Build gRPC endpoint and create the Rust-backed channel
        endpoint = _build_grpc_endpoint(self._host, secure)

        from pinecone import __version__
        from pinecone._grpc import GrpcChannel  # type: ignore[import-not-found]

        self._channel: GrpcChannelProtocol = GrpcChannel(
            endpoint,
            resolved_key,
            api_version,
            __version__,
            secure,
            timeout,
            connect_timeout,
            source_tag=source_tag,
            on_throttle=on_throttle,
        )

        self._executor = ThreadPoolExecutor()
        self._batch_executor: ThreadPoolExecutor | None = None
        self._batch_executor_workers: int = 0

        # REST HTTP client for records operations (integrated inference).
        # upsert_records and search use REST endpoints with no gRPC equivalent.
        from pinecone._internal.http_client import HTTPClient

        rest_config = PineconeConfig(
            api_key=resolved_key,
            host=self._host,
            timeout=timeout,
            source_tag=source_tag or "",
            ssl_verify=secure,
        )
        self._http = HTTPClient(rest_config, DATA_PLANE_API_VERSION)
        self._adapter = VectorsAdapter()

        logger.info("GrpcIndex client created for host %s", self._host)

    @property
    def host(self) -> str:
        """The data plane host URL for this index."""
        return self._host

    def _get_batch_executor(self, max_concurrency: int) -> ThreadPoolExecutor:
        if self._batch_executor is None or self._batch_executor_workers != max_concurrency:
            if self._batch_executor is not None:
                self._batch_executor.shutdown(wait=False)
            self._batch_executor = ThreadPoolExecutor(
                max_concurrency,
                thread_name_prefix="pinecone-grpc-batch-upsert",
            )
            self._batch_executor_workers = max_concurrency
        return self._batch_executor

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
        max_concurrency: int = 4,
        show_progress: bool = True,
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
            batch_size (int | None): If set, splits ``vectors`` into batches of
                this size and submits them in **parallel** via a
                ``ThreadPoolExecutor``. ``None`` (default) sends all vectors in
                a single channel call. Must be a positive integer when set.
            max_concurrency (int): Number of parallel threads used when
                ``batch_size`` is set. Default ``4``, range ``[1, 64]``. Ignored
                when ``batch_size`` is ``None``.
            show_progress (bool): If ``True`` and ``tqdm`` is installed, display a
                progress bar while submitting batches. Ignored when ``batch_size``
                is ``None``. Defaults to ``True``.
            timeout (float | None): Per-call timeout in seconds. Applied per batch
                when batching. None uses the client-level default.

        Returns:
            :class:`UpsertResponse` with the count of vectors upserted.

        Raises:
            :exc:`TypeError`: If a vector element is not a recognized format.
            :exc:`ValueError`: If a vector element is malformed.
            :exc:`PineconeValueError`: If ``batch_size`` is not a positive integer
                or ``max_concurrency`` is outside ``[1, 64]``.
            :exc:`PineconeTimeoutError`: If the call exceeds *timeout* or the server
                returns CANCELLED with a timeout cause.

        Notes:
            When ``batch_size`` is set, batches are submitted **in parallel** via a
            ``ThreadPoolExecutor`` of ``max_concurrency`` workers (default 4, range
            1–64). Per-batch retries are handled by the gRPC channel's own retry
            policy. **Partial failures do not raise** — the returned
            :class:`UpsertResponse` carries ``upserted_count``,
            ``failed_item_count``, ``errors``, and ``failed_items`` for inspection /
            retry. Pass ``response.failed_items`` back to ``upsert(...)`` to retry
            only the failures.

        Examples:

            .. code-block:: python

                from pinecone.grpc import GrpcIndex
                from pinecone.models.vectors.vector import Vector

                idx = GrpcIndex(host="article-search-abc123.svc.pinecone.io", api_key="...")
                response = idx.upsert(
                    vectors=[
                        Vector(
                            id="article-101",
                            values=[0.012, -0.087, 0.153, ...],  # 1536-dim
                        ),
                        ("article-102", [0.045, 0.021, -0.064, ...]),
                        {"id": "article-103", "values": [0.091, -0.032, 0.178, ...]},
                    ],
                    namespace="articles-en",
                )
                print(response.upserted_count)
        """
        if batch_size is None:
            built = [VectorFactory.build(v) for v in vectors]
            grpc_vectors = [_vector_to_grpc_dict(v) for v in built]
            logger.info("Upserting %d vectors via gRPC into namespace %r", len(built), namespace)
            result = self._channel.upsert(grpc_vectors, namespace or None, timeout_s=timeout)
            return UpsertResponse(upserted_count=result.get("upserted_count", 0))

        validate_batch_size(batch_size)
        require_in_range("max_concurrency", max_concurrency, 1, 64)

        built = [VectorFactory.build(v) for v in vectors]
        items: builtins.list[dict[str, Any]] = [_vector_to_grpc_dict(v) for v in built]

        def _operation(chunk: builtins.list[dict[str, Any]]) -> dict[str, Any]:
            return self._channel.upsert(chunk, namespace or None, timeout_s=timeout)

        batch_result = batch_execute(
            items=items,
            operation=_operation,
            batch_size=batch_size,
            max_concurrency=max_concurrency,
            show_progress=show_progress,
            desc="Upserting",
            executor=self._get_batch_executor(max_concurrency),
        )

        return UpsertResponse(
            upserted_count=batch_result.successful_item_count,
            total_item_count=batch_result.total_item_count,
            failed_item_count=batch_result.failed_item_count,
            total_batch_count=batch_result.total_batch_count,
            successful_batch_count=batch_result.successful_batch_count,
            failed_batch_count=batch_result.failed_batch_count,
            errors=batch_result.errors,
        )

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
            timeout (float | None): Per-call timeout in seconds. None uses the client-level default.

        Returns:
            :class:`QueryResponse` with matches, namespace, and usage info.

        Raises:
            :exc:`ValidationError`: If top_k is not between 1 and 10000, both vector
                and id are provided, or none of vector, id, or sparse_vector are provided.
            :exc:`PineconeTimeoutError`: If the call exceeds *timeout* or the server
                returns CANCELLED with a timeout cause.

        Examples:

            .. code-block:: python

                response = idx.query(
                    top_k=10,
                    vector=[0.012, -0.087, 0.153, ...],  # 1536-dim embedding
                )
                for match in response.matches:
                    print(match.id, match.score)
        """
        require_in_range("top_k", top_k, 1, 10_000)

        has_vector = vector is not None
        has_id = id is not None
        has_sparse = sparse_vector is not None
        if has_vector and has_id:
            raise ValidationError("Exactly one of vector or id must be provided, not both")
        if not has_vector and not has_id and not has_sparse:
            raise ValidationError("At least one of vector, id, or sparse_vector must be provided")

        # Convert SparseValues model to dict for GrpcChannel
        sv_dict: Mapping[str, Any] | None = None
        if sparse_vector is not None:
            if isinstance(sparse_vector, SparseValues):
                sv_dict = {
                    "indices": sparse_vector.indices,
                    "values": sparse_vector.values,
                }
            else:
                sv_dict = sparse_vector

        logger.info("Querying index via gRPC with top_k=%d", top_k)
        result = self._channel.query(
            top_k,
            vector=vector,
            id=id,
            namespace=namespace or None,
            filter=filter,
            include_values=include_values,
            include_metadata=include_metadata,
            sparse_vector=sv_dict,
            scan_factor=scan_factor,
            max_candidates=max_candidates,
            timeout_s=timeout,
        )

        matches = [_dict_to_scored_vector(m) for m in result.get("matches", [])]
        usage = _dict_to_usage(result.get("usage"))
        return QueryResponse(
            matches=matches,
            namespace=result.get("namespace", ""),
            usage=usage,
        )

    def fetch(
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
            timeout (float | None): Per-call timeout in seconds. None uses the client-level default.

        Returns:
            :class:`FetchResponse` with a map of vector IDs to Vector objects, namespace,
            and usage info.

        Raises:
            :exc:`ValidationError`: If ids is empty.
            :exc:`PineconeTimeoutError`: If the call exceeds *timeout* or the server
                returns CANCELLED with a timeout cause.

        Examples:

            .. code-block:: python

                response = idx.fetch(ids=["article-101", "article-102"])
                for vid, vec in response.vectors.items():
                    print(vid, vec.values)
        """
        if not ids:
            raise ValidationError("ids must be a non-empty list")

        logger.info("Fetching %d vectors via gRPC", len(ids))
        result = self._channel.fetch(ids, namespace=namespace or None, timeout_s=timeout)

        vectors: dict[str, Vector] = {}
        for vid, vdata in result.get("vectors", {}).items():
            vectors[vid] = _dict_to_vector(vid, vdata)

        usage = _dict_to_usage(result.get("usage"))
        return FetchResponse(
            vectors=vectors,
            namespace=result.get("namespace", ""),
            usage=usage,
        )

    def delete(
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

        Args:
            ids (list[str] | None): List of vector IDs to delete.
            delete_all (bool): If True, delete all vectors in the namespace.
            filter (dict[str, Any] | None): Metadata filter expression selecting vectors to delete.
            namespace (str): Namespace to delete from. Defaults to the default namespace.
            timeout (float | None): Per-call timeout in seconds. None uses the client-level default.

        Returns:
            None

        Raises:
            :exc:`ValidationError`: If zero or more than one deletion mode is specified.
            :exc:`PineconeTimeoutError`: If the call exceeds *timeout* or the server
                returns CANCELLED with a timeout cause.

        Examples:

            .. code-block:: python

                # Delete by IDs
                idx.delete(ids=["article-101", "article-102"])

                # Delete all vectors in a namespace
                idx.delete(delete_all=True, namespace="articles-deprecated")

                # Delete by metadata filter
                idx.delete(filter={"category": {"$eq": "obsolete"}})
        """
        mode_count = sum([ids is not None, delete_all, filter is not None])
        if mode_count == 0:
            raise ValidationError("Must specify one of ids, delete_all, or filter")
        if mode_count > 1:
            raise ValidationError(
                "Cannot combine ids, delete_all, and filter — specify exactly one"
            )

        logger.info("Deleting vectors via gRPC from namespace %r", namespace)
        self._channel.delete(
            ids=ids,
            delete_all=delete_all,
            namespace=namespace or None,
            filter=filter,
            timeout_s=timeout,
        )

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
        """Update vectors by ID or metadata filter.

        Args:
            id (str | None): ID of the vector to update.
            values (list[float] | None): New dense vector values.
            sparse_values (SparseValues | dict[str, Any] | None): New sparse vector.
            set_metadata (dict[str, Any] | None): Metadata fields to set or overwrite.
            namespace (str): Namespace to target. Defaults to the default namespace.
            filter (dict[str, Any] | None): Metadata filter expression selecting vectors to update.
            dry_run (bool): If True, return the count of records that would be
                affected without applying changes.
            timeout (float | None): Per-call timeout in seconds. None uses the client-level default.

        Returns:
            :class:`UpdateResponse` with matched_records count (when available).

        Raises:
            :exc:`ValidationError`: If both or neither of id and filter are provided.
            :exc:`PineconeTimeoutError`: If the call exceeds *timeout* or the server
                returns CANCELLED with a timeout cause.

        Examples:

            .. code-block:: python

                # Update by ID
                idx.update(id="article-101", values=[0.012, -0.087, 0.153, ...])

                # Bulk-update metadata by filter
                idx.update(
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

        # Convert SparseValues model to dict for GrpcChannel
        sv_dict: Mapping[str, Any] | None = None
        if sparse_values is not None:
            if isinstance(sparse_values, SparseValues):
                sv_dict = {
                    "indices": sparse_values.indices,
                    "values": sparse_values.values,
                }
            else:
                sv_dict = sparse_values

        logger.info("Updating vectors via gRPC in namespace %r", namespace)
        # The Rust channel's update() requires `id` as a positional string arg.
        # For filter-based updates id is None, so pass "" which the API ignores
        # when a filter is provided.
        result = self._channel.update(
            id if id is not None else "",
            values=values,
            sparse_values=sv_dict,
            set_metadata=set_metadata,
            namespace=namespace or None,
            filter=filter,
            dry_run=dry_run or None,
            timeout_s=timeout,
        )

        return UpdateResponse(matched_records=result.get("matched_records"))

    def list_paginated(
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
            timeout (float | None): Per-call timeout in seconds. None uses the client-level default.

        Returns:
            :class:`ListResponse` with vector IDs, pagination info, namespace, and usage.

        Raises:
            :exc:`PineconeTimeoutError`: If the call exceeds *timeout* or the server
                returns CANCELLED with a timeout cause.

        Examples:

            .. code-block:: python

                response = idx.list_paginated(prefix="doc1#", limit=50)
                for item in response.vectors:
                    print(item.id)
        """
        logger.info("Listing vectors via gRPC in namespace %r", namespace)
        result = self._channel.list(
            prefix=prefix,
            limit=limit,
            pagination_token=pagination_token,
            namespace=namespace or None,
            timeout_s=timeout,
        )

        vectors = [ListItem(id=v.get("id")) for v in result.get("vectors", [])]
        pagination_data = result.get("pagination")
        pagination = None
        if pagination_data is not None:
            pagination = Pagination(next=pagination_data.get("next"))
        usage = _dict_to_usage(result.get("usage"))

        return ListResponse(
            vectors=vectors,
            pagination=pagination,
            namespace=result.get("namespace", ""),
            usage=usage,
        )

    def list(
        self,
        *,
        prefix: str | None = None,
        limit: int | None = None,
        namespace: str = "",
        timeout: float | None = None,
    ) -> Iterator[ListResponse]:
        """List vector IDs in a namespace, automatically following pagination.

        Yields one ``ListResponse`` per page.

        Args:
            prefix (str | None): Return only IDs starting with this prefix.
            limit (int | None): Maximum number of IDs to return per page.
            namespace (str): Namespace to list from. Defaults to the default namespace.
            timeout (float | None): Per-call timeout in seconds applied to each page
                request. None uses the client-level default.

        Yields:
            :class:`ListResponse` for each page of results.

        Raises:
            :exc:`PineconeTimeoutError`: If any page call exceeds *timeout* or the
                server returns CANCELLED with a timeout cause.

        Examples:

            .. code-block:: python

                for page in idx.list(prefix="doc1#"):
                    for item in page.vectors:
                        print(item.id)
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

    def describe_index_stats(
        self,
        *,
        filter: Mapping[str, Any] | None = None,
        timeout: float | None = None,
    ) -> DescribeIndexStatsResponse:
        """Return statistics for this index.

        Args:
            filter (dict[str, Any] | None): Metadata filter expression. When
                provided, only vectors matching the filter are counted.

        Returns:
            :class:`DescribeIndexStatsResponse` with namespace summaries, dimension,
            total vector count, and fullness metrics.

        Examples:

            .. code-block:: python

                stats = idx.describe_index_stats()
                print(stats.total_vector_count, stats.dimension)

                # With filter — only count vectors matching the expression
                stats = idx.describe_index_stats(
                    filter={"genre": {"$eq": "drama"}}
                )
        """
        logger.info("Describing index stats via gRPC")
        result = self._channel.describe_index_stats(filter=filter, timeout_s=timeout)

        namespaces: dict[str, NamespaceSummary] = {}
        for ns_name, ns_data in result.get("namespaces", {}).items():
            namespaces[ns_name] = NamespaceSummary(
                vector_count=ns_data.get("vector_count", 0),
            )

        return DescribeIndexStatsResponse(
            namespaces=namespaces,
            dimension=result.get("dimension"),
            index_fullness=result.get("index_fullness", 0.0),
            total_vector_count=result.get("total_vector_count", 0),
            metric=result.get("metric"),
            vector_type=result.get("vector_type"),
            memory_fullness=result.get("memory_fullness"),
            storage_fullness=result.get("storage_fullness"),
        )

    def upsert_from_dataframe(
        self,
        df: pd.DataFrame,
        namespace: str = "",
        batch_size: int = 500,
        show_progress: bool = True,
    ) -> UpsertResponse:
        """Upsert vectors from a pandas DataFrame using async batching.

        Splits the DataFrame into batches of ``batch_size`` rows and submits
        each batch asynchronously via :meth:`upsert_async`, then aggregates
        the results.

        Args:
            df: A ``pandas.DataFrame`` with at least ``id`` and ``values``
                columns. ``sparse_values`` and ``metadata`` columns are
                included when present and non-None.
            namespace: Target namespace. Defaults to the default namespace.
            batch_size: Number of rows per upsert batch. Defaults to 500.
            show_progress: If ``True`` and ``tqdm`` is installed, display a
                progress bar. If ``tqdm`` is not installed, silently falls
                back to no progress bar.

        Returns:
            :class:`UpsertResponse` with the total count of vectors upserted across
            all batches.

        Raises:
            :exc:`RuntimeError`: If ``pandas`` is not installed.
            :exc:`PineconeValueError`: If *df* is not a ``pandas.DataFrame``.
            :exc:`PineconeValueError`: If *batch_size* is not a positive integer.

        Examples:

            .. code-block:: python

                import pandas as pd
                from pinecone.grpc import GrpcIndex

                idx = GrpcIndex(
                    host="article-search-abc123.svc.pinecone.io",
                    api_key="your-api-key",
                )
                df = pd.DataFrame([
                    {"id": "article-101", "values": [0.012, -0.087, 0.153]},
                    {"id": "article-102", "values": [0.045, 0.021, -0.064]},
                ])
                response = idx.upsert_from_dataframe(df)
                response.upserted_count

            .. code-block:: python

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
        """
        try:
            import pandas as pd
        except ImportError:
            raise RuntimeError(
                "pandas is required for upsert_from_dataframe. Install it with: pip install pandas"
            ) from None

        if not isinstance(df, pd.DataFrame):
            raise PineconeValueError("df must be a pandas DataFrame")

        validate_batch_size(batch_size)

        has_sparse = "sparse_values" in df.columns
        has_metadata = "metadata" in df.columns

        records: builtins.list[dict[str, Any]] = []
        for _, row in df.iterrows():
            record: dict[str, Any] = {"id": row["id"], "values": row["values"]}
            if has_sparse and row["sparse_values"] is not None:
                record["sparse_values"] = row["sparse_values"]
            if has_metadata and row["metadata"] is not None:
                record["metadata"] = row["metadata"]
            records.append(record)

        batches = chunked(records, batch_size)
        futures: builtins.list[PineconeFuture[UpsertResponse]] = [
            self.upsert_async(vectors=batch, namespace=namespace)
            for batch in with_progress(batches, show_progress=show_progress)
        ]

        total_count = 0
        for future in futures:
            result = future.result()
            total_count += result.upserted_count

        return UpsertResponse(upserted_count=total_count)

    # ------------------------------------------------------------------
    # Async (future-returning) variants
    # ------------------------------------------------------------------

    def upsert_async(
        self,
        *,
        vectors: Sequence[
            Vector
            | tuple[str, Sequence[float]]
            | tuple[str, Sequence[float], Mapping[str, Any]]
            | Mapping[str, Any]
        ],
        namespace: str = "",
        timeout: float | None = None,
    ) -> PineconeFuture[UpsertResponse]:
        """Submit an upsert operation and return a :class:`PineconeFuture`.

        Same parameters as :meth:`upsert`, including ``timeout (float | None)``
        which sets a per-call timeout in seconds.

        Returns:
            :class:`PineconeFuture` [:class:`UpsertResponse`] that resolves to
            the upsert result.

        Examples:

            .. code-block:: python

                future = index.upsert_async(
                    vectors=[("doc-42", [0.012, -0.087, 0.153])],
                )
                result = future.result()
                result.upserted_count  # 1
        """
        future: PineconeFuture[UpsertResponse] = PineconeFuture(
            self._executor.submit(
                self.upsert, vectors=vectors, namespace=namespace, timeout=timeout
            )
        )
        return future

    def query_async(
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
    ) -> PineconeFuture[QueryResponse]:
        """Submit a query operation and return a :class:`PineconeFuture`.

        Same parameters as :meth:`query`, including ``timeout (float | None)``
        which sets a per-call timeout in seconds.

        Returns:
            :class:`PineconeFuture` [:class:`QueryResponse`] that resolves to
            the query result containing scored matches.

        Examples:

            .. code-block:: python

                future = index.query_async(
                    vector=[0.012, -0.087, 0.153],
                    top_k=5,
                )
                result = future.result()
                result.matches[0].id    # 'doc-42'
                result.matches[0].score  # 0.95
        """
        future: PineconeFuture[QueryResponse] = PineconeFuture(
            self._executor.submit(
                self.query,
                top_k=top_k,
                vector=vector,
                id=id,
                namespace=namespace,
                filter=filter,
                include_values=include_values,
                include_metadata=include_metadata,
                sparse_vector=sparse_vector,
                scan_factor=scan_factor,
                max_candidates=max_candidates,
                timeout=timeout,
            )
        )
        return future

    def fetch_async(
        self,
        *,
        ids: Sequence[str],
        namespace: str = "",
        timeout: float | None = None,
    ) -> PineconeFuture[FetchResponse]:
        """Submit a fetch operation and return a :class:`PineconeFuture`.

        Same parameters as :meth:`fetch`, including ``timeout (float | None)``
        which sets a per-call timeout in seconds.

        Returns:
            :class:`PineconeFuture` [:class:`FetchResponse`] that resolves to
            the fetched vectors keyed by ID.

        Examples:

            .. code-block:: python

                future = index.fetch_async(ids=["doc-42", "doc-43"])
                result = future.result()
                result.vectors["doc-42"].values  # [0.012, -0.087, 0.153]
        """
        future: PineconeFuture[FetchResponse] = PineconeFuture(
            self._executor.submit(self.fetch, ids=ids, namespace=namespace, timeout=timeout)
        )
        return future

    def delete_async(
        self,
        *,
        ids: Sequence[str] | None = None,
        delete_all: bool = False,
        filter: Mapping[str, Any] | None = None,
        namespace: str = "",
        timeout: float | None = None,
    ) -> PineconeFuture[None]:
        """Submit a delete operation and return a :class:`PineconeFuture`.

        Same parameters as :meth:`delete`, including ``timeout (float | None)``
        which sets a per-call timeout in seconds.

        Returns:
            :class:`PineconeFuture` [None] that resolves when the delete
            operation completes.

        Examples:

            .. code-block:: python

                future = index.delete_async(ids=["doc-42", "doc-43"])
                future.result()

            .. code-block:: python

                future = index.delete_async(delete_all=True, namespace="docs")
                future.result()
        """
        future: PineconeFuture[None] = PineconeFuture(
            self._executor.submit(
                self.delete,
                ids=ids,
                delete_all=delete_all,
                filter=filter,
                namespace=namespace,
                timeout=timeout,
            )
        )
        return future

    def update_async(
        self,
        *,
        id: str | None = None,
        values: Sequence[float] | None = None,
        sparse_values: SparseValues | Mapping[str, Any] | None = None,
        set_metadata: Mapping[str, Any] | None = None,
        filter: Mapping[str, Any] | None = None,
        namespace: str = "",
        dry_run: bool = False,
        timeout: float | None = None,
    ) -> PineconeFuture[UpdateResponse]:
        """Submit an update call without blocking; returns a :class:`PineconeFuture`."""
        return PineconeFuture(
            self._executor.submit(
                self.update,
                id=id,
                values=values,
                sparse_values=sparse_values,
                set_metadata=set_metadata,
                filter=filter,
                namespace=namespace,
                dry_run=dry_run,
                timeout=timeout,
            )
        )

    def upsert_records(
        self,
        *,
        records: builtins.list[dict[str, Any]],
        namespace: str,
        timeout: float | None = None,
    ) -> UpsertRecordsResponse:
        """Upsert records for indexes with integrated inference.

        Records are sent as newline-delimited JSON (NDJSON) over REST. Embeddings
        are generated server-side. This method delegates to the REST endpoint
        because the Pinecone gRPC API does not expose a records upsert operation.

        Args:
            records: List of record dicts. Each must contain an ``_id`` or
                ``id`` field. Additional fields are passed through for
                server-side embedding.
            namespace (str): Target namespace (required). Use ``""`` for the
                default namespace.

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

                pc = Pinecone(api_key="YOUR_API_KEY")
                idx = pc.index("my-index", grpc=True)
                response = idx.upsert_records(
                    namespace="articles-en",
                    records=[
                        {"_id": "article-101", "text": "Vector DBs enable similarity search."},
                        {"_id": "article-102", "text": "RAG combines search with LLMs."},
                    ],
                )
                print(response.record_count)
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

        normalized: builtins.list[dict[str, Any]] = []
        for record in records:
            r = dict(record)
            if "_id" not in r and "id" in r:
                r["_id"] = r.pop("id")
            normalized.append(r)

        ndjson_lines = [orjson.dumps(r).decode("utf-8") for r in normalized]
        ndjson_body = "\n".join(ndjson_lines) + "\n"

        logger.info(
            "Upserting %d records into namespace %r (NDJSON via REST)", len(records), namespace
        )
        response = self._http.post(
            f"/records/namespaces/{namespace}/upsert",
            timeout=timeout,
            content=ndjson_body.encode("utf-8"),
            headers={"Content-Type": "application/x-ndjson"},
        )
        result = UpsertRecordsResponse(record_count=len(records))
        result.response_info = extract_response_info(response)
        return result

    def search(
        self,
        *,
        namespace: str,
        top_k: int,
        inputs: SearchInputs | Mapping[str, Any] | None = None,
        vector: Sequence[float] | None = None,
        id: str | None = None,
        filter: Mapping[str, Any] | None = None,
        fields: Sequence[str] | None = None,
        rerank: RerankConfig | Mapping[str, Any] | None = None,
        match_terms: Mapping[str, Any] | None = None,
        timeout: float | None = None,
    ) -> SearchRecordsResponse:
        """Search records by text, vector, or ID with optional reranking.

        Delegates to the REST endpoint because the Pinecone gRPC API does not
        expose a records search operation for integrated inference indexes.

        .. note::
           Use this method for indexes with integrated inference. For classic
           indexes where you provide your own vectors, use :meth:`query`.

        Args:
            namespace (str): Namespace to search in (required).
            top_k (int): Number of results to return (must be >= 1).
            inputs (SearchInputs | dict[str, Any] | None): Inputs for
                server-side embedding (e.g. ``{"text": "query text"}``).
            vector (list[float] | None): Dense query vector values.
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

                response = idx.search(
                    namespace="articles-en",
                    top_k=10,
                    inputs={"text": "benefits of vector databases for search"},
                )
                for hit in response.result.hits:
                    print(hit.id, hit.score)

            Search with reranking:

            .. code-block:: python

                response = idx.search(
                    namespace="articles-en",
                    top_k=10,
                    inputs={"text": "benefits of vector databases"},
                    rerank={
                        "model": "bge-reranker-v2-m3",
                        "rank_fields": ["text"],
                        "top_n": 5,
                    },
                )
                for hit in response.result.hits:
                    print(hit.id, hit.score)

        .. note::
           Use inline ``rerank`` when searching and reranking in a single call.
           Use ``pc.inference.rerank()`` when reranking results from a different
           source or when you need to rerank without searching.
        """
        if not isinstance(namespace, str):
            raise ValidationError("namespace must be a string")
        if not namespace or not namespace.strip():
            raise ValidationError("namespace must be a non-empty string")
        if top_k < 1:
            raise ValidationError(f"top_k must be a positive integer, got {top_k}")
        if rerank is not None:
            if "model" not in rerank:
                raise ValidationError("rerank requires 'model' to be specified")
            if "rank_fields" not in rerank:
                raise ValidationError("rerank requires 'rank_fields' to be specified")
        if inputs is None and vector is None and id is None:
            raise ValidationError(
                "At least one of inputs, vector, or id must be provided as a query source"
            )

        query_body: dict[str, Any] = {"top_k": top_k}
        if inputs is not None:
            query_body["inputs"] = inputs
        if vector is not None:
            query_body["vector"] = vector
        if id is not None:
            query_body["id"] = id
        if filter is not None:
            query_body["filter"] = filter
        if match_terms is not None:
            query_body["match_terms"] = match_terms

        body: dict[str, Any] = {"query": query_body}
        if fields is not None:
            body["fields"] = fields
        if rerank is not None:
            body["rerank"] = rerank

        logger.info("Searching namespace %r with top_k=%d (via REST)", namespace, top_k)
        response = self._http.post(
            f"/records/namespaces/{namespace}/search", timeout=timeout, json=body
        )
        result = self._adapter.to_search_response(response.content)
        result.response_info = extract_response_info(response)
        return result

    def search_records(
        self,
        *,
        namespace: str,
        top_k: int,
        inputs: SearchInputs | Mapping[str, Any] | None = None,
        vector: Sequence[float] | None = None,
        id: str | None = None,
        filter: Mapping[str, Any] | None = None,
        fields: Sequence[str] | None = None,
        rerank: RerankConfig | Mapping[str, Any] | None = None,
        match_terms: Mapping[str, Any] | None = None,
        timeout: float | None = None,
    ) -> SearchRecordsResponse:
        """Alias for :meth:`search`.

        Prefer calling :meth:`search` directly — this alias exists for backwards compatibility.
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
            timeout=timeout,
        )

    def list_namespaces_paginated(
        self,
        *,
        prefix: str | None = None,
        limit: int | None = None,
        pagination_token: str | None = None,
        timeout: float | None = None,
    ) -> ListNamespacesResponse:
        """Fetch a single page of namespace descriptions via gRPC.

        Args:
            prefix (str | None): Return only namespaces whose names start with this prefix.
            limit (int | None): Maximum number of namespaces to return in this page.
            pagination_token (str | None): Token from a previous response to fetch the next page.
            timeout (float | None): Per-call timeout in seconds.

        Returns:
            :class:`ListNamespacesResponse` with namespace descriptions, pagination info,
            and total count.
        """
        logger.info("Listing namespaces (paginated) via gRPC")
        result = self._channel.list_namespaces(
            prefix=prefix,
            limit=limit,
            pagination_token=pagination_token,
            timeout_s=timeout,
        )

        namespaces = [
            _dict_to_namespace_description(ns_data) for ns_data in result.get("namespaces", [])
        ]

        pagination: Pagination | None = None
        raw_pag = result.get("pagination")
        if raw_pag is not None:
            pagination = Pagination(next=raw_pag.get("next"))

        return ListNamespacesResponse(
            namespaces=namespaces,
            pagination=pagination,
            total_count=result.get("total_count", 0),
        )

    def list_namespaces(
        self,
        *,
        prefix: str | None = None,
        limit: int | None = None,
        timeout: float | None = None,
    ) -> Iterator[ListNamespacesResponse]:
        """List namespaces, automatically following pagination.

        Yields one :class:`ListNamespacesResponse` per page. The generator
        automatically follows pagination tokens until all pages have been
        retrieved.

        Args:
            prefix (str | None): Return only namespaces whose names start with this prefix.
            limit (int | None): Maximum number of namespaces to return per page.
            timeout (float | None): Per-call timeout in seconds.

        Yields:
            :class:`ListNamespacesResponse` for each page of results.

        Examples:
            .. code-block:: python

                for page in idx.list_namespaces(prefix="prod-"):
                    for ns in page.namespaces:
                        print(ns.name, ns.record_count)
        """
        pagination_token: str | None = None
        while True:
            page = self.list_namespaces_paginated(
                prefix=prefix,
                limit=limit,
                pagination_token=pagination_token,
                timeout=timeout,
            )
            if page.namespaces:
                yield page
            if page.pagination is not None and page.pagination.next is not None:
                pagination_token = page.pagination.next
            else:
                break

    def create_namespace(
        self,
        *,
        name: str,
        schema: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> NamespaceDescription:
        """Create a named namespace in the index via gRPC.

        Args:
            name (str): Name for the new namespace (must be non-empty).
            schema (dict[str, Any] | None): Optional schema configuration
                with metadata field indexing settings.
            timeout (float | None): Per-call timeout in seconds.

        Returns:
            :class:`NamespaceDescription` with the namespace name and record count.

        Raises:
            :exc:`ValidationError`: If the name is not a string or is empty/whitespace.
        """
        if not isinstance(name, str):
            raise ValidationError("namespace name must be a string")
        if not name or not name.strip():
            raise ValidationError("namespace name must be a non-empty string")

        logger.info("Creating namespace %r via gRPC", name)
        result = self._channel.create_namespace(name, schema, timeout_s=timeout)
        return _dict_to_namespace_description(result)

    def describe_namespace(
        self,
        *,
        name: str | None = None,
        timeout: float | None = None,
        **kwargs: str,
    ) -> NamespaceDescription:
        """Describe a namespace by name.

        Args:
            name (str): Name of the namespace to describe.
            timeout (float | None): Per-call timeout in seconds.

        Returns:
            :class:`NamespaceDescription` with the namespace name, record count,
            and schema information.

        Raises:
            :exc:`ValidationError`: If the name is not a string or is empty/whitespace.
            :exc:`TypeError`: If unexpected keyword arguments are passed.
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

        logger.info("Describing namespace %r via gRPC", effective)
        result = self._channel.describe_namespace(effective, timeout_s=timeout)
        return _dict_to_namespace_description(result)

    def delete_namespace(
        self,
        *,
        name: str | None = None,
        timeout: float | None = None,
        **kwargs: str,
    ) -> None:
        """Delete a namespace by name, removing all its vectors.

        Args:
            name (str): Name of the namespace to delete.
            timeout (float | None): Per-call timeout in seconds.

        Returns:
            None — a successful delete returns no payload.

        Raises:
            :exc:`ValidationError`: If the name is not a string or is empty/whitespace.
            :exc:`TypeError`: If unexpected keyword arguments are passed.
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

        logger.info("Deleting namespace %r via gRPC", effective)
        self._channel.delete_namespace(effective, timeout_s=timeout)

    def close(self) -> None:
        """Close the underlying gRPC channel, REST client, and release resources."""
        self._executor.shutdown(wait=True)
        if self._batch_executor is not None:
            self._batch_executor.shutdown(wait=False)
        self._http.close()
        if hasattr(self._channel, "close"):
            self._channel.close()

    def __enter__(self) -> GrpcIndex:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


# Legacy capitalisation alias (BCG-141).
GRPCIndex = GrpcIndex

# Legacy name (renamed from PineconeGrpcFuture in the rewrite — BCG-143).
PineconeGrpcFuture = PineconeFuture

from pinecone.grpc.pinecone_grpc import PineconeGRPC  # noqa: E402

__all__ = ["GRPCIndex", "GrpcIndex", "PineconeGRPC", "PineconeGrpcFuture"]
