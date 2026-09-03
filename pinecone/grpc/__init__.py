"""Synchronous gRPC data plane client for a Pinecone index."""

from __future__ import annotations

import builtins
import ipaddress
import logging
import os
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import quote

if TYPE_CHECKING:
    import pandas as pd  # type: ignore[import-untyped]

from pinecone._internal.adapters.imports_adapter import ImportsAdapter
from pinecone._internal.adapters.vectors_adapter import VectorsAdapter, extract_response_info
from pinecone._internal.adaptive import _AdaptiveLimiterRegistry
from pinecone._internal.batching import validate_batch_size
from pinecone._internal.bulk import bulk_execute_sync
from pinecone._internal.config import (
    GRPC_SCHEMES,
    PineconeConfig,
    RetryConfig,
    resolve_grpc_scheme,
)
from pinecone._internal.constants import DATA_PLANE_API_VERSION, DEFAULT_MAX_CONCURRENCY
from pinecone._internal.data_plane_helpers import _build_search_records_body, _validate_host
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
from pinecone._internal.vector_factory import VectorFactory, validate_vector_dict
from pinecone.errors.exceptions import (
    PineconeTimeoutError,
    PineconeValueError,
    ValidationError,
)
from pinecone.grpc._protocol import GrpcChannelProtocol
from pinecone.grpc.future import PineconeFuture
from pinecone.models.batch import BatchResult
from pinecone.models.imports.list import ImportList
from pinecone.models.imports.model import ImportModel, StartImportResponse
from pinecone.models.namespaces.models import (
    IndexedFields,
    ListNamespacesResponse,
    NamespaceDescription,
    NamespaceFieldConfig,
    NamespaceSchema,
)
from pinecone.models.vectors.query_aggregator import QueryNamespacesResults, QueryResultsAggregator
from pinecone.models.vectors.responses import (
    DescribeIndexStatsResponse,
    FetchByMetadataResponse,
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
from pinecone.models.vectors.search import (
    RerankConfig,
    SearchInputs,
    SearchQuery,
    SearchRecordsResponse,
)
from pinecone.models.vectors.sparse import SparseValues
from pinecone.models.vectors.usage import Usage
from pinecone.models.vectors.vector import ScoredVector, Vector

logger = logging.getLogger(__name__)


_PLAINTEXT_SAFE_NETWORKS: tuple[ipaddress.IPv4Network, ...] = (
    ipaddress.IPv4Network("10.0.0.0/8"),
    ipaddress.IPv4Network("172.16.0.0/12"),
    ipaddress.IPv4Network("192.168.0.0/16"),
)

_warned_about_plaintext_grpc = False


def _endpoint_hostname(host: str) -> str:
    """Extract the hostname from a bare ``host[:port]``, IPv6 brackets included."""
    bare = host.split("/", 1)[0]
    if bare.startswith("["):
        return bare[1:].split("]", 1)[0]
    if bare.count(":") > 1:
        return bare
    return bare.split(":", 1)[0]


def _is_plaintext_safe_host(hostname: str) -> bool:
    """Whether plaintext to *hostname* stays inside the caller's own network.

    Loopback and the RFC 1918 ranges qualify. A hostname that is not an IP
    address does not, apart from ``localhost``: a name has to be resolved to
    know where it points, and assuming the best of an unresolvable one would
    silence the warning exactly where it matters.
    """
    lowered = hostname.lower().rstrip(".")
    if lowered == "localhost" or lowered.endswith(".localhost"):
        return True
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        return False
    if address.is_loopback:
        return True
    return any(address in network for network in _PLAINTEXT_SAFE_NETWORKS)


def _warn_plaintext_grpc_once(host: str) -> None:
    """Warn the first time this process dials a public host without TLS."""
    global _warned_about_plaintext_grpc
    hostname = _endpoint_hostname(host)
    if _warned_about_plaintext_grpc or _is_plaintext_safe_host(hostname):
        return
    _warned_about_plaintext_grpc = True
    warnings.warn(
        f"The gRPC data plane is configured to dial {hostname} over http, so the "
        "API key and every request and response body travel unencrypted. Pass "
        'grpc_scheme="https" (and unset PINECONE_GRPC_SCHEME if it is set to '
        '"http") to encrypt the connection.',
        RuntimeWarning,
        stacklevel=4,
    )


def _build_grpc_endpoint(host: str, *, secure: bool, scheme: str | None) -> str:
    """Build a gRPC endpoint URL from a host string.

    Strips any existing scheme and applies *scheme*, or the one implied by
    *secure* when no scheme was configured.

    The scheme, not *secure*, is what decides whether the wire carries TLS:
    tonic runs a handshake only for an ``https`` endpoint, so a ``http``
    endpoint stays plaintext even with TLS material configured, and an
    ``https`` one without that material cannot connect at all. That pairing is
    rejected here rather than at the first call.

    A resolved ``http`` scheme against a host that is neither loopback nor
    RFC 1918 private emits a :exc:`RuntimeWarning` once per process, since the
    API key then crosses a public network in the clear.

    Raises:
        PineconeValueError: If *scheme* is ``"https"`` while *secure* is
            ``False``, or names anything other than ``http`` or ``https``.
    """
    if scheme is not None and scheme not in GRPC_SCHEMES:
        raise PineconeValueError(
            f"Invalid gRPC scheme {scheme!r}. Must be one of: {', '.join(GRPC_SCHEMES)}."
        )
    if scheme == "https" and not secure:
        raise PineconeValueError(
            'grpc_scheme="https" requires secure=True: an https endpoint needs the TLS '
            "material secure=False withholds, so the channel could not connect. Pass "
            'secure=True for a TLS data plane, or grpc_scheme="http" for a plaintext one.'
        )

    bare = host
    for prefix in ("https://", "http://"):
        if bare.startswith(prefix):
            bare = bare[len(prefix) :]
            break

    resolved = scheme if scheme is not None else ("https" if secure else "http")
    if resolved == "http":
        _warn_plaintext_grpc_once(bare)
    return f"{resolved}://{bare}"


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

    ``indexed_fields`` arrives as a bare list of names here, where the REST JSON
    nests the same names under a ``fields`` key — see
    ``namespace_description_to_py_dict`` in rust/src/transport.rs. Both shapes
    have to produce the same model, so the two readers cannot be collapsed.
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
        size_bytes=data.get("size_bytes", 0),
    )


# gRPC's retry defaults, which differ from REST's (3 / 0.25s / 60s). The counts and
# floor are what the Rust layer has always used; only the cap changed, from 1.6s — a
# value small enough to swallow a `grpc-retry-pushback-ms: 30000` hint from the server.
_GRPC_DEFAULT_MAX_RETRIES = 5
_GRPC_DEFAULT_BACKOFF_FACTOR = 0.1
_GRPC_DEFAULT_MAX_WAIT = 60.0


_warned_about_grpc_partial_failure = False


def _warn_grpc_partial_failure_once(response: UpsertResponse) -> None:
    """Announce the 10.0.0 change the first time a caller is affected by it.

    Only on gRPC: REST has aggregated since v9.0.0, so warning there would be
    noise about behavior that did not change.
    """
    global _warned_about_grpc_partial_failure
    if _warned_about_grpc_partial_failure:
        return
    _warned_about_grpc_partial_failure = True
    warnings.warn(
        f"{response.failed_item_count} of {response.total_item_count} vectors failed to "
        "upsert. As of 10.0.0 upsert_from_dataframe aggregates partial failures instead "
        "of raising: inspect response.errors and retry response.failed_items. Pass "
        'on_error="raise" to restore the previous behavior, or on_error="collect" to '
        "silence this warning.",
        stacklevel=3,
    )


def _upsert_response_from(batch_result: BatchResult) -> UpsertResponse:
    """Project a BatchResult onto the response shape callers already handle."""
    return UpsertResponse(
        upserted_count=batch_result.successful_item_count,
        total_item_count=batch_result.total_item_count,
        failed_item_count=batch_result.failed_item_count,
        total_batch_count=batch_result.total_batch_count,
        successful_batch_count=batch_result.successful_batch_count,
        failed_batch_count=batch_result.failed_batch_count,
        errors=batch_result.errors,
    )


def _bulk_gate_registry() -> Any:
    """Deferred import: the bulk registry pulls in gate machinery this module
    only needs once a client is constructed, not at import time."""
    from pinecone._internal.bulk import get_registry

    return get_registry()


def _limiter_host(host: str) -> str:
    """The key the Rust throttle callback reports under.

    ``self._host`` carries a scheme, because ``normalize_host`` adds one; the
    callback receives what ``parse_host_from_endpoint`` produced, which is the
    bare hostname. Registering a limiter under one and reporting throttles
    against the other would leave the limiter permanently at its ceiling.
    """
    bare = host
    for prefix in ("https://", "http://"):
        if bare.startswith(prefix):
            bare = bare[len(prefix) :]
            break
    for separator in (":", "/"):
        bare = bare.split(separator, 1)[0]
    return bare


def _as_sentence(text: str) -> str:
    """Close the wrapped message off, so appended guidance starts a new sentence.

    Without the period, ``deadline exceeded`` and the guidance become one clause,
    and anything matching on the leading clause picks up the run-specific timeout
    values from the guidance — which these messages do not promise to keep stable.
    """
    stripped = text.rstrip()
    if not stripped or stripped[-1] in ".!?":
        return stripped
    return f"{stripped}."


@keyword_only_methods
class GrpcIndex:
    """Synchronous gRPC data plane client targeting a specific Pinecone index.

    Reach it as ``pc.index(name="articles-en", grpc=True)``, which resolves the
    host for you, or construct it directly when you already know the host.

    It offers the same data-plane methods as :class:`~pinecone.index.Index` and
    is the one to reach for when throughput on a long ingest matters; on
    everything else :class:`~pinecone.index.Index` is the better default,
    because gRPC has no asyncio twin and needs a compiled extension. Three
    differences are visible in the code you write: the ``*_async`` methods here
    return a :class:`~pinecone.grpc.future.PineconeFuture` rather than
    something you ``await``; ``retry_config.retryable_status_codes`` has no
    effect, since this transport retries gRPC status codes rather than HTTP
    ones; and :meth:`upsert_records` and :meth:`search` still travel over REST,
    because the gRPC API has no records operations. See :doc:`/guides/grpc`.

    Args:
        host (str): The index-specific data plane host URL.
        api_key (str | None): Pinecone API key. Falls back to ``PINECONE_API_KEY`` env var.
        api_version (str): API version string. Defaults to the current data plane version.
        source_tag (str | None): Tag appended to the User-Agent string for request attribution.
        secure (bool): Whether the channel is given TLS material — system root
            certificates for gRPC, certificate verification for the REST calls this
            client makes alongside it. Defaults to ``True``. It supplies the default
            for ``grpc_scheme``, and ``grpc_scheme`` is what decides whether the wire
            is actually encrypted.
        grpc_scheme ("http" | "https" | None): URL scheme used to dial the data plane.
            State it when the data plane is reached over something other than public
            TLS — a plaintext gateway, an egress proxy, a private endpoint, or a local
            simulator — rather than leaving the SDK to assume one. ``None`` (default)
            takes the scheme from ``secure``: ``https`` when ``True``, ``http`` when
            ``False``. Falls back to the ``PINECONE_GRPC_SCHEME`` env var before that
            default applies. ``"https"`` requires ``secure=True``, since an https
            endpoint cannot connect without the TLS material ``secure=False``
            withholds. ``"http"`` with ``secure=True`` is a plaintext channel: the
            scheme, not the TLS material, decides what goes on the wire. A
            resolved ``http`` scheme against a host outside loopback and the
            RFC 1918 private ranges warns once per process, because the API key
            and every payload then cross a public network unencrypted.
        timeout (float): Deadline in seconds for a **single attempt** of a request, not
            for the call as a whole. Defaults to ``20.0``. A per-call ``timeout=`` does not
            replace it — the channel keeps this one too, so the shorter of the two governs.
        connect_timeout (float): Connection timeout in seconds. Defaults to ``1.0``.
        retry_config (RetryConfig | None): Retry policy for transient gRPC errors. Accepts
            the same :class:`~pinecone._internal.config.RetryConfig` REST uses. ``None``
            (default) uses the gRPC defaults: ``max_retries=5``, ``backoff_factor=0.1``,
            ``max_wait=60.0``, which differ from REST's — so a ``retry_config`` you leave
            unset on :class:`~pinecone.Pinecone` does not carry over here. Its
            ``retryable_status_codes`` field is **ignored on this transport**: it carries
            HTTP statuses, and the codes retried here are gRPC ones. See
            :doc:`/guides/retries`.
        proxy_url (str | None): HTTP proxy URL. gRPC traffic is tunnelled through it with
            HTTP CONNECT.
        limiter_registry (_AdaptiveLimiterRegistry | None): SDK-internal. Registry the
            bulk paths consult to back off under throttling. Wired by
            :meth:`Pinecone.index`; not intended for user configuration.

    Raises:
        :exc:`PineconeValueError`: If no API key can be resolved, the host is invalid,
            ``grpc_scheme`` names a scheme other than ``http`` or ``https``, or
            ``grpc_scheme="https"`` is combined with ``secure=False``.

    Examples:

        .. code-block:: python

            from pinecone.grpc import GrpcIndex

            idx = GrpcIndex(host="movie-recs-abc123.svc.pinecone.io", api_key="...")

        A data plane fronted by a plaintext gateway or served by a local
        simulator is dialled over ``http`` by saying so:

        .. code-block:: python

            idx = GrpcIndex(
                host="http://127.0.0.1:5085",
                api_key="...",
                grpc_scheme="http",
            )

    Note:
        **Four timeout layers apply to every gRPC call**, and only the first three bound a
        single request:

        1. **Connect** — ``connect_timeout``.
        2. **Per attempt** — ``timeout``, or a per-call ``timeout=``. This is a deadline on
           *one attempt*, not on the call. Both apply when a call passes its own, so the
           shorter of the two is what fires.
        3. **Retry budget** — ``retry_config.max_retries`` attempts after the first, with
           backoff between them.
        4. **Whole job** — for bulk methods only, ``total_timeout``.

        Layers 2 and 3 compound only across *retryable* failures, and this transport retries
        exactly three gRPC status codes: UNAVAILABLE, RESOURCE_EXHAUSTED, and ABORTED. So
        the multiplied worst case — every attempt burning nearly its full deadline and then
        failing with one of those — is what a lower ``max_retries`` shrinks.

        **An expiring deadline is not one of the three.** Layer 2 firing raises
        :exc:`~pinecone.errors.exceptions.PineconeTimeoutError` after a single attempt, so
        ``max_retries`` is not the knob for a timeout. Raise ``timeout=`` to give the server
        longer per attempt — raising the index-level ``timeout`` too if it is the lower of
        the two — or bound a bulk job with ``total_timeout``.

    .. seealso::
       :class:`~pinecone.index.Index` — the REST client, and the better default
       unless you are ingesting at volume. :doc:`/guides/grpc` compares the
       two, and :doc:`/guides/retries` gives the full retry policy for both.
    """

    def __init__(
        self,
        *,
        host: str,
        api_key: str | None = None,
        api_version: str = DATA_PLANE_API_VERSION,
        source_tag: str | None = None,
        secure: bool = True,
        grpc_scheme: Literal["http", "https"] | None = None,
        timeout: float = 20.0,
        connect_timeout: float = 1.0,
        retry_config: RetryConfig | None = None,
        proxy_url: str | None = None,
        on_throttle: Callable[[str], None] | None = None,
        limiter_registry: _AdaptiveLimiterRegistry | None = None,
    ) -> None:
        # Resolve API key: explicit arg > env var
        resolved_key = api_key or os.environ.get("PINECONE_API_KEY", "")
        if not resolved_key:
            raise ValidationError(
                "No API key provided. Pass api_key='...' or set the "
                "PINECONE_API_KEY environment variable."
            )

        # Validate and normalize host
        self._request_timeout = timeout
        self._host = _validate_host(host)
        self._limiter_host = _limiter_host(self._host)
        # A directly-constructed handle has no client behind it to supply one,
        # and without a registry the bulk paths do no adaptive backoff at all.
        # Per-handle state is weaker than per-client but far better than none.
        self._limiter_registry = limiter_registry or _AdaptiveLimiterRegistry()
        self._source_tag = source_tag

        # Build gRPC endpoint and create the Rust-backed channel
        endpoint = _build_grpc_endpoint(
            self._host, secure=secure, scheme=resolve_grpc_scheme(grpc_scheme)
        )

        from pinecone import __version__
        from pinecone._grpc import GrpcChannel  # type: ignore[import-not-found]

        # `retryable_status_codes` is deliberately not forwarded: it carries HTTP
        # statuses, and this transport retries a fixed set of tonic::Code values.
        # Forcing HTTP statuses through a gRPC channel would be meaningless.
        self._retry_config = retry_config or RetryConfig(
            max_retries=_GRPC_DEFAULT_MAX_RETRIES,
            backoff_factor=_GRPC_DEFAULT_BACKOFF_FACTOR,
            max_wait=_GRPC_DEFAULT_MAX_WAIT,
        )
        # The bulk gate must always hear throttles — it is how admission adapts —
        # so the transport callback feeds the process-global registry first and
        # any caller-supplied hook (explicit argument, or threaded through a
        # client-built RetryConfig) second, as observability.
        user_on_throttle = on_throttle or self._retry_config.on_throttle

        def resolved_on_throttle(throttled_host: str) -> None:
            _bulk_gate_registry().report_throttled(throttled_host)
            if user_on_throttle is not None:
                user_on_throttle(throttled_host)

        self._channel: GrpcChannelProtocol = GrpcChannel(
            endpoint,
            resolved_key,
            api_version,
            __version__,
            secure,
            timeout,
            connect_timeout,
            max_retries=self._retry_config.max_retries,
            backoff_factor_s=self._retry_config.backoff_factor,
            max_wait_s=self._retry_config.max_wait,
            source_tag=source_tag,
            proxy_url=proxy_url,
            on_throttle=resolved_on_throttle,
        )

        self._executor = ThreadPoolExecutor()

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
        self._imports_adapter = ImportsAdapter()

        logger.info("GrpcIndex client created for host %s", self._host)

    @property
    def host(self) -> str:
        """The data plane host URL for this index."""
        return self._host

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
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        show_progress: bool = True,
        timeout: float | None = None,
        total_timeout: float | None = None,
    ) -> UpsertResponse:
        """Upsert a batch of vectors into a namespace.

        If a vector with the same ID already exists in the namespace, it is
        overwritten.

        One request is capped both on the number of vectors it carries and on
        its encoded size, and with wide vectors or heavy metadata the size cap
        is usually the one reached first. Pass ``batch_size`` to split a long
        sequence into requests that stay under both.

        Args:
            vectors: Sequence of vectors to upsert. Each element can be a
                ``Vector`` instance, a tuple of ``(id, values)`` or
                ``(id, values, metadata)``, or a dict with ``id``, ``values``,
                and optional ``sparse_values`` / ``metadata`` keys.
            namespace (str): Target namespace. Defaults to the default
                (empty-string) namespace.
            batch_size (int | None): If set, splits ``vectors`` into batches of
                this size and submits them in **parallel**. ``None`` (default)
                sends all vectors in a single request. Must be a positive
                integer when set.
            max_concurrency (int): Number of parallel threads used when
                ``batch_size`` is set. Default ``8``, range ``[1, 64]``. Ignored
                when ``batch_size`` is ``None``.
            show_progress (bool): If ``True`` and ``tqdm`` is installed, display a
                progress bar while submitting batches. Ignored when ``batch_size``
                is ``None``. Defaults to ``True``.
            timeout (float | None): Per-call timeout in seconds. Applied per batch
                when batching. None uses the client-level default.
            total_timeout (float | None): Deadline in seconds for the whole
                batched operation (only meaningful with ``batch_size``). On
                expiry no further batches are submitted; batches already in
                flight are awaited and never cancelled; unsent batches are
                reported in ``failed_items``. ``None`` (default) means no
                deadline.

        Returns:
            :class:`~pinecone.models.vectors.responses.UpsertResponse` with ``upserted_count``. With
            ``batch_size`` set it also carries ``failed_item_count``,
            ``errors``, and ``failed_items``: a batch that fails does not raise,
            so check ``failed_item_count`` and hand ``failed_items`` straight
            back to :meth:`upsert` to retry only what did not land. Upserts are
            idempotent by vector ID, so a retry that overlaps is harmless.

        Raises:
            :exc:`PineconeTypeError`: If a vector element is not a recognized format.
            :exc:`PineconeValueError`: If a vector element is malformed, if
                ``batch_size`` is not a positive integer, or if
                ``max_concurrency`` is outside ``[1, 64]``.
            :exc:`ApiError`: If one request exceeds the server's cap on vectors
                per request or on encoded request size. Lower ``batch_size``
                and retry.

        Examples:
            Each element can be a :class:`~pinecone.models.vectors.vector.Vector`, a ``(id,
            values)`` tuple, or a dict — the three forms below are interchangeable, and the values
            are truncated here for length:

            .. code-block:: python

                from pinecone.grpc import GrpcIndex
                from pinecone.models.vectors.vector import Vector

                idx = GrpcIndex(host="article-search-abc123.svc.pinecone.io", api_key="...")
                response = idx.upsert(
                    vectors=[
                        Vector(id="article-101", values=[0.012, -0.087, 0.153]),
                        ("article-102", [0.045, 0.021, -0.064]),
                        {"id": "article-103", "values": [0.091, -0.032, 0.178]},
                    ],
                    namespace="articles-en",
                )
                print(response.upserted_count)

            For a long sequence, set ``batch_size`` and read the failure fields
            rather than relying on an exception:

            .. code-block:: python

                response = idx.upsert(
                    vectors=all_vectors,
                    namespace="articles-en",
                    batch_size=200,
                    total_timeout=600.0,
                )
                if response.failed_item_count:
                    idx.upsert(vectors=response.failed_items, namespace="articles-en")

        .. seealso::
           :meth:`upsert_records` — for an index with integrated inference,
           where you send text and the server embeds it.
           :meth:`start_import` — for a one-off load of millions of vectors
           already sitting in cloud storage.
           :doc:`/migration/v10-grpc-partial-failures` — how to read the
           partial-failure fields, and what changed for callers who expected a
           raise.
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

        batch_result = bulk_execute_sync(
            items=items,
            operation=_operation,
            batch_size=batch_size,
            max_concurrency=max_concurrency,
            show_progress=show_progress,
            desc="Upserting",
            host=self._limiter_host,
            total_timeout=total_timeout,
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

        Use this on an index you upsert your own vectors into. An index that
        carries a document schema is read through :meth:`search` instead, which
        embeds the query text server-side.

        Args:
            top_k (int): Number of results to return, 1-10000.
            vector (list[float] | None): Dense query vector values.
            id (str | None): ID of a stored vector to use as the query.
            namespace (str): Namespace to query. Defaults to the default namespace.
            filter (dict[str, Any] | None): Metadata filter expression.
            include_values (bool): Whether to include vector values in results.
            include_metadata (bool): Whether to include metadata in results.
            sparse_vector (SparseValues | dict[str, Any] | None): Sparse query vector
                with indices and values.
            scan_factor (float | None): Recall/latency trade for dedicated read
                node (DRN) indexes — a multiplier on how much of the index is
                scanned. Above 1 scans more and favours recall; below 1 scans
                less and favours latency. Omit to let the server choose.
            max_candidates (int | None): Recall/latency trade for dedicated read
                node (DRN) indexes — caps how many candidates are reranked before
                ``top_k`` is taken. Must be at least ``top_k``: a smaller value is
                rejected rather than clamped, since it could not fill the page.
            timeout (float | None): Per-call timeout in seconds. None uses the client-level default.

        Returns:
            :class:`~pinecone.models.vectors.responses.QueryResponse` with matches, namespace, and
            usage info.

        Raises:
            :exc:`PineconeValueError`: If top_k is not between 1 and 10000, ``id``
                is combined with ``vector`` or ``sparse_vector``, none of
                ``vector``, ``id``, or ``sparse_vector`` is provided, or ``id``
                is not a legal vector ID.
            :exc:`ApiError`: If ``scan_factor`` or ``max_candidates`` is out of
                range, or the index is not a dense DRN index — both knobs are
                rejected on on-demand indexes and on sparse indexes.

        Examples:

            .. code-block:: python

                response = idx.query(
                    top_k=10,
                    vector=[0.012, -0.087, 0.153, ...],  # 1536-dim embedding
                )
                for match in response.matches:
                    print(match.id, match.score)

        .. seealso::
           :meth:`search` — for an index with integrated inference, where you
           send query text and the server embeds it.
           :meth:`query_namespaces` — to run the same query across several
           namespaces and merge the results.
        """
        require_in_range("top_k", top_k, 1, QUERY_TOP_K_MAX)
        require_query_selectors(vector=vector, id=id, sparse_vector=sparse_vector)
        if id is not None:
            require_valid_vector_id("id", id)

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
        """Query multiple namespaces in parallel and return merged top results.

        Fans out individual ``query()`` calls across all given namespaces
        using a thread pool, then merges results via a heap-based aggregator
        that returns the overall top-k matches ranked by the specified metric.

        Args:
            vector: Dense query vector values. Required for dense and hybrid
                indexes; omit for sparse-only indexes (use *sparse_vector* instead).
            namespaces: Namespaces to query (must be non-empty). Duplicates
                are removed while preserving order.
            metric: The metric the index was created with — ``"cosine"``,
                ``"euclidean"``, or ``"dotproduct"``. It decides which
                direction counts as better when the per-namespace results are
                merged, and ``"euclidean"`` is the one where lower wins. Name
                the wrong one and the merge is not rejected, it is just ordered
                backwards.
            top_k: Maximum number of results to return. Defaults to 10.
            filter: Metadata filter expression applied to every namespace.
            include_values: Whether to include vector values in results.
            include_metadata: Whether to include metadata in results.
            sparse_vector: Sparse query vector with indices and values.
                Required for sparse-only indexes when *vector* is omitted.
            scan_factor: Recall/latency trade for dedicated read node (DRN)
                indexes — a multiplier on how much of the index is scanned.
                Above 1 scans more and favours recall; below 1 scans less and
                favours latency. Applied to every namespace queried.
            max_candidates: Recall/latency trade for dedicated read node (DRN)
                indexes — caps how many candidates are reranked before ``top_k``
                is taken, per namespace. Must be at least ``top_k``.

        Returns:
            :class:`~pinecone.models.vectors.query_aggregator.QueryNamespacesResults` with the
            merged top-k matches, total usage, and per-namespace usage.

        Raises:
            :exc:`PineconeValueError`: If *namespaces* is empty, if both
                *vector* and *sparse_vector* are absent/empty, or if *metric*
                is not a recognized value.
            :exc:`ApiError`: If any individual namespace query fails.

        Examples:
            .. code-block:: python

                results = idx.query_namespaces(
                    vector=[0.012, -0.087, 0.153],
                    namespaces=["articles-en", "articles-fr", "articles-de"],
                    metric="cosine",
                    top_k=10,
                )
                for match in results.matches:
                    print(match.id, match.score)

            On a sparse-only index, send ``sparse_vector`` instead and rank by
            ``"dotproduct"``:

            .. code-block:: python

                results = idx.query_namespaces(
                    sparse_vector={"indices": [412, 8871, 20114], "values": [0.42, 0.19, 0.08]},
                    namespaces=["articles-en", "articles-fr"],
                    metric="dotproduct",
                    top_k=10,
                )

        .. seealso::
           :meth:`query` — one namespace, and the only form that takes an
           ``id`` as the query.
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
        """Fetch vectors by their IDs from a namespace.

        Args:
            ids (list[str]): List of vector IDs to fetch (must be non-empty).
            namespace (str): Namespace to fetch from. Defaults to the default namespace.
            timeout (float | None): Per-call timeout in seconds. None uses the client-level default.

        Returns:
            :class:`~pinecone.models.vectors.responses.FetchResponse` with a map of vector IDs to
            Vector objects, namespace, and usage info.

        Raises:
            :exc:`PineconeValueError`: If ids is empty or any ID is not 1-512
                ASCII characters without a NUL.

        Examples:

            .. code-block:: python

                response = idx.fetch(
                    ids=["article-101", "article-102"],
                    namespace="articles-en",
                )
                for vid, vec in response.vectors.items():
                    print(vid, len(vec.values))

        .. seealso::
           :meth:`fetch_by_metadata` — when you know what the vectors look
           like but not their IDs.
        """
        require_valid_vector_ids("ids", ids)

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

    def fetch_by_metadata(
        self,
        *,
        filter: Mapping[str, Any],
        namespace: str = "",
        limit: int | None = None,
        pagination_token: str | None = None,
        timeout: float | None = None,
    ) -> FetchByMetadataResponse:
        """Fetch vectors matching a metadata filter expression.

        Args:
            filter: Metadata filter expression (required, at least one condition).
            namespace: Namespace to fetch from. Defaults to the default namespace.
            limit: Maximum number of vectors to return per page, 1-10000.
                Omit to let the server choose the page size.
            pagination_token: Token from a previous response to fetch the next page.
            timeout (float | None): Per-call timeout in seconds.

        Returns:
            :class:`~pinecone.models.vectors.responses.FetchByMetadataResponse` with matched
            vectors, namespace, usage, and pagination token for the next page (if any).

        Raises:
            :exc:`PineconeValueError`: If ``filter`` is empty or ``limit`` falls
                outside 1-10000.

        Examples:

            .. code-block:: python

                page = idx.fetch_by_metadata(
                    filter={"topic": {"$eq": "science"}},
                    namespace="articles-en",
                    limit=50,
                )
                for vid, vec in page.vectors.items():
                    print(vid, vec.metadata)
                next_token = page.pagination.next if page.pagination else None

        .. seealso::
           :meth:`fetch` — when you already know the IDs, and want them all in
           one response rather than a page at a time.
           :doc:`/guides/pagination` — following ``pagination.next``.
        """
        if limit is not None:
            require_valid_fetch_by_metadata_limit("limit", limit)
        require_non_empty_filter(
            "filter", filter, server_message=FETCH_BY_METADATA_EMPTY_FILTER_MESSAGE
        )

        logger.info("Fetching vectors by metadata via gRPC")
        result = self._channel.fetch_by_metadata(
            namespace=namespace or None,
            filter=filter,
            limit=limit,
            pagination_token=pagination_token,
            timeout_s=timeout,
        )

        vectors: dict[str, Vector] = {}
        for vid, vdata in result.get("vectors", {}).items():
            vectors[vid] = _dict_to_vector(vid, vdata)

        pagination_data = result.get("pagination")
        pagination = None
        if pagination_data is not None:
            pagination = Pagination(next=pagination_data.get("next"))

        return FetchByMetadataResponse(
            vectors=vectors,
            namespace=result.get("namespace", ""),
            usage=_dict_to_usage(result.get("usage")),
            pagination=pagination,
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

        A by-filter delete selects on metadata alone, so a text-match operator
        (``$match_phrase``, ``$match_all``, ``$match_any``) in the filter is
        rejected rather than ignored — evaluated there it would match everything
        and widen the delete to every record the rest of the filter admits. Text
        matching belongs in :meth:`search`.

        A by-filter delete also reads before it writes, so a dedicated index
        scaled to zero replicas refuses it; add replicas first. Deleting by ID or
        with ``delete_all`` is unaffected.

        Args:
            ids (list[str] | None): List of vector IDs to delete.
            delete_all (bool): If True, delete all vectors in the namespace.
            filter (dict[str, Any] | None): Metadata filter expression selecting vectors to delete.
            namespace (str): Namespace to delete from. Defaults to the default namespace.
            timeout (float | None): Per-call timeout in seconds. None uses the client-level default.

        Returns:
            None

        Raises:
            :exc:`PineconeValueError`: If zero or more than one deletion mode is
                specified, any ID is not a legal vector ID, or ``filter`` is empty.
            :exc:`ApiError`: If a by-filter delete uses a text-match operator, or
                the index is a dedicated index scaled to zero replicas.

        Examples:
            Delete named vectors:

            .. code-block:: python

                idx.delete(ids=["article-101", "article-102"], namespace="articles-en")

            Delete everything a metadata filter selects:

            .. code-block:: python

                idx.delete(filter={"category": {"$eq": "obsolete"}}, namespace="articles-en")

            Empty a namespace entirely. There is no undo and no dry run — every
            vector in it goes:

            .. code-block:: python

                idx.delete(delete_all=True, namespace="articles-deprecated")

        .. seealso::
           :meth:`delete_namespace` — removes the namespace itself, not just
           the vectors in it.
        """
        require_delete_selectors(ids=ids, delete_all=delete_all, filter=filter)
        if ids is not None:
            require_valid_vector_ids("ids", ids)
        if filter is not None:
            require_non_empty_filter("filter", filter, server_message=DELETE_EMPTY_FILTER_MESSAGE)

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

        A by-filter update selects on metadata alone, so a text-match operator
        (``$match_phrase``, ``$match_all``, ``$match_any``) in the filter is
        rejected rather than ignored — evaluated there it would match everything
        and widen the patch to every record the rest of the filter admits. Text
        matching belongs in :meth:`search`.

        A by-filter update also reads before it writes, so a dedicated index
        scaled to zero replicas refuses it; add replicas first. Updating by ID is
        unaffected.

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
            :class:`~pinecone.models.vectors.responses.UpdateResponse` with matched_records count
            (when available).

        Raises:
            :exc:`PineconeValueError`: If both or neither of id and filter are
                provided, if ``filter`` is combined with ``values`` or
                ``sparse_values``, if ``filter`` is empty, or if ``id`` is not
                a legal vector ID.
            :exc:`ApiError`: If a by-filter update uses a text-match operator, or
                the index is a dedicated index scaled to zero replicas.

        Examples:
            Replace one vector's values, leaving its metadata as it was:

            .. code-block:: python

                idx.update(
                    id="article-101",
                    values=[0.012, -0.087, 0.153],
                    namespace="articles-en",
                )

            Set metadata on every record a filter selects. Fields you do not
            name in ``set_metadata`` are left alone:

            .. code-block:: python

                response = idx.update(
                    filter={"topic": {"$eq": "science"}},
                    set_metadata={"reviewed_by": "editorial-team"},
                    namespace="articles-en",
                )
                print(response.matched_records)

            Pass ``dry_run=True`` first to see how many records a filter would
            touch before touching them:

            .. code-block:: python

                preview = idx.update(
                    filter={"topic": {"$eq": "science"}},
                    set_metadata={"reviewed_by": "editorial-team"},
                    namespace="articles-en",
                    dry_run=True,
                )
                print(preview.matched_records)
        """
        require_update_selectors(id=id, filter=filter, values=values, sparse_values=sparse_values)
        if id is not None:
            require_valid_vector_id("id", id)
        if filter is not None:
            require_non_empty_filter("filter", filter, server_message=UPDATE_EMPTY_FILTER_MESSAGE)

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
            limit (int | None): Maximum number of IDs to return in this page, 1-100.
            pagination_token (str | None): Token from a previous response to fetch the next page.
            namespace (str): Namespace to list from. Defaults to the default namespace.
            timeout (float | None): Per-call timeout in seconds. None uses the client-level default.

        Returns:
            :class:`~pinecone.models.vectors.responses.ListResponse` with vector IDs, pagination
            info, namespace, and usage.

        Raises:
            :exc:`PineconeValueError`: If ``prefix`` is not legal or ``limit``
                falls outside 1-100.

        Examples:

            .. code-block:: python

                page = idx.list_paginated(prefix="article-2024#", namespace="articles-en")
                for item in page.vectors:
                    print(item.id)
                next_token = page.pagination.next if page.pagination else None

        .. seealso::
           :meth:`list` — the same walk with the tokens handled for you.
           :doc:`/guides/pagination` — when to drive the tokens yourself.
        """
        if prefix is not None:
            require_valid_id_prefix("prefix", prefix)
        if limit is not None:
            require_valid_list_limit("limit", limit)

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
            :class:`~pinecone.models.vectors.responses.ListResponse` for each page of results.

        Raises:
            :exc:`PineconeValueError`: If ``prefix`` is not legal or ``limit``
                falls outside 1-100.

        Examples:

            .. code-block:: python

                for page in idx.list(prefix="article-2024#", namespace="articles-en"):
                    for item in page.vectors:
                        print(item.id)

        .. seealso::
           :meth:`list_paginated` — one page at a time, when you need to
           persist a token between calls. See :doc:`/guides/pagination`.
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
            filter (dict[str, Any] | None): Metadata filter expression. Accepted
                for API compatibility, but a non-empty filter is rejected for
                every index type, so the call fails instead of returning
                filtered counts. Leave it unset: the statistics returned always
                describe the whole index.
            timeout (float | None): Per-call timeout in seconds. None uses the
                client-level default.

        Returns:
            :class:`~pinecone.models.vectors.responses.DescribeIndexStatsResponse` with namespace
            summaries, dimension, total vector count, and fullness metrics.

        Raises:
            :exc:`ApiError`: If a non-empty ``filter`` is provided, since it is
                rejected for every index type.

        Examples:

            .. code-block:: python

                stats = idx.describe_index_stats()
                print(stats.total_vector_count, stats.dimension)
                for name, summary in stats.namespaces.items():
                    print(name, summary.vector_count)

        .. seealso::
           :meth:`list_namespaces` — per-namespace record counts plus
           ``size_bytes``, which this does not report.
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

    def _timeout_guidance(self, timeout: float | None) -> str:
        """Say which of the four layers fired, with the value that was in effect.

        A timeout here is the entire diagnostic surface for a batch job that died
        partway through, and "deadline exceeded" on its own does not say which
        knob to turn.
        """
        index_level = self._request_timeout
        if timeout is None:
            deadline = f"the index-level timeout of {index_level}s"
        else:
            # The channel keeps the Endpoint-level deadline it was built with even
            # when a call passes its own, so the shorter of the two is what fired.
            # Naming only the per-call value points at the wrong number and the
            # wrong knob whenever the index-level one is smaller.
            deadline = (
                f"{min(timeout, index_level)}s, the shorter of timeout={timeout} and the "
                f"index-level timeout of {index_level}s — both apply to every call"
            )
        return (
            f"The per-attempt deadline fired: {deadline}. It was not the connect timeout "
            f"and not total_timeout. Timeouts are not retried on this transport, which "
            f"retries only UNAVAILABLE, RESOURCE_EXHAUSTED and ABORTED, so this batch "
            f"failed after a single attempt and retry_config.max_retries is not the knob "
            f"to change. Raise timeout= to give the server longer per attempt — raising "
            f"the index-level timeout= too if it is the lower of the two — or set "
            f"total_timeout= to bound the whole ingest. Upserts are idempotent by vector "
            f"id, so retrying the same rows is safe."
        )

    def upsert_from_dataframe(
        self,
        df: pd.DataFrame,
        namespace: str = "",
        batch_size: int = 500,
        show_progress: bool = True,
        timeout: float | None = None,
        *,
        max_concurrency: int | None = None,
        total_timeout: float | None = None,
        on_error: Literal["raise", "collect"] | None = None,
    ) -> UpsertResponse:
        """Upsert vectors from a pandas DataFrame.

        Splits the DataFrame into batches of ``batch_size`` rows, submits
        batches in parallel, and aggregates the results into a single
        response.

        Args:
            df: A ``pandas.DataFrame`` with at least ``id`` and ``values``
                columns. ``sparse_values`` and ``metadata`` columns are
                included when present and non-None.
            namespace: Target namespace. Defaults to the default namespace.
            batch_size: Number of rows per upsert batch. Defaults to 500.
            show_progress: If ``True`` and ``tqdm`` is installed, display a
                progress bar. The bar advances as batches *complete*. If ``tqdm``
                is not installed, silently falls back to no progress bar.
            max_concurrency: Number of batches in flight at once, range
                ``[1, 64]``. ``None`` (default) uses ``8`` — flat and identical
                across every transport and machine, so throughput is
                reproducible across hosts. The host's adaptive limit still
                applies underneath; raise this only when the backend has
                headroom for a larger committed retry burst.
            on_error: What to do when some batches fail. ``"collect"`` returns
                an :class:`~pinecone.models.vectors.responses.UpsertResponse`
                carrying ``failed_item_count``, ``errors`` and
                ``failed_items``, so the caller can retry only what failed —
                the same contract the REST client has had since v9.0.0.
                ``"raise"`` re-raises the lowest-indexed batch failure, after all
                batches have settled, with the partial result attached to the
                exception's ``response`` attribute. ``None`` (default) behaves as
                ``"collect"`` and additionally warns once per process when a
                partial failure occurs, since this method used to raise; pass
                ``"collect"`` explicitly to silence that.
            total_timeout: Deadline in seconds for the **whole ingest**, as opposed
                to *timeout*, which bounds a single attempt of a single batch. On
                expiry no further batches are submitted; batches already in flight
                are allowed to settle rather than being abandoned, since dropping
                them client-side would not stop the server from applying them.
                :exc:`~pinecone.errors.exceptions.PineconeTimeoutError` is then raised carrying the
                partial
                :class:`~pinecone.models.vectors.responses.UpsertResponse` on its ``response``
                attribute, whose ``failed_items`` are the rows that were never sent. ``None``
                (default) means the ingest is bounded only by the per-batch deadlines.
            timeout: Deadline in seconds for a single *attempt* of a single
                batch — not for the batch, and not for the DataFrame. A batch
                that keeps failing on a retryable code is retried, so its
                wall-clock can exceed this several times over; a batch whose
                attempt runs out of time is *not* retried and fails after one
                attempt, so a larger *timeout* is the fix for a timeout and
                ``max_retries`` is not. ``None`` (default) uses the ``timeout``
                the index was constructed with. See the four timeout layers on
                :class:`~pinecone.grpc.GrpcIndex` and :doc:`/guides/retries`.

        Returns:
            :class:`~pinecone.models.vectors.responses.UpsertResponse`
            with the total count of vectors upserted across all batches.

        Raises:
            :exc:`RuntimeError`: If ``pandas`` is not installed. It is not an SDK
                dependency; install it yourself with ``pip install pandas``.
            :exc:`PineconeValueError`: If *df* is not a ``pandas.DataFrame`` or
                *batch_size* is not a positive integer.
            :exc:`~pinecone.errors.exceptions.PineconeTimeoutError`: If a batch exceeds *timeout* on
            the server,
                or if *total_timeout* expires before every batch is submitted. In
                the latter case the exception carries the partial
                :class:`~pinecone.models.vectors.responses.UpsertResponse` on its ``response``
                attribute.

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

            Give each batch a longer server-side deadline for large or slow
            ingests, and check what failed rather than waiting for a raise:

            .. code-block:: python

                response = idx.upsert_from_dataframe(
                    df,
                    batch_size=200,
                    timeout=120.0,
                    on_error="collect",
                )
                if response.failed_item_count:
                    idx.upsert(vectors=response.failed_items, batch_size=200)

        .. seealso::
           :meth:`upsert` — the same batching without the pandas dependency.
           :doc:`/guides/bulk-ingest` — choosing ``batch_size``,
           ``max_concurrency``, and ``total_timeout``.

        .. versionchanged:: 10.0.0
           Partial failures are aggregated into the response rather than
           raised, matching :meth:`upsert` with ``batch_size`` and the REST
           client. The old raise discarded the partial count, so no caller
           could tell what had landed. Pass ``on_error="raise"`` to keep the
           previous behavior. See
           :doc:`/migration/v10-grpc-partial-failures`.
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
            raise PineconeValueError(
                f"df must be a pandas DataFrame, got {type(df).__name__}. Build one with "
                "columns ['id', 'values'] and optionally ['sparse_values', 'metadata'], "
                "e.g. pd.DataFrame([{'id': 'v1', 'values': [0.1, 0.2]}])."
            )

        validate_batch_size(batch_size)

        resolved_on_error = _resolve_on_error(on_error)
        resolved_concurrency = (
            DEFAULT_MAX_CONCURRENCY if max_concurrency is None else max_concurrency
        )
        require_in_range("max_concurrency", resolved_concurrency, 1, 64)

        records: builtins.list[dict[str, Any]] = extract_records(df)

        # Validate before submitting anything, so a malformed row cannot leave
        # part of the frame ingested. VectorFactory would otherwise do this
        # inside a worker thread, after earlier batches had already landed.
        for row, record in enumerate(records):
            validate_vector_dict(record, row=row)

        def _upsert_batch(batch: builtins.list[dict[str, Any]]) -> dict[str, Any]:
            return self._channel.upsert(batch, namespace or None, timeout_s=timeout)

        batch_result = bulk_execute_sync(
            items=records,
            operation=_upsert_batch,
            batch_size=batch_size,
            max_concurrency=resolved_concurrency,
            show_progress=show_progress,
            desc="Upserting",
            host=self._limiter_host,
            total_timeout=total_timeout,
        )

        response = _upsert_response_from(batch_result)

        if batch_result.timed_out:
            message = (
                f"total_timeout of {total_timeout}s expired after "
                f"{response.upserted_count} of {batch_result.total_item_count} vectors were "
                f"upserted; retry the remainder with response.failed_items"
            )
            if resolved_on_error == "raise":
                raise PineconeTimeoutError(message, response=response)
            logger.warning(message)
            return response

        if batch_result.errors:
            if resolved_on_error == "raise":
                # All batches have settled by the time bulk_execute_sync returns,
                # so nothing is left running server-side when this propagates.
                error = min(batch_result.errors, key=lambda err: err.batch_index).error
                if isinstance(error, PineconeTimeoutError):
                    raise PineconeTimeoutError(
                        f"{_as_sentence(str(error))} {self._timeout_guidance(timeout)}",
                        response=response,
                    ) from error
                error.response = response  # type: ignore[attr-defined]
                raise error
            if on_error is None:
                _warn_grpc_partial_failure_once(response)

        return response

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
        """Send one upsert request without waiting for it.

        A narrower :meth:`upsert`: it sends exactly one request, so there is no
        ``batch_size`` and none of the batching arguments that go with it. To
        overlap several requests, issue several of these and collect the
        futures.

        Args:
            vectors: The vectors to upsert, in any of the forms
                :meth:`upsert` accepts.
            namespace (str): Target namespace. Defaults to the default
                (empty-string) namespace.
            timeout (float | None): Per-attempt deadline in seconds for the
                request itself, unrelated to the deadline you later pass to
                :meth:`PineconeFuture.result() <pinecone.grpc.future.PineconeFuture.result>`.
                ``None`` uses the index-level
                default.

        Returns:
            :class:`~pinecone.grpc.future.PineconeFuture` resolving to an
            :class:`~pinecone.models.vectors.responses.UpsertResponse`.

        Examples:

            .. code-block:: python

                future = idx.upsert_async(
                    vectors=[("article-101", [0.012, -0.087, 0.153])],
                    namespace="articles-en",
                )
                print(future.result().upserted_count)
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
        """Send one query without waiting for it, as :meth:`query` otherwise would.

        Takes the same arguments as :meth:`query`; only the return type
        differs. Reach for it to have several queries in flight at once — over
        different namespaces, or with different filters.

        Returns:
            :class:`~pinecone.grpc.future.PineconeFuture` resolving to a
            :class:`~pinecone.models.vectors.responses.QueryResponse`.

        Examples:

            .. code-block:: python

                future = idx.query_async(
                    vector=[0.012, -0.087, 0.153],
                    top_k=5,
                    namespace="articles-en",
                )
                for match in future.result().matches:
                    print(match.id, match.score)
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
        """Send one fetch without waiting for it, as :meth:`fetch` otherwise would.

        Takes the same arguments as :meth:`fetch`; only the return type
        differs. Reach for it to fetch from several namespaces at once.

        Returns:
            :class:`~pinecone.grpc.future.PineconeFuture` resolving to a
            :class:`~pinecone.models.vectors.responses.FetchResponse`.

        Examples:

            .. code-block:: python

                future = idx.fetch_async(
                    ids=["article-101", "article-102"],
                    namespace="articles-en",
                )
                for vid, vec in future.result().vectors.items():
                    print(vid, len(vec.values))
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
        """Send one delete without waiting for it, as :meth:`delete` otherwise would.

        Takes the same arguments as :meth:`delete`; only the return type
        differs. Note that the delete is already on its way when this returns:
        dropping the future does not call it back, and
        :meth:`PineconeFuture.cancel() <pinecone.grpc.future.PineconeFuture.cancel>` only helps
        before a worker thread picks
        it up.

        Returns:
            :class:`~pinecone.grpc.future.PineconeFuture` resolving to ``None``
            once the delete has been accepted. Collect it even though there is
            no payload — that is where a failure surfaces.

        Examples:

            .. code-block:: python

                future = idx.delete_async(
                    ids=["article-101", "article-102"],
                    namespace="articles-en",
                )
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
        """Send one update without waiting for it, as :meth:`update` otherwise would.

        Takes the same arguments as :meth:`update`; only the return type
        differs.

        Returns:
            :class:`~pinecone.grpc.future.PineconeFuture` resolving to an
            :class:`~pinecone.models.vectors.responses.UpdateResponse`.

        Examples:

            .. code-block:: python

                future = idx.update_async(
                    id="article-101",
                    values=[0.012, -0.087, 0.153],
                    namespace="articles-en",
                )
                future.result()
        """
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

    def query_namespaces_async(
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
    ) -> PineconeFuture[QueryNamespacesResults]:
        """Start a :meth:`query_namespaces` fan-out without waiting for it.

        Takes the same arguments as :meth:`query_namespaces`; only the return
        type differs. The fan-out across namespaces already happens on its own
        thread pool, so this is worth it only to overlap the whole fan-out with
        other work.

        Returns:
            :class:`~pinecone.grpc.future.PineconeFuture` resolving to a
            :class:`~pinecone.models.vectors.query_aggregator.QueryNamespacesResults`.

        Examples:

            .. code-block:: python

                future = idx.query_namespaces_async(
                    vector=[0.012, -0.087, 0.153],
                    namespaces=["articles-en", "articles-fr", "articles-de"],
                    metric="cosine",
                    top_k=10,
                )
                for match in future.result(timeout=30.0).matches:
                    print(match.id, match.score)
        """
        return PineconeFuture(
            self._executor.submit(
                self.query_namespaces,
                vector=vector,
                namespaces=namespaces,
                metric=metric,
                top_k=top_k,
                filter=filter,
                include_values=include_values,
                include_metadata=include_metadata,
                sparse_vector=sparse_vector,
                scan_factor=scan_factor,
                max_candidates=max_candidates,
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

        Embeddings are generated server-side from the fields you provide, so
        each record carries source data (e.g. text) rather than precomputed
        vector values. Like :meth:`search`, this call travels over REST even on
        a ``GrpcIndex``, because the gRPC API has no records operations.

        Args:
            records: List of record dicts. Each must contain an ``_id`` or
                ``id`` field. Additional fields are passed through for
                server-side embedding.
            namespace (str): Target namespace (required). Unlike :meth:`upsert`,
                namespace has no default because the records API requires an
                explicit namespace (must be non-empty).
            timeout (float | None): Per-request deadline in seconds. ``None``
                uses the ``timeout`` the index was constructed with.

        Returns:
            :class:`~pinecone.models.vectors.responses.UpsertRecordsResponse` whose ``record_count``
            is how many records the client sent, counted locally — not a server confirmation that
            each one embedded.

        Raises:
            :exc:`PineconeValueError`: If namespace is not a string or is empty/whitespace,
                records is empty, or a record is missing an identifier field.

        Examples:

            .. code-block:: python

                idx = pc.index(name="articles-en", grpc=True)
                response = idx.upsert_records(
                    namespace="published",
                    records=[
                        {"_id": "article-101", "text": "Vector DBs enable similarity search."},
                        {"_id": "article-102", "text": "RAG combines search with LLMs."},
                    ],
                )
                print(response.record_count)

        .. seealso::
           :meth:`upsert` — for an index you embed for yourself, and the only
           one of the two with client-side batching.
           :meth:`search` — the matching read path for these records.
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
            f"/records/namespaces/{quote(namespace, safe='')}/upsert",
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

        Use this on an index with integrated inference: you send query text and the server embeds
        it. This call travels over REST even on a ``GrpcIndex``, because the gRPC API has no records
        search — so a ``retry_config`` you passed to :class:`~pinecone.grpc.GrpcIndex` does not
        govern it, and it retries on the REST data plane's own fixed terms (:doc:`/guides/retries`).

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
                configuration with ``model`` (required), ``rank_fields`` (required), and optional
                ``top_n``, ``parameters``, ``query`` keys. Use
                :class:`~pinecone.models.vectors.search.RerankConfig` for IDE autocompletion.
            match_terms (dict[str, Any] | None): Term-matching constraint for
                sparse search. Requires keys ``"strategy"`` (currently only
                ``"all"``) and ``"terms"`` (list of strings).
                Valid only on a text query — combined with ``vector`` or ``id``
                it is rejected — and only on a sparse index whose embedding model
                supports it; the server names the supported model when it
                refuses. ``None`` disables term matching.
            timeout (float | None): Per-request deadline in seconds. ``None``
                uses the ``timeout`` the index was constructed with.
            query (dict[str, Any] | None): Legacy query body containing
                ``top_k`` plus one of ``inputs``, ``vector``, or ``id``. Prefer
                passing these fields directly.

        Returns:
            :class:`~pinecone.models.vectors.search.SearchRecordsResponse` whose ``result.hits`` are
            :class:`~pinecone.models.vectors.search.Hit` objects — read
            ``hit.id``, ``hit.score``, and ``hit.fields`` — and whose ``usage``
            reports what the search, and any rerank, consumed.

        Raises:
            :exc:`PineconeValueError`: If ``namespace`` is not a string, ``top_k < 1``,
                or ``rerank`` is missing required keys.

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

        .. seealso::
           :meth:`query` — for an index you upsert your own vectors into.
           :meth:`Inference.rerank() <pinecone.client.inference.Inference.rerank>` — for reranking
           results that came
           from somewhere other than this index, or reranking without
           searching; the inline ``rerank`` above covers the single-call case.
        """
        if not isinstance(namespace, str):
            raise ValidationError("namespace must be a string")
        if not namespace or not namespace.strip():
            raise ValidationError("namespace must be a non-empty string")
        body = _build_search_records_body(
            method_name="GrpcIndex.search",
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

        logger.info(
            "Searching namespace %r with top_k=%d (via REST)",
            namespace,
            body["query"]["top_k"],
        )
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
        """Alias for :meth:`search`, kept for callers written against the old name.

        Identical arguments, identical behavior — it forwards straight to
        :meth:`search`, which is where the arguments are documented. Prefer
        :meth:`search` in new code.

        Examples:

            .. code-block:: python

                response = idx.search_records(
                    namespace="articles-en",
                    top_k=10,
                    inputs={"text": "benefits of vector databases for search"},
                )
                for hit in response.result.hits:
                    print(hit.id, hit.score)

        .. seealso::
           :meth:`search` — the current name, and the full argument reference.
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

    def list_namespaces_paginated(
        self,
        *,
        prefix: str | None = None,
        limit: int | None = None,
        pagination_token: str | None = None,
        timeout: float | None = None,
    ) -> ListNamespacesResponse:
        """Fetch a single page of namespace descriptions.

        Args:
            prefix (str | None): Return only namespaces whose names start with this
                prefix. Must be ASCII, must not contain the NUL character, and must
                be at most 512 characters. The empty prefix matches every namespace.
            limit (int | None): Maximum number of namespaces to return in this page,
                1-100.
            pagination_token (str | None): Token from a previous response to fetch the next page.
            timeout (float | None): Per-call timeout in seconds.

        Returns:
            :class:`~pinecone.models.namespaces.models.ListNamespacesResponse` with namespace
            descriptions, pagination info, and total count. Each description carries ``size_bytes``.

        Raises:
            :exc:`PineconeValueError`: If *prefix* or *limit* violates the rules
                above. Raised locally, before the request is sent, with the same
                message the REST and asyncio clients raise.

        Examples:
            .. code-block:: python

                page = idx.list_namespaces_paginated(prefix="articles-", limit=50)
                for ns in page.namespaces:
                    print(ns.name, ns.record_count, ns.size_bytes)
                next_token = page.pagination.next if page.pagination else None

        .. seealso::
           :meth:`list_namespaces` — the same walk with the tokens handled for
           you. See :doc:`/guides/pagination`.
        """
        if prefix is not None:
            require_valid_namespace_prefix("prefix", prefix)
        if limit is not None:
            require_valid_namespace_limit("limit", limit)

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

        Yields one :class:`~pinecone.models.namespaces.models.ListNamespacesResponse` per page. The
        generator automatically follows pagination tokens until all pages have been retrieved.

        Args:
            prefix (str | None): Return only namespaces whose names start with this
                prefix. Must be ASCII, must not contain the NUL character, and must
                be at most 512 characters. The empty prefix matches every namespace.
            limit (int | None): Maximum number of namespaces to return per page, 1-100.
            timeout (float | None): Per-call timeout in seconds.

        Yields:
            :class:`~pinecone.models.namespaces.models.ListNamespacesResponse` for each page of
            results. Each
            :class:`~pinecone.models.namespaces.models.NamespaceDescription` carries ``size_bytes``.

        Raises:
            :exc:`PineconeValueError`: If *prefix* or *limit* violates the rules
                above. Raised on the first iteration, before the request is sent.

        Examples:
            .. code-block:: python

                for page in idx.list_namespaces(prefix="articles-"):
                    for ns in page.namespaces:
                        print(ns.name, ns.record_count, ns.size_bytes)

        .. seealso::
           :meth:`list_namespaces_paginated` — one page at a time, when you
           need to persist a token between calls.
           :meth:`describe_namespace` — for a single namespace, though prefer
           this method for more than one.
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
        """Create a named namespace in the index.

        Args:
            name (str): Name for the new namespace. Must be ASCII, must not
                contain the NUL character, and must be 1-512 characters long.
                ``__default__`` is reserved and cannot be created: it names the
                namespace requests address when they omit a namespace, so it
                always exists.
            schema (dict[str, Any] | None): Optional metadata-index configuration,
                ``{"fields": {<field>: {"filterable": True}}}``. Omitting it does
                not mean "index everything": the namespace inherits the index's
                own metadata-index configuration, so an index that restricts which
                fields are indexed passes that restriction on. Supply *schema* to
                override the inherited configuration for this namespace, indexing
                exactly the fields listed. ``filterable`` is required on each field
                and must be ``True`` — to leave a field unindexed, omit it from
                ``fields``.
            timeout (float | None): Per-call timeout in seconds.

        Returns:
            :class:`~pinecone.models.namespaces.models.NamespaceDescription` with the namespace
            name, record count, schema, indexed fields, and ``size_bytes``.

        Raises:
            :exc:`PineconeValueError`: If *name* violates the rules above, or
                *schema* is malformed. Raised locally, before the request is
                sent, with the same message the REST and asyncio clients raise.

        Examples:
            .. code-block:: python

                ns = idx.create_namespace(name="articles-en")
                print(ns.name, ns.record_count, ns.size_bytes)

            Restrict which metadata fields this namespace indexes, overriding
            what it would otherwise inherit from the index:

            .. code-block:: python

                ns = idx.create_namespace(
                    name="articles-fr",
                    schema={"fields": {"topic": {"filterable": True}}},
                )

        .. seealso::
           :meth:`describe_namespace` — read back the schema and indexed fields
           the namespace ended up with.
        """
        require_creatable_namespace_name("name", name)
        if schema is not None:
            require_valid_namespace_schema("schema", schema)

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

        This operation is rate limited per index, independently of the other
        namespace operations. Prefer :meth:`list_namespaces` when describing more
        than one namespace: it returns the same information for every namespace
        in a single request and is not subject to that limit.

        Args:
            name (str): Name of the namespace to describe. Must be ASCII, must not
                contain the NUL character, and must be 1-512 characters long.
                ``__default__`` is accepted and describes the namespace requests
                address when they omit one.
            timeout (float | None): Per-call timeout in seconds.

        Returns:
            :class:`~pinecone.models.namespaces.models.NamespaceDescription` with the namespace
            name, record count, schema, indexed fields, and ``size_bytes``.

        Raises:
            :exc:`PineconeValueError`: If *name* violates the rules above.
                Raised locally, before the request is sent, with the same
                message the REST and asyncio clients raise.
            :exc:`TypeError`: If unexpected keyword arguments are passed.

        Examples:
            .. code-block:: python

                ns = idx.describe_namespace(name="articles-en")
                print(ns.name, ns.record_count, ns.size_bytes)

        .. seealso::
           :meth:`list_namespaces` — the same fields for every namespace in one
           request, and not subject to this method's rate limit.
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
            name (str): Name of the namespace to delete. Must be ASCII, must not
                contain the NUL character, and must be 1-512 characters long.
            timeout (float | None): Per-call timeout in seconds.

        Returns:
            None — a successful delete returns no payload.

        Raises:
            :exc:`PineconeValueError`: If *name* violates the rules above.
                Raised locally, before the request is sent, with the same
                message the REST and asyncio clients raise.
            :exc:`TypeError`: If unexpected keyword arguments are passed.

        Examples:
            .. code-block:: python

                idx.delete_namespace(name="articles-en")

        .. seealso::
           :meth:`delete` with ``delete_all=True`` — empties a namespace but
           keeps the namespace itself, and its schema.
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

        logger.info("Deleting namespace %r via gRPC", effective)
        self._channel.delete_namespace(effective, timeout_s=timeout)

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
        """Start a bulk import operation from an external data source.

        Initiates an asynchronous bulk import of vectors from cloud storage
        into the index. The import runs server-side; use :meth:`describe_import`
        to poll for progress and completion.

        .. note::
           The import URI must point to a directory of Parquet files in cloud
           storage. Each Parquet file must follow the Pinecone-required schema.
           See
           `Pinecone import docs <https://docs.pinecone.io/guides/data/understanding-imports>`_
           for the required Parquet schema and supported storage formats.

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
            :class:`~pinecone.models.imports.model.StartImportResponse` with the ID of the created
            import operation.

        Raises:
            :exc:`PineconeValueError`: If ``error_mode`` is supplied but not
                ``"continue"`` or ``"abort"``.
            :exc:`ApiError`: If ``uri`` is empty or longer than the server
                accepts, uses an unsupported scheme, is an ``s3://`` URI on an
                index not hosted on AWS, or names an S3 directory bucket, which
                imports do not support.

        Examples:
            The call returns as soon as the import is accepted, so poll
            :meth:`describe_import` for the outcome:

            .. code-block:: python

                import time

                response = idx.start_import(uri="s3://my-bucket/vectors/")
                import_op = idx.describe_import(response.id)
                while import_op.status not in ("Completed", "Failed", "Cancelled"):
                    time.sleep(10)
                    import_op = idx.describe_import(response.id)
                print(import_op.status, import_op.records_imported)

            Skip unreadable records rather than failing the whole import:

            .. code-block:: python

                response = idx.start_import(
                    uri="s3://my-bucket/vectors/",
                    error_mode="continue",
                )

        .. seealso::
           - :meth:`upsert` — for upserting vectors directly in small
             batches (single request per call).
           - :meth:`upsert_records` — for indexes with integrated inference
             (text in, server-side embedding).
           - :meth:`upsert_from_dataframe` — for loading vectors from a
             pandas DataFrame with automatic batching.
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
            id: Import operation ID. Integers are converted to strings silently.

        Returns:
            :class:`~pinecone.models.imports.model.ImportModel` with the import operation details.

        Raises:
            :exc:`PineconeValueError`: If the ID is empty or exceeds 1000 characters.

        Examples:
            .. code-block:: python

                import_op = idx.describe_import("import-123")
                print(import_op.status, import_op.percent_complete)

        .. seealso::
           :meth:`list_imports` — every import on this index, without knowing
           an ID.
        """
        str_id = self._validate_import_id(id)
        logger.info("Describing import %s", str_id)
        response = self._http.get(f"/bulk/imports/{quote(str_id, safe='')}")
        return self._imports_adapter.to_import_model(response.content)

    def cancel_import(self, id: str | int) -> None:
        """Cancel a running bulk import operation by ID.

        Args:
            id (str | int): ID of the import to cancel, as returned by
                :meth:`start_import`. Integers are converted to strings silently.

        Returns:
            None — a successful cancellation returns no payload.

        Raises:
            :exc:`PineconeValueError`: If the ID is empty or exceeds 1000 characters.

        Examples:
            .. code-block:: python

                idx.cancel_import("import-123")

        .. seealso::
           :meth:`describe_import` — poll it afterwards to confirm the import
           reached ``"Cancelled"``.
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
        """List bulk import operations, automatically following pagination.

        Yields individual :class:`~pinecone.models.imports.model.ImportModel` objects, fetching
        additional pages transparently until all results have been returned. Prefer
        :meth:`list_imports_paginated` to control pagination yourself.

        Args:
            limit (int | None): Maximum number of imports per page. Omit to let
                the server choose the page size.
            pagination_token (str | None): Token to resume pagination
                from a previous call.

        Yields:
            :class:`~pinecone.models.imports.model.ImportModel` for each import operation.

        Examples:
            .. code-block:: python

                for imp in idx.list_imports():
                    print(imp.id, imp.status)

        .. seealso::
           :meth:`list_imports_paginated` — one page at a time, when you need
           to persist a token between calls. See :doc:`/guides/pagination`.
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
        """Fetch a single page of bulk import operations.

        Returns an :class:`~pinecone.models.imports.list.ImportList` for one page. The caller is
        responsible for managing the pagination token. Prefer :meth:`list_imports` to have
        pagination handled automatically.

        Args:
            limit (int | None): Maximum number of imports to return in this page.
            pagination_token (str | None): Token from a previous response to
                fetch the next page.

        Returns:
            :class:`~pinecone.models.imports.list.ImportList` for the requested page, iterable over
            its
            :class:`~pinecone.models.imports.model.ImportModel` entries. Its ``pagination.next``
            field holds the token for the next page, or ``None`` once there are no more.

        Examples:
            .. code-block:: python

                page = idx.list_imports_paginated(limit=10)
                for imp in page:
                    print(imp.id, imp.status)
                next_token = page.pagination.next if page.pagination else None

        .. seealso::
           :meth:`list_imports` — the same walk with the tokens handled for
           you. See :doc:`/guides/pagination`.
        """
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if pagination_token is not None:
            params["paginationToken"] = pagination_token

        response = self._http.get("/bulk/imports", params=params)
        return self._imports_adapter.to_import_list(response.content)

    def close(self) -> None:
        """Close the connection to the index and release background resources.

        Waits for any in-flight ``*_async`` submissions to finish, then closes
        the network connection. Call this when you are done issuing requests
        through this client and are not using it as a context manager.

        Examples:
            .. code-block:: python

                idx = pc.index(name="articles-en", grpc=True)
                idx.upsert(vectors=all_vectors, namespace="published")
                idx.close()

        .. seealso::
           :meth:`__enter__` — using the client as a context manager closes it
           for you, including on the way out of an exception.
        """
        self._executor.shutdown(wait=True)
        self._http.close()
        if hasattr(self._channel, "close"):
            self._channel.close()

    def __enter__(self) -> GrpcIndex:
        """Enter a context manager block, returning this client unchanged.

        Examples:
            .. code-block:: python

                with pc.index(name="articles-en", grpc=True) as idx:
                    idx.upsert(vectors=all_vectors, namespace="published")
        """
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit the context manager block, calling :meth:`close`."""
        self.close()


# Legacy capitalisation alias (BCG-141).
GRPCIndex = GrpcIndex

# Legacy name (renamed from PineconeGrpcFuture in the rewrite — BCG-143).
PineconeGrpcFuture = PineconeFuture

from pinecone.grpc.pinecone_grpc import PineconeGRPC  # noqa: E402

__all__ = ["GRPCIndex", "GrpcIndex", "PineconeGRPC", "PineconeGrpcFuture"]
