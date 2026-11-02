"""Synchronous gRPC data plane client for a Pinecone index."""

from __future__ import annotations

import builtins
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
    check_api_key_is_header_safe,
    resolve_grpc_scheme,
    validate_proxy_url,
)
from pinecone._internal.constants import DATA_PLANE_API_VERSION, DEFAULT_MAX_CONCURRENCY
from pinecone._internal.data_plane_helpers import (
    _build_search_records_body,
    _is_plaintext_safe_host,
    _validate_host,
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
from pinecone._internal.vector_factory import VectorFactory, validate_vector_dict
from pinecone.errors.exceptions import (
    PineconeTimeoutError,
    PineconeTypeError,
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


_warned_about_plaintext_grpc = False


def _endpoint_hostname(host: str) -> str:
    """Extract the hostname from a bare ``host[:port]``, IPv6 brackets included."""
    bare = host.split("/", 1)[0]
    if bare.startswith("["):
        return bare[1:].split("]", 1)[0]
    if bare.count(":") > 1:
        return bare
    return bare.split(":", 1)[0]


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

    Reach it as ``pc.index(name="article-search", grpc=True)``, which resolves the
    host for you, or construct it directly when you already know the host.

    It offers the vector, namespace and import methods of
    :class:`~pinecone.index.Index` (a document index needs
    :attr:`Index.documents <pinecone.Index.documents>` on the REST client) and
    is the one to reach for when throughput on a long ingest matters; on
    everything else :class:`~pinecone.index.Index` is the better default,
    because gRPC has no asyncio twin and needs a compiled extension. Three
    differences are visible in the code you write: the ``*_async`` methods here
    return a :class:`~pinecone.grpc.future.PineconeFuture` rather than
    something you ``await``; ``retry_config.retryable_status_codes`` has no
    effect, since this transport retries gRPC status codes rather than HTTP
    ones; and :meth:`upsert_records` and :meth:`search` still travel over REST,
    because the gRPC API has no records operations. See
    `Using the gRPC client <https://sdk.pinecone.io/python/guides/grpc.html>`__.

    Args:
        host (str): The index-specific data plane host, e.g.
            ``"article-search-abc123.svc.pinecone.io"``.
        api_key (str | None): Pinecone API key. Falls back to ``PINECONE_API_KEY`` env var.
        api_version (str): API version string. Defaults to the current data plane version.
        source_tag (str | None): Tag appended to the User-Agent string for request attribution.
        secure (bool): Whether the channel is given TLS material: system root
            certificates for gRPC, and certificate verification for the REST calls
            this client also makes. Defaults to ``True``. It supplies the default for
            ``grpc_scheme``, and ``grpc_scheme`` decides whether the wire is
            encrypted.
        grpc_scheme (str | None): URL scheme used to dial the data plane,
            ``"http"`` or ``"https"``. Set it when the data plane is reached over
            something other than public TLS (a plaintext gateway, an egress proxy, a
            private endpoint, or a local simulator) rather than leaving the SDK to
            assume one. ``None`` (default) reads the ``PINECONE_GRPC_SCHEME`` env var,
            then falls back to ``https`` when ``secure`` is ``True`` and ``http``
            when it is ``False``. ``"https"`` requires ``secure=True``, since an
            https endpoint cannot connect without the TLS material ``secure=False``
            withholds. ``"http"`` with ``secure=True`` is a plaintext channel: the
            scheme, not the TLS material, decides what goes on the wire. A resolved
            ``http`` scheme against a host outside loopback and the RFC 1918
            private ranges warns once per process, because the API key and every
            payload then cross the network unencrypted.
        timeout (float): Deadline in seconds for a **single attempt** of a request,
            not for the call as a whole. Defaults to ``20.0``. A per-call
            ``timeout=`` does not replace it: the channel keeps this one too, so the
            shorter of the two governs.
        connect_timeout (float): Connection timeout in seconds. Defaults to ``1.0``.
        retry_config (RetryConfig | None): Retry policy for transient gRPC errors.
            Accepts the same :class:`~pinecone._internal.config.RetryConfig` REST
            uses. ``None`` (default) uses the gRPC defaults (``max_retries=5``,
            ``backoff_factor=0.1``, ``max_wait=60.0``), which differ from REST's, so
            a ``retry_config`` you leave unset on :class:`~pinecone.Pinecone` does
            not carry over here. Its ``retryable_status_codes`` field is **ignored
            on this transport**: it carries HTTP statuses, and the codes retried
            here are gRPC ones. See
            `Retries and resilience <https://sdk.pinecone.io/python/guides/retries.html>`__.
        proxy_url (str | None): HTTP proxy URL, e.g.
            ``"http://proxy.internal:3128"``. gRPC traffic is tunnelled through it
            with HTTP CONNECT. Credentials in the URL, e.g.
            ``"http://user:pass@proxy.internal:3128"``, are percent-decoded and sent
            as ``Proxy-Authorization: Basic ...``. Only ``http://`` proxies are
            supported: the CONNECT request always goes to the proxy in plaintext,
            so any other scheme (``https://``, ``socks5://``, ...) or a URL without
            a scheme raises :exc:`PineconeValueError`.
        on_throttle (Callable[[str], None] | None): Called with the bare host name
            each time a request fails with a retryable gRPC status, whether or not
            it is retried. Use it for observability, e.g. counting throttles per
            host; the SDK's own backoff runs whether or not you set it. It may be
            called from any thread. ``None``
            (default) falls back to ``retry_config.on_throttle``.
        limiter_registry (_AdaptiveLimiterRegistry | None): SDK-internal. Registry
            the bulk paths consult to back off under throttling. Wired by
            :meth:`~pinecone.Pinecone.index`; not intended for user configuration.

    Raises:
        :exc:`PineconeValueError`: If no API key can be resolved, the key is blank or
            contains a control or non-ASCII character (leading and trailing spaces and
            tabs are allowed and sent as given), the host is invalid,
            ``grpc_scheme`` names a scheme other than ``http`` or ``https``, or
            ``grpc_scheme="https"`` is combined with ``secure=False``, or
            ``proxy_url`` is not an ``http://`` URL (gRPC does not support
            ``https://`` or SOCKS proxies; REST does), has no host or an invalid port,
            has surrounding whitespace, or has an unencoded ``#``, ``/`` or ``?`` in
            the username or password.

    Examples:

        .. code-block:: python

            import os

            from pinecone import Pinecone

            pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
            idx = pc.index(name="article-search", grpc=True)

        Construct it directly when you already know the host. A data plane fronted
        by a plaintext gateway or served by a local simulator is dialled over
        ``http`` by saying so:

        .. code-block:: python

            from pinecone.grpc import GrpcIndex

            idx = GrpcIndex(
                host="http://127.0.0.1:5085",
                api_key=os.environ["PINECONE_API_KEY"],
                grpc_scheme="http",
            )

    Note:
        **Four timeout layers apply to every gRPC call**, and only the first three bound a
        single request:

        1. **Connect**: ``connect_timeout``.
        2. **Per attempt**: ``timeout``, or a per-call ``timeout=``. This is a deadline on
           *one attempt*, not on the call. Both apply when a call passes its own, so the
           shorter of the two is what fires.
        3. **Retry budget**: ``retry_config.max_retries`` attempts after the first, with
           backoff between them.
        4. **Whole job**: for bulk methods only, ``total_timeout``.

        Layers 2 and 3 compound only across *retryable* failures, and this transport
        retries exactly three gRPC status codes: UNAVAILABLE, RESOURCE_EXHAUSTED, and
        ABORTED. A lower ``max_retries`` shrinks the worst case, where every attempt
        uses nearly its full deadline and then fails with one of those.

        **An expiring deadline is not one of the three.** Layer 2 firing raises
        :exc:`~pinecone.errors.exceptions.PineconeTimeoutError` after a single attempt, so
        ``max_retries`` is not the knob for a timeout. Raise ``timeout=`` to give the
        server longer per attempt (and the index-level ``timeout`` too, if it is the
        lower of the two), or bound a bulk job with ``total_timeout``.

    .. seealso::
       - :class:`~pinecone.index.Index`: the REST client, and the better default
         unless you are ingesting at volume.
       - `Using the gRPC client <https://sdk.pinecone.io/python/guides/grpc.html>`__:
         compares the two transports.
       - `Retries and resilience <https://sdk.pinecone.io/python/guides/retries.html>`__:
         the full retry policy for both.
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
        # gRPC metadata carries edge spaces and tabs, and the server accepts them.
        check_api_key_is_header_safe(resolved_key.strip(" \t") or resolved_key)
        if proxy_url:
            validate_proxy_url(proxy_url, grpc=True)

        # Validate and normalize host
        self._request_timeout = timeout
        self._host = _validate_host(host)
        self._limiter_host = _limiter_host(self._host)
        # A directly-constructed handle has no client behind it to supply one,
        # and without a registry the bulk paths do no adaptive backoff at all.
        # Per-handle state is weaker than per-client but far better than none.
        self._limiter_registry = limiter_registry or _AdaptiveLimiterRegistry()
        self._source_tag = source_tag

        # Built before the channel and executor so an invalid timeout raises
        # before anything that needs closing exists.
        rest_config = PineconeConfig(
            api_key=resolved_key,
            host=self._host,
            timeout=timeout,
            source_tag=source_tag or "",
            ssl_verify=secure,
        )

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

        # REST HTTP client for records operations (legacy Records API).
        # upsert_records and search use REST endpoints with no gRPC equivalent.
        from pinecone._internal.http_client import HTTPClient

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

        Serves indexes created with ``dimension``/``metric`` and legacy integrated
        indexes; on a document index use :attr:`Index.documents
        <pinecone.Index.documents>` on the REST client instead. A vector whose ID
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

        Raises:
            :exc:`PineconeTypeError`: If a vector element is not one of the forms listed
                above.
            :exc:`PineconeValueError`: If a vector element is malformed, *batch_size* is
                not a positive integer, or *max_concurrency* is out of range.
            :exc:`ApiError`: If ``batch_size`` is ``None`` and the request exceeds the
                server's cap on vectors per request or on encoded size. Set a
                ``batch_size`` and retry.

        Examples:
            All three vector forms are interchangeable within one call. The values
            below are truncated to three floats for the page; pass your index's full
            dimension.

            .. code-block:: python

                from pinecone import Vector
                from pinecone.grpc import GrpcIndex

                idx = GrpcIndex(
                    host="article-search-abc123.svc.pinecone.io",
                    api_key="your-api-key",
                )
                response = idx.upsert(
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

                response = idx.upsert(
                    vectors=embeddings,
                    namespace="articles-en",
                    batch_size=100,
                )
                if response.has_errors:
                    idx.upsert(vectors=response.failed_items, namespace="articles-en")

        .. seealso::
           - :meth:`upsert_records`: text in, embedded server-side, for a legacy
             integrated index.
           - :meth:`upsert_from_dataframe`: the same write from a pandas DataFrame,
             batched for you.
           - :meth:`start_import`: millions of vectors from cloud storage, server-side
             and asynchronous.
           - `gRPC partial failures
             <https://sdk.pinecone.io/python/migration/v10-grpc-partial-failures.html>`__:
             what changed for callers who expected a batch failure to raise.
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
        """Query a namespace for the nearest neighbors of a vector you supply.

        Serves indexes created with ``dimension``/``metric`` and legacy integrated
        indexes; on a document index use :attr:`Index.documents
        <pinecone.Index.documents>` on the REST client instead. You supply the query
        vector; nothing is embedded for you. Pass a dense *vector*, a *sparse_vector*,
        both for a hybrid query, or the *id* of a stored vector to use as the query.

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

                response = idx.query(
                    top_k=5,
                    vector=[0.012, -0.087, 0.153],
                    namespace="articles-en",
                )
                for match in response.matches:
                    print(match.id, match.score)

            A ``filter`` narrows the search before ranking, and ``include_metadata``
            returns the fields it selected on:

            .. code-block:: python

                response = idx.query(
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

                results = idx.query_namespaces(
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

                results = idx.query_namespaces(
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

        Serves indexes created with ``dimension``/``metric`` and legacy integrated
        indexes; on a document index use :attr:`Index.documents
        <pinecone.Index.documents>` on the REST client instead. A lookup, not a search:
        nothing is ranked and no score is returned. An ID that is not in the namespace
        is silently absent from the result, so compare the keys you got back against the
        ones you asked for.

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
                response = idx.fetch(ids=wanted, namespace="articles-en")
                for vid, vec in response.vectors.items():
                    print(vid, vec.metadata)
                print("not in this namespace:", set(wanted) - set(response.vectors))

        .. seealso::
           - :meth:`fetch_by_metadata`: when you know the metadata you want rather
             than the IDs.
           - :meth:`query`: when you want the nearest vectors rather than named ones.
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
        """Fetch one page of the vectors whose metadata matches a filter.

        Serves indexes created with ``dimension``/``metric`` and legacy integrated
        indexes; on a document index use :attr:`Index.documents
        <pinecone.Index.documents>` on the REST client instead. A lookup, not a search:
        matches are not ranked and carry no score. One page is returned per call, so
        follow ``pagination.next`` to reach the rest.

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

                response = idx.fetch_by_metadata(
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
        """Delete vectors from a namespace by ID, by filter, or all of them.

        Serves indexes created with ``dimension``/``metric`` and legacy integrated
        indexes; on a document index use :attr:`Index.documents
        <pinecone.Index.documents>` on the REST client instead. Pass exactly one
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

                idx.delete(
                    ids=["article-101", "article-102"],
                    namespace="articles-en",
                )

            By metadata filter, which deletes every vector the filter admits:

            .. code-block:: python

                idx.delete(
                    filter={"status": {"$eq": "retracted"}},
                    namespace="articles-en",
                )

            Emptying a whole namespace cannot be undone:

            .. code-block:: python

                idx.delete(delete_all=True, namespace="articles-staging")

        .. seealso::
           :meth:`delete_namespace`: removes the namespace along with everything in it,
           where ``delete_all=True`` empties it and leaves it in place.
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
        """Patch one vector by ID, or patch metadata across a filter.

        Serves indexes created with ``dimension``/``metric`` and legacy integrated
        indexes; on a document index use :attr:`Index.documents
        <pinecone.Index.documents>` on the REST client instead. A partial update: fields
        you do not mention keep the values they had, so ``set_metadata={"year": 2021}``
        leaves every other metadata key in place. Pass exactly one selector, *id* or
        *filter*; a by-filter update is metadata-only, since *values* and
        *sparse_values* belong to one record. The write applies asynchronously, so a
        read straight afterwards can still see the old value.

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

        .. seealso::
           :meth:`upsert`: replaces a whole vector rather than patching it, and
           creates it if the ID is new.
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
        """Fetch one page of vector IDs, holding the pagination token yourself.

        Serves indexes created with ``dimension``/``metric`` and legacy integrated indexes;
        on a document index use :attr:`Index.documents <pinecone.Index.documents>` on the
        REST client instead. Returns IDs only, without values or metadata. :meth:`list`
        walks the pages for you; use this one when you need to keep the token between calls.

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

                response = idx.list_paginated(
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
        """List vector IDs in a namespace, a page at a time.

        Serves indexes created with ``dimension``/``metric`` and legacy integrated indexes;
        on a document index use :attr:`Index.documents <pinecone.Index.documents>` on the
        REST client instead. Returns IDs only, without values or metadata. Yields one page
        per iteration and follows the pagination tokens itself. Nothing is requested until
        you iterate, so a bad *prefix* or *limit* is not reported until then either.

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

                for page in idx.list(prefix="article-2024#", namespace="articles-en"):
                    ids = [item.id for item in page.vectors]
                    fetched = idx.fetch(ids=ids, namespace="articles-en")
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

                stats = idx.describe_index_stats()
                print(stats.dimension, stats.index_fullness)
                for name, summary in stats.namespaces.items():
                    print(name, summary.vector_count)
        .. seealso::
           :meth:`list_namespaces`: per-namespace record counts alongside each
           namespace's schema and ``size_bytes``.
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
        """Upsert vectors from a pandas DataFrame, in batches.

        Serves indexes created with ``dimension``/``metric`` and legacy integrated
        indexes; on a document index use :attr:`Index.documents
        <pinecone.Index.documents>` on the REST client instead. Each row becomes one
        vector, and every row is validated before the first batch is sent, so a
        malformed row cannot leave part of the frame ingested. A batch that fails is
        reported on the response rather than raised unless you pass
        ``on_error="raise"``. ``pandas`` is not an SDK dependency; install it
        yourself.

        Args:
            df: A ``pandas.DataFrame`` with ``id`` and ``values`` columns and, optionally,
                ``sparse_values`` and ``metadata`` columns. A missing cell (``None`` or
                ``NaN``) in an optional column leaves that field off the vector.
            namespace (str): Target namespace, e.g. ``"articles-en"``. Defaults to the
                index's default namespace.
            batch_size (int): Rows per upsert request, e.g. ``100``. Defaults to ``500``.
                Lower it for wide vectors or large metadata.
            show_progress (bool): When ``True`` (default) and ``tqdm`` is installed,
                display a progress bar that advances as batches complete.
            timeout (float | None): Deadline in seconds for one attempt of one batch,
                e.g. ``120.0``. A batch that keeps failing on a retryable code is
                retried, so its wall-clock time can exceed this, but an attempt that runs
                out of time is not retried: raise *timeout*, not ``max_retries``, to fix
                a timeout. ``None`` (default) uses the ``timeout`` the index was
                constructed with. See the timeout layers on
                :class:`~pinecone.grpc.GrpcIndex` and
                `Retries and resilience <https://sdk.pinecone.io/python/guides/retries.html>`__.
            max_concurrency (int | None): Most batch requests this call keeps in flight
                at once, e.g. ``16``. ``None`` (default) uses ``8``. A process-wide limit
                per host also applies and holds concurrency lower when the host pushes
                back; see
                `Bulk ingest <https://sdk.pinecone.io/python/guides/bulk-ingest.html>`__.
            total_timeout (float | None): Deadline in seconds for the whole ingest, e.g.
                ``600.0``. On expiry no further batches are submitted, batches already in
                flight are awaited (dropping them client-side would not stop the server
                from applying them), and unsent rows are reported in ``failed_items``.
                ``None`` (default) means no deadline. Zero or a negative value raises
                :exc:`~pinecone.errors.exceptions.PineconeValueError`.
            on_error (str | None): What to do when some batches fail. ``"collect"``
                returns the response with ``errors`` and ``failed_items`` filled in.
                This holds when every batch fails too: nothing raises, so check
                ``response.has_errors``. ``"raise"`` raises the lowest-indexed batch
                failure, or :exc:`~pinecone.errors.exceptions.PineconeTimeoutError` when
                *total_timeout* expired, once every batch has settled, with the partial
                response on the exception's ``response`` attribute. ``None`` (default)
                behaves as ``"collect"`` and also warns once per process on a partial
                failure, because this method raised before 10.0; pass ``"collect"`` to
                silence the warning.

        Returns:
            :class:`~pinecone.models.vectors.responses.UpsertResponse` with
            ``upserted_count`` totalled across the batches that landed, plus
            ``has_errors``, ``errors`` and ``failed_items`` for the ones that did not.
            ``failed_items`` can be passed straight back to :meth:`upsert`.

        Raises:
            :exc:`RuntimeError`: If ``pandas`` is not installed.
            :exc:`PineconeValueError`: If *df* is not a ``pandas.DataFrame`` or lacks an
                ``id`` or ``values`` column, a row is malformed, *batch_size* is not a
                positive integer, *max_concurrency* is out of range, or *on_error* is not
                ``"collect"`` or ``"raise"``.
            :exc:`~pinecone.errors.exceptions.PineconeTimeoutError`: If
                ``on_error="raise"`` and a batch timed out or *total_timeout* expired.

        Examples:
            A ``metadata`` column is optional; where it is present its dict lands on the
            vector as written. The values below are truncated to three floats for the
            page; pass your index's full dimension.

            .. code-block:: python

                import pandas as pd
                from pinecone.grpc import GrpcIndex

                idx = GrpcIndex(
                    host="article-search-abc123.svc.pinecone.io",
                    api_key="your-api-key",
                )
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
                if response.has_errors:
                    idx.upsert(vectors=response.failed_items, namespace="articles-en")

        .. seealso::
           - :meth:`upsert`: the same batched write from a list of vectors, with no
             ``pandas`` dependency.
           - :meth:`start_import`: millions of vectors from cloud storage, server-side
             and asynchronous.

        .. versionchanged:: 10.0
           Partial failures are aggregated into the response rather than raised,
           matching :meth:`upsert` with ``batch_size`` and the REST client. Pass
           ``on_error="raise"`` to keep the previous behavior. See
           `gRPC partial failures <https://sdk.pinecone.io/python/migration/v10-grpc-partial-failures.html>`__.
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
            namespace (str): Target namespace, e.g. ``"articles-en"``. Defaults to the
                index's default namespace.
            timeout (float | None): Timeout in seconds for one attempt of the request,
                e.g. ``30.0``. It does not bound how long
                :meth:`PineconeFuture.result() <pinecone.grpc.future.PineconeFuture.result>`
                waits. ``None`` (default) uses the client-level default.

        Returns:
            :class:`~pinecone.grpc.future.PineconeFuture` resolving to an
            :class:`~pinecone.models.vectors.responses.UpsertResponse`. Its ``result()``
            raises whatever :meth:`upsert` would have raised.

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
        differs. Reach for it to have several queries in flight at once, over
        different namespaces or with different filters.

        Returns:
            :class:`~pinecone.grpc.future.PineconeFuture` resolving to a
            :class:`~pinecone.models.vectors.responses.QueryResponse`. Its ``result()``
            raises whatever :meth:`query` would have raised.

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
            :class:`~pinecone.models.vectors.responses.FetchResponse`. Its ``result()``
            raises whatever :meth:`fetch` would have raised.

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
        differs. The delete is sent as soon as a worker thread picks it up.
        Dropping the future does not stop it, and
        :meth:`PineconeFuture.cancel() <pinecone.grpc.future.PineconeFuture.cancel>`
        succeeds only before that.

        Returns:
            :class:`~pinecone.grpc.future.PineconeFuture` resolving to ``None``
            once the server accepts the delete. Collect it anyway: its ``result()``
            raises whatever :meth:`delete` would have raised.

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
            :class:`~pinecone.models.vectors.responses.UpdateResponse`. Its ``result()``
            raises whatever :meth:`update` would have raised.

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
        type differs. :meth:`query_namespaces` already queries the namespaces
        concurrently, so this is worth it only to overlap the whole fan-out with
        other work.

        Returns:
            :class:`~pinecone.grpc.future.PineconeFuture` resolving to a
            :class:`~pinecone.models.vectors.query_aggregator.QueryNamespacesResults`.
            Its ``result()`` raises whatever :meth:`query_namespaces` would have raised.

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
        """Upsert text records into a legacy integrated index (legacy Records API).

        Serves legacy integrated indexes, created with
        :meth:`~pinecone.client.indexes.Indexes.create_for_model`: the server embeds
        each record's text with the index's model. On a document index use the REST
        client's :attr:`Index.documents <pinecone.index.Index.documents>` instead.
        Upserts apply asynchronously, so a record may not be searchable immediately.

        ``GrpcIndex`` sends this call over REST, because the gRPC API has no records
        operations.

        Args:
            records (list[dict[str, Any]]): Records to write, each with an ``_id``
                (or ``id``) string plus the fields to store, e.g.
                ``{"_id": "article-101", "text": "Vector databases for search."}``.
                The index embeds the field its embed configuration maps and stores
                the rest with the record.
            namespace (str): Target namespace, e.g. ``"articles-en"``. Required: the legacy
                Records API has no implicit default namespace, so pass ``"__default__"`` to
                write to it.
            timeout (float | None): Per-request timeout in seconds, e.g. ``10.0``.
                ``None`` uses the timeout the index was constructed with.

        Returns:
            :class:`~pinecone.models.vectors.responses.UpsertRecordsResponse` whose
            ``record_count`` is the number of records the client sent.

        Raises:
            :exc:`PineconeValueError`: If *namespace* is not a non-empty string,
                *records* is empty, or a record has no ``_id``/``id`` field. Raised
                before any request is made.

        Examples:
            .. code-block:: python

                idx = pc.index(name="article-search", grpc=True)
                response = idx.upsert_records(
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
        """Search records by text, vector, or ID, optionally reranking (legacy Records API).

        Give exactly one query source. Text in *inputs* needs a legacy integrated
        index, created with
        :meth:`~pinecone.client.indexes.Indexes.create_for_model`, whose model
        embeds it server-side. A *vector* or *id* query also works on an index
        created with ``dimension``/``metric``, though :meth:`query` is the usual
        call there. On a document index use the REST client's
        :attr:`Index.documents <pinecone.index.Index.documents>` instead.

        ``GrpcIndex`` sends this call over REST, because the gRPC API has no records
        search, so it retries on the REST data plane's fixed terms rather than the
        ``retry_config`` passed to :class:`~pinecone.grpc.GrpcIndex`; see
        `Retries and resilience <https://sdk.pinecone.io/python/guides/retries.html>`__.

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
                ``None`` uses the timeout the index was constructed with.
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

                idx = pc.index(name="article-search", grpc=True)
                response = idx.search(
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

                response = idx.search(
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
        """Alias for :meth:`search` (legacy Records API), kept for callers written against 9.x.

        Every argument, return value and error is :meth:`search`'s. Call that one in
        new code.
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
            timeout (float | None): Per-call timeout in seconds.

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

                response = idx.list_namespaces_paginated(prefix="movies-", limit=10)
                for ns in response.namespaces:
                    print(ns.name, ns.record_count, ns.size_bytes)
                next_token = response.pagination.next if response.pagination else None

        .. seealso::
           - :meth:`list_namespaces`: the same listing with the token handled for
             you.
           - `Pagination <https://sdk.pinecone.io/python/guides/pagination.html>`__:
             how the SDK pages generally.
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
            timeout (float | None): Per-call timeout in seconds.

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

                for page in idx.list_namespaces(prefix="movies-"):
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
            timeout (float | None): Per-call timeout in seconds.

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
           - :meth:`describe_namespace`: the same description for a namespace that
             already exists.
           - :meth:`list_namespaces`: the namespaces the index already has.
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
            timeout (float | None): Per-call timeout in seconds.

        Returns:
            :class:`~pinecone.models.namespaces.models.NamespaceDescription` with the
            namespace ``name``, ``record_count``, ``schema``, ``indexed_fields`` and
            ``size_bytes``. ``size_bytes`` is approximate.

        Raises:
            :exc:`PineconeValueError`: If *name* violates the rules above. Raised
                before any request is sent.
            :exc:`TypeError`: If unexpected keyword arguments are passed.
            :exc:`NotFoundError`: no namespace of that name exists on the index.
            :exc:`ApiError`: with ``status_code`` 429 when this operation's
                per-index limit was exceeded. Use :meth:`list_namespaces` to
                describe many namespaces.

        Examples:
            .. code-block:: python

                ns = idx.describe_namespace(name="movies-en")
                print(ns.name, ns.record_count, ns.size_bytes)

            On an index created with ``dimension``/``metric``, the namespace that
            unnamespaced requests address answers to ``__default__``:

            .. code-block:: python

                ns = idx.describe_namespace(name="__default__")
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

                idx.delete_namespace(name="movies-deprecated")

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

        logger.info("Deleting namespace %r via gRPC", effective)
        self._channel.delete_namespace(effective, timeout_s=timeout)

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

    def start_import(
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
            The call returns as soon as the import is accepted, so poll
            :meth:`describe_import` for the outcome:

            .. code-block:: python

                import time

                response = idx.start_import(uri="s3://acme-exports/articles/")
                import_op = idx.describe_import(response.id)
                while import_op.status not in ("Completed", "Failed", "Cancelled"):
                    time.sleep(10)
                    import_op = idx.describe_import(response.id)
                print(import_op.status, import_op.records_imported)

            Skip unreadable records rather than failing the whole import:

            .. code-block:: python

                response = idx.start_import(
                    uri="s3://acme-exports/articles/",
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
        response = self._http.post("/bulk/imports", json=body)
        return self._imports_adapter.to_start_import_response(response.content)

    def describe_import(self, id: str | int) -> ImportModel:
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

                import_op = idx.describe_import("1")
                print(import_op.status, import_op.percent_complete)

        .. seealso::
           - :meth:`start_import`: starts an import and returns the ``id`` these
             methods take.
           - :meth:`list_imports`: every import on the index, when you do not have an
             ``id``.
        """
        str_id = self._validate_import_id(id)
        logger.info("Describing import %s", str_id)
        response = self._http.get(f"/bulk/imports/{quote(str_id, safe='')}")
        return self._imports_adapter.to_import_model(response.content)

    def cancel_import(self, id: str | int) -> None:
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

                idx.cancel_import("1")

        .. seealso::
           - :meth:`start_import`: starts an import and returns the ``id`` these
             methods take.
           - :meth:`list_imports`: every import on the index, when you do not have an
             ``id``.
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

                for imp in idx.list_imports():
                    print(imp.id, imp.status)

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

                page = idx.list_imports_paginated(limit=10)
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
           :meth:`__enter__`: using the client as a context manager closes it
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
