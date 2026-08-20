"""Indexes namespace — schema-based control-plane operations (2026-07 API)."""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from pinecone._internal.adapters.indexes_adapter import IndexesAdapter
from pinecone._internal.index_migration import (
    reject_legacy_configure_kwargs,
    reject_legacy_create_kwargs,
)
from pinecone._internal.indexes_helpers import poll_index_until_ready, resolve_enum_value
from pinecone._internal.validation import (
    require_non_empty,
    require_one_of,
    require_positive,
    require_valid_resource_name,
    validate_index_tags,
)
from pinecone.errors.exceptions import (
    NotFoundError,
    PineconeTimeoutError,
    PineconeValueError,
)
from pinecone.models.indexes.index import IndexModel
from pinecone.models.indexes.requests import ConfigureIndexRequest, CreateIndexRequest
from pinecone.models.indexes.schema import IndexSchema
from pinecone.models.pagination import Page, Paginator

if TYPE_CHECKING:
    from pinecone._internal.http_client import HTTPClient

logger = logging.getLogger(__name__)

_POLL_INTERVAL_SECONDS = 5
_JSON_HEADERS = {"Content-Type": "application/json"}
_DELETION_PROTECTION_VALUES = ("disabled", "enabled")


def _require_fields_dict(param: str, schema: dict[str, Any] | IndexSchema) -> None:
    """Require a ``{"fields": {...}}``-shaped schema with at least one field."""
    if isinstance(schema, IndexSchema):
        return
    if not schema:
        raise PineconeValueError(f"{param} cannot be an empty dict")
    fields = schema.get("fields")
    if not isinstance(fields, dict) or "fields" not in schema:
        raise PineconeValueError(
            f"{param} must be a dict of the form {{'fields': {{'<field-name>': "
            "{...}}}} mapping field names to searched-field configurations "
            "(dense_vector, sparse_vector, or string with full_text_search). "
            "Metadata-only fields are not declared at create time in the 2026-07 "
            "API; they are indexed automatically at upsert."
        )
    if not fields:
        raise PineconeValueError(
            f"{param}['fields'] cannot be empty: the 2026-07 API requires at least "
            "one searched field (dense_vector, sparse_vector, or string with "
            "full_text_search)."
        )


def _require_non_empty_dict(param: str, value: dict[str, Any] | None) -> None:
    if value is not None and not value:
        raise PineconeValueError(f"{param} cannot be an empty dict")


class Indexes:
    """Control-plane operations for Pinecone indexes (2026-07 API).

    Provides ``list``, ``describe``, ``exists``, ``create``,
    ``create_for_model``, ``delete``, and ``configure`` methods.

    .. versionchanged:: 10.0
       Graduated to the 2026-07 schema-based API. ``create()`` takes
       ``schema=``/``deployment=`` instead of ``spec=``/``dimension=``/
       ``metric=``/``vector_type=``; ``configure()`` nests pod scaling under
       ``deployment=`` and removed ``embed=``; ``list()`` returns a
       :class:`~pinecone.models.pagination.Paginator`.

    .. seealso::
       Use :meth:`Pinecone.index(name) <pinecone.Pinecone.index>` to get a
       data-plane client for vector operations on a specific index.

    Args:
        http (HTTPClient): HTTP client for making API requests.

    Examples:

        .. code-block:: python

            from pinecone import Pinecone

            pc = Pinecone(api_key="your-api-key")
            names = [idx.name for idx in pc.indexes.list()]
    """

    def __init__(self, http: HTTPClient, host_cache: dict[str, str] | None = None) -> None:
        self._http = http
        self._adapter = IndexesAdapter()
        self._host_cache: dict[str, str] = host_cache if host_cache is not None else {}

    def __repr__(self) -> str:
        """Return developer-friendly representation."""
        return "Indexes()"

    def list(
        self,
        *,
        limit: int | None = None,
        pagination_token: str | None = None,
    ) -> Paginator[IndexModel]:
        """List all indexes in the project.

        The 2026-07 server returns all indexes in a single page. The returned
        :class:`~pinecone.models.pagination.Paginator` always yields exactly
        one page and then terminates; the paginator interface is used for
        consistency with other list methods and forward compatibility.

        .. versionchanged:: 10.0
           Returns a :class:`~pinecone.models.pagination.Paginator` instead of
           an ``IndexList``. Iteration keeps working; replace
           ``pc.indexes.list().names()`` with
           ``[idx.name for idx in pc.indexes.list()]``.

        Args:
            limit: Maximum number of items to yield. Must be a positive
                integer. ``None`` yields all items.
            pagination_token: Token to resume pagination from a previous call.
                ``None`` starts from the beginning.

        Returns:
            :class:`~pinecone.models.pagination.Paginator` over
            :class:`IndexModel` instances.

        Raises:
            :exc:`PineconeValueError`: If *limit* is zero or negative.
            :exc:`ApiError`: If the API returns an error response.

        Examples:
            >>> for index in pc.indexes.list():  # doctest: +SKIP
            ...     print(index.name)
        """
        if limit is not None:
            require_positive("limit", limit)

        def fetch_page(token: str | None) -> Page[IndexModel]:
            logger.info("Listing indexes")
            response = self._http.get("/indexes")
            items = list(self._adapter.to_index_list(response.content))
            logger.debug("Listed %d indexes", len(items))
            return Page(items=items, pagination_token=None)

        return Paginator(fetch_page=fetch_page, initial_token=pagination_token, limit=limit)

    def describe(self, name: str) -> IndexModel:
        """Get detailed information about a named index.

        After a successful call the host URL is cached internally for
        later data-plane client construction.

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
            >>> desc = pc.indexes.describe("my-index")
            >>> desc.host  # doctest: +SKIP
            'https://my-index.svc.pinecone.io'
        """
        require_non_empty("name", name)
        logger.info("Describing index %r", name)
        response = self._http.get(f"/indexes/{name}")
        model = self._adapter.to_index_model(response.content)
        if model.host is not None:
            self._host_cache[name] = model.host
        logger.debug("Described index %r (host=%s)", name, model.host)
        return model

    def exists(self, name: str) -> bool:
        """Check whether a named index exists.

        Uses describe internally; returns ``True`` on success and
        ``False`` when a 404 is returned.

        .. versionchanged:: 10.0
           An empty *name* now raises :exc:`PineconeValueError` instead of
           returning ``False``.

        Args:
            name (str): The name of the index to check.

        Returns:
            True if the index exists, False otherwise.

        Raises:
            :exc:`PineconeValueError`: If *name* is empty.
            :exc:`ApiError`: If the API returns an error other than 404.

        Examples:
            >>> pc.indexes.exists("my-index")  # doctest: +SKIP
            True
        """
        require_non_empty("name", name)
        try:
            self.describe(name)
            return True
        except NotFoundError:
            return False

    def delete(self, name: str, *, timeout: int | None = None) -> None:
        """Delete an index by name.

        After sending the delete request, removes the cached host URL
        for the index. By default, polls every 5 seconds until the index
        disappears with no upper time bound.

        Args:
            name (str): The name of the index to delete.
            timeout (int | None): Seconds to wait for the index to disappear.
                Use ``None`` (default) to poll indefinitely until the index
                is gone. Use a positive int to poll with a deadline.
                Use ``-1`` to return immediately without polling.

        Raises:
            :exc:`PineconeValueError`: If *name* is empty.
            :exc:`NotFoundError`: If the index does not exist.
            :exc:`ForbiddenError`: If deletion protection is enabled on the index.
            :exc:`PineconeTimeoutError`: If the index still exists after *timeout* seconds.
            :exc:`ApiError`: If the API returns another error response.

        Examples:
            .. code-block:: python

                pc.indexes.delete("my-index")

                # Wait up to 60 seconds for deletion to complete
                pc.indexes.delete("my-index", timeout=60)
        """
        require_non_empty("name", name)
        logger.info("Deleting index %r", name)
        self._http.delete(f"/indexes/{name}")
        self._host_cache.pop(name, None)
        logger.debug("Deleted index %r", name)

        if timeout == -1:
            return

        start = time.monotonic()
        while True:
            try:
                self.describe(name)
            except NotFoundError:
                self._host_cache.pop(name, None)
                return
            if timeout is not None:
                elapsed = time.monotonic() - start
                if elapsed >= timeout:
                    raise PineconeTimeoutError(f"Index '{name}' still exists after {timeout}s")
            time.sleep(_POLL_INTERVAL_SECONDS)

    def create(
        self,
        *,
        schema: dict[str, Any] | IndexSchema | None = None,
        name: str | None = None,
        deployment: dict[str, Any] | None = None,
        read_capacity: dict[str, Any] | None = None,
        deletion_protection: str | None = None,
        tags: Mapping[str, str] | None = None,
        cmek_id: str | None = None,
        timeout: int | None = None,
        **legacy_kwargs: Any,
    ) -> IndexModel:
        """Create a new Pinecone index (2026-07 schema-based API).

        The index shape is declared as a ``schema`` of named, typed fields.
        Every field must be searched — ``dense_vector``, ``sparse_vector``,
        or ``string`` with ``full_text_search``. Metadata-only fields are not
        declared at create time; they are indexed automatically at upsert.
        The schema cannot be modified after creation.

        .. versionchanged:: 10.0
           Replaces the 2025-10 signature. Before/after:

           .. code-block:: python

               # 9.x
               pc.indexes.create(name="movies", dimension=1536,
                                 spec=ServerlessSpec(cloud="aws", region="us-east-1"))
               # 10.x
               pc.indexes.create(
                   name="movies",
                   schema={"fields": {"embedding": {
                       "type": "dense_vector", "dimension": 1536, "metric": "cosine"}}},
                   deployment={"deployment_type": "managed",
                               "cloud": "aws", "region": "us-east-1"},
               )

           Legacy keyword arguments (``spec``, ``dimension``, ``metric``,
           ``vector_type``, ``pods``, ``metadata_config``,
           ``source_collection``, ``source_backup_id``) raise a
           :exc:`~pinecone.errors.exceptions.PineconeTypeError` showing the
           equivalent 2026-07 call.

        Args:
            schema: Required index schema. A dict with a ``"fields"`` key
                mapping field names to typed configurations, e.g.::

                    {
                        "fields": {
                            "embedding": {"type": "dense_vector",
                                          "dimension": 1536, "metric": "cosine"},
                            "body": {"type": "string",
                                     "full_text_search": {"language": "en"}},
                        }
                    }

                Also accepts the dict produced by
                :class:`~pinecone.schema_builder.SchemaBuilder` or an
                :class:`~pinecone.models.indexes.schema.IndexSchema`. Field
                names must be 1-64 characters and must not begin with ``_``
                or ``$``. Field-type rules (at most one dense and one sparse
                vector field; metadata types rejected) are enforced by the
                server and surfaced verbatim.
            name: Optional name for the index. 1-45 characters matching
                ``^[a-z0-9]([a-z0-9-]*[a-z0-9])?$``. If omitted, the server
                assigns one.
            deployment: Optional deployment configuration dict discriminated
                on ``"deployment_type"`` (``"managed"`` | ``"pod"`` |
                ``"byoc"``). Omitted defaults server-side to managed on AWS
                ``us-east-1``. Pod deployments must include all of
                ``environment``, ``pod_type``, ``replicas``, and ``shards``
                (the server rejects omissions with 422).
            read_capacity: Optional read capacity dict —
                ``{"mode": "OnDemand"}`` or ``{"mode": "Dedicated",
                "dedicated": {"node_type": ..., "scaling": ...,
                "manual": {"replicas": ..., "shards": ...}}}``.
            deletion_protection: ``"enabled"`` or ``"disabled"`` (server
                default ``"disabled"``).
            tags: Optional key-value tags. Keys: 1-80 ASCII alphanumerics,
                ``_`` or ``-``; values: 0-120 printable ASCII characters;
                at most 20 tags.
            cmek_id: Optional customer-managed encryption key ID. Incompatible
                with pod deployments and with ``full_text_search`` fields.
            timeout: Seconds to wait for the index to become ready. ``None``
                (default) polls indefinitely every 5 seconds. A positive int
                polls with a deadline. ``-1`` returns immediately without
                polling.

        Returns:
            :class:`IndexModel` describing the created index (ready, unless
            ``timeout=-1`` was passed).

        Raises:
            :exc:`PineconeValueError`: If inputs fail client-side validation
                (empty or invalid name, malformed or empty schema/deployment/
                read_capacity dicts, invalid tags or deletion_protection).
            :exc:`PineconeTypeError`: If legacy 2025-10 keyword arguments are
                passed; the message shows the equivalent 2026-07 call.
            :exc:`IndexInitFailedError`: If the index fails to initialise.
            :exc:`PineconeTimeoutError`: If the index is not ready before the deadline.
            :exc:`ApiError`: If the API returns an error response; server
                validation messages are surfaced verbatim.

        Examples:
            >>> pc.indexes.create(  # doctest: +SKIP
            ...     name="movie-recommendations",
            ...     schema={"fields": {"embedding": {
            ...         "type": "dense_vector", "dimension": 1536, "metric": "cosine"}}},
            ... )
        """
        reject_legacy_create_kwargs(legacy_kwargs, name)

        if schema is None:
            raise PineconeValueError(
                "schema is required: pass schema={'fields': {'<field-name>': {...}}} "
                "declaring at least one searched field (dense_vector, sparse_vector, "
                "or string with full_text_search)."
            )
        _require_fields_dict("schema", schema)
        if name is not None:
            require_valid_resource_name("name", name)
        _require_non_empty_dict("deployment", deployment)
        _require_non_empty_dict("read_capacity", read_capacity)
        if tags is not None and not tags:
            raise PineconeValueError("tags cannot be an empty dict")
        validate_index_tags(tags)
        resolved_dp: str | None = None
        if deletion_protection is not None:
            resolved_dp = resolve_enum_value(deletion_protection)
            require_one_of("deletion_protection", resolved_dp, _DELETION_PROTECTION_VALUES)

        request = CreateIndexRequest(
            schema=schema,
            name=name,
            deployment=deployment,
            read_capacity=read_capacity,
            deletion_protection=resolved_dp,
            tags=dict(tags) if tags is not None else None,
            cmek_id=cmek_id,
        )

        logger.info("Creating index name=%r", name)
        response = self._http.post(
            "/indexes",
            content=self._adapter.to_create_request(request),
            headers=_JSON_HEADERS,
        )
        model = self._adapter.to_index_model(response.content)
        logger.debug("Created index %r", model.name)

        if timeout != -1:
            model = self._poll_until_ready(model.name, timeout)
        return model

    def create_for_model(
        self,
        *,
        name: str,
        cloud: str,
        region: str,
        embed: Mapping[str, Any] | Any,
        deletion_protection: str | None = None,
        tags: Mapping[str, str] | None = None,
        schema: dict[str, Any] | None = None,
        read_capacity: dict[str, Any] | None = None,
        timeout: int | None = None,
    ) -> IndexModel:
        """Create a serverless index with an integrated embedding model.

        Pinecone embeds text written to the mapped field automatically at
        upsert time and embeds queries at read time using the same model. In
        the returned index, the embedding configuration surfaces as a
        ``semantic_text`` field in ``schema``, named after the ``field_map``
        text entry.

        .. note::
           The 2026-07 wire shape for this operation is the
           ``cloud``/``region``/``embed`` form (apis@5f808858 restored it;
           the backend accepts only this shape). The published build of the
           2026-07 OAS is stale here — see the SPEC-vs-BACKEND question
           linked from the migration guide.

        Args:
            name: Required name for the index (1-45 characters,
                ``^[a-z0-9]([a-z0-9-]*[a-z0-9])?$``).
            cloud: Public cloud provider — ``"aws"``, ``"gcp"``, or ``"azure"``.
            region: Cloud region (e.g. ``"us-east-1"``).
            embed: Embedding configuration. A dict (or
                :class:`~pinecone.models.indexes.specs.EmbedConfig` /
                :class:`~pinecone.inference.models.index_embed.IndexEmbed`)
                with required ``model`` and ``field_map`` (e.g.
                ``{"text": "chunk_text"}``) and optional ``metric``,
                ``dimension``, ``read_parameters``, ``write_parameters``.
                The model cannot be changed after creation.
            deletion_protection: ``"enabled"`` or ``"disabled"``.
            tags: Optional key-value tags (same limits as :meth:`create`).
            schema: Optional metadata schema dict for filterable metadata
                fields, e.g. ``{"fields": {"genre": {"filterable": True}}}``.
                A bare field map is wrapped in ``{"fields": ...}``.
            read_capacity: Optional read capacity dict (see :meth:`create`).
            timeout: Readiness polling — same semantics as :meth:`create`.

        Returns:
            :class:`IndexModel` describing the created index.

        Raises:
            :exc:`PineconeValueError`: If *name*, *cloud*, *region*, or
                *embed* fail client-side validation.
            :exc:`ApiError`: If the API returns an error response.

        Examples:
            >>> pc.indexes.create_for_model(  # doctest: +SKIP
            ...     name="semantic-search",
            ...     cloud="aws",
            ...     region="us-east-1",
            ...     embed={"model": "multilingual-e5-large",
            ...            "field_map": {"text": "chunk_text"}},
            ... )
        """
        require_valid_resource_name("name", name)
        cloud_str = resolve_enum_value(cloud)
        region_str = resolve_enum_value(region)
        require_non_empty("cloud", cloud_str)
        require_non_empty("region", region_str)

        embed_body = self._embed_to_body(embed)
        if tags is not None and not tags:
            raise PineconeValueError("tags cannot be an empty dict")
        validate_index_tags(tags)
        _require_non_empty_dict("read_capacity", read_capacity)

        body: dict[str, Any] = {
            "name": name,
            "cloud": cloud_str,
            "region": region_str,
            "embed": embed_body,
        }
        if deletion_protection is not None:
            resolved_dp = resolve_enum_value(deletion_protection)
            require_one_of("deletion_protection", resolved_dp, _DELETION_PROTECTION_VALUES)
            body["deletion_protection"] = resolved_dp
        if tags is not None:
            body["tags"] = dict(tags)
        if schema is not None:
            if not schema:
                raise PineconeValueError("schema cannot be an empty dict")
            wrapped = schema if list(schema.keys()) == ["fields"] else {"fields": schema}
            body["schema"] = wrapped
        if read_capacity is not None:
            body["read_capacity"] = read_capacity

        logger.info("Creating integrated index name=%r", name)
        response = self._http.post("/indexes/create-for-model", json=body)
        model = self._adapter.to_index_model(response.content)
        logger.debug("Created integrated index %r", model.name)

        if timeout != -1:
            model = self._poll_until_ready(model.name, timeout)
        return model

    def configure(
        self,
        name: str,
        *,
        deployment: dict[str, Any] | None = None,
        schema: dict[str, Any] | IndexSchema | None = None,
        read_capacity: dict[str, Any] | None = None,
        deletion_protection: str | None = None,
        tags: Mapping[str, str] | None = None,
        **legacy_kwargs: Any,
    ) -> IndexModel:
        """Configure an existing index (2026-07 API).

        Only the fields you provide are updated; omitted parameters are left
        unchanged on the server.

        .. versionchanged:: 10.0
           Before/after:

           .. code-block:: python

               # 9.x
               pc.indexes.configure("my-index", replicas=4, pod_type="p1.x2")
               # 10.x
               pc.indexes.configure("my-index",
                                    deployment={"replicas": 4, "pod_type": "p1.x2"})

           ``embed=`` was removed entirely (the 2025-10 convert-to-integrated
           flow no longer exists), ``serverless_read_capacity=`` collapsed
           into the single top-level ``read_capacity=`` (which now applies to
           managed **and** BYOC indexes, not just BYOC), and the method
           returns the updated :class:`IndexModel` instead of ``None``.

        .. note::
           *Schema updates:* the client no longer restricts ``schema=`` to
           ``semantic_text`` fields (decision: shape validation stays local,
           server policy stays on the server). The 2026-07 server accepts
           only ``semantic_text`` parameter updates in a PATCH schema and
           rejects anything else — and since ``semantic_text`` fields cannot
           be created via ``create()`` in 2026-07, this path is effectively
           unreachable except for indexes made by ``create_for_model``.
           Server errors are surfaced verbatim.

        Args:
            name: Name of the index to configure.
            deployment: Pod-scaling updates for pod-based indexes —
                ``{"replicas": int, "pod_type": str}`` (either or both). Must
                not include ``"deployment_type"``: deployment type,
                cloud/region, and environment cannot be changed after
                creation.
            schema: Schema updates. Only ``semantic_text`` field parameters
                (``read_parameters``/``write_parameters``) are updatable
                server-side; see note above.
            read_capacity: Updated read capacity dict —
                ``{"mode": "OnDemand"}`` or ``{"mode": "Dedicated",
                "dedicated": {...}}``. Applies to managed and BYOC indexes.
            deletion_protection: ``"enabled"`` or ``"disabled"``.
            tags: Tag updates, merged with existing tags on the server. Set a
                value to ``""`` to delete that tag key.

        Returns:
            :class:`IndexModel` reflecting the updated index state. Some
            changes (read capacity, pod scaling) apply asynchronously — check
            ``status``.

        Raises:
            :exc:`PineconeValueError`: If *name* is empty, all kwargs are
                ``None``, any dict kwarg is empty, *deployment* includes
                ``deployment_type``, or tags/deletion_protection are invalid.
            :exc:`PineconeTypeError`: If legacy 2025-10 keyword arguments
                (``replicas``, ``pod_type``, ``embed``, ``spec``,
                ``serverless_read_capacity``) are passed; the message shows
                the equivalent 2026-07 call.
            :exc:`NotFoundError`: If the index does not exist.
            :exc:`ApiError`: If the API returns another error response.

        Examples:
            >>> pc.indexes.configure("my-index", deployment={"replicas": 4})  # doctest: +SKIP
            >>> pc.indexes.configure("my-index", tags={"env": "prod"})  # doctest: +SKIP
        """
        require_non_empty("name", name)
        reject_legacy_configure_kwargs(legacy_kwargs, name)

        if (
            deployment is None
            and schema is None
            and read_capacity is None
            and deletion_protection is None
            and tags is None
        ):
            raise PineconeValueError(
                "at least one configuration parameter must be provided: "
                "deployment, schema, read_capacity, deletion_protection, or tags"
            )

        _require_non_empty_dict("deployment", deployment)
        if deployment is not None and "deployment_type" in deployment:
            raise PineconeValueError(
                "configure() deployment must not include 'deployment_type': the "
                "deployment type, cloud/region, and environment cannot be changed "
                "after creation. Pass only {'replicas': ..., 'pod_type': ...}."
            )
        if isinstance(schema, dict) and not schema:
            raise PineconeValueError("schema cannot be an empty dict")
        _require_non_empty_dict("read_capacity", read_capacity)
        if tags is not None and not tags:
            raise PineconeValueError("tags cannot be an empty dict")
        validate_index_tags(tags)
        resolved_dp: str | None = None
        if deletion_protection is not None:
            resolved_dp = resolve_enum_value(deletion_protection)
            require_one_of("deletion_protection", resolved_dp, _DELETION_PROTECTION_VALUES)

        request = ConfigureIndexRequest(
            deployment=deployment,
            schema=schema,
            read_capacity=read_capacity,
            deletion_protection=resolved_dp,
            tags=dict(tags) if tags is not None else None,
        )

        logger.info("Configuring index %r", name)
        response = self._http.patch(
            f"/indexes/{name}",
            content=self._adapter.to_configure_request(request),
            headers=_JSON_HEADERS,
        )
        model = self._adapter.to_index_model(response.content)
        logger.debug("Configured index %r", name)
        return model

    @staticmethod
    def _embed_to_body(embed: Mapping[str, Any] | Any) -> dict[str, Any]:
        """Normalize dict / EmbedConfig / IndexEmbed into the embed wire dict."""
        if isinstance(embed, Mapping):
            raw: dict[str, Any] = dict(embed)
        else:
            raw = {
                "model": getattr(embed, "model", None),
                "field_map": getattr(embed, "field_map", None),
                "metric": getattr(embed, "metric", None),
                "dimension": getattr(embed, "dimension", None),
                "read_parameters": getattr(embed, "read_parameters", None),
                "write_parameters": getattr(embed, "write_parameters", None),
            }
        model = raw.get("model")
        field_map = raw.get("field_map")
        if not model:
            raise PineconeValueError(
                "embed must include a non-empty 'model' (e.g. 'multilingual-e5-large')"
            )
        if not field_map:
            raise PineconeValueError(
                "embed must include a non-empty 'field_map' "
                "(e.g. {'text': 'chunk_text'}) naming the text field to embed"
            )
        body: dict[str, Any] = {
            "model": resolve_enum_value(model),
            "field_map": dict(field_map),
        }
        for key in ("metric", "dimension", "read_parameters", "write_parameters"):
            value = raw.get(key)
            if value:
                body[key] = resolve_enum_value(value) if key == "metric" else value
        return body

    def _poll_until_ready(self, name: str, timeout: int | None) -> IndexModel:
        """Poll describe() until the index is ready or timeout is reached."""
        return poll_index_until_ready(self.describe, name, timeout)
