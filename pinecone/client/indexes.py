"""Indexes namespace: schema-based control-plane operations."""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from pinecone._internal.adapters.backups_adapter import BackupsAdapter
from pinecone._internal.adapters.indexes_adapter import IndexesAdapter
from pinecone._internal.backups_helpers import backup_list_params
from pinecone._internal.index_migration import (
    reject_integrated_spec_create,
    reject_legacy_configure_kwargs,
    reject_legacy_create_kwargs,
)
from pinecone._internal.indexes_helpers import (
    poll_index_until_ready,
    poll_sleep_seconds,
    resolve_enum_value,
    validate_wait_timeout,
)
from pinecone._internal.legacy_index_translation import (
    legacy_pod_scaling,
    legacy_vector_schema,
    spec_to_deployment,
    spec_to_read_capacity,
)
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
from pinecone.models.backups.model import BackupModel
from pinecone.models.enums import Metric, PodType, VectorType
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
            "(dense_vector, sparse_vector, or string with full_text_search, embed, "
            "or sparse_embed). "
            "Metadata-only fields are not declared at create time in the 2026-07 "
            "API; they are indexed automatically at upsert."
        )
    if not fields:
        raise PineconeValueError(
            f"{param}['fields'] cannot be empty: the 2026-07 API requires at least "
            "one searched field (dense_vector, sparse_vector, or string with "
            "full_text_search, embed, or sparse_embed)."
        )


def _require_non_empty_dict(param: str, value: dict[str, Any] | None) -> None:
    if value is not None and not value:
        raise PineconeValueError(f"{param} cannot be an empty dict")


def _reject_legacy_metadata_schema(schema: dict[str, Any] | IndexSchema) -> None:
    """Raise a guided error for a 9.x metadata ``schema=`` value.

    On 9.x, ``create(schema=...)`` declared filterable metadata fields shaped
    ``{"fields": {"<name>": {"filterable": bool}}}``, with no ``"type"`` key. At
    2026-07 the same keyword declares searched fields instead, so silently
    reinterpreting one as the other would create a wrong index rather than
    fail loudly.
    """
    if not isinstance(schema, dict):
        return
    fields = schema.get("fields")
    if not isinstance(fields, dict) or not fields:
        return
    if all(isinstance(f, dict) and "type" not in f for f in fields.values()):
        raise PineconeValueError(
            f"schema={schema!r} looks like a 9.x metadata schema: none of its fields "
            "carry a 'type' key. At 2026-07, schema= declares searched fields instead "
            "(dense_vector, sparse_vector, or string with full_text_search, embed, or "
            "sparse_embed) — metadata "
            "fields are no longer declared at create time; they are indexed "
            "automatically at upsert. If you meant to declare a searched field, add "
            "'type': 'dense_vector' | 'sparse_vector' | 'string' to each field. See "
            "the migration guide: https://sdk.pinecone.io/python/migration/v10-migration.html"
        )


class Indexes:
    """Control-plane operations for Pinecone indexes.

    An index is the container your records live in: its searched fields are
    declared as a schema when you create it, and every query is aimed at one
    index. Reached as ``pc.indexes``; not constructed directly.

    The backup methods here are scoped to a single index.
    :class:`~pinecone.client.backups.Backups` (``pc.backups``) covers the
    project-wide backup listing plus ``delete``, which belong to no one index.

    Examples:

        .. code-block:: python

            from pinecone import Pinecone

            pc = Pinecone(api_key="your-api-key")
            names = [index.name for index in pc.indexes.list()]

    .. seealso::
       :meth:`Pinecone.index(name) <pinecone.Pinecone.index>`, the data-plane
       client for reads and writes against one index.

       `Error handling <https://sdk.pinecone.io/python/guides/error-handling.html>`__,
       the exceptions any method here can raise and how to handle them.

    .. versionchanged:: 10.0
       Graduated to the schema-based API. :meth:`create` takes
       ``schema=``/``deployment=`` instead of ``spec=``/``dimension=``/
       ``metric=``/``vector_type=``; :meth:`configure` nests pod scaling under
       ``deployment=`` and dropped ``embed=``; :meth:`list` returns a
       :class:`~pinecone.models.pagination.Paginator`; the index-scoped
       backup methods graduated from the preview namespace.
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
        """List every index in the project.

        Args:
            limit: Maximum number of indexes to yield. Must be a positive
                integer; ``None`` (the default) yields every index.
            pagination_token: Accepted so this listing shares the paginator
                interface of the other listings. The server issues no
                pagination token for indexes, so leave it ``None``.

        Returns:
            :class:`~pinecone.models.pagination.Paginator` over
            :class:`IndexModel` instances.

        Raises:
            :exc:`PineconeValueError`: If *limit* is zero or negative.

        Examples:
            >>> for index in pc.indexes.list():
            ...     print(index.name, index.status.state)

        .. versionchanged:: 10.0
           Returns a :class:`~pinecone.models.pagination.Paginator` instead of
           an ``IndexList``. Iteration keeps working; replace
           ``pc.indexes.list().names()`` with
           ``[index.name for index in pc.indexes.list()]``.
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

        Caches the index's host, so a later
        :meth:`Pinecone.index(name) <pinecone.Pinecone.index>` call for the
        same name skips its own describe round trip.

        Args:
            name (str): The name of the index to describe.

        Returns:
            :class:`IndexModel` with ``name``, ``host``, ``schema``,
            ``deployment``, ``read_capacity``, ``status``,
            ``deletion_protection``, and ``tags``.

        Raises:
            :exc:`PineconeValueError`: If *name* is empty.
            :exc:`PineconeTypeError`: If *name* is not a ``str``.
            :exc:`NotFoundError`: If the index does not exist.

        Examples:
            The returned host carries the ``https://`` scheme:

            >>> index = pc.indexes.describe("my-index")
            >>> index.host
            'https://my-index-abc123.svc.pinecone.io'
        """
        require_non_empty("name", name)
        logger.info("Describing index %r", name)
        response = self._http.get(f"/indexes/{quote(name, safe='')}")
        model = self._adapter.to_index_model(response.content)
        if model.host is not None:
            self._host_cache[name] = model.host
        logger.debug("Described index %r (host=%s)", name, model.host)
        return model

    def exists(self, name: str) -> bool:
        """Check whether a named index exists.

        Calls :meth:`describe` internally and returns ``False`` instead of
        raising when the index isn't found. Every other error propagates.

        Args:
            name (str): The name of the index to check.

        Returns:
            ``True`` if the index exists, ``False`` otherwise.

        Raises:
            :exc:`PineconeValueError`: If *name* is empty.
            :exc:`PineconeTypeError`: If *name* is not a ``str``.

        Examples:
            >>> pc.indexes.exists("my-index")
            True

        .. versionchanged:: 10.0
           An empty *name* now raises :exc:`PineconeValueError` instead of
           returning ``False``.
        """
        require_non_empty("name", name)
        try:
            self.describe(name)
            return True
        except NotFoundError:
            return False

    def delete(self, name: str, *, timeout: int | None = None) -> None:
        """Delete an index by name.

        Blocks until the index is gone, with no upper time bound unless you
        pass *timeout*.

        Args:
            name (str): The name of the index to delete.
            timeout (int | None): Seconds to wait for the index to disappear.
                Use ``None`` (default) to poll indefinitely until the index
                is gone. Use a positive int to poll with a deadline.
                Use ``-1`` to return immediately without polling.

        Raises:
            :exc:`PineconeValueError`: If *name* is empty or *timeout* is a
                negative value other than ``-1``.
            :exc:`PineconeTypeError`: If *name* is not a ``str``.
            :exc:`NotFoundError`: If the index does not exist.
            :exc:`ForbiddenError`: If deletion protection is enabled; clear
                it with :meth:`configure` first.
            :exc:`PineconeTimeoutError`: If the index still exists after
                *timeout* seconds.

        Examples:
            Delete an index and block until it is gone:

            .. code-block:: python

                pc.indexes.delete("my-index")

            Or bound the wait, so an index still present after a minute
            raises :exc:`PineconeTimeoutError` instead of polling forever:

            .. code-block:: python

                pc.indexes.delete("my-index", timeout=60)

            Passing ``timeout=-1`` returns as soon as the delete request is
            accepted, without polling; the index is still being torn down
            when the call returns.
        """
        require_non_empty("name", name)
        validate_wait_timeout(timeout)
        logger.info("Deleting index %r", name)
        self._http.delete(f"/indexes/{quote(name, safe='')}")
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
            time.sleep(poll_sleep_seconds(_POLL_INTERVAL_SECONDS, timeout, start))

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
        spec: Any = None,
        dimension: int | None = None,
        metric: Metric | str | None = None,
        vector_type: VectorType | str | None = None,
        **legacy_kwargs: Any,
    ) -> IndexModel:
        """Create a new index.

        An index's field layout is declared as a ``schema`` of named, typed
        fields. Every field in the schema must be a searched field:
        ``dense_vector``, ``sparse_vector``, or ``string`` with
        ``full_text_search``, ``embed`` or ``sparse_embed``. A string field
        with ``embed`` or ``sparse_embed`` supports integrated embedding: Pinecone
        embeds its text on write and the query text on search. The result is
        a document index; read and write it through ``index.documents``.
        Metadata-only fields aren't declared here; they're indexed
        automatically the first time they appear on an upserted record. The
        schema can't change once the index exists, and the call blocks until
        the index is ready unless you pass ``timeout=-1``.

        Args:
            schema: The index's field schema. Required unless the
                deprecated ``dimension=`` (with optional ``metric=``/
                ``vector_type=``) is used instead; the two are mutually
                exclusive. A dict with a ``"fields"`` key mapping field
                names to typed configurations::

                    {
                        "fields": {
                            "your_vector": {"type": "dense_vector",
                                            "dimension": 1536, "metric": "cosine"},
                            "body": {"type": "string",
                                     "full_text_search": {"language": "en",
                                                          "stemming": True}},
                        }
                    }

                Also accepts the dict produced by
                :class:`~pinecone.schema_builder.SchemaBuilder` or an
                :class:`~pinecone.models.indexes.schema.IndexSchema`. A
                hybrid index has to declare its ``sparse_vector`` field here;
                see the note after the examples.
                ``full_text_search.language`` accepts a fixed set of language
                codes (or their English names, default ``en``), but
                ``stop_words=True`` is not supported for every language: the
                server's 400 names the unsupported language, by its English
                name rather than the code you sent.
            name: Name for the index, in lowercase alphanumerics and hyphens,
                e.g. ``"movie-recommendations"``.
                The server assigns a name when omitted.
            deployment: Deployment configuration, discriminated on
                ``"deployment_type"``. For a managed index:
                ``{"deployment_type": "managed", "cloud": "aws", "region":
                "us-east-1"}``. For a pod-based index, ``"deployment_type":
                "pod"`` plus ``environment``, ``pod_type``, ``replicas``,
                and ``shards``. Defaults to a managed index on AWS
                ``us-east-1`` when omitted. Mutually exclusive with the
                deprecated ``spec=``.
            read_capacity: Read capacity for a managed or BYOC index, either
                ``{"mode": "OnDemand"}`` or ``{"mode": "Dedicated",
                "dedicated": {"node_type": ..., "scaling": ..., "manual":
                {"replicas": ..., "shards": ...}}}``.
            deletion_protection: ``"enabled"`` to block :meth:`delete` on
                this index until it's set back to ``"disabled"`` (the
                default).
            tags: Key-value tags to attach, e.g. ``{"env": "prod"}``. Pass
                ``None`` (the default) to attach none.
            cmek_id: ID of a customer-managed encryption key to encrypt the
                index with, e.g. ``"key-abc123"``.
            timeout: Seconds to wait for the index to become ready before
                returning. ``None`` (default) waits indefinitely, ``-1``
                returns immediately without waiting, and a positive number
                waits up to that many seconds. Any other negative value
                raises :exc:`PineconeValueError` before a request is sent.
            spec: **Deprecated.** A
                :class:`~pinecone.models.indexes.specs.ServerlessSpec`,
                :class:`~pinecone.models.indexes.specs.PodSpec`,
                :class:`~pinecone.models.indexes.specs.ByocSpec`, or the
                equivalent dict, translated into ``deployment=`` (and
                ``read_capacity=`` when the spec carries one). Mutually
                exclusive with ``deployment=``. Use :meth:`create_for_model`
                for ``IntegratedSpec``.

                .. deprecated:: 10.0
                   Pass ``deployment=`` directly instead.
            dimension: **Deprecated.** Dense vector width for the legacy
                path, translated into a single-field ``schema=``. Required
                when creating a dense index this way.

                .. deprecated:: 10.0
                   Declare a named field in ``schema=`` instead.
            metric: **Deprecated.** Similarity metric for the legacy dense
                path: ``"cosine"`` (default), ``"euclidean"``, or
                ``"dotproduct"``.

                .. deprecated:: 10.0
                   Set ``metric`` inside the ``schema=`` field declaration
                   instead.
            vector_type: **Deprecated.** ``"dense"`` (default) or
                ``"sparse"``, for the legacy path.

                .. deprecated:: 10.0
                   Declare a named ``dense_vector``/``sparse_vector`` field
                   in ``schema=`` instead.

        Returns:
            :class:`IndexModel` describing the created index, ready unless
            ``timeout=-1`` was passed.

        Raises:
            :exc:`PineconeValueError`: If neither ``schema=`` nor
                ``dimension=`` is given, or mutually exclusive arguments
                (``schema=`` with a legacy vector kwarg, or ``deployment=``
                with ``spec=``) are combined, or *timeout* is a negative value
                other than ``-1``.
            :exc:`PineconeTypeError`: If an unsupported legacy keyword (e.g.
                ``pods=``) or ``spec=IntegratedSpec(...)`` is passed.
            :exc:`IndexInitFailedError`: If the index fails to initialize.
            :exc:`PineconeTimeoutError`: If the index isn't ready before
                *timeout* elapses.

        Examples:
            A dense index on the default deployment (managed, AWS
            ``us-east-1``). The call waits for the index to become ready
            before returning:

            >>> index = pc.indexes.create(
            ...     name="movie-recommendations",
            ...     schema={"fields": {"your_vector": {
            ...         "type": "dense_vector", "dimension": 1536, "metric": "cosine"}}},
            ... )
            >>> index.status.ready
            True

            A hybrid index in a region of your choosing. Declare the
            ``sparse_vector`` field now; see the note below:

            >>> index = pc.indexes.create(
            ...     name="support-articles",
            ...     schema={"fields": {
            ...         "your_vector": {"type": "dense_vector",
            ...                         "dimension": 1024, "metric": "cosine"},
            ...         "keywords": {"type": "sparse_vector"},
            ...         "body": {"type": "string",
            ...                  "full_text_search": {"language": "en",
            ...                                       "stemming": True}},
            ...     }},
            ...     deployment={"deployment_type": "managed",
            ...                 "cloud": "aws", "region": "us-west-2"},
            ...     tags={"env": "prod"},
            ... )

            Integrated embedding on a ``string`` field, processed in the US
            for both documents and queries:

            >>> index = pc.indexes.create(
            ...     name="support-articles-us",
            ...     schema={"fields": {"body": {
            ...         "type": "string",
            ...         "embed": {
            ...             "model": "text-embedding-3-large",
            ...             "processing_location": "us",
            ...         },
            ...     }}},
            ... )

        .. note::
           A **hybrid** index must declare its ``sparse_vector`` field
           explicitly. A dense field with ``metric="dotproduct"`` does not
           accept sparse values on its own: the create succeeds, and
           only the sparse upserts are refused later. The field
           cannot be added by ``configure()``, so an index created
           without one has to be recreated. See
           `Migrating to V10 <https://sdk.pinecone.io/python/migration/v10-migration.html>`__.

        .. seealso::
           :meth:`create_for_model` creates a legacy integrated index, which
           requires the legacy Records API. For integrated embedding in a
           document index, declare ``embed`` / ``sparse_embed`` on a
           ``string`` field here instead.

        .. versionchanged:: 10.0
           ``spec=``, ``dimension=``, ``metric=``, and ``vector_type=`` are
           deprecated, keyword-only sugar for the current ``schema=``/
           ``deployment=`` arguments. ``pods=``, ``metadata_config=``,
           ``source_collection=``, ``source_backup_id=``, and
           ``spec=IntegratedSpec(...)`` have no equivalent here; use
           :meth:`create_for_model` for a legacy integrated index, or
           ``embed`` / ``sparse_embed`` on a ``string`` field for integrated
           embedding in a document index.
        """
        validate_wait_timeout(timeout)
        reject_legacy_create_kwargs(legacy_kwargs, name)
        reject_integrated_spec_create(spec, name)

        legacy_vector_kwargs_given = (
            dimension is not None or metric is not None or vector_type is not None
        )

        if schema is not None and legacy_vector_kwargs_given:
            conflicting = [
                kw
                for kw, val in (
                    ("dimension", dimension),
                    ("metric", metric),
                    ("vector_type", vector_type),
                )
                if val is not None
            ]
            raise PineconeValueError(
                "create() got both schema= and "
                f"{', '.join(f'{kw}=' for kw in conflicting)}: schema= is the 2026-07 "
                "field declaration and cannot be combined with the deprecated "
                "dimension=/metric=/vector_type= sugar. Pass one or the other."
            )

        if deployment is not None and spec is not None:
            raise PineconeValueError(
                "create() got both deployment= and spec=: deployment= is the 2026-07 "
                "deployment configuration and cannot be combined with the deprecated "
                "spec= sugar. Pass one or the other."
            )

        if schema is None and not legacy_vector_kwargs_given:
            raise PineconeValueError(
                "schema is required: pass schema={'fields': {'<field-name>': {...}}} "
                "declaring at least one searched field (dense_vector, sparse_vector, "
                "or string with full_text_search, embed, or sparse_embed), or, for "
                "backward compatibility, "
                "the deprecated dimension= keyword argument (optionally with metric= "
                "and vector_type=)."
            )

        resolved_schema: dict[str, Any] | IndexSchema
        if schema is None:
            resolved_schema = legacy_vector_schema(
                dimension=dimension, metric=metric, vector_type=vector_type
            )
        else:
            _reject_legacy_metadata_schema(schema)
            resolved_schema = schema
        _require_fields_dict("schema", resolved_schema)

        if name is not None:
            require_valid_resource_name("name", name)

        resolved_deployment = deployment
        resolved_read_capacity = read_capacity
        if spec is not None:
            resolved_deployment = spec_to_deployment(spec)
            if resolved_read_capacity is None:
                resolved_read_capacity = spec_to_read_capacity(spec)

        _require_non_empty_dict("deployment", resolved_deployment)
        _require_non_empty_dict("read_capacity", resolved_read_capacity)
        if tags is not None and not tags:
            raise PineconeValueError("tags cannot be an empty dict")
        validate_index_tags(tags)
        resolved_dp: str | None = None
        if deletion_protection is not None:
            resolved_dp = resolve_enum_value(deletion_protection)
            require_one_of("deletion_protection", resolved_dp, _DELETION_PROTECTION_VALUES)

        request = CreateIndexRequest(
            schema=resolved_schema,
            name=name,
            deployment=resolved_deployment,
            read_capacity=resolved_read_capacity,
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
        """Create a legacy integrated index, served by the legacy Records API.

        Pinecone embeds text written to the mapped field automatically at
        upsert time and embeds queries at read time using the same model. In
        the returned index, the embedding configuration surfaces as a
        ``semantic_text`` field in ``schema``, named after the ``field_map``
        text entry.

        This creates a legacy integrated index, and legacy integrated indexes
        require the legacy Records API
        (:meth:`~pinecone.index.Index.upsert_records` and
        :meth:`~pinecone.index.Index.search`). For new work, create a
        document index with integrated embedding instead: pass
        :meth:`create` a ``string`` field with ``embed`` or ``sparse_embed``
        and use ``index.documents``, which adds full-text search on the same
        fields and ``score_by``.

        Args:
            name: Name for the index, in lowercase alphanumerics and hyphens,
                e.g. ``"semantic-search"``.
            cloud: Public cloud provider: ``"aws"``, ``"gcp"``, or ``"azure"``.
            region: Cloud region, e.g. ``"us-east-1"``.
            embed: Embedding configuration. A dict (or
                :class:`~pinecone.models.indexes.specs.EmbedConfig` /
                :class:`~pinecone.inference.models.index_embed.IndexEmbed`)
                with required ``model`` and ``field_map`` (e.g.
                ``{"text": "chunk_text"}``) and optional ``metric``,
                ``dimension``, ``read_parameters``, ``write_parameters``.
                The model cannot be changed after creation.
            deletion_protection: ``"enabled"`` to block :meth:`delete` on
                this index until it's set back to ``"disabled"`` (the
                default).
            tags: Key-value tags to attach, e.g. ``{"env": "prod"}``. Pass
                ``None`` (the default) to attach none.
            schema: Filterable metadata fields, e.g.
                ``{"fields": {"genre": {"filterable": True}}}``. A bare field
                map is wrapped in ``{"fields": ...}`` for you.
            read_capacity: Read capacity for the index, in the same shape as
                for :meth:`create`.
            timeout: Seconds to wait for the index to become ready before
                returning. ``None`` (default) waits indefinitely, ``-1``
                returns immediately without waiting, and a positive number
                waits up to that many seconds. Any other negative value
                raises :exc:`PineconeValueError` before a request is sent.

        Returns:
            :class:`IndexModel` describing the created index, ready unless
            ``timeout=-1`` was passed.

        Raises:
            :exc:`PineconeValueError`: If *name*, *cloud*, *region*, *embed*,
                *tags*, *deletion_protection*, or *timeout* (a negative value
                other than ``-1``) fail client-side validation.

        Examples:
            The ``field_map`` text entry names the record field Pinecone
            embeds, and that same name is what the field is called in the
            returned schema (``chunk_text`` below):

            >>> index = pc.indexes.create_for_model(
            ...     name="semantic-search",
            ...     cloud="aws",
            ...     region="us-east-1",
            ...     embed={"model": "multilingual-e5-large",
            ...            "field_map": {"text": "chunk_text"}},
            ... )
            >>> index.schema.fields["chunk_text"].model
            'multilingual-e5-large'

        .. seealso::
           :meth:`create` creates an index you supply the vectors for
           yourself, declaring them as ``dense_vector``/``sparse_vector``
           fields in ``schema=``.
        """
        validate_wait_timeout(timeout)
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

        logger.info("Creating legacy integrated index name=%r", name)
        response = self._http.post("/indexes/create-for-model", json=body)
        model = self._adapter.to_index_model(response.content)
        logger.debug("Created legacy integrated index %r", model.name)

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
        replicas: int | None = None,
        pod_type: PodType | str | None = None,
        serverless_read_capacity: dict[str, Any] | None = None,
        **legacy_kwargs: Any,
    ) -> IndexModel:
        """Configure an existing index.

        Only the fields you provide are updated; omitted parameters are left
        unchanged on the server. Read capacity and pod scaling apply
        asynchronously, so the call returns while the change is still in
        flight.

        Args:
            name: Name of the index to configure.
            deployment: Pod-scaling updates for pod-based indexes:
                ``{"replicas": int, "pod_type": str}`` (either or both). Only
                those two keys are accepted; any other key is rejected by the
                server. ``"deployment_type"`` raises :exc:`PineconeValueError`
                client-side, since deployment type, cloud/region, and
                environment cannot be changed after creation.
            schema: Schema updates. On an index created with
                ``create(schema=...)`` the schema is fixed: nothing in it can
                be changed, including the ``embed``/``sparse_embed``
                parameters of a ``string`` field. On an index created with
                :meth:`create_for_model`, only ``semantic_text`` field
                parameters (``read_parameters``/``write_parameters``) can be
                updated; see the note after the examples.
            read_capacity: Updated read capacity dict, either
                ``{"mode": "OnDemand"}`` or ``{"mode": "Dedicated",
                "dedicated": {...}}``. Applies to managed and BYOC indexes.
            deletion_protection: ``"enabled"`` to block :meth:`delete` on
                this index, ``"disabled"`` to allow it again.
            tags: Tag updates, merged into the index's existing tags. Set a
                value to ``""`` to delete that key; keys you don't mention stay
                unchanged. The server applies its tag limit to the merged
                total, so adding tags to an index that already carries several
                can be rejected. ``{}`` is rejected client-side.
            replicas: **Deprecated.** Legacy pod-scaling replica count,
                translated into ``deployment={"replicas": ...}``. Mutually
                exclusive with *deployment*.
            pod_type: **Deprecated.** Legacy pod type, translated into
                ``deployment={"pod_type": ...}`` alongside *replicas*.
                Mutually exclusive with *deployment*.
            serverless_read_capacity: **Deprecated.** Legacy read-capacity
                keyword for managed indexes, translated straight into
                *read_capacity*. Mutually exclusive with *read_capacity*.

        Returns:
            :class:`IndexModel` reflecting the updated index state. Read
            ``status`` for how far an asynchronous change has got rather than
            assuming it landed.

        Raises:
            :exc:`PineconeValueError`: If *name* is empty, all kwargs are
                ``None``, any dict kwarg is empty, *deployment* includes
                ``deployment_type``, tags/deletion_protection are invalid, or
                *deployment*/*read_capacity* is combined with the deprecated
                keyword argument it translates to.
            :exc:`PineconeTypeError`: If *name* is not a ``str``, or if
                ``embed=`` or ``spec=`` is passed (neither has a translation
                here; the message shows the equivalent current call where one
                exists).
            :exc:`NotFoundError`: If the index does not exist.
            :exc:`ApiError`: Typically status 422 or 400, if *schema* is passed
                for an index created with ``create(schema=...)`` (even
                unchanged fields are refused), or if *deployment* has a key
                other than ``replicas`` or ``pod_type``. The message comes
                from the server and can read as a raw deserialization error
                rather than naming the field.

        Examples:
            Scale a pod-based index. Pod scaling is applied in the
            background, so read ``index.status`` on the returned model to
            see how far the change has got rather than assuming it landed:

            >>> index = pc.indexes.configure(
            ...     "legacy-recommender", deployment={"replicas": 4, "pod_type": "p1.x2"}
            ... )

            Tag updates merge into the tags the index already carries. Given
            an index tagged ``{"env": "staging", "team": "search"}``, the
            call below sets ``env``, deletes ``team`` (an empty value
            removes a key), and leaves every other tag as it was, so
            ``index.tags`` comes back ``{"env": "prod"}``:

            >>> index = pc.indexes.configure("my-index", tags={"env": "prod", "team": ""})

        .. note::
           Only ``semantic_text`` field parameters (``read_parameters``/
           ``write_parameters``) can be updated through ``schema=``; other
           field types can't be added, removed, or retyped after creation.
           Since :meth:`create` cannot declare a ``semantic_text`` field
           directly, this only applies to indexes created with
           :meth:`create_for_model`. On an index created with
           ``create(schema=...)`` the schema is fixed, and passing ``schema=``
           is rejected.

        .. versionchanged:: 10.0
           Before/after:

           .. code-block:: python

               # 9.x
               pc.indexes.configure("my-index", replicas=4, pod_type="p1.x2")
               # 10.x
               pc.indexes.configure("my-index",
                                    deployment={"replicas": 4, "pod_type": "p1.x2"})

           ``embed=`` is gone entirely, along with the convert-to-integrated
           flow it drove; ``replicas=``/``pod_type=``/
           ``serverless_read_capacity=`` remain as deprecated keyword-only
           sugar for ``deployment=``/``read_capacity=``; and the method
           returns the updated :class:`IndexModel` instead of ``None``.

        .. deprecated:: 10.0
           ``replicas=``, ``pod_type=``, and ``serverless_read_capacity=`` are
           translated into ``deployment=``/``read_capacity=`` rather than sent
           as-is, and cannot be combined with the argument they translate to;
           passing both raises :exc:`PineconeValueError`. New code should use
           ``deployment=``/``read_capacity=`` directly.
        """
        require_non_empty("name", name)
        reject_legacy_configure_kwargs(legacy_kwargs)

        if deployment is not None and (replicas is not None or pod_type is not None):
            raise PineconeValueError(
                "configure() cannot accept deployment= together with the deprecated "
                "replicas=/pod_type= pod-scaling keywords: pass pod scaling only one "
                "way, either deployment={'replicas': ..., 'pod_type': ...} or "
                "replicas=/pod_type=, not both."
            )
        if read_capacity is not None and serverless_read_capacity is not None:
            raise PineconeValueError(
                "configure() cannot accept read_capacity= together with the "
                "deprecated serverless_read_capacity= keyword: pass read capacity "
                "only one way, not both."
            )

        if replicas is not None or pod_type is not None:
            deployment = legacy_pod_scaling(replicas=replicas, pod_type=pod_type)
        if serverless_read_capacity is not None:
            read_capacity = serverless_read_capacity

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
            f"/indexes/{quote(name, safe='')}",
            content=self._adapter.to_configure_request(request),
            headers=_JSON_HEADERS,
        )
        model = self._adapter.to_index_model(response.content)
        logger.debug("Configured index %r", name)
        return model

    def create_backup(
        self,
        index_name: str,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> BackupModel:
        """Create a backup of an index.

        Index-scoped shortcut for
        :meth:`Backups.create <pinecone.client.backups.Backups.create>`, which does the same
        thing; the difference is that this one takes the index name
        positionally.

        Args:
            index_name: Name of the index to back up.
            name: Your own name for the backup, e.g. ``"nightly-20240115"``.
                Omit it and the server assigns one.
            description: Free-text note stored with the backup, e.g.
                ``"pre-reindex snapshot"``.

        Returns:
            :class:`~pinecone.models.backups.model.BackupModel` describing the
            new backup. ``status`` is typically ``"Initializing"`` right after
            creation; poll :meth:`describe_backup` until it reads ``"Ready"``
            before restoring from it.

        Raises:
            :exc:`PineconeValueError`: If *index_name* is empty.
            :exc:`PineconeTypeError`: If *index_name* is not a ``str``.
            :exc:`NotFoundError`: If the index does not exist.

        Examples:
            >>> backup = pc.indexes.create_backup("my-index", name="nightly-20240115")
            >>> backup.backup_id
            'bk-abc123'

        .. versionadded:: 10.0
           Graduated from ``pc.preview.indexes.create_backup``, now returning
           the single top-level
           :class:`~pinecone.models.backups.model.BackupModel`.
        """
        require_non_empty("index_name", index_name)

        body: dict[str, str] = {}
        if name is not None:
            body["name"] = name
        if description is not None:
            body["description"] = description

        logger.info("Creating backup for index %r", index_name)
        response = self._http.post(f"/indexes/{quote(index_name, safe='')}/backups", json=body)
        result = BackupsAdapter.to_backup(response.content)
        logger.debug("Created backup %r", result.backup_id)
        return result

    def list_backups(
        self,
        index_name: str,
        *,
        limit: int | None = None,
        pagination_token: str | None = None,
        include_deleted: bool | None = None,
    ) -> Paginator[BackupModel]:
        """List the backups of one index, following pages as you iterate.

        Args:
            index_name: Name of the index whose backups to list.
            limit: Maximum number of backups to yield across all pages. Must
                be a positive integer. ``None`` yields all backups. It also
                sets the requested page size, but only on a request that
                carries no pagination token: every later page is sized by the
                token, which already encodes it.
            pagination_token: Token from an earlier call, to resume where that
                call stopped. *limit* still caps the total yield, but it is
                not sent alongside a token (see above). See
                `Pagination <https://sdk.pinecone.io/python/guides/pagination.html>`__.
            include_deleted: When ``True``, include backups of every index
                that has ever used *index_name*, deleted ones included; those
                backups carry a non-``None``
                :attr:`~pinecone.models.backups.model.BackupModel.source_index_deleted_at`.
                When ``None`` (the default) the parameter is omitted entirely
                and the server's default (``false``) applies.

        Returns:
            :class:`~pinecone.models.pagination.Paginator` over
            :class:`~pinecone.models.backups.model.BackupModel` instances.

        Raises:
            :exc:`PineconeValueError`: If *index_name* is empty or *limit* is
                zero or negative.
            :exc:`PineconeTypeError`: If *index_name* is not a ``str``.
            :exc:`NotFoundError`: If *index_name* does not resolve to an
                active index, which is not the same as the name being
                unknown; see the note after the examples.

        Examples:
            >>> for backup in pc.indexes.list_backups("my-index"):
            ...     print(backup.backup_id, backup.status)
            bk-abc123 Ready

            Once every index that used a name has been deleted, that name's
            backups come back only with ``include_deleted=True``; without
            it this raises :exc:`NotFoundError`. They are the ones carrying
            a ``source_index_deleted_at``:

            >>> backups = pc.indexes.list_backups(
            ...     "legacy-catalog", include_deleted=True
            ... ).to_list()
            >>> [b.backup_id for b in backups if b.source_index_deleted_at]
            ['bk-old111']

        .. important::
           :exc:`NotFoundError` here does not necessarily mean *index_name*
           was never used. With *include_deleted* omitted or ``False``,
           *index_name* must resolve to an **active** index: if every index
           that used the name has since been deleted, this raises
           :exc:`NotFoundError` rather than returning an empty list. Retry
           with ``include_deleted=True`` to get those backups back; a
           :exc:`NotFoundError` there means the name was never used in this
           project.

        .. seealso::
           :meth:`Backups.list <pinecone.client.backups.Backups.list>` is the
           project-wide listing, where the index name is an optional filter. It hands back
           one page for you to drive the token yourself, rather than a
           paginator that follows the pages for you.

        .. versionadded:: 10.0
           Graduated from ``pc.preview.indexes.list_backups``, and gained
           *include_deleted*.
        """
        require_non_empty("index_name", index_name)
        if limit is not None:
            require_positive("limit", limit)

        def fetch_page(token: str | None) -> Page[BackupModel]:
            params = backup_list_params(
                limit=limit,
                pagination_token=token,
                include_deleted=include_deleted,
            )
            logger.info("Listing backups for index %r", index_name)
            response = self._http.get(
                f"/indexes/{quote(index_name, safe='')}/backups", params=params
            )
            result = BackupsAdapter.to_backup_list(response.content)
            next_token = result.pagination.next if result.pagination is not None else None
            return Page(items=list(result), pagination_token=next_token)

        return Paginator(fetch_page=fetch_page, initial_token=pagination_token, limit=limit)

    def describe_backup(self, backup_id: str) -> BackupModel:
        """Describe a backup by its ID.

        Backups are identified independently of any index, so despite living
        on ``indexes`` this takes a backup ID rather than an index name.

        Args:
            backup_id: Identifier of the backup to describe, as returned in
                ``backup_id`` by :meth:`create_backup`.

        Returns:
            :class:`~pinecone.models.backups.model.BackupModel` with the
            current state of the backup.

        Raises:
            :exc:`PineconeValueError`: If *backup_id* is empty.
            :exc:`PineconeTypeError`: If *backup_id* is not a ``str``.
            :exc:`NotFoundError`: If the backup does not exist.

        Examples:
            >>> backup = pc.indexes.describe_backup("bk-abc123")
            >>> backup.status
            'Ready'

        .. seealso::
           :meth:`Backups.describe <pinecone.client.backups.Backups.describe>` is the
           same lookup reached through ``pc.backups``, which takes ``backup_id`` as
           a keyword argument rather than positionally.

        .. versionadded:: 10.0
           Graduated from ``pc.preview.indexes.describe_backup``.
        """
        require_non_empty("backup_id", backup_id)
        logger.info("Describing backup %r", backup_id)
        response = self._http.get(f"/backups/{quote(backup_id, safe='')}")
        return BackupsAdapter.to_backup(response.content)

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
