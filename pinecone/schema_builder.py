"""Build the ``schema`` argument for creating an index.

A schema names the fields an index searches and says how each one is scored.
:class:`SchemaBuilder` assembles that structure a field at a time and hands
back a plain ``{"fields": {...}}`` dict rather than a model, so a key this SDK
version does not yet model still reaches the server unchanged.

What belongs in a create schema is narrower than what an index stores. Declare
the fields you search — a ``dense_vector`` field, a ``sparse_vector`` field,
and ``string`` fields with full-text search enabled — and at least one of
those is required. Everything else is metadata: put those values in the
documents you upsert and they are indexed for filtering automatically, with no
declaration at all. A schema may hold at most one ``dense_vector`` and one
``sparse_vector`` field.

The builder still has methods for the metadata-only declarations
(:meth:`SchemaBuilder.add_boolean_field`,
:meth:`SchemaBuilder.add_float_field`,
:meth:`SchemaBuilder.add_string_list_field`, and
:meth:`SchemaBuilder.add_string_field` without full-text search). The server
rejects all of them on create, and one rejected field fails the whole request.

Two field types have no method at all. Text embedded server-side is a
``semantic_text`` field, which a create schema does not accept — reach it
through :meth:`~pinecone.client.indexes.Indexes.create_for_model` instead.
``integer`` is response-only: describe and list return it for indexes that
pre-date numeric normalisation, but there is no ``integer`` variant to create
and the server answers one with a ``422``. The builder refuses it wherever a
raw ``type`` key could reach a field dict — see
:meth:`SchemaBuilder.add_custom_field` and the ``additional_options``
parameters — so replaying a described schema into a create request fails
locally with an explanation instead of remotely with a status code.

.. seealso::
   :meth:`~pinecone.client.indexes.Indexes.create` — the method that consumes
   the result, and the deprecated ``dimension=``/``metric=`` sugar this
   replaces.
"""

from __future__ import annotations

import copy
from typing import Any

_FTS_LANGUAGES_SHORT = frozenset(
    [
        "ar",
        "da",
        "de",
        "el",
        "en",
        "es",
        "fi",
        "fr",
        "hu",
        "it",
        "nl",
        "no",
        "pt",
        "ro",
        "ru",
        "sv",
        "ta",
        "tr",
    ]
)
_FTS_LANGUAGES_LONG_TO_SHORT: dict[str, str] = {
    "arabic": "ar",
    "danish": "da",
    "german": "de",
    "greek": "el",
    "english": "en",
    "spanish": "es",
    "finnish": "fi",
    "french": "fr",
    "hungarian": "hu",
    "italian": "it",
    "dutch": "nl",
    "norwegian": "no",
    "portuguese": "pt",
    "romanian": "ro",
    "russian": "ru",
    "swedish": "sv",
    "tamil": "ta",
    "turkish": "tr",
}


_FIELD_NAME_MAX_BYTES = 64
_DESCRIPTION_MAX_BYTES = 256
_DIMENSION_MIN = 1
_DIMENSION_MAX = 20000

_SPARSE_VECTOR_UNSUPPORTED_OPTIONS: dict[str, str] = {
    "metric": "a sparse vector field has no metric — sparse scoring is not configurable",
    "dimension": "a sparse vector field has no dimension — sparse vectors are variable-length",
}

_RESPONSE_ONLY_FIELD_TYPES: dict[str, str] = {
    "integer": (
        "'integer' appears only in describe/list responses, for indexes created "
        "before numeric values were normalised to float; the create schema has no "
        "integer variant. Drop the field from the create schema — numeric metadata "
        "is indexed for filtering automatically at upsert time, and the 'float' "
        "declaration add_float_field() writes is rejected on create too"
    )
}


def _validate_field_name(name: str) -> None:
    from pinecone.errors.exceptions import PineconeValueError

    if not name:
        raise PineconeValueError("Field name must be a non-empty string")
    try:
        byte_len = len(name.encode("utf-8"))
    except UnicodeEncodeError:
        raise PineconeValueError(
            f"Field name {name!r} is invalid: names must be valid Unicode "
            "(surrogate characters are not allowed)"
        ) from None
    if byte_len > _FIELD_NAME_MAX_BYTES:
        raise PineconeValueError(
            f"Field name '{name}' is too long: {byte_len} bytes (max {_FIELD_NAME_MAX_BYTES})"
        )


def _validate_description(name: str, description: str | None) -> None:
    from pinecone.errors.exceptions import PineconeValueError

    if description is None:
        return
    try:
        byte_len = len(description.encode("utf-8"))
    except UnicodeEncodeError:
        raise PineconeValueError(
            f"Field '{name}' description is invalid: descriptions must be valid "
            "Unicode (surrogate characters are not allowed)"
        ) from None
    if byte_len > _DESCRIPTION_MAX_BYTES:
        raise PineconeValueError(
            f"Field '{name}' description is too long: "
            f"{byte_len} bytes (max {_DESCRIPTION_MAX_BYTES})"
        )


def _validate_field_type(name: str, field: dict[str, Any]) -> None:
    """Reject field types the create-index schema does not accept.

    ``_RESPONSE_ONLY_FIELD_TYPES`` holds wire ``type`` values the API returns
    from describe/list but rejects on create, mapped to the remediation clause
    for that type. Copying a described schema straight into a create request
    is the way these reach the builder, so the message names the field, says
    the type is response-only, and states what to do instead — the server's
    own rejection is a plain-text ``422`` with no such guidance.
    """
    from pinecone.errors.exceptions import PineconeValueError

    field_type = field.get("type")
    if not isinstance(field_type, str):
        return
    remediation = _RESPONSE_ONLY_FIELD_TYPES.get(field_type)
    if remediation is None:
        return
    raise PineconeValueError(
        f"Field '{name}' has type '{field_type}', which is response-only and is "
        f"not accepted when creating an index: {remediation}."
    )


def _validate_sparse_vector_options(name: str, options: dict[str, Any]) -> None:
    """Refuse the keys a sparse vector field does not have.

    ``metric`` and ``dimension`` reach ``add_sparse_vector_field`` through
    ``**additional_options``, which exists to let genuinely new API keys
    through untouched. These two are not new: they are what 9.x's
    ``vector_type="sparse"`` implied, a create schema has nowhere to put
    either, and neither is echoed back by describe. Passing one through would
    read as configuration that took effect, so it is refused here instead.
    """
    from pinecone.errors.exceptions import PineconeValueError

    for key, reason in _SPARSE_VECTOR_UNSUPPORTED_OPTIONS.items():
        if key in options:
            raise PineconeValueError(
                f"Field '{name}' cannot declare '{key}': {reason}. Remove the "
                "argument — a sparse vector field accepts only a description."
            )


def _normalize_fts_language(language: str) -> str:
    """Return the canonical short-code form of a language input.

    Maps known long-form aliases (e.g. ``"english"``) and known short
    codes (e.g. ``"EN"``) to their canonical short-code form. Any value
    that does not match a known alias is returned unchanged so unknown
    or future languages pass through to the server unmodified — the
    server is the source of truth for which languages are supported.
    """
    if language in _FTS_LANGUAGES_SHORT:
        return language
    lowered = language.lower()
    if lowered in _FTS_LANGUAGES_LONG_TO_SHORT:
        return _FTS_LANGUAGES_LONG_TO_SHORT[lowered]
    if lowered in _FTS_LANGUAGES_SHORT:
        return lowered
    return language


class SchemaBuilder:
    """Assembles an index schema field by field.

    Construct one directly — ``SchemaBuilder()`` — then chain ``add_*``
    calls, each of which returns ``self``, and finish with :meth:`build`.
    Adding a field under a name already present replaces the earlier
    definition rather than raising, so a later call wins.

    Building the ``{"fields": {...}}`` dict by hand is equally valid and
    equally supported; the builder exists so the field types, their required
    keys, and the declarations the server refuses are checked as you write
    rather than on the create call.

    Examples:
        >>> from pinecone.schema_builder import SchemaBuilder
        >>> schema = (
        ...     SchemaBuilder()
        ...     .add_dense_vector_field("embedding", dimension=768, metric="cosine")
        ...     .add_string_field("title", full_text_search={"language": "en"})
        ...     .build()
        ... )
        >>> sorted(schema["fields"])
        ['embedding', 'title']

    Pass the result straight to the create call:

    .. code-block:: python

        pc.indexes.create(
            name="movie-recommendations",
            schema=schema,
            deployment={"deployment_type": "managed", "cloud": "aws",
                        "region": "us-east-1"},
        )

    .. seealso::
       :meth:`~pinecone.client.indexes.Indexes.create` — the ``schema``
       argument this builds, and the ``deployment`` argument that goes with
       it.
    """

    def __init__(self) -> None:
        self._fields: dict[str, dict[str, Any]] = {}

    def _set_field(self, name: str, field: dict[str, Any]) -> None:
        _validate_field_type(name, field)
        self._fields[name] = field

    def add_dense_vector_field(
        self,
        name: str,
        *,
        dimension: int,
        metric: str,
        description: str | None = None,
        **additional_options: Any,
    ) -> SchemaBuilder:
        """Add the field that holds an embedding, for vector similarity search.

        This is the field most indexes are built around, and a schema may
        hold at most one of them. ``dimension`` and ``metric`` are fixed for
        the life of the field, so they have to match the embedding model you
        intend to use.

        Args:
            name: Field name, up to 64 bytes; the documents you upsert carry
                their vector under this key, e.g. ``"embedding"``. Replaces
                any existing field with the same name.
            dimension: Length of the vectors this field stores — the output
                width of your embedding model, e.g. ``1536``. Must be between
                1 and 20000 inclusive; the SDK rejects anything else before
                the request goes out.
            metric: How similarity is scored — ``"cosine"``, ``"euclidean"``,
                or ``"dotproduct"``. ``"cosine"`` is what most text embedding
                models are trained for. See
                :class:`~pinecone.models.enums.Metric`.
            description: Human-readable note stored with the field, up to 256
                bytes. Optional.
            **additional_options: Extra parameters merged into the field dict
                last, for forward compatibility with new API features.

        Returns:
            ``self`` for method chaining.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`: If
                ``dimension`` is outside 1–20000, or if ``name`` or
                ``description`` exceeds its byte limit. Both limits count
                UTF-8 bytes rather than characters, so a name of non-ASCII
                text runs out sooner than its length suggests.

        Examples:
            >>> from pinecone.schema_builder import SchemaBuilder
            >>> schema = SchemaBuilder().add_dense_vector_field(
            ...     "embedding", dimension=1536, metric="cosine"
            ... ).build()
            >>> schema["fields"]["embedding"]
            {'type': 'dense_vector', 'dimension': 1536, 'metric': 'cosine'}
        """
        from pinecone.errors.exceptions import PineconeValueError

        _validate_field_name(name)
        _validate_description(name, description)
        if not (_DIMENSION_MIN <= dimension <= _DIMENSION_MAX):
            raise PineconeValueError(
                f"Field '{name}' has invalid dimension {dimension}: "
                f"dimension must be between {_DIMENSION_MIN} and {_DIMENSION_MAX} inclusive"
            )
        field: dict[str, Any] = {
            "type": "dense_vector",
            "dimension": dimension,
            "metric": metric,
        }
        if description is not None:
            field["description"] = description
        field.update(additional_options)
        self._set_field(name, field)
        return self

    def add_sparse_vector_field(
        self,
        name: str,
        *,
        description: str | None = None,
        **additional_options: Any,
    ) -> SchemaBuilder:
        """Add a sparse vector field, for keyword-weighted or learned-sparse search.

        Declare one alongside a dense vector field for hybrid search, or on
        its own for pure sparse retrieval. A schema may hold at most one.

        ``description`` is the only other key a create schema accepts here. A
        sparse vector field takes no ``metric`` — sparse scoring is not
        configurable — and no ``dimension``, because sparse vectors are
        variable-length. Passing either raises rather than putting a key on
        the wire that configures nothing.

        Args:
            name: Field name, up to 64 bytes, e.g. ``"keyword_terms"``.
                Replaces any existing field with the same name.
            description: Human-readable note stored with the field, up to 256
                bytes. Optional.
            **additional_options: Extra parameters merged into the field dict
                last, for forward compatibility with new API features.

        Returns:
            ``self`` for method chaining.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`: If
                ``metric`` or ``dimension`` is passed — a sparse vector field
                has neither — or if ``name`` or ``description`` exceeds its
                byte limit.

        Examples:
            >>> from pinecone.schema_builder import SchemaBuilder
            >>> schema = (
            ...     SchemaBuilder()
            ...     .add_dense_vector_field(
            ...         "embedding", dimension=1536, metric="dotproduct"
            ...     )
            ...     .add_sparse_vector_field("keyword_terms")
            ...     .build()
            ... )
            >>> schema["fields"]["keyword_terms"]
            {'type': 'sparse_vector'}
        """
        _validate_field_name(name)
        _validate_description(name, description)
        _validate_sparse_vector_options(name, additional_options)
        field: dict[str, Any] = {"type": "sparse_vector"}
        if description is not None:
            field["description"] = description
        field.update(additional_options)
        self._set_field(name, field)
        return self

    def add_string_field(
        self,
        name: str,
        *,
        full_text_search: bool | dict[str, Any] | None = None,
        language: str | None = None,
        stemming: bool | None = None,
        stop_words: bool | None = None,
        filterable: bool = False,
        description: str | None = None,
        **additional_options: Any,
    ) -> SchemaBuilder:
        """Add a string field searchable by keyword, using full-text search.

        Always pass ``full_text_search``: enabling it is what makes the field
        searched, and a string field without it is a metadata-only
        declaration the server refuses on create. Any of
        ``full_text_search=True``, a ``full_text_search`` dict, or one of the
        typed keyword arguments (``language``, ``stemming``, ``stop_words``)
        turns it on; where a dict and a keyword argument set the same key,
        the keyword argument wins.

        ``lowercase`` and ``max_term_len`` are managed for you and cannot be
        set here.

        Args:
            name: Field name, up to 64 bytes; the documents you upsert carry
                the text under this key, e.g. ``"title"``. Replaces any
                existing field with the same name.
            full_text_search: ``True`` or ``{}`` for full-text search with
                default analysis, or a dict of the config keys
                (``language``, ``stemming``, ``stop_words``, ``ngram``).
                ``None``, the default, leaves the field unsearched — see the
                note below before choosing it.
            language: Language whose rules drive tokenisation and analysis.
                Both short codes and their English names work — ``"en"`` and
                ``"english"`` are the same request, and the SDK sends the
                short form. It knows ``ar``, ``da``, ``de``, ``el``, ``en``,
                ``es``, ``fi``, ``fr``, ``hu``, ``it``, ``nl``, ``no``,
                ``pt``, ``ro``, ``ru``, ``sv``, ``ta`` and ``tr``, and
                passes anything else through untouched so a
                newly-supported language works without an SDK upgrade; the
                server decides what it accepts.
            stemming: Match words by their root, so a search for
                ``"running"`` also finds ``"run"``. Required when
                ``stop_words=True``.
            stop_words: Drop the language's most common words from the
                index. Requires ``stemming=True``. Not every language
                supports stop words, and the server is what rejects an
                unsupported pairing — the SDK does not pre-check it.
            filterable: Make the field filterable instead of searched.
                Requesting it together with ``full_text_search`` is the trap:
                a string field is one or the other, and the server keeps the
                filter, silently discards the search configuration, and
                reports no error. Sent on the wire including its ``False``
                default, except when ``full_text_search`` is enabled and you
                did not ask to be filterable.
            description: Human-readable note stored with the field, up to 256
                bytes. Optional.
            **additional_options: Extra parameters merged into the field dict
                last, for forward compatibility with new API features.

        Returns:
            ``self`` for method chaining.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`: If
                ``stop_words=True`` is requested without ``stemming=True``,
                if an ``ngram`` config is combined with ``stemming=True`` or
                ``stop_words=True``, or if ``name`` or ``description``
                exceeds its byte limit.

        Examples:
            Default analysis is enough for most text:

            >>> from pinecone.schema_builder import SchemaBuilder
            >>> schema = SchemaBuilder().add_string_field(
            ...     "title", full_text_search=True
            ... ).build()
            >>> schema["fields"]["title"]
            {'type': 'string', 'full_text_search': {}}

            Naming the language turns on that language's analysis, and the
            long form is accepted and normalised:

            >>> schema = SchemaBuilder().add_string_field(
            ...     "title", language="english", stemming=True, stop_words=True
            ... ).build()
            >>> schema["fields"]["title"]["full_text_search"]
            {'language': 'en', 'stemming': True, 'stop_words': True}

            Character n-grams match substrings rather than whole words, which
            is what autocomplete needs. They cannot be combined with stemming
            or stop words:

            >>> schema = SchemaBuilder().add_string_field(
            ...     "title", full_text_search={"ngram": {"min_gram": 2, "max_gram": 3}}
            ... ).build()
            >>> schema["fields"]["title"]["full_text_search"]
            {'ngram': {'min_gram': 2, 'max_gram': 3}}

        .. note::
           A string field with no ``full_text_search`` is a metadata-only
           declaration, and the server rejects those on create — a ``400``
           saying the schema only accepts fields used for search — whatever
           deployment you ask for, failing the whole request over the one
           field. Leave such fields out of the schema and put the values in
           the documents you upsert; they are indexed for filtering
           automatically.
        """
        from pinecone.errors.exceptions import PineconeValueError

        _validate_field_name(name)
        _validate_description(name, description)
        # Determine whether FTS is enabled by ANY of the inputs.
        fts_kwargs_provided = language is not None or stemming is not None or stop_words is not None
        fts_enabled = (
            full_text_search is True or isinstance(full_text_search, dict) or fts_kwargs_provided
        )

        fts_config: dict[str, Any] = {}
        if isinstance(full_text_search, dict):
            fts_config.update(full_text_search)
        if language is not None:
            fts_config["language"] = _normalize_fts_language(language)
        if stemming is not None:
            fts_config["stemming"] = stemming
        if stop_words is not None:
            fts_config["stop_words"] = stop_words

        # Pre-validate cross-field rules AFTER merging so we see the final
        # values users intended (whether they came from the dict or a kwarg).
        # The ngram check runs first, matching the server's validation order.
        if "ngram" in fts_config and (
            fts_config.get("stemming") is True or fts_config.get("stop_words") is True
        ):
            raise PineconeValueError(
                f"Field '{name}': ngram cannot be combined with stemming or stop_words"
            )
        if fts_config.get("stop_words") is True and fts_config.get("stemming") is not True:
            raise PineconeValueError(f"Field '{name}': stop_words requires stemming to be enabled")

        # If the dict supplied a language string, normalize it too (kwarg path
        # already normalized above; the dict path may not have).
        if "language" in fts_config and isinstance(fts_config["language"], str):
            fts_config["language"] = _normalize_fts_language(fts_config["language"])

        field: dict[str, Any] = {"type": "string"}
        if fts_enabled:
            field["full_text_search"] = fts_config
        # `filterable` selects the metadata variant on the wire, so the
        # `False` default is omitted only when `full_text_search` already
        # selects the search variant and the caller did not also ask to be
        # filterable — otherwise no variant would match this field at all.
        if filterable or not fts_enabled:
            field["filterable"] = filterable
        if description is not None:
            field["description"] = description
        field.update(additional_options)
        self._set_field(name, field)
        return self

    def add_string_list_field(
        self,
        name: str,
        *,
        filterable: bool = False,
        description: str | None = None,
        **additional_options: Any,
    ) -> SchemaBuilder:
        """Declare a list-of-strings field — which no index you can create accepts.

        A string-list field holds several strings per record, the shape
        tag-style metadata takes (``["sci-fi", "mystery"]``) when you want to
        filter on individual elements. Declaring it is what does not work:
        see the note below, and upsert the list as an ordinary document field
        instead.

        Args:
            name: Field name, up to 64 bytes, e.g. ``"tags"``. Replaces any
                existing field with the same name.
            filterable: Enable metadata filtering on the field. Always
                included in the built schema, whether ``True`` or ``False``.
            description: Human-readable note stored with the field, up to 256
                bytes. Optional.
            **additional_options: Extra parameters merged into the field dict
                last, for forward compatibility with new API features.

        Returns:
            ``self`` for method chaining.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`: If ``name``
                or ``description`` exceeds its byte limit.

        Examples:
            >>> from pinecone.schema_builder import SchemaBuilder
            >>> schema = SchemaBuilder().add_string_list_field(
            ...     "tags", filterable=True
            ... ).build()
            >>> schema["fields"]["tags"]
            {'type': 'string_list', 'filterable': True}

        .. note::
           ``string_list`` is a metadata-only declaration, and the server
           rejects those on create — a ``400`` saying the schema only accepts
           fields used for search — whatever deployment you ask for and
           whatever ``filterable`` says, failing the whole schema over the
           one field. Leave the field out and put the values in the documents
           you upsert; they are indexed for filtering automatically.
        """
        _validate_field_name(name)
        _validate_description(name, description)
        field: dict[str, Any] = {"type": "string_list", "filterable": filterable}
        if description is not None:
            field["description"] = description
        field.update(additional_options)
        self._set_field(name, field)
        return self

    def add_boolean_field(
        self,
        name: str,
        *,
        filterable: bool = False,
        description: str | None = None,
        **additional_options: Any,
    ) -> SchemaBuilder:
        """Declare a boolean field — which no index you can create accepts.

        Declaring the field is what does not work; a boolean is filterable
        once it is in a document. See the note below, and upsert the flag as
        an ordinary document field instead.

        Args:
            name: Field name, up to 64 bytes, e.g. ``"is_published"``.
                Replaces any existing field with the same name.
            filterable: Enable metadata filtering on the field. Always
                included in the built schema, whether ``True`` or ``False``.
            description: Human-readable note stored with the field, up to 256
                bytes. Optional.
            **additional_options: Extra parameters merged into the field dict
                last, for forward compatibility with new API features.

        Returns:
            ``self`` for method chaining.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`: If ``name``
                or ``description`` exceeds its byte limit.

        Examples:
            >>> from pinecone.schema_builder import SchemaBuilder
            >>> schema = SchemaBuilder().add_boolean_field(
            ...     "is_published", filterable=True
            ... ).build()
            >>> schema["fields"]["is_published"]
            {'type': 'boolean', 'filterable': True}

        .. note::
           ``boolean`` is a metadata-only declaration, and the server rejects
           those on create — a ``400`` saying the schema only accepts fields
           used for search — whatever deployment you ask for and whatever
           ``filterable`` says, failing the whole schema over the one field.
           Leave the field out and put the values in the documents you
           upsert; they are indexed for filtering automatically.
        """
        _validate_field_name(name)
        _validate_description(name, description)
        field: dict[str, Any] = {"type": "boolean", "filterable": filterable}
        if description is not None:
            field["description"] = description
        field.update(additional_options)
        self._set_field(name, field)
        return self

    def add_float_field(
        self,
        name: str,
        *,
        filterable: bool = False,
        description: str | None = None,
        **additional_options: Any,
    ) -> SchemaBuilder:
        """Declare a numeric field — which no index you can create accepts.

        This is the only numeric declaration there is: a create schema has no
        integer type, and whole numbers are stored and filtered as
        double-precision floats. Declaring the field is what does not work;
        numbers are filterable once they are in a document. See the note
        below, and upsert the value as an ordinary document field instead.

        Describe and list responses can still return ``integer`` for indexes
        that pre-date that normalisation — see
        :class:`~pinecone.models.indexes.schema.IntegerField`.

        Args:
            name: Field name, up to 64 bytes, e.g. ``"release_year"``.
                Replaces any existing field with the same name.
            filterable: Enable metadata filtering on the field. Always
                included in the built schema, whether ``True`` or ``False``.
            description: Human-readable note stored with the field, up to 256
                bytes. Optional.
            **additional_options: Extra parameters merged into the field dict
                last, for forward compatibility with new API features.

        Returns:
            ``self`` for method chaining.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`: If ``name``
                or ``description`` exceeds its byte limit.

        Examples:
            >>> from pinecone.schema_builder import SchemaBuilder
            >>> schema = SchemaBuilder().add_float_field(
            ...     "release_year", filterable=True
            ... ).build()
            >>> schema["fields"]["release_year"]
            {'type': 'float', 'filterable': True}

        .. note::
           ``float`` is a metadata-only declaration, and the server rejects
           those on create — a ``400`` saying the schema only accepts fields
           used for search — whatever deployment you ask for and whatever
           ``filterable`` says, failing the whole schema over the one field.
           Leave the field out and put the values in the documents you
           upsert; they are indexed for filtering automatically.
        """
        _validate_field_name(name)
        _validate_description(name, description)
        field: dict[str, Any] = {"type": "float", "filterable": filterable}
        if description is not None:
            field["description"] = description
        field.update(additional_options)
        self._set_field(name, field)
        return self

    def add_custom_field(
        self,
        name: str,
        field_definition: dict[str, Any],
    ) -> SchemaBuilder:
        """Store a raw field dict verbatim — the escape hatch.

        Two jobs: copying a field definition out of a describe response into
        a new index's schema, and declaring a field type a newer API version
        offers that this SDK does not model yet. The name is checked and the
        definition's ``type`` is rejected if it is response-only; the rest of
        the definition goes through untouched, so anything else wrong with it
        surfaces as a server error on create rather than here.

        Args:
            name: Field name, up to 64 bytes. Replaces any existing field
                with the same name.
            field_definition: The complete field definition, stored as-is and
                deep-copied into the built schema.

        Returns:
            ``self`` for method chaining.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`: If the
                field name is empty or over its byte limit, or if
                ``field_definition["type"]`` is response-only, which today
                means ``"integer"``.

        Examples:
            >>> from pinecone.schema_builder import SchemaBuilder
            >>> described = {"type": "dense_vector", "dimension": 1536,
            ...              "metric": "cosine"}
            >>> schema = SchemaBuilder().add_custom_field(
            ...     "embedding", described
            ... ).build()
            >>> schema["fields"]["embedding"]
            {'type': 'dense_vector', 'dimension': 1536, 'metric': 'cosine'}

        .. note::
           ``{"type": "integer"}`` is refused here rather than on the wire.
           ``integer`` comes back from describe and list for indexes created
           before numeric values were normalised to float, but there is no
           integer variant to create and the server answers a create request
           carrying one with a ``422``. When replaying a described schema
           into a create request, drop those fields —
           :meth:`add_float_field` is no help, because a ``float``
           declaration is rejected on create too, and numeric metadata needs
           no declaration.
        """
        _validate_field_name(name)
        self._set_field(name, field_definition)
        return self

    def build(self) -> dict[str, dict[str, Any]]:
        """Return the completed schema dict.

        The result is a deep copy of the builder's state, so writing into it
        — adding a forward-compatible key the SDK does not yet model, for
        instance — leaves the builder and every other result of ``build()``
        untouched. One builder can be reused across several indexes even when
        each schema is edited after the fact.

        Nothing here checks that the schema is complete. The server requires
        at least one searched field (``dense_vector``, ``sparse_vector``, or
        ``string`` with ``full_text_search``), and a schema without one is
        built and returned all the same, so that partial schemas can be
        inspected; the create call is where it fails.

        Returns:
            ``{"fields": {name: field_dict, ...}}`` ready to pass as the
            ``schema`` argument when creating an index.

        Examples:
            >>> from pinecone.schema_builder import SchemaBuilder
            >>> builder = SchemaBuilder().add_dense_vector_field(
            ...     "embedding", dimension=8, metric="cosine"
            ... )
            >>> schema = builder.build()
            >>> schema["fields"]["embedding"]["future_option"] = True
            >>> builder.build()["fields"]["embedding"]
            {'type': 'dense_vector', 'dimension': 8, 'metric': 'cosine'}
        """
        return {"fields": copy.deepcopy(self._fields)}


__all__ = ["SchemaBuilder"]
