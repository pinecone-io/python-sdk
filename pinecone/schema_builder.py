"""Build the ``schema`` argument for creating an index.

A schema names the fields an index searches and says how each one is scored.
:class:`SchemaBuilder` assembles that structure a field at a time and hands
back a plain ``{"fields": {...}}`` dict rather than a model, so a key this SDK
version does not yet model still reaches the server unchanged.

The builder helps you structure a schema and discover what each field type
takes; it is not a full validator. It catches a few mistakes locally, passes
dict-valued options such as ``embed`` and ``sparse_embed`` through as given,
and leaves the rest to the server, which is the source of truth: a schema it
rejects makes the create call raise
:class:`~pinecone.errors.exceptions.ApiError` carrying the server's message.

What belongs in a create schema is narrower than what an index stores. Declare
the fields you search (a ``dense_vector`` field, a ``sparse_vector`` field,
and ``string`` fields with full-text search enabled or with integrated
embedding through ``embed`` / ``sparse_embed``); at least one is required.
Everything else is metadata: put those values in the documents you upsert and
they are indexed for filtering automatically, with no declaration at all. An
index has one dense slot and one sparse slot: at most one field may be a
``dense_vector`` field or carry ``embed``, and at most one may be a
``sparse_vector`` field or carry ``sparse_embed``.

The builder still has methods for the metadata-only declarations
(:meth:`SchemaBuilder.add_boolean_field`,
:meth:`SchemaBuilder.add_float_field`,
:meth:`SchemaBuilder.add_string_list_field`, and
:meth:`SchemaBuilder.add_string_field` with no search key). The server
rejects all of them on create, and one rejected field fails the whole request.

Two field types have no method at all. For integrated embedding, pass
``embed`` or ``sparse_embed`` to :meth:`SchemaBuilder.add_string_field`; the
``semantic_text`` field type, which is how a legacy integrated index (created
via :meth:`~pinecone.client.indexes.Indexes.create_for_model`) reports its
embedded field, is not accepted in a create schema.
``integer`` is response-only: describe and list return it for indexes that
pre-date numeric normalisation, but a create schema has no ``integer`` type.
The builder refuses it wherever a raw ``type`` key could reach a field dict
(:meth:`SchemaBuilder.add_custom_field` and the ``additional_options``
parameters), so replaying a described schema into a create request fails
locally with an explanation instead of with a bare server status code.

.. seealso::
   :meth:`~pinecone.client.indexes.Indexes.create`: the method that consumes
   the result, and the deprecated ``dimension=``/``metric=`` sugar this
   replaces.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import msgspec

from pinecone._internal.indexes_helpers import lowercase_metric
from pinecone.models.indexes.schema import FieldEmbedConfig, FieldSparseEmbedConfig

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


def _embed_aspect(
    value: bool | Mapping[str, Any] | FieldEmbedConfig | FieldSparseEmbedConfig | None,
) -> dict[str, Any] | None:
    if value is None or value is False:
        return None
    if value is True:
        return {}
    if isinstance(value, msgspec.Struct):
        config = {k: v for k, v in msgspec.to_builtins(value).items() if v is not None}
    else:
        config = dict(value)
    if "metric" in config:
        config["metric"] = lowercase_metric(config["metric"])
    return config


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

    Construct one directly with ``SchemaBuilder()``, then chain ``add_*``
    calls, each of which returns ``self``, and finish with :meth:`build`.
    Adding a field under a name already present replaces the earlier
    definition rather than raising, so a later call wins.

    Building the ``{"fields": {...}}`` dict by hand is equally valid and
    equally supported; the builder exists to lay out the field types and
    their options, not to decide what the server accepts. It checks field
    name and description length, the dimension range and the required keys,
    and refuses the
    response-only ``integer`` type when a raw ``type`` key reaches a field
    dict (:meth:`add_custom_field`, ``additional_options``). It does not check
    which field combinations the server accepts, or the keys and values inside
    an ``embed`` or ``sparse_embed`` dict: the metadata-only methods, a second
    dense vector field, or ``embed={"modle": ...}`` are built as written, and
    the create call raises :class:`~pinecone.errors.exceptions.ApiError` (a
    ``400`` or ``422``) with the server's message.

    Examples:
        >>> from pinecone.schema_builder import SchemaBuilder
        >>> schema = (
        ...     SchemaBuilder()
        ...     .add_dense_vector_field("your_vector", dimension=768, metric="cosine")
        ...     .add_string_field(
        ...         "title", full_text_search={"language": "en", "stemming": True}
        ...     )
        ...     .build()
        ... )
        >>> sorted(schema["fields"])
        ['title', 'your_vector']

    Pass the result straight to the create call:

    .. code-block:: python

        pc.indexes.create(
            name="movie-recommendations",
            schema=schema,
            deployment={"deployment_type": "managed", "cloud": "aws",
                        "region": "us-east-1"},
        )

    .. seealso::
       :meth:`~pinecone.client.indexes.Indexes.create`: the ``schema``
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
        hold at most one of them. The builder does not count them: a second
        dense vector field is built as written and the server rejects it on
        create with a ``400``. ``dimension`` and ``metric`` are fixed for
        the life of the field, so they have to match the embedding model you
        intend to use.

        Args:
            name: Field name; the documents you upsert carry their vector
                under this key, e.g. ``"your_vector"``. Replaces any existing
                field with the same name.
            dimension: Length of the vectors this field stores: the output
                width of your embedding model, e.g. ``1536``. The SDK rejects
                a dimension outside the supported range before the request
                goes out.
            metric: How similarity is scored: ``"cosine"``, ``"euclidean"``,
                or ``"dotproduct"``. ``"cosine"`` is what most text embedding
                models are trained for. Capitals are lowercased before the
                request is sent. See :class:`~pinecone.models.enums.Metric`.
            description: Human-readable note stored with the field. Optional.
            **additional_options: Extra parameters merged into the field dict
                last, for forward compatibility with new API features.

        Returns:
            ``self`` for method chaining.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`: If
                ``dimension`` is outside the supported range, or if ``name``
                or ``description`` is too long. Both lengths count UTF-8
                bytes rather than characters, so non-ASCII text runs out
                sooner than its length suggests.

        Examples:
            >>> from pinecone.schema_builder import SchemaBuilder
            >>> schema = SchemaBuilder().add_dense_vector_field(
            ...     "your_vector", dimension=1536, metric="cosine"
            ... ).build()
            >>> schema["fields"]["your_vector"]
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
            "metric": lowercase_metric(metric),
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
        sparse vector field takes no ``metric``, because sparse scoring is not
        configurable, and no ``dimension``, because sparse vectors are
        variable-length. Passing either raises rather than putting a key on
        the wire that configures nothing.

        Args:
            name: Field name, e.g. ``"keyword_terms"``.
                Replaces any existing field with the same name.
            description: Human-readable note stored with the field. Optional.
            **additional_options: Extra parameters merged into the field dict
                last, for forward compatibility with new API features.

        Returns:
            ``self`` for method chaining.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`: If
                ``metric`` or ``dimension`` is passed, or if ``name`` or
                ``description`` is too long.

        Examples:
            >>> from pinecone.schema_builder import SchemaBuilder
            >>> schema = (
            ...     SchemaBuilder()
            ...     .add_dense_vector_field(
            ...         "your_vector", dimension=1536, metric="dotproduct"
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
        embed: bool | Mapping[str, Any] | FieldEmbedConfig | None = None,
        sparse_embed: bool | Mapping[str, Any] | FieldSparseEmbedConfig | None = None,
        store_text: bool | None = None,
        filterable: bool = False,
        description: str | None = None,
        **additional_options: Any,
    ) -> SchemaBuilder:
        """Add a string field searched by full-text search, integrated embedding, or both.

        Always pass at least one of ``full_text_search``, ``embed`` and
        ``sparse_embed``: they are what make the field searched, and a string
        field with none of them is a metadata-only declaration the server
        refuses on create. Any of ``full_text_search=True``, a
        ``full_text_search`` dict, or one of the typed keyword arguments
        (``language``, ``stemming``, ``stop_words``) turns full-text search
        on; where a dict and a keyword argument set the same key, the keyword
        argument wins.

        ``embed`` and ``sparse_embed`` enable integrated embedding for the field:
        Pinecone embeds the text into a dense or learned-sparse vector with a
        hosted model. Search
        the field with :class:`~pinecone.models.documents.score_by.EmbedQuery`
        or :class:`~pinecone.models.documents.score_by.SparseEmbedQuery`.
        The text is stored by default, and stored text is always full-text
        searchable, so ``embed`` alone also turns on full-text search with
        default settings. ``store_text=False`` embeds the text without
        storing it; see ``store_text`` below for what that gives up.

        Describe responses also show ``lowercase`` and ``max_term_len``; the
        server sets those and rejects them in a create schema.

        Args:
            name: Field name; the documents you upsert carry the text under
                this key, e.g. ``"title"``. Replaces any
                existing field with the same name.
            full_text_search: ``True`` or ``{}`` for full-text search with
                default analysis, or a dict of the config keys
                (``language``, ``stemming``, ``stop_words``, ``ngram``).
                ``None``, the default, leaves the field unsearched unless
                ``embed`` or ``sparse_embed`` is set; see the note below.
            language: Language whose rules drive tokenisation and analysis.
                Both short codes and their English names work: ``"en"`` and
                ``"english"`` are the same request, and the SDK sends the
                short form. A name the SDK does not recognise passes through
                unchanged, so a newly supported language works without an
                SDK upgrade; the server decides what it accepts.
            stemming: Match words by their root, so a search for
                ``"running"`` also finds ``"run"``. Required when
                ``stop_words=True``.
            stop_words: Drop the language's most common words from the
                index. Requires ``stemming=True``. Not every language
                supports stop words; the server rejects an unsupported
                pairing.
            embed: ``True`` or ``{}`` to embed the text with the default dense
                model at its default dimension and metric, or a dict or
                :class:`~pinecone.models.indexes.schema.FieldEmbedConfig` of
                ``model``, ``dimension``, ``metric`` (lowercased, as for a
                dense vector field), ``processing_location``,
                ``write_parameters`` and ``read_parameters``. A dict is sent
                as given, so a misspelled key fails only on create; the typed
                config rejects it when constructed. The field takes the index's one dense slot.
                Fixed at creation.
            sparse_embed: ``True`` or ``{}`` to embed the text with the
                default sparse model, or a dict or
                :class:`~pinecone.models.indexes.schema.FieldSparseEmbedConfig`
                of ``model``, ``write_parameters`` and ``read_parameters``. A
                dict is sent as given, as for ``embed``. The field takes the
                index's one sparse slot. Fixed at creation.
            store_text: Only meaningful as ``False``, on a field with ``embed``
                or ``sparse_embed``. ``None``, the default, leaves the key off
                and the server stores the text. ``False`` embeds the text
                without storing it, which can save space when the field is
                only used for similarity search, or keep sensitive text out of
                Pinecone. The field then has no full-text search: it can be
                searched only with an embed clause, and a fetch or search
                whose ``include_fields`` names it is rejected with a ``400``.
                The server rejects ``False`` on a field without ``embed`` or
                ``sparse_embed``, and together with ``full_text_search``, each
                with a ``400``. If you're not sure whether you'll need
                full-text search, keep the text stored, for more search
                flexibility later. Fixed at creation.
            filterable: Make the field filterable instead of searched. A
                string field is one or the other: the server rejects
                ``filterable=True`` alongside ``full_text_search``, ``embed``
                or ``sparse_embed`` with a ``400``. Sent on the wire including
                its ``False`` default, except when ``full_text_search``,
                ``embed``, ``sparse_embed`` or ``store_text`` is set and you
                did not ask to be filterable.
            description: Human-readable note stored with the field. Optional.
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

            Pinecone embeds the text of an ``embed`` field for you, here with
            a chosen model and dimension, and a sparse aspect can sit on the
            same field:

            >>> schema = SchemaBuilder().add_string_field(
            ...     "body",
            ...     embed={"model": "llama-text-embed-v2", "dimension": 512},
            ...     sparse_embed=True,
            ... ).build()
            >>> schema["fields"]["body"]["embed"]
            {'model': 'llama-text-embed-v2', 'dimension': 512}
            >>> schema["fields"]["body"]["sparse_embed"]
            {}

            ``store_text=False`` embeds the text without keeping it:

            >>> schema = SchemaBuilder().add_string_field(
            ...     "body", embed=True, store_text=False
            ... ).build()
            >>> schema["fields"]["body"]
            {'type': 'string', 'embed': {}, 'store_text': False}

        .. note::
           A string field with none of ``full_text_search``, ``embed`` and
           ``sparse_embed`` is a metadata-only
           declaration. The server rejects those on create with a ``400``
           saying the schema only accepts fields used for search, failing the
           whole request over the one field. Leave such fields out of the
           schema and put the values in the documents you upsert; they are
           indexed for filtering automatically.
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
        embed_config = _embed_aspect(embed)
        if embed_config is not None:
            field["embed"] = embed_config
        sparse_embed_config = _embed_aspect(sparse_embed)
        if sparse_embed_config is not None:
            field["sparse_embed"] = sparse_embed_config
        if store_text is not None:
            field["store_text"] = store_text
        search_configured = (
            fts_enabled
            or embed_config is not None
            or sparse_embed_config is not None
            or store_text is not None
        )
        # `filterable` selects the metadata variant on the wire, so the
        # `False` default is omitted only when a search key already selects
        # the search variant and the caller did not also ask to be
        # filterable — otherwise no variant would match this field at all.
        if filterable or not search_configured:
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
        """Declare a list-of-strings field, which the server rejects on create.

        A string-list field holds several strings per record, the shape
        tag-style metadata takes (``["sci-fi", "mystery"]``) when you want to
        filter on individual elements. Declaring it is what does not work:
        see the note below, and upsert the list as an ordinary document field
        instead.

        Args:
            name: Field name, e.g. ``"tags"``. Replaces any
                existing field with the same name.
            filterable: Enable metadata filtering on the field. Always
                included in the built schema, whether ``True`` or ``False``.
            description: Human-readable note stored with the field. Optional.
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
           ``string_list`` is a metadata-only declaration. The server rejects those
           on create with a ``400`` saying the schema only accepts fields
           used for search, whatever ``filterable`` says, failing the whole
           schema over the one field. Leave the field out and put the values
           in the documents you upsert; they are indexed for filtering
           automatically.
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
        """Declare a boolean field, which the server rejects on create.

        Declaring the field is what does not work; a boolean is filterable
        once it is in a document. See the note below, and upsert the flag as
        an ordinary document field instead.

        Args:
            name: Field name, e.g. ``"is_published"``.
                Replaces any existing field with the same name.
            filterable: Enable metadata filtering on the field. Always
                included in the built schema, whether ``True`` or ``False``.
            description: Human-readable note stored with the field. Optional.
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
           ``boolean`` is a metadata-only declaration. The server rejects those
           on create with a ``400`` saying the schema only accepts fields
           used for search, whatever ``filterable`` says, failing the whole
           schema over the one field. Leave the field out and put the values
           in the documents you upsert; they are indexed for filtering
           automatically.
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
        """Declare a numeric field, which the server rejects on create.

        This is the only numeric declaration there is: a create schema has no
        integer type, and whole numbers are stored and filtered as
        double-precision floats. Declaring the field is what does not work;
        numbers are filterable once they are in a document. See the note
        below, and upsert the value as an ordinary document field instead.

        Describe and list responses can still return ``integer`` for indexes
        that pre-date that normalisation; see
        :class:`~pinecone.models.indexes.schema.IntegerField`.

        Args:
            name: Field name, e.g. ``"release_year"``.
                Replaces any existing field with the same name.
            filterable: Enable metadata filtering on the field. Always
                included in the built schema, whether ``True`` or ``False``.
            description: Human-readable note stored with the field. Optional.
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
           ``float`` is a metadata-only declaration. The server rejects those
           on create with a ``400`` saying the schema only accepts fields
           used for search, whatever ``filterable`` says, failing the whole
           schema over the one field. Leave the field out and put the values
           in the documents you upsert; they are indexed for filtering
           automatically.
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
        """Store a raw field dict verbatim, as an escape hatch.

        Two jobs: copying a field definition out of a describe response into
        a new index's schema, and declaring a field type a newer API version
        offers that this SDK does not model yet. The name is checked and the
        definition's ``type`` is rejected if it is response-only; the rest of
        the definition goes through untouched, so anything else wrong with it
        surfaces as a server error on create rather than here. The one
        exception is a ``metric`` in capitals, which is lowercased when the
        create request is sent, not in the schema this builder returns.

        Args:
            name: Field name, e.g. ``"your_vector"``. Replaces any existing
                field with the same name.
            field_definition: The complete field definition, stored as-is and
                deep-copied into the built schema.

        Returns:
            ``self`` for method chaining.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`: If the
                field name is empty or too long, or if
                ``field_definition["type"]`` is response-only, which today
                means ``"integer"``.

        Examples:
            >>> from pinecone.schema_builder import SchemaBuilder
            >>> described = {"type": "dense_vector", "dimension": 1536,
            ...              "metric": "cosine"}
            >>> schema = SchemaBuilder().add_custom_field(
            ...     "your_vector", described
            ... ).build()
            >>> schema["fields"]["your_vector"]
            {'type': 'dense_vector', 'dimension': 1536, 'metric': 'cosine'}

        .. note::
           ``{"type": "integer"}`` is refused here rather than on the wire.
           ``integer`` comes back from describe and list for indexes created
           before numeric values were normalised to float, but a create
           schema has no ``integer`` type. When replaying a described schema
           into a create request, drop those fields.
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
        (adding a forward-compatible key the SDK does not yet model, for
        instance) leaves the builder and every other result of ``build()``
        untouched. One builder can be reused across several indexes even when
        each schema is edited after the fact.

        Nothing here checks that the schema is complete or that the server
        will accept it. The server requires
        at least one searched field (``dense_vector``, ``sparse_vector``, or
        ``string`` with ``full_text_search``, ``embed`` or ``sparse_embed``),
        and a schema without one is built and returned all the same, so that
        partial schemas can be inspected; the create call is where it fails.

        Returns:
            ``{"fields": {name: field_dict, ...}}`` ready to pass as the
            ``schema`` argument when creating an index.

        Examples:
            >>> from pinecone.schema_builder import SchemaBuilder
            >>> builder = SchemaBuilder().add_dense_vector_field(
            ...     "your_vector", dimension=8, metric="cosine"
            ... )
            >>> schema = builder.build()
            >>> schema["fields"]["your_vector"]["future_option"] = True
            >>> builder.build()["fields"]["your_vector"]
            {'type': 'dense_vector', 'dimension': 8, 'metric': 'cosine'}
        """
        return {"fields": copy.deepcopy(self._fields)}


__all__ = ["SchemaBuilder"]
