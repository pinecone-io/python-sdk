"""SchemaBuilder for constructing index schemas (2026-07 create-schema rules).

Returns a plain ``{"fields": {...}}`` dict (not a model) so forward-compatible
fields the SDK does not yet model can pass through unmodified.

Create-schema rules at API version ``2026-07``:

- Every schema must declare at least one **searched** field: ``dense_vector``,
  ``sparse_vector``, or ``string`` with ``full_text_search``.
- At most one ``dense_vector`` and at most one ``sparse_vector`` field per
  schema (server-enforced).
- On managed and BYOC indexes, metadata-only field declarations (``boolean``,
  ``float``, ``string_list``, and ``string`` without ``full_text_search``) are
  rejected by the server — metadata is indexed automatically at upsert time.
  Pod indexes are the exception: they still accept metadata field
  declarations.
- ``semantic_text`` fields are not accepted in create schemas at ``2026-07``
  and the builder does not offer a method for them.
"""

from __future__ import annotations

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


def _validate_field_name(name: str) -> None:
    from pinecone.errors.exceptions import PineconeValueError

    if not name:
        raise PineconeValueError("Field name must be a non-empty string")
    if name.startswith("$") or name.startswith("_"):
        raise PineconeValueError(
            f"Field name '{name}' is invalid: names cannot begin with '$' or '_'"
        )
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
    """Fluent builder for index schema dicts (API version ``2026-07``).

    Each ``add_*`` method appends or replaces a field definition and returns
    ``self`` so calls can be chained.  Call :meth:`build` at the end to
    obtain the ``{"fields": {...}}`` dict.

    Adding a field whose name already exists silently replaces the previous
    definition (last writer wins).

    A create schema declares the fields that are **searched**: a dense
    vector field, a sparse vector field, or string fields with full-text
    search enabled. On managed and BYOC indexes, every other field type is
    metadata — include those values in documents instead of the schema and
    they are indexed for filtering automatically at upsert time. Pod
    indexes still accept metadata field declarations
    (:meth:`add_boolean_field`, :meth:`add_float_field`,
    :meth:`add_string_list_field`, and :meth:`add_string_field` without
    full-text search).

    Examples:
        >>> from pinecone.schema_builder import SchemaBuilder
        >>> schema = (
        ...     SchemaBuilder()
        ...     .add_dense_vector_field("embedding", dimension=768, metric="cosine")
        ...     .add_string_field("title", full_text_search={"language": "en"})
        ...     .build()
        ... )
    """

    def __init__(self) -> None:
        self._fields: dict[str, dict[str, Any]] = {}

    def add_dense_vector_field(
        self,
        name: str,
        *,
        dimension: int,
        metric: str,
        description: str | None = None,
        **additional_options: Any,
    ) -> SchemaBuilder:
        """Add a dense vector field for similarity search.

        A schema may contain at most one dense vector field; the server
        rejects schemas with more than one.

        Args:
            name: Field name. Replaces any existing field with the same name.
            dimension: Vector dimensionality. Must be between 1 and 20000
                inclusive; values outside that range are rejected client-side.
            metric: Distance metric — ``"cosine"``, ``"euclidean"``, or
                ``"dotproduct"``.
            description: Optional human-readable description.
            **additional_options: Extra parameters merged into the field dict
                last, for forward compatibility with new API features.

        Returns:
            ``self`` for method chaining.

        Raises:
            PineconeValueError: If ``dimension`` is outside the range
                1–20000 inclusive.
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
        self._fields[name] = field
        return self

    def add_sparse_vector_field(
        self,
        name: str,
        *,
        description: str | None = None,
        **additional_options: Any,
    ) -> SchemaBuilder:
        """Add a sparse vector field for keyword-weighted or learned-sparse search.

        The wire type is ``"sparse_vector"``. The metric is fixed at
        ``"dotproduct"`` server-side and is not user-configurable. A schema
        may contain at most one sparse vector field; the server rejects
        schemas with more than one.

        Args:
            name: Field name. Replaces any existing field with the same name.
            description: Optional human-readable description.
            **additional_options: Extra parameters merged into the field dict
                last, for forward compatibility with new API features.

        Returns:
            ``self`` for method chaining.
        """
        _validate_field_name(name)
        _validate_description(name, description)
        field: dict[str, Any] = {
            "type": "sparse_vector",
            "metric": "dotproduct",
        }
        if description is not None:
            field["description"] = description
        field.update(additional_options)
        self._fields[name] = field
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
        """Add a string field for full-text search (or, on pod indexes, filtering).

        Full-text search is enabled by passing ``full_text_search=True``, a
        ``full_text_search`` dict, or any of the typed FTS keyword arguments
        (``language``, ``stemming``, ``stop_words``).

        .. important::

           At API version ``2026-07``, a string field **without**
           ``full_text_search`` is a metadata-only declaration, and the
           server rejects it when creating managed or BYOC indexes
           (``400``: the schema only accepts fields used for search).
           Pod indexes are the exception — they still accept string
           fields without ``full_text_search`` as filterable metadata
           declarations. For managed and BYOC indexes, omit metadata-only
           fields from the schema and include the values in documents;
           they are indexed for filtering automatically at upsert time.

        When both ``full_text_search`` dict and keyword arguments are provided,
        the keyword arguments take precedence for the same key.

        ``lowercase`` and ``max_term_len`` are server-managed and cannot be
        configured via the SDK.

        Args:
            name: Field name. Replaces any existing field with the same name.
            full_text_search: ``True`` or ``{}`` to enable FTS with server
                defaults, a ``dict`` of FTS-config keys (``language``,
                ``stemming``, ``stop_words``, ``ngram``), or ``None``
                (default) to leave FTS disabled — valid only for pod
                indexes; see the note above.
            language: Language for FTS tokenisation and analysis. Accepts
                ISO short codes or long-form aliases. Both ``"en"`` and
                ``"english"`` are valid; the SDK normalises known
                long-form aliases to the short-code form on the wire.
                Codes known to the SDK at this version: ``ar``, ``da``,
                ``de``, ``el``, ``en``, ``es``, ``fi``, ``fr``, ``hu``,
                ``it``, ``nl``, ``no``, ``pt``, ``ro``, ``ru``, ``sv``,
                ``ta``, ``tr`` (and their long-form aliases: ``arabic``,
                ``danish``, ``german``, ``greek``, ``english``,
                ``spanish``, ``finnish``, ``french``, ``hungarian``,
                ``italian``, ``dutch``, ``norwegian``, ``portuguese``,
                ``romanian``, ``russian``, ``swedish``, ``tamil``,
                ``turkish``). The SDK does not validate this value
                against that list — unknown codes are passed through
                unchanged so newly-supported languages work without an
                SDK upgrade. The server is the source of truth.
            stemming: Enable word stemming. Required when ``stop_words=True``.
            stop_words: Enable stop-word filtering. Requires
                ``stemming=True``. Not all languages support stop words;
                the server will reject unsupported combinations — the SDK
                does not pre-validate that rule.
            filterable: Enable metadata-filter support. ``False`` values are
                omitted from the wire payload.
            description: Optional human-readable description.
            **additional_options: Extra parameters merged into the field dict
                last, for forward compatibility with new API features.

        Returns:
            ``self`` for method chaining.

        Raises:
            PineconeValueError: If ``stop_words=True`` is requested without
                ``stemming=True``, or if an ``ngram`` config is combined
                with ``stemming=True`` or ``stop_words=True``.

        Examples:
            .. code-block:: python

                # Enable FTS with server defaults:
                builder.add_string_field("title", full_text_search=True)

                # Enable FTS with explicit kwargs:
                builder.add_string_field(
                    "title", language="en", stemming=True, stop_words=True
                )

                # Character n-gram tokenization (e.g. substring/autocomplete):
                builder.add_string_field(
                    "title",
                    full_text_search={"ngram": {"min_gram": 2, "max_gram": 3}},
                )
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
        if filterable:
            field["filterable"] = filterable
        if description is not None:
            field["description"] = description
        field.update(additional_options)
        self._fields[name] = field
        return self

    def add_string_list_field(
        self,
        name: str,
        *,
        filterable: bool = False,
        description: str | None = None,
        **additional_options: Any,
    ) -> SchemaBuilder:
        """Add a list-of-strings field for metadata filtering (pod indexes only).

        .. important::

           At API version ``2026-07``, ``string_list`` is a metadata-only
           declaration, and the server rejects it when creating managed or
           BYOC indexes (``400``: the schema only accepts fields used for
           search) — regardless of ``filterable``. Pod indexes are the
           exception and still accept this declaration. For managed and
           BYOC indexes, include list-of-string values in documents
           instead; they are indexed for filtering automatically at
           upsert time.

        String-list fields store a list of strings per row — useful for
        tag-style metadata (e.g. ``["sci-fi", "mystery"]``) that should be
        filterable against individual elements.

        The wire type is ``"string_list"``. ``filterable=False`` is omitted
        from the wire payload; ``None`` values are omitted as well.

        Args:
            name: Field name. Replaces any existing field with the same name.
            filterable: Enable metadata-filter support.
            description: Optional human-readable description.
            **additional_options: Extra parameters merged into the field dict
                last, for forward compatibility with new API features.

        Returns:
            ``self`` for method chaining.

        Examples:
            .. code-block:: python

                builder.add_string_list_field("tags", filterable=True)
        """
        _validate_field_name(name)
        _validate_description(name, description)
        field: dict[str, Any] = {"type": "string_list"}
        if filterable:
            field["filterable"] = filterable
        if description is not None:
            field["description"] = description
        field.update(additional_options)
        self._fields[name] = field
        return self

    def add_boolean_field(
        self,
        name: str,
        *,
        filterable: bool = False,
        description: str | None = None,
        **additional_options: Any,
    ) -> SchemaBuilder:
        """Add a boolean field for metadata filtering (pod indexes only).

        .. important::

           At API version ``2026-07``, ``boolean`` is a metadata-only
           declaration, and the server rejects it when creating managed or
           BYOC indexes (``400``: the schema only accepts fields used for
           search) — regardless of ``filterable``. Pod indexes are the
           exception and still accept this declaration. For managed and
           BYOC indexes, include boolean values in documents instead; they
           are indexed for filtering automatically at upsert time.

        The wire type is ``"boolean"``. ``filterable=False`` is omitted from
        the wire payload; ``None`` description is omitted as well.

        Args:
            name: Field name. Replaces any existing field with the same name.
            filterable: Enable metadata-filter support on this field.
            description: Optional human-readable description.
            **additional_options: Extra parameters merged into the field dict
                last, for forward compatibility with new API features.

        Returns:
            ``self`` for method chaining.

        Examples:
            .. code-block:: python

                builder.add_boolean_field("is_published", filterable=True)
        """
        _validate_field_name(name)
        _validate_description(name, description)
        field: dict[str, Any] = {"type": "boolean"}
        if filterable:
            field["filterable"] = filterable
        if description is not None:
            field["description"] = description
        field.update(additional_options)
        self._fields[name] = field
        return self

    def add_float_field(
        self,
        name: str,
        *,
        filterable: bool = False,
        description: str | None = None,
        **additional_options: Any,
    ) -> SchemaBuilder:
        """Add a numeric field for metadata filtering (pod indexes only).

        .. important::

           At API version ``2026-07``, ``float`` is a metadata-only
           declaration, and the server rejects it when creating managed or
           BYOC indexes (``400``: the schema only accepts fields used for
           search) — regardless of ``filterable``. Pod indexes are the
           exception and still accept this declaration. For managed and
           BYOC indexes, include numeric values in documents instead; they
           are indexed for filtering automatically at upsert time.

        The wire type is ``"float"``. The Pinecone API does not have a
        separate integer type; integers are stored and filtered as
        double-precision floats.

        Args:
            name: Field name. Replaces any existing field with the same name.
            filterable: Enable filtering on this field. ``False`` is omitted
                from the wire payload.
            description: Optional human-readable description.
            **additional_options: Extra parameters merged into the field dict
                last, for forward compatibility with new API features.

        Returns:
            ``self`` for method chaining.
        """
        _validate_field_name(name)
        _validate_description(name, description)
        field: dict[str, Any] = {"type": "float"}
        if filterable:
            field["filterable"] = filterable
        if description is not None:
            field["description"] = description
        field.update(additional_options)
        self._fields[name] = field
        return self

    def add_custom_field(
        self,
        name: str,
        field_definition: dict[str, Any],
    ) -> SchemaBuilder:
        """Escape hatch — store a raw field dict verbatim.

        Use when you need a field type the SDK does not yet model, or when
        experimenting with new API features before the SDK adds support.
        Only the field name is validated; the definition is not.

        Args:
            name: Field name. Replaces any existing field with the same name.
            field_definition: Complete field definition dict; stored as-is.

        Returns:
            ``self`` for method chaining.
        """
        _validate_field_name(name)
        self._fields[name] = field_definition
        return self

    def build(self) -> dict[str, dict[str, Any]]:
        """Return the completed schema dict.

        Returns a copy of the internal field dict so that subsequent
        ``add_*`` calls do not mutate a previously built result.

        The server requires at least one searched field (``dense_vector``,
        ``sparse_vector``, or ``string`` with ``full_text_search``) per
        schema; the builder does not enforce that here so partial schemas
        can be built and inspected.

        Returns:
            ``{"fields": {name: field_dict, ...}}`` ready to pass as the
            ``schema`` argument when creating an index.
        """
        return {"fields": dict(self._fields)}


__all__ = ["SchemaBuilder"]
