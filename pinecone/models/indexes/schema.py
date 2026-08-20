"""Index schema response models (2026-07 API).

These models represent typed schema fields returned by the index describe
and list endpoints.  They form a tagged union so msgspec can deserialise
the ``type`` discriminator field at decode time.

Legacy metadata fields (from indexes created before typed schemas were
introduced) carry no ``type`` key on the wire and are normalised to
:class:`LegacyMetadataField` during decode.
"""

from __future__ import annotations

from typing import Any

import msgspec
from msgspec import Struct

__all__ = [
    "BooleanField",
    "DenseVectorField",
    "FloatField",
    "FullTextSearchConfig",
    "IndexSchema",
    "IndexSchemaField",
    "IntegerField",
    "LegacyMetadataField",
    "NgramConfig",
    "SemanticTextField",
    "SparseVectorField",
    "StringField",
    "StringListField",
]

#: Discriminator injected during decode for schema fields that carry no
#: ``type`` key on the wire. Never sent by the API.
UNTYPED_FIELD_TAG = "__untyped__"


def _strip_untyped_tags(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {
            k: _strip_untyped_tags(v)
            for k, v in obj.items()
            if not (k == "type" and v == UNTYPED_FIELD_TAG)
        }
    if isinstance(obj, list):
        return [_strip_untyped_tags(item) for item in obj]
    return obj


def _tag_untyped_schema_fields(obj: Any) -> Any:
    """Inject the internal discriminator into ``schema.fields`` entries lacking ``type``.

    Legacy metadata fields arrive with no ``type`` discriminator; msgspec
    tagged unions cannot decode them without one.  Mutates and returns
    *obj*; a non-dict or a payload without a dict-shaped ``schema.fields``
    passes through untouched.
    """
    if not isinstance(obj, dict):
        return obj
    schema = obj.get("schema")
    if isinstance(schema, dict):
        fields = schema.get("fields")
        if isinstance(fields, dict):
            for field in fields.values():
                if isinstance(field, dict) and "type" not in field:
                    field["type"] = UNTYPED_FIELD_TAG
    return obj


class DenseVectorField(Struct, tag="dense_vector", tag_field="type", kw_only=True):
    """Dense vector field definition.

    Dense vectors are fixed-length floating-point vectors used for
    approximate nearest-neighbor (ANN) similarity search.

    Attributes:
        dimension: Number of dimensions in the vector (1-20000).
        metric: Distance metric — ``"cosine"``, ``"dotproduct"``, or
            ``"euclidean"``.
        description: Optional human-readable description of the field.
            Always present in responses; ``None`` when no description
            was given.

    Note:
        The ``type`` field is automatically set to ``"dense_vector"`` by
        msgspec's tagged union system and should not be included explicitly.
    """

    dimension: int
    metric: str
    description: str | None = None


class SparseVectorField(Struct, tag="sparse_vector", tag_field="type", kw_only=True):
    """Sparse vector field definition.

    Sparse vectors represent most values as zero and are stored as
    (indices, values) pairs.  Useful for keyword-based search (e.g. BM25).

    Attributes:
        description: Optional human-readable description of the field.

    Note:
        The ``type`` field is automatically set to ``"sparse_vector"`` by
        msgspec's tagged union system.
    """

    description: str | None = None


class SemanticTextField(Struct, tag="semantic_text", tag_field="type", kw_only=True):
    """Semantic text field with integrated embedding.

    Semantic text fields automatically embed text using a specified model,
    eliminating the need to generate embeddings separately.  In the
    ``2026-07`` API this field type cannot be declared at index creation;
    it appears in responses for indexes that already carry one (including
    indexes created via ``create_index_for_model``).

    Attributes:
        model: Embedding model name (e.g. ``"multilingual-e5-large"``).
        metric: Distance metric (``"cosine"``, ``"dotproduct"``, or
            ``"euclidean"``), or ``None`` to use the model default.
        description: Optional human-readable description of the field.
        read_parameters: Parameters forwarded to the embedding model on
            read operations (e.g. ``{"input_type": "query"}``), or ``None``.
        write_parameters: Parameters forwarded to the embedding model on
            write operations (e.g. ``{"input_type": "passage"}``), or ``None``.

    Note:
        The ``type`` field is automatically set to ``"semantic_text"`` by
        msgspec's tagged union system.
    """

    model: str
    metric: str | None = None
    description: str | None = None
    read_parameters: dict[str, Any] | None = None
    write_parameters: dict[str, Any] | None = None


class NgramConfig(Struct, kw_only=True):
    """Character n-gram tokenization configuration for a string field.

    When present, the field is tokenized into character n-grams instead of
    words (useful for substring matching and autocomplete).  Cannot be
    combined with ``stemming`` or ``stop_words``.

    Attributes:
        min_gram: Minimum n-gram length (1-10, no greater than ``max_gram``).
        max_gram: Maximum n-gram length (1-10, no less than ``min_gram``).
        prefix_only: When ``True``, only prefix n-grams anchored at the
            start of the token are generated. Defaults to ``False``.
    """

    min_gram: int
    max_gram: int
    prefix_only: bool = False


class FullTextSearchConfig(Struct, kw_only=True):
    """Full-text search configuration for a string field.

    Presence of this object on a :class:`StringField` indicates the field
    is full-text searchable; absence means it is not.  All keys are
    optional on create — an empty config (``FullTextSearchConfig()``) is
    valid and requests the server defaults.  Responses always carry
    ``language``, ``stemming``, and ``stop_words``.

    Attributes:
        language: Language used for text analysis, as a two-letter code or
            English name (e.g. ``"en"`` or ``"english"``). When ``None``,
            the server applies its default (``"en"``).
        stemming: Whether to stem tokens to root form during indexing.
            When ``None``, the server applies its default (``False``).
        stop_words: Whether to filter stop words during indexing. Requires
            ``stemming=True``. When ``None``, the server applies its
            default (``False``).
        ngram: Character n-gram tokenization configuration, or ``None``
            for word-based tokenization. Cannot be combined with
            ``stemming`` or ``stop_words``.
    """

    language: str | None = None
    stemming: bool | None = None
    stop_words: bool | None = None
    ngram: NgramConfig | None = None


class StringField(Struct, tag="string", tag_field="type", kw_only=True):
    """String field for full-text search or metadata filtering.

    In responses, string fields configured for full-text search include a
    ``full_text_search`` object; string fields used for metadata filtering
    only include a ``filterable`` flag.  At index creation, a string field
    must include ``full_text_search`` — metadata-only fields are not
    declared in the schema (pass them as record metadata instead).

    Attributes:
        description: Optional human-readable description of the field.
        filterable: Whether the field can be used in metadata filters.
            Defaults to ``False``.
        full_text_search: Full-text search configuration. Presence (even
            an empty config) indicates the field is full-text searchable;
            absence (``None``) means it is not.

    Note:
        The ``type`` field is automatically set to ``"string"`` by
        msgspec's tagged union system.
    """

    description: str | None = None
    filterable: bool = False
    full_text_search: FullTextSearchConfig | None = None


class StringListField(Struct, tag="string_list", tag_field="type", kw_only=True):
    """List-of-strings field for metadata filtering.

    Stores a list of strings per record — useful for tag-style metadata
    (e.g. ``["sci-fi", "mystery"]``) that should be filterable against
    individual elements.  Not declared at index creation; appears in
    responses for fields indexed automatically at upsert time.

    Attributes:
        description: Optional human-readable description of the field.
        filterable: Whether the field can be used in metadata filters.
            Defaults to ``False``.

    Note:
        The ``type`` field is automatically set to ``"string_list"`` by
        msgspec's tagged union system.
    """

    description: str | None = None
    filterable: bool = False


class BooleanField(Struct, tag="boolean", tag_field="type", kw_only=True):
    """Boolean field for metadata filtering.

    Not declared at index creation; appears in responses for fields
    indexed automatically at upsert time.

    Attributes:
        description: Optional human-readable description.
        filterable: Whether the field can be used in metadata filters.

    Note:
        The ``type`` field is automatically set to ``"boolean"`` by
        msgspec's tagged union system.
    """

    description: str | None = None
    filterable: bool = False


class IntegerField(Struct, tag="integer", tag_field="type", kw_only=True):
    """Legacy integer field.

    Numeric values are normalised to ``float`` at upsert time in current
    indexes; ``integer`` appears only in responses for indexes that
    pre-date that normalisation.

    Attributes:
        description: Optional human-readable description.
        filterable: Whether the field can be used in metadata filters.

    Note:
        The ``type`` field is automatically set to ``"integer"`` by
        msgspec's tagged union system.
    """

    description: str | None = None
    filterable: bool = False


class FloatField(Struct, tag="float", tag_field="type", kw_only=True):
    """Numeric (float) field for metadata filtering.

    Numeric fields store double-precision floating-point values and can be
    used for range filtering (e.g. ``year >= 2020``).  There is no separate
    integer type in the API — integers are stored and filtered as floats.
    Not declared at index creation; appears in responses for fields
    indexed automatically at upsert time.

    Attributes:
        description: Optional human-readable description of the field.
        filterable: Whether the field can be used in metadata filters.
            Defaults to ``False``.

    Note:
        The ``type`` field is automatically set to ``"float"`` by
        msgspec's tagged union system.
    """

    description: str | None = None
    filterable: bool = False


class LegacyMetadataField(Struct, tag=UNTYPED_FIELD_TAG, tag_field="type", kw_only=True):
    """Untyped metadata field from indexes that pre-date typed schemas.

    The original data type of the field (string, float, boolean, etc.) was
    not recorded — only the ``filterable`` flag is available.  On the wire
    these fields carry **no** ``type`` key.  This field type never appears
    in new indexes.

    Attributes:
        filterable: Whether the field is indexed for metadata filtering.

    Note:
        msgspec requires a discriminator for union decoding, so this class
        carries the internal tag ``"__untyped__"``.  The tag is an SDK
        artifact: it is stripped by :meth:`IndexSchema.to_dict` and never
        appears in API traffic, but ``msgspec.json.encode`` output of this
        class does include it.
    """

    filterable: bool


#: Union of all schema field types appearing in index responses.
#: Use this as the decode target when parsing a single field from JSON.
IndexSchemaField = (
    DenseVectorField
    | SparseVectorField
    | SemanticTextField
    | StringField
    | StringListField
    | BooleanField
    | IntegerField
    | FloatField
    | LegacyMetadataField
)


class IndexSchema(Struct, kw_only=True):
    """Index schema definition.

    The schema defines all fields in the index, including vector, text, and
    metadata fields.

    Attributes:
        fields: Mapping of field name to its typed field definition.
    """

    fields: dict[str, IndexSchemaField]

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict representation.

        Typed fields include their ``type`` discriminator; legacy untyped
        fields are emitted without a ``type`` key, matching the wire format.
        """
        result: dict[str, Any] = _strip_untyped_tags(msgspec.to_builtins(self))
        return result
