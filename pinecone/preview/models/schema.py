"""Preview schema field response models (2026-01.alpha API).

These models represent typed schema fields returned by the index describe
and list endpoints when using the preview API version.  They form a tagged
union so msgspec can deserialise the ``type`` discriminator field at decode
time.
"""

from __future__ import annotations

from typing import Any

from msgspec import Struct

__all__ = [
    "PreviewBooleanField",
    "PreviewDenseVectorField",
    "PreviewFullTextSearchConfig",
    "PreviewIntegerField",
    "PreviewLegacyIntegerField",
    "PreviewSchema",
    "PreviewSchemaField",
    "PreviewSemanticTextField",
    "PreviewSparseVectorField",
    "PreviewStringField",
    "PreviewStringListField",
]


class PreviewDenseVectorField(Struct, tag="dense_vector", tag_field="type", kw_only=True):
    """Dense vector field definition.

    .. admonition:: Preview
       :class: warning

       Uses Pinecone API version ``2026-01.alpha``.
       Preview surface is not covered by SemVer — signatures and behavior
       may change in any minor SDK release. Pin your SDK version when
       relying on preview features.

    Dense vectors are the traditional fixed-length vectors used for
    similarity search. Each dimension has a floating-point value.

    Attributes:
        dimension: Number of dimensions in the vector, or ``None`` if not
            specified.
        metric: Distance metric (``"cosine"``, ``"euclidean"``, or
            ``"dotproduct"``), or ``None`` if not specified.
        description: Optional human-readable description of the field.

    Note:
        The ``type`` field is automatically set to ``"dense_vector"`` by
        msgspec's tagged union system and should not be included explicitly.
    """

    dimension: int | None = None
    metric: str | None = None
    description: str | None = None


class PreviewSparseVectorField(Struct, tag="sparse_vector", tag_field="type", kw_only=True):
    """Sparse vector field definition.

    .. admonition:: Preview
       :class: warning

       Uses Pinecone API version ``2026-01.alpha``.
       Preview surface is not covered by SemVer — signatures and behavior
       may change in any minor SDK release. Pin your SDK version when
       relying on preview features.

    Sparse vectors represent most values as zero and are stored as
    (indices, values) pairs.  Useful for keyword-based search (e.g. BM25).

    Attributes:
        metric: Distance metric (typically ``"dotproduct"``), or ``None``
            if not specified.
        description: Optional human-readable description of the field.

    Note:
        The ``type`` field is automatically set to ``"sparse_vector"`` by
        msgspec's tagged union system.
    """

    metric: str | None = None
    description: str | None = None


class PreviewSemanticTextField(Struct, tag="semantic_text", tag_field="type", kw_only=True):
    """Semantic text field with integrated embedding.

    .. admonition:: Preview
       :class: warning

       Uses Pinecone API version ``2026-01.alpha``.
       Preview surface is not covered by SemVer — signatures and behavior
       may change in any minor SDK release. Pin your SDK version when
       relying on preview features.

    Semantic text fields automatically embed text using a specified model,
    eliminating the need to generate embeddings separately.

    Attributes:
        model: Embedding model name (e.g. ``"multilingual-e5-large"``), or
            ``None`` if not specified.
        metric: Distance metric (typically ``"cosine"``), or ``None`` if
            not specified.
        dimension: Vector dimension, or ``None`` if not specified.
        description: Optional human-readable description of the field.
        read_parameters: Parameters forwarded to the embedding model on
            read operations (e.g. ``{"input_type": "query"}``), or
            ``None``.
        write_parameters: Parameters forwarded to the embedding model on
            write operations (e.g. ``{"input_type": "passage"}``), or
            ``None``.

    Note:
        The ``type`` field is automatically set to ``"semantic_text"`` by
        msgspec's tagged union system.
    """

    model: str | None = None
    metric: str | None = None
    dimension: int | None = None
    description: str | None = None
    read_parameters: dict[str, Any] | None = None
    write_parameters: dict[str, Any] | None = None


class PreviewFullTextSearchConfig(Struct, kw_only=True):
    """Full-text search configuration for a string field.

    .. admonition:: Preview
       :class: warning

       Uses Pinecone API version ``2026-01.alpha``.
       Preview surface is not covered by SemVer — signatures and behavior
       may change in any minor SDK release. Pin your SDK version when
       relying on preview features.

    Presence of this object on a :class:`PreviewStringField` indicates the
    field is full-text searchable; absence means it is not. All keys are
    optional — an empty config (``PreviewFullTextSearchConfig()``) is
    valid and requests the server defaults.

    Attributes:
        language: BCP-47 language code (e.g. ``"en"``). When ``None``, the
            server applies its default (``"en"``).
        stemming: Whether to stem tokens to root form during indexing.
            When ``None``, the server applies its default (``False``).
        lowercase: Whether to lowercase tokens before indexing. When
            ``None``, the server applies its default (``True``).
        max_term_len: Maximum term length for indexing. When ``None``, the
            server applies its default (``40``).
        stop_words: Whether to filter stop words during indexing. When
            ``None``, the server applies its default (``False``).
    """

    language: str | None = None
    stemming: bool | None = None
    lowercase: bool | None = None
    max_term_len: int | None = None
    stop_words: bool | None = None


class PreviewStringField(Struct, tag="string", tag_field="type", kw_only=True):
    """String field for metadata or full-text search.

    .. admonition:: Preview
       :class: warning

       Uses Pinecone API version ``2026-01.alpha``.
       Preview surface is not covered by SemVer — signatures and behavior
       may change in any minor SDK release. Pin your SDK version when
       relying on preview features.

    String fields can be used for filtering (``filterable=True``) and/or
    full-text search (pass a :class:`PreviewFullTextSearchConfig` via
    ``full_text_search``). Presence of ``full_text_search`` — even an
    empty config — indicates the field is full-text searchable.

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
    full_text_search: PreviewFullTextSearchConfig | None = None


class PreviewStringListField(Struct, tag="string_list", tag_field="type", kw_only=True):
    """List-of-strings field for metadata filtering.

    .. admonition:: Preview
       :class: warning

       Uses Pinecone API version ``2026-01.alpha``.
       Preview surface is not covered by SemVer — signatures and behavior
       may change in any minor SDK release. Pin your SDK version when
       relying on preview features.

    Stores a list of strings per row — useful for tag-style metadata
    (e.g. ``["sci-fi", "mystery"]``) that should be filterable against
    individual elements.

    The wire type tag is ``"string_list"``. The prior tag ``"string[]"``
    is no longer accepted by the server.

    Attributes:
        description: Optional human-readable description of the field.
        filterable: Whether the field can be used in metadata filters.
            Defaults to ``False``.

    Note:
        The ``type`` field is automatically set to ``"string_list"`` by
        msgspec's tagged union system and should not be included
        explicitly.
    """

    description: str | None = None
    filterable: bool = False


class PreviewBooleanField(Struct, tag="boolean", tag_field="type", kw_only=True):
    """Boolean field for metadata filtering.

    .. admonition:: Preview
       :class: warning

       Uses Pinecone API version ``2026-01.alpha``.
       Preview surface is not covered by SemVer — signatures and behavior
       may change in any minor SDK release. Pin your SDK version when
       relying on preview features.

    Attributes:
        description: Optional human-readable description.
        filterable: Whether the field can be used in metadata filters.

    Note:
        The ``type`` field is automatically set to ``"boolean"`` by
        msgspec's tagged union system.
    """

    description: str | None = None
    filterable: bool = False


class PreviewLegacyIntegerField(Struct, tag="integer", tag_field="type", kw_only=True):
    """Legacy integer field (stored type from older API versions).

    .. admonition:: Preview
       :class: warning

       Uses Pinecone API version ``2026-01.alpha``.
       Preview surface is not covered by SemVer — signatures and behavior
       may change in any minor SDK release. Pin your SDK version when
       relying on preview features.

    New indexes use ``float`` for numeric fields; ``integer`` only
    appears in indexes migrated from older API versions.

    Attributes:
        description: Optional human-readable description.
        filterable: Whether the field can be used in metadata filters.

    Note:
        The ``type`` field is automatically set to ``"integer"`` by
        msgspec's tagged union system.
    """

    description: str | None = None
    filterable: bool = False


class PreviewIntegerField(Struct, tag="float", tag_field="type", kw_only=True):
    """Numeric (float) field for metadata filtering.

    .. admonition:: Preview
       :class: warning

       Uses Pinecone API version ``2026-01.alpha``.
       Preview surface is not covered by SemVer — signatures and behavior
       may change in any minor SDK release. Pin your SDK version when
       relying on preview features.

    Numeric fields store double-precision floating-point values and can be
    used for range filtering (e.g. ``year >= 2020``).  There is no separate
    integer type in the API — integers are stored and filtered as floats.
    Created via :meth:`~pinecone.schema_builder.SchemaBuilder.add_float_field`.

    Attributes:
        description: Optional human-readable description of the field.
        filterable: Whether the field can be used in metadata filters.
            Defaults to ``False``.

    Note:
        The wire type is ``"float"`` — the API normalises both
        ``"number"`` and ``"float"`` to ``"float"`` in responses.  The
        ``type`` field is automatically set to ``"float"`` by msgspec's
        tagged union system.
    """

    description: str | None = None
    filterable: bool = False


#: Union of all supported preview schema field types.
#: Use this as the decode target when parsing a single field from JSON.
PreviewSchemaField = (
    PreviewDenseVectorField
    | PreviewSparseVectorField
    | PreviewSemanticTextField
    | PreviewStringField
    | PreviewStringListField
    | PreviewBooleanField
    | PreviewLegacyIntegerField
    | PreviewIntegerField
)


class PreviewSchema(Struct, kw_only=True):
    """Index schema definition (preview).

    .. admonition:: Preview
       :class: warning

       Uses Pinecone API version ``2026-01.alpha``.
       Preview surface is not covered by SemVer — signatures and behavior
       may change in any minor SDK release. Pin your SDK version when
       relying on preview features.

    The schema defines all fields in the index, including vector, text, and
    metadata fields.

    Attributes:
        fields: Mapping of field name to its typed field definition.
    """

    fields: dict[str, PreviewSchemaField]
