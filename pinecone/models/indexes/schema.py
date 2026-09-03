"""The field types an index schema is built from.

:class:`IndexSchema` is the entry point and carries the canonical account of
what a field type is, which ones exist, and which of them you can declare when
creating an index. The classes below are the per-type detail.

Fields from indexes created before typed schemas carry no ``type`` on the wire
and are normalised to :class:`LegacyMetadataField` when decoded.
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
    """A fixed-width vector of floats, scored by similarity.

    The workhorse field type, and what a semantic search index is built on.
    Declare one per embedding model you intend to search with. Its ``type``
    in a ``schema=`` dict is ``"dense_vector"``; constructing this class
    instead sets that for you.

    Attributes:
        dimension: Width of the vector, matching the embedding model that
            produces it, e.g. ``1536``. Must be between 1 and 20000 —
            :class:`~pinecone.schema_builder.SchemaBuilder` rejects anything
            outside that range before the request leaves the client.
        metric: How similarity is scored — ``"cosine"``, ``"dotproduct"``, or
            ``"euclidean"``. Fixed for the life of the field; see
            :class:`~pinecone.models.enums.Metric` for how to choose.
        description: Free-text note about the field, or ``None`` when none was
            given. Always present in responses.

    Examples:
        The ``schema=`` entry that declares one:

        .. code-block:: python

            {"embedding": {"type": "dense_vector", "dimension": 1536,
                           "metric": "cosine"}}
    """

    dimension: int
    metric: str
    description: str | None = None


class SparseVectorField(Struct, tag="sparse_vector", tag_field="type", kw_only=True):
    """Variable-length index/value pairs, for keyword-style scoring.

    Where a dense field compares meaning, a sparse field compares terms, so an
    index that needs both declares both. There is nothing to configure: a
    sparse field has no dimension because it is variable-length, and no metric
    because sparse scoring is not adjustable. Its ``type`` in a ``schema=``
    dict is ``"sparse_vector"``.

    A hybrid index has to declare its sparse field up front. The field cannot
    be added later by ``configure``, so an index created without one has to be
    recreated.

    Attributes:
        description: Free-text note about the field, or ``None``.

    Examples:
        The ``schema=`` entry that declares one:

        .. code-block:: python

            {"keywords": {"type": "sparse_vector"}}
    """

    description: str | None = None


class SemanticTextField(Struct, tag="semantic_text", tag_field="type", kw_only=True):
    """Text that Pinecone embeds for you, on write and on read.

    With a semantic text field you upsert and query plain text and never
    handle a vector yourself — the difference from :class:`DenseVectorField`,
    where producing the embedding is your job.

    **Response-only, and there is exactly one way to get one:**
    :meth:`~pinecone.client.indexes.Indexes.create_for_model`, which names the
    field after the ``field_map`` text entry. Writing ``"type":
    "semantic_text"`` into a ``schema=`` you pass to :meth:`create
    <pinecone.client.indexes.Indexes.create>` is not the other way — the
    client sends it and the server rejects it, and because one bad field fails
    the whole schema the index is not created at all. The 9.x
    ``spec=IntegratedSpec(...)`` route is gone too; it raises
    :exc:`~pinecone.errors.exceptions.PineconeTypeError` naming
    ``create_for_model`` as the replacement. The model cannot be changed once
    the index exists.

    Attributes:
        model: Embedding model doing the work, e.g.
            ``"multilingual-e5-large"``.
        metric: How similarity is scored, or ``None`` when the field uses the
            model's own default.
        description: Free-text note about the field, or ``None``.
        read_parameters: Extra arguments passed to the model when embedding a
            query, e.g. ``{"input_type": "query"}``, or ``None``.
        write_parameters: Extra arguments passed to the model when embedding
            an upsert, e.g. ``{"input_type": "passage"}``, or ``None``.
    """

    model: str
    metric: str | None = None
    description: str | None = None
    read_parameters: dict[str, Any] | None = None
    write_parameters: dict[str, Any] | None = None


class NgramConfig(Struct, kw_only=True):
    """Tokenize a string field into character n-grams instead of words.

    Word tokenization matches whole words, so a search for ``head`` misses
    ``headphones``. N-gram tokenization indexes runs of characters instead,
    which is what makes substring matching and autocomplete work. It cannot be
    combined with ``stemming`` or ``stop_words``.

    Attributes:
        min_gram: Shortest run of characters to index. The shorter this is,
            the more aggressively short queries match.
        max_gram: Longest run of characters to index; no smaller than
            ``min_gram``.
        prefix_only: When ``True``, index only the runs anchored at the start
            of the token, which is what autocomplete wants. Defaults to
            ``False``.

    Examples:
        Substring matching on a product title:

        .. code-block:: python

            {"title": {"type": "string", "full_text_search": {
                "ngram": {"min_gram": 2, "max_gram": 3}}}}
    """

    min_gram: int
    max_gram: int
    prefix_only: bool = False


class FullTextSearchConfig(Struct, kw_only=True):
    """How a string field's text is analysed for full-text search.

    Its presence on a :class:`StringField` is what makes the field full-text
    searchable at all; ``None`` means it is not. Every key is optional, so an
    empty ``FullTextSearchConfig()`` is a valid way to say "searchable, server
    defaults please". Responses always report ``language``, ``stemming`` and
    ``stop_words`` as resolved values.

    Attributes:
        language: Language whose analysis rules to use, as a two-letter code
            or its English name — ``"en"`` and ``"english"`` are both
            accepted. ``None`` takes the server default of English.
        stemming: Fold tokens to their root form, so ``running`` matches
            ``run``. ``None`` takes the server default of off.
        stop_words: Drop common words like ``the`` from the index. Requires
            ``stemming=True``, and is not supported for every language — the
            rejection names the unsupported language by its English name
            rather than the code you sent. ``None`` takes the server default
            of off.
        ngram: A :class:`NgramConfig` to index character runs instead of
            words, or ``None`` for word tokenization. Mutually exclusive with
            ``stemming`` and ``stop_words``.
    """

    language: str | None = None
    stemming: bool | None = None
    stop_words: bool | None = None
    ngram: NgramConfig | None = None


class StringField(Struct, tag="string", tag_field="type", kw_only=True):
    """Text, either full-text searchable or filterable — never both.

    A string field you declare on create must carry a ``full_text_search``
    config, because search is the only reason the schema takes a string field.
    Text you only want to filter on is not declared at all: put it in the
    documents you upsert. Its ``type`` in a ``schema=`` dict is ``"string"``.

    In responses, a searchable field reports its ``full_text_search`` object
    and a filter-only field reports just ``filterable``.

    Attributes:
        description: Free-text note about the field, or ``None``.
        filterable: Whether the field can be used in metadata filters.
            Defaults to ``False``. Sending ``filterable=True`` alongside
            ``full_text_search`` does not give you both: the server keeps the
            filter, discards the search configuration, and reports no error
            for doing so, so the field silently comes back unsearchable.
        full_text_search: A :class:`FullTextSearchConfig` — its presence, even
            empty, is what makes the field searchable; ``None`` means it is
            not.

    Examples:
        The ``schema=`` entry that declares one:

        .. code-block:: python

            {"title": {"type": "string", "full_text_search": {}}}
    """

    description: str | None = None
    filterable: bool = False
    full_text_search: FullTextSearchConfig | None = None


class StringListField(Struct, tag="string_list", tag_field="type", kw_only=True):
    """Tag-style metadata: a list of strings, filterable per element.

    A filter on this field matches if any element matches, which is what makes
    it right for tags like ``["sci-fi", "mystery"]``.

    **Response-only.** You read one back for a field the server indexed for
    you at upsert time; sending ``string_list`` in a ``schema=`` on create is
    rejected, and one rejected field fails the whole schema. Upsert the list
    as an ordinary document value instead.

    Attributes:
        description: Free-text note about the field, or ``None``.
        filterable: Whether the field can be used in metadata filters.
    """

    description: str | None = None
    filterable: bool = False


class BooleanField(Struct, tag="boolean", tag_field="type", kw_only=True):
    """Boolean metadata, filterable.

    **Response-only.** You read one back for a field the server indexed for
    you at upsert time; sending ``boolean`` in a ``schema=`` on create is
    rejected, and one rejected field fails the whole schema. Upsert the value
    as an ordinary document value instead.

    Attributes:
        description: Free-text note about the field, or ``None``.
        filterable: Whether the field can be used in metadata filters.
    """

    description: str | None = None
    filterable: bool = False


class IntegerField(Struct, tag="integer", tag_field="type", kw_only=True):
    """Integer metadata on an index that predates numeric normalisation.

    **Response-only, and the one field type with a sharp edge.** Numbers are
    normalised to ``float`` at upsert time now, so ``integer`` only ever comes
    back from an older index. There is no ``integer`` type in the create
    schema, and sending one is rejected with a plain-text body rather than a
    structured API error — so the exception you catch will not tell you which
    field was at fault.

    This matters for describe-then-create: a schema read off an old index
    cannot be handed straight to :meth:`create
    <pinecone.client.indexes.Indexes.create>`. Drop its integer fields, since
    numeric metadata is indexed for filtering automatically at upsert time.
    :class:`~pinecone.schema_builder.SchemaBuilder` has no method for this
    type and refuses ``{"type": "integer"}`` passed through
    :meth:`~pinecone.schema_builder.SchemaBuilder.add_custom_field`, so
    building the schema that way fails client-side with an explanation
    instead.

    Attributes:
        description: Free-text note about the field, or ``None``.
        filterable: Whether the field can be used in metadata filters.
    """

    description: str | None = None
    filterable: bool = False


class FloatField(Struct, tag="float", tag_field="type", kw_only=True):
    """Numeric metadata, filterable and range-comparable.

    Double-precision throughout, which is why range filters like
    ``year >= 2020`` work on it and why there is no separate integer type:
    integers are stored and filtered as floats, and ``float`` is the only
    numeric type name the API uses.

    **Response-only.** You read one back for a field the server indexed for
    you at upsert time; sending ``float`` in a ``schema=`` on create is
    rejected, and one rejected field fails the whole schema. Upsert the number
    as an ordinary document value instead.

    Attributes:
        description: Free-text note about the field, or ``None``.
        filterable: Whether the field can be used in metadata filters.
    """

    description: str | None = None
    filterable: bool = False


class LegacyMetadataField(Struct, tag=UNTYPED_FIELD_TAG, tag_field="type", kw_only=True):
    """A metadata field from an index older than typed schemas.

    These fields carry no ``type`` at all on the wire, and their original data
    type was never recorded, so ``filterable`` is all there is to read. New
    indexes never produce one.

    Attributes:
        filterable: Whether the field is indexed for metadata filtering.

    Note:
        Decoding a union needs a discriminator, so instances carry the
        internal tag ``"__untyped__"``. :meth:`IndexSchema.to_dict` strips it
        and it never reaches the API, but ``msgspec.json.encode`` of this
        class does emit it.
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
    """Every field an index has, and what each one can do.

    The schema is where an index's shape is declared. You pass it as
    ``schema=`` when creating an index and read it back from
    :attr:`IndexModel.schema <pinecone.models.indexes.index.IndexModel.schema>`
    afterwards. Dimension, metric and vector type live inside a field
    declaration rather than on the index, which is what lets one index carry a
    dense field, a sparse field and searchable text at the same time.

    Each entry in :attr:`fields` names a *field type* through its ``type``
    key, and that type is what decides how the field can be searched.
    **A ``schema=`` you pass to :meth:`create
    <pinecone.client.indexes.Indexes.create>` declares searchable fields, and
    only those** — exactly the three types below. Metadata you merely want to
    filter on stays out of it: put it in the documents you upsert and it is
    indexed for filtering automatically. Pass a filter-only field anyway and
    the client rejects the call with
    :exc:`~pinecone.errors.exceptions.PineconeValueError`, saying the schema
    "looks like a 9.x metadata schema" because none of its fields carry a
    ``type``.

    The remaining types are things you read back, never things you ask for.
    Declaring one is rejected, and because one bad field fails the whole
    schema, a single response-only field turns the entire create call into an
    error rather than being ignored — which is what makes a describe-then-create
    round-trip fail.

    Declarable when you create an index:

    ``dense_vector``
        A fixed-width vector of floats, scored by a similarity metric. Carries
        ``dimension`` and ``metric`` — see :class:`DenseVectorField`.
    ``sparse_vector``
        Variable-length index/value pairs for keyword-style scoring, with no
        dimension and no choice of metric — see :class:`SparseVectorField`.
    ``string``
        Text made full-text searchable by a ``full_text_search`` config — see
        :class:`StringField`.

    Read back but not declarable — sending one on create is rejected:

    ``semantic_text``
        Text Pinecone embeds for you on write and on read.
        :meth:`~pinecone.client.indexes.Indexes.create_for_model` is the only
        way to get one — see :class:`SemanticTextField`.
    ``float``, ``boolean``, ``string_list``
        Numeric, boolean and tag-style metadata, indexed for filtering
        automatically at upsert time — see :class:`FloatField`,
        :class:`BooleanField`, :class:`StringListField`.
    ``integer``
        Numeric metadata on indexes predating the normalisation of numbers to
        float — see :class:`IntegerField`.

    A field from an index older than typed schemas arrives with no ``type`` at
    all and becomes a :class:`LegacyMetadataField`.

    .. note::
       :meth:`~pinecone.client.indexes.Indexes.create_for_model` also takes a
       ``schema=``, but it is a different, older-shaped parameter: that one
       does take filter-only metadata fields such as
       ``{"fields": {"genre": {"filterable": True}}}``, and sends them as
       given. Everything above describes ``create()``'s ``schema=``.

    Attributes:
        fields: Field name to its typed definition, one of the types above.

    Examples:
        >>> idx = pc.indexes.describe("semantic-search")
        >>> {name: type(f).__name__ for name, f in idx.schema.fields.items()}
        {'chunk_text': 'SemanticTextField'}

    .. seealso::
       :class:`~pinecone.schema_builder.SchemaBuilder` — assembles the
       ``schema=`` dict with one validated method per declarable field type,
       and refuses the response-only ones before you spend a round trip.
    """

    fields: dict[str, IndexSchemaField]

    def to_dict(self) -> dict[str, Any]:
        """Return the schema as the plain dict the API exchanges.

        Typed fields keep their ``type`` key; a
        :class:`LegacyMetadataField` is emitted without one, matching what the
        wire format actually looks like. Note that the result still needs its
        response-only fields removed before it can be passed back to
        :meth:`create <pinecone.client.indexes.Indexes.create>`.
        """
        result: dict[str, Any] = _strip_untyped_tags(msgspec.to_builtins(self))
        return result


def _encode_schema_for_request(schema: IndexSchema) -> dict[str, Any]:
    """Encode a typed schema for a create or configure request body.

    ``filterable`` is omitted from a :class:`StringField` that carries a
    ``full_text_search`` config and did not ask to be filterable.  The
    server reads a string field as filter-only metadata as soon as
    ``filterable`` is present and discards the full-text-search
    configuration, so emitting the ``False`` default would turn every
    full-text-search field into a filter-only one.  An explicit
    ``filterable=True`` is still sent as given.

    Every other field type, and a schema passed as a plain dict, is
    encoded unchanged.
    """
    encoded: dict[str, Any] = msgspec.to_builtins(schema)
    fields: dict[str, Any] = encoded["fields"]
    for name, field in schema.fields.items():
        if (
            isinstance(field, StringField)
            and field.full_text_search is not None
            and not field.filterable
        ):
            fields[name].pop("filterable")
    return encoded
