"""Property-based tests for index schema models (2026-07).

Two properties are pinned here:

* Any valid combination of schema-field variants round-trips through
  msgspec encode/decode without loss (model equality), and the ``to_dict``
  projection matches the wire shape modulo the internal legacy tag.
* Any create-style schema (0-1 dense, 0-1 sparse, 0-N fts-string fields)
  survives build -> encode -> decode -> to_dict losslessly, while invalid
  field names are rejected by CreateIndexRequest before any HTTP request
  could be made.
"""

from __future__ import annotations

import msgspec
from hypothesis import given
from hypothesis import strategies as st

from pinecone.models.indexes.requests import CreateIndexRequest
from pinecone.models.indexes.schema import (
    BooleanField,
    DenseVectorField,
    FloatField,
    FullTextSearchConfig,
    IndexSchema,
    IndexSchemaField,
    IntegerField,
    LegacyMetadataField,
    NgramConfig,
    SemanticTextField,
    SparseVectorField,
    StringField,
    StringListField,
)

_description = st.one_of(st.none(), st.text(min_size=0, max_size=20))
_metric = st.sampled_from(["cosine", "dotproduct", "euclidean"])

_ngram = st.builds(
    NgramConfig,
    min_gram=st.integers(min_value=1, max_value=10),
    max_gram=st.integers(min_value=1, max_value=10),
    prefix_only=st.booleans(),
)

_fts_config = st.builds(
    FullTextSearchConfig,
    language=st.one_of(st.none(), st.sampled_from(["en", "de", "french"])),
    stemming=st.one_of(st.none(), st.booleans()),
    stop_words=st.one_of(st.none(), st.booleans()),
    ngram=st.one_of(st.none(), _ngram),
)

_dense = st.builds(
    DenseVectorField,
    dimension=st.integers(min_value=1, max_value=20000),
    metric=_metric,
    description=_description,
)
_sparse = st.builds(SparseVectorField, description=_description)
_semantic = st.builds(
    SemanticTextField,
    model=st.sampled_from(["multilingual-e5-large", "llama-text-embed-v2"]),
    metric=st.one_of(st.none(), _metric),
    description=_description,
    read_parameters=st.one_of(st.none(), st.dictionaries(st.text(min_size=1), st.text())),
    write_parameters=st.one_of(st.none(), st.dictionaries(st.text(min_size=1), st.text())),
)
_string = st.builds(
    StringField,
    description=_description,
    filterable=st.booleans(),
    full_text_search=st.one_of(st.none(), _fts_config),
)
_string_list = st.builds(StringListField, description=_description, filterable=st.booleans())
_boolean = st.builds(BooleanField, description=_description, filterable=st.booleans())
_integer = st.builds(IntegerField, description=_description, filterable=st.booleans())
_float = st.builds(FloatField, description=_description, filterable=st.booleans())
_legacy = st.builds(LegacyMetadataField, filterable=st.booleans())

_any_field = st.one_of(
    _dense, _sparse, _semantic, _string, _string_list, _boolean, _integer, _float, _legacy
)

_valid_field_name = st.text(
    alphabet=st.characters(codec="ascii", categories=["L", "N"]),
    min_size=1,
    max_size=64,
).filter(lambda s: not s.startswith(("_", "$")))

_any_schema = st.builds(
    IndexSchema,
    fields=st.dictionaries(_valid_field_name, _any_field, max_size=6),
)


@given(schema=_any_schema)
def test_any_schema_field_combination_roundtrips_without_loss(schema: IndexSchema) -> None:
    encoded = msgspec.json.encode(schema)
    decoded = msgspec.json.decode(encoded, type=IndexSchema)
    assert decoded == schema

    as_dict = schema.to_dict()
    for name, field in schema.fields.items():
        wire = as_dict["fields"][name]
        if isinstance(field, LegacyMetadataField):
            assert "type" not in wire
        else:
            assert wire["type"] == msgspec.to_builtins(field)["type"]


@given(field=_any_field)
def test_single_field_roundtrips_through_union(field: IndexSchemaField) -> None:
    encoded = msgspec.json.encode(field)
    decoded = msgspec.json.decode(encoded, type=IndexSchemaField)
    assert decoded == field


_fts_string_for_create = st.builds(
    StringField,
    description=st.none(),
    filterable=st.booleans(),
    full_text_search=_fts_config,
)


@st.composite
def _create_schemas(draw: st.DrawFn) -> IndexSchema:
    fields: dict[str, IndexSchemaField] = {}
    names = iter(draw(st.lists(_valid_field_name, min_size=8, max_size=8, unique=True)))
    if draw(st.booleans()):
        fields[next(names)] = draw(_dense)
    if draw(st.booleans()):
        fields[next(names)] = draw(_sparse)
    for _ in range(draw(st.integers(min_value=0, max_value=4))):
        fields[next(names)] = draw(_fts_string_for_create)
    return IndexSchema(fields=fields)


@given(schema=_create_schemas())
def test_create_schema_build_encode_decode_to_dict_lossless(schema: IndexSchema) -> None:
    request = CreateIndexRequest(schema=schema)
    encoded = msgspec.json.encode(request)
    decoded_body = msgspec.json.decode(encoded, type=dict)

    redecoded = msgspec.convert(decoded_body["schema"], IndexSchema)
    assert redecoded == schema
    assert redecoded.to_dict() == schema.to_dict()


_invalid_field_names = st.one_of(
    st.just(""),
    st.just("f" * 65),
    _valid_field_name.map(lambda s: f"_{s}"),
    _valid_field_name.map(lambda s: f"${s}"),
)


@given(name=_invalid_field_names)
def test_invalid_field_names_rejected_before_any_http(name: str) -> None:
    import pytest

    with pytest.raises(ValueError, match="Invalid schema field name"):
        CreateIndexRequest(schema={"fields": {name: {"type": "dense_vector", "dimension": 3}}})
