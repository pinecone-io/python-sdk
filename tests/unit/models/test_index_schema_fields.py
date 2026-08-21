"""Unit tests for index schema field models (2026-07)."""

from __future__ import annotations

import msgspec
import msgspec.json
import pytest

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


def test_dense_vector_field_roundtrip() -> None:
    field = DenseVectorField(dimension=1536, metric="cosine", description="embeddings")
    assert field.dimension == 1536
    assert field.metric == "cosine"
    assert field.description == "embeddings"


def test_sparse_vector_field_defaults() -> None:
    field = SparseVectorField()
    assert field.description is None


@pytest.mark.parametrize(
    "raw",
    [b'{"type":"sparse_vector"}', b'{"type":"sparse_vector","description":"bm25"}'],
)
def test_sparse_vector_field_never_encodes_a_metric(raw: bytes) -> None:
    """#350: the model must not be able to hand a ``metric`` back to a caller.

    Both the description-present and description-absent payloads are checked,
    because ``description`` is the only optional key the field has and absence
    is the shape a create round-trip produces.
    """
    decoded = msgspec.json.decode(raw, type=IndexSchemaField)
    assert isinstance(decoded, SparseVectorField)

    reencoded = msgspec.json.decode(msgspec.json.encode(decoded))
    assert set(reencoded) == {"type", "description"}
    assert reencoded["type"] == "sparse_vector"


def test_sparse_vector_field_tolerates_a_metric_on_the_wire() -> None:
    """A server that starts sending one must not break an installed client."""
    decoded = msgspec.json.decode(
        b'{"type":"sparse_vector","metric":"dotproduct"}', type=IndexSchemaField
    )
    assert isinstance(decoded, SparseVectorField)
    assert decoded.description is None


def test_string_field_fts_and_filterable() -> None:
    field = StringField(
        full_text_search=FullTextSearchConfig(language="en"),
        filterable=True,
    )
    assert field.full_text_search is not None
    assert field.full_text_search.language == "en"
    assert field.filterable is True


def test_string_field_stop_words_config() -> None:
    field_default = StringField()
    assert field_default.full_text_search is None

    field_empty = StringField(full_text_search=FullTextSearchConfig())
    assert field_empty.full_text_search is not None
    assert field_empty.full_text_search.stop_words is None

    field_true = StringField(full_text_search=FullTextSearchConfig(stop_words=True))
    assert field_true.full_text_search is not None
    assert field_true.full_text_search.stop_words is True

    field_false = StringField(full_text_search=FullTextSearchConfig(stop_words=False))
    assert field_false.full_text_search is not None
    assert field_false.full_text_search.stop_words is False


def test_float_field_wire_type_is_float() -> None:
    raw = b'{"type": "float", "filterable": true}'
    field = msgspec.json.decode(raw, type=IndexSchemaField)
    assert isinstance(field, FloatField)
    assert field.filterable is True


def test_integer_field_wire_type_is_integer() -> None:
    raw = b'{"type": "integer", "filterable": true}'
    field = msgspec.json.decode(raw, type=IndexSchemaField)
    assert isinstance(field, IntegerField)
    assert field.filterable is True


def test_schema_field_union_decode() -> None:
    cases: list[tuple[bytes, type]] = [
        (
            b'{"type": "dense_vector", "dimension": 768, "metric": "cosine"}',
            DenseVectorField,
        ),
        (b'{"type": "sparse_vector"}', SparseVectorField),
        (b'{"type": "semantic_text", "model": "multilingual-e5-large"}', SemanticTextField),
        (b'{"type": "string", "filterable": true}', StringField),
        (b'{"type": "string_list", "filterable": true}', StringListField),
        (b'{"type": "boolean", "filterable": true}', BooleanField),
        (b'{"type": "float", "filterable": false}', FloatField),
        (b'{"type": "integer", "filterable": false}', IntegerField),
    ]
    for raw, expected_type in cases:
        result = msgspec.json.decode(raw, type=IndexSchemaField)
        assert isinstance(result, expected_type), f"Expected {expected_type}, got {type(result)}"


def test_index_schema_decode() -> None:
    raw = b"""{
        "fields": {
            "embedding": {"type": "dense_vector", "dimension": 1536, "metric": "cosine"},
            "title": {"type": "string", "full_text_search": {"language": "en"}},
            "year": {"type": "float", "filterable": true},
            "body": {"type": "semantic_text", "model": "multilingual-e5-large"}
        }
    }"""
    schema = msgspec.json.decode(raw, type=IndexSchema)
    embedding = schema.fields["embedding"]
    assert isinstance(embedding, DenseVectorField)
    assert embedding.dimension == 1536
    title = schema.fields["title"]
    assert isinstance(title, StringField)
    assert title.full_text_search is not None
    assert title.full_text_search.language == "en"
    year = schema.fields["year"]
    assert isinstance(year, FloatField)
    assert year.filterable is True
    assert isinstance(schema.fields["body"], SemanticTextField)


def test_semantic_text_field_with_parameters() -> None:
    field = SemanticTextField(
        model="multilingual-e5-large",
        metric="cosine",
        read_parameters={"input_type": "query"},
        write_parameters={"input_type": "passage"},
    )
    assert field.model == "multilingual-e5-large"
    assert field.read_parameters == {"input_type": "query"}
    assert field.write_parameters == {"input_type": "passage"}


def test_string_field_defaults() -> None:
    field = StringField()
    assert field.filterable is False
    assert field.full_text_search is None


def test_float_field_model_defaults() -> None:
    field = FloatField()
    assert field.description is None
    assert field.filterable is False


def test_schema_field_union_decode_dense_vector_with_all_fields() -> None:
    raw = (
        b'{"type": "dense_vector", "dimension": 512, "metric": "euclidean", "description": "test"}'
    )
    field = msgspec.json.decode(raw, type=IndexSchemaField)
    assert isinstance(field, DenseVectorField)
    assert field.dimension == 512
    assert field.metric == "euclidean"
    assert field.description == "test"


def test_index_schema_decodes_empty_fields() -> None:
    schema = msgspec.json.decode(b'{"fields": {}}', type=IndexSchema)
    assert schema.fields == {}
    assert isinstance(schema.fields, dict)


def test_full_text_search_config_all_defaults_none() -> None:
    cfg = FullTextSearchConfig()
    assert cfg.language is None
    assert cfg.stemming is None
    assert cfg.stop_words is None
    assert cfg.ngram is None


def test_string_field_decodes_empty_full_text_search_dict() -> None:
    raw = b'{"type": "string", "full_text_search": {}}'
    field = msgspec.json.decode(raw, type=IndexSchemaField)
    assert isinstance(field, StringField)
    assert field.full_text_search is not None
    assert field.full_text_search.language is None


def test_string_field_absent_full_text_search_is_none() -> None:
    raw = b'{"type": "string"}'
    field = msgspec.json.decode(raw, type=IndexSchemaField)
    assert isinstance(field, StringField)
    assert field.full_text_search is None


def test_string_field_decodes_populated_full_text_search() -> None:
    raw = (
        b'{"type": "string", "full_text_search": {"language": "en", '
        b'"stemming": true, "stop_words": true}}'
    )
    field = msgspec.json.decode(raw, type=IndexSchemaField)
    assert isinstance(field, StringField)
    cfg = field.full_text_search
    assert cfg is not None
    assert cfg.language == "en"
    assert cfg.stemming is True
    assert cfg.stop_words is True


def test_string_field_decodes_null_language() -> None:
    raw = b'{"type": "string", "full_text_search": {"language": null, "stemming": false, "stop_words": false}}'
    field = msgspec.json.decode(raw, type=IndexSchemaField)
    assert isinstance(field, StringField)
    assert field.full_text_search is not None
    assert field.full_text_search.language is None


def test_string_field_decodes_ngram_config() -> None:
    raw = (
        b'{"type": "string", "full_text_search": {"language": "en", "stemming": false, '
        b'"stop_words": false, "ngram": {"min_gram": 2, "max_gram": 3, "prefix_only": true}}}'
    )
    field = msgspec.json.decode(raw, type=IndexSchemaField)
    assert isinstance(field, StringField)
    cfg = field.full_text_search
    assert cfg is not None
    assert isinstance(cfg.ngram, NgramConfig)
    assert cfg.ngram.min_gram == 2
    assert cfg.ngram.max_gram == 3
    assert cfg.ngram.prefix_only is True


def test_ngram_prefix_only_defaults_false() -> None:
    ngram = msgspec.json.decode(b'{"min_gram": 1, "max_gram": 5}', type=NgramConfig)
    assert ngram.prefix_only is False


def test_string_list_field_defaults() -> None:
    field = StringListField()
    assert field.filterable is False
    assert field.description is None


def test_string_list_field_with_filterable_and_description() -> None:
    field = StringListField(filterable=True, description="tags")
    assert field.filterable is True
    assert field.description == "tags"


def test_string_list_field_decode_minimal() -> None:
    raw = b'{"type": "string_list"}'
    field = msgspec.json.decode(raw, type=IndexSchemaField)
    assert isinstance(field, StringListField)


def test_string_list_field_decode_filterable() -> None:
    raw = b'{"type": "string_list", "filterable": true, "description": "tags"}'
    field = msgspec.json.decode(raw, type=IndexSchemaField)
    assert isinstance(field, StringListField)
    assert field.filterable is True
    assert field.description == "tags"


def test_schema_field_union_rejects_old_string_array_tag() -> None:
    raw = b'{"type": "string[]"}'
    with pytest.raises(msgspec.ValidationError):
        msgspec.json.decode(raw, type=IndexSchemaField)


def test_boolean_field_decode() -> None:
    raw = b'{"type":"boolean","filterable":true}'
    field = msgspec.json.decode(raw, type=IndexSchemaField)
    assert isinstance(field, BooleanField)
    assert field.filterable is True


def test_boolean_field_decode_defaults() -> None:
    raw = b'{"type":"boolean"}'
    field = msgspec.json.decode(raw, type=IndexSchemaField)
    assert isinstance(field, BooleanField)
    assert field.filterable is False
    assert field.description is None


def test_boolean_field_decode_with_description() -> None:
    raw = b'{"type":"boolean","filterable":false,"description":"active flag"}'
    field = msgspec.json.decode(raw, type=IndexSchemaField)
    assert isinstance(field, BooleanField)
    assert field.description == "active flag"


def test_description_null_decodes_to_none() -> None:
    raw = b'{"type":"dense_vector","dimension":8,"metric":"cosine","description":null}'
    field = msgspec.json.decode(raw, type=IndexSchemaField)
    assert isinstance(field, DenseVectorField)
    assert field.description is None


def test_legacy_metadata_field_construct() -> None:
    field = LegacyMetadataField(filterable=True)
    assert field.filterable is True


def test_legacy_metadata_field_roundtrips_through_msgspec() -> None:
    field = LegacyMetadataField(filterable=True)
    encoded = msgspec.json.encode(field)
    decoded = msgspec.json.decode(encoded, type=IndexSchemaField)
    assert decoded == field


def test_index_schema_to_dict_strips_legacy_tag() -> None:
    schema = IndexSchema(fields={"old": LegacyMetadataField(filterable=False)})
    result = schema.to_dict()
    assert result["fields"]["old"] == {"filterable": False}
