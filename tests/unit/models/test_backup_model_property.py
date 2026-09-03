"""Property-based tests for BackupModel schema embedding (2026-07).

Pins the round trip a caller depends on when reading ``backup.schema``:
for any valid IndexSchema embedded in a backup payload, decode ->
``to_dict`` -> re-encode preserves both the schema field-name key set and
the concrete type of every field.
"""

from __future__ import annotations

from typing import Any

import msgspec
from hypothesis import given
from hypothesis import strategies as st

from pinecone.models.backups.model import BackupModel
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

_any_field = st.one_of(
    st.builds(
        DenseVectorField,
        dimension=st.integers(min_value=1, max_value=20000),
        metric=_metric,
        description=_description,
    ),
    st.builds(SparseVectorField, description=_description),
    st.builds(
        SemanticTextField,
        model=st.sampled_from(["multilingual-e5-large", "llama-text-embed-v2"]),
        metric=st.one_of(st.none(), _metric),
        description=_description,
        read_parameters=st.one_of(st.none(), st.dictionaries(st.text(min_size=1), st.text())),
        write_parameters=st.one_of(st.none(), st.dictionaries(st.text(min_size=1), st.text())),
    ),
    st.builds(
        StringField,
        description=_description,
        filterable=st.booleans(),
        full_text_search=st.one_of(st.none(), _fts_config),
    ),
    st.builds(StringListField, description=_description, filterable=st.booleans()),
    st.builds(BooleanField, description=_description, filterable=st.booleans()),
    st.builds(IntegerField, description=_description, filterable=st.booleans()),
    st.builds(FloatField, description=_description, filterable=st.booleans()),
    st.builds(LegacyMetadataField, filterable=st.booleans()),
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

_timestamp = st.one_of(st.none(), st.just("2025-03-05T12:00:00Z"))


def _backup_payload(schema: IndexSchema | None, deleted_at: str | None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "backup_id": "bkp-1",
        "source_index_name": "my-index",
        "source_index_id": "idx-1",
        "status": "Ready",
        "cloud": "aws",
        "region": "us-east-1",
        "created_at": "2025-03-01T09:00:00Z",
    }
    if schema is not None:
        payload["schema"] = msgspec.to_builtins(schema)
    if deleted_at is not None:
        payload["source_index_deleted_at"] = deleted_at
    return payload


@given(schema=_any_schema, deleted_at=_timestamp)
def test_embedded_schema_survives_decode_to_dict_reencode(
    schema: IndexSchema, deleted_at: str | None
) -> None:
    decoded = msgspec.convert(_backup_payload(schema, deleted_at), BackupModel)

    assert decoded.schema is not None
    assert set(decoded.schema.fields) == set(schema.fields)
    for name, field in schema.fields.items():
        assert type(decoded.schema.fields[name]) is type(field)
    assert decoded.source_index_deleted_at == deleted_at

    as_dict = decoded.to_dict()
    assert set(as_dict["schema"]["fields"]) == set(schema.fields)

    redecoded = msgspec.convert(_backup_payload(schema, deleted_at), BackupModel)
    assert redecoded.to_dict() == as_dict
    assert redecoded.schema == decoded.schema


@given(schema=_any_schema)
def test_to_dict_schema_matches_index_schema_projection(schema: IndexSchema) -> None:
    decoded = msgspec.convert(_backup_payload(schema, None), BackupModel)
    assert decoded.schema is not None
    assert decoded.to_dict()["schema"] == decoded.schema.to_dict()


@given(field=_any_field)
def test_dense_dimension_reads_the_single_dense_field(field: IndexSchemaField) -> None:
    schema = IndexSchema(fields={"solo": field})
    decoded = msgspec.convert(_backup_payload(schema, None), BackupModel)
    if isinstance(field, DenseVectorField):
        assert decoded.dense_dimension == field.dimension
    else:
        assert decoded.dense_dimension is None


@given(schema=_any_schema, deleted_at=_timestamp)
def test_optional_fields_absent_stay_none(schema: IndexSchema, deleted_at: str | None) -> None:
    decoded = msgspec.convert(_backup_payload(schema, deleted_at), BackupModel)
    assert decoded.name is None
    assert decoded.description is None
    assert decoded.record_count is None
    assert decoded.namespace_count is None
    assert decoded.size_bytes is None
    assert decoded.tags is None
