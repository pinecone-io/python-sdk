"""Unit tests for document score-by query models (2026-07)."""

from __future__ import annotations

import warnings

import msgspec
import pytest

from pinecone.models.documents import (
    DenseVectorQuery,
    DocumentScoringMethod,
    QueryStringQuery,
    SparseVectorQuery,
    TextQuery,
)
from pinecone.models.vectors.sparse import SparseValues


def test_text_query_wire_shape() -> None:
    q = TextQuery(fields=["body"], query="hello world")
    decoded = msgspec.json.decode(msgspec.json.encode(q))
    assert decoded["type"] == "text"
    assert decoded["fields"] == ["body"]
    assert decoded["query"] == "hello world"


def test_text_query_multiple_fields_wire_shape() -> None:
    q = TextQuery(fields=["title", "body"], query="hello world")
    decoded = msgspec.json.decode(msgspec.json.encode(q))
    assert decoded["fields"] == ["title", "body"]


def test_text_query_round_trip() -> None:
    raw = b'{"type": "text", "fields": ["body"], "query": "hello world"}'
    result = msgspec.json.decode(raw, type=TextQuery)
    assert isinstance(result, TextQuery)
    assert result.fields == ["body"]
    assert result.query == "hello world"


def test_text_query_deprecated_field_kwarg_migrates_to_fields() -> None:
    with pytest.warns(DeprecationWarning, match=r"`field` is deprecated"):
        q = TextQuery(field="title", query="hello")
    assert q.fields == ["title"]
    assert q.field is None


def test_text_query_deprecated_field_kwarg_keeps_wire_shape_clean() -> None:
    with pytest.warns(DeprecationWarning, match=r"`field` is deprecated"):
        q = TextQuery(field="title", query="hello")
    decoded = msgspec.json.decode(msgspec.json.encode(q))
    assert decoded == {"type": "text", "fields": ["title"], "query": "hello"}
    assert "field" not in decoded


def test_text_query_canonical_fields_emits_no_warning() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        q = TextQuery(fields=["title"], query="hello")
    assert q.fields == ["title"]


def test_text_query_rejects_both_field_and_fields() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with pytest.raises(ValueError, match=r"`fields=\[\.\.\.\]`.*`field=\.\.\.`.*not both"):
            TextQuery(field="title", fields=["body"], query="hello")


def test_text_query_rejects_missing_fields_with_backend_vocabulary() -> None:
    with pytest.raises(ValueError, match="Text scoring requires specifying at least one field"):
        TextQuery(query="hello")


def test_text_query_rejects_empty_fields_list_with_backend_vocabulary() -> None:
    with pytest.raises(ValueError, match="Text scoring requires specifying at least one field"):
        TextQuery(fields=[], query="hello")


def test_text_query_decodes_backend_field_variant() -> None:
    raw = b'{"type": "text", "field": "body", "query": "hello"}'
    with pytest.warns(DeprecationWarning, match=r"`field` is deprecated"):
        result = msgspec.json.decode(raw, type=TextQuery)
    assert result.fields == ["body"]
    assert result.field is None


def test_query_string_query_wire_shape() -> None:
    q = QueryStringQuery(query="robots AND adventure")
    decoded = msgspec.json.decode(msgspec.json.encode(q))
    assert decoded == {"type": "query_string", "query": "robots AND adventure"}


def test_query_string_query_round_trip() -> None:
    raw = b'{"type": "query_string", "query": "robots AND adventure"}'
    result = msgspec.json.decode(raw, type=QueryStringQuery)
    assert isinstance(result, QueryStringQuery)
    assert result.query == "robots AND adventure"


def test_query_string_query_rejects_field() -> None:
    with pytest.raises(ValueError, match=r"query_string.*must not specify 'field' or 'fields'"):
        QueryStringQuery(query="test", field="title")


def test_query_string_query_rejects_fields() -> None:
    with pytest.raises(ValueError, match=r"query_string.*must not specify 'field' or 'fields'"):
        QueryStringQuery(query="test", fields=["title"])


def test_dense_vector_query_wire_shape() -> None:
    q = DenseVectorQuery(field="embedding", values=[0.1, 0.2, 0.3])
    decoded = msgspec.json.decode(msgspec.json.encode(q))
    assert decoded["type"] == "dense_vector"
    assert decoded["field"] == "embedding"
    assert decoded["values"] == [0.1, 0.2, 0.3]


def test_dense_vector_query_round_trip() -> None:
    raw = b'{"type": "dense_vector", "field": "embedding", "values": [0.1, 0.2, 0.3]}'
    result = msgspec.json.decode(raw, type=DenseVectorQuery)
    assert isinstance(result, DenseVectorQuery)
    assert result.field == "embedding"
    assert result.values == [0.1, 0.2, 0.3]


def test_dense_vector_query_rejects_empty_field() -> None:
    with pytest.raises(
        ValueError, match="dense_vector scoring requires specifying the name of exactly one field"
    ):
        DenseVectorQuery(field="", values=[0.1])


def test_dense_vector_query_rejects_empty_values() -> None:
    with pytest.raises(ValueError, match="dense_vector scoring requires a non-empty 'values'"):
        DenseVectorQuery(field="embedding", values=[])


def test_sparse_vector_query_wire_shape_matches_2026_07_fixture() -> None:
    """Recorded-fixture check of the 2026-07 wire casing.

    Per the 2026-07 OAS (SparseValues :2319, DocumentScoringMethod :2931)
    and the backend serde structs (svc-docs-api/src/core/documents/mod.rs
    SparseValues/ScoringMethod), the wire keys are snake_case:
    ``sparse_values`` with nested ``indices`` and ``values``.
    """
    q = SparseVectorQuery(
        field="_sparse", sparse_values=SparseValues(indices=[0, 5, 10], values=[0.5, 0.3, 0.8])
    )
    assert msgspec.json.decode(msgspec.json.encode(q)) == {
        "type": "sparse_vector",
        "field": "_sparse",
        "sparse_values": {"indices": [0, 5, 10], "values": [0.5, 0.3, 0.8]},
    }


def test_sparse_vector_query_round_trip() -> None:
    raw = (
        b'{"type": "sparse_vector", "field": "_sparse", '
        b'"sparse_values": {"indices": [0, 5], "values": [0.5, 0.3]}}'
    )
    result = msgspec.json.decode(raw, type=SparseVectorQuery)
    assert isinstance(result, SparseVectorQuery)
    assert result.field == "_sparse"
    assert result.sparse_values.indices == [0, 5]
    assert result.sparse_values.values == [0.5, 0.3]


def test_sparse_vector_query_rejects_empty_field() -> None:
    with pytest.raises(
        ValueError, match="sparse_vector scoring requires specifying the name of exactly one field"
    ):
        SparseVectorQuery(field="", sparse_values=SparseValues(indices=[1], values=[0.9]))


def test_union_decodes_text_query() -> None:
    raw = b'{"type": "text", "fields": ["body"], "query": "hello"}'
    assert isinstance(msgspec.json.decode(raw, type=DocumentScoringMethod), TextQuery)


def test_union_decodes_query_string_query() -> None:
    raw = b'{"type": "query_string", "query": "robots"}'
    assert isinstance(msgspec.json.decode(raw, type=DocumentScoringMethod), QueryStringQuery)


def test_union_decodes_dense_vector_query() -> None:
    raw = b'{"type": "dense_vector", "field": "emb", "values": [0.1]}'
    assert isinstance(msgspec.json.decode(raw, type=DocumentScoringMethod), DenseVectorQuery)


def test_union_decodes_sparse_vector_query() -> None:
    raw = (
        b'{"type": "sparse_vector", "field": "_sparse", '
        b'"sparse_values": {"indices": [1], "values": [0.9]}}'
    )
    assert isinstance(msgspec.json.decode(raw, type=DocumentScoringMethod), SparseVectorQuery)


def test_union_importable_from_canonical_paths() -> None:
    import pinecone
    from pinecone.models.documents import score_by

    assert pinecone.DocumentScoringMethod is score_by.DocumentScoringMethod
    assert pinecone.models.DocumentScoringMethod is score_by.DocumentScoringMethod
