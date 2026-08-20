"""Unit tests for DocumentRecord and UpdateDocumentRecord (2026-07).

Includes the DocumentFieldValue-grammar property test: arbitrary field maps
round-trip decode/encode losslessly, and ``_id`` constraint violations
always raise.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from pinecone.models.documents import DocumentRecord, UpdateDocumentRecord


def test_record_from_dict() -> None:
    record = DocumentRecord({"_id": "doc-1", "title": "Rome", "year": 2019})
    assert record.id == "doc-1"
    assert record._id == "doc-1"
    assert record.get("title") == "Rome"
    assert record.to_dict() == {"_id": "doc-1", "title": "Rome", "year": 2019}


def test_record_from_kwargs() -> None:
    record = DocumentRecord(_id="doc-1", title="Rome")
    assert record.to_dict() == {"_id": "doc-1", "title": "Rome"}


def test_record_kwargs_override_dict() -> None:
    record = DocumentRecord({"_id": "doc-1", "title": "Old"}, title="New")
    assert record.get("title") == "New"


def test_record_to_json_round_trip() -> None:
    record = DocumentRecord({"_id": "doc-1", "tags": ["a", "b"], "ok": True})
    assert DocumentRecord(json.loads(record.to_json())) == record


def test_record_missing_id_raises() -> None:
    with pytest.raises(ValueError, match="'_id' is required"):
        DocumentRecord({"title": "no id"})


def test_record_non_string_id_raises() -> None:
    with pytest.raises(ValueError, match="'_id' is required and must be a string"):
        DocumentRecord({"_id": 42})


def test_record_empty_id_raises() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        DocumentRecord({"_id": ""})


def test_record_overlong_id_raises() -> None:
    with pytest.raises(ValueError, match="maximum length of 512"):
        DocumentRecord({"_id": "x" * 513})


def test_record_max_length_id_accepted() -> None:
    assert DocumentRecord({"_id": "x" * 512}).id == "x" * 512


def test_record_non_ascii_id_raises() -> None:
    with pytest.raises(ValueError, match="ASCII"):
        DocumentRecord({"_id": "docüment"})


def test_record_nul_byte_id_raises() -> None:
    with pytest.raises(ValueError, match="ASCII"):
        DocumentRecord({"_id": "doc\x00one"})


def test_record_unknown_fields_preserved() -> None:
    data = {
        "_id": "doc-1",
        "future_field": {"nested": True},
        "sv": {"indices": [1], "values": [0.5]},
    }
    assert DocumentRecord(data).to_dict() == data


def test_update_record_valid() -> None:
    record = UpdateDocumentRecord({"_id": "doc-1", "title": "New", "_remove_fields": ["old"]})
    assert record.id == "doc-1"
    assert record.remove_fields == ["old"]
    assert record.get("title") == "New"


def test_update_record_remove_fields_optional() -> None:
    record = UpdateDocumentRecord({"_id": "doc-1", "title": "New"})
    assert record.remove_fields is None


def test_update_record_id_validated() -> None:
    with pytest.raises(ValueError, match="'_id'"):
        UpdateDocumentRecord({"_remove_fields": ["a"]})


def test_update_record_remove_fields_must_be_string_list() -> None:
    with pytest.raises(ValueError, match="_remove_fields"):
        UpdateDocumentRecord({"_id": "doc-1", "_remove_fields": "title"})
    with pytest.raises(ValueError, match="_remove_fields"):
        UpdateDocumentRecord({"_id": "doc-1", "_remove_fields": [1, 2]})


def test_update_record_set_and_remove_same_field_raises() -> None:
    with pytest.raises(ValueError, match=r"both set and removed.*title"):
        UpdateDocumentRecord({"_id": "doc-1", "title": "New", "_remove_fields": ["title"]})


def test_update_record_wire_shape_round_trip() -> None:
    data = {"_id": "doc-1", "_remove_fields": ["content"], "title": "Updated title"}
    record = UpdateDocumentRecord(data)
    assert record.to_dict() == data
    assert UpdateDocumentRecord(json.loads(record.to_json())) == record


_valid_ids = st.text(
    alphabet=st.characters(min_codepoint=0x01, max_codepoint=0x7F), min_size=1, max_size=512
)
_invalid_ids = st.one_of(
    st.just(""),
    st.text(
        alphabet=st.characters(min_codepoint=0x01, max_codepoint=0x7F),
        min_size=513,
        max_size=600,
    ),
    st.text(min_size=1, max_size=50).filter(lambda s: any(ord(c) > 0x7F for c in s)),
    st.text(max_size=10).map(lambda s: s + "\x00"),
    st.integers(),
    st.none(),
    st.booleans(),
)

_scalar_values = st.one_of(
    st.text(max_size=20),
    st.integers(min_value=-(2**31), max_value=2**31),
    st.floats(allow_nan=False, allow_infinity=False, width=32),
    st.booleans(),
)
_sparse_values = st.builds(
    lambda indices, values: {"indices": indices, "values": values},
    st.lists(st.integers(min_value=0, max_value=2**32 - 1), min_size=1, max_size=5),
    st.lists(st.floats(allow_nan=False, allow_infinity=False, width=32), min_size=1, max_size=5),
)
_document_field_values = st.one_of(
    _scalar_values,
    st.lists(st.text(max_size=10), max_size=5),
    st.lists(st.floats(allow_nan=False, allow_infinity=False, width=32), max_size=5),
    _sparse_values,
)
_field_names = st.text(max_size=20).filter(lambda s: s != "_id")
_field_maps = st.dictionaries(_field_names, _document_field_values, max_size=8)


@given(doc_id=_valid_ids, fields=_field_maps)
def test_property_record_round_trips_losslessly(doc_id: str, fields: dict[str, Any]) -> None:
    """DocumentRecords built from the DocumentFieldValue grammar round-trip
    decode/encode losslessly through to_dict and JSON."""
    data = {"_id": doc_id, **fields}
    record = DocumentRecord(data)
    assert record.to_dict() == data
    assert DocumentRecord(json.loads(record.to_json())) == record


@given(bad_id=_invalid_ids, fields=_field_maps)
def test_property_invalid_id_always_raises(bad_id: Any, fields: dict[str, Any]) -> None:
    data = {**fields, "_id": bad_id}
    with pytest.raises(ValueError):
        DocumentRecord(data)
    with pytest.raises(ValueError):
        UpdateDocumentRecord(data)
