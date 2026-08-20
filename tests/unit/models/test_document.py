"""Unit tests for the Document dict-wrapper model (2026-07)."""

from __future__ import annotations

import json

import pytest
from hypothesis import given
from hypothesis import strategies as st

from pinecone.models.documents import Document


def test_typed_id() -> None:
    doc = Document({"_id": "doc-1", "_score": 0.9})
    assert doc.id == "doc-1"
    assert doc._id == "doc-1"


def test_typed_score() -> None:
    doc = Document({"_id": "doc-1", "_score": 0.75})
    assert doc.score == 0.75
    assert doc._score == 0.75


def test_typed_score_none_when_absent() -> None:
    doc = Document({"_id": "doc-1"})
    assert doc.score is None
    assert doc._score is None


def test_typed_score_coercion() -> None:
    doc = Document({"_id": "doc-1", "_score": 1})
    assert isinstance(doc.score, float)
    assert doc.score == 1.0
    assert doc._score == 1.0


def test_dynamic_attribute_access() -> None:
    doc = Document({"_id": "doc-1", "title": "Ancient Rome"})
    assert doc.title == "Ancient Rome"  # type: ignore[attr-defined]


def test_dynamic_attribute_missing_raises() -> None:
    doc = Document({"_id": "doc-1"})
    with pytest.raises(AttributeError):
        _ = doc.nonexistent  # type: ignore[attr-defined]


def test_get_existing_key() -> None:
    doc = Document({"_id": "doc-1", "category": "history"})
    assert doc.get("category") == "history"


def test_get_missing_key_returns_none() -> None:
    doc = Document({"_id": "doc-1"})
    assert doc.get("missing") is None


def test_get_missing_key_with_default() -> None:
    doc = Document({"_id": "doc-1"})
    assert doc.get("missing", "fallback") == "fallback"


def test_to_dict_returns_shallow_copy() -> None:
    data = {"_id": "doc-1", "_score": 0.5, "title": "Test"}
    doc = Document(data)
    result = doc.to_dict()
    assert result == data
    result["title"] = "Modified"
    assert doc.get("title") == "Test"


def test_to_json_roundtrip() -> None:
    data = {"_id": "doc-1", "_score": 0.5, "title": "Test"}
    doc = Document(data)
    parsed = json.loads(doc.to_json())
    assert parsed == data


def test_unknown_fields_survive_decode_to_dict_round_trip() -> None:
    data = {
        "_id": "doc-1",
        "_score": 0.5,
        "custom_nested": {"a": [1, 2, {"b": "c"}]},
        "unknown_future_field": "kept",
        "": "empty-key",
    }
    assert Document(data).to_dict() == data


def test_score_collision_typed_property_wins() -> None:
    doc = Document({"_id": "doc-1", "_score": 99.9})
    assert doc.score == 99.9
    assert doc.get("_score") == 99.9


def test_id_collision_typed_property_wins() -> None:
    doc = Document({"_id": "primary", "id": "secondary"})
    assert doc.id == "primary"
    assert doc._id == "primary"
    assert doc.get("_id") == "primary"
    assert doc.get("id") == "secondary"


def test_id_property_does_not_fall_back_to_user_field() -> None:
    doc = Document({"id": "user-field"})
    with pytest.raises(AttributeError):
        _ = doc.id
    with pytest.raises(AttributeError):
        _ = doc._id
    assert doc.get("id") == "user-field"
    assert doc.to_dict() == {"id": "user-field"}


def test_repr_no_extra_fields() -> None:
    doc = Document({"_id": "doc-1", "_score": 0.5})
    r = repr(doc)
    assert r.startswith("Document(")
    assert "_id='doc-1'" in r
    assert "score=0.5" in r
    assert "..." not in r


def test_repr_with_extra_fields() -> None:
    doc = Document({"_id": "doc-1", "_score": 0.5, "title": "Rome"})
    r = repr(doc)
    assert "_id='doc-1'" in r
    assert "..." in r


def test_equality_compares_underlying_data() -> None:
    assert Document({"_id": "a", "x": 1}) == Document({"_id": "a", "x": 1})
    assert Document({"_id": "a"}) != Document({"_id": "b"})


_reserved_keys = st.sampled_from(["_id", "_score", "id", "score"])
_field_keys = st.one_of(st.text(max_size=15), _reserved_keys)
_field_values = st.one_of(
    st.text(max_size=15),
    st.integers(min_value=-(2**31), max_value=2**31),
    st.floats(allow_nan=False, allow_infinity=False, width=32),
    st.booleans(),
    st.lists(st.text(max_size=5), max_size=4),
)
_field_dicts = st.dictionaries(_field_keys, _field_values, max_size=8)


@given(data=_field_dicts)
def test_property_get_to_dict_and_attribute_access_agree(data: dict[str, object]) -> None:
    """``get``, ``to_dict``, and attribute access agree for arbitrary field
    dicts, including the reserved names _id/_score/id/score."""
    doc = Document(data)
    assert doc.to_dict() == data
    for key, value in data.items():
        assert doc.get(key) == value
        assert doc.to_dict()[key] == value
        if key.isidentifier() and not hasattr(Document, key):
            assert getattr(doc, key) == value

    if isinstance(data.get("_id"), str) and data["_id"]:
        assert doc.id == data["_id"]
        assert doc._id == data["_id"]
    raw_score = data.get("_score")
    if isinstance(raw_score, (int, float)) and not isinstance(raw_score, bool):
        assert doc.score == float(raw_score)
        assert doc._score == float(raw_score)
