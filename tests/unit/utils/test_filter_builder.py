"""Tests for the filter builder utility."""

from __future__ import annotations

import re

import pytest

from pinecone.errors import PineconeError, PineconeTypeError, PineconeValueError
from pinecone.utils.filter_builder import Condition, Field


class TestFieldEquality:
    def test_field_eq_str(self) -> None:
        result = (Field("genre") == "drama").to_dict()
        assert result == {"genre": {"$eq": "drama"}}

    def test_field_eq_int(self) -> None:
        result = (Field("year") == 2024).to_dict()
        assert result == {"year": {"$eq": 2024}}

    def test_field_ne(self) -> None:
        result = (Field("status") != "archived").to_dict()
        assert result == {"status": {"$ne": "archived"}}

    def test_bool_value_eq(self) -> None:
        result = (Field("active") == True).to_dict()  # noqa: E712
        assert result == {"active": {"$eq": True}}


class TestFieldNumericComparison:
    def test_field_gt(self) -> None:
        result = Field("score").gt(0.5).to_dict()
        assert result == {"score": {"$gt": 0.5}}

    def test_field_gte(self) -> None:
        result = Field("count").gte(10).to_dict()
        assert result == {"count": {"$gte": 10}}

    def test_field_lt(self) -> None:
        result = Field("price").lt(100).to_dict()
        assert result == {"price": {"$lt": 100}}

    def test_field_lte(self) -> None:
        result = Field("rank").lte(5).to_dict()
        assert result == {"rank": {"$lte": 5}}

    def test_numeric_only_gt(self) -> None:
        with pytest.raises(TypeError, match="numeric"):
            Field("x").gt("text")  # type: ignore[arg-type]

    def test_numeric_only_gte(self) -> None:
        with pytest.raises(TypeError, match="numeric"):
            Field("x").gte("text")  # type: ignore[arg-type]

    def test_numeric_only_lt(self) -> None:
        with pytest.raises(TypeError, match="numeric"):
            Field("x").lt(True)


class TestFieldSetOperators:
    def test_field_is_in(self) -> None:
        result = Field("color").is_in(["red", "blue"]).to_dict()
        assert result == {"color": {"$in": ["red", "blue"]}}

    def test_field_not_in(self) -> None:
        result = Field("tag").not_in(["spam"]).to_dict()
        assert result == {"tag": {"$nin": ["spam"]}}

    def test_field_is_in_accepts_tuple(self) -> None:
        result = Field("color").is_in(("red", "blue")).to_dict()
        assert result == {"color": {"$in": ("red", "blue")}}

    def test_is_in_requires_list(self) -> None:
        with pytest.raises(TypeError, match="list"):
            Field("genre").is_in("drama")  # type: ignore[arg-type]

    def test_not_in_requires_list(self) -> None:
        with pytest.raises(TypeError, match="list"):
            Field("genre").not_in("drama")  # type: ignore[arg-type]


class TestFieldExists:
    def test_field_exists(self) -> None:
        result = Field("metadata_key").exists().to_dict()
        assert result == {"metadata_key": {"$exists": True}}


class TestLogicalCombinations:
    def test_and_combination(self) -> None:
        result = ((Field("a") == 1) & (Field("b") == 2)).to_dict()
        assert result == {"$and": [{"a": {"$eq": 1}}, {"b": {"$eq": 2}}]}

    def test_or_combination(self) -> None:
        result = ((Field("a") == 1) | (Field("b") == 2)).to_dict()
        assert result == {"$or": [{"a": {"$eq": 1}}, {"b": {"$eq": 2}}]}

    def test_and_flattening(self) -> None:
        result = (((Field("a") == 1) & (Field("b") == 2)) & (Field("c") == 3)).to_dict()
        assert result == {
            "$and": [
                {"a": {"$eq": 1}},
                {"b": {"$eq": 2}},
                {"c": {"$eq": 3}},
            ]
        }

    def test_or_flattening(self) -> None:
        result = (((Field("a") == 1) | (Field("b") == 2)) | (Field("c") == 3)).to_dict()
        assert result == {
            "$or": [
                {"a": {"$eq": 1}},
                {"b": {"$eq": 2}},
                {"c": {"$eq": 3}},
            ]
        }


class TestConditionEdgeCases:
    def test_empty_condition_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            Condition({}).to_dict()


class TestExceptionTypes:
    @pytest.mark.parametrize("op", ["gt", "gte", "lt", "lte"])
    def test_non_numeric_raises_pinecone_type_error(self, op: str) -> None:
        with pytest.raises(PineconeTypeError, match="numeric"):
            getattr(Field("x"), op)("text")

    @pytest.mark.parametrize("op", ["gt", "gte", "lt", "lte"])
    def test_non_numeric_is_catchable_as_pinecone_error(self, op: str) -> None:
        with pytest.raises(PineconeError):
            getattr(Field("x"), op)(True)

    @pytest.mark.parametrize("op", ["gt", "gte", "lt", "lte"])
    def test_non_numeric_is_still_catchable_as_type_error(self, op: str) -> None:
        with pytest.raises(TypeError):
            getattr(Field("x"), op)("text")

    def test_non_numeric_message_is_unchanged(self) -> None:
        with pytest.raises(PineconeTypeError) as excinfo:
            Field("x").gt("text")  # type: ignore[arg-type]
        assert str(excinfo.value) == "gt requires a numeric value (int or float), got str"

    def test_empty_condition_raises_pinecone_value_error(self) -> None:
        with pytest.raises(PineconeValueError, match="empty"):
            Condition({}).to_dict()

    def test_empty_condition_is_catchable_as_pinecone_error(self) -> None:
        with pytest.raises(PineconeError):
            Condition({}).to_dict()

    def test_empty_condition_message_is_unchanged(self) -> None:
        with pytest.raises(PineconeValueError) as excinfo:
            Condition({}).to_dict()
        assert str(excinfo.value) == "Cannot convert an empty condition to a filter dict"


class TestFieldTextMatch:
    @pytest.mark.parametrize("op", ["match_phrase", "match_all", "match_any"])
    def test_to_dict(self, op: str) -> None:
        cond = getattr(Field("title"), op)("black hole")
        assert isinstance(cond, Condition)
        assert cond.to_dict() == {"title": {f"${op}": "black hole"}}

    @pytest.mark.parametrize("op", ["match_phrase", "match_all", "match_any"])
    @pytest.mark.parametrize("bad", [1, 2.5, True, None, ["a", "b"], b"x"])
    def test_non_str_raises(self, op: str, bad: object) -> None:
        with pytest.raises(PineconeTypeError, match=op) as info:
            getattr(Field("title"), op)(bad)
        assert isinstance(info.value, TypeError)

    @pytest.mark.parametrize("op", ["match_phrase", "match_all", "match_any"])
    def test_empty_string_sent_as_given(self, op: str) -> None:
        assert getattr(Field("title"), op)("").to_dict() == {"title": {f"${op}": ""}}

    def test_composes_with_and(self) -> None:
        cond = Field("title").match_phrase("black hole") & Field("year").gte(2010)
        assert cond.to_dict() == {
            "$and": [{"title": {"$match_phrase": "black hole"}}, {"year": {"$gte": 2010}}]
        }

    def test_composes_with_or(self) -> None:
        cond = Field("title").match_all("a b") | Field("title").match_any("c")
        assert cond.to_dict() == {
            "$or": [{"title": {"$match_all": "a b"}}, {"title": {"$match_any": "c"}}]
        }

    @pytest.mark.parametrize("op", ["match_phrase", "match_all", "match_any"])
    def test_docstring_names_operations(self, op: str) -> None:
        doc = getattr(Field, op).__doc__ or ""
        assert "search" in doc and "fetch" in doc
        assert re.search(r"delete.*update.*reject.*400", doc, re.DOTALL)

    def test_class_and_module_docs_list_the_operators(self) -> None:
        import pinecone.utils.filter_builder as mod

        for name in ("match_phrase", "match_all", "match_any"):
            assert name in (Field.__doc__ or "")
        assert "text match" in (mod.__doc__ or "")
