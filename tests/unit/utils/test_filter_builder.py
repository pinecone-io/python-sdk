"""Tests for the filter builder utility."""

from __future__ import annotations

import pytest

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
            Field("x").lt(True)  # type: ignore[arg-type]


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
