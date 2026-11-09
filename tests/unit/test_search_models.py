"""Tests for search records response models."""

from __future__ import annotations

import pytest

from pinecone._internal.adapters.vectors_adapter import VectorsAdapter
from pinecone.models.vectors.search import (
    Hit,
    SearchRecordsResponse,
    SearchResult,
    SearchUsage,
)


class TestHit:
    def test_hit_fields(self) -> None:
        hit = Hit(id_="r1", score_=0.95, fields={"text": "hello"})
        assert hit.id == "r1"
        assert hit.score == 0.95
        assert hit.fields == {"text": "hello"}

    def test_hit_default_fields(self) -> None:
        hit = Hit(id_="r1", score_=0.5)
        assert hit.fields == {}

    def test_hit_bracket_access(self) -> None:
        hit = Hit(id_="r1", score_=0.95, fields={"text": "hello"})
        assert hit["id"] == "r1"
        assert hit["score"] == 0.95
        assert hit["fields"] == {"text": "hello"}

    def test_hit_bracket_missing_key(self) -> None:
        hit = Hit(id_="r1", score_=0.5)
        with pytest.raises(KeyError, match="nonexistent"):
            hit["nonexistent"]

    def test_hit_bracket_missing_key_lists_available_keys(self) -> None:
        hit = Hit(id_="r1", score_=0.5)
        with pytest.raises(KeyError) as excinfo:
            hit["nonexistent"]
        assert "available keys" in str(excinfo.value)

    def test_hit_bracket_wire_name_id_teaches_alias(self) -> None:
        """The wire format calls the field '_id'; the error must point at hit['id']."""
        hit = Hit(id_="r1", score_=0.5)
        with pytest.raises(KeyError) as excinfo:
            hit["_id"]
        message = str(excinfo.value)
        assert "wire name" in message
        assert "hit['id']" in message

    def test_hit_bracket_wire_name_score_teaches_alias(self) -> None:
        hit = Hit(id_="r1", score_=0.5)
        with pytest.raises(KeyError) as excinfo:
            hit["_score"]
        message = str(excinfo.value)
        assert "wire name" in message
        assert "hit['score']" in message


class TestSearchUsage:
    def test_search_usage_required_only(self) -> None:
        usage = SearchUsage(read_units=5)
        assert usage.read_units == 5
        assert usage.embed_total_tokens is None
        assert usage.rerank_units is None

    def test_search_usage_all_fields(self) -> None:
        usage = SearchUsage(read_units=5, embed_total_tokens=100, rerank_units=2)
        assert usage.read_units == 5
        assert usage.embed_total_tokens == 100
        assert usage.rerank_units == 2


class TestSearchResult:
    def test_search_result_empty_hits(self) -> None:
        result = SearchResult(hits=[])
        assert result.hits == []

    def test_search_result_default_hits(self) -> None:
        result = SearchResult()
        assert result.hits == []


class TestSearchRecordsResponse:
    def test_search_records_response_full(self) -> None:
        hit = Hit(id_="r1", score_=0.92, fields={"data": "text"})
        result = SearchResult(hits=[hit])
        usage = SearchUsage(read_units=5, embed_total_tokens=10)
        response = SearchRecordsResponse(result=result, usage=usage)

        assert len(response.result.hits) == 1
        assert response.result.hits[0].id == "r1"
        assert response.result.hits[0].score == 0.92
        assert response.result.hits[0].fields == {"data": "text"}
        assert response.usage.read_units == 5
        assert response.usage.embed_total_tokens == 10

    def test_search_records_response_bracket_access(self) -> None:
        result = SearchResult(hits=[])
        usage = SearchUsage(read_units=1)
        response = SearchRecordsResponse(result=result, usage=usage)

        assert response["result"] is result
        assert response["usage"] is usage

    def test_search_records_response_bracket_missing_key(self) -> None:
        result = SearchResult(hits=[])
        usage = SearchUsage(read_units=1)
        response = SearchRecordsResponse(result=result, usage=usage)

        with pytest.raises(KeyError, match="nonexistent"):
            response["nonexistent"]


class TestAdapterDecode:
    def test_adapter_decode(self) -> None:
        data = (
            b'{"result": {"hits": [{"_id": "r1", "_score": 0.92,'
            b' "fields": {"data": "text"}}]}, "usage": {"read_units": 5,'
            b' "embed_total_tokens": 10}}'
        )
        response = VectorsAdapter.to_search_response(data)

        assert isinstance(response, SearchRecordsResponse)
        assert len(response.result.hits) == 1
        assert response.result.hits[0].id == "r1"
        assert response.result.hits[0].score == 0.92
        assert response.result.hits[0].fields == {"data": "text"}
        assert response.usage.read_units == 5
        assert response.usage.embed_total_tokens == 10
        assert response.usage.rerank_units is None

    def test_adapter_decode_multiple_hits(self) -> None:
        data = (
            b'{"result": {"hits": ['
            b'{"_id": "r1", "_score": 0.95, "fields": {"text": "first"}},'
            b'{"_id": "r2", "_score": 0.80, "fields": {"text": "second"}}'
            b']}, "usage": {"read_units": 10}}'
        )
        response = VectorsAdapter.to_search_response(data)

        assert len(response.result.hits) == 2
        assert response.result.hits[0].id == "r1"
        assert response.result.hits[1].id == "r2"
        assert response.result.hits[1].score == 0.80

    def test_adapter_decode_empty_hits(self) -> None:
        data = b'{"result": {"hits": []}, "usage": {"read_units": 0}}'
        response = VectorsAdapter.to_search_response(data)

        assert response.result.hits == []
        assert response.usage.read_units == 0

    def test_adapter_decode_with_rerank_usage(self) -> None:
        data = (
            b'{"result": {"hits": [{"_id": "r1", "_score": 0.9, "fields": {}}]},'
            b' "usage": {"read_units": 5, "embed_total_tokens": 50, "rerank_units": 3}}'
        )
        response = VectorsAdapter.to_search_response(data)

        assert response.usage.read_units == 5
        assert response.usage.embed_total_tokens == 50
        assert response.usage.rerank_units == 3
