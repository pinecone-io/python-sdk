"""Tests for the search_records alias method on Index."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from pinecone.index import Index
from pinecone.models.vectors.search import (
    Hit,
    SearchRecordsResponse,
    SearchResult,
    SearchUsage,
)


def _make_index() -> Index:
    """Create an Index instance with mocked internals."""
    idx = object.__new__(Index)
    idx._http = MagicMock()  # type: ignore[attr-defined]
    idx._adapter = MagicMock()  # type: ignore[attr-defined]
    return idx


def _make_search_response() -> SearchRecordsResponse:
    """Build a minimal SearchRecordsResponse for tests."""
    return SearchRecordsResponse(
        result=SearchResult(hits=[Hit(id_="v1", score_=0.9, fields={})]),
        usage=SearchUsage(read_units=1),
    )


class TestSearchRecordsAlias:
    def test_search_records_delegates_to_search(self) -> None:
        idx = _make_index()
        expected = _make_search_response()
        with patch.object(idx, "search", return_value=expected) as mock_search:
            idx.search_records(namespace="ns", top_k=5, vector=[1.0])
            mock_search.assert_called_once_with(
                namespace="ns",
                top_k=5,
                inputs=None,
                vector=[1.0],
                id=None,
                filter=None,
                fields=None,
                rerank=None,
                match_terms=None,
                query=None,
                timeout=None,
            )

    def test_search_records_returns_search_result(self) -> None:
        idx = _make_index()
        expected = _make_search_response()
        with patch.object(idx, "search", return_value=expected):
            result = idx.search_records(namespace="ns", top_k=5, vector=[1.0])
            assert result is expected

    def test_search_records_passes_all_params(self) -> None:
        idx = _make_index()
        expected = _make_search_response()
        kwargs: dict[str, Any] = {
            "namespace": "ns",
            "top_k": 10,
            "inputs": {"text": "hello"},
            "vector": [0.1, 0.2],
            "id": "rec-1",
            "filter": {"genre": "comedy"},
            "fields": ["title", "genre"],
            "rerank": {"model": "bge-reranker", "rank_fields": ["text"]},
            "match_terms": {"strategy": "all", "terms": ["animal", "duck"]},
            "query": None,
            "timeout": None,
        }
        with patch.object(idx, "search", return_value=expected) as mock_search:
            idx.search_records(**kwargs)
            mock_search.assert_called_once_with(**kwargs)
