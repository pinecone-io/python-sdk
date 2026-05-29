"""Tests for AsyncIndex.search_records alias method."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from pinecone.async_client.async_index import AsyncIndex
from pinecone.models.vectors.search import SearchRecordsResponse


@pytest.mark.asyncio
async def test_search_records_delegates_to_search() -> None:
    """search_records delegates to search with the same kwargs."""
    idx = object.__new__(AsyncIndex)
    mock_response = AsyncMock(spec=SearchRecordsResponse)

    with patch.object(AsyncIndex, "search", new_callable=AsyncMock, return_value=mock_response):
        await idx.search_records(namespace="ns", top_k=5, vector=[1.0])
        AsyncIndex.search.assert_awaited_once_with(  # type: ignore[attr-defined]
            namespace="ns",
            top_k=5,
            vector=[1.0],
            inputs=None,
            id=None,
            filter=None,
            fields=None,
            rerank=None,
            match_terms=None,
            query=None,
            timeout=None,
        )


@pytest.mark.asyncio
async def test_search_records_returns_search_result() -> None:
    """search_records returns the same object that search returns."""
    idx = object.__new__(AsyncIndex)
    mock_response = AsyncMock(spec=SearchRecordsResponse)

    with patch.object(AsyncIndex, "search", new_callable=AsyncMock, return_value=mock_response):
        result = await idx.search_records(namespace="ns", top_k=5, vector=[1.0])

    assert result is mock_response


@pytest.mark.asyncio
async def test_search_records_passes_all_params() -> None:
    """search_records forwards every parameter to search."""
    idx = object.__new__(AsyncIndex)
    mock_response = AsyncMock(spec=SearchRecordsResponse)

    kwargs = {
        "namespace": "ns",
        "top_k": 10,
        "inputs": {"text": "hello"},
        "vector": [0.1, 0.2],
        "id": "vec-1",
        "filter": {"genre": "action"},
        "fields": ["title", "year"],
        "rerank": {"model": "bge-reranker-v2-m3", "rank_fields": ["text"]},
        "match_terms": {"strategy": "all", "terms": ["animal", "duck"]},
        "query": None,
        "timeout": None,
    }

    with patch.object(AsyncIndex, "search", new_callable=AsyncMock, return_value=mock_response):
        await idx.search_records(**kwargs)
        AsyncIndex.search.assert_awaited_once_with(**kwargs)  # type: ignore[attr-defined]
