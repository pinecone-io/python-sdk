"""Unit tests for AsyncIndex.describe_index_stats() method."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx

from pinecone.async_client.async_index import AsyncIndex
from pinecone.errors.exceptions import PineconeValueError
from pinecone.models.vectors.responses import DescribeIndexStatsResponse

INDEX_HOST = "my-index-abc123.svc.pinecone.io"
INDEX_HOST_HTTPS = f"https://{INDEX_HOST}"
STATS_URL = f"{INDEX_HOST_HTTPS}/describe_index_stats"

STATS_RESPONSE: dict[str, Any] = {
    "namespaces": {
        "ns1": {"vectorCount": 100},
        "ns2": {"vectorCount": 200},
    },
    "dimension": 128,
    "indexFullness": 0.5,
    "totalVectorCount": 300,
    "metric": "cosine",
    "vectorType": "dense",
}


def _make_stats_response(
    *,
    namespaces: dict[str, dict[str, Any]] | None = None,
    dimension: int = 128,
    index_fullness: float = 0.5,
    total_vector_count: int = 1000,
    metric: str | None = None,
    vector_type: str | None = None,
    memory_fullness: float | None = None,
    storage_fullness: float | None = None,
) -> dict[str, object]:
    """Build a realistic describe_index_stats API response payload."""
    result: dict[str, object] = {
        "namespaces": namespaces or {},
        "dimension": dimension,
        "indexFullness": index_fullness,
        "totalVectorCount": total_vector_count,
    }
    if metric is not None:
        result["metric"] = metric
    if vector_type is not None:
        result["vectorType"] = vector_type
    if memory_fullness is not None:
        result["memoryFullness"] = memory_fullness
    if storage_fullness is not None:
        result["storageFullness"] = storage_fullness
    return result


def _make_async_index() -> AsyncIndex:
    return AsyncIndex(host=INDEX_HOST, api_key="test-key")


class TestAsyncDescribeIndexStats:
    """Async describe_index_stats tests."""

    @respx.mock
    @pytest.mark.anyio
    async def test_async_describe_index_stats_basic(self) -> None:
        respx.post(STATS_URL).mock(
            return_value=httpx.Response(200, json=STATS_RESPONSE),
        )
        idx = _make_async_index()
        result = await idx.describe_index_stats()

        assert isinstance(result, DescribeIndexStatsResponse)
        assert result.dimension == 128
        assert result.total_vector_count == 300
        assert result.index_fullness == pytest.approx(0.5)
        assert result.metric == "cosine"
        assert len(result.namespaces) == 2
        assert result.namespaces["ns1"].vector_count == 100
        assert result.namespaces["ns2"].vector_count == 200

    @respx.mock
    @pytest.mark.anyio
    async def test_async_describe_index_stats_with_filter(self) -> None:
        route = respx.post(STATS_URL).mock(
            return_value=httpx.Response(200, json=STATS_RESPONSE),
        )
        idx = _make_async_index()
        await idx.describe_index_stats(filter={"genre": {"$eq": "sci-fi"}})

        request = route.calls.last.request
        body = json.loads(request.content)
        assert body["filter"] == {"genre": {"$eq": "sci-fi"}}

    @respx.mock
    @pytest.mark.anyio
    async def test_async_describe_index_stats_no_filter(self) -> None:
        route = respx.post(STATS_URL).mock(
            return_value=httpx.Response(200, json=STATS_RESPONSE),
        )
        idx = _make_async_index()
        await idx.describe_index_stats()

        request = route.calls.last.request
        body = json.loads(request.content)
        assert body == {}

    @pytest.mark.anyio
    async def test_async_describe_index_stats_keyword_only(self) -> None:
        idx = _make_async_index()
        with pytest.raises(PineconeValueError, match="keyword-only"):
            await idx.describe_index_stats({"genre": {"$eq": "sci-fi"}})  # type: ignore[misc]

    @respx.mock
    @pytest.mark.anyio
    async def test_async_describe_index_stats_with_metric_and_vector_type(self) -> None:
        respx.post(STATS_URL).mock(
            return_value=httpx.Response(
                200,
                json=_make_stats_response(metric="cosine", vector_type="dense"),
            ),
        )
        idx = _make_async_index()
        result = await idx.describe_index_stats()

        assert result.metric == "cosine"
        assert result.vector_type == "dense"

    @respx.mock
    @pytest.mark.anyio
    async def test_async_describe_index_stats_with_fullness_metrics(self) -> None:
        respx.post(STATS_URL).mock(
            return_value=httpx.Response(
                200,
                json=_make_stats_response(memory_fullness=0.3, storage_fullness=0.7),
            ),
        )
        idx = _make_async_index()
        result = await idx.describe_index_stats()

        assert result.memory_fullness == pytest.approx(0.3)
        assert result.storage_fullness == pytest.approx(0.7)

    @respx.mock
    @pytest.mark.anyio
    async def test_async_describe_index_stats_empty_namespaces(self) -> None:
        respx.post(STATS_URL).mock(
            return_value=httpx.Response(
                200,
                json=_make_stats_response(
                    namespaces={},
                    dimension=0,
                    total_vector_count=0,
                    index_fullness=0.0,
                ),
            ),
        )
        idx = _make_async_index()
        result = await idx.describe_index_stats()

        assert result.namespaces == {}
        assert result.total_vector_count == 0
        assert result.dimension == 0

    @respx.mock
    @pytest.mark.anyio
    async def test_async_describe_index_stats_multiple_namespaces(self) -> None:
        respx.post(STATS_URL).mock(
            return_value=httpx.Response(
                200,
                json=_make_stats_response(
                    namespaces={
                        "": {"vectorCount": 100},
                        "prod": {"vectorCount": 200},
                        "staging": {"vectorCount": 300},
                    },
                    total_vector_count=600,
                ),
            ),
        )
        idx = _make_async_index()
        result = await idx.describe_index_stats()

        assert len(result.namespaces) == 3
        assert result.namespaces[""].vector_count == 100
        assert result.namespaces["prod"].vector_count == 200
        assert result.namespaces["staging"].vector_count == 300
        assert result.total_vector_count == 600

    @respx.mock
    @pytest.mark.anyio
    async def test_async_describe_index_stats_bracket_access(self) -> None:
        respx.post(STATS_URL).mock(
            return_value=httpx.Response(
                200,
                json=_make_stats_response(dimension=256),
            ),
        )
        idx = _make_async_index()
        result = await idx.describe_index_stats()

        assert result["dimension"] == 256
        assert result["total_vector_count"] == 1000

    @respx.mock
    @pytest.mark.anyio
    async def test_async_describe_index_stats_bracket_access_missing_key(self) -> None:
        respx.post(STATS_URL).mock(
            return_value=httpx.Response(200, json=_make_stats_response()),
        )
        idx = _make_async_index()
        result = await idx.describe_index_stats()

        with pytest.raises(KeyError):
            result["nonexistent"]
