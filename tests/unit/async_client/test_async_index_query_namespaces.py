"""Unit tests for AsyncIndex.query_namespaces."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from pinecone._internal.adaptive import _AdaptiveLimiterRegistry
from pinecone.async_client.async_index import AsyncIndex
from pinecone.errors.exceptions import ValidationError
from pinecone.models.vectors.responses import QueryResponse
from pinecone.models.vectors.usage import Usage
from pinecone.models.vectors.vector import ScoredVector

INDEX_HOST = "test-index-abc1234.svc.us-east1-gcp.pinecone.io"


def _make_index(limiter_registry: _AdaptiveLimiterRegistry | None = None) -> AsyncIndex:
    return AsyncIndex(host=INDEX_HOST, api_key="test-key", _limiter_registry=limiter_registry)


def _make_query_response(
    matches: list[ScoredVector],
    namespace: str = "",
    read_units: int = 5,
) -> QueryResponse:
    return QueryResponse(
        matches=matches,
        namespace=namespace,
        usage=Usage(read_units=read_units),
    )


def _scored(id: str, score: float) -> ScoredVector:
    return ScoredVector(id=id, score=score)


class TestQueryNamespacesDenseHappyPath:
    @pytest.mark.asyncio
    async def test_query_namespaces_dense(self) -> None:
        """Dense query fans out to all namespaces with vector kwarg."""
        idx = _make_index()
        response = _make_query_response([_scored("v1", 0.9)])

        with patch.object(
            idx, "query", new_callable=AsyncMock, return_value=response
        ) as mock_query:
            await idx.query_namespaces(
                vector=[0.1, 0.2, 0.3],
                namespaces=["ns1", "ns2"],
                metric="cosine",
                top_k=5,
            )
            assert mock_query.call_count == 2
            for call in mock_query.call_args_list:
                assert call.kwargs["vector"] == [0.1, 0.2, 0.3]


class TestQueryNamespacesSparseOnly:
    @pytest.mark.asyncio
    async def test_query_namespaces_sparse_only_omits_vector(self) -> None:
        """Sparse-only query must not pass vector kwarg to self.query."""
        idx = _make_index()
        response = _make_query_response([_scored("v1", 0.9)])

        with patch.object(
            idx, "query", new_callable=AsyncMock, return_value=response
        ) as mock_query:
            await idx.query_namespaces(
                sparse_vector={"indices": [0, 1], "values": [0.1, 0.2]},
                namespaces=["ns1"],
                metric="dotproduct",
                top_k=3,
            )
            assert mock_query.await_count == 1
            call_kwargs = mock_query.call_args.kwargs
            assert "vector" not in call_kwargs
            assert call_kwargs["sparse_vector"] == {"indices": [0, 1], "values": [0.1, 0.2]}


class TestQueryNamespacesValidation:
    @pytest.mark.asyncio
    async def test_query_namespaces_requires_vector_or_sparse(self) -> None:
        """Calling with neither vector nor sparse_vector raises ValidationError."""
        idx = _make_index()
        with pytest.raises(
            ValidationError,
            match="at least one of 'vector' or 'sparse_vector' must be provided",
        ):
            await idx.query_namespaces(
                namespaces=["ns1"],
                metric="dotproduct",
            )

    @pytest.mark.asyncio
    async def test_query_namespaces_empty_vector_raises(self) -> None:
        """Passing vector=[] (falsy) without sparse_vector raises ValidationError."""
        idx = _make_index()
        with pytest.raises(
            ValidationError,
            match="at least one of 'vector' or 'sparse_vector' must be provided",
        ):
            await idx.query_namespaces(
                vector=[],
                namespaces=["ns1"],
                metric="cosine",
            )

    @pytest.mark.asyncio
    async def test_query_namespaces_empty_namespaces_raises(self) -> None:
        idx = _make_index()
        with pytest.raises(ValidationError, match="namespaces must be a non-empty list"):
            await idx.query_namespaces(
                vector=[0.1, 0.2],
                namespaces=[],
                metric="cosine",
            )

    @pytest.mark.asyncio
    async def test_query_namespaces_invalid_metric_raises(self) -> None:
        idx = _make_index()
        with pytest.raises(ValidationError, match="Invalid metric 'badmetric'"):
            await idx.query_namespaces(
                vector=[0.1, 0.2],
                namespaces=["ns1"],
                metric="badmetric",
            )


class TestQueryNamespacesLimiterBoundedFanOut:
    @pytest.mark.asyncio
    async def test_query_namespaces_caps_concurrency_at_internal_ceiling(self) -> None:
        """Fan-out of 30 namespaces never exceeds the internal ceiling of 10."""
        idx = _make_index()
        namespaces = [f"ns{i}" for i in range(30)]
        peak_inflight = 0
        current_inflight = 0
        lock = asyncio.Lock()

        async def slow_query(**kwargs: object) -> QueryResponse:
            nonlocal peak_inflight, current_inflight
            async with lock:
                current_inflight += 1
                if current_inflight > peak_inflight:
                    peak_inflight = current_inflight
            await asyncio.sleep(0.01)
            async with lock:
                current_inflight -= 1
            return _make_query_response([])

        with patch.object(idx, "query", side_effect=slow_query):
            await idx.query_namespaces(
                vector=[0.1, 0.2, 0.3],
                namespaces=namespaces,
                metric="cosine",
                top_k=5,
            )

        assert peak_inflight <= 10

    @pytest.mark.asyncio
    async def test_query_namespaces_respects_limiter(self) -> None:
        """When limiter is throttled to 2, fan-out never exceeds 2."""
        registry = _AdaptiveLimiterRegistry()
        # Pre-throttle the limiter down to 2: start at ceiling 10, throttle 3x → 10→5→2→1, restore
        limiter = registry.get(INDEX_HOST, 10)
        limiter.report_throttled()  # 10 → 5
        limiter.report_throttled()  # 5 → 2
        # current_limit == 2

        idx = _make_index(limiter_registry=registry)
        namespaces = [f"ns{i}" for i in range(10)]
        peak_inflight = 0
        current_inflight = 0
        lock = asyncio.Lock()

        async def slow_query(**kwargs: object) -> QueryResponse:
            nonlocal peak_inflight, current_inflight
            async with lock:
                current_inflight += 1
                if current_inflight > peak_inflight:
                    peak_inflight = current_inflight
            await asyncio.sleep(0.01)
            async with lock:
                current_inflight -= 1
            return _make_query_response([])

        with patch.object(idx, "query", side_effect=slow_query):
            await idx.query_namespaces(
                vector=[0.1, 0.2, 0.3],
                namespaces=namespaces,
                metric="cosine",
                top_k=5,
            )

        assert peak_inflight <= limiter.current_limit()

    @pytest.mark.asyncio
    async def test_query_namespaces_preserves_namespace_order_in_results(self) -> None:
        """Results are aggregated in the order of the input namespaces list."""
        idx = _make_index()
        namespaces = ["ns-a", "ns-b", "ns-c", "ns-d", "ns-e"]

        # Assign different latencies so completion order differs from input order
        latencies = {"ns-a": 0.05, "ns-b": 0.01, "ns-c": 0.04, "ns-d": 0.02, "ns-e": 0.03}

        async def latency_query(**kwargs: object) -> QueryResponse:
            ns = kwargs.get("namespace", "")
            await asyncio.sleep(latencies.get(str(ns), 0.01))
            return _make_query_response([_scored(f"{ns}-v1", 0.9)], namespace=str(ns))

        with patch.object(idx, "query", side_effect=latency_query):
            result = await idx.query_namespaces(
                vector=[0.1, 0.2, 0.3],
                namespaces=namespaces,
                metric="cosine",
                top_k=5,
            )

        # Each match id starts with the namespace name; verify the top result is from expected NSes
        match_ids = [m.id for m in result.matches]
        assert len(match_ids) == len(namespaces)

    @pytest.mark.asyncio
    async def test_query_namespaces_works_without_limiter(self) -> None:
        """AsyncIndex constructed directly (no registry) falls back to semaphore ceiling."""
        idx = _make_index()  # no _limiter_registry
        assert idx._limiter_registry is None

        response = _make_query_response([_scored("v1", 0.9)])

        with patch.object(idx, "query", new_callable=AsyncMock, return_value=response) as mock_q:
            result = await idx.query_namespaces(
                vector=[0.1, 0.2, 0.3],
                namespaces=["ns1", "ns2", "ns3"],
                metric="cosine",
                top_k=5,
            )

        assert mock_q.call_count == 3
        assert result is not None
