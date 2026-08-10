"""Unit tests for Index.describe_index_stats() method."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
import respx

from pinecone import Index
from pinecone.errors.exceptions import PineconeValueError
from pinecone.models.vectors.responses import DescribeIndexStatsResponse

INDEX_HOST = "test-index-abc1234.svc.us-east1-gcp.pinecone.io"
INDEX_HOST_HTTPS = f"https://{INDEX_HOST}"
STATS_URL = f"{INDEX_HOST_HTTPS}/describe_index_stats"


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


def _make_index() -> Index:
    return Index(host=INDEX_HOST, api_key="test-key")


# ---------------------------------------------------------------------------
# Successful stats retrieval
# ---------------------------------------------------------------------------


class TestDescribeIndexStatsSuccess:
    """describe_index_stats returns DescribeIndexStatsResponse with all fields."""

    @respx.mock
    def test_basic_stats(self) -> None:
        respx.post(STATS_URL).mock(
            return_value=httpx.Response(
                200,
                json=_make_stats_response(
                    namespaces={
                        "ns1": {"vectorCount": 500},
                        "ns2": {"vectorCount": 500},
                    },
                    dimension=128,
                    total_vector_count=1000,
                    index_fullness=0.5,
                ),
            ),
        )
        idx = _make_index()
        result = idx.describe_index_stats()

        assert isinstance(result, DescribeIndexStatsResponse)
        assert result.dimension == 128
        assert result.total_vector_count == 1000
        assert result.index_fullness == pytest.approx(0.5)
        assert len(result.namespaces) == 2
        assert result.namespaces["ns1"].vector_count == 500
        assert result.namespaces["ns2"].vector_count == 500

    @respx.mock
    def test_stats_with_metric_and_vector_type(self) -> None:
        respx.post(STATS_URL).mock(
            return_value=httpx.Response(
                200,
                json=_make_stats_response(
                    metric="cosine",
                    vector_type="dense",
                ),
            ),
        )
        idx = _make_index()
        result = idx.describe_index_stats()

        assert result.metric == "cosine"
        assert result.vector_type == "dense"

    @respx.mock
    def test_stats_with_fullness_metrics(self) -> None:
        respx.post(STATS_URL).mock(
            return_value=httpx.Response(
                200,
                json=_make_stats_response(
                    memory_fullness=0.3,
                    storage_fullness=0.7,
                ),
            ),
        )
        idx = _make_index()
        result = idx.describe_index_stats()

        assert result.memory_fullness == pytest.approx(0.3)
        assert result.storage_fullness == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# Filter handling
# ---------------------------------------------------------------------------


class TestDescribeIndexStatsFilter:
    """Stats with metadata filter (unified-vec-0021)."""

    @respx.mock
    def test_filter_sent_in_request_body(self) -> None:
        route = respx.post(STATS_URL).mock(
            return_value=httpx.Response(200, json=_make_stats_response()),
        )
        idx = _make_index()
        idx.describe_index_stats(filter={"genre": {"$eq": "drama"}})

        request = route.calls.last.request
        import json

        body = json.loads(request.content)
        assert body["filter"] == {"genre": {"$eq": "drama"}}

    @respx.mock
    def test_no_filter_sends_empty_body(self) -> None:
        route = respx.post(STATS_URL).mock(
            return_value=httpx.Response(200, json=_make_stats_response()),
        )
        idx = _make_index()
        idx.describe_index_stats()

        request = route.calls.last.request
        import json

        body = json.loads(request.content)
        assert "filter" not in body


# ---------------------------------------------------------------------------
# Empty namespaces
# ---------------------------------------------------------------------------


class TestDescribeIndexStatsEmpty:
    """Empty index returns empty namespaces dict."""

    @respx.mock
    def test_empty_namespaces(self) -> None:
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
        idx = _make_index()
        result = idx.describe_index_stats()

        assert result.namespaces == {}
        assert result.total_vector_count == 0
        assert result.dimension == 0


# ---------------------------------------------------------------------------
# Multiple namespaces
# ---------------------------------------------------------------------------


class TestDescribeIndexStatsMultipleNamespaces:
    """Multiple namespaces each have their own vector_count."""

    @respx.mock
    def test_multiple_namespaces(self) -> None:
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
        idx = _make_index()
        result = idx.describe_index_stats()

        assert len(result.namespaces) == 3
        assert result.namespaces[""].vector_count == 100
        assert result.namespaces["prod"].vector_count == 200
        assert result.namespaces["staging"].vector_count == 300
        assert result.total_vector_count == 600


# ---------------------------------------------------------------------------
# Bracket access
# ---------------------------------------------------------------------------


class TestDescribeIndexStatsBracketAccess:
    """Response supports bracket access."""

    @respx.mock
    def test_bracket_access(self) -> None:
        respx.post(STATS_URL).mock(
            return_value=httpx.Response(
                200,
                json=_make_stats_response(dimension=256),
            ),
        )
        idx = _make_index()
        result = idx.describe_index_stats()

        assert result["dimension"] == 256
        assert result["total_vector_count"] == 1000

    @respx.mock
    def test_bracket_access_missing_key(self) -> None:
        respx.post(STATS_URL).mock(
            return_value=httpx.Response(200, json=_make_stats_response()),
        )
        idx = _make_index()
        result = idx.describe_index_stats()

        with pytest.raises(KeyError):
            result["nonexistent_field"]


# ---------------------------------------------------------------------------
# Keyword-only enforcement
# ---------------------------------------------------------------------------


class TestDescribeIndexStatsKeywordOnly:
    """All params must be keyword-only."""

    def test_positional_args_rejected(self) -> None:
        idx = _make_index()
        with pytest.raises(PineconeValueError, match="keyword-only"):
            idx.describe_index_stats({"genre": {"$eq": "drama"}})  # type: ignore[misc]


# ---------------------------------------------------------------------------
# __repr__ tests
# ---------------------------------------------------------------------------


class TestDescribeIndexStatsRepr:
    def test_describe_index_stats_repr_summary(self) -> None:
        """Shows namespace count, not full namespace dump."""
        from pinecone.models.vectors.responses import NamespaceSummary

        resp = DescribeIndexStatsResponse(
            namespaces={
                "ns1": NamespaceSummary(vector_count=100),
                "ns2": NamespaceSummary(vector_count=200),
                "ns3": NamespaceSummary(vector_count=300),
            },
            dimension=1536,
            total_vector_count=600,
            metric="cosine",
        )
        r = repr(resp)
        assert (
            r
            == "DescribeIndexStatsResponse(dimension=1536, total_vector_count=600, metric='cosine', namespaces=3)"
        )
        assert "NamespaceSummary" not in r

    def test_describe_index_stats_repr_omits_none_dimension(self) -> None:
        """dimension omitted when None."""
        resp = DescribeIndexStatsResponse(
            namespaces={},
            dimension=None,
            total_vector_count=0,
        )
        r = repr(resp)
        assert "dimension" not in r
        assert "total_vector_count=0" in r
        assert "namespaces=0" in r

    def test_describe_index_stats_repr_omits_none_metric(self) -> None:
        """metric omitted when None."""
        resp = DescribeIndexStatsResponse(
            namespaces={},
            dimension=128,
            total_vector_count=50,
            metric=None,
        )
        r = repr(resp)
        assert "metric" not in r
        assert "dimension=128" in r
