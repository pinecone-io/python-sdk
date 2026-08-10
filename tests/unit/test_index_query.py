"""Unit tests for Index.query() method."""

from __future__ import annotations

import httpx
import pytest
import respx

from pinecone import Index
from pinecone.errors.exceptions import PineconeValueError, ValidationError
from pinecone.models.vectors.responses import QueryResponse

INDEX_HOST = "test-index-abc1234.svc.us-east1-gcp.pinecone.io"
INDEX_HOST_HTTPS = f"https://{INDEX_HOST}"
QUERY_URL = f"{INDEX_HOST_HTTPS}/query"


def _make_query_response(
    *,
    matches: list[dict[str, object]] | None = None,
    namespace: str = "",
    usage: dict[str, int] | None = None,
) -> dict[str, object]:
    """Build a realistic query API response payload."""
    return {
        "matches": matches or [],
        "namespace": namespace,
        "usage": usage or {"readUnits": 5},
    }


def _make_index() -> Index:
    return Index(host=INDEX_HOST, api_key="test-key")


# ---------------------------------------------------------------------------
# Successful queries
# ---------------------------------------------------------------------------


class TestQueryWithVector:
    """Query using a dense vector."""

    @respx.mock
    def test_returns_query_response(self) -> None:
        respx.post(QUERY_URL).mock(
            return_value=httpx.Response(
                200,
                json=_make_query_response(
                    matches=[
                        {"id": "vec1", "score": 0.95},
                        {"id": "vec2", "score": 0.80},
                    ],
                ),
            ),
        )
        idx = _make_index()
        result = idx.query(top_k=2, vector=[0.1, 0.2, 0.3])

        assert isinstance(result, QueryResponse)
        assert len(result.matches) == 2
        assert result.matches[0].id == "vec1"
        assert result.matches[0].score == pytest.approx(0.95)
        assert result.matches[1].id == "vec2"

    @respx.mock
    def test_request_body_correct(self) -> None:
        route = respx.post(QUERY_URL).mock(
            return_value=httpx.Response(200, json=_make_query_response()),
        )
        idx = _make_index()
        idx.query(top_k=5, vector=[1.0, 2.0])

        request = route.calls.last.request
        import orjson

        body = orjson.loads(request.content)
        assert body["topK"] == 5
        assert body["vector"] == [1.0, 2.0]
        assert body["includeValues"] is False
        assert body["includeMetadata"] is False


class TestQueryWithId:
    """Query using a stored vector ID."""

    @respx.mock
    def test_returns_query_response(self) -> None:
        respx.post(QUERY_URL).mock(
            return_value=httpx.Response(
                200,
                json=_make_query_response(
                    matches=[{"id": "vec1", "score": 1.0}],
                ),
            ),
        )
        idx = _make_index()
        result = idx.query(top_k=1, id="vec1")

        assert isinstance(result, QueryResponse)
        assert len(result.matches) == 1
        assert result.matches[0].id == "vec1"

    @respx.mock
    def test_request_body_has_id(self) -> None:
        route = respx.post(QUERY_URL).mock(
            return_value=httpx.Response(200, json=_make_query_response()),
        )
        idx = _make_index()
        idx.query(top_k=3, id="existing-vec")

        import orjson

        body = orjson.loads(route.calls.last.request.content)
        assert body["id"] == "existing-vec"
        assert "vector" not in body


# ---------------------------------------------------------------------------
# Namespace handling
# ---------------------------------------------------------------------------


class TestQueryNamespace:
    """Namespace targeting (unified-vec-0022)."""

    @respx.mock
    def test_default_namespace_omitted_from_body(self) -> None:
        """Empty namespace string means default — not sent in body."""
        route = respx.post(QUERY_URL).mock(
            return_value=httpx.Response(200, json=_make_query_response()),
        )
        idx = _make_index()
        idx.query(top_k=1, vector=[0.1])

        import orjson

        body = orjson.loads(route.calls.last.request.content)
        assert "namespace" not in body

    @respx.mock
    def test_explicit_namespace_in_body(self) -> None:
        route = respx.post(QUERY_URL).mock(
            return_value=httpx.Response(200, json=_make_query_response(namespace="prod")),
        )
        idx = _make_index()
        idx.query(top_k=1, vector=[0.1], namespace="prod")

        import orjson

        body = orjson.loads(route.calls.last.request.content)
        assert body["namespace"] == "prod"

    @respx.mock
    def test_response_includes_namespace(self) -> None:
        """unified-vec-0007: response includes queried namespace."""
        respx.post(QUERY_URL).mock(
            return_value=httpx.Response(200, json=_make_query_response(namespace="test-ns")),
        )
        idx = _make_index()
        result = idx.query(top_k=1, vector=[0.1], namespace="test-ns")
        assert result.namespace == "test-ns"


# ---------------------------------------------------------------------------
# Include flags
# ---------------------------------------------------------------------------


class TestQueryIncludeFlags:
    """unified-vec-0023: values/metadata only when requested."""

    @respx.mock
    def test_include_values_in_body(self) -> None:
        route = respx.post(QUERY_URL).mock(
            return_value=httpx.Response(200, json=_make_query_response()),
        )
        idx = _make_index()
        idx.query(top_k=1, vector=[0.1], include_values=True)

        import orjson

        body = orjson.loads(route.calls.last.request.content)
        assert body["includeValues"] is True

    @respx.mock
    def test_include_metadata_in_body(self) -> None:
        route = respx.post(QUERY_URL).mock(
            return_value=httpx.Response(200, json=_make_query_response()),
        )
        idx = _make_index()
        idx.query(top_k=1, vector=[0.1], include_metadata=True)

        import orjson

        body = orjson.loads(route.calls.last.request.content)
        assert body["includeMetadata"] is True


# ---------------------------------------------------------------------------
# Filter passthrough
# ---------------------------------------------------------------------------


class TestQueryFilter:
    """Filter dict is passed through in body."""

    @respx.mock
    def test_filter_in_body(self) -> None:
        route = respx.post(QUERY_URL).mock(
            return_value=httpx.Response(200, json=_make_query_response()),
        )
        idx = _make_index()
        my_filter = {"genre": {"$eq": "action"}}
        idx.query(top_k=5, vector=[0.1], filter=my_filter)

        import orjson

        body = orjson.loads(route.calls.last.request.content)
        assert body["filter"] == {"genre": {"$eq": "action"}}


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestQueryValidation:
    """Input validation (unified-vec-0038, 0039, 0040)."""

    def test_top_k_zero_raises(self) -> None:
        idx = _make_index()
        with pytest.raises(ValidationError, match="top_k must be a positive integer"):
            idx.query(top_k=0, vector=[0.1])

    def test_top_k_negative_raises(self) -> None:
        idx = _make_index()
        with pytest.raises(ValidationError, match="top_k must be a positive integer"):
            idx.query(top_k=-1, vector=[0.1])

    def test_both_vector_and_id_raises(self) -> None:
        idx = _make_index()
        with pytest.raises(ValidationError, match="not both"):
            idx.query(top_k=1, vector=[0.1], id="vec1")

    def test_neither_vector_nor_id_raises(self) -> None:
        idx = _make_index()
        with pytest.raises(ValidationError, match="At least one of vector, id, or sparse_vector"):
            idx.query(top_k=1)

    def test_positional_args_rejected(self) -> None:
        """unified-vec-0040: all params must be keyword-only."""
        idx = _make_index()
        with pytest.raises(PineconeValueError, match="keyword-only"):
            idx.query(10)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Sparse vector
# ---------------------------------------------------------------------------


class TestQuerySparseVector:
    """Sparse vector passthrough."""

    @respx.mock
    def test_sparse_vector_in_body(self) -> None:
        route = respx.post(QUERY_URL).mock(
            return_value=httpx.Response(200, json=_make_query_response()),
        )
        idx = _make_index()
        sparse = {"indices": [0, 3], "values": [0.5, 0.8]}
        idx.query(top_k=5, vector=[0.1], sparse_vector=sparse)

        import orjson

        body = orjson.loads(route.calls.last.request.content)
        assert body["sparseVector"] == {"indices": [0, 3], "values": [0.5, 0.8]}

    @respx.mock
    def test_sparse_vector_struct_in_body(self) -> None:
        from pinecone.models.vectors.sparse import SparseValues

        route = respx.post(QUERY_URL).mock(
            return_value=httpx.Response(200, json=_make_query_response()),
        )
        idx = _make_index()
        sparse = SparseValues(indices=[0, 1], values=[0.5, 0.5])
        idx.query(top_k=5, vector=[0.1], sparse_vector=sparse)

        import orjson

        body = orjson.loads(route.calls.last.request.content)
        assert body["sparseVector"] == {"indices": [0, 1], "values": [0.5, 0.5]}


# ---------------------------------------------------------------------------
# DRN parameters (scan_factor, max_candidates)
# ---------------------------------------------------------------------------


class TestQueryDrnParams:
    """DRN optimization parameters (unified-wf-0005)."""

    @respx.mock
    def test_query_scan_factor_included_in_body(self) -> None:
        route = respx.post(QUERY_URL).mock(
            return_value=httpx.Response(200, json=_make_query_response()),
        )
        idx = _make_index()
        idx.query(top_k=10, vector=[0.1], scan_factor=2.0)

        import orjson

        body = orjson.loads(route.calls.last.request.content)
        assert body["scanFactor"] == 2.0

    @respx.mock
    def test_query_max_candidates_included_in_body(self) -> None:
        route = respx.post(QUERY_URL).mock(
            return_value=httpx.Response(200, json=_make_query_response()),
        )
        idx = _make_index()
        idx.query(top_k=10, vector=[0.1], max_candidates=5000)

        import orjson

        body = orjson.loads(route.calls.last.request.content)
        assert body["maxCandidates"] == 5000

    @respx.mock
    def test_query_scan_factor_and_max_candidates_omitted_when_none(self) -> None:
        route = respx.post(QUERY_URL).mock(
            return_value=httpx.Response(200, json=_make_query_response()),
        )
        idx = _make_index()
        idx.query(top_k=10, vector=[0.1])

        import orjson

        body = orjson.loads(route.calls.last.request.content)
        assert "scanFactor" not in body
        assert "maxCandidates" not in body

    @respx.mock
    def test_query_both_drn_params(self) -> None:
        route = respx.post(QUERY_URL).mock(
            return_value=httpx.Response(200, json=_make_query_response()),
        )
        idx = _make_index()
        idx.query(top_k=10, vector=[0.1], scan_factor=1.5, max_candidates=3000)

        import orjson

        body = orjson.loads(route.calls.last.request.content)
        assert body["scanFactor"] == 1.5
        assert body["maxCandidates"] == 3000
