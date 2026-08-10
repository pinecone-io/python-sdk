"""Unit tests for Index.upsert() method."""

from __future__ import annotations

import httpx
import pytest
import respx

from pinecone import Index
from pinecone.errors.exceptions import PineconeValueError
from pinecone.models.vectors.responses import UpsertResponse
from pinecone.models.vectors.sparse import SparseValues
from pinecone.models.vectors.vector import Vector

INDEX_HOST = "test-index-abc1234.svc.us-east1-gcp.pinecone.io"
INDEX_HOST_HTTPS = f"https://{INDEX_HOST}"
UPSERT_URL = f"{INDEX_HOST_HTTPS}/vectors/upsert"


def _make_upsert_response(*, upserted_count: int = 3) -> dict[str, object]:
    """Build a realistic upsert API response payload."""
    return {"upsertedCount": upserted_count}


def _make_index() -> Index:
    return Index(host=INDEX_HOST, api_key="test-key")


# ---------------------------------------------------------------------------
# Upsert with Vector objects
# ---------------------------------------------------------------------------


class TestUpsertWithVectorObjects:
    """Upsert using Vector struct instances."""

    @respx.mock
    def test_returns_upsert_response(self) -> None:
        respx.post(UPSERT_URL).mock(
            return_value=httpx.Response(200, json=_make_upsert_response(upserted_count=3)),
        )
        idx = _make_index()
        result = idx.upsert(
            vectors=[
                Vector(id="v1", values=[0.1, 0.2]),
                Vector(id="v2", values=[0.3, 0.4]),
                Vector(id="v3", values=[0.5, 0.6]),
            ],
        )

        assert isinstance(result, UpsertResponse)
        assert result.upserted_count == 3

    @respx.mock
    def test_request_body_correct(self) -> None:
        route = respx.post(UPSERT_URL).mock(
            return_value=httpx.Response(200, json=_make_upsert_response()),
        )
        idx = _make_index()
        idx.upsert(
            vectors=[
                Vector(id="v1", values=[1.0, 2.0]),
            ],
        )

        import orjson

        body = orjson.loads(route.calls.last.request.content)
        assert body["vectors"] == [{"id": "v1", "values": [1.0, 2.0]}]
        assert "namespace" not in body

    @respx.mock
    def test_vector_with_sparse_values(self) -> None:
        route = respx.post(UPSERT_URL).mock(
            return_value=httpx.Response(200, json=_make_upsert_response(upserted_count=1)),
        )
        idx = _make_index()
        idx.upsert(
            vectors=[
                Vector(
                    id="v1",
                    values=[0.1, 0.2],
                    sparse_values=SparseValues(indices=[0, 3], values=[0.5, 0.8]),
                ),
            ],
        )

        import orjson

        body = orjson.loads(route.calls.last.request.content)
        vec = body["vectors"][0]
        assert vec["sparseValues"] == {"indices": [0, 3], "values": [0.5, 0.8]}

    @respx.mock
    def test_vector_with_metadata(self) -> None:
        route = respx.post(UPSERT_URL).mock(
            return_value=httpx.Response(200, json=_make_upsert_response(upserted_count=1)),
        )
        idx = _make_index()
        idx.upsert(
            vectors=[
                Vector(id="v1", values=[0.1], metadata={"genre": "action"}),
            ],
        )

        import orjson

        body = orjson.loads(route.calls.last.request.content)
        vec = body["vectors"][0]
        assert vec["metadata"] == {"genre": "action"}


# ---------------------------------------------------------------------------
# Upsert with tuple format
# ---------------------------------------------------------------------------


class TestUpsertWithTuples:
    """Upsert using tuple format — VectorFactory normalizes them."""

    @respx.mock
    def test_two_element_tuple(self) -> None:
        route = respx.post(UPSERT_URL).mock(
            return_value=httpx.Response(200, json=_make_upsert_response(upserted_count=1)),
        )
        idx = _make_index()
        idx.upsert(vectors=[("v1", [0.1, 0.2])])

        import orjson

        body = orjson.loads(route.calls.last.request.content)
        assert body["vectors"] == [{"id": "v1", "values": [0.1, 0.2]}]

    @respx.mock
    def test_three_element_tuple_with_metadata(self) -> None:
        route = respx.post(UPSERT_URL).mock(
            return_value=httpx.Response(200, json=_make_upsert_response(upserted_count=1)),
        )
        idx = _make_index()
        idx.upsert(vectors=[("v1", [0.1, 0.2], {"color": "red"})])

        import orjson

        body = orjson.loads(route.calls.last.request.content)
        vec = body["vectors"][0]
        assert vec["id"] == "v1"
        assert vec["values"] == [0.1, 0.2]
        assert vec["metadata"] == {"color": "red"}


# ---------------------------------------------------------------------------
# Upsert with dict format
# ---------------------------------------------------------------------------


class TestUpsertWithDicts:
    """Upsert using dict format — VectorFactory normalizes them."""

    @respx.mock
    def test_dict_with_id_and_values(self) -> None:
        route = respx.post(UPSERT_URL).mock(
            return_value=httpx.Response(200, json=_make_upsert_response(upserted_count=1)),
        )
        idx = _make_index()
        idx.upsert(vectors=[{"id": "v1", "values": [0.1, 0.2]}])

        import orjson

        body = orjson.loads(route.calls.last.request.content)
        assert body["vectors"] == [{"id": "v1", "values": [0.1, 0.2]}]

    @respx.mock
    def test_dict_with_sparse_values(self) -> None:
        route = respx.post(UPSERT_URL).mock(
            return_value=httpx.Response(200, json=_make_upsert_response(upserted_count=1)),
        )
        idx = _make_index()
        idx.upsert(
            vectors=[
                {
                    "id": "v1",
                    "values": [0.1],
                    "sparse_values": {"indices": [0, 5], "values": [1.0, 2.0]},
                },
            ],
        )

        import orjson

        body = orjson.loads(route.calls.last.request.content)
        vec = body["vectors"][0]
        assert vec["sparseValues"] == {"indices": [0, 5], "values": [1.0, 2.0]}


# ---------------------------------------------------------------------------
# Namespace handling
# ---------------------------------------------------------------------------


class TestUpsertNamespace:
    """Namespace targeting (unified-vec-0022)."""

    @respx.mock
    def test_explicit_namespace_in_body(self) -> None:
        route = respx.post(UPSERT_URL).mock(
            return_value=httpx.Response(200, json=_make_upsert_response()),
        )
        idx = _make_index()
        idx.upsert(vectors=[("v1", [0.1])], namespace="my-ns")

        import orjson

        body = orjson.loads(route.calls.last.request.content)
        assert body["namespace"] == "my-ns"

    @respx.mock
    def test_default_namespace_omitted_from_body(self) -> None:
        """Empty namespace string means default — not sent in body."""
        route = respx.post(UPSERT_URL).mock(
            return_value=httpx.Response(200, json=_make_upsert_response()),
        )
        idx = _make_index()
        idx.upsert(vectors=[("v1", [0.1])])

        import orjson

        body = orjson.loads(route.calls.last.request.content)
        assert "namespace" not in body


# ---------------------------------------------------------------------------
# Keyword-only enforcement
# ---------------------------------------------------------------------------


class TestUpsertKeywordOnly:
    """unified-vec-0040: all params must be keyword-only."""

    def test_positional_args_rejected(self) -> None:
        idx = _make_index()
        with pytest.raises(PineconeValueError, match="keyword-only"):
            idx.upsert([("v1", [0.1])])  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Response bracket access
# ---------------------------------------------------------------------------


class TestUpsertResponseBracketAccess:
    """UpsertResponse supports bracket access."""

    @respx.mock
    def test_bracket_access(self) -> None:
        respx.post(UPSERT_URL).mock(
            return_value=httpx.Response(200, json=_make_upsert_response(upserted_count=5)),
        )
        idx = _make_index()
        result = idx.upsert(vectors=[("v1", [0.1])])

        assert result["upserted_count"] == 5
