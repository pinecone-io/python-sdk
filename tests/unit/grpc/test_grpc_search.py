"""Unit tests for GrpcIndex.search() REST-delegation method."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest
import respx

from pinecone.grpc import GrpcIndex
from pinecone.errors.exceptions import ValidationError
from pinecone.models.vectors.search import SearchRecordsResponse

_MOCK_GRPC_MODULE_PATH = "pinecone._grpc"
_INDEX_HOST = "test-index-abc123.svc.pinecone.io"
_INDEX_HOST_HTTPS = f"https://{_INDEX_HOST}"
_SEARCH_NS = "test-ns"
_SEARCH_URL = f"{_INDEX_HOST_HTTPS}/records/namespaces/{_SEARCH_NS}/search"

_SEARCH_RESPONSE: dict[str, object] = {
    "result": {
        "hits": [
            {"_id": "r1", "_score": 0.95, "fields": {"chunk_text": "hello world"}},
            {"_id": "r2", "_score": 0.82, "fields": {"chunk_text": "foo bar"}},
        ]
    },
    "usage": {"read_units": 5, "embed_total_tokens": 10},
}


def _make_grpc_index() -> GrpcIndex:
    mock_channel = MagicMock()
    mock_module = MagicMock()
    mock_module.GrpcChannel.return_value = mock_channel
    with patch.dict("sys.modules", {_MOCK_GRPC_MODULE_PATH: mock_module}):
        return GrpcIndex(
            host=_INDEX_HOST,
            api_key="test-api-key",
        )


@pytest.fixture
def grpc_index() -> GrpcIndex:
    return _make_grpc_index()


class TestGrpcSearchDenseVectorWrapped:
    """GrpcIndex.search() must wrap bare list[float] vectors as {"values": [...]}."""

    @respx.mock
    def test_grpc_search_dense_vector_wrapped(self, grpc_index: GrpcIndex) -> None:
        """A bare list[float] vector must arrive as {"values": [...]} — not a bare list."""
        route = respx.post(_SEARCH_URL).mock(
            return_value=httpx.Response(200, json=_SEARCH_RESPONSE),
        )
        grpc_index.search(namespace=_SEARCH_NS, top_k=5, vector=[0.1, 0.2])

        body = json.loads(route.calls.last.request.content)
        assert body["query"]["vector"] == {"values": [0.1, 0.2]}, (
            "GrpcIndex.search() must wrap a bare list[float] as {'values': [...]}, "
            "not send a bare list — the REST endpoint requires dict form"
        )

    @respx.mock
    def test_grpc_search_dense_vector_not_bare_list(self, grpc_index: GrpcIndex) -> None:
        """Confirm the body does NOT contain a bare list at query.vector."""
        route = respx.post(_SEARCH_URL).mock(
            return_value=httpx.Response(200, json=_SEARCH_RESPONSE),
        )
        grpc_index.search(namespace=_SEARCH_NS, top_k=5, vector=[0.1, 0.2])

        body = json.loads(route.calls.last.request.content)
        assert not isinstance(body["query"]["vector"], list), (
            "query.vector must be a dict, not a bare list"
        )

    @respx.mock
    def test_grpc_search_dict_vector_passthrough(self, grpc_index: GrpcIndex) -> None:
        """A dict vector passed directly must be forwarded unchanged."""
        route = respx.post(_SEARCH_URL).mock(
            return_value=httpx.Response(200, json=_SEARCH_RESPONSE),
        )
        grpc_index.search(
            namespace=_SEARCH_NS,
            top_k=5,
            vector={"values": [0.3, 0.4]},
        )

        body = json.loads(route.calls.last.request.content)
        assert body["query"]["vector"] == {"values": [0.3, 0.4]}

    @respx.mock
    def test_grpc_search_inputs(self, grpc_index: GrpcIndex) -> None:
        """Text inputs are forwarded correctly."""
        route = respx.post(_SEARCH_URL).mock(
            return_value=httpx.Response(200, json=_SEARCH_RESPONSE),
        )
        grpc_index.search(
            namespace=_SEARCH_NS,
            top_k=10,
            inputs={"text": "hello world"},
        )

        body = json.loads(route.calls.last.request.content)
        assert body["query"]["inputs"] == {"text": "hello world"}
        assert body["query"]["top_k"] == 10

    @respx.mock
    def test_grpc_search_returns_search_records_response(
        self, grpc_index: GrpcIndex
    ) -> None:
        """search() returns a SearchRecordsResponse."""
        respx.post(_SEARCH_URL).mock(
            return_value=httpx.Response(200, json=_SEARCH_RESPONSE),
        )
        result = grpc_index.search(namespace=_SEARCH_NS, top_k=5, inputs={"text": "hi"})

        assert isinstance(result, SearchRecordsResponse)
        assert len(result.result.hits) == 2
        assert result.result.hits[0].id == "r1"
        assert result.usage.read_units == 5


class TestGrpcSearchValidation:
    """GrpcIndex.search() input validation."""

    def test_grpc_search_namespace_not_string_raises(self, grpc_index: GrpcIndex) -> None:
        with pytest.raises(ValidationError, match="namespace must be a string"):
            grpc_index.search(namespace=123, top_k=5, inputs={"text": "x"})  # type: ignore[arg-type]

    def test_grpc_search_namespace_empty_raises(self, grpc_index: GrpcIndex) -> None:
        with pytest.raises(ValidationError, match="namespace must be a non-empty string"):
            grpc_index.search(namespace="", top_k=5, inputs={"text": "x"})

    def test_grpc_search_namespace_whitespace_raises(self, grpc_index: GrpcIndex) -> None:
        with pytest.raises(ValidationError, match="namespace must be a non-empty string"):
            grpc_index.search(namespace="   ", top_k=5, inputs={"text": "x"})


class TestGrpcSearchRecordsAlias:
    """GrpcIndex.search_records() delegates to search(), inheriting the dense-vector fix."""

    @respx.mock
    def test_search_records_dense_vector_wrapped(self, grpc_index: GrpcIndex) -> None:
        """search_records() must also wrap list[float] as {'values': [...]}."""
        route = respx.post(_SEARCH_URL).mock(
            return_value=httpx.Response(200, json=_SEARCH_RESPONSE),
        )
        grpc_index.search_records(namespace=_SEARCH_NS, top_k=5, vector=[0.5, 0.6])

        body = json.loads(route.calls.last.request.content)
        assert body["query"]["vector"] == {"values": [0.5, 0.6]}
