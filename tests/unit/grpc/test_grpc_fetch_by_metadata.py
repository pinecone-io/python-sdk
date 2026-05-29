"""Unit tests for GrpcIndex.fetch_by_metadata() REST-delegation method."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest
import respx

from pinecone.grpc import GrpcIndex
from pinecone.models.vectors.responses import FetchByMetadataResponse

_MOCK_GRPC_MODULE_PATH = "pinecone._grpc"
_INDEX_HOST = "test-index-abc123.svc.pinecone.io"
_INDEX_HOST_HTTPS = f"https://{_INDEX_HOST}"
_FETCH_BY_META_URL = f"{_INDEX_HOST_HTTPS}/vectors/fetch_by_metadata"


def _make_grpc_index(mock_channel: MagicMock) -> GrpcIndex:
    mock_module = MagicMock()
    mock_module.GrpcChannel.return_value = mock_channel
    with patch.dict("sys.modules", {_MOCK_GRPC_MODULE_PATH: mock_module}):
        return GrpcIndex(
            host=_INDEX_HOST,
            api_key="test-api-key",
        )


def _make_response(
    *,
    vectors: dict[str, dict[str, Any]] | None = None,
    namespace: str = "",
    usage: dict[str, int] | None = None,
    pagination: dict[str, str] | None = None,
) -> dict[str, object]:
    resp: dict[str, object] = {
        "vectors": vectors or {},
        "namespace": namespace,
        "usage": usage or {"readUnits": 5},
    }
    if pagination is not None:
        resp["pagination"] = pagination
    return resp


@pytest.fixture
def mock_channel() -> MagicMock:
    return MagicMock()


@pytest.fixture
def grpc_index(mock_channel: MagicMock) -> GrpcIndex:
    return _make_grpc_index(mock_channel)


# ---------------------------------------------------------------------------
# Basic success
# ---------------------------------------------------------------------------


class TestGrpcFetchByMetadataBasic:
    """fetch_by_metadata returns FetchByMetadataResponse with vectors."""

    @respx.mock
    def test_fetch_by_metadata_basic(self, mock_channel: MagicMock) -> None:
        respx.post(_FETCH_BY_META_URL).mock(
            return_value=httpx.Response(
                200,
                json=_make_response(
                    vectors={
                        "vec1": {"id": "vec1", "values": [0.1, 0.2]},
                        "vec2": {"id": "vec2", "values": [0.3, 0.4]},
                    },
                ),
            ),
        )
        idx = _make_grpc_index(mock_channel)
        result = idx.fetch_by_metadata(filter={"genre": "comedy"})

        assert isinstance(result, FetchByMetadataResponse)
        assert len(result.vectors) == 2
        assert result.vectors["vec1"].id == "vec1"
        assert result.vectors["vec2"].id == "vec2"


# ---------------------------------------------------------------------------
# Request body construction
# ---------------------------------------------------------------------------


class TestGrpcFetchByMetadataRequestBody:
    """Verify the POST body is built correctly from parameters."""

    @respx.mock
    def test_fetch_by_metadata_with_limit_and_pagination(self, mock_channel: MagicMock) -> None:
        route = respx.post(_FETCH_BY_META_URL).mock(
            return_value=httpx.Response(200, json=_make_response()),
        )
        idx = _make_grpc_index(mock_channel)
        idx.fetch_by_metadata(
            filter={"genre": {"$eq": "comedy"}},
            limit=50,
            pagination_token="tok",
        )

        body = json.loads(route.calls.last.request.content)
        assert body["filter"] == {"genre": {"$eq": "comedy"}}
        assert body["limit"] == 50
        assert body["paginationToken"] == "tok"

    @respx.mock
    def test_fetch_by_metadata_sends_namespace(self, mock_channel: MagicMock) -> None:
        route = respx.post(_FETCH_BY_META_URL).mock(
            return_value=httpx.Response(200, json=_make_response()),
        )
        idx = _make_grpc_index(mock_channel)
        idx.fetch_by_metadata(filter={"a": 1}, namespace="my-ns")

        body = json.loads(route.calls.last.request.content)
        assert body["namespace"] == "my-ns"

    @respx.mock
    def test_fetch_by_metadata_omits_optional_fields(self, mock_channel: MagicMock) -> None:
        route = respx.post(_FETCH_BY_META_URL).mock(
            return_value=httpx.Response(200, json=_make_response()),
        )
        idx = _make_grpc_index(mock_channel)
        idx.fetch_by_metadata(filter={"a": 1})

        body = json.loads(route.calls.last.request.content)
        assert "namespace" not in body
        assert "limit" not in body
        assert "paginationToken" not in body


# ---------------------------------------------------------------------------
# Pagination response
# ---------------------------------------------------------------------------


class TestGrpcFetchByMetadataResponsePagination:
    """Verify pagination token is correctly deserialized."""

    @respx.mock
    def test_fetch_by_metadata_response_pagination(self, mock_channel: MagicMock) -> None:
        respx.post(_FETCH_BY_META_URL).mock(
            return_value=httpx.Response(
                200,
                json=_make_response(
                    vectors={"v1": {"id": "v1", "values": [0.1]}},
                    pagination={"next": "token123"},
                ),
            ),
        )
        idx = _make_grpc_index(mock_channel)
        result = idx.fetch_by_metadata(filter={"a": 1})

        assert result.pagination is not None
        assert result.pagination.next == "token123"

    @respx.mock
    def test_fetch_by_metadata_no_pagination(self, mock_channel: MagicMock) -> None:
        respx.post(_FETCH_BY_META_URL).mock(
            return_value=httpx.Response(
                200,
                json=_make_response(vectors={"v1": {"id": "v1", "values": [0.1]}}),
            ),
        )
        idx = _make_grpc_index(mock_channel)
        result = idx.fetch_by_metadata(filter={"a": 1})

        assert result.pagination is None


# ---------------------------------------------------------------------------
# Limit validation
# ---------------------------------------------------------------------------


class TestGrpcFetchByMetadataInvalidLimit:
    """limit must be >= 1; limit=0 or negative raises before any HTTP call."""

    def test_fetch_by_metadata_invalid_limit_zero(self, mock_channel: MagicMock) -> None:
        idx = _make_grpc_index(mock_channel)
        with pytest.raises(Exception, match="limit"):
            idx.fetch_by_metadata(filter={"a": "b"}, limit=0)

    def test_fetch_by_metadata_invalid_limit_negative(self, mock_channel: MagicMock) -> None:
        idx = _make_grpc_index(mock_channel)
        with pytest.raises(Exception, match="limit"):
            idx.fetch_by_metadata(filter={"a": "b"}, limit=-1)

    @respx.mock
    def test_fetch_by_metadata_limit_one_passes(self, mock_channel: MagicMock) -> None:
        respx.post(_FETCH_BY_META_URL).mock(
            return_value=httpx.Response(200, json=_make_response()),
        )
        idx = _make_grpc_index(mock_channel)
        # Should not raise
        idx.fetch_by_metadata(filter={"a": "b"}, limit=1)
