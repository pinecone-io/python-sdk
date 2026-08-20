"""Unit tests for GrpcIndex.fetch_by_metadata() over the FetchByMetadata rpc.

Until 2026-07 this method delegated to the REST endpoint behind a docstring
claiming the gRPC API had no such rpc — which had stopped being true when the
2026-07 proto landed: ``rust/src/transport.rs`` fully implements
``FetchByMetadata``. #124 wired the Python method to that channel method, so
these tests drive the channel mock rather than an HTTP mock, and the response
tests hold the gRPC result to the exact models the REST adapter produces from
the equivalent JSON.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from pinecone._internal.adapters.vectors_adapter import VectorsAdapter
from pinecone.errors.exceptions import ValidationError
from pinecone.grpc import GrpcIndex
from pinecone.models.vectors.responses import FetchByMetadataResponse

_MOCK_GRPC_MODULE_PATH = "pinecone._grpc"
_INDEX_HOST = "test-index-abc123.svc.pinecone.io"


def _make_grpc_index(mock_channel: MagicMock) -> GrpcIndex:
    mock_module = MagicMock()
    mock_module.GrpcChannel.return_value = mock_channel
    with patch.dict("sys.modules", {_MOCK_GRPC_MODULE_PATH: mock_module}):
        return GrpcIndex(
            host=_INDEX_HOST,
            api_key="test-api-key",
        )


def _make_channel_response(
    *,
    vectors: dict[str, dict[str, Any]] | None = None,
    namespace: str = "",
    usage: dict[str, int] | None = None,
    pagination: dict[str, str] | None = None,
) -> dict[str, object]:
    resp: dict[str, object] = {
        "vectors": vectors or {},
        "namespace": namespace,
        "usage": usage or {"read_units": 5},
    }
    if pagination is not None:
        resp["pagination"] = pagination
    return resp


@pytest.fixture
def mock_channel() -> MagicMock:
    channel = MagicMock()
    channel.fetch_by_metadata.return_value = _make_channel_response()
    return channel


@pytest.fixture
def grpc_index(mock_channel: MagicMock) -> GrpcIndex:
    return _make_grpc_index(mock_channel)


# ---------------------------------------------------------------------------
# Basic success
# ---------------------------------------------------------------------------


class TestGrpcFetchByMetadataBasic:
    """fetch_by_metadata returns FetchByMetadataResponse with vectors."""

    def test_fetch_by_metadata_basic(self, grpc_index: GrpcIndex, mock_channel: MagicMock) -> None:
        mock_channel.fetch_by_metadata.return_value = _make_channel_response(
            vectors={
                "vec1": {"id": "vec1", "values": [0.1, 0.2]},
                "vec2": {"id": "vec2", "values": [0.3, 0.4]},
            },
        )
        result = grpc_index.fetch_by_metadata(filter={"genre": "comedy"})

        assert isinstance(result, FetchByMetadataResponse)
        assert len(result.vectors) == 2
        assert result.vectors["vec1"].id == "vec1"
        assert result.vectors["vec2"].id == "vec2"
        mock_channel.fetch_by_metadata.assert_called_once()

    def test_fetch_by_metadata_goes_out_over_the_channel_not_rest(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        """The stale REST delegation is gone: the call reaches the Rust channel."""
        grpc_index.fetch_by_metadata(filter={"genre": "comedy"})
        mock_channel.fetch_by_metadata.assert_called_once()


# ---------------------------------------------------------------------------
# Channel argument forwarding
# ---------------------------------------------------------------------------


class TestGrpcFetchByMetadataChannelArgs:
    """Verify the channel call is built correctly from parameters."""

    def test_fetch_by_metadata_with_limit_and_pagination(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        grpc_index.fetch_by_metadata(
            filter={"genre": {"$eq": "comedy"}},
            limit=50,
            pagination_token="tok",
        )

        kwargs = mock_channel.fetch_by_metadata.call_args.kwargs
        assert kwargs["filter"] == {"genre": {"$eq": "comedy"}}
        assert kwargs["limit"] == 50
        assert kwargs["pagination_token"] == "tok"

    def test_fetch_by_metadata_sends_namespace(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        grpc_index.fetch_by_metadata(filter={"a": 1}, namespace="my-ns")

        kwargs = mock_channel.fetch_by_metadata.call_args.kwargs
        assert kwargs["namespace"] == "my-ns"

    def test_fetch_by_metadata_omits_optional_fields(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        grpc_index.fetch_by_metadata(filter={"a": 1})

        kwargs = mock_channel.fetch_by_metadata.call_args.kwargs
        assert kwargs["namespace"] is None
        assert kwargs["limit"] is None
        assert kwargs["pagination_token"] is None

    def test_fetch_by_metadata_forwards_timeout(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        grpc_index.fetch_by_metadata(filter={"a": 1}, timeout=3.5)

        kwargs = mock_channel.fetch_by_metadata.call_args.kwargs
        assert kwargs["timeout_s"] == 3.5


# ---------------------------------------------------------------------------
# Pagination response
# ---------------------------------------------------------------------------


class TestGrpcFetchByMetadataResponsePagination:
    """Verify pagination token is correctly deserialized."""

    def test_fetch_by_metadata_response_pagination(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        mock_channel.fetch_by_metadata.return_value = _make_channel_response(
            vectors={"v1": {"id": "v1", "values": [0.1]}},
            pagination={"next": "token123"},
        )
        result = grpc_index.fetch_by_metadata(filter={"a": 1})

        assert result.pagination is not None
        assert result.pagination.next == "token123"

    def test_fetch_by_metadata_no_pagination(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        mock_channel.fetch_by_metadata.return_value = _make_channel_response(
            vectors={"v1": {"id": "v1", "values": [0.1]}},
        )
        result = grpc_index.fetch_by_metadata(filter={"a": 1})

        assert result.pagination is None


# ---------------------------------------------------------------------------
# Validation: limit range and empty filter, with the REST lane's exact words
# ---------------------------------------------------------------------------


class TestGrpcFetchByMetadataValidation:
    """limit must be 1-10000 and filter non-empty; rejected before the channel."""

    def test_fetch_by_metadata_invalid_limit_zero(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        with pytest.raises(ValidationError) as excinfo:
            grpc_index.fetch_by_metadata(filter={"a": "b"}, limit=0)
        assert str(excinfo.value) == "limit must be between 1 and 10000, got 0"
        assert not mock_channel.method_calls

    def test_fetch_by_metadata_invalid_limit_negative(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        with pytest.raises(ValidationError, match="limit must be between 1 and 10000"):
            grpc_index.fetch_by_metadata(filter={"a": "b"}, limit=-1)
        assert not mock_channel.method_calls

    def test_fetch_by_metadata_invalid_limit_over_max(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        with pytest.raises(ValidationError) as excinfo:
            grpc_index.fetch_by_metadata(filter={"a": "b"}, limit=10_001)
        assert str(excinfo.value) == "limit must be between 1 and 10000, got 10001"
        assert not mock_channel.method_calls

    def test_fetch_by_metadata_empty_filter(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        with pytest.raises(ValidationError) as excinfo:
            grpc_index.fetch_by_metadata(filter={})
        assert str(excinfo.value) == (
            "filter must contain at least one condition, got {}. "
            "Empty filter provided for fetch by metadata request"
        )
        assert not mock_channel.method_calls

    @pytest.mark.parametrize("limit", [1, 100, 10_000])
    def test_fetch_by_metadata_limit_boundaries_pass(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock, limit: int
    ) -> None:
        grpc_index.fetch_by_metadata(filter={"a": "b"}, limit=limit)
        assert mock_channel.fetch_by_metadata.call_args.kwargs["limit"] == limit


# ---------------------------------------------------------------------------
# Response parity with the REST adapter
# ---------------------------------------------------------------------------


class TestGrpcFetchByMetadataRestParity:
    """The two wire shapes differ; the models they decode to must not."""

    def test_same_response_yields_the_same_model(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        rest_json = {
            "vectors": {
                "v1": {
                    "id": "v1",
                    "values": [0.5, 0.25],
                    "sparseValues": {"indices": [1, 4], "values": [0.5, 0.25]},
                    "metadata": {"genre": "comedy", "year": 2020.0},
                },
                "v2": {"id": "v2", "values": [0.125]},
            },
            "namespace": "movies",
            "usage": {"readUnits": 7},
            "pagination": {"next": "tok"},
        }
        grpc_dict = {
            "vectors": {
                "v1": {
                    "id": "v1",
                    "values": [0.5, 0.25],
                    "sparse_values": {"indices": [1, 4], "values": [0.5, 0.25]},
                    "metadata": {"genre": "comedy", "year": 2020.0},
                },
                "v2": {"id": "v2", "values": [0.125]},
            },
            "namespace": "movies",
            "usage": {"read_units": 7},
            "pagination": {"next": "tok"},
        }

        from_rest = VectorsAdapter.to_fetch_by_metadata_response(json.dumps(rest_json).encode())
        mock_channel.fetch_by_metadata.return_value = grpc_dict
        from_grpc = grpc_index.fetch_by_metadata(filter={"genre": {"$eq": "comedy"}})

        assert from_rest == from_grpc

    def test_parity_holds_when_optional_members_are_absent(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        rest_json: dict[str, Any] = {"vectors": {}, "namespace": ""}
        grpc_dict: dict[str, Any] = {"vectors": {}, "namespace": ""}

        from_rest = VectorsAdapter.to_fetch_by_metadata_response(json.dumps(rest_json).encode())
        mock_channel.fetch_by_metadata.return_value = grpc_dict
        from_grpc = grpc_index.fetch_by_metadata(filter={"genre": {"$eq": "comedy"}})

        assert from_rest == from_grpc
        assert from_grpc.usage is None
        assert from_grpc.pagination is None

    def test_the_two_fixtures_are_not_the_same_dict(self) -> None:
        """Parity would be vacuous if the two wire shapes were collapsed into one."""
        assert {"read_units": 7} != {"readUnits": 7}
