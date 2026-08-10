"""Unit tests for Index.fetch_by_metadata() method."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
import respx

from pinecone import Index
from pinecone.errors.exceptions import PineconeValueError
from pinecone.models.vectors.responses import FetchByMetadataResponse

INDEX_HOST = "test-index-abc1234.svc.us-east1-gcp.pinecone.io"
INDEX_HOST_HTTPS = f"https://{INDEX_HOST}"
FETCH_BY_META_URL = f"{INDEX_HOST_HTTPS}/vectors/fetch_by_metadata"


def _make_response(
    *,
    vectors: dict[str, dict[str, Any]] | None = None,
    namespace: str = "",
    usage: dict[str, int] | None = None,
    pagination: dict[str, str] | None = None,
) -> dict[str, object]:
    """Build a realistic fetch-by-metadata API response payload."""
    resp: dict[str, object] = {
        "vectors": vectors or {},
        "namespace": namespace,
        "usage": usage or {"readUnits": 5},
    }
    if pagination is not None:
        resp["pagination"] = pagination
    return resp


def _make_index() -> Index:
    return Index(host=INDEX_HOST, api_key="test-key")


# ---------------------------------------------------------------------------
# Basic success
# ---------------------------------------------------------------------------


class TestFetchByMetadataBasic:
    """fetch_by_metadata returns FetchByMetadataResponse with vectors."""

    @respx.mock
    def test_fetch_by_metadata_basic(self) -> None:
        respx.post(FETCH_BY_META_URL).mock(
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
        idx = _make_index()
        result = idx.fetch_by_metadata(filter={"genre": "comedy"})

        assert isinstance(result, FetchByMetadataResponse)
        assert len(result.vectors) == 2
        assert result.vectors["vec1"].id == "vec1"
        assert result.vectors["vec2"].id == "vec2"


# ---------------------------------------------------------------------------
# Request body construction
# ---------------------------------------------------------------------------


class TestFetchByMetadataRequestBody:
    """Verify the POST body is built correctly from parameters."""

    @respx.mock
    def test_fetch_by_metadata_sends_filter(self) -> None:
        route = respx.post(FETCH_BY_META_URL).mock(
            return_value=httpx.Response(200, json=_make_response()),
        )
        idx = _make_index()
        idx.fetch_by_metadata(filter={"genre": {"$eq": "comedy"}})

        request = route.calls.last.request
        import json

        body = json.loads(request.content)
        assert body["filter"] == {"genre": {"$eq": "comedy"}}

    @respx.mock
    def test_fetch_by_metadata_sends_namespace(self) -> None:
        route = respx.post(FETCH_BY_META_URL).mock(
            return_value=httpx.Response(200, json=_make_response()),
        )
        idx = _make_index()
        idx.fetch_by_metadata(filter={"a": 1}, namespace="my-ns")

        request = route.calls.last.request
        import json

        body = json.loads(request.content)
        assert body["namespace"] == "my-ns"

    @respx.mock
    def test_fetch_by_metadata_sends_limit(self) -> None:
        route = respx.post(FETCH_BY_META_URL).mock(
            return_value=httpx.Response(200, json=_make_response()),
        )
        idx = _make_index()
        idx.fetch_by_metadata(filter={"a": 1}, limit=50)

        request = route.calls.last.request
        import json

        body = json.loads(request.content)
        assert body["limit"] == 50

    @respx.mock
    def test_fetch_by_metadata_sends_pagination_token(self) -> None:
        route = respx.post(FETCH_BY_META_URL).mock(
            return_value=httpx.Response(200, json=_make_response()),
        )
        idx = _make_index()
        idx.fetch_by_metadata(filter={"a": 1}, pagination_token="abc")

        request = route.calls.last.request
        import json

        body = json.loads(request.content)
        assert body["paginationToken"] == "abc"

    @respx.mock
    def test_fetch_by_metadata_omits_optional_fields(self) -> None:
        route = respx.post(FETCH_BY_META_URL).mock(
            return_value=httpx.Response(200, json=_make_response()),
        )
        idx = _make_index()
        idx.fetch_by_metadata(filter={"a": 1})

        request = route.calls.last.request
        import json

        body = json.loads(request.content)
        assert "namespace" not in body
        assert "limit" not in body
        assert "paginationToken" not in body


# ---------------------------------------------------------------------------
# Pagination response
# ---------------------------------------------------------------------------


class TestFetchByMetadataResponsePagination:
    """Verify pagination token is correctly deserialized."""

    @respx.mock
    def test_fetch_by_metadata_response_pagination(self) -> None:
        respx.post(FETCH_BY_META_URL).mock(
            return_value=httpx.Response(
                200,
                json=_make_response(
                    vectors={"v1": {"id": "v1", "values": [0.1]}},
                    pagination={"next": "token123"},
                ),
            ),
        )
        idx = _make_index()
        result = idx.fetch_by_metadata(filter={"a": 1})

        assert result.pagination is not None
        assert result.pagination.next == "token123"

    @respx.mock
    def test_fetch_by_metadata_no_pagination(self) -> None:
        respx.post(FETCH_BY_META_URL).mock(
            return_value=httpx.Response(
                200,
                json=_make_response(vectors={"v1": {"id": "v1", "values": [0.1]}}),
            ),
        )
        idx = _make_index()
        result = idx.fetch_by_metadata(filter={"a": 1})

        assert result.pagination is None

    def test_positional_args_rejected(self) -> None:
        """All params must be keyword-only."""
        idx = _make_index()
        with pytest.raises(PineconeValueError, match="keyword-only"):
            idx.fetch_by_metadata({"a": 1})  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Limit validation
# ---------------------------------------------------------------------------


class TestFetchByMetadataLimitValidation:
    """limit must be >= 1; limit=0 or negative raises before any HTTP call."""

    def test_fetch_by_metadata_limit_validation_zero(self) -> None:
        idx = _make_index()
        with pytest.raises(Exception, match="limit"):
            idx.fetch_by_metadata(filter={"a": "b"}, limit=0)

    def test_fetch_by_metadata_limit_validation_negative(self) -> None:
        idx = _make_index()
        with pytest.raises(Exception, match="limit"):
            idx.fetch_by_metadata(filter={"a": "b"}, limit=-1)

    @respx.mock
    def test_fetch_by_metadata_limit_validation_one_passes(self) -> None:
        respx.post(FETCH_BY_META_URL).mock(
            return_value=httpx.Response(200, json=_make_response()),
        )
        idx = _make_index()
        idx.fetch_by_metadata(filter={"a": "b"}, limit=1)
