"""Unit tests for Index.fetch() method."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
import respx

from pinecone import Index
from pinecone.errors.exceptions import PineconeValueError, ValidationError
from pinecone.models.vectors.responses import FetchResponse

INDEX_HOST = "test-index-abc1234.svc.us-east1-gcp.pinecone.io"
INDEX_HOST_HTTPS = f"https://{INDEX_HOST}"
FETCH_URL = f"{INDEX_HOST_HTTPS}/vectors/fetch"


def _make_fetch_response(
    *,
    vectors: dict[str, dict[str, Any]] | None = None,
    namespace: str = "",
    usage: dict[str, int] | None = None,
) -> dict[str, object]:
    """Build a realistic fetch API response payload."""
    return {
        "vectors": vectors or {},
        "namespace": namespace,
        "usage": usage or {"readUnits": 5},
    }


def _make_index() -> Index:
    return Index(host=INDEX_HOST, api_key="test-key")


# ---------------------------------------------------------------------------
# Successful fetches
# ---------------------------------------------------------------------------


class TestFetchSuccess:
    """Fetch returns FetchResponse with vectors map."""

    @respx.mock
    def test_fetch_multiple_ids(self) -> None:
        respx.get(FETCH_URL).mock(
            return_value=httpx.Response(
                200,
                json=_make_fetch_response(
                    vectors={
                        "vec1": {"id": "vec1", "values": [0.1, 0.2, 0.3]},
                        "vec2": {"id": "vec2", "values": [0.4, 0.5, 0.6]},
                    },
                ),
            ),
        )
        idx = _make_index()
        result = idx.fetch(ids=["vec1", "vec2"])

        assert isinstance(result, FetchResponse)
        assert len(result.vectors) == 2
        assert result.vectors["vec1"].id == "vec1"
        assert result.vectors["vec1"].values == pytest.approx([0.1, 0.2, 0.3])
        assert result.vectors["vec2"].id == "vec2"

    @respx.mock
    def test_request_params_correct(self) -> None:
        """Verify ids are sent as repeated query params."""
        route = respx.get(FETCH_URL).mock(
            return_value=httpx.Response(200, json=_make_fetch_response()),
        )
        idx = _make_index()
        idx.fetch(ids=["vec1", "vec2"])

        request = route.calls.last.request
        # httpx sends repeated params as ?ids=vec1&ids=vec2
        assert request.url.params.multi_items() == [
            ("ids", "vec1"),
            ("ids", "vec2"),
        ] or set(request.url.params.get_list("ids")) == {"vec1", "vec2"}

    @respx.mock
    def test_response_includes_usage(self) -> None:
        respx.get(FETCH_URL).mock(
            return_value=httpx.Response(
                200,
                json=_make_fetch_response(usage={"readUnits": 10}),
            ),
        )
        idx = _make_index()
        result = idx.fetch(ids=["vec1"])

        assert result.usage is not None
        assert result.usage.read_units == 10


# ---------------------------------------------------------------------------
# Namespace handling
# ---------------------------------------------------------------------------


class TestFetchNamespace:
    """Namespace targeting (unified-vec-0022)."""

    @respx.mock
    def test_default_namespace_omitted_from_params(self) -> None:
        """Empty namespace string means default — not sent in params."""
        route = respx.get(FETCH_URL).mock(
            return_value=httpx.Response(200, json=_make_fetch_response()),
        )
        idx = _make_index()
        idx.fetch(ids=["vec1"])

        request = route.calls.last.request
        param_keys = [k for k, _ in request.url.params.multi_items()]
        assert "namespace" not in param_keys

    @respx.mock
    def test_explicit_namespace_in_params(self) -> None:
        route = respx.get(FETCH_URL).mock(
            return_value=httpx.Response(200, json=_make_fetch_response(namespace="prod")),
        )
        idx = _make_index()
        idx.fetch(ids=["vec1"], namespace="prod")

        request = route.calls.last.request
        assert request.url.params["namespace"] == "prod"

    @respx.mock
    def test_response_includes_namespace(self) -> None:
        respx.get(FETCH_URL).mock(
            return_value=httpx.Response(200, json=_make_fetch_response(namespace="test-ns")),
        )
        idx = _make_index()
        result = idx.fetch(ids=["vec1"], namespace="test-ns")
        assert result.namespace == "test-ns"


# ---------------------------------------------------------------------------
# Non-existent IDs (unified-vec-0053)
# ---------------------------------------------------------------------------


class TestFetchNonExistentIds:
    """Fetching IDs that do not exist returns empty vectors map."""

    @respx.mock
    def test_nonexistent_ids_return_empty_map(self) -> None:
        respx.get(FETCH_URL).mock(
            return_value=httpx.Response(
                200,
                json=_make_fetch_response(vectors={}, namespace=""),
            ),
        )
        idx = _make_index()
        result = idx.fetch(ids=["does-not-exist"])

        assert isinstance(result, FetchResponse)
        assert result.vectors == {}
        assert result.namespace == ""


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestFetchValidation:
    """Input validation."""

    def test_empty_ids_raises(self) -> None:
        idx = _make_index()
        with pytest.raises(ValidationError, match="ids must be a non-empty list"):
            idx.fetch(ids=[])

    def test_positional_args_rejected(self) -> None:
        """All params must be keyword-only."""
        idx = _make_index()
        with pytest.raises(PineconeValueError, match="keyword-only"):
            idx.fetch(["vec1"])  # type: ignore[misc]
