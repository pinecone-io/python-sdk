"""Unit tests for Index.delete() method."""

from __future__ import annotations

import httpx
import pytest
import respx

from pinecone import Index
from pinecone.errors.exceptions import PineconeValueError, ValidationError

INDEX_HOST = "test-index-abc1234.svc.us-east1-gcp.pinecone.io"
INDEX_HOST_HTTPS = f"https://{INDEX_HOST}"
DELETE_URL = f"{INDEX_HOST_HTTPS}/vectors/delete"


def _make_index() -> Index:
    return Index(host=INDEX_HOST, api_key="test-key")


# ---------------------------------------------------------------------------
# Delete by IDs
# ---------------------------------------------------------------------------


class TestDeleteByIds:
    """Delete vectors by ID list (unified-vec-0013)."""

    @respx.mock
    def test_sends_correct_body_and_returns_none(self) -> None:
        route = respx.post(DELETE_URL).mock(
            return_value=httpx.Response(200, json={}),
        )
        idx = _make_index()
        result = idx.delete(ids=["vec1", "vec2"])

        assert result is None

        import orjson

        body = orjson.loads(route.calls.last.request.content)
        assert body["ids"] == ["vec1", "vec2"]
        assert body["namespace"] == ""
        assert "deleteAll" not in body
        assert "filter" not in body


# ---------------------------------------------------------------------------
# Delete all
# ---------------------------------------------------------------------------


class TestDeleteAll:
    """Delete all vectors in a namespace (unified-vec-0013)."""

    @respx.mock
    def test_sends_delete_all_true(self) -> None:
        route = respx.post(DELETE_URL).mock(
            return_value=httpx.Response(200, json={}),
        )
        idx = _make_index()
        result = idx.delete(delete_all=True)

        assert result is None

        import orjson

        body = orjson.loads(route.calls.last.request.content)
        assert body["deleteAll"] is True
        assert body["namespace"] == ""
        assert "ids" not in body
        assert "filter" not in body


# ---------------------------------------------------------------------------
# Delete by filter
# ---------------------------------------------------------------------------


class TestDeleteByFilter:
    """Delete vectors matching a metadata filter (unified-vec-0013)."""

    @respx.mock
    def test_sends_filter_in_body(self) -> None:
        route = respx.post(DELETE_URL).mock(
            return_value=httpx.Response(200, json={}),
        )
        idx = _make_index()
        my_filter = {"genre": {"$eq": "action"}}
        result = idx.delete(filter=my_filter)

        assert result is None

        import orjson

        body = orjson.loads(route.calls.last.request.content)
        assert body["filter"] == {"genre": {"$eq": "action"}}
        assert body["namespace"] == ""
        assert "ids" not in body
        assert "deleteAll" not in body


# ---------------------------------------------------------------------------
# Namespace handling
# ---------------------------------------------------------------------------


class TestDeleteNamespace:
    """Namespace targeting (unified-vec-0022)."""

    @respx.mock
    def test_default_namespace_is_empty_string(self) -> None:
        route = respx.post(DELETE_URL).mock(
            return_value=httpx.Response(200, json={}),
        )
        idx = _make_index()
        idx.delete(ids=["vec1"])

        import orjson

        body = orjson.loads(route.calls.last.request.content)
        assert body["namespace"] == ""

    @respx.mock
    def test_explicit_namespace_in_body(self) -> None:
        route = respx.post(DELETE_URL).mock(
            return_value=httpx.Response(200, json={}),
        )
        idx = _make_index()
        idx.delete(ids=["vec1"], namespace="prod")

        import orjson

        body = orjson.loads(route.calls.last.request.content)
        assert body["namespace"] == "prod"


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestDeleteValidation:
    """Input validation (unified-vec-0041)."""

    def test_no_mode_raises(self) -> None:
        """Must specify at least one of ids, delete_all, filter."""
        idx = _make_index()
        with pytest.raises(ValidationError, match="Must specify one of"):
            idx.delete()

    def test_ids_and_delete_all_raises(self) -> None:
        idx = _make_index()
        with pytest.raises(ValidationError, match="Cannot combine"):
            idx.delete(ids=["vec1"], delete_all=True)

    def test_ids_and_filter_raises(self) -> None:
        idx = _make_index()
        with pytest.raises(ValidationError, match="Cannot combine"):
            idx.delete(ids=["vec1"], filter={"key": {"$eq": "val"}})

    def test_delete_all_and_filter_raises(self) -> None:
        idx = _make_index()
        with pytest.raises(ValidationError, match="Cannot combine"):
            idx.delete(delete_all=True, filter={"key": {"$eq": "val"}})

    def test_all_three_raises(self) -> None:
        idx = _make_index()
        with pytest.raises(ValidationError, match="Cannot combine"):
            idx.delete(ids=["vec1"], delete_all=True, filter={"k": {"$eq": "v"}})

    def test_positional_args_rejected(self) -> None:
        idx = _make_index()
        with pytest.raises(PineconeValueError, match="keyword-only"):
            idx.delete(["vec1"])  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Non-existent IDs (unified-vec-0032)
# ---------------------------------------------------------------------------


class TestDeleteNonExistentIds:
    """Deleting IDs that don't exist should not raise."""

    @respx.mock
    def test_non_existent_ids_succeed(self) -> None:
        """API returns 200 with empty body for non-existent IDs."""
        respx.post(DELETE_URL).mock(
            return_value=httpx.Response(200, json={}),
        )
        idx = _make_index()
        result = idx.delete(ids=["does-not-exist-1", "does-not-exist-2"])
        assert result is None
