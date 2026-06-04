"""Unit tests for Index.update() method."""

from __future__ import annotations

from typing import Any

import httpx
import orjson
import pytest
import respx

from pinecone import Index
from pinecone.errors.exceptions import ValidationError
from pinecone.models.vectors.responses import UpdateResponse

INDEX_HOST = "test-index-abc1234.svc.us-east1-gcp.pinecone.io"
INDEX_HOST_HTTPS = f"https://{INDEX_HOST}"
UPDATE_URL = f"{INDEX_HOST_HTTPS}/vectors/update"


def _make_update_response(*, matched_records: int | None = None) -> dict[str, Any]:
    """Build a realistic update API response payload."""
    return {"matchedRecords": matched_records}


def _make_index() -> Index:
    return Index(host=INDEX_HOST, api_key="test-key")


# ---------------------------------------------------------------------------
# Update by ID
# ---------------------------------------------------------------------------


class TestUpdateById:
    """Update a single vector by identifier."""

    @respx.mock
    def test_update_values_by_id(self) -> None:
        route = respx.post(UPDATE_URL).mock(
            return_value=httpx.Response(200, json=_make_update_response()),
        )
        idx = _make_index()
        result = idx.update(id="vec1", values=[0.1, 0.2])

        assert isinstance(result, UpdateResponse)
        body = orjson.loads(route.calls.last.request.content)
        assert body["id"] == "vec1"
        assert body["values"] == [0.1, 0.2]
        assert body["namespace"] == ""

    @respx.mock
    def test_update_metadata_by_id(self) -> None:
        route = respx.post(UPDATE_URL).mock(
            return_value=httpx.Response(200, json=_make_update_response()),
        )
        idx = _make_index()
        idx.update(id="vec1", set_metadata={"genre": "comedy"})

        body = orjson.loads(route.calls.last.request.content)
        assert body["setMetadata"] == {"genre": "comedy"}

    @respx.mock
    def test_update_sparse_values_by_id(self) -> None:
        route = respx.post(UPDATE_URL).mock(
            return_value=httpx.Response(200, json=_make_update_response()),
        )
        idx = _make_index()
        sparse = {"indices": [0, 3], "values": [0.1, 0.2]}
        idx.update(id="vec1", sparse_values=sparse)

        body = orjson.loads(route.calls.last.request.content)
        assert body["sparseValues"] == {"indices": [0, 3], "values": [0.1, 0.2]}

    @respx.mock
    def test_update_sparse_values_struct_by_id(self) -> None:
        from pinecone.models.vectors.sparse import SparseValues

        route = respx.post(UPDATE_URL).mock(
            return_value=httpx.Response(200, json=_make_update_response()),
        )
        idx = _make_index()
        sparse = SparseValues(indices=[0, 1], values=[0.5, 0.5])
        idx.update(id="vec1", sparse_values=sparse)

        body = orjson.loads(route.calls.last.request.content)
        assert body["sparseValues"] == {"indices": [0, 1], "values": [0.5, 0.5]}

    @respx.mock
    def test_update_with_namespace(self) -> None:
        route = respx.post(UPDATE_URL).mock(
            return_value=httpx.Response(200, json=_make_update_response()),
        )
        idx = _make_index()
        idx.update(id="vec1", values=[0.1], namespace="my-ns")

        body = orjson.loads(route.calls.last.request.content)
        assert body["namespace"] == "my-ns"

    @respx.mock
    def test_update_default_namespace_is_empty_string(self) -> None:
        route = respx.post(UPDATE_URL).mock(
            return_value=httpx.Response(200, json=_make_update_response()),
        )
        idx = _make_index()
        idx.update(id="vec1", values=[0.1])

        body = orjson.loads(route.calls.last.request.content)
        assert body["namespace"] == ""


# ---------------------------------------------------------------------------
# Update by filter
# ---------------------------------------------------------------------------


class TestUpdateByFilter:
    """Bulk-update metadata via metadata filter."""

    @respx.mock
    def test_update_metadata_by_filter(self) -> None:
        route = respx.post(UPDATE_URL).mock(
            return_value=httpx.Response(200, json=_make_update_response()),
        )
        idx = _make_index()
        idx.update(
            filter={"genre": {"$eq": "drama"}},
            set_metadata={"year": 2020},
        )

        body = orjson.loads(route.calls.last.request.content)
        assert body["filter"] == {"genre": {"$eq": "drama"}}
        assert body["setMetadata"] == {"year": 2020}
        assert "id" not in body

    @respx.mock
    def test_update_dry_run(self) -> None:
        route = respx.post(UPDATE_URL).mock(
            return_value=httpx.Response(200, json=_make_update_response(matched_records=42)),
        )
        idx = _make_index()
        result = idx.update(
            filter={"genre": {"$eq": "drama"}},
            set_metadata={"year": 2020},
            dry_run=True,
        )

        body = orjson.loads(route.calls.last.request.content)
        assert body["dryRun"] is True
        assert result.matched_records == 42


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestUpdateValidation:
    """Input validation for update()."""

    def test_both_id_and_filter_raises(self) -> None:
        idx = _make_index()
        with pytest.raises(ValidationError, match="not both"):
            idx.update(id="vec1", filter={"a": 1}, values=[0.1])

    def test_neither_id_nor_filter_raises(self) -> None:
        idx = _make_index()
        with pytest.raises(ValidationError, match="got neither"):
            idx.update(values=[0.1])

    def test_dry_run_with_id_raises(self) -> None:
        idx = _make_index()
        with pytest.raises(ValidationError, match="dry_run is only supported"):
            idx.update(id="vec1", set_metadata={"genre": "comedy"}, dry_run=True)

    @respx.mock
    def test_dry_run_not_in_body_when_false(self) -> None:
        route = respx.post(UPDATE_URL).mock(
            return_value=httpx.Response(200, json=_make_update_response()),
        )
        idx = _make_index()
        idx.update(id="vec1", values=[0.1])

        body = orjson.loads(route.calls.last.request.content)
        assert "dryRun" not in body
