"""Unit tests for GrpcIndex bulk import methods."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest
import respx

from pinecone.errors.exceptions import ValidationError
from pinecone.grpc import GrpcIndex
from pinecone.models.imports.list import ImportList
from pinecone.models.imports.model import ImportModel, StartImportResponse

_MOCK_GRPC_MODULE_PATH = "pinecone._grpc"
_INDEX_HOST = "test-index-abc123.svc.pinecone.io"
_INDEX_HOST_HTTPS = f"https://{_INDEX_HOST}"
_IMPORTS_URL = f"{_INDEX_HOST_HTTPS}/bulk/imports"


def _make_grpc_index(mock_channel: MagicMock) -> GrpcIndex:
    mock_module = MagicMock()
    mock_module.GrpcChannel.return_value = mock_channel
    with patch.dict("sys.modules", {_MOCK_GRPC_MODULE_PATH: mock_module}):
        return GrpcIndex(host=_INDEX_HOST, api_key="test-api-key")


def _import_payload(
    *,
    id: str = "import-123",
    uri: str = "s3://my-bucket/data/",
    status: str = "Pending",
    created_at: str = "2025-01-01T00:00:00Z",
) -> dict[str, Any]:
    return {"id": id, "uri": uri, "status": status, "createdAt": created_at}


def _list_payload(
    imports: list[dict[str, Any]],
    *,
    pagination_token: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {"data": imports}
    if pagination_token is not None:
        result["pagination"] = {"next": pagination_token}
    return result


@pytest.fixture
def mock_channel() -> MagicMock:
    return MagicMock()


@pytest.fixture
def grpc_index(mock_channel: MagicMock) -> GrpcIndex:
    return _make_grpc_index(mock_channel)


# ---------------------------------------------------------------------------
# start_import
# ---------------------------------------------------------------------------


class TestStartImport:
    @respx.mock
    def test_start_import(self, mock_channel: MagicMock) -> None:
        respx.post(_IMPORTS_URL).mock(
            return_value=httpx.Response(200, json={"id": "import-001"}),
        )
        idx = _make_grpc_index(mock_channel)
        result = idx.start_import(uri="s3://bucket/data/")

        assert isinstance(result, StartImportResponse)
        assert result.id == "import-001"

    @respx.mock
    def test_start_import_omits_error_mode_by_default(self, mock_channel: MagicMock) -> None:
        route = respx.post(_IMPORTS_URL).mock(
            return_value=httpx.Response(200, json={"id": "imp-x"}),
        )
        idx = _make_grpc_index(mock_channel)
        idx.start_import(uri="s3://bucket/data/")

        body = json.loads(route.calls.last.request.content)
        assert "errorMode" not in body

    @respx.mock
    def test_start_import_error_mode_abort(self, mock_channel: MagicMock) -> None:
        route = respx.post(_IMPORTS_URL).mock(
            return_value=httpx.Response(200, json={"id": "imp-abort"}),
        )
        idx = _make_grpc_index(mock_channel)
        idx.start_import(uri="s3://bucket/data/", error_mode="abort")

        body = json.loads(route.calls.last.request.content)
        assert body["errorMode"] == {"onError": "abort"}

    @respx.mock
    def test_start_import_error_mode_case_insensitive(self, mock_channel: MagicMock) -> None:
        route = respx.post(_IMPORTS_URL).mock(
            return_value=httpx.Response(200, json={"id": "imp-ci"}),
        )
        idx = _make_grpc_index(mock_channel)
        idx.start_import(uri="s3://bucket/data/", error_mode="ABORT")

        body = json.loads(route.calls.last.request.content)
        assert body["errorMode"] == {"onError": "abort"}

    def test_start_import_invalid_error_mode(self, mock_channel: MagicMock) -> None:
        idx = _make_grpc_index(mock_channel)
        with pytest.raises(ValidationError, match="error_mode"):
            idx.start_import(uri="s3://bucket/data/", error_mode="invalid")

    @respx.mock
    def test_start_import_with_integration_id(self, mock_channel: MagicMock) -> None:
        route = respx.post(_IMPORTS_URL).mock(
            return_value=httpx.Response(200, json={"id": "imp-int"}),
        )
        idx = _make_grpc_index(mock_channel)
        idx.start_import(uri="s3://bucket/data/", integration_id="int-456")

        body = json.loads(route.calls.last.request.content)
        assert body["integrationId"] == "int-456"


# ---------------------------------------------------------------------------
# describe_import
# ---------------------------------------------------------------------------


class TestDescribeImport:
    @respx.mock
    def test_describe_import(self, mock_channel: MagicMock) -> None:
        respx.get(f"{_IMPORTS_URL}/101").mock(
            return_value=httpx.Response(200, json=_import_payload(id="101", status="InProgress")),
        )
        idx = _make_grpc_index(mock_channel)
        result = idx.describe_import("101")

        assert isinstance(result, ImportModel)
        assert result.id == "101"
        assert result.status == "InProgress"

    @respx.mock
    def test_describe_import_int_id(self, mock_channel: MagicMock) -> None:
        route = respx.get(f"{_IMPORTS_URL}/42").mock(
            return_value=httpx.Response(200, json=_import_payload(id="42")),
        )
        idx = _make_grpc_index(mock_channel)
        result = idx.describe_import(42)

        assert isinstance(result, ImportModel)
        assert str(route.calls.last.request.url).endswith("/bulk/imports/42")

    def test_describe_import_empty_id_raises(self, mock_channel: MagicMock) -> None:
        idx = _make_grpc_index(mock_channel)
        with pytest.raises(ValidationError, match="import id"):
            idx.describe_import("")


# ---------------------------------------------------------------------------
# cancel_import
# ---------------------------------------------------------------------------


class TestCancelImport:
    @respx.mock
    def test_cancel_import(self, mock_channel: MagicMock) -> None:
        route = respx.delete(f"{_IMPORTS_URL}/101").mock(
            return_value=httpx.Response(202),
        )
        idx = _make_grpc_index(mock_channel)
        result = idx.cancel_import("101")

        assert result is None
        assert str(route.calls.last.request.url).endswith("/bulk/imports/101")

    def test_cancel_import_empty_id_raises(self, mock_channel: MagicMock) -> None:
        idx = _make_grpc_index(mock_channel)
        with pytest.raises(ValidationError, match="import id"):
            idx.cancel_import("")


# ---------------------------------------------------------------------------
# list_imports (auto-paginating)
# ---------------------------------------------------------------------------


class TestListImports:
    @respx.mock
    def test_list_imports_follows_pagination(self, mock_channel: MagicMock) -> None:
        page1 = [_import_payload(id="imp-1"), _import_payload(id="imp-2")]
        page2 = [_import_payload(id="imp-3")]
        respx.get(_IMPORTS_URL).mock(
            side_effect=[
                httpx.Response(200, json=_list_payload(page1, pagination_token="tok-abc")),
                httpx.Response(200, json=_list_payload(page2)),
            ],
        )
        idx = _make_grpc_index(mock_channel)
        results = list(idx.list_imports())

        assert len(results) == 3
        assert all(isinstance(r, ImportModel) for r in results)
        assert [r.id for r in results] == ["imp-1", "imp-2", "imp-3"]

    @respx.mock
    def test_list_imports_single_page(self, mock_channel: MagicMock) -> None:
        imports = [_import_payload(id="a"), _import_payload(id="b")]
        respx.get(_IMPORTS_URL).mock(
            return_value=httpx.Response(200, json=_list_payload(imports)),
        )
        idx = _make_grpc_index(mock_channel)
        results = list(idx.list_imports())

        assert len(results) == 2
        assert results[0].id == "a"
        assert results[1].id == "b"

    @respx.mock
    def test_list_imports_empty(self, mock_channel: MagicMock) -> None:
        respx.get(_IMPORTS_URL).mock(
            return_value=httpx.Response(200, json=_list_payload([])),
        )
        idx = _make_grpc_index(mock_channel)
        assert list(idx.list_imports()) == []

    @respx.mock
    def test_list_imports_passes_limit(self, mock_channel: MagicMock) -> None:
        route = respx.get(_IMPORTS_URL).mock(
            return_value=httpx.Response(200, json=_list_payload([])),
        )
        idx = _make_grpc_index(mock_channel)
        list(idx.list_imports(limit=10))

        assert route.calls.last.request.url.params["limit"] == "10"


# ---------------------------------------------------------------------------
# list_imports_paginated (single page)
# ---------------------------------------------------------------------------


class TestListImportsPaginated:
    @respx.mock
    def test_list_imports_paginated_returns_import_list(self, mock_channel: MagicMock) -> None:
        imports = [_import_payload(id="p1"), _import_payload(id="p2")]
        respx.get(_IMPORTS_URL).mock(
            return_value=httpx.Response(200, json=_list_payload(imports)),
        )
        idx = _make_grpc_index(mock_channel)
        result = idx.list_imports_paginated()

        assert isinstance(result, ImportList)
        assert len(result) == 2
        assert result[0].id == "p1"

    @respx.mock
    def test_list_imports_paginated_with_params(self, mock_channel: MagicMock) -> None:
        route = respx.get(_IMPORTS_URL).mock(
            return_value=httpx.Response(200, json=_list_payload([])),
        )
        idx = _make_grpc_index(mock_channel)
        idx.list_imports_paginated(limit=5, pagination_token="tok-xyz")

        params = route.calls.last.request.url.params
        assert params["limit"] == "5"
        assert params["paginationToken"] == "tok-xyz"
