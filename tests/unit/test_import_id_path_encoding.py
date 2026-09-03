"""The bulk-import routes carry the import id in the URL path, so it must be encoded.

``describe_import`` and ``cancel_import`` address ``/bulk/imports/{id}``. The
2026-07 spec constrains the ``id`` path parameter to
``type: string, minLength: 1, maxLength: 1000`` with **no** pattern
(``db_data_2026-07.oas.yaml:163-171`` @ apis ``5f808858``), so every character
is legal — including ``/``, ``?``, ``#``, ``%`` and the C0 controls, none of
which survive raw interpolation into a path:

- ``/`` silently addresses a different route (``a/b`` requests
  ``/bulk/imports/a/b``)
- ``?`` and ``#`` truncate the path
- a control character makes httpx reject the URL outright

So the segment is percent-encoded, exactly as the namespace routes (#119/#120)
and the records routes (#233) already do. These tests pin the encoded request
line for both bulk-import operations on all three lanes that build the path:
REST sync, REST async, and the gRPC wrapper (whose bulk-import methods go over
HTTP, not gRPC). Reverting the encoding fails every case but ``space``, which
httpx normalizes on its own.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest
import respx

from pinecone import AsyncIndex, Index
from pinecone.grpc import GrpcIndex

INDEX_HOST = "import-id-path-encoding-abc123.svc.pinecone.io"
BASE_URL = f"https://{INDEX_HOST}"
IMPORTS_URL = f"{BASE_URL}/bulk/imports"

_MOCK_GRPC_MODULE_PATH = "pinecone._grpc"

IMPORT_PAYLOAD: dict[str, Any] = {
    "id": "101",
    "uri": "s3://my-bucket/data/",
    "status": "InProgress",
    "createdAt": "2025-01-01T00:00:00Z",
}

ENCODING_CASES = [
    pytest.param("a/b", "a%2Fb", id="slash_cannot_change_the_route"),
    pytest.param("a b", "a%20b", id="space"),
    pytest.param("a?b#c", "a%3Fb%23c", id="query_and_fragment_delimiters"),
    pytest.param("100%", "100%25", id="percent"),
    pytest.param("\x01", "%01", id="control_character"),
    pytest.param("\x7f", "%7F", id="delete_character"),
]


@pytest.fixture
def index() -> Index:
    return Index(host=INDEX_HOST, api_key="test-key")


@pytest.fixture
async def async_index() -> AsyncIterator[AsyncIndex]:
    client = AsyncIndex(host=INDEX_HOST, api_key="test-key")
    yield client
    await client.close()


@pytest.fixture
def grpc_index() -> Iterator[GrpcIndex]:
    mock_module = MagicMock()
    mock_module.GrpcChannel.return_value = MagicMock()
    with patch.dict("sys.modules", {_MOCK_GRPC_MODULE_PATH: mock_module}):
        yield GrpcIndex(host=INDEX_HOST, api_key="test-key")


# ---------------------------------------------------------------------------
# REST sync
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("import_id", "encoded"), ENCODING_CASES)
@respx.mock
def test_describe_import_encodes_id_as_one_path_segment(
    index: Index, import_id: str, encoded: str
) -> None:
    route = respx.get(f"{IMPORTS_URL}/{encoded}").mock(
        return_value=httpx.Response(200, json=IMPORT_PAYLOAD)
    )

    index.describe_import(import_id)

    assert route.called
    assert route.calls.last.request.url.raw_path.decode() == f"/bulk/imports/{encoded}"


@pytest.mark.parametrize(("import_id", "encoded"), ENCODING_CASES)
@respx.mock
def test_cancel_import_encodes_id_as_one_path_segment(
    index: Index, import_id: str, encoded: str
) -> None:
    route = respx.delete(f"{IMPORTS_URL}/{encoded}").mock(return_value=httpx.Response(202))

    index.cancel_import(import_id)

    assert route.called
    assert route.calls.last.request.url.raw_path.decode() == f"/bulk/imports/{encoded}"


# ---------------------------------------------------------------------------
# REST async
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("import_id", "encoded"), ENCODING_CASES)
async def test_async_describe_import_encodes_id_as_one_path_segment(
    async_index: AsyncIndex, import_id: str, encoded: str, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.get(f"{IMPORTS_URL}/{encoded}").mock(
        return_value=httpx.Response(200, json=IMPORT_PAYLOAD)
    )

    await async_index.describe_import(import_id)

    assert route.called
    assert route.calls.last.request.url.raw_path.decode() == f"/bulk/imports/{encoded}"


@pytest.mark.parametrize(("import_id", "encoded"), ENCODING_CASES)
async def test_async_cancel_import_encodes_id_as_one_path_segment(
    async_index: AsyncIndex, import_id: str, encoded: str, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.delete(f"{IMPORTS_URL}/{encoded}").mock(
        return_value=httpx.Response(202, content=b"")
    )

    await async_index.cancel_import(import_id)

    assert route.called
    assert route.calls.last.request.url.raw_path.decode() == f"/bulk/imports/{encoded}"


# ---------------------------------------------------------------------------
# gRPC wrapper (bulk imports ride the HTTP lane)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("import_id", "encoded"), ENCODING_CASES)
@respx.mock
def test_grpc_describe_import_encodes_id_as_one_path_segment(
    grpc_index: GrpcIndex, import_id: str, encoded: str
) -> None:
    route = respx.get(f"{IMPORTS_URL}/{encoded}").mock(
        return_value=httpx.Response(200, json=IMPORT_PAYLOAD)
    )

    grpc_index.describe_import(import_id)

    assert route.called
    assert route.calls.last.request.url.raw_path.decode() == f"/bulk/imports/{encoded}"


@pytest.mark.parametrize(("import_id", "encoded"), ENCODING_CASES)
@respx.mock
def test_grpc_cancel_import_encodes_id_as_one_path_segment(
    grpc_index: GrpcIndex, import_id: str, encoded: str
) -> None:
    route = respx.delete(f"{IMPORTS_URL}/{encoded}").mock(return_value=httpx.Response(202))

    grpc_index.cancel_import(import_id)

    assert route.called
    assert route.calls.last.request.url.raw_path.decode() == f"/bulk/imports/{encoded}"
