"""The records routes carry the namespace in the URL path, so it must be encoded.

``GrpcIndex.upsert_records`` and ``GrpcIndex.search`` delegate to the REST
``/records/namespaces/{namespace}/...`` routes — the gRPC API has no records
rpcs — and interpolated the namespace raw, the identical defect #233 fixed on
the REST and asyncio clients. The 2026-07 namespace charset
(``^[\\x01-\\x7F]+$``, ``db_data_2026-07.oas.yaml:975-984`` @ apis ``5f808858``)
admits ``/``, ``?``, ``#``, ``%`` and the C0 controls, none of which survive
raw interpolation into a path. These cases bind the same table
``tests/unit/test_records_path_encoding.py`` pins for the other two lanes.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest
import respx

from pinecone.grpc import GrpcIndex
from tests.unit.test_records_path_encoding import ENCODING_CASES

_MOCK_GRPC_MODULE_PATH = "pinecone._grpc"
INDEX_HOST = "records-path-encoding-abc123.svc.pinecone.io"
RECORDS_URL = f"https://{INDEX_HOST}/records/namespaces"

SEARCH_RESPONSE: dict[str, Any] = {
    "result": {"hits": [{"_id": "r1", "_score": 0.5, "fields": {}}]},
    "usage": {"read_units": 1},
}


@pytest.fixture
def grpc_index() -> GrpcIndex:
    mock_module = MagicMock()
    mock_module.GrpcChannel.return_value = MagicMock()
    with patch.dict("sys.modules", {_MOCK_GRPC_MODULE_PATH: mock_module}):
        return GrpcIndex(host=INDEX_HOST, api_key="test-key")


@pytest.mark.parametrize(("namespace", "encoded"), ENCODING_CASES)
@respx.mock
def test_grpc_upsert_records_encodes_namespace_as_one_path_segment(
    grpc_index: GrpcIndex, namespace: str, encoded: str
) -> None:
    route = respx.post(f"{RECORDS_URL}/{encoded}/upsert").mock(return_value=httpx.Response(201))

    grpc_index.upsert_records(namespace=namespace, records=[{"_id": "r1", "text": "hello"}])

    assert route.called
    assert route.calls.last.request.url.raw_path.decode() == (
        f"/records/namespaces/{encoded}/upsert"
    )


@pytest.mark.parametrize(("namespace", "encoded"), ENCODING_CASES)
@respx.mock
def test_grpc_search_encodes_namespace_as_one_path_segment(
    grpc_index: GrpcIndex, namespace: str, encoded: str
) -> None:
    route = respx.post(f"{RECORDS_URL}/{encoded}/search").mock(
        return_value=httpx.Response(200, json=SEARCH_RESPONSE)
    )

    grpc_index.search(namespace=namespace, top_k=1, inputs={"text": "hello"})

    assert route.called
    assert route.calls.last.request.url.raw_path.decode() == (
        f"/records/namespaces/{encoded}/search"
    )
