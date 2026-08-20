"""2026-07 conformance for the 18 db_data operations carried over from 2025-10.

Ticket #91 is version-bump-only, and an op-by-op diff of
``db_data_2026-07.oas.yaml`` against its 2025-10 predecessor bears that out for
these 18: nothing wire-visible changes beyond the ``X-Pinecone-Api-Version``
default, newly documented 401 responses, response ``links``, and array
constraints respelled from ``minLength``/``maxLength`` to
``minItems``/``maxItems``. The request validation the same diff tightens and the
``size_bytes`` it adds to namespace descriptions belong to the M3 tickets
(#119-#124); the six ``documents`` operations 2026-07 introduces are new surface
and are deliberately not claimed here.

Every operation is claimed twice — once through :class:`Index`, once through
:class:`AsyncIndex` — because the header has to appear on the wire for both, and
both read it from the same ``DATA_PLANE_API_VERSION``.
``test_grpc_metadata_carries_the_same_constant`` closes the third transport: the
Rust channel is handed that same constant, and ``rust/src/transport.rs`` already
covers the interceptor that turns it into call metadata.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest
import respx

from pinecone._internal.adapters.imports_adapter import _ImportListEnvelope
from pinecone._internal.constants import DATA_PLANE_API_VERSION
from pinecone.async_client.async_index import AsyncIndex
from pinecone.index import Index
from pinecone.models.imports.model import ImportModel, StartImportResponse
from pinecone.models.namespaces.models import ListNamespacesResponse, NamespaceDescription
from pinecone.models.vectors.responses import (
    DescribeIndexStatsResponse,
    FetchByMetadataResponse,
    FetchResponse,
    ListResponse,
    QueryResponse,
    UpdateResponse,
    UpsertResponse,
)
from pinecone.models.vectors.search import SearchRecordsResponse
from tests.unit.conformance import api_op

INDEX_HOST = "conformance-index-abc123.svc.us-east-1-aws.pinecone.io"
BASE_URL = f"https://{INDEX_HOST}"
NAMESPACE = "conformance-ns"
IMPORT_ID = "101"

SPARSE: dict[str, Any] = {"indices": [17, 42], "values": [0.6, 0.4]}
VECTOR: dict[str, Any] = {
    "id": "vec-1",
    "values": [0.1, 0.2],
    "sparseValues": SPARSE,
    "metadata": {"genre": "documentary", "year": 2026},
}
VECTOR_INPUT: dict[str, Any] = {
    "id": "vec-1",
    "values": [0.1, 0.2],
    "sparse_values": SPARSE,
    "metadata": {"genre": "documentary", "year": 2026},
}

UPSERT: dict[str, Any] = {"upsertedCount": 2}
UPDATE: dict[str, Any] = {"matchedRecords": 1}
QUERY: dict[str, Any] = {
    "matches": [{**VECTOR, "score": 0.92}],
    "namespace": NAMESPACE,
    "usage": {"readUnits": 5},
}
FETCH: dict[str, Any] = {
    "vectors": {"vec-1": VECTOR},
    "namespace": NAMESPACE,
    "usage": {"readUnits": 1},
}
FETCH_BY_METADATA: dict[str, Any] = {**FETCH, "pagination": {"next": "page-2"}}
LIST_VECTORS: dict[str, Any] = {
    "vectors": [{"id": "vec-1"}],
    "pagination": {"next": "page-2"},
    "namespace": NAMESPACE,
    "usage": {"readUnits": 1},
}
STATS: dict[str, Any] = {
    "namespaces": {NAMESPACE: {"vectorCount": 80000}},
    "dimension": 1024,
    "indexFullness": 0.4,
    "totalVectorCount": 80000,
    "metric": "cosine",
    "vectorType": "dense",
    "memoryFullness": 0.25,
    "storageFullness": 0.75,
}
SEARCH: dict[str, Any] = {
    "result": {"hits": [{"_id": "rec-1", "_score": 0.81, "fields": {"chunk_text": "hello"}}]},
    "usage": {"read_units": 3, "embed_total_tokens": 12, "rerank_units": 1},
}
NAMESPACE_DESCRIPTION: dict[str, Any] = {
    "name": NAMESPACE,
    "record_count": 42,
    "schema": {"fields": {"genre": {"filterable": True}}},
    "indexed_fields": {"fields": ["genre"]},
}
LIST_NAMESPACES: dict[str, Any] = {
    "namespaces": [NAMESPACE_DESCRIPTION],
    "pagination": {"next": "page-2"},
    "total_count": 1,
}
IMPORT: dict[str, Any] = {
    "id": IMPORT_ID,
    "uri": "s3://bucket/prefix",
    "status": "InProgress",
    "createdAt": "2026-07-01T12:00:00Z",
    "finishedAt": "2026-07-01T12:30:00Z",
    "percentComplete": 42.5,
    "recordsImported": 34000,
    "error": "",
}
LIST_IMPORTS: dict[str, Any] = {"data": [IMPORT], "pagination": {"next": "page-2"}}

RECORDS: list[dict[str, Any]] = [{"_id": "rec-1", "chunk_text": "hello"}]
UPSERT_RECORDS_CLIENT_SIDE = ["record_count", "response_info"]


@pytest.fixture
def index() -> Iterator[Index]:
    client = Index(host=INDEX_HOST, api_key="conformance-key")
    yield client
    client.close()


@pytest.fixture
async def async_index() -> AsyncIterator[AsyncIndex]:
    client = AsyncIndex(host=INDEX_HOST, api_key="conformance-key")
    yield client
    await client.close()


def _conforms(
    claim: Any,
    route: respx.Route,
    model: type,
    payload: dict[str, Any],
    optional_absent: list[str],
) -> None:
    request = route.calls.last.request
    claim.assert_request(request)
    claim.assert_api_version(request)
    claim.assert_roundtrip(model, payload, optional_absent=optional_absent)


def _conforms_bodyless(
    claim: Any, route: respx.Route, returned: Any, client_side: list[str] | None = None
) -> None:
    request = route.calls.last.request
    claim.assert_request(request)
    claim.assert_api_version(request)
    claim.assert_no_response_body(returned, client_side=client_side or [])


@api_op("db_data:upsertVectors")
def test_upsert(claim: Any, index: Index, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.post(f"{BASE_URL}/vectors/upsert").mock(
        return_value=httpx.Response(200, json=UPSERT)
    )
    assert index.upsert(vectors=[VECTOR_INPUT], namespace=NAMESPACE).upserted_count == 2
    _conforms(claim, route, UpsertResponse, UPSERT, [])


@api_op("db_data:upsertVectors")
@pytest.mark.anyio
async def test_async_upsert(
    claim: Any, async_index: AsyncIndex, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.post(f"{BASE_URL}/vectors/upsert").mock(
        return_value=httpx.Response(200, json=UPSERT)
    )
    result = await async_index.upsert(vectors=[VECTOR_INPUT], namespace=NAMESPACE)
    assert result.upserted_count == 2
    _conforms(claim, route, UpsertResponse, UPSERT, [])


@api_op("db_data:queryVectors")
def test_query(claim: Any, index: Index, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.post(f"{BASE_URL}/query").mock(return_value=httpx.Response(200, json=QUERY))
    assert index.query(top_k=1, vector=[0.1, 0.2], namespace=NAMESPACE).matches[0].id == "vec-1"
    _conforms(claim, route, QueryResponse, QUERY, ["usage"])


@api_op("db_data:queryVectors")
@pytest.mark.anyio
async def test_async_query(
    claim: Any, async_index: AsyncIndex, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.post(f"{BASE_URL}/query").mock(return_value=httpx.Response(200, json=QUERY))
    result = await async_index.query(top_k=1, vector=[0.1, 0.2], namespace=NAMESPACE)
    assert result.matches[0].id == "vec-1"
    _conforms(claim, route, QueryResponse, QUERY, ["usage"])


@api_op("db_data:fetchVectors")
def test_fetch(claim: Any, index: Index, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.get(f"{BASE_URL}/vectors/fetch").mock(
        return_value=httpx.Response(200, json=FETCH)
    )
    assert index.fetch(ids=["vec-1"], namespace=NAMESPACE).vectors["vec-1"].id == "vec-1"
    _conforms(claim, route, FetchResponse, FETCH, ["usage"])


@api_op("db_data:fetchVectors")
@pytest.mark.anyio
async def test_async_fetch(
    claim: Any, async_index: AsyncIndex, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.get(f"{BASE_URL}/vectors/fetch").mock(
        return_value=httpx.Response(200, json=FETCH)
    )
    result = await async_index.fetch(ids=["vec-1"], namespace=NAMESPACE)
    assert result.vectors["vec-1"].id == "vec-1"
    _conforms(claim, route, FetchResponse, FETCH, ["usage"])


@api_op("db_data:fetch_vectors_by_metadata")
def test_fetch_by_metadata(claim: Any, index: Index, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.post(f"{BASE_URL}/vectors/fetch_by_metadata").mock(
        return_value=httpx.Response(200, json=FETCH_BY_METADATA)
    )
    result = index.fetch_by_metadata(filter={"genre": {"$eq": "documentary"}}, namespace=NAMESPACE)
    assert result.pagination is not None
    _conforms(claim, route, FetchByMetadataResponse, FETCH_BY_METADATA, ["pagination"])


@api_op("db_data:fetch_vectors_by_metadata")
@pytest.mark.anyio
async def test_async_fetch_by_metadata(
    claim: Any, async_index: AsyncIndex, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.post(f"{BASE_URL}/vectors/fetch_by_metadata").mock(
        return_value=httpx.Response(200, json=FETCH_BY_METADATA)
    )
    result = await async_index.fetch_by_metadata(
        filter={"genre": {"$eq": "documentary"}}, namespace=NAMESPACE
    )
    assert result.pagination is not None
    _conforms(claim, route, FetchByMetadataResponse, FETCH_BY_METADATA, ["pagination"])


@api_op("db_data:listVectors")
def test_list_vectors(claim: Any, index: Index, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.get(f"{BASE_URL}/vectors/list").mock(
        return_value=httpx.Response(200, json=LIST_VECTORS)
    )
    assert index.list_paginated(namespace=NAMESPACE).vectors[0].id == "vec-1"
    _conforms(claim, route, ListResponse, LIST_VECTORS, ["pagination"])


@api_op("db_data:listVectors")
@pytest.mark.anyio
async def test_async_list_vectors(
    claim: Any, async_index: AsyncIndex, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.get(f"{BASE_URL}/vectors/list").mock(
        return_value=httpx.Response(200, json=LIST_VECTORS)
    )
    result = await async_index.list_paginated(namespace=NAMESPACE)
    assert result.vectors[0].id == "vec-1"
    _conforms(claim, route, ListResponse, LIST_VECTORS, ["pagination"])


@api_op("db_data:updateVector")
def test_update(claim: Any, index: Index, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.post(f"{BASE_URL}/vectors/update").mock(
        return_value=httpx.Response(200, json=UPDATE)
    )
    assert index.update(id="vec-1", values=[0.1, 0.2], namespace=NAMESPACE).matched_records == 1
    _conforms(claim, route, UpdateResponse, UPDATE, ["matchedRecords"])


@api_op("db_data:updateVector")
@pytest.mark.anyio
async def test_async_update(
    claim: Any, async_index: AsyncIndex, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.post(f"{BASE_URL}/vectors/update").mock(
        return_value=httpx.Response(200, json=UPDATE)
    )
    result = await async_index.update(id="vec-1", values=[0.1, 0.2], namespace=NAMESPACE)
    assert result.matched_records == 1
    _conforms(claim, route, UpdateResponse, UPDATE, ["matchedRecords"])


@api_op("db_data:deleteVectors")
def test_delete(claim: Any, index: Index, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.post(f"{BASE_URL}/vectors/delete").mock(
        return_value=httpx.Response(200, json={})
    )
    _conforms_bodyless(claim, route, index.delete(ids=["vec-1"], namespace=NAMESPACE))


@api_op("db_data:deleteVectors")
@pytest.mark.anyio
async def test_async_delete(
    claim: Any, async_index: AsyncIndex, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.post(f"{BASE_URL}/vectors/delete").mock(
        return_value=httpx.Response(200, json={})
    )
    returned = await async_index.delete(ids=["vec-1"], namespace=NAMESPACE)
    _conforms_bodyless(claim, route, returned)


@api_op("db_data:describeIndexStats")
def test_describe_index_stats(claim: Any, index: Index, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.post(f"{BASE_URL}/describe_index_stats").mock(
        return_value=httpx.Response(200, json=STATS)
    )
    assert index.describe_index_stats().total_vector_count == 80000
    _conforms(claim, route, DescribeIndexStatsResponse, STATS, ["memoryFullness"])


@api_op("db_data:describeIndexStats")
@pytest.mark.anyio
async def test_async_describe_index_stats(
    claim: Any, async_index: AsyncIndex, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.post(f"{BASE_URL}/describe_index_stats").mock(
        return_value=httpx.Response(200, json=STATS)
    )
    result = await async_index.describe_index_stats()
    assert result.total_vector_count == 80000
    _conforms(claim, route, DescribeIndexStatsResponse, STATS, ["memoryFullness"])


@api_op("db_data:upsertRecordsNamespace")
def test_upsert_records(claim: Any, index: Index, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.post(f"{BASE_URL}/records/namespaces/{NAMESPACE}/upsert").mock(
        return_value=httpx.Response(201)
    )
    returned = index.upsert_records(records=RECORDS, namespace=NAMESPACE)
    assert returned.record_count == 1
    _conforms_bodyless(claim, route, returned, UPSERT_RECORDS_CLIENT_SIDE)


@api_op("db_data:upsertRecordsNamespace")
@pytest.mark.anyio
async def test_async_upsert_records(
    claim: Any, async_index: AsyncIndex, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.post(f"{BASE_URL}/records/namespaces/{NAMESPACE}/upsert").mock(
        return_value=httpx.Response(201)
    )
    returned = await async_index.upsert_records(records=RECORDS, namespace=NAMESPACE)
    assert returned.record_count == 1
    _conforms_bodyless(claim, route, returned, UPSERT_RECORDS_CLIENT_SIDE)


@api_op("db_data:searchRecordsNamespace")
def test_search_records(claim: Any, index: Index, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.post(f"{BASE_URL}/records/namespaces/{NAMESPACE}/search").mock(
        return_value=httpx.Response(200, json=SEARCH)
    )
    result = index.search(namespace=NAMESPACE, top_k=1, inputs={"text": "hello"})
    assert result.result.hits[0].id_ == "rec-1"
    _conforms(claim, route, SearchRecordsResponse, SEARCH, [])


@api_op("db_data:searchRecordsNamespace")
@pytest.mark.anyio
async def test_async_search_records(
    claim: Any, async_index: AsyncIndex, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.post(f"{BASE_URL}/records/namespaces/{NAMESPACE}/search").mock(
        return_value=httpx.Response(200, json=SEARCH)
    )
    result = await async_index.search(namespace=NAMESPACE, top_k=1, inputs={"text": "hello"})
    assert result.result.hits[0].id_ == "rec-1"
    _conforms(claim, route, SearchRecordsResponse, SEARCH, [])


@api_op("db_data:createNamespace")
def test_create_namespace(claim: Any, index: Index, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.post(f"{BASE_URL}/namespaces").mock(
        return_value=httpx.Response(200, json=NAMESPACE_DESCRIPTION)
    )
    assert index.create_namespace(name=NAMESPACE).name == NAMESPACE
    _conforms(claim, route, NamespaceDescription, NAMESPACE_DESCRIPTION, ["schema"])


@api_op("db_data:createNamespace")
@pytest.mark.anyio
async def test_async_create_namespace(
    claim: Any, async_index: AsyncIndex, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.post(f"{BASE_URL}/namespaces").mock(
        return_value=httpx.Response(200, json=NAMESPACE_DESCRIPTION)
    )
    result = await async_index.create_namespace(name=NAMESPACE)
    assert result.name == NAMESPACE
    _conforms(claim, route, NamespaceDescription, NAMESPACE_DESCRIPTION, ["schema"])


@api_op("db_data:describeNamespace")
def test_describe_namespace(claim: Any, index: Index, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.get(f"{BASE_URL}/namespaces/{NAMESPACE}").mock(
        return_value=httpx.Response(200, json=NAMESPACE_DESCRIPTION)
    )
    assert index.describe_namespace(name=NAMESPACE).record_count == 42
    _conforms(claim, route, NamespaceDescription, NAMESPACE_DESCRIPTION, ["schema"])


@api_op("db_data:describeNamespace")
@pytest.mark.anyio
async def test_async_describe_namespace(
    claim: Any, async_index: AsyncIndex, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.get(f"{BASE_URL}/namespaces/{NAMESPACE}").mock(
        return_value=httpx.Response(200, json=NAMESPACE_DESCRIPTION)
    )
    result = await async_index.describe_namespace(name=NAMESPACE)
    assert result.record_count == 42
    _conforms(claim, route, NamespaceDescription, NAMESPACE_DESCRIPTION, ["schema"])


@api_op("db_data:listNamespacesOperation")
def test_list_namespaces(claim: Any, index: Index, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.get(f"{BASE_URL}/namespaces").mock(
        return_value=httpx.Response(200, json=LIST_NAMESPACES)
    )
    assert index.list_namespaces_paginated().namespaces[0].name == NAMESPACE
    _conforms(claim, route, ListNamespacesResponse, LIST_NAMESPACES, ["pagination"])


@api_op("db_data:listNamespacesOperation")
@pytest.mark.anyio
async def test_async_list_namespaces(
    claim: Any, async_index: AsyncIndex, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.get(f"{BASE_URL}/namespaces").mock(
        return_value=httpx.Response(200, json=LIST_NAMESPACES)
    )
    result = await async_index.list_namespaces_paginated()
    assert result.namespaces[0].name == NAMESPACE
    _conforms(claim, route, ListNamespacesResponse, LIST_NAMESPACES, ["pagination"])


@api_op("db_data:deleteNamespace")
def test_delete_namespace(claim: Any, index: Index, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.delete(f"{BASE_URL}/namespaces/{NAMESPACE}").mock(
        return_value=httpx.Response(200, json={})
    )
    _conforms_bodyless(claim, route, index.delete_namespace(name=NAMESPACE))


@api_op("db_data:deleteNamespace")
@pytest.mark.anyio
async def test_async_delete_namespace(
    claim: Any, async_index: AsyncIndex, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.delete(f"{BASE_URL}/namespaces/{NAMESPACE}").mock(
        return_value=httpx.Response(200, json={})
    )
    returned = await async_index.delete_namespace(name=NAMESPACE)
    _conforms_bodyless(claim, route, returned)


@api_op("db_data:startBulkImport")
def test_start_import(claim: Any, index: Index, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.post(f"{BASE_URL}/bulk/imports").mock(
        return_value=httpx.Response(200, json={"id": IMPORT_ID})
    )
    assert index.start_import("s3://bucket/prefix").id == IMPORT_ID
    _conforms(claim, route, StartImportResponse, {"id": IMPORT_ID}, [])


@api_op("db_data:startBulkImport")
@pytest.mark.anyio
async def test_async_start_import(
    claim: Any, async_index: AsyncIndex, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.post(f"{BASE_URL}/bulk/imports").mock(
        return_value=httpx.Response(200, json={"id": IMPORT_ID})
    )
    result = await async_index.start_import("s3://bucket/prefix")
    assert result.id == IMPORT_ID
    _conforms(claim, route, StartImportResponse, {"id": IMPORT_ID}, [])


@api_op("db_data:describeBulkImport")
def test_describe_import(claim: Any, index: Index, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.get(f"{BASE_URL}/bulk/imports/{IMPORT_ID}").mock(
        return_value=httpx.Response(200, json=IMPORT)
    )
    assert index.describe_import(IMPORT_ID).status == "InProgress"
    _conforms(claim, route, ImportModel, IMPORT, ["percentComplete"])


@api_op("db_data:describeBulkImport")
@pytest.mark.anyio
async def test_async_describe_import(
    claim: Any, async_index: AsyncIndex, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.get(f"{BASE_URL}/bulk/imports/{IMPORT_ID}").mock(
        return_value=httpx.Response(200, json=IMPORT)
    )
    result = await async_index.describe_import(IMPORT_ID)
    assert result.status == "InProgress"
    _conforms(claim, route, ImportModel, IMPORT, ["percentComplete"])


@api_op("db_data:listBulkImports")
def test_list_imports(claim: Any, index: Index, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.get(f"{BASE_URL}/bulk/imports").mock(
        return_value=httpx.Response(200, json=LIST_IMPORTS)
    )
    assert index.list_imports_paginated()[0].id == IMPORT_ID
    _conforms(claim, route, _ImportListEnvelope, LIST_IMPORTS, ["pagination"])


@api_op("db_data:listBulkImports")
@pytest.mark.anyio
async def test_async_list_imports(
    claim: Any, async_index: AsyncIndex, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.get(f"{BASE_URL}/bulk/imports").mock(
        return_value=httpx.Response(200, json=LIST_IMPORTS)
    )
    result = await async_index.list_imports_paginated()
    assert result[0].id == IMPORT_ID
    _conforms(claim, route, _ImportListEnvelope, LIST_IMPORTS, ["pagination"])


@api_op("db_data:cancelBulkImport")
def test_cancel_import(claim: Any, index: Index, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.delete(f"{BASE_URL}/bulk/imports/{IMPORT_ID}").mock(
        return_value=httpx.Response(200, json={})
    )
    _conforms_bodyless(claim, route, index.cancel_import(IMPORT_ID))


@api_op("db_data:cancelBulkImport")
@pytest.mark.anyio
async def test_async_cancel_import(
    claim: Any, async_index: AsyncIndex, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.delete(f"{BASE_URL}/bulk/imports/{IMPORT_ID}").mock(
        return_value=httpx.Response(200, json={})
    )
    returned = await async_index.cancel_import(IMPORT_ID)
    _conforms_bodyless(claim, route, returned)


def test_grpc_metadata_carries_the_same_constant() -> None:
    module = MagicMock()
    with patch.dict("sys.modules", {"pinecone._grpc": module}):
        from pinecone.grpc import GrpcIndex

        GrpcIndex(host=INDEX_HOST, api_key="conformance-key")

    assert module.GrpcChannel.call_args.args[2] == DATA_PLANE_API_VERSION
    assert DATA_PLANE_API_VERSION == "2026-07"
