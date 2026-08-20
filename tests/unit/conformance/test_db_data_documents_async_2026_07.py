"""2026-07 conformance for the asyncio transport of the documents operations (#134).

The sync variants live in ``test_db_data_documents_2026_07.py``; both may claim
the same operation (see README, "Additional rules"), so these add no operation
ids to the coverage numerator. What they add is that nothing on the async side
asserted the 2026-07 version header or the endpoint shape before this file —
``AsyncIndex`` could have regressed to another version with a green suite.

Payloads, host, and namespace are imported from the sync module rather than
restated, so the two transports cannot drift apart in their fixtures. The
request bodies each async method builds are re-asserted here: the body is what
the async method itself emits, and that is the thing under test.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

import httpx
import orjson
import pytest
import respx

from pinecone._internal.adapters.documents_adapter import (
    _FetchDocumentsEnvelope,
    _ListDocumentsEnvelope,
    _SearchDocumentsEnvelope,
)
from pinecone.async_client.async_index import AsyncIndex
from pinecone.models.documents.responses import (
    DeleteDocumentsResponse,
    UpdateDocumentsResponse,
    UpsertDocumentsResponse,
)
from tests.unit.conformance import api_op
from tests.unit.conformance.test_db_data_documents_2026_07 import (
    BASE_URL,
    DOC_DELETE,
    DOC_FETCH,
    DOC_LIST,
    DOC_SEARCH,
    DOC_UPDATE,
    DOC_UPDATE_INPUT,
    DOC_UPSERT,
    DOCUMENTS_INPUT,
    INDEX_HOST,
    NAMESPACE,
)


@pytest.fixture
async def async_index() -> AsyncGenerator[AsyncIndex]:
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


@api_op("db_data:upsertDocuments")
async def test_async_upsert_documents(
    claim: Any, async_index: AsyncIndex, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.post(f"{BASE_URL}/namespaces/{NAMESPACE}/documents/upsert").mock(
        return_value=httpx.Response(202, json=DOC_UPSERT)
    )
    result = await async_index.upsert_documents(namespace=NAMESPACE, documents=DOCUMENTS_INPUT)
    assert result.upserted_count == 2
    assert orjson.loads(route.calls.last.request.content) == {"documents": DOCUMENTS_INPUT}
    _conforms(claim, route, UpsertDocumentsResponse, DOC_UPSERT, [])


@api_op("db_data:searchDocuments")
async def test_async_search_documents(
    claim: Any, async_index: AsyncIndex, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.post(f"{BASE_URL}/namespaces/{NAMESPACE}/documents/search").mock(
        return_value=httpx.Response(200, json=DOC_SEARCH)
    )
    result = await async_index.search_documents(
        namespace=NAMESPACE,
        top_k=10,
        score_by=[{"type": "text", "fields": ["content"], "query": "What is machine learning?"}],
        include_fields=["title", "content"],
    )
    assert result.matches[0]._id == "doc-1"
    assert result.matches[0].title == "Introduction to Machine Learning"
    assert orjson.loads(route.calls.last.request.content) == {
        "score_by": [{"type": "text", "query": "What is machine learning?", "fields": ["content"]}],
        "top_k": 10,
        "include_fields": ["title", "content"],
    }
    _conforms(claim, route, _SearchDocumentsEnvelope, DOC_SEARCH, [])


@api_op("db_data:fetchDocuments")
async def test_async_fetch_documents(
    claim: Any, async_index: AsyncIndex, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.post(f"{BASE_URL}/namespaces/{NAMESPACE}/documents/fetch").mock(
        return_value=httpx.Response(200, json=DOC_FETCH)
    )
    result = await async_index.fetch_documents(
        namespace=NAMESPACE, filter={"category": {"$eq": "news"}}
    )
    assert result.documents["doc-1"]._id == "doc-1"
    assert result.pagination is not None and result.pagination.next == "page-2"
    assert orjson.loads(route.calls.last.request.content) == {
        "filter": {"category": {"$eq": "news"}}
    }
    _conforms(claim, route, _FetchDocumentsEnvelope, DOC_FETCH, ["pagination"])


@api_op("db_data:deleteDocuments")
async def test_async_delete_documents(
    claim: Any, async_index: AsyncIndex, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.post(f"{BASE_URL}/namespaces/{NAMESPACE}/documents/delete").mock(
        return_value=httpx.Response(202, json=DOC_DELETE)
    )
    result = await async_index.delete_documents(
        namespace=NAMESPACE, filter={"category": {"$eq": "news"}}
    )
    assert result.matched_records == 42
    assert orjson.loads(route.calls.last.request.content) == {
        "filter": {"category": {"$eq": "news"}}
    }
    _conforms(claim, route, DeleteDocumentsResponse, DOC_DELETE, ["matched_records"])


@api_op("db_data:updateDocuments")
async def test_async_update_documents(
    claim: Any, async_index: AsyncIndex, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.post(f"{BASE_URL}/namespaces/{NAMESPACE}/documents/update").mock(
        return_value=httpx.Response(202, json=DOC_UPDATE)
    )
    result = await async_index.update_documents(namespace=NAMESPACE, documents=DOC_UPDATE_INPUT)
    assert result.matched_records == 42
    assert orjson.loads(route.calls.last.request.content) == {"documents": DOC_UPDATE_INPUT}
    _conforms(claim, route, UpdateDocumentsResponse, DOC_UPDATE, ["matched_records"])


@api_op("db_data:listDocuments")
async def test_async_list_documents(
    claim: Any, async_index: AsyncIndex, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.post(f"{BASE_URL}/namespaces/{NAMESPACE}/documents/list").mock(
        return_value=httpx.Response(200, json=DOC_LIST)
    )
    paginator = async_index.list_documents(namespace=NAMESPACE, prefix="doc-", limit=20)
    page = await anext(paginator.pages())
    assert [record.id for record in page.items] == ["doc-1", "doc-2"]
    assert page.pagination_token == "page-2"
    assert orjson.loads(route.calls.last.request.content) == {"prefix": "doc-", "limit": 20}
    _conforms(claim, route, _ListDocumentsEnvelope, DOC_LIST, ["pagination"])
