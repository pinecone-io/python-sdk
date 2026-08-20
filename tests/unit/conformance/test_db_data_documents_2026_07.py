"""2026-07 conformance for the documents operations graduated onto Index (#132, #135).

Claims the six document operations the sync ``Index`` implements —
``upsertDocuments``, ``searchDocuments``, ``fetchDocuments``,
``deleteDocuments`` (#132) and ``updateDocuments``, ``listDocuments`` (#135).
The async twin adds its own claims for the same operations.

Fixtures mirror the response examples and required-field sets of
``db_data_2026-07.oas.yaml`` (UpsertDocumentsResponse:2918,
SearchDocumentsResponse:3089, FetchDocumentsResponse:3182,
DeleteDocumentsResponse:3234, UpdateDocumentsResponse, ListDocumentsResponse).
The search, fetch, and list envelopes are open-schema or renamed-field on the
wire, so the round-trip runs through the internal msgspec envelopes the SDK
itself decodes with, the same way ``_ImportListEnvelope`` is claimed for
``listBulkImports``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import httpx
import pytest
import respx

from pinecone._internal.adapters.documents_adapter import (
    _FetchDocumentsEnvelope,
    _ListDocumentsEnvelope,
    _SearchDocumentsEnvelope,
)
from pinecone.index import Index
from pinecone.models.documents.responses import (
    DeleteDocumentsResponse,
    UpdateDocumentsResponse,
    UpsertDocumentsResponse,
)
from tests.unit.conformance import api_op

INDEX_HOST = "conformance-index-abc123.svc.us-east-1-aws.pinecone.io"
BASE_URL = f"https://{INDEX_HOST}"
NAMESPACE = "conformance-ns"

DOCUMENTS_INPUT: list[dict[str, Any]] = [
    {
        "_id": "doc-1",
        "content": "Machine learning is a subset of artificial intelligence.",
        "title": "Introduction to Machine Learning",
    },
    {
        "_id": "doc-2",
        "content": "Deep learning uses neural networks with many layers.",
        "title": "Deep Learning Fundamentals",
    },
]

DOC_UPSERT: dict[str, Any] = {"upserted_count": 2}
DOC_SEARCH: dict[str, Any] = {
    "matches": [
        {
            "_id": "doc-1",
            "_score": 0.9281134605407715,
            "title": "Introduction to Machine Learning",
        }
    ],
    "namespace": NAMESPACE,
    "usage": {"read_units": 5},
}
DOC_FETCH: dict[str, Any] = {
    "documents": {
        "doc-1": {
            "_id": "doc-1",
            "content": "Machine learning is a subset of artificial intelligence.",
            "title": "Introduction to Machine Learning",
        }
    },
    "pagination": {"next": "page-2"},
    "namespace": NAMESPACE,
    "usage": {"read_units": 5},
}
DOC_DELETE: dict[str, Any] = {"matched_records": 42}
DOC_UPDATE: dict[str, Any] = {"matched_records": 42}
DOC_UPDATE_INPUT: list[dict[str, Any]] = [
    {"_id": "doc-1", "title": "Updated title"},
    {"_id": "doc-2", "_remove_fields": ["content"]},
]
DOC_LIST: dict[str, Any] = {
    "documents": [{"_id": "doc-1"}, {"_id": "doc-2"}],
    "pagination": {"next": "page-2"},
    "namespace": NAMESPACE,
    "usage": {"read_units": 1},
}


@pytest.fixture
def index() -> Iterator[Index]:
    client = Index(host=INDEX_HOST, api_key="conformance-key")
    yield client
    client.close()


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
def test_upsert_documents(claim: Any, index: Index, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.post(f"{BASE_URL}/namespaces/{NAMESPACE}/documents/upsert").mock(
        return_value=httpx.Response(202, json=DOC_UPSERT)
    )
    result = index.upsert_documents(namespace=NAMESPACE, documents=DOCUMENTS_INPUT)
    assert result.upserted_count == 2
    _conforms(claim, route, UpsertDocumentsResponse, DOC_UPSERT, [])


@api_op("db_data:searchDocuments")
def test_search_documents(claim: Any, index: Index, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.post(f"{BASE_URL}/namespaces/{NAMESPACE}/documents/search").mock(
        return_value=httpx.Response(200, json=DOC_SEARCH)
    )
    result = index.search_documents(
        namespace=NAMESPACE,
        top_k=10,
        score_by=[{"type": "text", "fields": ["content"], "query": "What is machine learning?"}],
        include_fields=["title", "content"],
    )
    assert result.matches[0]._id == "doc-1"
    assert result.matches[0].title == "Introduction to Machine Learning"
    _conforms(claim, route, _SearchDocumentsEnvelope, DOC_SEARCH, [])


@api_op("db_data:fetchDocuments")
def test_fetch_documents(claim: Any, index: Index, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.post(f"{BASE_URL}/namespaces/{NAMESPACE}/documents/fetch").mock(
        return_value=httpx.Response(200, json=DOC_FETCH)
    )
    result = index.fetch_documents(namespace=NAMESPACE, filter={"category": {"$eq": "news"}})
    assert result.documents["doc-1"]._id == "doc-1"
    assert result.pagination is not None and result.pagination.next == "page-2"
    _conforms(claim, route, _FetchDocumentsEnvelope, DOC_FETCH, ["pagination"])


@api_op("db_data:deleteDocuments")
def test_delete_documents(claim: Any, index: Index, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.post(f"{BASE_URL}/namespaces/{NAMESPACE}/documents/delete").mock(
        return_value=httpx.Response(202, json=DOC_DELETE)
    )
    result = index.delete_documents(namespace=NAMESPACE, filter={"category": {"$eq": "news"}})
    assert result.matched_records == 42
    _conforms(claim, route, DeleteDocumentsResponse, DOC_DELETE, ["matched_records"])


@api_op("db_data:updateDocuments")
def test_update_documents(claim: Any, index: Index, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.post(f"{BASE_URL}/namespaces/{NAMESPACE}/documents/update").mock(
        return_value=httpx.Response(202, json=DOC_UPDATE)
    )
    result = index.update_documents(namespace=NAMESPACE, documents=DOC_UPDATE_INPUT)
    assert result.matched_records == 42
    _conforms(claim, route, UpdateDocumentsResponse, DOC_UPDATE, ["matched_records"])


@api_op("db_data:listDocuments")
def test_list_documents(claim: Any, index: Index, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.post(f"{BASE_URL}/namespaces/{NAMESPACE}/documents/list").mock(
        return_value=httpx.Response(200, json=DOC_LIST)
    )
    page = next(index.list_documents(namespace=NAMESPACE, prefix="doc-", limit=20).pages())
    assert [record.id for record in page.items] == ["doc-1", "doc-2"]
    assert page.pagination_token == "page-2"
    _conforms(claim, route, _ListDocumentsEnvelope, DOC_LIST, ["pagination"])
