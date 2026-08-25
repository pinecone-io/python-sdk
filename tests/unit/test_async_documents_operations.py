"""Behavior tests for the document operations on AsyncIndex (#134).

The async half of ``tests/unit/test_documents_operations.py``: wire bodies,
client-side validation vocabulary, namespace path-segment encoding, batch
aggregation semantics, optional-field-absent responses, and intact surfacing of
server 400 text. Byte-identity with the sync transport is pinned separately in
``tests/unit/test_async_documents_parity.py``; what is asserted here is that the
async transport is correct on its own terms, plus the two async-only contracts
the ticket names — no request is issued at handle creation, and ``close()`` is
idempotent.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
import respx

from pinecone.async_client.async_index import AsyncIndex
from pinecone.errors.exceptions import ApiError, PineconeValueError
from pinecone.models.batch import BatchResult
from pinecone.models.documents.document import DocumentRecord, UpdateDocumentRecord
from pinecone.models.documents.score_by import DenseVectorQuery, TextQuery

INDEX_HOST = "documents-index-abc123.svc.us-east-1-aws.pinecone.io"
BASE_URL = f"https://{INDEX_HOST}"
NS = "articles-en"

UPSERT_OK = httpx.Response(202, json={"upserted_count": 1})
SEARCH_OK = httpx.Response(
    200,
    json={
        "matches": [{"_id": "doc-1", "_score": 0.9, "title": "Rome"}],
        "namespace": NS,
        "usage": {"read_units": 5},
    },
)
FETCH_OK = httpx.Response(
    200,
    json={
        "documents": {"doc-1": {"_id": "doc-1", "title": "Rome"}},
        "namespace": NS,
        "usage": {"read_units": 1},
    },
)
DELETE_OK = httpx.Response(202, json={})
UPDATE_OK = httpx.Response(202, json={})
LIST_OK_LAST_PAGE = httpx.Response(
    200,
    json={"documents": [{"_id": "doc-1"}], "namespace": NS, "usage": {"read_units": 1}},
)


@pytest.fixture
async def index() -> AsyncIterator[AsyncIndex]:
    client = AsyncIndex(host=INDEX_HOST, api_key="test-key")
    yield client
    await client.close()


def _body(route: respx.Route) -> dict[str, Any]:
    return dict(json.loads(route.calls.last.request.content))


class TestUpsertDocumentsWire:
    @respx.mock
    async def test_body_carries_documents_verbatim(self, index: AsyncIndex) -> None:
        route = respx.post(f"{BASE_URL}/namespaces/{NS}/documents/upsert").mock(
            return_value=UPSERT_OK
        )
        docs = [{"_id": "doc-1", "title": "Rome", "year": 2026, "embedding": [0.1, 0.2]}]
        result = await index.documents.upsert(namespace=NS, documents=docs)
        assert result.upserted_count == 1
        assert _body(route) == {"documents": docs}

    @respx.mock
    async def test_accepts_document_record_instances(self, index: AsyncIndex) -> None:
        route = respx.post(f"{BASE_URL}/namespaces/{NS}/documents/upsert").mock(
            return_value=UPSERT_OK
        )
        await index.documents.upsert(
            namespace=NS, documents=[DocumentRecord({"_id": "doc-1", "title": "Rome"})]
        )
        assert _body(route) == {"documents": [{"_id": "doc-1", "title": "Rome"}]}

    @respx.mock
    async def test_response_info_extracted(self, index: AsyncIndex) -> None:
        respx.post(f"{BASE_URL}/namespaces/{NS}/documents/upsert").mock(
            return_value=httpx.Response(
                202,
                json={"upserted_count": 1},
                headers={"x-pinecone-request-id": "req-77"},
            )
        )
        result = await index.documents.upsert(namespace=NS, documents=[{"_id": "a"}])
        assert result.response_info is not None
        assert result.response_info.request_id == "req-77"


class TestNamespaceHandling:
    @respx.mock
    async def test_namespace_path_segment_is_url_encoded(self, index: AsyncIndex) -> None:
        route = respx.post(
            f"{BASE_URL}/namespaces/my%20ns%2Fv1/documents/upsert",
        ).mock(return_value=UPSERT_OK)
        await index.documents.upsert(namespace="my ns/v1", documents=[{"_id": "a"}])
        assert route.calls.last.request.url.raw_path.decode().startswith(
            "/namespaces/my%20ns%2Fv1/documents/upsert"
        )

    @pytest.mark.parametrize("namespace", ["", "   "])
    async def test_empty_namespace_rejected(self, index: AsyncIndex, namespace: str) -> None:
        with pytest.raises(PineconeValueError, match="namespace must be a non-empty string"):
            await index.documents.upsert(namespace=namespace, documents=[{"_id": "a"}])

    async def test_non_string_namespace_rejected(self, index: AsyncIndex) -> None:
        with pytest.raises(PineconeValueError, match="namespace must be a string"):
            await index.documents.fetch(namespace=7, ids=["a"])  # type: ignore[arg-type]


class TestUpsertDocumentsValidation:
    async def test_empty_documents_rejected(self, index: AsyncIndex) -> None:
        with pytest.raises(PineconeValueError, match="documents must be a non-empty list"):
            await index.documents.upsert(namespace=NS, documents=[])

    async def test_over_1000_documents_rejected(self, index: AsyncIndex) -> None:
        docs = [{"_id": f"doc-{i}"} for i in range(1001)]
        with pytest.raises(PineconeValueError, match="1000"):
            await index.documents.upsert(namespace=NS, documents=docs)

    @pytest.mark.parametrize(
        ("bad_doc", "fragment"),
        [
            ({"title": "no id"}, "required"),
            ({"_id": ""}, "must not be empty"),
            ({"_id": 42}, "must be a string"),
            ({"_id": "x" * 513}, "maximum length"),
            ({"_id": "ünïcode"}, "ASCII"),
        ],
    )
    async def test_invalid_id_names_position(
        self, index: AsyncIndex, bad_doc: dict[str, Any], fragment: str
    ) -> None:
        with pytest.raises(PineconeValueError, match="position 1") as excinfo:
            await index.documents.upsert(namespace=NS, documents=[{"_id": "ok"}, bad_doc])
        assert fragment in str(excinfo.value)

    async def test_duplicate_id_names_position(self, index: AsyncIndex) -> None:
        with pytest.raises(PineconeValueError, match=r"position 2.*duplicate '_id' 'doc-1'"):
            await index.documents.upsert(
                namespace=NS,
                documents=[{"_id": "doc-1"}, {"_id": "doc-2"}, {"_id": "doc-1"}],
            )

    async def test_non_dict_document_rejected(self, index: AsyncIndex) -> None:
        with pytest.raises(PineconeValueError, match=r"position 0.*got str"):
            await index.documents.upsert(namespace=NS, documents=["doc-1"])  # type: ignore[list-item]


class TestSearchDocuments:
    @respx.mock
    async def test_body_with_typed_score_by(self, index: AsyncIndex) -> None:
        route = respx.post(f"{BASE_URL}/namespaces/{NS}/documents/search").mock(
            return_value=SEARCH_OK
        )
        result = await index.documents.search(
            namespace=NS,
            top_k=5,
            score_by=[TextQuery(query="rome", fields=["content"])],
            include_fields=["title"],
            filter={"category": {"$eq": "tech"}},
        )
        assert result.matches[0].title == "Rome"
        assert result.usage is not None and result.usage.read_units == 5
        assert _body(route) == {
            "score_by": [{"type": "text", "query": "rome", "fields": ["content"]}],
            "top_k": 5,
            "include_fields": ["title"],
            "filter": {"category": {"$eq": "tech"}},
        }

    @respx.mock
    async def test_optional_fields_absent_stay_off_the_wire(self, index: AsyncIndex) -> None:
        route = respx.post(f"{BASE_URL}/namespaces/{NS}/documents/search").mock(
            return_value=SEARCH_OK
        )
        await index.documents.search(
            namespace=NS,
            top_k=5,
            score_by=[{"type": "query_string", "query": "title:(rome)"}],
        )
        assert _body(route) == {
            "score_by": [{"type": "query_string", "query": "title:(rome)"}],
            "top_k": 5,
        }

    @respx.mock
    async def test_empty_include_fields_sent_explicitly(self, index: AsyncIndex) -> None:
        route = respx.post(f"{BASE_URL}/namespaces/{NS}/documents/search").mock(
            return_value=SEARCH_OK
        )
        await index.documents.search(
            namespace=NS,
            top_k=5,
            score_by=[{"type": "query_string", "query": "rome"}],
            include_fields=[],
        )
        assert _body(route)["include_fields"] == []

    async def test_empty_score_by_rejected(self, index: AsyncIndex) -> None:
        with pytest.raises(PineconeValueError, match="at least one scoring method"):
            await index.documents.search(namespace=NS, top_k=5, score_by=[])

    async def test_over_100_score_by_clauses_rejected(self, index: AsyncIndex) -> None:
        clauses = [{"type": "query_string", "query": f"q{i}"} for i in range(101)]
        with pytest.raises(PineconeValueError, match="100"):
            await index.documents.search(namespace=NS, top_k=5, score_by=clauses)

    async def test_vector_clause_must_appear_alone(self, index: AsyncIndex) -> None:
        with pytest.raises(PineconeValueError, match="must appear alone"):
            await index.documents.search(
                namespace=NS,
                top_k=5,
                score_by=[
                    DenseVectorQuery(field="embedding", values=[0.1]),
                    TextQuery(query="rome", fields=["content"]),
                ],
            )

    @pytest.mark.parametrize("top_k", [0, -1, 10001])
    async def test_top_k_bounds(self, index: AsyncIndex, top_k: int) -> None:
        with pytest.raises(PineconeValueError, match="top_k"):
            await index.documents.search(
                namespace=NS,
                top_k=top_k,
                score_by=[{"type": "query_string", "query": "rome"}],
            )

    async def test_invalid_clause_dict_rejected(self, index: AsyncIndex) -> None:
        with pytest.raises(PineconeValueError):
            await index.documents.search(
                namespace=NS, top_k=5, score_by=[{"type": "bogus", "query": "x"}]
            )


class TestFetchDocuments:
    @respx.mock
    async def test_by_ids_body(self, index: AsyncIndex) -> None:
        route = respx.post(f"{BASE_URL}/namespaces/{NS}/documents/fetch").mock(
            return_value=FETCH_OK
        )
        result = await index.documents.fetch(namespace=NS, ids=["doc-1"], include_fields=["title"])
        assert result.documents["doc-1"].title == "Rome"
        assert result.pagination is None
        assert _body(route) == {"ids": ["doc-1"], "include_fields": ["title"]}

    @respx.mock
    async def test_by_filter_with_pagination_token_body(self, index: AsyncIndex) -> None:
        route = respx.post(f"{BASE_URL}/namespaces/{NS}/documents/fetch").mock(
            return_value=FETCH_OK
        )
        await index.documents.fetch(
            namespace=NS,
            filter={"category": {"$eq": "tech"}},
            pagination_token="tok-1",
        )
        assert _body(route) == {
            "filter": {"category": {"$eq": "tech"}},
            "pagination_token": "tok-1",
        }

    async def test_both_ids_and_filter_rejected_naming_both(self, index: AsyncIndex) -> None:
        with pytest.raises(PineconeValueError, match="'ids' and 'filter'"):
            await index.documents.fetch(namespace=NS, ids=["a"], filter={"x": {"$eq": 1}})

    async def test_neither_ids_nor_filter_rejected_naming_both(self, index: AsyncIndex) -> None:
        with pytest.raises(PineconeValueError, match="No 'ids' or 'filter'"):
            await index.documents.fetch(namespace=NS)

    async def test_pagination_token_without_filter_rejected(self, index: AsyncIndex) -> None:
        with pytest.raises(PineconeValueError, match="only valid together with 'filter'"):
            await index.documents.fetch(namespace=NS, ids=["a"], pagination_token="tok-1")

    async def test_empty_filter_rejected_naming_the_rule(self, index: AsyncIndex) -> None:
        with pytest.raises(PineconeValueError, match="non-empty object of filter predicates"):
            await index.documents.fetch(namespace=NS, filter={})

    async def test_over_1000_ids_rejected(self, index: AsyncIndex) -> None:
        with pytest.raises(PineconeValueError, match="1000"):
            await index.documents.fetch(namespace=NS, ids=[f"doc-{i}" for i in range(1001)])


class TestDeleteDocuments:
    @respx.mock
    async def test_by_ids_body_and_absent_matched_records(self, index: AsyncIndex) -> None:
        route = respx.post(f"{BASE_URL}/namespaces/{NS}/documents/delete").mock(
            return_value=DELETE_OK
        )
        result = await index.documents.delete(namespace=NS, ids=["doc-1", "doc-2"])
        assert result.matched_records is None
        assert _body(route) == {"ids": ["doc-1", "doc-2"]}

    @respx.mock
    async def test_delete_all_body(self, index: AsyncIndex) -> None:
        route = respx.post(f"{BASE_URL}/namespaces/{NS}/documents/delete").mock(
            return_value=DELETE_OK
        )
        await index.documents.delete(namespace=NS, delete_all=True)
        assert _body(route) == {"delete_all": True}

    @respx.mock
    async def test_by_filter_returns_matched_records(self, index: AsyncIndex) -> None:
        route = respx.post(f"{BASE_URL}/namespaces/{NS}/documents/delete").mock(
            return_value=httpx.Response(202, json={"matched_records": 7})
        )
        result = await index.documents.delete(namespace=NS, filter={"category": {"$eq": "old"}})
        assert result.matched_records == 7
        assert _body(route) == {"filter": {"category": {"$eq": "old"}}}

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"ids": ["a"], "filter": {"x": {"$eq": 1}}},
            {"ids": ["a"], "delete_all": True},
            {"filter": {"x": {"$eq": 1}}, "delete_all": True},
            {"ids": ["a"], "filter": {"x": {"$eq": 1}}, "delete_all": True},
        ],
    )
    async def test_more_than_one_selector_rejected(
        self, index: AsyncIndex, kwargs: dict[str, Any]
    ) -> None:
        with pytest.raises(PineconeValueError, match="mutually exclusive"):
            await index.documents.delete(namespace=NS, **kwargs)

    async def test_no_selector_rejected_naming_all_three(self, index: AsyncIndex) -> None:
        with pytest.raises(PineconeValueError, match="'ids', 'filter', or 'delete_all'"):
            await index.documents.delete(namespace=NS)

    async def test_empty_filter_rejected(self, index: AsyncIndex) -> None:
        with pytest.raises(PineconeValueError, match="non-empty object of filter predicates"):
            await index.documents.delete(namespace=NS, filter={})


class TestUpdateDocuments:
    @respx.mock
    async def test_per_id_body_carries_patches_verbatim(self, index: AsyncIndex) -> None:
        route = respx.post(f"{BASE_URL}/namespaces/{NS}/documents/update").mock(
            return_value=UPDATE_OK
        )
        patches: list[Any] = [
            {"_id": "doc-1", "title": "New title", "year": 2027},
            {"_id": "doc-2", "_remove_fields": ["content"]},
        ]
        result = await index.documents.update(namespace=NS, documents=patches)
        assert result.matched_records is None
        assert _body(route) == {"documents": patches}

    @respx.mock
    async def test_accepts_update_document_record_instances(self, index: AsyncIndex) -> None:
        route = respx.post(f"{BASE_URL}/namespaces/{NS}/documents/update").mock(
            return_value=UPDATE_OK
        )
        await index.documents.update(
            namespace=NS,
            documents=[UpdateDocumentRecord({"_id": "doc-1", "_remove_fields": ["content"]})],
        )
        assert _body(route) == {"documents": [{"_id": "doc-1", "_remove_fields": ["content"]}]}

    @respx.mock
    async def test_by_filter_body_and_matched_records(self, index: AsyncIndex) -> None:
        route = respx.post(f"{BASE_URL}/namespaces/{NS}/documents/update").mock(
            return_value=httpx.Response(202, json={"matched_records": 42})
        )
        result = await index.documents.update(
            namespace=NS,
            filter={"category": {"$eq": "news"}},
            set_fields={"category": "archive"},
            remove_fields=["content"],
        )
        assert result.matched_records == 42
        assert _body(route) == {
            "filter": {"category": {"$eq": "news"}},
            "set_fields": {"category": "archive"},
            "remove_fields": ["content"],
        }

    @respx.mock
    async def test_namespace_path_segment_is_url_encoded(self, index: AsyncIndex) -> None:
        route = respx.post(f"{BASE_URL}/namespaces/live%20ns%2Fv1/documents/update").mock(
            return_value=UPDATE_OK
        )
        await index.documents.update(namespace="live ns/v1", documents=[{"_id": "doc-1", "a": 1}])
        assert route.called

    @respx.mock
    async def test_null_field_value_passes_through_to_the_server(self, index: AsyncIndex) -> None:
        route = respx.post(f"{BASE_URL}/namespaces/{NS}/documents/update").mock(
            return_value=UPDATE_OK
        )
        await index.documents.update(namespace=NS, documents=[{"_id": "doc-1", "title": None}])
        assert _body(route) == {"documents": [{"_id": "doc-1", "title": None}]}

    async def test_empty_documents_rejected(self, index: AsyncIndex) -> None:
        with pytest.raises(PineconeValueError, match="non-empty list"):
            await index.documents.update(namespace=NS, documents=[])

    async def test_missing_id_names_the_position(self, index: AsyncIndex) -> None:
        with pytest.raises(PineconeValueError, match="Document at position 1: Document '_id'"):
            await index.documents.update(
                namespace=NS, documents=[{"_id": "doc-1"}, {"title": "no id at all"}]
            )

    async def test_duplicate_id_names_the_position(self, index: AsyncIndex) -> None:
        with pytest.raises(PineconeValueError, match="position 1 has duplicate '_id' 'doc-1'"):
            await index.documents.update(
                namespace=NS, documents=[{"_id": "doc-1"}, {"_id": "doc-1", "a": 1}]
            )

    async def test_field_both_set_and_removed_rejected(self, index: AsyncIndex) -> None:
        with pytest.raises(PineconeValueError, match="both set and removed"):
            await index.documents.update(
                namespace=NS,
                documents=[{"_id": "doc-1", "title": "New", "_remove_fields": ["title"]}],
            )

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"documents": [{"_id": "a"}], "filter": {"x": {"$eq": 1}}},
            {"documents": [{"_id": "a"}], "set_fields": {"y": 1}},
            {"documents": [{"_id": "a"}], "remove_fields": ["y"]},
        ],
    )
    async def test_documents_with_by_filter_fields_rejected(
        self, index: AsyncIndex, kwargs: dict[str, Any]
    ) -> None:
        with pytest.raises(PineconeValueError, match="mutually exclusive"):
            await index.documents.update(namespace=NS, **kwargs)

    async def test_no_selector_rejected_naming_both_shapes(self, index: AsyncIndex) -> None:
        with pytest.raises(PineconeValueError, match="No 'documents' or 'filter' provided"):
            await index.documents.update(namespace=NS)

    async def test_patch_fields_without_filter_rejected(self, index: AsyncIndex) -> None:
        with pytest.raises(PineconeValueError, match="only valid together with 'filter'"):
            await index.documents.update(namespace=NS, set_fields={"category": "archive"})

    async def test_filter_without_a_patch_rejected(self, index: AsyncIndex) -> None:
        with pytest.raises(PineconeValueError, match="must change something"):
            await index.documents.update(namespace=NS, filter={"category": {"$eq": "news"}})


class TestListDocuments:
    @respx.mock
    async def test_optional_fields_absent_stay_off_the_wire(self, index: AsyncIndex) -> None:
        route = respx.post(f"{BASE_URL}/namespaces/{NS}/documents/list").mock(
            return_value=LIST_OK_LAST_PAGE
        )
        records = await index.documents.list(namespace=NS).to_list()
        assert [record.id for record in records] == ["doc-1"]
        assert _body(route) == {}

    @respx.mock
    async def test_prefix_and_limit_reach_the_body(self, index: AsyncIndex) -> None:
        route = respx.post(f"{BASE_URL}/namespaces/{NS}/documents/list").mock(
            return_value=LIST_OK_LAST_PAGE
        )
        await index.documents.list(namespace=NS, prefix="doc-", limit=20).to_list()
        assert _body(route) == {"prefix": "doc-", "limit": 20}

    @respx.mock
    async def test_pagination_is_followed_across_pages(self, index: AsyncIndex) -> None:
        route = respx.post(f"{BASE_URL}/namespaces/{NS}/documents/list").mock(
            side_effect=[
                httpx.Response(
                    200,
                    json={
                        "documents": [{"_id": "doc-1"}, {"_id": "doc-2"}],
                        "pagination": {"next": "doc-2"},
                        "namespace": NS,
                        "usage": {"read_units": 1},
                    },
                ),
                httpx.Response(
                    200,
                    json={
                        "documents": [{"_id": "doc-3"}],
                        "namespace": NS,
                        "usage": {"read_units": 1},
                    },
                ),
            ]
        )
        paginator = index.documents.list(namespace=NS, prefix="doc-")
        assert [record.id async for record in paginator] == ["doc-1", "doc-2", "doc-3"]
        assert route.call_count == 2
        assert json.loads(route.calls[1].request.content) == {
            "prefix": "doc-",
            "pagination_token": "doc-2",
        }
        assert paginator.pagination_token is None

    @respx.mock
    async def test_initial_pagination_token_resumes(self, index: AsyncIndex) -> None:
        route = respx.post(f"{BASE_URL}/namespaces/{NS}/documents/list").mock(
            return_value=LIST_OK_LAST_PAGE
        )
        await index.documents.list(namespace=NS, pagination_token="doc-9").to_list()
        assert _body(route) == {"pagination_token": "doc-9"}

    @respx.mock
    async def test_no_request_issued_until_the_paginator_is_iterated(
        self, index: AsyncIndex
    ) -> None:
        route = respx.post(f"{BASE_URL}/namespaces/{NS}/documents/list").mock(
            return_value=LIST_OK_LAST_PAGE
        )
        paginator = index.documents.list(namespace=NS)
        assert not route.called
        await paginator.to_list()
        assert route.call_count == 1

    @respx.mock
    async def test_namespace_path_segment_is_url_encoded(self, index: AsyncIndex) -> None:
        route = respx.post(f"{BASE_URL}/namespaces/live%20ns%2Fv1/documents/list").mock(
            return_value=LIST_OK_LAST_PAGE
        )
        await index.documents.list(namespace="live ns/v1").to_list()
        assert route.called

    @pytest.mark.parametrize("limit", [0, -1, 101])
    async def test_limit_bounds_rejected_eagerly(self, index: AsyncIndex, limit: int) -> None:
        with pytest.raises(PineconeValueError, match="'limit' must be between 1 and 100"):
            index.documents.list(namespace=NS, limit=limit)

    async def test_overlong_prefix_rejected_eagerly(self, index: AsyncIndex) -> None:
        with pytest.raises(PineconeValueError, match="maximum length of 512"):
            index.documents.list(namespace=NS, prefix="x" * 513)

    @pytest.mark.parametrize("namespace", ["", "   "])
    async def test_empty_namespace_rejected_eagerly(
        self, index: AsyncIndex, namespace: str
    ) -> None:
        with pytest.raises(PineconeValueError, match="non-empty string"):
            index.documents.list(namespace=namespace)


class TestBatchUpsertDocuments:
    @respx.mock
    async def test_aggregates_across_batches(self, index: AsyncIndex) -> None:
        route = respx.post(f"{BASE_URL}/namespaces/{NS}/documents/upsert").mock(
            return_value=httpx.Response(202, json={"upserted_count": 2})
        )
        docs = [{"_id": f"doc-{i}"} for i in range(6)]
        result = await index.documents.batch_upsert(
            namespace=NS, documents=docs, batch_size=2, show_progress=False
        )
        assert isinstance(result, BatchResult)
        assert result.total_item_count == 6
        assert result.successful_item_count == 6
        assert result.failed_item_count == 0
        assert result.total_batch_count == 3
        assert route.call_count == 3
        assert all(len(json.loads(call.request.content)["documents"]) == 2 for call in route.calls)

    @respx.mock
    async def test_partial_failure_captured_not_raised(self, index: AsyncIndex) -> None:
        responses = iter(
            [
                httpx.Response(202, json={"upserted_count": 2}),
                httpx.Response(400, json={"code": 3, "message": "bad batch"}),
                httpx.Response(202, json={"upserted_count": 2}),
            ]
        )
        respx.post(f"{BASE_URL}/namespaces/{NS}/documents/upsert").mock(
            side_effect=lambda request: next(responses)
        )
        docs = [{"_id": f"doc-{i}"} for i in range(6)]
        result = await index.documents.batch_upsert(
            namespace=NS, documents=docs, batch_size=2, max_concurrency=1, show_progress=False
        )
        assert result.failed_batch_count == 1
        assert result.successful_item_count == 4
        assert result.failed_item_count == 2
        assert len(result.errors) == 1
        assert result.failed_items == [{"_id": "doc-2"}, {"_id": "doc-3"}]

    @respx.mock
    async def test_accepts_more_than_1000_documents(self, index: AsyncIndex) -> None:
        route = respx.post(f"{BASE_URL}/namespaces/{NS}/documents/upsert").mock(
            return_value=httpx.Response(202, json={"upserted_count": 500})
        )
        docs = [{"_id": f"doc-{i}"} for i in range(1500)]
        result = await index.documents.batch_upsert(
            namespace=NS, documents=docs, batch_size=500, show_progress=False
        )
        assert result.total_item_count == 1500
        assert route.call_count == 3

    async def test_batch_size_capped_at_1000(self, index: AsyncIndex) -> None:
        with pytest.raises(PineconeValueError, match="batch_size must be between 1 and 1000"):
            await index.documents.batch_upsert(
                namespace=NS, documents=[{"_id": "a"}], batch_size=1001
            )

    async def test_max_concurrency_bounds(self, index: AsyncIndex) -> None:
        with pytest.raises(PineconeValueError, match="max_concurrency must be between 1 and 64"):
            await index.documents.batch_upsert(
                namespace=NS, documents=[{"_id": "a"}], max_concurrency=0
            )

    async def test_cross_batch_duplicate_ids_rejected_up_front(self, index: AsyncIndex) -> None:
        docs = [{"_id": "doc-1"}, {"_id": "doc-2"}, {"_id": "doc-1"}]
        with pytest.raises(PineconeValueError, match="duplicate '_id'"):
            await index.documents.batch_upsert(namespace=NS, documents=docs, batch_size=1)

    async def test_no_max_workers_hatch(self, index: AsyncIndex) -> None:
        with pytest.raises(TypeError, match="max_workers"):
            await index.documents.batch_upsert(
                namespace=NS,
                documents=[{"_id": "a"}],
                max_workers=8,  # type: ignore[call-arg]
            )

    async def test_batches_run_concurrently(
        self, index: AsyncIndex, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        n_batches = 3
        all_started = asyncio.Event()
        gate = asyncio.Event()
        started = 0

        async def _post(path: str, **kwargs: Any) -> httpx.Response:
            nonlocal started
            started += 1
            if started >= n_batches:
                all_started.set()
            await gate.wait()
            count = len(kwargs["json"]["documents"])
            return httpx.Response(202, json={"upserted_count": count})

        monkeypatch.setattr(index._http, "post", _post)
        docs = [{"_id": f"doc-{i}"} for i in range(n_batches)]
        task = asyncio.create_task(
            index.documents.batch_upsert(
                namespace=NS, documents=docs, batch_size=1, show_progress=False
            )
        )
        await asyncio.wait_for(all_started.wait(), timeout=5.0)
        gate.set()
        result = await asyncio.wait_for(task, timeout=5.0)
        assert result.successful_item_count == n_batches


class TestServerErrorSurfacing:
    @respx.mock
    async def test_400_body_text_surfaced_intact(self, index: AsyncIndex) -> None:
        server_message = "Invalid request: score_by clauses of type dense_vector must appear alone"
        respx.post(f"{BASE_URL}/namespaces/{NS}/documents/search").mock(
            return_value=httpx.Response(400, json={"code": 3, "message": server_message})
        )
        with pytest.raises(ApiError) as excinfo:
            await index.documents.search(
                namespace=NS,
                top_k=5,
                score_by=[{"type": "query_string", "query": "rome"}],
            )
        assert server_message in str(excinfo.value)


class TestKeywordOnly:
    async def test_positional_call_raises_actionable_error(self, index: AsyncIndex) -> None:
        with pytest.raises(PineconeValueError, match="keyword-only"):
            await index.documents.upsert(NS, [{"_id": "a"}])  # type: ignore[misc]


class TestHandleLifecycle:
    @respx.mock
    async def test_no_request_issued_before_first_data_plane_call(self) -> None:
        route = respx.post(f"{BASE_URL}/namespaces/{NS}/documents/upsert").mock(
            return_value=UPSERT_OK
        )
        client = AsyncIndex(host=INDEX_HOST, api_key="test-key")
        assert len(respx.calls) == 0
        await client.documents.upsert(namespace=NS, documents=[{"_id": "a"}])
        assert route.call_count == 1
        await client.close()

    async def test_close_is_idempotent_before_any_request(self) -> None:
        client = AsyncIndex(host=INDEX_HOST, api_key="test-key")
        await client.close()
        await client.close()

    @respx.mock
    async def test_close_is_idempotent_after_a_request(self) -> None:
        respx.post(f"{BASE_URL}/namespaces/{NS}/documents/delete").mock(return_value=DELETE_OK)
        client = AsyncIndex(host=INDEX_HOST, api_key="test-key")
        await client.documents.delete(namespace=NS, delete_all=True)
        await client.close()
        await client.close()
