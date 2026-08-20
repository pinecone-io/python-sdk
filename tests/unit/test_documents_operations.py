"""Behavior tests for the document operations on the sync Index (#132).

Covers the wire bodies each method emits, the client-side validation
vocabulary (mutual exclusions, ``_id`` contract, bounds), namespace
path-segment encoding, batch aggregation semantics, optional-field-absent
responses, and intact surfacing of server 400 text.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
import respx

from pinecone.errors.exceptions import ApiError, PineconeValueError
from pinecone.index import Index
from pinecone.models.batch import BatchResult
from pinecone.models.documents.document import DocumentRecord
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


@pytest.fixture
def index() -> Iterator[Index]:
    client = Index(host=INDEX_HOST, api_key="test-key")
    yield client
    client.close()


def _body(route: respx.Route) -> dict[str, Any]:
    return dict(json.loads(route.calls.last.request.content))


class TestUpsertDocumentsWire:
    @respx.mock
    def test_body_carries_documents_verbatim(self, index: Index) -> None:
        route = respx.post(f"{BASE_URL}/namespaces/{NS}/documents/upsert").mock(
            return_value=UPSERT_OK
        )
        docs = [{"_id": "doc-1", "title": "Rome", "year": 2026, "embedding": [0.1, 0.2]}]
        result = index.upsert_documents(namespace=NS, documents=docs)
        assert result.upserted_count == 1
        assert _body(route) == {"documents": docs}

    @respx.mock
    def test_accepts_document_record_instances(self, index: Index) -> None:
        route = respx.post(f"{BASE_URL}/namespaces/{NS}/documents/upsert").mock(
            return_value=UPSERT_OK
        )
        index.upsert_documents(
            namespace=NS, documents=[DocumentRecord({"_id": "doc-1", "title": "Rome"})]
        )
        assert _body(route) == {"documents": [{"_id": "doc-1", "title": "Rome"}]}

    @respx.mock
    def test_response_info_extracted(self, index: Index) -> None:
        respx.post(f"{BASE_URL}/namespaces/{NS}/documents/upsert").mock(
            return_value=httpx.Response(
                202,
                json={"upserted_count": 1},
                headers={"x-pinecone-request-id": "req-77"},
            )
        )
        result = index.upsert_documents(namespace=NS, documents=[{"_id": "a"}])
        assert result.response_info is not None
        assert result.response_info.request_id == "req-77"


class TestNamespaceHandling:
    @respx.mock
    def test_namespace_path_segment_is_url_encoded(self, index: Index) -> None:
        route = respx.post(
            f"{BASE_URL}/namespaces/my%20ns%2Fv1/documents/upsert",
        ).mock(return_value=UPSERT_OK)
        index.upsert_documents(namespace="my ns/v1", documents=[{"_id": "a"}])
        assert route.calls.last.request.url.raw_path.decode().startswith(
            "/namespaces/my%20ns%2Fv1/documents/upsert"
        )

    @pytest.mark.parametrize("namespace", ["", "   "])
    def test_empty_namespace_rejected(self, index: Index, namespace: str) -> None:
        with pytest.raises(PineconeValueError, match="namespace must be a non-empty string"):
            index.upsert_documents(namespace=namespace, documents=[{"_id": "a"}])

    def test_non_string_namespace_rejected(self, index: Index) -> None:
        with pytest.raises(PineconeValueError, match="namespace must be a string"):
            index.fetch_documents(namespace=7, ids=["a"])  # type: ignore[arg-type]


class TestUpsertDocumentsValidation:
    def test_empty_documents_rejected(self, index: Index) -> None:
        with pytest.raises(PineconeValueError, match="documents must be a non-empty list"):
            index.upsert_documents(namespace=NS, documents=[])

    def test_over_1000_documents_rejected(self, index: Index) -> None:
        docs = [{"_id": f"doc-{i}"} for i in range(1001)]
        with pytest.raises(PineconeValueError, match="1000"):
            index.upsert_documents(namespace=NS, documents=docs)

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
    def test_invalid_id_names_position(
        self, index: Index, bad_doc: dict[str, Any], fragment: str
    ) -> None:
        with pytest.raises(PineconeValueError, match="position 1") as excinfo:
            index.upsert_documents(namespace=NS, documents=[{"_id": "ok"}, bad_doc])
        assert fragment in str(excinfo.value)

    def test_duplicate_id_names_position(self, index: Index) -> None:
        with pytest.raises(PineconeValueError, match=r"position 2.*duplicate '_id' 'doc-1'"):
            index.upsert_documents(
                namespace=NS,
                documents=[{"_id": "doc-1"}, {"_id": "doc-2"}, {"_id": "doc-1"}],
            )

    def test_non_dict_document_rejected(self, index: Index) -> None:
        with pytest.raises(PineconeValueError, match=r"position 0.*got str"):
            index.upsert_documents(namespace=NS, documents=["doc-1"])  # type: ignore[list-item]


class TestSearchDocuments:
    @respx.mock
    def test_body_with_typed_score_by(self, index: Index) -> None:
        route = respx.post(f"{BASE_URL}/namespaces/{NS}/documents/search").mock(
            return_value=SEARCH_OK
        )
        result = index.search_documents(
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
    def test_optional_fields_absent_stay_off_the_wire(self, index: Index) -> None:
        route = respx.post(f"{BASE_URL}/namespaces/{NS}/documents/search").mock(
            return_value=SEARCH_OK
        )
        index.search_documents(
            namespace=NS,
            top_k=5,
            score_by=[{"type": "query_string", "query": "title:(rome)"}],
        )
        assert _body(route) == {
            "score_by": [{"type": "query_string", "query": "title:(rome)"}],
            "top_k": 5,
        }

    @respx.mock
    def test_empty_include_fields_sent_explicitly(self, index: Index) -> None:
        route = respx.post(f"{BASE_URL}/namespaces/{NS}/documents/search").mock(
            return_value=SEARCH_OK
        )
        index.search_documents(
            namespace=NS,
            top_k=5,
            score_by=[{"type": "query_string", "query": "rome"}],
            include_fields=[],
        )
        assert _body(route)["include_fields"] == []

    def test_empty_score_by_rejected(self, index: Index) -> None:
        with pytest.raises(PineconeValueError, match="at least one scoring method"):
            index.search_documents(namespace=NS, top_k=5, score_by=[])

    def test_over_100_score_by_clauses_rejected(self, index: Index) -> None:
        clauses = [{"type": "query_string", "query": f"q{i}"} for i in range(101)]
        with pytest.raises(PineconeValueError, match="100"):
            index.search_documents(namespace=NS, top_k=5, score_by=clauses)

    def test_vector_clause_must_appear_alone(self, index: Index) -> None:
        with pytest.raises(PineconeValueError, match="must appear alone"):
            index.search_documents(
                namespace=NS,
                top_k=5,
                score_by=[
                    DenseVectorQuery(field="embedding", values=[0.1]),
                    TextQuery(query="rome", fields=["content"]),
                ],
            )

    @pytest.mark.parametrize("top_k", [0, -1, 10001])
    def test_top_k_bounds(self, index: Index, top_k: int) -> None:
        with pytest.raises(PineconeValueError, match="top_k"):
            index.search_documents(
                namespace=NS,
                top_k=top_k,
                score_by=[{"type": "query_string", "query": "rome"}],
            )

    def test_invalid_clause_dict_rejected(self, index: Index) -> None:
        with pytest.raises(PineconeValueError):
            index.search_documents(
                namespace=NS, top_k=5, score_by=[{"type": "bogus", "query": "x"}]
            )


class TestFetchDocuments:
    @respx.mock
    def test_by_ids_body(self, index: Index) -> None:
        route = respx.post(f"{BASE_URL}/namespaces/{NS}/documents/fetch").mock(
            return_value=FETCH_OK
        )
        result = index.fetch_documents(namespace=NS, ids=["doc-1"], include_fields=["title"])
        assert result.documents["doc-1"].title == "Rome"
        assert result.pagination is None
        assert _body(route) == {"ids": ["doc-1"], "include_fields": ["title"]}

    @respx.mock
    def test_by_filter_with_pagination_token_body(self, index: Index) -> None:
        route = respx.post(f"{BASE_URL}/namespaces/{NS}/documents/fetch").mock(
            return_value=FETCH_OK
        )
        index.fetch_documents(
            namespace=NS,
            filter={"category": {"$eq": "tech"}},
            pagination_token="tok-1",
        )
        assert _body(route) == {
            "filter": {"category": {"$eq": "tech"}},
            "pagination_token": "tok-1",
        }

    def test_both_ids_and_filter_rejected_naming_both(self, index: Index) -> None:
        with pytest.raises(PineconeValueError, match="'ids' and 'filter'"):
            index.fetch_documents(namespace=NS, ids=["a"], filter={"x": {"$eq": 1}})

    def test_neither_ids_nor_filter_rejected_naming_both(self, index: Index) -> None:
        with pytest.raises(PineconeValueError, match="No 'ids' or 'filter'"):
            index.fetch_documents(namespace=NS)

    def test_pagination_token_without_filter_rejected(self, index: Index) -> None:
        with pytest.raises(PineconeValueError, match="only valid together with 'filter'"):
            index.fetch_documents(namespace=NS, ids=["a"], pagination_token="tok-1")

    def test_empty_filter_rejected_naming_the_rule(self, index: Index) -> None:
        with pytest.raises(PineconeValueError, match="non-empty object of filter predicates"):
            index.fetch_documents(namespace=NS, filter={})

    def test_over_1000_ids_rejected(self, index: Index) -> None:
        with pytest.raises(PineconeValueError, match="1000"):
            index.fetch_documents(namespace=NS, ids=[f"doc-{i}" for i in range(1001)])


class TestDeleteDocuments:
    @respx.mock
    def test_by_ids_body_and_absent_matched_records(self, index: Index) -> None:
        route = respx.post(f"{BASE_URL}/namespaces/{NS}/documents/delete").mock(
            return_value=DELETE_OK
        )
        result = index.delete_documents(namespace=NS, ids=["doc-1", "doc-2"])
        assert result.matched_records is None
        assert _body(route) == {"ids": ["doc-1", "doc-2"]}

    @respx.mock
    def test_delete_all_body(self, index: Index) -> None:
        route = respx.post(f"{BASE_URL}/namespaces/{NS}/documents/delete").mock(
            return_value=DELETE_OK
        )
        index.delete_documents(namespace=NS, delete_all=True)
        assert _body(route) == {"delete_all": True}

    @respx.mock
    def test_by_filter_returns_matched_records(self, index: Index) -> None:
        route = respx.post(f"{BASE_URL}/namespaces/{NS}/documents/delete").mock(
            return_value=httpx.Response(202, json={"matched_records": 7})
        )
        result = index.delete_documents(namespace=NS, filter={"category": {"$eq": "old"}})
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
    def test_more_than_one_selector_rejected(self, index: Index, kwargs: dict[str, Any]) -> None:
        with pytest.raises(PineconeValueError, match="mutually exclusive"):
            index.delete_documents(namespace=NS, **kwargs)

    def test_no_selector_rejected_naming_all_three(self, index: Index) -> None:
        with pytest.raises(PineconeValueError, match="'ids', 'filter', or 'delete_all'"):
            index.delete_documents(namespace=NS)

    def test_empty_filter_rejected(self, index: Index) -> None:
        with pytest.raises(PineconeValueError, match="non-empty object of filter predicates"):
            index.delete_documents(namespace=NS, filter={})


class TestBatchUpsertDocuments:
    @respx.mock
    def test_aggregates_across_batches(self, index: Index) -> None:
        route = respx.post(f"{BASE_URL}/namespaces/{NS}/documents/upsert").mock(
            return_value=httpx.Response(202, json={"upserted_count": 2})
        )
        docs = [{"_id": f"doc-{i}"} for i in range(6)]
        result = index.batch_upsert_documents(
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
    def test_partial_failure_captured_not_raised(self, index: Index) -> None:
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
        result = index.batch_upsert_documents(
            namespace=NS, documents=docs, batch_size=2, max_concurrency=1, show_progress=False
        )
        assert result.failed_batch_count == 1
        assert result.successful_item_count == 4
        assert result.failed_item_count == 2
        assert len(result.errors) == 1
        assert result.failed_items == [{"_id": "doc-2"}, {"_id": "doc-3"}]

    @respx.mock
    def test_accepts_more_than_1000_documents(self, index: Index) -> None:
        route = respx.post(f"{BASE_URL}/namespaces/{NS}/documents/upsert").mock(
            return_value=httpx.Response(202, json={"upserted_count": 500})
        )
        docs = [{"_id": f"doc-{i}"} for i in range(1500)]
        result = index.batch_upsert_documents(
            namespace=NS, documents=docs, batch_size=500, show_progress=False
        )
        assert result.total_item_count == 1500
        assert route.call_count == 3

    def test_batch_size_capped_at_1000(self, index: Index) -> None:
        with pytest.raises(PineconeValueError, match="batch_size must be between 1 and 1000"):
            index.batch_upsert_documents(namespace=NS, documents=[{"_id": "a"}], batch_size=1001)

    def test_max_concurrency_bounds(self, index: Index) -> None:
        with pytest.raises(PineconeValueError, match="max_concurrency must be between 1 and 64"):
            index.batch_upsert_documents(namespace=NS, documents=[{"_id": "a"}], max_concurrency=0)

    def test_cross_batch_duplicate_ids_rejected_up_front(self, index: Index) -> None:
        docs = [{"_id": "doc-1"}, {"_id": "doc-2"}, {"_id": "doc-1"}]
        with pytest.raises(PineconeValueError, match="duplicate '_id'"):
            index.batch_upsert_documents(namespace=NS, documents=docs, batch_size=1)

    def test_no_max_workers_hatch(self, index: Index) -> None:
        with pytest.raises(TypeError, match="max_workers"):
            index.batch_upsert_documents(
                namespace=NS,
                documents=[{"_id": "a"}],
                max_workers=8,  # type: ignore[call-arg]
            )

    @respx.mock
    def test_executor_reused_across_calls(self, index: Index) -> None:
        respx.post(f"{BASE_URL}/namespaces/{NS}/documents/upsert").mock(
            return_value=httpx.Response(202, json={"upserted_count": 1})
        )
        index.batch_upsert_documents(namespace=NS, documents=[{"_id": "a"}], show_progress=False)
        first = index._batch_executor
        index.batch_upsert_documents(namespace=NS, documents=[{"_id": "b"}], show_progress=False)
        assert index._batch_executor is first


class TestServerErrorSurfacing:
    @respx.mock
    def test_400_body_text_surfaced_intact(self, index: Index) -> None:
        server_message = "Invalid request: score_by clauses of type dense_vector must appear alone"
        respx.post(f"{BASE_URL}/namespaces/{NS}/documents/search").mock(
            return_value=httpx.Response(400, json={"code": 3, "message": server_message})
        )
        with pytest.raises(ApiError) as excinfo:
            index.search_documents(
                namespace=NS,
                top_k=5,
                score_by=[{"type": "query_string", "query": "rome"}],
            )
        assert server_message in str(excinfo.value)


class TestKeywordOnly:
    def test_positional_call_raises_actionable_error(self, index: Index) -> None:
        with pytest.raises(PineconeValueError, match="keyword-only"):
            index.upsert_documents(NS, [{"_id": "a"}])  # type: ignore[misc]
