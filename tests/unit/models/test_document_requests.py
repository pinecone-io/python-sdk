"""Unit tests for document request envelopes (2026-07)."""

from __future__ import annotations

import msgspec
import orjson
import pytest

from pinecone.models.documents import (
    DeleteDocumentsRequest,
    DocumentRecord,
    FetchDocumentsRequest,
    ListDocumentsRequest,
    SearchDocumentsRequest,
    TextQuery,
    UpdateDocumentRecord,
    UpdateDocumentsRequest,
    UpsertDocumentsRequest,
)


def test_upsert_request_wire_shape() -> None:
    request = UpsertDocumentsRequest(
        documents=[
            {"_id": "doc-1", "title": "Rome"},
            DocumentRecord(_id="doc-2", title="Athens"),
        ]
    )
    assert orjson.loads(msgspec.json.encode(request)) == {
        "documents": [
            {"_id": "doc-1", "title": "Rome"},
            {"_id": "doc-2", "title": "Athens"},
        ]
    }


def test_upsert_request_validates_each_document_id() -> None:
    with pytest.raises(ValueError, match="'_id'"):
        UpsertDocumentsRequest(documents=[{"title": "no id"}])


def test_upsert_request_rejects_empty_documents() -> None:
    with pytest.raises(ValueError, match="at least one document"):
        UpsertDocumentsRequest(documents=[])


def test_upsert_request_rejects_more_than_1000_documents() -> None:
    docs = [{"_id": f"doc-{i}"} for i in range(1001)]
    with pytest.raises(ValueError, match="maximum limit of 1000 documents"):
        UpsertDocumentsRequest(documents=docs)


def test_search_request_wire_shape() -> None:
    request = SearchDocumentsRequest(
        score_by=[TextQuery(fields=["content"], query="What is machine learning?")],
        top_k=10,
        include_fields=["title", "content"],
    )
    assert orjson.loads(msgspec.json.encode(request)) == {
        "score_by": [{"type": "text", "fields": ["content"], "query": "What is machine learning?"}],
        "top_k": 10,
        "include_fields": ["title", "content"],
    }


def test_search_request_optional_fields_stay_off_the_wire() -> None:
    request = SearchDocumentsRequest(score_by=[TextQuery(fields=["content"], query="q")], top_k=5)
    encoded = orjson.loads(msgspec.json.encode(request))
    assert "include_fields" not in encoded
    assert "filter" not in encoded


def test_search_request_converts_dict_clauses_to_typed_variants() -> None:
    request = SearchDocumentsRequest(
        score_by=[{"type": "text", "fields": ["content"], "query": "q"}], top_k=5
    )
    assert isinstance(request.score_by[0], TextQuery)


def test_search_request_rejects_empty_score_by() -> None:
    with pytest.raises(ValueError, match="at least one scoring method"):
        SearchDocumentsRequest(score_by=[], top_k=5)


def test_search_request_rejects_more_than_100_scoring_methods() -> None:
    clauses = [{"type": "text", "fields": ["f"], "query": "q"}] * 101
    with pytest.raises(ValueError, match="maximum limit of 100 methods"):
        SearchDocumentsRequest(score_by=clauses, top_k=5)


def test_search_request_rejects_vector_clause_combined_with_others() -> None:
    clauses = [
        {"type": "dense_vector", "field": "emb", "values": [0.1]},
        {"type": "text", "fields": ["f"], "query": "q"},
    ]
    with pytest.raises(ValueError, match=r"must appear alone in.*score_by"):
        SearchDocumentsRequest(score_by=clauses, top_k=5)


def test_search_request_allows_combined_text_and_query_string() -> None:
    request = SearchDocumentsRequest(
        score_by=[
            {"type": "text", "fields": ["f"], "query": "q"},
            {"type": "query_string", "query": "a AND b"},
        ],
        top_k=5,
    )
    assert len(request.score_by) == 2


def test_search_request_rejects_out_of_range_top_k() -> None:
    clause = {"type": "text", "fields": ["f"], "query": "q"}
    with pytest.raises(ValueError, match="top_k"):
        SearchDocumentsRequest(score_by=[clause], top_k=0)
    with pytest.raises(ValueError, match="top_k"):
        SearchDocumentsRequest(score_by=[dict(clause)], top_k=10001)


def test_fetch_request_by_ids_wire_shape() -> None:
    request = FetchDocumentsRequest(ids=["doc-1", "doc-2"])
    assert orjson.loads(msgspec.json.encode(request)) == {"ids": ["doc-1", "doc-2"]}


def test_fetch_request_by_filter_with_pagination() -> None:
    request = FetchDocumentsRequest(filter={"category": {"$eq": "news"}}, pagination_token="tok")
    assert orjson.loads(msgspec.json.encode(request)) == {
        "filter": {"category": {"$eq": "news"}},
        "pagination_token": "tok",
    }


def test_fetch_request_rejects_limit() -> None:
    with pytest.raises(TypeError, match="Unexpected keyword argument 'limit'"):
        FetchDocumentsRequest(filter={"category": {"$eq": "news"}}, limit=50)  # type: ignore[call-arg]


def test_fetch_request_never_serializes_limit() -> None:
    assert "limit" not in FetchDocumentsRequest.__struct_fields__
    by_filter = FetchDocumentsRequest(filter={"category": {"$eq": "news"}}, pagination_token="tok")
    by_ids = FetchDocumentsRequest(ids=["doc-1"], include_fields=["title"])
    for request in (by_filter, by_ids):
        assert "limit" not in orjson.loads(msgspec.json.encode(request))


def test_fetch_request_docstring_does_not_claim_a_fixed_page_size() -> None:
    doc = " ".join((FetchDocumentsRequest.__doc__ or "").split())
    assert "the server chooses the page size" in doc
    assert "fixed" not in doc
    assert "10000" not in doc


def test_fetch_request_rejects_ids_with_filter() -> None:
    with pytest.raises(ValueError, match="'ids' and 'filter' fields are mutually exclusive"):
        FetchDocumentsRequest(ids=["doc-1"], filter={"a": {"$eq": 1}})


def test_fetch_request_rejects_neither_ids_nor_filter() -> None:
    with pytest.raises(ValueError, match="No 'ids' or 'filter' provided"):
        FetchDocumentsRequest()


def test_fetch_request_rejects_pagination_token_without_filter() -> None:
    with pytest.raises(ValueError, match="only valid together with 'filter'"):
        FetchDocumentsRequest(ids=["doc-1"], pagination_token="tok")


def test_fetch_request_rejects_empty_filter() -> None:
    with pytest.raises(ValueError, match="non-empty object of filter predicates"):
        FetchDocumentsRequest(filter={})


def test_fetch_request_rejects_more_than_1000_ids() -> None:
    with pytest.raises(ValueError, match="maximum limit of 1000 ids"):
        FetchDocumentsRequest(ids=[f"doc-{i}" for i in range(1001)])


def test_delete_request_by_ids_wire_shape() -> None:
    request = DeleteDocumentsRequest(ids=["doc-1"])
    assert orjson.loads(msgspec.json.encode(request)) == {"ids": ["doc-1"]}


def test_delete_request_delete_all_wire_shape() -> None:
    request = DeleteDocumentsRequest(delete_all=True)
    assert orjson.loads(msgspec.json.encode(request)) == {"delete_all": True}


def test_delete_request_rejects_filter_with_ids() -> None:
    with pytest.raises(ValueError, match="mutually exclusive with 'ids' and 'delete_all'"):
        DeleteDocumentsRequest(ids=["doc-1"], filter={"a": {"$eq": 1}})


def test_delete_request_rejects_filter_with_delete_all() -> None:
    with pytest.raises(ValueError, match="mutually exclusive with 'ids' and 'delete_all'"):
        DeleteDocumentsRequest(delete_all=True, filter={"a": {"$eq": 1}})


def test_delete_request_rejects_ids_with_delete_all() -> None:
    with pytest.raises(ValueError, match="'ids' and 'delete_all' fields are mutually exclusive"):
        DeleteDocumentsRequest(ids=["doc-1"], delete_all=True)


def test_delete_request_rejects_no_selector() -> None:
    with pytest.raises(ValueError, match="No 'ids', 'filter', or 'delete_all' provided"):
        DeleteDocumentsRequest()


def test_delete_request_rejects_empty_filter() -> None:
    with pytest.raises(ValueError, match="non-empty object of filter predicates"):
        DeleteDocumentsRequest(filter={})


def test_update_request_per_id_wire_shape() -> None:
    request = UpdateDocumentsRequest(
        documents=[
            {"_id": "doc-1", "title": "Updated title", "_remove_fields": ["content"]},
            UpdateDocumentRecord(_id="doc-2", category="archive"),
        ]
    )
    assert orjson.loads(msgspec.json.encode(request)) == {
        "documents": [
            {"_id": "doc-1", "title": "Updated title", "_remove_fields": ["content"]},
            {"_id": "doc-2", "category": "archive"},
        ]
    }


def test_update_request_by_filter_wire_shape() -> None:
    request = UpdateDocumentsRequest(
        filter={"category": {"$eq": "news"}},
        set_fields={"category": "archive"},
        remove_fields=["content"],
    )
    assert orjson.loads(msgspec.json.encode(request)) == {
        "filter": {"category": {"$eq": "news"}},
        "set_fields": {"category": "archive"},
        "remove_fields": ["content"],
    }


def test_update_request_rejects_documents_with_filter() -> None:
    with pytest.raises(ValueError, match="mutually exclusive with the by-filter fields"):
        UpdateDocumentsRequest(documents=[{"_id": "doc-1", "a": 1}], filter={"b": {"$eq": 2}})


def test_update_request_rejects_documents_with_set_fields() -> None:
    with pytest.raises(ValueError, match="mutually exclusive with the by-filter fields"):
        UpdateDocumentsRequest(documents=[{"_id": "doc-1", "a": 1}], set_fields={"b": 2})


def test_update_request_empty_patch_fields_are_ignored() -> None:
    request = UpdateDocumentsRequest(
        documents=[{"_id": "doc-1", "a": 1}], set_fields={}, remove_fields=[]
    )
    assert request.documents is not None


def test_update_request_rejects_patch_fields_without_filter() -> None:
    with pytest.raises(ValueError, match="only valid together with 'filter'"):
        UpdateDocumentsRequest(set_fields={"a": 1})


def test_update_request_rejects_filter_without_patch_fields() -> None:
    with pytest.raises(ValueError, match="must change something"):
        UpdateDocumentsRequest(filter={"a": {"$eq": 1}})


def test_update_request_rejects_neither_documents_nor_filter() -> None:
    with pytest.raises(ValueError, match="No 'documents' or 'filter' provided"):
        UpdateDocumentsRequest()


def test_update_request_rejects_empty_documents() -> None:
    with pytest.raises(ValueError, match="at least one document update"):
        UpdateDocumentsRequest(documents=[])


def test_update_request_validates_each_document() -> None:
    with pytest.raises(ValueError, match="both set and removed"):
        UpdateDocumentsRequest(
            documents=[{"_id": "doc-1", "title": "x", "_remove_fields": ["title"]}]
        )


def test_list_request_wire_shape() -> None:
    request = ListDocumentsRequest(prefix="doc-", limit=20)
    assert orjson.loads(msgspec.json.encode(request)) == {"prefix": "doc-", "limit": 20}


def test_list_request_empty_is_valid() -> None:
    assert orjson.loads(msgspec.json.encode(ListDocumentsRequest())) == {}


def test_list_request_rejects_out_of_range_limit() -> None:
    with pytest.raises(ValueError, match="between 1 and 100"):
        ListDocumentsRequest(limit=0)
    with pytest.raises(ValueError, match="between 1 and 100"):
        ListDocumentsRequest(limit=101)


def test_list_request_rejects_non_ascii_prefix() -> None:
    with pytest.raises(ValueError, match="ASCII"):
        ListDocumentsRequest(prefix="docü")


def test_list_request_rejects_overlong_prefix() -> None:
    with pytest.raises(ValueError, match="maximum length of 512"):
        ListDocumentsRequest(prefix="x" * 513)
