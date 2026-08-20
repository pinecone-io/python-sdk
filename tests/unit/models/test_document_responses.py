"""Unit tests for document response envelopes (2026-07).

Wire fixtures follow the response examples in
``apis/_build/2026-07/db_data_2026-07.oas.yaml`` (SearchDocumentsResponse
:3089, FetchDocumentsResponse :3182, DeleteDocumentsResponse :3234,
UpdateDocumentsResponse :3319, ListDocumentsResponse :3375,
UpsertDocumentsResponse :2918).
"""

from __future__ import annotations

import msgspec
import orjson

from pinecone.models.documents import (
    DeleteDocumentsResponse,
    Document,
    DocumentFetchUsage,
    DocumentSearchUsage,
    FetchDocumentsResponse,
    ListDocumentsResponse,
    ListedDocumentRecord,
    SearchDocumentsResponse,
    UpdateDocumentsResponse,
    UpsertDocumentsResponse,
)
from pinecone.models.response_info import ResponseInfo


def _info() -> ResponseInfo:
    return ResponseInfo(
        raw_headers={
            "x-pinecone-request-id": "req-1",
            "x-pinecone-lsn-reconciled": "42",
            "x-pinecone-lsn-committed": "50",
        }
    )


def test_upsert_response_spec_example_round_trip() -> None:
    decoded = msgspec.json.decode(b'{"upserted_count": 2}', type=UpsertDocumentsResponse)
    assert decoded.upserted_count == 2
    assert decoded.response_info is None


def test_upsert_response_carries_response_info() -> None:
    r = UpsertDocumentsResponse(upserted_count=3, response_info=_info())
    assert r.response_info is not None
    assert r.response_info.request_id == "req-1"


def test_delete_response_matched_records_for_filtered_delete() -> None:
    decoded = msgspec.json.decode(b'{"matched_records": 42}', type=DeleteDocumentsResponse)
    assert decoded.matched_records == 42


def test_delete_response_matched_records_absent_is_none() -> None:
    decoded = msgspec.json.decode(b"{}", type=DeleteDocumentsResponse)
    assert decoded.matched_records is None


def test_delete_response_matched_records_zero_means_no_matches() -> None:
    decoded = msgspec.json.decode(b'{"matched_records": 0}', type=DeleteDocumentsResponse)
    assert decoded.matched_records == 0


def test_update_response_matched_records_for_filtered_update() -> None:
    decoded = msgspec.json.decode(b'{"matched_records": 7}', type=UpdateDocumentsResponse)
    assert decoded.matched_records == 7


def test_update_response_matched_records_absent_is_none() -> None:
    decoded = msgspec.json.decode(b"{}", type=UpdateDocumentsResponse)
    assert decoded.matched_records is None


def test_search_response_from_dict_spec_example() -> None:
    payload = {
        "matches": [
            {
                "_id": "doc-1",
                "_score": 0.9281134605407715,
                "title": "Introduction to Machine Learning",
            }
        ],
        "namespace": "my-namespace",
        "usage": {"read_units": 5},
    }
    response = SearchDocumentsResponse.from_dict(payload)
    assert response.namespace == "my-namespace"
    assert response.usage == DocumentSearchUsage(read_units=5)
    assert len(response.matches) == 1
    assert response.matches[0].id == "doc-1"
    assert response.matches[0].score == 0.9281134605407715
    assert response.to_dict() == payload


def test_search_response_usage_absent_is_none() -> None:
    response = SearchDocumentsResponse.from_dict(
        {"matches": [{"_id": "doc-1", "_score": 0.5}], "namespace": "ns"}
    )
    assert response.usage is None


def test_search_response_unknown_match_fields_survive_round_trip() -> None:
    payload = orjson.dumps(
        {
            "matches": [{"_id": "doc-1", "_score": 0.8, "title": "Rome", "future": {"x": 1}}],
            "namespace": "ns",
            "usage": {"read_units": 1},
        }
    )
    response = SearchDocumentsResponse.from_dict(orjson.loads(payload))
    match = response.matches[0]
    assert match.title == "Rome"  # type: ignore[attr-defined]
    assert match.get("future") == {"x": 1}
    assert orjson.loads(orjson.dumps(response.to_dict())) == orjson.loads(payload)


def test_search_response_carries_response_info() -> None:
    response = SearchDocumentsResponse.from_dict(
        {"matches": [], "namespace": "ns"}, response_info=_info()
    )
    assert response.response_info is not None
    assert response.response_info.lsn_reconciled == 42
    assert response.response_info.lsn_committed == 50


def test_search_response_repr_and_html() -> None:
    response = SearchDocumentsResponse(
        matches=[Document({"_id": "doc-1", "_score": 0.9})],
        namespace="wiki",
        usage=DocumentSearchUsage(read_units=3),
    )
    assert repr(response).startswith("SearchDocumentsResponse(matches=1")
    assert "<table" in response._repr_html_()


def test_fetch_response_from_dict_populates_documents_keyed_by_id() -> None:
    response = FetchDocumentsResponse.from_dict(
        {
            "documents": {
                "doc-1": {"_id": "doc-1", "title": "Rome"},
                "doc-2": {"_id": "doc-2", "title": "Athens"},
            },
            "namespace": "wiki",
            "usage": {"read_units": 2},
        }
    )
    assert set(response.documents.keys()) == {"doc-1", "doc-2"}
    assert isinstance(response.documents["doc-1"], Document)
    assert response.documents["doc-1"].id == "doc-1"
    assert response.usage == DocumentFetchUsage(read_units=2)
    assert response.pagination is None


def test_fetch_response_missing_ids_omitted_from_map() -> None:
    response = FetchDocumentsResponse.from_dict(
        {"documents": {"doc-1": {"_id": "doc-1"}}, "namespace": "ns"}
    )
    assert "doc-99999" not in response.documents
    assert "doc-1" in response.documents
    assert response.usage is None


def test_fetch_response_models_pagination_token() -> None:
    payload = {
        "documents": {"doc-1": {"_id": "doc-1"}},
        "pagination": {"next": "Tm90aGluZyB0byBzZWUgaGVyZQo="},
        "namespace": "ns",
        "usage": {"read_units": 5},
    }
    response = FetchDocumentsResponse.from_dict(payload)
    assert response.pagination is not None
    assert response.pagination.next == "Tm90aGluZyB0byBzZWUgaGVyZQo="
    assert response.to_dict() == payload


def test_fetch_response_carries_response_info() -> None:
    response = FetchDocumentsResponse.from_dict(
        {"documents": {}, "namespace": "ns"}, response_info=_info()
    )
    assert response.response_info is not None
    assert response.response_info.lsn_reconciled == 42


def test_fetch_response_repr() -> None:
    response = FetchDocumentsResponse(documents={}, namespace="ns")
    assert repr(response).startswith("FetchDocumentsResponse(documents=0")


def test_listed_document_record_uses_underscore_id_on_the_wire() -> None:
    record = msgspec.json.decode(b'{"_id": "doc-1"}', type=ListedDocumentRecord)
    assert record.id == "doc-1"
    assert record._id == "doc-1"
    assert msgspec.json.encode(record) == b'{"_id":"doc-1"}'


def test_list_response_decode_round_trip() -> None:
    payload = orjson.dumps(
        {
            "documents": [{"_id": "doc-1"}, {"_id": "doc-2"}],
            "pagination": {"next": "token-1"},
            "namespace": "my-namespace",
            "usage": {"read_units": 1},
        }
    )
    response = msgspec.json.decode(payload, type=ListDocumentsResponse)
    assert [d.id for d in response.documents] == ["doc-1", "doc-2"]
    assert response.pagination is not None
    assert response.pagination.next == "token-1"
    assert response.namespace == "my-namespace"
    assert response.usage.read_units == 1


def test_list_response_pagination_absent_is_none() -> None:
    payload = orjson.dumps(
        {"documents": [{"_id": "doc-1"}], "namespace": "ns", "usage": {"read_units": 1}}
    )
    response = msgspec.json.decode(payload, type=ListDocumentsResponse)
    assert response.pagination is None
    assert response.response_info is None
