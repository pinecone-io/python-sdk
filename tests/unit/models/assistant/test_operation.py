"""Wire-format decode/round-trip tests for OperationModel and ListOperationsResponse."""

from __future__ import annotations

import json
from typing import Any

import msgspec
import pytest

from pinecone.models.assistant.list import ListOperationsResponse, _Pagination
from pinecone.models.assistant.operation import OperationModel

FULL_OPERATION: dict[str, Any] = {
    "id": "op-1234-abcd-5678",
    "operation_type": "upload_file",
    "file_id": "my-file-id-123",
    "status": "Completed",
    "created_on": "2025-10-01T12:30:00Z",
    "completed_on": "2025-10-01T12:35:00Z",
    "percent_complete": 100,
    "error_message": None,
    "ingestion_units": 50.0,
}

MINIMAL_OPERATION: dict[str, Any] = {"id": "op-1234-abcd-5678", "status": "Processing"}


def decode_operation(payload: dict[str, Any]) -> OperationModel:
    return msgspec.json.decode(json.dumps(payload).encode(), type=OperationModel)


def decode_list(payload: dict[str, Any]) -> ListOperationsResponse:
    return msgspec.json.decode(json.dumps(payload).encode(), type=ListOperationsResponse)


class TestOperationModelDecode:
    def test_full_payload_maps_every_field(self) -> None:
        op = decode_operation(FULL_OPERATION)
        assert op.operation_id == "op-1234-abcd-5678"
        assert op.operation_type == "upload_file"
        assert op.file_id == "my-file-id-123"
        assert op.status == "Completed"
        assert op.created_at == "2025-10-01T12:30:00Z"
        assert op.completed_on == "2025-10-01T12:35:00Z"
        assert op.percent_complete == 100
        assert op.error is None
        assert op.ingestion_units == 50.0

    def test_minimal_2026_04_body_still_parses(self) -> None:
        op = decode_operation(MINIMAL_OPERATION)
        assert op.operation_id == "op-1234-abcd-5678"
        assert op.status == "Processing"
        assert op.operation_type is None
        assert op.file_id is None
        assert op.created_at is None
        assert op.completed_on is None
        assert op.percent_complete is None
        assert op.error is None
        assert op.ingestion_units is None

    @pytest.mark.parametrize(
        "field",
        [
            "operation_type",
            "file_id",
            "created_on",
            "completed_on",
            "percent_complete",
            "error_message",
            "ingestion_units",
        ],
    )
    def test_each_optional_field_may_be_absent(self, field: str) -> None:
        payload = {k: v for k, v in FULL_OPERATION.items() if k != field}
        assert decode_operation(payload) is not None

    @pytest.mark.parametrize(
        "wire_key, attribute",
        [
            ("file_id", "file_id"),
            ("completed_on", "completed_on"),
            ("error_message", "error"),
            ("ingestion_units", "ingestion_units"),
        ],
    )
    def test_explicit_null_decodes_to_none(self, wire_key: str, attribute: str) -> None:
        payload = {**FULL_OPERATION, wire_key: None}
        assert getattr(decode_operation(payload), attribute) is None

    def test_error_message_populated_on_failure(self) -> None:
        payload = {
            **FULL_OPERATION,
            "status": "Failed",
            "percent_complete": 0,
            "ingestion_units": None,
            "error_message": "File processing failed: unsupported file format.",
        }
        op = decode_operation(payload)
        assert op.status == "Failed"
        assert op.error == "File processing failed: unsupported file format."

    def test_integer_ingestion_units_widen_to_float(self) -> None:
        op = decode_operation({**FULL_OPERATION, "ingestion_units": 3})
        assert op.ingestion_units == pytest.approx(3.0)

    def test_missing_required_field_raises(self) -> None:
        with pytest.raises(msgspec.ValidationError):
            decode_operation({"id": "op-1"})

    def test_encode_restores_wire_names(self) -> None:
        payload = json.loads(msgspec.json.encode(decode_operation(FULL_OPERATION)))
        assert payload == FULL_OPERATION

    def test_round_trip_of_minimal_body_omits_nothing_it_did_not_have(self) -> None:
        payload = json.loads(msgspec.json.encode(decode_operation(MINIMAL_OPERATION)))
        assert payload["id"] == "op-1234-abcd-5678"
        assert payload["status"] == "Processing"
        assert all(payload[key] is None for key in payload if key not in {"id", "status"})


class TestOperationModelDictLike:
    def test_dict_like_access(self) -> None:
        op = decode_operation(FULL_OPERATION)
        assert op["operation_type"] == "upload_file"
        assert "percent_complete" in op
        assert op.get("nope", "fallback") == "fallback"


class TestListOperationsResponseDecode:
    def test_full_page_decodes(self) -> None:
        resp = decode_list({"operations": [FULL_OPERATION], "pagination": {"next": "tok"}})
        assert len(resp.operations) == 1
        assert isinstance(resp.operations[0], OperationModel)
        assert resp.operations[0].operation_id == "op-1234-abcd-5678"
        assert resp.next == "tok"
        assert resp.next_token == "tok"

    def test_pagination_absent_means_exhausted(self) -> None:
        resp = decode_list({"operations": [MINIMAL_OPERATION]})
        assert resp.pagination is None
        assert resp.next is None
        assert resp.next_token is None

    def test_empty_page(self) -> None:
        resp = decode_list({"operations": []})
        assert resp.operations == []
        assert resp.next is None

    def test_operations_absent_raises(self) -> None:
        with pytest.raises(msgspec.ValidationError):
            decode_list({"pagination": {"next": "tok"}})

    def test_mixed_full_and_minimal_operations(self) -> None:
        resp = decode_list({"operations": [FULL_OPERATION, MINIMAL_OPERATION]})
        assert resp.operations[0].ingestion_units == 50.0
        assert resp.operations[1].ingestion_units is None

    def test_next_token_alias_matches_next(self) -> None:
        resp = ListOperationsResponse(operations=[], pagination=_Pagination(next="tok-2"))
        assert resp.next_token == resp.next == "tok-2"

    def test_encode_restores_wire_shape(self) -> None:
        resp = decode_list({"operations": [FULL_OPERATION], "pagination": {"next": "tok"}})
        assert json.loads(msgspec.json.encode(resp)) == {
            "operations": [FULL_OPERATION],
            "pagination": {"next": "tok"},
        }
