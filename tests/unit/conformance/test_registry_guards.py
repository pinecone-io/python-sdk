"""The two schema-category methods must each refuse the other's operations.

Without this, an operation with a real response schema could satisfy the
mandatory round-trip category by claiming to have no body, and coverage would
stop meaning anything. These tests carry no ``@api_op`` — they guard the
registry, they do not claim an operation.
"""

from __future__ import annotations

import pytest

from pinecone.models.admin.project import ProjectModel
from pinecone.models.response_info import ResponseInfo
from pinecone.models.vectors.responses import UpsertRecordsResponse
from tests.unit.conformance import ClaimRecorder, ConformanceError, manifest_operations

_WITH_BODY = "admin:fetch_project"
_WITHOUT_BODY = "admin:delete_project"
_EMPTY_OBJECT_BODY = "db_data:deleteVectors"
_CLIENT_SIDE_RETURN = "db_data:upsertRecordsNamespace"


def test_manifest_records_success_body_from_the_spec() -> None:
    operations = manifest_operations()
    assert operations[_WITH_BODY]["success_body"] is True
    assert operations[_WITHOUT_BODY]["success_body"] is False
    assert operations[_EMPTY_OBJECT_BODY]["success_body"] is False


def test_roundtrip_refuses_an_operation_with_no_success_body() -> None:
    recorder = ClaimRecorder([_WITHOUT_BODY])
    with pytest.raises(ConformanceError, match="use assert_no_response_body"):
        recorder.assert_roundtrip(ProjectModel, {}, optional_absent=[])


def test_no_response_body_refuses_an_operation_with_a_success_body() -> None:
    recorder = ClaimRecorder([_WITH_BODY])
    with pytest.raises(ConformanceError, match="use assert_roundtrip"):
        recorder.assert_no_response_body(None)


def test_no_response_body_rejects_a_non_none_return() -> None:
    recorder = ClaimRecorder([_WITHOUT_BODY])
    with pytest.raises(ConformanceError, match="instead of None"):
        recorder.assert_no_response_body({"deleted": True})


def test_client_side_accounts_for_a_struct_the_sdk_builds_itself() -> None:
    recorder = ClaimRecorder([_CLIENT_SIDE_RETURN])
    recorder.assert_no_response_body(
        UpsertRecordsResponse(record_count=3), client_side=["record_count"]
    )


def test_client_side_rejects_a_populated_field_it_does_not_name() -> None:
    returned = UpsertRecordsResponse(record_count=3)
    returned.response_info = ResponseInfo()
    recorder = ClaimRecorder([_CLIENT_SIDE_RETURN])
    with pytest.raises(ConformanceError, match="client_side does not account for"):
        recorder.assert_no_response_body(returned, client_side=["record_count"])


def test_client_side_rejects_names_the_struct_does_not_have() -> None:
    recorder = ClaimRecorder([_CLIENT_SIDE_RETURN])
    with pytest.raises(ConformanceError, match="does not have"):
        recorder.assert_no_response_body(
            UpsertRecordsResponse(record_count=3), client_side=["upserted_count"]
        )
