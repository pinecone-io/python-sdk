"""The two schema-category methods must each refuse the other's operations.

Without this, an operation with a real response schema could satisfy the
mandatory round-trip category by claiming to have no body, and coverage would
stop meaning anything. These tests carry no ``@api_op`` — they guard the
registry, they do not claim an operation.
"""

from __future__ import annotations

import pytest

from pinecone.models.admin.project import ProjectModel
from tests.unit.conformance import ClaimRecorder, ConformanceError, manifest_operations

_WITH_BODY = "admin:fetch_project"
_WITHOUT_BODY = "admin:delete_project"


def test_manifest_records_success_body_from_the_spec() -> None:
    operations = manifest_operations()
    assert operations[_WITH_BODY]["success_body"] is True
    assert operations[_WITHOUT_BODY]["success_body"] is False


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
