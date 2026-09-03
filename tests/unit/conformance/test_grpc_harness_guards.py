"""Guards keeping the gRPC harness and the manifest's rpc entries honest.

The manifest's grpc entries come from a line-oriented parse of the vendored
proto in ``scripts/api_coverage.py``; the harness's stubs come from protoc, a
full proto compiler. Cross-checking one against the other means neither can
drift silently: a manifest regenerated against a changed proto, a stale
vendored copy, or a parser bug that misreads a message all fail here. The
registry guards mirror ``test_registry_guards.py`` for the gRPC arm of the
schema category: a bodyless rpc must not round-trip an invented model, and a
body-carrying rpc must not dodge the round-trip.
"""

from __future__ import annotations

import pytest

from pinecone.models.vectors.responses import UpsertRecordsResponse, UpsertResponse
from tests.unit.conformance import ClaimRecorder, ConformanceError, manifest_operations
from tests.unit.conformance._grpc_harness import (
    SERVICE_NAME,
    VENDORED_PROTO,
    generated_modules,
)

_GRPC_WITH_BODY = "db_data_grpc:Upsert"
_GRPC_WITHOUT_BODY = "db_data_grpc:Delete"


def _grpc_entries() -> dict[str, dict[str, object]]:
    return {
        op_id: entry for op_id, entry in manifest_operations().items() if entry["kind"] == "grpc"
    }


def test_generated_stubs_compile_the_vendored_proto_bytes() -> None:
    _, _, proto_copy = generated_modules()
    assert proto_copy.read_bytes() == VENDORED_PROTO.read_bytes()


def test_generated_service_matches_the_manifest_rpc_set() -> None:
    pb2, _, _ = generated_modules()
    service = pb2.DESCRIPTOR.services_by_name[SERVICE_NAME]
    generated = {method.name for method in service.methods}
    manifest = {str(entry["rpc"]) for entry in _grpc_entries().values()}
    assert generated == manifest
    assert len(manifest) == 12


def test_manifest_rpc_entries_match_the_protoc_descriptor() -> None:
    pb2, _, _ = generated_modules()
    service = pb2.DESCRIPTOR.services_by_name[SERVICE_NAME]
    for op_id, entry in sorted(_grpc_entries().items()):
        method = service.methods_by_name[str(entry["rpc"])]
        assert entry["service"] == SERVICE_NAME, op_id
        assert entry["request"] == method.input_type.name, op_id
        assert entry["response"] == method.output_type.name, op_id
        assert entry["success_body"] is bool(method.output_type.fields), op_id


def test_manifest_records_success_body_from_the_proto() -> None:
    entries = _grpc_entries()
    assert entries[_GRPC_WITH_BODY]["success_body"] is True
    assert entries[_GRPC_WITHOUT_BODY]["success_body"] is False
    assert entries["db_data_grpc:DeleteNamespace"]["success_body"] is False


def test_grpc_roundtrip_refuses_a_bodyless_rpc() -> None:
    recorder = ClaimRecorder([_GRPC_WITHOUT_BODY])
    with pytest.raises(ConformanceError, match="use assert_no_response_body"):
        recorder.assert_roundtrip(UpsertResponse, {"upsertedCount": 0}, optional_absent=[])


def test_grpc_no_response_body_refuses_a_body_carrying_rpc() -> None:
    recorder = ClaimRecorder([_GRPC_WITH_BODY])
    with pytest.raises(ConformanceError, match="use assert_roundtrip"):
        recorder.assert_no_response_body(None)


def test_grpc_no_response_body_rejects_a_non_none_return() -> None:
    recorder = ClaimRecorder([_GRPC_WITHOUT_BODY])
    with pytest.raises(ConformanceError, match="instead of None"):
        recorder.assert_no_response_body({"deleted": True})


def test_grpc_no_response_body_rejects_client_side() -> None:
    recorder = ClaimRecorder([_GRPC_WITHOUT_BODY])
    with pytest.raises(ConformanceError, match="HTTP-only"):
        recorder.assert_no_response_body(
            UpsertRecordsResponse(record_count=3), client_side=["record_count"]
        )
