"""The fixture-vs-OAS-response-schema leg of assert_roundtrip (#189).

Before this leg existed, a claimed operation only proved the SDK round-trips
the test's own fixture — self-consistency, not spec conformance. These tests
guard the strengthened contract: a fixture no spec-conformant server could
send must fail, a registered divergence must validate against its documented
alternative shape, and a divergence without its question-issue reference must
fail rather than silently pass. They carry no ``@api_op`` — they guard the
registry, they do not claim an operation.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from pinecone.models.admin.token import TokenResponse
from pinecone.models.assistant.model import AssistantModel
from tests.unit.conformance import (
    ClaimRecorder,
    ConformanceError,
    manifest_operations,
    manifest_schemas,
    validate_response_payload,
)

_PLAIN_OP = "oauth:get_token"
_DIVERGENT_OP = "assistant_control:update_assistant"

TOKEN: dict[str, Any] = {
    "access_token": "guard-token",
    "token_type": "Bearer",
    "expires_in": 1800,
}

ASSISTANT: dict[str, Any] = {
    "name": "guard-assistant",
    "status": "Ready",
    "instructions": "Answer briefly.",
    "metadata": {"team": "Operations"},
    "host": "https://prod-1-data.ke.pinecone.io",
    "created_at": "2026-07-01T12:30:00Z",
    "updated_at": "2026-07-01T12:45:00Z",
}


def _entry(op_id: str) -> dict[str, Any]:
    return copy.deepcopy(manifest_operations()[op_id])


def test_conformant_fixture_passes() -> None:
    validate_response_payload(_PLAIN_OP, _entry(_PLAIN_OP), TOKEN)


def test_wrong_typed_fixture_fails() -> None:
    with pytest.raises(ConformanceError, match="does not conform to the spec response schema"):
        validate_response_payload(_PLAIN_OP, _entry(_PLAIN_OP), {**TOKEN, "expires_in": "soon"})


def test_fixture_missing_a_required_property_fails() -> None:
    payload = {key: value for key, value in TOKEN.items() if key != "access_token"}
    with pytest.raises(ConformanceError, match="'access_token' is a required property"):
        validate_response_payload(_PLAIN_OP, _entry(_PLAIN_OP), payload)


def test_fixture_with_a_key_the_spec_never_declared_fails() -> None:
    with pytest.raises(ConformanceError, match="does not conform to the spec response schema"):
        validate_response_payload(_PLAIN_OP, _entry(_PLAIN_OP), {**TOKEN, "scope": "admin"})


def test_wrong_shape_fixture_fails_through_assert_roundtrip() -> None:
    recorder = ClaimRecorder([_PLAIN_OP])
    with pytest.raises(ConformanceError, match="does not conform to the spec response schema"):
        recorder.assert_roundtrip(TokenResponse, {**TOKEN, "scope": "admin"}, optional_absent=[])


def test_divergent_op_validates_against_the_documented_alternative() -> None:
    entry = _entry(_DIVERGENT_OP)
    assert entry["divergence"]["issue"] == 170
    assert entry["divergence"]["response_schema"] == "assistant_control:Assistant"
    assert entry["response_schema"] == "assistant_control:UpdateAssistantResponse"
    validate_response_payload(_DIVERGENT_OP, entry, ASSISTANT)


def test_divergent_op_rejects_the_shape_the_oas_declares() -> None:
    spec_shape = {"assistant_name": "guard-assistant", "instructions": "Answer briefly."}
    with pytest.raises(ConformanceError, match="assistant_control:Assistant"):
        validate_response_payload(_DIVERGENT_OP, _entry(_DIVERGENT_OP), spec_shape)


def test_divergent_op_passes_through_assert_roundtrip() -> None:
    recorder = ClaimRecorder([_DIVERGENT_OP])
    recorder.assert_roundtrip(
        AssistantModel,
        ASSISTANT,
        optional_absent=["instructions", "metadata", "host", "created_at", "updated_at"],
    )


def test_divergence_without_an_issue_reference_fails() -> None:
    entry = _entry(_DIVERGENT_OP)
    del entry["divergence"]["issue"]
    with pytest.raises(ConformanceError, match="silent divergence exceptions are not allowed"):
        validate_response_payload(_DIVERGENT_OP, entry, ASSISTANT)


def test_divergence_with_a_non_numeric_issue_reference_fails() -> None:
    entry = _entry(_DIVERGENT_OP)
    entry["divergence"]["issue"] = "170"
    with pytest.raises(ConformanceError, match="silent divergence exceptions are not allowed"):
        validate_response_payload(_DIVERGENT_OP, entry, ASSISTANT)


def test_divergence_without_a_reason_fails() -> None:
    entry = _entry(_DIVERGENT_OP)
    entry["divergence"]["reason"] = ""
    with pytest.raises(ConformanceError, match="has no reason"):
        validate_response_payload(_DIVERGENT_OP, entry, ASSISTANT)


def test_missing_schema_key_fails_instead_of_skipping() -> None:
    entry = _entry(_PLAIN_OP)
    entry["response_schema"] = None
    with pytest.raises(ConformanceError, match="records no response schema"):
        validate_response_payload(_PLAIN_OP, entry, TOKEN)


def test_unknown_schema_key_fails_instead_of_skipping() -> None:
    entry = _entry(_PLAIN_OP)
    entry["response_schema"] = "oauth:NoSuchSchema"
    with pytest.raises(ConformanceError, match="is not in the manifest"):
        validate_response_payload(_PLAIN_OP, entry, TOKEN)


def test_every_bodied_operation_has_a_vendored_schema() -> None:
    schemas = manifest_schemas()
    for op_id, entry in manifest_operations().items():
        if entry["kind"] != "http":
            continue
        if entry["success_body"]:
            assert entry["response_schema"] in schemas, op_id
        else:
            assert entry["response_schema"] is None, op_id
        divergence = entry.get("divergence")
        if divergence is not None:
            assert isinstance(divergence["issue"], int), op_id
            assert divergence["reason"].strip(), op_id
            assert divergence["response_schema"] in schemas, op_id
