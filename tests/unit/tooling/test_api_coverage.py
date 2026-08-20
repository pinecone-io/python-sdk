"""Tests for scripts/api_coverage.py and the tests/unit/conformance/ machinery."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import httpx
import jsonschema
import msgspec
import pytest

from tests.unit.conformance import (
    CLAIMS,
    ClaimRecorder,
    ConformanceError,
    UnknownOperationError,
    api_op,
    manifest_operations,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "api_coverage.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("api_coverage", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cov = _load_script()


class SampleModel(msgspec.Struct):
    name: str
    dimension: int
    pagination: str | None = None
    metric: str | None = None


class SampleToken(msgspec.Struct):
    access_token: str
    token_type: str
    expires_in: int | None = None


class RequiredOnlyModel(msgspec.Struct):
    name: str


def _request(method: str, path: str, version: str | None = "2026-07") -> httpx.Request:
    headers = {"X-Pinecone-Api-Version": version} if version else {}
    return httpx.Request(method, f"https://api.pinecone.io{path}", headers=headers)


def _overall(stdout: str, kind: str) -> tuple[int, int]:
    """Parse ``overall <kind>: <covered>/<total>`` out of a --report run.

    Covered counts grow as each release lane lands its conformance tests, so
    these end-to-end tests assert the denominator and the invariants, never a
    snapshot of the numerator.
    """
    for line in stdout.splitlines():
        prefix = f"overall {kind}: "
        if line.startswith(prefix):
            covered, _, total = line.removeprefix(prefix).partition("/")
            return int(covered), int(total)
    raise AssertionError(f"no 'overall {kind}' line in --report output:\n{stdout}")


def test_parse_oas_file_extracts_operations(tmp_path: Path) -> None:
    oas = tmp_path / "widgets_2026-07.oas.yaml"
    oas.write_text(
        textwrap.dedent(
            """
            openapi: 3.0.3
            paths:
              /widgets:
                parameters:
                  - name: verbose
                    in: query
                get:
                  operationId: list_widgets
                  responses:
                    '200':
                      content:
                        application/json:
                          schema:
                            type: object
                            properties:
                              data:
                                type: array
                                items:
                                  type: string
                post:
                  operationId: create_widget
                  responses:
                    '201':
                      content:
                        application/json:
                          schema:
                            $ref: '#/components/schemas/Widget'
              /widgets/{widget_id}:
                delete:
                  operationId: delete_widget
                  responses:
                    '202':
                      description: accepted
                    '404':
                      content:
                        application/json:
                          schema:
                            type: object
                            properties:
                              error:
                                type: string
              /widgets/{widget_id}/retire:
                post:
                  operationId: retire_widget
                  responses:
                    '200':
                      content:
                        application/json:
                          schema:
                            $ref: '#/components/schemas/RetireWidgetResponse'
            components:
              schemas:
                Widget:
                  type: object
                  properties:
                    id:
                      type: string
                RetireWidgetResponse:
                  description: The response for the retire operation.
                  type: object
            """
        )
    )
    ops, schemas = cov.parse_oas_file(oas)
    assert ops == {
        "widgets:list_widgets": {
            "kind": "http",
            "method": "GET",
            "base_path": "",
            "path": "/widgets",
            "success_body": True,
            "response_schema": "widgets:list_widgets.response",
        },
        "widgets:create_widget": {
            "kind": "http",
            "method": "POST",
            "base_path": "",
            "path": "/widgets",
            "success_body": True,
            "response_schema": "widgets:Widget",
        },
        "widgets:delete_widget": {
            "kind": "http",
            "method": "DELETE",
            "base_path": "",
            "path": "/widgets/{widget_id}",
            "success_body": False,
            "response_schema": None,
        },
        "widgets:retire_widget": {
            "kind": "http",
            "method": "POST",
            "base_path": "",
            "path": "/widgets/{widget_id}/retire",
            "success_body": False,
            "response_schema": None,
        },
    }
    assert schemas == {
        "widgets:list_widgets.response": {
            "type": "object",
            "additionalProperties": False,
            "properties": {"data": {"type": "array", "items": {"type": "string"}}},
        },
        "widgets:Widget": {
            "type": "object",
            "additionalProperties": False,
            "properties": {"id": {"type": "string"}},
        },
    }


def test_parse_oas_file_records_the_server_base_path(tmp_path: Path) -> None:
    oas = tmp_path / "widgets_2026-07.oas.yaml"
    oas.write_text(
        textwrap.dedent(
            """
            openapi: 3.0.3
            servers:
            - url: https://api.pinecone.io/widget
            - url: https://eu.api.pinecone.io/widget
            paths:
              /widgets:
                get:
                  operationId: list_widgets
                  responses:
                    '200':
                      description: ok
            """
        )
    )
    ops, _ = cov.parse_oas_file(oas)
    assert ops["widgets:list_widgets"]["base_path"] == "/widget"


def test_parse_oas_file_rejects_servers_that_disagree_on_base_path(tmp_path: Path) -> None:
    oas = tmp_path / "widgets_2026-07.oas.yaml"
    oas.write_text(
        textwrap.dedent(
            """
            openapi: 3.0.3
            servers:
            - url: https://api.pinecone.io/widget
            - url: https://api.pinecone.io/gadget
            paths:
              /widgets:
                get:
                  operationId: list_widgets
                  responses:
                    '200':
                      description: ok
            """
        )
    )
    with pytest.raises(cov.SpecError, match="disagree on a base path"):
        cov.parse_oas_file(oas)


def test_assert_request_requires_the_server_base_path() -> None:
    recorder = ClaimRecorder(["assistant_control:list_assistants"])
    with pytest.raises(ConformanceError, match="does not match spec template"):
        recorder.assert_request(_request("GET", "/assistants"))
    recorder.assert_request(_request("GET", "/assistant/assistants"))


def test_parse_oas_file_requires_operation_ids(tmp_path: Path) -> None:
    oas = tmp_path / "widgets_2026-07.oas.yaml"
    oas.write_text("paths:\n  /widgets:\n    get:\n      summary: no id\n")
    with pytest.raises(cov.SpecError, match="no operationId"):
        cov.parse_oas_file(oas)


def test_parse_proto_file_extracts_rpcs(tmp_path: Path) -> None:
    proto = tmp_path / "db_data_2026-07.proto"
    proto.write_text(
        textwrap.dedent(
            """
            syntax = "proto3";
            message UpsertRequest {
              repeated Vector vectors = 1 [
                (google.api.field_behavior) = REQUIRED
              ];
            }
            message UpsertResponse {
              uint32 upserted_count = 1;
            }
            message DeleteRequest {
              repeated string ids = 1;
            }
            message DeleteResponse {}
            service VectorService {
              rpc Upsert(UpsertRequest) returns (UpsertResponse) {}
              rpc Delete(DeleteRequest) returns (DeleteResponse) {}
            }
            """
        )
    )
    ops = cov.parse_proto_file(proto)
    assert ops == {
        "db_data_grpc:Upsert": {
            "kind": "grpc",
            "service": "VectorService",
            "rpc": "Upsert",
            "request": "UpsertRequest",
            "response": "UpsertResponse",
            "success_body": True,
        },
        "db_data_grpc:Delete": {
            "kind": "grpc",
            "service": "VectorService",
            "rpc": "Delete",
            "request": "DeleteRequest",
            "response": "DeleteResponse",
            "success_body": False,
        },
    }


def test_parse_proto_file_rejects_rpc_outside_service(tmp_path: Path) -> None:
    proto = tmp_path / "db_data_2026-07.proto"
    proto.write_text("message Req {}\nmessage Resp {}\nrpc Orphan(Req) returns (Resp) {}\n")
    with pytest.raises(cov.SpecError, match="outside any service"):
        cov.parse_proto_file(proto)


def test_parse_proto_file_rejects_an_undefined_response_message(tmp_path: Path) -> None:
    proto = tmp_path / "db_data_2026-07.proto"
    proto.write_text(
        "message Req {}\nservice VectorService {\n  rpc Op(Req) returns (Ghost) {}\n}\n"
    )
    with pytest.raises(cov.SpecError, match="does not define"):
        cov.parse_proto_file(proto)


def test_proto_message_fields_ignores_comments_and_option_blocks() -> None:
    text = textwrap.dedent(
        """
        // message NotReal { string ghost = 1; }
        message Commented {
          // string ghost = 1;
        }
        message WithOptions {
          optional google.protobuf.Struct filter = 6 [
            (google.api.field_behavior) = REQUIRED
          ];
        }
        message OneLiner {}
        message CompactWithField { int32 x = 1; }
        message OpenWithField { int32 x = 1;
          int32 y = 2;
        }
        """
    )
    assert cov._proto_message_fields(text) == {
        "Commented": False,
        "WithOptions": True,
        "OneLiner": False,
        "CompactWithField": True,
        "OpenWithField": True,
    }


_BUNDLER_DOC = {
    "components": {
        "schemas": {
            "Leaf": {
                "type": "object",
                "properties": {"value": {"type": "string", "example": "v"}},
                "required": ["value"],
            },
            "Node": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "children": {
                        "type": "array",
                        "items": {"$ref": "#/components/schemas/Node"},
                    },
                },
            },
            "Envelope": {
                "type": "object",
                "description": "annotation to strip",
                "x-component-name": "Envelope",
                "properties": {
                    "leaf": {
                        "nullable": True,
                        "type": "object",
                        "allOf": [{"$ref": "#/components/schemas/Leaf"}],
                    },
                    "state": {"type": "string", "nullable": True},
                },
            },
        }
    }
}


def test_bundler_inlines_refs_merges_allof_translates_nullable_and_seals() -> None:
    bundled = cov._SchemaBundler(_BUNDLER_DOC).bundle({"$ref": "#/components/schemas/Envelope"})
    assert bundled == {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "leaf": {
                "type": ["object", "null"],
                "additionalProperties": False,
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
            "state": {"type": ["string", "null"]},
        },
    }


def test_bundler_relaxes_oneof_to_deduplicated_anyof() -> None:
    doc: dict = {"components": {"schemas": {}}}
    schema = {
        "type": "object",
        "properties": {
            "reference": {
                "oneOf": [
                    {
                        "type": "object",
                        "properties": {"file": {"type": "string"}},
                        "discriminator": {"propertyName": "type"},
                    },
                    {"type": "object", "properties": {"file": {"type": "string"}}},
                    {
                        "type": "object",
                        "properties": {"file": {"type": "string"}, "pages": {"type": "integer"}},
                    },
                ]
            }
        },
    }
    bundled = cov._SchemaBundler(doc).bundle(schema)
    variants = bundled["properties"]["reference"]["anyOf"]
    assert len(variants) == 2
    assert "oneOf" not in json.dumps(bundled)

    validator = jsonschema.Draft202012Validator(bundled)
    validator.validate({"reference": {"file": "doc.pdf"}})
    validator.validate({"reference": {"file": "doc.pdf", "pages": 3}})
    assert list(validator.iter_errors({"reference": {"unknown": 1}}))


def test_bundler_widens_both_type_and_enum_for_nullable() -> None:
    doc: dict = {"components": {"schemas": {}}}
    schema = {
        "type": "object",
        "properties": {
            "state": {"type": "string", "enum": ["on", "off"], "nullable": True},
        },
    }
    bundled = cov._SchemaBundler(doc).bundle(schema)
    assert bundled["properties"]["state"] == {
        "type": ["string", "null"],
        "enum": ["on", "off", None],
    }
    jsonschema.Draft202012Validator(bundled).validate({"state": None})


def test_bundler_keeps_cyclic_refs_as_local_defs() -> None:
    bundled = cov._SchemaBundler(_BUNDLER_DOC).bundle({"$ref": "#/components/schemas/Node"})
    assert bundled["$ref"] == "#/$defs/Node"
    node = bundled["$defs"]["Node"]
    assert node["properties"]["children"]["items"] == {"$ref": "#/$defs/Node"}
    assert node["additionalProperties"] is False

    validator = jsonschema.Draft202012Validator(bundled)
    validator.validate({"name": "root", "children": [{"name": "child", "children": []}]})
    assert list(validator.iter_errors({"name": "root", "children": [{"unknown": 1}]}))


_DIVERGENT_OAS = """
openapi: 3.0.3
paths:
  /widgets/{widget_id}:
    patch:
      operationId: update_widget
      responses:
        '200':
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/UpdateWidgetResponse'
components:
  schemas:
    UpdateWidgetResponse:
      type: object
      properties:
        widget_name:
          type: string
    Widget:
      type: object
      properties:
        id:
          type: string
      required:
      - id
"""


def test_parse_oas_file_applies_divergences(tmp_path: Path) -> None:
    oas = tmp_path / "widgets_2026-07.oas.yaml"
    oas.write_text(_DIVERGENT_OAS)
    divergences = {
        "widgets:update_widget": {
            "issue": 170,
            "reason": "backend returns the full Widget",
            "alternative_schema": "Widget",
        }
    }
    ops, schemas = cov.parse_oas_file(oas, divergences)
    entry = ops["widgets:update_widget"]
    assert entry["response_schema"] == "widgets:UpdateWidgetResponse"
    assert entry["divergence"] == {
        "issue": 170,
        "reason": "backend returns the full Widget",
        "response_schema": "widgets:Widget",
    }
    assert set(schemas) == {"widgets:UpdateWidgetResponse", "widgets:Widget"}


def test_parse_oas_file_rejects_divergence_naming_a_missing_component(tmp_path: Path) -> None:
    oas = tmp_path / "widgets_2026-07.oas.yaml"
    oas.write_text(_DIVERGENT_OAS)
    divergences = {
        "widgets:update_widget": {
            "issue": 170,
            "reason": "backend returns the full Widget",
            "alternative_schema": "NoSuchSchema",
        }
    }
    with pytest.raises(cov.SpecError, match="not in components/schemas"):
        cov.parse_oas_file(oas, divergences)


def _synthetic_specs_dir(tmp_path: Path) -> Path:
    specs = tmp_path / "specs"
    specs.mkdir()
    (specs / "widgets_2026-07.oas.yaml").write_text(_DIVERGENT_OAS)
    (specs / "db_data_2026-07.proto").write_text(
        'syntax = "proto3";\n'
        "message UpsertRequest {\n  repeated Vector vectors = 1;\n}\n"
        "message UpsertResponse {\n  uint32 upserted_count = 1;\n}\n"
        "service VectorService {\n"
        "  rpc Upsert(UpsertRequest) returns (UpsertResponse) {}\n}\n"
    )
    return specs


def test_manifest_generation_is_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    divergences_path = tmp_path / "divergences_2026-07.json"
    divergences_path.write_text(
        json.dumps(
            {
                "divergences": {
                    "widgets:update_widget": {
                        "issue": 170,
                        "reason": "backend returns the full Widget",
                        "alternative_schema": "Widget",
                    }
                }
            }
        )
    )
    monkeypatch.setattr(cov, "DIVERGENCES_PATH", divergences_path)
    specs = _synthetic_specs_dir(tmp_path)

    first = cov.derive_manifest(specs)
    second = cov.derive_manifest(specs)
    assert first == second
    assert cov.render_manifest(first) == cov.render_manifest(second)

    shuffled = {
        "api_version": first["api_version"],
        "operations": dict(reversed(list(first["operations"].items()))),
        "schemas": dict(reversed(list(first["schemas"].items()))),
    }
    assert cov.render_manifest(shuffled) == cov.render_manifest(first)
    assert json.loads(cov.render_manifest(first))["operations"] == first["operations"]


def test_derive_manifest_rejects_divergences_for_unknown_operations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    divergences_path = tmp_path / "divergences_2026-07.json"
    divergences_path.write_text(
        json.dumps(
            {
                "divergences": {
                    "widgets:no_such_op": {
                        "issue": 170,
                        "reason": "stale entry",
                        "alternative_schema": "Widget",
                    }
                }
            }
        )
    )
    monkeypatch.setattr(cov, "DIVERGENCES_PATH", divergences_path)
    specs = _synthetic_specs_dir(tmp_path)
    with pytest.raises(cov.SpecError, match="not in the specs"):
        cov.derive_manifest(specs)


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        ({"reason": "r", "alternative_schema": "Widget"}, "must reference a question"),
        (
            {"issue": True, "reason": "r", "alternative_schema": "Widget"},
            "must reference a question",
        ),
        ({"issue": 170, "reason": " ", "alternative_schema": "Widget"}, "non-empty 'reason'"),
        ({"issue": 170, "reason": "r"}, "alternative_schema"),
        (
            {"issue": 170, "reason": "r", "alternative_schema": "Widget", "extra": 1},
            "unknown keys",
        ),
    ],
)
def test_load_divergences_rejects_malformed_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entry: dict, message: str
) -> None:
    divergences_path = tmp_path / "divergences_2026-07.json"
    divergences_path.write_text(json.dumps({"divergences": {"widgets:update_widget": entry}}))
    monkeypatch.setattr(cov, "DIVERGENCES_PATH", divergences_path)
    with pytest.raises(cov.SpecError, match=message):
        cov.load_divergences()


def test_parse_oas_file_applies_a_base_path_override(tmp_path: Path) -> None:
    oas = tmp_path / "widgets_2026-07.oas.yaml"
    oas.write_text(
        textwrap.dedent(
            """
            openapi: 3.0.3
            servers:
            - url: https://{widget_host}
            paths:
              /widgets:
                get:
                  operationId: list_widgets
                  responses:
                    '200':
                      description: ok
            """
        )
    )
    overrides = {
        "widgets": {"issue": 173, "reason": "mounted under /widget", "base_path": "/widget"}
    }
    ops, _ = cov.parse_oas_file(oas, None, overrides)
    entry = ops["widgets:list_widgets"]
    assert entry["base_path"] == "/widget"
    assert entry["base_path_divergence"] == {
        "issue": 173,
        "reason": "mounted under /widget",
        "spec_base_path": "",
    }


def test_derive_manifest_rejects_base_path_overrides_for_unknown_surfaces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    divergences_path = tmp_path / "divergences_2026-07.json"
    divergences_path.write_text(
        json.dumps(
            {
                "divergences": {},
                "base_path_overrides": {
                    "gadgets": {"issue": 173, "reason": "stale entry", "base_path": "/gadget"}
                },
            }
        )
    )
    monkeypatch.setattr(cov, "DIVERGENCES_PATH", divergences_path)
    specs = _synthetic_specs_dir(tmp_path)
    with pytest.raises(cov.SpecError, match="surfaces not in the specs"):
        cov.derive_manifest(specs)


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        ({"reason": "r", "base_path": "/widget"}, "must reference a question"),
        ({"issue": True, "reason": "r", "base_path": "/widget"}, "must reference a question"),
        ({"issue": 173, "reason": " ", "base_path": "/widget"}, "non-empty 'reason'"),
        ({"issue": 173, "reason": "r"}, "must be a path"),
        ({"issue": 173, "reason": "r", "base_path": "widget"}, "must be a path"),
        ({"issue": 173, "reason": "r", "base_path": "/widget/"}, "must be a path"),
        ({"issue": 173, "reason": "r", "base_path": "/widget", "extra": 1}, "unknown keys"),
    ],
)
def test_load_base_path_overrides_rejects_malformed_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entry: dict, message: str
) -> None:
    divergences_path = tmp_path / "divergences_2026-07.json"
    divergences_path.write_text(json.dumps({"base_path_overrides": {"widgets": entry}}))
    monkeypatch.setattr(cov, "DIVERGENCES_PATH", divergences_path)
    with pytest.raises(cov.SpecError, match=message):
        cov.load_base_path_overrides()


def test_vendored_manifest_denominators() -> None:
    ops = manifest_operations()
    http = [op for op, entry in ops.items() if entry["kind"] == "http"]
    grpc = [op for op, entry in ops.items() if entry["kind"] == "grpc"]
    assert len(http) == 102
    assert len(grpc) == 12
    for op_id, entry in ops.items():
        surface, _, name = op_id.partition(":")
        assert surface and name
        if entry["kind"] == "http":
            assert entry["method"] and entry["path"].startswith("/")
        else:
            assert surface == "db_data_grpc"
            assert entry["service"] == "VectorService"
            assert entry["rpc"] == name


@pytest.mark.skipif(not cov.DEFAULT_SPECS_DIR.is_dir(), reason="2026-07 spec checkout not present")
def test_vendored_manifest_matches_live_specs() -> None:
    derived = cov.derive_manifest(cov.DEFAULT_SPECS_DIR)
    vendored = cov.load_manifest()
    assert vendored["operations"] == derived["operations"]
    assert vendored["schemas"] == derived["schemas"]


def test_version_constants_extraction() -> None:
    source = textwrap.dedent(
        """
        CONTROL_PLANE_API_VERSION: str = "2026-07"
        ASSISTANT_API_VERSION = "2025-10"
        API_VERSION_HEADER: str = "X-Pinecone-Api-Version"
        DEFAULT_BASE_URL: str = "https://api.pinecone.io"
        SOME_DATE = "2026-07"
        """
    )
    assert cov.version_constants(source) == {
        "CONTROL_PLANE_API_VERSION": "2026-07",
        "ASSISTANT_API_VERSION": "2025-10",
    }


def test_unexcepted_constants_honors_decision_comments() -> None:
    constants = {
        "CONTROL_PLANE_API_VERSION": "2026-07",
        "ASSISTANT_API_VERSION": "2025-10",
        "ADMIN_API_VERSION": "2025-10",
    }
    decisions = ["DECISION: ASSISTANT_API_VERSION stays at 2025-10 until assistants GA"]
    assert cov.unexcepted_constants(constants, decisions) == {"ADMIN_API_VERSION": "2025-10"}


def test_real_constants_file_parses() -> None:
    constants = cov.version_constants(cov.CONSTANTS_PATH.read_text())
    assert "CONTROL_PLANE_API_VERSION" in constants
    assert "API_VERSION_HEADER" not in constants


def test_coverage_counting_excludes_non_passing_tests() -> None:
    results = {
        "tests": {
            "t::a": {"ops": ["db_control:list_indexes"], "outcome": "passed"},
            "t::b": {"ops": ["db_control:create_index"], "outcome": "failed"},
            "t::c": {"ops": ["db_control:delete_index"], "outcome": "skipped"},
            "t::d": {"ops": ["db_control:describe_index"], "outcome": "collected"},
        }
    }
    assert cov.verified_ops(results) == {"db_control:list_indexes"}
    assert set(cov.claimed_ops(results)) == {
        "db_control:list_indexes",
        "db_control:create_index",
        "db_control:delete_index",
        "db_control:describe_index",
    }
    assert cov.unverified_claimed_tests(results) == {
        "t::b": "failed",
        "t::c": "skipped",
        "t::d": "collected",
    }


def test_api_op_rejects_unknown_operation() -> None:
    with pytest.raises(UnknownOperationError, match="not_a_real_op"):
        api_op("db_control:not_a_real_op")


def test_api_op_registers_and_marks() -> None:
    @api_op("db_control:list_indexes")
    def probe(claim: ClaimRecorder) -> None:
        pass

    assert "probe" in CLAIMS["db_control:list_indexes"][-1]
    marks = [m for m in probe.pytestmark if m.name == "api_op"]
    assert [m.args[0] for m in marks] == ["db_control:list_indexes"]


def test_recorder_happy_path_http() -> None:
    recorder = ClaimRecorder(["oauth:get_token"])
    request = _request("POST", "/oauth/token")
    recorder.assert_request(request)
    recorder.assert_api_version(request)
    recorder.assert_roundtrip(
        SampleToken,
        {"access_token": "tok", "token_type": "Bearer", "expires_in": 1800},
        optional_absent=["expires_in"],
    )
    recorder.assert_satisfied()


def test_recorder_matches_path_templates() -> None:
    recorder = ClaimRecorder(["db_control:describe_index"])
    recorder.assert_request(_request("GET", "/indexes/my-index"))
    with pytest.raises(ConformanceError, match="does not match spec template"):
        recorder.assert_request(_request("GET", "/indexes/my-index/backups"))


def test_recorder_rejects_wrong_method() -> None:
    recorder = ClaimRecorder(["db_control:list_indexes"])
    with pytest.raises(ConformanceError, match="expected method GET"):
        recorder.assert_request(_request("POST", "/indexes"))


def test_recorder_rejects_wrong_or_missing_version() -> None:
    recorder = ClaimRecorder(["db_control:list_indexes"])
    with pytest.raises(ConformanceError, match="expected '2026-07'"):
        recorder.assert_api_version(_request("GET", "/indexes", version="2025-10"))
    with pytest.raises(ConformanceError, match="is None"):
        recorder.assert_api_version(_request("GET", "/indexes", version=None))


def test_recorder_grpc_request_and_metadata() -> None:
    recorder = ClaimRecorder(["db_data_grpc:Upsert"])
    recorder.assert_grpc_request("/VectorService/Upsert")
    recorder.assert_api_version([("x-pinecone-api-version", "2026-07")])
    with pytest.raises(ConformanceError, match="call used"):
        recorder.assert_grpc_request("/VectorService/Query")
    with pytest.raises(ConformanceError, match="use assert_request"):
        ClaimRecorder(["db_control:list_indexes"]).assert_grpc_request("/VectorService/Upsert")


def test_recorder_roundtrip_detects_lost_fields() -> None:
    recorder = ClaimRecorder(["db_data_grpc:Upsert"])
    with pytest.raises(ConformanceError, match="lost in schema round-trip"):
        recorder.assert_roundtrip(
            SampleModel,
            {"name": "idx", "dimension": 2, "pagination": "tok", "extra_field": True},
            optional_absent=["pagination"],
        )


def test_recorder_roundtrip_requires_optional_absent_leg() -> None:
    recorder = ClaimRecorder(["db_data_grpc:Upsert"])
    with pytest.raises(ConformanceError, match="optional_absent must exercise at least one"):
        recorder.assert_roundtrip(
            SampleModel, {"name": "idx", "dimension": 2, "pagination": "tok"}, optional_absent=[]
        )
    with pytest.raises(ConformanceError, match="non-optional or unknown"):
        recorder.assert_roundtrip(
            SampleModel, {"name": "idx", "dimension": 2}, optional_absent=["name"]
        )
    with pytest.raises(ConformanceError, match="not in the payload"):
        recorder.assert_roundtrip(
            SampleModel, {"name": "idx", "dimension": 2}, optional_absent=["pagination"]
        )


def test_recorder_roundtrip_allows_required_only_models() -> None:
    recorder = ClaimRecorder(["db_data_grpc:Upsert"])
    recorder.assert_roundtrip(RequiredOnlyModel, {"name": "idx"}, optional_absent=[])


def test_recorder_roundtrip_allows_a_payload_with_no_optional_field() -> None:
    recorder = ClaimRecorder(["db_data_grpc:Upsert"])
    recorder.assert_roundtrip(SampleModel, {"name": "idx", "dimension": 2}, optional_absent=[])


def test_recorder_unsatisfied_claims_fail() -> None:
    recorder = ClaimRecorder(["db_control:list_indexes"])
    recorder.assert_api_version(_request("GET", "/indexes"))
    with pytest.raises(ConformanceError, match="missing mandatory assertions"):
        recorder.assert_satisfied()


def test_recorder_multi_op_requires_disambiguation() -> None:
    recorder = ClaimRecorder(["db_control:list_indexes", "db_control:create_index"])
    with pytest.raises(ConformanceError, match="pass op="):
        recorder.assert_request(_request("GET", "/indexes"))
    recorder.assert_request(_request("GET", "/indexes"), op="db_control:list_indexes")
    with pytest.raises(ConformanceError, match="not claimed by this test"):
        recorder.assert_request(_request("GET", "/indexes"), op="db_control:delete_index")


def _run(cmd: list[str], env_extra: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT), **(env_extra or {})}
    return subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True)  # noqa: S603


def test_claim_fixture_enforcement_end_to_end(tmp_path: Path) -> None:
    (tmp_path / "conftest.py").write_text(
        textwrap.dedent(
            """
            from tests.unit.conformance.conftest import (
                claim,
                pytest_collection_modifyitems,
                pytest_runtest_logreport,
                pytest_runtest_setup,
                pytest_sessionfinish,
            )
            """
        )
    )
    (tmp_path / "test_probe.py").write_text(
        textwrap.dedent(
            """
            from __future__ import annotations

            import httpx
            import msgspec

            from tests.unit.conformance import api_op


            class Probe(msgspec.Struct):
                a: int
                b: str | None = None


            def _req(method, path):
                return httpx.Request(
                    method,
                    f"https://api.pinecone.io{path}",
                    headers={"X-Pinecone-Api-Version": "2026-07"},
                )


            @api_op("db_data_grpc:Upsert")
            def test_satisfied(claim):
                claim.assert_grpc_request("/VectorService/Upsert")
                claim.assert_api_version([("x-pinecone-api-version", "2026-07")])
                claim.assert_roundtrip(Probe, {"a": 1, "b": "x"}, optional_absent=["b"])


            @api_op("db_control:create_index")
            def test_incomplete(claim):
                claim.assert_api_version(_req("POST", "/indexes"))


            @api_op("db_control:describe_index")
            def test_without_claim_fixture():
                pass
            """
        )
    )
    results_path = tmp_path / "results.json"
    proc = _run(
        [sys.executable, "-m", "pytest", str(tmp_path), "-q", "-p", "no:cacheprovider"],
        env_extra={"PINECONE_CONFORMANCE_RESULTS": str(results_path)},
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr

    outcomes = {
        nodeid.rsplit("::", 1)[-1]: info["outcome"]
        for nodeid, info in json.loads(results_path.read_text())["tests"].items()
    }
    assert outcomes == {
        "test_satisfied": "passed",
        "test_incomplete": "failed",
        "test_without_claim_fixture": "failed",
    }
    assert "missing mandatory assertions" in proc.stdout


def test_report_mode_end_to_end() -> None:
    proc = _run([sys.executable, str(SCRIPT_PATH), "--report"])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    http_covered, http_total = _overall(proc.stdout, "http")
    grpc_covered, grpc_total = _overall(proc.stdout, "grpc")
    assert (http_total, grpc_total) == (102, 12)
    assert 0 <= http_covered <= http_total
    assert 0 <= grpc_covered <= grpc_total


def test_gaps_mode_end_to_end() -> None:
    report = _run([sys.executable, str(SCRIPT_PATH), "--report"])
    assert report.returncode == 0, report.stdout + report.stderr
    http_covered, http_total = _overall(report.stdout, "http")
    grpc_covered, grpc_total = _overall(report.stdout, "grpc")

    proc = _run([sys.executable, str(SCRIPT_PATH), "--gaps"])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    gaps = proc.stdout.split()
    uncovered = (http_total - http_covered) + (grpc_total - grpc_covered)
    assert len(gaps) == uncovered
    assert set(gaps) <= set(manifest_operations())
    assert "db_data_grpc:Upsert" not in gaps
