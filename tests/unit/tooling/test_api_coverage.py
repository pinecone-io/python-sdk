"""Tests for scripts/api_coverage.py and the tests/unit/conformance/ machinery."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from collections.abc import Callable
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


EXEMPT_OP = "db_metrics:fetch_prometheus_targets"
OTHER_OP = "db_control:list_indexes"

# Spelled out rather than read from cov.RELEASE_MILESTONES, so wiring the sum to
# the wrong milestones fails these tests instead of relabelling them.
RELEASE_MILESTONE_TITLES = (
    "2026-07 M1: Foundations & version bumps",
    "2026-07 M2: Shared models & validation",
    "2026-07 M3: Transport implementations",
    "2026-07 M4: Graduation cleanup, docs & release notes",
)


def _ops(*op_ids: str) -> dict[str, dict[str, object]]:
    return {op_id: {"kind": "http"} for op_id in op_ids}


def _milestone_payload(counts: tuple[int, int, int, int]) -> list[dict[str, object]]:
    """The four release milestones at *counts*, alongside unrelated ones."""
    return [
        *(
            {"title": title, "open_issues": count, "closed_issues": 0}
            for title, count in zip(RELEASE_MILESTONE_TITLES, counts, strict=True)
        ),
        {"title": "9.2.0", "open_issues": 7, "closed_issues": 10},
        {"title": "Bulk core rewrite", "open_issues": 3, "closed_issues": 9},
    ]


def test_the_db_metrics_exemption_is_named_and_cites_its_decision() -> None:
    assert set(cov.COVERAGE_EXEMPTIONS) == {EXEMPT_OP}
    assert EXEMPT_OP in manifest_operations()
    assert "issuecomment-5365004482" in cov.COVERAGE_EXEMPTIONS[EXEMPT_OP]


def test_coverage_status_honors_the_named_exemption() -> None:
    status = cov.coverage_status(_ops(EXEMPT_OP, OTHER_OP), {OTHER_OP})
    assert status.ok
    assert status.problems == []
    assert status.missing == []
    assert f"1 deliberate omission(s): {EXEMPT_OP}" in status.detail
    assert "uncovered" not in status.detail


def test_coverage_status_still_fails_a_different_uncovered_operation() -> None:
    status = cov.coverage_status(_ops(EXEMPT_OP, OTHER_OP), set())
    assert not status.ok
    assert status.problems == []
    assert status.missing == [OTHER_OP]
    assert "1 uncovered (run --gaps)" in status.detail


def test_coverage_status_excuses_only_the_operation_the_exemption_names() -> None:
    """One named operation, not a tolerance of one: which op is excused matters.

    A count allowance would report the same *number* of gaps here while
    excusing the wrong operation, so this asserts identity, not arithmetic.
    """
    third = "db_control:create_index"
    status = cov.coverage_status(_ops(EXEMPT_OP, OTHER_OP, third), set())
    assert not status.ok
    assert status.missing == sorted([OTHER_OP, third])
    assert EXEMPT_OP not in status.missing
    assert "2 uncovered (run --gaps)" in status.detail


def test_coverage_status_reports_a_stale_exemption() -> None:
    ops = _ops(EXEMPT_OP, OTHER_OP)
    status = cov.coverage_status(ops, set(ops))
    assert not status.ok
    assert len(status.problems) == 1
    assert "STALE" in status.problems[0]
    assert EXEMPT_OP in status.problems[0]


def test_coverage_status_reports_an_exemption_for_an_operation_off_the_specs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        cov.COVERAGE_EXEMPTIONS, "gone:removed_operation", "https://example/decision"
    )
    status = cov.coverage_status(_ops(OTHER_OP), {OTHER_OP})
    assert not status.ok
    assert any(
        "gone:removed_operation" in p and "not in the 2026-07 specs" in p for p in status.problems
    )


def test_the_epoch_condition_reads_the_four_release_milestones() -> None:
    assert cov.RELEASE_MILESTONES == RELEASE_MILESTONE_TITLES
    assert "issuecomment-5365004630" in cov.RELEASE_MILESTONES_DECISION


def test_the_epoch_condition_no_longer_reads_87s_capped_sub_issue_list() -> None:
    assert not hasattr(cov, "fetch_open_subissues")


def test_release_milestone_status_sums_only_the_release_milestones() -> None:
    ok, message = cov.release_milestone_status(_milestone_payload((0, 0, 16, 15)))
    assert not ok
    assert "31 open issues" in message
    assert "2026-07 M3 16" in message
    assert "2026-07 M4 15" in message


def test_release_milestone_status_passes_only_when_every_milestone_is_empty() -> None:
    ok, message = cov.release_milestone_status(_milestone_payload((0, 0, 0, 0)))
    assert ok
    assert "zero open issues across the 4 2026-07 milestones" in message
    assert not cov.release_milestone_status(_milestone_payload((0, 0, 0, 1)))[0]


def test_release_milestone_status_fails_when_a_release_milestone_is_missing() -> None:
    dropped = RELEASE_MILESTONE_TITLES[1]
    payload = [m for m in _milestone_payload((0, 0, 0, 0)) if m["title"] != dropped]
    ok, message = cov.release_milestone_status(payload)
    assert not ok
    assert dropped in message
    assert "not found in this repo" in message


def test_release_milestone_status_counts_a_later_release_milestone_too() -> None:
    """A fifth 2026-07 milestone cannot hold work the gate never looks at."""
    payload = [
        *_milestone_payload((0, 0, 0, 0)),
        {"title": "2026-07 M5: late addition", "open_issues": 2, "closed_issues": 0},
    ]
    ok, message = cov.release_milestone_status(payload)
    assert not ok
    assert "2 open issues across the 5 2026-07 milestones" in message


def test_the_real_manifest_has_exactly_one_deliberate_omission() -> None:
    """The exempt op is a named id, and it is the only one the gate excuses.

    A bare 101/102 would read the same if a different operation silently
    dropped out, so this asserts which operation is missing, not how many.
    """
    ops = manifest_operations()
    required = set(ops) - {EXEMPT_OP}

    status = cov.coverage_status(ops, required)
    assert status.ok
    assert status.missing == []
    assert f"deliberate omission(s): {EXEMPT_OP}" in status.detail

    dropped = cov.coverage_status(ops, required - {OTHER_OP})
    assert not dropped.ok
    assert dropped.missing == [OTHER_OP]


# Spelled out rather than read from cov.VACUOUS_VERSION_HEADER, so widening or
# narrowing the annotation fails these tests instead of relabelling them.
VACUOUS_OPS = (
    "assistant_evaluation:metrics_alignment",
    "db_data:cancelBulkImport",
    "db_data:describeBulkImport",
    "db_data:listBulkImports",
    "db_data:startBulkImport",
)
VACUOUS_OP = "db_data:listBulkImports"


def test_the_vacuous_version_header_annotation_names_exactly_the_five_known_operations() -> None:
    assert set(cov.VACUOUS_VERSION_HEADER) == set(VACUOUS_OPS)
    assert set(VACUOUS_OPS) <= set(manifest_operations())
    assert set(cov.VACUOUS_VERSION_HEADER.values()) == {348}
    assert "issuecomment-5365418366" in cov.VACUOUS_VERSION_HEADER_EVIDENCE


def test_the_annotation_is_not_an_exemption_and_does_not_excuse_coverage() -> None:
    """Annotated operations still have to be covered, and still count as covered.

    The whole point of annotating rather than removing is that method+path and
    schema round-trip remain real coverage. An annotation that quietly excused
    an operation would be a coverage regression dressed up as a label.
    """
    assert not set(cov.VACUOUS_VERSION_HEADER) & set(cov.COVERAGE_EXEMPTIONS)
    ops = manifest_operations()
    required = set(ops) - set(cov.COVERAGE_EXEMPTIONS)

    assert cov.coverage_status(ops, required).ok

    for op_id in VACUOUS_OPS:
        dropped = cov.coverage_status(ops, required - {op_id})
        assert not dropped.ok, op_id
        assert dropped.missing == [op_id]


def test_the_real_manifest_has_no_stale_vacuous_annotation() -> None:
    ops = manifest_operations()
    assert cov.vacuous_version_header_problems(ops, set(ops)) == []
    assert cov.vacuous_version_header_ops(ops) == cov.VACUOUS_VERSION_HEADER


def test_a_vacuous_annotation_for_an_operation_off_the_specs_is_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(cov.VACUOUS_VERSION_HEADER, "gone:removed_operation", 348)
    problems = cov.vacuous_version_header_problems(_ops(OTHER_OP), {OTHER_OP})
    assert any("gone:removed_operation" in p and "not in the 2026-07 specs" in p for p in problems)
    assert cov.vacuous_version_header_ops(_ops(OTHER_OP)) == {}


def test_a_vacuous_annotation_whose_claim_stopped_passing_is_reported() -> None:
    """The annotation qualifies a passing claim; with no claim it annotates nothing."""
    ops = _ops(*VACUOUS_OPS, OTHER_OP)
    problems = cov.vacuous_version_header_problems(ops, set(ops) - {VACUOUS_OP})
    assert len(problems) == 1
    assert VACUOUS_OP in problems[0]
    assert "annotates nothing" in problems[0]
    assert cov.vacuous_version_header_problems(ops, set(ops)) == []


def test_the_coverage_line_records_the_vacuous_assertions_next_to_the_number() -> None:
    """The number is where the claim is loudest, so the qualification goes there."""
    ops = _ops(*VACUOUS_OPS, OTHER_OP, EXEMPT_OP)
    status = cov.coverage_status(ops, set(ops) - {EXEMPT_OP})
    assert status.ok
    assert "6/7 operations verified" in status.detail
    assert f"1 deliberate omission(s): {EXEMPT_OP}" in status.detail
    assert "on 5 operation(s) the version-header assertion is vacuous" in status.detail

    unannotated = cov.coverage_status(_ops(OTHER_OP, EXEMPT_OP), {OTHER_OP})
    assert unannotated.ok
    assert "vacuous" not in unannotated.detail


def test_report_annotates_the_surfaces_and_lists_the_operations(
    capsys: pytest.CaptureFixture[str],
) -> None:
    ops = manifest_operations()
    cov.print_report(ops, set(ops), basis="test")
    out = capsys.readouterr().out

    assert "assistant_evaluation    1 / 1   (1 of 1 with a vacuous version assertion)" in out
    db_data = next(line for line in out.splitlines() if line.startswith("  db_data  "))
    assert "4 of 24 with a vacuous version assertion" in db_data
    assert cov.VACUOUS_VERSION_HEADER_EVIDENCE in out
    for op_id in VACUOUS_OPS:
        assert f"vacuous version assertion: {op_id} — " in out
    assert "issues/348" in out


def test_report_says_nothing_about_vacuity_when_nothing_is_annotated(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cov, "VACUOUS_VERSION_HEADER", {})
    cov.print_report(_ops(OTHER_OP), {OTHER_OP}, basis="test")
    assert "vacuous" not in capsys.readouterr().out


def test_every_annotated_registration_carries_the_note_at_the_claim_site() -> None:
    """The vacuity is visible where the claim is made, not only in the tooling.

    A reader of the conformance test must be able to see that its version
    assertion is empty. A comment can rot silently, so this pins the reference
    to the annotation list and to the issue that establishes the fact.
    """
    for op_id, issue in cov.VACUOUS_VERSION_HEADER.items():
        claiming = [
            path
            for path in sorted(cov.CONFORMANCE_DIR.glob("test_*.py"))
            if f'@api_op("{op_id}")' in path.read_text()
        ]
        assert claiming, f"no conformance test registers {op_id}"
        for path in claiming:
            text = path.read_text()
            assert "VACUOUS_VERSION_HEADER" in text, path.name
            assert f"#{issue}" in text, path.name


def test_an_annotated_operation_still_has_to_assert_the_version_header() -> None:
    """Annotating is not exempting: dropping the assertion still fails the claim.

    If the endpoint is gated later this assertion becomes meaningful, so it has
    to already be there — and the recorder has to keep refusing a claim that
    skips it.
    """
    recorder = ClaimRecorder([VACUOUS_OP])
    with pytest.raises(ConformanceError, match="x-pinecone-api-version"):
        recorder.assert_api_version(_request("GET", "/bulk/imports", version="2025-10"))

    recorder.assert_request(_request("GET", "/bulk/imports"))
    with pytest.raises(ConformanceError, match=f"{VACUOUS_OP}: api_version"):
        recorder.assert_satisfied()


def _state(state: str, labels: list[str]) -> Callable[[int], tuple[str, set[str]]]:
    def lookup(issue: int) -> tuple[str, set[str]]:
        assert issue == 348
        return state, set(labels)

    return lookup


def test_the_gate_passes_the_annotation_while_its_question_issue_is_open() -> None:
    ops = _ops(*VACUOUS_OPS, OTHER_OP)
    conditions = cov.version_header_conditions(
        ops, set(ops), _state("OPEN", ["question", "lane:rest"])
    )
    assert [ok for ok, _ in conditions] == [True]
    assert "5 operation(s) annotated" in conditions[0][1]
    assert "open question issue #348" in conditions[0][1]
    for op_id in VACUOUS_OPS:
        assert op_id in conditions[0][1]


def test_the_gate_fails_an_annotation_whose_question_issue_has_closed() -> None:
    """A closed issue means the fact moved; the annotation must not outlive it."""
    ops = _ops(*VACUOUS_OPS, OTHER_OP)
    conditions = cov.version_header_conditions(ops, set(ops), _state("CLOSED", ["question"]))
    assert [ok for ok, _ in conditions] == [False]
    assert "which is CLOSED" in conditions[0][1]
    assert "stale" in conditions[0][1]


def test_the_gate_fails_an_annotation_pointed_at_a_non_question_issue() -> None:
    ops = _ops(*VACUOUS_OPS, OTHER_OP)
    conditions = cov.version_header_conditions(ops, set(ops), _state("OPEN", ["bug"]))
    assert [ok for ok, _ in conditions] == [False]
    assert "not a 'question' issue" in conditions[0][1]


def test_the_gate_fails_when_the_issue_lookup_cannot_run() -> None:
    def explode(issue: int) -> tuple[str, set[str]]:
        raise FileNotFoundError("gh")

    ops = _ops(*VACUOUS_OPS, OTHER_OP)
    conditions = cov.version_header_conditions(ops, set(ops), explode)
    assert [ok for ok, _ in conditions] == [False]
    assert "could not check #348 via gh" in conditions[0][1]


def test_the_gate_reports_a_stale_annotation_alongside_the_issue_check() -> None:
    ops = _ops(*VACUOUS_OPS, OTHER_OP)
    conditions = cov.version_header_conditions(
        ops, set(ops) - {VACUOUS_OP}, _state("OPEN", ["question"])
    )
    failed = [message for ok, message in conditions if not ok]
    assert len(failed) == 1
    assert VACUOUS_OP in failed[0]
    assert "annotates nothing" in failed[0]


def test_the_gate_says_so_when_no_operation_is_annotated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cov, "VACUOUS_VERSION_HEADER", {})
    conditions = cov.version_header_conditions(_ops(OTHER_OP), {OTHER_OP}, _state("OPEN", []))
    assert conditions == [(True, "version header: no vacuous version assertions registered")]


def test_a_wholly_stale_registry_never_reads_as_nothing_registered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale annotation must not coexist with a PASS on the same check.

    Every entry naming an operation that has left the specs filters down to an
    empty map. Keyed off that map rather than the registry, the gate would emit
    the stale-annotation FAIL *and* a 'no vacuous version assertions registered'
    PASS — a self-contradicting verdict, and the same read-it-as-reassuring
    failure the annotation exists to prevent (Bugbot, #439).
    """
    monkeypatch.setattr(cov, "VACUOUS_VERSION_HEADER", {"gone:removed_operation": 348})
    conditions = cov.version_header_conditions(_ops(OTHER_OP), {OTHER_OP}, _state("OPEN", []))
    assert [ok for ok, _ in conditions] == [False]
    assert "gone:removed_operation" in conditions[0][1]
    assert not any(
        "no vacuous version assertions registered" in message for _, message in conditions
    )


def _gate_with_stubbed_lookups(
    monkeypatch: pytest.MonkeyPatch, issue_states: dict[int, tuple[str, set[str]]]
) -> Callable[[], int]:
    """A --gate run whose GitHub and pytest lookups are stubbed healthy."""
    ops = manifest_operations()
    monkeypatch.setattr(cov, "run_verify", lambda o: (True, set(o) - {EXEMPT_OP}, []))
    monkeypatch.setattr(cov, "fetch_decision_comments", lambda issue: [])
    monkeypatch.setattr(
        cov, "fetch_issue_state", lambda issue: issue_states.get(issue, ("OPEN", {"question"}))
    )
    monkeypatch.setattr(cov, "fetch_milestones", lambda: _milestone_payload((0, 0, 0, 0)))
    monkeypatch.setattr(cov, "fetch_open_release_issues", lambda: [])
    return lambda: cov.mode_gate(ops, 87)


def test_the_gate_prints_the_version_header_condition(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The condition has to be wired into the gate, not merely available to it."""
    exit_code = _gate_with_stubbed_lookups(monkeypatch, {})()
    out = capsys.readouterr().out
    assert exit_code == 0, out
    assert "[PASS] version header: 5 operation(s) annotated" in out
    assert "open question issue #348" in out
    assert "gate: PASS" in out


def test_the_gate_fails_when_the_annotations_tracking_issue_closes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = _gate_with_stubbed_lookups(monkeypatch, {348: ("CLOSED", {"question"})})()
    out = capsys.readouterr().out
    assert exit_code == 1
    assert "[FAIL] version header:" in out
    assert "which is CLOSED" in out
    assert "gate: FAIL" in out


def test_fetch_issue_state_returns_the_state_and_label_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cov,
        "_gh",
        lambda args: json.dumps(
            {"state": "OPEN", "labels": [{"name": "question"}, {"name": "lane:rest"}]}
        ),
    )
    assert cov.fetch_issue_state(348) == ("OPEN", {"question", "lane:rest"})


def _issue(number: int, labels: list[str], milestone: str | None) -> dict[str, object]:
    return {
        "number": number,
        "labels": [{"name": name} for name in labels],
        "milestone": {"title": milestone} if milestone else None,
    }


def test_unclassified_release_issues_flags_a_ticket_the_milestone_sum_cannot_see() -> None:
    """Unmilestoned does not imply `question` — the converse is where work hides."""
    issues = [
        _issue(371, ["release/2026-07", "lane:rest", "breaking-change"], None),
        _issue(374, ["release/2026-07", "bug", "lane:models"], None),
        _issue(368, ["release/2026-07", "question"], None),
        _issue(366, ["release/2026-07", "lane:tooling"], "2026-07 M4: Graduation cleanup"),
    ]
    assert cov.unclassified_release_issues(issues, epoch_issue=87) == [371, 374]


def test_unclassified_release_issues_excepts_the_epoch_parent_by_number() -> None:
    """#87 carries the release label with no milestone and no `question` label."""
    issues = [_issue(87, ["release/2026-07"], None)]
    assert cov.unclassified_release_issues(issues, epoch_issue=87) == []
    assert cov.unclassified_release_issues(issues, epoch_issue=999) == [87]


def test_unclassified_release_issues_accepts_either_legitimate_state() -> None:
    issues = [
        _issue(1, ["release/2026-07"], "2026-07 M3: Transport implementations"),
        _issue(2, ["release/2026-07", "question"], None),
    ]
    assert cov.unclassified_release_issues(issues, epoch_issue=87) == []


def test_unclassified_release_issues_flags_a_ticket_on_a_non_release_milestone() -> None:
    """Having *a* milestone is not classification — the sum must be the one counting it.

    `release_milestone_status` only sums `2026-07*` titles, so a release ticket
    parked on `9.2.0` reads as classified while being counted by nobody. The
    two functions have to agree on which milestones count or the gap between
    them is where work disappears.
    """
    issues = [
        _issue(500, ["release/2026-07", "lane:rest"], "9.2.0"),
        _issue(501, ["release/2026-07", "bug"], "Bulk core rewrite"),
        _issue(502, ["release/2026-07"], "2026-07 M4: Graduation cleanup"),
    ]
    assert cov.unclassified_release_issues(issues, epoch_issue=87) == [500, 501]


def test_fetch_milestones_follows_pagination_past_the_first_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A release milestone on page two still reaches the sum and holds it nonzero."""
    pages = {
        1: [{"title": f"m{i}", "open_issues": 0} for i in range(cov._PAGE_SIZE)],
        2: [{"title": "2026-07 M9: late", "open_issues": 4}],
    }
    requested: list[str] = []

    def fake_gh(args: list[str]) -> str:
        requested.append(args[-1])
        page = int(args[-1].split("page=")[-1])
        return json.dumps(pages.get(page, []))

    monkeypatch.setattr(cov, "_gh", fake_gh)
    milestones = cov.fetch_milestones()
    assert len(milestones) == cov._PAGE_SIZE + 1
    assert {"title": "2026-07 M9: late", "open_issues": 4} in milestones
    assert len(requested) == 2
    assert not cov.release_milestone_status(milestones)[0]


def test_fetch_milestones_refuses_a_listing_that_never_terminates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full = [{"title": f"m{i}", "open_issues": 0} for i in range(cov._PAGE_SIZE)]
    monkeypatch.setattr(cov, "_gh", lambda args: json.dumps(full))
    with pytest.raises(cov.SpecError, match="did not terminate"):
        cov.fetch_milestones()


def test_fetch_open_release_issues_refuses_a_possibly_truncated_listing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    at_limit = [{"number": n, "labels": [], "milestone": None} for n in range(cov._ISSUE_LIMIT)]
    monkeypatch.setattr(cov, "_gh", lambda args: json.dumps(at_limit))
    with pytest.raises(cov.SpecError, match="fetch limit"):
        cov.fetch_open_release_issues()


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
    assert f"deliberate omission: {EXEMPT_OP} — " in proc.stdout
    assert "db_metrics" in proc.stdout
    assert f"vacuous version assertion: {len(VACUOUS_OPS)} of {http_total + grpc_total} " in (
        proc.stdout
    )
    for op_id in VACUOUS_OPS:
        assert f"vacuous version assertion: {op_id} — " in proc.stdout


# The #327 sweep for unit tests near the 5s ceiling turned this up as the only
# genuinely CPU-bound one: two `scripts/api_coverage.py` subprocesses, each an
# interpreter start plus a parse of every 2026-07 OAS. Measured 1.65s at 94% CPU,
# so unlike the eight wall-clock-bound tests #327 marked, it scales with runner
# speed — the same profile as the grpc dataframe property test that amplified
# >3.3x local->CI and blew the 5s default (#306). 3x margin is not enough on that
# profile; 30s is ~18x.
@pytest.mark.timeout(30)
def test_gaps_mode_end_to_end() -> None:
    report = _run([sys.executable, str(SCRIPT_PATH), "--report"])
    assert report.returncode == 0, report.stdout + report.stderr
    http_covered, http_total = _overall(report.stdout, "http")
    grpc_covered, grpc_total = _overall(report.stdout, "grpc")

    proc = _run([sys.executable, str(SCRIPT_PATH), "--gaps"])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    lines = proc.stdout.splitlines()
    gaps = [line.split("\t", 1)[0] for line in lines]
    uncovered = (http_total - http_covered) + (grpc_total - grpc_covered)
    assert len(gaps) == uncovered
    assert set(gaps) <= set(manifest_operations())
    assert "db_data_grpc:Upsert" not in gaps
    # The exempt operation stays listed — annotated, not hidden, so an
    # unexplained 101/102 cannot be mistaken for a regression.
    assert EXEMPT_OP in gaps
    annotated = next(line for line in lines if line.startswith(EXEMPT_OP))
    assert "deliberate omission" in annotated
    assert cov.COVERAGE_EXEMPTIONS[EXEMPT_OP] in annotated
