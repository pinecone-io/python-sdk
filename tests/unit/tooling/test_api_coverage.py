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
                post:
                  operationId: create_widget
                  responses:
                    '201':
                      content:
                        application/json:
                          schema:
                            type: object
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
            """
        )
    )
    ops = cov.parse_oas_file(oas)
    assert ops == {
        "widgets:list_widgets": {
            "kind": "http",
            "method": "GET",
            "base_path": "",
            "path": "/widgets",
            "success_body": True,
        },
        "widgets:create_widget": {
            "kind": "http",
            "method": "POST",
            "base_path": "",
            "path": "/widgets",
            "success_body": True,
        },
        "widgets:delete_widget": {
            "kind": "http",
            "method": "DELETE",
            "base_path": "",
            "path": "/widgets/{widget_id}",
            "success_body": False,
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
    ops = cov.parse_oas_file(oas)
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
            service VectorService {
              rpc Upsert(UpsertRequest) returns (UpsertResponse) {}
              rpc Query(QueryRequest) returns (QueryResponse) {}
            }
            """
        )
    )
    ops = cov.parse_proto_file(proto)
    assert ops == {
        "db_data_grpc:Upsert": {"kind": "grpc", "service": "VectorService", "rpc": "Upsert"},
        "db_data_grpc:Query": {"kind": "grpc", "service": "VectorService", "rpc": "Query"},
    }


def test_parse_proto_file_rejects_rpc_outside_service(tmp_path: Path) -> None:
    proto = tmp_path / "db_data_2026-07.proto"
    proto.write_text("rpc Orphan(Req) returns (Resp) {}\n")
    with pytest.raises(cov.SpecError, match="outside any service"):
        cov.parse_proto_file(proto)


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
    assert cov.derive_operations(cov.DEFAULT_SPECS_DIR) == cov.load_manifest()


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
    recorder = ClaimRecorder(["db_control:list_indexes"])
    request = _request("GET", "/indexes")
    recorder.assert_request(request)
    recorder.assert_api_version(request)
    recorder.assert_roundtrip(
        SampleModel,
        {"name": "idx", "dimension": 2, "pagination": "tok"},
        optional_absent=["pagination"],
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
    recorder = ClaimRecorder(["db_control:list_indexes"])
    with pytest.raises(ConformanceError, match="lost in schema round-trip"):
        recorder.assert_roundtrip(
            SampleModel,
            {"name": "idx", "dimension": 2, "pagination": "tok", "extra_field": True},
            optional_absent=["pagination"],
        )


def test_recorder_roundtrip_requires_optional_absent_leg() -> None:
    recorder = ClaimRecorder(["db_control:list_indexes"])
    with pytest.raises(ConformanceError, match="optional_absent must exercise at least one"):
        recorder.assert_roundtrip(SampleModel, {"name": "idx", "dimension": 2}, optional_absent=[])
    with pytest.raises(ConformanceError, match="non-optional or unknown"):
        recorder.assert_roundtrip(
            SampleModel, {"name": "idx", "dimension": 2}, optional_absent=["name"]
        )
    with pytest.raises(ConformanceError, match="not in the payload"):
        recorder.assert_roundtrip(
            SampleModel, {"name": "idx", "dimension": 2}, optional_absent=["pagination"]
        )


def test_recorder_roundtrip_allows_required_only_models() -> None:
    recorder = ClaimRecorder(["db_control:list_indexes"])
    recorder.assert_roundtrip(RequiredOnlyModel, {"name": "idx"}, optional_absent=[])


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


            @api_op("db_control:list_indexes")
            def test_satisfied(claim):
                req = _req("GET", "/indexes")
                claim.assert_request(req)
                claim.assert_api_version(req)
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
    assert "db_data_grpc:Upsert" in gaps
