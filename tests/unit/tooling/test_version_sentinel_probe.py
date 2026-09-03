"""Tests for scripts/version_sentinel_probe.py.

Hermetic: the probe's HTTP client takes an injected transport, so every
verdict here is decided by the oracle rather than by prod. The live
positive control (#312, #319) is exercised by the nightly workflow, not by
this suite — a CI gate must never depend on an API key.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import httpx
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "version_sentinel_probe.py"


def _load_script():
    """Import the script by path.

    The module must be in ``sys.modules`` *before* it executes: ``@dataclass``
    resolves annotations through ``sys.modules[cls.__module__]``, so a
    not-yet-registered module makes the decorator raise.
    """
    spec = importlib.util.spec_from_file_location("version_sentinel_probe", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


probe = _load_script()


def _response(status: int, content_type: str, body: bytes = b"") -> httpx.Response:
    return httpx.Response(status, headers={"content-type": content_type}, content=body)


class TestSignature:
    def test_error_message_is_excluded_so_an_echoing_server_cannot_pass(self) -> None:
        """The load-bearing exclusion: a message that echoes the request.

        Upstream replaced the bare ``Invalid API version`` with a message
        that interpolates the version you sent. A message-sensitive
        comparison would then call two rejections "different" and report
        PASS on a server that rejects us outright.
        """
        pinned = _response(
            403,
            "text/plain; charset=utf-8",
            json.dumps(
                {"error": {"code": "FORBIDDEN", "message": "Unsupported API version '2026-07'."}}
            ).encode(),
        )
        bogus = _response(
            403,
            "text/plain; charset=utf-8",
            json.dumps(
                {"error": {"code": "FORBIDDEN", "message": "Unsupported API version '9999-99'."}}
            ).encode(),
        )
        assert probe.signature(pinned) == probe.signature(bogus)

    def test_content_type_parameters_are_dropped(self) -> None:
        assert probe.normalize_content_type("Application/JSON; charset=utf-8") == "application/json"
        assert probe.normalize_content_type(None) == ""

    @pytest.mark.parametrize(
        ("body", "expected"),
        [
            (b'{"error": {"code": "FORBIDDEN"}}', "FORBIDDEN"),
            (b'{"code": 7, "message": "nope"}', "7"),
            (b'{"error": "access_denied"}', "access_denied"),
            (b'{"message": "no code at all"}', None),
            (b"", None),
            (b"<html>not json</html>", None),
            (b"[1, 2, 3]", None),
        ],
    )
    def test_error_code_across_every_envelope_prod_emits(
        self, body: bytes, expected: str | None
    ) -> None:
        assert probe.extract_error_code(body) == expected


class TestDiagnosis:
    def test_a_successful_pinned_response_is_unversioned(self) -> None:
        assert probe.diagnose(probe.Signature(200, "application/json", None)).startswith(
            "unversioned"
        )

    def test_a_failing_pinned_response_is_unrecognized(self) -> None:
        assert probe.diagnose(probe.Signature(404, "text/html", None)).startswith("unrecognized")


class TestSpecEnumeration:
    """The denominator is derived from the specs, never hand-listed."""

    @staticmethod
    def _write_spec(tmp_path: Path, paths: dict[str, object], server: str) -> Path:
        spec = tmp_path / f"demo_{probe.API_VERSION}.oas.yaml"
        spec.write_text(json.dumps({"servers": [{"url": server}], "paths": paths}))
        return spec

    def test_every_path_and_method_becomes_an_operation(self, tmp_path: Path) -> None:
        self._write_spec(
            tmp_path,
            {
                "/things": {
                    "get": {"operationId": "listThings"},
                    "post": {"operationId": "createThing"},
                },
                "/things/{thing_id}": {"delete": {"operationId": "deleteThing"}},
            },
            "https://api.example.invalid",
        )
        surface, url, operations = probe.parse_spec(
            next(tmp_path.glob(f"*_{probe.API_VERSION}.oas.yaml"))
        )
        assert surface == "demo"
        assert url == "https://api.example.invalid"
        assert {op.key for op in operations} == {
            "demo:GET /things",
            "demo:POST /things",
            "demo:DELETE /things/{thing_id}",
        }

    def test_required_query_parameters_are_filled_from_the_spec(self, tmp_path: Path) -> None:
        """An omitted required param yields the same 4xx at both versions.

        That is a false FAIL manufactured by the probe, so the parameter has
        to be filled — which ``GET /vectors/fetch`` (``ids``) proved live.
        """
        self._write_spec(
            tmp_path,
            {
                "/fetch": {
                    "get": {
                        "operationId": "fetch",
                        "parameters": [
                            {"in": "query", "name": "ids", "required": True, "schema": {}},
                            {"in": "query", "name": "optional", "schema": {}},
                        ],
                    }
                }
            },
            "https://api.example.invalid",
        )
        _, _, operations = probe.parse_spec(next(tmp_path.glob("*.oas.yaml")))
        assert operations[0].required_query == f"ids={probe.PLACEHOLDER_STRING}"

    def test_probe_variants_add_a_distinctly_keyed_row(self, tmp_path: Path) -> None:
        self._write_spec(
            tmp_path, {"/models": {"get": {"operationId": "listModels"}}}, "https://x.invalid"
        )
        spec = next(tmp_path.glob("*.oas.yaml"))
        monkey = dict(probe.PROBE_VARIANTS)
        monkey["demo:GET /models"] = ("type=embed",)
        original, probe.PROBE_VARIANTS = probe.PROBE_VARIANTS, monkey
        try:
            _, _, operations = probe.parse_spec(spec)
        finally:
            probe.PROBE_VARIANTS = original
        assert [op.key for op in operations] == [
            "demo:GET /models",
            "demo:GET /models?type=embed",
        ]

    def test_path_placeholders_prefer_the_specs_own_example(self) -> None:
        doc: dict[str, object] = {}
        assert probe.placeholder_for({"example": "101", "schema": {}}, doc) == "101"
        assert probe.placeholder_for({"schema": {"format": "uuid"}}, doc) == probe.PLACEHOLDER_UUID
        assert probe.placeholder_for({"schema": {}}, doc) == probe.PLACEHOLDER_STRING

    def test_fill_path_substitutes_every_brace_token(self) -> None:
        assert probe.fill_path("/a/{x}/b/{y}", {"x": "1", "y": "2"}) == "/a/1/b/2"


class TestRunVerdicts:
    """End-to-end verdicts over an injected transport."""

    @staticmethod
    def _spec(tmp_path: Path, paths: dict[str, object]) -> None:
        (tmp_path / f"demo_{probe.API_VERSION}.oas.yaml").write_text(
            json.dumps({"servers": [{"url": "https://api.example.invalid"}], "paths": paths})
        )

    def test_identical_responses_fail_and_differing_responses_pass(self, tmp_path: Path) -> None:
        self._spec(
            tmp_path,
            {
                "/blind": {"get": {"operationId": "blind"}},
                "/healthy": {"get": {"operationId": "healthy"}},
            },
        )

        def handler(request: httpx.Request) -> httpx.Response:
            bogus = request.headers[probe.API_VERSION_HEADER] == probe.BOGUS_VERSION
            if request.url.path == "/blind":
                return _response(404, "text/html")
            if bogus:
                return _response(403, "text/plain", b'{"error": {"code": "FORBIDDEN"}}')
            return _response(200, "application/json", b"{}")

        results = {
            r.key: r
            for r in probe.run(
                specs_dir=tmp_path,
                api_key="unused",
                transport=httpx.MockTransport(handler),
            )
        }
        assert results["demo:GET /blind"].verdict == probe.FAIL
        assert results["demo:GET /healthy"].verdict == probe.PASS

    def test_the_pinned_request_carries_the_pinned_version_and_the_key(
        self, tmp_path: Path
    ) -> None:
        self._spec(tmp_path, {"/x": {"get": {"operationId": "x"}}})
        seen: list[tuple[str, str]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append((request.headers[probe.API_VERSION_HEADER], request.headers["Api-Key"]))
            return _response(200, "application/json", b"{}")

        probe.run(specs_dir=tmp_path, api_key="sekret", transport=httpx.MockTransport(handler))
        assert seen == [(probe.API_VERSION, "sekret"), (probe.BOGUS_VERSION, "sekret")]

    def test_unallowlisted_writes_are_skipped_with_a_reported_reason(self, tmp_path: Path) -> None:
        """A silent absence is the #295 failure mode, so a skip must be a row."""
        self._spec(
            tmp_path,
            {
                "/things": {"post": {"operationId": "create"}},
                "/things/{id}": {
                    "delete": {"operationId": "delete"},
                    "patch": {"operationId": "update"},
                    "put": {"operationId": "replace"},
                },
            },
        )

        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError(f"a write was probed: {request.method} {request.url}")

        results = probe.run(
            specs_dir=tmp_path, api_key="unused", transport=httpx.MockTransport(handler)
        )
        assert len(results) == 4
        assert {r.verdict for r in results} == {probe.SKIP}
        assert all(r.detail.startswith("mutating:") for r in results)

    def test_an_unresolvable_templated_host_skips_with_the_reason_why(self, tmp_path: Path) -> None:
        (tmp_path / f"demo_{probe.API_VERSION}.oas.yaml").write_text(
            json.dumps(
                {
                    "servers": [{"url": "https://{index_host}"}],
                    "paths": {"/namespaces": {"get": {"operationId": "listNamespaces"}}},
                }
            )
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return _response(401, "text/plain", b'{"error": {"code": "UNAUTHENTICATED"}}')

        results = probe.run(
            specs_dir=tmp_path, api_key="unused", transport=httpx.MockTransport(handler)
        )
        assert len(results) == 1
        assert results[0].verdict == probe.SKIP
        assert "unresolved-host" in results[0].detail
        assert "401" in results[0].detail

    def test_an_index_host_override_avoids_the_control_plane_entirely(self, tmp_path: Path) -> None:
        (tmp_path / f"demo_{probe.API_VERSION}.oas.yaml").write_text(
            json.dumps(
                {
                    "servers": [{"url": "https://{index_host}"}],
                    "paths": {"/namespaces": {"get": {"operationId": "listNamespaces"}}},
                }
            )
        )
        hosts: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            hosts.append(request.url.host)
            bogus = request.headers[probe.API_VERSION_HEADER] == probe.BOGUS_VERSION
            return _response(403 if bogus else 200, "application/json", b"{}")

        results = probe.run(
            specs_dir=tmp_path,
            api_key="unused",
            index_host="idx.example.invalid",
            transport=httpx.MockTransport(handler),
        )
        assert [r.verdict for r in results] == [probe.PASS]
        assert set(hosts) == {"idx.example.invalid"}

    def test_a_transport_error_fails_rather_than_disappearing(self, tmp_path: Path) -> None:
        self._spec(tmp_path, {"/x": {"get": {"operationId": "x"}}})

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("no route")

        results = probe.run(
            specs_dir=tmp_path, api_key="unused", transport=httpx.MockTransport(handler)
        )
        assert results[0].verdict == probe.FAIL
        assert "transport error" in results[0].detail

    def test_missing_specs_raise_rather_than_reporting_an_empty_all_green(
        self, tmp_path: Path
    ) -> None:
        transport = httpx.MockTransport(lambda request: _response(200, "application/json"))
        with pytest.raises(probe.ProbeError, match="no \\*_2026-07"):
            probe.run(specs_dir=tmp_path, api_key="unused", transport=transport)


class TestPositiveControl:
    """An all-green run of this tool would be worthless — prove it can fail."""

    @staticmethod
    def _result(key: str, verdict: str) -> object:
        surface, rest = key.split(":", 1)
        method, path = rest.split(" ", 1)
        query = ""
        if "?" in path:
            path, query = path.split("?", 1)
        return probe.Result(
            operation=probe.Operation(
                surface=surface, method=method, path=path, operation_id="", query=query
            ),
            verdict=verdict,
        )

    def _all_expected(self) -> list[object]:
        return [self._result(k, probe.FAIL) for k in probe.CONTROL_EXPECT_FAIL] + [
            self._result(k, probe.PASS) for k in probe.CONTROL_EXPECT_PASS
        ]

    def test_the_known_answers_satisfy_the_control(self) -> None:
        assert probe.check_positive_control(self._all_expected()) == []

    def test_the_control_names_the_ops_from_312_and_319(self) -> None:
        assert probe.CONTROL_EXPECT_FAIL == (
            "inference:GET /models",
            "inference:GET /models?type=embed",
            "assistant_control:GET /assistants",
        )
        assert probe.CONTROL_EXPECT_PASS == ("inference:POST /embed", "inference:POST /rerank")

    def test_an_all_green_run_violates_the_control(self) -> None:
        results = [
            self._result(k, probe.PASS)
            for k in probe.CONTROL_EXPECT_FAIL + probe.CONTROL_EXPECT_PASS
        ]
        problems = probe.check_positive_control(results)
        assert len(problems) == len(probe.CONTROL_EXPECT_FAIL)
        assert all("expected FAIL, got PASS" in p for p in problems)

    def test_a_control_op_that_was_never_probed_is_a_violation(self) -> None:
        problems = probe.check_positive_control([])
        assert len(problems) == 5
        assert all("was not probed at all" in p for p in problems)


class TestReporting:
    def _results(self) -> list[object]:
        return [
            probe.Result(
                operation=probe.Operation("inference", "GET", "/models", "listModels"),
                verdict=probe.FAIL,
                pinned=probe.Signature(404, "text/html", None),
                bogus=probe.Signature(404, "text/html", None),
                detail="unrecognized: x",
            ),
            probe.Result(
                operation=probe.Operation("inference", "POST", "/embed", "embed"),
                verdict=probe.PASS,
                pinned=probe.Signature(200, "application/json", None),
                bogus=probe.Signature(403, "text/plain", "FORBIDDEN"),
            ),
            probe.Result(
                operation=probe.Operation("db_control", "POST", "/indexes", "createIndex"),
                verdict=probe.SKIP,
                detail="mutating: ...",
            ),
        ]

    def test_counts_cover_every_row(self) -> None:
        assert probe.counts(self._results()) == {probe.PASS: 1, probe.FAIL: 1, probe.SKIP: 1}

    def test_text_report_lists_failures_first_and_states_the_oracle(self) -> None:
        text = probe.render_text(self._results(), control=[])
        assert "IDENTICAL" in text
        assert text.index("inference:GET /models") < text.index("inference:POST /embed")
        assert "1 PASS   1 FAIL   1 SKIP" in text

    def test_every_row_appears_in_every_renderer(self) -> None:
        results = self._results()
        for text in (
            probe.render_text(results, None),
            probe.render_markdown(results, None),
            probe.render_json(results, None),
        ):
            for result in results:
                assert result.key in text

    def test_json_report_is_machine_readable_and_carries_the_control(self) -> None:
        payload = json.loads(probe.render_json(self._results(), control=["boom"]))
        assert payload["api_version"] == probe.API_VERSION
        assert payload["bogus_version"] == probe.BOGUS_VERSION
        assert payload["positive_control"] == {"ok": False, "problems": ["boom"]}
        assert len(payload["operations"]) == 3

    def test_a_violated_control_is_shouted_in_both_human_renderers(self) -> None:
        for text in (
            probe.render_text(self._results(), control=["boom"]),
            probe.render_markdown(self._results(), control=["boom"]),
        ):
            assert "VIOLATED" in text
            assert "boom" in text


class TestReadOnlyByConstruction:
    def test_the_allowlist_only_names_operations_the_specs_declare(self) -> None:
        """A stale allowlist entry is dead weight that hides a real skip."""
        specs_dir = probe.DEFAULT_SPECS_DIR
        if not specs_dir.is_dir():
            pytest.skip(f"specs checkout not present at {specs_dir}")
        declared = set()
        for spec_file in specs_dir.glob(f"*_{probe.API_VERSION}.oas.yaml"):
            _, _, operations = probe.parse_spec(spec_file)
            declared.update(op.body_key for op in operations)
        assert set(probe.READ_ONLY_POSTS) <= declared

    def test_the_allowlist_holds_no_write_verbs(self) -> None:
        for key in probe.READ_ONLY_POSTS:
            assert key.split(":", 1)[1].startswith("POST ")
