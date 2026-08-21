#!/usr/bin/env python3
"""2026-07 API conformance coverage: report, gaps, verify, and the release gate.

The denominator is derived from the 2026-07 spec files themselves — every
operation in ``~/workspace/apis/_build/2026-07/*.oas.yaml`` (keyed
``<surface>:<operationId>``) plus every rpc in ``db_data_2026-07.proto``
(keyed ``db_data_grpc:<RpcName>``). Nothing is hand-maintained: a vendored
manifest (``tests/unit/conformance/manifest_2026-07.json``) lets the
``@api_op`` decorator validate claims in CI without the specs checkout, and
``--verify`` / ``--gate`` re-derive the denominator from the specs and fail
if the manifest has drifted.

Coverage is claimed by tests in ``tests/unit/conformance/`` decorated with
``@api_op("<surface>:<operationId>")``. Static modes (``--report``,
``--gaps``) count claims discovered by pytest collection; ``--verify`` and
``--gate`` count an operation as covered only when at least one claiming
test actually passed (see tests/unit/conformance/README.md for the
assertion contract each test must satisfy).

Each proto rpc entry records its service, request and response message
names, and — mirroring the HTTP ``success_body`` flag — whether the response
message declares any field, so a fieldless response (``DeleteResponse {}``)
is asserted as "the SDK returned None" rather than round-tripped through an
invented model.

The manifest also vendors, per HTTP operation with a success response body,
the OAS response schema (refs resolved, OAS 3.0 ``nullable`` translated,
declared objects sealed with ``additionalProperties: false``) so
``assert_roundtrip`` can validate each test's fixture against the spec
before round-tripping it. Operations where the SDK deliberately implements
backend behavior over the OAS carry a ``divergence`` entry — sourced from
the hand-maintained ``tests/unit/conformance/divergences_2026-07.json``,
each referencing an open SPEC-vs-BACKEND question issue — that switches
fixture validation to the documented alternative schema. No silent
exceptions: ``--gate`` checks every referenced issue is still an open
``question`` issue.

Usage:

    uv run python scripts/api_coverage.py --report
    uv run python scripts/api_coverage.py --gaps
    uv run python scripts/api_coverage.py --verify
    uv run python scripts/api_coverage.py --gate
    uv run python scripts/api_coverage.py --write-manifest

A handful of operations are cut from the release by decision rather than
left uncovered by accident. Those are named one at a time in
:data:`COVERAGE_EXEMPTIONS`, each pointing at the ``DECISION:`` comment that
cut it. An exemption excuses exactly the operation it names — a *different*
uncovered operation still fails the gate — and it stays visible in
``--report`` / ``--gaps`` output rather than disappearing from the
denominator, because an unexplained 101/102 is indistinguishable from a
regression. An exemption that has become stale (the operation gained
passing coverage after all) fails the gate too, so it gets deleted.

``--gate`` exits 0 only when all of the following hold:

1. every OAS operation and proto rpc is covered by a passing conformance
   test, except operations named in :data:`COVERAGE_EXEMPTIONS`
2. no claimed conformance test failed or was skipped, and no exemption is
   stale
3. every version constant in pinecone/_internal/constants.py is 2026-07 or
   excepted by a ``DECISION:`` comment on the epoch issue
4. rust/proto/db_data_2026-07.proto exists and rust/build.rs references it
5. the four :data:`RELEASE_MILESTONES` together have zero open issues, and
   every open ``release/2026-07`` issue is classified into that count or
   exempted from it — an unmilestoned, unlabelled ticket is invisible to the
   sum, so it fails the gate instead of passing quietly
6. every divergence exception and base-path override references an open
   ``question`` issue
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, NamedTuple
from urllib.parse import urlparse

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFORMANCE_DIR = REPO_ROOT / "tests" / "unit" / "conformance"
MANIFEST_PATH = CONFORMANCE_DIR / "manifest_2026-07.json"
DIVERGENCES_PATH = CONFORMANCE_DIR / "divergences_2026-07.json"
CONSTANTS_PATH = REPO_ROOT / "pinecone" / "_internal" / "constants.py"
RUST_PROTO_PATH = REPO_ROOT / "rust" / "proto" / "db_data_2026-07.proto"
RUST_BUILD_RS_PATH = REPO_ROOT / "rust" / "build.rs"

API_VERSION = "2026-07"
DEFAULT_SPECS_DIR = Path.home() / "workspace" / "apis" / "_build" / API_VERSION
DEFAULT_EPOCH_ISSUE = 87
GRPC_SURFACE = "db_data_grpc"
RESULTS_ENV = "PINECONE_CONFORMANCE_RESULTS"

_ISSUE_URL = "https://github.com/pinecone-io/python-sdk-internal/issues"

# db_metrics:fetch_prometheus_targets — the Prometheus service-discovery
# surface does not ship in 2026-07, so HTTP coverage lands at 101/102 by
# decision. Delete this entry when the surface is picked up.
COVERAGE_EXEMPTIONS: dict[str, str] = {
    "db_metrics:fetch_prometheus_targets": f"{_ISSUE_URL}/87#issuecomment-5365004482",
}

RELEASE_MILESTONES_DECISION = f"{_ISSUE_URL}/87#issuecomment-5365004630"
RELEASE_LABEL = f"release/{API_VERSION}"
_PAGE_SIZE = 100
_MAX_PAGES = 100
_ISSUE_LIMIT = 1000

# Milestones, not #87's sub-issue list: #87 hit GitHub's 100-sub-issue cap, so
# closing a ticket stopped moving that count and "zero open" could never
# become true. ``question`` issues carry no milestone and so are already
# outside this count — a label filter here would be papering over a
# mis-milestoned ticket.
RELEASE_MILESTONES: tuple[str, ...] = (
    f"{API_VERSION} M1: Foundations & version bumps",
    f"{API_VERSION} M2: Shared models & validation",
    f"{API_VERSION} M3: Transport implementations",
    f"{API_VERSION} M4: Graduation cleanup, docs & release notes",
)

OperationMap = dict[str, dict[str, Any]]

HTTP_METHODS = ("get", "put", "post", "delete", "options", "head", "patch", "trace")
_VERSION_VALUE_RE = re.compile(r"\d{4}-\d{2}")
_RPC_RE = re.compile(r"^\s*rpc\s+(\w+)\s*\(\s*([\w.]+)\s*\)\s*returns\s*\(\s*([\w.]+)\s*\)")
_SERVICE_RE = re.compile(r"^\s*service\s+(\w+)\s*\{")
_MESSAGE_RE = re.compile(r"^\s*message\s+(\w+)\s*\{")
_FIELD_DECL_RE = re.compile(
    r"^\s*(?:repeated\s+|optional\s+)?(?:map\s*<[^>]+>|[\w.]+)\s+\w+\s*=\s*\d+\s*[;\[]"
)


class SpecError(RuntimeError):
    """A spec file could not be parsed into an unambiguous operation list."""


_SHAPE_KEYWORDS = ("properties", "items", "additionalProperties", "allOf", "oneOf", "anyOf", "enum")
_ANNOTATION_KEYWORDS = frozenset(
    {"description", "example", "examples", "title", "externalDocs", "xml", "deprecated"}
)
_COMPOSITION_KEYWORDS = ("allOf", "anyOf", "oneOf", "not")
_CHILD_LIST_KEYWORDS = ("allOf", "anyOf", "oneOf")


def _deref(schema: Any, doc: dict[str, Any], seen: frozenset[str] = frozenset()) -> Any:
    while isinstance(schema, dict) and isinstance(schema.get("$ref"), str):
        ref = schema["$ref"]
        if not ref.startswith("#/") or ref in seen:
            return schema
        seen = seen | {ref}
        target: Any = doc
        for part in ref.removeprefix("#/").split("/"):
            if not isinstance(target, dict) or part not in target:
                return schema
            target = target[part]
        schema = target
    return schema


def _conveys_fields(schema: Any, doc: dict[str, Any]) -> bool:
    """Whether *schema* describes any field a model could carry or lose.

    A bare ``type: object`` with no ``properties`` — what db_data's
    ``DeleteResponse`` and ``CancelImportResponse`` are — describes nothing, so
    there is no round-trip to make. Counting it as a body would force a test to
    invent a throwaway empty model, which is the inflation ``success_body``
    exists to prevent.
    """
    schema = _deref(schema, doc)
    if not isinstance(schema, dict):
        return False
    if any(schema.get(keyword) for keyword in _SHAPE_KEYWORDS):
        return True
    return schema.get("type") not in (None, "object")


def success_response_schema(operation: dict[str, Any], doc: dict[str, Any]) -> Any:
    """The schema of *operation*'s success response body, or None.

    The first 2xx (lowest status) with a field-conveying media schema wins;
    ``application/json`` is preferred when a response declares several media
    types (the chat operations also declare ``text/event-stream``). ``None``
    means the operation answers with an empty body — 202/204 deletes and bare
    ``type: object`` acknowledgements — where inventing a throwaway model
    would only inflate coverage.
    """
    responses = operation.get("responses") or {}
    for status in sorted(responses, key=str):
        response = responses[status]
        if not str(status).startswith("2") or not isinstance(response, dict):
            continue
        content = response.get("content") or {}
        for media_type in sorted(content, key=lambda m: (m != "application/json", m)):
            media = content[media_type]
            if isinstance(media, dict) and _conveys_fields(media.get("schema"), doc):
                return media.get("schema")
    return None


class _SchemaBundler:
    """Turns one OAS 3.0 response schema into a self-contained JSON Schema.

    Five transformations, in order, all at manifest-generation time so the
    test-time validator stays a plain ``jsonschema`` call:

    - **refs resolved**: non-cyclic ``$ref``s are inlined; cyclic ones become
      local ``#/$defs/<Name>`` entries so recursion cannot loop.
    - **oneOf relaxed to anyOf**: see :func:`_relax_oneof` — with
      ``discriminator`` stripped, exactly-one-match would fail spec-valid
      payloads whose variants inline to overlapping schemas.
    - **allOf merged**: the specs use single-branch ``allOf`` as a nullable-
      wrapper idiom; branches that are plain object schemas merge into their
      parent. Anything unmergeable is kept verbatim (and stays unsealed).
    - **nullable translated**: OAS 3.0 ``nullable: true`` has no JSON Schema
      meaning, so it becomes ``type: [T, "null"]`` / an enum ``null`` / an
      ``anyOf`` wrapper; ``type`` and ``enum`` are widened independently
      when both are present.
    - **annotations stripped**: descriptions, examples, and ``x-*`` extensions
      carry no constraints and would bloat the vendored manifest.

    Sealing (``additionalProperties: false`` on objects that declare
    ``properties`` without addressing extras) happens in :func:`_seal` after
    bundling: it is what makes a fixture carrying keys the spec never declared
    fail validation instead of silently counting as coverage.
    """

    def __init__(self, doc: dict[str, Any]) -> None:
        self._doc = doc
        self._defs: dict[str, Any] = {}
        self._cyclic: set[str] = set()

    def bundle(self, schema: Any) -> dict[str, Any]:
        out = self._convert(schema, frozenset())
        if not isinstance(out, dict):
            raise SpecError(f"response schema did not convert to an object: {out!r}")
        if self._defs:
            out["$defs"] = dict(sorted(self._defs.items()))
        _seal(out)
        return out

    def _resolve(self, ref: str) -> Any:
        target: Any = self._doc
        for part in ref.removeprefix("#/").split("/"):
            if not isinstance(target, dict) or part not in target:
                raise SpecError(f"unresolvable $ref {ref!r}")
            target = target[part]
        return target

    def _convert(self, schema: Any, in_progress: frozenset[str]) -> Any:
        if not isinstance(schema, dict):
            return schema
        ref = schema.get("$ref")
        if isinstance(ref, str):
            if not ref.startswith("#/"):
                raise SpecError(f"external $ref {ref!r} is not supported")
            name = ref.rsplit("/", 1)[-1]
            if ref in in_progress:
                self._cyclic.add(ref)
                return {"$ref": f"#/$defs/{name}"}
            converted = self._convert(self._resolve(ref), in_progress | {ref})
            if ref in self._cyclic:
                self._defs[name] = converted
                return {"$ref": f"#/$defs/{name}"}
            return converted

        out: dict[str, Any] = {}
        for key, value in schema.items():
            if key in _ANNOTATION_KEYWORDS or key.startswith("x-") or key == "discriminator":
                continue
            if key == "properties" and isinstance(value, dict):
                out[key] = {k: self._convert(v, in_progress) for k, v in value.items()}
            elif key in ("items", "additionalProperties", "not") and isinstance(value, dict):
                out[key] = self._convert(value, in_progress)
            elif key in _CHILD_LIST_KEYWORDS and isinstance(value, list):
                out[key] = [self._convert(v, in_progress) for v in value]
            else:
                out[key] = value
        return _apply_nullable(_merge_allof(_relax_oneof(out)))


def _relax_oneof(node: dict[str, Any]) -> dict[str, Any]:
    """Rewrite ``oneOf`` as a deduplicated ``anyOf``.

    ``discriminator`` is stripped during bundling, so OAS variants that
    differed only by it inline to identical (or overlapping, once sealed)
    schemas — under ``oneOf``'s exactly-one-match rule a spec-valid payload
    then matches several branches and *fails*. The property fixture
    validation needs is "matches at least one documented variant", which is
    ``anyOf``.
    """
    branches = node.get("oneOf")
    if not isinstance(branches, list):
        return node
    deduped: list[Any] = list(node.get("anyOf", []))
    seen = {json.dumps(b, sort_keys=True) for b in deduped}
    for branch in branches:
        fingerprint = json.dumps(branch, sort_keys=True)
        if fingerprint not in seen:
            seen.add(fingerprint)
            deduped.append(branch)
    out = {key: value for key, value in node.items() if key != "oneOf"}
    out["anyOf"] = deduped
    return out


def _merge_allof(node: dict[str, Any]) -> dict[str, Any]:
    branches = node.get("allOf")
    if not isinstance(branches, list):
        return node
    if any(key in node for key in ("anyOf", "oneOf", "not", "$ref")):
        return node
    merged = {key: value for key, value in node.items() if key != "allOf"}
    if "properties" in merged:
        merged["properties"] = dict(merged["properties"])
    for branch in branches:
        if not isinstance(branch, dict):
            return node
        if any(key in branch for key in _COMPOSITION_KEYWORDS) or "$ref" in branch:
            return node
        if branch.get("type", "object") != "object":
            return node
        for key, value in branch.items():
            if key == "properties":
                properties = merged.setdefault("properties", {})
                for prop, prop_schema in value.items():
                    if prop in properties and properties[prop] != prop_schema:
                        return node
                    properties[prop] = prop_schema
            elif key == "required":
                merged["required"] = sorted(set(merged.get("required", [])) | set(value))
            elif key not in merged:
                merged[key] = value
            elif merged[key] != value:
                return node
    merged.setdefault("type", "object")
    return merged


def _apply_nullable(node: dict[str, Any]) -> dict[str, Any]:
    if node.pop("nullable", None) is not True:
        return node
    handled = False
    if isinstance(node.get("type"), str):
        node["type"] = [node["type"], "null"]
        handled = True
    if isinstance(node.get("enum"), list):
        if None not in node["enum"]:
            node["enum"] = [*node["enum"], None]
        handled = True
    if not handled:
        return {"anyOf": [node, {"type": "null"}]}
    return node


def _seal(node: Any) -> None:
    if isinstance(node, list):
        for child in node:
            _seal(child)
        return
    if not isinstance(node, dict):
        return
    if (
        "properties" in node
        and "additionalProperties" not in node
        and not any(key in node for key in _COMPOSITION_KEYWORDS)
    ):
        node["additionalProperties"] = False
    for key in ("properties", "$defs"):
        for child in node.get(key, {}).values():
            _seal(child)
    for key in ("items", "additionalProperties", "not"):
        if isinstance(node.get(key), dict):
            _seal(node[key])
    for key in _CHILD_LIST_KEYWORDS:
        _seal(node.get(key, []))


def server_base_path(doc: dict[str, Any], name: str) -> str:
    """The path component shared by every ``servers`` URL of a spec.

    ``assistant_control`` and ``assistant_evaluation`` mount their operations
    under ``/assistant``, so the path a request actually carries is this plus
    the operation's path. Recorded in the manifest so ``assert_request`` keeps
    comparing whole paths instead of suffixes.

    A spec whose ``servers`` URL omits a prefix the deployed surface really
    carries — ``assistant_data``, whose ``https://{assistant_host}`` hides the
    ``/assistant`` mount — is corrected by an issue-referenced entry in
    ``base_path_overrides`` (see :func:`load_base_path_overrides`), not here.
    """
    servers = doc.get("servers") or []
    bases = {
        urlparse(server["url"]).path.rstrip("/")
        for server in servers
        if isinstance(server, dict) and isinstance(server.get("url"), str)
    }
    if len(bases) > 1:
        raise SpecError(f"{name}: servers disagree on a base path: {sorted(bases)}")
    return bases.pop() if bases else ""


def _record_schema(
    schemas: dict[str, Any], key: str, bundled: dict[str, Any], context: str
) -> None:
    if key in schemas and schemas[key] != bundled:
        raise SpecError(f"{context}: schema key {key!r} maps to two different shapes")
    schemas[key] = bundled


def load_divergences() -> dict[str, dict[str, Any]]:
    """The hand-maintained divergence exception list, structurally validated.

    Every entry must name an operation, reference a question issue by number,
    give a reason, and name the alternative component schema the SDK actually
    implements — the manifest generator refuses anything less, so a silent or
    unattributed exception cannot exist.
    """
    if not DIVERGENCES_PATH.exists():
        return {}
    with DIVERGENCES_PATH.open() as f:
        doc = json.load(f)
    divergences = doc.get("divergences")
    if not isinstance(divergences, dict):
        raise SpecError(f"{DIVERGENCES_PATH.name}: top-level 'divergences' object is required")
    for op_id, entry in divergences.items():
        if not isinstance(entry, dict):
            raise SpecError(f"{DIVERGENCES_PATH.name}: {op_id}: entry must be an object")
        issue = entry.get("issue")
        if not isinstance(issue, int) or isinstance(issue, bool) or issue <= 0:
            raise SpecError(
                f"{DIVERGENCES_PATH.name}: {op_id}: 'issue' must reference a question "
                "issue by positive number — no silent divergence exceptions"
            )
        if not isinstance(entry.get("reason"), str) or not entry["reason"].strip():
            raise SpecError(f"{DIVERGENCES_PATH.name}: {op_id}: a non-empty 'reason' is required")
        if not isinstance(entry.get("alternative_schema"), str) or not entry["alternative_schema"]:
            raise SpecError(
                f"{DIVERGENCES_PATH.name}: {op_id}: 'alternative_schema' must name the "
                "component schema the SDK actually implements"
            )
        unknown = set(entry) - {"issue", "reason", "alternative_schema"}
        if unknown:
            raise SpecError(f"{DIVERGENCES_PATH.name}: {op_id}: unknown keys {sorted(unknown)}")
    return {str(op_id): dict(entry) for op_id, entry in divergences.items()}


def load_base_path_overrides() -> dict[str, dict[str, Any]]:
    """Per-surface corrections to the base path derived from ``servers``.

    Same contract as the divergence list: an override must name the surface,
    reference a question issue by number, and give a reason, so a spec the SDK
    deliberately contradicts on the *request* side stays as visible as one it
    contradicts on the response side.
    """
    if not DIVERGENCES_PATH.exists():
        return {}
    with DIVERGENCES_PATH.open() as f:
        doc = json.load(f)
    overrides = doc.get("base_path_overrides", {})
    if not isinstance(overrides, dict):
        raise SpecError(f"{DIVERGENCES_PATH.name}: 'base_path_overrides' must be an object")
    for surface, entry in overrides.items():
        if not isinstance(entry, dict):
            raise SpecError(f"{DIVERGENCES_PATH.name}: {surface}: entry must be an object")
        issue = entry.get("issue")
        if not isinstance(issue, int) or isinstance(issue, bool) or issue <= 0:
            raise SpecError(
                f"{DIVERGENCES_PATH.name}: {surface}: 'issue' must reference a question "
                "issue by positive number — no silent base-path overrides"
            )
        if not isinstance(entry.get("reason"), str) or not entry["reason"].strip():
            raise SpecError(f"{DIVERGENCES_PATH.name}: {surface}: a non-empty 'reason' is required")
        base_path = entry.get("base_path")
        if (
            not isinstance(base_path, str)
            or not base_path.startswith("/")
            or base_path != base_path.rstrip("/")
        ):
            raise SpecError(
                f"{DIVERGENCES_PATH.name}: {surface}: 'base_path' must be a path starting "
                "with '/' and carrying no trailing slash"
            )
        unknown = set(entry) - {"issue", "reason", "base_path"}
        if unknown:
            raise SpecError(f"{DIVERGENCES_PATH.name}: {surface}: unknown keys {sorted(unknown)}")
    return {str(surface): dict(entry) for surface, entry in overrides.items()}


def parse_oas_file(
    path: Path,
    divergences: dict[str, dict[str, Any]] | None = None,
    base_path_overrides: dict[str, dict[str, Any]] | None = None,
) -> tuple[OperationMap, dict[str, Any]]:
    surface = path.name.removesuffix(f"_{API_VERSION}.oas.yaml")
    divergences = divergences or {}
    with path.open() as f:
        doc = yaml.safe_load(f)
    spec_base_path = server_base_path(doc, path.name)
    override = (base_path_overrides or {}).get(surface)
    base_path = override["base_path"] if override is not None else spec_base_path
    ops: OperationMap = {}
    schemas: dict[str, Any] = {}
    for url_path, path_item in (doc.get("paths") or {}).items():
        if not isinstance(path_item, dict):
            continue
        for method in HTTP_METHODS:
            operation = path_item.get(method)
            if not isinstance(operation, dict):
                continue
            operation_id = operation.get("operationId")
            if not operation_id:
                raise SpecError(f"{path.name}: {method.upper()} {url_path} has no operationId")
            op_id = f"{surface}:{operation_id}"
            if op_id in ops:
                raise SpecError(f"{path.name}: duplicate operationId {operation_id!r}")
            schema = success_response_schema(operation, doc)
            entry: dict[str, Any] = {
                "kind": "http",
                "method": method.upper(),
                "base_path": base_path,
                "path": url_path,
                "success_body": schema is not None,
                "response_schema": None,
            }
            if override is not None:
                entry["base_path_divergence"] = {
                    "issue": override["issue"],
                    "reason": override["reason"],
                    "spec_base_path": spec_base_path,
                }
            if schema is not None:
                ref = schema.get("$ref") if isinstance(schema, dict) else None
                if isinstance(ref, str) and list(schema) == ["$ref"]:
                    key = f"{surface}:{ref.rsplit('/', 1)[-1]}"
                else:
                    key = f"{surface}:{operation_id}.response"
                _record_schema(schemas, key, _SchemaBundler(doc).bundle(schema), path.name)
                entry["response_schema"] = key
            divergence = divergences.get(op_id)
            if divergence is not None:
                component = divergence["alternative_schema"]
                components = (doc.get("components") or {}).get("schemas") or {}
                if component not in components:
                    raise SpecError(
                        f"{path.name}: divergence for {op_id} names alternative schema "
                        f"{component!r}, which is not in components/schemas"
                    )
                alt_key = f"{surface}:{component}"
                bundled = _SchemaBundler(doc).bundle({"$ref": f"#/components/schemas/{component}"})
                _record_schema(schemas, alt_key, bundled, path.name)
                entry["divergence"] = {
                    "issue": divergence["issue"],
                    "reason": divergence["reason"],
                    "response_schema": alt_key,
                }
            ops[op_id] = entry
    if not ops:
        raise SpecError(f"{path.name}: no operations found")
    return ops, schemas


def _proto_message_fields(text: str) -> dict[str, bool]:
    """Map each top-level proto message name to whether it declares any field.

    This is what gives gRPC rpcs the same ``success_body`` signal HTTP
    operations get from the OAS: an rpc answering with a fieldless message
    (``DeleteResponse {}``) has no round-trip to make, and a conformance test
    for it must assert the SDK returned ``None`` instead of inventing a
    throwaway model. The scanner strips ``//`` comments and tracks brace depth,
    so option blocks and ``oneof`` groups do not confuse it.
    """
    messages: dict[str, bool] = {}
    current: str | None = None
    depth = 0
    for raw in text.splitlines():
        line = raw.split("//", 1)[0]
        if current is None:
            message_match = _MESSAGE_RE.match(line)
            if message_match:
                name = message_match.group(1)
                if name in messages:
                    raise SpecError(f"duplicate message {name!r} in proto")
                remainder = line.split("{", 1)[1]
                messages[name] = bool(_FIELD_DECL_RE.match(remainder))
                current = name
                depth = line.count("{") - line.count("}")
                if depth <= 0:
                    current = None
            continue
        if _FIELD_DECL_RE.match(line):
            messages[current] = True
        depth += line.count("{") - line.count("}")
        if depth <= 0:
            current = None
    return messages


def parse_proto_file(path: Path) -> OperationMap:
    text = path.read_text()
    message_has_fields = _proto_message_fields(text)
    service = ""
    ops: OperationMap = {}
    for line in text.splitlines():
        service_match = _SERVICE_RE.match(line)
        if service_match:
            service = service_match.group(1)
            continue
        rpc_match = _RPC_RE.match(line)
        if rpc_match:
            if not service:
                raise SpecError(f"{path.name}: rpc {rpc_match.group(1)} outside any service")
            rpc, request, response = rpc_match.groups()
            op_id = f"{GRPC_SURFACE}:{rpc}"
            if op_id in ops:
                raise SpecError(f"{path.name}: duplicate rpc {rpc!r}")
            for message in (request, response):
                if message not in message_has_fields:
                    raise SpecError(
                        f"{path.name}: rpc {rpc} references message {message!r}, "
                        "which the proto does not define"
                    )
            ops[op_id] = {
                "kind": "grpc",
                "service": service,
                "rpc": rpc,
                "request": request,
                "response": response,
                "success_body": message_has_fields[response],
            }
    if not ops:
        raise SpecError(f"{path.name}: no rpcs found")
    return ops


def derive_manifest(specs_dir: Path) -> dict[str, Any]:
    oas_files = sorted(specs_dir.glob(f"*_{API_VERSION}.oas.yaml"))
    proto_file = specs_dir / f"db_data_{API_VERSION}.proto"
    if not oas_files:
        raise SpecError(f"no *_{API_VERSION}.oas.yaml files in {specs_dir}")
    if not proto_file.exists():
        raise SpecError(f"{proto_file} not found")
    divergences = load_divergences()
    base_path_overrides = load_base_path_overrides()
    ops: OperationMap = {}
    schemas: dict[str, Any] = {}
    surfaces: set[str] = set()
    for oas in oas_files:
        surfaces.add(oas.name.removesuffix(f"_{API_VERSION}.oas.yaml"))
        file_ops, file_schemas = parse_oas_file(oas, divergences, base_path_overrides)
        ops.update(file_ops)
        for key, bundled in file_schemas.items():
            _record_schema(schemas, key, bundled, oas.name)
    ops.update(parse_proto_file(proto_file))
    unconsumed = sorted(set(divergences) - set(ops))
    if unconsumed:
        raise SpecError(
            f"{DIVERGENCES_PATH.name} lists divergences for operations not in the specs: "
            f"{unconsumed}"
        )
    unknown_surfaces = sorted(set(base_path_overrides) - surfaces)
    if unknown_surfaces:
        raise SpecError(
            f"{DIVERGENCES_PATH.name} lists base-path overrides for surfaces not in the specs: "
            f"{unknown_surfaces}"
        )
    return {"api_version": API_VERSION, "operations": ops, "schemas": schemas}


def load_manifest() -> dict[str, Any]:
    with MANIFEST_PATH.open() as f:
        manifest: dict[str, Any] = json.load(f)
    return manifest


def render_manifest(manifest: dict[str, Any]) -> str:
    ops = manifest["operations"]
    schemas = manifest["schemas"]
    lines = [
        "{",
        f'  "api_version": "{API_VERSION}",',
        '  "generated_by": "uv run python scripts/api_coverage.py --write-manifest",',
        '  "operations": {',
    ]
    lines.append(
        ",\n".join(
            f"    {json.dumps(op_id)}: {json.dumps(ops[op_id], sort_keys=True)}"
            for op_id in sorted(ops)
        )
    )
    lines.extend(["  },", '  "schemas": {'])
    lines.append(
        ",\n".join(
            f"    {json.dumps(key)}: {json.dumps(schemas[key], sort_keys=True)}"
            for key in sorted(schemas)
        )
    )
    lines.extend(["  }", "}", ""])
    return "\n".join(lines)


def write_manifest(manifest: dict[str, Any], path: Path = MANIFEST_PATH) -> None:
    path.write_text(render_manifest(manifest))


def surface_of(op_id: str) -> str:
    return op_id.split(":", 1)[0]


def exempt_operations(ops: OperationMap) -> dict[str, str]:
    return {op_id: url for op_id, url in COVERAGE_EXEMPTIONS.items() if op_id in ops}


def exemption_problems(ops: OperationMap, covered: set[str]) -> list[str]:
    """Every reason an entry in :data:`COVERAGE_EXEMPTIONS` should be deleted.

    An exemption for an operation the specs no longer contain, or for one that
    has since gained passing coverage, is reported rather than ignored: a
    forgotten exemption is a hole in the denominator that nothing else would
    ever surface.
    """
    problems: list[str] = []
    for op_id, url in sorted(COVERAGE_EXEMPTIONS.items()):
        if op_id not in ops:
            problems.append(
                f"exemption names {op_id}, which is not in the {API_VERSION} specs — "
                f"delete the exemption ({url})"
            )
        elif op_id in covered:
            problems.append(
                f"exemption for {op_id} is STALE: the operation now has passing "
                f"conformance coverage — delete the exemption ({url})"
            )
    return problems


class CoverageStatus(NamedTuple):
    ok: bool
    detail: str
    problems: list[str]
    missing: list[str]


def coverage_status(ops: OperationMap, verified: set[str]) -> CoverageStatus:
    """Whether coverage clears the bar, and exactly which operations fall short.

    Exemptions excuse the operations they name and nothing else — ``missing``
    is a set difference, not a count allowance, so a different uncovered
    operation still fails.
    """
    exempt = exempt_operations(ops)
    problems = exemption_problems(ops, verified)
    missing = sorted(set(ops) - verified - set(exempt))
    detail = f"{len(verified)}/{len(ops)} operations verified by passing conformance tests"
    if exempt:
        detail += f"; {len(exempt)} deliberate omission(s): {', '.join(sorted(exempt))}"
    if missing:
        detail += f"; {len(missing)} uncovered (run --gaps)"
    return CoverageStatus(not missing and not problems, detail, problems, missing)


def run_conformance_suite(collect_only: bool) -> tuple[int, dict[str, Any]]:
    """Run pytest over tests/unit/conformance/ and return (exit_code, results).

    The conformance conftest writes a results JSON (claims per test plus
    outcomes) to the path named by ``PINECONE_CONFORMANCE_RESULTS``. With
    ``collect_only`` the outcomes stay ``"collected"`` — the static claims
    view used by --report / --gaps.
    """
    fd, results_path = tempfile.mkstemp(prefix="conformance-", suffix=".json")
    os.close(fd)
    cmd = [sys.executable, "-m", "pytest", str(CONFORMANCE_DIR), "-q", "-p", "no:cacheprovider"]
    if collect_only:
        cmd.append("--collect-only")
    env = {**os.environ, RESULTS_ENV: results_path}
    try:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
        results_file = Path(results_path)
        raw = results_file.read_text() if results_file.exists() else ""
        results: dict[str, Any] = json.loads(raw) if raw else {"tests": {}}
    finally:
        Path(results_path).unlink(missing_ok=True)
    if proc.returncode not in (0, 1, 5):
        print(proc.stdout, file=sys.stderr)
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"pytest exited with unexpected code {proc.returncode}")
    return proc.returncode, results


def claimed_ops(results: dict[str, Any]) -> dict[str, list[str]]:
    claims: dict[str, list[str]] = {}
    for nodeid, info in results["tests"].items():
        for op_id in info["ops"]:
            claims.setdefault(op_id, []).append(nodeid)
    return claims


def verified_ops(results: dict[str, Any]) -> set[str]:
    verified: set[str] = set()
    for info in results["tests"].values():
        if info["outcome"] == "passed":
            verified.update(info["ops"])
    return verified


def unverified_claimed_tests(results: dict[str, Any]) -> dict[str, str]:
    return {
        nodeid: info["outcome"]
        for nodeid, info in results["tests"].items()
        if info["outcome"] != "passed"
    }


def version_constants(source: str) -> dict[str, str]:
    """API-version constants assigned in a constants module.

    A constant counts when its name contains ``API_VERSION`` and its value
    looks like a version string (``YYYY-MM``) — this keeps header-name
    constants like ``API_VERSION_HEADER`` out of the gate.
    """
    constants: dict[str, str] = {}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets: list[ast.expr] = [node.target]
            value = node.value
        elif isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
        else:
            continue
        if not (isinstance(value, ast.Constant) and isinstance(value.value, str)):
            continue
        if not _VERSION_VALUE_RE.fullmatch(value.value):
            continue
        for target in targets:
            if isinstance(target, ast.Name) and "API_VERSION" in target.id:
                constants[target.id] = value.value
    return constants


def unexcepted_constants(constants: dict[str, str], decision_texts: list[str]) -> dict[str, str]:
    return {
        name: value
        for name, value in constants.items()
        if value != API_VERSION and not any(name in text for text in decision_texts)
    }


def _gh(args: list[str]) -> str:
    proc = subprocess.run(["gh", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    return proc.stdout


def fetch_decision_comments(epoch_issue: int) -> list[str]:
    payload = json.loads(_gh(["issue", "view", str(epoch_issue), "--json", "comments"]))
    return [
        comment["body"] for comment in payload.get("comments", []) if "DECISION:" in comment["body"]
    ]


def fetch_milestones() -> list[dict[str, Any]]:
    """Every milestone in the repo, open and closed, following pagination to the end.

    A page cap that truncates silently is precisely what broke the check this
    replaces. If a release milestone holding open issues fell off page one
    while the four expected titles were empty, the epoch condition would go
    green and the work would stay invisible — so this follows pages until one
    comes back short, and refuses to guess if that never happens.
    """
    milestones: list[dict[str, Any]] = []
    for page in range(1, _MAX_PAGES + 1):
        batch: list[dict[str, Any]] = json.loads(
            _gh(
                [
                    "api",
                    f"repos/:owner/:repo/milestones?state=all&per_page={_PAGE_SIZE}&page={page}",
                ]
            )
        )
        milestones.extend(batch)
        if len(batch) < _PAGE_SIZE:
            return milestones
    raise SpecError(
        f"milestone listing did not terminate within {_MAX_PAGES} pages of {_PAGE_SIZE} — "
        "refusing to report a possibly-truncated count"
    )


def release_milestone_status(milestones: list[dict[str, Any]]) -> tuple[bool, str]:
    """The epoch stop condition: zero open issues across the release milestones.

    Every milestone titled for this release counts, so a fifth one added later
    cannot hold open work the gate never looks at. On top of that, all four
    :data:`RELEASE_MILESTONES` must exist — a renamed or deleted milestone would
    otherwise drop silently out of the sum and turn the gate green by making
    work invisible, which is exactly what the sub-issue cap did.
    """
    counted = {
        str(m.get("title")): int(m.get("open_issues") or 0)
        for m in milestones
        if str(m.get("title") or "").startswith(API_VERSION)
    }
    missing = [title for title in RELEASE_MILESTONES if title not in counted]
    if missing:
        return False, (
            f"milestone(s) not found in this repo: {'; '.join(missing)} — the open-issue "
            f"sum cannot be trusted (decision: {RELEASE_MILESTONES_DECISION})"
        )
    breakdown = ", ".join(
        f"{title.split(':', 1)[0]} {count}" for title, count in sorted(counted.items())
    )
    total = sum(counted.values())
    if total:
        return False, (
            f"{total} open issues across the {len(counted)} {API_VERSION} milestones "
            f"({breakdown}); run `gh issue list --milestone <title>` for the list"
        )
    return True, (
        f"zero open issues across the {len(counted)} {API_VERSION} milestones ({breakdown})"
    )


def fetch_open_release_issues() -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = json.loads(
        _gh(
            [
                "issue",
                "list",
                "--state",
                "open",
                "--label",
                RELEASE_LABEL,
                "--limit",
                str(_ISSUE_LIMIT),
                "--json",
                "number,labels,milestone",
            ]
        )
    )
    if len(issues) >= _ISSUE_LIMIT:
        raise SpecError(
            f"open {RELEASE_LABEL} issues hit the {_ISSUE_LIMIT} fetch limit, so the listing "
            "may be truncated — raise the limit rather than triaging a partial list"
        )
    return issues


def unclassified_release_issues(issues: list[dict[str, Any]], epoch_issue: int) -> list[int]:
    """Open release issues the milestone sum cannot see.

    Two states are legitimate: a milestone :func:`release_milestone_status`
    actually counts, or the ``question`` label, which marks an upstream
    SPEC-vs-BACKEND item as out of scope. Anything else is invisible to the
    sum — and that includes a release ticket parked on some *other* milestone
    (``9.2.0``), which reads as classified while being counted by nobody.

    The test is therefore "does the sum see it", not "does it have a
    milestone": these two functions must agree on which milestones count, or
    the gap between them is where work disappears. "Question issues carry no
    milestone" does not give the converse either, so an unlabelled finding
    filed mid-task is invisible too — that is the sub-issue cap's failure one
    ticket at a time, a bet on filing discipline rather than a measure. The
    epoch issue is the one legitimate exception: it is the parent, not a work
    ticket.
    """
    unclassified: list[int] = []
    for issue in issues:
        number = int(issue["number"])
        if number == epoch_issue:
            continue
        milestone = issue.get("milestone") or {}
        if str(milestone.get("title") or "").startswith(API_VERSION):
            continue
        if "question" in {label["name"] for label in issue.get("labels") or []}:
            continue
        unclassified.append(number)
    return sorted(unclassified)


def rust_proto_wired() -> tuple[bool, str]:
    if not RUST_PROTO_PATH.exists():
        return False, f"{RUST_PROTO_PATH.relative_to(REPO_ROOT)} does not exist"
    if RUST_PROTO_PATH.name not in RUST_BUILD_RS_PATH.read_text():
        return False, f"rust/build.rs does not reference {RUST_PROTO_PATH.name}"
    return True, "vendored proto present and wired in rust/build.rs"


def resolve_operations(specs_dir: Path, *, strict: bool) -> OperationMap:
    """The operation map, preferring live specs and checking manifest drift.

    ``strict`` (used by --verify / --gate) requires the specs checkout; the
    read-only modes fall back to the vendored manifest so they work in CI.
    Drift is checked over operations AND vendored response schemas — a stale
    schema would quietly validate fixtures against yesterday's contract.
    """
    if specs_dir.is_dir():
        derived = derive_manifest(specs_dir)
        if not MANIFEST_PATH.exists():
            raise SpecError(
                f"{MANIFEST_PATH.relative_to(REPO_ROOT)} missing — run --write-manifest"
            )
        vendored = load_manifest()
        for section in ("operations", "schemas"):
            if vendored.get(section) != derived[section]:
                raise SpecError(
                    f"vendored manifest {section} are stale relative to the specs in "
                    f"{specs_dir} — run --write-manifest and commit the result"
                )
        operations: OperationMap = derived["operations"]
        return operations
    if strict:
        raise SpecError(f"specs directory {specs_dir} not found (pass --specs-dir)")
    print(f"note: {specs_dir} not found; using vendored manifest", file=sys.stderr)
    manifest_operations: OperationMap = load_manifest()["operations"]
    return manifest_operations


def print_report(ops: OperationMap, covered: set[str], basis: str) -> None:
    surfaces: dict[str, list[str]] = {}
    for op_id in ops:
        surfaces.setdefault(surface_of(op_id), []).append(op_id)
    exempt = exempt_operations(ops)
    print(f"conformance coverage ({basis})")
    width = max(len(s) for s in surfaces)
    for surface in sorted(surfaces):
        total = len(surfaces[surface])
        done = sum(1 for op_id in surfaces[surface] if op_id in covered)
        omitted = sum(1 for op_id in surfaces[surface] if op_id in exempt)
        suffix = f"   ({omitted} deliberate omission)" if omitted else ""
        print(f"  {surface:<{width}}  {done:>3} / {total}{suffix}")
    http_ops = [op for op, entry in ops.items() if entry["kind"] == "http"]
    grpc_ops = [op for op, entry in ops.items() if entry["kind"] == "grpc"]
    http_done = sum(1 for op in http_ops if op in covered)
    grpc_done = sum(1 for op in grpc_ops if op in covered)
    print(f"overall http: {http_done}/{len(http_ops)}")
    print(f"overall grpc: {grpc_done}/{len(grpc_ops)}")
    for op_id, url in sorted(exempt.items()):
        print(f"deliberate omission: {op_id} — {url}")


def mode_report(ops: OperationMap) -> int:
    _, results = run_conformance_suite(collect_only=True)
    print_report(ops, set(claimed_ops(results)), basis="claimed by collected tests")
    return 0


def mode_gaps(ops: OperationMap) -> int:
    _, results = run_conformance_suite(collect_only=True)
    claims = claimed_ops(results)
    exempt = exempt_operations(ops)
    for op_id in sorted(ops):
        if op_id in claims:
            continue
        if op_id in exempt:
            print(f"{op_id}\t[deliberate omission — {exempt[op_id]}]")
        else:
            print(op_id)
    return 0


def run_verify(ops: OperationMap) -> tuple[bool, set[str], list[str]]:
    exit_code, results = run_conformance_suite(collect_only=False)
    problems: list[str] = []
    if exit_code not in (0, 5):
        problems.append(f"pytest exited {exit_code} (test failures in tests/unit/conformance/)")
    for nodeid, outcome in sorted(unverified_claimed_tests(results).items()):
        problems.append(f"claimed test did not pass ({outcome}): {nodeid}")
    unknown = set(claimed_ops(results)) - set(ops)
    problems.extend(
        f"claim for operation not in the 2026-07 specs: {op_id}" for op_id in sorted(unknown)
    )
    verified = verified_ops(results) & set(ops)
    return not problems, verified, problems


def mode_verify(ops: OperationMap) -> int:
    green, verified, problems = run_verify(ops)
    print_report(ops, verified, basis="verified by passing tests")
    for problem in problems:
        print(f"FAIL: {problem}")
    return 0 if green else 1


def mode_gate(ops: OperationMap, epoch_issue: int) -> int:
    conditions: list[tuple[bool, str]] = []

    green, verified, problems = run_verify(ops)
    status = coverage_status(ops, verified)
    conditions.append((green and status.ok, f"conformance: {status.detail}"))
    conditions.extend(
        (False, f"conformance: {problem}") for problem in [*problems, *status.problems]
    )

    constants = version_constants(CONSTANTS_PATH.read_text())
    try:
        decisions = fetch_decision_comments(epoch_issue)
        stale = unexcepted_constants(constants, decisions)
        if stale:
            conditions.extend(
                (
                    False,
                    f"constants: {name} = {value!r} is not {API_VERSION} and has no "
                    f"DECISION exception on #{epoch_issue}",
                )
                for name, value in sorted(stale.items())
            )
        else:
            conditions.append(
                (True, f"constants: all version constants at {API_VERSION} or excepted")
            )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        conditions.append((False, f"constants: could not fetch DECISION comments via gh: {exc}"))

    wired, why = rust_proto_wired()
    conditions.append((wired, f"rust proto: {why}"))

    divergent = {
        op_id: entry["divergence"] for op_id, entry in ops.items() if entry.get("divergence")
    }
    for surface, override in sorted(load_base_path_overrides().items()):
        divergent[f"{surface}:<base_path>"] = override
    for op_id, divergence in sorted(divergent.items()):
        issue = divergence["issue"]
        try:
            payload = json.loads(_gh(["issue", "view", str(issue), "--json", "state,labels"]))
            labels = {label["name"] for label in payload.get("labels", [])}
            if payload.get("state") != "OPEN":
                conditions.append(
                    (
                        False,
                        f"divergence: {op_id} references issue #{issue}, which is "
                        f"{payload.get('state')}; resolve the divergence or reopen the question",
                    )
                )
            elif "question" not in labels:
                conditions.append(
                    (False, f"divergence: {op_id} references #{issue}, not a 'question' issue")
                )
            else:
                conditions.append(
                    (True, f"divergence: {op_id} references open question issue #{issue}")
                )
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            conditions.append((False, f"divergence: could not check #{issue} via gh: {exc}"))
    if not divergent:
        conditions.append((True, "divergence: no divergence exceptions registered"))

    try:
        ok, message = release_milestone_status(fetch_milestones())
        conditions.append((ok, f"epoch: {message}"))
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        KeyError,
        ValueError,
        SpecError,
    ) as exc:
        conditions.append((False, f"epoch: could not list milestones via gh: {exc}"))

    try:
        unclassified = unclassified_release_issues(fetch_open_release_issues(), epoch_issue)
        if unclassified:
            numbers = ", ".join(f"#{number}" for number in unclassified)
            conditions.append(
                (
                    False,
                    f"triage: {len(unclassified)} open {RELEASE_LABEL} issue(s) carry neither a "
                    f"{API_VERSION} milestone nor the 'question' label, so the milestone sum "
                    f"cannot see them: {numbers}",
                )
            )
        else:
            conditions.append(
                (
                    True,
                    f"triage: every open {RELEASE_LABEL} issue is on a {API_VERSION} milestone "
                    f"or marked 'question' (#{epoch_issue} excepted as the epoch parent)",
                )
            )
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        KeyError,
        ValueError,
        SpecError,
    ) as exc:
        conditions.append((False, f"triage: could not list {RELEASE_LABEL} issues via gh: {exc}"))

    all_ok = all(ok for ok, _ in conditions)
    for ok, message in conditions:
        print(f"[{'PASS' if ok else 'FAIL'}] {message}")
    print(f"gate: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--report", action="store_true", help="covered/total per surface")
    mode.add_argument("--gaps", action="store_true", help="uncovered operation ids, one per line")
    mode.add_argument(
        "--verify", action="store_true", help="run the conformance suite; claimed tests must pass"
    )
    mode.add_argument("--gate", action="store_true", help="the epoch stop condition (exit 0/1)")
    mode.add_argument(
        "--write-manifest",
        action="store_true",
        help="regenerate tests/unit/conformance/manifest_2026-07.json from the specs",
    )
    parser.add_argument(
        "--specs-dir",
        type=Path,
        default=DEFAULT_SPECS_DIR,
        help=f"2026-07 spec checkout (default: {DEFAULT_SPECS_DIR})",
    )
    parser.add_argument(
        "--epoch-issue",
        type=int,
        default=DEFAULT_EPOCH_ISSUE,
        help=f"epoch issue number for the DECISION-comment check (default: {DEFAULT_EPOCH_ISSUE})",
    )
    args = parser.parse_args(argv)

    try:
        if args.write_manifest:
            if not args.specs_dir.is_dir():
                raise SpecError(f"specs directory {args.specs_dir} not found")
            manifest = derive_manifest(args.specs_dir)
            write_manifest(manifest)
            ops = manifest["operations"]
            http = sum(1 for entry in ops.values() if entry["kind"] == "http")
            grpc = len(ops) - http
            print(
                f"wrote {MANIFEST_PATH.relative_to(REPO_ROOT)} ({http} http ops, {grpc} rpcs, "
                f"{len(manifest['schemas'])} response schemas)"
            )
            return 0

        strict = args.verify or args.gate
        ops = resolve_operations(args.specs_dir, strict=strict)
    except SpecError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.report:
        return mode_report(ops)
    if args.gaps:
        return mode_gaps(ops)
    if args.verify:
        return mode_verify(ops)
    return mode_gate(ops, args.epoch_issue)


if __name__ == "__main__":
    sys.exit(main())
