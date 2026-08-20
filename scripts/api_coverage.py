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

Usage:

    uv run python scripts/api_coverage.py --report
    uv run python scripts/api_coverage.py --gaps
    uv run python scripts/api_coverage.py --verify
    uv run python scripts/api_coverage.py --gate
    uv run python scripts/api_coverage.py --write-manifest

``--gate`` exits 0 only when all of the following hold:

1. every OAS operation and proto rpc is covered by a passing conformance test
2. no claimed conformance test failed or was skipped
3. every version constant in pinecone/_internal/constants.py is 2026-07 or
   excepted by a ``DECISION:`` comment on the epoch issue
4. rust/proto/db_data_2026-07.proto exists and rust/build.rs references it
5. the epoch issue has zero open sub-issues
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
from typing import Any
from urllib.parse import urlparse

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFORMANCE_DIR = REPO_ROOT / "tests" / "unit" / "conformance"
MANIFEST_PATH = CONFORMANCE_DIR / "manifest_2026-07.json"
CONSTANTS_PATH = REPO_ROOT / "pinecone" / "_internal" / "constants.py"
RUST_PROTO_PATH = REPO_ROOT / "rust" / "proto" / "db_data_2026-07.proto"
RUST_BUILD_RS_PATH = REPO_ROOT / "rust" / "build.rs"

API_VERSION = "2026-07"
DEFAULT_SPECS_DIR = Path.home() / "workspace" / "apis" / "_build" / API_VERSION
DEFAULT_EPOCH_ISSUE = 87
GRPC_SURFACE = "db_data_grpc"
RESULTS_ENV = "PINECONE_CONFORMANCE_RESULTS"

OperationMap = dict[str, dict[str, Any]]

HTTP_METHODS = ("get", "put", "post", "delete", "options", "head", "patch", "trace")
_VERSION_VALUE_RE = re.compile(r"\d{4}-\d{2}")
_RPC_RE = re.compile(r"^\s*rpc\s+(\w+)\s*\(")
_SERVICE_RE = re.compile(r"^\s*service\s+(\w+)\s*\{")


class SpecError(RuntimeError):
    """A spec file could not be parsed into an unambiguous operation list."""


def declares_success_body(operation: dict[str, Any]) -> bool:
    """Whether any 2xx response of *operation* declares a response body.

    Recorded in the manifest so ``ClaimRecorder`` can distinguish an operation
    with a response schema (round-trip mandatory) from one that answers with an
    empty body, where inventing a throwaway model would only inflate coverage.
    """
    for status, response in (operation.get("responses") or {}).items():
        if not str(status).startswith("2") or not isinstance(response, dict):
            continue
        if response.get("content"):
            return True
    return False


def server_base_path(doc: dict[str, Any], name: str) -> str:
    """The path component shared by every ``servers`` URL of a spec.

    ``assistant_control`` and ``assistant_evaluation`` mount their operations
    under ``/assistant``, so the path a request actually carries is this plus
    the operation's path. Recorded in the manifest so ``assert_request`` keeps
    comparing whole paths instead of suffixes.
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


def parse_oas_file(path: Path) -> OperationMap:
    surface = path.name.removesuffix(f"_{API_VERSION}.oas.yaml")
    with path.open() as f:
        doc = yaml.safe_load(f)
    base_path = server_base_path(doc, path.name)
    ops: OperationMap = {}
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
            ops[op_id] = {
                "kind": "http",
                "method": method.upper(),
                "base_path": base_path,
                "path": url_path,
                "success_body": declares_success_body(operation),
            }
    if not ops:
        raise SpecError(f"{path.name}: no operations found")
    return ops


def parse_proto_file(path: Path) -> OperationMap:
    service = ""
    ops: OperationMap = {}
    for line in path.read_text().splitlines():
        service_match = _SERVICE_RE.match(line)
        if service_match:
            service = service_match.group(1)
            continue
        rpc_match = _RPC_RE.match(line)
        if rpc_match:
            if not service:
                raise SpecError(f"{path.name}: rpc {rpc_match.group(1)} outside any service")
            rpc = rpc_match.group(1)
            op_id = f"{GRPC_SURFACE}:{rpc}"
            if op_id in ops:
                raise SpecError(f"{path.name}: duplicate rpc {rpc!r}")
            ops[op_id] = {"kind": "grpc", "service": service, "rpc": rpc}
    if not ops:
        raise SpecError(f"{path.name}: no rpcs found")
    return ops


def derive_operations(specs_dir: Path) -> OperationMap:
    oas_files = sorted(specs_dir.glob(f"*_{API_VERSION}.oas.yaml"))
    proto_file = specs_dir / f"db_data_{API_VERSION}.proto"
    if not oas_files:
        raise SpecError(f"no *_{API_VERSION}.oas.yaml files in {specs_dir}")
    if not proto_file.exists():
        raise SpecError(f"{proto_file} not found")
    ops: OperationMap = {}
    for oas in oas_files:
        ops.update(parse_oas_file(oas))
    ops.update(parse_proto_file(proto_file))
    return ops


def load_manifest() -> OperationMap:
    with MANIFEST_PATH.open() as f:
        manifest = json.load(f)
    operations: OperationMap = manifest["operations"]
    return operations


def write_manifest(ops: OperationMap) -> None:
    lines = [
        "{",
        f'  "api_version": "{API_VERSION}",',
        '  "generated_by": "uv run python scripts/api_coverage.py --write-manifest",',
        '  "operations": {',
    ]
    entries = [
        f"    {json.dumps(op_id)}: {json.dumps(ops[op_id], sort_keys=True)}"
        for op_id in sorted(ops)
    ]
    lines.append(",\n".join(entries))
    lines.extend(["  }", "}", ""])
    MANIFEST_PATH.write_text("\n".join(lines))


def surface_of(op_id: str) -> str:
    return op_id.split(":", 1)[0]


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


def fetch_open_subissues(epoch_issue: int) -> list[dict[str, Any]]:
    repo = json.loads(_gh(["repo", "view", "--json", "nameWithOwner"]))["nameWithOwner"]
    owner, name = repo.split("/", 1)
    query = """
    query($owner: String!, $name: String!, $number: Int!, $cursor: String) {
      repository(owner: $owner, name: $name) {
        issue(number: $number) {
          subIssues(first: 100, after: $cursor) {
            pageInfo { hasNextPage endCursor }
            nodes { number state title }
          }
        }
      }
    }
    """
    open_issues: list[dict[str, Any]] = []
    cursor: str | None = None
    while True:
        args = [
            "api",
            "graphql",
            "-f",
            f"query={query}",
            "-f",
            f"owner={owner}",
            "-f",
            f"name={name}",
            "-F",
            f"number={epoch_issue}",
        ]
        if cursor:
            args += ["-f", f"cursor={cursor}"]
        data = json.loads(_gh(args))
        sub = data["data"]["repository"]["issue"]["subIssues"]
        open_issues.extend(n for n in sub["nodes"] if n["state"] == "OPEN")
        if not sub["pageInfo"]["hasNextPage"]:
            break
        cursor = sub["pageInfo"]["endCursor"]
    return open_issues


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
    """
    if specs_dir.is_dir():
        ops = derive_operations(specs_dir)
        if not MANIFEST_PATH.exists():
            raise SpecError(
                f"{MANIFEST_PATH.relative_to(REPO_ROOT)} missing — run --write-manifest"
            )
        if load_manifest() != ops:
            raise SpecError(
                "vendored manifest is stale relative to the specs in "
                f"{specs_dir} — run --write-manifest and commit the result"
            )
        return ops
    if strict:
        raise SpecError(f"specs directory {specs_dir} not found (pass --specs-dir)")
    print(f"note: {specs_dir} not found; using vendored manifest", file=sys.stderr)
    return load_manifest()


def print_report(ops: OperationMap, covered: set[str], basis: str) -> None:
    surfaces: dict[str, list[str]] = {}
    for op_id in ops:
        surfaces.setdefault(surface_of(op_id), []).append(op_id)
    print(f"conformance coverage ({basis})")
    width = max(len(s) for s in surfaces)
    for surface in sorted(surfaces):
        total = len(surfaces[surface])
        done = sum(1 for op_id in surfaces[surface] if op_id in covered)
        print(f"  {surface:<{width}}  {done:>3} / {total}")
    http_ops = [op for op, entry in ops.items() if entry["kind"] == "http"]
    grpc_ops = [op for op, entry in ops.items() if entry["kind"] == "grpc"]
    http_done = sum(1 for op in http_ops if op in covered)
    grpc_done = sum(1 for op in grpc_ops if op in covered)
    print(f"overall http: {http_done}/{len(http_ops)}")
    print(f"overall grpc: {grpc_done}/{len(grpc_ops)}")


def mode_report(ops: OperationMap) -> int:
    _, results = run_conformance_suite(collect_only=True)
    print_report(ops, set(claimed_ops(results)), basis="claimed by collected tests")
    return 0


def mode_gaps(ops: OperationMap) -> int:
    _, results = run_conformance_suite(collect_only=True)
    claims = claimed_ops(results)
    for op_id in sorted(ops):
        if op_id not in claims:
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
    missing = sorted(set(ops) - verified)
    coverage_ok = green and not missing
    detail = f"{len(verified)}/{len(ops)} operations verified by passing conformance tests"
    if missing:
        detail += f"; {len(missing)} uncovered (run --gaps)"
    conditions.append((coverage_ok, f"conformance: {detail}"))
    conditions.extend((False, f"conformance: {problem}") for problem in problems)

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

    try:
        open_subissues = fetch_open_subissues(epoch_issue)
        if open_subissues:
            numbers = ", ".join(f"#{issue['number']}" for issue in open_subissues)
            conditions.append(
                (
                    False,
                    f"epoch: {len(open_subissues)} open sub-issues on #{epoch_issue}: {numbers}",
                )
            )
        else:
            conditions.append((True, f"epoch: zero open sub-issues on #{epoch_issue}"))
    except (subprocess.CalledProcessError, FileNotFoundError, KeyError) as exc:
        conditions.append((False, f"epoch: could not list sub-issues via gh: {exc}"))

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
        help=f"epoch issue number for DECISION / sub-issue checks (default: {DEFAULT_EPOCH_ISSUE})",
    )
    args = parser.parse_args(argv)

    try:
        if args.write_manifest:
            if not args.specs_dir.is_dir():
                raise SpecError(f"specs directory {args.specs_dir} not found")
            ops = derive_operations(args.specs_dir)
            write_manifest(ops)
            http = sum(1 for entry in ops.values() if entry["kind"] == "http")
            grpc = len(ops) - http
            print(f"wrote {MANIFEST_PATH.relative_to(REPO_ROOT)} ({http} http ops, {grpc} rpcs)")
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
