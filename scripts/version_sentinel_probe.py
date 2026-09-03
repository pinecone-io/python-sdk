#!/usr/bin/env python3
"""Bogus-version sentinel probe: does the server actually know the version we pin?

Run it::

    uv run python scripts/version_sentinel_probe.py
    uv run python scripts/version_sentinel_probe.py --positive-control
    uv run python scripts/version_sentinel_probe.py --markdown >> "$GITHUB_STEP_SUMMARY"

WHY THIS EXISTS
===============

``scripts/api_coverage.py`` answers "does the SDK speak the documented
shape". It asserts method, path, the ``X-Pinecone-Api-Version`` header, and
schema round-trip **against fixtures**. A fixture cannot be stale, so
conformance is *structurally* blind to the question this script asks:

    does the deployment we are pointed at actually recognise ``2026-07``?

Both are needed. During the 2026-07 epoch conformance reported
``assistant_control`` 5/5 and ``inference`` 4/4 while, in production,
``GET /assistant/assistants`` returned ``403 Invalid API version`` (#312)
and ``GET /models`` returned a bare empty ``404`` (#319) — for weeks. Both
were found by accident. Green scorecards over a dead deployment is the exact
failure this script is designed to make impossible to miss.

THE ORACLE — AND WHY IT IS A *COMPARISON*, NOT A STATUS-CODE CHECK
=================================================================

For each operation declared in ``_build/2026-07/*.oas.yaml``, send the same
minimal request **twice**: once with ``X-Pinecone-Api-Version: 2026-07``
(whatever the SDK pins) and once with a deliberately invalid sentinel such
as ``9999-99``. Then compare the reduced signature::

    (status, content-type, error-code)

**Identical signature => FAIL: the server did not distinguish our pinned
version from a version that cannot possibly exist.**

The next maintainer's instinct will be "just assert 200" or "just assert not
403". Both are wrong, and each of tonight's two findings breaks one of them:

* A status check cannot see #319. ``GET /models`` returns ``404`` at
  ``2026-07`` because ``api_version_from_headers()`` did
  ``.unwrap_or_default()`` over an unknown version string and
  ``#[default]`` sits on ``V202404``, which routes to an empty router. There
  is no error, no gate, and nothing anomalous about a 404 on a REST API. The
  only thing that betrays it is that ``9999-99`` — a string no server should
  ever accept — produces *byte-identical* output.
* A "must not be 403" check cannot see a healthy server either: a correct
  deployment answers the *sentinel* with ``403 Unsupported API version
  '9999-99'``. 403 is the signal of health at the bogus version and the
  signal of sickness at the pinned one. Only the comparison knows which is
  which.

The comparison is also indifferent to failure mode, which is why one check
covers both findings: #312's loud middleware rejection and #319's silent
misclassification both collapse to "the two responses are the same".

WHAT IS DELIBERATELY EXCLUDED FROM THE SIGNATURE
------------------------------------------------

**The error message.** This is the subtle part, so it is worth spelling out.
Today prod's KE control plane answers both versions with the bare string
``Invalid API version``, so messages happen to match. But that constant was
deleted upstream on 2026-07-10 in favour of an actionable
``Unsupported API version '<v>'. Supported versions: ...`` — which *echoes
the version you sent*. The moment KE deploys any build carrying the new
message but still lacking ``V202607`` in its enum, a message-sensitive
comparison would see ``'2026-07'`` vs ``'9999-99'``, call the two responses
different, and report **PASS on a server that rejects us**. Interpolating
the request back into the response makes message text a mirror, not
evidence. The same reasoning excludes date/trace/request-id headers.

**The response body — for a second, stronger, and independent reason:
at least one probed operation is genuinely nondeterministic.** #348 called
``POST /evaluation/metrics/alignment`` six times at a single *fixed*
version and got two distinct bodies back (``alignment: 1.0`` and
``alignment: 0.0``) — LLM-sampling noise inside the eval service, not a
version effect. A body-sensitive oracle would disagree with itself on that
endpoint roughly five runs in six, reporting an intermittent, unreproducible
FAIL on a server that is doing nothing wrong — the hardest kind of false
positive to diagnose, because rerunning it can just as easily "confirm" the
false alarm as dispel it. **Do not add response-body comparison, including
behind a flag: an oracle that is wrong on a nondeterministic endpoint five-
sixths of the time is worse than one that is silent on bodies altogether.**

This is not the only body-shaped trap #348 found, and the other one is why
body comparison would not even be sufficient if it were safe: byte-identical
bodies do not prove a version is respected. ``GET /bulk/imports`` and
``GET /bulk/imports/{id}`` compare byte-identical at ``2026-07`` and at the
bogus version — but the bodies are degenerate (``{"data": []}``, a plain
"not found" 404), and an empty list or a 404 would also compare equal on a
*correctly* gated endpoint with nothing to return. Identity there is
necessary, not sufficient, so it was never a candidate for the signature.
What actually distinguished "this route ignores the version header" from
"this route is gated and legitimately has nothing to show" was a same-host
differential outside this script: on the identical host/auth/transcoding
path, ``GET /vectors/list`` and ``GET /namespaces`` answer the bogus version
with ``403 {"code": 7, "message": "Unsupported API version..."}`` while
``/bulk/imports`` answers ``200``. If a future body-shaped ambiguity needs
resolving, that differential against a known-gated sibling on the same host
— not a body comparison in this probe — is the technique, and #348 is the
worked example.

Content-type is kept, and normalised to a bare media type (parameters like
``charset`` dropped): it is what separates #319's ``text/html`` blank page
from a structured ``application/json`` error, and dropping parameters only
ever makes two responses *more* likely to compare equal — the conservative
direction, since equality is what raises the alarm.

Error code is kept, and read from all three envelopes prod actually uses:
``{"error": {"code": "FORBIDDEN"}}`` (control plane), ``{"code": 7}``
(data plane, gRPC-derived), and ``{"error": "access_denied"}`` (the OAuth
host). An absent code is ``None``, which compares fine.

DIAGNOSES ON FAILURE
--------------------

A FAIL carries one of two labels, derived purely from the pinned response —
no expectations, nothing to keep up to date:

``unrecognized``
    The pinned request failed and the sentinel failed the same way. #312 and
    #319 are both this. Read the printed signatures: if the pinned side is a
    ``401``/``UNAUTHENTICATED``, the probe may simply have been stopped at
    the auth layer before any version gate could speak, which is a weaker
    (but still unresolved) result rather than a proven version bug.

``unversioned``
    The pinned request *succeeded* and the sentinel succeeded identically —
    the route never reads the version header at all. Looks healthy, is not:
    we have no evidence the response we got carries 2026-07 semantics rather
    than some default epoch's. This is #319's failure mode wearing a 200.

SKIPS ARE REPORTED, NEVER OMITTED
=================================

#295 was a green CI run over a ~94% inert suite: the tests skipped
themselves and the *absence* was the whole problem. So every operation in
the specs appears in this script's output exactly once, and an operation
that was not probed prints ``SKIP`` with the reason. The reasons are:

``mutating``
    A ``POST``/``PUT``/``PATCH``/``DELETE`` that is not on the read-only
    allowlist below. Probing it would create or destroy real resources, so
    it is declined rather than guessed at. This is the honest cost of the
    "no writes" constraint, and it is stated per operation rather than
    hidden in a total.

``unresolved-host``
    The spec's ``servers`` URL is templated on a per-resource host
    (``{index_host}``, ``{assistant_host}``) and no host could be resolved.
    The reason carries *why* — and when the reason is "the control-plane
    call needed to resolve it is itself rejected at 2026-07", that is a
    finding, not a gap.

READ-ONLY BY CONSTRUCTION
=========================

Every probe is a read: ``GET``, or a ``POST`` from ``READ_ONLY_POSTS``
below, which is the only hand-maintained table in this script. Note what it
does and does not contain: **request inputs, never expected outputs.** It
cannot rot into a lie the way a golden fixture can, because nothing in it is
compared against anything. A ``POST`` the specs add later is not silently
assumed safe — it lands in ``SKIP mutating`` and shows up in the table.

Path parameters come from the spec's own ``example`` when it has one,
otherwise a synthetic placeholder that is meant not to exist. A genuine
``404 NOT_FOUND`` at the pinned version is a perfectly good oracle input:
it differs from the sentinel's ``403``, which is exactly the discrimination
being measured.

A minimal *valid* body matters more than it looks. An invalid one is not a
shortcut: ``POST /assistant/evaluation/metrics/alignment`` with ``{}``
returns the same ``422`` at both versions, because body validation runs
ahead of the version gate — a false FAIL. Bodies here were checked live
against prod for exactly this.

OPERATIONAL SHAPE
=================

This needs a live API key, and the standing constraint on this release is
that **no CI gate may depend on one**. So it is a scheduled, non-gating job:
``.github/workflows/version-sentinel-probe.yml`` runs nightly on a schedule
plus ``workflow_dispatch``, has no ``pull_request`` trigger, and cannot
therefore gate a merge. It writes the table to the job summary and exits
non-zero when anything FAILs, so a red nightly is the alarm.

There is deliberately no known-failures baseline file. One would silence the
alarm the day it was written and rot exactly like the fixtures this script
exists to supplement. While #312 and #319 are undeployed upstream, this job
is red, and it should be.

The API key is read from the environment or ``.env`` and never printed.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit

import httpx
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

API_VERSION = "2026-07"
BOGUS_VERSION = "9999-99"
API_VERSION_HEADER = "X-Pinecone-Api-Version"
DEFAULT_SPECS_DIR = Path.home() / "workspace" / "apis" / "_build" / API_VERSION
CONTROL_BASE_URL = "https://api.pinecone.io"

HTTP_METHODS = ("get", "put", "post", "delete", "options", "head", "patch", "trace")

PLACEHOLDER_STRING = "probe-nonexistent-331"
PLACEHOLDER_UUID = "00000000-0000-0000-0000-000000000000"

READ_ONLY_POSTS: dict[str, dict[str, Any]] = {
    "inference:POST /embed": {
        "model": "multilingual-e5-large",
        "inputs": [{"text": "version sentinel probe"}],
        "parameters": {"input_type": "passage"},
    },
    "inference:POST /rerank": {
        "model": "bge-reranker-v2-m3",
        "query": "version sentinel probe",
        "documents": [{"id": "1", "text": "version sentinel probe"}],
    },
    "assistant_evaluation:POST /evaluation/metrics/alignment": {
        "question": "version sentinel probe",
        "answer": "version sentinel probe",
        "ground_truth_answer": "version sentinel probe",
    },
    "db_data:POST /describe_index_stats": {},
    "db_data:POST /query": {"topK": 1},
    "db_data:POST /vectors/fetch_by_metadata": {"filter": {}, "limit": 1},
    "db_data:POST /records/namespaces/{namespace}/search": {
        "query": {"top_k": 1, "inputs": {"text": "version sentinel probe"}}
    },
    "db_data:POST /namespaces/{namespace}/documents/search": {
        # ``query_string`` rather than ``text``: a text scoring method needs a
        # field declared full-text-searchable in that index's schema, which no
        # arbitrary probe target can promise.
        "score_by": [{"type": "query_string", "query": "version sentinel probe"}],
        "top_k": 1,
    },
    "db_data:POST /namespaces/{namespace}/documents/fetch": {"ids": [PLACEHOLDER_STRING]},
    "db_data:POST /namespaces/{namespace}/documents/list": {"limit": 1},
    "assistant_data:POST /chat/{assistant_name}/context": {
        "messages": [{"role": "user", "content": "version sentinel probe"}]
    },
    "oauth:POST /oauth/token": {
        "grant_type": "client_credentials",
        "client_id": PLACEHOLDER_STRING,
        "client_secret": PLACEHOLDER_STRING,
    },
}

PROBE_VARIANTS: dict[str, tuple[str, ...]] = {
    # #319 validated ``GET /models?type=embed`` separately from the bare
    # ``GET /models`` because the SDK calls both spellings, so the positive
    # control has to cover both.
    "inference:GET /models": ("type=embed",),
}

CONTROL_EXPECT_FAIL = (
    "inference:GET /models",
    "inference:GET /models?type=embed",
    # https://api.pinecone.io/assistant/assistants — the spec path is
    # ``/assistants`` because assistant_control's server URL carries the
    # ``/assistant`` prefix.
    "assistant_control:GET /assistants",
)
CONTROL_EXPECT_PASS = (
    "inference:POST /embed",
    "inference:POST /rerank",
)

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"


class ProbeError(RuntimeError):
    """The probe could not be set up (missing specs, missing key)."""


@dataclass(frozen=True)
class Signature:
    """The reduced response identity the oracle compares.

    Deliberately excludes message text, body, and per-request headers — see
    the module docstring for why including any of them turns a broken server
    into a PASS. The body exclusion is the one to read before "fixing" it:
    #348 found a probed operation whose body is genuinely nondeterministic,
    which would make a body-sensitive oracle intermittently wrong rather than
    reliably right.
    """

    status: int
    content_type: str
    error_code: str | None

    def render(self) -> str:
        return f"{self.status} {self.content_type or '-'} {self.error_code or '-'}"


@dataclass
class Operation:
    """One path+method declared by a 2026-07 spec, plus its probe variant."""

    surface: str
    method: str
    path: str
    operation_id: str
    query: str = ""
    required_query: str = ""

    @property
    def key(self) -> str:
        suffix = f"?{self.query}" if self.query else ""
        return f"{self.surface}:{self.method} {self.path}{suffix}"

    @property
    def body_key(self) -> str:
        return f"{self.surface}:{self.method} {self.path}"


@dataclass
class Result:
    operation: Operation
    verdict: str
    url: str = ""
    pinned: Signature | None = None
    bogus: Signature | None = None
    detail: str = ""

    @property
    def key(self) -> str:
        return self.operation.key


@dataclass
class Surface:
    name: str
    base_url: str | None
    unresolved: str = ""
    operations: list[Operation] = field(default_factory=list)


def _main_worktree_root() -> Path | None:
    """Repo root of the **main** working tree, even when run from a worktree.

    Mirrors ``tests/integration/conftest.py``: a linked worktree's ``.git``
    is a file, so the current tree's root is not the checkout that holds
    ``.env``. ``git rev-parse --git-common-dir`` resolves back to the real
    ``.git`` in both cases.
    """
    git = shutil.which("git")
    if git is None:
        return None
    try:
        proc = subprocess.run(
            [git, "rev-parse", "--git-common-dir"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    raw = proc.stdout.strip()
    if not raw:
        return None
    common_dir = Path(raw)
    if not common_dir.is_absolute():
        common_dir = (REPO_ROOT / common_dir).resolve()
    return common_dir.parent


def _env_files() -> list[Path]:
    override = os.getenv("PINECONE_SDK_ENV_FILE")
    if override:
        return [Path(override).expanduser()]
    candidates = [REPO_ROOT / ".env"]
    main_root = _main_worktree_root()
    if main_root is not None and (main_root / ".env") not in candidates:
        candidates.append(main_root / ".env")
    return candidates


def resolve_api_key() -> str:
    """The live API key, from the environment or the first ``.env`` found.

    Never logged, never echoed, never written to the report.
    """
    key = os.getenv("PINECONE_API_KEY", "").strip()
    if key:
        return key
    for path in _env_files():
        if not path.is_file():
            continue
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line.startswith("PINECONE_API_KEY="):
                continue
            value = line.split("=", 1)[1].strip().strip("'\"")
            if value:
                return value
    raise ProbeError(
        "no PINECONE_API_KEY in the environment or .env. This probe needs a live key "
        "by design; it is a scheduled job, never a CI gate."
    )


def _deref(node: Any, doc: dict[str, Any]) -> Any:
    seen: set[str] = set()
    while isinstance(node, dict) and isinstance(node.get("$ref"), str):
        ref = node["$ref"]
        if not ref.startswith("#/") or ref in seen:
            return node
        seen.add(ref)
        target: Any = doc
        for part in ref.removeprefix("#/").split("/"):
            if not isinstance(target, dict) or part not in target:
                return node
            target = target[part]
        node = target
    return node


def surface_name(path: Path) -> str:
    return path.name.removesuffix(".oas.yaml").removesuffix(f"_{API_VERSION}")


def server_url(doc: dict[str, Any]) -> str:
    servers = doc.get("servers") or []
    if not servers or not isinstance(servers[0], dict):
        raise ProbeError("spec declares no servers[0].url")
    return str(servers[0]["url"]).rstrip("/")


def parse_spec(path: Path) -> tuple[str, str, list[Operation]]:
    """``(surface, servers[0].url, operations)`` for one 2026-07 OAS file."""
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise ProbeError(f"{path.name}: not a mapping")
    surface = surface_name(path)
    operations: list[Operation] = []
    for spec_path, item in sorted((doc.get("paths") or {}).items()):
        if not isinstance(item, dict):
            continue
        for method, op in item.items():
            if method not in HTTP_METHODS or not isinstance(op, dict):
                continue
            base = Operation(
                surface=surface,
                method=method.upper(),
                path=str(spec_path),
                operation_id=str(op.get("operationId") or ""),
                required_query=required_query_string(item, op, doc),
            )
            operations.append(base)
            operations.extend(
                Operation(
                    surface=surface,
                    method=base.method,
                    path=base.path,
                    operation_id=base.operation_id,
                    query=query,
                    required_query=base.required_query,
                )
                for query in PROBE_VARIANTS.get(base.key, ())
            )
    return surface, server_url(doc), operations


def required_query_string(item: dict[str, Any], op: dict[str, Any], doc: dict[str, Any]) -> str:
    """Placeholders for an operation's ``required`` query parameters.

    Omitting one is not a harmless shortcut: ``GET /vectors/fetch`` without
    ``ids`` answers ``400`` at *both* versions, because request validation
    runs ahead of the version gate — a false FAIL manufactured by the probe
    rather than found in the API.
    """
    pairs: list[str] = []
    for param in list(item.get("parameters") or []) + list(op.get("parameters") or []):
        resolved = _deref(param, doc)
        if not isinstance(resolved, dict):
            continue
        if resolved.get("in") != "query" or not resolved.get("required"):
            continue
        name = str(resolved.get("name") or "")
        if name:
            pairs.append(f"{quote(name)}={quote(placeholder_for(resolved, doc))}")
    return "&".join(pairs)


def path_parameters(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Every path parameter declared in a spec, keyed by name."""
    out: dict[str, dict[str, Any]] = {}
    for item in (doc.get("paths") or {}).values():
        if not isinstance(item, dict):
            continue
        declared = list(item.get("parameters") or [])
        for method, op in item.items():
            if method in HTTP_METHODS and isinstance(op, dict):
                declared.extend(op.get("parameters") or [])
        for param in declared:
            resolved = _deref(param, doc)
            if not isinstance(resolved, dict) or resolved.get("in") != "path":
                continue
            name = str(resolved.get("name") or "")
            if name:
                out[name] = resolved
    return out


def placeholder_for(param: dict[str, Any], doc: dict[str, Any]) -> str:
    """A value for one path parameter, preferring the spec's own ``example``.

    Preferring the spec keeps placeholders derived rather than
    hand-maintained; the synthetic fallbacks are chosen to name nothing that
    exists, so the worst case is a 404.
    """
    schema = _deref(param.get("schema") or {}, doc)
    for candidate in (param.get("example"), schema.get("example"), schema.get("default")):
        if isinstance(candidate, str) and candidate:
            return candidate
    if isinstance(schema, dict) and schema.get("format") == "uuid":
        return PLACEHOLDER_UUID
    return PLACEHOLDER_STRING


def fill_path(spec_path: str, placeholders: dict[str, str]) -> str:
    out = spec_path
    for name, value in placeholders.items():
        out = out.replace("{" + name + "}", value)
    return out


def _templated(url: str) -> bool:
    return "{" in url


def resolve_index_host(
    client: httpx.Client, api_key: str, override: str | None
) -> tuple[str | None, str]:
    """Base URL for ``db_data``'s ``https://{index_host}`` server.

    Read-only: lists indexes and picks a ready one. Never creates anything.
    """
    if override:
        return f"https://{override.removeprefix('https://').rstrip('/')}", ""
    try:
        response = client.get(
            f"{CONTROL_BASE_URL}/indexes",
            headers={"Api-Key": api_key, API_VERSION_HEADER: API_VERSION},
        )
    except httpx.HTTPError as exc:
        return None, f"GET /indexes raised {type(exc).__name__}"
    if response.status_code != 200:
        return None, f"GET /indexes returned {response.status_code} at {API_VERSION}"
    try:
        indexes = response.json().get("indexes") or []
    except ValueError:
        return None, "GET /indexes returned a non-JSON body"
    for index in indexes:
        if not isinstance(index, dict):
            continue
        if (index.get("status") or {}).get("ready") and index.get("host"):
            return f"https://{index['host']}", ""
    return None, "the project has no ready index to read from (set PINECONE_PROBE_INDEX_HOST)"


def resolve_assistant_host(
    client: httpx.Client, api_key: str, override: str | None
) -> tuple[str | None, str]:
    """Base URL for ``assistant_data``'s ``https://{assistant_host}`` server.

    Mirrors ``Assistants._data_plane_http``: the data-plane base is the
    assistant's ``host`` plus an ``/assistant`` prefix. Resolution goes
    through the assistant control plane, so when control is itself rejected
    at 2026-07 (#312) the skip reason says exactly that.
    """
    if override:
        base = override.removeprefix("https://").rstrip("/")
        if not base.endswith("/assistant"):
            base = f"{base}/assistant"
        return f"https://{base}", ""
    try:
        response = client.get(
            f"{CONTROL_BASE_URL}/assistant/assistants",
            headers={"Api-Key": api_key, API_VERSION_HEADER: API_VERSION},
        )
    except httpx.HTTPError as exc:
        return None, f"GET /assistant/assistants raised {type(exc).__name__}"
    if response.status_code != 200:
        return (
            None,
            f"GET /assistant/assistants returned {response.status_code} at {API_VERSION} "
            "(assistant control is itself the subject of #312)",
        )
    try:
        assistants = response.json().get("assistants") or []
    except ValueError:
        return None, "GET /assistant/assistants returned a non-JSON body"
    for assistant in assistants:
        if isinstance(assistant, dict) and assistant.get("host"):
            host = str(assistant["host"]).removeprefix("https://").rstrip("/")
            return f"https://{host}/assistant", ""
    return None, "the project has no assistant to read from (set PINECONE_PROBE_ASSISTANT_HOST)"


def normalize_content_type(raw: str | None) -> str:
    """Bare lowercase media type; parameters like ``charset`` dropped.

    Dropping parameters can only make two responses compare *equal*, which
    is the direction that raises an alarm rather than hides one.
    """
    if not raw:
        return ""
    return raw.split(";", 1)[0].strip().lower()


def extract_error_code(body: bytes) -> str | None:
    """The error code, across every envelope prod actually emits.

    ``{"error": {"code": "FORBIDDEN"}}`` on the control plane,
    ``{"code": 7}`` on the (gRPC-derived) data plane, and
    ``{"error": "access_denied"}`` on the OAuth host. Anything else is
    ``None``, which compares perfectly well.
    """
    try:
        doc = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(doc, dict):
        return None
    error = doc.get("error")
    if isinstance(error, dict):
        code = error.get("code")
        return None if code is None else str(code)
    if isinstance(error, str) and error:
        return error
    code = doc.get("code")
    return None if code is None else str(code)


def signature(response: httpx.Response) -> Signature:
    return Signature(
        status=response.status_code,
        content_type=normalize_content_type(response.headers.get("content-type")),
        error_code=extract_error_code(response.content),
    )


def diagnose(pinned: Signature) -> str:
    """Label a FAIL from the pinned response alone — no expectations stored."""
    if 200 <= pinned.status < 300:
        return "unversioned: the route served us identically at a version that cannot exist"
    return "unrecognized: the server did not distinguish 2026-07 from a bogus version"


def _request(
    client: httpx.Client,
    api_key: str,
    method: str,
    url: str,
    version: str,
    body: dict[str, Any] | None,
) -> httpx.Response:
    headers = {"Api-Key": api_key, "Accept": "application/json", API_VERSION_HEADER: version}
    content = None
    if body is not None:
        headers["Content-Type"] = "application/json"
        content = json.dumps(body).encode()
    return client.request(method, url, headers=headers, content=content)


def probe_operation(
    client: httpx.Client,
    api_key: str,
    operation: Operation,
    url: str,
    body: dict[str, Any] | None,
) -> Result:
    """Send the paired request and compare the two signatures."""
    try:
        pinned = signature(_request(client, api_key, operation.method, url, API_VERSION, body))
        bogus = signature(_request(client, api_key, operation.method, url, BOGUS_VERSION, body))
    except httpx.HTTPError as exc:
        return Result(
            operation=operation,
            verdict=FAIL,
            url=url,
            detail=f"transport error: {type(exc).__name__}: {exc}",
        )
    if pinned == bogus:
        return Result(
            operation=operation,
            verdict=FAIL,
            url=url,
            pinned=pinned,
            bogus=bogus,
            detail=diagnose(pinned),
        )
    return Result(operation=operation, verdict=PASS, url=url, pinned=pinned, bogus=bogus)


def build_surfaces(
    specs_dir: Path,
    client: httpx.Client,
    api_key: str,
    index_host: str | None,
    assistant_host: str | None,
) -> tuple[list[Surface], dict[str, dict[str, str]]]:
    """Every surface in *specs_dir* with its resolved base URL and operations."""
    spec_files = sorted(specs_dir.glob(f"*_{API_VERSION}.oas.yaml"))
    if not spec_files:
        raise ProbeError(f"no *_{API_VERSION}.oas.yaml under {specs_dir}")
    surfaces: list[Surface] = []
    placeholders: dict[str, dict[str, str]] = {}
    for spec_file in spec_files:
        surface_id, url, operations = parse_spec(spec_file)
        doc = yaml.safe_load(spec_file.read_text(encoding="utf-8"))
        placeholders[surface_id] = {
            name: placeholder_for(param, doc) for name, param in path_parameters(doc).items()
        }
        base: str | None = url
        unresolved = ""
        if _templated(url):
            template = urlsplit(url).netloc or url
            if "{index_host}" in template:
                base, unresolved = resolve_index_host(client, api_key, index_host)
            elif "{assistant_host}" in template:
                base, unresolved = resolve_assistant_host(client, api_key, assistant_host)
            else:
                base, unresolved = None, f"unsupported server template {url!r}"
            if base is None:
                unresolved = f"{url} could not be resolved: {unresolved}"
        surfaces.append(
            Surface(name=surface_id, base_url=base, unresolved=unresolved, operations=operations)
        )
    return surfaces, placeholders


def run(
    specs_dir: Path,
    api_key: str,
    only: str | None = None,
    index_host: str | None = None,
    assistant_host: str | None = None,
    timeout: float = 60.0,
    transport: httpx.BaseTransport | None = None,
) -> list[Result]:
    """Probe every operation in the specs, returning one Result each.

    Every operation appears in the output exactly once. Nothing is dropped:
    an operation that was not probed comes back as ``SKIP`` with a reason,
    because a silent absence is the #295 failure mode.

    *transport* exists so the unit tests can exercise the verdict logic
    hermetically; production callers leave it as ``None``.
    """
    results: list[Result] = []
    with httpx.Client(timeout=timeout, follow_redirects=False, transport=transport) as client:
        surfaces, placeholders = build_surfaces(
            specs_dir, client, api_key, index_host, assistant_host
        )
        for surface in surfaces:
            for operation in surface.operations:
                if only and only not in operation.key:
                    continue
                if surface.base_url is None:
                    results.append(
                        Result(
                            operation=operation,
                            verdict=SKIP,
                            detail=f"unresolved-host: {surface.unresolved}",
                        )
                    )
                    continue
                body: dict[str, Any] | None = None
                if operation.method != "GET":
                    if operation.body_key not in READ_ONLY_POSTS:
                        results.append(
                            Result(
                                operation=operation,
                                verdict=SKIP,
                                detail=(
                                    f"mutating: {operation.method} is not on the read-only "
                                    "allowlist; probing it could create or destroy a resource"
                                ),
                            )
                        )
                        continue
                    body = READ_ONLY_POSTS[operation.body_key]
                path = fill_path(operation.path, placeholders.get(surface.name, {}))
                url = f"{surface.base_url}{path}"
                query = "&".join(q for q in (operation.required_query, operation.query) if q)
                if query:
                    url = f"{url}?{query}"
                results.append(probe_operation(client, api_key, operation, url, body))
    return results


def counts(results: list[Result]) -> dict[str, int]:
    return {
        verdict: sum(1 for r in results if r.verdict == verdict) for verdict in (PASS, FAIL, SKIP)
    }


def check_positive_control(results: list[Result]) -> list[str]:
    """Violations of the known-answer control. Empty list means trustworthy.

    An all-green run of this script proves nothing. These five operations
    have independently determined answers (#312, #319); if the run does not
    reproduce them, the run is broken, not the API.
    """
    by_key = {r.key: r for r in results}
    problems: list[str] = []
    for key in CONTROL_EXPECT_FAIL:
        result = by_key.get(key)
        if result is None:
            problems.append(f"{key}: expected FAIL, was not probed at all")
        elif result.verdict != FAIL:
            problems.append(f"{key}: expected FAIL, got {result.verdict}")
    for key in CONTROL_EXPECT_PASS:
        result = by_key.get(key)
        if result is None:
            problems.append(f"{key}: expected PASS, was not probed at all")
        elif result.verdict != PASS:
            problems.append(f"{key}: expected PASS, got {result.verdict}")
    return problems


def _ordered(results: list[Result]) -> list[Result]:
    return sorted(results, key=lambda r: (r.verdict != FAIL, r.key))


def render_text(results: list[Result], control: list[str] | None) -> str:
    width = max((len(r.key) for r in results), default=10)
    lines = [
        f"Bogus-version sentinel probe: {API_VERSION} vs {BOGUS_VERSION}",
        "",
        "FAIL means the two responses were IDENTICAL, i.e. the server did not",
        "distinguish our pinned version from one that cannot exist. See the module",
        "docstring for why the comparison, not the status code, is the test.",
        "",
        f"{'VERDICT':<7}  {'OPERATION':<{width}}  {'PINNED':<26}  {'BOGUS':<26}  DETAIL",
    ]
    for result in _ordered(results):
        pinned = result.pinned.render() if result.pinned else "-"
        bogus = result.bogus.render() if result.bogus else "-"
        lines.append(
            f"{result.verdict:<7}  {result.key:<{width}}  {pinned:<26}  "
            f"{bogus:<26}  {result.detail}"
        )
    tally = counts(results)
    lines += [
        "",
        f"{tally[PASS]} PASS   {tally[FAIL]} FAIL   {tally[SKIP]} SKIP (reported above, "
        f"never omitted)   {len(results)} rows total",
    ]
    if control is not None:
        lines.append("")
        if control:
            lines.append("POSITIVE CONTROL VIOLATED — this run is not trustworthy:")
            lines += [f"  - {problem}" for problem in control]
        else:
            lines.append(
                "Positive control OK: reproduced the three known FAILs (#312, #319) "
                "and the two known PASSes."
            )
    return "\n".join(lines)


def render_markdown(results: list[Result], control: list[str] | None) -> str:
    tally = counts(results)
    lines = [
        f"## Bogus-version sentinel probe — `{API_VERSION}` vs `{BOGUS_VERSION}`",
        "",
        f"**{tally[PASS]} PASS · {tally[FAIL]} FAIL · {tally[SKIP]} SKIP** "
        f"over {len(results)} rows.",
        "",
        "A **FAIL** means the pinned-version and bogus-version responses were "
        "*identical* — the server did not distinguish `2026-07` from a version that "
        "cannot exist. That comparison, not any status code, is the test.",
        "",
    ]
    if control is not None:
        if control:
            lines += ["### Positive control VIOLATED", ""]
            lines += [f"- {problem}" for problem in control]
        else:
            lines += [
                "### Positive control OK",
                "",
                "Reproduced the three known FAILs (#312, #319) and the two known PASSes.",
            ]
        lines.append("")
    lines += [
        "| Verdict | Operation | Pinned | Bogus | Detail |",
        "| --- | --- | --- | --- | --- |",
    ]
    for result in _ordered(results):
        pinned = result.pinned.render() if result.pinned else "—"
        bogus = result.bogus.render() if result.bogus else "—"
        lines.append(
            f"| {result.verdict} | `{result.key}` | `{pinned}` | `{bogus}` | {result.detail} |"
        )
    return "\n".join(lines)


def render_json(results: list[Result], control: list[str] | None) -> str:
    payload: dict[str, Any] = {
        "api_version": API_VERSION,
        "bogus_version": BOGUS_VERSION,
        "counts": counts(results),
        "operations": [
            {
                "key": r.key,
                "surface": r.operation.surface,
                "method": r.operation.method,
                "path": r.operation.path,
                "query": r.operation.query,
                "operation_id": r.operation.operation_id,
                "verdict": r.verdict,
                "url": r.url,
                "pinned": None if r.pinned is None else r.pinned.render(),
                "bogus": None if r.bogus is None else r.bogus.render(),
                "detail": r.detail,
            }
            for r in sorted(results, key=lambda r: r.key)
        ],
    }
    if control is not None:
        payload["positive_control"] = {"ok": not control, "problems": control}
    return json.dumps(payload, indent=2)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Probe every 2026-07 operation twice — pinned version and a bogus sentinel — "
            "and FAIL any whose two responses are identical."
        )
    )
    parser.add_argument("--specs-dir", type=Path, default=None, help=f"default {DEFAULT_SPECS_DIR}")
    parser.add_argument("--only", default=None, help="probe only keys matching this substring")
    parser.add_argument("--index-host", default=None, help="host for db_data's {index_host}")
    parser.add_argument(
        "--assistant-host", default=None, help="host for assistant_data's {assistant_host}"
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--positive-control",
        action="store_true",
        help="also assert the known-answer control (#312, #319); exit 3 if it does not hold",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--markdown", action="store_true", help="emit a GitHub job summary table")
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="also write JSON here (so one run can emit every format)",
    )
    parser.add_argument(
        "--markdown-out", type=Path, default=None, help="also write the markdown table here"
    )
    args = parser.parse_args(argv)

    specs_dir = (
        args.specs_dir or Path(os.getenv("PINECONE_SPECS_DIR", str(DEFAULT_SPECS_DIR))).expanduser()
    )

    try:
        api_key = resolve_api_key()
        results = run(
            specs_dir=specs_dir,
            api_key=api_key,
            only=args.only,
            index_host=args.index_host or os.getenv("PINECONE_PROBE_INDEX_HOST") or None,
            assistant_host=(
                args.assistant_host or os.getenv("PINECONE_PROBE_ASSISTANT_HOST") or None
            ),
            timeout=args.timeout,
        )
    except ProbeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    control = check_positive_control(results) if args.positive_control else None
    if args.json:
        print(render_json(results, control))
    elif args.markdown:
        print(render_markdown(results, control))
    else:
        print(render_text(results, control))

    for path, render in ((args.json_out, render_json), (args.markdown_out, render_markdown)):
        if path is not None:
            path.write_text(render(results, control) + "\n", encoding="utf-8")

    if control:
        return 3
    return 1 if counts(results)[FAIL] else 0


if __name__ == "__main__":
    sys.exit(main())
