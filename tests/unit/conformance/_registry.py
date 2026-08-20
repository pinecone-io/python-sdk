"""Claim registry and assertion recorder for 2026-07 conformance tests.

The three assertion categories every conformance test must satisfy — and
why they are recorded rather than free-form — are documented in this
package's README.md. Expected method/path (HTTP) and service/rpc (gRPC)
come from the vendored manifest, not from the test, so a test cannot claim
an operation while asserting a different endpoint.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import cache, lru_cache
from pathlib import Path
from typing import Any, TypeVar

import jsonschema
import msgspec
import pytest

EXPECTED_API_VERSION = "2026-07"
API_VERSION_HEADER = "x-pinecone-api-version"

MANIFEST_PATH = Path(__file__).resolve().parent / f"manifest_{EXPECTED_API_VERSION}.json"

_CATEGORIES = ("request", "api_version", "roundtrip")

OperationMap = dict[str, dict[str, Any]]

F = TypeVar("F", bound=Callable[..., Any])

CLAIMS: dict[str, list[str]] = {}


class UnknownOperationError(ValueError):
    """An @api_op id that does not exist in the 2026-07 manifest."""


class ConformanceError(AssertionError):
    """A conformance assertion failed or a required assertion is missing."""


@lru_cache(maxsize=1)
def _manifest() -> dict[str, Any]:
    with MANIFEST_PATH.open() as f:
        manifest: dict[str, Any] = json.load(f)
    return manifest


def manifest_operations() -> OperationMap:
    operations: OperationMap = _manifest()["operations"]
    return operations


def manifest_schemas() -> dict[str, Any]:
    schemas: dict[str, Any] = _manifest()["schemas"]
    return schemas


@cache
def _schema_validator(schema_key: str) -> jsonschema.protocols.Validator:
    schema = manifest_schemas().get(schema_key)
    if schema is None:
        raise ConformanceError(
            f"response schema {schema_key!r} is not in the manifest — regenerate it "
            "with scripts/api_coverage.py --write-manifest"
        )
    return jsonschema.Draft202012Validator(schema)


def validate_response_payload(op_id: str, entry: Mapping[str, Any], payload: Any) -> None:
    """Validate a test fixture against the OAS response schema for *op_id*.

    This is what makes a claimed operation mean spec conformance rather than
    self-consistency: the payload the test mocks must be a response a
    spec-conformant server could send. A ``divergence`` entry switches
    validation to the documented alternative schema, but only when it carries
    the question-issue reference and reason the exception list requires — a
    tampered or incomplete exception fails rather than silently passing.
    """
    divergence = entry.get("divergence")
    if divergence is not None:
        issue = divergence.get("issue")
        if not isinstance(issue, int) or isinstance(issue, bool) or issue <= 0:
            raise ConformanceError(
                f"{op_id}: divergence entry does not reference a question issue number; "
                "silent divergence exceptions are not allowed (see "
                "tests/unit/conformance/divergences_2026-07.json)"
            )
        if not isinstance(divergence.get("reason"), str) or not divergence["reason"].strip():
            raise ConformanceError(f"{op_id}: divergence entry has no reason")
        schema_key = divergence.get("response_schema")
    else:
        schema_key = entry.get("response_schema")
    if not isinstance(schema_key, str):
        raise ConformanceError(
            f"{op_id}: manifest records no response schema for this operation — "
            "regenerate it with scripts/api_coverage.py --write-manifest"
        )
    errors = sorted(
        _schema_validator(schema_key).iter_errors(payload),
        key=lambda e: ([str(p) for p in e.absolute_path], e.message),
    )
    if errors:
        details = "; ".join(
            f"{'/'.join(str(p) for p in error.absolute_path) or '<root>'}: {error.message}"
            for error in errors[:5]
        )
        suffix = f" (+{len(errors) - 5} more)" if len(errors) > 5 else ""
        raise ConformanceError(
            f"{op_id}: fixture does not conform to the spec response schema "
            f"{schema_key}: {details}{suffix}"
        )


def _expected_base_path(op_id: str, entry: Mapping[str, Any]) -> str:
    """The path prefix the request must carry, honouring a registered override.

    A surface whose spec ``servers`` URL omits the prefix the deployed API is
    really mounted under carries a ``base_path_divergence``. It is held to the
    same standard as a response-schema divergence — question issue by number
    plus a reason — so an unattributed or tampered override fails the test
    rather than quietly relaxing the path assertion.
    """
    divergence = entry.get("base_path_divergence")
    if divergence is None:
        return str(entry["base_path"])
    issue = divergence.get("issue")
    if not isinstance(issue, int) or isinstance(issue, bool) or issue <= 0:
        raise ConformanceError(
            f"{op_id}: base_path_divergence does not reference a question issue number; "
            "silent base-path overrides are not allowed (see "
            "tests/unit/conformance/divergences_2026-07.json)"
        )
    if not isinstance(divergence.get("reason"), str) or not divergence["reason"].strip():
        raise ConformanceError(f"{op_id}: base_path_divergence has no reason")
    return str(entry["base_path"])


def api_op(op_id: str) -> Callable[[F], F]:
    """Register a conformance test as a claim for one manifest operation.

    Stacks: apply once per operation the test genuinely covers. Registration
    happens at import time; an id absent from the manifest fails collection
    immediately rather than silently counting toward nothing.
    """
    operations = manifest_operations()
    if op_id not in operations:
        raise UnknownOperationError(
            f"{op_id!r} is not an operation in the {EXPECTED_API_VERSION} manifest "
            f"({MANIFEST_PATH.name}). Expected '<surface>:<operationId>' or "
            f"'db_data_grpc:<RpcName>'."
        )

    def register(fn: F) -> F:
        CLAIMS.setdefault(op_id, []).append(fn.__qualname__)
        marked: F = pytest.mark.api_op(op_id)(fn)
        return marked

    return register


def _path_template_regex(template: str) -> re.Pattern[str]:
    parts = [
        "[^/]+" if part.startswith("{") and part.endswith("}") else re.escape(part)
        for part in template.split("/")
    ]
    return re.compile("^" + "/".join(parts) + "$")


def _header_value(headers: Any, name: str) -> str | None:
    if hasattr(headers, "headers"):
        headers = headers.headers
    if isinstance(headers, Mapping):
        items: Iterable[tuple[Any, Any]] = headers.items()
    else:
        items = headers
    for key, value in items:
        key_str = key.decode() if isinstance(key, bytes) else str(key)
        if key_str.lower() == name:
            return value.decode() if isinstance(value, bytes) else str(value)
    return None


def _assert_subset(expected: Any, actual: Any, crumb: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise ConformanceError(f"{crumb}: expected an object, round-tripped {type(actual)}")
        for key, value in expected.items():
            if key not in actual:
                raise ConformanceError(f"{crumb}.{key}: lost in schema round-trip")
            _assert_subset(value, actual[key], f"{crumb}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ConformanceError(f"{crumb}: list changed shape in round-trip")
        for i, value in enumerate(expected):
            _assert_subset(value, actual[i], f"{crumb}[{i}]")
    elif expected != actual:
        raise ConformanceError(f"{crumb}: {expected!r} became {actual!r} in round-trip")


class ClaimRecorder:
    """Records which mandatory assertion categories a test has satisfied.

    ``assert_satisfied`` (run by the ``claim`` fixture at teardown) fails
    the test unless every claimed operation had all three categories
    genuinely asserted — this is what makes inflated coverage detectable.
    """

    def __init__(self, op_ids: Sequence[str]) -> None:
        if not op_ids:
            raise ConformanceError("ClaimRecorder needs at least one @api_op claim")
        operations = manifest_operations()
        unknown = [op for op in op_ids if op not in operations]
        if unknown:
            raise UnknownOperationError(f"unknown operation ids: {unknown}")
        self._ops = {op: operations[op] for op in op_ids}
        self._satisfied: set[tuple[str, str]] = set()

    def _resolve(self, op: str | None) -> str:
        if op is not None:
            if op not in self._ops:
                raise ConformanceError(f"{op!r} is not claimed by this test ({sorted(self._ops)})")
            return op
        if len(self._ops) > 1:
            raise ConformanceError(
                f"test claims multiple operations {sorted(self._ops)}; pass op=..."
            )
        return next(iter(self._ops))

    def assert_request(self, request: Any, *, op: str | None = None) -> None:
        """The actual HTTP request used the method and path the spec defines."""
        op_id = self._resolve(op)
        entry = self._ops[op_id]
        if entry["kind"] != "http":
            raise ConformanceError(f"{op_id} is a gRPC rpc; use assert_grpc_request")
        actual_method = str(request.method).upper()
        actual_path = request.url.path
        expected_path = _expected_base_path(op_id, entry) + entry["path"]
        if actual_method != entry["method"]:
            raise ConformanceError(
                f"{op_id}: expected method {entry['method']}, request used {actual_method}"
            )
        if not _path_template_regex(expected_path).match(actual_path):
            raise ConformanceError(
                f"{op_id}: path {actual_path!r} does not match spec template {expected_path!r}"
            )
        self._satisfied.add((op_id, "request"))

    def assert_grpc_request(self, full_method: str, *, op: str | None = None) -> None:
        """The actual gRPC call used the full method name the proto defines."""
        op_id = self._resolve(op)
        entry = self._ops[op_id]
        if entry["kind"] != "grpc":
            raise ConformanceError(f"{op_id} is an HTTP operation; use assert_request")
        expected = f"/{entry['service']}/{entry['rpc']}"
        if full_method != expected:
            raise ConformanceError(f"{op_id}: expected {expected!r}, call used {full_method!r}")
        self._satisfied.add((op_id, "request"))

    def assert_api_version(self, headers_or_metadata: Any, *, op: str | None = None) -> None:
        """The request carried X-Pinecone-Api-Version: 2026-07 (gRPC: metadata).

        Accepts an httpx.Request, a mapping, or an iterable of key/value
        pairs (gRPC metadata).
        """
        op_id = self._resolve(op)
        value = _header_value(headers_or_metadata, API_VERSION_HEADER)
        if value != EXPECTED_API_VERSION:
            raise ConformanceError(
                f"{op_id}: {API_VERSION_HEADER} is {value!r}, expected {EXPECTED_API_VERSION!r}"
            )
        self._satisfied.add((op_id, "api_version"))

    def assert_roundtrip(
        self,
        model_cls: type[msgspec.Struct],
        payload: dict[str, Any],
        *,
        optional_absent: Sequence[str],
        op: str | None = None,
    ) -> None:
        """The payload conforms to the OAS response schema and survives
        decode -> model -> re-encode without loss.

        For HTTP operations the payload is first validated against the
        operation's vendored response schema (or its registered divergence
        alternative), so a fixture no spec-conformant server could send
        cannot claim coverage. gRPC rpcs have no OAS schema and skip that
        leg — their fixture-side guarantee comes from the conformance
        harness, whose responses are protoc-built messages from the vendored
        proto. An rpc whose response message declares no fields
        (``DeleteResponse {}``) must use :meth:`assert_no_response_body`
        instead, exactly like a bodyless HTTP operation.

        ``optional_absent`` names top-level optional fields to strip for the
        absent-field leg: the reduced payload must still decode, and the
        model must not invent values for the stripped keys. It must name at
        least one field whenever the payload carries any optional field, so
        the absent-field contract cannot be skipped by omission. A payload
        that carries none — the spec declares only required properties, or
        only ones this model treats as required — has already proved every
        optional field tolerates absence just by decoding.
        """
        op_id = self._resolve(op)
        entry = self._ops[op_id]
        if not entry["success_body"]:
            declared = (
                "the spec declares no success response body"
                if entry["kind"] == "http"
                else "the proto response message declares no fields"
            )
            raise ConformanceError(f"{op_id}: {declared}; use assert_no_response_body")
        if entry["kind"] == "http":
            validate_response_payload(op_id, entry, payload)
        if not (isinstance(model_cls, type) and issubclass(model_cls, msgspec.Struct)):
            raise ConformanceError(f"{op_id}: assert_roundtrip needs a msgspec.Struct type")

        optional_fields = {
            field.encode_name for field in msgspec.structs.fields(model_cls) if not field.required
        }
        payload_optional = optional_fields & set(payload)
        if payload_optional and not optional_absent:
            raise ConformanceError(
                f"{op_id}: the payload carries optional {model_cls.__name__} fields "
                f"{sorted(payload_optional)}; optional_absent must exercise at least one"
            )
        unknown_fields = set(optional_absent) - optional_fields
        if unknown_fields:
            raise ConformanceError(
                f"{op_id}: optional_absent names non-optional or unknown fields "
                f"{sorted(unknown_fields)}"
            )
        missing_from_payload = set(optional_absent) - set(payload)
        if missing_from_payload:
            raise ConformanceError(
                f"{op_id}: optional_absent fields {sorted(missing_from_payload)} are not in "
                "the payload, so stripping them proves nothing"
            )

        decoded = msgspec.convert(payload, type=model_cls)
        _assert_subset(payload, msgspec.to_builtins(decoded), crumb=model_cls.__name__)

        if optional_absent:
            reduced = {k: v for k, v in payload.items() if k not in set(optional_absent)}
            decoded_reduced = msgspec.convert(reduced, type=model_cls)
            reencoded = msgspec.to_builtins(decoded_reduced)
            _assert_subset(reduced, reencoded, crumb=f"{model_cls.__name__}(reduced)")
            invented = {
                key: reencoded[key]
                for key in optional_absent
                if reencoded.get(key) == payload[key] and payload[key] is not None
            }
            if invented:
                raise ConformanceError(
                    f"{op_id}: absent optional fields came back with the stripped values "
                    f"{invented}; the round-trip is not exercising absence"
                )

        self._satisfied.add((op_id, "roundtrip"))

    def assert_no_response_body(
        self, returned: Any, *, client_side: Sequence[str] = (), op: str | None = None
    ) -> None:
        """The empty-body leg of the schema contract, for 202/204-style operations.

        Satisfies the ``roundtrip`` category only for operations the spec gives
        no success response body — for gRPC, rpcs whose proto response message
        declares no fields (``DeleteResponse {}``). ``returned`` is the SDK
        call's return value and must be ``None``: the alternative —
        round-tripping a throwaway empty model — would let any operation dodge
        the schema category.

        ``client_side`` is the escape hatch for the SDK methods that answer a
        bodyless operation with a struct they build themselves — a caller-side
        count, header-derived response metadata. It must name every field that
        comes back populated; any other populated field could only have come
        from a body the spec does not declare, so it fails. No gRPC method
        builds such a struct, so ``client_side`` stays HTTP-only.
        """
        op_id = self._resolve(op)
        entry = self._ops[op_id]
        if entry["kind"] == "grpc":
            if entry["success_body"]:
                raise ConformanceError(
                    f"{op_id}: the proto response message declares fields; use assert_roundtrip"
                )
            if client_side:
                raise ConformanceError(f"{op_id}: client_side is HTTP-only")
            if returned is not None:
                raise ConformanceError(
                    f"{op_id}: the proto response message declares no fields, but the SDK "
                    f"returned {returned!r} instead of None"
                )
            self._satisfied.add((op_id, "roundtrip"))
            return
        if entry["success_body"]:
            raise ConformanceError(
                f"{op_id}: the spec declares a success response body; use assert_roundtrip"
            )
        if returned is not None:
            if not client_side:
                raise ConformanceError(
                    f"{op_id}: spec declares no success response body, but the SDK returned "
                    f"{returned!r} instead of None"
                )
            if not isinstance(returned, msgspec.Struct):
                raise ConformanceError(
                    f"{op_id}: client_side only describes msgspec structs, not {type(returned)}"
                )
            names = {field.name for field in msgspec.structs.fields(type(returned))}
            unknown = sorted(set(client_side) - names)
            if unknown:
                raise ConformanceError(
                    f"{op_id}: client_side names fields {unknown} that "
                    f"{type(returned).__name__} does not have"
                )
            populated = {name for name in names if getattr(returned, name) is not None}
            unexpected = sorted(populated - set(client_side))
            if unexpected:
                raise ConformanceError(
                    f"{op_id}: spec declares no success response body, but "
                    f"{type(returned).__name__} came back with {unexpected} populated, which "
                    "client_side does not account for"
                )
        self._satisfied.add((op_id, "roundtrip"))

    def assert_satisfied(self) -> None:
        missing = [
            f"{op_id}: {category}"
            for op_id in sorted(self._ops)
            for category in _CATEGORIES
            if (op_id, category) not in self._satisfied
        ]
        if missing:
            raise ConformanceError(
                "conformance claim is incomplete; missing mandatory assertions: "
                + ", ".join(missing)
                + " (see tests/unit/conformance/README.md)"
            )
