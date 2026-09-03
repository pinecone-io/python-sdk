"""In-process gRPC test server capturing the Rust channel's real rpcs (#253).

The db_data_grpc conformance claims need three assertions no channel mock can
make honestly: the rpc method actually invoked, the ``x-pinecone-api-version``
metadata actually carried, and a request/response schema round-trip per the
vendored ``rust/proto/db_data_2026-07.proto``. This harness makes them
observable by running a genuine ``grpc.server`` on a loopback ephemeral port
and pointing the SDK's real transport at it — the Rust tonic client, its
``MetadataInterceptor``, and its prost codecs are all on the wire, and the
server decodes what arrives with protoc-generated code, an implementation
independent of prost, so an encoding that diverged from the vendored proto
would fail to decode rather than pass by construction.

The message/service stubs are regenerated from the vendored proto on every
test session — never vendored — so they cannot go stale (#189 contract);
``test_grpc_harness_guards.py`` cross-checks the generated descriptor against
the manifest's rpc entries. The proto is compiled under an import-safe module
basename because its real filename contains a hyphen; the copy is byte-for-
byte the vendored file, which the guard tests also pin.
"""

from __future__ import annotations

import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import grpc

REPO_ROOT = Path(__file__).resolve().parents[3]
VENDORED_PROTO = REPO_ROOT / "rust" / "proto" / "db_data_2026-07.proto"
SERVICE_NAME = "VectorService"

_GOOGLE_API_STUBS = ("google/api/annotations.proto", "google/api/field_behavior.proto")
_MODULE_BASENAME = "db_data_2026_07_conformance"

_generate_lock = threading.Lock()
_generated: tuple[ModuleType, ModuleType, Path] | None = None


def _generate() -> tuple[ModuleType, ModuleType, Path]:
    import importlib
    import importlib.resources

    from grpc_tools import protoc

    out_dir = Path(tempfile.mkdtemp(prefix="conformance-grpc-stubs-"))
    src_dir = out_dir / "proto_src"
    src_dir.mkdir()
    proto_copy = src_dir / f"{_MODULE_BASENAME}.proto"
    proto_copy.write_bytes(VENDORED_PROTO.read_bytes())

    proto_root = VENDORED_PROTO.parent
    well_known = str(importlib.resources.files("grpc_tools") / "_proto")
    include_args = [f"-I{src_dir}", f"-I{proto_root}", f"-I{well_known}"]

    rc = protoc.main(
        [
            "protoc",
            *include_args,
            f"--python_out={out_dir}",
            f"--grpc_python_out={out_dir}",
            str(proto_copy),
        ]
    )
    if rc != 0:
        raise RuntimeError(f"protoc failed ({rc}) generating stubs from {VENDORED_PROTO}")
    rc = protoc.main(
        [
            "protoc",
            *include_args,
            f"--python_out={out_dir}",
            *(str(proto_root / stub) for stub in _GOOGLE_API_STUBS),
        ]
    )
    if rc != 0:
        raise RuntimeError(f"protoc failed ({rc}) generating google.api stubs")

    sys.path.insert(0, str(out_dir))
    pb2 = importlib.import_module(f"{_MODULE_BASENAME}_pb2")
    pb2_grpc = importlib.import_module(f"{_MODULE_BASENAME}_pb2_grpc")
    return pb2, pb2_grpc, proto_copy


def generated_modules() -> tuple[ModuleType, ModuleType, Path]:
    """The ``(pb2, pb2_grpc, proto_copy)`` protoc generated from the vendored proto."""
    global _generated
    with _generate_lock:
        if _generated is None:
            _generated = _generate()
        return _generated


@dataclass
class CapturedCall:
    """One rpc as the server saw it: real method path, metadata, and request."""

    method: str
    metadata: tuple[tuple[str, str], ...]
    request: Any = None


class _CaptureInterceptor(grpc.ServerInterceptor):  # type: ignore[misc]
    """Records each call's ``:path`` method and invocation metadata.

    This runs on the server before handler dispatch, so what it records is what
    the tonic client put on the wire — not anything the test constructed.
    """

    def __init__(self, harness: VectorServiceHarness) -> None:
        self._harness = harness

    def intercept_service(self, continuation: Any, handler_call_details: Any) -> Any:
        self._harness._record(
            CapturedCall(
                method=handler_call_details.method,
                metadata=tuple(handler_call_details.invocation_metadata),
            )
        )
        return continuation(handler_call_details)


class VectorServiceHarness:
    """A real ``VectorService`` on ``127.0.0.1:<ephemeral>`` with queued responses.

    Tests queue protoc-built response messages with :meth:`respond`, drive the
    SDK, then read :meth:`single_call` for the captured method, metadata, and
    decoded request. An rpc arriving with no queued response fails the call
    with an unretryable status, so a misconfigured test fails fast instead of
    hanging or silently passing.
    """

    def __init__(self) -> None:
        self.pb2, self.pb2_grpc, self.proto_copy = generated_modules()
        self.calls: list[CapturedCall] = []
        self._responses: dict[str, list[Any]] = {}
        self._lock = threading.Lock()
        self._server = grpc.server(
            ThreadPoolExecutor(max_workers=4, thread_name_prefix="conformance-grpc"),
            interceptors=[_CaptureInterceptor(self)],
        )
        self.pb2_grpc.add_VectorServiceServicer_to_server(self._build_servicer(), self._server)
        self.port: int = self._server.add_insecure_port("127.0.0.1:0")

    @property
    def host(self) -> str:
        return f"127.0.0.1:{self.port}"

    def start(self) -> None:
        self._server.start()

    def stop(self) -> None:
        self._server.stop(grace=None)

    def reset(self) -> None:
        with self._lock:
            self.calls.clear()
            self._responses.clear()

    def respond(self, rpc: str, message: Any) -> None:
        with self._lock:
            self._responses.setdefault(rpc, []).append(message)

    def single_call(self) -> CapturedCall:
        with self._lock:
            if len(self.calls) != 1:
                raise AssertionError(
                    f"expected exactly one captured rpc, saw {[call.method for call in self.calls]}"
                )
            return self.calls[0]

    def rpc_names(self) -> list[str]:
        service = self.pb2.DESCRIPTOR.services_by_name[SERVICE_NAME]
        return [method.name for method in service.methods]

    def _record(self, call: CapturedCall) -> None:
        with self._lock:
            self.calls.append(call)

    def _handle(self, rpc: str, request: Any) -> Any:
        with self._lock:
            if self.calls:
                self.calls[-1].request = request
            queue = self._responses.get(rpc)
            if not queue:
                raise AssertionError(f"no response queued for rpc {rpc!r}")
            return queue.pop(0)

    def _build_servicer(self) -> Any:
        harness = self

        def make_handler(rpc: str) -> Any:
            def handler(self: Any, request: Any, context: Any) -> Any:
                return harness._handle(rpc, request)

            return handler

        methods = {rpc: make_handler(rpc) for rpc in self.rpc_names()}
        servicer_cls = type(
            "_RecordingVectorService",
            (self.pb2_grpc.VectorServiceServicer,),
            methods,
        )
        return servicer_cls()
