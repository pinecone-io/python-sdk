"""Sync/async parity for the 2026-07 backup operations (#113 ∥ #114).

``Backups`` and ``AsyncBackups`` validate through the same callables, build
their query strings through the same ``pinecone/_internal/backups_helpers.py``,
and decode through the same ``BackupsAdapter``, so the two transports should
differ only in ``await``. These tests hold them to that on the axes a transport
port can quietly break: identical request snapshots on the wire (method, path,
query, body, and the version header), identical signatures, and byte-identical
exception types and messages for the client-side rejection matrix.

Follows the pattern of ``tests/unit/test_async_namespace_parity.py`` (#228).

Two divergences are asserted rather than papered over:

* Examples inside the docstrings *must* differ — the sync lane shows
  ``pc = Pinecone(...)`` doctests, the async lane ``async with``. So docstring
  parity is enforced over the caller-facing contract sections (``Args`` and the
  ``.. important::`` / ``.. versionchanged::`` directives) rather than the whole
  string.
* The ``Raises`` sections name the same class under two spellings: sync says
  ``ValidationError``, async says ``PineconeValueError``, and the former is a
  deprecated alias of the latter. Normalised here rather than renamed in the
  sync lane, which is #113's shipped copy and out of scope for this ticket.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import AsyncGenerator, Awaitable, Callable, Generator
from typing import Any

import httpx
import pytest
import respx

from pinecone._internal.config import PineconeConfig
from pinecone._internal.constants import API_VERSION_HEADER, CONTROL_PLANE_API_VERSION
from pinecone._internal.http_client import AsyncHTTPClient, HTTPClient
from pinecone.async_client.backups import AsyncBackups
from pinecone.client.backups import Backups

BASE_URL = "https://api.test.pinecone.io"

_METHODS = ["create", "list", "describe", "get", "delete"]

_CALLS: dict[str, dict[str, Any]] = {
    "create": {"index_name": "my-index", "name": "nightly", "description": "before reindex"},
    "list": {"index_name": "my-index", "limit": 5, "pagination_token": "tok-1"},
    "describe": {"backup_id": "bkp_123abc"},
    "get": {"backup_id": "bkp_123abc"},
    "delete": {"backup_id": "bkp_123abc"},
}

_LIST_VARIANTS: list[dict[str, Any]] = [
    {},
    {"index_name": "my-index"},
    {"index_name": "my-index", "include_deleted": True},
    {"index_name": "my-index", "include_deleted": False},
    {"index_name": "my-index", "limit": 1, "pagination_token": "tok-2"},
    {"index_name": "my-index", "include_deleted": True, "limit": 100},
    {"limit": 50, "pagination_token": "tok-3"},
]

_ERROR_CASES: list[tuple[str, dict[str, Any]]] = [
    ("create", {"index_name": ""}),
    ("create", {"index_name": "   "}),
    ("list", {"include_deleted": True}),
    ("list", {"include_deleted": False}),
    ("list", {"include_deleted": True, "limit": 10}),
    ("describe", {"backup_id": ""}),
    ("describe", {"backup_id": "   "}),
    ("get", {"backup_id": ""}),
    ("delete", {"backup_id": ""}),
    ("delete", {"backup_id": "   "}),
]

BACKUP_PAYLOAD: dict[str, Any] = {
    "backup_id": "bkp_123abc",
    "source_index_name": "my-index",
    "source_index_id": "idx_456",
    "name": "backup_2025_03_15",
    "status": "Ready",
    "cloud": "aws",
    "region": "us-east-1",
    "record_count": 120000,
    "namespace_count": 3,
    "size_bytes": 10000000,
    "created_at": "2025-03-15T10:30:00Z",
}


@pytest.fixture
def sync_backups() -> Generator[Backups]:
    config = PineconeConfig(api_key="parity-key", host=BASE_URL)
    http = HTTPClient(config, CONTROL_PLANE_API_VERSION)
    yield Backups(http=http)
    http.close()


@pytest.fixture
async def async_backups() -> AsyncGenerator[AsyncBackups]:
    config = PineconeConfig(api_key="parity-key", host=BASE_URL)
    http = AsyncHTTPClient(config, CONTROL_PLANE_API_VERSION)
    yield AsyncBackups(http=http)
    await http.close()


def _register_routes() -> None:
    respx.post(f"{BASE_URL}/indexes/my-index/backups").mock(
        return_value=httpx.Response(201, json=BACKUP_PAYLOAD)
    )
    respx.get(f"{BASE_URL}/indexes/my-index/backups").mock(
        return_value=httpx.Response(200, json={"data": [BACKUP_PAYLOAD]})
    )
    respx.get(f"{BASE_URL}/backups").mock(
        return_value=httpx.Response(200, json={"data": [BACKUP_PAYLOAD]})
    )
    respx.get(f"{BASE_URL}/backups/bkp_123abc").mock(
        return_value=httpx.Response(200, json=BACKUP_PAYLOAD)
    )
    respx.delete(f"{BASE_URL}/backups/bkp_123abc").mock(return_value=httpx.Response(202))


def _snapshot(request: httpx.Request) -> dict[str, Any]:
    return {
        "method": request.method,
        "raw_path": request.url.raw_path.decode(),
        "query": dict(request.url.params),
        "body": request.content.decode() if request.content else None,
        "api_version": request.headers[API_VERSION_HEADER],
    }


def _raised(call: Callable[[], object]) -> tuple[type[BaseException], str]:
    try:
        call()
    except Exception as exc:
        return type(exc), str(exc)
    raise AssertionError("expected the call to raise, it returned instead")


async def _raised_async(call: Callable[[], Awaitable[object]]) -> tuple[type[BaseException], str]:
    try:
        await call()
    except Exception as exc:
        return type(exc), str(exc)
    raise AssertionError("expected the call to raise, it returned instead")


def _section(docstring: str | None, heading: str) -> str:
    """Return the ``heading:`` block of a Google-style docstring, dedented."""
    assert docstring is not None, f"missing docstring, cannot compare {heading}"
    lines = inspect.cleandoc(docstring).splitlines()
    try:
        start = lines.index(f"{heading}:")
    except ValueError:
        return ""
    body: list[str] = []
    for line in lines[start + 1 :]:
        if line and not line.startswith(" "):
            break
        body.append(line.rstrip())
    return "\n".join(body).strip()


def _directives(docstring: str | None) -> list[str]:
    """Return the reST directive blocks (``.. name::`` + body) in order."""
    assert docstring is not None
    text = inspect.cleandoc(docstring)
    blocks = re.findall(r"^\.\. (\w+)::(.*?)(?=^\S|^\.\. |\Z)", text, re.MULTILINE | re.DOTALL)
    return [f"{name}::{' '.join(body.split())}" for name, body in blocks]


@pytest.mark.parametrize("method_name", sorted(_CALLS))
@respx.mock
async def test_request_snapshot_parity(
    method_name: str, sync_backups: Backups, async_backups: AsyncBackups
) -> None:
    _register_routes()
    kwargs = _CALLS[method_name]

    getattr(sync_backups, method_name)(**kwargs)
    sync_snapshot = _snapshot(respx.calls.last.request)

    await getattr(async_backups, method_name)(**kwargs)
    async_snapshot = _snapshot(respx.calls.last.request)

    assert len(respx.calls) == 2, "each transport must have issued exactly one request"
    assert async_snapshot == sync_snapshot
    assert async_snapshot["api_version"] == CONTROL_PLANE_API_VERSION


@pytest.mark.parametrize("kwargs", _LIST_VARIANTS, ids=range(len(_LIST_VARIANTS)))
@respx.mock
async def test_list_query_string_parity(
    kwargs: dict[str, Any], sync_backups: Backups, async_backups: AsyncBackups
) -> None:
    """Both lanes build the listing query through the same shared helper."""
    _register_routes()

    sync_backups.list(**kwargs)
    sync_snapshot = _snapshot(respx.calls.last.request)

    await async_backups.list(**kwargs)
    async_snapshot = _snapshot(respx.calls.last.request)

    assert async_snapshot == sync_snapshot


@pytest.mark.parametrize("method_name", _METHODS)
def test_parameter_parity(method_name: str) -> None:
    sync_params = dict(inspect.signature(getattr(Backups, method_name)).parameters)
    async_params = dict(inspect.signature(getattr(AsyncBackups, method_name)).parameters)

    assert set(sync_params) == set(async_params), (
        f"{method_name}: parameter names differ — "
        f"sync-only={set(sync_params) - set(async_params)}, "
        f"async-only={set(async_params) - set(sync_params)}"
    )

    for name, sync_param in sync_params.items():
        async_param = async_params[name]
        assert sync_param.kind == async_param.kind, (
            f"{method_name}.{name}: kind differs (sync={sync_param.kind}, async={async_param.kind})"
        )
        assert sync_param.default == async_param.default, (
            f"{method_name}.{name}: default differs "
            f"(sync={sync_param.default!r}, async={async_param.default!r})"
        )
        assert str(sync_param.annotation) == str(async_param.annotation), (
            f"{method_name}.{name}: annotation differs "
            f"(sync={sync_param.annotation}, async={async_param.annotation})"
        )


@pytest.mark.parametrize("method_name", _METHODS)
def test_return_annotation_parity(method_name: str) -> None:
    sync_return = str(inspect.signature(getattr(Backups, method_name)).return_annotation)
    async_return = str(inspect.signature(getattr(AsyncBackups, method_name)).return_annotation)

    assert async_return == sync_return, (
        f"{method_name}: return annotation differs (sync={sync_return}, async={async_return})"
    )


@pytest.mark.parametrize("method_name", _METHODS)
def test_keyword_only_parity(method_name: str) -> None:
    for cls in (Backups, AsyncBackups):
        params = inspect.signature(getattr(cls, method_name)).parameters
        positional = [
            name
            for name, param in params.items()
            if name != "self"
            and param.kind not in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.VAR_KEYWORD)
        ]
        assert positional == [], (
            f"{cls.__name__}.{method_name} must be keyword-only, found {positional}"
        )


@pytest.mark.parametrize("method_name", _METHODS)
def test_sync_is_blocking_and_async_is_a_coroutine(method_name: str) -> None:
    assert inspect.iscoroutinefunction(getattr(AsyncBackups, method_name))
    assert not inspect.iscoroutinefunction(getattr(Backups, method_name))


def test_no_public_method_drift() -> None:
    def public(cls: type) -> set[str]:
        return {
            name
            for name, _ in inspect.getmembers(cls, callable)
            if not name.startswith("_") or name == "__repr__"
        }

    assert public(Backups) == public(AsyncBackups)


@pytest.mark.parametrize("method_name", _METHODS)
def test_args_section_parity(method_name: str) -> None:
    """The per-parameter contract must read identically in both lanes."""
    sync_args = _section(getattr(Backups, method_name).__doc__, "Args")
    async_args = _section(getattr(AsyncBackups, method_name).__doc__, "Args")

    assert async_args == sync_args, f"{method_name}: Args section differs"


@pytest.mark.parametrize("method_name", _METHODS)
def test_returns_section_parity(method_name: str) -> None:
    sync_returns = _section(getattr(Backups, method_name).__doc__, "Returns")
    async_returns = _section(getattr(AsyncBackups, method_name).__doc__, "Returns")

    assert async_returns == sync_returns, f"{method_name}: Returns section differs"


@pytest.mark.parametrize("method_name", _METHODS)
def test_raises_section_parity_modulo_the_deprecated_alias(method_name: str) -> None:
    sync_raises = _section(getattr(Backups, method_name).__doc__, "Raises").replace(
        "ValidationError", "PineconeValueError"
    )
    async_raises = _section(getattr(AsyncBackups, method_name).__doc__, "Raises").replace(
        "ValidationError", "PineconeValueError"
    )

    assert async_raises == sync_raises, f"{method_name}: Raises section differs"


def test_list_directive_parity() -> None:
    """The 2026-07 ``versionchanged`` and 404-semantics notes must match verbatim."""
    sync_directives = _directives(Backups.list.__doc__)
    async_directives = _directives(AsyncBackups.list.__doc__)

    assert sync_directives == async_directives
    assert any(d.startswith("versionchanged::") for d in sync_directives)
    assert any("include_deleted=True" in d for d in sync_directives)


@pytest.mark.parametrize(
    ("method_name", "kwargs"),
    _ERROR_CASES,
    ids=[f"{name}-{i}" for i, (name, _) in enumerate(_ERROR_CASES)],
)
async def test_validation_error_parity(
    method_name: str,
    kwargs: dict[str, Any],
    sync_backups: Backups,
    async_backups: AsyncBackups,
) -> None:
    sync_type, sync_message = _raised(lambda: getattr(sync_backups, method_name)(**kwargs))
    async_type, async_message = await _raised_async(
        lambda: getattr(async_backups, method_name)(**kwargs)
    )

    assert async_type is sync_type
    assert async_message == sync_message


# ---------------------------------------------------------------------------
# Top-level shim parity (#133, orchestrator-routed from #114): the async
# Pinecone.create_index_from_backup gained read_capacity and the legacy
# list_backups shim gained include_deleted — both must match the sync
# top-level signatures parameter-for-parameter.
# ---------------------------------------------------------------------------

_SHIM_METHODS = ["create_index_from_backup", "list_backups"]


@pytest.mark.parametrize("method_name", _SHIM_METHODS)
def test_top_level_shim_parameter_parity(method_name: str) -> None:
    from pinecone import Pinecone
    from pinecone.async_client.pinecone import AsyncPinecone

    sync_params = dict(inspect.signature(getattr(Pinecone, method_name)).parameters)
    async_params = dict(inspect.signature(getattr(AsyncPinecone, method_name)).parameters)

    assert set(sync_params) == set(async_params), (
        f"{method_name}: parameter names differ — "
        f"sync-only={set(sync_params) - set(async_params)}, "
        f"async-only={set(async_params) - set(sync_params)}"
    )

    for name, sync_param in sync_params.items():
        async_param = async_params[name]
        assert sync_param.kind == async_param.kind, (
            f"{method_name}.{name}: kind differs (sync={sync_param.kind}, async={async_param.kind})"
        )
        assert sync_param.default == async_param.default, (
            f"{method_name}.{name}: default differs "
            f"(sync={sync_param.default!r}, async={async_param.default!r})"
        )
        assert str(sync_param.annotation) == str(async_param.annotation), (
            f"{method_name}.{name}: annotation differs "
            f"(sync={sync_param.annotation}, async={async_param.annotation})"
        )
