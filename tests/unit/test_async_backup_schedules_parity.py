"""Sync/async parity for the 2026-07 backup-schedule operations (#115 ∥ #116).

``BackupSchedules`` and ``AsyncBackupSchedules`` validate through the same
callables, build their query strings and bodies through the same
``pinecone/_internal/backups_helpers.py`` and the same request models, and
decode through the same ``BackupSchedulesAdapter``, so the two transports
should differ only in ``await``. These tests hold them to that on the axes a
transport port can quietly break: identical request snapshots on the wire
(method, path, query, body, and the version header), identical signatures, and
byte-identical exception types and messages for the client-side rejection
matrix — including the 403 plan-hint annotation, which is the one place the SDK
rewrites a server message.

Follows the pattern of ``tests/unit/test_async_backups_parity.py`` (#240) and
``tests/unit/test_async_assistant_files_parity.py`` (#268), which is also where
the ``AsyncPaginator`` annotation normalisation comes from.

Three divergences are asserted rather than papered over:

* Examples inside the docstrings *must* differ — the sync lane shows
  ``pc = Pinecone(...)`` doctests, the async lane ``async with``. So docstring
  parity is enforced over the caller-facing contract sections (``Args``,
  ``Returns``, ``Raises``, ``Note``) and the ``.. important:: / .. warning:: /
  .. note::`` directives rather than the whole string.
* ``iter_schedules`` and ``iter_history`` return ``Paginator`` on the sync lane
  and ``AsyncPaginator`` on the async one, in both the annotation and the
  ``Returns`` prose. Normalised, not exempted.
* Those two are the only methods that are *not* coroutine functions on the
  async lane: they return the paginator without a round trip, so awaiting them
  would be awaiting the wrong thing. Pinned explicitly below.
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
from pinecone.async_client.backup_schedules import AsyncBackupSchedules
from pinecone.client.backup_schedules import BackupSchedules

BASE_URL = "https://api.test.pinecone.io"

SCHEDULE_ID = "e88f7273-42aa-47e9-af73-593827136867"

_AWAITED_METHODS = ["create", "delete", "describe", "get", "history", "list", "update"]

_ITERATOR_METHODS = ["iter_history", "iter_schedules"]

_METHODS = sorted([*_AWAITED_METHODS, *_ITERATOR_METHODS])

_CALLS: dict[str, dict[str, Any]] = {
    "create": {
        "index_name": "my-index",
        "name": "daily-compliance-backup",
        "frequency": "daily",
        "retention_days": 90,
    },
    "list": {"index_name": "my-index", "limit": 5, "pagination_token": "tok-1"},
    "describe": {"schedule_id": SCHEDULE_ID},
    "get": {"schedule_id": SCHEDULE_ID},
    "update": {"schedule_id": SCHEDULE_ID, "frequency": "weekly", "retention_days": 30},
    "delete": {"schedule_id": SCHEDULE_ID},
    "history": {"schedule_id": SCHEDULE_ID, "limit": 3, "pagination_token": "tok-2"},
}

_LIST_VARIANTS: list[dict[str, Any]] = [
    {"index_name": "my-index"},
    {"index_name": "my-index", "limit": 1},
    {"index_name": "my-index", "pagination_token": "tok-9"},
    {"index_name": "my-index", "limit": 100, "pagination_token": "tok-10"},
]

_HISTORY_VARIANTS: list[dict[str, Any]] = [
    {"schedule_id": SCHEDULE_ID},
    {"schedule_id": SCHEDULE_ID, "limit": 1},
    {"schedule_id": SCHEDULE_ID, "pagination_token": "tok-11"},
    {"schedule_id": SCHEDULE_ID, "limit": 50, "pagination_token": "tok-12"},
]

_UPDATE_VARIANTS: list[dict[str, Any]] = [
    {"schedule_id": SCHEDULE_ID},
    {"schedule_id": SCHEDULE_ID, "enabled": False},
    {"schedule_id": SCHEDULE_ID, "enabled": True},
    {"schedule_id": SCHEDULE_ID, "frequency": "monthly"},
    {"schedule_id": SCHEDULE_ID, "retention_days": 14},
    {"schedule_id": SCHEDULE_ID, "frequency": "daily", "retention_days": 1, "enabled": True},
]

_CREATE_VARIANTS: list[dict[str, Any]] = [
    {"index_name": "my-index", "name": "s", "frequency": "daily", "retention_days": 1},
    {"index_name": "my-index", "name": "s", "frequency": "weekly", "retention_days": 365},
    {"index_name": "my-index", "name": "a" * 29, "frequency": "monthly", "retention_days": 400},
]

_AWAITED_ERROR_CASES: list[tuple[str, dict[str, Any]]] = [
    ("create", {"index_name": "", "name": "s", "frequency": "daily", "retention_days": 1}),
    ("create", {"index_name": "   ", "name": "s", "frequency": "daily", "retention_days": 1}),
    ("create", {"index_name": "i", "name": "", "frequency": "daily", "retention_days": 1}),
    ("create", {"index_name": "i", "name": "   ", "frequency": "daily", "retention_days": 1}),
    ("create", {"index_name": "i", "name": "s", "frequency": "hourly", "retention_days": 1}),
    ("create", {"index_name": "i", "name": "s", "frequency": "0 6 * * *", "retention_days": 1}),
    ("create", {"index_name": "i", "name": "s", "frequency": "DAILY", "retention_days": 1}),
    ("create", {"index_name": "i", "name": "s", "frequency": "", "retention_days": 1}),
    ("create", {"index_name": "i", "name": "s", "frequency": "daily", "retention_days": 0}),
    ("create", {"index_name": "i", "name": "s", "frequency": "daily", "retention_days": -365}),
    ("describe", {"schedule_id": ""}),
    ("describe", {"schedule_id": "   "}),
    ("get", {"schedule_id": ""}),
    ("delete", {"schedule_id": ""}),
    ("delete", {"schedule_id": "   "}),
    ("update", {"schedule_id": ""}),
    ("update", {"schedule_id": SCHEDULE_ID, "frequency": "yearly"}),
    ("update", {"schedule_id": SCHEDULE_ID, "retention_days": 0}),
    ("update", {"schedule_id": SCHEDULE_ID, "retention_days": -1}),
    ("list", {"index_name": ""}),
    ("list", {"index_name": "my-index", "limit": 0}),
    ("list", {"index_name": "my-index", "limit": -1}),
    ("history", {"schedule_id": ""}),
    ("history", {"schedule_id": SCHEDULE_ID, "limit": 0}),
    ("history", {"schedule_id": SCHEDULE_ID, "limit": -1}),
]

_ITERATOR_ERROR_CASES: list[tuple[str, dict[str, Any]]] = [
    ("iter_schedules", {"index_name": ""}),
    ("iter_schedules", {"index_name": "   "}),
    ("iter_schedules", {"index_name": "my-index", "limit": 0}),
    ("iter_schedules", {"index_name": "my-index", "limit": -1}),
    ("iter_history", {"schedule_id": ""}),
    ("iter_history", {"schedule_id": "   "}),
    ("iter_history", {"schedule_id": SCHEDULE_ID, "limit": 0}),
    ("iter_history", {"schedule_id": SCHEDULE_ID, "limit": -1}),
]

SCHEDULE_PAYLOAD: dict[str, Any] = {
    "schedule_id": SCHEDULE_ID,
    "name": "daily-compliance-backup",
    "index_id": "8cbf7ba6-4135-438e-a3c3-4a89a3298905",
    "project_id": "71ce31ea-75f7-45d6-a147-ef67f661a1b0",
    "schedule_type": "time-based",
    "frequency": "daily",
    "retention_expire_after_days": 90,
    "enabled": True,
    "next_scheduled_run": "2026-04-03T06:00:00+00:00",
    "created_at": "2026-04-02T18:22:56.712605+00:00",
}

HISTORY_PAYLOAD: dict[str, Any] = {
    "backup_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "source_index_id": "8cbf7ba6-4135-438e-a3c3-4a89a3298905",
    "source_index_name": "my-index",
    "name": "daily-compliance-backup-20260403T060000Z",
    "status": "Ready",
    "cloud": "aws",
    "region": "us-east-1",
    "record_count": 500000,
    "namespace_count": 1,
    "size_bytes": 104857600,
    "created_at": "2026-04-03T06:00:00+00:00",
}

PLAN_GATE_403: dict[str, Any] = {
    "error": {
        "code": "PERMISSION_DENIED",
        "message": "Scheduled backups are not available for your plan",
    },
    "status": 403,
}

NON_PLAN_403: dict[str, Any] = {
    "error": {
        "code": "PERMISSION_DENIED",
        "message": "API key does not have write access to this project.",
    },
    "status": 403,
}


@pytest.fixture
def sync_schedules() -> Generator[BackupSchedules]:
    config = PineconeConfig(api_key="parity-key", host=BASE_URL)
    http = HTTPClient(config, CONTROL_PLANE_API_VERSION)
    yield BackupSchedules(http=http)
    http.close()


@pytest.fixture
async def async_schedules() -> AsyncGenerator[AsyncBackupSchedules]:
    config = PineconeConfig(api_key="parity-key", host=BASE_URL)
    http = AsyncHTTPClient(config, CONTROL_PLANE_API_VERSION)
    yield AsyncBackupSchedules(http=http)
    await http.close()


def _register_routes() -> None:
    respx.post(re.compile(rf"{BASE_URL}/indexes/[^/]+/backup-schedules")).mock(
        return_value=httpx.Response(201, json=SCHEDULE_PAYLOAD)
    )
    respx.get(re.compile(rf"{BASE_URL}/indexes/[^/]+/backup-schedules")).mock(
        return_value=httpx.Response(200, json={"data": [SCHEDULE_PAYLOAD]})
    )
    respx.get(f"{BASE_URL}/backup-schedules/{SCHEDULE_ID}/history").mock(
        return_value=httpx.Response(200, json={"data": [HISTORY_PAYLOAD]})
    )
    respx.get(f"{BASE_URL}/backup-schedules/{SCHEDULE_ID}").mock(
        return_value=httpx.Response(200, json=SCHEDULE_PAYLOAD)
    )
    respx.patch(f"{BASE_URL}/backup-schedules/{SCHEDULE_ID}").mock(
        return_value=httpx.Response(200, json=SCHEDULE_PAYLOAD)
    )
    respx.delete(f"{BASE_URL}/backup-schedules/{SCHEDULE_ID}").mock(
        return_value=httpx.Response(204)
    )


def _snapshot(request: httpx.Request) -> dict[str, Any]:
    return {
        "method": request.method,
        "raw_path": request.url.raw_path.decode(),
        "query": dict(request.url.params),
        "body": request.content.decode() if request.content else None,
        "api_version": request.headers[API_VERSION_HEADER],
    }


def _comparable(value: Any) -> str:
    return str(value).replace("AsyncPaginator", "Paginator")


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
    method_name: str, sync_schedules: BackupSchedules, async_schedules: AsyncBackupSchedules
) -> None:
    _register_routes()
    kwargs = _CALLS[method_name]

    getattr(sync_schedules, method_name)(**kwargs)
    sync_snapshot = _snapshot(respx.calls.last.request)

    await getattr(async_schedules, method_name)(**kwargs)
    async_snapshot = _snapshot(respx.calls.last.request)

    assert len(respx.calls) == 2, "each transport must have issued exactly one request"
    assert async_snapshot == sync_snapshot
    assert async_snapshot["api_version"] == CONTROL_PLANE_API_VERSION


@pytest.mark.parametrize("kwargs", _CREATE_VARIANTS, ids=range(len(_CREATE_VARIANTS)))
@respx.mock
async def test_create_body_parity(
    kwargs: dict[str, Any],
    sync_schedules: BackupSchedules,
    async_schedules: AsyncBackupSchedules,
) -> None:
    """Both lanes build the nested create body through the same request model."""
    _register_routes()

    sync_schedules.create(**kwargs)
    sync_snapshot = _snapshot(respx.calls.last.request)

    await async_schedules.create(**kwargs)
    async_snapshot = _snapshot(respx.calls.last.request)

    assert async_snapshot == sync_snapshot


@pytest.mark.parametrize("kwargs", _UPDATE_VARIANTS, ids=range(len(_UPDATE_VARIANTS)))
@respx.mock
async def test_update_body_parity(
    kwargs: dict[str, Any],
    sync_schedules: BackupSchedules,
    async_schedules: AsyncBackupSchedules,
) -> None:
    """Sparse PATCH must drop exactly the same fields on both lanes."""
    _register_routes()

    sync_schedules.update(**kwargs)
    sync_snapshot = _snapshot(respx.calls.last.request)

    await async_schedules.update(**kwargs)
    async_snapshot = _snapshot(respx.calls.last.request)

    assert async_snapshot == sync_snapshot


@pytest.mark.parametrize("kwargs", _LIST_VARIANTS, ids=range(len(_LIST_VARIANTS)))
@respx.mock
async def test_list_query_string_parity(
    kwargs: dict[str, Any],
    sync_schedules: BackupSchedules,
    async_schedules: AsyncBackupSchedules,
) -> None:
    _register_routes()

    sync_schedules.list(**kwargs)
    sync_snapshot = _snapshot(respx.calls.last.request)

    await async_schedules.list(**kwargs)
    async_snapshot = _snapshot(respx.calls.last.request)

    assert async_snapshot == sync_snapshot


@pytest.mark.parametrize("kwargs", _HISTORY_VARIANTS, ids=range(len(_HISTORY_VARIANTS)))
@respx.mock
async def test_history_query_string_parity(
    kwargs: dict[str, Any],
    sync_schedules: BackupSchedules,
    async_schedules: AsyncBackupSchedules,
) -> None:
    _register_routes()

    sync_schedules.history(**kwargs)
    sync_snapshot = _snapshot(respx.calls.last.request)

    await async_schedules.history(**kwargs)
    async_snapshot = _snapshot(respx.calls.last.request)

    assert async_snapshot == sync_snapshot


@pytest.mark.parametrize("kwargs", _LIST_VARIANTS, ids=range(len(_LIST_VARIANTS)))
@respx.mock
async def test_iter_schedules_request_parity(
    kwargs: dict[str, Any],
    sync_schedules: BackupSchedules,
    async_schedules: AsyncBackupSchedules,
) -> None:
    """The paginators must send the same first page request as each other."""
    _register_routes()

    list(sync_schedules.iter_schedules(**kwargs))
    sync_snapshot = _snapshot(respx.calls.last.request)

    await async_schedules.iter_schedules(**kwargs).to_list()
    async_snapshot = _snapshot(respx.calls.last.request)

    assert async_snapshot == sync_snapshot


@pytest.mark.parametrize("kwargs", _HISTORY_VARIANTS, ids=range(len(_HISTORY_VARIANTS)))
@respx.mock
async def test_iter_history_request_parity(
    kwargs: dict[str, Any],
    sync_schedules: BackupSchedules,
    async_schedules: AsyncBackupSchedules,
) -> None:
    _register_routes()

    list(sync_schedules.iter_history(**kwargs))
    sync_snapshot = _snapshot(respx.calls.last.request)

    await async_schedules.iter_history(**kwargs).to_list()
    async_snapshot = _snapshot(respx.calls.last.request)

    assert async_snapshot == sync_snapshot


@respx.mock
async def test_multi_page_token_following_parity(
    sync_schedules: BackupSchedules, async_schedules: AsyncBackupSchedules
) -> None:
    """Both paginators walk the same pages and send the same tokens, in order."""
    pages = [
        {
            "data": [{**SCHEDULE_PAYLOAD, "schedule_id": "s-1"}],
            "pagination": {"next": "tok-2"},
        },
        {"data": [{**SCHEDULE_PAYLOAD, "schedule_id": "s-2"}], "pagination": {"next": "tok-3"}},
        {"data": [{**SCHEDULE_PAYLOAD, "schedule_id": "s-3"}], "pagination": None},
    ]
    route = respx.get(f"{BASE_URL}/indexes/my-index/backup-schedules").mock(
        side_effect=[httpx.Response(200, json=page) for page in pages * 2]
    )

    sync_ids = [s.schedule_id for s in sync_schedules.iter_schedules(index_name="my-index")]
    sync_requests = [_snapshot(call.request) for call in route.calls]

    async_ids = [s.schedule_id async for s in async_schedules.iter_schedules(index_name="my-index")]
    async_requests = [_snapshot(call.request) for call in route.calls][len(sync_requests) :]

    assert async_ids == sync_ids == ["s-1", "s-2", "s-3"]
    assert async_requests == sync_requests


@respx.mock
async def test_plan_gated_403_annotation_parity(
    sync_schedules: BackupSchedules, async_schedules: AsyncBackupSchedules
) -> None:
    """The hint is appended by shared code, so both lanes must word it identically."""
    respx.post(f"{BASE_URL}/indexes/my-index/backup-schedules").mock(
        return_value=httpx.Response(403, json=PLAN_GATE_403)
    )

    sync_type, sync_message = _raised(
        lambda: sync_schedules.create(
            index_name="my-index", name="s", frequency="daily", retention_days=1
        )
    )
    async_type, async_message = await _raised_async(
        lambda: async_schedules.create(
            index_name="my-index", name="s", frequency="daily", retention_days=1
        )
    )

    assert async_type is sync_type
    assert async_message == sync_message


@respx.mock
async def test_a_403_that_is_not_the_plan_gate_is_left_alone_on_both_lanes(
    sync_schedules: BackupSchedules, async_schedules: AsyncBackupSchedules
) -> None:
    respx.get(f"{BASE_URL}/backup-schedules/{SCHEDULE_ID}").mock(
        return_value=httpx.Response(403, json=NON_PLAN_403)
    )

    sync_type, sync_message = _raised(lambda: sync_schedules.describe(schedule_id=SCHEDULE_ID))
    async_type, async_message = await _raised_async(
        lambda: async_schedules.describe(schedule_id=SCHEDULE_ID)
    )

    assert async_type is sync_type
    assert async_message == sync_message


@respx.mock
async def test_the_paginators_annotate_the_403_identically(
    sync_schedules: BackupSchedules, async_schedules: AsyncBackupSchedules
) -> None:
    respx.get(f"{BASE_URL}/indexes/my-index/backup-schedules").mock(
        return_value=httpx.Response(403, json=PLAN_GATE_403)
    )

    sync_type, sync_message = _raised(
        lambda: list(sync_schedules.iter_schedules(index_name="my-index"))
    )
    async_type, async_message = await _raised_async(
        lambda: async_schedules.iter_schedules(index_name="my-index").to_list()
    )

    assert async_type is sync_type
    assert async_message == sync_message


@pytest.mark.parametrize("method_name", _METHODS)
def test_parameter_parity(method_name: str) -> None:
    sync_params = dict(inspect.signature(getattr(BackupSchedules, method_name)).parameters)
    async_params = dict(inspect.signature(getattr(AsyncBackupSchedules, method_name)).parameters)

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
        assert _comparable(sync_param.annotation) == _comparable(async_param.annotation), (
            f"{method_name}.{name}: annotation differs "
            f"(sync={sync_param.annotation}, async={async_param.annotation})"
        )


@pytest.mark.parametrize("method_name", _METHODS)
def test_return_annotation_parity(method_name: str) -> None:
    sync_return = inspect.signature(getattr(BackupSchedules, method_name)).return_annotation
    async_return = inspect.signature(getattr(AsyncBackupSchedules, method_name)).return_annotation

    assert _comparable(sync_return) == _comparable(async_return), (
        f"{method_name}: return annotation differs (sync={sync_return}, async={async_return})"
    )


@pytest.mark.parametrize("method_name", _METHODS)
def test_keyword_only_parity(method_name: str) -> None:
    for cls in (BackupSchedules, AsyncBackupSchedules):
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


@pytest.mark.parametrize("method_name", _AWAITED_METHODS)
def test_sync_is_blocking_and_async_is_a_coroutine(method_name: str) -> None:
    assert inspect.iscoroutinefunction(getattr(AsyncBackupSchedules, method_name))
    assert not inspect.iscoroutinefunction(getattr(BackupSchedules, method_name))


@pytest.mark.parametrize("method_name", _ITERATOR_METHODS)
def test_the_paginator_factories_are_plain_calls_on_both_lanes(method_name: str) -> None:
    """They hand back a paginator without a round trip, so neither lane awaits."""
    assert not inspect.iscoroutinefunction(getattr(AsyncBackupSchedules, method_name))
    assert not inspect.iscoroutinefunction(getattr(BackupSchedules, method_name))


def test_no_public_method_drift() -> None:
    def public(cls: type) -> set[str]:
        return {
            name
            for name, _ in inspect.getmembers(cls, callable)
            if not name.startswith("_") or name == "__repr__"
        }

    assert public(BackupSchedules) == public(AsyncBackupSchedules)


@pytest.mark.parametrize("method_name", _METHODS)
@pytest.mark.parametrize("heading", ["Args", "Returns", "Raises", "Note"])
def test_docstring_contract_section_parity(method_name: str, heading: str) -> None:
    """The caller-facing contract must read identically in both lanes."""
    sync_section = _comparable(_section(getattr(BackupSchedules, method_name).__doc__, heading))
    async_section = _comparable(
        _section(getattr(AsyncBackupSchedules, method_name).__doc__, heading)
    )

    assert async_section == sync_section, f"{method_name}: {heading} section differs"


@pytest.mark.parametrize("method_name", _METHODS)
def test_directive_parity(method_name: str) -> None:
    """The 28-char footgun, the re-enable warning, and the retry caveat must match."""
    sync_directives = _directives(getattr(BackupSchedules, method_name).__doc__)
    async_directives = _directives(getattr(AsyncBackupSchedules, method_name).__doc__)

    assert async_directives == sync_directives, f"{method_name}: directives differ"


def test_the_documented_footguns_survive_on_both_lanes() -> None:
    """A rewrite that quietly dropped a caveat would leave these assertions red."""
    create = _directives(AsyncBackupSchedules.create.__doc__)
    update = _directives(AsyncBackupSchedules.update.__doc__)
    delete = _directives(AsyncBackupSchedules.delete.__doc__)

    assert any("28 characters or fewer" in d for d in create)
    assert any("immediately" in d and "enqueues a backup run" in d for d in update)
    assert any("not safe to retry blindly" in d for d in delete)


@pytest.mark.parametrize(
    ("method_name", "kwargs"),
    _AWAITED_ERROR_CASES,
    ids=[f"{name}-{i}" for i, (name, _) in enumerate(_AWAITED_ERROR_CASES)],
)
async def test_validation_error_parity(
    method_name: str,
    kwargs: dict[str, Any],
    sync_schedules: BackupSchedules,
    async_schedules: AsyncBackupSchedules,
) -> None:
    sync_type, sync_message = _raised(lambda: getattr(sync_schedules, method_name)(**kwargs))
    async_type, async_message = await _raised_async(
        lambda: getattr(async_schedules, method_name)(**kwargs)
    )

    assert async_type is sync_type
    assert async_message == sync_message


@pytest.mark.parametrize(
    ("method_name", "kwargs"),
    _ITERATOR_ERROR_CASES,
    ids=[f"{name}-{i}" for i, (name, _) in enumerate(_ITERATOR_ERROR_CASES)],
)
async def test_iterator_validation_error_parity(
    method_name: str,
    kwargs: dict[str, Any],
    sync_schedules: BackupSchedules,
    async_schedules: AsyncBackupSchedules,
) -> None:
    """Both lanes reject eagerly, before a paginator is handed back."""
    sync_type, sync_message = _raised(lambda: getattr(sync_schedules, method_name)(**kwargs))
    async_type, async_message = _raised(lambda: getattr(async_schedules, method_name)(**kwargs))

    assert async_type is sync_type
    assert async_message == sync_message
