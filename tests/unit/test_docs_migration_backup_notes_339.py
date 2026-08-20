"""Executes the code blocks in ``docs/migration/v10-2026-07-backup-models.md``.

The examples are read out of the published guide and run, rather than
transcribed into Python literals here, so a transcription cannot drift from the
text a user reads.

Blocks are classified by content: the one opening ``# 9.x`` documents the
removed ``BackupModel`` attributes and must raise ``AttributeError`` naming the
replacement; the rest must reach a stubbed control plane. Async twins are
checked twice — the source must be its sync neighbour word-for-word modulo
``await``, and the two must put identical bytes on the wire.
"""

from __future__ import annotations

import ast
import inspect
import json
import re
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from pinecone import AsyncPinecone, Pinecone
from pinecone._internal.adapters.backups_adapter import BackupsAdapter
from pinecone.errors.exceptions import PineconeValueError
from tests.factories import make_backup_response, make_index_response

BASE_URL = "https://api.test.pinecone.io"
GUIDE = Path(__file__).resolve().parents[2] / "docs/migration/v10-2026-07-backup-models.md"

SCHEDULE_ID = "e88f7273-42aa-47e9-af73-593827136867"

SCHEDULE: dict[str, Any] = {
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

EMPTY_PAGE: dict[str, Any] = {"data": [], "pagination": None}

Block = tuple[str, str]


def _blocks() -> list[Block]:
    sources = [
        m.group(1) for m in re.finditer(r"```python\n(.*?)```", GUIDE.read_text(), re.DOTALL)
    ]
    assert sources, f"no python blocks found in {GUIDE}"
    return [
        ("9x" if s.lstrip().startswith("# 9.x") else "async" if "await " in s else "sync", s)
        for s in sources
    ]


BLOCKS = _blocks()
LEGACY = [(i, s) for i, (kind, s) in enumerate(BLOCKS) if kind == "9x"]
CURRENT = [(i, s) for i, (kind, s) in enumerate(BLOCKS) if kind != "9x"]
ASYNC = [(i, s) for i, (kind, s) in enumerate(BLOCKS) if kind == "async"]


def _stub() -> None:
    ready = make_index_response(name="product-search-restored")
    respx.get(url__regex=rf"{BASE_URL}/backups(\?.*)?$").mock(
        return_value=httpx.Response(200, json=EMPTY_PAGE)
    )
    respx.get(url__regex=rf"{BASE_URL}/restore-jobs(\?.*)?$").mock(
        return_value=httpx.Response(200, json=EMPTY_PAGE)
    )
    respx.get(url__regex=rf"{BASE_URL}/indexes/[^/]+/backups(\?.*)?$").mock(
        return_value=httpx.Response(200, json=EMPTY_PAGE)
    )
    respx.get(url__regex=rf"{BASE_URL}/indexes/[^/]+/backup-schedules(\?.*)?$").mock(
        return_value=httpx.Response(200, json=EMPTY_PAGE)
    )
    respx.get(url__regex=rf"{BASE_URL}/backup-schedules/[^/]+/history(\?.*)?$").mock(
        return_value=httpx.Response(200, json=EMPTY_PAGE)
    )
    respx.post(url__regex=rf"{BASE_URL}/backups/[^/]+/create-index$").mock(
        return_value=httpx.Response(201, json={"restore_job_id": "rj-1", "index_id": "ix-1"})
    )
    respx.post(url__regex=rf"{BASE_URL}/indexes/[^/]+/backup-schedules$").mock(
        return_value=httpx.Response(201, json=SCHEDULE)
    )
    respx.patch(url__regex=rf"{BASE_URL}/backup-schedules/[^/]+$").mock(
        return_value=httpx.Response(200, json=SCHEDULE)
    )
    respx.get(url__regex=rf"{BASE_URL}/indexes/[^/]+$").mock(
        return_value=httpx.Response(200, json=ready)
    )
    respx.patch(url__regex=rf"{BASE_URL}/indexes/[^/]+$").mock(
        return_value=httpx.Response(200, json=ready)
    )


def _fingerprint(start: int = 0) -> list[tuple[str, str, bytes]]:
    """*start* skips calls made earlier in the same ``respx.mock`` context, so a
    sync and an async run of the same block can be compared inside one test."""
    return [
        (call.request.method, str(call.request.url), bytes(call.request.content))
        for call in list(respx.calls)[start:]
    ]


def _run(source: str, namespace: dict[str, Any]) -> Any:
    code = compile(source, str(GUIDE), "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    return eval(code, namespace)  # noqa: S307


def _sync_requests(source: str) -> list[tuple[str, str, bytes]]:
    _stub()
    start = len(respx.calls)
    _run(source, {"pc": Pinecone(api_key="key", host=BASE_URL)})
    return _fingerprint(start)


async def _async_requests(source: str) -> list[tuple[str, str, bytes]]:
    _stub()
    start = len(respx.calls)
    async with AsyncPinecone(api_key="key", host=BASE_URL) as pc:
        result = _run(source, {"pc": pc})
        if inspect.isawaitable(result):
            await result
    return _fingerprint(start)


def _backup_with_typed_schema() -> Any:
    payload = make_backup_response(
        schema={
            "fields": {"embedding": {"type": "dense_vector", "dimension": 1536, "metric": "cosine"}}
        }
    )
    payload.pop("dimension")
    payload.pop("metric")
    return BackupsAdapter().to_backup(json.dumps(payload).encode())


@pytest.mark.parametrize(("index", "source"), CURRENT, ids=[str(i) for i, _ in CURRENT])
@pytest.mark.asyncio
@respx.mock
async def test_current_example_reaches_the_control_plane(index: int, source: str) -> None:
    sent = await _async_requests(source) if "await " in source else _sync_requests(source)
    assert sent, f"block {index} issued no request"


@pytest.mark.parametrize(("index", "source"), LEGACY, ids=[str(i) for i, _ in LEGACY])
def test_9x_attribute_access_raises_naming_the_replacement(index: int, source: str) -> None:
    backup = _backup_with_typed_schema()
    legacy, _, current = source.partition("# 10.x")
    for attr, replacement in (("dimension", "dense_dimension"), ("metric", "schema.fields")):
        assert f"backup.{attr}" in legacy, f"block {index} no longer reads backup.{attr}"
        with pytest.raises(AttributeError) as excinfo:
            getattr(backup, attr)
        assert replacement in str(excinfo.value)
    _run(current.strip(), {"backup": backup})


@pytest.mark.parametrize(("index", "source"), ASYNC, ids=[str(i) for i, _ in ASYNC])
def test_async_block_mirrors_its_sync_twin_word_for_word(index: int, source: str) -> None:
    twin_kind, twin_source = BLOCKS[index - 1]
    assert twin_kind == "sync", f"block {index} has no sync twin immediately before it"
    assert source.replace("await ", "") == twin_source


@pytest.mark.parametrize(("index", "source"), ASYNC, ids=[str(i) for i, _ in ASYNC])
@pytest.mark.asyncio
@respx.mock
async def test_async_block_puts_identical_bytes_on_the_wire(index: int, source: str) -> None:
    assert await _async_requests(source) == _sync_requests(BLOCKS[index - 1][1])


@respx.mock
def test_restore_example_omits_read_capacity() -> None:
    _stub()
    pc = Pinecone(api_key="key", host=BASE_URL)
    pc.create_index_from_backup(name="product-search-restored", backup_id="bk-abc123", timeout=-1)
    restore = next(r for r in _fingerprint() if r[0] == "POST")
    assert b"read_capacity" not in restore[2]


@respx.mock
def test_include_deleted_is_rejected_on_the_project_wide_listing() -> None:
    _stub()
    pc = Pinecone(api_key="key", host=BASE_URL)
    with pytest.raises(PineconeValueError):
        pc.backups.list(include_deleted=True)


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda pc: pc.backups.list(limit=7, pagination_token="T"), id="backups"),
        pytest.param(
            lambda pc: pc.backups.list(index_name="ix", limit=7, pagination_token="T"),
            id="index-backups",
        ),
        pytest.param(
            lambda pc: pc.backup_schedules.list(index_name="ix", limit=7, pagination_token="T"),
            id="schedules",
        ),
        pytest.param(
            lambda pc: pc.backup_schedules.history(
                schedule_id=SCHEDULE_ID, limit=7, pagination_token="T"
            ),
            id="history",
        ),
        pytest.param(
            lambda pc: pc.restore_jobs.list(limit=7, pagination_token="T"), id="restore-jobs"
        ),
    ],
)
@respx.mock
def test_limit_is_dropped_alongside_a_token_on_every_listing(call: Any) -> None:
    """Asserted per endpoint: routing through one helper today is not a guarantee
    that each of the five call sites keeps doing so."""
    _stub()
    call(Pinecone(api_key="key", host=BASE_URL))
    query = httpx.URL(_fingerprint()[-1][1]).query.decode()
    assert "paginationToken=T" in query
    assert "limit" not in query
