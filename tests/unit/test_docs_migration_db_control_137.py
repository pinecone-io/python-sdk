"""Executes the code blocks in ``docs/migration/v10-2026-07-db-control.md`` (#137).

The examples are read out of the guide and run, rather than transcribed into
Python literals here — a transcription can drift from the published text,
which is the one thing a migration guide cannot afford.

Blocks are classified by content: one opening with ``# 9.x`` must raise the
guided ``PineconeTypeError`` (exact messages are pinned in
``test_indexes_helpers.py`` and ``client/test_indexes_configure.py``); the
rest must reach a stubbed control plane. Async twins are checked twice — the
source must be its sync neighbour word-for-word modulo ``await``, and the two
must put identical bytes on the wire.
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

from pinecone import AsyncPinecone, Pinecone, ServerlessSpec
from pinecone.errors.exceptions import PineconeTypeError
from tests.factories import make_index_response

BASE_URL = "https://api.test.pinecone.io"
GUIDE = Path(__file__).resolve().parents[2] / "docs/migration/v10-2026-07-db-control.md"

CREATE_FIELDS = {
    "name",
    "schema",
    "deployment",
    "read_capacity",
    "tags",
    "deletion_protection",
    "cmek_id",
}
CONFIGURE_FIELDS = {"deployment", "schema", "read_capacity", "tags", "deletion_protection"}
REMOVED_FIELDS = {"spec", "dimension", "metric", "vector_type", "pods", "metadata_config", "embed"}

# Excluded from CREATE_FIELDS deliberately: the server rejects both
# unconditionally, so the allowlist should keep failing any example sending one.
SERVER_REJECTED_CREATE_FIELDS = {"source_collection", "source_backup_id"}

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
    ready = make_index_response(name="movies")
    respx.get(url__regex=rf"{BASE_URL}/indexes/[^/]+$").mock(
        return_value=httpx.Response(200, json=ready)
    )
    respx.patch(url__regex=rf"{BASE_URL}/indexes/[^/]+$").mock(
        return_value=httpx.Response(200, json=ready)
    )
    respx.post(f"{BASE_URL}/indexes").mock(return_value=httpx.Response(201, json=ready))


def _written_body() -> bytes:
    """Skip the readiness GETs that ``create``'s default polling adds after the write."""
    writes = [call for call in respx.calls if call.request.method in ("POST", "PATCH")]
    assert writes, "example issued no POST or PATCH"
    return bytes(writes[-1].request.content)


def _run(source: str, namespace: dict[str, Any]) -> Any:
    code = compile(source, str(GUIDE), "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    return eval(code, namespace)  # noqa: S307


def _sync_body(source: str) -> bytes:
    _stub()
    _run(source, {"pc": Pinecone(api_key="key", host=BASE_URL), "ServerlessSpec": ServerlessSpec})
    return _written_body()


async def _async_body(source: str) -> bytes:
    _stub()
    async with AsyncPinecone(api_key="key", host=BASE_URL) as pc:
        result = _run(source, {"pc": pc, "ServerlessSpec": ServerlessSpec})
        if inspect.isawaitable(result):
            await result
    return _written_body()


@pytest.mark.parametrize(("index", "source"), CURRENT, ids=[str(i) for i, _ in CURRENT])
@pytest.mark.asyncio
@respx.mock
async def test_current_example_reaches_the_control_plane(index: int, source: str) -> None:
    raw = await _async_body(source) if "await " in source else _sync_body(source)
    body = json.loads(raw)
    assert set(body) <= CREATE_FIELDS | CONFIGURE_FIELDS, f"block {index} sent unknown fields"
    assert not set(body) & REMOVED_FIELDS, f"block {index} still sends a 2025-10 field"


@pytest.mark.parametrize(("index", "source"), LEGACY, ids=[str(i) for i, _ in LEGACY])
def test_9x_example_raises_the_guided_error(index: int, source: str) -> None:
    namespace: dict[str, Any] = {
        "pc": Pinecone(api_key="key", host=BASE_URL),
        "ServerlessSpec": ServerlessSpec,
    }
    with pytest.raises(PineconeTypeError) as excinfo:
        _run(source, namespace)
    assert "docs/migration/" in str(excinfo.value)


@pytest.mark.parametrize(("index", "source"), ASYNC, ids=[str(i) for i, _ in ASYNC])
def test_async_block_mirrors_its_sync_twin_word_for_word(index: int, source: str) -> None:
    twin_kind, twin_source = BLOCKS[index - 1]
    assert twin_kind == "sync", f"block {index} has no sync twin immediately before it"
    assert source.replace("await ", "") == twin_source


@pytest.mark.parametrize(("index", "source"), ASYNC, ids=[str(i) for i, _ in ASYNC])
@pytest.mark.asyncio
@respx.mock
async def test_async_block_puts_identical_bytes_on_the_wire(index: int, source: str) -> None:
    assert await _async_body(source) == _sync_body(BLOCKS[index - 1][1])


@respx.mock
def test_create_for_model_still_sends_the_legacy_cloud_region_embed_shape() -> None:
    route = respx.post(f"{BASE_URL}/indexes/create-for-model").mock(
        return_value=httpx.Response(201, json=make_index_response(name="docs"))
    )
    pc = Pinecone(api_key="key", host=BASE_URL)
    pc.indexes.create_for_model(
        name="docs",
        cloud="aws",
        region="us-east-1",
        embed={"model": "multilingual-e5-large", "field_map": {"text": "chunk_text"}},
        timeout=-1,
    )
    assert route.calls.last.request.read() == (
        b'{"name":"docs","cloud":"aws","region":"us-east-1",'
        b'"embed":{"model":"multilingual-e5-large","field_map":{"text":"chunk_text"}}}'
    )


def test_create_allowlist_covers_the_whole_sendable_create_surface() -> None:
    """``CREATE_FIELDS`` gates every docs ticket's examples, not just this file's.

    An author whose valid example carries a field missing from the allowlist
    reads "sent unknown fields" and concludes their example is wrong, when the
    allowlist is. So derive the expectation from the request model instead of
    restating it: adding a field to ``CreateIndexRequest`` now fails here until
    the allowlist is widened (or the field is named as server-rejected).
    """
    from pinecone.models.indexes.requests import CreateIndexRequest

    sendable = set(CreateIndexRequest.__struct_fields__) - SERVER_REJECTED_CREATE_FIELDS
    assert sendable == CREATE_FIELDS, (
        "CREATE_FIELDS is out of step with CreateIndexRequest; a valid documented "
        "example would fail with a misleading 'unknown fields' message"
    )


def test_configure_allowlist_covers_the_whole_sendable_configure_surface() -> None:
    from pinecone.models.indexes.requests import ConfigureIndexRequest

    assert set(ConfigureIndexRequest.__struct_fields__) == CONFIGURE_FIELDS


def test_read_capacity_kwarg_is_deliberately_not_intercepted() -> None:
    """The WARNING box's premise: none of these configure() kwargs raise."""
    from pinecone._internal.index_migration import LEGACY_CONFIGURE_KWARGS

    assert {"embed", "spec"} == LEGACY_CONFIGURE_KWARGS
    for restored in ("replicas", "pod_type", "serverless_read_capacity", "read_capacity"):
        assert restored not in LEGACY_CONFIGURE_KWARGS
