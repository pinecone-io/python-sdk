"""Executes the hybrid-sparse section of ``docs/migration/v10-migration.md`` (#332).

Same discipline as ``test_docs_migration_db_control_137.py``: the examples are
read out of the published guide and run, never transcribed here, so a
transcription cannot drift from what a reader copies.

The section makes four claims this file holds to:

1. The deprecated ``dimension=``/``metric=``/``spec=`` sugar still accepts the
   9.x hybrid call (#500), but only declares the dense field — no
   ``sparse_vector`` field appears on the wire.
2. The 2026-07 replacement puts both vector fields on the wire, and the sparse
   one carries neither ``metric`` nor ``dimension``.
3. The async twin is its sync neighbour word-for-word modulo ``await`` and puts
   identical bytes on the wire.
4. The ``SchemaBuilder`` chain the section shows declares the same pair as the
   executed dict form and emits the same sparse field (#350). Its warning is
   gone, so what is pinned now is the agreement between the two spellings.

Plus the byte-comparison the ticket asks for: the hybrid sentence added to
``create()``'s docstring must be identical in the sync and async clients.
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
from pinecone.schema_builder import SchemaBuilder
from tests.factories import make_index_response

BASE_URL = "https://api.test.pinecone.io"
DOCS = Path(__file__).resolve().parents[2] / "docs"
GUIDE = DOCS / "migration/v10-migration.md"
DB_CONTROL = DOCS / "migration/v10-migration.md"
INDEX_MODEL = DOCS / "migration/v10-migration.md"

ANCHOR = "(sparse-writes)="
SECTION_END = "(db-data-breaking-changes)="


def _section() -> str:
    """The guide text from the ``(sparse-writes)=`` target to the end of that section."""
    text = GUIDE.read_text()
    assert ANCHOR in text, f"{GUIDE} lost the {ANCHOR} target the other guides link to"
    return text.split(ANCHOR, 1)[1].split(SECTION_END, 1)[0]


def _blocks() -> list[tuple[str, str]]:
    sources = [m.group(1) for m in re.finditer(r"```python\n(.*?)```", _section(), re.DOTALL)]
    assert sources, f"no python blocks found under {ANCHOR} in {GUIDE}"

    def kind(source: str) -> str:
        if source.lstrip().startswith("# Deprecated sugar"):
            return "deprecated"
        if "PineconeValueError" in source:
            return "error-demo"
        if "SchemaBuilder(" in source:
            return "builder"
        return "async" if "await " in source else "sync"

    return [(kind(s), s) for s in sources]


BLOCKS = _blocks()
LEGACY = [(i, s) for i, (k, s) in enumerate(BLOCKS) if k == "deprecated"]
CREATES = [(i, s) for i, (k, s) in enumerate(BLOCKS) if k in ("sync", "async")]
ASYNC = [(i, s) for i, (k, s) in enumerate(BLOCKS) if k == "async"]
BUILDER = [(i, s) for i, (k, s) in enumerate(BLOCKS) if k == "builder"]
ERROR_DEMO = [(i, s) for i, (k, s) in enumerate(BLOCKS) if k == "error-demo"]


def test_the_section_still_carries_every_kind_of_block_this_file_checks() -> None:
    """A block silently deleted from the guide must not silently drop its test."""
    assert LEGACY, "the deprecated-sugar before-example is gone"
    assert ASYNC, "the async tab is gone"
    assert len(CREATES) == 2, f"expected one sync and one async create, got {len(CREATES)}"
    assert len(BUILDER) == 1, f"expected exactly one SchemaBuilder block, got {len(BUILDER)}"
    assert len(ERROR_DEMO) == 1, f"expected exactly one error-demo block, got {len(ERROR_DEMO)}"


def _printed_error(source: str) -> str:
    lines = source.strip().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("# PineconeValueError:"))
    parts = [lines[start].split(":", 1)[1].strip()]
    for line in lines[start + 1 :]:
        if not line.startswith("#"):
            break
        parts.append(line.lstrip("#").strip())
    return " ".join(parts)


@pytest.mark.parametrize(("index", "source"), ERROR_DEMO, ids=[str(i) for i, _ in ERROR_DEMO])
def test_the_no_metric_error_block_raises_the_message_the_section_prints(
    index: int, source: str
) -> None:
    from pinecone.errors.exceptions import PineconeValueError

    code = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("#"))
    with pytest.raises(PineconeValueError) as excinfo:
        eval(  # noqa: S307
            compile(code, str(GUIDE), "eval"), {"SchemaBuilder": SchemaBuilder}
        )
    assert str(excinfo.value) == _printed_error(source), f"block {index}"


def _stub() -> None:
    ready = make_index_response(
        name="hybrid",
        schema={
            "fields": {
                "embedding": {"type": "dense_vector", "dimension": 1536, "metric": "dotproduct"},
                "sparse_terms": {"type": "sparse_vector"},
            }
        },
    )
    respx.get(url__regex=rf"{BASE_URL}/indexes/[^/]+$").mock(
        return_value=httpx.Response(200, json=ready)
    )
    respx.post(f"{BASE_URL}/indexes").mock(return_value=httpx.Response(201, json=ready))


def _written_body() -> bytes:
    writes = [call for call in respx.calls if call.request.method == "POST"]
    assert writes, "example issued no POST"
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


@pytest.mark.parametrize(("index", "source"), LEGACY, ids=[str(i) for i, _ in LEGACY])
@respx.mock
def test_deprecated_hybrid_create_still_works_but_declares_no_sparse_field(
    index: int, source: str
) -> None:
    raw = _sync_body(source)
    fields = json.loads(raw)["schema"]["fields"]

    dense = [name for name, f in fields.items() if f["type"] == "dense_vector"]
    sparse = [name for name, f in fields.items() if f["type"] == "sparse_vector"]
    assert dense == ["_values"], f"block {index}: expected the reserved dense field, got {dense}"
    assert not sparse, f"block {index}: expected no sparse_vector field, got {sparse}"
    assert fields["_values"]["metric"] == "dotproduct"


@pytest.mark.parametrize(("index", "source"), CREATES, ids=[str(i) for i, _ in CREATES])
@pytest.mark.asyncio
@respx.mock
async def test_hybrid_create_declares_both_vector_fields(index: int, source: str) -> None:
    raw = await _async_body(source) if "await " in source else _sync_body(source)
    fields = json.loads(raw)["schema"]["fields"]

    dense = [name for name, f in fields.items() if f["type"] == "dense_vector"]
    sparse = [name for name, f in fields.items() if f["type"] == "sparse_vector"]
    assert len(dense) == 1, f"block {index}: expected one dense_vector field, got {dense}"
    assert len(sparse) == 1, f"block {index}: expected one sparse_vector field, got {sparse}"

    assert fields[dense[0]]["metric"] == "dotproduct", (
        f"block {index}: the section is about hybrid indexes, so the dense field "
        "must still be dotproduct"
    )
    # pc-types/src/index_schema_def.rs:333-335 — the sparse create field holds
    # a description and nothing else, so any other key configures nothing.
    assert set(fields[sparse[0]]) == {"type"}, (
        f"block {index}: sparse field sent {sorted(fields[sparse[0]])}; the 2026-07 "
        "create schema accepts only type and description"
    )


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


@pytest.mark.parametrize(("index", "source"), BUILDER, ids=[str(i) for i, _ in BUILDER])
def test_builder_block_declares_the_same_pair_as_the_executed_create(
    index: int, source: str
) -> None:
    namespace: dict[str, Any] = {"SchemaBuilder": SchemaBuilder}
    _run(source, namespace)
    fields = namespace["schema"]["fields"]
    assert {f["type"] for f in fields.values()} == {"dense_vector", "sparse_vector"}


@pytest.mark.parametrize(("index", "source"), BUILDER, ids=[str(i) for i, _ in BUILDER])
def test_builder_block_emits_the_same_sparse_field_as_the_executed_create(
    index: int, source: str
) -> None:
    """#350: the two spellings the section offers must agree on the sparse field.

    The guide now says the chain and the dict "put identical bytes on the wire",
    so the claim is checked against the builder's own output rather than
    trusted — that sentence is the reason the #350 warning could be deleted.
    """
    namespace: dict[str, Any] = {"SchemaBuilder": SchemaBuilder}
    _run(source, namespace)
    sparse = next(f for f in namespace["schema"]["fields"].values() if f["type"] == "sparse_vector")
    assert set(sparse) == {"type"}, (
        f"block {index}: the builder chain emitted {sorted(sparse)}; the guide "
        "promises it matches the dict form's {'type': 'sparse_vector'}"
    )


def test_the_guide_no_longer_warns_callers_off_the_builder() -> None:
    """The #350 warning and its issue link must not outlive the bug they described."""
    section = _section()
    assert "issues/350" not in section
    assert "#350" not in section
    assert "issues/350" not in DB_CONTROL.read_text()


def _hybrid_docstring_block(create: Any) -> str:
    doc = inspect.getdoc(create)
    assert doc is not None
    match = re.search(
        r"A \*\*hybrid\*\* index must declare.*?v10-migration\.md``\.", doc, re.DOTALL
    )
    assert match, "create() no longer documents the hybrid sparse-field requirement"
    return match.group(0)


def test_create_docstrings_are_byte_identical_across_the_two_lanes() -> None:
    from pinecone.async_client.indexes import AsyncIndexes
    from pinecone.client.indexes import Indexes

    sync_block = _hybrid_docstring_block(Indexes.create).encode()
    async_block = _hybrid_docstring_block(AsyncIndexes.create).encode()
    assert sync_block == async_block


def test_the_docstring_and_the_guide_agree_on_the_two_load_bearing_facts() -> None:
    """Both say the failure is late, and both say the field cannot be added later."""
    from pinecone.client.indexes import Indexes

    block = _hybrid_docstring_block(Indexes.create)
    assert "only the sparse upserts are" in block
    assert "cannot be added by ``configure()``" in block

    section = _section()
    assert "cannot add the sparse field afterwards" in section
    assert "no signal at create time" in section


def test_the_sibling_guides_link_to_this_section() -> None:
    """``sphinx-build -W`` fails these links if the target moves; this names why they exist."""
    target = "](#sparse-writes)"
    assert target in DB_CONTROL.read_text(), (
        "the db_control hybrid warning must point at the full write-up, not just the issue"
    )
    assert target in INDEX_MODEL.read_text()


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def test_the_section_names_the_error_that_actually_surfaces() -> None:
    section = _flat(_section())
    assert "document schema, so writes must go through the documents" in section


def test_the_section_quotes_the_stale_server_message_and_flags_it() -> None:
    """The misleading text is reproduced and named, not paraphrased away."""
    section = _flat(_section())
    assert "only indexes that are sparse or using dotproduct are supported" in section
