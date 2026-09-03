"""Executes the create-limits section of ``docs/migration/v10-migration.md`` (#334).

Same discipline as ``test_docs_migration_db_control_137.py`` and
``test_docs_migration_sparse_hybrid_332.py``: the examples are read out of the
published guide and run, never transcribed here, so a transcription cannot
drift from what a reader copies. ``_137`` already executes every block in this
file generically; this module pins the *specific* claims the #334 section makes,
which a generic "reaches the control plane" assertion cannot see.

The section documents five things, each of which one test below holds to:

1. A schema field ``description`` is capped in UTF-8 bytes, not characters.
2. ``full_text_search.language`` is an 18-value set and ``stop_words`` is gated
   to 13 of them, so ``language="tr"`` alone must go out on the wire unmolested.
3. A ``language`` sent alongside ``ngram`` is accepted by the SDK and dropped by
   the server, so both keys must appear in the request.
4. A ``""`` tag value must reach the server rather than being stripped
   client-side — that is what makes the server's delete-on-create rule
   observable from Python at all.
5. The two CMEK rules report different status codes.

Plus the byte-comparisons the ticket asks for: every docstring block #334
touches must be identical in the sync and async clients.
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
from tests.factories import make_index_response

BASE_URL = "https://api.test.pinecone.io"
GUIDE = Path(__file__).resolve().parents[2] / "docs/migration/v10-migration.md"

ANCHOR = "(create-limits)="
SECTION_END = "(vector-data)="


def _section() -> str:
    text = GUIDE.read_text()
    assert ANCHOR in text, f"{GUIDE} lost the {ANCHOR} target"
    return text.split(ANCHOR, 1)[1].split(SECTION_END, 1)[0]


def _blocks() -> list[str]:
    sources = [m.group(1) for m in re.finditer(r"```python\n(.*?)```", _section(), re.DOTALL)]
    assert sources, f"no python blocks found under {ANCHOR} in {GUIDE}"
    return sources


BLOCKS = _blocks()
ASYNC = [(i, s) for i, s in enumerate(BLOCKS) if "await " in s]


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
    writes = [call for call in respx.calls if call.request.method in ("POST", "PATCH")]
    assert writes, "example issued no POST or PATCH"
    return bytes(writes[-1].request.content)


def _run(source: str, namespace: dict[str, Any]) -> Any:
    code = compile(source, str(GUIDE), "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    return eval(code, namespace)  # noqa: S307


def _sync_body(source: str) -> bytes:
    _stub()
    _run(source, {"pc": Pinecone(api_key="key", host=BASE_URL)})
    return _written_body()


async def _async_body(source: str) -> bytes:
    _stub()
    async with AsyncPinecone(api_key="key", host=BASE_URL) as pc:
        result = _run(source, {"pc": pc})
        if inspect.isawaitable(result):
            await result
    return _written_body()


@respx.mock
def _body(source: str) -> dict[str, Any]:
    return dict(json.loads(_sync_body(source)))


def _find(predicate: Any) -> dict[str, Any]:
    for source in BLOCKS:
        if "await " in source:
            continue
        body = _body(source)
        if predicate(body):
            return body
    raise AssertionError("no block in the #334 section matches")


def test_the_section_still_carries_the_five_blocks_this_file_checks() -> None:
    """A block silently deleted from the guide must not silently drop its test."""
    assert len(BLOCKS) == 5, f"expected 5 python blocks under {ANCHOR}, got {len(BLOCKS)}"
    assert len(ASYNC) == 1, f"expected exactly one async block, got {len(ASYNC)}"


def test_the_documented_description_stays_inside_the_byte_cap() -> None:
    """The cap is bytes, so the example must be measured in bytes too."""
    body = _find(lambda b: "description" in json.dumps(b))
    field = body["schema"]["fields"]["embedding"]
    assert len(field["description"].encode("utf-8")) <= 256

    section = _section()
    assert "256 bytes of UTF-8, not 256 characters" in section

    # The description cap is client-enforced (schema_builder._DESCRIPTION_MAX_BYTES) so
    # the guide keeps its number; the full_text_search field cap is enforced nowhere in
    # this repo, so the guide names the constraint rather than a value it cannot verify.
    assert "capped on how many `full_text_search` fields" in section


def test_a_stop_word_less_language_still_goes_out_unmolested() -> None:
    """``language="tr"`` is valid alone, so nothing client-side may reject it."""
    body = _find(lambda b: "full_text_search" in json.dumps(b))
    assert body["schema"]["fields"]["body"]["full_text_search"] == {"language": "tr"}


def test_ngram_and_language_are_both_sent_because_the_server_drops_one() -> None:
    """The SDK does not pre-empt the server's silent replacement."""
    body = _find(lambda b: "ngram" in json.dumps(b))
    config = body["schema"]["fields"]["title"]["full_text_search"]
    assert config["language"] == "tr"
    assert config["ngram"] == {"min_gram": 2, "max_gram": 4}


def test_the_empty_tag_value_reaches_the_wire_on_create() -> None:
    """If the SDK stripped it, the server's delete-on-create rule would be moot."""
    body = _find(lambda b: b.get("tags", {}).get("owner") == "")
    assert body["tags"] == {"env": "prod", "owner": ""}


@pytest.mark.asyncio
@respx.mock
async def test_the_configure_block_deletes_a_key_on_both_lanes() -> None:
    index, source = ASYNC[0]
    twin = BLOCKS[index - 1]
    assert source.replace("await ", "") == twin
    assert await _async_body(source) == _sync_body(twin)
    assert json.loads(_sync_body(twin))["tags"] == {"team": "search", "owner": ""}


def test_the_section_keeps_the_two_cmek_status_codes_apart() -> None:
    """The per-request rule and the per-project rule are two different checks."""
    section = re.sub(r"\s+", " ", _section())
    assert "is a 400 per request" in section
    assert "is a 412 per project" in section
    assert "regardless of whether the request carries a `cmek_id`" in section


def _flat(doc: str) -> str:
    return re.sub(r"\s+", " ", doc)


def test_the_language_constraint_names_the_error_not_an_enumerated_list() -> None:
    """Style rule: name the constraint and the error, never the exhaustive value list."""
    from pinecone.client.indexes import Indexes

    doc = _flat(inspect.getdoc(Indexes.create) or "")
    assert "accepts a fixed set of language codes" in doc
    assert "not supported for every language" in doc
    assert "ar da de" not in doc
    assert "18 values" not in doc
    assert "13-value subset" not in doc


def test_the_docstring_describes_rather_than_quotes_the_stop_words_error() -> None:
    """A verbatim server message goes stale; describing *why* it differs does not."""
    from pinecone.client.indexes import Indexes

    doc = _flat(inspect.getdoc(Indexes.create) or "")
    assert "the server's 400 names the unsupported language" in doc
    assert "by its English name rather than the code you sent" in doc
    assert "stop_words is not supported for language 'turkish'" not in doc


def _args(fn: Any) -> bytes:
    doc = inspect.getdoc(fn) or ""
    match = re.search(r"^Args:\n(.*?)^(?:Returns|Raises):", doc, re.DOTALL | re.MULTILINE)
    assert match, f"{fn.__qualname__} has no Args block"
    return match.group(1).encode()


def _raises(fn: Any) -> bytes:
    doc = inspect.getdoc(fn) or ""
    match = re.search(r"^Raises:\n(.*?)^Examples:", doc, re.DOTALL | re.MULTILINE)
    assert match, f"{fn.__qualname__} has no Raises block"
    return match.group(1).encode()


def test_create_and_configure_prose_is_byte_identical_across_the_two_lanes() -> None:
    from pinecone.async_client.indexes import AsyncIndexes
    from pinecone.client.indexes import Indexes

    assert _args(Indexes.create) == _args(AsyncIndexes.create)
    assert _raises(Indexes.create) == _raises(AsyncIndexes.create)
    assert _args(Indexes.configure) == _args(AsyncIndexes.configure)


def test_the_schedule_type_paragraph_is_byte_identical_across_the_two_lanes() -> None:
    import pinecone.async_client.backup_schedules as async_module
    import pinecone.client.backup_schedules as sync_module
    from pinecone.models.backups.schedules import BackupScheduleModel

    def paragraph(module: Any) -> bytes:
        doc = module.__doc__ or ""
        start = doc.index("The SDK always sends")
        return doc[start : doc.index("\n\n", start)].encode()

    block = paragraph(sync_module)
    assert paragraph(async_module) == block
    assert b'``"time-based"``' in block
    assert b"x-enum" not in block

    attribute = " ".join((inspect.getdoc(BackupScheduleModel) or "").split())
    assert "the server does not constrain the field" in attribute
    assert "another client can report something else" in attribute
    assert "x-enum" not in attribute


def test_the_configure_docstring_says_the_tag_cap_counts_the_merge() -> None:
    """ "at most 20 tags" reads as a per-request cap; it is not."""
    from pinecone.client.indexes import Indexes

    doc = inspect.getdoc(Indexes.configure) or ""
    assert "applied to the **merged**" in doc
    assert "a 400 naming 23" not in doc
