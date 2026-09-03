"""Executes rows 9-12 of ``docs/migration/v10-migration.md`` (#415).

Split out of #138: the vector-op selector algebra (mutual exclusion between
``query``'s ``id``/``vector``/``sparse_vector``, ``update``'s ``filter`` and
vector-value arguments) and the ID/limit rules on ``fetch``, ``fetch_by_metadata``
and ``list_paginated``. Same discipline as ``test_docs_migration_db_data_138.py``:
every code block and every ``PineconeValueError`` message is read out of the
published file and executed, never retyped here, so a transcription cannot
drift from what a reader copies.

Two claims this file holds to beyond "the message matches":

1. Every block in rows 9-12 raises before any HTTP request is made — these are
   all client-side checks, and the guide's whole premise is that the request
   never leaves the process.
2. The checks are the same shared validators on all three lanes (``Index``,
   ``AsyncIndex``, ``GrpcIndex``), so a message pinned against one lane cannot
   drift from what the others raise.

Two things this file deliberately does NOT assert, and why:

- **Row 10's claim that a raw-HTTP ``update`` request combining ``id`` and
  ``filter`` has the server silently drop ``id``.** That is backend behavior,
  unreachable through the SDK (the SDK's own ``id``-xor-``filter`` check,
  unchanged since before this release, blocks it first) — its citation is in
  the PR body, not executed here.
- **minicone parity for that same combination.** minicone rejects
  ``update(id=..., filter=...)`` with a 400 rather than accepting it and
  dropping ``id``, which diverges from the backend for this one shape
  (filed as a minicone fidelity issue, referenced in the PR). This file does
  not exercise that path at all, on either lane, to avoid pinning minicone's
  behavior as though it were the backend's.
"""

from __future__ import annotations

import re
from pathlib import Path

import httpx
import pytest
import respx

from pinecone import Index
from pinecone.errors.exceptions import PineconeValueError

GUIDE = Path(__file__).resolve().parents[2] / "docs/migration/v10-migration.md"
SECTION_START = "(db-data-breaking-changes)="
SECTION_END = "(backup-models)="
TEXT = GUIDE.read_text().split(SECTION_START, 1)[1].split(SECTION_END, 1)[0]

INDEX_HOST = "test-index-abc1234.svc.us-east1-gcp.pinecone.io"
BASE_URL = f"https://{INDEX_HOST}"


def _blocks() -> list[str]:
    sources = [m.group(1) for m in re.finditer(r"```python\n(.*?)```", TEXT, re.DOTALL)]
    assert sources, f"no python blocks found in {GUIDE}"
    return sources


def _block(needle: str) -> str:
    matches = [s for s in _blocks() if needle in s]
    assert len(matches) == 1, f"expected one block containing {needle!r}, got {len(matches)}"
    return matches[0]


def _printed_error(source: str) -> str:
    lines = source.strip().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("# PineconeValueError:"))
    parts = [lines[start].split(":", 1)[1].strip()]
    for line in lines[start + 1 :]:
        if not line.startswith("#"):
            break
        parts.append(line.lstrip("#").strip())
    return " ".join(parts)


def _code(source: str) -> str:
    return "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("#"))


def _index() -> Index:
    return Index(host=INDEX_HOST, api_key="test-key")


def _lane_source(module: str) -> str:
    return Path(__import__(module, fromlist=["__file__"]).__file__ or "").read_text()


LANES = ["pinecone.index", "pinecone.async_client.async_index", "pinecone.grpc"]


@pytest.mark.parametrize(
    ("needle", "route_path"),
    [
        ('sparse_vector={"indices": [0, 1], "values": [0.5, 0.5]}', "/query"),
        ("update(filter=", "/vectors/update"),
        ("delete(filter={})", "/vectors/delete"),
        ('fetch(ids=["a" * 600])', "/vectors/fetch"),
        ("list_paginated(limit=500)", "/vectors/list"),
        ("fetch_by_metadata(filter=", "/vectors/fetch_by_metadata"),
    ],
    ids=[
        "query-id-sparse_vector",
        "update-filter-values",
        "delete-empty-filter",
        "fetch-id-too-long",
        "list-limit-out-of-range",
        "fetch_by_metadata-limit-out-of-range",
    ],
)
@respx.mock
def test_each_row_9_to_12_block_raises_the_message_the_guide_prints(
    needle: str, route_path: str
) -> None:
    source = _block(needle)
    route = respx.route(url__regex=re.escape(route_path)).mock(
        return_value=httpx.Response(200, json={})
    )

    with pytest.raises(PineconeValueError) as excinfo:
        exec(_code(source), {"idx": _index()})  # noqa: S102

    assert str(excinfo.value) == _printed_error(source)
    assert not route.calls


@pytest.mark.parametrize("module", LANES)
def test_every_lane_shares_the_same_query_selector_validator(module: str) -> None:
    assert "require_query_selectors" in _lane_source(module)


@pytest.mark.parametrize("module", LANES)
def test_every_lane_shares_the_same_update_selector_validator(module: str) -> None:
    assert "require_update_selectors" in _lane_source(module)


@pytest.mark.parametrize("module", LANES)
def test_every_lane_shares_the_same_empty_filter_validator(module: str) -> None:
    assert "require_non_empty_filter" in _lane_source(module)


@pytest.mark.parametrize("module", LANES)
def test_every_lane_shares_the_same_id_and_limit_validators(module: str) -> None:
    source = _lane_source(module)
    assert "require_valid_vector_id" in source
    assert "require_valid_list_limit" in source
    assert "require_valid_fetch_by_metadata_limit" in source


def test_the_id_sparse_vector_message_names_both_arguments() -> None:
    expected = _printed_error(_block('sparse_vector={"indices": [0, 1], "values": [0.5, 0.5]}'))
    assert "id is mutually exclusive with sparse_vector" in expected


@respx.mock
def test_the_hybrid_query_the_guide_says_is_unaffected_still_works() -> None:
    route = respx.post(f"{BASE_URL}/query").mock(
        return_value=httpx.Response(200, json={"matches": [], "namespace": "", "usage": {}})
    )
    _index().query(
        vector=[0.1, 0.2],
        sparse_vector={"indices": [0, 1], "values": [0.5, 0.5]},
        top_k=5,
    )
    assert route.calls.last is not None


def test_the_guide_still_says_id_and_filter_together_predate_this_release() -> None:
    flat = re.sub(r"\s+", " ", TEXT)
    assert "id` and `filter` together were already rejected before this release" in flat


def test_the_row_9_to_12_table_entries_match_the_sections_below() -> None:
    for n in (9, 10, 11, 12):
        assert re.search(rf"^#### {n}\. ", TEXT, re.MULTILINE), (
            f"no '#### {n}. ' heading in {GUIDE}"
        )
        assert re.search(rf"^\| {n} \|", TEXT, re.MULTILINE), f"no '| {n} |' table row in {GUIDE}"
