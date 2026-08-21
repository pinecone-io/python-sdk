"""Executes ``docs/migration/v10-2026-07-query-param-enums.md`` (#371).

Same discipline as ``test_docs_migration_inference_enums_296.py``: the guide's
table and code blocks are read out of the published file and run, never
transcribed here, so a transcription cannot drift from what a reader copies.

The guide makes four claims this file holds to:

1. Every ``query sent now`` cell is the query string the SDK really produces for
   that argument.
2. Every ``query sent before`` cell is really what httpx's ``str()``-based query
   encoder produces for that argument, which is what the old code path sent.
   When a Python release or a ``StrEnum`` migration makes that stop being true,
   this fails and the guide's premise needs rewriting.
3. Passing ``.value`` and passing the member produce the same URL.
4. The mangled literal is rejected with the error the guide prints, before any
   request is made.

The guide also quotes the server's wording for the rejected value. That string
comes from the backend, not from this SDK, so what is checkable here is that the
SDK surfaces it unaltered — the citation for the wording itself is in the PR body.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from pinecone import Pinecone, VectorType
from pinecone.errors.exceptions import ApiError, PineconeValueError
from tests.factories import make_error_response

BASE_URL = "https://api.test.pinecone.io"
GUIDE = Path(__file__).resolve().parents[2] / "docs/migration/v10-2026-07-query-param-enums.md"
NAMESPACE: dict[str, Any] = {"VectorType": VectorType}
MODEL_LIST: dict[str, Any] = {"models": []}


def _table_rows() -> list[tuple[str, str, str]]:
    """The (argument, sent-before, sent-now) rows of the guide's comparison table."""
    rows = []
    for line in GUIDE.read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 3 or set(cells[0]) == {"-"} or "sent before" in cells[1]:
            continue
        if not all(c.startswith("`") and c.endswith("`") for c in cells):
            continue
        rows.append(tuple(c.strip("`") for c in cells))
    assert len(rows) == 3, f"expected 3 comparison rows in {GUIDE}, got {len(rows)}"
    return rows  # type: ignore[return-value]


def _blocks(language: str = "python") -> list[str]:
    sources = [
        m.group(1) for m in re.finditer(rf"```{language}\n(.*?)```", GUIDE.read_text(), re.DOTALL)
    ]
    assert sources, f"no {language} blocks found in {GUIDE}"
    return sources


ROWS = _table_rows()
CALLS = [s for s in _blocks() if "pc.inference." in s]


def _wire_query(argument: str) -> str:
    """Run the real SDK call for one table row and return the query it sent."""
    route = respx.get(f"{BASE_URL}/models").mock(return_value=httpx.Response(200, json=MODEL_LIST))
    pc = Pinecone(api_key="key", host=BASE_URL)
    pc.inference.list_models(vector_type=eval(argument, dict(NAMESPACE)))  # noqa: S307
    return str(route.calls.last.request.url.params)


@pytest.mark.parametrize(("argument", "before", "now"), ROWS, ids=[r[0] for r in ROWS])
@respx.mock
def test_the_table_says_what_the_sdk_really_sends(argument: str, before: str, now: str) -> None:
    assert _wire_query(argument) == now


@pytest.mark.parametrize(("argument", "before", "now"), ROWS, ids=[r[0] for r in ROWS])
def test_the_before_column_is_still_what_httpx_produces(
    argument: str, before: str, now: str
) -> None:
    """The old code path handed the argument straight to httpx, which uses ``str()``."""
    unresolved = {"vector_type": eval(argument, dict(NAMESPACE))}  # noqa: S307
    assert str(httpx.QueryParams(unresolved)) == before


@respx.mock
def test_the_value_spelling_the_guide_keeps_is_identical_to_the_member() -> None:
    """The guide tells readers ``.value`` needs no cleanup; this is that promise."""
    source = next(s for s in CALLS if ".value" in s)
    route = respx.get(f"{BASE_URL}/models").mock(return_value=httpx.Response(200, json=MODEL_LIST))
    pc = Pinecone(api_key="key", host=BASE_URL)

    exec(source, {**NAMESPACE, "pc": pc})  # noqa: S102
    with_value = str(route.calls.last.request.url)

    pc.inference.list_models(type="embed", vector_type=VectorType.DENSE)
    assert str(route.calls.last.request.url) == with_value


@respx.mock
def test_the_mangled_literal_example_raises_the_error_the_guide_prints() -> None:
    source = next(s for s in CALLS if "PineconeValueError" in s)
    expected = re.search(r"# PineconeValueError: (.+)", source)
    assert expected, "the mangled-literal block no longer shows its error message"

    route = respx.get(f"{BASE_URL}/models").mock(return_value=httpx.Response(200, json=MODEL_LIST))
    with pytest.raises(PineconeValueError) as excinfo:
        exec(source, {**NAMESPACE, "pc": Pinecone(api_key="key", host=BASE_URL)})  # noqa: S102

    assert str(excinfo.value) == expected.group(1).strip()
    assert not route.calls


@respx.mock
def test_the_sdk_surfaces_the_server_wording_the_guide_tells_readers_to_grep_for() -> None:
    message = next(s for s in _blocks("text")).strip()
    respx.get(f"{BASE_URL}/models").mock(
        return_value=httpx.Response(400, json=make_error_response(400, message))
    )

    with pytest.raises(ApiError) as excinfo:
        Pinecone(api_key="key", host=BASE_URL).inference.list_models(vector_type=VectorType.DENSE)

    assert excinfo.value.message == message


def test_the_guide_says_the_fix_covers_both_lanes_and_the_facade() -> None:
    text = GUIDE.read_text()
    assert re.search(r"both the sync and the async\s+client", text)
    assert "pc.inference.model.list()" in text
