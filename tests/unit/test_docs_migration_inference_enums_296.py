"""Executes ``docs/migration/v10-2026-07-inference-model-enums.md`` (#296).

Same discipline as ``test_docs_migration_sparse_hybrid_332.py``: the guide's
table and code blocks are read out of the published file and run, never
transcribed here, so a transcription cannot drift from what a reader copies.

The guide makes four claims this file holds to:

1. Every ``"model" sent now`` cell is the string the SDK really puts on the wire
   for that argument.
2. Every ``"model" sent before`` cell is really what ``str()`` produced, which
   is what the old code path sent. When a Python release or a ``StrEnum``
   migration makes that stop being true, this fails and the guide's premise
   needs rewriting.
3. Passing ``.value`` and passing the member produce byte-identical requests.
4. The literal mangled string is forwarded verbatim and raises ``NotFoundError``
   with the message the guide prints.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import httpx
import orjson
import pytest
import respx

from pinecone import AsyncPinecone, EmbedModel, Pinecone, RerankModel
from pinecone.errors.exceptions import NotFoundError
from tests.factories import make_embed_response, make_error_response, make_rerank_response

BASE_URL = "https://api.test.pinecone.io"
GUIDE = Path(__file__).resolve().parents[2] / "docs/migration/v10-2026-07-inference-model-enums.md"
NAMESPACE: dict[str, Any] = {"EmbedModel": EmbedModel, "RerankModel": RerankModel}


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


def _wire_model(argument: str) -> str:
    """Run the real SDK call for one table row and return the model it sent."""
    model = eval(argument, dict(NAMESPACE))  # noqa: S307
    is_rerank = "Rerank" in argument
    path, payload = (
        ("/rerank", make_rerank_response()) if is_rerank else ("/embed", make_embed_response())
    )
    route = respx.post(f"{BASE_URL}{path}").mock(return_value=httpx.Response(200, json=payload))
    pc = Pinecone(api_key="key", host=BASE_URL)
    if is_rerank:
        pc.inference.rerank(model=model, query="q", documents=["doc"])
    else:
        pc.inference.embed(model=model, inputs=["hello"])
    return str(orjson.loads(route.calls.last.request.content)["model"])


@pytest.mark.parametrize(("argument", "before", "now"), ROWS, ids=[r[0] for r in ROWS])
@respx.mock
def test_the_table_says_what_the_sdk_really_sends(argument: str, before: str, now: str) -> None:
    assert _wire_model(argument) == now


@pytest.mark.parametrize(("argument", "before", "now"), ROWS, ids=[r[0] for r in ROWS])
def test_the_before_column_is_still_what_str_produces(argument: str, before: str, now: str) -> None:
    """The old code path was ``str(model)``, so ``str()`` is what the column records."""
    assert str(eval(argument, dict(NAMESPACE))) == before  # noqa: S307


@pytest.mark.asyncio
@respx.mock
async def test_the_value_spelling_the_guide_keeps_is_byte_identical_to_the_member() -> None:
    """The guide tells readers ``.value`` needs no cleanup; this is that promise."""
    source = next(s for s in CALLS if ".value" in s)
    route = respx.post(f"{BASE_URL}/embed").mock(
        return_value=httpx.Response(200, json=make_embed_response())
    )

    pc = Pinecone(api_key="key", host=BASE_URL)
    exec(source, {**NAMESPACE, "pc": pc})  # noqa: S102
    with_value = bytes(route.calls.last.request.content)

    pc.inference.embed(model=EmbedModel.Multilingual_E5_Large, inputs=["hello"])
    with_member = bytes(route.calls.last.request.content)

    async with AsyncPinecone(api_key="key", host=BASE_URL) as apc:
        await apc.inference.embed(model=EmbedModel.Multilingual_E5_Large, inputs=["hello"])
    from_async = bytes(route.calls.last.request.content)

    assert with_value == with_member == from_async


@respx.mock
def test_the_plain_string_example_reaches_the_wire_unchanged() -> None:
    """A model id no installed member covers must still be accepted."""
    source = next(s for s in CALLS if "some-newer" in s)
    route = respx.post(f"{BASE_URL}/embed").mock(
        return_value=httpx.Response(200, json=make_embed_response())
    )

    exec(source, {**NAMESPACE, "pc": Pinecone(api_key="key", host=BASE_URL)})  # noqa: S102

    sent = orjson.loads(route.calls.last.request.content)["model"]
    assert f'"{sent}"' in source


@respx.mock
def test_the_mangled_string_example_raises_the_error_the_guide_prints() -> None:
    source = next(s for s in CALLS if "NotFoundError" in s)
    expected = re.search(r"# NotFoundError: (.+)", source)
    assert expected, "the mangled-string block no longer shows its error message"
    message = expected.group(1).strip()

    route = respx.post(f"{BASE_URL}/embed").mock(
        return_value=httpx.Response(404, json=make_error_response(404, message))
    )

    with pytest.raises(NotFoundError) as excinfo:
        exec(source, {**NAMESPACE, "pc": Pinecone(api_key="key", host=BASE_URL)})  # noqa: S102

    assert excinfo.value.message == message
    assert orjson.loads(route.calls.last.request.content)["model"] in message


def test_the_guide_names_the_log_string_a_reader_would_grep_for() -> None:
    text = GUIDE.read_text()
    assert "Model 'EmbedModel.Multilingual_E5_Large' not found" in text
    assert re.search(r"both the sync and the async\s+client", text)
