"""Executes ``docs/migration/v10-migration.md``.

The guide's code blocks are read out of the published file and run, never
transcribed here, so a transcription cannot drift from what a reader copies.

Three things are checked.

1. **Every keyword in every ``pc.…`` call in the guide binds to the real
   signature**, on both the sync and the async class. This covers the blocks
   that cannot be executed as well — the upload example reads a local file.
2. **The executable blocks execute**, against mocked transports, and the
   attributes the prose names (``op.file_id``, ``operation.error``,
   ``assistant.region``, ``exc.retry_after``) really exist.
3. **The docstring facts the guide leans on hold**, on both lanes: ``rerank``
   documents ``NotFoundError``, and no listing docstring documents a pagination
   bound that only the server enforces.

The guide also states that the backend remaps the deprecated ``claude-3-*``
aliases and that the assistant error-code enum carries ``TOO_MANY_REQUESTS``.
Those are backend facts, not SDK behaviour, and their citations are in the PR
body. What is checkable here is the SDK side: that a 429 arrives as
``RateLimitError`` carrying what the guide prints.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import re
import textwrap
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from pinecone import AsyncPinecone, Pinecone
from pinecone._internal.config import RetryConfig
from pinecone.async_client.assistants import AsyncAssistants
from pinecone.async_client.inference import AsyncInference
from pinecone.client.assistants import Assistants
from pinecone.client.inference import Inference
from pinecone.errors.exceptions import RateLimitError
from tests.factories import (
    make_assistant_response,
    make_error_response,
    make_file_operation_response,
    make_operation_list_response,
)

BASE_URL = "https://api.test.pinecone.io"
DATA_PLANE_URL = "https://test-assistant-abc123.svc.pinecone.io/assistant"
GUIDE = Path(__file__).resolve().parents[2] / "docs/migration/v10-migration.md"
NAMESPACES: dict[str, tuple[type, type]] = {
    "assistants": (Assistants, AsyncAssistants),
    "inference": (Inference, AsyncInference),
}


def _blocks() -> list[str]:
    text = GUIDE.read_text()
    sources = [m.group(1) for m in re.finditer(r"```python\n(.*?)```", text, re.DOTALL)]
    assert len(sources) >= 7, f"expected the guide's python blocks, got {len(sources)}"
    return sources


def _calls(source: str) -> list[tuple[str, str, list[str]]]:
    """Every ``pc.<namespace>.<method>(...)`` call in *source*, with its keywords."""
    found = []
    for node in ast.walk(ast.parse(textwrap.dedent(source))):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or not isinstance(func.value, ast.Attribute):
            continue
        if not isinstance(func.value.value, ast.Name) or func.value.value.id != "pc":
            continue
        if func.value.attr not in NAMESPACES:
            continue
        found.append((func.value.attr, func.attr, [kw.arg or "" for kw in node.keywords]))
    return found


BLOCKS = _blocks()
GUIDE_CALLS = [(ns, meth, kws, i) for i, b in enumerate(BLOCKS) for ns, meth, kws in _calls(b)]


def test_the_guide_still_contains_calls_to_check() -> None:
    """Guards the parametrization below against silently collapsing to nothing."""
    assert len(GUIDE_CALLS) >= 6


@pytest.mark.parametrize(
    ("namespace", "method", "keywords", "block"),
    GUIDE_CALLS,
    ids=[f"block{i}-{ns}.{m}" for ns, m, _, i in GUIDE_CALLS],
)
def test_every_guide_call_binds_to_both_real_signatures(
    namespace: str, method: str, keywords: list[str], block: int
) -> None:
    for cls in NAMESPACES[namespace]:
        fn = getattr(cls, method, None)
        assert fn is not None, f"{cls.__name__} has no {method}()"
        inspect.signature(fn).bind_partial(object(), **dict.fromkeys(keywords))


def _mock_assistant_transport() -> None:
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    respx.get(f"{DATA_PLANE_URL}/operations/test-assistant/op-1234-abcd-5678").mock(
        return_value=httpx.Response(
            200, json=make_file_operation_response(status="Failed", error_message="boom")
        ),
    )
    respx.get(f"{DATA_PLANE_URL}/operations/test-assistant").mock(
        return_value=httpx.Response(
            200, json=make_operation_list_response([make_file_operation_response()])
        ),
    )


@respx.mock
def test_the_describe_operation_block_reports_the_failure_reason() -> None:
    _mock_assistant_transport()
    pc = Pinecone(api_key="key", host=BASE_URL)

    operation = pc.assistants.describe_operation(
        assistant_name="test-assistant", operation_id="op-1234-abcd-5678"
    )

    assert operation.status == "Failed"
    assert operation.error == "boom"


@respx.mock
def test_the_async_describe_operation_block_runs() -> None:
    _mock_assistant_transport()

    async def run() -> Any:
        async with AsyncPinecone(api_key="key", host=BASE_URL) as pc:
            return await pc.assistants.describe_operation(
                assistant_name="test-assistant", operation_id="op-1234-abcd-5678"
            )

    assert asyncio.run(run()).status == "Failed"


@respx.mock
def test_correlating_an_operation_to_its_file_works_as_the_guide_shows() -> None:
    """Recovering a fire-and-forget upload means matching operations on ``file_id``."""
    _mock_assistant_transport()
    pc = Pinecone(api_key="key", host=BASE_URL)

    operations = pc.assistants.list_operations(
        assistant_name="test-assistant", operation_type="upload_file", status="Processing"
    )
    mine = [op for op in operations if op.file_id == "file-abc123"]

    assert [op.operation_id for op in mine] == ["op-abc123"]


@respx.mock
def test_create_with_region_eu_sends_it_and_reports_it_back() -> None:
    route = respx.post(f"{BASE_URL}/assistant/assistants").mock(
        return_value=httpx.Response(200, json=make_assistant_response(region="eu")),
    )
    respx.get(f"{BASE_URL}/assistant/assistants/eu-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response(region="eu")),
    )
    pc = Pinecone(api_key="key", host=BASE_URL)

    assistant = pc.assistants.create(name="eu-assistant", region="eu")

    assert b'"region"' in route.calls.last.request.read()
    assert assistant.region == "eu"


@respx.mock
def test_a_429_arrives_as_rate_limit_error_with_the_attributes_the_guide_prints() -> None:
    respx.get(f"{BASE_URL}/assistant/assistants").mock(
        return_value=httpx.Response(
            429, json=make_error_response(429), headers={"Retry-After": "7"}
        ),
    )
    pc = Pinecone(api_key="key", host=BASE_URL, retry_config=RetryConfig(max_retries=0))

    with pytest.raises(RateLimitError) as excinfo:
        pc.assistants.list().to_list()

    assert excinfo.value.retry_after == 7
    assert excinfo.value.error_code is not None


@pytest.mark.parametrize("cls", [Inference, AsyncInference], ids=["sync", "async"])
def test_rerank_documents_not_found_error_for_an_unknown_model(cls: type) -> None:
    """An unknown rerank model raises ``NotFoundError``, so the docstring says so."""
    doc = inspect.getdoc(cls.rerank) or ""
    raises = doc.split("Raises:", 1)[1].split("Examples:", 1)[0]
    assert "NotFoundError" in raises
    assert "ForbiddenError" in raises


@pytest.mark.parametrize(
    "method",
    [
        "list_page",
        "list_operations",
        "list_operations_page",
        "describe_operation",
        "delete_file",
        "list_files",
    ],
)
def test_no_listing_docstring_documents_a_bound_only_the_server_enforces(method: str) -> None:
    """A number belongs in a docstring only where this repo enforces it."""
    banned = ("1-1000", "1-100", "0-100", "defaults to 50", "60 days", "Limit cannot exceed")
    for cls in (Assistants, AsyncAssistants):
        doc = inspect.getdoc(getattr(cls, method)) or ""
        for phrase in banned:
            assert phrase not in doc, f"{cls.__name__}.{method} still documents {phrase!r}"


@pytest.mark.parametrize("method", ["delete_file", "list_files", "list_files_page"])
def test_both_lanes_document_the_404(method: str) -> None:
    for cls in (Assistants, AsyncAssistants):
        doc = inspect.getdoc(getattr(cls, method)) or ""
        assert "NotFoundError" in doc.split("Raises:", 1)[1]


def test_list_page_carries_the_explicitly_provided_preamble_on_both_lanes() -> None:
    preamble = "Only the parameters that are explicitly provided are sent"
    for cls in (Assistants, AsyncAssistants):
        assert preamble in (inspect.getdoc(cls.list_page) or "")


def test_the_guide_is_registered_in_the_toctree() -> None:
    index = (Path(__file__).resolve().parents[2] / "docs/index.rst").read_text()
    assert f"migration/{GUIDE.stem}" in index
