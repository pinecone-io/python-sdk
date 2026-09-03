"""Sync/async parity for the assistant chat/context data plane (#127 ∥ #130).

``Assistants`` and ``AsyncAssistants`` build the chat, chat_completions and
context bodies with the same code and decode them through the same
``AssistantsAdapter``, so the two transports should differ only in ``await``.
These tests hold them to that on the axes a transport port can quietly break:
identical request snapshots on the wire (method, the ``/assistant``-prefixed
path, query, body, and the 2026-07 version header) for identical arguments;
identical signatures, defaults and return annotations modulo the
``Async*Stream`` return types; identical exception types and messages for the
client-side rejection matrix and for a backend 400; identical parsed chunk
sequences out of the streaming wrappers; and — the 2026-07 delta this ticket
carries — identical documented model lists in the two docstrings, so the enum
churn cannot land on one transport only.

The control-plane and evaluation surface has its own module,
``tests/unit/test_async_assistants_parity.py``; this one follows the request
snapshot pattern of ``tests/unit/test_async_documents_parity.py``.

Streaming is compared on parsed output rather than on iteration mechanics:
``ChatStream`` is a single-pass ``Iterator`` and ``AsyncChatStream`` a
single-pass ``AsyncIterator``, which is the one difference that is meant to
exist. The chunk structs they yield are the same msgspec types, so equality on
the drained sequence is the strongest available statement.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from typing import Any

import httpx
import orjson
import pytest
import respx

from pinecone._internal.config import PineconeConfig
from pinecone._internal.constants import API_VERSION_HEADER
from pinecone.async_client.assistants import (
    _STREAM_TIMEOUT_FLOOR_SECONDS as _ASYNC_STREAM_FLOOR,
)
from pinecone.async_client.assistants import (
    AsyncAssistants,
)
from pinecone.async_client.assistants import (
    _stream_timeout as _async_stream_timeout,
)
from pinecone.client.assistants import (
    _STREAM_TIMEOUT_FLOOR_SECONDS as _SYNC_STREAM_FLOOR,
)
from pinecone.client.assistants import (
    Assistants,
)
from pinecone.client.assistants import (
    _stream_timeout as _sync_stream_timeout,
)
from pinecone.models.assistant.streaming import (
    AsyncChatCompletionStream,
    AsyncChatStream,
    ChatCompletionStream,
    ChatStream,
)

BASE_URL = "https://api.test.pinecone.io"
CONTROL_URL = f"{BASE_URL}/assistant"
DATA_HOST = "https://prod-1-data.ke.pinecone.io"
DATA_URL = f"{DATA_HOST}/assistant"

ASSISTANT_NAME = "parity-assistant"

ASSISTANT: dict[str, Any] = {
    "name": ASSISTANT_NAME,
    "status": "Ready",
    "host": DATA_HOST,
    "region": "us",
}

_METHODS = ["chat", "chat_completions", "context"]
_STREAMING_METHODS = ["chat", "chat_completions"]

# The 2026-07 x-enum per endpoint, hardcoded from the spec on purpose: importing
# it from the docstring under test would let a wrong list certify itself.
CHAT_MODELS = ("gpt-4o", "gpt-4.1", "gpt-5", "o4-mini", "claude-sonnet-4-5", "gemini-2.5-pro")
CHAT_COMPLETION_MODELS = ("gpt-4o", "gpt-4.1", "o4-mini", "claude-sonnet-4-5", "gemini-2.5-pro")
DEPRECATED_ALIASES = ("claude-3-5-sonnet", "claude-3-7-sonnet")

_ENDPOINT_MODELS = {"chat": CHAT_MODELS, "chat_completions": CHAT_COMPLETION_MODELS}

_CALLS: dict[str, dict[str, Any]] = {
    "chat": {
        "assistant_name": ASSISTANT_NAME,
        "messages": [{"content": "What is Pinecone?"}],
        "model": "gpt-5",
        "temperature": 0.25,
        "filter": {"category": {"$eq": "docs"}},
        "include_highlights": True,
        "context_options": {"top_k": 12},
    },
    "chat_completions": {
        "assistant_name": ASSISTANT_NAME,
        "messages": [{"role": "system", "content": "Be brief."}, {"content": "What is Pinecone?"}],
        "model": "claude-sonnet-4-5",
        "temperature": 0.5,
        "filter": {"category": {"$eq": "docs"}},
    },
    "context": {
        "assistant_name": ASSISTANT_NAME,
        "query": "What is Pinecone?",
        "top_k": 20,
        "snippet_size": 512,
        "multimodal": True,
        "include_binary_content": False,
        "filter": {"category": {"$eq": "docs"}},
    },
}

_ERROR_CASES: list[tuple[str, dict[str, Any]]] = [
    (
        "chat",
        {
            "assistant_name": ASSISTANT_NAME,
            "messages": [{"content": "hi"}],
            "stream": True,
            "json_response": True,
        },
    ),
    ("chat", {"assistant_name": ASSISTANT_NAME, "messages": [{"role": "user"}]}),
    ("chat_completions", {"assistant_name": ASSISTANT_NAME, "messages": [{"role": "user"}]}),
    (
        "context",
        {"assistant_name": ASSISTANT_NAME, "query": "q", "messages": [{"content": "hi"}]},
    ),
    ("context", {"assistant_name": ASSISTANT_NAME}),
    ("context", {"assistant_name": ASSISTANT_NAME, "query": ""}),
    ("context", {"assistant_name": ASSISTANT_NAME, "messages": []}),
    ("context", {"assistant_name": ASSISTANT_NAME, "query": "", "messages": []}),
    ("context", {"assistant_name": ASSISTANT_NAME, "query": "q", "top_k": -1}),
    ("context", {"assistant_name": ASSISTANT_NAME, "query": "q", "snippet_size": -1}),
    ("context", {"assistant_name": ASSISTANT_NAME, "messages": [{"role": "user"}]}),
]

CHAT_SSE = (
    b'data: {"type":"message_start","id":"c1","model":"gpt-5","role":"assistant",'
    b'"context_snippet_count":16}\n\n'
    b'data: {"type":"content_chunk","id":"c1","model":"gpt-5","delta":{"content":"Pinecone is "}}\n\n'
    b'data: {"type":"content_chunk","id":"c1","model":"gpt-5",'
    b'"delta":{"content":"a managed vector database."}}\n\n'
    b'data: {"type":"citation","id":"c1","model":"gpt-5",'
    b'"citation":{"position":37,"references":[]}}\n\n'
    b'data: {"type":"message_end","id":"c1","model":"gpt-5","finish_reason":"tool_calls",'
    b'"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}\n\n'
    b"data: [DONE]\n\n"
)

COMPLETION_SSE = (
    b'data: {"id":"k1","model":"claude-sonnet-4-5",'
    b'"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}\n\n'
    b'data: {"id":"k1","model":"claude-sonnet-4-5",'
    b'"choices":[{"index":0,"delta":{"content":"Pinecone is "},"finish_reason":null}]}\n\n'
    b'data: {"id":"k1","model":"claude-sonnet-4-5",'
    b'"choices":[{"index":0,"delta":{"content":"a managed vector database."},'
    b'"finish_reason":null}]}\n\n'
    b'data: {"id":"k1","model":"claude-sonnet-4-5",'
    b'"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],'
    b'"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}\n\n'
    b"data: [DONE]\n\n"
)

CHAT_RESPONSE: dict[str, Any] = {
    "id": "c1",
    "model": "gpt-5",
    "finish_reason": "tool_calls",
    "message": {"role": "assistant", "content": "Pinecone is a managed vector database."},
    "citations": [],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    "context_snippet_count": 16,
    "content_filter_results": {"spec": "openai", "results": {}},
}

COMPLETION_RESPONSE: dict[str, Any] = {
    "id": "k1",
    "model": "claude-sonnet-4-5",
    "choices": [
        {
            "index": 0,
            "finish_reason": "tool_calls",
            "message": {"role": "assistant", "content": "Pinecone is a managed vector database."},
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}

CONTEXT_RESPONSE: dict[str, Any] = {
    "id": "x1",
    "snippets": [],
    "usage": {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10},
}

BACKEND_400_MESSAGE = (
    "Invalid model `claude-3-5-sonnet`. Expected one of: gpt-4o, gpt-4.1, o4-mini, "
    "gpt-5, claude-sonnet-4-5, gemini-2.5-pro."
)


def _comparable(annotation: Any) -> str:
    """Erase the one difference the two transports are allowed to have."""
    return (
        str(annotation)
        .replace("AsyncChatCompletionStream", "ChatCompletionStream")
        .replace("AsyncChatStream", "ChatStream")
    )


def _model_doc_block(method: Any) -> str:
    """The ``model`` entry of a method's Args section, dedented.

    Parity on the rendered prose is stronger than parity on a parsed list: it
    also catches a remap note or a validation caveat landing on one transport
    only.
    """
    lines = (inspect.getdoc(method) or "").splitlines()
    starts = [i for i, line in enumerate(lines) if line.strip().startswith("model (str):")]
    assert len(starts) == 1, "expected exactly one `model (str):` entry in the Args section"

    start = starts[0]
    indent = len(lines[start]) - len(lines[start].lstrip())
    block = [lines[start]]
    for line in lines[start + 1 :]:
        if line.strip() and len(line) - len(line.lstrip()) <= indent:
            break
        block.append(line)
    return " ".join(" ".join(block).split())


def _quoted_run(block: str, marker: str) -> tuple[str, ...]:
    """The comma-and-``and``-separated run of ``"quoted"`` names after *marker*.

    Reading the whole block would also pick up names the prose mentions in
    order to explain their *absence* — ``chat_completions`` names ``gpt-5``
    only to say the spec does not document it there. So the run stops at the
    first gap that is anything other than a separator.
    """
    start = block.find(marker)
    assert start != -1, f"no {marker!r} in the documented model block"

    tail = block[start + len(marker) :]
    names: list[str] = []
    cursor = 0
    for match in re.finditer(r'``"([^"]+)"``', tail):
        gap = tail[cursor : match.start()]
        if names and not re.fullmatch(r"\s*,?\s*(and\s+)?", gap):
            break
        names.append(match.group(1))
        cursor = match.end()
    return tuple(names)


def _documented_models(method: Any) -> tuple[str, ...]:
    return _quoted_run(_model_doc_block(method), "documents for this endpoint are")


def _documented_aliases(method: Any) -> tuple[str, ...]:
    return _quoted_run(_model_doc_block(method), "removed aliases")


@pytest.fixture
def sync_assistants(respx_mock: respx.MockRouter) -> Iterator[Assistants]:
    respx_mock.get(f"{CONTROL_URL}/assistants/{ASSISTANT_NAME}").mock(
        return_value=httpx.Response(200, json=ASSISTANT)
    )
    client = Assistants(config=PineconeConfig(api_key="parity-key", host=BASE_URL))
    yield client
    client.close()


@pytest.fixture
async def async_assistants(respx_mock: respx.MockRouter) -> AsyncIterator[AsyncAssistants]:
    respx_mock.get(f"{CONTROL_URL}/assistants/{ASSISTANT_NAME}").mock(
        return_value=httpx.Response(200, json=ASSISTANT)
    )
    client = AsyncAssistants(config=PineconeConfig(api_key="parity-key", host=BASE_URL))
    yield client
    await client.close()


def _register_routes(respx_mock: respx.MockRouter) -> dict[str, respx.Route]:
    return {
        "chat": respx_mock.post(f"{DATA_URL}/chat/{ASSISTANT_NAME}").mock(
            return_value=httpx.Response(200, json=CHAT_RESPONSE)
        ),
        "chat_completions": respx_mock.post(
            f"{DATA_URL}/chat/{ASSISTANT_NAME}/chat/completions"
        ).mock(return_value=httpx.Response(200, json=COMPLETION_RESPONSE)),
        "context": respx_mock.post(f"{DATA_URL}/chat/{ASSISTANT_NAME}/context").mock(
            return_value=httpx.Response(200, json=CONTEXT_RESPONSE)
        ),
    }


def _snapshot(request: httpx.Request) -> dict[str, Any]:
    return {
        "method": request.method,
        "raw_path": request.url.raw_path.decode(),
        "query": dict(request.url.params),
        "body": orjson.loads(request.content) if request.content else None,
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


@pytest.mark.parametrize("method_name", _METHODS)
async def test_request_snapshot_parity(
    method_name: str,
    sync_assistants: Assistants,
    async_assistants: AsyncAssistants,
    respx_mock: respx.MockRouter,
) -> None:
    routes = _register_routes(respx_mock)
    route = routes[method_name]
    kwargs = _CALLS[method_name]

    getattr(sync_assistants, method_name)(**kwargs)
    await getattr(async_assistants, method_name)(**kwargs)

    assert len(route.calls) == 2, "each transport must have issued exactly one request"
    sync_snapshot = _snapshot(route.calls[0].request)
    async_snapshot = _snapshot(route.calls[1].request)

    assert async_snapshot == sync_snapshot
    assert async_snapshot["api_version"] == "2026-07"
    assert async_snapshot["raw_path"].startswith(f"/assistant/chat/{ASSISTANT_NAME}")


@pytest.mark.parametrize("method_name", _STREAMING_METHODS)
async def test_streaming_request_snapshot_parity(
    method_name: str,
    sync_assistants: Assistants,
    async_assistants: AsyncAssistants,
    respx_mock: respx.MockRouter,
) -> None:
    body = CHAT_SSE if method_name == "chat" else COMPLETION_SSE
    routes = _register_routes(respx_mock)
    route = routes[method_name]
    route.mock(return_value=httpx.Response(200, content=body))
    kwargs = {**_CALLS[method_name], "stream": True}

    sync_stream = getattr(sync_assistants, method_name)(**kwargs)
    list(sync_stream)
    async_stream = await getattr(async_assistants, method_name)(**kwargs)
    [chunk async for chunk in async_stream]

    assert len(route.calls) == 2
    sync_snapshot = _snapshot(route.calls[0].request)
    async_snapshot = _snapshot(route.calls[1].request)

    assert async_snapshot == sync_snapshot
    assert async_snapshot["api_version"] == "2026-07"
    assert async_snapshot["body"]["stream"] is True

    sync_timeout = route.calls[0].request.extensions["timeout"]
    async_timeout = route.calls[1].request.extensions["timeout"]
    assert async_timeout == sync_timeout
    assert sync_timeout["read"] == _SYNC_STREAM_FLOOR


def test_streaming_timeout_floor_parity() -> None:
    """The floor is declared once per transport module; the two must not drift."""
    assert _ASYNC_STREAM_FLOOR == _SYNC_STREAM_FLOOR
    assert _sync_stream_timeout(30.0, None) == _async_stream_timeout(30.0, None)
    assert _sync_stream_timeout(30.0, 7.5) == _async_stream_timeout(30.0, 7.5) == 7.5


async def test_chat_stream_yields_the_same_chunks(
    sync_assistants: Assistants,
    async_assistants: AsyncAssistants,
    respx_mock: respx.MockRouter,
) -> None:
    route = respx_mock.post(f"{DATA_URL}/chat/{ASSISTANT_NAME}")
    route.mock(return_value=httpx.Response(200, content=CHAT_SSE))
    kwargs = {"assistant_name": ASSISTANT_NAME, "messages": [{"content": "hi"}], "stream": True}

    sync_stream = sync_assistants.chat(**kwargs)
    async_stream = await async_assistants.chat(**kwargs)
    assert isinstance(sync_stream, ChatStream)
    assert isinstance(async_stream, AsyncChatStream)

    sync_chunks = list(sync_stream)
    async_chunks = [chunk async for chunk in async_stream]

    assert async_chunks == sync_chunks
    assert [chunk.type for chunk in async_chunks] == [
        "message_start",
        "content_chunk",
        "content_chunk",
        "citation",
        "message_end",
    ]
    assert async_chunks[-1].finish_reason == "tool_calls"


async def test_chat_completion_stream_yields_the_same_chunks(
    sync_assistants: Assistants,
    async_assistants: AsyncAssistants,
    respx_mock: respx.MockRouter,
) -> None:
    route = respx_mock.post(f"{DATA_URL}/chat/{ASSISTANT_NAME}/chat/completions")
    route.mock(return_value=httpx.Response(200, content=COMPLETION_SSE))
    kwargs = {"assistant_name": ASSISTANT_NAME, "messages": [{"content": "hi"}], "stream": True}

    sync_stream = sync_assistants.chat_completions(**kwargs)
    async_stream = await async_assistants.chat_completions(**kwargs)
    assert isinstance(sync_stream, ChatCompletionStream)
    assert isinstance(async_stream, AsyncChatCompletionStream)

    assert [chunk async for chunk in async_stream] == list(sync_stream)


async def test_chat_stream_collect_parity(
    sync_assistants: Assistants,
    async_assistants: AsyncAssistants,
    respx_mock: respx.MockRouter,
) -> None:
    route = respx_mock.post(f"{DATA_URL}/chat/{ASSISTANT_NAME}")
    route.mock(return_value=httpx.Response(200, content=CHAT_SSE))
    kwargs = {"assistant_name": ASSISTANT_NAME, "messages": [{"content": "hi"}], "stream": True}

    sync_stream = sync_assistants.chat(**kwargs)
    async_stream = await async_assistants.chat(**kwargs)
    assert isinstance(sync_stream, ChatStream)
    assert isinstance(async_stream, AsyncChatStream)

    assert await async_stream.collect() == sync_stream.collect()


async def test_non_streaming_response_parity(
    sync_assistants: Assistants,
    async_assistants: AsyncAssistants,
    respx_mock: respx.MockRouter,
) -> None:
    """The 2026-07 additions must land identically on both adapters' output."""
    _register_routes(respx_mock)

    sync_result = sync_assistants.chat(assistant_name=ASSISTANT_NAME, messages=[{"content": "hi"}])
    async_result = await async_assistants.chat(
        assistant_name=ASSISTANT_NAME, messages=[{"content": "hi"}]
    )

    assert async_result == sync_result
    assert async_result.context_snippet_count == 16
    assert async_result.content_filter_results == CHAT_RESPONSE["content_filter_results"]
    assert async_result.finish_reason == "tool_calls"


@pytest.mark.parametrize("method_name", _METHODS)
def test_parameter_parity(method_name: str) -> None:
    sync_params = dict(inspect.signature(getattr(Assistants, method_name)).parameters)
    async_params = dict(inspect.signature(getattr(AsyncAssistants, method_name)).parameters)

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
    sync_return = inspect.signature(getattr(Assistants, method_name)).return_annotation
    async_return = inspect.signature(getattr(AsyncAssistants, method_name)).return_annotation

    assert _comparable(sync_return) == _comparable(async_return), (
        f"{method_name}: return annotation differs (sync={sync_return}, async={async_return})"
    )


@pytest.mark.parametrize("method_name", _METHODS)
def test_keyword_only_parity(method_name: str) -> None:
    for cls in (Assistants, AsyncAssistants):
        params = inspect.signature(getattr(cls, method_name)).parameters
        positional = [
            name
            for name, param in params.items()
            if name != "self" and param.kind is not inspect.Parameter.KEYWORD_ONLY
        ]
        assert positional == [], (
            f"{cls.__name__}.{method_name} must be keyword-only, found {positional}"
        )


@pytest.mark.parametrize("method_name", sorted(_ENDPOINT_MODELS))
def test_documented_model_list_parity(method_name: str) -> None:
    """AC4: the enum churn must be documented identically on both transports."""
    sync_block = _model_doc_block(getattr(Assistants, method_name))
    async_block = _model_doc_block(getattr(AsyncAssistants, method_name))

    assert async_block == sync_block, (
        f"{method_name}: the documented model prose differs between transports"
    )


@pytest.mark.parametrize("method_name", sorted(_ENDPOINT_MODELS))
@pytest.mark.parametrize("cls", [Assistants, AsyncAssistants], ids=["sync", "async"])
def test_documented_models_match_the_spec_enum(cls: type, method_name: str) -> None:
    documented = _documented_models(getattr(cls, method_name))
    expected = _ENDPOINT_MODELS[method_name]

    assert documented == expected, (
        f"{cls.__name__}.{method_name}: documented models {documented} "
        f"do not match the 2026-07 x-enum {expected}"
    )


@pytest.mark.parametrize("method_name", sorted(_ENDPOINT_MODELS))
@pytest.mark.parametrize("cls", [Assistants, AsyncAssistants], ids=["sync", "async"])
def test_deprecated_aliases_are_documented_as_remapped(cls: type, method_name: str) -> None:
    """#220: the backend remaps the retired claude aliases, it does not reject them.

    Documenting them as rejected would send callers hunting for an error they
    will never see; documenting nothing would leave the remap invisible.
    """
    block = _model_doc_block(getattr(cls, method_name))

    assert _documented_aliases(getattr(cls, method_name)) == DEPRECATED_ALIASES
    assert "still accepted but deprecated" in block
    assert 'remaps them to ``"claude-sonnet-4-5"``' in block


@pytest.mark.parametrize("cls", [Assistants, AsyncAssistants], ids=["sync", "async"])
def test_gpt_5_is_documented_on_chat_only(cls: type) -> None:
    """The spec lists ``gpt-5`` on ChatRequest but not on SearchCompletions."""
    assert "gpt-5" in _documented_models(cls.chat)
    assert "gpt-5" not in _documented_models(cls.chat_completions)


@pytest.mark.parametrize(
    "method_name,kwargs",
    _ERROR_CASES,
    ids=[f"{name}-{i}" for i, (name, _) in enumerate(_ERROR_CASES)],
)
async def test_validation_error_parity(
    method_name: str,
    kwargs: dict[str, Any],
    sync_assistants: Assistants,
    async_assistants: AsyncAssistants,
) -> None:
    sync_type, sync_message = _raised(lambda: getattr(sync_assistants, method_name)(**kwargs))
    async_type, async_message = await _raised_async(
        lambda: getattr(async_assistants, method_name)(**kwargs)
    )

    assert async_type is sync_type
    assert async_message == sync_message


async def test_backend_400_error_parity(
    sync_assistants: Assistants,
    async_assistants: AsyncAssistants,
    respx_mock: respx.MockRouter,
) -> None:
    """A rejected model name must read identically whichever transport sent it."""
    respx_mock.post(f"{DATA_URL}/chat/{ASSISTANT_NAME}").mock(
        return_value=httpx.Response(
            400,
            json={
                "status": 400,
                "error": {"code": "INVALID_ARGUMENT", "message": BACKEND_400_MESSAGE},
            },
        )
    )
    kwargs = {
        "assistant_name": ASSISTANT_NAME,
        "messages": [{"content": "hi"}],
        "model": "claude-3-5-sonnet",
    }

    sync_type, sync_message = _raised(lambda: sync_assistants.chat(**kwargs))
    async_type, async_message = await _raised_async(lambda: async_assistants.chat(**kwargs))

    assert async_type is sync_type
    assert async_message == sync_message
    assert BACKEND_400_MESSAGE in async_message
