"""2026-07 conformance for the asyncio transport of the three assistant_data
chat/context operations.

The sync variants live in ``test_assistant_data_chat_2026_07.py``; both may
claim the same operation (see README, "Additional rules"), and these add no
operation ids to the coverage numerator. What they add is the guarantee that
``AsyncAssistants`` puts the same method, the same ``/assistant``-prefixed path
(#173) and the same ``X-Pinecone-Api-Version`` on the wire, decodes the same
2026-07 response additions through the shared adapter, and — the part only an
async test can prove — that ``AsyncChatStream``/``AsyncChatCompletionStream``
reassemble the documented ``StreamChatChunkModel`` transcript from an *async*
byte stream, including the ``tool_calls`` finish reason.

Every payload, transcript and expected chunk sequence is imported from the sync
module rather than restated, so the two transports cannot drift apart in the
fixtures.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Iterator
from typing import Any

import httpx
import orjson
import pytest
import respx
from hypothesis import given
from hypothesis import strategies as st

from pinecone._internal.config import PineconeConfig
from pinecone.async_client.assistants import AsyncAssistants
from pinecone.errors.exceptions import ApiError
from pinecone.models.assistant.chat import ChatCompletionResponse, ChatResponse
from pinecone.models.assistant.context import ContextResponse
from pinecone.models.assistant.streaming import AsyncChatCompletionStream, AsyncChatStream
from tests.unit.conformance import api_op
from tests.unit.conformance.test_assistant_data_chat_2026_07 import (
    ASSISTANT,
    ASSISTANT_NAME,
    BASE_URL,
    CHAT,
    CHAT_SSE_EVENTS,
    COMPLETION,
    COMPLETION_SSE_EVENTS,
    CONTENT_FILTER_RESULTS,
    CONTEXT,
    CONTROL_URL,
    DATA_URL,
    EXPECTED_CHAT_STREAM,
    FILE,
    REMOVED_MODEL_ERROR_MESSAGE,
    expected_path,
    sse_body,
    summarise,
)


async def _async_body(pieces: list[bytes]) -> AsyncIterator[bytes]:
    """Feed *pieces* to httpx as an async byte stream, one read per piece."""
    for piece in pieces:
        yield piece


@pytest.fixture
async def async_assistants(respx_mock: respx.MockRouter) -> AsyncGenerator[AsyncAssistants]:
    respx_mock.get(f"{CONTROL_URL}/assistants/{ASSISTANT_NAME}").mock(
        return_value=httpx.Response(200, json=ASSISTANT)
    )
    client = AsyncAssistants(config=PineconeConfig(api_key="conformance-key", host=BASE_URL))
    yield client
    await client.close()


@api_op("assistant_data:chat_assistant")
async def test_async_chat_assistant(
    claim: Any, async_assistants: AsyncAssistants, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.post(f"{DATA_URL}/chat/{ASSISTANT_NAME}").mock(
        return_value=httpx.Response(200, json=CHAT)
    )

    result = await async_assistants.chat(
        assistant_name=ASSISTANT_NAME,
        messages=[{"content": "What is Pinecone?"}],
        model="gpt-5",
    )
    assert isinstance(result, ChatResponse)
    assert result.finish_reason == "tool_calls"
    assert result.context_snippet_count == 16
    assert result.content_filter_results == CONTENT_FILTER_RESULTS
    assert result.citations[0].references[0].file.name == FILE["name"]

    request = route.calls.last.request
    assert orjson.loads(request.content) == {
        "messages": [{"role": "user", "content": "What is Pinecone?"}],
        "model": "gpt-5",
        "stream": False,
    }
    claim.assert_request(request)
    claim.assert_api_version(request)
    claim.assert_roundtrip(
        ChatResponse,
        CHAT,
        optional_absent=["context_snippet_count", "content_filter_results"],
    )


@api_op("assistant_data:chat_completion_assistant")
async def test_async_chat_completion_assistant(
    claim: Any, async_assistants: AsyncAssistants, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.post(f"{DATA_URL}/chat/{ASSISTANT_NAME}/chat/completions").mock(
        return_value=httpx.Response(200, json=COMPLETION)
    )

    result = await async_assistants.chat_completions(
        assistant_name=ASSISTANT_NAME,
        messages=[{"content": "What is Pinecone?"}],
        model="claude-sonnet-4-5",
    )
    assert isinstance(result, ChatCompletionResponse)
    assert result.choices[0].finish_reason == "tool_calls"

    request = route.calls.last.request
    assert orjson.loads(request.content) == {
        "messages": [{"role": "user", "content": "What is Pinecone?"}],
        "model": "claude-sonnet-4-5",
        "stream": False,
    }
    claim.assert_request(request)
    claim.assert_api_version(request)
    claim.assert_roundtrip(ChatCompletionResponse, COMPLETION, optional_absent=[])


@api_op("assistant_data:context_assistant")
async def test_async_context_assistant(
    claim: Any, async_assistants: AsyncAssistants, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.post(f"{DATA_URL}/chat/{ASSISTANT_NAME}/context").mock(
        return_value=httpx.Response(200, json=CONTEXT)
    )

    result = await async_assistants.context(
        assistant_name=ASSISTANT_NAME, query="What is Pinecone?", top_k=20
    )
    assert isinstance(result, ContextResponse)
    assert result.snippets[0].reference.type == "pdf"
    assert result.usage.total_tokens == 480

    request = route.calls.last.request
    assert orjson.loads(request.content) == {"query": "What is Pinecone?", "top_k": 20}
    claim.assert_request(request)
    claim.assert_api_version(request)
    claim.assert_roundtrip(ContextResponse, CONTEXT, optional_absent=["id"])


async def test_async_chat_streaming_transcript(
    async_assistants: AsyncAssistants, respx_mock: respx.MockRouter
) -> None:
    """The documented message_start -> content_chunk -> citation -> message_end stream."""
    route = respx_mock.post(f"{DATA_URL}/chat/{ASSISTANT_NAME}").mock(
        return_value=httpx.Response(200, content=sse_body(CHAT_SSE_EVENTS))
    )

    stream = await async_assistants.chat(
        assistant_name=ASSISTANT_NAME,
        messages=[{"content": "What is Pinecone?"}],
        model="gpt-5",
        stream=True,
    )
    assert isinstance(stream, AsyncChatStream)
    chunks = [chunk async for chunk in stream]

    assert summarise(chunks) == EXPECTED_CHAT_STREAM

    request = route.calls.last.request
    assert request.url.path == expected_path(
        "assistant_data:chat_assistant", f"/chat/{ASSISTANT_NAME}"
    )
    assert request.headers["x-pinecone-api-version"] == "2026-07"
    assert orjson.loads(request.content)["stream"] is True


async def test_async_chat_stream_text_helper_yields_only_content(
    async_assistants: AsyncAssistants, respx_mock: respx.MockRouter
) -> None:
    """``AsyncChatStream.text()`` drops the start/citation/end frames.

    The transcript carries three non-content frames around two content deltas;
    a caller printing ``text()`` must see the message and nothing else.
    """
    respx_mock.post(f"{DATA_URL}/chat/{ASSISTANT_NAME}").mock(
        return_value=httpx.Response(200, content=sse_body(CHAT_SSE_EVENTS))
    )

    stream = await async_assistants.chat(
        assistant_name=ASSISTANT_NAME,
        messages=[{"content": "What is Pinecone?"}],
        stream=True,
    )
    assert isinstance(stream, AsyncChatStream)
    fragments = [fragment async for fragment in stream.text()]

    assert fragments == ["Pinecone is ", "a managed vector database."]


async def test_async_chat_completions_streaming_transcript(
    async_assistants: AsyncAssistants, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.post(f"{DATA_URL}/chat/{ASSISTANT_NAME}/chat/completions").mock(
        return_value=httpx.Response(200, content=sse_body(COMPLETION_SSE_EVENTS))
    )

    stream = await async_assistants.chat_completions(
        assistant_name=ASSISTANT_NAME,
        messages=[{"content": "What is Pinecone?"}],
        model="claude-sonnet-4-5",
        stream=True,
    )
    assert isinstance(stream, AsyncChatCompletionStream)
    chunks = [chunk async for chunk in stream]

    assert len(chunks) == len(COMPLETION_SSE_EVENTS)
    assert chunks[0].choices[0].delta.role == "assistant"
    joined = "".join(chunk.choices[0].delta.content or "" for chunk in chunks)
    assert joined == "Pinecone is a managed vector database."
    assert chunks[-1].choices[0].finish_reason == "tool_calls"
    assert chunks[-1].usage is not None
    assert chunks[-1].usage.total_tokens == 2641

    request = route.calls.last.request
    assert request.url.path == expected_path(
        "assistant_data:chat_completion_assistant",
        f"/chat/{ASSISTANT_NAME}/chat/completions",
    )
    assert request.headers["x-pinecone-api-version"] == "2026-07"
    assert orjson.loads(request.content)["stream"] is True


async def test_async_context_request_carries_the_api_version(
    async_assistants: AsyncAssistants, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.post(f"{DATA_URL}/chat/{ASSISTANT_NAME}/context").mock(
        return_value=httpx.Response(200, json=CONTEXT)
    )

    await async_assistants.context(
        assistant_name=ASSISTANT_NAME, messages=[{"content": "What is Pinecone?"}]
    )

    assert route.calls.last.request.headers["x-pinecone-api-version"] == "2026-07"


async def test_async_removed_model_400_surfaces_the_backend_message_verbatim(
    async_assistants: AsyncAssistants, respx_mock: respx.MockRouter
) -> None:
    """The async transport must not reword or truncate the backend's 400.

    Byte-identical to the sync assertion — the message text is imported from
    the sync module, so a divergence in either error path fails here.
    """
    respx_mock.post(f"{DATA_URL}/chat/{ASSISTANT_NAME}").mock(
        return_value=httpx.Response(
            400,
            json={
                "status": 400,
                "error": {"code": "INVALID_ARGUMENT", "message": REMOVED_MODEL_ERROR_MESSAGE},
            },
        )
    )

    with pytest.raises(ApiError) as excinfo:
        await async_assistants.chat(
            assistant_name=ASSISTANT_NAME,
            messages=[{"content": "What is Pinecone?"}],
            model="claude-3-5-sonnet",
        )

    assert excinfo.value.message == REMOVED_MODEL_ERROR_MESSAGE
    assert excinfo.value.error_code == "INVALID_ARGUMENT"
    assert REMOVED_MODEL_ERROR_MESSAGE in str(excinfo.value)
    assert "claude-sonnet-4-5" in str(excinfo.value)


async def test_async_chat_response_without_the_new_fields_decodes(
    async_assistants: AsyncAssistants, respx_mock: respx.MockRouter
) -> None:
    """A 2025-10-shaped body leaves the two 2026-07 additions at ``None``."""
    dropped = ("context_snippet_count", "content_filter_results")
    older = {key: value for key, value in CHAT.items() if key not in dropped}
    respx_mock.post(f"{DATA_URL}/chat/{ASSISTANT_NAME}").mock(
        return_value=httpx.Response(200, json=older)
    )

    result = await async_assistants.chat(
        assistant_name=ASSISTANT_NAME, messages=[{"content": "What is Pinecone?"}]
    )
    assert isinstance(result, ChatResponse)
    assert result.context_snippet_count is None
    assert result.content_filter_results is None


@pytest.fixture(scope="module")
def property_loop() -> Iterator[asyncio.AbstractEventLoop]:
    """One event loop for every example of the property test below (#345).

    ``asyncio.run`` builds and tears down a loop per example, which is setup
    rather than anything the property asserts.
    """
    loop = asyncio.new_event_loop()
    yield loop
    # asyncio.run() does this for the loop it owns; a hand-rolled loop has to
    # do it too, or the last example's response iterators are collected after
    # the loop is gone and each one reports an un-awaited aclose().
    loop.run_until_complete(loop.shutdown_asyncgens())
    loop.close()


@pytest.fixture(scope="module")
def property_assistants(
    property_loop: asyncio.AbstractEventLoop,
    hermetic_pinecone_env_module: None,
) -> Iterator[AsyncAssistants]:
    """One client reused by every example of the property test below.

    The client builds its control-plane and data-plane HTTP clients on first
    use, and each one parses the whole CA bundle into a fresh
    ``ssl.SSLContext`` — the work that dominated this test and the part that
    amplifies on slower CI runners (#345). None of it is exercised: respx
    intercepts at the transport, so no example opens a socket. Every example
    still streams its own transcript through its own routes; only the client
    build leaves the loop, so the data-plane host is resolved once instead of
    once per example.

    It is built on ``property_loop`` so the lazily created ``AsyncClient``
    belongs to the loop the examples run on.
    """
    client = AsyncAssistants(config=PineconeConfig(api_key="conformance-key", host=BASE_URL))
    yield client
    property_loop.run_until_complete(client.close())


@given(sizes=st.lists(st.integers(min_value=1, max_value=64), min_size=1, max_size=24))
def test_async_sse_chunk_boundaries_do_not_change_the_parsed_stream(
    sizes: list[int],
    property_assistants: AsyncAssistants,
    property_loop: asyncio.AbstractEventLoop,
) -> None:
    """Chunk boundaries are a transport artifact on the async path too.

    ``aiter_lines`` has to reassemble by line rather than by read, exactly as
    ``iter_lines`` does. Splitting the same transcript at arbitrary offsets
    must yield the chunk sequence the sync property test pins.
    """
    body = sse_body(CHAT_SSE_EVENTS)
    pieces: list[bytes] = []
    offset = 0
    index = 0
    while offset < len(body):
        size = sizes[index % len(sizes)]
        pieces.append(body[offset : offset + size])
        offset += size
        index += 1

    async def check() -> None:
        with respx.mock:
            respx.get(f"{CONTROL_URL}/assistants/{ASSISTANT_NAME}").mock(
                return_value=httpx.Response(200, json=ASSISTANT)
            )
            respx.post(f"{DATA_URL}/chat/{ASSISTANT_NAME}").mock(
                return_value=httpx.Response(200, content=_async_body(pieces))
            )
            stream = await property_assistants.chat(
                assistant_name=ASSISTANT_NAME,
                messages=[{"content": "What is Pinecone?"}],
                stream=True,
            )
            assert isinstance(stream, AsyncChatStream)
            chunks = [chunk async for chunk in stream]

        assert summarise(chunks) == EXPECTED_CHAT_STREAM

    property_loop.run_until_complete(check())
