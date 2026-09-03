"""2026-07 conformance for the three assistant_data chat/context operations.

``assistant_data_2026-07.oas.yaml`` reworks the chat surface: ``ChatRequest``
drops ``claude-3-5-sonnet``/``claude-3-7-sonnet`` and gains ``gpt-5`` and
``claude-sonnet-4-5``; ``SearchCompletions`` gains ``claude-sonnet-4-5`` but not
``gpt-5``; ``finish_reason`` replaces ``function_call`` with ``tool_calls``
everywhere it appears; ``ChatModel`` gains ``context_snippet_count`` and
``content_filter_results``; and the ``text/event-stream`` response with its
``StreamChatChunkModel`` discriminated union is formally documented. The
backend routes 2026-07 chat/context to the v202604 handlers
(``svc-knowledge-engine/src/search/service/routes/mod.rs:21``,
pinecone-db@f6fd0a4019).

These tests pin method and path — including the ``/assistant`` prefix the SDK
really sends, which the spec's ``servers`` URL omits (#173, registered as a
``base_path_overrides`` entry) — the ``X-Pinecone-Api-Version`` header, the
response schemas, and the SSE transcript both streaming wrappers must
reassemble.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import httpx
import orjson
import pytest
import respx
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from pinecone._internal.config import PineconeConfig
from pinecone.client.assistants import Assistants
from pinecone.errors.exceptions import ApiError
from pinecone.models.assistant.chat import ChatCompletionResponse, ChatResponse
from pinecone.models.assistant.context import ContextResponse
from pinecone.models.assistant.streaming import (
    ChatCompletionStream,
    ChatStream,
    ChatStreamChunk,
    StreamCitationChunk,
    StreamContentChunk,
    StreamMessageStart,
)
from tests.unit.conformance import (
    ClaimRecorder,
    ConformanceError,
    api_op,
    manifest_operations,
)
from tests.unit.conformance._registry import _expected_base_path

BASE_URL = "https://api.test.pinecone.io"
CONTROL_URL = f"{BASE_URL}/assistant"
DATA_HOST = "https://prod-1-data.ke.pinecone.io"
DATA_URL = f"{DATA_HOST}/assistant"

ASSISTANT_NAME = "chat-conformance"
CHAT_ID = "00000000000000002fe0c02e20be1c6a"
COMPLETION_ID = "00000000000000002fe0c02e20be1c6b"

ASSISTANT: dict[str, Any] = {
    "name": ASSISTANT_NAME,
    "status": "Ready",
    "host": DATA_HOST,
    "region": "us",
}

FILE: dict[str, Any] = {
    "id": "ae79e447-b89e-4994-994b-3232ca52a654",
    "name": "pinecone-guide.pdf",
    "size": 25000,
    "status": "Available",
    "multimodal": False,
    "metadata": None,
    "signed_url": None,
    "created_on": "2026-07-01T00:00:00Z",
    "updated_on": "2026-07-01T00:01:00Z",
}

CONTENT_FILTER_RESULTS: dict[str, Any] = {
    "spec": "openai",
    "results": {"hate": {"filtered": False, "severity": "safe"}},
}

CHAT: dict[str, Any] = {
    "id": CHAT_ID,
    "model": "gpt-5",
    "finish_reason": "tool_calls",
    "message": {"role": "assistant", "content": "Pinecone is a managed vector database."},
    "citations": [
        {
            "position": 37,
            "references": [
                {
                    "file": FILE,
                    "pages": [1, 2],
                    "highlight": {"type": "text", "content": "a managed vector database"},
                }
            ],
        }
    ],
    "usage": {"prompt_tokens": 2506, "completion_tokens": 135, "total_tokens": 2641},
    "context_snippet_count": 16,
    "content_filter_results": CONTENT_FILTER_RESULTS,
}

COMPLETION: dict[str, Any] = {
    "id": COMPLETION_ID,
    "model": "claude-sonnet-4-5",
    "choices": [
        {
            "index": 0,
            "finish_reason": "tool_calls",
            "message": {"role": "assistant", "content": "Pinecone is a managed vector database."},
        }
    ],
    "usage": {"prompt_tokens": 2506, "completion_tokens": 135, "total_tokens": 2641},
}

CONTEXT: dict[str, Any] = {
    "id": "00000000000000002fe0c02e20be1c6c",
    "snippets": [
        {
            "type": "text",
            "content": "Pinecone is a managed vector database for AI applications.",
            "score": 0.913,
            "reference": {"type": "pdf", "file": FILE, "pages": [1, 2]},
        }
    ],
    "usage": {"prompt_tokens": 480, "completion_tokens": 0, "total_tokens": 480},
}

CHAT_SSE_EVENTS: tuple[dict[str, Any], ...] = (
    {
        "type": "message_start",
        "id": CHAT_ID,
        "model": "gpt-5",
        "role": "assistant",
        "context_snippet_count": 16,
    },
    {
        "type": "content_chunk",
        "id": CHAT_ID,
        "model": "gpt-5",
        "delta": {"content": "Pinecone is "},
    },
    {
        "type": "content_chunk",
        "id": CHAT_ID,
        "model": "gpt-5",
        "delta": {"content": "a managed vector database."},
    },
    {
        "type": "citation",
        "id": CHAT_ID,
        "model": "gpt-5",
        "citation": {"position": 37, "references": [{"file": FILE, "pages": [1, 2]}]},
    },
    {
        "type": "message_end",
        "id": CHAT_ID,
        "model": "gpt-5",
        "finish_reason": "tool_calls",
        "usage": {"prompt_tokens": 2506, "completion_tokens": 135, "total_tokens": 2641},
    },
)

EXPECTED_CHAT_STREAM: list[tuple[Any, ...]] = [
    ("message_start", "gpt-5", "assistant", 16),
    ("content_chunk", CHAT_ID, "gpt-5", "Pinecone is "),
    ("content_chunk", CHAT_ID, "gpt-5", "a managed vector database."),
    ("citation", CHAT_ID, "gpt-5", 37, 1),
    ("message_end", CHAT_ID, "gpt-5", "tool_calls", 2641),
]

COMPLETION_SSE_EVENTS: tuple[dict[str, Any], ...] = (
    {
        "id": COMPLETION_ID,
        "model": "claude-sonnet-4-5",
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    },
    {
        "id": COMPLETION_ID,
        "model": "claude-sonnet-4-5",
        "choices": [{"index": 0, "delta": {"content": "Pinecone is "}, "finish_reason": None}],
    },
    {
        "id": COMPLETION_ID,
        "model": "claude-sonnet-4-5",
        "choices": [
            {"index": 0, "delta": {"content": "a managed vector database."}, "finish_reason": None}
        ],
    },
    {
        "id": COMPLETION_ID,
        "model": "claude-sonnet-4-5",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
        "usage": {"prompt_tokens": 2506, "completion_tokens": 135, "total_tokens": 2641},
    },
)

REMOVED_MODEL_ERROR_MESSAGE = (
    "Invalid model `claude-3-5-sonnet`. Expected one of: gpt-4o, gpt-4.1, o4-mini, "
    "gpt-5, claude-sonnet-4-5, gemini-2.5-pro."
)


def sse_body(events: Sequence[dict[str, Any]]) -> bytes:
    """The wire bytes a spec-conformant ``text/event-stream`` response carries."""
    events_bytes = b"".join(b"data: " + orjson.dumps(event) + b"\n\n" for event in events)
    return events_bytes + b"data: [DONE]\n\n"


def summarise(chunks: Sequence[ChatStreamChunk]) -> list[tuple[Any, ...]]:
    """The identifying fields of each parsed chunk, in arrival order."""
    summary: list[tuple[Any, ...]] = []
    for chunk in chunks:
        if isinstance(chunk, StreamMessageStart):
            summary.append(("message_start", chunk.model, chunk.role, chunk.context_snippet_count))
        elif isinstance(chunk, StreamContentChunk):
            summary.append(("content_chunk", chunk.id, chunk.model, chunk.delta.content))
        elif isinstance(chunk, StreamCitationChunk):
            summary.append(
                (
                    "citation",
                    chunk.id,
                    chunk.model,
                    chunk.citation.position,
                    len(chunk.citation.references),
                )
            )
        else:
            summary.append(
                (
                    "message_end",
                    chunk.id,
                    chunk.model,
                    chunk.finish_reason,
                    chunk.usage.total_tokens if chunk.usage is not None else None,
                )
            )
    return summary


def expected_path(op_id: str, suffix: str) -> str:
    entry = manifest_operations()[op_id]
    return _expected_base_path(op_id, entry) + suffix


@pytest.fixture
def assistants(respx_mock: respx.MockRouter) -> Iterator[Assistants]:
    respx_mock.get(f"{CONTROL_URL}/assistants/{ASSISTANT_NAME}").mock(
        return_value=httpx.Response(200, json=ASSISTANT)
    )
    client = Assistants(config=PineconeConfig(api_key="conformance-key", host=BASE_URL))
    yield client
    client.close()


@api_op("assistant_data:chat_assistant")
def test_chat_assistant(claim: Any, assistants: Assistants, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.post(f"{DATA_URL}/chat/{ASSISTANT_NAME}").mock(
        return_value=httpx.Response(200, json=CHAT)
    )

    result = assistants.chat(
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
def test_chat_completion_assistant(
    claim: Any, assistants: Assistants, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.post(f"{DATA_URL}/chat/{ASSISTANT_NAME}/chat/completions").mock(
        return_value=httpx.Response(200, json=COMPLETION)
    )

    result = assistants.chat_completions(
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
def test_context_assistant(
    claim: Any, assistants: Assistants, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.post(f"{DATA_URL}/chat/{ASSISTANT_NAME}/context").mock(
        return_value=httpx.Response(200, json=CONTEXT)
    )

    result = assistants.context(assistant_name=ASSISTANT_NAME, query="What is Pinecone?", top_k=20)
    assert isinstance(result, ContextResponse)
    assert result.snippets[0].reference.type == "pdf"
    assert result.usage.total_tokens == 480

    request = route.calls.last.request
    assert orjson.loads(request.content) == {"query": "What is Pinecone?", "top_k": 20}
    claim.assert_request(request)
    claim.assert_api_version(request)
    claim.assert_roundtrip(ContextResponse, CONTEXT, optional_absent=["id"])


def test_chat_streaming_transcript(assistants: Assistants, respx_mock: respx.MockRouter) -> None:
    """The documented message_start -> content_chunk -> citation -> message_end stream."""
    route = respx_mock.post(f"{DATA_URL}/chat/{ASSISTANT_NAME}").mock(
        return_value=httpx.Response(200, content=sse_body(CHAT_SSE_EVENTS))
    )

    stream = assistants.chat(
        assistant_name=ASSISTANT_NAME,
        messages=[{"content": "What is Pinecone?"}],
        model="gpt-5",
        stream=True,
    )
    assert isinstance(stream, ChatStream)
    chunks = list(stream)

    assert summarise(chunks) == EXPECTED_CHAT_STREAM

    request = route.calls.last.request
    assert request.url.path == expected_path(
        "assistant_data:chat_assistant", f"/chat/{ASSISTANT_NAME}"
    )
    assert request.headers["x-pinecone-api-version"] == "2026-07"
    assert orjson.loads(request.content)["stream"] is True


def test_chat_completions_streaming_transcript(
    assistants: Assistants, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.post(f"{DATA_URL}/chat/{ASSISTANT_NAME}/chat/completions").mock(
        return_value=httpx.Response(200, content=sse_body(COMPLETION_SSE_EVENTS))
    )

    stream = assistants.chat_completions(
        assistant_name=ASSISTANT_NAME,
        messages=[{"content": "What is Pinecone?"}],
        model="claude-sonnet-4-5",
        stream=True,
    )
    assert isinstance(stream, ChatCompletionStream)
    chunks = list(stream)

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


def test_context_request_carries_the_api_version(
    assistants: Assistants, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.post(f"{DATA_URL}/chat/{ASSISTANT_NAME}/context").mock(
        return_value=httpx.Response(200, json=CONTEXT)
    )

    assistants.context(assistant_name=ASSISTANT_NAME, messages=[{"content": "What is Pinecone?"}])

    assert route.calls.last.request.headers["x-pinecone-api-version"] == "2026-07"


def test_removed_model_400_surfaces_the_backend_message_verbatim(
    assistants: Assistants, respx_mock: respx.MockRouter
) -> None:
    """A rejected model name must reach the caller with the allowed values intact.

    The backend lists what it accepts in the 400 body; truncating or rewording
    it is the difference between a caller self-correcting and a caller guessing.
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
        assistants.chat(
            assistant_name=ASSISTANT_NAME,
            messages=[{"content": "What is Pinecone?"}],
            model="claude-3-5-sonnet",
        )

    assert excinfo.value.message == REMOVED_MODEL_ERROR_MESSAGE
    assert excinfo.value.error_code == "INVALID_ARGUMENT"
    assert REMOVED_MODEL_ERROR_MESSAGE in str(excinfo.value)
    assert "claude-sonnet-4-5" in str(excinfo.value)


def test_chat_response_without_the_new_fields_decodes(
    assistants: Assistants, respx_mock: respx.MockRouter
) -> None:
    """A 2025-10-shaped body leaves the two 2026-07 additions at ``None``."""
    dropped = ("context_snippet_count", "content_filter_results")
    older = {key: value for key, value in CHAT.items() if key not in dropped}
    respx_mock.post(f"{DATA_URL}/chat/{ASSISTANT_NAME}").mock(
        return_value=httpx.Response(200, json=older)
    )

    result = assistants.chat(
        assistant_name=ASSISTANT_NAME, messages=[{"content": "What is Pinecone?"}]
    )
    assert isinstance(result, ChatResponse)
    assert result.context_snippet_count is None
    assert result.content_filter_results is None


@settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(sizes=st.lists(st.integers(min_value=1, max_value=64), min_size=1, max_size=24))
def test_sse_chunk_boundaries_do_not_change_the_parsed_stream(sizes: list[int]) -> None:
    """Chunk boundaries are a transport artifact, not part of the message.

    A server may flush one SSE event across any number of reads, so the parser
    has to reassemble by line rather than by read. Splitting the same transcript
    at arbitrary offsets must yield an identical chunk sequence.
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

    with respx.mock:
        respx.get(f"{CONTROL_URL}/assistants/{ASSISTANT_NAME}").mock(
            return_value=httpx.Response(200, json=ASSISTANT)
        )
        respx.post(f"{DATA_URL}/chat/{ASSISTANT_NAME}").mock(
            return_value=httpx.Response(200, content=iter(pieces))
        )
        client = Assistants(config=PineconeConfig(api_key="conformance-key", host=BASE_URL))
        try:
            stream = client.chat(
                assistant_name=ASSISTANT_NAME,
                messages=[{"content": "What is Pinecone?"}],
                stream=True,
            )
            assert isinstance(stream, ChatStream)
            chunks = list(stream)
        finally:
            client.close()

    assert summarise(chunks) == EXPECTED_CHAT_STREAM


def test_manifest_records_the_assistant_data_base_path_override() -> None:
    entry = manifest_operations()["assistant_data:chat_assistant"]
    assert entry["base_path"] == "/assistant"
    assert entry["base_path_divergence"]["issue"] == 173
    assert entry["base_path_divergence"]["spec_base_path"] == ""


def test_base_path_override_without_an_issue_number_is_refused() -> None:
    entry = dict(manifest_operations()["assistant_data:chat_assistant"])
    entry["base_path_divergence"] = {"reason": "because"}
    with pytest.raises(ConformanceError, match="does not reference a question issue"):
        _expected_base_path("assistant_data:chat_assistant", entry)


def test_base_path_override_without_a_reason_is_refused() -> None:
    entry = dict(manifest_operations()["assistant_data:chat_assistant"])
    entry["base_path_divergence"] = {"issue": 173, "reason": "  "}
    with pytest.raises(ConformanceError, match="no reason"):
        _expected_base_path("assistant_data:chat_assistant", entry)


def test_assert_request_rejects_the_unprefixed_spec_path() -> None:
    recorder = ClaimRecorder(["assistant_data:chat_assistant"])
    request = httpx.Request("POST", f"{DATA_HOST}/chat/{ASSISTANT_NAME}")
    with pytest.raises(ConformanceError, match="does not match spec template"):
        recorder.assert_request(request)
