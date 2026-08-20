"""Unit tests for the 2026-07 chat response additions.

Covers ``ChatResponse.content_filter_results`` / ``ChatResponse.context_snippet_count``
and ``StreamMessageStart.context_snippet_count``: wire decoding with and without
the fields, ``to_dict`` round-tripping, and repr compactness.
"""

from __future__ import annotations

from typing import Any

import msgspec

from pinecone._internal.adapters.assistants_adapter import AssistantsAdapter
from pinecone.models.assistant.chat import ChatMessage, ChatResponse, ChatUsage
from pinecone.models.assistant.streaming import ChatStreamChunk, StreamMessageStart

_OPENAI_FILTER: dict[str, Any] = {
    "spec": "openai",
    "results": {
        "hate": {"filtered": False, "severity": "safe"},
        "violence": {"filtered": False, "severity": "safe"},
    },
}

_GEMINI_FILTER: dict[str, Any] = {
    "spec": "gemini",
    "results": [
        {"category": "HARM_CATEGORY_HARASSMENT", "probability": "NEGLIGIBLE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "probability": "NEGLIGIBLE"},
    ],
}

_HUGE_FILTER: dict[str, Any] = {"spec": "openai", "results": {"blob": "x" * 100_000}}


def _chat_payload(**extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": "r-1",
        "model": "gpt-4o-2024-11-20",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "message": {"role": "assistant", "content": "Hi"},
        "finish_reason": "stop",
        "citations": [],
    }
    payload.update(extra)
    return payload


def _decode_chat(payload: dict[str, Any]) -> ChatResponse:
    return AssistantsAdapter.to_chat_response(msgspec.json.encode(payload))


def _message_start_payload(**extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": "message_start",
        "id": "c-1",
        "model": "gpt-4o-2024-11-20",
        "role": "assistant",
    }
    payload.update(extra)
    return payload


def _decode_chunk(payload: dict[str, Any]) -> ChatStreamChunk:
    return msgspec.json.decode(msgspec.json.encode(payload), type=ChatStreamChunk)


class TestChatResponseNewFields:
    def test_decodes_both_new_fields(self) -> None:
        resp = _decode_chat(
            _chat_payload(context_snippet_count=16, content_filter_results=_OPENAI_FILTER)
        )
        assert resp.context_snippet_count == 16
        assert resp.content_filter_results == _OPENAI_FILTER

    def test_fields_default_none_when_absent(self) -> None:
        resp = _decode_chat(_chat_payload())
        assert resp.context_snippet_count is None
        assert resp.content_filter_results is None

    def test_zero_snippet_count_is_distinct_from_none(self) -> None:
        resp = _decode_chat(_chat_payload(context_snippet_count=0))
        assert resp.context_snippet_count == 0
        assert resp.context_snippet_count is not None

    def test_gemini_filter_shape_preserved(self) -> None:
        resp = _decode_chat(_chat_payload(content_filter_results=_GEMINI_FILTER))
        assert resp.content_filter_results == _GEMINI_FILTER

    def test_unmodelled_filter_shape_preserved(self) -> None:
        """A provider payload the SDK has never seen survives decoding untouched."""
        payload: dict[str, Any] = {"spec": "future-llm", "results": "blocked"}
        resp = _decode_chat(_chat_payload(content_filter_results=payload))
        assert resp.content_filter_results == payload

    def test_defaults_keep_direct_construction_working(self) -> None:
        resp = ChatResponse(
            id="r-1",
            model="m",
            usage=ChatUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            message=ChatMessage(role="assistant", content="Hi"),
            finish_reason="stop",
            citations=[],
        )
        assert resp.context_snippet_count is None
        assert resp.content_filter_results is None


class TestChatResponseToDict:
    def test_to_dict_includes_new_keys_when_present(self) -> None:
        resp = _decode_chat(
            _chat_payload(context_snippet_count=3, content_filter_results=_OPENAI_FILTER)
        )
        result = resp.to_dict()
        assert result["context_snippet_count"] == 3
        assert result["content_filter_results"] == _OPENAI_FILTER

    def test_to_dict_includes_new_keys_as_none_when_absent(self) -> None:
        result = _decode_chat(_chat_payload()).to_dict()
        assert result["context_snippet_count"] is None
        assert result["content_filter_results"] is None

    def test_round_trip_with_new_fields(self) -> None:
        payload = _chat_payload(context_snippet_count=7, content_filter_results=_GEMINI_FILTER)
        first = _decode_chat(payload).to_dict()
        second = _decode_chat(first).to_dict()
        assert first == second
        assert second["context_snippet_count"] == 7
        assert second["content_filter_results"] == _GEMINI_FILTER

    def test_round_trip_without_new_fields(self) -> None:
        first = _decode_chat(_chat_payload()).to_dict()
        second = _decode_chat(first).to_dict()
        assert first == second
        assert second["context_snippet_count"] is None

    def test_filter_results_nested_dict_is_plain(self) -> None:
        resp = _decode_chat(_chat_payload(content_filter_results=_OPENAI_FILTER))
        results = resp.to_dict()["content_filter_results"]
        assert isinstance(results, dict)
        assert isinstance(results["results"]["hate"], dict)

    def test_dict_like_access(self) -> None:
        resp = _decode_chat(_chat_payload(context_snippet_count=2))
        assert resp["context_snippet_count"] == 2
        assert "content_filter_results" in resp
        assert resp.get("context_snippet_count") == 2


class TestChatResponseRepr:
    def test_repr_shows_snippet_count_when_present(self) -> None:
        resp = _decode_chat(_chat_payload(context_snippet_count=16))
        assert "context_snippet_count=16" in repr(resp)

    def test_repr_omits_snippet_count_when_absent(self) -> None:
        assert "context_snippet_count" not in repr(_decode_chat(_chat_payload()))

    def test_repr_shows_zero_snippet_count(self) -> None:
        resp = _decode_chat(_chat_payload(context_snippet_count=0))
        assert "context_snippet_count=0" in repr(resp)

    def test_repr_stays_bounded_with_huge_filter_results(self) -> None:
        resp = _decode_chat(_chat_payload(content_filter_results=_HUGE_FILTER))
        assert len(repr(resp)) < 1000

    def test_repr_pretty_bounded_with_huge_filter_results(self) -> None:
        from IPython.lib.pretty import pretty

        resp = _decode_chat(_chat_payload(content_filter_results=_HUGE_FILTER))
        assert len(pretty(resp)) < 2000

    def test_repr_html_bounded_with_huge_filter_results(self) -> None:
        resp = _decode_chat(_chat_payload(content_filter_results=_HUGE_FILTER))
        assert len(resp._repr_html_()) < 10_000

    def test_repr_html_shows_snippet_count(self) -> None:
        resp = _decode_chat(_chat_payload(context_snippet_count=4))
        assert "Context snippets" in resp._repr_html_()

    def test_repr_html_omits_snippet_count_when_absent(self) -> None:
        assert "Context snippets" not in _decode_chat(_chat_payload())._repr_html_()


class TestStreamMessageStartSnippetCount:
    def test_decodes_snippet_count(self) -> None:
        chunk = _decode_chunk(_message_start_payload(context_snippet_count=16))
        assert isinstance(chunk, StreamMessageStart)
        assert chunk.context_snippet_count == 16

    def test_snippet_count_defaults_none(self) -> None:
        chunk = _decode_chunk(_message_start_payload())
        assert isinstance(chunk, StreamMessageStart)
        assert chunk.context_snippet_count is None

    def test_zero_snippet_count_decodes(self) -> None:
        chunk = _decode_chunk(_message_start_payload(context_snippet_count=0))
        assert isinstance(chunk, StreamMessageStart)
        assert chunk.context_snippet_count == 0

    def test_unknown_wire_keys_still_ignored(self) -> None:
        """content_filter_results is modelled as of #222; genuinely unknown keys stay ignored."""
        chunk = _decode_chunk(
            _message_start_payload(
                context_snippet_count=1,
                content_filter_results=_OPENAI_FILTER,
                some_future_key={"nested": True},
            )
        )
        assert isinstance(chunk, StreamMessageStart)
        assert chunk.context_snippet_count == 1
        assert chunk.content_filter_results == _OPENAI_FILTER

    def test_to_dict_round_trip_with_snippet_count(self) -> None:
        chunk = _decode_chunk(_message_start_payload(context_snippet_count=5))
        result = chunk.to_dict()
        assert result["context_snippet_count"] == 5
        assert result["role"] == "assistant"

    def test_to_dict_round_trip_without_snippet_count(self) -> None:
        result = _decode_chunk(_message_start_payload()).to_dict()
        assert result["context_snippet_count"] is None

    def test_repr_shows_snippet_count_when_present(self) -> None:
        chunk = _decode_chunk(_message_start_payload(context_snippet_count=16))
        assert "context_snippet_count=16" in repr(chunk)

    def test_repr_omits_snippet_count_when_absent(self) -> None:
        assert "context_snippet_count" not in repr(_decode_chunk(_message_start_payload()))

    def test_repr_pretty_shows_snippet_count(self) -> None:
        from IPython.lib.pretty import pretty

        chunk = _decode_chunk(_message_start_payload(context_snippet_count=9))
        assert "context_snippet_count=9" in pretty(chunk)

    def test_repr_html_shows_snippet_count(self) -> None:
        chunk = _decode_chunk(_message_start_payload(context_snippet_count=9))
        assert "Context snippets" in chunk._repr_html_()

    def test_repr_html_omits_snippet_count_when_absent(self) -> None:
        assert "Context snippets" not in _decode_chunk(_message_start_payload())._repr_html_()
