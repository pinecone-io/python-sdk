"""Unit tests for the 2026-07 streaming chunk additions.

Covers ``content_filter_results`` on ``StreamMessageStart`` /
``StreamContentChunk`` / ``StreamMessageEnd`` and ``StreamMessageStart.id``:
wire decoding with and without each field, ``to_dict`` round-tripping, and
repr compactness against unbounded provider payloads.
"""

from __future__ import annotations

from typing import Any

import msgspec
import pytest

from pinecone.models.assistant.streaming import (
    ChatStreamChunk,
    StreamContentChunk,
    StreamMessageEnd,
    StreamMessageStart,
)

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

_UNMODELLED_FILTER: dict[str, Any] = {
    "spec": "some-future-provider",
    "results": {"nested": {"deeply": ["a", 1, True, None]}},
}

_HUGE_FILTER: dict[str, Any] = {"spec": "openai", "results": {"blob": "x" * 100_000}}

CHUNK_ID = "00000000000000002fe0c02e20be1c6a"


def _message_start_payload(**extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": "message_start",
        "id": CHUNK_ID,
        "model": "gpt-4o-2024-11-20",
        "role": "assistant",
    }
    payload.update(extra)
    return payload


def _content_chunk_payload(**extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": "content_chunk",
        "id": CHUNK_ID,
        "model": "gpt-4o-2024-11-20",
        "delta": {"content": "Pinecone is "},
    }
    payload.update(extra)
    return payload


def _message_end_payload(**extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": "message_end",
        "id": CHUNK_ID,
        "model": "gpt-4o-2024-11-20",
        "finish_reason": "stop",
    }
    payload.update(extra)
    return payload


def _decode_chunk(payload: dict[str, Any]) -> ChatStreamChunk:
    return msgspec.json.decode(msgspec.json.encode(payload), type=ChatStreamChunk)


_BUILDERS = {
    "message_start": _message_start_payload,
    "content_chunk": _content_chunk_payload,
    "message_end": _message_end_payload,
}

_TYPES: dict[str, type] = {
    "message_start": StreamMessageStart,
    "content_chunk": StreamContentChunk,
    "message_end": StreamMessageEnd,
}

_CHUNK_KINDS = tuple(_BUILDERS)


@pytest.mark.parametrize("kind", _CHUNK_KINDS)
class TestContentFilterResults:
    def test_decodes_openai_filter(self, kind: str) -> None:
        chunk = _decode_chunk(_BUILDERS[kind](content_filter_results=_OPENAI_FILTER))
        assert isinstance(chunk, _TYPES[kind])
        assert chunk.content_filter_results == _OPENAI_FILTER

    def test_defaults_none_when_absent(self, kind: str) -> None:
        chunk = _decode_chunk(_BUILDERS[kind]())
        assert isinstance(chunk, _TYPES[kind])
        assert chunk.content_filter_results is None

    def test_gemini_filter_shape_preserved(self, kind: str) -> None:
        chunk = _decode_chunk(_BUILDERS[kind](content_filter_results=_GEMINI_FILTER))
        assert chunk.content_filter_results == _GEMINI_FILTER

    def test_unmodelled_filter_shape_preserved(self, kind: str) -> None:
        chunk = _decode_chunk(_BUILDERS[kind](content_filter_results=_UNMODELLED_FILTER))
        assert chunk.content_filter_results == _UNMODELLED_FILTER

    def test_empty_filter_dict_is_distinct_from_none(self, kind: str) -> None:
        chunk = _decode_chunk(_BUILDERS[kind](content_filter_results={}))
        assert chunk.content_filter_results == {}
        assert chunk.content_filter_results is not None

    def test_to_dict_includes_filter_when_present(self, kind: str) -> None:
        result = _decode_chunk(_BUILDERS[kind](content_filter_results=_OPENAI_FILTER)).to_dict()
        assert result["content_filter_results"] == _OPENAI_FILTER

    def test_to_dict_includes_filter_as_none_when_absent(self, kind: str) -> None:
        result = _decode_chunk(_BUILDERS[kind]()).to_dict()
        assert result["content_filter_results"] is None

    def test_round_trip_with_filter(self, kind: str) -> None:
        original = _decode_chunk(_BUILDERS[kind](content_filter_results=_GEMINI_FILTER))
        reencoded = _decode_chunk({**original.to_dict(), "type": kind})
        assert reencoded.content_filter_results == _GEMINI_FILTER
        assert reencoded == original

    def test_round_trip_without_filter(self, kind: str) -> None:
        original = _decode_chunk(_BUILDERS[kind]())
        reencoded = _decode_chunk({**original.to_dict(), "type": kind})
        assert reencoded.content_filter_results is None
        assert reencoded == original

    def test_filter_results_nested_value_is_plain(self, kind: str) -> None:
        chunk = _decode_chunk(_BUILDERS[kind](content_filter_results=_OPENAI_FILTER))
        assert chunk.content_filter_results is not None
        assert isinstance(chunk.content_filter_results["results"], dict)

    def test_repr_omits_filter_results(self, kind: str) -> None:
        chunk = _decode_chunk(_BUILDERS[kind](content_filter_results=_HUGE_FILTER))
        assert "content_filter_results" not in repr(chunk)
        assert len(repr(chunk)) < 1000

    def test_repr_pretty_bounded_with_huge_filter(self, kind: str) -> None:
        from IPython.lib.pretty import pretty

        chunk = _decode_chunk(_BUILDERS[kind](content_filter_results=_HUGE_FILTER))
        rendered = pretty(chunk)
        assert "content_filter_results" in rendered
        assert len(rendered) < 2000

    def test_repr_html_bounded_with_huge_filter(self, kind: str) -> None:
        chunk = _decode_chunk(_BUILDERS[kind](content_filter_results=_HUGE_FILTER))
        html = chunk._repr_html_()
        assert "Content filter results" in html
        assert len(html) < 4000

    def test_repr_html_omits_filter_when_absent(self, kind: str) -> None:
        html = _decode_chunk(_BUILDERS[kind]())._repr_html_()
        assert "Content filter results" not in html


class TestStreamMessageStartId:
    def test_decodes_id(self) -> None:
        chunk = _decode_chunk(_message_start_payload())
        assert isinstance(chunk, StreamMessageStart)
        assert chunk.id == CHUNK_ID

    def test_id_defaults_none_when_absent(self) -> None:
        payload = _message_start_payload()
        del payload["id"]
        chunk = _decode_chunk(payload)
        assert isinstance(chunk, StreamMessageStart)
        assert chunk.id is None

    def test_id_matches_the_rest_of_the_stream(self) -> None:
        start = _decode_chunk(_message_start_payload())
        end = _decode_chunk(_message_end_payload())
        assert start.id == end.id

    def test_to_dict_includes_id_when_present(self) -> None:
        assert _decode_chunk(_message_start_payload()).to_dict()["id"] == CHUNK_ID

    def test_to_dict_includes_id_as_none_when_absent(self) -> None:
        payload = _message_start_payload()
        del payload["id"]
        assert _decode_chunk(payload).to_dict()["id"] is None

    def test_round_trip_with_id(self) -> None:
        original = _decode_chunk(_message_start_payload())
        reencoded = _decode_chunk({**original.to_dict(), "type": "message_start"})
        assert reencoded == original

    def test_round_trip_without_id(self) -> None:
        payload = _message_start_payload()
        del payload["id"]
        original = _decode_chunk(payload)
        reencoded = _decode_chunk({**original.to_dict(), "type": "message_start"})
        assert reencoded == original
        assert isinstance(reencoded, StreamMessageStart)
        assert reencoded.id is None

    def test_repr_shows_id_when_present(self) -> None:
        assert f"id={CHUNK_ID!r}" in repr(_decode_chunk(_message_start_payload()))

    def test_repr_omits_id_when_absent(self) -> None:
        payload = _message_start_payload()
        del payload["id"]
        assert "id=" not in repr(_decode_chunk(payload))

    def test_repr_pretty_shows_id(self) -> None:
        from IPython.lib.pretty import pretty

        assert CHUNK_ID in pretty(_decode_chunk(_message_start_payload()))

    def test_repr_pretty_omits_id_when_absent(self) -> None:
        from IPython.lib.pretty import pretty

        payload = _message_start_payload()
        del payload["id"]
        assert "id=" not in pretty(_decode_chunk(payload))

    def test_repr_html_shows_id(self) -> None:
        assert CHUNK_ID in _decode_chunk(_message_start_payload())._repr_html_()

    def test_repr_html_omits_id_when_absent(self) -> None:
        payload = _message_start_payload()
        del payload["id"]
        assert "Id:" not in _decode_chunk(payload)._repr_html_()


class TestFullStreamWithNewFields:
    def test_documented_transcript_carries_every_new_field(self) -> None:
        events = (
            _message_start_payload(context_snippet_count=16, content_filter_results=_OPENAI_FILTER),
            _content_chunk_payload(content_filter_results=_GEMINI_FILTER),
            _message_end_payload(
                usage={"prompt_tokens": 2506, "completion_tokens": 135, "total_tokens": 2641},
                content_filter_results=_UNMODELLED_FILTER,
            ),
        )
        chunks = [_decode_chunk(event) for event in events]
        assert [chunk.content_filter_results for chunk in chunks] == [
            _OPENAI_FILTER,
            _GEMINI_FILTER,
            _UNMODELLED_FILTER,
        ]
        assert chunks[0].id == CHUNK_ID

    def test_older_server_transcript_leaves_every_new_field_none(self) -> None:
        start_payload = _message_start_payload()
        del start_payload["id"]
        chunks = [
            _decode_chunk(start_payload),
            _decode_chunk(_content_chunk_payload()),
            _decode_chunk(_message_end_payload()),
        ]
        assert all(chunk.content_filter_results is None for chunk in chunks)
        assert isinstance(chunks[0], StreamMessageStart)
        assert chunks[0].id is None
