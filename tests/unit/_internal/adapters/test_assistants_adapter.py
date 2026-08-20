from __future__ import annotations

import msgspec
import orjson
import pytest

from pinecone._internal.adapters.assistants_adapter import AssistantsAdapter
from pinecone.errors.exceptions import ResponseParsingError

_VALID_PAYLOAD: dict[str, object] = {
    "metrics": {
        "correctness": 0.9,
        "completeness": 0.8,
        "alignment": 0.85,
    },
    "reasoning": {
        "evaluated_facts": [
            {"fact": {"content": "The sky is blue."}, "entailment": "entailed"},
        ]
    },
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    },
}


class TestToAlignmentResultPositive:
    def test_decodes_the_spec_response_shape(self) -> None:
        result = AssistantsAdapter.to_alignment_result(orjson.dumps(_VALID_PAYLOAD))
        assert result.scores.correctness == 0.9
        assert result.scores.completeness == 0.8
        assert result.scores.alignment == 0.85
        assert [(f.fact, f.entailment, f.reasoning) for f in result.facts] == [
            ("The sky is blue.", "entailed", "")
        ]
        assert result.usage.total_tokens == 15

    def test_coerces_numeric_strings(self) -> None:
        payload = {
            **_VALID_PAYLOAD,
            "metrics": {"correctness": "0.42", "completeness": 0.8, "alignment": 0.85},
        }
        result = AssistantsAdapter.to_alignment_result(orjson.dumps(payload))
        assert result.scores.correctness == 0.42


class TestToAlignmentResultNegative:
    def test_metrics_not_dict_raises_response_parsing_error(self) -> None:
        payload = {**_VALID_PAYLOAD, "metrics": [1, 2, 3]}
        data = orjson.dumps(payload)
        with pytest.raises(ResponseParsingError, match="Failed to parse") as exc_info:
            AssistantsAdapter.to_alignment_result(data)
        assert isinstance(exc_info.value.cause, msgspec.ValidationError)
        assert "metrics" in str(exc_info.value.cause)

    def test_reasoning_not_dict_raises_response_parsing_error(self) -> None:
        payload = {**_VALID_PAYLOAD, "reasoning": "not-a-dict"}
        data = orjson.dumps(payload)
        with pytest.raises(ResponseParsingError, match="Failed to parse") as exc_info:
            AssistantsAdapter.to_alignment_result(data)
        assert isinstance(exc_info.value.cause, msgspec.ValidationError)
        assert "reasoning" in str(exc_info.value.cause)

    def test_evaluated_facts_not_list_raises_response_parsing_error(self) -> None:
        payload = {
            **_VALID_PAYLOAD,
            "reasoning": {"evaluated_facts": {"not": "a list"}},
        }
        data = orjson.dumps(payload)
        with pytest.raises(ResponseParsingError, match="Failed to parse") as exc_info:
            AssistantsAdapter.to_alignment_result(data)
        assert isinstance(exc_info.value.cause, msgspec.ValidationError)
        assert "evaluated_facts" in str(exc_info.value.cause)

    def test_usage_not_dict_raises_response_parsing_error(self) -> None:
        payload = {**_VALID_PAYLOAD, "usage": 42}
        data = orjson.dumps(payload)
        with pytest.raises(ResponseParsingError, match="Failed to parse") as exc_info:
            AssistantsAdapter.to_alignment_result(data)
        assert isinstance(exc_info.value.cause, msgspec.ValidationError)
        assert "usage" in str(exc_info.value.cause)

    def test_missing_key_raises_response_parsing_error(self) -> None:
        payload = {k: v for k, v in _VALID_PAYLOAD.items() if k != "metrics"}
        data = orjson.dumps(payload)
        with pytest.raises(ResponseParsingError, match="Failed to parse") as exc_info:
            AssistantsAdapter.to_alignment_result(data)
        assert isinstance(exc_info.value.cause, msgspec.ValidationError)
        assert "metrics" in str(exc_info.value.cause)

    def test_invalid_numeric_string_raises_response_parsing_error(self) -> None:
        payload = {
            **_VALID_PAYLOAD,
            "metrics": {
                "correctness": "not-a-number",
                "completeness": 0.8,
                "alignment": 0.85,
            },
        }
        data = orjson.dumps(payload)
        with pytest.raises(ResponseParsingError, match="Failed to parse") as exc_info:
            AssistantsAdapter.to_alignment_result(data)
        assert isinstance(exc_info.value.cause, msgspec.ValidationError)
        assert "correctness" in str(exc_info.value.cause)
