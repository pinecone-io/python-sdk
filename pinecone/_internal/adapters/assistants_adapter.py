"""Adapter for Assistants API responses."""

from __future__ import annotations

from msgspec import Struct

from pinecone._internal.adapters._decode import decode_response, decode_response_lax
from pinecone.models.assistant.chat import ChatCompletionResponse, ChatResponse, ChatUsage
from pinecone.models.assistant.context import ContextResponse
from pinecone.models.assistant.evaluation import AlignmentResult, AlignmentScores, EntailmentResult
from pinecone.models.assistant.file_model import AssistantFileModel
from pinecone.models.assistant.list import (
    ListAssistantsResponse,
    ListFilesResponse,
    ListOperationsResponse,
)
from pinecone.models.assistant.model import AssistantModel
from pinecone.models.assistant.operation import OperationModel


class _Fact(Struct, kw_only=True):
    content: str


class _EvaluatedFact(Struct, kw_only=True):
    fact: _Fact
    entailment: str
    # Not in assistant_evaluation_2026-07.oas.yaml's EvaluatedFact, but the
    # evaluation service returns it and EntailmentResult.reasoning exposes it.
    reasoning: str = ""


class _Reasoning(Struct, kw_only=True):
    evaluated_facts: list[_EvaluatedFact]


class _Metrics(Struct, kw_only=True):
    correctness: float
    completeness: float
    alignment: float


class _TokenCounts(Struct, kw_only=True):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class _AlignmentResponse(Struct, kw_only=True):
    """Wire shape of the 200 body of ``POST /evaluation/metrics/alignment``.

    Kept separate from :class:`AlignmentResult` because the SDK-facing model
    renames ``metrics`` to ``scores`` and flattens ``reasoning`` to ``facts``.
    """

    metrics: _Metrics
    reasoning: _Reasoning
    usage: _TokenCounts


class AssistantsAdapter:
    """Transforms raw API JSON into AssistantModel / ListAssistantsResponse instances."""

    @staticmethod
    def to_assistant(data: bytes) -> AssistantModel:
        """Decode raw JSON bytes into an AssistantModel."""
        return decode_response(data, AssistantModel)

    @staticmethod
    def to_assistant_list(data: bytes) -> ListAssistantsResponse:
        """Decode raw JSON bytes into a ListAssistantsResponse."""
        return decode_response(data, ListAssistantsResponse)

    @staticmethod
    def to_file(data: bytes) -> AssistantFileModel:
        """Decode raw JSON bytes into an AssistantFileModel."""
        return decode_response(data, AssistantFileModel)

    @staticmethod
    def to_operation(data: bytes) -> OperationModel:
        """Decode raw JSON bytes into an OperationModel."""
        return decode_response(data, OperationModel)

    @staticmethod
    def to_operation_list(data: bytes) -> ListOperationsResponse:
        """Decode raw JSON bytes into a ListOperationsResponse."""
        return decode_response(data, ListOperationsResponse)

    @staticmethod
    def to_file_list(data: bytes) -> ListFilesResponse:
        """Decode raw JSON bytes into a ListFilesResponse."""
        return decode_response(data, ListFilesResponse)

    @staticmethod
    def to_chat_response(data: bytes) -> ChatResponse:
        """Decode raw JSON bytes into a ChatResponse."""
        return decode_response(data, ChatResponse)

    @staticmethod
    def to_chat_completion_response(data: bytes) -> ChatCompletionResponse:
        """Decode raw JSON bytes into a ChatCompletionResponse."""
        return decode_response(data, ChatCompletionResponse)

    @staticmethod
    def to_context_response(data: bytes) -> ContextResponse:
        """Decode raw JSON bytes into a ContextResponse."""
        return decode_response(data, ContextResponse)

    @staticmethod
    def to_alignment_result(data: bytes) -> AlignmentResult:
        """Decode raw JSON bytes into an AlignmentResult.

        Transforms the API response shape (``metrics``, ``reasoning``, ``usage``)
        into the SDK model shape (``scores``, ``facts``, ``usage``).
        """
        wire = decode_response_lax(data, _AlignmentResponse)
        return AlignmentResult(
            scores=AlignmentScores(
                correctness=wire.metrics.correctness,
                completeness=wire.metrics.completeness,
                alignment=wire.metrics.alignment,
            ),
            facts=[
                EntailmentResult(
                    fact=item.fact.content,
                    entailment=item.entailment,
                    reasoning=item.reasoning,
                )
                for item in wire.reasoning.evaluated_facts
            ],
            usage=ChatUsage(
                prompt_tokens=wire.usage.prompt_tokens,
                completion_tokens=wire.usage.completion_tokens,
                total_tokens=wire.usage.total_tokens,
            ),
        )
