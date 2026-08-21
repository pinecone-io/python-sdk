"""Adapter for Inference API responses and request normalization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any

from msgspec import Struct

from pinecone._internal.adapters._decode import convert_response, decode_response
from pinecone.errors.exceptions import PineconeTypeError, ValidationError
from pinecone.models.inference.embed import (
    DenseEmbedding,
    EmbeddingsList,
    EmbedUsage,
    SparseEmbedding,
)
from pinecone.models.inference.model_list import ModelInfoList
from pinecone.models.inference.models import ModelInfo
from pinecone.models.inference.rerank import RerankResult


class _EmbedEnvelope(Struct, kw_only=True):
    """Internal envelope for the embed response.

    Decodes the top-level fields while keeping data items as raw dicts
    so we can dispatch on vector_type before decoding individual items.
    """

    model: str
    vector_type: str
    data: list[Any]
    usage: EmbedUsage


class _ModelListEnvelope(Struct, kw_only=True):
    """Internal envelope for the list-models response."""

    models: list[ModelInfo] = []


class InferenceAdapter:
    """Transforms raw API JSON into inference model instances."""

    @staticmethod
    def to_embeddings_list(data: bytes) -> EmbeddingsList:
        """Decode raw JSON bytes from the embed endpoint into an EmbeddingsList.

        The embed response contains a ``vector_type`` field that determines
        whether items in ``data`` are dense or sparse embeddings. This method
        first decodes the envelope, then converts items to the correct type.
        """
        envelope = decode_response(data, _EmbedEnvelope)

        if envelope.vector_type == "sparse":
            embeddings: list[DenseEmbedding] | list[SparseEmbedding] = [
                convert_response(item, SparseEmbedding) for item in envelope.data
            ]
        else:
            embeddings = [convert_response(item, DenseEmbedding) for item in envelope.data]

        return EmbeddingsList(
            model=envelope.model,
            vector_type=envelope.vector_type,
            data=embeddings,
            usage=envelope.usage,
        )

    @staticmethod
    def to_rerank_result(data: bytes) -> RerankResult:
        """Decode raw JSON bytes into a RerankResult."""
        return decode_response(data, RerankResult)

    @staticmethod
    def to_model_info(data: bytes) -> ModelInfo:
        """Decode raw JSON bytes into a ModelInfo."""
        return decode_response(data, ModelInfo)

    @staticmethod
    def to_model_info_list(data: bytes) -> ModelInfoList:
        """Decode raw JSON bytes from the list-models endpoint into a ModelInfoList."""
        envelope = decode_response(data, _ModelListEnvelope)
        return ModelInfoList(envelope.models)


def resolve_model_id(model: str | Enum) -> str:
    """Return the wire identifier for a model.

    ``EmbedModel``/``RerankModel`` are ``(str, Enum)`` mixins, so ``str(member)``
    yields ``"EmbedModel.X"`` rather than the model id — read ``.value`` instead.
    """
    return str(model.value) if isinstance(model, Enum) else str(model)


def normalize_embed_inputs(
    inputs: str | Sequence[str] | Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Normalize embed inputs into the API's expected format.

    Accepts a single string, any Sequence of strings, or any Sequence of
    Mappings. Tuples and other Sequence-conforming types all work.

    Args:
        inputs: A single string, Sequence of strings, or Sequence of Mappings.

    Returns:
        A list of dicts with ``"text"`` keys.

    Raises:
        ValidationError: If inputs is an empty Sequence.
        PineconeTypeError: If inputs is not a recognized type or contains mixed types.
    """
    if isinstance(inputs, str):
        return [{"text": inputs}]
    if not isinstance(inputs, Sequence):
        raise PineconeTypeError(
            f"Expected str, Sequence[str], or Sequence[Mapping[str, Any]], "
            f"got {type(inputs).__name__}"
        )
    items = list(inputs)
    if len(items) == 0:
        raise ValidationError("inputs must not be empty")
    if not all(isinstance(item, (str, Mapping)) for item in items):
        raise PineconeTypeError("each input must be a string or mapping")
    first = items[0]
    if isinstance(first, str):
        if not all(isinstance(item, str) for item in items):
            raise PineconeTypeError("each input must be a string or mapping")
        return [{"text": s} for s in items]
    if not all(isinstance(item, Mapping) for item in items):
        raise PineconeTypeError("each input must be a string or mapping")
    return [dict(item) for item in items]  # type: ignore[arg-type]


def normalize_rerank_documents(
    documents: Sequence[str] | Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Normalize rerank documents into the API's expected format.

    Accepts any Sequence of strings or Sequence of Mappings.

    Args:
        documents: A Sequence of strings or Sequence of Mappings.

    Returns:
        A list of dicts suitable for the rerank API.  String inputs are wrapped as
        ``{"text": s}``; Mapping inputs are copied as-is.

    Raises:
        PineconeTypeError: If documents is not a Sequence (or is a string)
            or contains invalid element types.
        ValidationError: If documents is empty.
    """
    if isinstance(documents, str) or not isinstance(documents, Sequence):
        raise PineconeTypeError("documents must be a Sequence of strings or Sequence of mappings")
    items = list(documents)
    if len(items) == 0:
        raise ValidationError("documents must not be empty")
    if not all(isinstance(d, (str, Mapping)) for d in items):
        raise PineconeTypeError("each document must be a string or mapping")
    first = items[0]
    if isinstance(first, str):
        if not all(isinstance(d, str) for d in items):
            raise PineconeTypeError("each document must be a string or mapping")
        return [{"text": s} for s in items]
    if not all(isinstance(d, Mapping) for d in items):
        raise PineconeTypeError("each document must be a string or mapping")
    return [dict(d) for d in items]  # type: ignore[arg-type]
