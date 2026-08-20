"""Async Inference namespace — embed, rerank, and model operations."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from functools import cached_property
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from pinecone._internal.adapters.inference_adapter import (
    InferenceAdapter,
    normalize_embed_inputs,
    normalize_rerank_documents,
)
from pinecone._internal.constants import INFERENCE_API_VERSION
from pinecone._internal.validation import require_non_empty, require_one_of
from pinecone.errors.exceptions import ValidationError
from pinecone.models import enums as _enums

if TYPE_CHECKING:
    from pinecone._internal.config import PineconeConfig
    from pinecone.models.inference.embed import EmbeddingsList
    from pinecone.models.inference.model_list import ModelInfoList
    from pinecone.models.inference.models import ModelInfo
    from pinecone.models.inference.rerank import RerankResult

logger = logging.getLogger(__name__)

_DEFAULT_RANK_FIELDS: list[str] = ["text"]


class AsyncModelResource:
    """Lazily-initialized resource for listing and getting inference model info.

    Accessed via ``pc.inference.model``.

    Args:
        inference (AsyncInference): The parent async inference namespace that
            handles HTTP requests on behalf of this resource.

    Examples:

        .. code-block:: python

            # List all available models
            from pinecone import AsyncPinecone
            async with AsyncPinecone(api_key="your-api-key") as pc:
                models = await pc.inference.model.list()
                models.names()

        .. code-block:: python

            # Get details about a specific model
            async with AsyncPinecone(api_key="your-api-key") as pc:
                info = await pc.inference.model.get("multilingual-e5-large")
                print(info.type)
    """

    def __init__(self, inference: AsyncInference) -> None:
        self._inference = inference

    async def list(
        self,
        *,
        type: str | None = None,
        vector_type: str | None = None,
    ) -> ModelInfoList:
        """List available inference models.

        Delegates to :meth:`~AsyncInference.list_models`.

        Args:
            type (str | None): Filter by model type (``"embed"`` or ``"rerank"``).
            vector_type (str | None): Filter by vector type
                (``"dense"`` or ``"sparse"``). Only relevant when ``type="embed"``.

        Returns:
            A :class:`ModelInfoList` supporting iteration, len(), and ``.names()``.

        Raises:
            :exc:`ApiError`: If the API returns an error response.

        Examples:

            .. code-block:: python

                from pinecone import AsyncPinecone
                async with AsyncPinecone(api_key="your-api-key") as pc:
                    models = await pc.inference.model.list()
                    embed_models = await pc.inference.model.list(type="embed")
        """
        return await self._inference.list_models(type=type, vector_type=vector_type)

    async def get(self, model: str | None = None, **kwargs: str) -> ModelInfo:
        """Get detailed information about a specific model.

        Delegates to :meth:`~AsyncInference.get_model`.

        Args:
            model (str): The model identifier to look up.

        Returns:
            A :class:`ModelInfo` with full model details.

        Raises:
            :exc:`NotFoundError`: If the model does not exist.
            :exc:`ApiError`: If the API returns another error response.

        Examples:

            .. code-block:: python

                from pinecone import AsyncPinecone
                async with AsyncPinecone(api_key="your-api-key") as pc:
                    info = await pc.inference.model.get("multilingual-e5-large")
                    print(info.type)
        """
        model_name: str | None = kwargs.pop("model_name", None)
        if kwargs:
            raise TypeError(f"get() got unexpected keyword arguments: {sorted(kwargs)!r}")
        if model is not None and model_name is not None:
            raise ValidationError("Provide either model= or model_name=, not both")
        effective: str = model or model_name or ""
        return await self._inference.get_model(model=effective)


class AsyncInference:
    """Asynchronous operations for Pinecone inference (embed & rerank).

    Provides async methods to generate embeddings and rerank documents
    using Pinecone's hosted models.

    Args:
        config (PineconeConfig): SDK configuration used to construct an
            async HTTP client targeting the inference API version.

    Examples:

        .. code-block:: python

            from pinecone import AsyncPinecone

            async with AsyncPinecone(api_key="your-api-key") as pc:
                embeddings = await pc.inference.embed(
                    model="multilingual-e5-large",
                    inputs=["Hello, world!"],
                )
    """

    EmbedModel = _enums.EmbedModel
    RerankModel = _enums.RerankModel

    def __init__(self, config: PineconeConfig) -> None:
        from pinecone._internal.http_client import AsyncHTTPClient

        self._http = AsyncHTTPClient(config, INFERENCE_API_VERSION)
        self._adapter = InferenceAdapter()

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._http.close()

    def __repr__(self) -> str:
        """Return developer-friendly representation."""
        return "AsyncInference()"

    @cached_property
    def model(self) -> AsyncModelResource:
        """Lazily-initialized resource for listing and getting model info.

        Returns:
            A :class:`AsyncModelResource` that exposes ``.list()`` and ``.get()`` methods.

        Examples:
            .. code-block:: python

                from pinecone import AsyncPinecone
                async with AsyncPinecone(api_key="your-api-key") as pc:
                    models = await pc.inference.model.list()
                    info = await pc.inference.model.get("multilingual-e5-large")
        """
        return AsyncModelResource(self)

    async def embed(
        self,
        model: _enums.EmbedModel | str,
        inputs: str | Sequence[str] | Sequence[Mapping[str, Any]],
        parameters: Mapping[str, Any] | None = None,
    ) -> EmbeddingsList:
        """Generate embeddings for the provided inputs.

        Args:
            model (EmbedModel | str): Embedding model name.
            inputs (str | Sequence[str] | Sequence[Mapping[str, Any]]): Text inputs.
                A single string is automatically wrapped. Any Sequence type
                (list, tuple, etc.) of strings or Mappings is accepted.
            parameters (Mapping[str, Any] | None): Model-specific parameters
                (e.g., ``{"input_type": "passage", "truncate": "END"}``).

        Returns:
            An :class:`EmbeddingsList` with ``.data``, ``.model``, and ``.usage``.

        Raises:
            :exc:`PineconeValueError`: If *model* is empty or *inputs* is empty.
            :exc:`PineconeTypeError`: If *inputs* has an invalid type.
            :exc:`ApiError`: If the API returns an error response.
            :exc:`PineconeConnectionError`: If a network-level connection
                fails (DNS, refused, transport error).
            :exc:`PineconeTimeoutError`: If the request exceeds the configured timeout.

        Examples:
            .. code-block:: python

                from pinecone import AsyncPinecone
                async with AsyncPinecone(api_key="your-api-key") as pc:
                    embeddings = await pc.inference.embed(
                        model="multilingual-e5-large",
                        inputs=["Hello, world!"],
                        parameters={"input_type": "passage"},
                    )
        """
        require_non_empty("model", str(model))
        normalized_inputs = normalize_embed_inputs(inputs)

        body: dict[str, Any] = {
            "model": str(model),
            "inputs": normalized_inputs,
        }
        if parameters is not None:
            body["parameters"] = parameters

        logger.info("Generating embeddings with model %r", str(model))
        response = await self._http.post("/embed", json=body)
        result = self._adapter.to_embeddings_list(response.content)
        logger.debug("Generated %d embeddings", len(result.data))
        return result

    async def rerank(
        self,
        model: _enums.RerankModel | str,
        query: str,
        documents: Sequence[str] | Sequence[Mapping[str, Any]],
        rank_fields: Sequence[str] = _DEFAULT_RANK_FIELDS,
        return_documents: bool = True,
        top_n: int | None = None,
        parameters: Mapping[str, Any] | None = None,
    ) -> RerankResult:
        """Rerank documents by relevance to a query.

        Args:
            model (RerankModel | str): Reranking model name.
            query (str): Query text to rank against.
            documents (Sequence[str] | Sequence[Mapping[str, Any]]): Documents to rank.
                Strings are auto-wrapped as ``{"text": ...}``. Any Sequence
                type (list, tuple, etc.) is accepted.
            rank_fields (Sequence[str]): Document fields to rank on.
                Defaults to ``["text"]``.
            return_documents (bool): Include document text in response.
                Defaults to ``True``.
            top_n (int | None): Number of top documents to return.
                ``None`` returns all.
            parameters (dict[str, Any] | None): Model-specific parameters.

        Returns:
            A :class:`RerankResult` with ``.data`` and ``.usage``.

        Raises:
            :exc:`PineconeValueError`: If *model*, *query*, or *documents* is empty.
            :exc:`PineconeTypeError`: If *documents* has an invalid type.
            :exc:`ApiError`: If the API returns an error response.
            :exc:`PineconeConnectionError`: If a network-level connection
                fails (DNS, refused, transport error).
            :exc:`PineconeTimeoutError`: If the request exceeds the configured timeout.

        Examples:
            .. code-block:: python

                from pinecone import AsyncPinecone
                async with AsyncPinecone(api_key="your-api-key") as pc:
                    result = await pc.inference.rerank(
                        model="bge-reranker-v2-m3",
                        query="Tell me about tech companies",
                        documents=["Apple is a fruit.", "Acme Inc. revolutionized tech."],
                        top_n=1,
                    )
        """
        require_non_empty("model", str(model))
        require_non_empty("query", query)
        normalized_docs = normalize_rerank_documents(documents)

        body: dict[str, Any] = {
            "model": str(model),
            "query": query,
            "documents": normalized_docs,
            "rank_fields": rank_fields,
            "return_documents": return_documents,
        }
        if top_n is not None:
            body["top_n"] = top_n
        if parameters is not None:
            body["parameters"] = parameters

        logger.info("Reranking %d documents with model %r", len(normalized_docs), str(model))
        response = await self._http.post("/rerank", json=body)
        result = self._adapter.to_rerank_result(response.content)
        logger.debug("Reranked documents, got %d results", len(result.data))
        return result

    async def list_models(
        self,
        *,
        type: str | None = None,
        vector_type: str | None = None,
    ) -> ModelInfoList:
        """List available inference models.

        Args:
            type (str | None): Filter by model type (``"embed"`` or ``"rerank"``).
            vector_type (str | None): Filter by vector type
                (``"dense"`` or ``"sparse"``). Only relevant when ``type="embed"``.

        Returns:
            A :class:`ModelInfoList` supporting iteration, len(), and ``.names()``.

        Raises:
            :exc:`PineconeValueError`: If *type* or *vector_type* is not a valid value.
            :exc:`ApiError`: If the API returns an error response.
            :exc:`PineconeConnectionError`: If a network-level connection
                fails (DNS, refused, transport error).
            :exc:`PineconeTimeoutError`: If the request exceeds the configured timeout.

        Examples:

            .. code-block:: python

                from pinecone import AsyncPinecone
                async with AsyncPinecone(api_key="your-api-key") as pc:
                    models = await pc.inference.list_models()
                    models.names()
        """
        if type is not None:
            require_one_of("type", type, ("embed", "rerank"))
        if vector_type is not None:
            require_one_of("vector_type", vector_type, ("dense", "sparse"))
        if type == "rerank" and vector_type is not None:
            raise ValidationError("vector_type is not supported when type='rerank'")

        params: dict[str, Any] = {}
        if type is not None:
            params["type"] = type
        if vector_type is not None:
            params["vector_type"] = vector_type

        logger.info("Listing models")
        response = await self._http.get("/models", params=params)
        result = self._adapter.to_model_info_list(response.content)
        logger.debug("Listed %d models", len(result))
        return result

    async def get_model(
        self,
        *,
        model: str | None = None,
        **kwargs: str,
    ) -> ModelInfo:
        """Get detailed information about a specific model.

        Pass the canonical model name. The lookup is an exact match against the
        model's canonical name and does not resolve aliases, so an alias such as
        ``nvidia/llama-text-embed-v2`` raises :exc:`NotFoundError` even though
        ``embed`` accepts it. The name is percent-encoded into the path, so an
        alias containing ``/`` reaches this route and 404s rather than 405s.

        Args:
            model (str): The model identifier to look up. Use the canonical
                model name, not an alias.

        Returns:
            A :class:`ModelInfo` with full model details.

        Raises:
            :exc:`PineconeValueError`: If *model* is empty.
            :exc:`NotFoundError`: If the model does not exist, or if *model* is
                an alias rather than a canonical model name.
            :exc:`ApiError`: If the API returns another error response.
            :exc:`PineconeConnectionError`: If a network-level connection
                fails (DNS, refused, transport error).
            :exc:`PineconeTimeoutError`: If the request exceeds the configured timeout.

        Examples:
            .. code-block:: python

                from pinecone import AsyncPinecone
                async with AsyncPinecone(api_key="your-api-key") as pc:
                    model_info = await pc.inference.get_model(
                        model="multilingual-e5-large",
                    )
                    model_info.type
        """
        model_name: str | None = kwargs.pop("model_name", None)
        if kwargs:
            raise TypeError(f"get_model() got unexpected keyword arguments: {sorted(kwargs)!r}")
        if model is not None and model_name is not None:
            raise ValidationError("Provide either model= or model_name=, not both")
        effective: str = model or model_name or ""
        require_non_empty("model", effective)
        logger.info("Describing model %r", effective)
        response = await self._http.get(f"/models/{quote(effective, safe='')}")
        result = self._adapter.to_model_info(response.content)
        logger.debug("Described model %r", effective)
        return result
