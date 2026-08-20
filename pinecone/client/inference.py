"""Inference namespace — embed and rerank operations."""

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


class ModelResource:
    """Lazily-initialized resource for listing and getting inference model info.

    Accessed via ``pc.inference.model``.

    Args:
        inference (Inference): The parent inference namespace that handles
            HTTP requests on behalf of this resource.

    Examples:
        List all available models:

        >>> from pinecone import Pinecone
        >>> pc = Pinecone(api_key="your-api-key")
        >>> models = pc.inference.model.list()
        >>> models.names()  # doctest: +SKIP
        ['multilingual-e5-large', 'pinecone-sparse-english-v0']

        Get details about a specific model:

        >>> info = pc.inference.model.get("multilingual-e5-large")
        >>> info.type
        'embed'
    """

    def __init__(self, inference: Inference) -> None:
        self._inference = inference

    def list(
        self,
        *,
        type: str | None = None,
        vector_type: str | None = None,
    ) -> ModelInfoList:
        """List available inference models.

        Delegates to :meth:`~Inference.list_models`.

        Args:
            type (str | None): Filter by model type (``"embed"`` or ``"rerank"``).
            vector_type (str | None): Filter by vector type
                (``"dense"`` or ``"sparse"``). Only relevant when ``type="embed"``.

        Returns:
            A :class:`ModelInfoList` supporting iteration, len(), and ``.names()``.

        Raises:
            :exc:`ApiError`: If the API returns an error response.

        Examples:
            >>> from pinecone import Pinecone
            >>> pc = Pinecone(api_key="your-api-key")
            >>> models = pc.inference.model.list()
            >>> embed_models = pc.inference.model.list(type="embed")
        """
        return self._inference.list_models(type=type, vector_type=vector_type)

    def get(self, model: str | None = None, **kwargs: str) -> ModelInfo:
        """Get detailed information about a specific model.

        Delegates to :meth:`~Inference.get_model`.

        Args:
            model (str): The model identifier to look up.

        Returns:
            A :class:`ModelInfo` with full model details.

        Raises:
            :exc:`NotFoundError`: If the model does not exist.
            :exc:`ApiError`: If the API returns another error response.

        Examples:
            >>> from pinecone import Pinecone
            >>> pc = Pinecone(api_key="your-api-key")
            >>> info = pc.inference.model.get("multilingual-e5-large")
            >>> info.type
            'embed'
        """
        model_name: str | None = kwargs.pop("model_name", None)
        if kwargs:
            raise TypeError(f"get() got unexpected keyword arguments: {sorted(kwargs)!r}")
        if model is not None and model_name is not None:
            raise ValidationError("Provide either model= or model_name=, not both")
        effective: str = model or model_name or ""
        return self._inference.get_model(model=effective)


class Inference:
    """Control-plane operations for Pinecone inference (embed & rerank).

    Provides methods to generate embeddings and rerank documents using
    Pinecone's hosted models.

    Args:
        config (PineconeConfig): SDK configuration used to construct an
            HTTP client targeting the inference API version.

    Examples:

        .. code-block:: python

            from pinecone import Pinecone

            pc = Pinecone(api_key="your-api-key")
            embeddings = pc.inference.embed(
                model="multilingual-e5-large", inputs=["Hello, world!"]
            )
    """

    EmbedModel = _enums.EmbedModel
    RerankModel = _enums.RerankModel

    def __init__(self, config: PineconeConfig) -> None:
        from pinecone._internal.http_client import HTTPClient

        self._http = HTTPClient(config, INFERENCE_API_VERSION)
        self._adapter = InferenceAdapter()

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._http.close()

    def __repr__(self) -> str:
        """Return developer-friendly representation."""
        return "Inference()"

    @cached_property
    def model(self) -> ModelResource:
        """Lazily-initialized resource for listing and getting model info.

        Returns:
            A :class:`ModelResource` that exposes ``.list()`` and ``.get()`` methods.

        Examples:
            >>> from pinecone import Pinecone
            >>> pc = Pinecone(api_key="your-api-key")
            >>> models = pc.inference.model.list()
            >>> info = pc.inference.model.get("multilingual-e5-large")
        """
        return ModelResource(self)

    def embed(
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
                To discover valid parameters for a model, call
                :meth:`get_model`::

                    pc.inference.get_model(model="multilingual-e5-large").supported_parameters

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
            >>> from pinecone import Pinecone
            >>> pc = Pinecone(api_key="your-api-key")
            >>> embeddings = pc.inference.embed(
            ...     model="multilingual-e5-large",
            ...     inputs=["Hello, world!"],
            ...     parameters={"input_type": "passage"},
            ... )
            >>> len(embeddings.data)
            1

        .. note::
           To store embeddings in a Pinecone index, extract the raw vector
           values and pass them to :meth:`~pinecone.Index.upsert`::

               values = embeddings.data[0].values
               index.upsert(vectors=[("doc-1", values)])

           Alternatively, use an index with integrated inference
           (``IntegratedSpec``) and call :meth:`~pinecone.Index.upsert_records`
           to let Pinecone handle embedding server-side — no manual embed step
           required.
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
        response = self._http.post("/embed", json=body)
        result = self._adapter.to_embeddings_list(response.content)
        logger.debug("Generated %d embeddings", len(result.data))
        return result

    def rerank(
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
            parameters (Mapping[str, Any] | None): Model-specific parameters.
                To discover valid parameters for a model, call
                :meth:`get_model`::

                    pc.inference.get_model(model="bge-reranker-v2-m3").supported_parameters

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
            >>> from pinecone import Pinecone
            >>> pc = Pinecone(api_key="your-api-key")
            >>> result = pc.inference.rerank(
            ...     model="bge-reranker-v2-m3",
            ...     query="Tell me about tech companies",
            ...     documents=["Apple is a fruit.", "Acme Inc. revolutionized tech."],
            ...     top_n=1,
            ... )
            >>> result.data[0].score  # doctest: +SKIP
            0.95
        """
        require_non_empty("model", str(model))
        require_non_empty("query", query)
        normalized_docs = normalize_rerank_documents(documents)
        if top_n is not None and top_n < 1:
            raise ValidationError("top_n must be >= 1")

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
        response = self._http.post("/rerank", json=body)
        result = self._adapter.to_rerank_result(response.content)
        logger.debug("Reranked documents, got %d results", len(result.data))
        return result

    def list_models(
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
            >>> from pinecone import Pinecone
            >>> pc = Pinecone(api_key="your-api-key")
            >>> models = pc.inference.list_models()  # doctest: +SKIP
            >>> models.names()  # doctest: +SKIP
            ['multilingual-e5-large', 'pinecone-sparse-english-v0']

            >>> embed_models = pc.inference.list_models(type="embed")  # doctest: +SKIP
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
        response = self._http.get("/models", params=params)
        result = self._adapter.to_model_info_list(response.content)
        logger.debug("Listed %d models", len(result))
        return result

    def get_model(
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
            >>> from pinecone import Pinecone
            >>> pc = Pinecone(api_key="your-api-key")
            >>> model_info = pc.inference.get_model(model="multilingual-e5-large")
            >>> model_info.type
            'embed'
        """
        model_name: str | None = kwargs.pop("model_name", None)
        if kwargs:
            raise TypeError(f"get_model() got unexpected keyword arguments: {sorted(kwargs)!r}")
        if model is not None and model_name is not None:
            raise ValidationError("Provide either model= or model_name=, not both")
        effective: str = model or model_name or ""
        require_non_empty("model", effective)
        logger.info("Describing model %r", effective)
        response = self._http.get(f"/models/{quote(effective, safe='')}")
        result = self._adapter.to_model_info(response.content)
        logger.debug("Described model %r", effective)
        return result
