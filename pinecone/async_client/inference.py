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
    resolve_model_id,
)
from pinecone._internal.constants import INFERENCE_API_VERSION
from pinecone._internal.validation import (
    require_non_empty,
    require_one_of,
    require_rerank_top_n,
)
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
    """Discovery for the embedding and reranking models a project can use.

    Reached as ``pc.inference.model``. Its two methods are the same operations
    as :meth:`AsyncInference.list_models` and :meth:`AsyncInference.get_model` —
    take whichever reads better at the call site. Not constructed directly.

    Examples:
        An unfiltered listing spans both model types — embedding models and
        reranking models alike:

        .. code-block:: python

            from pinecone import AsyncPinecone

            async with AsyncPinecone(api_key="your-api-key") as pc:
                models = await pc.inference.model.list()
                print(models.names())
    """

    def __init__(self, inference: AsyncInference) -> None:
        self._inference = inference

    async def list(
        self,
        *,
        type: str | None = None,
        vector_type: str | None = None,
    ) -> ModelInfoList:
        """List the inference models available to this project.

        Delegates to :meth:`AsyncInference.list_models`.

        Args:
            type (str | None): Restrict the listing to one model type,
                ``"embed"`` or ``"rerank"``. Omit it to get both.
            vector_type (str | None): Restrict embedding models to those
                producing ``"dense"`` or ``"sparse"`` vectors. Carries meaning
                only alongside ``type="embed"``.

        Returns:
            :class:`~pinecone.models.inference.model_list.ModelInfoList` — a
            sequence of :class:`~pinecone.models.inference.models.ModelInfo`
            supporting iteration, indexing and ``len()``, plus ``names()`` when
            you want the model identifiers alone.

        Raises:
            :exc:`PineconeValueError`: If *type* or *vector_type* is not one of
                the values above, or if *vector_type* is paired with
                ``type="rerank"`` — the client rejects that pairing rather than
                ignoring it.

        Examples:

            .. code-block:: python

                from pinecone import AsyncPinecone

                async with AsyncPinecone(api_key="your-api-key") as pc:
                    for info in await pc.inference.model.list():
                        print(info.model, info.type)

            Narrow to the embedding models that produce sparse vectors:

            .. code-block:: python

                async with AsyncPinecone(api_key="your-api-key") as pc:
                    sparse = await pc.inference.model.list(
                        type="embed",
                        vector_type="sparse",
                    )
                    print(sparse.names())
        """
        return await self._inference.list_models(type=type, vector_type=vector_type)

    async def get(self, model: str | None = None, **kwargs: str) -> ModelInfo:
        """Describe one inference model.

        Delegates to :meth:`AsyncInference.get_model`.

        Args:
            model (str): The model name to look up, e.g.
                ``"multilingual-e5-large"``. Call :meth:`list` for the names
                currently available.
            model_name (str): Deprecated alias for *model*. Passing both raises
                :exc:`PineconeValueError`.

        Returns:
            :class:`~pinecone.models.inference.models.ModelInfo` with
            ``supported_parameters`` (the keys this model accepts in a
            *parameters* argument), ``type``, and — for embedding models —
            ``vector_type``, ``default_dimension`` and
            ``supported_dimensions``.

        Raises:
            :exc:`PineconeValueError`: If *model* is empty, or if both *model*
                and *model_name* are given.
            :exc:`TypeError`: If any keyword argument other than those above is
                passed.
            :exc:`NotFoundError`: If no model of that name exists.

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
    """Embedding and reranking against Pinecone's hosted models.

    Reached as ``pc.inference``. Call these when you want the vectors or the
    scores in your own hands — to store somewhere else, to embed a query
    yourself, or to rerank candidates that came from another system. If instead
    you want Pinecone to embed on your behalf, build an index with
    :class:`~pinecone.models.indexes.specs.IntegratedSpec` and use
    :meth:`~pinecone.async_client.async_index.AsyncIndex.upsert_records`, which
    needs no explicit embed step. Not constructed directly.

    Examples:

        .. code-block:: python

            from pinecone import AsyncPinecone

            async with AsyncPinecone(api_key="your-api-key") as pc:
                embeddings = await pc.inference.embed(
                    model="multilingual-e5-large",
                    inputs=["Vector databases index embeddings for similarity search."],
                    parameters={"input_type": "passage"},
                )
                print(len(embeddings))

    .. seealso::
       :doc:`/guides/error-handling` — the exceptions every method here can
       raise, and how to retry them.
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
        """Model discovery for this namespace.

        Returns:
            An :class:`AsyncModelResource` exposing ``list()`` and ``get()``.

        Examples:
            .. code-block:: python

                from pinecone import AsyncPinecone

                async with AsyncPinecone(api_key="your-api-key") as pc:
                    info = await pc.inference.model.get("multilingual-e5-large")
                    print(info.default_dimension)

                    models = await pc.inference.model.list()
                    print(models.names())
        """
        return AsyncModelResource(self)

    async def embed(
        self,
        model: _enums.EmbedModel | str,
        inputs: str | Sequence[str] | Sequence[Mapping[str, Any]],
        parameters: Mapping[str, Any] | None = None,
    ) -> EmbeddingsList:
        """Generate embeddings for the provided inputs.

        Many models are asymmetric — they embed a stored passage and a search
        query differently — so where a model accepts ``input_type``, pass it in
        *parameters*, or the query and the corpus will not line up.

        Args:
            model (EmbedModel | str): Embedding model name, e.g.
                ``"multilingual-e5-large"``. An
                :class:`~pinecone.models.enums.EmbedModel` member is accepted
                too; call :meth:`list_models` with ``type="embed"`` for the
                names currently available.
            inputs (str | Sequence[str] | Sequence[Mapping[str, Any]]): The text
                to embed. Any sequence (list, tuple) of strings or mappings; a
                bare string is wrapped for you and still comes back as a
                one-item result rather than a lone embedding.
            parameters (Mapping[str, Any] | None): Model-specific parameters
                (e.g., ``{"input_type": "passage", "truncate": "END"}``).
                Call :meth:`get_model` and read ``supported_parameters`` to
                discover the keys a given model accepts.

        Returns:
            :class:`~pinecone.models.inference.embed.EmbeddingsList` — one
            embedding per input, in input order. Iterating it (or indexing into
            it) yields the embeddings themselves, and ``data`` holds the same
            list. ``vector_type`` says which shape they are and so which fields
            they carry: :class:`~pinecone.models.inference.embed.DenseEmbedding`
            has ``values``, while
            :class:`~pinecone.models.inference.embed.SparseEmbedding` has
            ``sparse_values`` and ``sparse_indices``. ``model`` names the model
            that served the request, and ``usage.total_tokens`` the tokens
            counted for it.

        Raises:
            :exc:`PineconeValueError`: If *model* is empty or *inputs* is empty.
            :exc:`PineconeTypeError`: If *inputs* has an invalid type.
            :exc:`NotFoundError`: If *model* is not available to this project —
                either no such model exists, or the project is not authorized to
                use it. The error does not distinguish the two cases.

        Examples:
            Embed the text you intend to store. ``input_type="passage"`` is the
            corpus side of a search:

            .. code-block:: python

                from pinecone import AsyncPinecone

                async with AsyncPinecone(api_key="your-api-key") as pc:
                    embeddings = await pc.inference.embed(
                        model="multilingual-e5-large",
                        inputs=[
                            "Vector databases index embeddings for similarity search.",
                            "Reranking reorders candidate results by relevance.",
                        ],
                        parameters={"input_type": "passage"},
                    )
                    print(len(embeddings), embeddings.vector_type)

            Embed the search query with ``input_type="query"``. The two are
            not interchangeable — a query embedded as a passage will not land
            where the model expects it:

            .. code-block:: python

                async with AsyncPinecone(api_key="your-api-key") as pc:
                    query = await pc.inference.embed(
                        model="multilingual-e5-large",
                        inputs="How does reranking work?",
                        parameters={"input_type": "query"},
                    )
                    print(len(query.data))

        .. note::
           To store these vectors in a Pinecone index, read the values off each
           embedding and pass them to
           :meth:`~pinecone.async_client.async_index.AsyncIndex.upsert`::

               idx = await pc.index(name="product-search")
               values = embeddings.data[0].values
               await idx.upsert(vectors=[("doc-1", values)])

           ``values`` exists only on the dense shape. A sparse embedding model
           returns :class:`~pinecone.models.inference.embed.SparseEmbedding`
           objects, which carry ``sparse_values`` and ``sparse_indices`` and
           have no ``values`` field — reading ``.values`` on one hands back a
           dict-view method rather than a vector, and raises nothing to warn
           you. Branch on ``embeddings.vector_type`` when the model is not
           fixed in advance.

        .. seealso::
           :meth:`~pinecone.async_client.async_index.AsyncIndex.upsert_records`
           — on an index built with
           :class:`~pinecone.models.indexes.specs.IntegratedSpec`, Pinecone
           embeds the records for you and no call here is needed.
        """
        model_id = resolve_model_id(model)
        require_non_empty("model", model_id)
        normalized_inputs = normalize_embed_inputs(inputs)

        body: dict[str, Any] = {
            "model": model_id,
            "inputs": normalized_inputs,
        }
        if parameters is not None:
            body["parameters"] = parameters

        logger.info("Generating embeddings with model %r", model_id)
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
            model (RerankModel | str): Reranking model name, e.g.
                ``"bge-reranker-v2-m3"``. A
                :class:`~pinecone.models.enums.RerankModel` member is accepted
                too; call :meth:`list_models` with ``type="rerank"`` for the
                names currently available.
            query (str): The text the documents are scored against.
            documents (Sequence[str] | Sequence[Mapping[str, Any]]): Documents to rank.
                Any sequence (list, tuple) of strings or mappings. A bare
                string is wrapped as ``{"text": ...}``, which is what the
                default *rank_fields* scores on.
            rank_fields (Sequence[str]): The document keys to score, e.g.
                ``["summary"]`` when the text lives under ``summary``. Defaults
                to ``["text"]``.
            return_documents (bool): Send each document back in its result.
                Leave it ``True`` to read ``.document``; set it ``False`` when
                you already hold the documents and want only ``index`` and
                ``score``.
            top_n (int | None): Keep only the *n* best-scoring documents.
                ``None``, the default, returns a result for every document.
            parameters (Mapping[str, Any] | None): Model-specific parameters.
                Call :meth:`get_model` and read ``supported_parameters`` to
                discover the keys a given model accepts.

        Returns:
            :class:`~pinecone.models.inference.rerank.RerankResult` whose
            ``data`` is a list of
            :class:`~pinecone.models.inference.rerank.RankedDocument` ordered by
            descending ``score``. Each one carries the ``index`` it held in
            *documents* and, unless *return_documents* is ``False``, the
            ``document`` itself. ``model`` names the model that served the
            request, and ``usage.rerank_units`` the units counted for it.

        Raises:
            :exc:`PineconeValueError`: If *model*, *query*, or *documents* is
                empty, or *top_n* is less than 1.
            :exc:`PineconeTypeError`: If *documents* has an invalid type.
            :exc:`NotFoundError`: If *model* does not name a model the API
                serves. A typo in the model name surfaces here, so check this
                before assuming the request body was at fault.
            :exc:`ForbiddenError`: If the project is not authorized to use
                *model*, including when *model* has been deprecated.

        Examples:
            Rank a list of strings against the query. ``result.data`` comes back
            ordered by descending relevance, not by the order the documents were
            passed in, so read ``.index`` to map a result back to its position
            in *documents* — the top hit below is the second document, so its
            ``.index`` is ``1``, not ``0``:

            .. code-block:: python

                from pinecone import AsyncPinecone

                async with AsyncPinecone(api_key="your-api-key") as pc:
                    result = await pc.inference.rerank(
                        model="bge-reranker-v2-m3",
                        query="Tell me about tech companies",
                        documents=["Apple is a fruit.", "Acme Inc. revolutionized tech."],
                        top_n=1,
                    )
                    top = result.data[0]
                    print(top.index, top.score, top.document["text"])

            Pass mappings instead when you want your own identifiers back
            alongside the scores. Every key other than the ones named in
            *rank_fields* rides along untouched and comes back in
            ``.document``:

            .. code-block:: python

                async with AsyncPinecone(api_key="your-api-key") as pc:
                    result = await pc.inference.rerank(
                        model="bge-reranker-v2-m3",
                        query="Tell me about tech companies",
                        documents=[
                            {"id": "doc-1", "summary": "Apple is a fruit."},
                            {"id": "doc-2", "summary": "Acme Inc. revolutionized tech."},
                        ],
                        rank_fields=["summary"],
                        top_n=1,
                    )
                    print(result.data[0].document["id"])

        .. note::
           The model you request may not be the model that serves the request —
           Pinecone may substitute a different one. ``result.model`` reports
           which one did, so read it there rather than assuming it echoes
           *model*.

        .. seealso::
           :meth:`~pinecone.async_client.async_index.AsyncIndex.search_records`
           — its ``rerank`` argument reranks that search's own hits in one round
           trip. Reach for the method here when the candidates came from
           somewhere else.
        """
        model_id = resolve_model_id(model)
        require_non_empty("model", model_id)
        require_non_empty("query", query)
        normalized_docs = normalize_rerank_documents(documents)
        require_rerank_top_n(top_n)

        body: dict[str, Any] = {
            "model": model_id,
            "query": query,
            "documents": normalized_docs,
            "rank_fields": rank_fields,
            "return_documents": return_documents,
        }
        if top_n is not None:
            body["top_n"] = top_n
        if parameters is not None:
            body["parameters"] = parameters

        logger.info("Reranking %d documents with model %r", len(normalized_docs), model_id)
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
        """List the inference models available to this project.

        Args:
            type (str | None): Restrict the listing to one model type,
                ``"embed"`` or ``"rerank"``. Omit it to get both.
            vector_type (str | None): Restrict embedding models to those
                producing ``"dense"`` or ``"sparse"`` vectors. Carries meaning
                only alongside ``type="embed"``.

        Returns:
            :class:`~pinecone.models.inference.model_list.ModelInfoList` — a
            sequence of :class:`~pinecone.models.inference.models.ModelInfo`
            supporting iteration, indexing and ``len()``, plus ``names()`` when
            you want the model identifiers alone.

        Raises:
            :exc:`PineconeValueError`: If *type* or *vector_type* is not one of
                the values above, or if *vector_type* is paired with
                ``type="rerank"`` — the client rejects that pairing rather than
                ignoring it.

        Examples:

            .. code-block:: python

                from pinecone import AsyncPinecone

                async with AsyncPinecone(api_key="your-api-key") as pc:
                    models = await pc.inference.list_models()
                    print(models.names())

            Narrow to the embedding models that produce sparse vectors:

            .. code-block:: python

                async with AsyncPinecone(api_key="your-api-key") as pc:
                    sparse = await pc.inference.list_models(
                        type="embed",
                        vector_type="sparse",
                    )
                    print(sparse.names())
        """
        if type is not None:
            require_one_of("type", type, ("embed", "rerank"))
        if vector_type is not None:
            if type == "rerank":
                raise ValidationError("vector_type is not supported when type='rerank'")
            require_one_of("vector_type", vector_type, ("dense", "sparse"))

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
        """Describe one inference model.

        Args:
            model (str): The model name to look up, e.g.
                ``"multilingual-e5-large"``. Call :meth:`list_models` for the
                names currently available.
            model_name (str): Deprecated alias for *model*. Passing both raises
                :exc:`PineconeValueError`.

        Returns:
            :class:`~pinecone.models.inference.models.ModelInfo` with
            ``supported_parameters`` (the keys *parameters* accepts on
            :meth:`embed` and :meth:`rerank` for this model), ``type``, and —
            for embedding models — ``vector_type``, ``default_dimension`` and
            ``supported_dimensions``.

        Raises:
            :exc:`PineconeValueError`: If *model* is empty, or if both *model*
                and *model_name* are given.
            :exc:`TypeError`: If any keyword argument other than those above is
                passed.
            :exc:`NotFoundError`: If no model of that name exists.

        Examples:
            ``supported_parameters`` is what :meth:`embed` and :meth:`rerank`
            point at for discovering the keys their *parameters* argument
            accepts, and each entry names the values it will take:

            .. code-block:: python

                from pinecone import AsyncPinecone

                async with AsyncPinecone(api_key="your-api-key") as pc:
                    model_info = await pc.inference.get_model(
                        model="multilingual-e5-large",
                    )
                    print(model_info.type)
                    for p in model_info.supported_parameters:
                        print(p.parameter, p.allowed_values)
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
