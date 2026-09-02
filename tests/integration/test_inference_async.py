"""Integration tests for inference embed and rerank — async (REST) transport."""

from __future__ import annotations

import pytest

from pinecone import AsyncPinecone
from pinecone.errors import PineconeTypeError, PineconeValueError
from pinecone.errors.exceptions import ValidationError
from pinecone.models.inference.embed import SparseEmbedding

# ---------------------------------------------------------------------------
# embed (async)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_embed_single_string_async(async_client: AsyncPinecone) -> None:
    """async embed() with a single string input returns a 1-item EmbeddingsList."""
    result = await async_client.inference.embed(
        model="multilingual-e5-large",
        inputs=["Hello, world!"],
        parameters={"input_type": "passage", "truncate": "END"},
    )
    assert result.model == "multilingual-e5-large"
    assert result.vector_type == "dense"
    assert len(result) == 1
    assert result.usage.total_tokens > 0

    embedding = result.data[0]
    assert isinstance(embedding.values, list)
    assert len(embedding.values) > 0
    assert all(isinstance(v, float) for v in embedding.values)
    assert embedding.vector_type == "dense"


@pytest.mark.integration
@pytest.mark.anyio
async def test_embed_sparse_model_returns_sparse_embeddings_async(
    async_client: AsyncPinecone,
) -> None:
    """async embed() with pinecone-sparse-english-v0 returns SparseEmbedding objects.

    Async variant of test_embed_sparse_model_returns_sparse_embeddings. Verifies
    unified-enum-0006 and the sparse decode path in InferenceAdapter.
    """
    inputs = [
        "What is vector search?",
        "Pinecone is a managed vector database.",
    ]
    result = await async_client.inference.embed(
        model="pinecone-sparse-english-v0",
        inputs=inputs,
        parameters={"input_type": "passage"},
    )

    assert result.model == "pinecone-sparse-english-v0"
    assert result.vector_type == "sparse"
    assert len(result) == len(inputs)
    assert result.usage.total_tokens > 0

    for emb in result:
        assert isinstance(emb, SparseEmbedding)
        assert emb.vector_type == "sparse"
        assert isinstance(emb.sparse_values, list)
        assert isinstance(emb.sparse_indices, list)
        assert len(emb.sparse_values) > 0
        assert len(emb.sparse_indices) > 0
        assert len(emb.sparse_values) == len(emb.sparse_indices)
        assert all(isinstance(v, float) for v in emb.sparse_values)
        assert all(isinstance(i, int) for i in emb.sparse_indices)
        assert emb.sparse_tokens is None


@pytest.mark.integration
@pytest.mark.anyio
async def test_embed_list_of_dict_inputs_async(async_client: AsyncPinecone) -> None:
    """async embed() accepts a list of dicts as inputs and returns one embedding per dict.

    Async variant of test_embed_list_of_dict_inputs. Verifies unified-inf-0009:
    the embed operation accepts list[dict] inputs end-to-end with the real API.
    """
    inputs = [
        {"text": "What is a vector database?"},
        {"text": "Pinecone is a managed vector database service."},
    ]
    result = await async_client.inference.embed(
        model="multilingual-e5-large",
        inputs=inputs,
        parameters={"input_type": "passage"},
    )

    assert result.model == "multilingual-e5-large"
    assert result.vector_type == "dense"
    assert len(result) == len(inputs)
    assert result.usage.total_tokens > 0

    for emb in result:
        assert isinstance(emb.values, list)
        assert len(emb.values) > 0
        assert all(isinstance(v, float) for v in emb.values)

    assert result.data[0].values != result.data[1].values


@pytest.mark.integration
@pytest.mark.anyio
async def test_embed_multiple_inputs_async(async_client: AsyncPinecone) -> None:
    """async embed() with multiple inputs returns one embedding per input."""
    inputs = [
        "The quick brown fox jumps over the lazy dog.",
        "Pinecone is a vector database.",
        "Machine learning powers modern AI.",
    ]
    result = await async_client.inference.embed(
        model="multilingual-e5-large",
        inputs=inputs,
        parameters={"input_type": "passage"},
    )
    assert result.model == "multilingual-e5-large"
    assert result.vector_type == "dense"
    assert len(result) == len(inputs)
    assert result.usage.total_tokens > 0

    # All embeddings have the same dimension
    dims = [len(emb.values) for emb in result.data]
    assert len(set(dims)) == 1, f"Expected uniform dimension, got: {dims}"

    # Embeddings are not all zeros
    for emb in result.data:
        assert any(v != 0.0 for v in emb.values)


@pytest.mark.integration
@pytest.mark.anyio
async def test_embed_iterable_and_indexable_async(async_client: AsyncPinecone) -> None:
    """async: EmbeddingsList supports len(), iteration, and integer indexing."""
    result = await async_client.inference.embed(
        model="multilingual-e5-large",
        inputs=["foo", "bar"],
        parameters={"input_type": "passage"},
    )
    assert len(result) == 2

    # Integer indexing
    first = result[0]
    second = result[1]
    assert first.values != second.values

    # Iteration
    items = list(result)
    assert len(items) == 2

    # String key access on the list
    assert result["model"] == "multilingual-e5-large"
    assert "model" in result


@pytest.mark.integration
@pytest.mark.anyio
async def test_embed_bare_string_auto_wrapped_async(async_client: AsyncPinecone) -> None:
    """async embed() with a single bare string is auto-wrapped and returns 1 embedding.

    Verifies unified-inf-0008: The embed operation accepts a single string input
    and wraps it into a list automatically.
    """
    # Pass a bare string — NOT wrapped in a list
    result = await async_client.inference.embed(
        model="multilingual-e5-large",
        inputs="What is a vector database?",
        parameters={"input_type": "passage", "truncate": "END"},
    )

    # The bare string should have been auto-wrapped → 1 embedding returned
    assert result.model == "multilingual-e5-large"
    assert result.vector_type == "dense"
    assert len(result) == 1, "Bare string should be auto-wrapped to a single-element batch"
    assert result.usage.total_tokens > 0

    embedding = result.data[0]
    assert isinstance(embedding.values, list)
    assert len(embedding.values) > 0
    assert all(isinstance(v, float) for v in embedding.values)
    assert embedding.vector_type == "dense"

    # The result should be identical to passing the same string in a list
    result_as_list = await async_client.inference.embed(
        model="multilingual-e5-large",
        inputs=["What is a vector database?"],
        parameters={"input_type": "passage", "truncate": "END"},
    )
    assert len(result_as_list) == 1
    # Same string → same embedding shape and (approximately) same values.
    # Element-wise tolerance absorbs server-side non-determinism (e.g. mixed
    # precision across replicas); the SDK contract being verified here is
    # that the bare-string path doesn't transform the input differently from
    # the single-element-list path, not that the model is bit-deterministic.
    assert len(result.data[0].values) == len(result_as_list.data[0].values)
    assert result.data[0].values == pytest.approx(result_as_list.data[0].values, abs=1e-3)


# ---------------------------------------------------------------------------
# rerank (async)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_rerank_basic_async(async_client: AsyncPinecone) -> None:
    """async rerank() returns a RerankResult with ranked documents sorted by score."""
    query = "What is machine learning?"
    documents = [
        "Paris is the capital of France.",
        "Machine learning is a subset of AI that uses algorithms to learn from data.",
        "Deep learning uses neural networks with many layers.",
    ]
    result = await async_client.inference.rerank(
        model="bge-reranker-v2-m3",
        query=query,
        documents=documents,
    )

    assert result.model == "bge-reranker-v2-m3"
    assert len(result.data) == len(documents)
    assert result.usage.rerank_units > 0

    # Results are sorted by score descending
    scores = [doc.score for doc in result.data]
    assert scores == sorted(scores, reverse=True), f"Scores not sorted descending: {scores}"

    # Each result has index, score, and document
    for ranked in result.data:
        assert isinstance(ranked.index, int)
        assert 0 <= ranked.index < len(documents)
        assert isinstance(ranked.score, float)
        assert ranked.document is not None
        assert "text" in ranked.document

    # The most relevant document should be the ML one (original index 1)
    assert result.data[0].index == 1


@pytest.mark.integration
@pytest.mark.anyio
async def test_rerank_with_top_n_async(async_client: AsyncPinecone) -> None:
    """async rerank() with top_n limits the number of returned results."""
    documents = [
        "Machine learning is a branch of artificial intelligence.",
        "Paris is known for the Eiffel Tower.",
        "Neural networks mimic the human brain.",
        "The ocean covers 71% of Earth's surface.",
    ]
    result = await async_client.inference.rerank(
        model="bge-reranker-v2-m3",
        query="Tell me about AI and neural networks",
        documents=documents,
        top_n=2,
    )

    assert result.model == "bge-reranker-v2-m3"
    assert len(result.data) == 2
    assert result.usage.rerank_units > 0

    scores = [doc.score for doc in result.data]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.integration
@pytest.mark.anyio
async def test_rerank_return_documents_false_async(async_client: AsyncPinecone) -> None:
    """async rerank() with return_documents=False omits document text from results."""
    documents = [
        "The sky is blue due to Rayleigh scattering.",
        "Vector databases store high-dimensional embeddings.",
    ]
    result = await async_client.inference.rerank(
        model="bge-reranker-v2-m3",
        query="How do vector databases work?",
        documents=documents,
        return_documents=False,
    )

    assert len(result.data) == len(documents)
    for ranked in result.data:
        assert ranked.document is None


@pytest.mark.integration
@pytest.mark.anyio
async def test_rerank_string_inputs_auto_wrapped_async(async_client: AsyncPinecone) -> None:
    """async rerank() with plain string documents auto-wraps them as {text: ...}.

    Verifies claim unified-inf-0013: both sync and async rerank() accept plain strings
    and the SDK transparently wraps them as {"text": <string>} before sending to the API.
    """
    documents = [
        "Machine learning enables computers to learn from experience.",
        "Cooking pasta requires boiling water.",
    ]
    result = await async_client.inference.rerank(
        model="bge-reranker-v2-m3",
        query="How does machine learning work?",
        documents=documents,
        return_documents=True,
    )

    assert len(result.data) == len(documents)
    for ranked in result.data:
        # SDK wraps strings as {"text": ...} before sending
        assert ranked.document is not None
        assert "text" in ranked.document


# ---------------------------------------------------------------------------
# list_models / get_model (async)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_list_models_returns_nonempty_list_async(async_client: AsyncPinecone) -> None:
    """async list_models() returns a ModelInfoList with at least one known model."""
    result = await async_client.inference.list_models()

    assert len(result) > 0
    names = result.names()
    assert isinstance(names, list)
    assert "multilingual-e5-large" in names


@pytest.mark.integration
@pytest.mark.anyio
async def test_list_models_supports_iteration_and_indexing_async(
    async_client: AsyncPinecone,
) -> None:
    """async: ModelInfoList supports len(), iteration, integer indexing, and 'models' key."""
    result = await async_client.inference.list_models()

    # Integer indexing
    first = result[0]
    assert isinstance(first.model, str)
    assert len(first.model) > 0

    # Iteration produces ModelInfo objects
    items = list(result)
    assert len(items) == len(result)
    assert all(hasattr(m, "model") for m in items)

    # String key access
    models_list = result["models"]
    assert isinstance(models_list, list)
    assert len(models_list) == len(result)


@pytest.mark.integration
@pytest.mark.anyio
async def test_list_models_filter_by_type_embed_async(async_client: AsyncPinecone) -> None:
    """async list_models(type='embed') returns only embed models."""
    result = await async_client.inference.list_models(type="embed")

    assert len(result) > 0
    for model_info in result:
        assert model_info.type == "embed"


@pytest.mark.integration
@pytest.mark.anyio
async def test_list_models_filter_by_type_rerank_async(async_client: AsyncPinecone) -> None:
    """async list_models(type='rerank') returns only rerank models."""
    result = await async_client.inference.list_models(type="rerank")

    assert len(result) > 0
    for model_info in result:
        assert model_info.type == "rerank"


@pytest.mark.integration
@pytest.mark.anyio
async def test_get_model_returns_model_info_async(async_client: AsyncPinecone) -> None:
    """async get_model() returns a ModelInfo with name, vector_type, and default_dimension."""
    model_info = await async_client.inference.get_model(model_name="multilingual-e5-large")

    # Required fields
    assert model_info.model == "multilingual-e5-large"
    assert model_info.type == "embed"

    # Alias property
    assert model_info.name == "multilingual-e5-large"

    # Embed-specific fields
    assert model_info.vector_type is not None
    assert model_info.default_dimension is not None
    assert isinstance(model_info.default_dimension, int)
    assert model_info.default_dimension > 0

    # Bracket access
    assert model_info["model"] == "multilingual-e5-large"
    assert model_info["name"] == "multilingual-e5-large"  # alias
    assert "model" in model_info
    assert "name" in model_info  # alias


@pytest.mark.integration
@pytest.mark.anyio
async def test_get_model_rerank_model_async(async_client: AsyncPinecone) -> None:
    """async get_model() works for rerank models; vector_type and default_dimension are None."""
    model_info = await async_client.inference.get_model(model_name="bge-reranker-v2-m3")

    assert model_info.model == "bge-reranker-v2-m3"
    assert model_info.type == "rerank"
    # Rerank models don't produce vectors
    assert model_info.vector_type is None
    assert model_info.default_dimension is None


# ---------------------------------------------------------------------------
# embed — input validation (async, client-side, no API call)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_embed_inputs_validation_rest_async(async_client: AsyncPinecone) -> None:
    """async embed() raises PineconeValueError for empty inputs and PineconeTypeError for wrong type.

    Verifies unified-inf-0016 (empty list rejected) and unified-inf-0017 (non-list rejected).
    These validations fire client-side in normalize_embed_inputs() before any HTTP request.
    Async variant of test_embed_inputs_validation_rest.
    """
    model = "multilingual-e5-large"

    # unified-inf-0016: empty list must be rejected
    with pytest.raises(PineconeValueError):
        await async_client.inference.embed(model=model, inputs=[])

    # unified-inf-0017: plain integer rejected
    with pytest.raises(PineconeTypeError):
        await async_client.inference.embed(model=model, inputs=42)  # type: ignore[arg-type]

    # unified-inf-0017: mixed list (string + integer) rejected
    with pytest.raises(PineconeTypeError):
        await async_client.inference.embed(model=model, inputs=["valid", 999])  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# rerank — input validation (async, client-side, no API call)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_rerank_documents_validation_rest_async(async_client: AsyncPinecone) -> None:
    """async rerank() raises PineconeValueError for empty docs, PineconeTypeError for non-list.

    Verifies unified-inf-0018 (empty list rejected) and unified-inf-0019 (non-list rejected).
    These validations happen client-side before any HTTP request is made.
    """
    # unified-inf-0018: empty list must be rejected
    with pytest.raises(PineconeValueError):
        await async_client.inference.rerank(
            model="bge-reranker-v2-m3",
            query="test query",
            documents=[],
        )

    # unified-inf-0019: non-list documents (plain string) must be rejected
    with pytest.raises(PineconeTypeError):
        await async_client.inference.rerank(
            model="bge-reranker-v2-m3",
            query="test query",
            documents="not a list",  # type: ignore[arg-type]
        )

    # unified-inf-0019: non-list documents (integer) must be rejected
    with pytest.raises(PineconeTypeError):
        await async_client.inference.rerank(
            model="bge-reranker-v2-m3",
            query="test query",
            documents=42,  # type: ignore[arg-type]
        )


@pytest.mark.integration
@pytest.mark.anyio
async def test_rerank_top_n_validation_async(async_client: AsyncPinecone) -> None:
    """async rerank() raises ValidationError for top_n < 1.

    Async counterpart of test_rerank_top_n_validation (sync). Both surfaces
    reject an out-of-range top_n client-side, so neither sends the value on to
    the backend.
    """
    with pytest.raises(ValidationError, match="top_n must be >= 1"):
        await async_client.inference.rerank(
            model="bge-reranker-v2-m3",
            query="test query",
            documents=["doc"],
            top_n=-1,
        )

    with pytest.raises(ValidationError, match="top_n must be >= 1"):
        await async_client.inference.rerank(
            model="bge-reranker-v2-m3",
            query="test query",
            documents=["doc"],
            top_n=0,
        )


@pytest.mark.integration
@pytest.mark.anyio
async def test_rerank_dict_documents_with_rank_fields_async(async_client: AsyncPinecone) -> None:
    """async rerank() with list-of-dict documents and custom rank_fields passes dicts through.

    Verifies unified-inf-0010 (dict document path) on the async transport: when documents
    are passed as dicts, they are sent to the API without wrapping, and rank_fields selects
    which field to rank on.
    """
    query = "machine learning and artificial intelligence"
    documents = [
        {"body": "The Eiffel Tower is a famous landmark in Paris, France.", "category": "travel"},
        {"body": "Machine learning is a branch of AI that learns from data.", "category": "tech"},
        {"body": "Neural networks are inspired by the human brain.", "category": "tech"},
        {"body": "The best pizza is from Naples, Italy.", "category": "food"},
    ]

    result = await async_client.inference.rerank(
        model="bge-reranker-v2-m3",
        query=query,
        documents=documents,
        rank_fields=["body"],  # non-default field — default is ["text"]
        return_documents=True,
    )

    assert result.model == "bge-reranker-v2-m3"
    assert len(result.data) == len(documents)
    assert result.usage.rerank_units > 0

    # Scores must be sorted descending
    scores = [item.score for item in result.data]
    assert scores == sorted(scores, reverse=True), f"Scores not sorted descending: {scores}"

    # Each ranked result must have an index into the original list and a document
    for ranked in result.data:
        assert isinstance(ranked.index, int)
        assert 0 <= ranked.index < len(documents)
        assert isinstance(ranked.score, float)
        assert ranked.document is not None

        # Dict documents are passed through as-is (not wrapped in {"text": ...})
        # so the original "body" and "category" keys must be present in the returned doc
        assert "body" in ranked.document, (
            f"'body' key missing from ranked.document: {ranked.document}"
        )
        assert "category" in ranked.document, (
            f"'category' key missing from ranked.document: {ranked.document}"
        )

    # The most relevant document should be one of the AI/ML ones (indexes 1 or 2)
    assert result.data[0].index in {1, 2}, (
        f"Expected AI/ML doc to rank first, got index {result.data[0].index}"
    )


# ---------------------------------------------------------------------------
# list_models — vector_type filter and invalid-value validation (async)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_list_models_filter_by_vector_type_and_invalid_values_async(
    async_client: AsyncPinecone,
) -> None:
    """async list_models() supports vector_type filter and rejects invalid type/vector_type values.

    Verifies unified-inf-0020 on the async transport: only "embed"/"rerank" accepted for type,
    only "dense"/"sparse" accepted for vector_type. Invalid values raise PineconeValueError
    client-side.
    """
    # vector_type="dense" → only dense embed models returned
    dense_result = await async_client.inference.list_models(vector_type="dense")
    assert len(dense_result) > 0
    for model_info in dense_result:
        assert model_info.vector_type == "dense", (
            f"Expected vector_type='dense', got {model_info.vector_type!r} for model {model_info.model!r}"
        )

    # vector_type="sparse" → only sparse embed models returned
    sparse_result = await async_client.inference.list_models(vector_type="sparse")
    assert len(sparse_result) > 0
    for model_info in sparse_result:
        assert model_info.vector_type == "sparse", (
            f"Expected vector_type='sparse', got {model_info.vector_type!r} for model {model_info.model!r}"
        )

    # invalid type → PineconeValueError, no HTTP call
    with pytest.raises(PineconeValueError):
        await async_client.inference.list_models(type="invalid_type")

    # invalid vector_type → PineconeValueError, no HTTP call
    with pytest.raises(PineconeValueError):
        await async_client.inference.list_models(vector_type="invalid_vector_type")


@pytest.mark.integration
@pytest.mark.anyio
async def test_list_models_rerank_vector_type_combination_async(
    async_client: AsyncPinecone,
) -> None:
    """async list_models(type='rerank', vector_type=...) raises ValidationError client-side.

    Async counterpart of test_list_models_rerank_vector_type_combination (sync).
    Verifies D6 fix: the combination type='rerank' + any vector_type is rejected
    before any HTTP request, matching the backend's InvalidArgument behavior.
    """
    with pytest.raises(ValidationError, match="vector_type is not supported"):
        await async_client.inference.list_models(type="rerank", vector_type="dense")

    with pytest.raises(ValidationError, match="vector_type is not supported"):
        await async_client.inference.list_models(type="rerank", vector_type="sparse")


# ---------------------------------------------------------------------------
# get_model full structure — description, supported_parameters, modality, etc. (async)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_get_model_full_structure_async(async_client: AsyncPinecone) -> None:
    """async get_model() populates ALL ModelInfo fields including description,
    supported_parameters, modality, max_sequence_length, max_batch_size,
    provider_name, and supported_metrics.

    Verifies unified-rs-0016: ModelInfo contains name, description, type,
    supported_parameters (list of ModelInfoSupportedParameter), optional vector
    type, optional default dimension, modality, optional max sequence length,
    optional max batch size, provider name, optional supported dimensions, and
    supported metrics.

    Area tag: model-info-full-structure-async
    Transport: rest-async
    """
    from pinecone.models.inference.models import ModelInfoSupportedParameter

    model_info = await async_client.inference.get_model(model_name="multilingual-e5-large")

    # --- description / short_description alias ---
    assert isinstance(model_info.short_description, str)
    assert len(model_info.short_description) > 0
    assert model_info.description == model_info.short_description
    assert model_info["description"] == model_info.short_description

    # --- supported_parameters: non-empty list of ModelInfoSupportedParameter ---
    assert isinstance(model_info.supported_parameters, list)
    assert len(model_info.supported_parameters) > 0, (
        "multilingual-e5-large should expose at least one supported parameter"
    )
    for param in model_info.supported_parameters:
        assert isinstance(param, ModelInfoSupportedParameter)
        assert isinstance(param.parameter, str) and len(param.parameter) > 0
        assert isinstance(param.type, str) and len(param.type) > 0
        assert isinstance(param.value_type, str) and len(param.value_type) > 0
        assert isinstance(param.required, bool)
        assert param.allowed_values is None or isinstance(param.allowed_values, list)
        assert param.min is None or isinstance(param.min, (int, float))
        assert param.max is None or isinstance(param.max, (int, float))
        assert param["parameter"] == param.parameter

    # --- modality: non-None string (text model) ---
    assert model_info.modality is not None
    assert isinstance(model_info.modality, str)
    assert len(model_info.modality) > 0

    # --- max_sequence_length: positive int or None ---
    if model_info.max_sequence_length is not None:
        assert isinstance(model_info.max_sequence_length, int)
        assert model_info.max_sequence_length > 0

    # --- max_batch_size: positive int or None ---
    if model_info.max_batch_size is not None:
        assert isinstance(model_info.max_batch_size, int)
        assert model_info.max_batch_size > 0

    # --- provider_name: non-None string ---
    assert model_info.provider_name is not None
    assert isinstance(model_info.provider_name, str)
    assert len(model_info.provider_name) > 0

    # --- supported_metrics: non-empty list of strings for an embed model ---
    assert model_info.supported_metrics is not None
    assert isinstance(model_info.supported_metrics, list)
    assert len(model_info.supported_metrics) > 0
    for metric in model_info.supported_metrics:
        assert isinstance(metric, str) and len(metric) > 0

    # --- supported_dimensions: list of positive ints or None ---
    if model_info.supported_dimensions is not None:
        assert isinstance(model_info.supported_dimensions, list)
        assert len(model_info.supported_dimensions) > 0
        for dim in model_info.supported_dimensions:
            assert isinstance(dim, int) and dim > 0


# ---------------------------------------------------------------------------
# model dict-like access error behaviors on real API responses (async)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_model_dict_keyerror_and_readonly_async(async_client: AsyncPinecone) -> None:
    """Dict-like access on model objects from real async API responses raises correct errors.

    Uses list_models() (no resources created or cleaned up) to obtain a real
    ModelInfo object deserialized from the live API, then verifies:

    - unified-model-0007: model["nonexistent_key"] raises KeyError
    - unified-model-0010: model["model"] = "x" raises TypeError (read-only)

    Also verifies positive behaviors work on real deserialized data:
    - model["model"] returns a string (existing key subscript access)
    - "model" in model → True
    - "totally_absent_key" in model → False
    """
    models = await async_client.inference.list_models()
    assert len(models) > 0, "list_models() must return at least one model for this test"

    model_info = models[0]

    # --- unified-model-0007: KeyError on non-existent key ---
    with pytest.raises(KeyError):
        _ = model_info["nonexistent_key_xyz"]

    # --- unified-model-0010: no item assignment (read-only) ---
    with pytest.raises(TypeError):
        model_info["model"] = "injected"  # type: ignore[index]

    # Positive: existing-key access and membership tests work on real data
    assert isinstance(model_info["model"], str)
    assert len(model_info["model"]) > 0
    assert "model" in model_info
    assert "totally_absent_key_xyz" not in model_info


# ---------------------------------------------------------------------------
# embed query vs passage (async) — input_type parameter parity
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_embed_query_vs_passage_async(async_client: AsyncPinecone) -> None:
    """async embed() with input_type='query' vs 'passage' produces different embeddings.

    Async counterpart of test_embed_query_vs_passage (sync). Verifies:
    - unified-inf-0013: Both sync and async variants of all inference operations
      are available with identical parameters.

    Passes the same text twice using different input_type values and asserts
    the resulting embedding vectors are not equal — confirming that the
    parameters dict is correctly forwarded to the API in the async path.
    No index resources are created.
    """
    text = ["vector databases enable semantic search"]

    query_result = await async_client.inference.embed(
        model="multilingual-e5-large",
        inputs=text,
        parameters={"input_type": "query"},
    )
    passage_result = await async_client.inference.embed(
        model="multilingual-e5-large",
        inputs=text,
        parameters={"input_type": "passage"},
    )

    assert query_result.model == "multilingual-e5-large"
    assert passage_result.model == "multilingual-e5-large"

    query_vec = query_result.data[0].values
    passage_vec = passage_result.data[0].values

    # Same dimension — both inputs are the same text / same model
    assert len(query_vec) == len(passage_vec)
    assert len(query_vec) > 0

    # Different values — input_type semantically changes the embedding
    assert query_vec != passage_vec, (
        "Expected input_type='query' and input_type='passage' to produce different "
        "embedding vectors for the same text, but they were identical."
    )
