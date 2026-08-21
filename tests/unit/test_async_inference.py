"""Unit tests for AsyncInference namespace — async embed, rerank, and model methods."""

from __future__ import annotations

import logging
from unittest.mock import patch

import httpx
import orjson
import pytest
import respx

from pinecone._internal.config import PineconeConfig
from pinecone._internal.constants import API_VERSION_HEADER
from pinecone.async_client.inference import AsyncInference
from pinecone.errors.exceptions import (
    ApiError,
    ForbiddenError,
    NotFoundError,
    PineconeValueError,
    ValidationError,
)
from pinecone.models.enums import EmbedModel, RerankModel
from pinecone.models.inference.embed import EmbeddingsList, SparseEmbedding
from pinecone.models.inference.model_list import ModelInfoList
from pinecone.models.inference.models import ModelInfo
from pinecone.models.inference.rerank import RerankResult
from tests.factories import (
    make_embed_response,
    make_error_response,
    make_model_info,
    make_model_list_response,
    make_rerank_response,
)

BASE_URL = "https://api.test.pinecone.io"


@pytest.fixture
def config() -> PineconeConfig:
    return PineconeConfig(api_key="test-key", host=BASE_URL)


@pytest.fixture
def inference(config: PineconeConfig) -> AsyncInference:
    return AsyncInference(config=config)


# ---------------------------------------------------------------------------
# embed()
# ---------------------------------------------------------------------------


@respx.mock
@pytest.mark.asyncio
async def test_async_embed_returns_embeddings_list_with_model_and_usage(
    inference: AsyncInference,
) -> None:
    """embed() returns an EmbeddingsList whose model, data, and usage fields reflect the response body."""
    route = respx.post(f"{BASE_URL}/embed").mock(
        return_value=httpx.Response(200, json=make_embed_response()),
    )

    result = await inference.embed("multilingual-e5-large", ["hello"])

    assert isinstance(result, EmbeddingsList)
    assert result.model == "multilingual-e5-large"
    assert len(result.data) == 1
    # Values + token count come from make_embed_response() factory.
    assert result.data[0].values == [0.1, 0.2, 0.3]
    assert result.usage.total_tokens == 205
    assert route.called


@respx.mock
@pytest.mark.asyncio
async def test_async_embed_wraps_bare_string_input_as_text_dict(inference: AsyncInference) -> None:
    """A bare string input is normalized to a one-element list of {"text": ...} dicts in the request body."""
    route = respx.post(f"{BASE_URL}/embed").mock(
        return_value=httpx.Response(200, json=make_embed_response()),
    )

    result = await inference.embed("multilingual-e5-large", "hello")

    assert isinstance(result, EmbeddingsList)
    import orjson

    body = orjson.loads(route.calls[0].request.content)
    assert body["inputs"] == [{"text": "hello"}]


@pytest.mark.asyncio
async def test_async_embed_empty_raises(inference: AsyncInference) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        await inference.embed("multilingual-e5-large", [])


@respx.mock
@pytest.mark.asyncio
async def test_async_embed_dict_inputs(inference: AsyncInference) -> None:
    route = respx.post(f"{BASE_URL}/embed").mock(
        return_value=httpx.Response(200, json=make_embed_response()),
    )

    result = await inference.embed("multilingual-e5-large", [{"text": "hello"}])

    assert isinstance(result, EmbeddingsList)
    body = orjson.loads(route.calls[0].request.content)
    assert body["inputs"] == [{"text": "hello"}]


@pytest.mark.asyncio
async def test_async_embed_empty_model_raises(inference: AsyncInference) -> None:
    with pytest.raises(ValidationError, match="non-empty"):
        await inference.embed("", ["hello"])


@respx.mock
@pytest.mark.asyncio
async def test_async_embed_forwards_parameters_to_request_body(inference: AsyncInference) -> None:
    route = respx.post(f"{BASE_URL}/embed").mock(
        return_value=httpx.Response(200, json=make_embed_response()),
    )

    await inference.embed(
        "multilingual-e5-large",
        ["hello"],
        parameters={"input_type": "passage", "truncate": "END"},
    )

    body = orjson.loads(route.calls[0].request.content)
    assert body["parameters"] == {"input_type": "passage", "truncate": "END"}


@respx.mock
@pytest.mark.asyncio
async def test_async_embed_accepts_embed_model_enum(inference: AsyncInference) -> None:
    route = respx.post(f"{BASE_URL}/embed").mock(
        return_value=httpx.Response(200, json=make_embed_response()),
    )

    result = await inference.embed(EmbedModel.Multilingual_E5_Large, ["hello"])

    assert isinstance(result, EmbeddingsList)
    body = orjson.loads(route.calls.last.request.content)
    assert body["model"] == "multilingual-e5-large"


# ---------------------------------------------------------------------------
# rerank()
# ---------------------------------------------------------------------------


@respx.mock
@pytest.mark.asyncio
async def test_async_rerank_returns_rerank_result_with_model_and_usage(
    inference: AsyncInference,
) -> None:
    """rerank() returns a RerankResult whose model, data, and usage fields reflect the response body."""
    route = respx.post(f"{BASE_URL}/rerank").mock(
        return_value=httpx.Response(200, json=make_rerank_response()),
    )

    result = await inference.rerank(
        model="bge-reranker-v2-m3",
        query="Tell me about tech companies",
        documents=[{"text": "Acme Inc. revolutionized tech."}],
        top_n=2,
    )

    assert isinstance(result, RerankResult)
    assert result.model == "bge-reranker-v2-m3"
    assert len(result.data) == 2
    # Score + unit count come from make_rerank_response() factory.
    assert result.data[0].score == 0.95
    assert result.usage.rerank_units == 1
    assert route.called


@respx.mock
@pytest.mark.asyncio
@pytest.mark.parametrize("top_n", [0, -1])
async def test_async_rerank_top_n_below_one_raises_before_any_request(
    inference: AsyncInference, top_n: int
) -> None:
    route = respx.post(f"{BASE_URL}/rerank").mock(
        return_value=httpx.Response(200, json=make_rerank_response()),
    )

    with pytest.raises(PineconeValueError, match="top_n must be >= 1") as exc_info:
        await inference.rerank(
            model="bge-reranker-v2-m3",
            query="test query",
            documents=["doc"],
            top_n=top_n,
        )

    assert type(exc_info.value) is PineconeValueError
    assert not route.called


@pytest.mark.asyncio
async def test_async_rerank_empty_docs_raises(inference: AsyncInference) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        await inference.rerank(
            model="bge-reranker-v2-m3",
            query="test query",
            documents=[],
        )


@respx.mock
@pytest.mark.asyncio
async def test_async_rerank_wraps_bare_string_documents_as_text_dicts(
    inference: AsyncInference,
) -> None:
    route = respx.post(f"{BASE_URL}/rerank").mock(
        return_value=httpx.Response(200, json=make_rerank_response()),
    )

    await inference.rerank(
        model="bge-reranker-v2-m3",
        query="test query",
        documents=["doc1", "doc2"],
    )

    body = orjson.loads(route.calls[0].request.content)
    assert body["documents"] == [{"text": "doc1"}, {"text": "doc2"}]


@respx.mock
@pytest.mark.asyncio
async def test_async_rerank_defaults_rank_fields_to_text_when_omitted(
    inference: AsyncInference,
) -> None:
    route = respx.post(f"{BASE_URL}/rerank").mock(
        return_value=httpx.Response(200, json=make_rerank_response()),
    )

    await inference.rerank(
        model="bge-reranker-v2-m3",
        query="test query",
        documents=["doc1"],
    )

    body = orjson.loads(route.calls[0].request.content)
    assert body["rank_fields"] == ["text"]


@pytest.mark.asyncio
async def test_async_rerank_empty_model_raises(inference: AsyncInference) -> None:
    with pytest.raises(ValidationError, match="non-empty"):
        await inference.rerank(
            model="",
            query="test query",
            documents=["doc1"],
        )


@pytest.mark.asyncio
async def test_async_rerank_empty_query_raises(inference: AsyncInference) -> None:
    with pytest.raises(ValidationError, match="non-empty"):
        await inference.rerank(
            model="bge-reranker-v2-m3",
            query="",
            documents=["doc1"],
        )


@pytest.mark.asyncio
async def test_async_rerank_mixed_types_raises(inference: AsyncInference) -> None:
    with pytest.raises(TypeError, match="string or mapping"):
        await inference.rerank(
            model="bge-reranker-v2-m3",
            query="test query",
            documents=["a string", 123],  # type: ignore[list-item]
        )


@pytest.mark.asyncio
async def test_async_rerank_non_list_documents_raises(inference: AsyncInference) -> None:
    with pytest.raises(TypeError, match="Sequence"):
        await inference.rerank(
            model="bge-reranker-v2-m3",
            query="test query",
            documents="not a list",  # type: ignore[arg-type]
        )


@respx.mock
@pytest.mark.asyncio
async def test_async_rerank_tuple_documents_accepted(inference: AsyncInference) -> None:
    respx.post(f"{BASE_URL}/rerank").mock(
        return_value=httpx.Response(200, json=make_rerank_response()),
    )

    result = await inference.rerank(
        model="bge-reranker-v2-m3",
        query="test query",
        documents=("a", "b"),  # type: ignore[arg-type]
    )
    assert isinstance(result, RerankResult)


@respx.mock
@pytest.mark.asyncio
async def test_async_rerank_accepts_rerank_model_enum(inference: AsyncInference) -> None:
    route = respx.post(f"{BASE_URL}/rerank").mock(
        return_value=httpx.Response(200, json=make_rerank_response()),
    )

    result = await inference.rerank(
        model=RerankModel.Bge_Reranker_V2_M3,
        query="test query",
        documents=["doc1"],
    )

    assert isinstance(result, RerankResult)
    body = orjson.loads(route.calls.last.request.content)
    assert body["model"] == "bge-reranker-v2-m3"


# ---------------------------------------------------------------------------
# list_models()
# ---------------------------------------------------------------------------


@respx.mock
@pytest.mark.asyncio
async def test_async_list_models(inference: AsyncInference) -> None:
    route = respx.get(f"{BASE_URL}/models").mock(
        return_value=httpx.Response(200, json=make_model_list_response()),
    )

    result = await inference.list_models()

    assert isinstance(result, ModelInfoList)
    assert len(result) == 2
    assert result.names() == ["multilingual-e5-large", "bge-reranker-v2-m3"]
    assert route.called


@pytest.mark.asyncio
async def test_async_list_models_invalid_type_raises(inference: AsyncInference) -> None:
    with pytest.raises(ValidationError, match="must be one of"):
        await inference.list_models(type="invalid")


@pytest.mark.asyncio
async def test_async_list_models_invalid_vector_type_raises(inference: AsyncInference) -> None:
    with pytest.raises(ValidationError, match="must be one of"):
        await inference.list_models(vector_type="invalid")


@pytest.mark.asyncio
async def test_async_list_models_rerank_vector_type_raises(inference: AsyncInference) -> None:
    with pytest.raises(ValidationError, match="vector_type is not supported"):
        await inference.list_models(type="rerank", vector_type="dense")


@respx.mock
@pytest.mark.asyncio
async def test_async_list_models_filter_by_type(inference: AsyncInference) -> None:
    route = respx.get(f"{BASE_URL}/models").mock(
        return_value=httpx.Response(200, json=make_model_list_response()),
    )

    await inference.list_models(type="embed")

    assert route.called
    request = route.calls[0].request
    assert request.url.params["type"] == "embed"


@respx.mock
@pytest.mark.asyncio
async def test_async_list_models_filter_by_vector_type(inference: AsyncInference) -> None:
    route = respx.get(f"{BASE_URL}/models").mock(
        return_value=httpx.Response(200, json=make_model_list_response()),
    )

    await inference.list_models(vector_type="sparse")

    assert route.called
    request = route.calls[0].request
    assert request.url.params["vector_type"] == "sparse"


@respx.mock
@pytest.mark.asyncio
async def test_async_list_models_both_filters(inference: AsyncInference) -> None:
    route = respx.get(f"{BASE_URL}/models").mock(
        return_value=httpx.Response(200, json=make_model_list_response()),
    )

    await inference.list_models(type="embed", vector_type="sparse")

    request = route.calls[0].request
    assert request.url.params["type"] == "embed"
    assert request.url.params["vector_type"] == "sparse"


# ---------------------------------------------------------------------------
# get_model()
# ---------------------------------------------------------------------------


@respx.mock
@pytest.mark.asyncio
async def test_async_get_model(inference: AsyncInference) -> None:
    route = respx.get(f"{BASE_URL}/models/multilingual-e5-large").mock(
        return_value=httpx.Response(200, json=make_model_info()),
    )

    result = await inference.get_model(model="multilingual-e5-large")

    assert isinstance(result, ModelInfo)
    assert result.model == "multilingual-e5-large"
    assert result.type == "embed"
    assert route.called


@pytest.mark.asyncio
async def test_async_get_model_empty_name_raises(inference: AsyncInference) -> None:
    with pytest.raises(ValidationError, match="non-empty"):
        await inference.get_model(model="")


@respx.mock
@pytest.mark.asyncio
async def test_async_get_model_legacy_model_name_kwarg(inference: AsyncInference) -> None:
    route = respx.get(f"{BASE_URL}/models/multilingual-e5-large").mock(
        return_value=httpx.Response(200, json=make_model_info()),
    )

    result = await inference.get_model(model_name="multilingual-e5-large")

    assert isinstance(result, ModelInfo)
    assert result.model == "multilingual-e5-large"
    assert route.called


@pytest.mark.asyncio
async def test_async_get_model_conflict_raises(inference: AsyncInference) -> None:
    with pytest.raises(ValidationError, match="model= or model_name="):
        await inference.get_model(model="foo", model_name="bar")


@pytest.mark.asyncio
async def test_async_get_model_unexpected_kwarg_raises(inference: AsyncInference) -> None:
    with pytest.raises(TypeError, match="unexpected keyword arguments"):
        await inference.get_model(model_alias="foo")


# ---------------------------------------------------------------------------
# API version header
# ---------------------------------------------------------------------------


@respx.mock
@pytest.mark.asyncio
async def test_async_all_operations_send_the_2026_07_version_header(
    inference: AsyncInference,
) -> None:
    embed = respx.post(f"{BASE_URL}/embed").mock(
        return_value=httpx.Response(200, json=make_embed_response()),
    )
    rerank = respx.post(f"{BASE_URL}/rerank").mock(
        return_value=httpx.Response(200, json=make_rerank_response()),
    )
    list_models = respx.get(f"{BASE_URL}/models").mock(
        return_value=httpx.Response(200, json=make_model_list_response()),
    )
    get_model = respx.get(f"{BASE_URL}/models/multilingual-e5-large").mock(
        return_value=httpx.Response(200, json=make_model_info()),
    )

    await inference.embed("multilingual-e5-large", ["hello"])
    await inference.rerank("bge-reranker-v2-m3", "q", ["a", "b"])
    await inference.list_models()
    await inference.get_model(model="multilingual-e5-large")

    for route in (embed, rerank, list_models, get_model):
        assert route.calls.last.request.headers[API_VERSION_HEADER] == "2026-07"


@respx.mock
@pytest.mark.asyncio
async def test_async_model_resource_sends_the_2026_07_version_header(
    inference: AsyncInference,
) -> None:
    list_models = respx.get(f"{BASE_URL}/models").mock(
        return_value=httpx.Response(200, json=make_model_list_response()),
    )
    get_model = respx.get(f"{BASE_URL}/models/multilingual-e5-large").mock(
        return_value=httpx.Response(200, json=make_model_info()),
    )

    await inference.model.list()
    await inference.model.get("multilingual-e5-large")

    for route in (list_models, get_model):
        assert route.calls.last.request.headers[API_VERSION_HEADER] == "2026-07"


# ---------------------------------------------------------------------------
# pinecone.inference.inference_asyncio
# ---------------------------------------------------------------------------


@respx.mock
@pytest.mark.asyncio
async def test_inference_asyncio_shim_resolves_and_sends_2026_07(
    config: PineconeConfig,
) -> None:
    from pinecone.inference.inference_asyncio import AsyncioInference

    assert AsyncioInference is AsyncInference

    route = respx.post(f"{BASE_URL}/embed").mock(
        return_value=httpx.Response(200, json=make_embed_response()),
    )
    shimmed = AsyncioInference(config=config)
    try:
        await shimmed.embed("multilingual-e5-large", ["hello"])
    finally:
        await shimmed.close()

    assert route.calls.last.request.headers[API_VERSION_HEADER] == "2026-07"


# ---------------------------------------------------------------------------
# Lazy property on AsyncPinecone
# ---------------------------------------------------------------------------


def test_async_inference_lazy_property() -> None:
    """Accessing .inference twice returns the same instance."""
    with patch.dict("os.environ", {"PINECONE_API_KEY": "test-key"}):
        from pinecone.async_client.pinecone import AsyncPinecone

        pc = AsyncPinecone(api_key="test-key")
        first = pc.inference
        second = pc.inference
        assert first is second


# ---------------------------------------------------------------------------
# repr and class attributes
# ---------------------------------------------------------------------------


def test_async_inference_repr(inference: AsyncInference) -> None:
    assert repr(inference) == "AsyncInference()"


def test_async_inference_class_attributes() -> None:
    """AsyncInference exposes EmbedModel and RerankModel as class attributes."""
    assert AsyncInference.EmbedModel is EmbedModel
    assert AsyncInference.RerankModel is RerankModel


# ---------------------------------------------------------------------------
# model cached_property / AsyncModelResource
# ---------------------------------------------------------------------------


def test_async_inference_model_cached_property(config: PineconeConfig) -> None:
    """Accessing .model twice returns the same AsyncModelResource instance."""
    inference = AsyncInference(config=config)
    first = inference.model
    second = inference.model
    assert first is second


@respx.mock
@pytest.mark.asyncio
async def test_async_inference_model_list(inference: AsyncInference) -> None:
    route = respx.get(f"{BASE_URL}/models").mock(
        return_value=httpx.Response(200, json=make_model_list_response()),
    )

    result = await inference.model.list()

    assert isinstance(result, ModelInfoList)
    assert len(result) == 2
    assert route.called


@respx.mock
@pytest.mark.asyncio
async def test_async_inference_model_list_with_filters(inference: AsyncInference) -> None:
    route = respx.get(f"{BASE_URL}/models").mock(
        return_value=httpx.Response(200, json=make_model_list_response()),
    )

    await inference.model.list(type="embed", vector_type="dense")

    request = route.calls[0].request
    assert request.url.params["type"] == "embed"
    assert request.url.params["vector_type"] == "dense"


@respx.mock
@pytest.mark.asyncio
async def test_async_inference_model_get(inference: AsyncInference) -> None:
    route = respx.get(f"{BASE_URL}/models/multilingual-e5-large").mock(
        return_value=httpx.Response(200, json=make_model_info()),
    )

    result = await inference.model.get("multilingual-e5-large")

    assert isinstance(result, ModelInfo)
    assert result.model == "multilingual-e5-large"
    assert route.called


@respx.mock
@pytest.mark.asyncio
async def test_async_inference_model_get_legacy_model_name_kwarg(inference: AsyncInference) -> None:
    route = respx.get(f"{BASE_URL}/models/multilingual-e5-large").mock(
        return_value=httpx.Response(200, json=make_model_info()),
    )

    result = await inference.model.get(model_name="multilingual-e5-large")

    assert isinstance(result, ModelInfo)
    assert result.model == "multilingual-e5-large"
    assert route.called


@pytest.mark.asyncio
async def test_async_inference_model_get_conflict_raises(inference: AsyncInference) -> None:
    with pytest.raises(ValidationError, match="model= or model_name="):
        await inference.model.get(model="foo", model_name="bar")


@pytest.mark.asyncio
async def test_async_inference_model_get_unexpected_kwarg_raises(inference: AsyncInference) -> None:
    with pytest.raises(TypeError, match="unexpected keyword arguments"):
        await inference.model.get(model_alias="foo")


# ---------------------------------------------------------------------------
# 2026-07 inference alignment (#257) — async mirror of tests/unit/test_inference.py
# ---------------------------------------------------------------------------

_NEW_2026_07_EMBED_MODELS = [
    (EmbedModel.Llama_Text_Embed_V2, "llama-text-embed-v2"),
    (EmbedModel.Pinecone_Sparse_Multilingual_V0, "pinecone-sparse-multilingual-v0"),
]


@respx.mock
@pytest.mark.asyncio
@pytest.mark.parametrize(("member", "wire_value"), _NEW_2026_07_EMBED_MODELS)
async def test_async_embed_serializes_2026_07_embed_model_ids(
    inference: AsyncInference, member: EmbedModel, wire_value: str
) -> None:
    """The two model ids added for 2026-07 reach the wire and round-trip back.

    Mirrors the sync case.
    """
    route = respx.post(f"{BASE_URL}/embed").mock(
        return_value=httpx.Response(200, json=make_embed_response(model=wire_value)),
    )

    result = await inference.embed(member, ["hello"])

    body = orjson.loads(route.calls[0].request.content)
    assert body["model"] == wire_value
    assert result.model == wire_value
    assert route.calls.last.request.headers[API_VERSION_HEADER] == "2026-07"


@respx.mock
@pytest.mark.asyncio
async def test_async_embed_deserializes_sparse_response_without_sparse_tokens(
    inference: AsyncInference,
) -> None:
    """A sparse embed response decodes with sparse_tokens absent."""
    respx.post(f"{BASE_URL}/embed").mock(
        return_value=httpx.Response(
            200,
            json=make_embed_response(
                model="pinecone-sparse-multilingual-v0",
                vector_type="sparse",
                data=[
                    {
                        "sparse_values": [0.5, 0.25],
                        "sparse_indices": [10, 20],
                        "vector_type": "sparse",
                    }
                ],
            ),
        ),
    )

    result = await inference.embed(EmbedModel.Pinecone_Sparse_Multilingual_V0, ["hello"])

    assert result.vector_type == "sparse"
    embedding = result.data[0]
    assert isinstance(embedding, SparseEmbedding)
    assert embedding.sparse_indices == [10, 20]
    assert embedding.sparse_values == [0.5, 0.25]
    assert embedding.sparse_tokens is None


@respx.mock
@pytest.mark.asyncio
async def test_async_embed_deserializes_sparse_response_with_sparse_tokens(
    inference: AsyncInference,
) -> None:
    respx.post(f"{BASE_URL}/embed").mock(
        return_value=httpx.Response(
            200,
            json=make_embed_response(
                model="pinecone-sparse-multilingual-v0",
                vector_type="sparse",
                data=[
                    {
                        "sparse_values": [0.5],
                        "sparse_indices": [10],
                        "sparse_tokens": ["hello"],
                        "vector_type": "sparse",
                    }
                ],
            ),
        ),
    )

    result = await inference.embed(EmbedModel.Pinecone_Sparse_Multilingual_V0, ["hello"])

    embedding = result.data[0]
    assert isinstance(embedding, SparseEmbedding)
    assert embedding.sparse_tokens == ["hello"]


@respx.mock
@pytest.mark.asyncio
async def test_async_embed_deserializes_dense_response_for_llama_text_embed_v2(
    inference: AsyncInference,
) -> None:
    respx.post(f"{BASE_URL}/embed").mock(
        return_value=httpx.Response(200, json=make_embed_response(model="llama-text-embed-v2")),
    )

    result = await inference.embed(EmbedModel.Llama_Text_Embed_V2, ["hello"])

    assert result.vector_type == "dense"
    assert result.data[0].values == [0.1, 0.2, 0.3]


@respx.mock
@pytest.mark.asyncio
async def test_async_embed_unknown_parameters_key_surfaces_plain_text_422(
    inference: AsyncInference,
) -> None:
    """An unknown ``parameters`` key is rejected by the server, not by the SDK."""
    plain_text = (
        "Failed to deserialize the JSON body into the target type: "
        "parameters: unknown field `not_a_real_param`, expected one of "
        "`input_type`, `truncate`, `dimension`, `return_tokens`, "
        "`max_tokens_per_sequence` at line 1 column 74"
    )
    route = respx.post(f"{BASE_URL}/embed").mock(
        return_value=httpx.Response(422, text=plain_text),
    )

    with pytest.raises(ApiError) as excinfo:
        await inference.embed(
            EmbedModel.Llama_Text_Embed_V2, ["hello"], parameters={"not_a_real_param": 1}
        )

    assert excinfo.value.status_code == 422
    assert excinfo.value.message == plain_text
    assert excinfo.value.error_code is None
    body = orjson.loads(route.calls[0].request.content)
    assert body["parameters"] == {"not_a_real_param": 1}


@respx.mock
@pytest.mark.asyncio
async def test_async_embed_project_not_authorized_is_masked_as_404(
    inference: AsyncInference,
) -> None:
    """embed reports an authorization failure as 404, unlike rerank's 403."""
    message = "Model 'llama-text-embed-v2' not found"
    respx.post(f"{BASE_URL}/embed").mock(
        return_value=httpx.Response(404, json=make_error_response(404, message)),
    )

    with pytest.raises(NotFoundError) as excinfo:
        await inference.embed(EmbedModel.Llama_Text_Embed_V2, ["hello"])

    assert excinfo.value.status_code == 404
    assert excinfo.value.message == message


@respx.mock
@pytest.mark.asyncio
async def test_async_rerank_project_not_authorized_surfaces_403(
    inference: AsyncInference,
) -> None:
    message = "Project is not authorized to use model bge-reranker-v2-m3"
    respx.post(f"{BASE_URL}/rerank").mock(
        return_value=httpx.Response(403, json=make_error_response(403, message)),
    )

    with pytest.raises(ForbiddenError) as excinfo:
        await inference.rerank(model=RerankModel.Bge_Reranker_V2_M3, query="q", documents=["doc"])

    assert excinfo.value.status_code == 403
    assert excinfo.value.message == message


@respx.mock
@pytest.mark.asyncio
async def test_async_rerank_deprecated_pinecone_rerank_v0_surfaces_403(
    inference: AsyncInference,
) -> None:
    """pinecone-rerank-v0 is deprecated server-side and 403s off the allow-list."""
    message = (
        "Model pinecone-rerank-v0 has been deprecated, please use a different "
        "reranking model from (https://docs.pinecone.io/models)"
    )
    respx.post(f"{BASE_URL}/rerank").mock(
        return_value=httpx.Response(403, json=make_error_response(403, message)),
    )

    with pytest.raises(ForbiddenError) as excinfo:
        await inference.rerank(model=RerankModel.Pinecone_Rerank_V0, query="q", documents=["doc"])

    assert excinfo.value.status_code == 403
    assert "has been deprecated" in excinfo.value.message


@respx.mock
@pytest.mark.asyncio
async def test_async_rerank_response_model_may_differ_from_requested_model(
    inference: AsyncInference,
) -> None:
    """cohere-rerank-3.5 can be served by cohere-rerank-4-fast."""
    respx.post(f"{BASE_URL}/rerank").mock(
        return_value=httpx.Response(200, json=make_rerank_response(model="cohere-rerank-4-fast")),
    )

    result = await inference.rerank(
        model=RerankModel.Cohere_Rerank_3_5, query="q", documents=["doc"]
    )

    assert result.model == "cohere-rerank-4-fast"


@respx.mock
@pytest.mark.asyncio
async def test_async_rerank_unknown_parameters_key_surfaces_plain_text_422(
    inference: AsyncInference,
) -> None:
    plain_text = (
        "Failed to deserialize the JSON body into the target type: "
        "parameters: unknown field `not_a_real_param`, expected one of "
        "`truncate`, `max_chunks_per_doc`, `max_tokens_per_doc` at line 1 column 90"
    )
    respx.post(f"{BASE_URL}/rerank").mock(
        return_value=httpx.Response(422, text=plain_text),
    )

    with pytest.raises(ApiError) as excinfo:
        await inference.rerank(
            model=RerankModel.Bge_Reranker_V2_M3,
            query="q",
            documents=["doc"],
            parameters={"not_a_real_param": 1},
        )

    assert excinfo.value.status_code == 422
    assert excinfo.value.message == plain_text
    assert excinfo.value.error_code is None


# ---------------------------------------------------------------------------
# Enum members serialize as model ids, not as "EmbedModel.X" (#296)
# async mirror of tests/unit/test_inference.py
# ---------------------------------------------------------------------------


@respx.mock
@pytest.mark.asyncio
@pytest.mark.parametrize("member", list(EmbedModel), ids=lambda m: m.name)
async def test_async_embed_sends_the_model_id_for_every_embed_model_member(
    inference: AsyncInference, member: EmbedModel
) -> None:
    """Parametrized over the whole enum so a member added later is covered too."""
    route = respx.post(f"{BASE_URL}/embed").mock(
        return_value=httpx.Response(200, json=make_embed_response(model=member.value)),
    )

    result = await inference.embed(member, ["hello"])

    request = route.calls.last.request
    assert orjson.loads(request.content) == {
        "model": member.value,
        "inputs": [{"text": "hello"}],
    }
    assert b"EmbedModel." not in request.content
    assert result.model == member.value
    assert request.headers[API_VERSION_HEADER] == "2026-07"


@respx.mock
@pytest.mark.asyncio
@pytest.mark.parametrize("member", list(RerankModel), ids=lambda m: m.name)
async def test_async_rerank_sends_the_model_id_for_every_rerank_model_member(
    inference: AsyncInference, member: RerankModel
) -> None:
    route = respx.post(f"{BASE_URL}/rerank").mock(
        return_value=httpx.Response(200, json=make_rerank_response(model=member.value)),
    )

    result = await inference.rerank(model=member, query="q", documents=["doc"])

    request = route.calls.last.request
    assert orjson.loads(request.content) == {
        "model": member.value,
        "query": "q",
        "documents": [{"text": "doc"}],
        "rank_fields": ["text"],
        "return_documents": True,
    }
    assert b"RerankModel." not in request.content
    assert result.model == member.value
    assert request.headers[API_VERSION_HEADER] == "2026-07"


@respx.mock
@pytest.mark.asyncio
async def test_async_embed_still_sends_a_plain_model_string_unchanged(
    inference: AsyncInference,
) -> None:
    """Both parameters accept a plain string, including an id this SDK has no member for."""
    route = respx.post(f"{BASE_URL}/embed").mock(
        return_value=httpx.Response(200, json=make_embed_response(model="future-embed-model-v9")),
    )

    await inference.embed("future-embed-model-v9", ["hello"])

    assert orjson.loads(route.calls.last.request.content)["model"] == "future-embed-model-v9"


@respx.mock
@pytest.mark.asyncio
async def test_async_rerank_still_sends_a_plain_model_string_unchanged(
    inference: AsyncInference,
) -> None:
    route = respx.post(f"{BASE_URL}/rerank").mock(
        return_value=httpx.Response(200, json=make_rerank_response(model="future-rerank-model-v9")),
    )

    await inference.rerank(model="future-rerank-model-v9", query="q", documents=["doc"])

    assert orjson.loads(route.calls.last.request.content)["model"] == "future-rerank-model-v9"


@respx.mock
@pytest.mark.asyncio
async def test_async_mangled_model_id_this_release_stopped_sending_still_404s(
    inference: AsyncInference,
) -> None:
    """What callers saw before this fix, reproduced by sending the old string by hand."""
    old_wire_value = "EmbedModel.Multilingual_E5_Large"
    message = f"Model '{old_wire_value}' not found."
    route = respx.post(f"{BASE_URL}/embed").mock(
        return_value=httpx.Response(404, json=make_error_response(404, message)),
    )

    with pytest.raises(NotFoundError) as excinfo:
        await inference.embed(old_wire_value, ["hello"])

    assert orjson.loads(route.calls.last.request.content)["model"] == old_wire_value
    assert excinfo.value.message == message


@respx.mock
@pytest.mark.asyncio
async def test_async_logs_name_the_model_id_not_the_enum_member(
    inference: AsyncInference, caplog: pytest.LogCaptureFixture
) -> None:
    """An operator grepping the logs for a model id has to find it there too."""
    respx.post(f"{BASE_URL}/embed").mock(
        return_value=httpx.Response(200, json=make_embed_response()),
    )
    respx.post(f"{BASE_URL}/rerank").mock(
        return_value=httpx.Response(200, json=make_rerank_response()),
    )

    with caplog.at_level(logging.INFO, logger="pinecone.async_client.inference"):
        await inference.embed(EmbedModel.Multilingual_E5_Large, ["hello"])
        await inference.rerank(model=RerankModel.Bge_Reranker_V2_M3, query="q", documents=["doc"])

    assert "'multilingual-e5-large'" in caplog.text
    assert "'bge-reranker-v2-m3'" in caplog.text
    assert "EmbedModel." not in caplog.text
    assert "RerankModel." not in caplog.text
