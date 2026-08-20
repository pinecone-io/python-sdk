"""Unit tests for AsyncInference namespace — async embed, rerank, and model methods."""

from __future__ import annotations

from unittest.mock import patch

import httpx
import orjson
import pytest
import respx

from pinecone._internal.config import PineconeConfig
from pinecone._internal.constants import API_VERSION_HEADER
from pinecone.async_client.inference import AsyncInference
from pinecone.errors.exceptions import ValidationError
from pinecone.models.enums import EmbedModel, RerankModel
from pinecone.models.inference.embed import EmbeddingsList
from pinecone.models.inference.model_list import ModelInfoList
from pinecone.models.inference.models import ModelInfo
from pinecone.models.inference.rerank import RerankResult
from tests.factories import (
    make_embed_response,
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
    respx.post(f"{BASE_URL}/embed").mock(
        return_value=httpx.Response(200, json=make_embed_response()),
    )

    result = await inference.embed(EmbedModel.Multilingual_E5_Large, ["hello"])

    assert isinstance(result, EmbeddingsList)


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
    respx.post(f"{BASE_URL}/rerank").mock(
        return_value=httpx.Response(200, json=make_rerank_response()),
    )

    result = await inference.rerank(
        model=RerankModel.Bge_Reranker_V2_M3,
        query="test query",
        documents=["doc1"],
    )

    assert isinstance(result, RerankResult)


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
