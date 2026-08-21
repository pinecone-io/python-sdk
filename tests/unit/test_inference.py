"""Unit tests for Inference namespace — embed and rerank methods."""

from __future__ import annotations

import logging
from unittest.mock import patch

import httpx
import orjson
import pytest
import respx

from pinecone._internal.config import PineconeConfig
from pinecone._internal.constants import API_VERSION_HEADER
from pinecone.client.inference import Inference
from pinecone.errors.exceptions import (
    ApiError,
    ForbiddenError,
    NotFoundError,
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
def inference(config: PineconeConfig) -> Inference:
    return Inference(config=config)


# ---------------------------------------------------------------------------
# embed()
# ---------------------------------------------------------------------------


@respx.mock
def test_embed_returns_embeddings_list_with_model_and_usage(inference: Inference) -> None:
    """embed() returns an EmbeddingsList whose model, data, and usage fields reflect the response body."""
    route = respx.post(f"{BASE_URL}/embed").mock(
        return_value=httpx.Response(200, json=make_embed_response()),
    )

    result = inference.embed("multilingual-e5-large", ["hello"])

    assert isinstance(result, EmbeddingsList)
    assert result.model == "multilingual-e5-large"
    assert len(result.data) == 1
    # Values + token count come from make_embed_response() factory.
    assert result.data[0].values == [0.1, 0.2, 0.3]
    assert result.usage.total_tokens == 205
    assert route.called


@respx.mock
def test_embed_single_string_input(inference: Inference) -> None:
    """A bare string input is normalized to a one-element list of {"text": ...} dicts in the request body."""
    route = respx.post(f"{BASE_URL}/embed").mock(
        return_value=httpx.Response(200, json=make_embed_response()),
    )

    result = inference.embed("multilingual-e5-large", "hello")

    assert isinstance(result, EmbeddingsList)
    # Verify the bare string was normalized to a list of dicts
    request = route.calls[0].request
    import orjson

    body = orjson.loads(request.content)
    assert body["inputs"] == [{"text": "hello"}]


@respx.mock
def test_embed_dict_inputs(inference: Inference) -> None:
    route = respx.post(f"{BASE_URL}/embed").mock(
        return_value=httpx.Response(200, json=make_embed_response()),
    )

    result = inference.embed("multilingual-e5-large", [{"text": "hello"}])

    assert isinstance(result, EmbeddingsList)
    import orjson

    body = orjson.loads(route.calls[0].request.content)
    assert body["inputs"] == [{"text": "hello"}]


def test_embed_empty_inputs_raises(inference: Inference) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        inference.embed("multilingual-e5-large", [])


def test_embed_empty_model_raises(inference: Inference) -> None:
    with pytest.raises(ValidationError, match="non-empty"):
        inference.embed("", ["hello"])


@respx.mock
def test_embed_forwards_parameters_to_request_body(inference: Inference) -> None:
    route = respx.post(f"{BASE_URL}/embed").mock(
        return_value=httpx.Response(200, json=make_embed_response()),
    )

    inference.embed(
        "multilingual-e5-large",
        ["hello"],
        parameters={"input_type": "passage", "truncate": "END"},
    )

    import orjson

    body = orjson.loads(route.calls[0].request.content)
    assert body["parameters"] == {"input_type": "passage", "truncate": "END"}


@respx.mock
def test_embed_accepts_embed_model_enum(inference: Inference) -> None:
    route = respx.post(f"{BASE_URL}/embed").mock(
        return_value=httpx.Response(200, json=make_embed_response()),
    )

    result = inference.embed(EmbedModel.Multilingual_E5_Large, ["hello"])

    assert isinstance(result, EmbeddingsList)
    body = orjson.loads(route.calls.last.request.content)
    assert body["model"] == "multilingual-e5-large"


# ---------------------------------------------------------------------------
# rerank()
# ---------------------------------------------------------------------------


@respx.mock
def test_rerank_returns_rerank_result_with_model_and_usage(inference: Inference) -> None:
    """rerank() returns a RerankResult whose model, data, and usage fields reflect the response body."""
    route = respx.post(f"{BASE_URL}/rerank").mock(
        return_value=httpx.Response(200, json=make_rerank_response()),
    )

    result = inference.rerank(
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
def test_rerank_wraps_bare_string_documents_as_text_dicts(inference: Inference) -> None:
    route = respx.post(f"{BASE_URL}/rerank").mock(
        return_value=httpx.Response(200, json=make_rerank_response()),
    )

    inference.rerank(
        model="bge-reranker-v2-m3",
        query="test query",
        documents=["doc1", "doc2"],
    )

    import orjson

    body = orjson.loads(route.calls[0].request.content)
    assert body["documents"] == [{"text": "doc1"}, {"text": "doc2"}]


@respx.mock
def test_rerank_defaults_rank_fields_to_text_when_omitted(inference: Inference) -> None:
    route = respx.post(f"{BASE_URL}/rerank").mock(
        return_value=httpx.Response(200, json=make_rerank_response()),
    )

    inference.rerank(
        model="bge-reranker-v2-m3",
        query="test query",
        documents=["doc1"],
    )

    import orjson

    body = orjson.loads(route.calls[0].request.content)
    assert body["rank_fields"] == ["text"]


def test_rerank_empty_documents_raises(inference: Inference) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        inference.rerank(
            model="bge-reranker-v2-m3",
            query="test query",
            documents=[],
        )


def test_rerank_empty_model_raises(inference: Inference) -> None:
    with pytest.raises(ValidationError, match="non-empty"):
        inference.rerank(
            model="",
            query="test query",
            documents=["doc1"],
        )


def test_rerank_empty_query_raises(inference: Inference) -> None:
    with pytest.raises(ValidationError, match="non-empty"):
        inference.rerank(
            model="bge-reranker-v2-m3",
            query="",
            documents=["doc1"],
        )


def test_rerank_non_list_documents_raises(inference: Inference) -> None:
    with pytest.raises(TypeError, match="Sequence"):
        inference.rerank(
            model="bge-reranker-v2-m3",
            query="test query",
            documents="not a list",  # type: ignore[arg-type]
        )


def test_rerank_top_n_negative_raises(inference: Inference) -> None:
    with pytest.raises(ValidationError, match="top_n must be >= 1"):
        inference.rerank(
            model="bge-reranker-v2-m3",
            query="test query",
            documents=["doc"],
            top_n=-1,
        )


def test_rerank_top_n_zero_raises(inference: Inference) -> None:
    with pytest.raises(ValidationError, match="top_n must be >= 1"):
        inference.rerank(
            model="bge-reranker-v2-m3",
            query="test query",
            documents=["doc"],
            top_n=0,
        )


@respx.mock
def test_rerank_top_n_one_accepted(inference: Inference) -> None:
    respx.post(f"{BASE_URL}/rerank").mock(
        return_value=httpx.Response(200, json=make_rerank_response()),
    )
    result = inference.rerank(
        model="bge-reranker-v2-m3",
        query="test query",
        documents=["doc"],
        top_n=1,
    )
    assert isinstance(result, RerankResult)


@respx.mock
def test_rerank_tuple_documents_accepted(inference: Inference) -> None:
    respx.post(f"{BASE_URL}/rerank").mock(
        return_value=httpx.Response(200, json=make_rerank_response()),
    )

    result = inference.rerank(
        model="bge-reranker-v2-m3",
        query="test query",
        documents=("a", "b"),  # type: ignore[arg-type]
    )
    assert isinstance(result, RerankResult)


def test_rerank_mixed_types_raises(inference: Inference) -> None:
    with pytest.raises(TypeError, match="string or mapping"):
        inference.rerank(
            model="bge-reranker-v2-m3",
            query="test query",
            documents=["a string", 123],  # type: ignore[list-item]
        )


@respx.mock
def test_rerank_accepts_rerank_model_enum(inference: Inference) -> None:
    route = respx.post(f"{BASE_URL}/rerank").mock(
        return_value=httpx.Response(200, json=make_rerank_response()),
    )

    result = inference.rerank(
        model=RerankModel.Bge_Reranker_V2_M3,
        query="test query",
        documents=["doc1"],
    )

    assert isinstance(result, RerankResult)
    body = orjson.loads(route.calls.last.request.content)
    assert body["model"] == "bge-reranker-v2-m3"


# ---------------------------------------------------------------------------
# Lazy property on Pinecone client
# ---------------------------------------------------------------------------


def test_inference_lazy_property() -> None:
    """Accessing .inference twice returns the same instance."""
    with patch.dict("os.environ", {"PINECONE_API_KEY": "test-key"}):
        from pinecone._client import Pinecone

        pc = Pinecone(api_key="test-key")
        first = pc.inference
        second = pc.inference
        assert first is second


def test_inference_repr(inference: Inference) -> None:
    assert repr(inference) == "Inference()"


def test_inference_class_attributes() -> None:
    """Inference exposes EmbedModel and RerankModel as class attributes."""
    assert Inference.EmbedModel is EmbedModel
    assert Inference.RerankModel is RerankModel


# ---------------------------------------------------------------------------
# list_models()
# ---------------------------------------------------------------------------


@respx.mock
def test_list_models_no_filter(inference: Inference) -> None:
    route = respx.get(f"{BASE_URL}/models").mock(
        return_value=httpx.Response(200, json=make_model_list_response()),
    )

    result = inference.list_models()

    assert isinstance(result, ModelInfoList)
    assert len(result) == 2
    assert result.names() == ["multilingual-e5-large", "bge-reranker-v2-m3"]
    assert route.called


@respx.mock
def test_list_models_filter_by_type(inference: Inference) -> None:
    route = respx.get(f"{BASE_URL}/models").mock(
        return_value=httpx.Response(200, json=make_model_list_response()),
    )

    inference.list_models(type="embed")

    assert route.called
    request = route.calls[0].request
    assert request.url.params["type"] == "embed"


@respx.mock
def test_list_models_filter_by_vector_type(inference: Inference) -> None:
    route = respx.get(f"{BASE_URL}/models").mock(
        return_value=httpx.Response(200, json=make_model_list_response()),
    )

    inference.list_models(vector_type="dense")

    assert route.called
    request = route.calls[0].request
    assert request.url.params["vector_type"] == "dense"


def test_list_models_invalid_type_raises(inference: Inference) -> None:
    with pytest.raises(ValidationError, match="must be one of"):
        inference.list_models(type="invalid")


def test_list_models_invalid_vector_type_raises(inference: Inference) -> None:
    with pytest.raises(ValidationError, match="must be one of"):
        inference.list_models(vector_type="invalid")


def test_list_models_rerank_vector_type_raises(inference: Inference) -> None:
    with pytest.raises(ValidationError, match="vector_type is not supported"):
        inference.list_models(type="rerank", vector_type="dense")


@respx.mock
def test_list_models_both_filters(inference: Inference) -> None:
    route = respx.get(f"{BASE_URL}/models").mock(
        return_value=httpx.Response(200, json=make_model_list_response()),
    )

    inference.list_models(type="embed", vector_type="sparse")

    request = route.calls[0].request
    assert request.url.params["type"] == "embed"
    assert request.url.params["vector_type"] == "sparse"


# ---------------------------------------------------------------------------
# get_model()
# ---------------------------------------------------------------------------


@respx.mock
def test_get_model_success(inference: Inference) -> None:
    route = respx.get(f"{BASE_URL}/models/multilingual-e5-large").mock(
        return_value=httpx.Response(200, json=make_model_info()),
    )

    result = inference.get_model(model="multilingual-e5-large")

    assert isinstance(result, ModelInfo)
    assert result.model == "multilingual-e5-large"
    assert result.type == "embed"
    assert result.vector_type == "dense"
    assert result.default_dimension == 1024
    assert result.supported_dimensions == [256, 512, 1024]
    assert result.modality == "text"
    assert result.max_sequence_length == 512
    assert result.max_batch_size == 96
    assert result.provider_name == "Pinecone"
    assert result.supported_metrics == ["cosine", "euclidean"]
    assert len(result.supported_parameters) == 2
    assert result.supported_parameters[0].parameter == "input_type"
    assert result.supported_parameters[0].required is True
    assert result.supported_parameters[0].allowed_values == ["passage", "query"]
    assert result.supported_parameters[1].default == "END"
    assert route.called


def test_get_model_empty_name_raises(inference: Inference) -> None:
    with pytest.raises(ValidationError, match="non-empty"):
        inference.get_model(model="")


@respx.mock
def test_get_model_legacy_model_name_kwarg(inference: Inference) -> None:
    route = respx.get(f"{BASE_URL}/models/multilingual-e5-large").mock(
        return_value=httpx.Response(200, json=make_model_info()),
    )

    result = inference.get_model(model_name="multilingual-e5-large")

    assert isinstance(result, ModelInfo)
    assert result.model == "multilingual-e5-large"
    assert route.called


def test_get_model_conflict_raises(inference: Inference) -> None:
    with pytest.raises(ValidationError, match="model= or model_name="):
        inference.get_model(model="foo", model_name="bar")


def test_get_model_unexpected_kwarg_raises(inference: Inference) -> None:
    with pytest.raises(TypeError, match="unexpected keyword arguments"):
        inference.get_model(model_alias="foo")


# ---------------------------------------------------------------------------
# model cached_property / ModelResource
# ---------------------------------------------------------------------------


def test_inference_model_cached_property(config: PineconeConfig) -> None:
    inference = Inference(config=config)
    assert inference.model is inference.model


@respx.mock
def test_inference_model_list(inference: Inference) -> None:
    route = respx.get(f"{BASE_URL}/models").mock(
        return_value=httpx.Response(200, json=make_model_list_response()),
    )

    result = inference.model.list()

    assert isinstance(result, ModelInfoList)
    assert len(result) == 2
    assert route.called


@respx.mock
def test_inference_model_list_with_filters(inference: Inference) -> None:
    route = respx.get(f"{BASE_URL}/models").mock(
        return_value=httpx.Response(200, json=make_model_list_response()),
    )

    inference.model.list(type="embed", vector_type="dense")

    request = route.calls[0].request
    assert request.url.params["type"] == "embed"
    assert request.url.params["vector_type"] == "dense"


@respx.mock
def test_inference_model_get(inference: Inference) -> None:
    route = respx.get(f"{BASE_URL}/models/multilingual-e5-large").mock(
        return_value=httpx.Response(200, json=make_model_info()),
    )

    result = inference.model.get("multilingual-e5-large")

    assert isinstance(result, ModelInfo)
    assert result.model == "multilingual-e5-large"
    assert route.called


@respx.mock
def test_inference_model_get_legacy_model_name_kwarg(inference: Inference) -> None:
    route = respx.get(f"{BASE_URL}/models/multilingual-e5-large").mock(
        return_value=httpx.Response(200, json=make_model_info()),
    )

    result = inference.model.get(model_name="multilingual-e5-large")

    assert isinstance(result, ModelInfo)
    assert result.model == "multilingual-e5-large"
    assert route.called


def test_inference_model_get_conflict_raises(inference: Inference) -> None:
    with pytest.raises(ValidationError, match="model= or model_name="):
        inference.model.get(model="foo", model_name="bar")


def test_inference_model_get_unexpected_kwarg_raises(inference: Inference) -> None:
    with pytest.raises(TypeError, match="unexpected keyword arguments"):
        inference.model.get(model_alias="foo")


# ---------------------------------------------------------------------------
# API version header
# ---------------------------------------------------------------------------


@respx.mock
def test_all_operations_send_the_2026_07_version_header(inference: Inference) -> None:
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

    inference.embed("multilingual-e5-large", ["hello"])
    inference.rerank("bge-reranker-v2-m3", "q", ["a", "b"])
    inference.list_models()
    inference.get_model(model="multilingual-e5-large")

    for route in (embed, rerank, list_models, get_model):
        assert route.calls.last.request.headers[API_VERSION_HEADER] == "2026-07"


# ---------------------------------------------------------------------------
# Error surfacing
# ---------------------------------------------------------------------------


@respx.mock
def test_embed_400_surfaces_the_server_message_verbatim(inference: Inference) -> None:
    """A parameter-constraint 400 reaches the caller with the server's wording intact.

    The message is the only place the server can name which supported_parameter
    was violated, so paraphrasing or truncating it would cost the caller the one
    detail that makes the error actionable.
    """
    message = (
        "input_type must be one of ['query', 'passage'] "
        "(model multilingual-e5-large, parameter type one_of)"
    )
    respx.post(f"{BASE_URL}/embed").mock(
        return_value=httpx.Response(400, json=make_error_response(400, message)),
    )

    with pytest.raises(ApiError) as excinfo:
        inference.embed("multilingual-e5-large", ["hello"], parameters={"input_type": "nonsense"})

    assert excinfo.value.message == message
    assert excinfo.value.status_code == 400
    assert excinfo.value.error_code == "INVALID_ARGUMENT"


# ---------------------------------------------------------------------------
# 2026-07 inference alignment (#257)
# ---------------------------------------------------------------------------

_NEW_2026_07_EMBED_MODELS = [
    (EmbedModel.Llama_Text_Embed_V2, "llama-text-embed-v2"),
    (EmbedModel.Pinecone_Sparse_Multilingual_V0, "pinecone-sparse-multilingual-v0"),
]


@respx.mock
@pytest.mark.parametrize(("member", "wire_value"), _NEW_2026_07_EMBED_MODELS)
def test_embed_serializes_2026_07_embed_model_ids(
    inference: Inference, member: EmbedModel, wire_value: str
) -> None:
    """The two model ids added for 2026-07 reach the wire and round-trip back."""
    route = respx.post(f"{BASE_URL}/embed").mock(
        return_value=httpx.Response(200, json=make_embed_response(model=wire_value)),
    )

    result = inference.embed(member, ["hello"])

    body = orjson.loads(route.calls[0].request.content)
    assert body["model"] == wire_value
    assert result.model == wire_value
    assert route.calls.last.request.headers[API_VERSION_HEADER] == "2026-07"


@respx.mock
def test_embed_deserializes_sparse_response_without_sparse_tokens(inference: Inference) -> None:
    """A sparse embed response decodes with sparse_tokens absent.

    pinecone-sparse-multilingual-v0 only returns sparse_tokens when the caller
    asks for them via parameters, so the absent case is the default one.
    """
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

    result = inference.embed(EmbedModel.Pinecone_Sparse_Multilingual_V0, ["hello"])

    assert result.vector_type == "sparse"
    embedding = result.data[0]
    assert isinstance(embedding, SparseEmbedding)
    assert embedding.sparse_indices == [10, 20]
    assert embedding.sparse_values == [0.5, 0.25]
    assert embedding.sparse_tokens is None


@respx.mock
def test_embed_deserializes_sparse_response_with_sparse_tokens(inference: Inference) -> None:
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

    result = inference.embed(EmbedModel.Pinecone_Sparse_Multilingual_V0, ["hello"])

    embedding = result.data[0]
    assert isinstance(embedding, SparseEmbedding)
    assert embedding.sparse_tokens == ["hello"]


@respx.mock
def test_embed_deserializes_dense_response_for_llama_text_embed_v2(inference: Inference) -> None:
    respx.post(f"{BASE_URL}/embed").mock(
        return_value=httpx.Response(200, json=make_embed_response(model="llama-text-embed-v2")),
    )

    result = inference.embed(EmbedModel.Llama_Text_Embed_V2, ["hello"])

    assert result.vector_type == "dense"
    assert result.data[0].values == [0.1, 0.2, 0.3]


@respx.mock
def test_embed_unknown_parameters_key_surfaces_plain_text_422(inference: Inference) -> None:
    """An unknown ``parameters`` key is rejected by the server, not by the SDK.

    The inference wire structs deny unknown fields, so the request never reaches
    a handler: the framework rejects it with a 422 whose body is plain text
    rather than the Pinecone JSON error envelope. That leaves error_code unset,
    so callers have only the message to go on — which is why the SDK forwards it
    verbatim instead of validating the keys itself.
    """
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
        inference.embed(
            EmbedModel.Llama_Text_Embed_V2, ["hello"], parameters={"not_a_real_param": 1}
        )

    assert excinfo.value.status_code == 422
    assert excinfo.value.message == plain_text
    assert excinfo.value.error_code is None
    body = orjson.loads(route.calls[0].request.content)
    assert body["parameters"] == {"not_a_real_param": 1}


@respx.mock
def test_embed_project_not_authorized_is_masked_as_404(inference: Inference) -> None:
    """embed reports an authorization failure as 404, unlike rerank's 403.

    Pinned so the divergence stays visible: a NotFoundError from embed does not
    imply the model name is wrong.
    """
    message = "Model 'llama-text-embed-v2' not found"
    respx.post(f"{BASE_URL}/embed").mock(
        return_value=httpx.Response(404, json=make_error_response(404, message)),
    )

    with pytest.raises(NotFoundError) as excinfo:
        inference.embed(EmbedModel.Llama_Text_Embed_V2, ["hello"])

    assert excinfo.value.status_code == 404
    assert excinfo.value.message == message


@respx.mock
def test_rerank_project_not_authorized_surfaces_403(inference: Inference) -> None:
    message = "Project is not authorized to use model bge-reranker-v2-m3"
    respx.post(f"{BASE_URL}/rerank").mock(
        return_value=httpx.Response(403, json=make_error_response(403, message)),
    )

    with pytest.raises(ForbiddenError) as excinfo:
        inference.rerank(model=RerankModel.Bge_Reranker_V2_M3, query="q", documents=["doc"])

    assert excinfo.value.status_code == 403
    assert excinfo.value.message == message


@respx.mock
def test_rerank_deprecated_pinecone_rerank_v0_surfaces_403(inference: Inference) -> None:
    """pinecone-rerank-v0 is deprecated server-side and 403s off the allow-list."""
    message = (
        "Model pinecone-rerank-v0 has been deprecated, please use a different "
        "reranking model from (https://docs.pinecone.io/models)"
    )
    respx.post(f"{BASE_URL}/rerank").mock(
        return_value=httpx.Response(403, json=make_error_response(403, message)),
    )

    with pytest.raises(ForbiddenError) as excinfo:
        inference.rerank(model=RerankModel.Pinecone_Rerank_V0, query="q", documents=["doc"])

    assert excinfo.value.status_code == 403
    assert "has been deprecated" in excinfo.value.message


@respx.mock
def test_rerank_response_model_may_differ_from_requested_model(inference: Inference) -> None:
    """cohere-rerank-3.5 can be served by cohere-rerank-4-fast.

    The backend rewrites the model under a rollout flag and reports what
    actually served the request, so the SDK must not assume an echo.
    """
    respx.post(f"{BASE_URL}/rerank").mock(
        return_value=httpx.Response(200, json=make_rerank_response(model="cohere-rerank-4-fast")),
    )

    result = inference.rerank(model=RerankModel.Cohere_Rerank_3_5, query="q", documents=["doc"])

    assert result.model == "cohere-rerank-4-fast"


@respx.mock
def test_rerank_unknown_parameters_key_surfaces_plain_text_422(inference: Inference) -> None:
    plain_text = (
        "Failed to deserialize the JSON body into the target type: "
        "parameters: unknown field `not_a_real_param`, expected one of "
        "`truncate`, `max_chunks_per_doc`, `max_tokens_per_doc` at line 1 column 90"
    )
    respx.post(f"{BASE_URL}/rerank").mock(
        return_value=httpx.Response(422, text=plain_text),
    )

    with pytest.raises(ApiError) as excinfo:
        inference.rerank(
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
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("member", list(EmbedModel), ids=lambda m: m.name)
def test_str_of_an_embed_model_member_is_not_the_model_id(member: EmbedModel) -> None:
    """The language behaviour this fix exists for.

    If a future Python release or a ``StrEnum`` migration makes ``str(member)``
    return the value, this fails — and ``resolve_model_id`` can be retired.
    """
    assert str(member) == f"EmbedModel.{member.name}"
    assert str(member) != member.value


@pytest.mark.parametrize("member", list(RerankModel), ids=lambda m: m.name)
def test_str_of_a_rerank_model_member_is_not_the_model_id(member: RerankModel) -> None:
    assert str(member) == f"RerankModel.{member.name}"
    assert str(member) != member.value


@respx.mock
@pytest.mark.parametrize("member", list(EmbedModel), ids=lambda m: m.name)
def test_embed_sends_the_model_id_for_every_embed_model_member(
    inference: Inference, member: EmbedModel
) -> None:
    """Parametrized over the whole enum so a member added later is covered too."""
    route = respx.post(f"{BASE_URL}/embed").mock(
        return_value=httpx.Response(200, json=make_embed_response(model=member.value)),
    )

    result = inference.embed(member, ["hello"])

    request = route.calls.last.request
    assert orjson.loads(request.content) == {
        "model": member.value,
        "inputs": [{"text": "hello"}],
    }
    assert b"EmbedModel." not in request.content
    assert result.model == member.value
    assert request.headers[API_VERSION_HEADER] == "2026-07"


@respx.mock
@pytest.mark.parametrize("member", list(RerankModel), ids=lambda m: m.name)
def test_rerank_sends_the_model_id_for_every_rerank_model_member(
    inference: Inference, member: RerankModel
) -> None:
    route = respx.post(f"{BASE_URL}/rerank").mock(
        return_value=httpx.Response(200, json=make_rerank_response(model=member.value)),
    )

    result = inference.rerank(model=member, query="q", documents=["doc"])

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
def test_embed_still_sends_a_plain_model_string_unchanged(inference: Inference) -> None:
    """Both parameters accept a plain string, including an id this SDK has no member for."""
    route = respx.post(f"{BASE_URL}/embed").mock(
        return_value=httpx.Response(200, json=make_embed_response(model="future-embed-model-v9")),
    )

    inference.embed("future-embed-model-v9", ["hello"])

    assert orjson.loads(route.calls.last.request.content)["model"] == "future-embed-model-v9"


@respx.mock
def test_rerank_still_sends_a_plain_model_string_unchanged(inference: Inference) -> None:
    route = respx.post(f"{BASE_URL}/rerank").mock(
        return_value=httpx.Response(200, json=make_rerank_response(model="future-rerank-model-v9")),
    )

    inference.rerank(model="future-rerank-model-v9", query="q", documents=["doc"])

    assert orjson.loads(route.calls.last.request.content)["model"] == "future-rerank-model-v9"


@respx.mock
def test_the_mangled_model_id_this_release_stopped_sending_still_404s(
    inference: Inference,
) -> None:
    """What callers saw before this fix, reproduced by sending the old string by hand.

    The SDK forwards the name verbatim rather than repairing it, so the failure
    is the server's ``NotFoundError`` and the message names the bogus id.
    """
    old_wire_value = "EmbedModel.Multilingual_E5_Large"
    message = f"Model '{old_wire_value}' not found."
    route = respx.post(f"{BASE_URL}/embed").mock(
        return_value=httpx.Response(404, json=make_error_response(404, message)),
    )

    with pytest.raises(NotFoundError) as excinfo:
        inference.embed(old_wire_value, ["hello"])

    assert orjson.loads(route.calls.last.request.content)["model"] == old_wire_value
    assert excinfo.value.message == message


@respx.mock
def test_the_logs_name_the_model_id_not_the_enum_member(
    inference: Inference, caplog: pytest.LogCaptureFixture
) -> None:
    """An operator grepping the logs for a model id has to find it there too."""
    respx.post(f"{BASE_URL}/embed").mock(
        return_value=httpx.Response(200, json=make_embed_response()),
    )
    respx.post(f"{BASE_URL}/rerank").mock(
        return_value=httpx.Response(200, json=make_rerank_response()),
    )

    with caplog.at_level(logging.INFO, logger="pinecone.client.inference"):
        inference.embed(EmbedModel.Multilingual_E5_Large, ["hello"])
        inference.rerank(model=RerankModel.Bge_Reranker_V2_M3, query="q", documents=["doc"])

    assert "'multilingual-e5-large'" in caplog.text
    assert "'bge-reranker-v2-m3'" in caplog.text
    assert "EmbedModel." not in caplog.text
    assert "RerankModel." not in caplog.text
