"""2026-07 conformance for the asyncio transport of the four inference operations.

The sync variants live in ``test_inference_2026_07.py``; both may claim the same
operation (see README, "Additional rules"), so these add no operation ids to the
coverage numerator. What they add is that nothing on the async side asserted the
version before this file — ``AsyncInference`` could have regressed to 2025-10
with a green suite.

Payloads and expected shapes are imported from the sync module rather than
restated, so the two transports cannot drift apart in their fixtures. The
request bodies are re-asserted rather than imported: the body is what the async
method itself builds, and that is the thing under test.

The embed round-trip is recorded against ``_EmbedEnvelope``, and
``DenseEmbedding.vector_type`` is still absent from ``optional_absent``, for the
reasons ``test_inference_2026_07.py`` documents — msgspec cannot convert into a
union of two array types, and the model's ``"dense"`` default trips the
absent-field guard on a spec-required field (#94's finding, not a 2026-07
change). As there, the envelope leg alone would let the items through
untouched, so both embed tests also assert ``result.to_dict() == payload``: a
full trip out through the real async decode path and back. The sparse item
round-trip is a plain msgspec assertion, not ``claim.assert_roundtrip``,
because claim payloads are validated against the operation's response schema
and an embedding item is not a response body.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

import httpx
import msgspec
import orjson
import pytest
import respx

from pinecone._internal.adapters.inference_adapter import _EmbedEnvelope, _ModelListEnvelope
from pinecone._internal.config import PineconeConfig
from pinecone.async_client.inference import AsyncInference
from pinecone.models.enums import EmbedModel, RerankModel
from pinecone.models.inference.embed import SparseEmbedding
from pinecone.models.inference.models import ModelInfo
from pinecone.models.inference.rerank import RerankResult
from tests.unit.conformance import api_op
from tests.unit.conformance.test_inference_2026_07 import (
    BASE_URL,
    DOCUMENTS,
    EMBED_DENSE,
    EMBED_SPARSE,
    MODEL_INFO,
    MODEL_INFO_OPTIONALS,
    MODEL_LIST,
    MODEL_NAME,
    RERANK,
    SPARSE_ITEM,
)


@pytest.fixture
async def async_inference() -> AsyncGenerator[AsyncInference]:
    client = AsyncInference(config=PineconeConfig(api_key="conformance-key", host=BASE_URL))
    yield client
    await client.close()


@api_op("inference:embed")
async def test_async_embed_dense(
    claim: Any, async_inference: AsyncInference, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.post(f"{BASE_URL}/embed").mock(
        return_value=httpx.Response(200, json=EMBED_DENSE)
    )

    result = await async_inference.embed(
        model=MODEL_NAME,
        inputs=["the quick brown fox"],
        parameters={"input_type": "passage", "truncate": "END", "dimension": 512},
    )
    assert result.to_dict() == EMBED_DENSE

    request = route.calls.last.request
    assert orjson.loads(request.content) == {
        "model": MODEL_NAME,
        "inputs": [{"text": "the quick brown fox"}],
        "parameters": {"input_type": "passage", "truncate": "END", "dimension": 512},
    }
    claim.assert_request(request)
    claim.assert_api_version(request)
    claim.assert_roundtrip(_EmbedEnvelope, EMBED_DENSE, optional_absent=[])


@api_op("inference:embed")
async def test_async_embed_sparse(
    claim: Any, async_inference: AsyncInference, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.post(f"{BASE_URL}/embed").mock(
        return_value=httpx.Response(200, json=EMBED_SPARSE)
    )

    result = await async_inference.embed(
        model="pinecone-sparse-english-v0", inputs="the quick brown fox"
    )
    assert result.to_dict() == EMBED_SPARSE

    request = route.calls.last.request
    assert orjson.loads(request.content) == {
        "model": "pinecone-sparse-english-v0",
        "inputs": [{"text": "the quick brown fox"}],
    }
    claim.assert_request(request)
    claim.assert_api_version(request)
    claim.assert_roundtrip(_EmbedEnvelope, EMBED_SPARSE, optional_absent=[])
    item = msgspec.convert(SPARSE_ITEM, type=SparseEmbedding)
    assert msgspec.to_builtins(item) == SPARSE_ITEM
    reduced = {k: v for k, v in SPARSE_ITEM.items() if k != "sparse_tokens"}
    rebuilt = msgspec.to_builtins(msgspec.convert(reduced, type=SparseEmbedding))
    assert {key: rebuilt[key] for key in reduced} == reduced
    assert rebuilt.get("sparse_tokens") is None


@api_op("inference:rerank")
async def test_async_rerank(
    claim: Any, async_inference: AsyncInference, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.post(f"{BASE_URL}/rerank").mock(
        return_value=httpx.Response(200, json=RERANK)
    )

    result = await async_inference.rerank(
        model="bge-reranker-v2-m3",
        query="What is the capital of France?",
        documents=DOCUMENTS,
        rank_fields=["text", "title"],
        return_documents=True,
        top_n=2,
        parameters={"truncate": "END"},
    )
    assert result.to_dict() == RERANK

    request = route.calls.last.request
    assert orjson.loads(request.content) == {
        "model": "bge-reranker-v2-m3",
        "query": "What is the capital of France?",
        "documents": DOCUMENTS,
        "rank_fields": ["text", "title"],
        "return_documents": True,
        "top_n": 2,
        "parameters": {"truncate": "END"},
    }
    claim.assert_request(request)
    claim.assert_api_version(request)
    claim.assert_roundtrip(RerankResult, RERANK, optional_absent=[])


@api_op("inference:embed")
async def test_async_embed_with_an_embed_model_member(
    claim: Any, async_inference: AsyncInference, respx_mock: respx.MockRouter
) -> None:
    """Async twin of the sync module's enum-member embed claim (#296)."""
    route = respx_mock.post(f"{BASE_URL}/embed").mock(
        return_value=httpx.Response(200, json=EMBED_DENSE)
    )

    result = await async_inference.embed(
        model=EmbedModel.Multilingual_E5_Large, inputs=["the quick brown fox"]
    )
    assert result.to_dict() == EMBED_DENSE

    request = route.calls.last.request
    assert orjson.loads(request.content) == {
        "model": MODEL_NAME,
        "inputs": [{"text": "the quick brown fox"}],
    }
    claim.assert_request(request)
    claim.assert_api_version(request)
    claim.assert_roundtrip(_EmbedEnvelope, EMBED_DENSE, optional_absent=[])


@api_op("inference:rerank")
async def test_async_rerank_with_a_rerank_model_member(
    claim: Any, async_inference: AsyncInference, respx_mock: respx.MockRouter
) -> None:
    """Async twin of the sync module's enum-member rerank claim (#296)."""
    route = respx_mock.post(f"{BASE_URL}/rerank").mock(
        return_value=httpx.Response(200, json=RERANK)
    )

    result = await async_inference.rerank(
        model=RerankModel.Bge_Reranker_V2_M3,
        query="What is the capital of France?",
        documents=DOCUMENTS,
    )
    assert result.to_dict() == RERANK

    request = route.calls.last.request
    assert orjson.loads(request.content) == {
        "model": "bge-reranker-v2-m3",
        "query": "What is the capital of France?",
        "documents": DOCUMENTS,
        "rank_fields": ["text"],
        "return_documents": True,
    }
    claim.assert_request(request)
    claim.assert_api_version(request)
    claim.assert_roundtrip(RerankResult, RERANK, optional_absent=[])


@api_op("inference:list_models")
async def test_async_list_models(
    claim: Any, async_inference: AsyncInference, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.get(f"{BASE_URL}/models").mock(
        return_value=httpx.Response(200, json=MODEL_LIST)
    )

    result = await async_inference.list_models(type="embed", vector_type="dense")
    assert result.names() == [MODEL_NAME]

    request = route.calls.last.request
    assert dict(request.url.params) == {"type": "embed", "vector_type": "dense"}
    claim.assert_request(request)
    claim.assert_api_version(request)
    claim.assert_roundtrip(_ModelListEnvelope, MODEL_LIST, optional_absent=["models"])


@api_op("inference:get_model")
async def test_async_get_model(
    claim: Any, async_inference: AsyncInference, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.get(f"{BASE_URL}/models/{MODEL_NAME}").mock(
        return_value=httpx.Response(200, json=MODEL_INFO)
    )

    result = await async_inference.get_model(model=MODEL_NAME)
    assert result.model == MODEL_NAME
    assert result.supported_metrics == ["cosine", "euclidean", "dotproduct"]
    assert [p.parameter for p in result.supported_parameters] == [
        "input_type",
        "dimension",
        "return_tokens",
    ]
    assert result.supported_parameters[1].min == 256
    assert result.supported_parameters[2].default is True

    request = route.calls.last.request
    assert request.url.path == f"/models/{MODEL_NAME}"
    claim.assert_request(request)
    claim.assert_api_version(request)
    claim.assert_roundtrip(ModelInfo, MODEL_INFO, optional_absent=MODEL_INFO_OPTIONALS)
