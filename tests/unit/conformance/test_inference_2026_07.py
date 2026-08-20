"""2026-07 conformance for the four inference operations.

Ticket #94 is version-bump-only, and the evidence is stronger here than for the
other surfaces: a key-sorted semantic diff of ``inference_2026-07.oas.yaml``
against its 2025-10 predecessor is *empty* once the ``X-Pinecone-Api-Version``
default and ``info.version`` are set aside — the raw textual diff is pure
alphabetical reordering of the schema block. The backend agrees: 2026-07 is
routed to the same handlers 2026-04 uses
(``svc-global-apis/src/inference/routes/mod.rs:32`` for ``/models``,
``inference-server/src/inference/routes/mod.rs:64`` for ``/embed`` and
``/rerank``, pinecone-db@f6fd0a4019).

So these tests pin what a bump can still break: method, path, query and path
parameters, the exact request bodies (which must be byte-for-byte what 2025-10
sent), the version header now reading 2026-07, and the response schemas.

The embed round-trip is recorded against ``_EmbedEnvelope`` rather than
``EmbeddingsList`` because the latter's ``data`` is a union of two array types,
which msgspec cannot convert into directly — the SDK's own adapter decodes the
envelope first and dispatches on ``vector_type`` for exactly that reason. The
envelope leg alone would let the embedding items through untouched, so both
embed tests additionally assert ``result.to_dict() == payload``: a full trip
out through the SDK's real decode path and back, items included. The sparse test
also round-trips the item through ``SparseEmbedding`` directly — as a plain
msgspec assertion, because ``claim.assert_roundtrip`` payloads are validated
against the operation's response schema and an embedding item is not a
response body.

The client under test is a real :class:`Inference`, so the version on the wire
comes from ``INFERENCE_API_VERSION`` and not from this file. Async inference
reads the same constant; ticket #95 claims these operations for that transport.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import httpx
import msgspec
import orjson
import pytest
import respx

from pinecone._internal.adapters.inference_adapter import _EmbedEnvelope, _ModelListEnvelope
from pinecone._internal.config import PineconeConfig
from pinecone._internal.constants import DEFAULT_BASE_URL
from pinecone.client.inference import Inference
from pinecone.models.inference.embed import SparseEmbedding
from pinecone.models.inference.models import ModelInfo
from pinecone.models.inference.rerank import RerankResult
from tests.unit.conformance import api_op

BASE_URL = DEFAULT_BASE_URL
MODEL_NAME = "multilingual-e5-large"

EMBED_DENSE: dict[str, Any] = {
    "model": MODEL_NAME,
    "vector_type": "dense",
    "data": [{"values": [0.1, 0.2, 0.3], "vector_type": "dense"}],
    "usage": {"total_tokens": 205},
}

SPARSE_ITEM: dict[str, Any] = {
    "sparse_values": [0.6, 0.4],
    "sparse_indices": [17, 42],
    "sparse_tokens": ["quick", "fox"],
    "vector_type": "sparse",
}

EMBED_SPARSE: dict[str, Any] = {
    "model": "pinecone-sparse-english-v0",
    "vector_type": "sparse",
    "data": [SPARSE_ITEM],
    "usage": {"total_tokens": 12},
}

DOCUMENTS: list[dict[str, Any]] = [
    {"id": "1", "text": "Paris is the capital of France.", "title": "France"},
    {"id": "2", "text": "Berlin is the capital of Germany.", "title": "Germany"},
]

RERANK: dict[str, Any] = {
    "model": "bge-reranker-v2-m3",
    "data": [
        {"index": 0, "score": 0.95, "document": DOCUMENTS[0]},
        {"index": 1, "score": 0.45, "document": DOCUMENTS[1]},
    ],
    "usage": {"rerank_units": 1},
}

MODEL_INFO: dict[str, Any] = {
    "model": MODEL_NAME,
    "short_description": "A multilingual embedding model.",
    "type": "embed",
    "vector_type": "dense",
    "default_dimension": 1024,
    "supported_dimensions": [256, 512, 1024],
    "modality": "text",
    "max_sequence_length": 512,
    "max_batch_size": 96,
    "provider_name": "NVIDIA",
    "supported_metrics": ["cosine", "euclidean", "dotproduct"],
    "supported_parameters": [
        {
            "parameter": "input_type",
            "type": "one_of",
            "value_type": "string",
            "required": True,
            "allowed_values": ["passage", "query"],
        },
        {
            "parameter": "dimension",
            "type": "numeric_range",
            "value_type": "integer",
            "required": False,
            "min": 256,
            "max": 1024,
            "default": 1024,
        },
        {
            "parameter": "return_tokens",
            "type": "any",
            "value_type": "boolean",
            "required": False,
            "default": True,
        },
    ],
}

MODEL_INFO_OPTIONALS = [
    "vector_type",
    "default_dimension",
    "supported_dimensions",
    "modality",
    "max_sequence_length",
    "max_batch_size",
    "provider_name",
    "supported_metrics",
]

MODEL_LIST: dict[str, Any] = {"models": [MODEL_INFO]}


@pytest.fixture
def inference() -> Iterator[Inference]:
    client = Inference(config=PineconeConfig(api_key="conformance-key", host=BASE_URL))
    yield client
    client.close()


@api_op("inference:embed")
def test_embed_dense(claim: Any, inference: Inference, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.post(f"{BASE_URL}/embed").mock(
        return_value=httpx.Response(200, json=EMBED_DENSE)
    )

    result = inference.embed(
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
def test_embed_sparse(claim: Any, inference: Inference, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.post(f"{BASE_URL}/embed").mock(
        return_value=httpx.Response(200, json=EMBED_SPARSE)
    )

    result = inference.embed(model="pinecone-sparse-english-v0", inputs="the quick brown fox")
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
def test_rerank(claim: Any, inference: Inference, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.post(f"{BASE_URL}/rerank").mock(
        return_value=httpx.Response(200, json=RERANK)
    )

    result = inference.rerank(
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


@api_op("inference:list_models")
def test_list_models(claim: Any, inference: Inference, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.get(f"{BASE_URL}/models").mock(
        return_value=httpx.Response(200, json=MODEL_LIST)
    )

    result = inference.list_models(type="embed", vector_type="dense")
    assert result.names() == [MODEL_NAME]

    request = route.calls.last.request
    assert request.url.params["type"] == "embed"
    assert request.url.params["vector_type"] == "dense"
    claim.assert_request(request)
    claim.assert_api_version(request)
    claim.assert_roundtrip(_ModelListEnvelope, MODEL_LIST, optional_absent=["models"])


@api_op("inference:get_model")
def test_get_model(claim: Any, inference: Inference, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.get(f"{BASE_URL}/models/{MODEL_NAME}").mock(
        return_value=httpx.Response(200, json=MODEL_INFO)
    )

    result = inference.get_model(model=MODEL_NAME)
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
