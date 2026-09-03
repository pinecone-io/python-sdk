"""Sync/async parity for the four inference operations.

``Inference`` and ``AsyncInference`` read the same ``INFERENCE_API_VERSION`` and
talk to the same handlers, so they should differ only in ``await``. These tests
hold them to that on the three axes a version bump can quietly break: identical
request snapshots on the wire (method, path, query, body, version header) for
identical arguments, identical signatures, and identical exception types and
messages for the failures a caller can trigger before any request is made.

The request-snapshot cases are the async half of AC #2 — the sync bodies pinned
in ``tests/unit/conformance/test_inference_2026_07.py`` are compared against
what the async client emits for the same call, rather than being restated a
third time.

Follows the pattern of ``tests/unit/test_async_assistants_parity.py``.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
import orjson
import pytest
import respx

from pinecone._internal.config import PineconeConfig
from pinecone._internal.constants import API_VERSION_HEADER
from pinecone.async_client.inference import AsyncInference, AsyncModelResource
from pinecone.client.inference import Inference, ModelResource
from tests.factories import (
    make_embed_response,
    make_model_info,
    make_model_list_response,
    make_rerank_response,
)

BASE_URL = "https://api.test.pinecone.io"
MODEL_NAME = "multilingual-e5-large"

_METHODS = ["close", "embed", "get_model", "list_models", "rerank"]
_MODEL_RESOURCE_METHODS = ["get", "list"]

_CALLS: dict[str, dict[str, Any]] = {
    "embed": {
        "model": MODEL_NAME,
        "inputs": ["the quick brown fox"],
        "parameters": {"input_type": "passage", "truncate": "END", "dimension": 512},
    },
    "rerank": {
        "model": "bge-reranker-v2-m3",
        "query": "What is the capital of France?",
        "documents": [{"id": "1", "text": "Paris is the capital of France.", "title": "France"}],
        "rank_fields": ["text", "title"],
        "return_documents": True,
        "top_n": 2,
        "parameters": {"truncate": "END"},
    },
    "list_models": {"type": "embed", "vector_type": "dense"},
    "get_model": {"model": MODEL_NAME},
}

_ERROR_CASES: list[tuple[str, dict[str, Any]]] = [
    ("embed", {"model": "", "inputs": ["hello"]}),
    ("embed", {"model": MODEL_NAME, "inputs": []}),
    ("embed", {"model": MODEL_NAME, "inputs": 17}),
    ("rerank", {"model": "", "query": "q", "documents": ["d"]}),
    ("rerank", {"model": "m", "query": "", "documents": ["d"]}),
    ("rerank", {"model": "m", "query": "q", "documents": []}),
    ("rerank", {"model": "m", "query": "q", "documents": "not a list"}),
    ("rerank", {"model": "m", "query": "q", "documents": ["a string", 123]}),
    ("rerank", {"model": "m", "query": "q", "documents": ["d"], "top_n": 0}),
    ("rerank", {"model": "m", "query": "q", "documents": ["d"], "top_n": -1}),
    ("list_models", {"type": "invalid"}),
    ("list_models", {"vector_type": "invalid"}),
    ("list_models", {"type": "rerank", "vector_type": "dense"}),
    ("get_model", {"model": ""}),
    ("get_model", {"model": "foo", "model_name": "bar"}),
    ("get_model", {"model_alias": "foo"}),
]


@pytest.fixture
def config() -> PineconeConfig:
    return PineconeConfig(api_key="test-key", host=BASE_URL)


@pytest.fixture
def sync_inference(config: PineconeConfig) -> Inference:
    return Inference(config=config)


@pytest.fixture
def async_inference(config: PineconeConfig) -> AsyncInference:
    return AsyncInference(config=config)


def _register_routes() -> None:
    respx.post(f"{BASE_URL}/embed").mock(
        return_value=httpx.Response(200, json=make_embed_response()),
    )
    respx.post(f"{BASE_URL}/rerank").mock(
        return_value=httpx.Response(200, json=make_rerank_response()),
    )
    respx.get(f"{BASE_URL}/models").mock(
        return_value=httpx.Response(200, json=make_model_list_response()),
    )
    respx.get(f"{BASE_URL}/models/{MODEL_NAME}").mock(
        return_value=httpx.Response(200, json=make_model_info()),
    )


def _snapshot(request: httpx.Request) -> dict[str, Any]:
    return {
        "method": request.method,
        "path": request.url.path,
        "query": dict(request.url.params),
        "body": orjson.loads(request.content) if request.content else None,
        "api_version": request.headers[API_VERSION_HEADER],
    }


def _raised(call: Callable[[], object]) -> tuple[type[BaseException], str]:
    try:
        call()
    except Exception as exc:
        return type(exc), str(exc)
    raise AssertionError("expected the call to raise, it returned instead")


async def _raised_async(call: Callable[[], Awaitable[object]]) -> tuple[type[BaseException], str]:
    try:
        await call()
    except Exception as exc:
        return type(exc), str(exc)
    raise AssertionError("expected the call to raise, it returned instead")


@pytest.mark.parametrize("method_name", sorted(_CALLS))
@respx.mock
async def test_request_snapshot_parity(
    method_name: str, sync_inference: Inference, async_inference: AsyncInference
) -> None:
    _register_routes()
    kwargs = _CALLS[method_name]

    getattr(sync_inference, method_name)(**kwargs)
    sync_snapshot = _snapshot(respx.calls.last.request)

    await getattr(async_inference, method_name)(**kwargs)
    async_snapshot = _snapshot(respx.calls.last.request)

    assert len(respx.calls) == 2, "each transport must have issued exactly one request"
    assert async_snapshot == sync_snapshot


@pytest.mark.parametrize("method_name", _METHODS)
def test_parameter_parity(method_name: str) -> None:
    sync_params = dict(inspect.signature(getattr(Inference, method_name)).parameters)
    async_params = dict(inspect.signature(getattr(AsyncInference, method_name)).parameters)

    assert set(sync_params) == set(async_params), (
        f"{method_name}: parameter names differ — "
        f"sync-only={set(sync_params) - set(async_params)}, "
        f"async-only={set(async_params) - set(sync_params)}"
    )

    for name, sync_param in sync_params.items():
        async_param = async_params[name]
        assert sync_param.kind == async_param.kind, (
            f"{method_name}.{name}: kind differs (sync={sync_param.kind}, async={async_param.kind})"
        )
        assert sync_param.default == async_param.default, (
            f"{method_name}.{name}: default differs "
            f"(sync={sync_param.default!r}, async={async_param.default!r})"
        )
        assert str(sync_param.annotation) == str(async_param.annotation), (
            f"{method_name}.{name}: annotation differs "
            f"(sync={sync_param.annotation}, async={async_param.annotation})"
        )


@pytest.mark.parametrize("method_name", _METHODS)
def test_return_annotation_parity(method_name: str) -> None:
    sync_return = inspect.signature(getattr(Inference, method_name)).return_annotation
    async_return = inspect.signature(getattr(AsyncInference, method_name)).return_annotation

    assert str(sync_return) == str(async_return), (
        f"{method_name}: return annotation differs (sync={sync_return}, async={async_return})"
    )


@pytest.mark.parametrize("method_name", _MODEL_RESOURCE_METHODS)
def test_model_resource_signature_parity(method_name: str) -> None:
    sync_signature = inspect.signature(getattr(ModelResource, method_name))
    async_signature = inspect.signature(getattr(AsyncModelResource, method_name))

    assert str(sync_signature) == str(async_signature)


@pytest.mark.parametrize(
    "method_name,kwargs",
    _ERROR_CASES,
    ids=[f"{name}-{sorted(kwargs.items())}" for name, kwargs in _ERROR_CASES],
)
async def test_validation_error_parity(
    method_name: str,
    kwargs: dict[str, Any],
    sync_inference: Inference,
    async_inference: AsyncInference,
) -> None:
    sync_type, sync_message = _raised(lambda: getattr(sync_inference, method_name)(**kwargs))
    async_type, async_message = await _raised_async(
        lambda: getattr(async_inference, method_name)(**kwargs)
    )

    assert async_type is sync_type
    assert async_message == sync_message
