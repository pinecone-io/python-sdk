"""``get_model`` carries the model id in the URL path, so it must be encoded.

Inference model ids include provider-prefixed aliases such as
``nvidia/llama-text-embed-v2`` (``pc-inference-commons/src/lib.rs:14``
@ ``f6fd0a4019``). Raw interpolation into ``/models/{model}`` turns the ``/``
into a path separator: the GET misses the ``/models/:model`` route entirely and
falls through to the POST-only proxy, so the caller sees ``405`` instead of a
model lookup. Control characters make httpx reject the URL outright.

So the segment is percent-encoded, exactly as the namespace, records, and
documents routes already do (#119/#120, #233). These tests pin the encoded
request line for both transports; reverting the encoding fails every case.

The gateway's ``get_model`` is an exact match on the canonical model name with
no alias resolution (``routes/base/models.rs:212-221``), so an encoded alias
still 404s — that is a server-side contract the docstring calls out, not
something encoding can fix. What encoding buys is that the request reaches the
route at all.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from pinecone._internal.config import PineconeConfig
from pinecone.async_client.inference import AsyncInference
from pinecone.client.inference import Inference
from tests.factories import make_model_info

BASE_URL = "https://api.test.pinecone.io"

ENCODING_CASES = [
    pytest.param(
        "nvidia/llama-text-embed-v2",
        "nvidia%2Fllama-text-embed-v2",
        id="provider_prefixed_alias_slash",
    ),
    pytest.param("a/b", "a%2Fb", id="slash_cannot_change_the_route"),
    pytest.param("a b", "a%20b", id="space"),
    pytest.param("a?b#c", "a%3Fb%23c", id="query_and_fragment_delimiters"),
    pytest.param("100%", "100%25", id="percent"),
    pytest.param("\x01", "%01", id="control_character"),
    pytest.param("\x7f", "%7F", id="delete_character"),
]


@pytest.fixture
def config() -> PineconeConfig:
    return PineconeConfig(api_key="test-key", host=BASE_URL)


@pytest.fixture
def inference(config: PineconeConfig) -> Inference:
    return Inference(config=config)


@pytest.fixture
def async_inference(config: PineconeConfig) -> AsyncInference:
    return AsyncInference(config=config)


@pytest.mark.parametrize(("model", "encoded"), ENCODING_CASES)
@respx.mock
def test_get_model_encodes_model_as_one_path_segment(
    inference: Inference, model: str, encoded: str
) -> None:
    route = respx.get(f"{BASE_URL}/models/{encoded}").mock(
        return_value=httpx.Response(200, json=make_model_info(model=model))
    )

    inference.get_model(model=model)

    assert route.called
    assert route.calls.last.request.url.raw_path.decode() == f"/models/{encoded}"


@pytest.mark.parametrize(("model", "encoded"), ENCODING_CASES)
@respx.mock
def test_get_model_encodes_model_name_alias_as_one_path_segment(
    inference: Inference, model: str, encoded: str
) -> None:
    route = respx.get(f"{BASE_URL}/models/{encoded}").mock(
        return_value=httpx.Response(200, json=make_model_info(model=model))
    )

    inference.get_model(model_name=model)

    assert route.called
    assert route.calls.last.request.url.raw_path.decode() == f"/models/{encoded}"


@pytest.mark.parametrize(("model", "encoded"), ENCODING_CASES)
@respx.mock
@pytest.mark.asyncio
async def test_async_get_model_encodes_model_as_one_path_segment(
    async_inference: AsyncInference, model: str, encoded: str
) -> None:
    route = respx.get(f"{BASE_URL}/models/{encoded}").mock(
        return_value=httpx.Response(200, json=make_model_info(model=model))
    )

    await async_inference.get_model(model=model)

    assert route.called
    assert route.calls.last.request.url.raw_path.decode() == f"/models/{encoded}"


@pytest.mark.parametrize(("model", "encoded"), ENCODING_CASES)
@respx.mock
@pytest.mark.asyncio
async def test_async_get_model_encodes_model_name_alias_as_one_path_segment(
    async_inference: AsyncInference, model: str, encoded: str
) -> None:
    route = respx.get(f"{BASE_URL}/models/{encoded}").mock(
        return_value=httpx.Response(200, json=make_model_info(model=model))
    )

    await async_inference.get_model(model_name=model)

    assert route.called
    assert route.calls.last.request.url.raw_path.decode() == f"/models/{encoded}"
