"""Wire-level tests for the Pinecone.create_index_for_model backcompat shim.

Verifies the deletion_protection default: 9.x sent "disabled" when the
caller omitted the argument, and an explicit "disabled" was never dropped.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from pinecone import AsyncPinecone, Pinecone
from tests.factories import make_index_response

BASE_URL = "https://api.pinecone.io"
EMBED = {"model": "multilingual-e5-large", "field_map": {"text": "chunk_text"}}


@pytest.fixture
def pc() -> Pinecone:
    return Pinecone(api_key="test-key")


@pytest.fixture
def async_pc() -> AsyncPinecone:
    return AsyncPinecone(api_key="test-key")


def _mock_route() -> respx.Route:
    return respx.post(f"{BASE_URL}/indexes/create-for-model").mock(
        return_value=httpx.Response(201, json=make_index_response(name="semantic-index"))
    )


@respx.mock
def test_create_index_for_model_defaults_deletion_protection_to_disabled_on_wire(
    pc: Pinecone,
) -> None:
    route = _mock_route()

    pc.create_index_for_model(
        name="semantic-index", cloud="aws", region="us-east-1", embed=EMBED, timeout=-1
    )

    body = json.loads(route.calls.last.request.content)
    assert body["deletion_protection"] == "disabled"


@respx.mock
def test_create_index_for_model_forwards_explicit_disabled_on_wire(pc: Pinecone) -> None:
    route = _mock_route()

    pc.create_index_for_model(
        name="semantic-index",
        cloud="aws",
        region="us-east-1",
        embed=EMBED,
        deletion_protection="disabled",
        timeout=-1,
    )

    body = json.loads(route.calls.last.request.content)
    assert body["deletion_protection"] == "disabled"


@respx.mock
def test_create_index_for_model_forwards_explicit_enabled_on_wire(pc: Pinecone) -> None:
    route = _mock_route()

    pc.create_index_for_model(
        name="semantic-index",
        cloud="aws",
        region="us-east-1",
        embed=EMBED,
        deletion_protection="enabled",
        timeout=-1,
    )

    body = json.loads(route.calls.last.request.content)
    assert body["deletion_protection"] == "enabled"


@respx.mock
@pytest.mark.asyncio
async def test_async_create_index_for_model_defaults_deletion_protection_to_disabled_on_wire(
    async_pc: AsyncPinecone,
) -> None:
    route = _mock_route()

    await async_pc.create_index_for_model(
        name="semantic-index", cloud="aws", region="us-east-1", embed=EMBED, timeout=-1
    )

    body = json.loads(route.calls.last.request.content)
    assert body["deletion_protection"] == "disabled"

    await async_pc.close()
