"""Unit tests for AsyncIndexes.create_for_model() — POST /indexes/create-for-model.

Async mirror of tests/unit/client/test_indexes_create_for_model.py. The
2026-07 wire shape for this operation is the legacy cloud/region/embed form:
apis@5f808858 restored it at the source level
(src/release/db/control/resources/indexes/CreateIndexForModel.yaml) and the
backend accepts only this shape (pinecone-db v202607/indexes.rs
test_create_for_model @ f6fd0a40). The published _build OAS is stale here.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import respx

from pinecone._internal.config import PineconeConfig
from pinecone._internal.constants import CONTROL_PLANE_API_VERSION
from pinecone._internal.http_client import AsyncHTTPClient
from pinecone.async_client.indexes import AsyncIndexes
from pinecone.errors.exceptions import PineconeValueError
from pinecone.models.indexes.index import IndexModel
from pinecone.models.indexes.specs import EmbedConfig
from tests.factories import make_index_response

BASE_URL = "https://api.test.pinecone.io"

EMBED = {"model": "multilingual-e5-large", "field_map": {"text": "chunk_text"}}

SEMANTIC_RESPONSE = make_index_response(
    name="semantic-index",
    schema={
        "fields": {
            "chunk_text": {"type": "semantic_text", "model": "multilingual-e5-large"},
        }
    },
)


@pytest.fixture
async def async_http_client() -> AsyncGenerator[AsyncHTTPClient]:
    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    client = AsyncHTTPClient(config, CONTROL_PLANE_API_VERSION)
    yield client
    await client.close()


@pytest.fixture
def indexes(async_http_client: AsyncHTTPClient) -> AsyncIndexes:
    return AsyncIndexes(http=async_http_client)


def _mock_route() -> respx.Route:
    return respx.post(f"{BASE_URL}/indexes/create-for-model").mock(
        return_value=httpx.Response(201, json=SEMANTIC_RESPONSE)
    )


# ---------------------------------------------------------------------------
# Wire shape
# ---------------------------------------------------------------------------


@respx.mock
async def test_create_for_model_posts_legacy_shape(indexes: AsyncIndexes) -> None:
    route = _mock_route()

    result = await indexes.create_for_model(
        name="semantic-index", cloud="aws", region="us-east-1", embed=EMBED, timeout=-1
    )

    request = route.calls.last.request
    assert request.url.path == "/indexes/create-for-model"
    assert request.headers.get("X-Pinecone-Api-Version") == CONTROL_PLANE_API_VERSION
    body = json.loads(request.content)
    assert body == {
        "name": "semantic-index",
        "cloud": "aws",
        "region": "us-east-1",
        "embed": EMBED,
    }
    assert isinstance(result, IndexModel)


@respx.mock
async def test_create_for_model_emits_source_spec_example_body(indexes: AsyncIndexes) -> None:
    """Reproduces the create_index_for_model example from the source spec.

    (src/release/db/control/resources/indexes/CreateIndexForModel.yaml
    example 'index-for-model' @ apis 5f808858.)
    """
    route = _mock_route()

    await indexes.create_for_model(
        name="multilingual-e5-large-index",
        cloud="gcp",
        region="us-east1",
        deletion_protection="enabled",
        embed={
            "model": "multilingual-e5-large",
            "metric": "cosine",
            "field_map": {"text": "your-text-field"},
        },
        timeout=-1,
    )

    body = json.loads(route.calls.last.request.content)
    assert body == {
        "name": "multilingual-e5-large-index",
        "cloud": "gcp",
        "region": "us-east1",
        "deletion_protection": "enabled",
        "embed": {
            "model": "multilingual-e5-large",
            "metric": "cosine",
            "field_map": {"text": "your-text-field"},
        },
    }


@respx.mock
async def test_create_for_model_full_body(indexes: AsyncIndexes) -> None:
    route = _mock_route()

    await indexes.create_for_model(
        name="semantic-index",
        cloud="aws",
        region="us-east-1",
        embed={
            "model": "multilingual-e5-large",
            "field_map": {"text": "chunk_text"},
            "read_parameters": {"input_type": "query", "truncate": "NONE"},
            "write_parameters": {"input_type": "passage"},
        },
        tags={"env": "prod"},
        schema={"genre": {"filterable": True}},
        read_capacity={"mode": "OnDemand"},
        timeout=-1,
    )

    body = json.loads(route.calls.last.request.content)
    assert body["embed"]["read_parameters"] == {"input_type": "query", "truncate": "NONE"}
    assert body["embed"]["write_parameters"] == {"input_type": "passage"}
    assert body["tags"] == {"env": "prod"}
    assert body["read_capacity"] == {"mode": "OnDemand"}
    assert body["schema"] == {"fields": {"genre": {"filterable": True}}}


@respx.mock
async def test_create_for_model_wrapped_metadata_schema_passes_through(
    indexes: AsyncIndexes,
) -> None:
    route = _mock_route()

    await indexes.create_for_model(
        name="semantic-index",
        cloud="aws",
        region="us-east-1",
        embed=EMBED,
        schema={"fields": {"genre": {"filterable": True}}},
        timeout=-1,
    )

    body = json.loads(route.calls.last.request.content)
    assert body["schema"] == {"fields": {"genre": {"filterable": True}}}


@respx.mock
async def test_create_for_model_accepts_embed_config(indexes: AsyncIndexes) -> None:
    route = _mock_route()

    await indexes.create_for_model(
        name="semantic-index",
        cloud="aws",
        region="us-east-1",
        embed=EmbedConfig(model="multilingual-e5-large", field_map={"text": "chunk_text"}),
        timeout=-1,
    )

    body = json.loads(route.calls.last.request.content)
    assert body["embed"] == EMBED


@respx.mock
async def test_create_for_model_accepts_index_embed(indexes: AsyncIndexes) -> None:
    from pinecone.inference.models.index_embed import IndexEmbed

    route = _mock_route()

    await indexes.create_for_model(
        name="semantic-index",
        cloud="aws",
        region="us-east-1",
        embed=IndexEmbed(model="multilingual-e5-large", field_map={"text": "chunk_text"}),
        timeout=-1,
    )

    body = json.loads(route.calls.last.request.content)
    assert body["embed"]["model"] == "multilingual-e5-large"
    assert body["embed"]["field_map"] == {"text": "chunk_text"}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


async def test_create_for_model_missing_model_raises(indexes: AsyncIndexes) -> None:
    with pytest.raises(PineconeValueError, match="'model'"):
        await indexes.create_for_model(
            name="x", cloud="aws", region="us-east-1", embed={"field_map": {"text": "t"}}
        )


async def test_create_for_model_missing_field_map_raises(indexes: AsyncIndexes) -> None:
    with pytest.raises(PineconeValueError, match="'field_map'"):
        await indexes.create_for_model(
            name="x", cloud="aws", region="us-east-1", embed={"model": "multilingual-e5-large"}
        )


async def test_create_for_model_invalid_name_raises(indexes: AsyncIndexes) -> None:
    with pytest.raises(PineconeValueError, match="invalid characters"):
        await indexes.create_for_model(
            name="Bad_Name", cloud="aws", region="us-east-1", embed=EMBED
        )


async def test_create_for_model_empty_cloud_raises(indexes: AsyncIndexes) -> None:
    with pytest.raises(PineconeValueError, match="cloud"):
        await indexes.create_for_model(name="x", cloud="", region="us-east-1", embed=EMBED)


async def test_create_for_model_invalid_deletion_protection_raises(indexes: AsyncIndexes) -> None:
    with pytest.raises(PineconeValueError, match="deletion_protection"):
        await indexes.create_for_model(
            name="x", cloud="aws", region="us-east-1", embed=EMBED, deletion_protection="on"
        )


# ---------------------------------------------------------------------------
# Readiness polling
# ---------------------------------------------------------------------------


@respx.mock
async def test_create_for_model_polls_until_ready_by_default(indexes: AsyncIndexes) -> None:
    respx.post(f"{BASE_URL}/indexes/create-for-model").mock(
        return_value=httpx.Response(
            201,
            json=make_index_response(
                name="semantic-index", status={"ready": False, "state": "Initializing"}
            ),
        )
    )
    describe_route = respx.get(f"{BASE_URL}/indexes/semantic-index").mock(
        return_value=httpx.Response(
            200,
            json=make_index_response(
                name="semantic-index", status={"ready": True, "state": "Ready"}
            ),
        )
    )

    with patch("pinecone._internal.indexes_helpers.asyncio.sleep", new_callable=AsyncMock):
        result = await indexes.create_for_model(
            name="semantic-index", cloud="aws", region="us-east-1", embed=EMBED
        )

    assert describe_route.called
    assert result.status.ready is True
