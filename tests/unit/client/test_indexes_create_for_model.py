"""Unit tests for Indexes.create_for_model() — POST /indexes/create-for-model.

The 2026-07 wire shape for this operation is the legacy cloud/region/embed
form: apis@5f808858 restored it at the source level
(src/release/db/control/resources/indexes/CreateIndexForModel.yaml) and the
backend accepts only this shape (pinecone-db v202607/indexes.rs
test_create_for_model @ f6fd0a40). The published _build OAS is stale here.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import httpx
import pytest
import respx

from pinecone._internal.config import PineconeConfig
from pinecone._internal.constants import CONTROL_PLANE_API_VERSION
from pinecone._internal.http_client import HTTPClient
from pinecone.client.indexes import Indexes
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
def http_client() -> HTTPClient:
    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    return HTTPClient(config, CONTROL_PLANE_API_VERSION)


@pytest.fixture
def indexes(http_client: HTTPClient) -> Indexes:
    return Indexes(http=http_client)


def _mock_route() -> respx.Route:
    return respx.post(f"{BASE_URL}/indexes/create-for-model").mock(
        return_value=httpx.Response(201, json=SEMANTIC_RESPONSE)
    )


# ---------------------------------------------------------------------------
# Wire shape
# ---------------------------------------------------------------------------


@respx.mock
def test_create_for_model_posts_legacy_shape(indexes: Indexes) -> None:
    route = _mock_route()

    result = indexes.create_for_model(
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
def test_create_for_model_emits_source_spec_example_body(indexes: Indexes) -> None:
    """Reproduces the create_index_for_model example from the source spec.

    (src/release/db/control/resources/indexes/CreateIndexForModel.yaml
    example 'index-for-model' @ apis 5f808858.)
    """
    route = _mock_route()

    indexes.create_for_model(
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
def test_create_for_model_full_body(indexes: Indexes) -> None:
    route = _mock_route()

    indexes.create_for_model(
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
def test_create_for_model_wrapped_metadata_schema_passes_through(indexes: Indexes) -> None:
    route = _mock_route()

    indexes.create_for_model(
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
def test_create_for_model_accepts_embed_config(indexes: Indexes) -> None:
    route = _mock_route()

    indexes.create_for_model(
        name="semantic-index",
        cloud="aws",
        region="us-east-1",
        embed=EmbedConfig(model="multilingual-e5-large", field_map={"text": "chunk_text"}),
        timeout=-1,
    )

    body = json.loads(route.calls.last.request.content)
    assert body["embed"] == EMBED


@respx.mock
def test_create_for_model_accepts_index_embed(indexes: Indexes) -> None:
    from pinecone.inference.models.index_embed import IndexEmbed

    route = _mock_route()

    indexes.create_for_model(
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


def test_create_for_model_missing_model_raises(indexes: Indexes) -> None:
    with pytest.raises(PineconeValueError, match="'model'"):
        indexes.create_for_model(
            name="x", cloud="aws", region="us-east-1", embed={"field_map": {"text": "t"}}
        )


def test_create_for_model_missing_field_map_raises(indexes: Indexes) -> None:
    with pytest.raises(PineconeValueError, match="'field_map'"):
        indexes.create_for_model(
            name="x", cloud="aws", region="us-east-1", embed={"model": "multilingual-e5-large"}
        )


def test_create_for_model_invalid_name_raises(indexes: Indexes) -> None:
    with pytest.raises(PineconeValueError, match="invalid characters"):
        indexes.create_for_model(name="Bad_Name", cloud="aws", region="us-east-1", embed=EMBED)


def test_create_for_model_empty_cloud_raises(indexes: Indexes) -> None:
    with pytest.raises(PineconeValueError, match="cloud"):
        indexes.create_for_model(name="x", cloud="", region="us-east-1", embed=EMBED)


def test_create_for_model_invalid_deletion_protection_raises(indexes: Indexes) -> None:
    with pytest.raises(PineconeValueError, match="deletion_protection"):
        indexes.create_for_model(
            name="x", cloud="aws", region="us-east-1", embed=EMBED, deletion_protection="on"
        )


# ---------------------------------------------------------------------------
# Readiness polling
# ---------------------------------------------------------------------------


@respx.mock
def test_create_for_model_polls_until_ready_by_default(indexes: Indexes) -> None:
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

    with patch("pinecone._internal.indexes_helpers.time.sleep"):
        result = indexes.create_for_model(
            name="semantic-index", cloud="aws", region="us-east-1", embed=EMBED
        )

    assert describe_route.called
    assert result.status.ready is True
