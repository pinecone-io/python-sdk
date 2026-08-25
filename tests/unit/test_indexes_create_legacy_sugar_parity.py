"""Sync/async parity for the #500 deprecated create() sugar.

``Indexes.create()`` and ``AsyncIndexes.create()`` both call the same
``pinecone/_internal/legacy_index_translation.py`` functions to turn
``spec=``/``dimension=``/``metric=``/``vector_type=`` into ``schema=``/
``deployment=``/``read_capacity=``. Ticket #500 ships sync and async
together specifically so the two request bodies cannot drift; this file
pins that identical inputs on both lanes produce byte-identical wire bodies,
covering the shapes named in the ticket's acceptance criteria (dense,
sparse, pod, byoc, dict-spec, and read-capacity placement).
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
import respx

from pinecone._internal.config import PineconeConfig
from pinecone._internal.constants import CONTROL_PLANE_API_VERSION
from pinecone._internal.http_client import AsyncHTTPClient, HTTPClient
from pinecone.async_client.indexes import AsyncIndexes
from pinecone.client.indexes import Indexes
from pinecone.models.indexes.specs import ByocSpec, PodSpec, ServerlessSpec
from tests.factories import make_index_response

BASE_URL = "https://api.test.pinecone.io"

_CALLS: dict[str, dict[str, Any]] = {
    "dense_with_spec": {
        "name": "movies",
        "dimension": 1536,
        "metric": "cosine",
        "spec": ServerlessSpec(cloud="aws", region="us-east-1"),
        "timeout": -1,
    },
    "sparse_with_spec": {
        "name": "sparse-index",
        "vector_type": "sparse",
        "metric": "dotproduct",
        "spec": ServerlessSpec(cloud="aws", region="us-east-1"),
        "timeout": -1,
    },
    "dimension_only_metric_defaults": {
        "dimension": 8,
        "spec": ServerlessSpec(cloud="aws", region="us-east-1"),
        "timeout": -1,
    },
    "pod_spec": {
        "name": "pods",
        "dimension": 8,
        "spec": PodSpec(environment="us-east1-gcp", pod_type="p1.x2", replicas=2, shards=3),
        "timeout": -1,
    },
    "byoc_spec": {
        "name": "byoc",
        "dimension": 8,
        "spec": ByocSpec(environment="aws-us-east-1-b921"),
        "timeout": -1,
    },
    "dict_spec": {
        "name": "movies",
        "dimension": 3,
        "spec": {"serverless": {"cloud": "aws", "region": "eu-west-1"}},
        "timeout": -1,
    },
    "spec_read_capacity_lands_top_level": {
        "dimension": 8,
        "spec": ServerlessSpec(cloud="aws", region="us-east-1", read_capacity={"mode": "OnDemand"}),
        "timeout": -1,
    },
    "explicit_read_capacity_overrides_spec": {
        "dimension": 8,
        "spec": ServerlessSpec(cloud="aws", region="us-east-1", read_capacity={"mode": "OnDemand"}),
        "read_capacity": {"mode": "Dedicated", "dedicated": {"node_type": "t1"}},
        "timeout": -1,
    },
    "schema_and_deployment_unaffected": {
        "name": "explicit",
        "schema": {
            "fields": {"embedding": {"type": "dense_vector", "dimension": 3, "metric": "cosine"}}
        },
        "deployment": {"deployment_type": "managed", "cloud": "aws", "region": "us-east-1"},
        "timeout": -1,
    },
}


@pytest.fixture
def http_client() -> HTTPClient:
    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    return HTTPClient(config, CONTROL_PLANE_API_VERSION)


@pytest.fixture
async def async_http_client() -> AsyncHTTPClient:
    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    client = AsyncHTTPClient(config, CONTROL_PLANE_API_VERSION)
    yield client
    await client.close()


def _sync_body(http_client: HTTPClient, kwargs: dict[str, Any]) -> bytes:
    indexes = Indexes(http=http_client)
    with respx.mock:
        route = respx.post(f"{BASE_URL}/indexes").mock(
            return_value=httpx.Response(201, json=make_index_response())
        )
        indexes.create(**kwargs)
        return bytes(route.calls.last.request.content)


async def _async_body(async_http_client: AsyncHTTPClient, kwargs: dict[str, Any]) -> bytes:
    indexes = AsyncIndexes(http=async_http_client)
    with respx.mock:
        route = respx.post(f"{BASE_URL}/indexes").mock(
            return_value=httpx.Response(201, json=make_index_response())
        )
        await indexes.create(**kwargs)
        return bytes(route.calls.last.request.content)


@pytest.mark.parametrize("case", sorted(_CALLS))
async def test_create_body_is_byte_identical_on_sync_and_async(
    case: str, http_client: HTTPClient, async_http_client: AsyncHTTPClient
) -> None:
    kwargs = _CALLS[case]
    sync_body = _sync_body(http_client, kwargs)
    async_body = await _async_body(async_http_client, kwargs)
    assert sync_body == async_body, f"case {case!r}: sync/async bodies diverged"
