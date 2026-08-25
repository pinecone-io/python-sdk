"""Sync/async parity for configure()'s deprecated pod/read-capacity sugar (#501).

``Indexes.configure()`` and ``AsyncIndexes.configure()`` both translate
``replicas=``/``pod_type=``/``serverless_read_capacity=`` through the same
``pinecone/_internal/legacy_index_translation.py`` helpers, so identical
inputs must put byte-identical PATCH bodies on the wire on both transports.
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
from tests.factories import make_index_response

BASE_URL = "https://api.test.pinecone.io"

_CASES: dict[str, dict[str, Any]] = {
    "replicas-only": {"replicas": 3},
    "pod-type-only": {"pod_type": "p1.x2"},
    "replicas-and-pod-type": {"replicas": 3, "pod_type": "p1.x2"},
    "serverless-read-capacity": {"serverless_read_capacity": {"mode": "OnDemand"}},
    "replicas-with-tags": {"replicas": 5, "tags": {"env": "prod"}},
}


@pytest.mark.parametrize("case_name", sorted(_CASES))
async def test_sync_and_async_configure_put_identical_bytes_on_the_wire(case_name: str) -> None:
    kwargs = _CASES[case_name]

    with respx.mock:
        route = respx.patch(f"{BASE_URL}/indexes/test-index").mock(
            return_value=httpx.Response(200, json=make_index_response())
        )
        sync_config = PineconeConfig(api_key="test-key", host=BASE_URL)
        Indexes(http=HTTPClient(sync_config, CONTROL_PLANE_API_VERSION)).configure(
            "test-index", **kwargs
        )
        sync_body = bytes(route.calls.last.request.content)

    with respx.mock:
        route = respx.patch(f"{BASE_URL}/indexes/test-index").mock(
            return_value=httpx.Response(200, json=make_index_response())
        )
        async_config = PineconeConfig(api_key="test-key", host=BASE_URL)
        async_http_client = AsyncHTTPClient(async_config, CONTROL_PLANE_API_VERSION)
        try:
            await AsyncIndexes(http=async_http_client).configure("test-index", **kwargs)
        finally:
            await async_http_client.close()
        async_body = bytes(route.calls.last.request.content)

    assert sync_body == async_body
    assert sync_body != b""
