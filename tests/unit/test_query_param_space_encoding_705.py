"""Query parameters must percent-encode spaces as ``%20``, not ``+`` (#705).

httpx serializes query parameters with ``application/x-www-form-urlencoded``
(``quote_plus``), encoding spaces as ``+``. Pinecone endpoints such as
``/vectors/fetch``, ``/vectors/list``, and ``/namespaces`` expect RFC 3986
``%20`` encoding (matching v8.x / urllib3) so IDs, namespaces, and prefixes
containing spaces match on the server.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from pinecone import AsyncPinecone, Pinecone
from pinecone._internal.config import PineconeConfig
from pinecone._internal.http_client import AsyncHTTPClient, HTTPClient

INDEX_HOST = "https://idx-space-test.svc.pinecone.io"


def _sync_http() -> HTTPClient:
    return HTTPClient(PineconeConfig(api_key="test-key", host=INDEX_HOST), api_version="2026-07")


def _async_http() -> AsyncHTTPClient:
    return AsyncHTTPClient(
        PineconeConfig(api_key="test-key", host=INDEX_HOST), api_version="2026-07"
    )


class TestSyncQuerySpaceEncoding:
    @respx.mock
    def test_get_encodes_spaces_as_percent20_and_preserves_literal_plus(self) -> None:
        route = respx.get(url__startswith=f"{INDEX_HOST}/vectors/fetch").mock(
            return_value=httpx.Response(200, json={"vectors": {}})
        )
        _sync_http().get(
            "/vectors/fetch",
            params={"ids": ["doc 42", "a+b"], "namespace": "my namespace"},
        )
        assert route.called
        raw_path = route.calls.last.request.url.raw_path.decode("ascii")
        assert "ids=doc%2042" in raw_path
        assert "ids=a%2Bb" in raw_path
        assert "namespace=my%20namespace" in raw_path
        assert "+" not in raw_path.split("?", 1)[1]

    @respx.mock
    def test_get_appends_to_existing_query_string(self) -> None:
        route = respx.get(url__startswith=f"{INDEX_HOST}/vectors/list").mock(
            return_value=httpx.Response(200, json={"vectors": []})
        )
        _sync_http().get("/vectors/list?limit=10", params={"namespace": "team alpha"})
        assert route.called
        raw_path = route.calls.last.request.url.raw_path.decode("ascii")
        assert raw_path == "/vectors/list?limit=10&namespace=team%20alpha"

    @respx.mock
    def test_index_fetch_list_paginated_and_list_namespaces_use_percent20(self) -> None:
        fetch_route = respx.get(url__startswith=f"{INDEX_HOST}/vectors/fetch").mock(
            return_value=httpx.Response(200, json={"vectors": {}, "namespace": "my ns"})
        )
        list_route = respx.get(url__startswith=f"{INDEX_HOST}/vectors/list").mock(
            return_value=httpx.Response(200, json={"vectors": [], "namespace": "my ns"})
        )
        ns_route = respx.get(url__startswith=f"{INDEX_HOST}/namespaces").mock(
            return_value=httpx.Response(200, json={"namespaces": []})
        )

        pc = Pinecone(api_key="test-key")
        idx = pc.Index(host=INDEX_HOST)

        idx.fetch(ids=["doc 42"], namespace="my ns")
        idx.list_paginated(prefix="doc ", namespace="my ns")
        idx.list_namespaces_paginated(prefix="my ")

        assert (
            fetch_route.calls.last.request.url.raw_path.decode("ascii")
            == "/vectors/fetch?ids=doc%2042&namespace=my%20ns"
        )
        assert "prefix=doc%20" in list_route.calls.last.request.url.raw_path.decode("ascii")
        assert "namespace=my%20ns" in list_route.calls.last.request.url.raw_path.decode("ascii")
        assert "prefix=my%20" in ns_route.calls.last.request.url.raw_path.decode("ascii")


class TestAsyncQuerySpaceEncoding:
    @pytest.mark.asyncio
    @respx.mock
    async def test_async_get_encodes_spaces_as_percent20(self) -> None:
        route = respx.get(url__startswith=f"{INDEX_HOST}/vectors/fetch").mock(
            return_value=httpx.Response(200, json={"vectors": {}})
        )
        client = _async_http()
        try:
            await client.get(
                "/vectors/fetch",
                params={"ids": ["doc 42"], "namespace": "my namespace"},
            )
        finally:
            await client.close()

        assert route.called
        raw_path = route.calls.last.request.url.raw_path.decode("ascii")
        assert raw_path == "/vectors/fetch?ids=doc%2042&namespace=my%20namespace"

    @pytest.mark.asyncio
    @respx.mock
    async def test_async_index_fetch_uses_percent20(self) -> None:
        fetch_route = respx.get(url__startswith=f"{INDEX_HOST}/vectors/fetch").mock(
            return_value=httpx.Response(200, json={"vectors": {}, "namespace": "my ns"})
        )
        async with AsyncPinecone(api_key="test-key") as pc:
            async with pc.IndexAsyncio(host=INDEX_HOST) as idx:
                await idx.fetch(ids=["doc 42"], namespace="my ns")

        assert (
            fetch_route.calls.last.request.url.raw_path.decode("ascii")
            == "/vectors/fetch?ids=doc%2042&namespace=my%20ns"
        )
