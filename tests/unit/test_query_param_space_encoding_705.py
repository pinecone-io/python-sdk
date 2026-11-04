"""A space in a query value must reach the server as ``%20``, not ``+`` (#705).

httpx encodes a space in a query value as ``+``. Pinecone's API reads a ``+`` as a
literal plus sign, so ``fetch(ids=["doc 42"])`` asked for ``doc+42``, and
``list_paginated(namespace="my ns")`` listed a namespace called ``my+ns``. Nothing
matched and no error was raised. 8.1.2 sent ``%20``; 9.0.0 moved to httpx and lost it.

The fix is ``_fold_params_into_path`` at the HTTP boundary, so these tests come in
two layers: every method of both HTTP clients that forwards ``params=``, and the
data-plane calls from the report, ``fetch``, ``list_paginated`` and
``list_namespaces(prefix=...)``, on the sync and async lanes.

Every assertion is on the raw request target, which is what the server parses.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
import respx

from pinecone import AsyncIndex, Index, VectorType
from pinecone._internal.config import PineconeConfig
from pinecone._internal.http_client import AsyncHTTPClient, HTTPClient

BASE_URL = "https://api.test.pinecone.io"
INDEX_HOST = "test-index-abc1234.svc.us-east1-gcp.pinecone.io"
INDEX_URL = f"https://{INDEX_HOST}"


def _sync_http() -> HTTPClient:
    return HTTPClient(PineconeConfig(api_key="key", host=BASE_URL), api_version="2026-07")


def _async_http() -> AsyncHTTPClient:
    return AsyncHTTPClient(PineconeConfig(api_key="key", host=BASE_URL), api_version="2026-07")


def _target(route: respx.Route) -> str:
    return route.calls.last.request.url.raw_path.decode()


class TestHttpBoundary:
    @pytest.mark.parametrize("method", ["get", "post", "put", "patch", "delete"])
    @respx.mock
    def test_sync_space_is_percent_20(self, method: str) -> None:
        route = respx.route(method=method.upper(), url__startswith=BASE_URL).mock(
            return_value=httpx.Response(200, json={})
        )
        kwargs: dict[str, Any] = {} if method in ("get", "delete") else {"json": {"x": 1}}
        getattr(_sync_http(), method)("/things", params={"namespace": "my ns"}, **kwargs)
        assert _target(route) == "/things?namespace=my%20ns"

    @pytest.mark.parametrize("method", ["get", "post", "put", "patch", "delete"])
    @pytest.mark.asyncio
    @respx.mock
    async def test_async_space_is_percent_20(self, method: str) -> None:
        route = respx.route(method=method.upper(), url__startswith=BASE_URL).mock(
            return_value=httpx.Response(200, json={})
        )
        kwargs: dict[str, Any] = {} if method in ("get", "delete") else {"json": {"x": 1}}
        await getattr(_async_http(), method)("/things", params={"namespace": "my ns"}, **kwargs)
        assert _target(route) == "/things?namespace=my%20ns"

    @respx.mock
    def test_a_literal_plus_stays_distinct_from_a_space(self) -> None:
        route = respx.get(f"{BASE_URL}/things").mock(return_value=httpx.Response(200, json={}))
        _sync_http().get("/things", params={"ids": ["a b", "a+b", "a%b"]})
        assert _target(route) == "/things?ids=a%20b&ids=a%2Bb&ids=a%25b"
        assert route.calls.last.request.url.params.get_list("ids") == ["a b", "a+b", "a%b"]

    @respx.mock
    def test_enum_members_are_still_resolved(self) -> None:
        route = respx.get(f"{BASE_URL}/models").mock(return_value=httpx.Response(200, json={}))
        _sync_http().get("/models", params={"vector_type": VectorType.DENSE, "q": "a b"})
        assert _target(route) == "/models?vector_type=dense&q=a%20b"

    @respx.mock
    def test_params_are_appended_to_a_query_already_in_the_path(self) -> None:
        route = respx.get(f"{BASE_URL}/things").mock(return_value=httpx.Response(200, json={}))
        _sync_http().get("/things?a=1", params={"b": "x y"})
        assert _target(route) == "/things?a=1&b=x%20y"

    @respx.mock
    def test_no_params_leaves_the_path_alone(self) -> None:
        route = respx.get(f"{BASE_URL}/things").mock(return_value=httpx.Response(200, json={}))
        _sync_http().get("/things")
        assert _target(route) == "/things"

    @respx.mock
    def test_empty_params_add_no_question_mark(self) -> None:
        route = respx.get(f"{BASE_URL}/things").mock(return_value=httpx.Response(200, json={}))
        _sync_http().get("/things", params={})
        assert _target(route) == "/things"


class TestDataPlaneCalls:
    @respx.mock
    def test_fetch_sends_spaces_as_percent_20(self) -> None:
        route = respx.get(f"{INDEX_URL}/vectors/fetch").mock(
            return_value=httpx.Response(200, json={"vectors": {}, "namespace": "my ns"})
        )
        Index(host=INDEX_HOST, api_key="k").fetch(ids=["doc 42", "doc+42"], namespace="my ns")
        target = _target(route)
        assert "+" not in target.replace("%2B", "")
        assert route.calls.last.request.url.params.get_list("ids") == ["doc 42", "doc+42"]
        assert "namespace=my%20ns" in target
        assert "ids=doc%2042" in target
        assert "ids=doc%2B42" in target

    @respx.mock
    def test_list_paginated_sends_spaces_as_percent_20(self) -> None:
        route = respx.get(f"{INDEX_URL}/vectors/list").mock(
            return_value=httpx.Response(200, json={"vectors": [], "namespace": "my ns"})
        )
        Index(host=INDEX_HOST, api_key="k").list_paginated(namespace="my ns", prefix="doc ")
        assert "namespace=my%20ns" in _target(route)
        assert "prefix=doc%20" in _target(route)

    @respx.mock
    def test_list_namespaces_prefix_sends_spaces_as_percent_20(self) -> None:
        route = respx.get(f"{INDEX_URL}/namespaces").mock(
            return_value=httpx.Response(200, json={"namespaces": [], "total_count": 0})
        )
        list(Index(host=INDEX_HOST, api_key="k").list_namespaces(prefix="my ns"))
        assert "prefix=my%20ns" in _target(route)

    @pytest.mark.asyncio
    @respx.mock
    async def test_async_fetch_sends_spaces_as_percent_20(self) -> None:
        route = respx.get(f"{INDEX_URL}/vectors/fetch").mock(
            return_value=httpx.Response(200, json={"vectors": {}, "namespace": "my ns"})
        )
        await AsyncIndex(host=INDEX_HOST, api_key="k").fetch(ids=["doc 42"], namespace="my ns")
        target = _target(route)
        assert "ids=doc%2042" in target
        assert "namespace=my%20ns" in target

    @pytest.mark.asyncio
    @respx.mock
    async def test_async_list_paginated_sends_spaces_as_percent_20(self) -> None:
        route = respx.get(f"{INDEX_URL}/vectors/list").mock(
            return_value=httpx.Response(200, json={"vectors": [], "namespace": "my ns"})
        )
        await AsyncIndex(host=INDEX_HOST, api_key="k").list_paginated(namespace="my ns")
        assert "namespace=my%20ns" in _target(route)
