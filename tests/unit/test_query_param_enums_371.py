"""Enum members must reach the query string as their values (#371).

``VectorType`` is a ``(str, Enum)`` mixin and httpx encodes every query value
with ``str()``, which for that kind of enum returns ``"VectorType.DENSE"``
rather than ``"dense"``. ``require_one_of`` validates the argument without
normalizing it, so a member passed to ``list_models(vector_type=...)`` survived
validation and went on the wire mangled.

The fix is ``_prepare_params`` at the HTTP boundary, not at the one call site,
so these tests come in two layers:

* the boundary itself, exercised through every method of both clients that
  forwards ``params=`` — this is what makes query parameters on *other*
  surfaces, and ones added later, immune;
* ``list_models`` and the ``pc.inference.model.list`` facade end to end on both
  lanes, which is the reported bug.

Every assertion is on the real request URL, never on a helper's return value.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
import respx

from pinecone import AsyncPinecone, Pinecone, VectorType
from pinecone._internal.config import PineconeConfig
from pinecone._internal.http_client import AsyncHTTPClient, HTTPClient
from pinecone.errors.exceptions import PineconeValueError
from pinecone.models.enums import Metric

BASE_URL = "https://api.test.pinecone.io"
MODEL_LIST: dict[str, Any] = {"models": []}


def _sync_http() -> HTTPClient:
    return HTTPClient(PineconeConfig(api_key="key", host=BASE_URL), api_version="2026-07")


def _async_http() -> AsyncHTTPClient:
    return AsyncHTTPClient(PineconeConfig(api_key="key", host=BASE_URL), api_version="2026-07")


class TestTheStringEnumPremise:
    """Pins why the fix has to exist. If these fail, ``_prepare_params`` can go."""

    def test_str_of_a_member_is_still_the_mangled_name(self) -> None:
        assert str(VectorType.DENSE) == "VectorType.DENSE"
        assert VectorType.DENSE.value == "dense"

    def test_httpx_still_encodes_an_unresolved_member_with_str(self) -> None:
        """httpx is the mangling encoder; this is the wire value the bug produced."""
        assert str(httpx.QueryParams({"vector_type": VectorType.DENSE})) == (
            "vector_type=VectorType.DENSE"
        )


class TestSyncBoundary:
    @respx.mock
    def test_get_resolves_members(self) -> None:
        route = respx.get(f"{BASE_URL}/models").mock(return_value=httpx.Response(200, json={}))
        _sync_http().get("/models", params={"vector_type": VectorType.SPARSE})
        assert dict(route.calls.last.request.url.params) == {"vector_type": "sparse"}

    @respx.mock
    def test_post_resolves_members(self) -> None:
        route = respx.post(f"{BASE_URL}/files/a").mock(return_value=httpx.Response(200, json={}))
        _sync_http().post("/files/a", params={"metric": Metric.COSINE}, json={"x": 1})
        assert dict(route.calls.last.request.url.params) == {"metric": "cosine"}

    @respx.mock
    def test_put_resolves_members(self) -> None:
        route = respx.put(f"{BASE_URL}/things/1").mock(return_value=httpx.Response(200, json={}))
        _sync_http().put("/things/1", params={"metric": Metric.EUCLIDEAN}, json={"x": 1})
        assert dict(route.calls.last.request.url.params) == {"metric": "euclidean"}

    @respx.mock
    def test_patch_resolves_members(self) -> None:
        route = respx.patch(f"{BASE_URL}/things/1").mock(return_value=httpx.Response(200, json={}))
        _sync_http().patch("/things/1", params={"metric": Metric.DOTPRODUCT}, json={"x": 1})
        assert dict(route.calls.last.request.url.params) == {"metric": "dotproduct"}

    @respx.mock
    def test_delete_resolves_members(self) -> None:
        route = respx.delete(f"{BASE_URL}/things/1").mock(return_value=httpx.Response(200, json={}))
        _sync_http().delete("/things/1", params={"metric": Metric.COSINE})
        assert dict(route.calls.last.request.url.params) == {"metric": "cosine"}


@pytest.mark.asyncio
class TestAsyncBoundary:
    @respx.mock
    async def test_get_resolves_members(self) -> None:
        route = respx.get(f"{BASE_URL}/models").mock(return_value=httpx.Response(200, json={}))
        client = _async_http()
        try:
            await client.get("/models", params={"vector_type": VectorType.SPARSE})
        finally:
            await client.close()
        assert dict(route.calls.last.request.url.params) == {"vector_type": "sparse"}

    @respx.mock
    async def test_post_resolves_members(self) -> None:
        route = respx.post(f"{BASE_URL}/files/a").mock(return_value=httpx.Response(200, json={}))
        client = _async_http()
        try:
            await client.post("/files/a", params={"metric": Metric.COSINE}, json={"x": 1})
        finally:
            await client.close()
        assert dict(route.calls.last.request.url.params) == {"metric": "cosine"}

    @respx.mock
    async def test_put_resolves_members(self) -> None:
        route = respx.put(f"{BASE_URL}/things/1").mock(return_value=httpx.Response(200, json={}))
        client = _async_http()
        try:
            await client.put("/things/1", params={"metric": Metric.EUCLIDEAN}, json={"x": 1})
        finally:
            await client.close()
        assert dict(route.calls.last.request.url.params) == {"metric": "euclidean"}

    @respx.mock
    async def test_patch_resolves_members(self) -> None:
        route = respx.patch(f"{BASE_URL}/things/1").mock(return_value=httpx.Response(200, json={}))
        client = _async_http()
        try:
            await client.patch("/things/1", params={"metric": Metric.DOTPRODUCT}, json={"x": 1})
        finally:
            await client.close()
        assert dict(route.calls.last.request.url.params) == {"metric": "dotproduct"}

    @respx.mock
    async def test_delete_resolves_members(self) -> None:
        route = respx.delete(f"{BASE_URL}/things/1").mock(return_value=httpx.Response(200, json={}))
        client = _async_http()
        try:
            await client.delete("/things/1", params={"metric": Metric.COSINE})
        finally:
            await client.close()
        assert dict(route.calls.last.request.url.params) == {"metric": "cosine"}


class TestOtherParamShapes:
    @respx.mock
    def test_a_list_of_members_resolves_element_by_element(self) -> None:
        route = respx.get(f"{BASE_URL}/models").mock(return_value=httpx.Response(200, json={}))
        _sync_http().get("/models", params={"vector_type": list(VectorType)})
        assert route.calls.last.request.url.params.get_list("vector_type") == ["dense", "sparse"]

    @respx.mock
    def test_a_sequence_of_pairs_resolves_too(self) -> None:
        """httpx accepts pairs as well as a mapping, so the boundary must handle both."""
        route = respx.get(f"{BASE_URL}/models").mock(return_value=httpx.Response(200, json={}))
        _sync_http().get("/models", params=[("vector_type", VectorType.DENSE), ("limit", 10)])
        assert dict(route.calls.last.request.url.params) == {"vector_type": "dense", "limit": "10"}

    @respx.mock
    def test_an_already_encoded_queryparams_keeps_its_repeated_keys(self) -> None:
        """``httpx.QueryParams`` stringifies on construction, so there is nothing
        left to resolve — and rebuilding it as a dict would drop the second ``a``.
        """
        route = respx.get(f"{BASE_URL}/models").mock(return_value=httpx.Response(200, json={}))
        _sync_http().get("/models", params=httpx.QueryParams([("a", "1"), ("a", "2")]))
        assert route.calls.last.request.url.params.get_list("a") == ["1", "2"]

    @respx.mock
    def test_a_raw_query_string_is_passed_through(self) -> None:
        route = respx.get(f"{BASE_URL}/models").mock(return_value=httpx.Response(200, json={}))
        _sync_http().get("/models", params="a=1&a=2")
        assert route.calls.last.request.url.params.get_list("a") == ["1", "2"]

    @respx.mock
    def test_non_enum_values_are_untouched(self) -> None:
        route = respx.get(f"{BASE_URL}/models").mock(return_value=httpx.Response(200, json={}))
        _sync_http().get("/models", params={"limit": 10, "include_deleted": True, "p": "tok"})
        assert dict(route.calls.last.request.url.params) == {
            "limit": "10",
            "include_deleted": "true",
            "p": "tok",
        }


@pytest.mark.parametrize("member", list(VectorType), ids=lambda m: m.name)
class TestListModelsEndToEnd:
    @respx.mock
    def test_sync_list_models(self, member: VectorType) -> None:
        route = respx.get(f"{BASE_URL}/models").mock(
            return_value=httpx.Response(200, json=MODEL_LIST)
        )
        pc = Pinecone(api_key="key", host=BASE_URL)
        pc.inference.list_models(type="embed", vector_type=member)
        url = route.calls.last.request.url
        assert url.params["vector_type"] == member.value
        assert "VectorType" not in str(url)

    @respx.mock
    def test_sync_model_list_facade(self, member: VectorType) -> None:
        route = respx.get(f"{BASE_URL}/models").mock(
            return_value=httpx.Response(200, json=MODEL_LIST)
        )
        pc = Pinecone(api_key="key", host=BASE_URL)
        pc.inference.model.list(type="embed", vector_type=member)
        assert route.calls.last.request.url.params["vector_type"] == member.value

    @pytest.mark.asyncio
    @respx.mock
    async def test_async_list_models(self, member: VectorType) -> None:
        route = respx.get(f"{BASE_URL}/models").mock(
            return_value=httpx.Response(200, json=MODEL_LIST)
        )
        async with AsyncPinecone(api_key="key", host=BASE_URL) as pc:
            await pc.inference.list_models(type="embed", vector_type=member)
        url = route.calls.last.request.url
        assert url.params["vector_type"] == member.value
        assert "VectorType" not in str(url)

    @pytest.mark.asyncio
    @respx.mock
    async def test_async_model_list_facade(self, member: VectorType) -> None:
        route = respx.get(f"{BASE_URL}/models").mock(
            return_value=httpx.Response(200, json=MODEL_LIST)
        )
        async with AsyncPinecone(api_key="key", host=BASE_URL) as pc:
            await pc.inference.model.list(type="embed", vector_type=member)
        assert route.calls.last.request.url.params["vector_type"] == member.value

    @respx.mock
    def test_the_member_and_its_value_produce_the_same_url(self, member: VectorType) -> None:
        route = respx.get(f"{BASE_URL}/models").mock(
            return_value=httpx.Response(200, json=MODEL_LIST)
        )
        pc = Pinecone(api_key="key", host=BASE_URL)
        pc.inference.list_models(vector_type=member)
        from_member = str(route.calls.last.request.url)
        pc.inference.list_models(vector_type=member.value)
        assert str(route.calls.last.request.url) == from_member


class TestTheOldShapeStillFails:
    """The SDK does not repair the mangled name; it rejects it before the request."""

    @respx.mock
    def test_the_mangled_literal_is_rejected_without_a_request(self) -> None:
        route = respx.get(f"{BASE_URL}/models").mock(
            return_value=httpx.Response(200, json=MODEL_LIST)
        )
        pc = Pinecone(api_key="key", host=BASE_URL)

        with pytest.raises(PineconeValueError) as excinfo:
            pc.inference.list_models(vector_type="VectorType.DENSE")

        assert str(excinfo.value) == (
            "vector_type must be one of 'dense', 'sparse', got 'VectorType.DENSE'"
        )
        assert not route.calls

    @pytest.mark.asyncio
    @respx.mock
    async def test_the_mangled_literal_is_rejected_on_the_async_lane_too(self) -> None:
        route = respx.get(f"{BASE_URL}/models").mock(
            return_value=httpx.Response(200, json=MODEL_LIST)
        )
        async with AsyncPinecone(api_key="key", host=BASE_URL) as pc:
            with pytest.raises(PineconeValueError):
                await pc.inference.list_models(vector_type="VectorType.SPARSE")
        assert not route.calls
