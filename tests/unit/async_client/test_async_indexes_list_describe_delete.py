"""Unit tests for AsyncIndexes.list/describe/exists/delete (2026-07).

Async mirror of tests/unit/client/test_indexes_list_describe_delete.py:
list() returns an AsyncPaginator over IndexModel; describe/exists/delete keep
their 2025-10 flows against 2026-07 response fixtures, including host-cache
maintenance and post-delete polling.
"""

from __future__ import annotations

import itertools
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import respx

from pinecone._internal.config import PineconeConfig
from pinecone._internal.constants import CONTROL_PLANE_API_VERSION
from pinecone._internal.http_client import AsyncHTTPClient
from pinecone.async_client.indexes import AsyncIndexes
from pinecone.errors.exceptions import (
    NotFoundError,
    PineconeTimeoutError,
    PineconeValueError,
)
from pinecone.models.indexes.index import IndexModel
from pinecone.models.pagination import AsyncPaginator
from tests.factories import make_error_response, make_index_list_response, make_index_response

BASE_URL = "https://api.test.pinecone.io"


@pytest.fixture
async def async_http_client() -> AsyncGenerator[AsyncHTTPClient]:
    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    client = AsyncHTTPClient(config, CONTROL_PLANE_API_VERSION)
    yield client
    await client.close()


@pytest.fixture
def indexes(async_http_client: AsyncHTTPClient) -> AsyncIndexes:
    return AsyncIndexes(http=async_http_client)


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


@respx.mock
async def test_list_returns_async_paginator_over_index_models(indexes: AsyncIndexes) -> None:
    route = respx.get(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(200, json=make_index_list_response())
    )

    result = indexes.list()

    assert isinstance(result, AsyncPaginator)
    items = [item async for item in result]
    assert route.called
    assert len(items) == 1
    assert isinstance(items[0], IndexModel)
    assert items[0].name == "test-index"
    assert route.calls.last.request.headers.get("X-Pinecone-Api-Version") == (
        CONTROL_PLANE_API_VERSION
    )


@respx.mock
async def test_list_is_lazy_until_iterated(indexes: AsyncIndexes) -> None:
    route = respx.get(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(200, json=make_index_list_response())
    )

    paginator = indexes.list()
    assert route.call_count == 0
    _ = [item async for item in paginator]
    assert route.call_count == 1


@respx.mock
async def test_list_empty(indexes: AsyncIndexes) -> None:
    respx.get(f"{BASE_URL}/indexes").mock(return_value=httpx.Response(200, json={"indexes": []}))

    assert [item async for item in indexes.list()] == []


@respx.mock
async def test_list_yields_single_page(indexes: AsyncIndexes) -> None:
    respx.get(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(200, json=make_index_list_response())
    )

    pages = [page async for page in indexes.list().pages()]

    assert len(pages) == 1
    assert pages[0].pagination_token is None
    assert len(pages[0].items) == 1


@respx.mock
async def test_list_respects_limit(indexes: AsyncIndexes) -> None:
    payload = {
        "indexes": [
            make_index_response(name=f"index-{i}", host=f"index-{i}.svc.pinecone.io")
            for i in range(3)
        ]
    }
    respx.get(f"{BASE_URL}/indexes").mock(return_value=httpx.Response(200, json=payload))

    items = [item async for item in indexes.list(limit=2)]

    assert [item.name for item in items] == ["index-0", "index-1"]


async def test_list_zero_limit_raises(indexes: AsyncIndexes) -> None:
    with pytest.raises(PineconeValueError, match="limit"):
        indexes.list(limit=0)


@respx.mock
async def test_list_skips_unparseable_index_with_warning(
    indexes: AsyncIndexes, caplog: pytest.LogCaptureFixture
) -> None:
    """The resilient list decode (question #177, parked) is preserved as-is."""
    payload = {
        "indexes": [
            make_index_response(),
            {"name": "broken-index", "schema": {"fields": {"f": {"type": "no-such-type"}}}},
        ]
    }
    respx.get(f"{BASE_URL}/indexes").mock(return_value=httpx.Response(200, json=payload))

    with caplog.at_level("WARNING"):
        items = [item async for item in indexes.list()]

    assert [item.name for item in items] == ["test-index"]
    assert "broken-index" in caplog.text


@respx.mock
async def test_list_decodes_legacy_untyped_schema_fields(indexes: AsyncIndexes) -> None:
    payload = {
        "indexes": [
            make_index_response(
                schema={
                    "fields": {
                        "embedding": {"type": "dense_vector", "dimension": 3, "metric": "cosine"},
                        "genre": {"filterable": True},
                    }
                }
            )
        ]
    }
    respx.get(f"{BASE_URL}/indexes").mock(return_value=httpx.Response(200, json=payload))

    items = [item async for item in indexes.list()]

    assert len(items) == 1
    assert "genre" in items[0].schema.fields


# ---------------------------------------------------------------------------
# describe / exists
# ---------------------------------------------------------------------------


@respx.mock
async def test_describe_index(indexes: AsyncIndexes) -> None:
    route = respx.get(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(200, json=make_index_response())
    )

    result = await indexes.describe("test-index")

    assert route.called
    assert isinstance(result, IndexModel)
    assert result.name == "test-index"
    assert result.host == "https://test-index-abc1234.svc.us-east1-gcp.pinecone.io"


@respx.mock
async def test_describe_caches_host(indexes: AsyncIndexes) -> None:
    respx.get(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(200, json=make_index_response())
    )

    await indexes.describe("test-index")

    assert indexes._host_cache["test-index"] == (
        "https://test-index-abc1234.svc.us-east1-gcp.pinecone.io"
    )


@respx.mock
async def test_describe_nonexistent_raises_not_found(indexes: AsyncIndexes) -> None:
    respx.get(f"{BASE_URL}/indexes/no-such-index").mock(
        return_value=httpx.Response(404, json=make_error_response(404, "Index not found"))
    )

    with pytest.raises(NotFoundError):
        await indexes.describe("no-such-index")


async def test_describe_empty_name_raises(indexes: AsyncIndexes) -> None:
    with pytest.raises(PineconeValueError, match="name"):
        await indexes.describe("")


@respx.mock
async def test_exists_true(indexes: AsyncIndexes) -> None:
    respx.get(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(200, json=make_index_response())
    )

    assert await indexes.exists("test-index") is True


@respx.mock
async def test_exists_false_on_404(indexes: AsyncIndexes) -> None:
    respx.get(f"{BASE_URL}/indexes/gone").mock(
        return_value=httpx.Response(404, json=make_error_response(404, "Index not found"))
    )

    assert await indexes.exists("gone") is False


async def test_exists_empty_name_raises(indexes: AsyncIndexes) -> None:
    """Graduated behavior: empty name raises instead of returning False."""
    with pytest.raises(PineconeValueError, match="name"):
        await indexes.exists("")


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


@respx.mock
async def test_delete_sends_delete_and_evicts_host_cache(indexes: AsyncIndexes) -> None:
    respx.get(f"{BASE_URL}/indexes/test-index").mock(
        side_effect=[
            httpx.Response(200, json=make_index_response()),
            httpx.Response(404, json=make_error_response(404, "Not found")),
        ]
    )
    delete_route = respx.delete(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(202)
    )

    await indexes.describe("test-index")
    assert "test-index" in indexes._host_cache

    with patch("pinecone.async_client.indexes.asyncio.sleep", new_callable=AsyncMock):
        await indexes.delete("test-index")

    assert delete_route.called
    assert "test-index" not in indexes._host_cache


@respx.mock
async def test_delete_timeout_negative_one_skips_polling(indexes: AsyncIndexes) -> None:
    respx.delete(f"{BASE_URL}/indexes/test-index").mock(return_value=httpx.Response(202))
    describe_route = respx.get(f"{BASE_URL}/indexes/test-index")

    await indexes.delete("test-index", timeout=-1)

    assert describe_route.call_count == 0


@respx.mock
async def test_delete_polls_until_not_found(indexes: AsyncIndexes) -> None:
    respx.delete(f"{BASE_URL}/indexes/test-index").mock(return_value=httpx.Response(202))
    respx.get(f"{BASE_URL}/indexes/test-index").mock(
        side_effect=[
            httpx.Response(200, json=make_index_response()),
            httpx.Response(404, json=make_error_response(404, "Not found")),
        ]
    )

    with patch("pinecone.async_client.indexes.asyncio.sleep", new_callable=AsyncMock):
        await indexes.delete("test-index", timeout=60)


@respx.mock
async def test_delete_timeout_raises(indexes: AsyncIndexes) -> None:
    respx.delete(f"{BASE_URL}/indexes/test-index").mock(return_value=httpx.Response(202))
    respx.get(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(200, json=make_index_response())
    )

    with (
        patch("pinecone.async_client.indexes.asyncio.sleep", new_callable=AsyncMock),
        patch(
            "pinecone.async_client.indexes.time.monotonic",
            side_effect=itertools.count(start=0.0, step=0.5).__next__,
        ),
        pytest.raises(PineconeTimeoutError, match="still exists after"),
    ):
        await indexes.delete("test-index", timeout=1)


async def test_delete_empty_name_raises(indexes: AsyncIndexes) -> None:
    with pytest.raises(PineconeValueError, match="name"):
        await indexes.delete("")
