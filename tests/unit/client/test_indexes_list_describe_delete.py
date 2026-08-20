"""Unit tests for Indexes.list/describe/exists/delete (2026-07).

list() returns a Paginator over IndexModel; describe/exists/delete keep
their 2025-10 flows against 2026-07 response fixtures, including host-cache
maintenance and post-delete polling.
"""

from __future__ import annotations

import itertools
from unittest.mock import patch

import httpx
import pytest
import respx

from pinecone._internal.config import PineconeConfig
from pinecone._internal.constants import CONTROL_PLANE_API_VERSION
from pinecone._internal.http_client import HTTPClient
from pinecone.client.indexes import Indexes
from pinecone.errors.exceptions import (
    NotFoundError,
    PineconeTimeoutError,
    PineconeValueError,
)
from pinecone.models.indexes.index import IndexModel
from pinecone.models.pagination import Paginator
from tests.factories import make_error_response, make_index_list_response, make_index_response

BASE_URL = "https://api.test.pinecone.io"


@pytest.fixture
def http_client() -> HTTPClient:
    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    return HTTPClient(config, CONTROL_PLANE_API_VERSION)


@pytest.fixture
def indexes(http_client: HTTPClient) -> Indexes:
    return Indexes(http=http_client)


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


@respx.mock
def test_list_returns_paginator_over_index_models(indexes: Indexes) -> None:
    route = respx.get(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(200, json=make_index_list_response())
    )

    result = indexes.list()

    assert isinstance(result, Paginator)
    items = list(result)
    assert route.called
    assert len(items) == 1
    assert isinstance(items[0], IndexModel)
    assert items[0].name == "test-index"
    assert route.calls.last.request.headers.get("X-Pinecone-Api-Version") == (
        CONTROL_PLANE_API_VERSION
    )


@respx.mock
def test_list_is_lazy_until_iterated(indexes: Indexes) -> None:
    route = respx.get(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(200, json=make_index_list_response())
    )

    paginator = indexes.list()
    assert route.call_count == 0
    list(paginator)
    assert route.call_count == 1


@respx.mock
def test_list_empty(indexes: Indexes) -> None:
    respx.get(f"{BASE_URL}/indexes").mock(return_value=httpx.Response(200, json={"indexes": []}))

    assert list(indexes.list()) == []


@respx.mock
def test_list_yields_single_page(indexes: Indexes) -> None:
    respx.get(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(200, json=make_index_list_response())
    )

    pages = list(indexes.list().pages())

    assert len(pages) == 1
    assert pages[0].pagination_token is None
    assert len(pages[0].items) == 1


@respx.mock
def test_list_respects_limit(indexes: Indexes) -> None:
    payload = {
        "indexes": [
            make_index_response(name=f"index-{i}", host=f"index-{i}.svc.pinecone.io")
            for i in range(3)
        ]
    }
    respx.get(f"{BASE_URL}/indexes").mock(return_value=httpx.Response(200, json=payload))

    items = list(indexes.list(limit=2))

    assert [item.name for item in items] == ["index-0", "index-1"]


def test_list_zero_limit_raises(indexes: Indexes) -> None:
    with pytest.raises(PineconeValueError, match="limit"):
        indexes.list(limit=0)


@respx.mock
def test_list_skips_unparseable_index_with_warning(
    indexes: Indexes, caplog: pytest.LogCaptureFixture
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
        items = list(indexes.list())

    assert [item.name for item in items] == ["test-index"]
    assert "broken-index" in caplog.text


@respx.mock
def test_list_decodes_legacy_untyped_schema_fields(indexes: Indexes) -> None:
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

    items = list(indexes.list())

    assert len(items) == 1
    assert "genre" in items[0].schema.fields


# ---------------------------------------------------------------------------
# describe / exists
# ---------------------------------------------------------------------------


@respx.mock
def test_describe_index(indexes: Indexes) -> None:
    route = respx.get(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(200, json=make_index_response())
    )

    result = indexes.describe("test-index")

    assert route.called
    assert isinstance(result, IndexModel)
    assert result.name == "test-index"
    assert result.host == "https://test-index-abc1234.svc.us-east1-gcp.pinecone.io"


@respx.mock
def test_describe_caches_host(indexes: Indexes) -> None:
    respx.get(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(200, json=make_index_response())
    )

    indexes.describe("test-index")

    assert indexes._host_cache["test-index"] == (
        "https://test-index-abc1234.svc.us-east1-gcp.pinecone.io"
    )


@respx.mock
def test_describe_nonexistent_raises_not_found(indexes: Indexes) -> None:
    respx.get(f"{BASE_URL}/indexes/no-such-index").mock(
        return_value=httpx.Response(404, json=make_error_response(404, "Index not found"))
    )

    with pytest.raises(NotFoundError):
        indexes.describe("no-such-index")


def test_describe_empty_name_raises(indexes: Indexes) -> None:
    with pytest.raises(PineconeValueError, match="name"):
        indexes.describe("")


@respx.mock
def test_exists_true(indexes: Indexes) -> None:
    respx.get(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(200, json=make_index_response())
    )

    assert indexes.exists("test-index") is True


@respx.mock
def test_exists_false_on_404(indexes: Indexes) -> None:
    respx.get(f"{BASE_URL}/indexes/gone").mock(
        return_value=httpx.Response(404, json=make_error_response(404, "Index not found"))
    )

    assert indexes.exists("gone") is False


def test_exists_empty_name_raises(indexes: Indexes) -> None:
    """Graduated behavior: empty name raises instead of returning False."""
    with pytest.raises(PineconeValueError, match="name"):
        indexes.exists("")


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


@respx.mock
def test_delete_sends_delete_and_evicts_host_cache(indexes: Indexes) -> None:
    respx.get(f"{BASE_URL}/indexes/test-index").mock(
        side_effect=[
            httpx.Response(200, json=make_index_response()),
            httpx.Response(404, json=make_error_response(404, "Not found")),
        ]
    )
    delete_route = respx.delete(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(202)
    )

    indexes.describe("test-index")
    assert "test-index" in indexes._host_cache

    with patch("pinecone.client.indexes.time.sleep"):
        indexes.delete("test-index")

    assert delete_route.called
    assert "test-index" not in indexes._host_cache


@respx.mock
def test_delete_timeout_negative_one_skips_polling(indexes: Indexes) -> None:
    respx.delete(f"{BASE_URL}/indexes/test-index").mock(return_value=httpx.Response(202))
    describe_route = respx.get(f"{BASE_URL}/indexes/test-index")

    indexes.delete("test-index", timeout=-1)

    assert describe_route.call_count == 0


@respx.mock
def test_delete_polls_until_not_found(indexes: Indexes) -> None:
    respx.delete(f"{BASE_URL}/indexes/test-index").mock(return_value=httpx.Response(202))
    respx.get(f"{BASE_URL}/indexes/test-index").mock(
        side_effect=[
            httpx.Response(200, json=make_index_response()),
            httpx.Response(404, json=make_error_response(404, "Not found")),
        ]
    )

    with patch("pinecone.client.indexes.time.sleep"):
        indexes.delete("test-index", timeout=60)


@respx.mock
def test_delete_timeout_raises(indexes: Indexes) -> None:
    respx.delete(f"{BASE_URL}/indexes/test-index").mock(return_value=httpx.Response(202))
    respx.get(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(200, json=make_index_response())
    )

    with (
        patch("pinecone.client.indexes.time.sleep"),
        patch(
            "pinecone.client.indexes.time.monotonic",
            side_effect=itertools.count(start=0.0, step=0.5).__next__,
        ),
        pytest.raises(PineconeTimeoutError, match="still exists after"),
    ):
        indexes.delete("test-index", timeout=1)


def test_delete_empty_name_raises(indexes: Indexes) -> None:
    with pytest.raises(PineconeValueError, match="name"):
        indexes.delete("")
