"""Unit tests for Indexes namespace — list, describe, exists."""

from __future__ import annotations

import httpx
import pytest
import respx

from pinecone._internal.config import PineconeConfig
from pinecone._internal.constants import CONTROL_PLANE_API_VERSION
from pinecone._internal.http_client import HTTPClient
from pinecone.client.indexes import Indexes
from pinecone.errors.exceptions import NotFoundError, ValidationError
from pinecone.models.indexes.index import IndexModel
from pinecone.models.indexes.list import IndexList
from tests.factories import (
    make_error_response,
    make_index_list_response,
    make_index_response,
)

BASE_URL = "https://api.test.pinecone.io"


@pytest.fixture
def http_client() -> HTTPClient:
    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    return HTTPClient(config, CONTROL_PLANE_API_VERSION)


@pytest.fixture
def indexes(http_client: HTTPClient) -> Indexes:
    return Indexes(http=http_client)


# ---------------------------------------------------------------------------
# list()
# ---------------------------------------------------------------------------


@respx.mock
def test_list_indexes(indexes: Indexes) -> None:
    respx.get(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(200, json=make_index_list_response()),
    )

    result = indexes.list()

    assert isinstance(result, IndexList)
    assert len(result) == 1
    assert result[0].name == "test-index"
    assert result.names() == ["test-index"]

    # verify iteration
    names = [idx.name for idx in result]
    assert names == ["test-index"]


@respx.mock
def test_list_indexes_empty(indexes: Indexes) -> None:
    respx.get(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(200, json={"indexes": []}),
    )

    result = indexes.list()

    assert isinstance(result, IndexList)
    assert len(result) == 0
    assert result.names() == []


# ---------------------------------------------------------------------------
# describe()
# ---------------------------------------------------------------------------


@respx.mock
def test_describe_index(indexes: Indexes) -> None:
    respx.get(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(200, json=make_index_response()),
    )

    result = indexes.describe("test-index")

    assert isinstance(result, IndexModel)
    assert result.name == "test-index"
    assert result.host == "https://test-index-abc1234.svc.us-east1-gcp.pinecone.io"
    assert result.deletion_protection == "disabled"
    assert result.status.ready is True
    assert result.status.state == "Ready"
    assert result.deployment.cloud == "aws"  # type: ignore[union-attr]
    embedding = result.schema.fields["embedding"]
    assert embedding.dimension == 1536  # type: ignore[union-attr]
    assert embedding.metric == "cosine"  # type: ignore[union-attr]
    # bracket access
    assert result["name"] == "test-index"
    assert result["schema"] is result.schema


@respx.mock
def test_describe_index_caches_host(indexes: Indexes) -> None:
    respx.get(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(200, json=make_index_response()),
    )

    indexes.describe("test-index")

    assert "test-index" in indexes._host_cache
    assert (
        indexes._host_cache["test-index"]
        == "https://test-index-abc1234.svc.us-east1-gcp.pinecone.io"
    )


@respx.mock
def test_describe_nonexistent_index(indexes: Indexes) -> None:
    respx.get(f"{BASE_URL}/indexes/no-such-index").mock(
        return_value=httpx.Response(
            404,
            json=make_error_response(404, "Index not found"),
        ),
    )

    with pytest.raises(NotFoundError):
        indexes.describe("no-such-index")


def test_describe_empty_name_raises(indexes: Indexes) -> None:
    with pytest.raises(ValidationError) as exc_info:
        indexes.describe("")

    assert "name" in str(exc_info.value)
    assert "non-empty" in str(exc_info.value)


# ---------------------------------------------------------------------------
# exists()
# ---------------------------------------------------------------------------


@respx.mock
def test_exists_true(indexes: Indexes) -> None:
    respx.get(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(200, json=make_index_response()),
    )

    assert indexes.exists("test-index") is True


@respx.mock
def test_exists_false(indexes: Indexes) -> None:
    respx.get(f"{BASE_URL}/indexes/no-such-index").mock(
        return_value=httpx.Response(
            404,
            json=make_error_response(404, "Index not found"),
        ),
    )

    assert indexes.exists("no-such-index") is False


def test_exists_empty_name_returns_false(indexes: Indexes) -> None:
    assert indexes.exists("") is False


# ---------------------------------------------------------------------------
# resilient list decode
# ---------------------------------------------------------------------------


@respx.mock
def test_list_skips_unparseable_index_with_warning(
    indexes: Indexes, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    good = make_index_response(name="good-index")
    bad = make_index_response(name="bad-index")
    bad["schema"] = {"fields": {"weird": {"type": "quantum_vector"}}}
    respx.get(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(200, json={"indexes": [good, bad]}),
    )

    with caplog.at_level(logging.WARNING):
        result = indexes.list()

    assert result.names() == ["good-index"]
    assert any("bad-index" in record.getMessage() for record in caplog.records)


@respx.mock
def test_list_decodes_legacy_untyped_schema_fields(indexes: Indexes) -> None:
    legacy = make_index_response(name="legacy-index")
    legacy["schema"] = {"fields": {"old_meta": {"filterable": True}}}
    respx.get(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(200, json={"indexes": [legacy]}),
    )

    result = indexes.list()

    assert result.names() == ["legacy-index"]
    from pinecone.models.indexes.schema import LegacyMetadataField

    field = result[0].schema.fields["old_meta"]
    assert isinstance(field, LegacyMetadataField)
    assert field.filterable is True
