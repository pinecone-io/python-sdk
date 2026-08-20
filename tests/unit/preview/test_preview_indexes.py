"""Unit tests for PreviewIndexes — create/describe/list/delete/configure/exists."""

from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest
import respx

from pinecone._internal.config import PineconeConfig
from pinecone.errors.exceptions import ForbiddenError, NotFoundError, PineconeTimeoutError
from pinecone.models.pagination import Page, Paginator
from pinecone.preview._internal.constants import INDEXES_API_VERSION
from pinecone.preview.indexes import PreviewIndexes
from pinecone.preview.models.indexes import PreviewIndexModel

BASE_URL = "https://api.test.pinecone.io"

_MINIMAL_SCHEMA: dict = {"fields": {"e": {"type": "dense_vector", "dimension": 4}}}

_INDEX_RESPONSE: dict = {
    "name": "test-index",
    "host": "test-index-xyz.svc.pinecone.io",
    "status": {"ready": True, "state": "Ready"},
    "schema": {
        "fields": {
            "e": {"type": "dense_vector", "dimension": 4},
            "title": {
                "type": "string",
                "full_text_search": {
                    "language": "en",
                    "stemming": False,
                    "lowercase": True,
                    "max_term_len": 40,
                    "stop_words": False,
                },
            },
        }
    },
    "deployment": {
        "deployment_type": "managed",
        "environment": "us-east-1-aws",
        "cloud": "aws",
        "region": "us-east-1",
    },
    "deletion_protection": "disabled",
    "tags": {"env": "test"},
}

_NOT_FOUND_RESPONSE: dict = {
    "error": {"code": "NOT_FOUND", "message": "Index 'test-index' not found."},
    "status": 404,
}

_FORBIDDEN_RESPONSE: dict = {
    "error": {"code": "FORBIDDEN", "message": "Deletion protection is enabled."},
    "status": 403,
}


@pytest.fixture
def indexes() -> PreviewIndexes:
    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    return PreviewIndexes(config=config)


# ---------------------------------------------------------------------------
# create
# ---------------------------------------------------------------------------


@respx.mock
def test_create_sends_api_version_header(indexes: PreviewIndexes) -> None:
    """create() sends X-Pinecone-Api-Version: 2026-01.alpha."""
    route = respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(201, json=_INDEX_RESPONSE)
    )

    indexes.create(schema=_MINIMAL_SCHEMA)

    assert route.called
    assert route.calls.last.request.headers.get("X-Pinecone-Api-Version") == INDEXES_API_VERSION


@respx.mock
def test_create_sends_correct_body_shape(indexes: PreviewIndexes) -> None:
    """create() sends schema in request body with None fields omitted."""
    import orjson

    route = respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(201, json=_INDEX_RESPONSE)
    )

    indexes.create(
        schema=_MINIMAL_SCHEMA,
        name="my-index",
        deletion_protection="enabled",
        tags={"env": "test"},
    )

    body = orjson.loads(route.calls.last.request.content)
    assert body["name"] == "my-index"
    assert body["deletion_protection"] == "enabled"
    assert body["tags"] == {"env": "test"}
    assert "read_capacity" not in body
    assert "deployment" not in body


@respx.mock
def test_create_returns_preview_index_model(indexes: PreviewIndexes) -> None:
    """create() deserializes the response into a PreviewIndexModel."""
    respx.post(f"{BASE_URL}/indexes").mock(return_value=httpx.Response(201, json=_INDEX_RESPONSE))

    result = indexes.create(schema=_MINIMAL_SCHEMA)

    assert isinstance(result, PreviewIndexModel)
    assert result.name == "test-index"
    assert result.host == "test-index-xyz.svc.pinecone.io"
    assert result.status.state == "Ready"
    assert result.status.ready is True


# ---------------------------------------------------------------------------
# describe
# ---------------------------------------------------------------------------


@respx.mock
def test_describe_parses_full_response(indexes: PreviewIndexes) -> None:
    """describe() parses a full server response including nested server-added defaults."""
    respx.get(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(200, json=_INDEX_RESPONSE)
    )

    result = indexes.describe("test-index")

    assert isinstance(result, PreviewIndexModel)
    assert result.name == "test-index"
    assert result.deletion_protection == "disabled"
    assert result.tags == {"env": "test"}

    from pinecone.preview.models.schema import PreviewStringField

    title_field = result.schema.fields["title"]
    assert isinstance(title_field, PreviewStringField)
    assert title_field.full_text_search is not None
    assert title_field.full_text_search.language == "en"
    assert title_field.full_text_search.lowercase is True
    assert title_field.full_text_search.max_term_len == 40


@respx.mock
def test_describe_raises_not_found_on_404(indexes: PreviewIndexes) -> None:
    """describe() raises NotFoundError when the server returns 404."""
    respx.get(f"{BASE_URL}/indexes/missing").mock(
        return_value=httpx.Response(404, json=_NOT_FOUND_RESPONSE)
    )

    with pytest.raises(NotFoundError):
        indexes.describe("missing")


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


@respx.mock
def test_list_paginator_yields_models_across_multiple_pages(indexes: PreviewIndexes) -> None:
    """list() paginator yields PreviewIndexModel instances; multiple page calls work correctly."""
    page1 = {
        "indexes": [
            {
                "name": "idx-1",
                "host": "idx-1.svc.pinecone.io",
                "status": {"ready": True, "state": "Ready"},
                "schema": {"fields": {"e": {"type": "dense_vector", "dimension": 4}}},
                "deployment": {
                    "deployment_type": "managed",
                    "environment": "us-east-1-aws",
                    "cloud": "aws",
                    "region": "us-east-1",
                },
                "deletion_protection": "disabled",
            }
        ]
    }
    page2 = {
        "indexes": [
            {
                "name": "idx-2",
                "host": "idx-2.svc.pinecone.io",
                "status": {"ready": False, "state": "Initializing"},
                "schema": {"fields": {"v": {"type": "dense_vector", "dimension": 8}}},
                "deployment": {
                    "deployment_type": "managed",
                    "environment": "us-east-1-aws",
                    "cloud": "aws",
                    "region": "us-east-1",
                },
                "deletion_protection": "disabled",
            }
        ]
    }

    respx.get(f"{BASE_URL}/indexes").mock(
        side_effect=[
            httpx.Response(200, json=page1),
            httpx.Response(200, json=page2),
        ]
    )

    result = indexes.list()
    assert isinstance(result, Paginator)

    pages = list(result.pages())
    assert len(pages) >= 1
    assert all(isinstance(item, PreviewIndexModel) for page in pages for item in page.items)


@respx.mock
def test_list_returns_paginator_type(indexes: PreviewIndexes) -> None:
    """list() returns a Paginator over PreviewIndexModel."""
    respx.get(f"{BASE_URL}/indexes").mock(return_value=httpx.Response(200, json={"indexes": []}))

    result = indexes.list()
    assert isinstance(result, Paginator)


@respx.mock
def test_list_pages_each_have_items(indexes: PreviewIndexes) -> None:
    """list().pages() yields Page objects with items list."""
    respx.get(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(200, json={"indexes": [_INDEX_RESPONSE]})
    )

    pages = list(indexes.list().pages())
    assert len(pages) == 1
    assert isinstance(pages[0], Page)
    assert len(pages[0].items) == 1
    assert isinstance(pages[0].items[0], PreviewIndexModel)


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


@respx.mock
def test_delete_polls_until_404(indexes: PreviewIndexes) -> None:
    """delete() polls describe() until NotFoundError, then returns None."""
    respx.delete(f"{BASE_URL}/indexes/test-index").mock(return_value=httpx.Response(204))

    describe_route = respx.get(f"{BASE_URL}/indexes/test-index")
    describe_route.side_effect = [
        httpx.Response(200, json=_INDEX_RESPONSE),
        httpx.Response(200, json=_INDEX_RESPONSE),
        httpx.Response(404, json=_NOT_FOUND_RESPONSE),
    ]

    with patch("time.sleep"):
        result = indexes.delete("test-index")

    assert result is None
    assert describe_route.call_count == 3


@respx.mock
def test_delete_returns_immediately_with_timeout_minus_one(indexes: PreviewIndexes) -> None:
    """delete("name", timeout=-1) returns without polling."""
    respx.delete(f"{BASE_URL}/indexes/test-index").mock(return_value=httpx.Response(204))

    with patch("time.sleep", side_effect=AssertionError("should not sleep")):
        indexes.delete("test-index", timeout=-1)


@respx.mock
def test_delete_raises_timeout_error(indexes: PreviewIndexes) -> None:
    """delete() raises PineconeTimeoutError when the timeout expires."""
    respx.delete(f"{BASE_URL}/indexes/test-index").mock(return_value=httpx.Response(204))
    respx.get(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(200, json=_INDEX_RESPONSE)
    )

    monotonic_values = iter([0.0, 10.0])
    with (
        patch("time.sleep"),
        patch("time.monotonic", side_effect=monotonic_values),
        pytest.raises(PineconeTimeoutError),
    ):
        indexes.delete("test-index", timeout=5)


@respx.mock
def test_delete_raises_forbidden_when_protection_enabled(indexes: PreviewIndexes) -> None:
    """delete() raises ForbiddenError when the server returns 403."""
    respx.delete(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(403, json=_FORBIDDEN_RESPONSE)
    )

    with pytest.raises(ForbiddenError):
        indexes.delete("test-index", timeout=-1)


# ---------------------------------------------------------------------------
# configure
# ---------------------------------------------------------------------------


@respx.mock
def test_configure_patch_only_sends_provided_fields(indexes: PreviewIndexes) -> None:
    """configure() only sends fields that are not None in the PATCH body."""
    import orjson

    route = respx.patch(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(200, json=_INDEX_RESPONSE)
    )

    indexes.configure("test-index", deletion_protection="enabled")

    body = orjson.loads(route.calls.last.request.content)
    assert body == {"deletion_protection": "enabled"}
    assert "schema" not in body
    assert "tags" not in body
    assert "read_capacity" not in body


@respx.mock
def test_configure_sends_multiple_fields(indexes: PreviewIndexes) -> None:
    """configure() with multiple args sends all of them in the PATCH body."""
    import orjson

    route = respx.patch(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(200, json=_INDEX_RESPONSE)
    )

    indexes.configure(
        "test-index",
        deletion_protection="disabled",
        tags={"env": "prod"},
    )

    body = orjson.loads(route.calls.last.request.content)
    assert body["deletion_protection"] == "disabled"
    assert body["tags"] == {"env": "prod"}


@respx.mock
def test_configure_returns_preview_index_model(indexes: PreviewIndexes) -> None:
    """configure() returns an updated PreviewIndexModel."""
    respx.patch(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(200, json=_INDEX_RESPONSE)
    )

    result = indexes.configure("test-index", deletion_protection="disabled")

    assert isinstance(result, PreviewIndexModel)


# ---------------------------------------------------------------------------
# exists
# ---------------------------------------------------------------------------


@respx.mock
def test_exists_returns_false_on_404(indexes: PreviewIndexes) -> None:
    """exists() returns False when describe() raises NotFoundError."""
    respx.get(f"{BASE_URL}/indexes/missing").mock(
        return_value=httpx.Response(404, json=_NOT_FOUND_RESPONSE)
    )

    assert indexes.exists("missing") is False


@respx.mock
def test_exists_returns_true_when_index_found(indexes: PreviewIndexes) -> None:
    """exists() returns True when describe() succeeds."""
    respx.get(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(200, json=_INDEX_RESPONSE)
    )

    assert indexes.exists("test-index") is True


# ---------------------------------------------------------------------------
# Forward-compatibility: unknown field types and extra options pass through
# ---------------------------------------------------------------------------


@respx.mock
def test_create_accepts_unknown_field_type(indexes: PreviewIndexes) -> None:
    """create() does not raise ValidationError for unknown field types (escape hatch)."""
    import orjson

    from pinecone.schema_builder import SchemaBuilder

    route = respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(201, json=_INDEX_RESPONSE)
    )

    schema = SchemaBuilder().add_custom_field("x", {"type": "new_type"}).build()
    result = indexes.create(schema=schema, name="i")

    assert isinstance(result, PreviewIndexModel)
    body = orjson.loads(route.calls.last.request.content)
    assert body["schema"]["fields"]["x"] == {"type": "new_type"}


# ---------------------------------------------------------------------------
# Preview namespace wiring
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# source_collection / source_backup_id / cmek_id parameters
# ---------------------------------------------------------------------------


def test_preview_create_index_request_source_fields() -> None:
    """PreviewCreateIndexRequest serializes source_collection, source_backup_id, cmek_id."""
    import orjson

    from pinecone.preview._internal.adapters.indexes import PreviewCreateIndexAdapter
    from pinecone.preview.models.requests import PreviewCreateIndexRequest

    req = PreviewCreateIndexRequest(
        schema={"fields": {"vec": {"type": "dense_vector", "dimension": 128}}},
        source_collection="my-collection",
        source_backup_id="backup-abc",
        cmek_id="key-xyz",
    )
    body = orjson.loads(PreviewCreateIndexAdapter.to_request(req))
    assert body["source_collection"] == "my-collection"
    assert body["source_backup_id"] == "backup-abc"
    assert body["cmek_id"] == "key-xyz"


@respx.mock
def test_preview_create_sends_source_fields_in_body(indexes: PreviewIndexes) -> None:
    """create() sends source_collection, source_backup_id, and cmek_id in request body."""
    import orjson

    route = respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(201, json=_INDEX_RESPONSE)
    )

    indexes.create(
        schema=_MINIMAL_SCHEMA,
        source_collection="my-collection",
        source_backup_id="backup-abc",
        cmek_id="key-xyz",
    )

    body = orjson.loads(route.calls.last.request.content)
    assert body["source_collection"] == "my-collection"
    assert body["source_backup_id"] == "backup-abc"
    assert body["cmek_id"] == "key-xyz"


@respx.mock
def test_preview_create_omits_source_fields_when_not_set(indexes: PreviewIndexes) -> None:
    """create() omits source_collection, source_backup_id, cmek_id when not provided."""
    import orjson

    route = respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(201, json=_INDEX_RESPONSE)
    )

    indexes.create(schema=_MINIMAL_SCHEMA)

    body = orjson.loads(route.calls.last.request.content)
    assert "source_collection" not in body
    assert "source_backup_id" not in body
    assert "cmek_id" not in body


def test_preview_indexes_property_is_lazily_initialized() -> None:
    """Preview.indexes returns a cached PreviewIndexes instance."""
    from pinecone._internal.config import PineconeConfig as _PineconeConfig
    from pinecone._internal.http_client import HTTPClient
    from pinecone.preview import Preview

    config = _PineconeConfig(api_key="test-key", host=BASE_URL)
    http = HTTPClient(config, "2025-10")
    preview = Preview(http=http, config=config)

    first = preview.indexes
    second = preview.indexes

    assert isinstance(first, PreviewIndexes)
    assert first is second
