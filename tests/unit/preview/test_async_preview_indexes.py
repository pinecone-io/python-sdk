"""Unit tests for AsyncPreviewIndexes — create/describe/list/delete/configure/exists."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest
import respx

from pinecone._internal.config import PineconeConfig
from pinecone.errors.exceptions import ForbiddenError, NotFoundError, PineconeTimeoutError
from pinecone.preview._internal.constants import INDEXES_API_VERSION
from pinecone.preview.async_indexes import AsyncPreviewIndexes
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
def indexes() -> AsyncPreviewIndexes:
    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    return AsyncPreviewIndexes(config=config)


# ---------------------------------------------------------------------------
# create
# ---------------------------------------------------------------------------


@respx.mock
async def test_create_sends_api_version_header(indexes: AsyncPreviewIndexes) -> None:
    """create() sends X-Pinecone-Api-Version: 2026-01.alpha."""
    route = respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(201, json=_INDEX_RESPONSE)
    )

    await indexes.create(schema=_MINIMAL_SCHEMA)

    assert route.called
    assert route.calls.last.request.headers.get("X-Pinecone-Api-Version") == INDEXES_API_VERSION


@respx.mock
async def test_create_sends_correct_body_shape(indexes: AsyncPreviewIndexes) -> None:
    """create() sends schema in request body with None fields omitted."""
    import orjson

    route = respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(201, json=_INDEX_RESPONSE)
    )

    await indexes.create(
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
async def test_create_returns_preview_index_model(indexes: AsyncPreviewIndexes) -> None:
    """create() deserializes the response into a PreviewIndexModel."""
    respx.post(f"{BASE_URL}/indexes").mock(return_value=httpx.Response(201, json=_INDEX_RESPONSE))

    result = await indexes.create(schema=_MINIMAL_SCHEMA)

    assert isinstance(result, PreviewIndexModel)
    assert result.name == "test-index"
    assert result.host == "test-index-xyz.svc.pinecone.io"
    assert result.status.state == "Ready"
    assert result.status.ready is True


# ---------------------------------------------------------------------------
# describe
# ---------------------------------------------------------------------------


@respx.mock
async def test_describe_parses_full_response(indexes: AsyncPreviewIndexes) -> None:
    """describe() parses a full server response including nested server-added defaults."""
    respx.get(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(200, json=_INDEX_RESPONSE)
    )

    result = await indexes.describe("test-index")

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
async def test_describe_raises_not_found_on_404(indexes: AsyncPreviewIndexes) -> None:
    """describe() raises NotFoundError when the server returns 404."""
    respx.get(f"{BASE_URL}/indexes/missing").mock(
        return_value=httpx.Response(404, json=_NOT_FOUND_RESPONSE)
    )

    with pytest.raises(NotFoundError):
        await indexes.describe("missing")


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


@respx.mock
async def test_delete_polls_until_404(indexes: AsyncPreviewIndexes) -> None:
    """delete() polls describe() until NotFoundError, then returns None."""
    respx.delete(f"{BASE_URL}/indexes/test-index").mock(return_value=httpx.Response(204))

    describe_route = respx.get(f"{BASE_URL}/indexes/test-index")
    describe_route.side_effect = [
        httpx.Response(200, json=_INDEX_RESPONSE),
        httpx.Response(200, json=_INDEX_RESPONSE),
        httpx.Response(404, json=_NOT_FOUND_RESPONSE),
    ]

    with patch("asyncio.sleep", new_callable=AsyncMock):
        result = await indexes.delete("test-index")

    assert result is None
    assert describe_route.call_count == 3


@respx.mock
async def test_delete_returns_immediately_with_timeout_minus_one(
    indexes: AsyncPreviewIndexes,
) -> None:
    """delete("name", timeout=-1) returns without polling."""
    respx.delete(f"{BASE_URL}/indexes/test-index").mock(return_value=httpx.Response(204))

    async def _should_not_sleep(_: float) -> None:
        raise AssertionError("should not sleep")

    with patch("asyncio.sleep", side_effect=_should_not_sleep):
        await indexes.delete("test-index", timeout=-1)


@respx.mock
async def test_delete_raises_timeout_error(indexes: AsyncPreviewIndexes) -> None:
    """delete() raises PineconeTimeoutError when the timeout expires."""
    respx.delete(f"{BASE_URL}/indexes/test-index").mock(return_value=httpx.Response(204))
    respx.get(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(200, json=_INDEX_RESPONSE)
    )

    monotonic_values = iter([0.0, 10.0])
    with (
        patch("asyncio.sleep", new_callable=AsyncMock),
        patch("time.monotonic", side_effect=monotonic_values),
        pytest.raises(PineconeTimeoutError),
    ):
        await indexes.delete("test-index", timeout=5)


@respx.mock
async def test_delete_raises_forbidden_when_protection_enabled(
    indexes: AsyncPreviewIndexes,
) -> None:
    """delete() raises ForbiddenError when the server returns 403."""
    respx.delete(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(403, json=_FORBIDDEN_RESPONSE)
    )

    with pytest.raises(ForbiddenError):
        await indexes.delete("test-index", timeout=-1)


# ---------------------------------------------------------------------------
# configure
# ---------------------------------------------------------------------------


@respx.mock
async def test_configure_patch_only_sends_provided_fields(indexes: AsyncPreviewIndexes) -> None:
    """configure() only sends fields that are not None in the PATCH body."""
    import orjson

    route = respx.patch(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(200, json=_INDEX_RESPONSE)
    )

    await indexes.configure("test-index", deletion_protection="enabled")

    body = orjson.loads(route.calls.last.request.content)
    assert body == {"deletion_protection": "enabled"}
    assert "schema" not in body
    assert "tags" not in body
    assert "read_capacity" not in body


@respx.mock
async def test_configure_sends_multiple_fields(indexes: AsyncPreviewIndexes) -> None:
    """configure() with multiple args sends all of them in the PATCH body."""
    import orjson

    route = respx.patch(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(200, json=_INDEX_RESPONSE)
    )

    await indexes.configure(
        "test-index",
        deletion_protection="disabled",
        tags={"env": "prod"},
    )

    body = orjson.loads(route.calls.last.request.content)
    assert body["deletion_protection"] == "disabled"
    assert body["tags"] == {"env": "prod"}


@respx.mock
async def test_configure_returns_preview_index_model(indexes: AsyncPreviewIndexes) -> None:
    """configure() returns an updated PreviewIndexModel."""
    respx.patch(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(200, json=_INDEX_RESPONSE)
    )

    result = await indexes.configure("test-index", deletion_protection="disabled")

    assert isinstance(result, PreviewIndexModel)


# ---------------------------------------------------------------------------
# exists
# ---------------------------------------------------------------------------


@respx.mock
async def test_exists_returns_false_on_404(indexes: AsyncPreviewIndexes) -> None:
    """exists() returns False when describe() raises NotFoundError."""
    respx.get(f"{BASE_URL}/indexes/missing").mock(
        return_value=httpx.Response(404, json=_NOT_FOUND_RESPONSE)
    )

    assert await indexes.exists("missing") is False


@respx.mock
async def test_exists_returns_true_when_index_found(indexes: AsyncPreviewIndexes) -> None:
    """exists() returns True when describe() succeeds."""
    respx.get(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(200, json=_INDEX_RESPONSE)
    )

    assert await indexes.exists("test-index") is True


# ---------------------------------------------------------------------------
# Forward-compatibility: unknown field types and extra options pass through
# ---------------------------------------------------------------------------


@respx.mock
async def test_async_create_accepts_unknown_field_type(indexes: AsyncPreviewIndexes) -> None:
    """create() does not raise ValidationError for unknown field types (escape hatch)."""
    import orjson

    from pinecone.schema_builder import SchemaBuilder

    route = respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(201, json=_INDEX_RESPONSE)
    )

    schema = SchemaBuilder().add_custom_field("x", {"type": "new_type"}).build()
    result = await indexes.create(schema=schema, name="i")

    assert isinstance(result, PreviewIndexModel)
    body = orjson.loads(route.calls.last.request.content)
    assert body["schema"]["fields"]["x"] == {"type": "new_type"}


# ---------------------------------------------------------------------------
# AsyncPreview namespace wiring
# ---------------------------------------------------------------------------


def test_async_preview_indexes_property_is_lazily_initialized() -> None:
    """AsyncPreview.indexes returns a cached AsyncPreviewIndexes instance."""
    from pinecone._internal.config import PineconeConfig as _PineconeConfig
    from pinecone._internal.http_client import AsyncHTTPClient
    from pinecone.preview import AsyncPreview

    config = _PineconeConfig(api_key="test-key", host=BASE_URL)
    http = AsyncHTTPClient(config, "2025-10")
    preview = AsyncPreview(http=http, config=config)

    first = preview.indexes
    second = preview.indexes

    assert isinstance(first, AsyncPreviewIndexes)
    assert first is second
