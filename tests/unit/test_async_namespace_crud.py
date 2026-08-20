"""Unit tests for AsyncIndex.create_namespace, describe_namespace, delete_namespace."""

from __future__ import annotations

import httpx
import orjson
import pytest
import respx

from pinecone import AsyncIndex
from pinecone.errors.exceptions import ValidationError
from pinecone.models.namespaces.models import NamespaceDescription

INDEX_HOST = "test-index-abc1234.svc.us-east1-gcp.pinecone.io"
INDEX_HOST_HTTPS = f"https://{INDEX_HOST}"
NS_URL = f"{INDEX_HOST_HTTPS}/namespaces"


def _make_async_index() -> AsyncIndex:
    return AsyncIndex(host=INDEX_HOST, api_key="test-key")


# ---------------------------------------------------------------------------
# create_namespace
# ---------------------------------------------------------------------------


class TestAsyncCreateNamespace:
    """Async create namespace (unified-ns-0001, unified-ns-0002)."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_create_namespace_basic(self) -> None:
        respx.post(NS_URL).mock(
            return_value=httpx.Response(200, json={"name": "ns1", "record_count": 0}),
        )
        idx = _make_async_index()
        result = await idx.create_namespace(name="ns1")

        assert isinstance(result, NamespaceDescription)
        assert result.name == "ns1"
        assert result.record_count == 0

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_create_namespace_with_schema(self) -> None:
        respx.post(NS_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "name": "ns1",
                    "record_count": 0,
                    "schema": {"fields": {"genre": {"filterable": True}}},
                },
            ),
        )
        idx = _make_async_index()
        result = await idx.create_namespace(
            name="ns1",
            schema={"fields": {"genre": {"filterable": True}}},
        )

        assert result.schema is not None
        assert "genre" in result.schema.fields
        assert result.schema.fields["genre"].filterable is True

        # Verify schema was included in request body
        body = orjson.loads(respx.calls.last.request.content)
        assert body["name"] == "ns1"
        assert body["schema"] == {"fields": {"genre": {"filterable": True}}}

    @pytest.mark.asyncio
    async def test_async_create_namespace_empty_name_raises(self) -> None:
        idx = _make_async_index()
        with pytest.raises(ValidationError, match="non-empty string"):
            await idx.create_namespace(name="")

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_create_namespace_whitespace_name_is_sent(self) -> None:
        """A whitespace-only name is legal: the rule is ASCII/no-NUL/1-512, not "non-blank".

        ``db_data_2026-07.oas.yaml:2039`` (``^[\\x01-\\x7F]+$``) and
        ``pc-validation/src/data_plane/mod.rs:66-82`` both accept it, so the SDK
        must not refuse a name the server would take.
        """
        route = respx.post(NS_URL).mock(
            return_value=httpx.Response(200, json={"name": "  ", "record_count": 0}),
        )
        idx = _make_async_index()
        result = await idx.create_namespace(name="  ")

        assert result.name == "  "
        assert orjson.loads(route.calls.last.request.content)["name"] == "  "


# ---------------------------------------------------------------------------
# describe_namespace
# ---------------------------------------------------------------------------


class TestAsyncDescribeNamespace:
    """Async describe namespace (unified-ns-0003)."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_describe_namespace(self) -> None:
        respx.get(f"{NS_URL}/ns1").mock(
            return_value=httpx.Response(200, json={"name": "ns1", "record_count": 500}),
        )
        idx = _make_async_index()
        result = await idx.describe_namespace(name="ns1")

        assert isinstance(result, NamespaceDescription)
        assert result.name == "ns1"
        assert result.record_count == 500

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_describe_namespace_accepts_legacy_namespace_kwarg(self) -> None:
        respx.get(f"{NS_URL}/movies").mock(
            return_value=httpx.Response(200, json={"name": "movies", "record_count": 10}),
        )
        idx = _make_async_index()
        result = await idx.describe_namespace(namespace="movies")

        assert isinstance(result, NamespaceDescription)
        assert result.name == "movies"
        assert result.record_count == 10

    @pytest.mark.asyncio
    async def test_async_describe_namespace_rejects_both_kwargs(self) -> None:
        from pinecone.errors.exceptions import ValidationError

        idx = _make_async_index()
        with pytest.raises(ValidationError, match="either name= or namespace="):
            await idx.describe_namespace(name="a", namespace="b")  # type: ignore[call-arg]

    @pytest.mark.asyncio
    async def test_async_describe_namespace_rejects_neither_kwarg(self) -> None:
        idx = _make_async_index()
        with pytest.raises(ValidationError, match="non-empty string"):
            await idx.describe_namespace()  # type: ignore[call-arg]

    @pytest.mark.asyncio
    async def test_async_describe_namespace_rejects_unknown_kwargs(self) -> None:
        idx = _make_async_index()
        with pytest.raises(TypeError, match="unexpected keyword arguments"):
            await idx.describe_namespace(name="x", bogus="y")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# delete_namespace
# ---------------------------------------------------------------------------


class TestAsyncDeleteNamespace:
    """Async delete namespace (unified-ns-0004)."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_delete_namespace(self) -> None:
        route = respx.delete(f"{NS_URL}/ns1").mock(
            return_value=httpx.Response(200, json={}),
        )
        idx = _make_async_index()
        result = await idx.delete_namespace(name="ns1")

        assert result is None
        assert route.called

    @respx.mock
    @pytest.mark.asyncio
    async def test_delete_namespace_accepts_legacy_namespace_kwarg(self) -> None:
        route = respx.delete(f"{NS_URL}/movies").mock(
            return_value=httpx.Response(200, json={}),
        )
        idx = _make_async_index()
        result = await idx.delete_namespace(namespace="movies")  # type: ignore[call-arg]

        assert result is None
        assert route.called

    @pytest.mark.asyncio
    async def test_delete_namespace_rejects_both_kwargs(self) -> None:
        idx = _make_async_index()
        with pytest.raises(ValidationError, match="either name= or namespace="):
            await idx.delete_namespace(name="a", namespace="b")  # type: ignore[call-arg]

    @pytest.mark.asyncio
    async def test_delete_namespace_rejects_neither_kwarg(self) -> None:
        idx = _make_async_index()
        with pytest.raises(ValidationError, match="non-empty string"):
            await idx.delete_namespace()  # type: ignore[call-arg]

    @pytest.mark.asyncio
    async def test_delete_namespace_rejects_unknown_kwargs(self) -> None:
        idx = _make_async_index()
        with pytest.raises(TypeError, match="unexpected keyword arguments"):
            await idx.delete_namespace(name="x", bogus="y")  # type: ignore[call-arg]

    @respx.mock
    @pytest.mark.asyncio
    async def test_delete_namespace_legacy_kwarg_with_timeout(self) -> None:
        route = respx.delete(f"{NS_URL}/x").mock(
            return_value=httpx.Response(200, json={}),
        )
        idx = _make_async_index()
        await idx.delete_namespace(namespace="x", timeout=5.0)  # type: ignore[call-arg]

        assert route.called


# ---------------------------------------------------------------------------
# Validation shared across methods
# ---------------------------------------------------------------------------


class TestAsyncNamespaceValidation:
    """Validation: name must be a string and non-empty (unified-ns-0010, unified-ns-0011)."""

    @pytest.mark.asyncio
    async def test_async_namespace_name_not_string_raises(self) -> None:
        idx = _make_async_index()
        with pytest.raises(ValidationError, match="must be a string"):
            await idx.create_namespace(name=123)  # type: ignore[arg-type]

        with pytest.raises(ValidationError, match="must be a string"):
            await idx.describe_namespace(name=123)  # type: ignore[arg-type]

        with pytest.raises(ValidationError, match="must be a string"):
            await idx.delete_namespace(name=123)  # type: ignore[arg-type]
