"""Unit tests for AsyncIndex.upsert_records()."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from pinecone.async_client.async_index import AsyncIndex
from pinecone.errors.exceptions import PineconeTypeError, ValidationError
from pinecone.models.vectors.responses import UpsertRecordsResponse

INDEX_HOST = "my-index-abc123.svc.pinecone.io"
INDEX_HOST_HTTPS = f"https://{INDEX_HOST}"
UPSERT_URL = f"{INDEX_HOST_HTTPS}/records/namespaces/test-ns/upsert"


def _make_async_index() -> AsyncIndex:
    return AsyncIndex(host=INDEX_HOST, api_key="test-key")


class TestAsyncUpsertRecords:
    @respx.mock
    @pytest.mark.anyio
    async def test_async_upsert_records_basic(self) -> None:
        respx.post(UPSERT_URL).mock(return_value=httpx.Response(201, content=b""))
        idx = _make_async_index()
        result = await idx.upsert_records(
            namespace="test-ns",
            records=[{"_id": "r1", "text": "hello"}, {"_id": "r2", "text": "world"}],
        )
        assert isinstance(result, UpsertRecordsResponse)
        assert result.record_count == 2

    @respx.mock
    @pytest.mark.anyio
    async def test_async_upsert_records_ndjson_content_type(self) -> None:
        route = respx.post(UPSERT_URL).mock(return_value=httpx.Response(201, content=b""))
        idx = _make_async_index()
        await idx.upsert_records(
            namespace="test-ns",
            records=[{"_id": "r1", "text": "hello"}],
        )
        request = route.calls[0].request
        assert request.headers["content-type"] == "application/x-ndjson"

    @respx.mock
    @pytest.mark.anyio
    async def test_async_upsert_records_ndjson_body_format(self) -> None:
        route = respx.post(UPSERT_URL).mock(return_value=httpx.Response(201, content=b""))
        idx = _make_async_index()
        await idx.upsert_records(
            namespace="test-ns",
            records=[{"_id": "r1", "text": "hello"}, {"_id": "r2", "text": "world"}],
        )
        request = route.calls[0].request
        body = request.content.decode("utf-8")
        lines = body.strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"_id": "r1", "text": "hello"}
        assert json.loads(lines[1]) == {"_id": "r2", "text": "world"}

    @respx.mock
    @pytest.mark.anyio
    async def test_async_upsert_records_id_alias(self) -> None:
        route = respx.post(UPSERT_URL).mock(return_value=httpx.Response(201, content=b""))
        idx = _make_async_index()
        await idx.upsert_records(
            namespace="test-ns",
            records=[{"id": "r1", "text": "hello"}],
        )
        request = route.calls[0].request
        body = request.content.decode("utf-8")
        parsed = json.loads(body.strip())
        assert "_id" in parsed
        assert "id" not in parsed

    @pytest.mark.anyio
    async def test_async_upsert_records_namespace_not_string(self) -> None:
        idx = _make_async_index()
        with pytest.raises(ValidationError, match="namespace must be a string"):
            await idx.upsert_records(namespace=123, records=[{"_id": "r1"}])  # type: ignore[arg-type]

    @pytest.mark.anyio
    async def test_async_upsert_records_namespace_empty_string(self) -> None:
        idx = _make_async_index()
        with pytest.raises(ValidationError, match="namespace must be a non-empty string"):
            await idx.upsert_records(namespace="", records=[{"_id": "r1"}])

    @pytest.mark.anyio
    async def test_async_upsert_records_namespace_whitespace_only(self) -> None:
        idx = _make_async_index()
        with pytest.raises(ValidationError, match="namespace must be a non-empty string"):
            await idx.upsert_records(namespace="   ", records=[{"_id": "r1"}])

    @pytest.mark.anyio
    async def test_async_upsert_records_empty_list(self) -> None:
        idx = _make_async_index()
        with pytest.raises(ValidationError, match="records must be a non-empty list"):
            await idx.upsert_records(namespace="test-ns", records=[])

    @pytest.mark.anyio
    async def test_async_upsert_records_missing_id(self) -> None:
        idx = _make_async_index()
        with pytest.raises(ValidationError, match="must contain an '_id' or 'id' field"):
            await idx.upsert_records(
                namespace="test-ns",
                records=[{"text": "no id"}],
            )

    @pytest.mark.anyio
    async def test_async_upsert_records_keyword_only(self) -> None:
        idx = _make_async_index()
        with pytest.raises(TypeError):
            await idx.upsert_records([{"_id": "r1"}], "ns")  # type: ignore[misc]

    @respx.mock
    @pytest.mark.anyio
    async def test_async_upsert_records_both_id_fields_strips_id(self) -> None:
        """When both '_id' and 'id' are present, 'id' is dropped and '_id' wins."""
        route = respx.post(UPSERT_URL).mock(return_value=httpx.Response(201, content=b""))
        idx = _make_async_index()
        await idx.upsert_records(
            namespace="test-ns",
            records=[{"_id": "wins", "id": "loses", "text": "hello"}],
        )
        body = route.calls.last.request.content.decode("utf-8")
        parsed = json.loads(body.strip())
        assert parsed["_id"] == "wins"
        assert "id" not in parsed

    @pytest.mark.anyio
    async def test_async_upsert_records_id_must_be_string(self) -> None:
        """Records where '_id' is not a string raise ValidationError."""
        idx = _make_async_index()
        with pytest.raises(ValidationError, match="'_id' must be a string"):
            await idx.upsert_records(
                namespace="test-ns",
                records=[{"_id": 123, "text": "hello"}],
            )

    @pytest.mark.anyio
    async def test_async_upsert_records_positional_args_raise_teaching_error(self) -> None:
        """Positional misuse names the keyword-only call shape, not a bare TypeError."""
        idx = _make_async_index()
        with pytest.raises(PineconeTypeError) as excinfo:
            await idx.upsert_records("test-ns", [{"_id": "r1", "text": "hello"}])  # type: ignore[misc]
        message = str(excinfo.value)
        assert "keyword arguments only" in message
        assert 'upsert_records(namespace="...", records=[...])' in message
