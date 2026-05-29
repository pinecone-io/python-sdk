"""Unit tests for HTTPClient and _raise_for_status."""

from __future__ import annotations

import logging
import random
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import orjson
import pytest
import respx

from pinecone._internal.config import PineconeConfig, RetryConfig
from pinecone._internal.constants import API_VERSION_HEADER
from pinecone._internal.http_client import (
    AsyncHTTPClient,
    HTTPClient,
    _AsyncRetryTransport,
    _build_headers,
    _compute_backoff,
    _compute_retry_after_delay,
    _encode_json,
    _log_curl,
    _prepare_json_kwargs,
    _raise_for_status,
    _redact_headers,
    _RetryTransport,
)
from pinecone.errors.exceptions import (
    ApiError,
    ConflictError,
    ForbiddenError,
    NotFoundError,
    PineconeConnectionError,
    PineconeError,
    PineconeTimeoutError,
    ServiceError,
    UnauthorizedError,
)

# ---------------------------------------------------------------------------
# _build_headers
# ---------------------------------------------------------------------------


class TestBuildHeaders:
    def test_includes_api_key(self) -> None:
        config = PineconeConfig(api_key="test-key")
        headers = _build_headers(config, "2025-10")
        assert headers["Api-Key"] == "test-key"

    def test_includes_api_version_header(self) -> None:
        config = PineconeConfig(api_key="k")
        headers = _build_headers(config, "2025-10")
        assert headers[API_VERSION_HEADER] == "2025-10"

    def test_includes_user_agent(self) -> None:
        config = PineconeConfig(api_key="k")
        headers = _build_headers(config, "2025-10")
        assert "User-Agent" in headers
        assert headers["User-Agent"].startswith("python-client-")

    def test_merges_additional_headers(self) -> None:
        config = PineconeConfig(api_key="k", additional_headers={"X-Custom": "value"})
        headers = _build_headers(config, "2025-10")
        assert headers["X-Custom"] == "value"

    def test_additional_headers_override_defaults(self) -> None:
        """User-supplied additional headers can override built-in headers."""
        config = PineconeConfig(api_key="k", additional_headers={"User-Agent": "custom-agent"})
        headers = _build_headers(config, "2025-10")
        assert headers["User-Agent"] == "custom-agent"

    def test_no_additional_headers(self) -> None:
        config = PineconeConfig(api_key="k")
        headers = _build_headers(config, "2025-10")
        # Should have exactly the three standard headers
        assert set(headers.keys()) == {"Api-Key", API_VERSION_HEADER, "User-Agent"}


# ---------------------------------------------------------------------------
# _raise_for_status
# ---------------------------------------------------------------------------


class TestRaiseForStatus:
    @pytest.mark.parametrize(
        ("status_code", "exception_type"),
        [
            (401, UnauthorizedError),
            (403, ForbiddenError),
            (404, NotFoundError),
            (409, ConflictError),
            (500, ServiceError),
            (502, ServiceError),
            (422, ApiError),
        ],
    )
    def test_status_code_to_exception(
        self, status_code: int, exception_type: type[ApiError]
    ) -> None:
        """Each status code maps to its specific exception type."""
        response = httpx.Response(status_code, json={"message": "oops"})
        with pytest.raises(exception_type) as exc_info:
            _raise_for_status(response)
        assert exc_info.value.status_code == status_code

    def test_extracts_message_from_json_body(self) -> None:
        response = httpx.Response(400, json={"message": "bad input"})
        with pytest.raises(ApiError) as exc_info:
            _raise_for_status(response)
        assert exc_info.value.message == "bad input"

    def test_fallback_message_when_body_not_json(self) -> None:
        response = httpx.Response(500, text="Internal Server Error")
        with pytest.raises(ApiError) as exc_info:
            _raise_for_status(response)
        assert exc_info.value.message == "Internal Server Error"

    def test_fallback_message_when_no_message_key(self) -> None:
        # body["error"] is a string (not a dict) — not a canonical Pinecone error shape.
        # New extraction falls through to raw text body (the JSON string itself).
        response = httpx.Response(500, json={"error": "something"})
        with pytest.raises(ApiError) as exc_info:
            _raise_for_status(response)
        assert exc_info.value.status_code == 500
        assert exc_info.value.message  # non-empty; exact text is implementation detail

    def test_body_attribute_contains_parsed_json(self) -> None:
        body = {"message": "conflict", "details": "already exists"}
        response = httpx.Response(409, json=body)
        with pytest.raises(ConflictError) as exc_info:
            _raise_for_status(response)
        assert exc_info.value.body == body

    def test_body_is_none_when_not_json(self) -> None:
        response = httpx.Response(500, text="oops")
        with pytest.raises(ApiError) as exc_info:
            _raise_for_status(response)
        assert exc_info.value.body is None

    def test_success_response_does_not_raise(self) -> None:
        response = httpx.Response(200, json={"ok": True})
        _raise_for_status(response)  # Should not raise

    def test_raise_for_status_403_forbidden(self) -> None:
        response = httpx.Response(403, json={"message": "Forbidden"})
        with pytest.raises(ForbiddenError) as exc_info:
            _raise_for_status(response)
        assert exc_info.value.status_code == 403

    def test_raise_for_status_500_service_error(self) -> None:
        response = httpx.Response(500, json={"message": "Internal error"})
        with pytest.raises(ServiceError) as exc_info:
            _raise_for_status(response)
        assert exc_info.value.status_code == 500

    def test_raise_for_status_502_service_error(self) -> None:
        response = httpx.Response(502)
        with pytest.raises(ServiceError) as exc_info:
            _raise_for_status(response)
        assert exc_info.value.status_code == 502

    def test_raise_for_status_503_service_error(self) -> None:
        response = httpx.Response(503)
        with pytest.raises(ServiceError) as exc_info:
            _raise_for_status(response)
        assert exc_info.value.status_code == 503

    def test_raise_for_status_422_generic_api_error(self) -> None:
        response = httpx.Response(422)
        with pytest.raises(ApiError) as exc_info:
            _raise_for_status(response)
        assert not isinstance(exc_info.value, ServiceError)

    def test_service_error_inherits_from_api_error(self) -> None:
        assert issubclass(ServiceError, ApiError)
        assert issubclass(ServiceError, PineconeError)

    def test_forbidden_error_inherits_from_api_error(self) -> None:
        assert issubclass(ForbiddenError, ApiError)
        assert issubclass(ForbiddenError, PineconeError)


# ---------------------------------------------------------------------------
# HTTPClient — sync
# ---------------------------------------------------------------------------

BASE_URL = "https://api.pinecone.io"


def _make_sync_client() -> HTTPClient:
    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    return HTTPClient(config, api_version="2025-10")


class TestHTTPClientGet:
    @respx.mock
    def test_get_success(self) -> None:
        respx.get(f"{BASE_URL}/indexes").mock(
            return_value=httpx.Response(200, json={"indexes": []})
        )
        client = _make_sync_client()
        resp = client.get("/indexes")
        assert resp.status_code == 200
        assert resp.json() == {"indexes": []}

    @respx.mock
    def test_get_raises_on_error(self) -> None:
        respx.get(f"{BASE_URL}/indexes/missing").mock(
            return_value=httpx.Response(404, json={"message": "not found"})
        )
        client = _make_sync_client()
        with pytest.raises(NotFoundError):
            client.get("/indexes/missing")


class TestHTTPClientPost:
    @respx.mock
    def test_post_success(self) -> None:
        respx.post(f"{BASE_URL}/indexes").mock(
            return_value=httpx.Response(201, json={"name": "my-index"})
        )
        client = _make_sync_client()
        resp = client.post("/indexes", json={"name": "my-index"})
        assert resp.status_code == 201

    @respx.mock
    def test_post_raises_on_conflict(self) -> None:
        respx.post(f"{BASE_URL}/indexes").mock(
            return_value=httpx.Response(409, json={"message": "already exists"})
        )
        client = _make_sync_client()
        with pytest.raises(ConflictError):
            client.post("/indexes", json={"name": "dup"})


class TestHTTPClientDelete:
    @respx.mock
    def test_delete_success(self) -> None:
        respx.delete(f"{BASE_URL}/indexes/foo").mock(return_value=httpx.Response(202, json={}))
        client = _make_sync_client()
        resp = client.delete("/indexes/foo")
        assert resp.status_code == 202

    @respx.mock
    def test_delete_raises_on_unauthorized(self) -> None:
        respx.delete(f"{BASE_URL}/indexes/foo").mock(
            return_value=httpx.Response(401, json={"message": "bad key"})
        )
        client = _make_sync_client()
        with pytest.raises(UnauthorizedError):
            client.delete("/indexes/foo")


class TestHTTPClientClose:
    def test_close_closes_underlying_client(self) -> None:
        client = _make_sync_client()
        mock_inner = MagicMock()
        client._client = mock_inner
        client.close()
        mock_inner.close.assert_called_once()


# ---------------------------------------------------------------------------
# AsyncHTTPClient
# ---------------------------------------------------------------------------


def _make_async_client() -> AsyncHTTPClient:
    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    return AsyncHTTPClient(config, api_version="2025-10")


class TestAsyncHTTPClientGet:
    @respx.mock
    @pytest.mark.asyncio
    async def test_get_success(self) -> None:
        respx.get(f"{BASE_URL}/indexes").mock(
            return_value=httpx.Response(200, json={"indexes": []})
        )
        client = _make_async_client()
        try:
            resp = await client.get("/indexes")
            assert resp.status_code == 200
        finally:
            await client.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_get_raises_on_error(self) -> None:
        respx.get(f"{BASE_URL}/indexes/missing").mock(
            return_value=httpx.Response(404, json={"message": "not found"})
        )
        client = _make_async_client()
        try:
            with pytest.raises(NotFoundError):
                await client.get("/indexes/missing")
        finally:
            await client.close()


class TestAsyncHTTPClientPost:
    @respx.mock
    @pytest.mark.asyncio
    async def test_post_raises_on_conflict(self) -> None:
        respx.post(f"{BASE_URL}/indexes").mock(
            return_value=httpx.Response(409, json={"message": "conflict"})
        )
        client = _make_async_client()
        try:
            with pytest.raises(ConflictError):
                await client.post("/indexes", json={"name": "dup"})
        finally:
            await client.close()


class TestAsyncHTTPClientClose:
    @pytest.mark.asyncio
    async def test_close_when_no_client_created(self) -> None:
        """close() should not error when no requests were made."""
        client = _make_async_client()
        await client.close()  # _client is None — should be a no-op

    @respx.mock
    @pytest.mark.asyncio
    async def test_close_closes_underlying_client(self) -> None:
        respx.get(f"{BASE_URL}/ping").mock(return_value=httpx.Response(200))
        client = _make_async_client()
        await client.get("/ping")  # Force client creation
        assert client._client is not None
        await client.close()


# ---------------------------------------------------------------------------
# _encode_json / _prepare_json_kwargs
# ---------------------------------------------------------------------------


class TestEncodeJson:
    def test_returns_bytes(self) -> None:
        result = _encode_json({"key": "value"})
        assert isinstance(result, bytes)

    def test_output_matches_orjson(self) -> None:
        data = {"vectors": [{"id": "v1", "values": [0.1, 0.2]}]}
        assert _encode_json(data) == orjson.dumps(data)

    def test_handles_nested_structures(self) -> None:
        data = {"a": [1, 2, {"b": True, "c": None}]}
        parsed = orjson.loads(_encode_json(data))
        assert parsed == data


class TestPrepareJsonKwargs:
    def test_replaces_json_with_content(self) -> None:
        kwargs: dict[str, object] = {"json": {"name": "idx"}}
        result = _prepare_json_kwargs(kwargs)
        assert "json" not in result
        assert result["content"] == orjson.dumps({"name": "idx"})
        assert result["headers"]["Content-Type"] == "application/json"  # type: ignore[index]

    def test_preserves_existing_headers(self) -> None:
        kwargs: dict[str, object] = {
            "json": {"x": 1},
            "headers": {"X-Custom": "val"},
        }
        result = _prepare_json_kwargs(kwargs)
        assert result["headers"]["X-Custom"] == "val"  # type: ignore[index]
        assert result["headers"]["Content-Type"] == "application/json"  # type: ignore[index]

    def test_noop_when_no_json_key(self) -> None:
        kwargs: dict[str, object] = {"params": {"limit": "10"}}
        result = _prepare_json_kwargs(kwargs)
        assert result == {"params": {"limit": "10"}}


# ---------------------------------------------------------------------------
# Sync orjson serialization — verify request body is orjson-encoded
# ---------------------------------------------------------------------------


class TestHTTPClientOrjsonPost:
    @respx.mock
    def test_post_sends_orjson_encoded_body(self) -> None:
        """POST with json= should send orjson-serialized bytes, not stdlib json."""
        route = respx.post(f"{BASE_URL}/vectors/upsert").mock(
            return_value=httpx.Response(200, json={"upsertedCount": 1})
        )
        client = _make_sync_client()
        payload = {"vectors": [{"id": "v1", "values": [0.1, 0.2, 0.3]}]}
        client.post("/vectors/upsert", json=payload)

        request = route.calls[0].request
        assert request.content == orjson.dumps(payload)
        assert request.headers["content-type"] == "application/json"


class TestHTTPClientOrjsonPut:
    @respx.mock
    def test_put_sends_orjson_encoded_body(self) -> None:
        route = respx.put(f"{BASE_URL}/things/1").mock(
            return_value=httpx.Response(200, json={"ok": True})
        )
        client = _make_sync_client()
        payload = {"name": "updated"}
        client.put("/things/1", json=payload)

        request = route.calls[0].request
        assert request.content == orjson.dumps(payload)


class TestHTTPClientOrjsonPatch:
    @respx.mock
    def test_patch_sends_orjson_encoded_body(self) -> None:
        route = respx.patch(f"{BASE_URL}/indexes/idx").mock(
            return_value=httpx.Response(200, json={"name": "idx"})
        )
        client = _make_sync_client()
        payload = {"replicas": 2}
        client.patch("/indexes/idx", json=payload)

        request = route.calls[0].request
        assert request.content == orjson.dumps(payload)


# ---------------------------------------------------------------------------
# Async orjson serialization — verify request body is orjson-encoded
# ---------------------------------------------------------------------------


class TestAsyncHTTPClientOrjsonPost:
    @respx.mock
    @pytest.mark.asyncio
    async def test_post_sends_orjson_encoded_body(self) -> None:
        route = respx.post(f"{BASE_URL}/vectors/upsert").mock(
            return_value=httpx.Response(200, json={"upsertedCount": 1})
        )
        client = _make_async_client()
        payload = {"vectors": [{"id": "v1", "values": [0.1, 0.2, 0.3]}]}
        try:
            await client.post("/vectors/upsert", json=payload)
        finally:
            await client.close()

        request = route.calls[0].request
        assert request.content == orjson.dumps(payload)
        assert request.headers["content-type"] == "application/json"


class TestAsyncHTTPClientOrjsonPatch:
    @respx.mock
    @pytest.mark.asyncio
    async def test_patch_sends_orjson_encoded_body(self) -> None:
        route = respx.patch(f"{BASE_URL}/indexes/idx").mock(
            return_value=httpx.Response(200, json={"name": "idx"})
        )
        client = _make_async_client()
        try:
            await client.patch("/indexes/idx", json={"replicas": 2})
        finally:
            await client.close()

        request = route.calls[0].request
        assert request.content == orjson.dumps({"replicas": 2})


# ---------------------------------------------------------------------------
# Transport error wrapping — sync
# ---------------------------------------------------------------------------


class TestHTTPClientTransportErrors:
    def test_connect_error_raises_connection_error(self) -> None:
        client = _make_sync_client()
        with respx.mock:
            respx.get(f"{BASE_URL}/indexes").mock(
                side_effect=httpx.ConnectError("Connection refused")
            )
            with pytest.raises(PineconeConnectionError, match="Connection refused") as exc_info:
                client.get("/indexes")
            assert isinstance(exc_info.value, PineconeError)
            assert isinstance(exc_info.value.__cause__, httpx.ConnectError)

    def test_read_timeout_raises_timeout_error(self) -> None:
        client = _make_sync_client()
        with respx.mock:
            respx.get(f"{BASE_URL}/indexes").mock(side_effect=httpx.ReadTimeout("Read timed out"))
            with pytest.raises(PineconeTimeoutError, match="Read timed out") as exc_info:
                client.get("/indexes")
            assert isinstance(exc_info.value, PineconeError)
            assert isinstance(exc_info.value.__cause__, httpx.ReadTimeout)

    def test_connect_timeout_raises_timeout_error(self) -> None:
        client = _make_sync_client()
        with respx.mock:
            respx.post(f"{BASE_URL}/vectors/upsert").mock(
                side_effect=httpx.ConnectTimeout("Connect timed out")
            )
            with pytest.raises(PineconeTimeoutError):
                client.post("/vectors/upsert", json={"vectors": []})

    def test_pool_timeout_raises_timeout_error(self) -> None:
        client = _make_sync_client()
        with respx.mock:
            respx.get(f"{BASE_URL}/indexes").mock(side_effect=httpx.PoolTimeout("Pool exhausted"))
            with pytest.raises(PineconeTimeoutError):
                client.get("/indexes")

    def test_write_error_raises_connection_error(self) -> None:
        client = _make_sync_client()
        with respx.mock:
            respx.put(f"{BASE_URL}/things").mock(side_effect=httpx.WriteError("Write failed"))
            with pytest.raises(PineconeConnectionError):
                client.put("/things", json={"x": 1})

    def test_transport_error_on_patch(self) -> None:
        client = _make_sync_client()
        with respx.mock:
            respx.patch(f"{BASE_URL}/indexes/idx").mock(
                side_effect=httpx.ConnectError("DNS failure")
            )
            with pytest.raises(PineconeConnectionError):
                client.patch("/indexes/idx", json={"replicas": 2})

    def test_transport_error_on_delete(self) -> None:
        client = _make_sync_client()
        with respx.mock:
            respx.delete(f"{BASE_URL}/indexes/foo").mock(
                side_effect=httpx.ConnectError("Connection reset")
            )
            with pytest.raises(PineconeConnectionError):
                client.delete("/indexes/foo")

    def test_transport_error_on_post(self) -> None:
        client = _make_sync_client()
        with respx.mock:
            respx.post(f"{BASE_URL}/v").mock(side_effect=httpx.ConnectError("dns fail"))
            with pytest.raises(PineconeConnectionError) as exc_info:
                client.post("/v", json={"x": 1})
            assert isinstance(exc_info.value.__cause__, httpx.ConnectError)

    def test_timeout_on_put(self) -> None:
        client = _make_sync_client()
        with respx.mock:
            respx.put(f"{BASE_URL}/v").mock(side_effect=httpx.ReadTimeout("put timeout"))
            with pytest.raises(PineconeTimeoutError):
                client.put("/v", json={"x": 1})

    def test_timeout_on_patch(self) -> None:
        client = _make_sync_client()
        with respx.mock:
            respx.patch(f"{BASE_URL}/v").mock(side_effect=httpx.ConnectTimeout("patch timeout"))
            with pytest.raises(PineconeTimeoutError):
                client.patch("/v", json={"x": 1})

    def test_timeout_on_delete(self) -> None:
        client = _make_sync_client()
        with respx.mock:
            respx.delete(f"{BASE_URL}/v").mock(side_effect=httpx.ReadTimeout("delete timeout"))
            with pytest.raises(PineconeTimeoutError):
                client.delete("/v")

    def test_transport_error_during_stream_is_wrapped(self) -> None:
        client = _make_sync_client()
        with respx.mock:
            respx.get(f"{BASE_URL}/stream").mock(side_effect=httpx.ReadTimeout("stream timeout"))
            with pytest.raises(PineconeTimeoutError), client.stream("GET", "/stream"):
                pass


class TestTransportErrorHierarchy:
    def test_timeout_error_is_pinecone_error(self) -> None:
        assert issubclass(PineconeTimeoutError, PineconeError)

    def test_connection_error_is_pinecone_error(self) -> None:
        assert issubclass(PineconeConnectionError, PineconeError)


# ---------------------------------------------------------------------------
# Transport error wrapping — async
# ---------------------------------------------------------------------------


class TestAsyncHTTPClientTransportErrors:
    @pytest.mark.asyncio
    async def test_connect_error_raises_connection_error(self) -> None:
        client = _make_async_client()
        with respx.mock:
            respx.get(f"{BASE_URL}/indexes").mock(
                side_effect=httpx.ConnectError("Connection refused")
            )
            with pytest.raises(PineconeConnectionError, match="Connection refused") as exc_info:
                await client.get("/indexes")
            assert isinstance(exc_info.value.__cause__, httpx.ConnectError)

    @pytest.mark.asyncio
    async def test_read_timeout_raises_timeout_error(self) -> None:
        client = _make_async_client()
        with respx.mock:
            respx.get(f"{BASE_URL}/indexes").mock(side_effect=httpx.ReadTimeout("Read timed out"))
            with pytest.raises(PineconeTimeoutError, match="Read timed out") as exc_info:
                await client.get("/indexes")
            assert isinstance(exc_info.value.__cause__, httpx.ReadTimeout)

    @pytest.mark.asyncio
    async def test_connect_error_on_post(self) -> None:
        client = _make_async_client()
        with respx.mock:
            respx.post(f"{BASE_URL}/vectors/upsert").mock(side_effect=httpx.ConnectError("refused"))
            with pytest.raises(PineconeConnectionError):
                await client.post("/vectors/upsert", json={"vectors": []})

    @pytest.mark.asyncio
    async def test_timeout_on_delete(self) -> None:
        client = _make_async_client()
        with respx.mock:
            respx.delete(f"{BASE_URL}/indexes/foo").mock(side_effect=httpx.ReadTimeout("timed out"))
            with pytest.raises(PineconeTimeoutError):
                await client.delete("/indexes/foo")

    @pytest.mark.asyncio
    async def test_transport_error_on_put(self) -> None:
        client = _make_async_client()
        with respx.mock:
            respx.put(f"{BASE_URL}/things").mock(side_effect=httpx.WriteError("broken pipe"))
            with pytest.raises(PineconeConnectionError):
                await client.put("/things", json={"x": 1})

    @pytest.mark.asyncio
    async def test_transport_error_on_patch(self) -> None:
        client = _make_async_client()
        with respx.mock:
            respx.patch(f"{BASE_URL}/indexes/idx").mock(
                side_effect=httpx.ConnectError("DNS failure")
            )
            with pytest.raises(PineconeConnectionError):
                await client.patch("/indexes/idx", json={"replicas": 2})

    @pytest.mark.asyncio
    async def test_async_transport_error_on_post(self) -> None:
        client = _make_async_client()
        with respx.mock:
            respx.post(f"{BASE_URL}/v").mock(side_effect=httpx.ConnectError("dns fail"))
            with pytest.raises(PineconeConnectionError) as exc_info:
                await client.post("/v", json={"x": 1})
            assert isinstance(exc_info.value.__cause__, httpx.ConnectError)

    @pytest.mark.asyncio
    async def test_async_timeout_on_put(self) -> None:
        client = _make_async_client()
        with respx.mock:
            respx.put(f"{BASE_URL}/v").mock(side_effect=httpx.ReadTimeout("put timeout"))
            with pytest.raises(PineconeTimeoutError):
                await client.put("/v", json={"x": 1})

    @pytest.mark.asyncio
    async def test_async_timeout_on_patch(self) -> None:
        client = _make_async_client()
        with respx.mock:
            respx.patch(f"{BASE_URL}/v").mock(side_effect=httpx.ConnectTimeout("patch timeout"))
            with pytest.raises(PineconeTimeoutError):
                await client.patch("/v", json={"x": 1})

    @pytest.mark.asyncio
    async def test_async_timeout_on_delete(self) -> None:
        client = _make_async_client()
        with respx.mock:
            respx.delete(f"{BASE_URL}/v").mock(side_effect=httpx.ReadTimeout("delete timeout"))
            with pytest.raises(PineconeTimeoutError):
                await client.delete("/v")

    @pytest.mark.asyncio
    async def test_async_timeout_on_post(self) -> None:
        client = _make_async_client()
        with respx.mock:
            respx.post(f"{BASE_URL}/v").mock(side_effect=httpx.ReadTimeout("post timeout"))
            with pytest.raises(PineconeTimeoutError):
                await client.post("/v", json={"x": 1})

    @pytest.mark.asyncio
    async def test_async_transport_error_on_delete(self) -> None:
        client = _make_async_client()
        with respx.mock:
            respx.delete(f"{BASE_URL}/v").mock(side_effect=httpx.ConnectError("connection reset"))
            with pytest.raises(PineconeConnectionError):
                await client.delete("/v")

    @pytest.mark.asyncio
    async def test_async_transport_error_during_stream_is_wrapped(self) -> None:
        client = _make_async_client()
        with respx.mock:
            respx.get(f"{BASE_URL}/stream").mock(side_effect=httpx.ReadTimeout("stream timeout"))
            with pytest.raises(PineconeTimeoutError):
                async with client.stream("GET", "/stream"):
                    pass


# ---------------------------------------------------------------------------
# _redact_headers / _log_curl redaction
# ---------------------------------------------------------------------------


class TestRedactHeaders:
    def test_redacts_api_key(self) -> None:
        headers = {"Api-Key": "sk-secret-123", "User-Agent": "test"}
        result = _redact_headers(headers)
        assert result["Api-Key"] == "***"
        assert result["User-Agent"] == "test"

    def test_redacts_authorization(self) -> None:
        headers = {"Authorization": "Bearer tok", "Accept": "application/json"}
        result = _redact_headers(headers)
        assert result["Authorization"] == "***"
        assert result["Accept"] == "application/json"

    def test_redacts_proxy_authorization(self) -> None:
        headers = {"Proxy-Authorization": "Basic abc"}
        result = _redact_headers(headers)
        assert result["Proxy-Authorization"] == "***"

    def test_case_insensitive(self) -> None:
        headers = {"api-key": "secret", "API-KEY": "secret2", "Api-Key": "secret3"}
        result = _redact_headers(headers)
        for v in result.values():
            assert v == "***"

    def test_non_sensitive_headers_pass_through(self) -> None:
        headers = {"Content-Type": "application/json", "X-Custom": "value"}
        result = _redact_headers(headers)
        assert result == headers

    def test_returns_copy(self) -> None:
        headers = {"Api-Key": "secret"}
        result = _redact_headers(headers)
        assert result is not headers


class TestLogCurlRedactsApiKey:
    def test_log_curl_redacts_api_key(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("PINECONE_DEBUG_CURL", "1")
        headers = {
            "Api-Key": "sk-super-secret",
            "Authorization": "Bearer my-token",
            "Proxy-Authorization": "Basic creds",
            "User-Agent": "test-agent",
        }
        with caplog.at_level(logging.DEBUG, logger="pinecone._internal.http_client"):
            _log_curl("GET", "https://api.pinecone.io/indexes", headers)
        output = caplog.text
        assert "sk-super-secret" not in output
        assert "my-token" not in output
        assert "Basic creds" not in output
        assert "***" in output
        assert "test-agent" in output

    def test_log_curl_noop_without_env(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.delenv("PINECONE_DEBUG_CURL", raising=False)
        with caplog.at_level(logging.DEBUG, logger="pinecone._internal.http_client"):
            _log_curl("GET", "https://api.pinecone.io/indexes", {"Api-Key": "secret"})
        assert "curl" not in caplog.text


# ---------------------------------------------------------------------------
# _RetryTransport on_throttle callback
# ---------------------------------------------------------------------------


class _SequencedTransport(httpx.BaseTransport):
    """Sync mock transport that yields pre-built responses in order."""

    def __init__(self, responses: list[httpx.Response]) -> None:
        self._responses = iter(responses)

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        return next(self._responses)


class _AsyncSequencedTransport(httpx.AsyncBaseTransport):
    """Async mock transport that yields pre-built responses in order."""

    def __init__(self, responses: list[httpx.Response]) -> None:
        self._responses = iter(responses)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return next(self._responses)


class TestRetryTransportThrottle:
    def test_on_throttle_called_on_429(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("pinecone._internal.http_client.time.sleep", lambda _: None)
        mock_callback = MagicMock()
        config = RetryConfig(max_retries=2, on_throttle=mock_callback)
        inner = _SequencedTransport([httpx.Response(429), httpx.Response(200)])
        transport = _RetryTransport(transport=inner, retry_config=config)  # type: ignore[arg-type]
        request = httpx.Request("GET", "https://api.pinecone.io/indexes")

        response = transport.handle_request(request)

        assert response.status_code == 200
        mock_callback.assert_called_once_with("api.pinecone.io")

    def test_on_throttle_called_each_retry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("pinecone._internal.http_client.time.sleep", lambda _: None)
        mock_callback = MagicMock()
        # max_retries=2 → 3 total attempts (initial + 2 retries)
        config = RetryConfig(max_retries=2, on_throttle=mock_callback)
        inner = _SequencedTransport([httpx.Response(429), httpx.Response(429), httpx.Response(429)])
        transport = _RetryTransport(transport=inner, retry_config=config)  # type: ignore[arg-type]
        request = httpx.Request("GET", "https://api.pinecone.io/indexes")

        response = transport.handle_request(request)

        assert response.status_code == 429
        assert mock_callback.call_count == 3
        mock_callback.assert_called_with("api.pinecone.io")

    def test_on_throttle_none_is_default_and_does_not_break(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("pinecone._internal.http_client.time.sleep", lambda _: None)
        config = RetryConfig(max_retries=2)
        inner = _SequencedTransport([httpx.Response(429), httpx.Response(200)])
        transport = _RetryTransport(transport=inner, retry_config=config)  # type: ignore[arg-type]
        request = httpx.Request("GET", "https://api.pinecone.io/indexes")

        response = transport.handle_request(request)

        assert response.status_code == 200

    def test_on_throttle_exception_is_swallowed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("pinecone._internal.http_client.time.sleep", lambda _: None)

        def bad_callback(host: str) -> None:
            raise RuntimeError("Limiter exploded!")

        config = RetryConfig(max_retries=2, on_throttle=bad_callback)
        inner = _SequencedTransport([httpx.Response(429), httpx.Response(200)])
        transport = _RetryTransport(transport=inner, retry_config=config)  # type: ignore[arg-type]
        request = httpx.Request("GET", "https://api.pinecone.io/indexes")

        response = transport.handle_request(request)

        assert response.status_code == 200


class TestAsyncRetryTransportThrottle:
    @pytest.mark.asyncio
    async def test_on_throttle_called_on_429(self) -> None:
        mock_callback = MagicMock()
        config = RetryConfig(max_retries=2, on_throttle=mock_callback)
        inner = _AsyncSequencedTransport([httpx.Response(429), httpx.Response(200)])
        transport = _AsyncRetryTransport(transport=inner, retry_config=config)  # type: ignore[arg-type]
        request = httpx.Request("GET", "https://api.pinecone.io/indexes")

        with patch("asyncio.sleep", new=AsyncMock()):
            response = await transport.handle_async_request(request)

        assert response.status_code == 200
        mock_callback.assert_called_once_with("api.pinecone.io")


# ---------------------------------------------------------------------------
# Jitter distribution — helpers
# ---------------------------------------------------------------------------


def _make_sync_retry_transport(retry_config: RetryConfig | None = None) -> _RetryTransport:
    return _RetryTransport(
        transport=httpx.HTTPTransport(),
        retry_config=retry_config or RetryConfig(),
    )


def _make_async_retry_transport(
    retry_config: RetryConfig | None = None,
) -> _AsyncRetryTransport:
    return _AsyncRetryTransport(
        transport=httpx.AsyncHTTPTransport(),
        retry_config=retry_config or RetryConfig(),
    )


# ---------------------------------------------------------------------------
# Jitter distribution — sync
# ---------------------------------------------------------------------------


class TestRetryJitterDistribution:
    def test_backoff_first_attempt_delay_range(self) -> None:
        import statistics

        random.seed(42)
        t = _make_sync_retry_transport()
        delays = [_compute_backoff(t._config, 0, None) for _ in range(200)]
        assert all(0.25 <= d <= 0.75 for d in delays), (
            f"out-of-range: {[d for d in delays if not (0.25 <= d <= 0.75)]}"
        )
        assert len({round(d, 6) for d in delays}) > 50, (
            "delays are too clustered — RNG not actually random"
        )
        assert statistics.stdev(delays) > 0.1, (
            "delay distribution too tight — smear may have regressed"
        )

    def test_backoff_evolves_with_prev_delay(self) -> None:
        random.seed(42)
        cfg = RetryConfig(max_wait=60.0)
        delays = [_compute_backoff(cfg, 2, 5.0) for _ in range(200)]
        # Decorrelated jitter: uniform(0.25, min(60.0, 5.0*3)) = uniform(0.25, 15.0)
        assert all(0.25 <= d <= 15.0 for d in delays)
        assert any(d > 5.0 for d in delays), "upper bound did not expand with prev_delay"

    def test_backoff_capped_at_max_wait(self) -> None:
        random.seed(42)
        cfg = RetryConfig(max_wait=10.0)
        delays = [_compute_backoff(cfg, 0, 100.0) for _ in range(50)]
        # prev_delay=100 would push upper to 300, but max_wait caps at 10.0
        assert all(d <= 10.0 for d in delays)

    def test_retry_after_smear_range(self) -> None:
        random.seed(42)
        t = _make_sync_retry_transport()
        response = httpx.Response(429, headers={"retry-after": "60"})
        delays = [_compute_retry_after_delay(t._config, response, 0, None) for _ in range(200)]
        # Smear: delay in [60, 60 + 0.5*60) = [60, 90)
        assert all(60.0 <= d < 90.0 for d in delays), (
            f"out-of-range delays: {[d for d in delays if not (60.0 <= d < 90.0)]}"
        )
        assert len({round(d, 6) for d in delays}) > 50
        assert any(d > 60.5 for d in delays)

    def test_retry_after_falls_back_to_backoff_when_invalid(self) -> None:
        random.seed(42)
        t = _make_sync_retry_transport()
        response = httpx.Response(429, headers={"retry-after": "Fri, 31 Dec 2026 23:59:59 GMT"})
        delays = [_compute_retry_after_delay(t._config, response, 0, None) for _ in range(50)]
        # HTTP-date is not parseable as float; falls back to backoff which is uniform(0.25, 0.75)
        assert all(0.25 <= d <= 0.75 for d in delays)

    def test_retry_after_falls_back_to_backoff_when_missing(self) -> None:
        random.seed(42)
        t = _make_sync_retry_transport()
        response = httpx.Response(500)
        delays = [_compute_retry_after_delay(t._config, response, 0, None) for _ in range(50)]
        # Falls back to backoff: uniform(0.25, 0.75)
        assert all(0.25 <= d <= 0.75 for d in delays)

    def test_negative_retry_after_falls_back_to_backoff(self) -> None:
        random.seed(42)
        t = _make_sync_retry_transport()
        response = httpx.Response(429, headers={"retry-after": "-1"})
        delays = [_compute_retry_after_delay(t._config, response, 0, None) for _ in range(50)]
        # Negative values are ignored; falls back to backoff: uniform(0.25, 0.75)
        assert all(0.25 <= d <= 0.75 for d in delays)


# ---------------------------------------------------------------------------
# Jitter distribution — async parity
# ---------------------------------------------------------------------------


class TestAsyncRetryJitterDistribution:
    def test_backoff_first_attempt_delay_range(self) -> None:
        import statistics

        random.seed(42)
        t = _make_async_retry_transport()
        delays = [_compute_backoff(t._config, 0, None) for _ in range(200)]
        assert all(0.25 <= d <= 0.75 for d in delays), (
            f"out-of-range: {[d for d in delays if not (0.25 <= d <= 0.75)]}"
        )
        assert len({round(d, 6) for d in delays}) > 50, (
            "delays are too clustered — RNG not actually random"
        )
        assert statistics.stdev(delays) > 0.1, (
            "delay distribution too tight — smear may have regressed"
        )

    def test_retry_after_smear_range(self) -> None:
        random.seed(42)
        t = _make_async_retry_transport()
        response = httpx.Response(429, headers={"retry-after": "60"})
        delays = [_compute_retry_after_delay(t._config, response, 0, None) for _ in range(200)]
        # Smear: delay in [60, 60 + 0.5*60) = [60, 90)
        assert all(60.0 <= d < 90.0 for d in delays), (
            f"out-of-range delays: {[d for d in delays if not (60.0 <= d < 90.0)]}"
        )
        assert len({round(d, 6) for d in delays}) > 50
        assert any(d > 60.5 for d in delays)
