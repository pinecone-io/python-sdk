"""Unit tests for debug logging and PINECONE_DEBUG_CURL functionality."""

from __future__ import annotations

import logging

import httpx
import pytest
import respx

from pinecone._internal.config import PineconeConfig
from pinecone._internal.http_client import HTTPClient, _log_curl


class TestPineconeDebugEnvVar:
    def test_pinecone_debug_enables_debug_level(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PINECONE_DEBUG", "1")
        # Reset the logger level before reimporting
        pinecone_logger = logging.getLogger("pinecone")
        original_level = pinecone_logger.level
        try:
            pinecone_logger.setLevel(logging.WARNING)
            # Re-execute the init logic
            import importlib

            import pinecone

            importlib.reload(pinecone)
            assert logging.getLogger("pinecone").level == logging.DEBUG
        finally:
            pinecone_logger.setLevel(original_level)

    def test_pinecone_debug_not_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("PINECONE_DEBUG", raising=False)
        pinecone_logger = logging.getLogger("pinecone")
        original_level = pinecone_logger.level
        try:
            pinecone_logger.setLevel(logging.WARNING)
            import importlib

            import pinecone

            importlib.reload(pinecone)
            # Logger level should remain at WARNING, not be forced to DEBUG
            assert logging.getLogger("pinecone").level != logging.DEBUG
        finally:
            pinecone_logger.setLevel(original_level)


class TestLogCurl:
    def test_curl_logging_disabled_by_default(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.delenv("PINECONE_DEBUG_CURL", raising=False)
        with caplog.at_level(logging.DEBUG, logger="pinecone._internal.http_client"):
            _log_curl("GET", "https://api.pinecone.io/test", {"Api-Key": "test-key"})
        assert "curl" not in caplog.text

    def test_curl_logging_when_enabled(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("PINECONE_DEBUG_CURL", "1")
        with caplog.at_level(logging.DEBUG, logger="pinecone._internal.http_client"):
            _log_curl(
                "GET",
                "https://api.pinecone.io/test",
                {"Api-Key": "test-key"},
            )
        assert "curl -X GET" in caplog.text
        assert "https://api.pinecone.io/test" in caplog.text

    def test_curl_logging_redacts_api_key(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("PINECONE_DEBUG_CURL", "1")
        with caplog.at_level(logging.DEBUG, logger="pinecone._internal.http_client"):
            _log_curl(
                "GET",
                "https://api.pinecone.io/test",
                {"Api-Key": "my-secret-key-12345"},
            )
        assert "my-secret-key-12345" not in caplog.text
        assert "Api-Key: ***" in caplog.text

    def test_curl_logging_omits_body_contents(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        # The raw request body must never be logged: it can carry user vector
        # and metadata payloads (potentially PII). Only the byte count is shown.
        monkeypatch.setenv("PINECONE_DEBUG_CURL", "1")
        body = b'{"vectors": [{"id": "v1"}]}'
        with caplog.at_level(logging.DEBUG, logger="pinecone._internal.http_client"):
            _log_curl(
                "POST",
                "https://api.pinecone.io/vectors/upsert",
                {"Api-Key": "test-key", "Content-Type": "application/json"},
                body=body,
            )
        assert '{"vectors": [{"id": "v1"}]}' not in caplog.text
        assert f"{len(body)} byte body omitted" in caplog.text

    def test_curl_logging_includes_all_headers(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("PINECONE_DEBUG_CURL", "1")
        with caplog.at_level(logging.DEBUG, logger="pinecone._internal.http_client"):
            _log_curl(
                "GET",
                "https://api.pinecone.io/test",
                {"Api-Key": "key", "X-Custom": "val"},
            )
        assert "-H 'Api-Key: ***'" in caplog.text
        assert "-H 'X-Custom: val'" in caplog.text


class TestHTTPClientCurlLogging:
    @respx.mock
    def test_get_logs_curl_when_enabled(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("PINECONE_DEBUG_CURL", "1")
        respx.get("https://api.pinecone.io/test").mock(return_value=httpx.Response(200, json={}))
        config = PineconeConfig(api_key="test-key")
        client = HTTPClient(config, api_version="2025-10")
        with caplog.at_level(logging.DEBUG, logger="pinecone._internal.http_client"):
            try:
                client.get("/test")
            finally:
                client.close()
        assert "curl -X GET" in caplog.text
        assert "/test" in caplog.text

    @respx.mock
    def test_post_logs_curl_with_body(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("PINECONE_DEBUG_CURL", "1")
        respx.post("https://api.pinecone.io/vectors/upsert").mock(
            return_value=httpx.Response(200, json={"upsertedCount": 1})
        )
        config = PineconeConfig(api_key="test-key")
        client = HTTPClient(config, api_version="2025-10")
        with caplog.at_level(logging.DEBUG, logger="pinecone._internal.http_client"):
            try:
                client.post("/vectors/upsert", json={"vectors": [{"id": "v1"}]})
            finally:
                client.close()
        assert "curl -X POST" in caplog.text
        assert "byte body omitted" in caplog.text
        # Raw request payload must not appear in logs.
        assert '"id": "v1"' not in caplog.text

    @respx.mock
    def test_no_curl_output_when_disabled(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.delenv("PINECONE_DEBUG_CURL", raising=False)
        respx.get("https://api.pinecone.io/test").mock(return_value=httpx.Response(200, json={}))
        config = PineconeConfig(api_key="test-key")
        client = HTTPClient(config, api_version="2025-10")
        with caplog.at_level(logging.DEBUG, logger="pinecone._internal.http_client"):
            try:
                client.get("/test")
            finally:
                client.close()
        assert "curl" not in caplog.text
