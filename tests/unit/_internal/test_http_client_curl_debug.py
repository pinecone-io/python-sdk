"""Tests for _log_curl debug output including per-request headers."""

from __future__ import annotations

import logging
import os
import shlex
from unittest.mock import patch

import httpx
import pytest
import respx

from pinecone._internal.config import PineconeConfig
from pinecone._internal.http_client import HTTPClient, _log_curl

INDEX_HOST = "https://idx-abc.svc.pinecone.io"


def test_log_curl_includes_content_type_header(caplog: pytest.LogCaptureFixture) -> None:
    with (
        patch.dict(os.environ, {"PINECONE_DEBUG_CURL": "1"}),
        caplog.at_level(logging.DEBUG, logger="pinecone._internal.http_client"),
    ):
        _log_curl(
            "POST",
            "https://idx.svc.pinecone.io/ns/documents/search",
            headers={"Api-Key": "secret", "Content-Type": "application/json"},
            body=b'{"top_k": 5}',
        )
    assert "Content-Type: application/json" in caplog.text


def test_log_curl_redacts_api_key(caplog: pytest.LogCaptureFixture) -> None:
    with (
        patch.dict(os.environ, {"PINECONE_DEBUG_CURL": "1"}),
        caplog.at_level(logging.DEBUG, logger="pinecone._internal.http_client"),
    ):
        _log_curl(
            "POST",
            "https://idx.svc.pinecone.io/ns/documents/search",
            headers={"Api-Key": "super-secret-key", "Content-Type": "application/json"},
            body=b"{}",
        )
    assert "super-secret-key" not in caplog.text
    assert "***" in caplog.text


@respx.mock
def test_http_client_post_logs_content_type_in_curl(caplog: pytest.LogCaptureFixture) -> None:
    respx.post(f"{INDEX_HOST}/test").mock(return_value=httpx.Response(200, json={}))
    config = PineconeConfig(api_key="test-key", host=INDEX_HOST)
    client = HTTPClient(config=config, api_version="2026-01")
    with (
        patch.dict(os.environ, {"PINECONE_DEBUG_CURL": "1"}),
        caplog.at_level(logging.DEBUG, logger="pinecone._internal.http_client"),
    ):
        client.post("/test", json={"hello": "world"})
    assert "Content-Type: application/json" in caplog.text


def test_log_curl_quotes_single_quotes_in_url_headers_and_body(
    caplog: pytest.LogCaptureFixture,
) -> None:
    body = b"""{"text": "it's here'; touch /tmp/pwned; echo '"}"""
    url = "https://idx.svc.pinecone.io/vectors/fetch?ids=a'b"
    with (
        patch.dict(os.environ, {"PINECONE_DEBUG_CURL": "1"}),
        caplog.at_level(logging.DEBUG, logger="pinecone._internal.http_client"),
    ):
        _log_curl(
            "POST",
            url,
            headers={"Api-Key": "secret", "X-Note": "o'clock"},
            body=body,
        )
    curl_cmd = caplog.records[-1].getMessage().split("\n", 1)[1]
    assert shlex.split(curl_cmd) == [
        "curl",
        "-X",
        "POST",
        url,
        "-H",
        "Api-Key: ***",
        "-H",
        "X-Note: o'clock",
        "-d",
        body.decode(),
    ]
