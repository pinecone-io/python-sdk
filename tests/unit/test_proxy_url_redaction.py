"""Proxy URL userinfo credentials must not appear in reprs or error messages."""

from __future__ import annotations

import pytest

from pinecone._internal.config import PineconeConfig
from pinecone.config.config import Config
from pinecone.errors.exceptions import PineconeValueError

_SECRET = "SENTINEL-SECRET-1234"
_PROXY_URL = f"http://proxyuser:{_SECRET}@proxy.example.com:8080"


def test_pinecone_config_repr_redacts_proxy_userinfo() -> None:
    result = repr(PineconeConfig(api_key="pcsk_abc123", proxy_url=_PROXY_URL))
    assert _SECRET not in result
    assert "proxyuser" not in result
    assert "proxy_url='http://***@proxy.example.com:8080'" in result


def test_pinecone_config_repr_keeps_proxy_url_without_userinfo() -> None:
    result = repr(PineconeConfig(api_key="pcsk_abc123", proxy_url="http://proxy:8080/a@b"))
    assert "proxy_url='http://proxy:8080/a@b'" in result


def test_config_shim_repr_redacts_proxy_userinfo() -> None:
    result = repr(Config(api_key="pcsk_abc123", proxy_url=_PROXY_URL))
    assert _SECRET not in result
    assert "proxyuser" not in result
    assert "proxy_url='http://***@proxy.example.com:8080'" in result


def test_config_shim_repr_none_proxy_url() -> None:
    assert "proxy_url=None" in repr(Config())


def test_grpc_invalid_proxy_url_error_redacts_userinfo() -> None:
    grpc = pytest.importorskip("pinecone._grpc", reason="Rust extension not available")
    with pytest.raises(PineconeValueError) as exc_info:
        grpc.GrpcChannel(
            "https://test-index-abc123.svc.us-east-1-aws.pinecone.io:443",
            "test-api-key",
            "2025-10",
            "0.1.0",
            secure=False,
            proxy_url=f"http://proxyuser:{_SECRET}@proxy example.com:8080",
        )
    message = str(exc_info.value)
    assert _SECRET not in message
    assert "proxyuser" not in message
    assert "Invalid proxy URL 'http://***@proxy example.com:8080'" in message
