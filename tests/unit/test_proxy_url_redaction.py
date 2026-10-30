"""Proxy URL userinfo credentials must not appear in reprs or error messages."""

from __future__ import annotations

import pytest

from pinecone._internal.config import PineconeConfig, _redact_url_userinfo
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
    result = repr(PineconeConfig(api_key="pcsk_abc123", proxy_url="http://proxy:8080/path"))
    assert "proxy_url='http://proxy:8080/path'" in result


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("http://user:pa#ss@proxy:8080", "http://***@proxy:8080"),
        ("http://user:pa/ss@proxy:8080", "http://***@proxy:8080"),
        ("http://user:pa?ss@proxy:8080", "http://***@proxy:8080"),
        ("http://user:pa ss/x@proxy:8080", "http://***@proxy:8080"),
        ("user:pass@proxy:8080/?next=http://x", "***@proxy:8080/?next=http://x"),
        ("http://proxy:8080/a@b", "http://***@b"),
        ("http://user:p@ss@proxy:8080", "http://***@proxy:8080"),
        ("http://user:p%40ss@proxy:8080", "http://***@proxy:8080"),
        ("http://user@proxy:8080", "http://***@proxy:8080"),
        ("user:pass@proxy:8080", "***@proxy:8080"),
        ("http://user:pass@[::1]:8080", "http://***@[::1]:8080"),
        ("http://user:pass@[fe80::1%25en0]:8080", "http://***@[fe80::1%25en0]:8080"),
        ("HTTP://user:pass@proxy:8080", "HTTP://***@proxy:8080"),
        ("socks5://user:pass@proxy:1080", "socks5://***@proxy:1080"),
        ("http://[::1]:8080", "http://[::1]:8080"),
        ("http://proxy:8080", "http://proxy:8080"),
    ],
)
def test_redact_url_userinfo(url: str, expected: str) -> None:
    assert _redact_url_userinfo(url) == expected


@pytest.mark.parametrize("password", ["pa#ss", "pa/ss", "pa?ss"])
def test_config_reprs_redact_password_with_delimiter(password: str) -> None:
    proxy_url = f"http://user:{password}@proxy:8080"
    for config in (
        PineconeConfig(api_key="pcsk_abc123", proxy_url=proxy_url),
        Config(proxy_url=proxy_url),
    ):
        result = repr(config)
        assert password not in result
        assert "proxy_url='http://***@proxy:8080'" in result


def test_config_shim_repr_redacts_proxy_userinfo() -> None:
    result = repr(Config(api_key="pcsk_abc123", proxy_url=_PROXY_URL))
    assert _SECRET not in result
    assert "proxyuser" not in result
    assert "proxy_url='http://***@proxy.example.com:8080'" in result


def test_config_shim_repr_none_proxy_url() -> None:
    assert "proxy_url=None" in repr(Config())


@pytest.mark.parametrize(
    ("proxy_url", "secret", "redacted"),
    [
        (
            f"http://proxyuser:{_SECRET}@proxy example.com:8080",
            _SECRET,
            "http://***@proxy example.com:8080",
        ),
        ("http://proxyuser:pa ss/x@proxy:8080", "pa ss/x", "http://***@proxy:8080"),
        ("http://proxyuser:pa ss?x@proxy:8080", "pa ss?x", "http://***@proxy:8080"),
        ("http://proxyuser:pa ss#x@proxy:8080", "pa ss#x", "http://***@proxy:8080"),
    ],
)
def test_grpc_invalid_proxy_url_error_redacts_userinfo(
    proxy_url: str, secret: str, redacted: str
) -> None:
    grpc = pytest.importorskip("pinecone._grpc", reason="Rust extension not available")
    with pytest.raises(PineconeValueError) as exc_info:
        grpc.GrpcChannel(
            "https://test-index-abc123.svc.us-east-1-aws.pinecone.io:443",
            "test-api-key",
            "2025-10",
            "0.1.0",
            secure=False,
            proxy_url=proxy_url,
        )
    message = str(exc_info.value)
    assert secret not in message
    assert "proxyuser" not in message
    assert f"Invalid proxy URL '{redacted}'" in message
