"""A malformed ``proxy_url`` fails at client construction without leaking the password."""

from __future__ import annotations

import importlib.util
import logging
import traceback
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from pinecone import Admin, AsyncPinecone, Pinecone
from pinecone._internal.config import PineconeConfig, validate_proxy_url
from pinecone.async_client.async_index import AsyncIndex
from pinecone.errors.exceptions import PineconeValueError
from pinecone.index import Index

_SECRET = "SENTINEL-SECRET-1234"
_INDEX_HOST = "https://idx.svc.example.test"


def _no_network(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("the token exchange must not be attempted")


def _admin(proxy_url: str) -> object:
    return Admin(client_id="id", client_secret="secret", proxy_url=proxy_url)


_CONSTRUCTORS: dict[str, Callable[[str], object]] = {
    "PineconeConfig": lambda url: PineconeConfig(api_key="k", proxy_url=url),
    "Pinecone": lambda url: Pinecone(api_key="k", proxy_url=url),
    "AsyncPinecone": lambda url: AsyncPinecone(api_key="k", proxy_url=url),
    "Index": lambda url: Index(host=_INDEX_HOST, api_key="k", proxy_url=url),
    "AsyncIndex": lambda url: AsyncIndex(host=_INDEX_HOST, api_key="k", proxy_url=url),
    "Admin": _admin,
}

_INVALID = {
    "scheme-less": (
        f"proxyuser:{_SECRET}@proxy.corp:3128",
        "Invalid proxy URL '***@proxy.corp:3128': it has no scheme; "
        "start it with http:// or https://",
    ),
    "host:port only": (
        "proxy.corp:3128",
        "Invalid proxy URL 'proxy.corp:3128': it has no scheme; start it with http:// or https://",
    ),
    "ftp": (
        f"ftp://proxyuser:{_SECRET}@proxy.corp:1080",
        "Invalid proxy URL 'ftp://***@proxy.corp:1080': "
        "scheme must be 'http', 'https', 'socks5' or 'socks5h', got 'ftp'",
    ),
    "raw slash in password": (
        f"http://proxyuser:{_SECRET}/x@proxy.corp:3128",
        "Invalid proxy URL 'http://***': it could not be parsed; percent-encode",
    ),
    "raw hash in password": (
        f"http://proxyuser:{_SECRET}#x@proxy.corp:3128",
        "Invalid proxy URL 'http://***': it could not be parsed; percent-encode",
    ),
    "raw question mark in password": (
        f"http://proxyuser:{_SECRET}?x@proxy.corp:3128",
        "Invalid proxy URL 'http://***': it could not be parsed; percent-encode",
    ),
    "percent-encoded separator": (
        f"http://proxyuser:{_SECRET}%40proxy.corp:3128",
        "Invalid proxy URL 'http://***': it could not be parsed; "
        "separate the credentials from the host with a plain '@'",
    ),
    "fullwidth separator": (
        f"http://proxyuser:{_SECRET}\uff20proxy.corp:3128",
        "Invalid proxy URL 'http://***': it could not be parsed; "
        "separate the credentials from the host with a plain '@'",
    ),
    "non-numeric port": (
        f"http://proxyuser:{_SECRET}@proxy.corp:31x8",
        "Invalid proxy URL 'http://***': the port must be a number from 1 to 65535",
    ),
    "port out of range": (
        f"http://proxyuser:{_SECRET}@proxy.corp:99999",
        "Invalid proxy URL 'http://***@proxy.corp:99999': "
        "the port must be a number from 1 to 65535",
    ),
    "port zero": (
        "http://proxy.corp:0",
        "Invalid proxy URL 'http://proxy.corp:0': the port must be a number from 1 to 65535",
    ),
    "unbracketed IPv6": (
        f"http://proxyuser:{_SECRET}@fe80::1:3128",
        "Invalid proxy URL 'http://***': an IPv6 address must be in brackets",
    ),
    "trailing newline": (
        f"http://proxyuser:{_SECRET}@proxy.corp:3128\n",
        "Invalid proxy URL 'http://***': it has leading or trailing whitespace; remove it",
    ),
    "leading space": (
        f" http://proxyuser:{_SECRET}@proxy.corp:3128",
        "Invalid proxy URL '***@proxy.corp:3128': it has leading or trailing whitespace",
    ),
    "embedded control character": (
        f"http://proxyuser:{_SECRET}@proxy\r\nX-Injected: 1",
        "Invalid proxy URL 'http://***': it contains a control character",
    ),
    "numeric password before raw hash": (
        f"http://proxyuser:1234#{_SECRET}@proxy.corp:3128",
        "Invalid proxy URL 'http://***@proxy.corp:3128': it has an '@' after the host; ",
    ),
    "no host": (
        "http://:3128",
        "Invalid proxy URL 'http://:3128': it has no host",
    ),
}


def _assert_secret_absent(exc: BaseException, caplog: pytest.LogCaptureFixture) -> None:
    assert exc.__cause__ is None
    assert exc.__context__ is None
    for text in (str(exc), repr(exc), "".join(traceback.format_exception(exc)), caplog.text):
        assert _SECRET not in text
        assert "proxyuser" not in text


@pytest.mark.parametrize("client", list(_CONSTRUCTORS))
@pytest.mark.parametrize("case", list(_INVALID))
def test_invalid_proxy_url_raises_redacted_error_at_construction(
    client: str, case: str, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(httpx.Client, "post", _no_network)
    url, expected = _INVALID[case]
    caplog.set_level(logging.DEBUG)
    with pytest.raises(PineconeValueError) as exc_info:
        _CONSTRUCTORS[client](url)
    assert str(exc_info.value).startswith(expected)
    _assert_secret_absent(exc_info.value, caplog)


@pytest.mark.parametrize("case", ["raw slash in password", "numeric password before raw hash"])
def test_parse_failures_name_the_percent_encoding_fix(case: str) -> None:
    with pytest.raises(PineconeValueError, match="percent-encode reserved characters"):
        validate_proxy_url(_INVALID[case][0])


@pytest.mark.parametrize(
    "url",
    [
        "http://proxy.corp:3128",
        "https://proxy.corp:3128",
        "HTTP://proxy.corp:3128",
        "http://proxy.corp:3128/",
        "http://proxy:8080/path",
        f"http://proxyuser:{_SECRET}@proxy.corp:3128",
        "http://proxyuser:s3c%23ret%2Fx%40y@proxy.corp:3128",
        "http://user:pass@[::1]:3128",
        "http://10.0.0.1:3128",
    ],
)
@pytest.mark.parametrize("client", ["PineconeConfig", "Pinecone", "AsyncPinecone", "Index"])
def test_valid_proxy_url_is_accepted(client: str, url: str) -> None:
    _CONSTRUCTORS[client](url)


def test_empty_proxy_url_means_no_proxy() -> None:
    assert PineconeConfig(api_key="k", proxy_url="").proxy_url == ""


@pytest.mark.parametrize("scheme", ["socks5", "socks5h", "SOCKS5"])
def test_socks_proxy_is_accepted_on_rest_and_rejected_on_grpc(scheme: str) -> None:
    url = f"{scheme}://proxyuser:{_SECRET}@proxy.corp:1080"
    validate_proxy_url(url)
    assert PineconeConfig(api_key="k", proxy_url=url).proxy_url == url
    with pytest.raises(PineconeValueError) as exc_info:
        validate_proxy_url(url, grpc=True)
    message = str(exc_info.value)
    assert message.startswith(
        f"gRPC supports only http:// proxies; got '{scheme}://***@proxy.corp:1080'"
    )
    assert message.endswith("which also supports https:// and SOCKS proxies")
    assert _SECRET not in message


def test_socks_without_socksio_keeps_httpx_import_error() -> None:
    if importlib.util.find_spec("socksio") is not None:
        pytest.skip("socksio is installed")
    with pytest.raises(ImportError, match="socksio") as exc_info:
        Pinecone(api_key="k", proxy_url=f"socks5://proxyuser:{_SECRET}@proxy.corp:1080")
    assert _SECRET not in "".join(traceback.format_exception(exc_info.value))


@pytest.mark.parametrize(
    "url", ["http://proxy.corp:31x8", "http://::1:3128", "http://proxy.corp:0"]
)
def test_percent_encoding_hint_only_when_url_has_userinfo(url: str) -> None:
    with pytest.raises(PineconeValueError) as exc_info:
        validate_proxy_url(url)
    assert "percent-encode" not in str(exc_info.value)


def test_grpc_index_rejects_proxy_before_describing_the_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pc = Pinecone(api_key="k", proxy_url=f"https://proxyuser:{_SECRET}@proxy.corp:3128")
    monkeypatch.setattr(pc.indexes, "describe", _no_network)
    with pytest.raises(PineconeValueError, match="gRPC supports only http:// proxies") as exc_info:
        pc.index(name="product-search", grpc=True)
    assert _SECRET not in "".join(traceback.format_exception(exc_info.value))


def test_parser_error_is_not_on_the_exception_chain() -> None:
    with pytest.raises(PineconeValueError) as exc_info:
        validate_proxy_url(f"http://proxyuser:{_SECRET}/x@proxy.corp:3128")
    assert exc_info.value.__context__ is None
    assert exc_info.value.__cause__ is None
