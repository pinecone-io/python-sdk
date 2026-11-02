"""gRPC sends proxy credentials on the CONNECT request.

Drives the real Rust channel through a local fake proxy that records each CONNECT
and refuses it with a 407, so no tunnel or TLS handshake is ever attempted.
"""

from __future__ import annotations

import base64
import socket
import threading
import traceback
from collections.abc import Callable, Iterator

import pytest

from pinecone import Pinecone
from pinecone._internal.config import RetryConfig
from pinecone.errors.exceptions import PineconeConnectionError, PineconeValueError
from pinecone.grpc import GrpcIndex

pytest.importorskip("pinecone._grpc", reason="Rust extension not available")

_SECRET = "SENTINEL-SECRET-1234"
_HOST = "idx.svc.example.test"


class _FakeProxy:
    def __init__(self) -> None:
        self._sock = socket.socket()
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen()
        self.port = self._sock.getsockname()[1]
        self.requests: list[str] = []
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        while True:
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            with conn:
                head = b""
                while b"\r\n\r\n" not in head:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    head += chunk
                self.requests.append(head.decode("latin-1"))
                conn.sendall(b"HTTP/1.1 407 Proxy Authentication Required\r\n\r\n")

    def close(self) -> None:
        self._sock.close()

    def proxy_authorization(self) -> list[str | None]:
        values: list[str | None] = []
        for request in self.requests:
            assert request.startswith(f"CONNECT {_HOST}:443 HTTP/1.1\r\n"), request
            found = None
            for line in request.split("\r\n")[1:]:
                name, _, value = line.partition(": ")
                if name.lower() == "proxy-authorization":
                    found = value
            values.append(found)
        return values


@pytest.fixture
def proxy() -> Iterator[_FakeProxy]:
    server = _FakeProxy()
    yield server
    server.close()


def _basic(credentials: str) -> str:
    return "Basic " + base64.b64encode(credentials.encode()).decode()


def _describe_through(idx: GrpcIndex) -> str:
    with pytest.raises(PineconeConnectionError) as exc_info:
        idx.describe_index_stats()
    return str(exc_info.value)


def _grpc_index(proxy_url: str) -> GrpcIndex:
    return GrpcIndex(
        host=f"https://{_HOST}",
        api_key="test-api-key",
        retry_config=RetryConfig(max_retries=0),
        proxy_url=proxy_url,
    )


def test_credentialed_proxy_url_sends_basic_auth(proxy: _FakeProxy) -> None:
    idx = _grpc_index(f"http://proxyuser:{_SECRET}@127.0.0.1:{proxy.port}")
    message = _describe_through(idx)
    assert proxy.proxy_authorization() == [_basic(f"proxyuser:{_SECRET}")]
    assert _SECRET not in message


def test_percent_encoded_credentials_are_decoded(proxy: _FakeProxy) -> None:
    idx = _grpc_index(f"http://us%40r:p%3As%2Fs%20w@127.0.0.1:{proxy.port}")
    _describe_through(idx)
    assert proxy.proxy_authorization() == [_basic("us@r:p:s/s w")]


def test_proxy_url_without_userinfo_sends_no_auth(proxy: _FakeProxy) -> None:
    idx = _grpc_index(f"http://127.0.0.1:{proxy.port}")
    _describe_through(idx)
    assert proxy.proxy_authorization() == [None]


@pytest.mark.parametrize("scheme", ["https", "HTTPS", "socks5", "socks5h", "ftp"])
def test_non_http_proxy_scheme_is_rejected_before_connect(
    proxy: _FakeProxy, scheme: str, caplog: pytest.LogCaptureFixture
) -> None:
    proxy_url = f"{scheme}://proxyuser:{_SECRET}@127.0.0.1:{proxy.port}"
    with pytest.raises(PineconeValueError) as exc_info:
        _grpc_index(proxy_url)
    exc = exc_info.value
    assert str(exc) == (
        f"gRPC supports only http:// proxies; got '{scheme}://***@127.0.0.1:{proxy.port}'; "
        "use an http:// proxy URL (traffic inside the tunnel is still TLS) or use REST "
        "(grpc=False), which also supports https:// and SOCKS proxies"
    )
    for text in (str(exc), repr(exc), "".join(traceback.format_exception(exc)), caplog.text):
        assert _SECRET not in text
    assert proxy.requests == []


def test_scheme_less_proxy_url_is_rejected(proxy: _FakeProxy) -> None:
    with pytest.raises(PineconeValueError) as exc_info:
        _grpc_index(f"proxyuser:{_SECRET}@127.0.0.1:{proxy.port}")
    assert str(exc_info.value).startswith("gRPC supports only http:// proxies; got '")
    assert _SECRET not in "".join(traceback.format_exception(exc_info.value))
    assert proxy.requests == []


def test_pinecone_index_grpc_rejects_https_proxy(proxy: _FakeProxy) -> None:
    pc = Pinecone(
        api_key="test-api-key",
        proxy_url=f"https://proxyuser:{_SECRET}@127.0.0.1:{proxy.port}",
    )
    with pytest.raises(PineconeValueError, match="gRPC supports only http:// proxies") as exc_info:
        pc.index(host=f"https://{_HOST}", grpc=True)
    assert _SECRET not in "".join(traceback.format_exception(exc_info.value))
    assert proxy.requests == []


@pytest.mark.parametrize(
    ("password", "reason"),
    [
        (f"{_SECRET}/x", "'http://***': it could not be parsed; "),
        (f"{_SECRET}#x", "'http://***': it could not be parsed; "),
        (f"1234#{_SECRET}", "'http://***@127.0.0.1:{port}': it has an '@' after the host; "),
    ],
)
def test_unencoded_password_is_rejected_before_connect(
    proxy: _FakeProxy, password: str, reason: str, caplog: pytest.LogCaptureFixture
) -> None:
    with pytest.raises(PineconeValueError) as exc_info:
        _grpc_index(f"http://proxyuser:{password}@127.0.0.1:{proxy.port}")
    exc = exc_info.value
    assert str(exc).startswith(f"Invalid proxy URL {reason.format(port=proxy.port)}")
    assert exc.__cause__ is None
    assert exc.__context__ is None
    for text in (str(exc), repr(exc), "".join(traceback.format_exception(exc)), caplog.text):
        assert _SECRET not in text
    assert proxy.requests == []


@pytest.mark.parametrize(
    "scheme", ["http", "HTTP", "Http", "https", "HTTPS", "socks5", "socks5h", "ftp", "ws"]
)
def test_python_and_rust_proxy_scheme_rules_agree(scheme: str) -> None:
    from pinecone._grpc import GrpcChannel  # type: ignore[import-not-found]

    from pinecone._internal.config import validate_proxy_url

    proxy_url = f"{scheme}://proxyuser:{_SECRET}@127.0.0.1:3128"

    def accepted(check: Callable[[], object]) -> bool:
        try:
            check()
        except PineconeValueError as exc:
            assert str(exc).startswith("gRPC supports only http:// proxies; got '")
            assert _SECRET not in str(exc)
            return False
        return True

    python_ok = accepted(lambda: validate_proxy_url(proxy_url, grpc=True))
    rust_ok = accepted(
        lambda: GrpcChannel(
            f"https://{_HOST}:443",
            "test-api-key",
            "2025-10",
            "0.0.0",
            proxy_url=proxy_url,
        )
    )
    assert python_ok == rust_ok == (scheme.lower() == "http")
