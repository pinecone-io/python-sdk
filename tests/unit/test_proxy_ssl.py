"""Tests for proxy and SSL configuration wiring in HTTPClient and AsyncHTTPClient."""

from __future__ import annotations

import ssl
from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from pinecone._internal.config import PineconeConfig
from pinecone._internal.http_client import (
    AsyncHTTPClient,
    HTTPClient,
    _AsyncRetryTransport,
    _build_socket_options,
    _RetryTransport,
)

API_VERSION = "2025-10"


def _make_config(**overrides: object) -> PineconeConfig:
    defaults = {"api_key": "test-key", "host": "https://api.example.com"}
    defaults.update(overrides)
    return PineconeConfig(**defaults)  # type: ignore[arg-type]


def _socket_ssl_context(client: httpx.Client | httpx.AsyncClient) -> ssl.SSLContext:
    """Return the SSL context of the transport that will actually open the socket.

    The mock-based tests above assert what was handed to ``httpx.Client``, which
    is not the same thing: httpx discards its own ``verify`` argument whenever an
    explicit ``transport`` is supplied. These helpers reach the transport httpx
    selects for a request and read the context it will connect with.
    """
    transport = client._transport_for_url(httpx.URL("https://api.example.com"))
    inner = getattr(transport, "_transport", transport)
    return inner._pool._ssl_context  # type: ignore[no-any-return, union-attr]


def _selected_transport(
    client: httpx.Client | httpx.AsyncClient,
) -> httpx.BaseTransport | httpx.AsyncBaseTransport:
    """Return the transport httpx will actually use for a request to our host.

    #447: a proxy handed to ``httpx.Client``/``httpx.AsyncClient`` makes httpx
    mount a second transport for proxied requests, and this is the method
    (``Client._transport_for_url``) that returns that mount instead of the
    ``_RetryTransport`` we built — silently dropping retries and socket
    options for every proxied request. Reading it back here is the same
    boundary ``_socket_ssl_context`` reads for #421.
    """
    return client._transport_for_url(httpx.URL("https://api.example.com"))


def _pool_socket_options(client: httpx.Client | httpx.AsyncClient) -> object:
    """Return the socket_options set on the pool of the transport httpx selects."""
    transport = _selected_transport(client)
    inner = getattr(transport, "_transport", transport)
    return inner._pool._socket_options  # type: ignore[union-attr]


def _one_ca_pem() -> str:
    """A single valid PEM certificate, taken from the SDK's own default trust store.

    Evaluated at import time, before any test patches ``httpx.Client`` out from
    under the client this reads the certificate from.
    """
    context = _socket_ssl_context(HTTPClient(_make_config(), API_VERSION)._client)
    return ssl.DER_cert_to_PEM_cert(context.get_ca_certs(binary_form=True)[0])


ONE_CA_PEM = _one_ca_pem()


def _sync_context(**overrides: object) -> ssl.SSLContext:
    return _socket_ssl_context(HTTPClient(_make_config(**overrides), API_VERSION)._client)


def _async_context(**overrides: object) -> ssl.SSLContext:
    client = AsyncHTTPClient(_make_config(**overrides), API_VERSION)
    return _socket_ssl_context(client._ensure_client())


def _sync_client(**overrides: object) -> httpx.Client:
    return HTTPClient(_make_config(**overrides), API_VERSION)._client


def _async_client(**overrides: object) -> httpx.AsyncClient:
    return AsyncHTTPClient(_make_config(**overrides), API_VERSION)._ensure_client()


class TestSyncProxyUrl:
    @patch("pinecone._internal.http_client.httpx.HTTPTransport")
    def test_sync_proxy_url_reaches_the_transport(self, mock_transport_cls: MagicMock) -> None:
        config = _make_config(proxy_url="http://proxy:8080")
        HTTPClient(config, API_VERSION)
        _, kwargs = mock_transport_cls.call_args
        assert kwargs["proxy"] == "http://proxy:8080"

    @patch("pinecone._internal.http_client.httpx.HTTPTransport")
    def test_sync_no_proxy_by_default(self, mock_transport_cls: MagicMock) -> None:
        config = _make_config()
        HTTPClient(config, API_VERSION)
        _, kwargs = mock_transport_cls.call_args
        assert kwargs["proxy"] is None

    @patch("pinecone._internal.http_client.httpx.HTTPTransport")
    def test_sync_proxy_headers_become_an_httpx_proxy(self, mock_transport_cls: MagicMock) -> None:
        config = _make_config(
            proxy_url="http://proxy:8080",
            proxy_headers={"Proxy-Authorization": "Basic abc"},
        )
        HTTPClient(config, API_VERSION)
        _, kwargs = mock_transport_cls.call_args
        proxy = kwargs["proxy"]
        assert isinstance(proxy, httpx.Proxy)
        assert proxy.headers["Proxy-Authorization"] == "Basic abc"

    @patch("pinecone._internal.http_client.httpx.Client")
    def test_sync_client_receives_no_proxy_kwarg(self, mock_client_cls: MagicMock) -> None:
        config = _make_config(proxy_url="http://proxy:8080")
        HTTPClient(config, API_VERSION)
        _, kwargs = mock_client_cls.call_args
        assert "proxy" not in kwargs


class TestSyncSSL:
    @patch("pinecone._internal.http_client.httpx.HTTPTransport")
    @patch("pinecone._internal.http_client.httpx.Client")
    def test_sync_ssl_ca_certs_reaches_the_transport(
        self, mock_client_cls: MagicMock, mock_transport_cls: MagicMock, tmp_path: Path
    ) -> None:
        bundle = tmp_path / "cert.pem"
        bundle.write_text(ONE_CA_PEM)
        config = _make_config(ssl_ca_certs=str(bundle))
        HTTPClient(config, API_VERSION)
        _, transport_kwargs = mock_transport_cls.call_args
        _, client_kwargs = mock_client_cls.call_args
        assert isinstance(transport_kwargs["verify"], ssl.SSLContext)
        assert "verify" not in client_kwargs

    @patch("pinecone._internal.http_client.httpx.HTTPTransport")
    def test_sync_ssl_verify_false(self, mock_transport_cls: MagicMock) -> None:
        config = _make_config(ssl_verify=False)
        HTTPClient(config, API_VERSION)
        _, kwargs = mock_transport_cls.call_args
        assert kwargs["verify"] is False

    @patch("pinecone._internal.http_client.httpx.HTTPTransport")
    def test_sync_ssl_verify_true_by_default(self, mock_transport_cls: MagicMock) -> None:
        config = _make_config()
        HTTPClient(config, API_VERSION)
        _, kwargs = mock_transport_cls.call_args
        assert kwargs["verify"] is True


class TestAsyncProxyUrl:
    @patch("pinecone._internal.http_client.httpx.AsyncHTTPTransport")
    def test_async_proxy_url_reaches_the_transport(self, mock_transport_cls: MagicMock) -> None:
        config = _make_config(proxy_url="http://proxy:8080")
        AsyncHTTPClient(config, API_VERSION)._ensure_client()
        _, kwargs = mock_transport_cls.call_args
        assert kwargs["proxy"] == "http://proxy:8080"

    @patch("pinecone._internal.http_client.httpx.AsyncHTTPTransport")
    def test_async_proxy_headers_become_an_httpx_proxy(self, mock_transport_cls: MagicMock) -> None:
        config = _make_config(
            proxy_url="http://proxy:8080",
            proxy_headers={"Proxy-Authorization": "Basic abc"},
        )
        AsyncHTTPClient(config, API_VERSION)._ensure_client()
        _, kwargs = mock_transport_cls.call_args
        proxy = kwargs["proxy"]
        assert isinstance(proxy, httpx.Proxy)
        assert proxy.headers["Proxy-Authorization"] == "Basic abc"

    @patch("pinecone._internal.http_client.httpx.AsyncClient")
    def test_async_client_receives_no_proxy_kwarg(self, mock_client_cls: MagicMock) -> None:
        config = _make_config(proxy_url="http://proxy:8080")
        AsyncHTTPClient(config, API_VERSION)._ensure_client()
        _, kwargs = mock_client_cls.call_args
        assert "proxy" not in kwargs


class TestAsyncSSL:
    @patch("pinecone._internal.http_client.httpx.AsyncHTTPTransport")
    @patch("pinecone._internal.http_client.httpx.AsyncClient")
    def test_async_ssl_ca_certs_reaches_the_transport(
        self, mock_client_cls: MagicMock, mock_transport_cls: MagicMock, tmp_path: Path
    ) -> None:
        bundle = tmp_path / "cert.pem"
        bundle.write_text(ONE_CA_PEM)
        config = _make_config(ssl_ca_certs=str(bundle))
        AsyncHTTPClient(config, API_VERSION)._ensure_client()
        _, transport_kwargs = mock_transport_cls.call_args
        _, client_kwargs = mock_client_cls.call_args
        assert isinstance(transport_kwargs["verify"], ssl.SSLContext)
        assert "verify" not in client_kwargs

    @patch("pinecone._internal.http_client.httpx.AsyncHTTPTransport")
    def test_async_ssl_verify_false(self, mock_transport_cls: MagicMock) -> None:
        config = _make_config(ssl_verify=False)
        client = AsyncHTTPClient(config, API_VERSION)
        client._ensure_client()
        _, kwargs = mock_transport_cls.call_args
        assert kwargs["verify"] is False


class TestSSLConfigReachesTheSocket:
    """#421: ssl_ca_certs and ssl_verify must configure the transport, not just the client."""

    @pytest.mark.parametrize("context_for", [_sync_context, _async_context], ids=["sync", "async"])
    def test_default_verifies_against_the_default_bundle(
        self, context_for: Callable[..., ssl.SSLContext]
    ) -> None:
        context = context_for()
        assert context.verify_mode is ssl.CERT_REQUIRED
        assert context.check_hostname is True
        assert context.get_ca_certs() != []

    @pytest.mark.parametrize("context_for", [_sync_context, _async_context], ids=["sync", "async"])
    def test_ssl_verify_false_disables_verification(
        self, context_for: Callable[..., ssl.SSLContext]
    ) -> None:
        context = context_for(ssl_verify=False)
        assert context.verify_mode is ssl.CERT_NONE
        assert context.check_hostname is False

    @pytest.mark.parametrize("context_for", [_sync_context, _async_context], ids=["sync", "async"])
    def test_ssl_ca_certs_file_replaces_the_trust_store(
        self, context_for: Callable[..., ssl.SSLContext], tmp_path: Path
    ) -> None:
        bundle = tmp_path / "one-ca.pem"
        bundle.write_text(ONE_CA_PEM)
        context = context_for(ssl_ca_certs=str(bundle))
        assert context.verify_mode is ssl.CERT_REQUIRED
        assert len(context.get_ca_certs()) == 1

    @pytest.mark.parametrize("context_for", [_sync_context, _async_context], ids=["sync", "async"])
    def test_ssl_ca_certs_directory_replaces_the_trust_store(
        self, context_for: Callable[..., ssl.SSLContext], tmp_path: Path
    ) -> None:
        context = context_for(ssl_ca_certs=str(tmp_path))
        assert context.verify_mode is ssl.CERT_REQUIRED
        assert context.get_ca_certs() == []

    @pytest.mark.parametrize("build", [_sync_context, _async_context], ids=["sync", "async"])
    def test_missing_ssl_ca_certs_file_raises(
        self, build: Callable[..., ssl.SSLContext], tmp_path: Path
    ) -> None:
        with pytest.raises(FileNotFoundError):
            build(ssl_ca_certs=str(tmp_path / "absent.pem"))

    @pytest.mark.parametrize("context_for", [_sync_context, _async_context], ids=["sync", "async"])
    def test_unparseable_ssl_ca_certs_file_raises(
        self, context_for: Callable[..., ssl.SSLContext], tmp_path: Path
    ) -> None:
        bundle = tmp_path / "garbage.pem"
        bundle.write_text("not a certificate\n")
        with pytest.raises(ssl.SSLError):
            context_for(ssl_ca_certs=str(bundle))

    @pytest.mark.parametrize("context_for", [_sync_context, _async_context], ids=["sync", "async"])
    def test_ssl_ca_certs_takes_precedence_over_ssl_verify_false(
        self, context_for: Callable[..., ssl.SSLContext], tmp_path: Path
    ) -> None:
        bundle = tmp_path / "one-ca.pem"
        bundle.write_text(ONE_CA_PEM)
        context = context_for(ssl_ca_certs=str(bundle), ssl_verify=False)
        assert context.verify_mode is ssl.CERT_REQUIRED
        assert len(context.get_ca_certs()) == 1


class TestProxyDoesNotBypassRetryOrSocketOptions:
    """#447: proxy_url must not route requests around _RetryTransport or socket tuning."""

    @pytest.mark.parametrize("client_for", [_sync_client, _async_client], ids=["sync", "async"])
    def test_retry_transport_selected_without_proxy(
        self, client_for: Callable[..., httpx.Client | httpx.AsyncClient]
    ) -> None:
        transport = _selected_transport(client_for())
        assert isinstance(transport, (_RetryTransport, _AsyncRetryTransport))

    @pytest.mark.parametrize("client_for", [_sync_client, _async_client], ids=["sync", "async"])
    def test_retry_transport_still_selected_with_proxy(
        self, client_for: Callable[..., httpx.Client | httpx.AsyncClient]
    ) -> None:
        transport = _selected_transport(client_for(proxy_url="http://proxy:8080"))
        assert isinstance(transport, (_RetryTransport, _AsyncRetryTransport))

    @pytest.mark.parametrize("client_for", [_sync_client, _async_client], ids=["sync", "async"])
    def test_socket_options_reach_the_pool_with_proxy(
        self, client_for: Callable[..., httpx.Client | httpx.AsyncClient]
    ) -> None:
        client = client_for(proxy_url="http://proxy:8080")
        assert _pool_socket_options(client) == _build_socket_options()

    @pytest.mark.parametrize("client_for", [_sync_client, _async_client], ids=["sync", "async"])
    def test_pool_is_proxy_aware_when_proxy_url_set(
        self, client_for: Callable[..., httpx.Client | httpx.AsyncClient]
    ) -> None:
        client = client_for(proxy_url="http://proxy:8080")
        transport = _selected_transport(client)
        inner = getattr(transport, "_transport", transport)
        assert "Proxy" in type(inner._pool).__name__

    @pytest.mark.parametrize("client_for", [_sync_client, _async_client], ids=["sync", "async"])
    def test_pool_is_plain_without_proxy_url(
        self, client_for: Callable[..., httpx.Client | httpx.AsyncClient]
    ) -> None:
        client = client_for()
        transport = _selected_transport(client)
        inner = getattr(transport, "_transport", transport)
        assert "Proxy" not in type(inner._pool).__name__
