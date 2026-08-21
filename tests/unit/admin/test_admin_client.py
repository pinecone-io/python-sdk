"""Unit tests for the Admin client constructor and OAuth token flow."""

from __future__ import annotations

import ssl
from typing import Any
from unittest.mock import patch

import httpx
import pytest
import respx
from httpx import Response

from pinecone import __version__
from pinecone._internal.constants import DEFAULT_BASE_URL
from pinecone.admin.admin import _OAUTH_URL, Admin
from pinecone.errors.exceptions import (
    ApiError,
    PineconeConnectionError,
    PineconeTimeoutError,
    ValidationError,
)


def _token_response(token: str = "test-access-token") -> dict[str, Any]:
    """Return a valid OAuth token response payload."""
    return {
        "access_token": token,
        "token_type": "Bearer",
        "expires_in": 1800,
    }


class TestAdminValidation:
    """Test validation of client_id and client_secret."""

    def test_admin_requires_client_id(self) -> None:
        with pytest.raises(ValidationError, match="client_id"):
            Admin(client_secret="secret")

    def test_admin_requires_client_secret(self) -> None:
        with pytest.raises(ValidationError, match="client_secret"):
            Admin(client_id="id")

    def test_admin_rejects_empty_client_id(self) -> None:
        with pytest.raises(ValidationError, match="client_id"):
            Admin(client_id="", client_secret="secret")

    def test_admin_rejects_empty_client_secret(self) -> None:
        with pytest.raises(ValidationError, match="client_secret"):
            Admin(client_id="id", client_secret="")

    def test_admin_rejects_whitespace_client_id(self) -> None:
        with pytest.raises(ValidationError, match="client_id"):
            Admin(client_id="   ", client_secret="secret")

    def test_admin_rejects_whitespace_client_secret(self) -> None:
        with pytest.raises(ValidationError, match="client_secret"):
            Admin(client_id="id", client_secret="   ")

    def test_admin_no_env_fallback_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("PINECONE_CLIENT_ID", raising=False)
        monkeypatch.delenv("PINECONE_CLIENT_SECRET", raising=False)
        with pytest.raises(ValidationError, match="client_id"):
            Admin()


class TestAdminEnvVarFallback:
    """Test environment variable fallback for credentials."""

    @respx.mock
    def test_admin_env_var_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PINECONE_CLIENT_ID", "env-client-id")
        monkeypatch.setenv("PINECONE_CLIENT_SECRET", "env-client-secret")

        respx.post(_OAUTH_URL).mock(return_value=Response(200, json=_token_response()))

        admin = Admin()
        assert admin._http is not None
        admin.close()

    @respx.mock
    def test_admin_explicit_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PINECONE_CLIENT_ID", "env-id")
        monkeypatch.setenv("PINECONE_CLIENT_SECRET", "env-secret")

        route = respx.post(_OAUTH_URL).mock(return_value=Response(200, json=_token_response()))

        admin = Admin(client_id="explicit-id", client_secret="explicit-secret")
        admin.close()

        # Verify the explicit credentials were sent, not the env vars
        request = route.calls.last.request
        import orjson

        body = orjson.loads(request.content)
        assert body["client_id"] == "explicit-id"
        assert body["client_secret"] == "explicit-secret"


class TestAdminTokenFetch:
    """Test the OAuth token exchange flow."""

    @respx.mock
    def test_admin_fetches_bearer_token(self) -> None:
        route = respx.post(_OAUTH_URL).mock(
            return_value=Response(200, json=_token_response("my-bearer-token"))
        )

        admin = Admin(client_id="test-id", client_secret="test-secret")

        # Verify the OAuth request was made with correct body
        assert route.called
        request = route.calls.last.request
        import orjson

        body = orjson.loads(request.content)
        assert body["client_id"] == "test-id"
        assert body["client_secret"] == "test-secret"
        assert body["grant_type"] == "client_credentials"
        assert body["audience"] == "https://api.pinecone.io/"

        # Verify the Bearer token is used in the HTTPClient headers
        auth_header = admin._http._headers.get("Authorization")
        assert auth_header == "Bearer my-bearer-token"

        admin.close()

    @respx.mock
    def test_admin_token_request_includes_api_version(self) -> None:
        route = respx.post(_OAUTH_URL).mock(return_value=Response(200, json=_token_response()))

        admin = Admin(client_id="test-id", client_secret="test-secret")

        request = route.calls.last.request
        assert request.headers["X-Pinecone-Api-Version"] == "2026-07"

        admin.close()

    @respx.mock
    def test_admin_token_fetch_401_raises_api_error(self) -> None:
        respx.post(_OAUTH_URL).mock(
            return_value=Response(
                401,
                json={"error": "access_denied", "error_description": "Unauthorized"},
            )
        )

        with pytest.raises(ApiError, match="Unauthorized") as exc_info:
            Admin(client_id="bad-id", client_secret="bad-secret")

        assert exc_info.value.status_code == 401

    @respx.mock
    def test_admin_token_fetch_403_raises_api_error(self) -> None:
        respx.post(_OAUTH_URL).mock(
            return_value=Response(
                403,
                json={"error": "forbidden", "error_description": "Forbidden"},
            )
        )

        with pytest.raises(ApiError, match="Forbidden") as exc_info:
            Admin(client_id="bad-id", client_secret="bad-secret")

        assert exc_info.value.status_code == 403

    @respx.mock
    def test_admin_token_fetch_missing_access_token(self) -> None:
        respx.post(_OAUTH_URL).mock(
            return_value=Response(200, json={"token_type": "Bearer", "expires_in": 1800})
        )

        with pytest.raises(ApiError, match="missing access_token"):
            Admin(client_id="test-id", client_secret="test-secret")

    @respx.mock
    def test_admin_token_fetch_error_without_description(self) -> None:
        respx.post(_OAUTH_URL).mock(return_value=Response(400, json={"error": "invalid_request"}))

        with pytest.raises(ApiError, match="invalid_request") as exc_info:
            Admin(client_id="test-id", client_secret="test-secret")

        assert exc_info.value.status_code == 400


class TestAdminHeaders:
    """Test that the Admin client configures correct headers."""

    @respx.mock
    def test_admin_sets_api_version_header(self) -> None:
        respx.post(_OAUTH_URL).mock(return_value=Response(200, json=_token_response()))

        admin = Admin(client_id="test-id", client_secret="test-secret")
        assert admin._http._headers["X-Pinecone-Api-Version"] == "2026-07"
        admin.close()

    @respx.mock
    def test_admin_additional_headers(self) -> None:
        respx.post(_OAUTH_URL).mock(return_value=Response(200, json=_token_response()))

        admin = Admin(
            client_id="test-id",
            client_secret="test-secret",
            additional_headers={"X-Custom": "value"},
        )
        assert admin._http._headers["X-Custom"] == "value"
        assert admin._http._headers["Authorization"] == "Bearer test-access-token"
        admin.close()


class TestAdminApiKeyNotLeaked:
    """Test that the Admin client does not leak data-plane API keys."""

    @respx.mock
    def test_admin_does_not_include_api_key_header(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When PINECONE_API_KEY is set, Admin must NOT send it as Api-Key."""
        monkeypatch.setenv("PINECONE_API_KEY", "data-plane-key-12345")

        respx.post(_OAUTH_URL).mock(return_value=Response(200, json=_token_response()))

        admin = Admin(client_id="test-id", client_secret="test-secret")
        assert "Api-Key" not in admin._http._headers
        assert admin._http._headers["Authorization"] == "Bearer test-access-token"
        admin.close()

    @respx.mock
    def test_admin_api_key_empty_without_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without PINECONE_API_KEY env var, Api-Key header is still absent."""
        monkeypatch.delenv("PINECONE_API_KEY", raising=False)

        respx.post(_OAUTH_URL).mock(return_value=Response(200, json=_token_response()))

        admin = Admin(client_id="test-id", client_secret="test-secret")
        assert "Api-Key" not in admin._http._headers
        admin.close()


class TestAdminOAuthUserAgent:
    """Test that the OAuth token request includes a User-Agent header."""

    @respx.mock
    def test_oauth_request_includes_user_agent(self) -> None:
        route = respx.post(_OAUTH_URL).mock(return_value=Response(200, json=_token_response()))

        admin = Admin(client_id="test-id", client_secret="test-secret")

        request = route.calls.last.request
        ua = request.headers.get("User-Agent", "")
        assert ua.startswith("python-client-")

        admin.close()


class TestAdminSourceTag:
    """Test that source_tag is correctly threaded into User-Agent headers."""

    @respx.mock
    def test_source_tag_appears_in_oauth_user_agent(self) -> None:
        route = respx.post(_OAUTH_URL).mock(return_value=Response(200, json=_token_response()))

        admin = Admin(client_id="test-id", client_secret="test-secret", source_tag="myapp")

        ua = route.calls.last.request.headers["User-Agent"]
        assert ua.startswith("python-client-")
        assert "source_tag=myapp" in ua

        admin.close()

    @respx.mock
    def test_source_tag_appears_in_admin_api_user_agent(self) -> None:
        respx.post(_OAUTH_URL).mock(return_value=Response(200, json=_token_response()))
        admin_route = respx.get(f"{DEFAULT_BASE_URL}/admin/organizations").mock(
            return_value=Response(200, json={"data": []})
        )

        admin = Admin(client_id="test-id", client_secret="test-secret", source_tag="myapp")
        admin.organizations.list()

        ua = admin_route.calls.last.request.headers["User-Agent"]
        assert "source_tag=myapp" in ua

        admin.close()

    @respx.mock
    def test_no_source_tag_omits_suffix(self) -> None:
        route = respx.post(_OAUTH_URL).mock(return_value=Response(200, json=_token_response()))

        admin = Admin(client_id="test-id", client_secret="test-secret")

        ua = route.calls.last.request.headers["User-Agent"]
        assert ua == f"python-client-{__version__}"

        admin.close()


class TestAdminProxyAndSsl:
    """Test that proxy_url and ssl_verify are accepted and forwarded."""

    @respx.mock
    def test_admin_accepts_proxy_url(self) -> None:
        respx.post(_OAUTH_URL).mock(return_value=Response(200, json=_token_response()))

        admin = Admin(
            client_id="test-id",
            client_secret="test-secret",
            proxy_url="http://proxy.example.com:8080",
        )
        assert admin._http._config.proxy_url == "http://proxy.example.com:8080"
        admin.close()

    @respx.mock
    def test_admin_accepts_ssl_verify_false(self) -> None:
        respx.post(_OAUTH_URL).mock(return_value=Response(200, json=_token_response()))

        admin = Admin(
            client_id="test-id",
            client_secret="test-secret",
            ssl_verify=False,
        )
        assert admin._http._config.ssl_verify is False
        admin.close()

    @staticmethod
    def _oauth_ssl_context(**kwargs: Any) -> ssl.SSLContext:
        """The SSL context the OAuth token exchange's own transport connects with.

        The token exchange builds a separate ``httpx.Client``, so it does not
        inherit the fix applied to the config-driven client (#421). Capturing the
        transport instance is the only way to see what it will really connect
        with: ``httpx.Client`` discards its own ``verify`` when handed one.
        """
        respx.post(_OAUTH_URL).mock(return_value=Response(200, json=_token_response()))
        created: list[httpx.HTTPTransport] = []
        real = httpx.HTTPTransport

        def spy(**transport_kwargs: Any) -> httpx.HTTPTransport:
            transport = real(**transport_kwargs)
            created.append(transport)
            return transport

        with patch("pinecone.admin.admin.httpx.HTTPTransport", spy):
            admin = Admin(client_id="test-id", client_secret="test-secret", **kwargs)
        admin.close()
        assert created, "the OAuth token exchange built no transport"
        return created[0]._pool._ssl_context  # type: ignore[attr-defined, no-any-return]

    @respx.mock
    def test_admin_ssl_verify_false_reaches_the_oauth_transport(self) -> None:
        context = self._oauth_ssl_context(ssl_verify=False)
        assert context.verify_mode is ssl.CERT_NONE
        assert context.check_hostname is False

    @respx.mock
    def test_admin_oauth_transport_verifies_by_default(self) -> None:
        context = self._oauth_ssl_context()
        assert context.verify_mode is ssl.CERT_REQUIRED
        assert context.check_hostname is True

    @staticmethod
    def _oauth_transport(**kwargs: Any) -> httpx.HTTPTransport:
        """The transport the OAuth token exchange's ``_RetryTransport`` wraps.

        Mirrors ``_oauth_ssl_context``: the token exchange builds its own
        ``httpx.Client``/``_RetryTransport`` pair, so #447 (a proxy silently
        bypassing retries and socket options) has to be checked on this
        transport too, not just on the config-driven client.
        """
        respx.post(_OAUTH_URL).mock(return_value=Response(200, json=_token_response()))
        created: list[httpx.HTTPTransport] = []
        real = httpx.HTTPTransport

        def spy(**transport_kwargs: Any) -> httpx.HTTPTransport:
            transport = real(**transport_kwargs)
            created.append(transport)
            return transport

        with patch("pinecone.admin.admin.httpx.HTTPTransport", spy):
            admin = Admin(client_id="test-id", client_secret="test-secret", **kwargs)
        admin.close()
        assert created, "the OAuth token exchange built no transport"
        return created[0]

    @respx.mock
    def test_admin_oauth_proxy_reaches_the_transport(self) -> None:
        transport = self._oauth_transport(proxy_url="http://proxy.example.com:8080")
        assert "Proxy" in type(transport._pool).__name__  # type: ignore[attr-defined]

    @respx.mock
    def test_admin_oauth_transport_has_no_proxy_by_default(self) -> None:
        transport = self._oauth_transport()
        assert "Proxy" not in type(transport._pool).__name__  # type: ignore[attr-defined]

    @respx.mock
    def test_admin_oauth_client_receives_no_proxy_or_verify_kwarg(self) -> None:
        respx.post(_OAUTH_URL).mock(return_value=Response(200, json=_token_response()))
        real_client = httpx.Client
        captured: list[dict[str, Any]] = []

        def spy(**client_kwargs: Any) -> httpx.Client:
            captured.append(client_kwargs)
            return real_client(**client_kwargs)

        with patch("pinecone.admin.admin.httpx.Client", spy):
            admin = Admin(
                client_id="test-id",
                client_secret="test-secret",
                proxy_url="http://proxy.example.com:8080",
            )
        admin.close()
        assert captured, "the OAuth token exchange built no client"
        assert "proxy" not in captured[0]
        assert "verify" not in captured[0]


class TestAdminContextManager:
    """Test context manager support."""

    @respx.mock
    def test_admin_context_manager(self) -> None:
        respx.post(_OAUTH_URL).mock(return_value=Response(200, json=_token_response()))

        with Admin(client_id="test-id", client_secret="test-secret") as admin:
            assert admin._http is not None


class TestAdminOAuthNetworkErrors:
    """Test network-level errors during OAuth token fetch."""

    @respx.mock
    def test_oauth_network_read_timeout_raises_pinecone_timeout_error(self) -> None:
        respx.post(_OAUTH_URL).mock(side_effect=httpx.ReadTimeout("read timed out"))

        with pytest.raises(PineconeTimeoutError, match="read timed out") as exc_info:
            Admin(client_id="test-id", client_secret="test-secret")

        assert isinstance(exc_info.value.__cause__, httpx.ReadTimeout)

    @respx.mock
    def test_oauth_network_connect_error_raises_pinecone_connection_error(self) -> None:
        respx.post(_OAUTH_URL).mock(side_effect=httpx.ConnectError("connection refused"))

        with pytest.raises(PineconeConnectionError, match="connection refused") as exc_info:
            Admin(client_id="test-id", client_secret="test-secret")

        assert isinstance(exc_info.value.__cause__, httpx.ConnectError)

    @respx.mock
    def test_oauth_network_pool_timeout_raises_pinecone_timeout_error(self) -> None:
        respx.post(_OAUTH_URL).mock(side_effect=httpx.PoolTimeout("pool exhausted"))

        with pytest.raises(PineconeTimeoutError):
            Admin(client_id="test-id", client_secret="test-secret")


class TestAdminOAuthNonJsonErrorBody:
    """Test error handling when OAuth error response body is not valid JSON."""

    @respx.mock
    def test_oauth_non_json_error_with_html_body_falls_back_to_default_message(self) -> None:
        respx.post(_OAUTH_URL).mock(
            return_value=Response(
                502,
                text="<html>Bad Gateway</html>",
                headers={"content-type": "text/html"},
            )
        )

        with pytest.raises(ApiError, match="OAuth token request failed") as exc_info:
            Admin(client_id="test-id", client_secret="test-secret")

        assert exc_info.value.status_code == 502
        assert exc_info.value.body is None

    @respx.mock
    def test_oauth_non_json_error_with_empty_body_falls_back_to_default_message(self) -> None:
        respx.post(_OAUTH_URL).mock(return_value=Response(500, text=""))

        with pytest.raises(ApiError, match="OAuth token request failed") as exc_info:
            Admin(client_id="test-id", client_secret="test-secret")

        assert exc_info.value.status_code == 500
        assert exc_info.value.body is None


class TestAdminUrlOverrides:
    """Test the host and oauth_url overrides used to target simulators."""

    @pytest.fixture(autouse=True)
    def _clear_host_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("PINECONE_CONTROLLER_HOST", raising=False)

    @respx.mock
    def test_defaults_unchanged(self) -> None:
        oauth = respx.post(_OAUTH_URL).mock(return_value=Response(200, json=_token_response()))
        orgs = respx.get(f"{DEFAULT_BASE_URL}/admin/organizations").mock(
            return_value=Response(200, json={"data": []})
        )

        admin = Admin(client_id="test-id", client_secret="test-secret")
        admin.organizations.list()

        assert admin._http._config.host == DEFAULT_BASE_URL
        assert oauth.called
        assert orgs.called
        admin.close()

    @respx.mock
    def test_host_override_targets_admin_requests(self) -> None:
        respx.post(_OAUTH_URL).mock(return_value=Response(200, json=_token_response()))
        orgs = respx.get("http://localhost:5080/admin/organizations").mock(
            return_value=Response(200, json={"data": []})
        )

        admin = Admin(
            client_id="test-id",
            client_secret="test-secret",
            host="http://localhost:5080",
        )
        admin.organizations.list()

        assert admin._http._config.host == "http://localhost:5080"
        assert orgs.called
        admin.close()

    @respx.mock
    def test_host_override_leaves_oauth_url_alone(self) -> None:
        oauth = respx.post(_OAUTH_URL).mock(return_value=Response(200, json=_token_response()))

        admin = Admin(
            client_id="test-id",
            client_secret="test-secret",
            host="http://localhost:5080",
        )

        assert str(oauth.calls.last.request.url) == _OAUTH_URL
        admin.close()

    @respx.mock
    def test_host_env_var_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PINECONE_CONTROLLER_HOST", "http://localhost:5080")
        respx.post(_OAUTH_URL).mock(return_value=Response(200, json=_token_response()))

        admin = Admin(client_id="test-id", client_secret="test-secret")

        assert admin._http._config.host == "http://localhost:5080"
        admin.close()

    @respx.mock
    def test_explicit_host_beats_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PINECONE_CONTROLLER_HOST", "http://env-host:5080")
        respx.post(_OAUTH_URL).mock(return_value=Response(200, json=_token_response()))

        admin = Admin(
            client_id="test-id",
            client_secret="test-secret",
            host="http://explicit-host:5080",
        )

        assert admin._http._config.host == "http://explicit-host:5080"
        admin.close()

    @respx.mock
    def test_host_without_scheme_gets_https(self) -> None:
        respx.post(_OAUTH_URL).mock(return_value=Response(200, json=_token_response()))

        admin = Admin(
            client_id="test-id",
            client_secret="test-secret",
            host="admin.example.com",
        )

        assert admin._http._config.host == "https://admin.example.com"
        admin.close()

    @respx.mock
    def test_oauth_url_override(self) -> None:
        override = "http://localhost:5080/oauth/token"
        oauth = respx.post(override).mock(return_value=Response(200, json=_token_response()))
        production = respx.post(_OAUTH_URL).mock(
            return_value=Response(200, json=_token_response("production-token"))
        )

        admin = Admin(
            client_id="test-id",
            client_secret="test-secret",
            oauth_url=override,
        )

        assert oauth.called
        assert not production.called
        assert admin._http._headers["Authorization"] == "Bearer test-access-token"
        admin.close()

    @respx.mock
    def test_oauth_url_override_leaves_host_alone(self) -> None:
        respx.post("http://localhost:5080/oauth/token").mock(
            return_value=Response(200, json=_token_response())
        )

        admin = Admin(
            client_id="test-id",
            client_secret="test-secret",
            oauth_url="http://localhost:5080/oauth/token",
        )

        assert admin._http._config.host == DEFAULT_BASE_URL
        admin.close()

    @respx.mock
    def test_oauth_url_without_scheme_gets_https(self) -> None:
        oauth = respx.post("https://login.example.com/oauth/token").mock(
            return_value=Response(200, json=_token_response())
        )

        admin = Admin(
            client_id="test-id",
            client_secret="test-secret",
            oauth_url="login.example.com/oauth/token",
        )

        assert oauth.called
        admin.close()

    @respx.mock
    def test_oauth_url_override_keeps_production_audience(self) -> None:
        oauth = respx.post("http://localhost:5080/oauth/token").mock(
            return_value=Response(200, json=_token_response())
        )

        admin = Admin(
            client_id="test-id",
            client_secret="test-secret",
            oauth_url="http://localhost:5080/oauth/token",
        )

        import orjson

        body = orjson.loads(oauth.calls.last.request.content)
        assert body["audience"] == "https://api.pinecone.io/"
        assert body["grant_type"] == "client_credentials"
        admin.close()

    @respx.mock
    def test_both_overrides_target_one_simulator(self) -> None:
        oauth = respx.post("http://localhost:5080/oauth/token").mock(
            return_value=Response(200, json=_token_response("sim-token"))
        )
        orgs = respx.get("http://localhost:5080/admin/organizations").mock(
            return_value=Response(200, json={"data": []})
        )

        admin = Admin(
            client_id="test-id",
            client_secret="test-secret",
            host="http://localhost:5080",
            oauth_url="http://localhost:5080/oauth/token",
        )
        admin.organizations.list()

        assert oauth.called
        assert orgs.called
        assert orgs.calls.last.request.headers["Authorization"] == "Bearer sim-token"
        admin.close()

    @respx.mock
    def test_default_host_path_still_suppresses_api_key_header(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Filling in the default host must not resurrect the PINECONE_API_KEY fallback."""
        monkeypatch.setenv("PINECONE_API_KEY", "data-plane-key-12345")
        respx.post(_OAUTH_URL).mock(return_value=Response(200, json=_token_response()))

        admin = Admin(client_id="test-id", client_secret="test-secret")

        assert "Api-Key" not in admin._http._headers
        assert admin._http._config.host == DEFAULT_BASE_URL
        admin.close()

    @respx.mock
    def test_host_override_still_suppresses_api_key_header(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PINECONE_API_KEY", "data-plane-key-12345")
        respx.post(_OAUTH_URL).mock(return_value=Response(200, json=_token_response()))

        admin = Admin(
            client_id="test-id",
            client_secret="test-secret",
            host="http://localhost:5080",
        )

        assert "Api-Key" not in admin._http._headers
        admin.close()

    @respx.mock
    def test_overrides_are_keyword_only(self) -> None:
        respx.post(_OAUTH_URL).mock(return_value=Response(200, json=_token_response()))

        with pytest.raises(TypeError):
            Admin("test-id", "test-secret", "http://localhost:5080")  # type: ignore[misc]
