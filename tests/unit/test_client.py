"""Unit tests for the Pinecone client constructor and configuration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pinecone import Pinecone
from pinecone._internal.adaptive import _AdaptiveLimiterRegistry
from pinecone._internal.config import RetryConfig
from pinecone._internal.constants import DEFAULT_BASE_URL
from pinecone.errors.exceptions import ValidationError


class TestBasicConstruction:
    """Test basic client construction with explicit arguments."""

    def test_create_with_api_key(self) -> None:
        pc = Pinecone(api_key="test-key")
        assert pc.config.api_key == "test-key"

    def test_default_host(self) -> None:
        pc = Pinecone(api_key="test-key")
        assert pc.config.host == DEFAULT_BASE_URL

    def test_custom_host(self) -> None:
        pc = Pinecone(api_key="test-key", host="https://custom.example.com")
        assert pc.config.host == "https://custom.example.com"

    def test_custom_timeout(self) -> None:
        pc = Pinecone(api_key="test-key", timeout=60.0)
        assert pc.config.timeout == 60.0

    def test_additional_headers(self) -> None:
        headers = {"X-Custom": "value"}
        pc = Pinecone(api_key="test-key", additional_headers=headers)
        assert pc.config.additional_headers == {"X-Custom": "value"}

    def test_proxy_url(self) -> None:
        pc = Pinecone(api_key="test-key", proxy_url="http://proxy:8080")
        assert pc.config.proxy_url == "http://proxy:8080"

    def test_ssl_ca_certs(self) -> None:
        pc = Pinecone(api_key="test-key", ssl_ca_certs="/path/to/certs.pem")
        assert pc.config.ssl_ca_certs == "/path/to/certs.pem"

    def test_ssl_verify_default_true(self) -> None:
        pc = Pinecone(api_key="test-key")
        assert pc.config.ssl_verify is True

    def test_ssl_verify_false(self) -> None:
        pc = Pinecone(api_key="test-key", ssl_verify=False)
        assert pc.config.ssl_verify is False


class TestEnvVarFallback:
    """Test environment variable fallbacks."""

    def test_api_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PINECONE_API_KEY", "env-api-key")
        pc = Pinecone()
        assert pc.config.api_key == "env-api-key"

    def test_explicit_api_key_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PINECONE_API_KEY", "env-api-key")
        pc = Pinecone(api_key="explicit-key")
        assert pc.config.api_key == "explicit-key"

    def test_host_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PINECONE_CONTROLLER_HOST", "https://custom-host.example.com")
        pc = Pinecone(api_key="test-key")
        assert pc.config.host == "https://custom-host.example.com"

    def test_explicit_host_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PINECONE_CONTROLLER_HOST", "https://env-host.example.com")
        pc = Pinecone(api_key="test-key", host="https://explicit-host.example.com")
        assert pc.config.host == "https://explicit-host.example.com"

    def test_host_env_gets_https_prefix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PINECONE_CONTROLLER_HOST", "custom-host.example.com")
        pc = Pinecone(api_key="test-key")
        assert pc.config.host == "https://custom-host.example.com"

    def test_additional_headers_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PINECONE_ADDITIONAL_HEADERS", '{"X-Env": "from-env"}')
        pc = Pinecone(api_key="test-key")
        assert pc.config.additional_headers == {"X-Env": "from-env"}

    def test_explicit_headers_override_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PINECONE_ADDITIONAL_HEADERS", '{"X-Env": "from-env"}')
        pc = Pinecone(api_key="test-key", additional_headers={"X-Custom": "explicit"})
        assert pc.config.additional_headers == {"X-Custom": "explicit"}


class TestValidation:
    """Test validation errors."""

    def test_no_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("PINECONE_API_KEY", raising=False)
        with pytest.raises(ValidationError, match="No API key provided"):
            Pinecone()

    def test_no_api_key_error_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("PINECONE_API_KEY", raising=False)
        with pytest.raises(ValidationError) as exc_info:
            Pinecone()
        assert "PINECONE_API_KEY" in str(exc_info.value)


class TestHostNormalization:
    """Test host normalization behavior."""

    def test_bare_hostname_gets_https(self) -> None:
        pc = Pinecone(api_key="test-key", host="custom.example.com")
        assert pc.config.host == "https://custom.example.com"

    def test_https_preserved(self) -> None:
        pc = Pinecone(api_key="test-key", host="https://custom.example.com")
        assert pc.config.host == "https://custom.example.com"

    def test_http_preserved(self) -> None:
        pc = Pinecone(api_key="test-key", host="http://localhost:8080")
        assert pc.config.host == "http://localhost:8080"

    def test_idempotent(self) -> None:
        pc = Pinecone(api_key="test-key", host="https://already-prefixed.com")
        assert pc.config.host == "https://already-prefixed.com"


class TestSourceTagNormalization:
    """Test source tag normalization."""

    def test_uppercase_lowered(self) -> None:
        pc = Pinecone(api_key="test-key", source_tag="MyApp")
        assert pc.config.source_tag == "myapp"

    def test_special_chars_stripped(self) -> None:
        pc = Pinecone(api_key="test-key", source_tag="my-app@v2!")
        assert pc.config.source_tag == "myappv2"

    def test_spaces_to_underscores(self) -> None:
        pc = Pinecone(api_key="test-key", source_tag="my app name")
        assert pc.config.source_tag == "my_app_name"

    def test_colons_preserved(self) -> None:
        pc = Pinecone(api_key="test-key", source_tag="app:v2")
        assert pc.config.source_tag == "app:v2"

    def test_underscores_preserved(self) -> None:
        pc = Pinecone(api_key="test-key", source_tag="my_app")
        assert pc.config.source_tag == "my_app"

    def test_combined_normalization(self) -> None:
        pc = Pinecone(api_key="test-key", source_tag="My App@V2! test:tag")
        assert pc.config.source_tag == "my_appv2_test:tag"

    def test_none_source_tag(self) -> None:
        pc = Pinecone(api_key="test-key", source_tag=None)
        assert pc.config.source_tag == ""


class TestUnrecognizedKwargs:
    """Test that unrecognized kwargs raise TypeError."""

    def test_openapi_config_raises(self) -> None:
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            Pinecone(api_key="test-key", openapi_config="something")  # type: ignore[call-arg]

    def test_pool_threads_accepted(self) -> None:
        # pool_threads= is a backcompat shim (BCG-020); it is silently consumed.
        pc = Pinecone(api_key="test-key", pool_threads=4)  # type: ignore[call-arg]
        assert pc is not None

    def test_index_api_raises(self) -> None:
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            Pinecone(api_key="test-key", index_api="something")  # type: ignore[call-arg]

    def test_typo_raises(self) -> None:
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            Pinecone(api_key="test-key", timout=30)  # type: ignore[call-arg]


class TestContextManager:
    """Test context manager protocol."""

    def test_context_manager(self) -> None:
        with Pinecone(api_key="test-key") as pc:
            assert pc.config.api_key == "test-key"

    def test_close(self) -> None:
        pc = Pinecone(api_key="test-key")
        pc.close()


class TestSharedHostCache:
    """Test that Pinecone and Indexes share the same host cache."""

    def test_indexes_shares_host_cache_with_client(self) -> None:
        """Indexes must use the same cache dict as Pinecone."""
        pc = Pinecone(api_key="test-key")
        indexes = pc.indexes
        assert indexes._host_cache is pc._host_cache

    def test_indexes_describe_populates_client_cache(self) -> None:
        """Writing to Indexes._host_cache is visible from Pinecone._host_cache."""
        pc = Pinecone(api_key="test-key")
        indexes = pc.indexes
        # Simulate what describe() does internally
        indexes._host_cache["test-index"] = "test-host.svc.pinecone.io"
        assert pc._host_cache.get("test-index") == "test-host.svc.pinecone.io"


class TestIndexFactory:
    """Test Pinecone.index() factory method."""

    @patch("pinecone.index.Index")
    def test_host_based_skips_describe(self, mock_index_cls: MagicMock) -> None:
        pc = Pinecone(api_key="test-key", additional_headers={"X-Test": "val"}, timeout=45.0)
        mock_idx = MagicMock()
        mock_index_cls.return_value = mock_idx

        result = pc.index(host="foo.svc.pinecone.io")

        assert result is mock_idx
        mock_index_cls.assert_called_once_with(
            host="foo.svc.pinecone.io",
            api_key="test-key",
            additional_headers={"X-Test": "val"},
            timeout=45.0,
            proxy_url="",
            proxy_headers={},
            ssl_ca_certs=None,
            ssl_verify=True,
            source_tag="",
            connection_pool_maxsize=0,
        )

    @patch("pinecone.index.Index")
    def test_name_cached_returns_index_without_describe(self, mock_index_cls: MagicMock) -> None:
        pc = Pinecone(api_key="test-key", timeout=20.0)
        pc._host_cache["my-index"] = "cached.host.pinecone.io"
        mock_idx = MagicMock()
        mock_index_cls.return_value = mock_idx

        result = pc.index(name="my-index")

        assert result is mock_idx
        mock_index_cls.assert_called_once_with(
            host="cached.host.pinecone.io",
            api_key="test-key",
            additional_headers={},
            timeout=20.0,
            proxy_url="",
            proxy_headers={},
            ssl_ca_certs=None,
            ssl_verify=True,
            source_tag="",
            connection_pool_maxsize=0,
        )

    @patch("pinecone.index.Index")
    def test_name_describe_resolves_host_and_caches(self, mock_index_cls: MagicMock) -> None:
        pc = Pinecone(api_key="test-key")
        mock_idx = MagicMock()
        mock_index_cls.return_value = mock_idx

        mock_desc = MagicMock()
        mock_desc.host = "resolved.host.pinecone.io"

        mock_indexes = MagicMock()
        mock_indexes.describe.return_value = mock_desc
        pc._indexes = mock_indexes

        result = pc.index(name="my-index")

        assert result is mock_idx
        mock_indexes.describe.assert_called_once_with("my-index")
        assert pc._host_cache["my-index"] == "resolved.host.pinecone.io"
        mock_index_cls.assert_called_once_with(
            host="resolved.host.pinecone.io",
            api_key="test-key",
            additional_headers={},
            timeout=30.0,
            proxy_url="",
            proxy_headers={},
            ssl_ca_certs=None,
            ssl_verify=True,
            source_tag="",
            connection_pool_maxsize=0,
        )

    def test_no_name_or_host_raises_validation_error(self) -> None:
        pc = Pinecone(api_key="test-key")
        with pytest.raises(ValidationError, match="Either name or host"):
            pc.index()

    @patch("pinecone.index.Index")
    def test_host_based_passes_config_fields(self, mock_index_cls: MagicMock) -> None:
        headers = {"X-Custom": "header-val"}
        pc = Pinecone(api_key="my-api-key", additional_headers=headers, timeout=99.0)
        mock_index_cls.return_value = MagicMock()

        pc.index(host="host.svc.pinecone.io")

        mock_index_cls.assert_called_once_with(
            host="host.svc.pinecone.io",
            api_key="my-api-key",
            additional_headers={"X-Custom": "header-val"},
            timeout=99.0,
            proxy_url="",
            proxy_headers={},
            ssl_ca_certs=None,
            ssl_verify=True,
            source_tag="",
            connection_pool_maxsize=0,
        )

    @patch("pinecone.index.Index")
    def test_name_cached_passes_config_fields(self, mock_index_cls: MagicMock) -> None:
        headers = {"X-Custom": "header-val"}
        pc = Pinecone(api_key="my-api-key", additional_headers=headers, timeout=99.0)
        pc._host_cache["idx"] = "cached.host"
        mock_index_cls.return_value = MagicMock()

        pc.index(name="idx")

        mock_index_cls.assert_called_once_with(
            host="cached.host",
            api_key="my-api-key",
            additional_headers={"X-Custom": "header-val"},
            timeout=99.0,
            proxy_url="",
            proxy_headers={},
            ssl_ca_certs=None,
            ssl_verify=True,
            source_tag="",
            connection_pool_maxsize=0,
        )

    @patch("pinecone._internal.http_client.HTTPClient")
    @patch("pinecone.index.Index")
    def test_index_propagates_proxy_ssl_source_tag(
        self, mock_index_cls: MagicMock, _mock_http: MagicMock
    ) -> None:
        pc = Pinecone(
            api_key="test-key",
            proxy_url="http://proxy:8080",
            proxy_headers={"Proxy-Authorization": "Basic abc"},
            ssl_ca_certs="/path/to/certs.pem",
            ssl_verify=False,
            source_tag="my_app",
            connection_pool_maxsize=10,
        )
        mock_index_cls.return_value = MagicMock()

        pc.index(host="foo.svc.pinecone.io")

        mock_index_cls.assert_called_once_with(
            host="foo.svc.pinecone.io",
            api_key="test-key",
            additional_headers={},
            timeout=30.0,
            proxy_url="http://proxy:8080",
            proxy_headers={"Proxy-Authorization": "Basic abc"},
            ssl_ca_certs="/path/to/certs.pem",
            ssl_verify=False,
            source_tag="my_app",
            connection_pool_maxsize=10,
        )

    @patch("pinecone._internal.http_client.HTTPClient")
    @patch("pinecone.index.Index")
    def test_index_by_name_propagates_proxy_ssl_source_tag(
        self, mock_index_cls: MagicMock, _mock_http: MagicMock
    ) -> None:
        pc = Pinecone(
            api_key="test-key",
            proxy_url="http://proxy:8080",
            ssl_ca_certs="/path/to/certs.pem",
            ssl_verify=False,
            source_tag="my_app",
            connection_pool_maxsize=10,
        )
        pc._host_cache["my-index"] = "cached.host.pinecone.io"
        mock_index_cls.return_value = MagicMock()

        pc.index(name="my-index")

        mock_index_cls.assert_called_once_with(
            host="cached.host.pinecone.io",
            api_key="test-key",
            additional_headers={},
            timeout=30.0,
            proxy_url="http://proxy:8080",
            proxy_headers={},
            ssl_ca_certs="/path/to/certs.pem",
            ssl_verify=False,
            source_tag="my_app",
            connection_pool_maxsize=10,
        )


class TestIndexFactoryGrpc:
    """Test Pinecone.index(grpc=True) factory method."""

    @patch("pinecone.grpc.GrpcIndex")
    def test_grpc_true_returns_grpc_index(self, mock_grpc_cls: MagicMock) -> None:
        pc = Pinecone(api_key="test-key")
        mock_grpc_idx = MagicMock()
        mock_grpc_cls.return_value = mock_grpc_idx

        result = pc.index(host="foo.svc.pinecone.io", grpc=True)

        assert result is mock_grpc_idx
        mock_grpc_cls.assert_called_once()
        _, call_kwargs = mock_grpc_cls.call_args
        assert call_kwargs["host"] == "foo.svc.pinecone.io"
        assert call_kwargs["api_key"] == "test-key"
        assert call_kwargs["source_tag"] is None
        # on_throttle must be wired to the registry's report_throttled method
        on_throttle = call_kwargs["on_throttle"]
        assert callable(on_throttle)
        assert on_throttle.__self__ is pc._limiter_registry

    @patch("pinecone.grpc.GrpcIndex")
    def test_grpc_true_passes_source_tag(self, mock_grpc_cls: MagicMock) -> None:
        pc = Pinecone(api_key="test-key", source_tag="my_app")
        mock_grpc_cls.return_value = MagicMock()

        pc.index(host="foo.svc.pinecone.io", grpc=True)

        mock_grpc_cls.assert_called_once()
        _, call_kwargs = mock_grpc_cls.call_args
        assert call_kwargs["host"] == "foo.svc.pinecone.io"
        assert call_kwargs["api_key"] == "test-key"
        assert call_kwargs["source_tag"] == "my_app"
        assert callable(call_kwargs["on_throttle"])

    @patch("pinecone.grpc.GrpcIndex")
    def test_grpc_true_with_name_cached(self, mock_grpc_cls: MagicMock) -> None:
        pc = Pinecone(api_key="test-key")
        pc._host_cache["my-index"] = "cached.host.pinecone.io"
        mock_grpc_cls.return_value = MagicMock()

        pc.index(name="my-index", grpc=True)

        mock_grpc_cls.assert_called_once()
        _, call_kwargs = mock_grpc_cls.call_args
        assert call_kwargs["host"] == "cached.host.pinecone.io"
        assert call_kwargs["api_key"] == "test-key"
        assert call_kwargs["source_tag"] is None
        assert callable(call_kwargs["on_throttle"])

    @patch("pinecone.grpc.GrpcIndex")
    def test_grpc_true_with_name_describe(self, mock_grpc_cls: MagicMock) -> None:
        pc = Pinecone(api_key="test-key")
        mock_grpc_cls.return_value = MagicMock()

        mock_desc = MagicMock()
        mock_desc.host = "resolved.host.pinecone.io"
        mock_indexes = MagicMock()
        mock_indexes.describe.return_value = mock_desc
        pc._indexes = mock_indexes

        pc.index(name="my-index", grpc=True)

        mock_indexes.describe.assert_called_once_with("my-index")
        assert pc._host_cache["my-index"] == "resolved.host.pinecone.io"
        mock_grpc_cls.assert_called_once()
        _, call_kwargs = mock_grpc_cls.call_args
        assert call_kwargs["host"] == "resolved.host.pinecone.io"
        assert call_kwargs["api_key"] == "test-key"
        assert call_kwargs["source_tag"] is None
        assert callable(call_kwargs["on_throttle"])

    @patch("pinecone.index.Index")
    def test_grpc_false_default_returns_http_index(self, mock_index_cls: MagicMock) -> None:
        pc = Pinecone(api_key="test-key")
        mock_idx = MagicMock()
        mock_index_cls.return_value = mock_idx

        result = pc.index(host="foo.svc.pinecone.io")

        assert result is mock_idx

    def test_grpc_true_no_name_or_host_raises(self) -> None:
        pc = Pinecone(api_key="test-key")
        with pytest.raises(ValidationError, match="Either name or host"):
            pc.index(grpc=True)


class TestClientLifecycle:
    """Test that close() properly cleans up all HTTP clients."""

    def test_close_without_inference_access(self) -> None:
        """close() works when inference was never accessed."""
        pc = Pinecone(api_key="test-key")
        assert pc._inference is None
        pc.close()

    def test_close_closes_inference_http_client(self) -> None:
        """close() closes the Inference namespace's HTTP client."""
        pc = Pinecone(api_key="test-key")
        # Trigger lazy creation of the inference namespace
        _ = pc.inference
        assert pc._inference is not None
        assert not pc._inference._http._client.is_closed
        pc.close()
        assert pc._inference._http._client.is_closed

    def test_context_manager_closes_inference(self) -> None:
        """Exiting a context manager closes the Inference HTTP client."""
        with Pinecone(api_key="test-key") as pc:
            _ = pc.inference
            assert not pc._inference._http._client.is_closed  # type: ignore[union-attr]
        assert pc._inference._http._client.is_closed  # type: ignore[union-attr]


class TestPineconeLimiterWiring:
    """Test that _AdaptiveLimiterRegistry is wired into the RetryConfig on_throttle callback."""

    def test_pinecone_has_limiter_registry(self) -> None:
        pc = Pinecone(api_key="test-key")
        assert pc._limiter_registry is not None
        assert isinstance(pc._limiter_registry, _AdaptiveLimiterRegistry)
        pc._limiter_registry.report_throttled("nonexistent-host")  # no-op, no error

    def test_pinecone_retry_config_has_on_throttle_wired(self) -> None:
        pc = Pinecone(api_key="test-key")
        assert pc._config.retry_config.on_throttle is not None
        pc._config.retry_config.on_throttle("test-host")

    def test_on_throttle_forwards_to_registry(self) -> None:
        pc = Pinecone(api_key="test-key")
        registry = pc._limiter_registry
        # Seed a limiter so report_throttled has something to decrement
        limiter = registry.get("test-host", 8)
        assert limiter.current_limit() == 8
        # Calling on_throttle with the host should halve the limiter's limit
        pc._config.retry_config.on_throttle("test-host")  # type: ignore[misc]
        assert limiter.current_limit() == 4

    def test_user_retry_config_is_preserved(self) -> None:
        user_cfg = RetryConfig(max_retries=10, backoff_factor=3.0)
        pc = Pinecone(api_key="test-key", retry_config=user_cfg)
        applied = pc._config.retry_config
        assert applied.max_retries == 10
        assert applied.backoff_factor == 3.0
        assert applied.on_throttle is not None  # injected by client

    def test_each_client_gets_independent_registry(self) -> None:
        pc1 = Pinecone(api_key="test-key")
        pc2 = Pinecone(api_key="test-key")
        assert pc1._limiter_registry is not pc2._limiter_registry
