"""Unit tests for AsyncPinecone client."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pinecone._internal.adaptive import _AdaptiveLimiterRegistry
from pinecone._internal.config import RetryConfig
from pinecone.async_client.async_index import AsyncIndex
from pinecone.async_client.pinecone import AsyncPinecone
from pinecone.errors.exceptions import ValidationError


def test_async_pinecone_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    with pytest.raises(ValidationError, match="No API key"):
        AsyncPinecone()


def test_async_pinecone_accepts_api_key() -> None:
    pc = AsyncPinecone(api_key="test-key")
    assert pc.config.api_key == "test-key"


def test_async_pinecone_unrecognized_kwargs() -> None:
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        AsyncPinecone(api_key="test-key", openapi_config={})  # type: ignore[call-arg]


def test_async_pinecone_default_host() -> None:
    pc = AsyncPinecone(api_key="test-key")
    assert pc.config.host == "https://api.pinecone.io"


def test_async_pinecone_custom_host() -> None:
    pc = AsyncPinecone(api_key="test-key", host="https://custom.pinecone.io")
    assert pc.config.host == "https://custom.pinecone.io"


def test_async_pinecone_indexes_property() -> None:
    pc = AsyncPinecone(api_key="test-key")
    from pinecone.async_client.indexes import AsyncIndexes

    indexes = pc.indexes
    assert isinstance(indexes, AsyncIndexes)
    # Verify lazy caching — same instance returned
    assert pc.indexes is indexes


async def test_async_pinecone_context_manager() -> None:
    async with AsyncPinecone(api_key="test-key") as pc:
        assert pc.config.api_key == "test-key"


async def test_async_pinecone_close() -> None:
    pc = AsyncPinecone(api_key="test-key")
    await pc.close()


class TestEnvVarFallback:
    """Test environment variable fallbacks for AsyncPinecone."""

    def test_api_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PINECONE_API_KEY", "env-api-key")
        pc = AsyncPinecone()
        assert pc.config.api_key == "env-api-key"

    def test_explicit_api_key_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PINECONE_API_KEY", "env-api-key")
        pc = AsyncPinecone(api_key="explicit-key")
        assert pc.config.api_key == "explicit-key"

    def test_host_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PINECONE_CONTROLLER_HOST", "https://custom-host.example.com")
        pc = AsyncPinecone(api_key="test-key")
        assert pc.config.host == "https://custom-host.example.com"

    def test_host_env_gets_https_prefix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PINECONE_CONTROLLER_HOST", "custom-host.example.com")
        pc = AsyncPinecone(api_key="test-key")
        assert pc.config.host == "https://custom-host.example.com"

    def test_additional_headers_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PINECONE_ADDITIONAL_HEADERS", '{"X-Env": "from-env"}')
        pc = AsyncPinecone(api_key="test-key")
        assert pc.config.additional_headers == {"X-Env": "from-env"}

    def test_explicit_headers_override_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PINECONE_ADDITIONAL_HEADERS", '{"X-Env": "from-env"}')
        pc = AsyncPinecone(api_key="test-key", additional_headers={"X-Custom": "explicit"})
        assert pc.config.additional_headers == {"X-Custom": "explicit"}

    def test_source_tag_normalization(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PINECONE_API_KEY", "env-key")
        pc = AsyncPinecone(source_tag="My App@V2! test:tag")
        assert pc.config.source_tag == "my_appv2_test:tag"

    def test_proxy_url_param(self) -> None:
        pc = AsyncPinecone(api_key="test-key", proxy_url="http://proxy:8080")
        assert pc.config.proxy_url == "http://proxy:8080"

    def test_ssl_ca_certs_param(self) -> None:
        pc = AsyncPinecone(api_key="test-key", ssl_ca_certs="/path/to/certs.pem")
        assert pc.config.ssl_ca_certs == "/path/to/certs.pem"

    def test_ssl_verify_param(self) -> None:
        pc = AsyncPinecone(api_key="test-key", ssl_verify=False)
        assert pc.config.ssl_verify is False


class TestSharedHostCache:
    """Test that AsyncPinecone and AsyncIndexes share the same host cache."""

    def test_indexes_shares_host_cache_with_client(self) -> None:
        """AsyncIndexes must use the same cache dict as AsyncPinecone."""
        pc = AsyncPinecone(api_key="test-key")
        indexes = pc.indexes
        assert indexes._host_cache is pc._host_cache

    def test_indexes_describe_populates_client_cache(self) -> None:
        """Writing to AsyncIndexes._host_cache is visible from AsyncPinecone._host_cache."""
        pc = AsyncPinecone(api_key="test-key")
        indexes = pc.indexes
        # Simulate what describe() does internally
        indexes._host_cache["test-index"] = "test-host.svc.pinecone.io"
        assert pc._host_cache.get("test-index") == "test-host.svc.pinecone.io"


class TestAsyncIndexFactory:
    """Test AsyncPinecone.index() propagates config to AsyncIndex."""

    @patch("pinecone._internal.http_client.AsyncHTTPClient")
    @patch("pinecone.async_client.async_index.AsyncIndex")
    async def test_index_by_host_propagates_config(
        self, mock_cls: MagicMock, _mock_http: MagicMock
    ) -> None:
        pc = AsyncPinecone(
            api_key="test-key",
            proxy_url="http://proxy:8080",
            ssl_ca_certs="/path/to/certs.pem",
            ssl_verify=False,
            source_tag="my_app",
            connection_pool_maxsize=10,
        )
        mock_cls.return_value = MagicMock()

        await pc.index(host="foo.svc.pinecone.io")

        mock_cls.assert_called_once_with(
            host="foo.svc.pinecone.io",
            api_key="test-key",
            additional_headers={},
            timeout=30.0,
            proxy_url="http://proxy:8080",
            proxy_headers={},
            ssl_ca_certs="/path/to/certs.pem",
            ssl_verify=False,
            source_tag="my_app",
            connection_pool_maxsize=10,
            _limiter_registry=pc._limiter_registry,
        )

    @patch("pinecone._internal.http_client.AsyncHTTPClient")
    @patch("pinecone.async_client.async_index.AsyncIndex")
    async def test_index_by_name_cached_propagates_config(
        self, mock_cls: MagicMock, _mock_http: MagicMock
    ) -> None:
        pc = AsyncPinecone(
            api_key="test-key",
            proxy_url="http://proxy:8080",
            ssl_ca_certs="/path/to/certs.pem",
            ssl_verify=False,
            source_tag="my_app",
            connection_pool_maxsize=10,
        )
        pc._host_cache["my-index"] = "cached.host.pinecone.io"
        mock_cls.return_value = MagicMock()

        await pc.index(name="my-index")

        mock_cls.assert_called_once_with(
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
            _limiter_registry=pc._limiter_registry,
        )

    @patch("pinecone.async_client.async_index.AsyncIndex")
    async def test_index_defaults_propagated(self, mock_cls: MagicMock) -> None:
        pc = AsyncPinecone(api_key="test-key")
        mock_cls.return_value = MagicMock()

        await pc.index(host="foo.svc.pinecone.io")

        mock_cls.assert_called_once_with(
            host="foo.svc.pinecone.io",
            api_key="test-key",
            additional_headers={},
            timeout=30.0,
            proxy_url="",
            proxy_headers={},
            ssl_ca_certs=None,
            ssl_verify=True,
            source_tag="",
            connection_pool_maxsize=0,
            _limiter_registry=pc._limiter_registry,
        )


class TestAsyncIndexAutoResolve:
    """Test that AsyncPinecone.index() auto-resolves hosts via describe."""

    async def test_index_by_host_skips_describe(self) -> None:
        """Providing host= returns AsyncIndex without calling describe."""
        pc = AsyncPinecone(api_key="test-key")
        pc.indexes.describe = AsyncMock()  # type: ignore[method-assign]

        result = await pc.index(host="abc.svc.pinecone.io")

        assert isinstance(result, AsyncIndex)
        pc.indexes.describe.assert_not_called()
        await result.close()

    async def test_index_by_name_resolves_via_describe(self) -> None:
        """Providing name= on cache miss triggers describe and caches the host."""
        pc = AsyncPinecone(api_key="test-key")
        index_model = MagicMock()
        index_model.host = "resolved.svc.pinecone.io"
        pc.indexes.describe = AsyncMock(return_value=index_model)  # type: ignore[method-assign]

        result = await pc.index(name="my-index")

        pc.indexes.describe.assert_awaited_once_with("my-index")
        assert isinstance(result, AsyncIndex)
        assert pc._host_cache["my-index"] == "resolved.svc.pinecone.io"
        await result.close()

    async def test_index_by_name_uses_cache(self) -> None:
        """Providing name= on cache hit does not call describe."""
        pc = AsyncPinecone(api_key="test-key")
        pc._host_cache["cached-idx"] = "cached.svc.pinecone.io"
        pc.indexes.describe = AsyncMock()  # type: ignore[method-assign]

        result = await pc.index(name="cached-idx")

        pc.indexes.describe.assert_not_called()
        assert isinstance(result, AsyncIndex)
        await result.close()

    async def test_index_no_args_raises_validation_error(self) -> None:
        """Calling index() with neither name nor host raises ValidationError."""
        pc = AsyncPinecone(api_key="test-key")

        with pytest.raises(ValidationError, match="Either name or host must be provided"):
            await pc.index()


class TestAsyncClientLifecycle:
    """Test that close() properly cleans up all HTTP clients."""

    async def test_close_without_inference_access(self) -> None:
        """close() works when inference was never accessed."""
        pc = AsyncPinecone(api_key="test-key")
        assert pc._inference is None
        await pc.close()

    async def test_close_closes_inference_http_client(self) -> None:
        """close() closes the AsyncInference namespace's HTTP client."""
        pc = AsyncPinecone(api_key="test-key")
        # Trigger lazy creation of the inference namespace
        inference = pc.inference
        assert pc._inference is not None
        # Mock the close method to verify it gets called
        inference.close = AsyncMock()  # type: ignore[method-assign]
        await pc.close()
        inference.close.assert_awaited_once()

    async def test_context_manager_closes_inference(self) -> None:
        """Exiting an async context manager closes the Inference HTTP client."""
        async with AsyncPinecone(api_key="test-key") as pc:
            inference = pc.inference
            inference.close = AsyncMock()  # type: ignore[method-assign]
        inference.close.assert_awaited_once()


class TestAsyncPineconeLimiterWiring:
    """Test that _AdaptiveLimiterRegistry is wired into RetryConfig on_throttle for AsyncPinecone."""

    def test_async_pinecone_has_limiter_registry(self) -> None:
        pc = AsyncPinecone(api_key="test-key")
        assert pc._limiter_registry is not None
        assert isinstance(pc._limiter_registry, _AdaptiveLimiterRegistry)
        pc._limiter_registry.report_throttled("nonexistent-host")  # no-op, no error

    def test_async_pinecone_retry_config_has_on_throttle_wired(self) -> None:
        pc = AsyncPinecone(api_key="test-key")
        assert pc._config.retry_config.on_throttle is not None
        pc._config.retry_config.on_throttle("test-host")

    def test_on_throttle_forwards_to_registry(self) -> None:
        pc = AsyncPinecone(api_key="test-key")
        registry = pc._limiter_registry
        limiter = registry.get("test-host", 8)
        assert limiter.current_limit() == 8
        pc._config.retry_config.on_throttle("test-host")  # type: ignore[misc]
        assert limiter.current_limit() == 4

    def test_user_retry_config_is_preserved(self) -> None:
        user_cfg = RetryConfig(max_retries=10, backoff_factor=3.0)
        pc = AsyncPinecone(api_key="test-key", retry_config=user_cfg)
        applied = pc._config.retry_config
        assert applied.max_retries == 10
        assert applied.backoff_factor == 3.0
        assert applied.on_throttle is not None  # injected by client

    def test_each_async_client_gets_independent_registry(self) -> None:
        pc1 = AsyncPinecone(api_key="test-key")
        pc2 = AsyncPinecone(api_key="test-key")
        assert pc1._limiter_registry is not pc2._limiter_registry
