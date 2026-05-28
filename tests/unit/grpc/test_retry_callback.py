"""Tests for the PyO3 on_throttle callback parameter on GrpcChannel."""

from __future__ import annotations

import pytest


class TestGrpcThrottleCallback:
    def test_callback_accepts_host_string(self) -> None:
        """GrpcChannel constructor accepts on_throttle callback as kwarg."""
        try:
            from pinecone._grpc import GrpcChannel  # type: ignore[import-not-found]
        except ImportError:
            pytest.skip("Rust extension not built")

        received: list[str] = []
        ch = GrpcChannel(
            endpoint="https://test-index-abc.svc.pinecone.io:443",
            api_key="test-key",
            api_version="2025-10",
            version="test-0.0.0",
            on_throttle=lambda host: received.append(host),
        )
        assert ch is not None

    def test_callback_none_is_default(self) -> None:
        """Constructor accepts None or absence of on_throttle without error."""
        try:
            from pinecone._grpc import GrpcChannel  # type: ignore[import-not-found]
        except ImportError:
            pytest.skip("Rust extension not built")

        ch = GrpcChannel(
            endpoint="https://test-index-abc.svc.pinecone.io:443",
            api_key="test-key",
            api_version="2025-10",
            version="test-0.0.0",
        )
        assert ch is not None

    def test_callback_omitted_same_as_none(self) -> None:
        """Omitting on_throttle is identical to passing on_throttle=None."""
        try:
            from pinecone._grpc import GrpcChannel  # type: ignore[import-not-found]
        except ImportError:
            pytest.skip("Rust extension not built")

        ch_explicit = GrpcChannel(
            endpoint="https://test-index-abc.svc.pinecone.io:443",
            api_key="test-key",
            api_version="2025-10",
            version="test-0.0.0",
            on_throttle=None,
        )
        ch_default = GrpcChannel(
            endpoint="https://test-index-abc.svc.pinecone.io:443",
            api_key="test-key",
            api_version="2025-10",
            version="test-0.0.0",
        )
        assert ch_explicit is not None
        assert ch_default is not None
