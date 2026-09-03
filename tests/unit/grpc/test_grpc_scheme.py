"""Unit tests for the gRPC endpoint scheme configuration.

The scheme, not ``secure``, decides whether the wire carries TLS: tonic runs a
handshake only for an ``https`` endpoint. These tests pin the three things that
follow from that — an explicitly configured scheme reaches the channel, the
default is unchanged for callers who configure nothing, and the one pairing the
transport cannot dial is refused up front.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from pinecone import Pinecone
from pinecone._internal.config import PineconeConfig, resolve_grpc_scheme
from pinecone.errors.exceptions import PineconeValueError
from pinecone.grpc import GrpcIndex, _build_grpc_endpoint

_MOCK_GRPC_MODULE_PATH = "pinecone._grpc"

PLAINTEXT_HOST = "http://127.0.0.1:5085"


@pytest.fixture(autouse=True)
def _no_scheme_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PINECONE_GRPC_SCHEME", raising=False)


@pytest.fixture
def channel_module() -> Iterator[MagicMock]:
    """Patch in a fake ``pinecone._grpc``, since GrpcChannel is imported lazily."""
    module = MagicMock()
    module.GrpcChannel.return_value = MagicMock()
    with patch.dict("sys.modules", {_MOCK_GRPC_MODULE_PATH: module}):
        yield module


def _channel_args(module: MagicMock) -> tuple[Any, ...]:
    return tuple(module.GrpcChannel.call_args[0])


class TestEndpointBuilder:
    def test_explicit_http_scheme_survives(self) -> None:
        assert _build_grpc_endpoint(PLAINTEXT_HOST, secure=True, scheme="http") == PLAINTEXT_HOST

    def test_explicit_https_scheme_replaces_a_plaintext_host(self) -> None:
        assert (
            _build_grpc_endpoint(PLAINTEXT_HOST, secure=True, scheme="https")
            == "https://127.0.0.1:5085"
        )

    @pytest.mark.filterwarnings("ignore:The gRPC data plane is configured:RuntimeWarning")
    @pytest.mark.parametrize(
        ("secure", "expected"),
        [(True, "https://idx.svc.pinecone.io"), (False, "http://idx.svc.pinecone.io")],
    )
    def test_unset_scheme_follows_secure(self, secure: bool, expected: str) -> None:
        assert _build_grpc_endpoint("idx.svc.pinecone.io", secure=secure, scheme=None) == expected

    def test_https_without_tls_material_is_refused(self) -> None:
        with pytest.raises(PineconeValueError, match="requires secure=True"):
            _build_grpc_endpoint(PLAINTEXT_HOST, secure=False, scheme="https")

    def test_unknown_scheme_is_refused(self) -> None:
        with pytest.raises(PineconeValueError, match="Invalid gRPC scheme"):
            _build_grpc_endpoint(PLAINTEXT_HOST, secure=True, scheme="grpc")


class TestGrpcIndexScheme:
    def test_plaintext_data_plane_is_dialled_over_http(self, channel_module: MagicMock) -> None:
        """The regression: a plaintext data plane must not be dialled over https.

        Without an explicit scheme the endpoint came from ``secure`` alone, so
        the only way to reach a plaintext data plane was to drop TLS material
        the sibling REST client also needs.
        """
        GrpcIndex(host=PLAINTEXT_HOST, api_key="test-key", grpc_scheme="http")

        args = _channel_args(channel_module)
        assert args[0] == PLAINTEXT_HOST
        assert args[4] is True

    def test_scheme_defaults_to_https_for_an_unconfigured_caller(
        self, channel_module: MagicMock
    ) -> None:
        GrpcIndex(host=PLAINTEXT_HOST, api_key="test-key")

        assert _channel_args(channel_module)[0] == "https://127.0.0.1:5085"

    def test_secure_false_still_yields_http(self, channel_module: MagicMock) -> None:
        GrpcIndex(host="idx.svc.pinecone.io", api_key="test-key", secure=False)

        args = _channel_args(channel_module)
        assert args[0] == "http://idx.svc.pinecone.io"
        assert args[4] is False

    def test_env_var_supplies_the_scheme(
        self, channel_module: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PINECONE_GRPC_SCHEME", "http")

        GrpcIndex(host=PLAINTEXT_HOST, api_key="test-key")

        assert _channel_args(channel_module)[0] == PLAINTEXT_HOST

    def test_explicit_scheme_beats_the_env_var(
        self, channel_module: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PINECONE_GRPC_SCHEME", "http")

        GrpcIndex(host=PLAINTEXT_HOST, api_key="test-key", grpc_scheme="https")

        assert _channel_args(channel_module)[0] == "https://127.0.0.1:5085"

    def test_https_with_secure_false_is_refused(self, channel_module: MagicMock) -> None:
        with pytest.raises(PineconeValueError, match="requires secure=True"):
            GrpcIndex(host=PLAINTEXT_HOST, api_key="test-key", secure=False, grpc_scheme="https")


class TestClientForwardsTheScheme:
    def test_client_scheme_reaches_the_channel(self, channel_module: MagicMock) -> None:
        pc = Pinecone(api_key="test-key", grpc_scheme="http")

        with patch.object(pc, "_resolve_index_host", return_value=PLAINTEXT_HOST):
            pc.index(name="plaintext-index", grpc=True)

        assert _channel_args(channel_module)[0] == PLAINTEXT_HOST

    def test_client_default_is_https(self, channel_module: MagicMock) -> None:
        pc = Pinecone(api_key="test-key")

        with patch.object(pc, "_resolve_index_host", return_value=PLAINTEXT_HOST):
            pc.index(name="plaintext-index", grpc=True)

        assert _channel_args(channel_module)[0] == "https://127.0.0.1:5085"

    def test_client_reads_the_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PINECONE_GRPC_SCHEME", "http")

        assert Pinecone(api_key="test-key")._config.grpc_scheme == "http"

    def test_invalid_env_var_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PINECONE_GRPC_SCHEME", "gopher")

        with pytest.raises(PineconeValueError, match="Invalid gRPC scheme"):
            Pinecone(api_key="test-key")


class TestResolveGrpcScheme:
    def test_unset_resolves_to_none(self) -> None:
        assert resolve_grpc_scheme(None) is None

    def test_blank_env_var_resolves_to_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PINECONE_GRPC_SCHEME", "  ")

        assert resolve_grpc_scheme(None) is None

    def test_config_normalizes_on_construction(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PINECONE_GRPC_SCHEME", "https")

        assert PineconeConfig(api_key="k").grpc_scheme == "https"
