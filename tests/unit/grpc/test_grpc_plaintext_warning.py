"""The one-time warning that a gRPC channel is dialling a public host in cleartext.

``PINECONE_GRPC_SCHEME=http`` outranks ``secure=True`` on purpose — explicit
configuration wins in this SDK — so the only signal a caller gets that the API
key is crossing a public network unencrypted is this warning. These tests pin
that it fires where it matters, stays quiet on the networks a plaintext data
plane legitimately lives on, and never becomes per-call noise.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import pinecone.grpc as grpc_module
from pinecone.grpc import GrpcIndex, _build_grpc_endpoint

_MOCK_GRPC_MODULE_PATH = "pinecone._grpc"

PUBLIC_HOST = "idx-abc123.svc.pinecone.io"


@pytest.fixture(autouse=True)
def _reset_warning_state(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv("PINECONE_GRPC_SCHEME", raising=False)
    grpc_module._warned_about_plaintext_grpc = False
    yield
    grpc_module._warned_about_plaintext_grpc = False


@pytest.fixture
def channel_module() -> Iterator[MagicMock]:
    module = MagicMock()
    module.GrpcChannel.return_value = MagicMock()
    with patch.dict("sys.modules", {_MOCK_GRPC_MODULE_PATH: module}):
        yield module


def _recorded(host: str, **kwargs: Any) -> list[warnings.WarningMessage]:
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        _build_grpc_endpoint(host, secure=kwargs.pop("secure", True), **kwargs)
    return list(record)


class TestPublicHost:
    def test_http_against_a_public_host_warns(self) -> None:
        with pytest.warns(RuntimeWarning, match="unencrypted") as record:
            _build_grpc_endpoint(PUBLIC_HOST, secure=True, scheme="http")

        message = str(record[0].message)
        assert PUBLIC_HOST in message
        assert 'grpc_scheme="https"' in message
        assert "PINECONE_GRPC_SCHEME" in message

    def test_the_port_is_not_mistaken_for_part_of_the_host(self) -> None:
        with pytest.warns(RuntimeWarning) as record:
            _build_grpc_endpoint(f"{PUBLIC_HOST}:50051", secure=True, scheme="http")

        assert f"dial {PUBLIC_HOST} over http" in str(record[0].message)

    def test_secure_false_with_no_scheme_also_warns(self) -> None:
        assert len(_recorded(PUBLIC_HOST, secure=False, scheme=None)) == 1

    def test_a_public_ip_warns(self) -> None:
        assert len(_recorded("203.0.113.7:50051", scheme="http")) == 1

    def test_an_unresolvable_hostname_is_treated_as_public(self) -> None:
        assert len(_recorded("grpc-gateway.internal.example:50051", scheme="http")) == 1


class TestHttps:
    def test_https_does_not_warn(self) -> None:
        assert _recorded(PUBLIC_HOST, secure=True, scheme="https") == []

    def test_the_default_scheme_does_not_warn(self) -> None:
        assert _recorded(PUBLIC_HOST, secure=True, scheme=None) == []


class TestLocalAndPrivateHosts:
    @pytest.mark.parametrize(
        "host",
        [
            "localhost",
            "localhost:5085",
            "127.0.0.1",
            "127.0.0.1:5085",
            "127.0.0.53",
            "[::1]:5085",
            "::1",
            "10.0.0.7:50051",
            "172.16.4.9",
            "172.31.255.255:50051",
            "192.168.1.5",
            "192.168.1.5:50051",
        ],
    )
    def test_plaintext_to_a_local_or_private_host_is_silent(self, host: str) -> None:
        assert _recorded(host, scheme="http") == []

    def test_a_host_just_outside_rfc_1918_still_warns(self) -> None:
        assert len(_recorded("172.32.0.1:50051", scheme="http")) == 1


class TestFiresOnce:
    def test_a_second_plaintext_endpoint_is_silent(self) -> None:
        assert len(_recorded(PUBLIC_HOST, scheme="http")) == 1
        assert _recorded("other-index.svc.pinecone.io", scheme="http") == []

    def test_a_local_endpoint_does_not_consume_the_one_warning(self) -> None:
        assert _recorded("127.0.0.1:5085", scheme="http") == []
        assert len(_recorded(PUBLIC_HOST, scheme="http")) == 1


class TestThroughGrpcIndex:
    def test_the_env_var_alone_is_enough_to_warn(
        self, channel_module: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PINECONE_GRPC_SCHEME", "http")

        with pytest.warns(RuntimeWarning, match="unencrypted"):
            GrpcIndex(host=PUBLIC_HOST, api_key="test-key")

        assert channel_module.GrpcChannel.call_args[0][0] == f"http://{PUBLIC_HOST}"

    def test_the_warning_points_at_the_callers_own_line(
        self, channel_module: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PINECONE_GRPC_SCHEME", "http")

        with pytest.warns(RuntimeWarning) as record:
            GrpcIndex(host=PUBLIC_HOST, api_key="test-key")

        assert record[0].filename == __file__

    def test_a_local_simulator_stays_silent(self, channel_module: MagicMock) -> None:
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            GrpcIndex(host="http://127.0.0.1:5085", api_key="test-key", grpc_scheme="http")

        assert list(record) == []

    def test_the_https_default_stays_silent(self, channel_module: MagicMock) -> None:
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            GrpcIndex(host=PUBLIC_HOST, api_key="test-key")

        assert list(record) == []
