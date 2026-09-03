"""Tests for TCP socket options: keep-alive and Nagle's algorithm."""

from __future__ import annotations

import socket
from unittest.mock import patch

import pytest

from pinecone._internal.http_client import _build_socket_options

# ``socket.TCP_KEEPIDLE`` is Linux-only in CPython; macOS exposes ``TCP_KEEPALIVE``
# instead. Patching ``sys.platform`` is therefore not enough to exercise the Linux
# branch of ``_build_socket_options`` from a macOS host — the constant simply is not
# there. Tests that need it are skipped where it is missing; CI runs on Linux, so
# they always execute there with their assertions intact.
_HAS_TCP_KEEPIDLE = hasattr(socket, "TCP_KEEPIDLE")


class TestBuildSocketOptions:
    def test_keepalive_enabled(self) -> None:
        opts = _build_socket_options()
        assert (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1) in opts

    def test_nagle_disabled(self) -> None:
        opts = _build_socket_options()
        assert (socket.IPPROTO_TCP, socket.TCP_NODELAY, 1) in opts

    @pytest.mark.skipif(
        not _HAS_TCP_KEEPIDLE, reason="socket.TCP_KEEPIDLE is Linux-only in CPython"
    )
    def test_linux_keepalive_params(self) -> None:
        with patch("pinecone._internal.http_client.sys") as mock_sys:
            mock_sys.platform = "linux"
            opts = _build_socket_options()
        assert (socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 300) in opts
        assert (socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 60) in opts
        assert (socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 4) in opts

    def test_darwin_keepalive_params(self) -> None:
        with patch("pinecone._internal.http_client.sys") as mock_sys:
            mock_sys.platform = "darwin"
            opts = _build_socket_options()
        # Exact list asserts "no idle, no count" without naming Linux-only TCP_KEEPIDLE.
        assert opts == [
            (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),
            (socket.IPPROTO_TCP, socket.TCP_NODELAY, 1),
            (socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 60),
        ]

    def test_windows_minimal_options(self) -> None:
        with patch("pinecone._internal.http_client.sys") as mock_sys:
            mock_sys.platform = "win32"
            opts = _build_socket_options()
        # Only keepalive enable and nodelay
        assert len(opts) == 2
        assert (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1) in opts
        assert (socket.IPPROTO_TCP, socket.TCP_NODELAY, 1) in opts
