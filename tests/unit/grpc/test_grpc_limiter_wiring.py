"""Tests that GrpcIndex wires the throttle callback to the _AdaptiveLimiterRegistry."""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import MagicMock, patch

_MOCK_GRPC_MODULE_PATH = "pinecone._grpc"


def _make_grpc_index_with_mock(
    on_throttle: Callable[[str], None] | None = None,
) -> tuple[MagicMock, MagicMock]:
    """Return (mock_channel_class, grpc_index_instance) for inspection."""
    from pinecone.grpc import GrpcIndex

    mock_channel = MagicMock()
    mock_module = MagicMock()
    mock_module.GrpcChannel.return_value = mock_channel
    with patch.dict("sys.modules", {_MOCK_GRPC_MODULE_PATH: mock_module}):
        idx = GrpcIndex(
            host="https://test-idx-abc.svc.pinecone.io",
            api_key="test-key",
            on_throttle=on_throttle,
        )
    return mock_module.GrpcChannel, idx


def test_grpc_index_construction_passes_on_throttle_none_by_default() -> None:
    mock_channel_cls, _ = _make_grpc_index_with_mock(on_throttle=None)
    _, kwargs = mock_channel_cls.call_args
    assert kwargs.get("on_throttle") is None


def test_grpc_index_construction_passes_on_throttle_callable() -> None:
    sentinel: list[str] = []

    def callback(host: str) -> None:
        sentinel.append(host)

    mock_channel_cls, _ = _make_grpc_index_with_mock(on_throttle=callback)
    _, kwargs = mock_channel_cls.call_args
    assert kwargs.get("on_throttle") is callback


def test_pinecone_client_wires_registry_to_grpc_index() -> None:
    """GrpcIndex constructed via Pinecone.index(grpc=True) receives the registry callback."""
    from pinecone import Pinecone

    pc = Pinecone(api_key="test-key")

    mock_channel = MagicMock()
    mock_module = MagicMock()
    mock_module.GrpcChannel.return_value = mock_channel

    with (
        patch.dict("sys.modules", {_MOCK_GRPC_MODULE_PATH: mock_module}),
        patch.object(pc, "_resolve_index_host", return_value="test-idx-abc.svc.pinecone.io"),
    ):
        pc.index(host="test-idx-abc.svc.pinecone.io", grpc=True)

    _, kwargs = mock_module.GrpcChannel.call_args
    on_throttle = kwargs.get("on_throttle")
    assert on_throttle is not None, "on_throttle should be wired from the limiter registry"
    assert callable(on_throttle)
    # Bound methods create a new object on each attribute access, so compare via __func__/__self__
    assert on_throttle.__func__ is pc._limiter_registry.report_throttled.__func__
    assert on_throttle.__self__ is pc._limiter_registry


def test_grpc_index_direct_construction_on_throttle_none_does_not_raise() -> None:
    """GrpcIndex constructed directly (no parent client) works fine with on_throttle=None."""
    from pinecone.grpc import GrpcIndex

    mock_channel = MagicMock()
    mock_module = MagicMock()
    mock_module.GrpcChannel.return_value = mock_channel
    with patch.dict("sys.modules", {_MOCK_GRPC_MODULE_PATH: mock_module}):
        GrpcIndex(
            host="https://direct-construction.svc.pinecone.io",
            api_key="test-key",
        )

    _, kwargs = mock_module.GrpcChannel.call_args
    assert kwargs.get("on_throttle") is None
