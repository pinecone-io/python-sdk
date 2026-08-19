"""Tests that GrpcIndex wires the throttle callback to the _AdaptiveLimiterRegistry."""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import MagicMock, patch

from pinecone._internal.adaptive import _AdaptiveLimiterRegistry
from pinecone.models.batch import BatchResult

_MOCK_GRPC_MODULE_PATH = "pinecone._grpc"


def _empty_batch_result() -> BatchResult:
    return BatchResult(
        total_item_count=2,
        successful_item_count=2,
        failed_item_count=0,
        total_batch_count=1,
        successful_batch_count=1,
        failed_batch_count=0,
        errors=[],
        response_info=None,
    )


def _rust_style_host(host: str) -> str:
    """Reproduce parse_host_from_endpoint: strip scheme, then port and path."""
    bare = host.split("://", 1)[-1]
    return bare.split(":", 1)[0].split("/", 1)[0]


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


class TestLimiterReachesTheBatchPath:
    """The callback half was already wired; the consulting half was not."""

    @staticmethod
    def _index_with_registry(registry: _AdaptiveLimiterRegistry) -> tuple[MagicMock, object]:
        from pinecone.grpc import GrpcIndex

        mock_channel = MagicMock()
        mock_channel.upsert.return_value = {"upserted_count": 1}
        mock_module = MagicMock()
        mock_module.GrpcChannel.return_value = mock_channel
        with patch.dict("sys.modules", {_MOCK_GRPC_MODULE_PATH: mock_module}):
            idx = GrpcIndex(
                host="https://test-idx-abc.svc.pinecone.io",
                api_key="test-key",
                on_throttle=registry.report_throttled,
                limiter_registry=registry,
            )
        return mock_channel, idx

    def test_registry_and_host_are_passed_to_batch_execute(self) -> None:
        registry = _AdaptiveLimiterRegistry()
        _, idx = self._index_with_registry(registry)

        with patch("pinecone.grpc.batch_execute", autospec=True) as spy:
            spy.return_value = _empty_batch_result()
            idx.upsert(
                vectors=[{"id": "v1", "values": [0.1]}, {"id": "v2", "values": [0.2]}],
                batch_size=1,
                show_progress=False,
            )

        kwargs = spy.call_args[1]
        assert kwargs["limiter_registry"] is registry
        assert kwargs["host"] == "test-idx-abc.svc.pinecone.io"

    def test_the_host_key_matches_what_the_throttle_callback_reports(self) -> None:
        """Keying on self._host would leave the limiter pinned at its ceiling.

        self._host keeps the https:// prefix normalize_host adds, while the Rust
        callback reports the bare hostname parse_host_from_endpoint produced. Two
        keys means the batch path consults a limiter no throttle ever reaches.
        """
        registry = _AdaptiveLimiterRegistry()
        _, idx = self._index_with_registry(registry)

        with patch("pinecone.grpc.batch_execute", autospec=True) as spy:
            spy.return_value = _empty_batch_result()
            idx.upsert(vectors=[{"id": "v1", "values": [0.1]}], batch_size=1, show_progress=False)

        passed_host = spy.call_args[1]["host"]
        assert passed_host == _rust_style_host(idx.host)
        assert passed_host != idx.host, "the two keys are only equal if the scheme is gone"

    def test_sustained_throttling_lowers_the_limit_the_batch_path_gates_on(self) -> None:
        registry = _AdaptiveLimiterRegistry()
        mock_channel, idx = self._index_with_registry(registry)
        host = _rust_style_host(idx.host)
        ceiling = 16

        observed: list[int] = []

        def _throttling_upsert(vectors, namespace, **_):
            # What the Rust retry loop does on a retryable error before it
            # retries and, here, succeeds.
            registry.report_throttled(host)
            observed.append(registry.get(host, ceiling).current_limit())
            return {"upserted_count": len(vectors)}

        mock_channel.upsert.side_effect = _throttling_upsert

        vectors = [{"id": f"v{i}", "values": [float(i)]} for i in range(80)]
        idx.upsert(vectors=vectors, batch_size=1, max_concurrency=ceiling, show_progress=False)

        assert observed, "the fake channel was never called"
        assert min(observed) < ceiling, "throttling never reached the limiter the gate reads"
        assert registry.get(host, ceiling).current_limit() < ceiling

    def test_limiter_recovers_after_throttling_stops(self) -> None:
        """AIMD's increase half — batch_execute reports successes, not just failures."""
        registry = _AdaptiveLimiterRegistry()
        _, idx = self._index_with_registry(registry)
        host = _rust_style_host(idx.host)

        limiter = registry.get(host, 16)
        for _ in range(4):
            limiter.report_throttled()
        floor = limiter.current_limit()
        assert floor < 16

        vectors = [{"id": f"v{i}", "values": [float(i)]} for i in range(60)]
        idx.upsert(vectors=vectors, batch_size=1, max_concurrency=16, show_progress=False)

        assert registry.get(host, 16).current_limit() > floor

    def test_no_registry_means_no_gating(self) -> None:
        """A bare GrpcIndex has no client behind it and must still work."""
        from pinecone.grpc import GrpcIndex

        mock_channel = MagicMock()
        mock_channel.upsert.return_value = {"upserted_count": 1}
        mock_module = MagicMock()
        mock_module.GrpcChannel.return_value = mock_channel
        with patch.dict("sys.modules", {_MOCK_GRPC_MODULE_PATH: mock_module}):
            idx = GrpcIndex(host="https://test-idx-abc.svc.pinecone.io", api_key="test-key")

        result = idx.upsert(
            vectors=[{"id": "v1", "values": [0.1]}], batch_size=1, show_progress=False
        )

        assert result.upserted_count == 1


class TestClientWiresTheRegistry:
    def test_index_grpc_true_passes_the_client_registry(self) -> None:
        from pinecone import Pinecone

        pc = Pinecone(api_key="test-key")
        mock_module = MagicMock()
        mock_module.GrpcChannel.return_value = MagicMock()
        with patch.dict("sys.modules", {_MOCK_GRPC_MODULE_PATH: mock_module}):
            idx = pc.index(host="test-idx-abc.svc.pinecone.io", grpc=True)

        assert idx._limiter_registry is pc._limiter_registry
