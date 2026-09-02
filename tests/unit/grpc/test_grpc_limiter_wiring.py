"""Tests that GrpcIndex wires throttles and admission to the bulk gate.

Since the bulk-core rewrite (#69/#70) the mechanism is the process-global
gate registry: the transport's throttle callback always feeds it, and the
bulk path always gates on it — there is no unwired configuration to test
for anymore, only that the wiring reaches the one registry.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from pinecone._internal.bulk import bulk_execute_sync, get_registry
from pinecone.models.batch import BatchResult

_MOCK_GRPC_MODULE_PATH = "pinecone._grpc"
BARE_HOST = "test-idx-abc.svc.pinecone.io"


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
    on_throttle: Any = None,
) -> tuple[MagicMock, Any]:
    from pinecone.grpc import GrpcIndex

    mock_channel = MagicMock()
    mock_channel.upsert.return_value = {"upserted_count": 1}
    mock_module = MagicMock()
    mock_module.GrpcChannel.return_value = mock_channel
    with patch.dict("sys.modules", {_MOCK_GRPC_MODULE_PATH: mock_module}):
        idx = GrpcIndex(
            host=f"https://{BARE_HOST}",
            api_key="test-key",
            on_throttle=on_throttle,
        )
    return mock_module.GrpcChannel, idx


def test_channel_always_gets_a_throttle_callback() -> None:
    """The gate must always hear throttles, so the callback is never None —
    even for a directly-constructed index with no user hook."""
    mock_channel_cls, _ = _make_grpc_index_with_mock(on_throttle=None)
    _, kwargs = mock_channel_cls.call_args
    callback = kwargs.get("on_throttle")
    assert callable(callback)


def test_channel_callback_feeds_the_global_gate() -> None:
    mock_channel_cls, _ = _make_grpc_index_with_mock(on_throttle=None)
    callback = mock_channel_cls.call_args[1]["on_throttle"]
    gate = get_registry().get(BARE_HOST)
    before = gate.limit
    callback(BARE_HOST)
    assert gate.limit < before


def test_user_on_throttle_is_composed_not_replaced() -> None:
    """A caller-supplied hook still fires, and the gate still hears."""
    seen: list[str] = []
    mock_channel_cls, _ = _make_grpc_index_with_mock(on_throttle=seen.append)
    callback = mock_channel_cls.call_args[1]["on_throttle"]
    gate = get_registry().get(BARE_HOST)
    before = gate.limit
    callback(BARE_HOST)
    assert seen == [BARE_HOST]
    assert gate.limit < before


def test_retry_config_on_throttle_is_composed() -> None:
    """A hook threaded through a client-built RetryConfig is not dropped."""
    from pinecone import RetryConfig
    from pinecone.grpc import GrpcIndex

    seen: list[str] = []
    config = RetryConfig(max_retries=2, backoff_factor=0.1, on_throttle=seen.append)
    mock_module = MagicMock()
    mock_module.GrpcChannel.return_value = MagicMock()
    with patch.dict("sys.modules", {_MOCK_GRPC_MODULE_PATH: mock_module}):
        GrpcIndex(host=f"https://{BARE_HOST}", api_key="test-key", retry_config=config)
    callback = mock_module.GrpcChannel.call_args[1]["on_throttle"]
    callback(BARE_HOST)
    assert seen == [BARE_HOST]


def test_pinecone_client_grpc_index_throttles_reach_the_gate() -> None:
    from pinecone import Pinecone

    pc = Pinecone(api_key="test-key")
    mock_module = MagicMock()
    mock_module.GrpcChannel.return_value = MagicMock()
    with (
        patch.dict("sys.modules", {_MOCK_GRPC_MODULE_PATH: mock_module}),
        patch.object(pc, "_resolve_index_host", return_value=BARE_HOST),
    ):
        pc.index(host=BARE_HOST, grpc=True)

    callback = mock_module.GrpcChannel.call_args[1]["on_throttle"]
    gate = get_registry().get(BARE_HOST)
    before = gate.limit
    callback(BARE_HOST)
    assert gate.limit < before


class TestGateReachesTheBatchPath:
    """The consulting half: bulk calls admit through the same gate the
    throttle callback feeds — same registry, same bare-host key."""

    def test_bare_host_is_passed_to_the_engine(self) -> None:
        _, idx = _make_grpc_index_with_mock()
        with patch("pinecone.grpc.bulk_execute_sync", autospec=True) as spy:
            spy.return_value = _empty_batch_result()
            idx.upsert(
                vectors=[{"id": "v1", "values": [0.1]}, {"id": "v2", "values": [0.2]}],
                batch_size=1,
                show_progress=False,
            )
        passed_host = spy.call_args[1]["host"]
        assert passed_host == _rust_style_host(idx.host)
        assert passed_host != idx.host, "the two keys are only equal if the scheme is gone"

    def test_sustained_throttling_lowers_the_limit_the_engine_gates_on(self) -> None:
        mock_channel_cls, idx = _make_grpc_index_with_mock()
        mock_channel = mock_channel_cls.return_value
        callback = mock_channel_cls.call_args[1]["on_throttle"]
        gate = get_registry().get(BARE_HOST)
        ceiling = gate.limit

        observed: list[int] = []

        def _throttling_upsert(vectors: Any, namespace: Any, **_: Any) -> dict[str, int]:
            callback(BARE_HOST)
            observed.append(gate.limit)
            return {"upserted_count": len(vectors)}

        mock_channel.upsert.side_effect = _throttling_upsert
        vectors = [{"id": f"v{i}", "values": [float(i)]} for i in range(40)]
        idx.upsert(vectors=vectors, batch_size=1, max_concurrency=16, show_progress=False)

        assert observed, "the fake channel was never called"
        assert min(observed) < ceiling, "throttling never reached the gate the engine reads"

    def test_gate_recovers_after_throttling_stops(self) -> None:
        """AIMD's increase half — the engine reports successes, not just failures."""
        _, idx = _make_grpc_index_with_mock()
        gate = get_registry().get(BARE_HOST)
        for _ in range(6):
            gate.report_throttled()
        floor = gate.limit
        assert floor < 16

        vectors = [{"id": f"v{i}", "values": [float(i)]} for i in range(60)]
        idx.upsert(vectors=vectors, batch_size=1, max_concurrency=16, show_progress=False)

        assert gate.limit > floor

    def test_direct_construction_is_gated_too(self) -> None:
        """No client, no configuration — the bulk path still admits through
        the global gate, because there is no ungated path anymore (#57)."""
        _, idx = _make_grpc_index_with_mock()
        with patch("pinecone.grpc.bulk_execute_sync", autospec=True) as spy:
            spy.return_value = _empty_batch_result()
            idx.upsert(vectors=[{"id": "v1", "values": [0.1]}], batch_size=1, show_progress=False)
        assert spy.call_args[1]["host"] == BARE_HOST

    def test_bare_index_upsert_works(self) -> None:
        _, idx = _make_grpc_index_with_mock()
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
            idx = pc.index(host=BARE_HOST, grpc=True)

        assert idx._limiter_registry is pc._limiter_registry


def test_engine_is_the_real_one() -> None:
    """Guard against the spy target drifting from the import the code uses."""
    import pinecone.grpc as grpc_mod

    assert grpc_mod.bulk_execute_sync is bulk_execute_sync
