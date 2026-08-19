"""Unit tests for per-call timeout on GrpcIndex data-plane methods.

Verifies that:
1. Each method accepts a `timeout` kwarg and forwards it as `timeout_s` to the channel.
2. When `timeout=None`, `timeout_s=None` is passed (channel default applies).
3. DEADLINE_EXCEEDED in the exception message raises PineconeTimeoutError.
4. PineconeTimeoutError is catchable as TimeoutError (built-in).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pinecone._internal.config import RetryConfig
from pinecone.errors.exceptions import PineconeConnectionError, PineconeTimeoutError
from pinecone.grpc import GrpcIndex

_MOCK_GRPC_MODULE_PATH = "pinecone._grpc"


def _make_grpc_index(mock_channel: MagicMock) -> GrpcIndex:
    mock_module = MagicMock()
    mock_module.GrpcChannel.return_value = mock_channel
    with patch.dict("sys.modules", {_MOCK_GRPC_MODULE_PATH: mock_module}):
        return GrpcIndex(
            host="test-index-abc123.svc.pinecone.io",
            api_key="test-api-key",
        )


@pytest.fixture
def mock_channel() -> MagicMock:
    ch = MagicMock()
    ch.query.return_value = {"matches": [], "namespace": ""}
    ch.fetch.return_value = {"vectors": {}, "namespace": ""}
    ch.upsert.return_value = {"upserted_count": 1}
    ch.delete.return_value = {}
    ch.update.return_value = {}
    ch.list.return_value = {"vectors": [], "namespace": ""}
    return ch


@pytest.fixture
def grpc_index(mock_channel: MagicMock) -> GrpcIndex:
    return _make_grpc_index(mock_channel)


class TestTimeoutForwarding:
    """Each method forwards timeout kwarg as timeout_s to the underlying channel."""

    def test_query_forwards_timeout(self, grpc_index: GrpcIndex, mock_channel: MagicMock) -> None:
        grpc_index.query(top_k=5, vector=[0.1, 0.2], timeout=2.5)
        _, kwargs = mock_channel.query.call_args
        assert kwargs.get("timeout_s") == 2.5

    def test_query_none_timeout_passes_none(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        grpc_index.query(top_k=5, vector=[0.1, 0.2])
        _, kwargs = mock_channel.query.call_args
        assert kwargs.get("timeout_s") is None

    def test_fetch_forwards_timeout(self, grpc_index: GrpcIndex, mock_channel: MagicMock) -> None:
        grpc_index.fetch(ids=["v1"], timeout=3.0)
        _, kwargs = mock_channel.fetch.call_args
        assert kwargs.get("timeout_s") == 3.0

    def test_fetch_none_timeout_passes_none(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        grpc_index.fetch(ids=["v1"])
        _, kwargs = mock_channel.fetch.call_args
        assert kwargs.get("timeout_s") is None

    def test_upsert_forwards_timeout(self, grpc_index: GrpcIndex, mock_channel: MagicMock) -> None:
        grpc_index.upsert(vectors=[{"id": "v1", "values": [0.1, 0.2]}], timeout=5.0)
        _args, kwargs = mock_channel.upsert.call_args
        assert kwargs.get("timeout_s") == 5.0

    def test_upsert_none_timeout_passes_none(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        grpc_index.upsert(vectors=[{"id": "v1", "values": [0.1, 0.2]}])
        _args, kwargs = mock_channel.upsert.call_args
        assert kwargs.get("timeout_s") is None

    def test_delete_forwards_timeout(self, grpc_index: GrpcIndex, mock_channel: MagicMock) -> None:
        grpc_index.delete(ids=["v1"], timeout=1.0)
        _, kwargs = mock_channel.delete.call_args
        assert kwargs.get("timeout_s") == 1.0

    def test_delete_none_timeout_passes_none(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        grpc_index.delete(ids=["v1"])
        _, kwargs = mock_channel.delete.call_args
        assert kwargs.get("timeout_s") is None

    def test_update_forwards_timeout(self, grpc_index: GrpcIndex, mock_channel: MagicMock) -> None:
        grpc_index.update(id="v1", values=[0.1, 0.2], timeout=4.0)
        _, kwargs = mock_channel.update.call_args
        assert kwargs.get("timeout_s") == 4.0

    def test_update_none_timeout_passes_none(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        grpc_index.update(id="v1", values=[0.1, 0.2])
        _, kwargs = mock_channel.update.call_args
        assert kwargs.get("timeout_s") is None

    def test_list_paginated_forwards_timeout(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        grpc_index.list_paginated(timeout=2.0)
        _, kwargs = mock_channel.list.call_args
        assert kwargs.get("timeout_s") == 2.0

    def test_list_paginated_none_timeout_passes_none(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        grpc_index.list_paginated()
        _, kwargs = mock_channel.list.call_args
        assert kwargs.get("timeout_s") is None


class TestDeadlineExceededRaisesPineconeTimeoutError:
    """PineconeTimeoutError from Rust propagates unchanged through GrpcIndex methods.

    The Rust transport raises PineconeTimeoutError directly when a gRPC
    DEADLINE_EXCEEDED status is received. The Python layer no longer
    does any exception mapping — it just calls self._channel.method(...)
    and lets the typed exception propagate.
    """

    def _deadline_channel(self) -> MagicMock:
        ch = MagicMock()
        exc = PineconeTimeoutError("deadline exceeded after 20s")
        ch.query.side_effect = exc
        ch.fetch.side_effect = exc
        ch.upsert.side_effect = exc
        ch.delete.side_effect = exc
        ch.update.side_effect = exc
        ch.list.side_effect = exc
        return ch

    def test_query_deadline_raises_timeout_error(self) -> None:
        idx = _make_grpc_index(self._deadline_channel())
        with pytest.raises(PineconeTimeoutError):
            idx.query(top_k=5, vector=[0.1, 0.2], timeout=0.001)

    def test_fetch_deadline_raises_timeout_error(self) -> None:
        idx = _make_grpc_index(self._deadline_channel())
        with pytest.raises(PineconeTimeoutError):
            idx.fetch(ids=["v1"], timeout=0.001)

    def test_upsert_deadline_raises_timeout_error(self) -> None:
        idx = _make_grpc_index(self._deadline_channel())
        with pytest.raises(PineconeTimeoutError):
            idx.upsert(vectors=[{"id": "v1", "values": [0.1, 0.2]}], timeout=0.001)

    def test_delete_deadline_raises_timeout_error(self) -> None:
        idx = _make_grpc_index(self._deadline_channel())
        with pytest.raises(PineconeTimeoutError):
            idx.delete(ids=["v1"], timeout=0.001)

    def test_update_deadline_raises_timeout_error(self) -> None:
        idx = _make_grpc_index(self._deadline_channel())
        with pytest.raises(PineconeTimeoutError):
            idx.update(id="v1", values=[0.1, 0.2], timeout=0.001)

    def test_list_paginated_deadline_raises_timeout_error(self) -> None:
        idx = _make_grpc_index(self._deadline_channel())
        with pytest.raises(PineconeTimeoutError):
            idx.list_paginated(timeout=0.001)

    def test_pinecone_timeout_error_is_catchable_as_builtin_timeout_error(self) -> None:
        """PineconeTimeoutError inherits from TimeoutError for broad exception handlers."""
        idx = _make_grpc_index(self._deadline_channel())
        with pytest.raises(TimeoutError):
            idx.query(top_k=5, vector=[0.1, 0.2], timeout=0.001)


class TestRetryConfigReachesChannel:
    """`retry_config` must actually arrive at the channel constructor.

    Before A2 the Python call site stopped at connect_timeout, so even the
    max_retries the Rust constructor already accepted was unreachable.
    """

    @staticmethod
    def _channel_kwargs(**index_kwargs: object) -> dict[str, object]:
        mock_module = MagicMock()
        mock_module.GrpcChannel.return_value = MagicMock()
        with patch.dict("sys.modules", {_MOCK_GRPC_MODULE_PATH: mock_module}):
            GrpcIndex(
                host="test-index-abc123.svc.pinecone.io",
                api_key="test-api-key",
                **index_kwargs,  # type: ignore[arg-type]
            )
        _, kwargs = mock_module.GrpcChannel.call_args
        return dict(kwargs)

    def test_defaults_are_grpcs_own_not_rests(self) -> None:
        """gRPC keeps 5 retries and a 0.1s floor; only the 1.6s cap changed."""
        kwargs = self._channel_kwargs()
        assert kwargs["max_retries"] == 5
        assert kwargs["backoff_factor_s"] == 0.1
        assert kwargs["max_wait_s"] == 60.0

    def test_explicit_retry_config_is_forwarded(self) -> None:
        kwargs = self._channel_kwargs(
            retry_config=RetryConfig(max_retries=2, backoff_factor=0.5, max_wait=30.0)
        )
        assert kwargs["max_retries"] == 2
        assert kwargs["backoff_factor_s"] == 0.5
        assert kwargs["max_wait_s"] == 30.0

    def test_max_retries_zero_is_forwarded_not_treated_as_unset(self) -> None:
        """0 disables retries. It must not be swallowed by an `or 5` fallback."""
        kwargs = self._channel_kwargs(retry_config=RetryConfig(max_retries=0))
        assert kwargs["max_retries"] == 0

    def test_proxy_url_is_forwarded(self) -> None:
        kwargs = self._channel_kwargs(proxy_url="http://proxy.example.com:8080")
        assert kwargs["proxy_url"] == "http://proxy.example.com:8080"

    def test_retryable_status_codes_are_not_forwarded(self) -> None:
        """HTTP statuses are meaningless to a gRPC channel; codes stay fixed."""
        kwargs = self._channel_kwargs(
            retry_config=RetryConfig(retryable_status_codes=frozenset({429}))
        )
        assert not any("status" in key for key in kwargs)

    def test_on_throttle_from_retry_config_is_not_dropped(self) -> None:
        """The client wires the limiter hook onto RetryConfig; honor it."""

        def _hook(host: str) -> None:
            pass

        kwargs = self._channel_kwargs(retry_config=RetryConfig(on_throttle=_hook))
        assert kwargs["on_throttle"] is _hook

    def test_explicit_on_throttle_wins_over_retry_config(self) -> None:
        def _from_config(host: str) -> None:
            pass

        def _explicit(host: str) -> None:
            pass

        kwargs = self._channel_kwargs(
            retry_config=RetryConfig(on_throttle=_from_config), on_throttle=_explicit
        )
        assert kwargs["on_throttle"] is _explicit


class TestMaxRetriesZeroRaisesImmediately:
    """With retries disabled, one failing attempt surfaces straight away."""

    def test_single_attempt_on_retryable_failure(self) -> None:
        mock_channel = MagicMock()
        mock_channel.upsert.side_effect = PineconeConnectionError("UNAVAILABLE: no backend")
        mock_module = MagicMock()
        mock_module.GrpcChannel.return_value = mock_channel
        with patch.dict("sys.modules", {_MOCK_GRPC_MODULE_PATH: mock_module}):
            idx = GrpcIndex(
                host="test-index-abc123.svc.pinecone.io",
                api_key="test-api-key",
                retry_config=RetryConfig(max_retries=0),
            )

        with pytest.raises(PineconeConnectionError):
            idx.upsert(vectors=[{"id": "v1", "values": [0.1, 0.2]}])

        # Retries live in Rust; the Python layer must not add a loop of its own.
        assert mock_channel.upsert.call_count == 1


class TestClientThreadsRetryConfigToGrpc:
    """Pinecone(retry_config=...) must reach a grpc=True index."""

    def test_explicit_config_reaches_grpc_index(self) -> None:
        from pinecone import Pinecone

        pc = Pinecone(api_key="test-api-key", retry_config=RetryConfig(max_retries=1))
        mock_module = MagicMock()
        mock_module.GrpcChannel.return_value = MagicMock()
        with patch.dict("sys.modules", {_MOCK_GRPC_MODULE_PATH: mock_module}):
            pc.index(host="test-index-abc123.svc.pinecone.io", grpc=True)

        _, kwargs = mock_module.GrpcChannel.call_args
        assert kwargs["max_retries"] == 1

    def test_unset_config_leaves_grpc_defaults_alone(self) -> None:
        """REST defaults to 3 retries; gRPC must not silently inherit that."""
        from pinecone import Pinecone

        pc = Pinecone(api_key="test-api-key")
        mock_module = MagicMock()
        mock_module.GrpcChannel.return_value = MagicMock()
        with patch.dict("sys.modules", {_MOCK_GRPC_MODULE_PATH: mock_module}):
            pc.index(host="test-index-abc123.svc.pinecone.io", grpc=True)

        _, kwargs = mock_module.GrpcChannel.call_args
        assert kwargs["max_retries"] == 5
