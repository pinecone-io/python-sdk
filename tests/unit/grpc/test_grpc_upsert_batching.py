"""Unit tests for GrpcIndex.upsert() batch_size and show_progress parameters (BCG-090)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pinecone.errors.exceptions import PineconeValueError
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


def _make_vectors(n: int) -> list[tuple[str, list[float]]]:
    return [(f"v{i}", [float(i)]) for i in range(n)]


@pytest.fixture
def mock_channel() -> MagicMock:
    ch = MagicMock()
    ch.upsert.return_value = {"upserted_count": 1}
    return ch


@pytest.fixture
def grpc_index(mock_channel: MagicMock) -> GrpcIndex:
    return _make_grpc_index(mock_channel)


class TestGrpcUpsertBatching:
    def test_upsert_no_batch_size_calls_channel_once(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        """With batch_size=None, all 100 vectors go in a single channel call."""
        mock_channel.upsert.return_value = {"upserted_count": 100}
        vectors = _make_vectors(100)
        result = grpc_index.upsert(vectors=vectors, batch_size=None)
        assert mock_channel.upsert.call_count == 1
        assert result.upserted_count == 100

    def test_upsert_with_batch_size_calls_channel_per_batch(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        """250 vectors at batch_size=100 should produce exactly 3 channel upsert calls."""
        mock_channel.upsert.return_value = {"upserted_count": 100}
        vectors = _make_vectors(250)
        grpc_index.upsert(vectors=vectors, batch_size=100, show_progress=False)
        assert mock_channel.upsert.call_count == 3

    def test_upsert_with_batch_size_aggregates_response(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        """upserted_count should equal the number of items in successful batches."""
        mock_channel.upsert.return_value = {"upserted_count": 100}
        vectors = _make_vectors(250)
        result = grpc_index.upsert(vectors=vectors, batch_size=100, show_progress=False)
        assert result.upserted_count == 250

    def test_upsert_invalid_batch_size_raises(self, grpc_index: GrpcIndex) -> None:
        """batch_size of 0, -1, or a float should raise PineconeValueError."""
        vectors = _make_vectors(5)
        with pytest.raises(PineconeValueError, match="batch_size must be a positive integer"):
            grpc_index.upsert(vectors=vectors, batch_size=0)
        with pytest.raises(PineconeValueError, match="batch_size must be a positive integer"):
            grpc_index.upsert(vectors=vectors, batch_size=-1)
        with pytest.raises(PineconeValueError, match="batch_size must be a positive integer"):
            grpc_index.upsert(vectors=vectors, batch_size=1.5)  # type: ignore[arg-type]

    def test_upsert_invalid_max_concurrency_raises(self, grpc_index: GrpcIndex) -> None:
        """max_concurrency outside [1, 64] should raise PineconeValueError."""
        vectors = _make_vectors(5)
        with pytest.raises(PineconeValueError):
            grpc_index.upsert(vectors=vectors, batch_size=2, max_concurrency=0)
        with pytest.raises(PineconeValueError):
            grpc_index.upsert(vectors=vectors, batch_size=2, max_concurrency=65)
        with pytest.raises(PineconeValueError):
            grpc_index.upsert(vectors=vectors, batch_size=2, max_concurrency=-1)

    def test_upsert_max_concurrency_default_is_4(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        """Default max_concurrency of 4 is forwarded to _get_batch_executor."""
        mock_channel.upsert.return_value = {"upserted_count": 5}
        vectors = _make_vectors(10)
        with patch.object(
            grpc_index, "_get_batch_executor", wraps=grpc_index._get_batch_executor
        ) as mock_exec:
            grpc_index.upsert(vectors=vectors, batch_size=5, show_progress=False)
            mock_exec.assert_called_once_with(4)

    def test_upsert_max_concurrency_explicit(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        """Explicit max_concurrency=8 is forwarded to _get_batch_executor."""
        mock_channel.upsert.return_value = {"upserted_count": 5}
        vectors = _make_vectors(10)
        with patch.object(
            grpc_index, "_get_batch_executor", wraps=grpc_index._get_batch_executor
        ) as mock_exec:
            grpc_index.upsert(vectors=vectors, batch_size=5, max_concurrency=8, show_progress=False)
            mock_exec.assert_called_once_with(8)

    def test_upsert_show_progress_false_does_not_import_tqdm(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        """When show_progress=False, tqdm must not be imported."""
        mock_channel.upsert.return_value = {"upserted_count": 5}
        vectors = _make_vectors(10)
        with patch.dict("sys.modules", {"tqdm": None, "tqdm.auto": None}):
            result = grpc_index.upsert(vectors=vectors, batch_size=5, show_progress=False)
        assert result.upserted_count == 10

    def test_upsert_partial_failure_returns_rich_response(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        """When a batch fails, returns UpsertResponse with has_errors=True, no raise."""
        err = RuntimeError("gRPC error on batch 1")
        call_count = 0

        def side_effect(
            chunk: list[dict[str, object]], ns: object, *, timeout_s: object
        ) -> dict[str, int]:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise err
            return {"upserted_count": len(chunk)}

        mock_channel.upsert.side_effect = side_effect
        vectors = _make_vectors(200)
        result = grpc_index.upsert(vectors=vectors, batch_size=100, show_progress=False)

        assert result.has_errors is True
        assert result.failed_batch_count == 1
        assert result.failed_item_count == 100
        assert result.successful_batch_count == 1
        assert result.errors[0].error is err

    def test_upsert_partial_failure_failed_items_list(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        """failed_items returns items from failed batches, ready for retry."""
        call_count = 0

        def side_effect(
            chunk: list[dict[str, object]], ns: object, *, timeout_s: object
        ) -> dict[str, int]:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("failure")
            return {"upserted_count": len(chunk)}

        mock_channel.upsert.side_effect = side_effect
        vectors = _make_vectors(200)
        result = grpc_index.upsert(vectors=vectors, batch_size=100, show_progress=False)

        assert result.has_errors is True
        assert len(result.failed_items) == 100

    def test_upsert_namespace_forwarded_per_batch(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        """The namespace argument must appear in every per-batch channel call."""
        mock_channel.upsert.return_value = {"upserted_count": 5}
        vectors = _make_vectors(10)
        grpc_index.upsert(vectors=vectors, namespace="my-ns", batch_size=5, show_progress=False)
        assert mock_channel.upsert.call_count == 2
        for c in mock_channel.upsert.call_args_list:
            assert c[0][1] == "my-ns"

    def test_upsert_timeout_forwarded_per_batch(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        """timeout=5.0 should be forwarded to each per-batch channel call."""
        mock_channel.upsert.return_value = {"upserted_count": 5}
        vectors = _make_vectors(10)
        grpc_index.upsert(vectors=vectors, batch_size=5, timeout=5.0, show_progress=False)
        assert mock_channel.upsert.call_count == 2
        for c in mock_channel.upsert.call_args_list:
            assert c[1].get("timeout_s") == 5.0

    def test_upsert_empty_vectors_with_batch_size(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        """Empty vector list with batch_size set should make zero channel calls."""
        result = grpc_index.upsert(vectors=[], batch_size=100, show_progress=False)
        assert mock_channel.upsert.call_count == 0
        assert result.upserted_count == 0

    def test_upsert_executor_is_cached_across_calls(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        """Same max_concurrency reuses the same ThreadPoolExecutor instance."""
        mock_channel.upsert.return_value = {"upserted_count": 5}
        vectors = _make_vectors(10)
        grpc_index.upsert(vectors=vectors, batch_size=5, max_concurrency=4, show_progress=False)
        executor_first = grpc_index._batch_executors[4]
        grpc_index.upsert(vectors=vectors, batch_size=5, max_concurrency=4, show_progress=False)
        executor_second = grpc_index._batch_executors[4]
        assert executor_first is executor_second

    def test_each_max_concurrency_keeps_its_own_executor(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        """The pools coexist rather than one replacing the other.

        upsert() and upsert_from_dataframe() have different concurrency defaults
        and can run concurrently on one handle, so shutting a pool down because
        the other caller asked for a different size would raise "cannot schedule
        new futures after shutdown" in whichever was still submitting.
        """
        mock_channel.upsert.return_value = {"upserted_count": 5}
        vectors = _make_vectors(10)
        grpc_index.upsert(vectors=vectors, batch_size=5, max_concurrency=4, show_progress=False)
        executor_first = grpc_index._batch_executors[4]
        grpc_index.upsert(vectors=vectors, batch_size=5, max_concurrency=8, show_progress=False)

        assert grpc_index._batch_executors[8] is not executor_first
        assert grpc_index._batch_executors[4] is executor_first
        assert executor_first.submit(lambda: "alive").result() == "alive"

    def test_close_shuts_down_every_batch_executor(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        """close() must shut down each pool that was created."""
        mock_channel.upsert.return_value = {"upserted_count": 5}
        vectors = _make_vectors(10)
        grpc_index.upsert(vectors=vectors, batch_size=5, max_concurrency=4, show_progress=False)
        grpc_index.upsert(vectors=vectors, batch_size=5, max_concurrency=8, show_progress=False)
        executors = list(grpc_index._batch_executors.values())
        assert len(executors) == 2

        grpc_index.close()

        for executor in executors:
            with pytest.raises(RuntimeError):
                executor.submit(lambda: None)
