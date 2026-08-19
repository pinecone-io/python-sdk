"""Unit tests for GrpcIndex.upsert_from_dataframe()."""

from __future__ import annotations

import builtins
import inspect
import os
import threading
from concurrent.futures import Future
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from pinecone.errors.exceptions import PineconeValueError
from pinecone.grpc import GrpcIndex, _default_max_concurrency
from pinecone.models.vectors.responses import UpsertResponse

_MOCK_GRPC_MODULE_PATH = "pinecone._grpc"


def _make_grpc_index(mock_channel: MagicMock) -> GrpcIndex:
    """Helper to create a GrpcIndex with a mocked GrpcChannel."""
    mock_module = MagicMock()
    mock_module.GrpcChannel.return_value = mock_channel
    with patch.dict("sys.modules", {_MOCK_GRPC_MODULE_PATH: mock_module}):
        return GrpcIndex(
            host="test-index-abc123.svc.pinecone.io",
            api_key="test-api-key",
        )


def _upsert_response(upserted_count: int) -> UpsertResponse:
    return UpsertResponse(upserted_count=upserted_count)


def _upsert_call_count(ch: MagicMock) -> int:
    """Number of batches that reached the channel.

    `upsert_from_dataframe` submits every batch concurrently, and Mock only
    updates `call_count` and `call_args_list` atomically on Python 3.13+. On
    3.10-3.12 a concurrent `call_count += 1` loses increments, so `call_count`
    undercounts. `list.append` is atomic on every version, which makes the
    length of `call_args_list` the reliable count.
    """
    return len(ch.upsert.call_args_list)


@pytest.fixture
def mock_channel() -> MagicMock:
    ch = MagicMock()
    ch.upsert.return_value = {"upserted_count": 500}
    return ch


@pytest.fixture
def grpc_index(mock_channel: MagicMock) -> GrpcIndex:
    return _make_grpc_index(mock_channel)


class TestGrpcDataframeUpsert:
    """Tests for GrpcIndex.upsert_from_dataframe()."""

    def test_dataframe_1200_rows_creates_3_batches(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        """1200 rows at batch_size=500 should produce 3 async calls: 500+500+200."""
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame(
            {
                "id": [f"v{i}" for i in range(1200)],
                "values": [[float(i)] for i in range(1200)],
            }
        )
        mock_channel.upsert.side_effect = [
            {"upserted_count": 500},
            {"upserted_count": 500},
            {"upserted_count": 200},
        ]

        result = grpc_index.upsert_from_dataframe(df, show_progress=False)

        assert _upsert_call_count(mock_channel) == 3
        batch_sizes = [len(call[0][0]) for call in mock_channel.upsert.call_args_list]
        assert sorted(batch_sizes) == [200, 500, 500]
        assert result.upserted_count == 1200

    def test_default_batch_size_is_500(self) -> None:
        """The default batch_size parameter should be 500."""
        sig = inspect.signature(GrpcIndex.upsert_from_dataframe)
        assert sig.parameters["batch_size"].default == 500

    def test_custom_batch_size(self, grpc_index: GrpcIndex, mock_channel: MagicMock) -> None:
        """A custom batch_size should control how many rows per async call."""
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame(
            {
                "id": [f"v{i}" for i in range(10)],
                "values": [[float(i)] for i in range(10)],
            }
        )
        mock_channel.upsert.return_value = {"upserted_count": 3}

        grpc_index.upsert_from_dataframe(df, batch_size=3, show_progress=False)

        assert _upsert_call_count(mock_channel) == 4
        batch_sizes = [len(call[0][0]) for call in mock_channel.upsert.call_args_list]
        assert sorted(batch_sizes) == [1, 3, 3, 3]

    def test_results_aggregated(self, grpc_index: GrpcIndex, mock_channel: MagicMock) -> None:
        """upserted_count should be summed across all batches."""
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame(
            {
                "id": [f"v{i}" for i in range(100)],
                "values": [[float(i)] for i in range(100)],
            }
        )
        mock_channel.upsert.side_effect = [
            {"upserted_count": 50},
            {"upserted_count": 50},
        ]

        result = grpc_index.upsert_from_dataframe(df, batch_size=50, show_progress=False)

        assert result.upserted_count == 100

    def test_namespace_passed_to_each_batch(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        """The namespace argument should be forwarded to every upsert call."""
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame(
            {
                "id": ["v1", "v2"],
                "values": [[0.1, 0.2], [0.3, 0.4]],
            }
        )
        mock_channel.upsert.return_value = {"upserted_count": 2}

        grpc_index.upsert_from_dataframe(df, namespace="my-ns", show_progress=False)

        for call in mock_channel.upsert.call_args_list:
            assert call[0][1] == "my-ns"

    def test_single_batch_when_rows_fit(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        """DataFrame with fewer rows than batch_size should produce one call."""
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame(
            {
                "id": ["v1", "v2", "v3"],
                "values": [[0.1], [0.2], [0.3]],
            }
        )
        mock_channel.upsert.return_value = {"upserted_count": 3}

        result = grpc_index.upsert_from_dataframe(df, show_progress=False)

        assert _upsert_call_count(mock_channel) == 1
        assert result.upserted_count == 3

    def test_invalid_df_raises(self, grpc_index: GrpcIndex) -> None:
        """Non-DataFrame input should raise PineconeValueError."""
        pytest.importorskip("pandas")
        with pytest.raises(PineconeValueError, match="df must be a pandas DataFrame"):
            grpc_index.upsert_from_dataframe([1, 2, 3])  # type: ignore[arg-type]

    def test_batch_size_zero_raises(self, grpc_index: GrpcIndex) -> None:
        """batch_size=0 should raise PineconeValueError."""
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"id": ["v1"], "values": [[0.1]]})
        with pytest.raises(PineconeValueError, match="batch_size must be a positive integer"):
            grpc_index.upsert_from_dataframe(df, batch_size=0)

    def test_batch_size_negative_raises(self, grpc_index: GrpcIndex) -> None:
        """Negative batch_size should raise PineconeValueError."""
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"id": ["v1"], "values": [[0.1]]})
        with pytest.raises(PineconeValueError, match="batch_size must be a positive integer"):
            grpc_index.upsert_from_dataframe(df, batch_size=-1)

    def test_timeout_defaults_to_none(self) -> None:
        """The timeout parameter should default to None."""
        sig = inspect.signature(GrpcIndex.upsert_from_dataframe)
        assert sig.parameters["timeout"].default is None

    def test_timeout_forwarded_to_each_batch(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        """timeout should be threaded through to every channel upsert call."""
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame(
            {
                "id": [f"v{i}" for i in range(5)],
                "values": [[float(i)] for i in range(5)],
            }
        )
        mock_channel.upsert.return_value = {"upserted_count": 2}

        grpc_index.upsert_from_dataframe(df, batch_size=2, timeout=42.0, show_progress=False)

        assert _upsert_call_count(mock_channel) == 3
        for call in mock_channel.upsert.call_args_list:
            assert call.kwargs["timeout_s"] == 42.0

    def test_default_timeout_none_forwarded_to_channel(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        """With no timeout, None is forwarded so the channel applies its default."""
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"id": ["v1", "v2"], "values": [[0.1], [0.2]]})
        mock_channel.upsert.return_value = {"upserted_count": 2}

        grpc_index.upsert_from_dataframe(df, show_progress=False)

        assert mock_channel.upsert.call_args.kwargs["timeout_s"] is None

    def test_client_side_wait_is_unbounded(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        """Collection waits for the server instead of imposing a client deadline.

        The bug behind the customer's report: collection used
        PineconeFuture.result()'s 5s default, so any batch slower than that
        raised PineconeTimeoutError even though the upsert had succeeded. The
        assertion is on the waiting, not on which future type does it.
        """
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"id": ["v1", "v2"], "values": [[0.1], [0.2]]})
        mock_channel.upsert.return_value = {"upserted_count": 1}

        waits: list[object] = []
        real_result = Future.result

        def _recording_result(self: Future[object], timeout: float | None = None) -> object:
            waits.append(timeout)
            return real_result(self, timeout)

        with patch.object(Future, "result", _recording_result):
            result = grpc_index.upsert_from_dataframe(df, batch_size=1, show_progress=False)

        assert result.upserted_count == 2
        assert waits, "collection did not wait on the batches at all"
        assert all(w is None for w in waits), f"collection imposed a client-side deadline: {waits}"

    def test_metadata_and_sparse_values_forwarded(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        """metadata and sparse_values columns should be forwarded to upsert."""
        pd = pytest.importorskip("pandas")
        sparse = {"indices": [0, 2], "values": [0.5, 0.8]}
        df = pd.DataFrame(
            {
                "id": ["v1"],
                "values": [[0.1, 0.2]],
                "sparse_values": [sparse],
                "metadata": [{"genre": "rock"}],
            }
        )
        mock_channel.upsert.return_value = {"upserted_count": 1}

        grpc_index.upsert_from_dataframe(df, show_progress=False)

        vec = mock_channel.upsert.call_args[0][0][0]
        assert vec["sparse_values"] == sparse
        assert vec["metadata"] == {"genre": "rock"}


class TestGrpcDataframePandasMissing:
    """The pandas-missing path must point at the installable extra."""

    def test_missing_pandas_names_the_extra(
        self, grpc_index: GrpcIndex, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real_import = builtins.__import__

        def _fake_import(name: str, *args: object, **kwargs: object) -> ModuleType:
            if name == "pandas":
                raise ImportError("No module named 'pandas'")
            return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(builtins, "__import__", _fake_import)

        with pytest.raises(RuntimeError, match="pip install pandas") as excinfo:
            grpc_index.upsert_from_dataframe("not-a-df")

        message = str(excinfo.value)
        assert "pandas is required" in message
        assert "not a dependency of this SDK" in message


class TestGrpcDataframeMissingCells:
    """Rows that omit an optional key leave NaN behind, not None."""

    def test_nan_metadata_does_not_reach_the_channel(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        """A NaN metadata cell used to surface as `metadata must be a dict, got float`."""
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame(
            [
                {"id": "v1", "values": [0.1], "metadata": {"genre": "rock"}},
                {"id": "v2", "values": [0.2]},
            ]
        )
        mock_channel.upsert.return_value = {"upserted_count": 2}

        grpc_index.upsert_from_dataframe(df, show_progress=False)

        sent = {v["id"]: v for v in mock_channel.upsert.call_args[0][0]}
        assert sent["v1"]["metadata"] == {"genre": "rock"}
        assert "metadata" not in sent["v2"]

    def test_nan_sparse_values_does_not_reach_the_channel(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        pd = pytest.importorskip("pandas")
        sparse = {"indices": [1], "values": [0.5]}
        df = pd.DataFrame(
            [
                {"id": "v1", "values": [0.1], "sparse_values": sparse},
                {"id": "v2", "values": [0.2]},
            ]
        )
        mock_channel.upsert.return_value = {"upserted_count": 2}

        grpc_index.upsert_from_dataframe(df, show_progress=False)

        sent = {v["id"]: v for v in mock_channel.upsert.call_args[0][0]}
        assert sent["v1"]["sparse_values"] == sparse
        assert "sparse_values" not in sent["v2"]


class TestGrpcDataframeConcurrency:
    """Submission is bounded, and the bound is a parameter."""

    def test_default_max_concurrency_matches_the_old_unbounded_pool(self) -> None:
        """min(32, cpu+4) is what a default ThreadPoolExecutor already gave."""
        expected = min(32, (os.cpu_count() or 1) + 4)

        assert _default_max_concurrency() == expected

    def test_default_is_forwarded_to_the_batch_executor(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"id": ["v1", "v2"], "values": [[0.1], [0.2]]})
        mock_channel.upsert.return_value = {"upserted_count": 1}

        with patch.object(
            GrpcIndex,
            "_get_batch_executor",
            autospec=True,
            side_effect=GrpcIndex._get_batch_executor,
        ) as spy:
            grpc_index.upsert_from_dataframe(df, batch_size=1, show_progress=False)

        assert spy.call_args[0][1] == _default_max_concurrency()

    def test_explicit_max_concurrency_is_forwarded(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"id": ["v1", "v2"], "values": [[0.1], [0.2]]})
        mock_channel.upsert.return_value = {"upserted_count": 1}

        with patch.object(
            GrpcIndex,
            "_get_batch_executor",
            autospec=True,
            side_effect=GrpcIndex._get_batch_executor,
        ) as spy:
            grpc_index.upsert_from_dataframe(
                df, batch_size=1, show_progress=False, max_concurrency=3
            )

        assert spy.call_args[0][1] == 3

    @pytest.mark.parametrize("bad", [0, -1, 65])
    def test_out_of_range_max_concurrency_raises(self, grpc_index: GrpcIndex, bad: int) -> None:
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"id": ["v1"], "values": [[0.1]]})

        with pytest.raises(PineconeValueError, match="max_concurrency"):
            grpc_index.upsert_from_dataframe(df, max_concurrency=bad, show_progress=False)

    def test_in_flight_batches_never_exceed_max_concurrency(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        """The starvation fix: 2000 batches no longer all queue at once."""
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame(
            {"id": [f"v{i}" for i in range(60)], "values": [[float(i)] for i in range(60)]}
        )

        lock = threading.Lock()
        inflight = 0
        peak = 0

        def _tracking_upsert(vectors: list[dict[str, object]], namespace: str | None, **_: object):
            nonlocal inflight, peak
            with lock:
                inflight += 1
                peak = max(peak, inflight)
            try:
                return {"upserted_count": len(vectors)}
            finally:
                with lock:
                    inflight -= 1

        mock_channel.upsert.side_effect = _tracking_upsert

        grpc_index.upsert_from_dataframe(df, batch_size=1, max_concurrency=4, show_progress=False)

        assert peak <= 4, f"{peak} batches were in flight with max_concurrency=4"

    def test_max_concurrency_is_keyword_only(self) -> None:
        """Positional df is what keeps this method out of the keyword-only guard."""
        sig = inspect.signature(GrpcIndex.upsert_from_dataframe)

        assert sig.parameters["max_concurrency"].kind is inspect.Parameter.KEYWORD_ONLY
        assert sig.parameters["df"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD


class TestBatchExecutorIsNotSharedAcrossSizes:
    """upsert() defaults to 4 workers, upsert_from_dataframe() to min(32, cpu+4)."""

    def test_different_sizes_get_different_pools(self, grpc_index: GrpcIndex) -> None:
        four = grpc_index._get_batch_executor(4)
        sixteen = grpc_index._get_batch_executor(16)

        assert four is not sixteen

    def test_same_size_reuses_one_pool(self, grpc_index: GrpcIndex) -> None:
        assert grpc_index._get_batch_executor(8) is grpc_index._get_batch_executor(8)

    def test_asking_for_another_size_does_not_shut_the_first_down(
        self, grpc_index: GrpcIndex
    ) -> None:
        """Shutting it down mid-submit raises "cannot schedule new futures"."""
        four = grpc_index._get_batch_executor(4)
        grpc_index._get_batch_executor(16)

        assert four.submit(lambda: "still usable").result() == "still usable"

    def test_close_shuts_down_every_pool(self, grpc_index: GrpcIndex) -> None:
        pools = [grpc_index._get_batch_executor(n) for n in (2, 4, 8)]

        grpc_index.close()

        for pool in pools:
            with pytest.raises(RuntimeError):
                pool.submit(lambda: None)


class TestGrpcDataframeDoesNotStarveTheSharedExecutor:
    def test_batches_do_not_queue_on_the_shared_async_executor(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        """A 2000-batch frame used to queue 2000 tasks ahead of every *_async call."""
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame(
            {"id": [f"v{i}" for i in range(10)], "values": [[float(i)] for i in range(10)]}
        )
        mock_channel.upsert.return_value = {"upserted_count": 1}

        shared = grpc_index._executor
        with patch.object(shared, "submit", wraps=shared.submit) as shared_submit:
            grpc_index.upsert_from_dataframe(df, batch_size=1, show_progress=False)

        assert shared_submit.call_count == 0


class TestGrpcDataframeProgressTracksCompletion:
    def test_bar_advances_once_per_completed_batch(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        """The bar used to wrap submission, so it filled before the ingest ran."""
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame(
            {"id": [f"v{i}" for i in range(6)], "values": [[float(i)] for i in range(6)]}
        )
        mock_channel.upsert.return_value = {"upserted_count": 2}

        updates: list[int] = []
        completed_at_update: list[int] = []

        class _RecordingBar:
            def update(self, n: int = 1) -> None:
                updates.append(n)
                completed_at_update.append(len(mock_channel.upsert.call_args_list))

            def set_postfix_str(self, s: str) -> None:
                pass

            def close(self) -> None:
                pass

        with patch("pinecone._internal.batch._create_progress_bar", return_value=_RecordingBar()):
            grpc_index.upsert_from_dataframe(df, batch_size=2, show_progress=True)

        assert sum(updates) == 3
        # Each update lands after a batch actually reached the channel, never before.
        assert all(seen >= i + 1 for i, seen in enumerate(completed_at_update))
