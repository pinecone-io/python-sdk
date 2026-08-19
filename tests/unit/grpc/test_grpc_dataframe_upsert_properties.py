"""Property-based characterization of GrpcIndex.upsert_from_dataframe().

Two kinds of tests live here:

* Invariant / metamorphic properties that hold regardless of how partial
  failures are eventually handled — batching is a valid partition, every input
  row reaches the channel exactly once with its payload intact, the upserted
  set is invariant under batch_size, and timeout is threaded to every batch.

* Characterization tests that pin the CURRENT all-or-nothing-raise behavior on
  partial failure. That behavior diverges from upsert(batch_size=...), which
  aggregates per-batch failures without raising, and is slated to change in
  issue #26 — these tests must be revisited when it does.

Assertions inspect the arguments handed to the (mocked) channel, which
originate from the input DataFrame, rather than the mock's synthesized return
value — so they test the batching/marshalling logic, not the mock.
"""

from __future__ import annotations

import math
from collections import Counter
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pinecone.errors.exceptions import PineconeTimeoutError
from pinecone.grpc import GrpcIndex

_MOCK_GRPC_MODULE_PATH = "pinecone._grpc"


def _make_grpc_index(mock_channel: MagicMock) -> GrpcIndex:
    mock_module = MagicMock()
    mock_module.GrpcChannel.return_value = mock_channel
    with patch.dict("sys.modules", {_MOCK_GRPC_MODULE_PATH: mock_module}):
        return GrpcIndex(host="test-index-abc123.svc.pinecone.io", api_key="test-api-key")


def _count_by_batch_channel() -> MagicMock:
    ch = MagicMock()
    ch.upsert.side_effect = lambda vectors, namespace, timeout_s=None: {
        "upserted_count": len(vectors)
    }
    return ch


def _make_df(pd, n_rows: int):
    return pd.DataFrame(
        {"id": [f"v{i}" for i in range(n_rows)], "values": [[float(i)] for i in range(n_rows)]}
    )


def _upsert_call_count(ch: MagicMock) -> int:
    """Number of batches that reached the channel.

    `upsert_from_dataframe` submits every batch concurrently, and Mock only
    updates `call_count` and `call_args_list` atomically on Python 3.13+. On
    3.10-3.12 a concurrent `call_count += 1` loses increments, so `call_count`
    undercounts. `list.append` is atomic on every version, which makes the
    length of `call_args_list` the reliable count.
    """
    return len(ch.upsert.call_args_list)


def _channel_vectors(ch: MagicMock):
    """Every vector dict handed to channel.upsert, flattened across batches."""
    return [v for c in ch.upsert.call_args_list for v in c.args[0]]


def _side_effect_factory(batch_size: int, failing: frozenset[int], exc_factory):
    """Per-batch side effect keyed on the batch's own contents.

    A list-style side_effect is racy here: batches run concurrently, so
    deriving the batch index from the first vector's id assigns failures
    deterministically regardless of execution order.
    """

    def _upsert(chunk, *args, **kwargs):
        batch_idx = int(chunk[0]["id"][1:]) // batch_size
        if batch_idx in failing:
            raise exc_factory(batch_idx)
        return {"upserted_count": len(chunk)}

    return _upsert


# Integer-valued floats survive any float32 marshalling exactly, so payload
# equality checks stay deterministic.
_int_float = st.integers(min_value=-1000, max_value=1000).map(float)
_values = st.lists(_int_float, min_size=1, max_size=6)
_metadata = st.dictionaries(
    keys=st.sampled_from(["genre", "year", "topic", "rank"]),
    values=st.one_of(st.integers(-100, 100), st.text(max_size=5)),
    max_size=3,
)


@st.composite
def _payloads(draw):
    values = draw(st.lists(_values, min_size=1, max_size=25))
    n = len(values)
    metas = draw(st.lists(st.none() | _metadata, min_size=n, max_size=n))
    batch_size = draw(st.integers(min_value=1, max_value=10))
    return values, metas, batch_size


def _payload_df(pd, values, metas):
    return pd.DataFrame(
        {
            "id": [f"v{i}" for i in range(len(values))],
            "values": values,
            "metadata": metas,
        }
    )


@st.composite
def _cases(draw):
    n_rows = draw(st.integers(min_value=1, max_value=40))
    batch_size = draw(st.integers(min_value=1, max_value=15))
    n_batches = math.ceil(n_rows / batch_size)
    failing = draw(st.sets(st.integers(min_value=0, max_value=n_batches - 1), max_size=n_batches))
    return n_rows, batch_size, frozenset(failing), n_batches


class TestUpsertFromDataframePartitionProperties:
    """Batching covers every row exactly once, with payloads intact."""

    @settings(max_examples=200, deadline=None)
    @given(
        n_rows=st.integers(min_value=0, max_value=3000),
        batch_size=st.integers(min_value=1, max_value=750),
    )
    def test_batches_form_a_valid_partition(self, n_rows: int, batch_size: int) -> None:
        pd = pytest.importorskip("pandas")
        ch = _count_by_batch_channel()
        with _make_grpc_index(ch) as idx:
            result = idx.upsert_from_dataframe(
                _make_df(pd, n_rows), batch_size=batch_size, show_progress=False
            )

        sizes = [len(c.args[0]) for c in ch.upsert.call_args_list]
        assert _upsert_call_count(ch) == (math.ceil(n_rows / batch_size) if n_rows else 0)
        assert all(0 < s <= batch_size for s in sizes)
        assert len([s for s in sizes if s != batch_size]) <= 1
        # Coverage, not just cardinality: every input id reaches the channel
        # exactly once (no drops, dups, or corruption).
        assert Counter(v["id"] for v in _channel_vectors(ch)) == Counter(
            f"v{i}" for i in range(n_rows)
        )
        assert result.upserted_count == n_rows

    @settings(max_examples=150, deadline=None)
    @given(payload=_payloads())
    def test_every_row_reaches_channel_with_payload_intact(self, payload) -> None:
        pd = pytest.importorskip("pandas")
        values, metas, batch_size = payload
        ch = _count_by_batch_channel()
        with _make_grpc_index(ch) as idx:
            idx.upsert_from_dataframe(
                _payload_df(pd, values, metas), batch_size=batch_size, show_progress=False
            )

        got = {v["id"]: (tuple(v["values"]), v.get("metadata")) for v in _channel_vectors(ch)}
        expected = {f"v{i}": (tuple(values[i]), metas[i]) for i in range(len(values))}
        assert got == expected

    @settings(max_examples=100, deadline=None)
    @given(payload=_payloads(), other_batch_size=st.integers(min_value=1, max_value=10))
    def test_upserted_set_is_invariant_under_batch_size(
        self, payload, other_batch_size: int
    ) -> None:
        """Metamorphic: regrouping the same rows into different batch sizes
        does not change the (id, values) multiset that reaches the channel."""
        pd = pytest.importorskip("pandas")
        values, metas, batch_size = payload
        df = _payload_df(pd, values, metas)

        def upserted(bs: int) -> Counter:
            ch = _count_by_batch_channel()
            with _make_grpc_index(ch) as idx:
                idx.upsert_from_dataframe(df, batch_size=bs, show_progress=False)
            return Counter((v["id"], tuple(v["values"])) for v in _channel_vectors(ch))

        assert upserted(batch_size) == upserted(other_batch_size)

    @settings(max_examples=100, deadline=None)
    @given(
        timeout=st.one_of(
            st.none(),
            st.floats(min_value=0, max_value=600, allow_nan=False, allow_infinity=False),
        ),
        n_rows=st.integers(min_value=1, max_value=20),
        batch_size=st.integers(min_value=1, max_value=10),
    )
    def test_timeout_threaded_to_every_batch(self, timeout, n_rows: int, batch_size: int) -> None:
        pd = pytest.importorskip("pandas")
        ch = _count_by_batch_channel()
        with _make_grpc_index(ch) as idx:
            idx.upsert_from_dataframe(
                _make_df(pd, n_rows), batch_size=batch_size, timeout=timeout, show_progress=False
            )

        assert _upsert_call_count(ch) >= 1
        for c in ch.upsert.call_args_list:
            assert c.kwargs["timeout_s"] == timeout


class TestUpsertFromDataframePartialFailureProperties:
    """Pins the current all-or-nothing-raise contract (see issue #26)."""

    @settings(max_examples=50, deadline=None)
    @given(case=_cases())
    def test_returns_full_count_iff_no_batch_fails(self, case) -> None:
        pd = pytest.importorskip("pandas")
        n_rows, batch_size, failing, _ = case
        ch = MagicMock()
        ch.upsert.side_effect = _side_effect_factory(
            batch_size, failing, lambda i: RuntimeError(f"batch {i} boom")
        )
        with _make_grpc_index(ch) as idx:
            df = _make_df(pd, n_rows)
            if not failing:
                result = idx.upsert_from_dataframe(df, batch_size=batch_size, show_progress=False)
                assert result.upserted_count == n_rows
            else:
                with pytest.raises(RuntimeError):
                    idx.upsert_from_dataframe(df, batch_size=batch_size, show_progress=False)

    @settings(max_examples=50, deadline=None)
    @given(case=_cases())
    def test_original_exception_propagates_unwrapped(self, case) -> None:
        pd = pytest.importorskip("pandas")
        n_rows, batch_size, failing, _ = case
        if not failing:
            return
        ch = MagicMock()
        ch.upsert.side_effect = _side_effect_factory(
            batch_size, failing, lambda i: PineconeTimeoutError("deadline exceeded")
        )
        with _make_grpc_index(ch) as idx:
            with pytest.raises(PineconeTimeoutError):
                idx.upsert_from_dataframe(
                    _make_df(pd, n_rows), batch_size=batch_size, show_progress=False
                )

    @settings(max_examples=40, deadline=None)
    @given(case=_cases())
    def test_failed_batches_are_not_cancelled(self, case) -> None:
        pd = pytest.importorskip("pandas")
        n_rows, batch_size, failing, n_batches = case
        if not failing:
            return
        ch = MagicMock()
        ch.upsert.side_effect = _side_effect_factory(
            batch_size, failing, lambda i: RuntimeError(f"batch {i} boom")
        )
        with _make_grpc_index(ch) as idx:
            with pytest.raises(RuntimeError):
                idx.upsert_from_dataframe(
                    _make_df(pd, n_rows), batch_size=batch_size, show_progress=False
                )

            idx._executor.shutdown(wait=True)
            assert _upsert_call_count(ch) == n_batches

    def test_earlier_successes_are_discarded_on_later_failure(self) -> None:
        pd = pytest.importorskip("pandas")
        ch = MagicMock()
        ch.upsert.side_effect = _side_effect_factory(
            batch_size=2, failing=frozenset({1}), exc_factory=lambda i: RuntimeError("boom")
        )
        with _make_grpc_index(ch) as idx:
            with pytest.raises(RuntimeError):
                idx.upsert_from_dataframe(_make_df(pd, 6), batch_size=2, show_progress=False)
