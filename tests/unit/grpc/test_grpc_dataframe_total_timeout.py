"""total_timeout — a deadline for the whole ingest, not for one attempt.

`timeout` bounds a single attempt of a single batch. Nothing bounded the job,
so "fail my pipeline if this hasn't finished in N minutes" had no expression at
any layer.

The fakes here block on `threading.Event` rather than sleeping: the unit
conftest patches `time.sleep` through the module object, which no-ops it for the
whole suite.
"""

from __future__ import annotations

import contextlib
import inspect
import threading
from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from pinecone.errors.exceptions import PineconeTimeoutError
from pinecone.grpc import GrpcIndex

pd = pytest.importorskip("pandas")

_MOCK_GRPC_MODULE_PATH = "pinecone._grpc"

# The expiring deadline every stalling test hands to the ingest.
# `_released_after_expiry` times itself against this, so the two cannot drift:
# a release timed against a different number either lands before the deadline
# (nothing expires) or falls through to `_StallingChannel`'s 2.0s backstop.
_TOTAL_TIMEOUT = 0.05

# How long past the stall `_released_after_expiry` keeps the fake blocked, on
# top of `_TOTAL_TIMEOUT`. It only has to cover the ingest waking from its own
# expired `as_completed` timeout and cancelling the queued batches, which is
# sub-millisecond work.
_RELEASE_MARGIN = 0.2

# Measured 0.26s per stalling test — `_TOTAL_TIMEOUT` + `_RELEASE_MARGIN` plus
# overhead. 15s is deliberate headroom, not a multiple of that: a releaser
# thread that never sees the stall waits 5s and the stalled `upsert` then pays
# its 2.0s backstop, so a degraded run legitimately costs ~7s and the global 5s
# unit default would misreport it as a hang. Class-scoped rather than
# module-level: nothing here is order-dependent, so the four tests that never
# stall keep the 5s guard.
_DEADLINE_TIMEOUT = pytest.mark.timeout(15)


def _grpc_index(mock_channel: MagicMock) -> GrpcIndex:
    mock_module = MagicMock()
    mock_module.GrpcChannel.return_value = mock_channel
    with patch.dict("sys.modules", {_MOCK_GRPC_MODULE_PATH: mock_module}):
        return GrpcIndex(host="test-index-abc123.svc.pinecone.io", api_key="test-api-key")


def _frame(rows: int) -> Any:
    return pd.DataFrame(
        {"id": [f"v{i}" for i in range(rows)], "values": [[float(i)] for i in range(rows)]}
    )


class _StallingChannel:
    """Accepts a fixed number of batches, then blocks past the deadline.

    Each accepted batch records what it was handed, so a test can assert the
    reported counts match what the fake actually took.
    """

    def __init__(self, accept: int) -> None:
        self.accepted: list[list[dict[str, Any]]] = []
        self.stalled = threading.Event()
        self.released = threading.Event()
        self._accept = accept
        self._lock = threading.Lock()

    def upsert(
        self, vectors: list[dict[str, Any]], namespace: str | None, **_: Any
    ) -> dict[str, Any]:
        with self._lock:
            index = len(self.accepted)
            if index < self._accept:
                self.accepted.append(vectors)
                return {"upserted_count": len(vectors)}
        self.stalled.set()
        self.released.wait(timeout=2)
        with self._lock:
            self.accepted.append(vectors)
        return {"upserted_count": len(vectors)}


@contextlib.contextmanager
def _released_after_expiry(channel: _StallingChannel) -> Iterator[None]:
    """Unblock the stalled batch, but not before the ingest's deadline has gone.

    A test cannot release the fake itself: `upsert_from_dataframe` does not
    return until the in-flight batch settles, and that batch is the one blocked
    in `_StallingChannel.upsert`, so any release written after the call has
    already been overtaken by the 2.0s backstop.

    Releasing the instant the stall is observed would be too early — the batch
    would land inside the budget and nothing would expire. Waiting
    `_TOTAL_TIMEOUT` + `_RELEASE_MARGIN` past the stall puts the release
    strictly after the deadline, since the deadline starts before the stall,
    which is what keeps expiry genuinely happening with a batch in flight.

    `Event.wait` rather than `time.sleep`, for the reason given in the module
    docstring.
    """

    def _release_once_expired() -> None:
        if channel.stalled.wait(timeout=5):
            channel.released.wait(timeout=_TOTAL_TIMEOUT + _RELEASE_MARGIN)
        channel.released.set()

    releaser = threading.Thread(target=_release_once_expired)
    releaser.start()
    try:
        yield
    finally:
        channel.released.set()
        releaser.join(timeout=5)


class TestSignature:
    def test_total_timeout_is_keyword_only_and_defaults_to_none(self) -> None:
        sig = inspect.signature(GrpcIndex.upsert_from_dataframe)
        param = sig.parameters["total_timeout"]

        assert param.kind is inspect.Parameter.KEYWORD_ONLY
        assert param.default is None

    def test_no_total_timeout_means_no_deadline(self) -> None:
        mock_channel = MagicMock()
        mock_channel.upsert.return_value = {"upserted_count": 1}
        index = _grpc_index(mock_channel)

        result = index.upsert_from_dataframe(_frame(4), batch_size=1, show_progress=False)

        assert result.upserted_count == 4


@_DEADLINE_TIMEOUT
class TestExpiry:
    def test_expiry_raises_with_the_partial_result_attached(self) -> None:
        channel = _StallingChannel(accept=2)
        index = _grpc_index(MagicMock(upsert=channel.upsert))

        with _released_after_expiry(channel), pytest.raises(PineconeTimeoutError) as excinfo:
            index.upsert_from_dataframe(
                _frame(6),
                batch_size=1,
                show_progress=False,
                max_concurrency=1,
                total_timeout=_TOTAL_TIMEOUT,
                on_error="raise",
            )

        response = excinfo.value.response
        assert response is not None
        # Two batches were taken outright and a third was in flight when the
        # deadline fired; all three are allowed to settle and are counted.
        assert response.upserted_count == 3
        assert response.failed_item_count == 3
        assert response.upserted_count + response.failed_item_count == 6
        assert "total_timeout" in str(excinfo.value)

    def test_unsent_rows_are_reported_for_retry(self) -> None:
        channel = _StallingChannel(accept=2)
        index = _grpc_index(MagicMock(upsert=channel.upsert))

        with _released_after_expiry(channel), pytest.raises(PineconeTimeoutError) as excinfo:
            index.upsert_from_dataframe(
                _frame(6),
                batch_size=1,
                show_progress=False,
                max_concurrency=1,
                total_timeout=_TOTAL_TIMEOUT,
                on_error="raise",
            )

        unsent = {item["id"] for item in excinfo.value.response.failed_items}
        assert unsent == {"v3", "v4", "v5"}

    def test_reported_counts_match_what_the_fake_accepted(self) -> None:
        channel = _StallingChannel(accept=3)
        index = _grpc_index(MagicMock(upsert=channel.upsert))

        with _released_after_expiry(channel), pytest.raises(PineconeTimeoutError) as excinfo:
            index.upsert_from_dataframe(
                _frame(8),
                batch_size=1,
                show_progress=False,
                max_concurrency=1,
                total_timeout=_TOTAL_TIMEOUT,
                on_error="raise",
            )

        assert excinfo.value.response.upserted_count == len(channel.accepted)

    def test_the_in_flight_batch_settles_and_the_queued_ones_are_never_sent(self) -> None:
        """Dropping a running batch client-side would not stop the server applying it.

        So the one in flight at expiry is awaited and counted, while the ones
        still queued behind it are cancelled before they can reach the channel.
        """
        channel = _StallingChannel(accept=2)
        index = _grpc_index(MagicMock(upsert=channel.upsert))

        with _released_after_expiry(channel), pytest.raises(PineconeTimeoutError) as excinfo:
            index.upsert_from_dataframe(
                _frame(6),
                batch_size=1,
                show_progress=False,
                max_concurrency=1,
                total_timeout=_TOTAL_TIMEOUT,
                on_error="raise",
            )

        assert channel.stalled.is_set(), "no batch was in flight when the deadline fired"

        sent = {item["id"] for batch in channel.accepted for item in batch}
        assert sent == {"v0", "v1", "v2"}
        assert excinfo.value.response.upserted_count == len(sent)


class TestNothingLeftToRetry:
    """An elapsed deadline with everything landed is not a failure."""

    def test_no_timeout_when_the_in_flight_batches_were_the_last_ones(self) -> None:
        """Raising here would hand the caller an empty set of items to retry.

        Easy to hit whenever the batch count is at most max_concurrency, which
        the default makes common for modest frames.
        """
        channel = _StallingChannel(accept=0)
        index = _grpc_index(MagicMock(upsert=channel.upsert))

        def _release_once_stalled() -> None:
            channel.stalled.wait(timeout=5)
            channel.released.set()

        releaser = threading.Thread(target=_release_once_stalled)
        releaser.start()
        try:
            result = index.upsert_from_dataframe(
                _frame(4),
                batch_size=1,
                show_progress=False,
                max_concurrency=4,
                total_timeout=0.05,
            )
        finally:
            channel.released.set()
            releaser.join(timeout=5)

        assert result.upserted_count == 4
        assert result.failed_item_count == 0
        assert result.failed_items == []


@_DEADLINE_TIMEOUT
class TestExpiryUnderCollect:
    """Expiry follows on_error, like any other partial failure."""

    def test_default_returns_the_partial_result_instead_of_raising(self) -> None:
        channel = _StallingChannel(accept=2)
        index = _grpc_index(MagicMock(upsert=channel.upsert))

        with _released_after_expiry(channel):
            response = index.upsert_from_dataframe(
                _frame(6),
                batch_size=1,
                show_progress=False,
                max_concurrency=1,
                total_timeout=_TOTAL_TIMEOUT,
            )

        assert response.upserted_count == 3
        assert {item["id"] for item in response.failed_items} == {"v3", "v4", "v5"}


class TestGenerousDeadline:
    def test_a_deadline_that_does_not_expire_changes_nothing(self) -> None:
        mock_channel = MagicMock()
        mock_channel.upsert.return_value = {"upserted_count": 2}
        index = _grpc_index(MagicMock(upsert=mock_channel.upsert))

        result = index.upsert_from_dataframe(
            _frame(6), batch_size=2, show_progress=False, total_timeout=30.0
        )

        assert result.upserted_count == 6
        assert result.failed_item_count == 0


@_DEADLINE_TIMEOUT
class TestDeadlineDuringLimiterWait:
    """The budget has to bound the wait for a concurrency slot, not just the work.

    With a limiter wired, submission blocks until the limiter allows another
    in-flight batch. A deadline checked only before that wait can be overrun by
    it, and the batch would then be submitted anyway.
    """

    def test_expiry_while_waiting_for_a_slot_stops_submission(self) -> None:
        from pinecone._internal.adaptive import _AdaptiveLimiterRegistry

        registry = _AdaptiveLimiterRegistry()
        host = "test-index-abc123.svc.pinecone.io"
        limiter = registry.get(host, 1)
        for _ in range(4):
            limiter.report_throttled()
        assert limiter.current_limit() == 1

        channel = _StallingChannel(accept=0)
        mock_module = MagicMock()
        mock_module.GrpcChannel.return_value = MagicMock(upsert=channel.upsert)
        with patch.dict("sys.modules", {_MOCK_GRPC_MODULE_PATH: mock_module}):
            index = GrpcIndex(
                host=host,
                api_key="test-api-key",
                on_throttle=registry.report_throttled,
                limiter_registry=registry,
            )

        with _released_after_expiry(channel), pytest.raises(PineconeTimeoutError) as excinfo:
            index.upsert_from_dataframe(
                _frame(6),
                batch_size=1,
                show_progress=False,
                max_concurrency=1,
                total_timeout=_TOTAL_TIMEOUT,
                on_error="raise",
            )

        response = excinfo.value.response
        assert response.failed_item_count >= 4, "submission kept going after the budget expired"
        assert len(channel.accepted) <= 2
