"""Engine behavior: result-contract parity with batch_execute, plus the
failure paths that motivated the rewrite — each reproduced bug from the old
implementation appears here as a regression test."""

from __future__ import annotations

import threading
import time
from typing import Any
from unittest.mock import patch

from pinecone._internal.bulk.core import STALL_CONSECUTIVE_FAILURES
from pinecone._internal.bulk.engine import bulk_execute_sync
from pinecone._internal.bulk.registry import get_registry

HOST = "engine-test.svc.pinecone.io"


def _items(n: int) -> list[dict[str, Any]]:
    return [{"id": str(i)} for i in range(n)]


def test_all_success_counts() -> None:
    result = bulk_execute_sync(
        items=_items(25),
        operation=lambda batch: {"upserted_count": len(batch)},
        batch_size=10,
        max_concurrency=4,
        show_progress=False,
        host=HOST,
    )
    assert result.successful_item_count == 25
    assert result.failed_item_count == 0
    assert result.total_batch_count == 3
    assert result.successful_batch_count == 3
    assert not result.timed_out


def test_operation_error_is_collected_not_raised() -> None:
    def op(batch: list[dict[str, Any]]) -> Any:
        if batch[0]["id"] == "10":
            raise RuntimeError("boom on batch 1")
        return {"upserted_count": len(batch)}

    result = bulk_execute_sync(
        items=_items(30),
        operation=op,
        batch_size=10,
        max_concurrency=2,
        show_progress=False,
        host=HOST,
    )
    assert result.successful_item_count == 20
    assert result.failed_item_count == 10
    assert [e.batch_index for e in result.errors] == [1]
    assert "boom" in result.errors[0].error_message


def test_submit_raising_releases_the_slot_and_later_calls_proceed() -> None:
    """THE reproduced deadlock: acquire before submit, submit raises, the
    release lived in a finally that never ran — with the limit floored, every
    later acquire hung forever. Now the slot returns in the except path and
    the gate is quiescent afterwards."""
    gate = get_registry().get(HOST)
    while gate.limit > 1:
        gate.report_throttled()

    calls = {"n": 0}

    from concurrent.futures import ThreadPoolExecutor

    real_submit = ThreadPoolExecutor.submit

    def flaky_submit(self: ThreadPoolExecutor, fn: Any, *args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("cannot schedule new futures after shutdown")
        return real_submit(self, fn, *args, **kwargs)

    with patch.object(ThreadPoolExecutor, "submit", flaky_submit):
        result = bulk_execute_sync(
            items=_items(6),
            operation=lambda batch: {"upserted_count": len(batch)},
            batch_size=2,
            max_concurrency=2,
            show_progress=False,
            host=HOST,
            total_timeout=30,
        )

    assert result.failed_item_count == 2
    assert result.successful_item_count == 4
    assert gate.quiescent(), "slot leaked on the submit-raise path"

    followup = bulk_execute_sync(
        items=_items(4),
        operation=lambda batch: {"upserted_count": len(batch)},
        batch_size=2,
        max_concurrency=2,
        show_progress=False,
        host=HOST,
        total_timeout=10,
    )
    assert followup.successful_item_count == 4
    assert not followup.timed_out, "a leaked slot would have starved this call"


def test_deadline_during_gate_wait_abandons_unsent() -> None:
    gate = get_registry().get(HOST)
    while gate.limit > 1:
        gate.report_throttled()

    started = threading.Event()

    def slow(batch: list[dict[str, Any]]) -> Any:
        started.set()
        time.sleep(0.4)
        return {"upserted_count": len(batch)}

    result = bulk_execute_sync(
        items=_items(8),
        operation=slow,
        batch_size=2,
        max_concurrency=4,
        show_progress=False,
        host=HOST,
        total_timeout=0.2,
    )
    assert result.timed_out
    assert result.successful_item_count + result.failed_item_count == 8
    assert result.failed_item_count >= 4, "unsent batches must land in errors"
    assert gate.quiescent()


def test_stall_abandons_remainder_loudly() -> None:
    gate = get_registry().get(HOST)
    while gate.limit > 1:
        gate.report_throttled()

    calls = {"n": 0}

    def dying(batch: list[dict[str, Any]]) -> Any:
        calls["n"] += 1
        raise RuntimeError("UNAVAILABLE: backend gone")

    result = bulk_execute_sync(
        items=_items(40),
        operation=dying,
        batch_size=2,
        max_concurrency=4,
        show_progress=False,
        host=HOST,
    )
    assert calls["n"] <= STALL_CONSECUTIVE_FAILURES + 2, (
        f"kept sending after the backend was clearly dead: {calls['n']} calls"
    )
    assert result.failed_item_count == 40
    assert not result.timed_out
    stall_errors = [e for e in result.errors if "backend appears unavailable" in e.error_message]
    assert stall_errors, "abandoned batches must say why"
    assert result.total_batch_count == len(result.errors)
    assert gate.quiescent()


def test_peak_concurrency_respects_both_bounds() -> None:
    gate = get_registry().get(HOST)
    gate.report_throttled()

    peak = 0
    current = 0
    lock = threading.Lock()

    def observing(batch: list[dict[str, Any]]) -> Any:
        nonlocal peak, current
        with lock:
            current += 1
            peak = max(peak, current)
        time.sleep(0.02)
        with lock:
            current -= 1
        return {"upserted_count": len(batch)}

    peak_limit = gate.limit
    bulk_execute_sync(
        items=_items(40),
        operation=observing,
        batch_size=2,
        max_concurrency=8,
        show_progress=False,
        host=HOST,
    )
    assert peak <= max(peak_limit, gate.limit), f"peak {peak} exceeded the gate trajectory"


def test_empty_items_short_circuits() -> None:
    result = bulk_execute_sync(
        items=[],
        operation=lambda b: None,
        batch_size=10,
        show_progress=False,
        host=HOST,
    )
    assert result.total_item_count == 0
    assert result.total_batch_count == 0


def test_dispositions_and_counters_on_mixed_outcomes() -> None:
    gate = get_registry().get(HOST)

    def op(batch: list[dict[str, Any]]) -> Any:
        if batch[0]["id"] == "0":
            from pinecone.errors.exceptions import PineconeValueError

            raise PineconeValueError("dimension mismatch")
        if batch[0]["id"] == "2":
            gate.report_throttled()
        return {"upserted_count": len(batch)}

    result = bulk_execute_sync(
        items=_items(12),
        operation=op,
        batch_size=2,
        max_concurrency=4,
        show_progress=False,
        host=HOST,
    )
    assert result.successful_item_count == 10
    poison = [e for e in result.errors if not e.retryable]
    assert len(poison) == 1 and poison[0].disposition == "rejected"
    assert result.throttle_event_count == 1, "the in-window throttle must be counted"
    assert result.final_limit == gate.limit
    assert 1 <= result.peak_inflight <= 4


def test_stall_errors_carry_abandoned_disposition() -> None:
    gate = get_registry().get(HOST)
    while gate.limit > 1:
        gate.report_throttled()

    def dying(batch: list[dict[str, Any]]) -> Any:
        raise RuntimeError("UNAVAILABLE")

    result = bulk_execute_sync(
        items=_items(20),
        operation=dying,
        batch_size=2,
        max_concurrency=2,
        show_progress=False,
        host=HOST,
    )
    abandoned = [e for e in result.errors if e.disposition == "abandoned"]
    rejected = [e for e in result.errors if e.disposition == "rejected"]
    assert abandoned, "stall-abandoned batches must be labeled"
    assert rejected, "the batches that actually failed stay rejected"
    assert all(e.retryable for e in abandoned)


def test_deadline_unsent_batches_carry_unsent_disposition() -> None:
    import threading

    gate = get_registry().get(HOST)
    while gate.limit > 1:
        gate.report_throttled()
    release = threading.Event()

    def slow(batch: list[dict[str, Any]]) -> Any:
        release.wait(2.0)
        return {"upserted_count": len(batch)}

    result = bulk_execute_sync(
        items=_items(8),
        operation=slow,
        batch_size=2,
        max_concurrency=4,
        show_progress=False,
        host=HOST,
        total_timeout=0.2,
    )
    release.set()
    unsent = [e for e in result.errors if e.disposition == "unsent"]
    assert unsent, "deadline-expired batches must be labeled unsent"
    assert all(e.retryable for e in unsent)
