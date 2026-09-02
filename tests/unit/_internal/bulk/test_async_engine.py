"""Async-engine behavior: result-contract parity with the sync engine, plus
the async-specific fault paths — cancellation mid-drive, deadline during the
gate wait, stall, and the counters/dispositions the sync twin guarantees."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from pinecone._internal.bulk.async_engine import bulk_execute_async
from pinecone._internal.bulk.core import STALL_CONSECUTIVE_FAILURES
from pinecone._internal.bulk.registry import get_registry

HOST = "async-engine-test.svc.pinecone.io"


def _items(n: int) -> list[dict[str, Any]]:
    return [{"id": str(i)} for i in range(n)]


async def _ok(batch: list[dict[str, Any]]) -> Any:
    return {"upserted_count": len(batch)}


def test_all_success_counts_and_parity_fields() -> None:
    async def run() -> Any:
        return await bulk_execute_async(
            items=_items(25),
            operation=_ok,
            batch_size=10,
            max_concurrency=4,
            show_progress=False,
            host=HOST,
        )

    result = asyncio.run(run())
    assert result.successful_item_count == 25
    assert result.total_batch_count == 3
    assert not result.timed_out
    assert result.final_limit is not None
    assert 1 <= result.peak_inflight <= 3
    assert get_registry().get(HOST).quiescent()


def test_errors_sorted_and_classified() -> None:
    from pinecone.errors.exceptions import PineconeValueError

    async def op(batch: list[dict[str, Any]]) -> Any:
        if batch[0]["id"] == "4":
            raise RuntimeError("transient blip")
        if batch[0]["id"] == "0":
            raise PineconeValueError("dimension mismatch")
        return {"upserted_count": len(batch)}

    async def run() -> Any:
        return await bulk_execute_async(
            items=_items(12),
            operation=op,
            batch_size=2,
            max_concurrency=3,
            show_progress=False,
            host=HOST,
        )

    result = asyncio.run(run())
    assert [e.batch_index for e in result.errors] == [0, 2]
    by_idx = {e.batch_index: e for e in result.errors}
    assert by_idx[0].retryable is False
    assert by_idx[2].retryable is True
    assert all(e.disposition == "rejected" for e in result.errors)


def test_peak_concurrency_respects_gate_limit() -> None:
    gate = get_registry().get(HOST)
    gate.report_throttled()

    peak = 0
    current = 0

    async def observing(batch: list[dict[str, Any]]) -> Any:
        nonlocal peak, current
        current += 1
        peak = max(peak, current)
        await asyncio.sleep(0.01)
        current -= 1
        return {"upserted_count": len(batch)}

    limit_before = gate.limit

    async def run() -> Any:
        return await bulk_execute_async(
            items=_items(30),
            operation=observing,
            batch_size=2,
            max_concurrency=8,
            show_progress=False,
            host=HOST,
        )

    asyncio.run(run())
    assert peak <= max(limit_before, gate.limit)
    assert gate.quiescent()


def test_total_timeout_abandons_unsent_and_never_cancels_inflight() -> None:
    gate = get_registry().get(HOST)
    while gate.limit > 1:
        gate.report_throttled()

    completed: list[int] = []

    async def slow(batch: list[dict[str, Any]]) -> Any:
        await asyncio.sleep(0.4)
        completed.append(int(batch[0]["id"]))
        return {"upserted_count": len(batch)}

    async def run() -> Any:
        return await bulk_execute_async(
            items=_items(8),
            operation=slow,
            batch_size=2,
            max_concurrency=4,
            show_progress=False,
            host=HOST,
            total_timeout=0.2,
        )

    result = asyncio.run(run())
    assert result.timed_out
    assert completed, "in-flight batches must be awaited, not cancelled"
    unsent = [e for e in result.errors if e.disposition == "unsent"]
    assert unsent and all(e.retryable for e in unsent)
    assert result.successful_item_count + result.failed_item_count == 8
    assert gate.quiescent()


def test_stall_abandons_remainder_with_bounded_attempts() -> None:
    gate = get_registry().get(HOST)
    while gate.limit > 1:
        gate.report_throttled()
    calls = {"n": 0}

    async def dying(batch: list[dict[str, Any]]) -> Any:
        calls["n"] += 1
        raise RuntimeError("UNAVAILABLE: backend gone")

    async def run() -> Any:
        return await bulk_execute_async(
            items=_items(40),
            operation=dying,
            batch_size=2,
            max_concurrency=4,
            show_progress=False,
            host=HOST,
        )

    result = asyncio.run(run())
    assert calls["n"] <= STALL_CONSECUTIVE_FAILURES + 2
    assert result.failed_item_count == 40
    abandoned = [e for e in result.errors if e.disposition == "abandoned"]
    assert abandoned
    assert gate.quiescent()


def test_external_cancellation_reraises_and_releases_slots() -> None:
    gate = get_registry().get(HOST)
    started = asyncio.Event()

    async def slow(batch: list[dict[str, Any]]) -> Any:
        started.set()
        await asyncio.sleep(5)
        return {"upserted_count": len(batch)}

    async def run() -> None:
        task = asyncio.ensure_future(
            bulk_execute_async(
                items=_items(10),
                operation=slow,
                batch_size=2,
                max_concurrency=2,
                show_progress=False,
                host=HOST,
            )
        )
        await started.wait()
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    assert gate.quiescent(), "cancellation leaked a slot"


def test_deadline_during_gate_wait_with_zero_free_slots() -> None:
    gate = get_registry().get(HOST)
    while gate.limit > 1:
        gate.report_throttled()
    _, held = gate.acquire()
    assert held is not None

    async def run() -> Any:
        t0 = time.monotonic()
        result = await bulk_execute_async(
            items=_items(4),
            operation=_ok,
            batch_size=2,
            max_concurrency=2,
            show_progress=False,
            host=HOST,
            total_timeout=0.3,
        )
        assert time.monotonic() - t0 < 5.0
        return result

    result = asyncio.run(run())
    held.release()
    assert result.timed_out
    assert result.failed_item_count == 4
    assert gate.quiescent()
