"""Fault-injection matrix for the gate shell: every way a waiter can die,
on both flavors, must return its slot. These are the cases the pure state
machine cannot exercise — real threads, real event loops, real timers."""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from pinecone._internal.bulk.core import STALL_CONSECUTIVE_FAILURES, AcquireOutcome
from pinecone._internal.bulk.gate import HostGate


def test_sync_acquire_release_roundtrip() -> None:
    gate = HostGate(initial_limit=2)
    outcome, slot = gate.acquire()
    assert outcome is AcquireOutcome.GRANTED and slot is not None
    assert gate.inflight == 1
    slot.release()
    assert gate.inflight == 0


def test_release_is_idempotent() -> None:
    gate = HostGate(initial_limit=2)
    _, slot = gate.acquire()
    assert slot is not None
    slot.release()
    slot.release()
    slot.release()
    assert gate.inflight == 0


def test_sync_deadline_in_wait_times_out_without_leak() -> None:
    gate = HostGate(initial_limit=1)
    _, held = gate.acquire()
    assert held is not None
    t0 = time.monotonic()
    outcome, slot = gate.acquire(deadline=time.monotonic() + 0.2)
    assert outcome is AcquireOutcome.TIMED_OUT and slot is None
    assert 0.15 < time.monotonic() - t0 < 2.0
    held.release()
    assert gate.quiescent()


def test_sync_waiter_woken_by_cross_thread_release() -> None:
    gate = HostGate(initial_limit=1)
    _, held = gate.acquire()
    assert held is not None
    got: list[AcquireOutcome] = []

    def waiter() -> None:
        outcome, slot = gate.acquire(deadline=time.monotonic() + 5)
        got.append(outcome)
        if slot is not None:
            slot.release()

    t = threading.Thread(target=waiter)
    t.start()
    time.sleep(0.05)
    held.release()
    t.join(timeout=5)
    assert got == [AcquireOutcome.GRANTED]
    assert gate.quiescent()


def test_concurrent_sync_waiters_never_exceed_limit() -> None:
    gate = HostGate(initial_limit=3)
    peak = 0
    current = 0
    lock = threading.Lock()

    def worker(_: int) -> None:
        nonlocal peak, current
        outcome, slot = gate.acquire(deadline=time.monotonic() + 10)
        assert outcome is AcquireOutcome.GRANTED and slot is not None
        with lock:
            current += 1
            peak = max(peak, current)
        time.sleep(0.01)
        with lock:
            current -= 1
        slot.release()

    with ThreadPoolExecutor(16) as pool:
        list(pool.map(worker, range(24)))
    assert peak <= 3
    assert gate.quiescent()


def test_async_acquire_and_cross_thread_wakeup() -> None:
    gate = HostGate(initial_limit=1)

    async def run() -> None:
        _, held = await gate.acquire_async()
        assert held is not None

        async def waiter() -> AcquireOutcome:
            outcome, slot = await gate.acquire_async(deadline=time.monotonic() + 5)
            if slot is not None:
                slot.release()
            return outcome

        task = asyncio.create_task(waiter())
        await asyncio.sleep(0.05)
        threading.Thread(target=held.release).start()
        assert await task is AcquireOutcome.GRANTED

    asyncio.run(run())
    assert gate.quiescent()


def test_async_cancel_before_grant_leaves_no_leak() -> None:
    gate = HostGate(initial_limit=1)

    async def run() -> None:
        _, held = await gate.acquire_async()
        assert held is not None
        task = asyncio.create_task(gate.acquire_async())
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        held.release()

    asyncio.run(run())
    assert gate.quiescent()


def test_async_cancel_after_grant_returns_slot_to_next_waiter() -> None:
    """The GH-90155 shape: waiter A is granted, then cancelled before it
    observes the grant; the slot must pass to waiter B, not leak."""
    gate = HostGate(initial_limit=1)

    async def run() -> None:
        _, held = await gate.acquire_async()
        assert held is not None
        task_a = asyncio.create_task(gate.acquire_async())
        task_b = asyncio.create_task(gate.acquire_async())
        await asyncio.sleep(0.05)
        held.release()
        task_a.cancel()
        try:
            await task_a
        except asyncio.CancelledError:
            pass
        outcome, slot = await asyncio.wait_for(task_b, timeout=5)
        assert outcome is AcquireOutcome.GRANTED and slot is not None
        slot.release()

    asyncio.run(run())
    assert gate.quiescent()


def test_closed_loop_waiter_is_abandoned_and_slot_regranted() -> None:
    gate = HostGate(initial_limit=1)
    _, held = gate.acquire()
    assert held is not None

    async def park() -> None:
        task = asyncio.create_task(gate.acquire_async())
        await asyncio.sleep(0.05)
        task.cancel()

    asyncio.run(park())
    held.release()
    outcome, slot = gate.acquire(deadline=time.monotonic() + 2)
    assert outcome is AcquireOutcome.GRANTED and slot is not None
    slot.release()
    assert gate.quiescent()


def test_pushback_hold_expiry_wakes_async_waiter_with_zero_traffic() -> None:
    """The quiet-failure case from review: a lone Retry-After with no other
    traffic must not park an ingest forever — the waiter's own timer is the
    wake source."""
    gate = HostGate(initial_limit=4)
    gate.report_throttled(pushback_seconds=0.3)

    async def run() -> float:
        t0 = time.monotonic()
        outcome, slot = await gate.acquire_async(deadline=time.monotonic() + 10)
        assert outcome is AcquireOutcome.GRANTED and slot is not None
        slot.release()
        return time.monotonic() - t0

    elapsed = asyncio.run(run())
    assert 0.25 < elapsed < 5.0
    assert gate.quiescent()


def test_pushback_hold_expiry_wakes_sync_waiter() -> None:
    gate = HostGate(initial_limit=4)
    gate.report_throttled(pushback_seconds=0.3)
    t0 = time.monotonic()
    outcome, slot = gate.acquire(deadline=time.monotonic() + 10)
    assert outcome is AcquireOutcome.GRANTED and slot is not None
    assert 0.25 < time.monotonic() - t0 < 5.0
    slot.release()
    assert gate.quiescent()


def test_stall_refuses_new_waiters_and_wakes_queued_ones() -> None:
    gate = HostGate(initial_limit=1)
    gate.report_throttled()
    _, held = gate.acquire()
    assert held is not None
    woken: list[AcquireOutcome] = []

    def queued_waiter() -> None:
        outcome, slot = gate.acquire(deadline=time.monotonic() + 10)
        woken.append(outcome)
        if slot is not None:
            slot.release()

    t = threading.Thread(target=queued_waiter)
    t.start()
    time.sleep(0.05)
    for _ in range(STALL_CONSECUTIVE_FAILURES):
        gate.report_failure()
    t.join(timeout=5)
    assert woken == [AcquireOutcome.STALLED]
    outcome, slot = gate.acquire()
    assert outcome is AcquireOutcome.STALLED and slot is None
    held.release()
    gate.report_success()
    outcome, slot = gate.acquire()
    assert outcome is AcquireOutcome.GRANTED and slot is not None
    slot.release()
    assert gate.quiescent()


def test_two_sequential_event_loops_on_one_gate() -> None:
    gate = HostGate(initial_limit=2)

    async def use_once() -> None:
        outcome, slot = await gate.acquire_async(deadline=time.monotonic() + 5)
        assert outcome is AcquireOutcome.GRANTED and slot is not None
        slot.release()

    asyncio.run(use_once())
    asyncio.run(use_once())
    assert gate.quiescent()


def test_mixed_sync_and_async_waiters_share_fifo() -> None:
    gate = HostGate(initial_limit=1)
    _, held = gate.acquire()
    assert held is not None
    order: list[str] = []

    def sync_waiter() -> None:
        outcome, slot = gate.acquire(deadline=time.monotonic() + 10)
        assert outcome is AcquireOutcome.GRANTED and slot is not None
        order.append("sync")
        slot.release()

    t = threading.Thread(target=sync_waiter)
    t.start()
    time.sleep(0.05)

    async def async_waiter() -> None:
        outcome, slot = await gate.acquire_async(deadline=time.monotonic() + 10)
        assert outcome is AcquireOutcome.GRANTED and slot is not None
        order.append("async")
        slot.release()

    def release_soon() -> None:
        time.sleep(0.05)
        held.release()

    threading.Thread(target=release_soon).start()
    asyncio.run(async_waiter())
    t.join(timeout=5)
    assert sorted(order) == ["async", "sync"]
    assert gate.quiescent()


def test_hold_set_after_sync_waiter_parked_wakes_at_hold_expiry() -> None:
    gate = HostGate(initial_limit=1)
    _, held = gate.acquire()
    assert held is not None
    result: list[float] = []

    def waiter() -> None:
        t0 = time.monotonic()
        outcome, slot = gate.acquire(deadline=time.monotonic() + 10)
        assert outcome is AcquireOutcome.GRANTED and slot is not None
        result.append(time.monotonic() - t0)
        slot.release()

    t = threading.Thread(target=waiter)
    t.start()
    time.sleep(0.05)
    gate.report_throttled(pushback_seconds=0.3)
    held.release()
    t.join(timeout=5)
    assert result and 0.2 < result[0] < 5.0, f"waiter waited {result} — deadline, not hold expiry"
    assert gate.quiescent()


def test_hold_set_after_async_waiter_parked_wakes_at_hold_expiry() -> None:
    gate = HostGate(initial_limit=1)

    async def run() -> float:
        _, held = await gate.acquire_async()
        assert held is not None

        async def waiter() -> float:
            t0 = time.monotonic()
            outcome, slot = await gate.acquire_async(deadline=time.monotonic() + 10)
            assert outcome is AcquireOutcome.GRANTED and slot is not None
            slot.release()
            return time.monotonic() - t0

        task = asyncio.create_task(waiter())
        await asyncio.sleep(0.05)
        gate.report_throttled(pushback_seconds=0.3)
        held.release()
        return await asyncio.wait_for(task, timeout=8)

    elapsed = asyncio.run(run())
    assert 0.2 < elapsed < 5.0, f"async waiter waited {elapsed}s — deadline, not hold expiry"
    assert gate.quiescent()


def test_stall_via_throttle_edge_wakes_queued_waiter_and_blocks_grants() -> None:
    gate = HostGate(initial_limit=2)
    _, a = gate.acquire()
    _, b = gate.acquire()
    assert a is not None and b is not None
    for _ in range(STALL_CONSECUTIVE_FAILURES):
        gate.report_failure()
    assert not gate.stalled
    woken: list[AcquireOutcome] = []

    def queued_waiter() -> None:
        outcome, slot = gate.acquire(deadline=time.monotonic() + 10)
        woken.append(outcome)
        if slot is not None:
            slot.release()

    t = threading.Thread(target=queued_waiter)
    t.start()
    time.sleep(0.05)
    gate.report_throttled()
    assert gate.stalled
    t.join(timeout=5)
    assert woken == [AcquireOutcome.STALLED]
    a.release()
    b.release()
    assert gate.quiescent()
