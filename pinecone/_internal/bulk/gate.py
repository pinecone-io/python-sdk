"""The waiting shell around :class:`~pinecone._internal.bulk.core.GateCore`.

Invariant this module exists to honor: throttle, success, and release signals
arrive on arbitrary threads — the gRPC ``on_throttle`` callback fires from a
tokio worker, done-callbacks fire on pool workers — so every wakeup into
asyncio goes through ``loop.call_soon_threadsafe``, and no loop-bound
primitive (``asyncio.Event``, ``asyncio.Condition``) ever lives on the gate.
Each async waiter carries its own ``(loop, future)``, captured at wait time,
which is what makes multiple (sequential or concurrent) event loops work.

Sync and async waiters share one FIFO in the core — under a process-global
registry, one process running both client flavors against one host is a
shipped scenario, and separate pools would lose wakeups or starve one kind.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from typing import Any

from pinecone._internal.bulk.core import AcquireOutcome, GateCore, Waiter


class Slot:
    """One granted admission. ``release`` is idempotent and never raises —
    it runs inside executor/task done-callback machinery that swallows
    exceptions, where a raising release would silently corrupt accounting."""

    __slots__ = ("_gate", "_released")

    def __init__(self, gate: HostGate) -> None:
        self._gate = gate
        self._released = False

    def release(self, *_args: Any) -> None:
        if self._released:
            return
        self._released = True
        with contextlib.suppress(Exception):
            self._gate._release()


class HostGate:
    """Per-host adaptive admission. All state behind one lock; the pure core
    decides, this class only waits and wakes."""

    __slots__ = ("_cond", "_core", "_lock")

    def __init__(self, initial_limit: int | None = None) -> None:
        from pinecone._internal.bulk.core import GATE_CEILING

        self._core = GateCore(initial_limit if initial_limit is not None else GATE_CEILING)
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

    @property
    def limit(self) -> int:
        return self._core.limit

    @property
    def inflight(self) -> int:
        return self._core.inflight

    @property
    def stalled(self) -> bool:
        return self._core.stalled

    @property
    def throttle_events(self) -> int:
        return self._core.throttle_events

    def quiescent(self) -> bool:
        with self._lock:
            return self._core.quiescent()

    def acquire(self, deadline: float | None = None) -> tuple[AcquireOutcome, Slot | None]:
        """Blocking acquire. ``deadline`` is an absolute ``time.monotonic``
        instant; ``None`` waits indefinitely (the caller's engine owns any
        overall budget)."""
        with self._cond:
            w = self._core.request("sync", time.monotonic())
            while not w.granted and not w.stalled:
                now = time.monotonic()
                if deadline is not None and now >= deadline:
                    self._deliver(self._core.abandon(w, now))
                    return (AcquireOutcome.TIMED_OUT, None)
                timeout = None if deadline is None else deadline - now
                hold = self._core.hold_remaining(now)
                if hold is not None:
                    timeout = hold if timeout is None else min(timeout, hold)
                self._cond.wait(timeout)
                self._deliver(self._core.tick(time.monotonic()))
            if w.stalled:
                return (AcquireOutcome.STALLED, None)
            return (AcquireOutcome.GRANTED, Slot(self))

    async def acquire_async(
        self, deadline: float | None = None
    ) -> tuple[AcquireOutcome, Slot | None]:
        """Async acquire; 3.10-clean (no asyncio.timeout). The waiter's own
        ``call_later`` timer is the wake source for deadline and hold expiry —
        with zero other traffic nothing else would ever wake it."""
        loop = asyncio.get_running_loop()
        with self._lock:
            w = self._core.request("async", time.monotonic())
            if w.granted:
                return (AcquireOutcome.GRANTED, Slot(self))
            if w.stalled:
                return (AcquireOutcome.STALLED, None)
            fut: asyncio.Future[None] = loop.create_future()
            w.delivery = (loop, fut)
            timer = self._arm_timer(loop, fut, w, deadline)
        try:
            await fut
        except asyncio.CancelledError:
            with self._lock:
                self._deliver(self._core.abandon(w, time.monotonic()))
            raise
        finally:
            if timer is not None:
                timer.cancel()
        if w.stalled:
            return (AcquireOutcome.STALLED, None)
        if w.granted:
            return (AcquireOutcome.GRANTED, Slot(self))
        return (AcquireOutcome.TIMED_OUT, None)

    def _arm_timer(
        self,
        loop: asyncio.AbstractEventLoop,
        fut: asyncio.Future[None],
        w: Waiter,
        deadline: float | None,
    ) -> asyncio.TimerHandle | None:
        now = time.monotonic()
        delays = []
        if deadline is not None:
            delays.append(deadline - now)
        hold = self._core.hold_remaining(now)
        if hold is not None:
            delays.append(hold)
        if not delays:
            return None

        def _on_timer() -> None:
            with self._lock:
                self._deliver(self._core.tick(time.monotonic()))
                if fut.done() or w.granted or w.stalled:
                    return
                now2 = time.monotonic()
                if deadline is not None and now2 >= deadline:
                    self._core.abandon(w, now2)
                    fut.set_result(None)
                    return
                self._arm_timer(loop, fut, w, deadline)

        return loop.call_later(max(0.0, min(delays)), _on_timer)

    def report_throttled(self, pushback_seconds: float | None = None) -> None:
        """A hold set after waiters parked must reach them: sync waiters
        computed their wait timeout at entry, async waiters armed their timer
        from the hold visible then — without a poke here, an at-capacity
        queue that drains during the hold parks until its deadline (or
        forever). notify_all makes sync waiters recompute; each parked async
        waiter gets its timer re-armed on its own loop."""
        with self._cond:
            self._deliver(self._core.report_throttled(time.monotonic(), pushback_seconds))
            self._cond.notify_all()
            for w in list(self._core._waiters):
                if w.kind != "async" or w.delivery is None:
                    continue
                loop, fut = w.delivery
                with contextlib.suppress(RuntimeError):
                    loop.call_soon_threadsafe(self._refresh_timer, w, fut, loop)

    def report_success(self) -> None:
        with self._cond:
            self._deliver(self._core.report_success(time.monotonic()))

    def report_failure(self) -> None:
        with self._cond:
            self._deliver(self._core.report_failure(time.monotonic()))

    def _release(self) -> None:
        with self._cond:
            self._deliver(self._core.release(time.monotonic()))

    def _deliver(self, woken: list[Waiter]) -> None:
        """Deliver grants/stall-wakes. Caller holds the lock. Sync waiters
        share the condition (notify_all + own-flag recheck, since a Condition
        cannot target one waiter); async waiters get a threadsafe callback
        whose closure re-checks the future — a waiter cancelled between grant
        and delivery must hand its slot back, not leak it."""
        notify_sync = False
        for w in woken:
            if w.kind == "sync" or w.delivery is None:
                notify_sync = True
                continue
            loop, fut = w.delivery
            try:
                loop.call_soon_threadsafe(self._finish_async, w, fut)
            except RuntimeError:
                more = self._core.abandon(w, time.monotonic())
                if more:
                    self._deliver(more)
        if notify_sync:
            self._cond.notify_all()

    def _refresh_timer(
        self, w: Waiter, fut: asyncio.Future[None], loop: asyncio.AbstractEventLoop
    ) -> None:
        with self._lock:
            if fut.done() or w.granted or w.stalled:
                return
            self._arm_timer(loop, fut, w, None)

    def _finish_async(self, w: Waiter, fut: asyncio.Future[None]) -> None:
        if fut.cancelled():
            with self._lock:
                self._deliver(self._core.abandon(w, time.monotonic()))
            return
        if not fut.done():
            fut.set_result(None)
