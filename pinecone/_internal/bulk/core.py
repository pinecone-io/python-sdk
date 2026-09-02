"""The pure admission core: deterministic state transitions, no I/O, no clocks.

Every decision the per-host gate makes lives here as a total function of
(state, event, now). The shell in ``gate.py`` supplies one ``threading.Lock``,
the two waiter flavors, and wakeup delivery — nothing else. Time is an input
(``now: float``, from ``time.monotonic()``), never read here, so the whole
state space is drivable from a Hypothesis rule-based state machine with a
simulated clock.

The gate admits while ``inflight < limit``. Per-call ``max_concurrency``
bounds are the engine's job (it never has more than its own bound
outstanding), so global admission is min(caller bound, limit) without the
gate tracking per-caller state.

AIMD spec (panel-reviewed, issue #69):
- increase +1 after a limit-sized success streak, counted only while the gate
  was limit-bound during the streak — an unused limit must not drift upward;
- one multiplicative decrease per in-flight epoch, halved from flight size
  (``min(limit, inflight)``), so one batch burning six in-channel retries
  cannot halve the limit six times while its siblings succeed;
- a pushback hint (Retry-After / grpc-retry-pushback-ms) blocks admission
  until it elapses;
- stall detector: at the floor, consecutive all-failed settles with zero
  successes mean the backend is down — waiters are refused so callers abandon
  their remainder instead of queueing on a dead host.
"""

from __future__ import annotations

import contextlib
import enum
from collections import deque
from typing import Any

GATE_CEILING = 64
STALL_CONSECUTIVE_FAILURES = 4
"""Failed settles at the floor before the gate reports stalled.

The design note says 2x the floor admission (= 2); 4 adds a robustness
margin against declaring a backend dead on a two-sample blip. Each settle
already represents a fully retried batch, so 4 settles is ~24 wire attempts.
"""

STALL_COOLDOWN_SECONDS = 30.0
"""How long a tripped stall refuses admission before the gate probes again.

A stall must be a cool-down, not a terminal state: the gate lives in a
process-global registry, so a permanent refusal would brick the host key
for the process lifetime after one outage (issue #150). While the cool-down
runs, callers fail fast and abandon — the panel-intended behavior. When it
elapses, the failure streak resets on the next request and the gate probes
fresh from the floor: the limit is still 1, so recovery against a
still-dead backend costs one serial batch per window, and each failed
settle while stalled pushes the window out again.
"""


class AcquireOutcome(enum.Enum):
    GRANTED = "granted"
    TIMED_OUT = "timed_out"
    STALLED = "stalled"


class Waiter:
    """One queued acquire. The core owns ``granted``/``stalled``; ``delivery``
    is shell-owned state the core never reads."""

    __slots__ = ("delivery", "granted", "kind", "stalled")

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.granted = False
        self.stalled = False
        self.delivery: Any = None

    def outcome(self) -> AcquireOutcome:
        if self.granted:
            return AcquireOutcome.GRANTED
        if self.stalled:
            return AcquireOutcome.STALLED
        return AcquireOutcome.TIMED_OUT


class GateCore:
    """Deterministic gate state. Callers hold the shell's lock."""

    __slots__ = (
        "_bound_since_change",
        "_consecutive_failures",
        "_epoch_settles_remaining",
        "_hold_until",
        "_inflight",
        "_limit",
        "_stalled_until",
        "_success_streak",
        "_throttle_events",
        "_waiters",
    )

    def __init__(self, initial_limit: int = GATE_CEILING) -> None:
        if not 1 <= initial_limit <= GATE_CEILING:
            raise ValueError(f"initial_limit must be in [1, {GATE_CEILING}]")
        self._limit = initial_limit
        self._inflight = 0
        self._waiters: deque[Waiter] = deque()
        self._success_streak = 0
        self._bound_since_change = False
        self._epoch_settles_remaining = 0
        self._hold_until: float | None = None
        self._consecutive_failures = 0
        self._stalled_until: float | None = None
        self._throttle_events = 0

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def inflight(self) -> int:
        return self._inflight

    @property
    def waiting(self) -> int:
        return len(self._waiters)

    @property
    def throttle_events(self) -> int:
        return self._throttle_events

    @property
    def stalled(self) -> bool:
        return self._limit == 1 and self._consecutive_failures >= STALL_CONSECUTIVE_FAILURES

    def quiescent(self) -> bool:
        return self._inflight == 0 and not self._waiters

    def hold_remaining(self, now: float) -> float | None:
        """Seconds until the pushback hold lifts, or None when no hold is active."""
        if self._hold_until is None or now >= self._hold_until:
            return None
        return self._hold_until - now

    def _admissible(self, now: float) -> bool:
        if self._hold_until is not None:
            if now < self._hold_until:
                return False
            self._hold_until = None
        return self._inflight < self._limit

    def request(self, kind: str, now: float) -> Waiter:
        """A caller wants a slot. Grants immediately when admissible and the
        queue is empty (FIFO — nobody barges past an existing waiter);
        otherwise the waiter is queued, or refused outright when stalled."""
        w = Waiter(kind)
        self._maybe_recover(now)
        if self.stalled:
            w.stalled = True
            return w
        if not self._waiters and self._admissible(now):
            w.granted = True
            self._inflight += 1
        else:
            self._bound_since_change = True
            self._waiters.append(w)
        return w

    def _maybe_recover(self, now: float) -> None:
        """A stall is a cool-down, not a verdict (issue #150): once
        ``stalled_until`` passes, the failure streak resets and the next
        request probes the backend from the floor. The ``None`` guard keeps
        even an unforeseen stall-shaped state recoverable — a permanent
        refusal in a process-global gate is the one unacceptable outcome."""
        if not self.stalled:
            return
        if self._stalled_until is None or now >= self._stalled_until:
            self._consecutive_failures = 0
            self._stalled_until = None

    def release(self, now: float) -> list[Waiter]:
        """A granted slot settled (any outcome). Frees capacity, advances the
        decrease epoch, and grants to queued waiters in FIFO order."""
        if self._inflight <= 0:
            raise AssertionError("release without a matching grant")
        self._inflight -= 1
        if self._epoch_settles_remaining > 0:
            self._epoch_settles_remaining -= 1
        return self._grant_waiters(now)

    def abandon(self, waiter: Waiter, now: float) -> list[Waiter]:
        """A waiter gave up (deadline, cancellation, closed loop). If it was
        already granted, the slot is returned and re-granted — 'granted but
        never observed' must not leak (the GH-90155 class)."""
        if waiter.granted:
            waiter.granted = False
            return self.release(now)
        with contextlib.suppress(ValueError):
            self._waiters.remove(waiter)
        return []

    def report_throttled(self, now: float, pushback_seconds: float | None = None) -> list[Waiter]:
        """The host said slow down. One decrease per in-flight epoch; halve
        from flight size. A pushback hint additionally blocks admission until
        it elapses (jitter is the transport's job — one policy, applied once).

        Returns waiters woken by a stall flip: dropping the limit to 1 can
        make an existing failure streak count as stalled, and queued waiters
        must learn that the same way report_failure's edge tells them."""
        self._success_streak = 0
        self._throttle_events += 1
        if pushback_seconds is not None and pushback_seconds > 0:
            hold = now + pushback_seconds
            if self._hold_until is None or hold > self._hold_until:
                self._hold_until = hold
        if self._epoch_settles_remaining > 0:
            return []
        was_stalled = self.stalled
        flight = self._inflight if self._inflight > 0 else self._limit
        self._limit = max(1, min(self._limit, flight) // 2)
        self._epoch_settles_remaining = self._inflight
        self._bound_since_change = False
        if self.stalled and not was_stalled:
            self._stalled_until = now + STALL_COOLDOWN_SECONDS
            return self._wake_all_stalled()
        return []

    def report_success(self, now: float) -> list[Waiter]:
        """A batch settled successfully. Recovers the limit only when the gate
        was actually limit-bound during the streak — idle headroom must not
        accumulate (else the next decrease starts epochs above anyone's use)."""
        self._consecutive_failures = 0
        self._success_streak += 1
        if self._success_streak >= self._limit:
            self._success_streak = 0
            if self._bound_since_change and self._limit < GATE_CEILING:
                self._limit += 1
                self._bound_since_change = False
                return self._grant_waiters(now)
        return []

    def report_failure(self, now: float) -> list[Waiter]:
        """A batch settled with every attempt failed (not necessarily
        throttled — a dead backend returns UNAVAILABLE, not 429). Feeds the
        stall detector; on the flip to stalled, queued waiters are woken so
        their callers can abandon instead of waiting on a dead host."""
        was_stalled = self.stalled
        self._consecutive_failures += 1
        if self.stalled:
            self._stalled_until = now + STALL_COOLDOWN_SECONDS
        if self.stalled and not was_stalled:
            return self._wake_all_stalled()
        return []

    def tick(self, now: float) -> list[Waiter]:
        """The clock advanced past something (hold expiry). Grants whatever
        is now admissible."""
        return self._grant_waiters(now)

    def _wake_all_stalled(self) -> list[Waiter]:
        woken = list(self._waiters)
        self._waiters.clear()
        for w in woken:
            w.stalled = True
        return woken

    def _grant_waiters(self, now: float) -> list[Waiter]:
        if self.stalled:
            return []
        granted: list[Waiter] = []
        while self._waiters and self._admissible(now):
            w = self._waiters.popleft()
            w.granted = True
            self._inflight += 1
            granted.append(w)
        return granted
