"""Stateful property tests for the pure admission core.

The core is a deterministic function of (state, event, now), so Hypothesis
can drive it through arbitrary event schedules with a simulated clock and
check the invariants after every step — no threads, no sleeps, shrinking
counterexamples. This is the machine that would have caught the _inflight
slot leak, the grant-then-cancel race, and the six-halvings-per-batch AIMD
bug, all of which were event-ordering defects.
"""

from __future__ import annotations

import hypothesis.strategies as st
from hypothesis import settings
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule

from pinecone._internal.bulk.core import GATE_CEILING, GateCore, Waiter


class GateMachine(RuleBasedStateMachine):
    def __init__(self) -> None:
        super().__init__()
        self.core = GateCore(initial_limit=8)
        self.now = 0.0
        self.outstanding: list[Waiter] = []
        self.queued: list[Waiter] = []
        self.limit_decreases = 0
        self.releases_since_decrease = 0
        self.inflight_at_last_decrease = 0
        self.bound_seen_since_change = False
        self.last_limit = self.core.limit

    def _absorb(self, woken: list[Waiter]) -> None:
        for w in woken:
            if w.granted:
                assert w in self.queued, "granted a waiter the model never queued"
                self.queued.remove(w)
                self.outstanding.append(w)
            elif w.stalled:
                assert w in self.queued
                self.queued.remove(w)

    @rule()
    def request(self) -> None:
        pre_inflight = self.core.inflight
        pre_limit = self.core.limit
        w = self.core.request("sync", self.now)
        if w.granted:
            assert pre_inflight < pre_limit, "grant while at/over the limit"
            assert not self.queued, "grant barged past a queued waiter"
            self.outstanding.append(w)
        elif w.stalled:
            assert self.core.stalled
        else:
            self.queued.append(w)
            self.bound_seen_since_change = True

    @precondition(lambda self: self.outstanding)
    @rule(data=st.data())
    def release(self, data: st.DataObject) -> None:
        idx = data.draw(st.integers(0, len(self.outstanding) - 1))
        self.outstanding.pop(idx)
        self.releases_since_decrease += 1
        self._absorb(self.core.release(self.now))

    @precondition(lambda self: self.queued)
    @rule(data=st.data())
    def abandon_queued(self, data: st.DataObject) -> None:
        idx = data.draw(st.integers(0, len(self.queued) - 1))
        w = self.queued.pop(idx)
        self._absorb(self.core.abandon(w, self.now))

    @precondition(lambda self: self.outstanding)
    @rule(data=st.data())
    def abandon_granted(self, data: st.DataObject) -> None:
        idx = data.draw(st.integers(0, len(self.outstanding) - 1))
        w = self.outstanding.pop(idx)
        self.releases_since_decrease += 1
        self._absorb(self.core.abandon(w, self.now))

    @rule(pushback=st.one_of(st.none(), st.floats(0.1, 30.0)))
    def throttle(self, pushback: float | None) -> None:
        pre_limit = self.core.limit
        pre_inflight = self.core.inflight
        self._absorb(self.core.report_throttled(self.now, pushback))
        if self.core.limit < pre_limit:
            if self.inflight_at_last_decrease > 0 and self.limit_decreases > 0:
                assert self.releases_since_decrease >= self.inflight_at_last_decrease, (
                    "second decrease before the first epoch's flight settled"
                )
            expected_flight = pre_inflight if pre_inflight > 0 else pre_limit
            assert self.core.limit == max(1, min(pre_limit, expected_flight) // 2)
            self.limit_decreases += 1
            self.releases_since_decrease = 0
            self.inflight_at_last_decrease = pre_inflight
            self.bound_seen_since_change = False

    @rule()
    def success(self) -> None:
        pre_limit = self.core.limit
        self._absorb(self.core.report_success(self.now))
        if self.core.limit > pre_limit:
            assert self.core.limit == pre_limit + 1, "increase must be additive"
            assert self.bound_seen_since_change, "limit grew while never bound"
            self.bound_seen_since_change = False

    @rule()
    def failure(self) -> None:
        self._absorb(self.core.report_failure(self.now))

    @rule(dt=st.floats(0.01, 40.0))
    def advance_clock(self, dt: float) -> None:
        self.now += dt
        self._absorb(self.core.tick(self.now))

    @invariant()
    def stall_means_empty_queue(self) -> None:
        """Every stall edge must wake the queue: a queued waiter surviving
        into a stalled state waits on a host the gate has declared dead."""
        if self.core.stalled:
            assert self.core.waiting == 0

    @invariant()
    def limit_in_range(self) -> None:
        assert 1 <= self.core.limit <= GATE_CEILING

    @invariant()
    def inflight_matches_model(self) -> None:
        assert self.core.inflight == len(self.outstanding)
        assert self.core.inflight >= 0

    @invariant()
    def queue_matches_model(self) -> None:
        assert self.core.waiting == len(self.queued)

    @invariant()
    def no_admissible_starvation(self) -> None:
        """If capacity exists, no hold is active, and nobody is stalled, the
        queue must be empty — a queued waiter with a free admissible slot is
        a lost wakeup."""
        if (
            self.queued
            and self.core.inflight < self.core.limit
            and self.core.hold_remaining(self.now) is None
            and not self.core.stalled
        ):
            raise AssertionError("waiter queued while a slot was admissible")

    def teardown(self) -> None:
        for w in list(self.queued):
            self.core.abandon(w, self.now)
        for _ in list(self.outstanding):
            self.core.release(self.now)
        assert self.core.quiescent(), "gate not quiescent after full drain"
        assert self.core.inflight == 0


TestGateMachine = GateMachine.TestCase
TestGateMachine.settings = settings(max_examples=200, stateful_step_count=60, deadline=None)
