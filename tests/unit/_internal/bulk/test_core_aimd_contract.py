"""Exact AIMD-trajectory contract tests, written from mutation findings.

A moonbuggy run over core.py (2026-08-20) showed the whole suite — unit,
property, machine, and storm — passes with limit GROWTH disabled entirely
(``_bound_since_change = True`` mutated to ``False``), with the ceiling
guard loosened, with the stall threshold off by one, and with a granted
waiter's abandonment leaving it marked granted. Ranges and invariants were
pinned; exact trajectories were not. These tests pin them.
"""

from __future__ import annotations

from pinecone._internal.bulk.core import (
    GATE_CEILING,
    STALL_CONSECUTIVE_FAILURES,
    GateCore,
)


def _drain_one(core: GateCore, now: float = 0.0) -> None:
    granted = core.request("sync", now)
    assert granted.granted
    core.release(now)


class TestGrowth:
    def test_bound_gate_grows_by_one_after_a_limit_sized_streak(self) -> None:
        core = GateCore(initial_limit=2)
        first = core.request("sync", 0.0)
        second = core.request("sync", 0.0)
        queued = core.request("sync", 0.0)
        assert first.granted and second.granted and not queued.granted

        core.report_success(1.0)
        assert core.limit == 2, "streak of 1 must not grow a limit of 2"
        core.report_success(2.0)
        assert core.limit == 3, "bound gate + limit-sized streak must grow by exactly 1"

        core.release(3.0)
        core.release(3.0)
        assert queued.granted
        core.release(3.0)

    def test_unbound_gate_never_grows(self) -> None:
        core = GateCore(initial_limit=2)
        for _ in range(10):
            _drain_one(core)
            core.report_success(0.0)
        assert core.limit == 2, "idle headroom must not accumulate"

    def test_growth_stops_exactly_at_the_ceiling(self) -> None:
        core = GateCore(initial_limit=GATE_CEILING)
        waiters = [core.request("sync", 0.0) for _ in range(GATE_CEILING + 1)]
        assert sum(1 for w in waiters if w.granted) == GATE_CEILING
        for _ in range(GATE_CEILING):
            core.report_success(1.0)
        assert core.limit == GATE_CEILING, "the ceiling is a hard bound"
        for w in waiters:
            if w.granted:
                core.release(2.0)
            else:
                core.abandon(w, 2.0)

    def test_throttle_resets_the_success_streak(self) -> None:
        core = GateCore(initial_limit=2)
        blocked = core.request("sync", 0.0)
        blocked2 = core.request("sync", 0.0)
        queued = core.request("sync", 0.0)
        core.report_success(0.0)
        core.report_throttled(1.0)
        core.report_success(2.0)
        assert core.limit < 3, "one post-throttle success must not complete a pre-throttle streak"
        for w in (blocked, blocked2, queued):
            core.abandon(w, 3.0)
        while core.inflight:
            core.release(3.0)

    def test_fresh_gate_counters_start_at_zero(self) -> None:
        core = GateCore(initial_limit=1)
        assert core.throttle_events == 0
        core.report_throttled(0.0)
        assert core.throttle_events == 1, "the counter is absolute, anchored at zero"


class TestDecreaseEpoch:
    def test_exactly_one_decrease_per_inflight_epoch(self) -> None:
        core = GateCore(initial_limit=4)
        for _ in range(4):
            assert core.request("sync", 0.0).granted
        core.report_throttled(0.0)
        assert core.limit == 2, "halved from flight size 4"
        core.report_throttled(0.0)
        core.report_throttled(0.0)
        assert core.limit == 2, "same epoch: further throttles must not halve again"
        core.release(1.0)
        core.release(1.0)
        core.release(1.0)
        core.report_throttled(1.0)
        assert core.limit == 2, "epoch of 4 settles has one left; still no second decrease"
        core.release(1.0)
        core.report_throttled(2.0)
        assert core.limit == 1, "epoch over: the next throttle decreases again"


class TestStallBoundary:
    def test_shipped_stall_constants(self) -> None:
        assert GATE_CEILING == 64
        assert STALL_CONSECUTIVE_FAILURES == 4

    def test_stall_needs_exactly_four_failures_at_the_floor(self) -> None:
        core = GateCore(initial_limit=1)
        for _ in range(3):
            core.report_failure(0.0)
        assert not core.stalled, "three failures is a blip, not a stall"
        core.report_failure(0.0)
        assert core.stalled

    def test_recovery_grants_a_full_fresh_detector_window(self) -> None:
        core = GateCore(initial_limit=1)
        for _ in range(4):
            core.report_failure(0.0)
        assert core.stalled

        probe = core.request("sync", 100.0)
        assert probe.granted, "cool-down elapsed: the probe must be admitted"
        core.release(100.0)
        for _ in range(3):
            core.report_failure(100.0)
        assert not core.stalled, "recovery must reset the streak to zero, not partially"
        core.report_failure(100.0)
        assert core.stalled


class TestStreakResetExactness:
    def test_throttle_resets_the_streak_to_zero_not_one(self) -> None:
        core = GateCore(initial_limit=4)
        for _ in range(4):
            assert core.request("sync", 0.0).granted
        core.request("sync", 0.0)
        core.report_success(0.0)
        core.report_throttled(1.0)
        assert core.limit == 2
        rebind = core.request("sync", 1.5)
        core.report_success(2.0)
        assert core.limit == 2, "streak must restart at zero: one success of two is not a streak"
        core.abandon(rebind, 3.0)
        while core.inflight:
            core.release(3.0)
        for w in list(core._waiters):
            core.abandon(w, 3.0)

    def test_growth_resets_the_streak_to_zero_not_one(self) -> None:
        core = GateCore(initial_limit=2)
        for _ in range(2):
            assert core.request("sync", 0.0).granted
        core.request("sync", 0.0)
        core.report_success(0.0)
        core.report_success(0.0)
        assert core.limit == 3
        rebind = core.request("sync", 0.5)
        core.report_success(1.0)
        core.report_success(1.0)
        assert core.limit == 3, "the next growth needs a FULL new limit-sized streak"
        core.abandon(rebind, 2.0)
        while core.inflight:
            core.release(2.0)
        for w in list(core._waiters):
            core.abandon(w, 2.0)

    def test_growth_consumes_the_bound_flag(self) -> None:
        core = GateCore(initial_limit=2)
        for _ in range(2):
            assert core.request("sync", 0.0).granted
        queued = core.request("sync", 0.0)
        core.report_success(0.0)
        core.report_success(0.0)
        assert core.limit == 3
        core.abandon(queued, 1.0)
        while core.inflight:
            core.release(1.0)
        for _ in range(6):
            core.report_success(2.0)
        assert core.limit == 3, "growth spends the bound evidence; unbound streaks must not stack"


class TestReleaseGuard:
    def test_release_without_a_grant_is_an_assertion_error(self) -> None:
        import pytest

        core = GateCore(initial_limit=1)
        with pytest.raises(AssertionError, match="release without a matching grant"):
            core.release(0.0)


class TestAbandonGranted:
    def test_abandoned_granted_waiter_reports_timeout_and_returns_the_slot(self) -> None:
        core = GateCore(initial_limit=1)
        w = core.request("sync", 0.0)
        assert w.granted
        core.abandon(w, 0.0)
        assert not w.granted, "an abandoned grant must not read as GRANTED (GH-90155 class)"
        assert core.inflight == 0
        replacement = core.request("sync", 0.0)
        assert replacement.granted, "the abandoned slot must be reusable immediately"
        core.release(1.0)


class TestInstantBoundaries:
    def test_hold_lifts_at_exactly_the_hold_instant(self) -> None:
        core = GateCore(initial_limit=1)
        core.report_throttled(0.0, pushback_seconds=5.0)
        assert core.hold_remaining(5.0) is None
        w = core.request("sync", 5.0)
        assert w.granted, "admission resumes AT the hold instant, not after it"
        core.release(6.0)

    def test_stall_cooldown_ends_at_exactly_the_stamp(self) -> None:
        core = GateCore(initial_limit=1)
        for _ in range(4):
            core.report_failure(0.0)
        stamp = core._stalled_until
        assert stamp is not None
        w = core.request("sync", stamp)
        assert w.granted, "recovery happens AT stalled_until, not one tick later"
        core.release(stamp)
