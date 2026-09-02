"""Stall recovery (issue #150): a stall is a cool-down, not a verdict.

The regression these tests pin: the gate lives in a process-global registry,
and before the cool-down existed nothing ever reset the failure streak of a
stalled gate — one outage bricked the host key for the process lifetime.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from pinecone._internal.bulk import bulk_execute_sync, get_registry
from pinecone._internal.bulk.core import (
    STALL_CONSECUTIVE_FAILURES,
    STALL_COOLDOWN_SECONDS,
    AcquireOutcome,
    GateCore,
)
from pinecone._internal.bulk.gate import HostGate

HOST = "stall-recovery.example.com"


def _trip_stall(core: GateCore, now: float = 0.0) -> None:
    for _ in range(STALL_CONSECUTIVE_FAILURES):
        core.report_failure(now)
    assert core.stalled


class TestCoreRecovery:
    def test_stall_refuses_during_cooldown_then_recovers(self) -> None:
        core = GateCore(initial_limit=1)
        _trip_stall(core, now=0.0)

        refused = core.request("sync", now=STALL_COOLDOWN_SECONDS - 0.1)
        assert refused.stalled and not refused.granted

        probe = core.request("sync", now=STALL_COOLDOWN_SECONDS + 0.1)
        assert probe.granted and not probe.stalled
        assert not core.stalled
        core.release(STALL_COOLDOWN_SECONDS + 0.2)

    def test_failed_settle_while_stalled_extends_the_cooldown(self) -> None:
        core = GateCore(initial_limit=1)
        _trip_stall(core, now=0.0)

        core.report_failure(now=20.0)

        still_refused = core.request("sync", now=20.0 + STALL_COOLDOWN_SECONDS - 0.1)
        assert still_refused.stalled

        probe = core.request("sync", now=20.0 + STALL_COOLDOWN_SECONDS + 0.1)
        assert probe.granted
        core.release(20.0 + STALL_COOLDOWN_SECONDS + 0.2)

    def test_stall_via_throttle_edge_also_gets_a_cooldown(self) -> None:
        core = GateCore(initial_limit=2)
        for _ in range(STALL_CONSECUTIVE_FAILURES):
            core.report_failure(now=0.0)
        assert not core.stalled

        core.report_throttled(now=0.0)
        assert core.stalled

        assert core.request("sync", now=1.0).stalled
        probe = core.request("sync", now=STALL_COOLDOWN_SECONDS + 0.1)
        assert probe.granted
        core.release(STALL_COOLDOWN_SECONDS + 0.2)

    def test_stall_shaped_state_without_a_cooldown_stamp_recovers(self) -> None:
        """Defensive floor: no stall-shaped state may ever be permanent in a
        process-global gate, even one reached by a path that skipped the
        flip (the exact shape of the original bug)."""
        core = GateCore(initial_limit=1)
        core._consecutive_failures = STALL_CONSECUTIVE_FAILURES
        assert core.stalled
        assert core._stalled_until is None

        probe = core.request("sync", now=0.0)
        assert probe.granted
        core.release(0.1)

    def test_success_after_recovery_fully_clears_the_detector(self) -> None:
        core = GateCore(initial_limit=1)
        _trip_stall(core, now=0.0)

        probe = core.request("sync", now=STALL_COOLDOWN_SECONDS + 1.0)
        assert probe.granted
        core.report_success(STALL_COOLDOWN_SECONDS + 1.1)
        core.release(STALL_COOLDOWN_SECONDS + 1.2)

        for _ in range(STALL_CONSECUTIVE_FAILURES - 1):
            core.report_failure(STALL_COOLDOWN_SECONDS + 2.0)
        assert not core.stalled


class TestGateRecovery:
    def test_sync_acquire_recovers_after_cooldown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("pinecone._internal.bulk.core.STALL_COOLDOWN_SECONDS", 0.05)
        gate = HostGate(initial_limit=1)
        for _ in range(STALL_CONSECUTIVE_FAILURES):
            gate.report_failure()

        outcome, slot = gate.acquire(deadline=time.monotonic() + 1.0)
        assert outcome is AcquireOutcome.STALLED and slot is None

        time.sleep(0.06)
        outcome, slot = gate.acquire(deadline=time.monotonic() + 1.0)
        assert outcome is AcquireOutcome.GRANTED and slot is not None
        slot.release()

    @pytest.mark.asyncio
    async def test_async_acquire_recovers_after_cooldown(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("pinecone._internal.bulk.core.STALL_COOLDOWN_SECONDS", 0.05)
        gate = HostGate(initial_limit=1)
        for _ in range(STALL_CONSECUTIVE_FAILURES):
            gate.report_failure()

        outcome, slot = await gate.acquire_async(deadline=time.monotonic() + 1.0)
        assert outcome is AcquireOutcome.STALLED and slot is None

        await asyncio.sleep(0.06)
        outcome, slot = await gate.acquire_async(deadline=time.monotonic() + 1.0)
        assert outcome is AcquireOutcome.GRANTED and slot is not None
        slot.release()


class TestEngineCrossCallRecovery:
    """The original bug, end to end: call 1 stalls, call 2 (during cooldown)
    fails fast, call 3 (after cooldown, healthy backend) must succeed."""

    def test_stalled_host_serves_again_after_cooldown(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The cooldown is generous (5s) so the fail-fast middle call cannot
        flake on a slow runner; recovery is then forced by expiring the
        stamp directly instead of sleeping it out."""
        monkeypatch.setattr("pinecone._internal.bulk.core.STALL_COOLDOWN_SECONDS", 5.0)
        gate = get_registry().get(HOST)
        while gate.limit > 1:
            gate.report_throttled()

        def dying(batch: list[dict[str, Any]]) -> Any:
            raise RuntimeError("UNAVAILABLE: backend gone")

        items = [{"id": str(i)} for i in range(20)]
        first = bulk_execute_sync(
            items=items,
            operation=dying,
            batch_size=2,
            max_concurrency=2,
            show_progress=False,
            host=HOST,
        )
        assert first.stalled is True
        assert first.failed_item_count == 20

        calls = {"n": 0}

        def healthy(batch: list[dict[str, Any]]) -> Any:
            calls["n"] += 1
            return {"upserted_count": len(batch)}

        during_cooldown = bulk_execute_sync(
            items=items,
            operation=healthy,
            batch_size=2,
            max_concurrency=2,
            show_progress=False,
            host=HOST,
        )
        assert during_cooldown.stalled is True
        assert calls["n"] == 0, "cooldown must fail fast without touching the backend"

        gate._core._stalled_until = time.monotonic() - 1.0
        recovered = bulk_execute_sync(
            items=items,
            operation=healthy,
            batch_size=2,
            max_concurrency=2,
            show_progress=False,
            host=HOST,
        )
        assert recovered.stalled is False
        assert recovered.successful_item_count == 20
        assert calls["n"] == 10

    def test_result_stalled_flag_defaults_false_on_healthy_run(self) -> None:
        result = bulk_execute_sync(
            items=[{"id": "1"}],
            operation=lambda batch: {"upserted_count": len(batch)},
            batch_size=1,
            max_concurrency=1,
            show_progress=False,
            host=HOST,
        )
        assert result.stalled is False
