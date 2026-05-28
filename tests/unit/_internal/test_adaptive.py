from __future__ import annotations

import pytest

from pinecone._internal.adaptive import _AdaptiveLimiter, _AdaptiveLimiterRegistry


class TestAdaptiveLimiter:
    def test_initial_limit_is_ceiling(self) -> None:
        lim = _AdaptiveLimiter(ceiling=10)
        assert lim.current_limit() == 10
        assert lim.ceiling == 10

    def test_throttle_halves_limit(self) -> None:
        lim = _AdaptiveLimiter(ceiling=10)
        lim.report_throttled()
        assert lim.current_limit() == 5

    def test_throttle_floors_at_one(self) -> None:
        lim = _AdaptiveLimiter(ceiling=2)
        lim.report_throttled()  # 2 → 1
        lim.report_throttled()  # 1 → 1 (floor)
        lim.report_throttled()
        assert lim.current_limit() == 1

    def test_success_streak_increases_limit(self) -> None:
        lim = _AdaptiveLimiter(ceiling=10)
        lim.report_throttled()  # 10 → 5
        for _ in range(5):
            lim.report_success()
        assert lim.current_limit() == 6

    def test_success_streak_resets_on_throttle(self) -> None:
        lim = _AdaptiveLimiter(ceiling=10)
        lim.report_throttled()  # 10 → 5
        for _ in range(3):
            lim.report_success()
        lim.report_throttled()  # 5 → 2; streak reset
        for _ in range(2):
            lim.report_success()
        # Streak hit 2 → limit becomes 3 (not 4 — streak was reset)
        assert lim.current_limit() == 3

    def test_increase_caps_at_ceiling(self) -> None:
        lim = _AdaptiveLimiter(ceiling=2)
        # No throttle; just lots of successes
        for _ in range(100):
            lim.report_success()
        assert lim.current_limit() == 2

    def test_update_ceiling_clamps_limit(self) -> None:
        lim = _AdaptiveLimiter(ceiling=10)
        assert lim.current_limit() == 10
        lim.update_ceiling(3)
        assert lim.ceiling == 3
        assert lim.current_limit() == 3

    def test_update_ceiling_does_not_raise_limit(self) -> None:
        lim = _AdaptiveLimiter(ceiling=10)
        lim.report_throttled()  # 10 → 5
        lim.update_ceiling(20)  # ceiling moves up
        assert lim.ceiling == 20
        assert lim.current_limit() == 5  # AIMD-earned limit unchanged

    def test_invalid_ceiling_raises(self) -> None:
        with pytest.raises(ValueError):
            _AdaptiveLimiter(ceiling=0)
        with pytest.raises(ValueError):
            _AdaptiveLimiter(ceiling=-1)


class TestAdaptiveLimiterRegistry:
    def test_get_creates_limiter_on_first_call(self) -> None:
        reg = _AdaptiveLimiterRegistry()
        lim = reg.get("host-a.pinecone.io", ceiling=8)
        assert lim.ceiling == 8
        assert lim.current_limit() == 8

    def test_get_returns_same_instance_for_same_host(self) -> None:
        reg = _AdaptiveLimiterRegistry()
        a1 = reg.get("host-a.pinecone.io", ceiling=8)
        a2 = reg.get("host-a.pinecone.io", ceiling=8)
        assert a1 is a2

    def test_get_isolates_hosts(self) -> None:
        reg = _AdaptiveLimiterRegistry()
        a = reg.get("host-a.pinecone.io", ceiling=8)
        b = reg.get("host-b.pinecone.io", ceiling=8)
        assert a is not b
        a.report_throttled()
        assert a.current_limit() == 4
        assert b.current_limit() == 8

    def test_report_throttled_on_unknown_host_is_noop(self) -> None:
        reg = _AdaptiveLimiterRegistry()
        # Should not raise
        reg.report_throttled("unknown-host.pinecone.io")

    def test_get_with_different_ceiling_updates_existing(self) -> None:
        reg = _AdaptiveLimiterRegistry()
        a = reg.get("host-a.pinecone.io", ceiling=8)
        a.report_throttled()  # 8 → 4
        a2 = reg.get("host-a.pinecone.io", ceiling=16)
        assert a is a2
        assert a.ceiling == 16
        assert a.current_limit() == 4  # AIMD-earned limit unchanged

    def test_concurrent_access_invariants(self) -> None:
        """Spawn N threads alternately throttling and reporting success;
        assert no exceptions and 1 <= current_limit() <= ceiling throughout.

        Smoke-level concurrent-access check — not Hypothesis-grade — but it
        catches obvious failures (race on ``_limit``, missed ``notify``,
        silent ``AssertionError`` swallowing, deadlocks).
        """
        import threading
        import time

        reg = _AdaptiveLimiterRegistry()
        lim = reg.get("test-host", ceiling=16)
        errors: list[BaseException] = []
        stop = threading.Event()

        def thrash(action: str) -> None:
            try:
                while not stop.is_set():
                    if action == "throttle":
                        lim.report_throttled()
                    else:
                        lim.report_success()
                    cur = lim.current_limit()
                    assert 1 <= cur <= 16, f"limit out of bounds: {cur}"
            except BaseException as e:
                errors.append(e)

        threads = [threading.Thread(target=thrash, args=("throttle",)) for _ in range(4)]
        threads += [threading.Thread(target=thrash, args=("success",)) for _ in range(4)]
        for t in threads:
            t.start()
        time.sleep(0.5)
        stop.set()
        for t in threads:
            t.join(timeout=2.0)
            assert not t.is_alive(), "thread did not exit"
        assert not errors, f"errors during concurrent access: {errors}"
