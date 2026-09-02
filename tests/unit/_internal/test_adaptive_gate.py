"""The concurrency gate belongs to the limiter, so it is shared per host.

Two bulk calls against one index previously kept their own in-flight counters
while consulting a shared limiter, so each admitted `current_limit()` requests
and the pair ran at twice the bound the limiter believed it had imposed.
"""

from __future__ import annotations

import threading
import time

from pinecone._internal.adaptive import _AdaptiveLimiter, _AdaptiveLimiterRegistry


class TestAcquireRelease:
    def test_acquire_admits_up_to_the_limit(self) -> None:
        limiter = _AdaptiveLimiter(3)

        assert all(limiter.acquire(deadline=None) for _ in range(3))
        assert limiter.inflight == 3

    def test_acquire_blocks_past_the_limit_until_released(self) -> None:
        limiter = _AdaptiveLimiter(1)
        assert limiter.acquire()

        admitted = threading.Event()

        def _second() -> None:
            limiter.acquire()
            admitted.set()

        waiter = threading.Thread(target=_second, daemon=True)
        waiter.start()
        try:
            assert not admitted.wait(timeout=0.2), "second acquire should have blocked"
            limiter.release()
            assert admitted.wait(timeout=5), "release should have admitted the waiter"
        finally:
            waiter.join(timeout=5)

    def test_acquire_gives_up_at_the_deadline(self) -> None:
        limiter = _AdaptiveLimiter(1)
        assert limiter.acquire()

        started = time.monotonic()
        taken = limiter.acquire(deadline=time.monotonic() + 0.1)

        assert taken is False
        assert time.monotonic() - started < 5
        assert limiter.inflight == 1, "a refused acquire must not take a slot"

    def test_an_expired_deadline_is_refused_even_when_a_slot_is_free(self) -> None:
        limiter = _AdaptiveLimiter(4)

        assert limiter.acquire(deadline=time.monotonic() - 1) is False
        assert limiter.inflight == 0

    def test_throttling_wakes_waiters_so_they_re_evaluate(self) -> None:
        """A halved limit must not leave a waiter parked forever."""
        limiter = _AdaptiveLimiter(2)
        assert limiter.acquire()
        assert limiter.acquire()

        refused = threading.Event()

        def _third() -> None:
            if limiter.acquire(deadline=time.monotonic() + 3) is False:
                refused.set()

        waiter = threading.Thread(target=_third, daemon=True)
        waiter.start()
        try:
            limiter.report_throttled()
            limiter.release()
            assert refused.wait(timeout=5) or limiter.inflight <= limiter.current_limit()
        finally:
            waiter.join(timeout=5)


class TestSharedAcrossCallers:
    def test_two_callers_on_one_host_share_the_bound(self) -> None:
        registry = _AdaptiveLimiterRegistry()
        host = "idx.svc.pinecone.io"

        first = registry.get(host, 2)
        second = registry.get(host, 2)
        assert first is second

        assert first.acquire()
        assert second.acquire()
        assert first.acquire(deadline=time.monotonic() + 0.1) is False, (
            "the second caller's slots must count against the first's limit"
        )
        assert first.inflight == 2
