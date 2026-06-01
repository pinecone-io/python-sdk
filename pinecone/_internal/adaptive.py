"""Per-host adaptive concurrency limiter (AIMD).

Internal to the SDK. The transport calls ``report_throttled(host)`` on every
retryable response; the bulk paths read ``current_limit(host, ceiling)``
before dispatching work. The limiter self-tunes effective concurrency
between ``1`` and the user-provided ``max_concurrency`` ceiling.

Not thread-coordinated across processes. See ``docs/guides/retries.md``
for multi-process guidance.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict

logger = logging.getLogger(__name__)

# Cap on the per-client _AdaptiveLimiter registry. Realistic Pinecone projects
# have far fewer than this many distinct hosts, so a bound mostly protects
# long-running services that rotate through many indexes. LRU eviction is
# safe because evicted limiters' AIMD state is just an optimization; a
# fresh limiter rebuilds correctly from throttle/success signals.
_MAX_REGISTRY_SIZE = 256


class _AdaptiveLimiter:
    """AIMD state for a single host.

    Multiplicative decrease on throttle: ``limit = max(1, limit // 2)``.
    Additive increase on success: ``limit = min(ceiling, limit + 1)``
    after ``current_limit`` consecutive successes. The success counter
    resets on each throttle.
    """

    __slots__ = ("_ceiling", "_limit", "_lock", "_success_streak")

    def __init__(self, ceiling: int) -> None:
        if ceiling < 1:
            raise ValueError(f"ceiling must be >= 1, got {ceiling}")
        self._lock = threading.Lock()
        self._ceiling = ceiling
        self._limit = ceiling
        self._success_streak = 0

    @property
    def ceiling(self) -> int:
        return self._ceiling

    def current_limit(self) -> int:
        """Return the current effective concurrency limit (1 <= limit <= ceiling)."""
        return self._limit

    def report_throttled(self) -> None:
        """Halve the limit (floored at 1) and reset the success streak."""
        with self._lock:
            before = self._limit
            self._limit = max(1, self._limit // 2)
            self._success_streak = 0
        if before != self._limit:
            logger.debug(
                "AIMD limiter decreased: before=%d after=%d ceiling=%d",
                before,
                self._limit,
                self._ceiling,
            )

    def report_success(self) -> None:
        """Increment the success streak; bump limit by 1 if streak hits current limit."""
        with self._lock:
            self._success_streak += 1
            increased = False
            if self._success_streak >= self._limit:
                if self._limit < self._ceiling:
                    self._limit = self._limit + 1
                    increased = True
                self._success_streak = 0
        if increased:
            logger.debug(
                "AIMD limiter increased: now=%d ceiling=%d",
                self._limit,
                self._ceiling,
            )

    def update_ceiling(self, ceiling: int) -> None:
        """Re-anchor the ceiling (e.g., a later bulk call uses a different max_concurrency).

        Clamps the current limit to the new ceiling. Never raises the limit
        beyond what AIMD has earned — only the ceiling moves.
        """
        if ceiling < 1:
            raise ValueError(f"ceiling must be >= 1, got {ceiling}")
        with self._lock:
            self._ceiling = ceiling
            if self._limit > ceiling:
                self._limit = ceiling


class _AdaptiveLimiterRegistry:
    """Per-client registry of ``_AdaptiveLimiter`` instances, one per host.

    One instance lives on each ``Pinecone`` / ``AsyncPinecone`` client. The
    transport's ``on_throttle`` callback calls ``report_throttled(host)``;
    the bulk path calls ``get(host, ceiling).current_limit()`` before
    each batch dispatch.

    Capped at ``_MAX_REGISTRY_SIZE`` entries with LRU eviction on overflow.
    Evicted limiters lose their AIMD state, which is safe — a fresh limiter
    rebuilds correctly from subsequent throttle/success signals.
    """

    __slots__ = ("_ever_throttled", "_limiters", "_lock")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._limiters: OrderedDict[str, _AdaptiveLimiter] = OrderedDict()
        self._ever_throttled: set[str] = set()

    def get(self, host: str, ceiling: int) -> _AdaptiveLimiter:
        """Return the limiter for ``host``, creating one with ``ceiling`` if absent.

        If a limiter already exists with a different ceiling, the existing
        limiter's ceiling is updated to the new value (current limit
        stays unchanged unless it exceeds the new ceiling).
        """
        with self._lock:
            limiter = self._limiters.get(host)
            if limiter is None:
                limiter = _AdaptiveLimiter(ceiling)
                if len(self._limiters) >= _MAX_REGISTRY_SIZE:
                    evicted_host, _ = self._limiters.popitem(last=False)
                    self._ever_throttled.discard(evicted_host)
                self._limiters[host] = limiter
            else:
                if limiter.ceiling != ceiling:
                    limiter.update_ceiling(ceiling)
                self._limiters.move_to_end(host)
            return limiter

    def report_throttled(self, host: str) -> None:
        """Convenience: look up the limiter for ``host`` and decrement.

        If no limiter exists for the host yet (e.g., throttle arrived before
        any bulk call set a ceiling), this is a no-op. The bulk path will
        create one with the right ceiling on its first call.
        """
        with self._lock:
            limiter = self._limiters.get(host)
            if limiter is not None:
                self._limiters.move_to_end(host)
            first_time = host not in self._ever_throttled
            self._ever_throttled.add(host)
        if limiter is not None:
            limiter.report_throttled()
        if first_time:
            logger.info(
                "Rate limited by host=%s. Adaptive concurrency will reduce in-flight requests. "
                "See https://docs.pinecone.io/python/retries for details.",
                host,
            )
