"""Guards the escape hatches from the global sleep patch in conftest.py (#360).

``tests/unit/conftest.py`` no-ops ``time.sleep`` and ``asyncio.sleep`` for the
whole process, which is what keeps retry-backoff tests fast. The cost is that a
test needing real elapsed time silently measures nothing: a 0.5s concurrency
soak in ``test_adaptive.py`` ran for 0.0000s and passed. These tests pin both
halves of the contract — that the patch is still in force by default, and that
``real_sleep`` / ``real_async_sleep`` / ``suppress_retry_sleep`` still get you
out of it.

Every assertion here is made against ``time.monotonic``, and
``test_monotonic_is_not_patched`` establishes with a busy loop that that clock
is real. A sleep-based probe cannot verify any of this — it would be measuring
the patch.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable

import pytest

_LONG_ENOUGH_TO_NOTICE = 0.5
_SHORT_REAL_WAIT = 0.05
_NOOP_CEILING = 0.05


def test_monotonic_is_not_patched() -> None:
    """Ground truth for every other assertion in this file."""
    started = time.monotonic()
    while time.monotonic() - started < _SHORT_REAL_WAIT:
        pass
    assert time.monotonic() - started >= _SHORT_REAL_WAIT


def test_time_sleep_is_a_noop_by_default() -> None:
    started = time.monotonic()
    time.sleep(_LONG_ENOUGH_TO_NOTICE)
    elapsed = time.monotonic() - started
    assert elapsed < _NOOP_CEILING, (
        f"time.sleep({_LONG_ENOUGH_TO_NOTICE}) took {elapsed:.4f}s — the global "
        "patch is gone, and the retry-backoff tests are now paying real time"
    )


async def test_asyncio_sleep_is_also_a_noop_by_default() -> None:
    """The patch reaches ``asyncio.sleep`` too, via the shared module object."""
    started = time.monotonic()
    await asyncio.sleep(_LONG_ENOUGH_TO_NOTICE)
    assert time.monotonic() - started < _NOOP_CEILING


def test_real_sleep_fixture_actually_sleeps(real_sleep: Callable[[float], None]) -> None:
    started = time.monotonic()
    real_sleep(_SHORT_REAL_WAIT)
    elapsed = time.monotonic() - started
    assert elapsed >= _SHORT_REAL_WAIT, f"real_sleep returned after only {elapsed:.4f}s"


async def test_real_async_sleep_fixture_actually_sleeps(
    real_async_sleep: Callable[[float], Awaitable[None]],
) -> None:
    started = time.monotonic()
    await real_async_sleep(_SHORT_REAL_WAIT)
    assert time.monotonic() - started >= _SHORT_REAL_WAIT


class TestSuppressRetrySleepOptOut:
    """Overriding ``suppress_retry_sleep`` leaves both sleeps unpatched.

    Declared on a class here to keep the opt-out off the rest of the file;
    a module needing it for every test declares the same fixture at module
    level, as the retry-storm modules do.
    """

    @pytest.fixture
    def suppress_retry_sleep(self) -> bool:
        return False

    def test_time_sleep_is_real_when_opted_out(self) -> None:
        started = time.monotonic()
        time.sleep(_SHORT_REAL_WAIT)
        elapsed = time.monotonic() - started
        assert elapsed >= _SHORT_REAL_WAIT, (
            f"time.sleep returned after {elapsed:.4f}s — suppress_retry_sleep=False "
            "did not disable the patch"
        )

    async def test_asyncio_sleep_is_real_when_opted_out(self) -> None:
        started = time.monotonic()
        await asyncio.sleep(_SHORT_REAL_WAIT)
        assert time.monotonic() - started >= _SHORT_REAL_WAIT
