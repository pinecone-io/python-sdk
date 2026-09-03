"""Guards the retry-sleep patch in conftest.py and its escape hatches (#360, #79).

``tests/unit/conftest.py``'s autouse ``_no_retry_sleep`` no-ops the
``_retry_sleep`` / ``_async_retry_sleep`` seams in
``pinecone._internal.http_client``, which is what keeps retry-backoff tests
fast. It patches **only** those seams: patching ``time.sleep`` and
``asyncio.sleep`` process-wide is the #45 bug #79 fixed — it no-oped every
sleep in the process and let a 0.5s concurrency soak in ``test_adaptive.py``
run for 0.0000s and pass. So these tests pin three things: the seams really
are no-oped by default, the process-wide clocks really are left alone, and
``real_sleep`` / ``real_async_sleep`` / ``suppress_retry_sleep`` still get a
test out of the patch.

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

from pinecone._internal import http_client

_LONG_ENOUGH_TO_NOTICE = 0.5
_SHORT_REAL_WAIT = 0.05
_NOOP_CEILING = 0.05


def test_monotonic_is_not_patched() -> None:
    """Ground truth for every other assertion in this file."""
    started = time.monotonic()
    while time.monotonic() - started < _SHORT_REAL_WAIT:
        pass
    assert time.monotonic() - started >= _SHORT_REAL_WAIT


def test_retry_sleep_seam_is_a_noop_by_default() -> None:
    started = time.monotonic()
    http_client._retry_sleep(_LONG_ENOUGH_TO_NOTICE)
    elapsed = time.monotonic() - started
    assert elapsed < _NOOP_CEILING, (
        f"_retry_sleep({_LONG_ENOUGH_TO_NOTICE}) took {elapsed:.4f}s — the seam "
        "patch is gone, and the retry-backoff tests are now paying real time"
    )


async def test_async_retry_sleep_seam_is_also_a_noop_by_default() -> None:
    """The autouse patch covers the async seam too."""
    started = time.monotonic()
    await http_client._async_retry_sleep(_LONG_ENOUGH_TO_NOTICE)
    assert time.monotonic() - started < _NOOP_CEILING


def test_process_wide_time_sleep_is_left_alone() -> None:
    """The other half of #79: the patch must not reach ``time.sleep`` itself.

    A process-wide no-op is the #45 bug — it makes every self-timed test
    measure nothing. Restoring it turns five real-time tests green-for-nothing
    (bulk/test_engine, bulk/test_async_engine, bulk/test_stall_recovery x2,
    grpc/test_grpc_upsert_batching), which is how it was caught.
    """
    started = time.monotonic()
    time.sleep(_SHORT_REAL_WAIT)
    elapsed = time.monotonic() - started
    assert elapsed >= _SHORT_REAL_WAIT, (
        f"time.sleep returned after {elapsed:.4f}s — something is patching the "
        "process-wide clock, and self-timed tests now measure nothing (#45)"
    )


async def test_process_wide_asyncio_sleep_is_left_alone() -> None:
    """As above, for ``asyncio.sleep``."""
    started = time.monotonic()
    await asyncio.sleep(_SHORT_REAL_WAIT)
    assert time.monotonic() - started >= _SHORT_REAL_WAIT


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
    """Overriding ``suppress_retry_sleep`` leaves both seams unpatched.

    Declared on a class here to keep the opt-out off the rest of the file;
    a module needing it for every test declares the same fixture at module
    level, as the retry-storm modules do.
    """

    @pytest.fixture
    def suppress_retry_sleep(self) -> bool:
        return False

    def test_retry_sleep_seam_is_real_when_opted_out(self) -> None:
        started = time.monotonic()
        http_client._retry_sleep(_SHORT_REAL_WAIT)
        elapsed = time.monotonic() - started
        assert elapsed >= _SHORT_REAL_WAIT, (
            f"_retry_sleep returned after {elapsed:.4f}s — suppress_retry_sleep=False "
            "did not disable the patch"
        )

    async def test_async_retry_sleep_seam_is_real_when_opted_out(self) -> None:
        started = time.monotonic()
        await http_client._async_retry_sleep(_SHORT_REAL_WAIT)
        assert time.monotonic() - started >= _SHORT_REAL_WAIT
