"""Unit-test-wide fixtures.

The autouse fixtures here exist so the unit suite — the CI gate — does
not depend on the machine it runs on. Integration tests
(tests/integration/) are unaffected because this conftest is scoped to
tests/unit/.

**If you are writing a test that depends on real elapsed time, read
``_no_retry_sleep`` below first.** It replaces ``time.sleep`` and
``asyncio.sleep`` process-wide, so a sleep in your test does nothing and a
sleep-based probe proves nothing. Use the ``real_sleep`` /
``real_async_sleep`` fixtures, or opt the whole module out via
``suppress_retry_sleep``.
"""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Awaitable, Callable
from typing import Any

import pytest

_PINECONE_ENV_PREFIX = "PINECONE_"

# Bound at import, before any test can reassign the module attributes, so the
# real_sleep fixtures hand out callables the autouse patch below cannot reach.
_REAL_TIME_SLEEP: Callable[[float], None] = time.sleep
_REAL_ASYNCIO_SLEEP: Callable[[float], Awaitable[None]] = asyncio.sleep


@pytest.fixture(autouse=True)
def _hermetic_pinecone_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hide every ``PINECONE_*`` environment variable from unit tests (#353).

    The SDK falls back to the environment for credentials, host, headers
    and service-account secrets, so a developer with ``PINECONE_API_KEY``
    exported for live-suite work saw unit failures that were not theirs —
    a CI gate whose result depended on the ambient environment. Worse,
    importing tests/integration/conftest.py from a unit test runs its
    module-level ``load_env()``, which puts a real key into ``os.environ``
    for the rest of the session.

    Scrubbing by prefix, per test, closes both routes: it covers variables
    added to the SDK later, and because it runs per test it also undoes
    pollution introduced at collection time. Tests that want a variable
    set it themselves with ``monkeypatch.setenv``; pytest runs that after
    this fixture, so those tests keep working.
    """
    for name in [n for n in os.environ if n.startswith(_PINECONE_ENV_PREFIX)]:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def real_sleep() -> Callable[[float], None]:
    """The genuine ``time.sleep``, unaffected by ``_no_retry_sleep`` (#360).

    Ask for this fixture whenever your test's own logic needs wall-clock
    time to pass — a concurrency soak, a race window, a deadline. Calling
    ``time.sleep`` directly returns instantly and the test silently
    measures nothing.

    Pair it with a ``time.monotonic()`` assertion on the elapsed time.
    ``time.monotonic`` is *not* patched, so it is the one clock that can
    prove the wait really happened; a sleep-based probe cannot, because it
    is measuring the patch.
    """
    return _REAL_TIME_SLEEP


@pytest.fixture
def real_async_sleep() -> Callable[[float], Awaitable[None]]:
    """The genuine ``asyncio.sleep``. See ``real_sleep`` (#360).

    Note that the patched ``asyncio.sleep`` never yields to the event loop,
    so ``await asyncio.sleep(0)`` does not hand control to other tasks
    under ``tests/unit/`` either.
    """
    return _REAL_ASYNCIO_SLEEP


@pytest.fixture
def suppress_retry_sleep() -> bool:
    """Whether ``_no_retry_sleep`` patches the sleeps for this test.

    Override it with ``False`` in a module to keep real sleeps for every
    test in that file — needed when the *production* code path under test
    must actually wait, as in the retry-storm dispersal tests::

        @pytest.fixture
        def suppress_retry_sleep() -> bool:
            return False

    Prefer the ``real_sleep`` fixture when it is only your test body that
    needs to wait; opting out makes every retry in the module pay real
    backoff.

    The retry-storm modules predate this fixture and opt out by shadowing
    ``_no_retry_sleep`` itself with a no-op. That still works; this is the
    supported spelling because it does not require knowing how the patch is
    implemented.
    """
    return True


@pytest.fixture(autouse=True)
def _no_retry_sleep(monkeypatch: pytest.MonkeyPatch, suppress_retry_sleep: bool) -> None:
    """Skip real sleeps inside _RetryTransport and _AsyncRetryTransport.

    **This patch is global, not scoped to the SDK.** The targets read as
    ``pinecone._internal.http_client.time.sleep`` /
    ``...http_client.asyncio.sleep``, but those attributes resolve to the
    shared ``time`` and ``asyncio`` *module objects*, so for the duration
    of every unit test ``time.sleep`` and ``asyncio.sleep`` are no-ops
    everywhere in the process — including inside the test itself. That
    cost us a 0.5s concurrency soak that ran for 0.0000s and looked green
    (#360), and it defeats any attempt to verify timing with a sleep probe.

    To escape it: ``real_sleep`` / ``real_async_sleep`` for a test that
    needs to wait, or ``suppress_retry_sleep = False`` to disable the patch
    for a whole module. Tests that assert on sleep call counts
    (test_retry.py) instead layer their own
    @patch("pinecone._internal.http_client.time.sleep") on top of this
    fixture; pytest applies the test-local patch last so those Mock
    assertions remain valid.
    """
    if not suppress_retry_sleep:
        return

    async def _noop_async(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(
        "pinecone._internal.http_client.time.sleep",
        lambda *_a, **_kw: None,
    )
    monkeypatch.setattr(
        "pinecone._internal.http_client.asyncio.sleep",
        _noop_async,
    )
