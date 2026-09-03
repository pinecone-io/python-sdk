"""Unit-test-wide fixtures.

The autouse fixtures here exist so the unit suite — the CI gate — does
not depend on the machine it runs on: no ambient ``PINECONE_*``
configuration, no real retry backoff, and no process-global registry
state carried between tests. Integration tests (tests/integration/) are
unaffected because this conftest is scoped to tests/unit/.

**If you are writing a test that depends on the retry transport's own
backoff, read ``_no_retry_sleep`` below first.** It replaces the
``_retry_sleep`` / ``_async_retry_sleep`` seams in
``pinecone._internal.http_client``, so retries advance no clock. Your own
``time.sleep`` is untouched; where you want the retry path to wait for
real, opt the module out via ``suppress_retry_sleep``, and ``real_sleep`` /
``real_async_sleep`` hand you sleep callables no patch can reach.
"""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Awaitable, Callable, Generator, Iterator
from typing import Any

import pytest

_PINECONE_ENV_PREFIX = "PINECONE_"

# Bound at import, before any test can reassign the module attributes, so the
# real_sleep fixtures hand out callables the autouse patch below cannot reach.
_REAL_TIME_SLEEP: Callable[[float], None] = time.sleep
_REAL_ASYNCIO_SLEEP: Callable[[float], Awaitable[None]] = asyncio.sleep


@pytest.fixture(scope="session", autouse=True)
def _hermetic_pinecone_env_session() -> Iterator[None]:
    """Scrub ``PINECONE_*`` before any module- or session-scoped fixture runs (#426).

    ``_hermetic_pinecone_env`` below is function-scoped and ``hermetic_pinecone_env_module``
    is opt-in, so either one can be outrun: pytest instantiates higher-scoped
    fixtures first, so a *new* module- or session-scoped fixture that depends
    on neither — the shape of #345's shared property-test clients — sees the
    developer's ambient environment before either scrub gets a chance to act.
    Relying on every future fixture author to remember the opt-in is exactly
    the gap Cursor Bugbot caught twice and we didn't.

    Session scope is the one thing a module- or session-scoped fixture cannot
    outrun: it is autouse, so it is requested by the very first test collected
    under ``tests/unit/``, and pytest sets same-or-higher-scoped fixtures up
    before it gets to any module-scoped one in that same closure. Held for the
    whole session via ``MonkeyPatch.context()`` — like ``hermetic_pinecone_env_module``
    below — so a variable removed here stays removed; it is not re-added by
    this fixture, only by something later in the session (a lazy ``load_env()``
    import, a test that sets one itself), which the per-test scrub still exists
    to undo before the next test body runs.
    """
    with pytest.MonkeyPatch.context() as monkeypatch:
        for name in [n for n in os.environ if n.startswith(_PINECONE_ENV_PREFIX)]:
            monkeypatch.delenv(name, raising=False)
        yield


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
    pollution introduced at collection time — pollution introduced *after*
    session start, which ``_hermetic_pinecone_env_session`` above cannot see
    because it only runs once. Tests that want a variable set it themselves
    with ``monkeypatch.setenv``; pytest runs that after this fixture, so
    those tests keep working.
    """
    for name in [n for n in os.environ if n.startswith(_PINECONE_ENV_PREFIX)]:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(scope="module")
def hermetic_pinecone_env_module() -> Iterator[None]:
    """The same scrub as ``_hermetic_pinecone_env``, for module-scoped fixtures.

    Ask for this from any module- or session-scoped fixture that builds an SDK
    object and wants the scrub to run at its *own* setup point rather than
    relying on ``_hermetic_pinecone_env_session`` having already run once at
    session start — for instance, a fixture that needs a variable it just set
    with ``monkeypatch.setenv`` to be visible to itself but not leak beyond its
    module. ``_hermetic_pinecone_env`` above is function-scoped, and pytest
    sets higher-scoped fixtures up **first** — so a module-scoped fixture runs
    before that scrub and sees the developer's ambient ``PINECONE_*``
    variables. A client built there bakes them in for every test in the
    module: measured on #345's shared property-test clients,
    ``PINECONE_ADDITIONAL_HEADERS`` landed in the client's header set that way.

    Depending on this fixture puts the scrub back in front of the construction.
    It does not replace the per-test one, which still has to run to undo
    pollution introduced at collection time.
    """
    with pytest.MonkeyPatch.context() as monkeypatch:
        for name in [n for n in os.environ if n.startswith(_PINECONE_ENV_PREFIX)]:
            monkeypatch.delenv(name, raising=False)
        yield


@pytest.fixture
def real_sleep() -> Callable[[float], None]:
    """A ``time.sleep`` bound before any patching can reach it (#360).

    Ask for this fixture when your test's own logic needs wall-clock time
    to pass — a concurrency soak, a race window, a deadline — and you want
    that guaranteed independently of what any fixture, here or in a module,
    has done to ``time.sleep``.

    Pair it with a ``time.monotonic()`` assertion on the elapsed time:
    ``time.monotonic`` is never patched, so it is the clock that can prove
    the wait really happened.
    """
    return _REAL_TIME_SLEEP


@pytest.fixture
def real_async_sleep() -> Callable[[float], Awaitable[None]]:
    """An ``asyncio.sleep`` bound the same way. See ``real_sleep`` (#360)."""
    return _REAL_ASYNCIO_SLEEP


@pytest.fixture
def suppress_retry_sleep() -> bool:
    """Whether ``_no_retry_sleep`` patches the sleeps for this test.

    Override it with ``False`` in a module to keep real retry backoff for
    every test in that file — needed when the *production* retry path under
    test must actually wait, as in the retry-storm dispersal tests::

        @pytest.fixture
        def suppress_retry_sleep() -> bool:
            return False

    Opting out makes every retry in the module pay real backoff, so scope it
    to the modules that measure dispersal.

    The retry-storm modules predate this fixture and opt out by shadowing
    ``_no_retry_sleep`` itself with a no-op. That still works; this is the
    supported spelling because it does not require knowing how the patch is
    implemented.
    """
    return True


@pytest.fixture(autouse=True)
def _no_retry_sleep(monkeypatch: pytest.MonkeyPatch, suppress_retry_sleep: bool) -> None:
    """Skip real sleeps inside _RetryTransport and _AsyncRetryTransport.

    The targets are ``pinecone._internal.http_client._retry_sleep`` and
    ``_async_retry_sleep``, the module-level seams the retry transports call
    (#79). Patching the seams — rather than ``time.sleep`` and
    ``asyncio.sleep`` on the shared module objects, which no-oped every sleep
    in the process and let a 0.5s soak run for 0.0000s and look green (#45,
    #360) — keeps the no-op inside the retry transport, so a test that
    measures its own elapsed time still measures something.

    Tests that assert on sleep call counts (test_retry.py) layer their own
    @patch("pinecone._internal.http_client._retry_sleep") on the same seam;
    pytest applies the test-local patch last, so those Mock assertions
    remain valid.

    To keep real backoff: ``suppress_retry_sleep = False`` for a whole
    module. ``real_sleep`` / ``real_async_sleep`` hand a test sleep
    callables that no patching can reach.
    """
    if not suppress_retry_sleep:
        return

    def _noop(*_args: Any, **_kwargs: Any) -> None:
        return None

    async def _noop_async(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(
        "pinecone._internal.http_client._retry_sleep",
        _noop,
    )
    monkeypatch.setattr(
        "pinecone._internal.http_client._async_retry_sleep",
        _noop_async,
    )


@pytest.fixture(autouse=True)
def _fresh_retry_budgets() -> Generator[None, None, None]:
    """The retry-budget registry is process-global; without a reset, a
    failure-heavy test drains the shared bucket and silently disables
    retries in whatever test runs next (the gate-registry lesson, again)."""
    from pinecone._internal.http_client import get_budget_registry

    get_budget_registry()._reset()
    yield
    get_budget_registry()._reset()


@pytest.fixture(autouse=True)
def _fresh_gate_registry() -> Generator[None, None, None]:
    """The gate registry is process-global too. The bulk package's own
    conftest additionally asserts quiescence; this suite-wide reset only
    guarantees isolation — a halved limit or a stalled gate left by one test
    must not bleed into an unrelated test file (issue #156)."""
    from pinecone._internal.bulk.registry import get_registry

    get_registry()._reset()
    yield
    get_registry()._reset()
