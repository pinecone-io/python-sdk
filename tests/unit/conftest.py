"""Unit-test-wide fixtures.

Both autouse fixtures here exist so the unit suite — the CI gate — does
not depend on the machine it runs on. Integration tests
(tests/integration/) are unaffected because this conftest is scoped to
tests/unit/.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

_PINECONE_ENV_PREFIX = "PINECONE_"


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


@pytest.fixture(autouse=True)
def _no_retry_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip real sleeps inside _RetryTransport and _AsyncRetryTransport.

    Tests that assert on sleep call counts (test_retry.py) layer their
    own @patch("pinecone._internal.http_client.time.sleep") on top of
    this autouse fixture; pytest applies the test-local patch last so
    those Mock assertions remain valid.
    """

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
