"""Unit-test-wide fixtures.

This autouse fixture eliminates real wall-clock cost from retry backoff
in _RetryTransport / _AsyncRetryTransport. Unit tests that mock
httpx.TransportError or retryable status codes would otherwise pay
0.3-3.0s of real time.sleep / asyncio.sleep per test. Integration
tests (tests/integration/) are unaffected because this conftest is
scoped to tests/unit/.
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

import pytest


@pytest.fixture(autouse=True)
def _no_retry_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip real sleeps inside _RetryTransport and _AsyncRetryTransport.

    Tests that assert on sleep call counts (test_retry.py) layer their
    own @patch("pinecone._internal.http_client._retry_sleep") on top of
    this autouse fixture; pytest applies the test-local patch last so
    those Mock assertions remain valid.
    """

    async def _noop_async(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(
        "pinecone._internal.http_client._retry_sleep",
        lambda *_a, **_kw: None,
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
