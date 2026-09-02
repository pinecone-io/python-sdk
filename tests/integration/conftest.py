"""Shared fixtures for integration tests.

These tests make real API calls to Pinecone and require a .env file
at the SDK root with PINECONE_API_KEY set:

    echo 'PINECONE_API_KEY=your-api-key' > .env
    uv run pytest tests/integration/ -v -s

The .env is looked up in the **main working tree**, so runs from a git
worktree pick up the same file (see ``tests.live_suite.env_candidates``).
Override the location with ``PINECONE_SDK_ENV_FILE=/path/to/.env``.

Credential-gated groups
-----------------------
Every group below skips itself when its variable is unset. Only
``PINECONE_API_KEY`` is expected to be present in a normal run; the rest are
**deliberately out-of-band** and stay skipped locally and in CI:

``PINECONE_API_KEY``
    Gates ~543 of ~648 collected tests (~94%). Required for any meaningful
    run. Read from .env or the environment.
``PINECONE_DOCUMENTS_INDEX_HOST``
    Gates 14 tests (``test_documents.py``, ``test_async_documents.py``). Needs
    a pre-provisioned schema-based index host; out-of-band because the suite
    cannot create one for itself.
``PINECONE_CLIENT_ID`` / ``PINECONE_CLIENT_SECRET``
    Gates 10 tests (``test_admin.py``). Needs a service-account OAuth client;
    out-of-band because those credentials are org-admin scoped and are
    deliberately not distributed alongside the data-plane key.
``PINECONE_RETRY_SMOKE=1``
    Gates 3 tests (``test_retry_smoke.py``). Opt-in: drives live retry/backoff
    behavior and is slow by design.
``RUN_EXPENSIVE_TESTS=1``
    Gates 2 tests (``test_admin.py``). Opt-in: creates real cloud resources.

Two further tests are skipped unconditionally pending IPV-0004
(documents/delete returns 401 Unknown operation).

A run's actual pass/skip breakdown is printed at the end of the session by
``pytest_terminal_summary`` — a mostly-skipped run says so out loud rather
than reporting a misleading green.
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from collections.abc import AsyncGenerator, Generator
from pathlib import Path
from typing import Protocol

import pytest

from pinecone import AsyncPinecone, Pinecone
from pinecone._internal.constants import DEFAULT_BASE_URL
from tests.integration.legacy_index import (
    LegacyIndex,
    create_legacy_index,
    delete_legacy_index,
)
from tests.live_suite import load_env, write_coverage_summary


class LegacyIndexFactory(Protocol):
    def __call__(
        self,
        *,
        dimension: int | None = None,
        metric: str = "cosine",
        vector_type: str = "dense",
    ) -> LegacyIndex: ...


_HERE = Path(__file__).resolve().parent

# The global `timeout` in pyproject.toml is sized for the unit suite, but also
# applies here and silently overrides each test's own poll budget — nearly every
# test in this directory asks poll_until() to wait longer than 60s, so they only
# pass when the control plane happens to be fast. poll_until() already bounds
# itself with a descriptive TimeoutError, leaving pytest-timeout as a backstop
# for hangs *outside* polling; size it above the largest declared budget
# (currently 1020s) rather than below the smallest.
_DEFAULT_TIMEOUT_SECONDS = int(os.environ.get("PINECONE_TEST_TIMEOUT", "1800"))


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "integration: marks tests as real-API integration tests")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Size the timeout for a live backend (#306) and reject asyncio marks (#313).

    The global ini ``timeout`` is sized for unit tests, which mock every sleep
    away; a real round trip cannot honour it. Tests carrying their own marker
    are skipped, so the explicit values in this tree keep winning either way.

    Path-filtered because pytest hands a conftest hook the entire session's
    item list, not only the items collected beneath that conftest — an
    unfiltered loop would hand ``tests/unit`` the integration default too, and
    would reject ``tests/smoke``'s remaining asyncio marks as if they were ours.
    """
    offenders = []
    for item in items:
        if _HERE not in item.path.parents:
            continue
        if item.get_closest_marker("timeout") is None:
            item.add_marker(pytest.mark.timeout(_DEFAULT_TIMEOUT_SECONDS))
        if item.get_closest_marker("asyncio") is not None:
            offenders.append(item.nodeid)

    if offenders:
        listing = "\n  ".join(offenders)
        raise pytest.UsageError(
            "pytest.mark.asyncio is not allowed in tests/integration (#313):\n  "
            f"{listing}\n"
            "pytest-asyncio closes the event loop before running async fixture "
            "teardown, so the `await pc.close()` in the `async_client` fixture "
            "below raises 'RuntimeError: Event loop is closed' and every test "
            "sharing the fixture errors in teardown. Use pytest.mark.anyio, "
            "which sequences fixture finalization inside the loop's lifetime. "
            "See commit bd074083."
        )


_ENV_SOURCE = load_env(_HERE)

_CREDENTIAL_VARS = (
    "PINECONE_API_KEY",
    "PINECONE_CLIENT_ID",
    "PINECONE_CLIENT_SECRET",
    "PINECONE_DOCUMENTS_INDEX_HOST",
)


def pytest_terminal_summary(
    terminalreporter: pytest.TerminalReporter,
    exitstatus: int,
    config: pytest.Config,
) -> None:
    """Report what actually ran.

    Without this, a fully credential-starved run exits 0 with a wall of skips
    that nobody counts, and "integration tests green" gets read as coverage.
    See #295.
    """
    write_coverage_summary(
        terminalreporter,
        label="integration",
        env_source=_ENV_SOURCE,
        credential_vars=_CREDENTIAL_VARS,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MAX_POLL_TIMEOUT = float(os.environ.get("PINECONE_TEST_MAX_POLL_TIMEOUT", "0")) or None


def _capped_timeout(timeout: int) -> int:
    """Shrink a poll timeout under PINECONE_TEST_MAX_POLL_TIMEOUT, if set.

    Against a synchronous backend like minicone, a condition that's ever going
    to become true does so on the first check; a failing test otherwise burns
    its full real-API-sized timeout (up to 300s) doing nothing.
    """
    if _MAX_POLL_TIMEOUT is None:
        return timeout
    return int(min(timeout, _MAX_POLL_TIMEOUT))


def _capped_interval(interval: int) -> int | float:
    if _MAX_POLL_TIMEOUT is None:
        return interval
    return min(interval, max(_MAX_POLL_TIMEOUT / 10, 0.5))


def wait_for_ready(
    check_fn: object,
    *,
    timeout: int = 300,
    interval: int = 5,
    description: str = "resource",
) -> None:
    """Poll until check_fn() returns True or timeout expires."""
    timeout, interval = _capped_timeout(timeout), _capped_interval(interval)
    start = time.time()
    while time.time() - start < timeout:
        try:
            if check_fn():  # type: ignore[operator]
                return
        except Exception:
            pass
        time.sleep(interval)
    raise TimeoutError(f"{description} not ready after {timeout}s")


def poll_until(
    query_fn: object,
    check_fn: object,
    *,
    timeout: int = 60,
    interval: int = 3,
    description: str = "condition",
) -> object:
    """Poll query_fn() until check_fn(result) is True. Returns the final result."""
    timeout, interval = _capped_timeout(timeout), _capped_interval(interval)
    start = time.time()
    last_result = None
    while time.time() - start < timeout:
        try:
            last_result = query_fn()  # type: ignore[operator]
            if check_fn(last_result):  # type: ignore[operator]
                return last_result
        except Exception:
            pass
        time.sleep(interval)
    raise TimeoutError(f"{description} not satisfied after {timeout}s (last result: {last_result})")


def unique_name(prefix: str = "inttest") -> str:
    """Generate a unique resource name using timestamp + random suffix."""
    short_uuid = uuid.uuid4().hex[:8]
    return f"{prefix}-{int(time.time())}-{short_uuid}"


def cleanup_resource(
    delete_fn: object,
    resource_id: str,
    resource_type: str = "resource",
) -> None:
    """Best-effort cleanup of a named resource. Logs but never raises."""
    try:
        delete_fn()  # type: ignore[operator]
        print(f"  Cleaned up {resource_type}: {resource_id}")
    except Exception as exc:
        print(f"  WARNING: Failed to clean up {resource_type} {resource_id}: {exc}")


def ensure_index_deleted(
    client: Pinecone,
    name: str,
    *,
    timeout: int = 120,
    interval: int = 3,
) -> None:
    """Delete an index and poll until it disappears. Best-effort; never raises.

    Unlike ``cleanup_resource``, this waits for the backend to finish the
    asynchronous delete so the name is released before the test returns,
    which reduces cross-test index-quota flakes.

    Iterate the paginator; do not reach for a ``.indexes`` attribute. It has
    none, so every poll raised and this helper leaked every index it was
    asked to delete (#346).
    """
    timeout, interval = _capped_timeout(timeout), _capped_interval(interval)
    try:
        client.indexes.delete(name)
    except Exception as exc:
        print(f"  WARNING: delete call failed for index {name}: {exc}")

    start = time.monotonic()
    while time.monotonic() - start < timeout:
        try:
            existing = {i.name for i in client.indexes.list()}
            if name not in existing:
                print(f"  Cleaned up index: {name}")
                return
        except Exception as exc:
            print(f"  WARNING: indexes.list() failed during cleanup of {name}: {exc}")
        time.sleep(interval)

    print(f"  WARNING: index {name} still present after {timeout}s — may leak quota")


async def async_cleanup_resource(
    delete_fn: object,
    resource_id: str,
    resource_type: str = "resource",
) -> None:
    """Async best-effort cleanup. Logs but never raises."""
    try:
        await delete_fn()  # type: ignore[operator]
        print(f"  Cleaned up {resource_type}: {resource_id}")
    except Exception as exc:
        print(f"  WARNING: Failed to clean up {resource_type} {resource_id}: {exc}")


async def async_ensure_index_deleted(
    async_client: AsyncPinecone,
    name: str,
    *,
    timeout: int = 120,
    interval: int = 3,
) -> None:
    """Async version of :func:`ensure_index_deleted`. Best-effort; never raises."""
    timeout, interval = _capped_timeout(timeout), _capped_interval(interval)
    try:
        await async_client.indexes.delete(name)
    except Exception as exc:
        print(f"  WARNING: delete call failed for index {name}: {exc}")

    start = time.monotonic()
    while time.monotonic() - start < timeout:
        try:
            existing = {i.name async for i in async_client.indexes.list()}
            if name not in existing:
                print(f"  Cleaned up index: {name}")
                return
        except Exception as exc:
            print(f"  WARNING: indexes.list() failed during cleanup of {name}: {exc}")
        await asyncio.sleep(interval)

    print(f"  WARNING: index {name} still present after {timeout}s — may leak quota")


async def async_poll_until(
    query_fn: object,
    check_fn: object,
    *,
    timeout: int = 60,
    interval: int = 3,
    description: str = "condition",
) -> object:
    """Async version of poll_until."""
    timeout, interval = _capped_timeout(timeout), _capped_interval(interval)
    start = time.time()
    last_result = None
    while time.time() - start < timeout:
        try:
            last_result = await query_fn()  # type: ignore[operator]
            if check_fn(last_result):  # type: ignore[operator]
                return last_result
        except Exception:
            pass
        await asyncio.sleep(interval)
    raise TimeoutError(f"{description} not satisfied after {timeout}s (last result: {last_result})")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def api_key() -> str:
    """Pinecone API key from environment. Skips all tests if not set."""
    key = os.getenv("PINECONE_API_KEY")
    if not key:
        pytest.skip("PINECONE_API_KEY not set")
    return key


@pytest.fixture(scope="session")
def client(api_key: str) -> Pinecone:
    """Session-scoped Pinecone client."""
    return Pinecone(api_key=api_key)


@pytest.fixture
def client_pool() -> Pinecone:
    """Pinecone client with pool_threads set, opting into legacy
    async_req=True execution.
    """
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        pytest.skip("PINECONE_API_KEY not set")
    return Pinecone(api_key=api_key, pool_threads=4)


@pytest.fixture(scope="session")
def legacy_index_factory(api_key: str) -> Generator[LegacyIndexFactory, None, None]:
    """Create legacy (vectors-API) indexes, one per distinct shape.

    2026-07 cannot create an index the vectors API will serve, so any test
    that expects ``upsert`` / ``query`` / ``fetch`` to **succeed** has to get
    its index from :mod:`tests.integration.legacy_index` instead of
    ``pc.indexes.create``. See that module for the sanctioned pattern and for
    why the SDK's own create path is bypassed.

    Call it as ``legacy_index_factory(dimension=3)``, or
    ``legacy_index_factory(vector_type="sparse", metric="dotproduct")``.
    Results are cached per shape for the whole session and deleted at the end,
    so callers sharing a shape share one index — isolate with a per-test
    namespace, as the rest of this package does.

    Reads ``PINECONE_CONTROLLER_HOST`` so a run pointing the SDK at a
    non-default host (e.g. a local simulator) also creates the legacy index
    there instead of against production.
    """
    base_url = os.environ.get("PINECONE_CONTROLLER_HOST", DEFAULT_BASE_URL)
    created: dict[tuple[int | None, str, str], LegacyIndex] = {}

    def factory(
        *,
        dimension: int | None = None,
        metric: str = "cosine",
        vector_type: str = "dense",
    ) -> LegacyIndex:
        key = (dimension, metric, vector_type)
        if key not in created:
            created[key] = create_legacy_index(
                api_key,
                dimension=dimension,
                metric=metric,
                vector_type=vector_type,
                base_url=base_url,
            )
        return created[key]

    try:
        yield factory
    finally:
        for index in created.values():
            delete_legacy_index(api_key, index.name, base_url=base_url)


@pytest.fixture(scope="session")
def legacy_index_dim3(legacy_index_factory: LegacyIndexFactory) -> LegacyIndex:
    """A ready dim-3 cosine legacy index — the default shape for vector tests."""
    return legacy_index_factory(dimension=3)


@pytest.fixture
async def async_client(api_key: str) -> AsyncGenerator[AsyncPinecone, None]:
    """Function-scoped async Pinecone client (REST).

    Plain ``@pytest.fixture`` so pytest-anyio (anyio_mode="auto") owns both
    the test and fixture event-loop lifecycle. Integration tests use
    ``@pytest.mark.anyio`` (not ``@pytest.mark.asyncio``) so pytest-asyncio
    does not double-collect them, which would cause each test to run twice and
    exhaust index quota. See CI-0019 for context.
    """
    pc = AsyncPinecone(api_key=api_key)
    try:
        yield pc
    finally:
        await pc.close()
