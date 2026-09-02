"""Shared fixtures for integration tests.

These tests make real API calls to Pinecone and require a .env file
at the SDK root with PINECONE_API_KEY set:

    echo 'PINECONE_API_KEY=your-api-key' > .env
    cd sdks/python-sdk2 && uv run --with python-dotenv pytest tests/integration/ -v -s
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
from dotenv import load_dotenv

from pinecone import AsyncPinecone, Pinecone


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "integration: marks tests as real-API integration tests")


# Load .env from the SDK root (two levels up from tests/integration/)
_env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(_env_path)


# The global `timeout = 60` in pyproject.toml is sized for the unit suite, but
# also applies here and silently overrides each test's own poll budget — nearly
# every test in this directory asks poll_until() to wait longer than 60s, so
# they only pass when the control plane happens to be fast. poll_until() already
# bounds itself with a descriptive TimeoutError, leaving pytest-timeout as a
# backstop for hangs *outside* polling; size it above the largest declared
# budget (currently 1020s) rather than below the smallest.
_DEFAULT_TIMEOUT = int(os.environ.get("PINECONE_TEST_TIMEOUT", "1800"))

_HERE = Path(__file__).resolve().parent


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        path = getattr(item, "path", None)
        if path is None:
            continue
        try:
            Path(path).resolve().relative_to(_HERE)
        except ValueError:
            continue
        if item.get_closest_marker("timeout") is None:
            item.add_marker(pytest.mark.timeout(_DEFAULT_TIMEOUT))


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
    """
    timeout, interval = _capped_timeout(timeout), _capped_interval(interval)
    try:
        client.indexes.delete(name)
    except Exception as exc:
        print(f"  WARNING: delete call failed for index {name}: {exc}")

    start = time.monotonic()
    while time.monotonic() - start < timeout:
        try:
            listing = client.indexes.list()
            existing = {i.name for i in listing.indexes}
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
            listing = await async_client.indexes.list()
            existing = {i.name for i in listing.indexes}
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
