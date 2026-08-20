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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def wait_for_ready(
    check_fn: object,
    *,
    timeout: int = 300,
    interval: int = 5,
    description: str = "resource",
) -> None:
    """Poll until check_fn() returns True or timeout expires."""
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
