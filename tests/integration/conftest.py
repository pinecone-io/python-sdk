"""Shared fixtures for integration tests.

These tests make real API calls to Pinecone and require a .env file
at the SDK root with PINECONE_API_KEY set:

    echo 'PINECONE_API_KEY=your-api-key' > .env
    uv run pytest tests/integration/ -v -s

The .env is looked up in the **main working tree**, so runs from a git
worktree pick up the same file (see ``_env_candidates``). Override the
location with ``PINECONE_SDK_ENV_FILE=/path/to/.env``.

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
import shutil
import subprocess
import time
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
from dotenv import load_dotenv

from pinecone import AsyncPinecone, Pinecone

_HERE = Path(__file__).resolve().parent

_DEFAULT_TIMEOUT_SECONDS = 120


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "integration: marks tests as real-API integration tests")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Replace the global 5s timeout with a live-backend-sized one here (#306).

    ``timeout = 5`` is sized for unit tests, which mock every sleep away; a
    real round trip cannot honour it. Tests carrying their own marker are
    skipped, so the explicit values in this tree keep winning either way.

    Path-filtered because pytest hands a conftest hook the entire session's
    item list, not only the items collected beneath that conftest — an
    unfiltered loop would hand ``tests/unit`` the integration default too.
    """
    for item in items:
        if _HERE not in item.path.parents:
            continue
        if item.get_closest_marker("timeout") is None:
            item.add_marker(pytest.mark.timeout(_DEFAULT_TIMEOUT_SECONDS))


def _main_worktree_root() -> Path | None:
    """Repo root of the **main** working tree, even when run from a worktree.

    A linked worktree's ``.git`` is a file pointing into
    ``<main>/.git/worktrees/<name>``, so the current tree's root is *not* the
    repo root that holds ``.env``. ``git rev-parse --git-common-dir`` resolves
    back to the real ``.git`` directory in both cases; its parent is the main
    checkout root. Returns ``None`` outside a git checkout.
    """
    git = shutil.which("git")
    if git is None:
        return None
    try:
        proc = subprocess.run(  # noqa: S603 — fixed argv, git resolved via shutil.which
            [git, "rev-parse", "--git-common-dir"],
            cwd=_HERE,
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    raw = proc.stdout.strip()
    if not raw:
        return None
    # Relative in the main tree (e.g. "../../.git"), absolute in a worktree.
    common_dir = Path(raw)
    if not common_dir.is_absolute():
        common_dir = (_HERE / common_dir).resolve()
    return common_dir.parent


def _env_candidates() -> list[Path]:
    """.env locations to try, most specific first."""
    override = os.getenv("PINECONE_SDK_ENV_FILE")
    if override:
        return [Path(override).expanduser()]
    candidates = [_HERE.parent.parent / ".env"]
    main_root = _main_worktree_root()
    if main_root is not None:
        main_env = main_root / ".env"
        if main_env not in candidates:
            candidates.append(main_env)
    return candidates


def _load_env() -> str:
    """Load the first .env that exists. Returns a description for the summary."""
    candidates = _env_candidates()
    for path in candidates:
        if path.is_file():
            load_dotenv(path)
            return str(path)
    return "none found (tried: " + ", ".join(str(p) for p in candidates) + ")"


_ENV_SOURCE = _load_env()

_CREDENTIAL_VARS = (
    "PINECONE_API_KEY",
    "PINECONE_CLIENT_ID",
    "PINECONE_CLIENT_SECRET",
    "PINECONE_DOCUMENTS_INDEX_HOST",
)


def _skip_reason(report: pytest.TestReport | pytest.CollectReport) -> str:
    longrepr = report.longrepr
    if isinstance(longrepr, tuple) and len(longrepr) == 3:
        reason = str(longrepr[2])
    else:
        reason = str(longrepr)
    return reason.removeprefix("Skipped: ").strip() or "(no reason given)"


def _is_credential_skip(reason: str) -> bool:
    return any(var in reason for var in _CREDENTIAL_VARS)


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
    stats = terminalreporter.stats
    skipped = stats.get("skipped", [])
    ran = sum(
        len(stats.get(key, [])) for key in ("passed", "failed", "error", "xfailed", "xpassed")
    )
    collected = ran + len(skipped)

    reasons: dict[str, int] = {}
    for report in skipped:
        reason = _skip_reason(report)
        reasons[reason] = reasons.get(reason, 0) + 1
    credential_skips = sum(n for reason, n in reasons.items() if _is_credential_skip(reason))

    lines = [
        f".env source: {_ENV_SOURCE}",
        f"ran {ran} of {collected} collected"
        + (f" ({100 * ran // collected}%)" if collected else "")
        + f" — passed {len(stats.get('passed', []))}, "
        f"failed {len(stats.get('failed', []))}, errors {len(stats.get('error', []))}",
    ]
    if reasons:
        lines.append(f"skipped {len(skipped)}, by reason:")
        for reason, count in sorted(reasons.items(), key=lambda kv: (-kv[1], kv[0])):
            flag = "  <- CREDENTIALS MISSING" if _is_credential_skip(reason) else ""
            lines.append(f"  {count:>4}  {reason}{flag}")
    if credential_skips:
        pct = 100 * credential_skips // collected if collected else 0
        lines.append(
            f"WARNING: {credential_skips} tests ({pct}% of collected) never ran because "
            "credentials were missing. This result is NOT evidence of integration coverage."
        )

    terminalreporter.write_sep("=", "integration coverage summary")
    for line in lines:
        terminalreporter.write_line(line)

    step_summary = os.getenv("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a", encoding="utf-8") as fh:
            fh.write("### Integration coverage summary\n\n```\n")
            fh.write("\n".join(lines))
            fh.write("\n```\n")


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
