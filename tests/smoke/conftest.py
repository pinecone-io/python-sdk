"""Shared fixtures and configuration for the smoke-test suite.

Smoke tests are notebook-style end-to-end scripts that walk through one
scenario each, exercising every method in the integration-testing punchlist
at least once on the happy path. They share infrastructure with
``tests/integration/`` (API key, polling helpers, cleanup utilities) but
live in their own directory so they can be run independently.

Run all smoke tests::

    PINECONE_API_KEY=... uv run --with python-dotenv pytest tests/smoke/ -v -s

Run only the fastest priority-1+2 path::

    pytest tests/smoke/test_inference_sync.py tests/smoke/test_inference_async.py \\
           tests/smoke/test_deprecated_shims_sync.py tests/smoke/test_deprecated_shims_async.py

The .env is looked up in the **main working tree**, so runs from a git
worktree pick up the same file (see ``tests.live_suite.env_candidates``).
Override the location with ``PINECONE_SDK_ENV_FILE=/path/to/.env``. Where a
run's key came from, and how much of the suite it actually let run, is
printed at the end of every session by ``pytest_terminal_summary``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from pinecone import Pinecone

# Re-export shared helpers so smoke tests can import them from this conftest.
from tests.integration.conftest import (  # noqa: F401 — re-exported for tests
    async_cleanup_resource,
    async_client,
    async_ensure_index_deleted,
    async_poll_until,
    cleanup_resource,
    ensure_index_deleted,
    poll_until,
    unique_name,
    wait_for_ready,
)
from tests.live_suite import load_env, write_coverage_summary

_HERE = Path(__file__).resolve().parent

_CREDENTIAL_VARS = ("PINECONE_API_KEY",)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "smoke: end-to-end smoke tests that hit a real Pinecone backend",
    )


_ENV_SOURCE = load_env(_HERE)


def pytest_terminal_summary(
    terminalreporter: pytest.TerminalReporter,
    exitstatus: int,
    config: pytest.Config,
) -> None:
    """Report what actually ran, and where the key came from.

    17 of the 17 collectible smoke tests gate on ``PINECONE_API_KEY``, so a
    keyless run is a wall of skips that exits 0 — indistinguishable from a
    real pass unless something says so. The printed ``.env source`` line also
    self-guards the lookup above: a resolution that stops reaching the main
    working tree announces itself on every run instead of going quiet for
    months (#295, #315).
    """
    write_coverage_summary(
        terminalreporter,
        label="smoke",
        env_source=_ENV_SOURCE,
        credential_vars=_CREDENTIAL_VARS,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def api_key() -> str:
    """Pinecone API key from environment. Skips smoke tests if absent."""
    key = os.getenv("PINECONE_API_KEY")
    if not key:
        pytest.skip("PINECONE_API_KEY not set")
    return key


@pytest.fixture
def client(api_key: str) -> Pinecone:
    """Function-scoped sync client.

    Function scope (not session) keeps each smoke scenario isolated — a client
    closed inside one test must not affect the next.
    """
    return Pinecone(api_key=api_key)


# ---------------------------------------------------------------------------
# Smoke prefix — make orphan detection trivial
# ---------------------------------------------------------------------------

SMOKE_PREFIX = "smoke"
"""All resources created by smoke tests must start with this prefix.

The orphan-cleanup script uses this prefix to find and delete any resources
left behind by killed jobs.
"""
