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
    LegacyIndexFactory,
    async_cleanup_resource,
    async_client,
    async_ensure_index_deleted,
    async_poll_until,
    cleanup_resource,
    ensure_index_deleted,
    legacy_index_factory,
    poll_until,
    unique_name,
    wait_for_ready,
)
from tests.integration.legacy_index import LegacyIndex
from tests.live_suite import load_env, write_coverage_summary

_HERE = Path(__file__).resolve().parent

_CREDENTIAL_VARS = ("PINECONE_API_KEY",)

_SMOKE_TIMEOUT_SECONDS = 600
"""Wall-clock ceiling for one smoke test, sized from measurement (#347).

- Slowest measured smoke run is ``test_backups_sync`` at 43.19s on a live
  2026-07 backend, so this is ~14x the observed worst case.
- ``Indexes.create()`` defaults to ``timeout=None``, which polls for readiness
  **indefinitely**. Most smoke tests create without a timeout, so nothing
  inside the test bounds a stuck provision and this ceiling is the only bound.
- The bounded waits a smoke test *does* make sum to 510s in the worst module
  CI runs (``test_serverless_dense_sync``: 60 + 60 + 30 + 4x60 + 120). A lower
  ceiling would replace those helpers' diagnostic ``TimeoutError`` with an
  opaque pytest kill.
- Both smoke jobs cap the whole job above this (``timeout-minutes`` 20 and 30),
  so a hang fails as one named test, not an unattributed job kill.

The pod/collections modules are the exception — internal waits summing to
~4320s — and are ``--ignore``d by both smoke jobs, so sizing them is left to
whoever revives them.
"""


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "smoke: end-to-end smoke tests that hit a real Pinecone backend",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Raise the timeout for smoke tests, which provision real infrastructure (#347).

    Sibling of the ``tests/integration`` hook and path-filtered for the same
    reason: pytest hands a conftest hook the whole session's item list, not
    only the items collected beneath that conftest. An unfiltered loop would
    hand ``tests/unit`` this ceiling too, and ``timeout = 5`` there is a real
    CI gate — nine unit tests already sit within 1.8-3.1x of it (#345).

    Items carrying their own ``timeout`` marker are left alone, so an explicit
    per-test value keeps winning.
    """
    for item in items:
        if _HERE not in item.path.parents:
            continue
        if item.get_closest_marker("timeout") is None:
            item.add_marker(pytest.mark.timeout(_SMOKE_TIMEOUT_SECONDS))


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


SMOKE_VECTOR_DIM = 8
"""Dimension shared by every smoke module that exercises the vectors API.

One value, so all of them share a single index from the session-scoped
factory instead of provisioning one each.
"""


@pytest.fixture(scope="session")
def legacy_index_dim8(legacy_index_factory: LegacyIndexFactory) -> LegacyIndex:  # noqa: F811
    """A ready dim-8 cosine legacy (vectors-API) index.

    ``pc.indexes.create`` cannot produce an index the vectors API will serve —
    2026-07 always persists a schema, and a schema naming its own data fields
    is served by the documents API, which refuses ``upsert`` / ``query`` /
    ``fetch`` (#322). Smoke modules whose scenario *is* the vectors data plane
    therefore take their index from here.

    Sibling of ``legacy_index_dim3`` in ``tests/integration/conftest.py`` and
    built from the same session-scoped factory, so this directory shares that
    one implementation rather than carrying a second copy of it (#379). See
    :mod:`tests.integration.legacy_index` for the sanctioned pattern.

    Session-scoped and shared, so a caller must isolate itself with a
    per-test namespace and must not delete the index.

    The ``noqa: F811`` is unavoidable: ``tests/smoke`` is a sibling of
    ``tests/integration``, not a child, so the factory has to be re-exported
    above to be resolvable here — and ruff reads the parameter of the same
    name as a redefinition of that import.
    """
    return legacy_index_factory(dimension=SMOKE_VECTOR_DIM)


# ---------------------------------------------------------------------------
# Smoke prefix — make orphan detection trivial
# ---------------------------------------------------------------------------

SMOKE_PREFIX = "smoke"
"""All resources created by smoke tests must start with this prefix.

The orphan-cleanup script uses this prefix to find and delete any resources
left behind by killed jobs.
"""
