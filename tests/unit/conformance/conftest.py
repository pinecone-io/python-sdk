"""Fixture and hooks enforcing the conformance-claim contract.

scripts/api_coverage.py sets PINECONE_CONFORMANCE_RESULTS to a file path
and reads back, per claimed test, which operations it claims and whether it
passed. Outcome folding is deliberately pessimistic: any failed phase
(setup, call, or the claim-satisfaction check that runs at teardown) and
any skip leaves the test not-passed, so its claims do not count.
"""

from __future__ import annotations

import json
import os
from collections.abc import Generator
from typing import Any

import pytest

from tests.unit.conformance._registry import ClaimRecorder

RESULTS_ENV = "PINECONE_CONFORMANCE_RESULTS"

_tracked: dict[str, dict[str, Any]] = {}


def _claimed(item: pytest.Item) -> list[str]:
    return [marker.args[0] for marker in item.iter_markers("api_op")]


@pytest.fixture
def claim(request: pytest.FixtureRequest) -> Generator[ClaimRecorder, None, None]:
    ops = _claimed(request.node)
    if not ops:
        pytest.fail("the claim fixture requires at least one @api_op decorator")
    recorder = ClaimRecorder(ops)
    yield recorder
    recorder.assert_satisfied()


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        ops = _claimed(item)
        if ops:
            _tracked[item.nodeid] = {"ops": ops, "outcome": "collected"}


def pytest_runtest_setup(item: pytest.Item) -> None:
    if _claimed(item) and "claim" not in getattr(item, "fixturenames", ()):
        pytest.fail(
            "@api_op tests must take the `claim` fixture and satisfy its mandatory "
            "assertions (see tests/unit/conformance/README.md)",
            pytrace=False,
        )


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    info = _tracked.get(report.nodeid)
    if info is None:
        return
    if report.failed:
        info["outcome"] = "failed"
    elif report.skipped and info["outcome"] != "failed":
        info["outcome"] = "skipped"
    elif report.when == "call" and report.passed and info["outcome"] == "collected":
        info["outcome"] = "passed"


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    results_path = os.environ.get(RESULTS_ENV)
    if not results_path:
        return
    with open(results_path, "w") as f:
        json.dump({"tests": _tracked}, f, indent=2, sort_keys=True)
