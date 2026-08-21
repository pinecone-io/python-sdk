"""The smoke jobs' orphan-cleanup step must run, and must be audible when it doesn't.

``python tests/smoke/scripts/cleanup_orphans.py`` makes ``sys.path[0]`` the
script's own directory, so the script's ``tests.live_suite`` import cannot
resolve and it dies before deleting anything. Both smoke jobs invoked it that
way, under ``|| true`` *and* ``continue-on-error``, so the step reported success
while doing nothing for its entire existence (#412). The path form looks fine
in a synced checkout because the editable install adds the project root through
a ``.pth`` file; a job that installs a built wheel has no such entry.

A source scan plus one subprocess. It deliberately does not import either live
suite's conftest — that reads the developer's real ``.env`` into the process
(same reason ``test_smoke_ci_plugin_deps.py`` scans).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from tests.live_suite import ENV_FILE_OVERRIDE

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"

_CLEANUP_MODULE = "tests.smoke.scripts.cleanup_orphans"
_CLEANUP_SCRIPT_PATH = "tests/smoke/scripts/cleanup_orphans.py"


def _cleanup_steps() -> dict[str, dict[str, object]]:
    """Map ``<workflow file>::<job id>`` to that job's orphan-cleanup step.

    Keyed off the module or script name rather than the step's ``name``, so
    renaming the step cannot make this scan quietly stop finding it.
    """
    found: dict[str, dict[str, object]] = {}
    for path in sorted(WORKFLOW_DIR.glob("*.y*ml")):
        workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
        for job_id, job in (workflow.get("jobs") or {}).items():
            for step in job.get("steps") or []:
                run = str(step.get("run", ""))
                if "cleanup_orphans" in run:
                    found[f"{path.name}::{job_id}"] = step
    return found


def _commands(step: dict[str, object]) -> list[str]:
    """Lines of the step's script that invoke the cleanup, comments dropped.

    Without dropping comments the explanation of why the broken form is broken
    would itself trip the assertion that the broken form is absent.
    """
    return [
        line
        for line in str(step.get("run", "")).splitlines()
        if "cleanup_orphans" in line and not line.lstrip().startswith("#")
    ]


CLEANUP_STEPS = _cleanup_steps()


class TestDiscovery:
    """Guard the scan: an empty scan would make every check below pass."""

    def test_both_smoke_jobs_have_a_cleanup_step(self) -> None:
        assert set(CLEANUP_STEPS) == {
            "dev-publish.yml::smoke-tests",
            "release-readiness.yml::install-tests",
        }

    @pytest.mark.parametrize("job", sorted(CLEANUP_STEPS))
    def test_each_step_has_exactly_one_cleanup_command(self, job: str) -> None:
        assert len(_commands(CLEANUP_STEPS[job])) == 1


class TestTheImportsResolve:
    @pytest.mark.parametrize("job", sorted(CLEANUP_STEPS))
    def test_invoked_as_a_module_not_as_a_path(self, job: str) -> None:
        command = _commands(CLEANUP_STEPS[job])[0]
        assert f"-m {_CLEANUP_MODULE}" in command, (
            f"{job} must invoke the cleanup as `python -m {_CLEANUP_MODULE}`. "
            "Got: " + command.strip()
        )

    @pytest.mark.parametrize("job", sorted(CLEANUP_STEPS))
    def test_does_not_invoke_the_bare_script_path(self, job: str) -> None:
        command = _commands(CLEANUP_STEPS[job])[0]
        assert _CLEANUP_SCRIPT_PATH not in command, (
            f"{job} invokes the cleanup by path. sys.path[0] is then the "
            "script's own directory, so `from tests.live_suite import "
            "load_env` raises ModuleNotFoundError and nothing is cleaned up "
            "(#412)."
        )

    @pytest.mark.timeout(60)
    @pytest.mark.parametrize("job", sorted(CLEANUP_STEPS))
    def test_the_module_the_workflow_names_is_runnable(self, job: str) -> None:
        """Run the dotted name the workflow actually uses, from the repo root.

        Pins the workflow string to a real module: a rename or a move that left
        the workflows untouched would otherwise reproduce #412 exactly, and
        ``continue-on-error`` means CI would not say so. ``--help`` exits inside
        argparse, so nothing here reaches the network. The override keeps the
        subprocess off the developer's real ``.env``.
        """
        target = re.search(r"-m\s+(\S+)", _commands(CLEANUP_STEPS[job])[0])
        assert target is not None
        env = {**os.environ, ENV_FILE_OVERRIDE: str(REPO_ROOT / "no-such-.env")}
        proc = subprocess.run(  # noqa: S603 — argv is this repo's own workflow text
            [sys.executable, "-m", target.group(1), "--help"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=45,
            env=env,
            check=False,
        )
        assert proc.returncode == 0, f"{job}: {proc.stderr}"
        assert "--dry-run" in proc.stdout


class TestFailureStaysVisible:
    """A cleanup failure must not block the job, and must not be silent either."""

    @pytest.mark.parametrize("job", sorted(CLEANUP_STEPS))
    def test_cannot_fail_the_job(self, job: str) -> None:
        assert CLEANUP_STEPS[job].get("continue-on-error") is True

    @pytest.mark.parametrize("job", sorted(CLEANUP_STEPS))
    def test_runs_even_when_the_suite_failed(self, job: str) -> None:
        assert str(CLEANUP_STEPS[job].get("if", "")).strip() == "always()"

    @pytest.mark.parametrize("job", sorted(CLEANUP_STEPS))
    def test_exit_code_is_not_rewritten(self, job: str) -> None:
        command = _commands(CLEANUP_STEPS[job])[0]
        assert "|| true" not in command, (
            f"{job} pipes the cleanup through `|| true`. continue-on-error "
            "already keeps the step from failing the job; `|| true` on top of "
            "it only hides that the cleanup broke (#412)."
        )

    @pytest.mark.parametrize("job", sorted(CLEANUP_STEPS))
    def test_reports_what_it_found_to_the_run_summary(self, job: str) -> None:
        script = str(CLEANUP_STEPS[job].get("run", ""))
        assert "GITHUB_STEP_SUMMARY" in script, (
            f"{job} must publish the cleanup output to the run summary — the "
            "step reporting *something* is the property #412 was missing."
        )
