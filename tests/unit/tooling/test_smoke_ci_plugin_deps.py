"""The smoke CI jobs must install every pytest plugin their config needs.

Those two jobs install test dependencies by name against a *published wheel*
rather than syncing the dev group, so nothing links them to
``[tool.pytest.ini_options]``. #306's ``--strict-config`` makes that gap fatal:
an ini key whose owning plugin is absent is a usage error, and in pytest 9 it
is raised after collection, so the job emits the suite's collect-time skips and
then exits 4 without running a single collected test (#399).

A source scan, not a run. It deliberately does not import either live suite's
conftest — that reads the developer's real ``.env`` into the process (same
reason ``test_integration_async_marker_policy.py`` scans).

The maps below are the audit surface. A new ini key or a new plugin-owned
marker fails a test here, in the unit gate, instead of failing a wheel-install
job hours later with a message that looks nothing like a missing dependency.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
PYPROJECT = REPO_ROOT / "pyproject.toml"
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"
SMOKE_DIR = REPO_ROOT / "tests" / "smoke"

PYTEST_CORE_INI_KEYS = frozenset(
    {
        "addopts",
        "cache_dir",
        "consider_namespace_packages",
        "doctest_encoding",
        "doctest_optionflags",
        "empty_parameter_set_mark",
        "faulthandler_timeout",
        "filterwarnings",
        "junit_duration_report",
        "junit_family",
        "junit_logging",
        "junit_suite_name",
        "markers",
        "minversion",
        "norecursedirs",
        "python_classes",
        "python_files",
        "python_functions",
        "pythonpath",
        "required_plugins",
        "testpaths",
        "tmp_path_retention_count",
        "tmp_path_retention_policy",
        "usefixtures",
        "xfail_strict",
    }
)

# `anyio_mode` belongs to anyio's own `anyio.pytest_plugin`, not to the
# `pytest-anyio` distribution on PyPI, which is a 2021 stub that ships no
# plugin and only depends on anyio.
INI_KEY_OWNERS = {
    "anyio_mode": "anyio",
    "asyncio_default_fixture_loop_scope": "pytest-asyncio",
    "asyncio_default_test_loop_scope": "pytest-asyncio",
    "asyncio_mode": "pytest-asyncio",
    "timeout": "pytest-timeout",
    "timeout_func_only": "pytest-timeout",
    "timeout_method": "pytest-timeout",
}

PYTEST_CORE_MARKERS = frozenset(
    {"filterwarnings", "parametrize", "skip", "skipif", "usefixtures", "xfail"}
)

MARKER_OWNERS = {
    "anyio": "anyio",
    "asyncio": "pytest-asyncio",
    "benchmark": "pytest-benchmark",
    "timeout": "pytest-timeout",
}

_SMOKE_INVOCATION = "pytest tests/smoke/"


def _ini_option_keys() -> set[str]:
    """Top-level keys of ``[tool.pytest.ini_options]``.

    Read by regex rather than ``tomllib``: the unit suite is a CI gate on
    Python 3.10, which has no ``tomllib``, and the repo declares no TOML
    backport.
    """
    text = PYPROJECT.read_text(encoding="utf-8")
    start = text.index("[tool.pytest.ini_options]") + len("[tool.pytest.ini_options]")
    rest = text[start:]
    end = re.search(r"^\[", rest, re.MULTILINE)
    section = rest[: end.start()] if end else rest
    return set(re.findall(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=", section, re.MULTILINE))


def _ini_registered_markers() -> set[str]:
    text = PYPROJECT.read_text(encoding="utf-8")
    start = text.index("markers = [")
    section = text[start : text.index("]", start)]
    return set(re.findall(r'"([A-Za-z_][A-Za-z0-9_]*)', section))


def _addopts() -> str:
    text = PYPROJECT.read_text(encoding="utf-8")
    start = text.index("addopts = [")
    return text[start : text.index("]", start)]


def _markers_used_in_smoke() -> set[str]:
    used: set[str] = set()
    for source in sorted(SMOKE_DIR.rglob("*.py")):
        text = source.read_text(encoding="utf-8")
        used |= set(re.findall(r"pytest\.mark\.([A-Za-z_][A-Za-z0-9_]*)", text))
    return used


def _strip_shell_comments(script: str) -> str:
    """Drop whole-line ``#`` comments from a workflow ``run:`` block.

    Without this, a comment that merely *names* a package satisfies the
    assertion that the package is installed — which is how a mutation that
    deleted ``anyio`` from the install line first passed here.
    """
    return "\n".join(line for line in script.splitlines() if not line.lstrip().startswith("#"))


def _smoke_install_commands() -> dict[str, str]:
    """Map ``<workflow file>::<job id>`` to that job's concatenated pip installs.

    A job qualifies when one of its steps runs the smoke suite. Only its
    ``pip install`` steps are returned, so the ``pytest tests/smoke/``
    invocation itself cannot satisfy a package assertion.
    """
    found: dict[str, str] = {}
    for path in sorted(WORKFLOW_DIR.glob("*.y*ml")):
        workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
        for job_id, job in (workflow.get("jobs") or {}).items():
            steps = job.get("steps") or []
            runs = [_strip_shell_comments(str(step.get("run", ""))) for step in steps]
            if not any(_SMOKE_INVOCATION in run for run in runs):
                continue
            found[f"{path.name}::{job_id}"] = "\n".join(r for r in runs if "pip install" in r)
    return found


def _installs(command_text: str, distribution: str) -> bool:
    """Whether ``distribution`` is named as a package, with any version spec.

    The boundaries matter: without the lookbehind, ``anyio`` would be satisfied
    by ``pytest-anyio``, and without the lookahead, ``pytest`` would be
    satisfied by ``pytest-asyncio``.
    """
    pattern = rf"(?<![\w.-]){re.escape(distribution)}(?![\w-])"
    return re.search(pattern, command_text) is not None


SMOKE_INSTALLS = _smoke_install_commands()


class TestDiscovery:
    """Guard the scan itself: an empty scan would make every check below pass."""

    def test_both_smoke_jobs_are_found(self) -> None:
        assert set(SMOKE_INSTALLS) == {
            "dev-publish.yml::smoke-tests",
            "release-readiness.yml::install-tests",
        }

    @pytest.mark.parametrize("job", sorted(SMOKE_INSTALLS))
    def test_each_smoke_job_has_a_pip_install_step(self, job: str) -> None:
        assert "pip install" in SMOKE_INSTALLS[job]


class TestStrictConfigIsIntact:
    """#399's fix must not be the one #306 explicitly ruled out.

    Dropping ``--strict-config`` would make both smoke jobs green while
    restoring the silence that let ``timeout = 5`` sit inert for four months.
    """

    def test_strict_config_still_set(self) -> None:
        assert "--strict-config" in _addopts()

    def test_strict_markers_still_set(self) -> None:
        assert "--strict-markers" in _addopts()

    def test_global_timeout_ini_key_still_declared(self) -> None:
        assert "timeout" in _ini_option_keys()


class TestPluginOwnership:
    """Tripwires: a new ini key or marker must be classified, not ignored."""

    def test_every_ini_key_is_either_core_or_has_a_known_owner(self) -> None:
        unclassified = _ini_option_keys() - PYTEST_CORE_INI_KEYS - set(INI_KEY_OWNERS)
        assert not unclassified, (
            f"ini keys with no known owning plugin: {sorted(unclassified)}. "
            "Add each to INI_KEY_OWNERS (and to the smoke jobs' pip install) "
            "or to PYTEST_CORE_INI_KEYS if pytest itself registers it."
        )

    def test_every_marker_used_in_smoke_is_either_registered_or_owned(self) -> None:
        unclassified = (
            _markers_used_in_smoke()
            - PYTEST_CORE_MARKERS
            - _ini_registered_markers()
            - set(MARKER_OWNERS)
        )
        assert not unclassified, (
            f"markers used in tests/smoke/ with no owner: {sorted(unclassified)}. "
            "--strict-markers rejects an unregistered marker, so each needs an "
            "ini `markers` entry or an entry in MARKER_OWNERS."
        )


class TestSmokeJobsInstallWhatTheConfigNeeds:
    @pytest.mark.parametrize("job", sorted(SMOKE_INSTALLS))
    def test_installs_every_plugin_its_ini_keys_require(self, job: str) -> None:
        required = {INI_KEY_OWNERS[key] for key in _ini_option_keys() if key in INI_KEY_OWNERS}
        assert required, "no plugin-owned ini keys found — the scan is vacuous"
        missing = sorted(d for d in required if not _installs(SMOKE_INSTALLS[job], d))
        assert not missing, (
            f"{job} does not install {missing}, which own ini keys in "
            "[tool.pytest.ini_options]. --strict-config makes each a usage "
            "error after collection, so the job exits 4 having run nothing."
        )

    @pytest.mark.parametrize("job", sorted(SMOKE_INSTALLS))
    def test_installs_every_plugin_whose_markers_smoke_uses(self, job: str) -> None:
        required = {MARKER_OWNERS[m] for m in _markers_used_in_smoke() if m in MARKER_OWNERS}
        assert required, "no plugin-owned markers found in tests/smoke/ — the scan is vacuous"
        missing = sorted(d for d in required if not _installs(SMOKE_INSTALLS[job], d))
        assert not missing, (
            f"{job} does not install {missing}, whose markers tests/smoke/ "
            "applies. --strict-markers rejects an unregistered marker."
        )
