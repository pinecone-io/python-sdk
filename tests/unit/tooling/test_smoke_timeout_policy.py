"""Guards the per-directory pytest-timeout policy from CI (#347).

The policy lives in conftest hooks that only fire when somebody runs the tree
they guard, and neither ``tests/smoke`` nor ``tests/integration`` is a CI gate.
This is the gated copy. It is a source scan rather than an import because
importing either conftest would read the real ``.env`` into this process at
import time — the same reason ``test_integration_async_marker_policy.py`` scans
instead of importing. Regex over ``tomllib`` so it runs on Python 3.10.

The failure this exists to prevent: raising the global default to accommodate a
slow live test. ``timeout = 5`` is the pressure that got the nine unit tests of
#345 made fast rather than marked; a global raise would have hidden them
instead. Slow trees get their own ceiling instead.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SMOKE_CONFTEST = _REPO_ROOT / "tests" / "smoke" / "conftest.py"
_INTEGRATION_CONFTEST = _REPO_ROOT / "tests" / "integration" / "conftest.py"

_SLOWEST_MEASURED_SMOKE_SECONDS = 43.19


def test_global_pytest_timeout_default_is_still_five_seconds() -> None:
    text = (_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r"^timeout\s*=\s*(\d+)$", text, re.MULTILINE)
    assert match is not None, "no `timeout = N` in [tool.pytest.ini_options]"
    assert match.group(1) == "5", (
        f"global pytest timeout is {match.group(1)}s, expected 5s. See this "
        "module's docstring: give the slow tree its own ceiling instead."
    )


def test_smoke_ceiling_clears_the_slowest_measured_smoke_run() -> None:
    text = _SMOKE_CONFTEST.read_text(encoding="utf-8")
    match = re.search(r"^_SMOKE_TIMEOUT_SECONDS\s*=\s*(\d+)$", text, re.MULTILINE)
    assert match is not None, f"no _SMOKE_TIMEOUT_SECONDS in {_SMOKE_CONFTEST.name}"
    assert int(match.group(1)) > _SLOWEST_MEASURED_SMOKE_SECONDS


def test_both_slow_tree_hooks_stay_path_filtered() -> None:
    for conftest in (_SMOKE_CONFTEST, _INTEGRATION_CONFTEST):
        text = conftest.read_text(encoding="utf-8")
        assert "def pytest_collection_modifyitems(" in text, (
            f"{conftest.parent.name} lost its timeout hook; its tests fall back "
            "to the 5s unit-test default"
        )
        assert "_HERE not in item.path.parents" in text, (
            f"{conftest.parent.name}'s hook lost its path filter. pytest hands a "
            "conftest hook the whole session's item list, so an unfiltered loop "
            "applies this tree's ceiling to tests/unit as well."
        )
