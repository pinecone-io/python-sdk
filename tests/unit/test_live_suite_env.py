"""Guards for the live suites' shared ``.env`` resolution (#295, #315).

This lookup has silently broken twice — once in ``tests/integration`` and once
in ``tests/smoke`` — and both times the only symptom was a live suite quietly
skipping itself in every git worktree. Nothing asserted the behaviour, so
nothing caught it. These tests build a real worktree and assert it.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from tests.live_suite import ENV_FILE_OVERRIDE, env_candidates

pytestmark = pytest.mark.timeout(60)


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(  # noqa: S603 — fixed argv, git presence checked by the caller
        ["git", *args],  # noqa: S607
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.fixture
def repo_with_worktree(tmp_path: Path) -> tuple[Path, Path]:
    """``(main checkout root, linked worktree root)`` of a throwaway repo."""
    if shutil.which("git") is None:
        pytest.skip("git not available")
    main = tmp_path / "main"
    (main / "tests" / "smoke" / "scripts").mkdir(parents=True)
    # git tracks files, not directories; without these the linked worktree has
    # no tests/smoke/ for `git rev-parse` to run inside.
    (main / "tests" / "smoke" / "conftest.py").write_text("")
    (main / "tests" / "smoke" / "scripts" / "cleanup.py").write_text("")
    (main / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    _git(main, "init", "-q", "-b", "trunk")
    _git(main, "config", "user.email", "t@example.com")
    _git(main, "config", "user.name", "t")
    _git(main, "add", "-A")
    _git(main, "commit", "-qm", "init")
    linked = tmp_path / "linked"
    _git(main, "worktree", "add", "-q", "--detach", str(linked), "trunk")
    return main, linked


def test_main_checkout_resolves_its_own_root(repo_with_worktree: tuple[Path, Path]) -> None:
    main, _ = repo_with_worktree
    assert env_candidates(main / "tests" / "smoke") == [main / ".env"]


def test_worktree_falls_back_to_the_main_checkout(repo_with_worktree: tuple[Path, Path]) -> None:
    main, linked = repo_with_worktree
    assert env_candidates(linked / "tests" / "smoke") == [linked / ".env", main / ".env"]


def test_resolution_is_independent_of_directory_depth(
    repo_with_worktree: tuple[Path, Path],
) -> None:
    """Parent-counting is what let the three copies of this lookup diverge."""
    _, linked = repo_with_worktree
    deep = linked / "tests" / "smoke" / "scripts"
    assert env_candidates(deep) == env_candidates(linked / "tests" / "smoke")


def test_override_wins_and_suppresses_discovery(
    repo_with_worktree: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, linked = repo_with_worktree
    monkeypatch.setenv(ENV_FILE_OVERRIDE, "/custom/place/.env")
    assert env_candidates(linked / "tests" / "smoke") == [Path("/custom/place/.env")]


def test_falls_back_to_the_pyproject_marker_outside_git(tmp_path: Path) -> None:
    root = tmp_path / "unpacked"
    (root / "tests" / "smoke").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    assert root / ".env" in env_candidates(root / "tests" / "smoke")
