"""Shared plumbing for the credential-gated live suites (integration + smoke).

Both suites talk to a real Pinecone backend, both take their credentials from
a ``.env`` at the SDK root, and both resolved that file from the root of the
*current* working tree — which is never where it lives when a run starts
inside a git worktree. #295 fixed the copy in ``tests/integration/conftest``;
#315 found two more copies (``tests/smoke/conftest`` and the smoke
orphan-cleanup script) with the same defect. There is one implementation now
so the next fix lands everywhere at once.

Two pieces live here:

``load_env``
    Resolve and load the ``.env``, returning a human-readable description of
    where it came from (or of every path that was tried and missed).
``write_coverage_summary``
    End-of-session report of what actually ran. A credential-starved live
    suite otherwise exits 0 behind a wall of skips nobody counts, and
    "the live tests are green" gets read as coverage.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv

if TYPE_CHECKING:
    import pytest

ENV_FILE_OVERRIDE = "PINECONE_SDK_ENV_FILE"
"""Point this at a ``.env`` to bypass repo-root discovery entirely."""


def _git_roots(start: Path) -> tuple[Path | None, Path | None]:
    """``(current tree root, main working tree root)`` for the tree holding ``start``.

    A linked worktree's ``.git`` is a *file* pointing into
    ``<main>/.git/worktrees/<name>``, so the current tree's root is not the
    repo root that holds ``.env``. ``--show-toplevel`` gives the current
    tree's root (a worktree with its own ``.env`` should still win) and
    ``--git-common-dir`` resolves back to the real ``.git`` directory in both
    cases, whose parent is the main checkout root.

    Returns ``(None, None)`` outside a git checkout or without a git binary.
    """
    git = shutil.which("git")
    if git is None:
        return None, None
    try:
        proc = subprocess.run(  # noqa: S603 — fixed argv, git resolved via shutil.which
            [git, "rev-parse", "--show-toplevel", "--git-common-dir"],
            cwd=start,
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None, None
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if len(lines) != 2:
        return None, None
    toplevel = Path(lines[0])
    # Relative in the main tree (e.g. "../../.git"), absolute in a worktree.
    common_dir = Path(lines[1])
    if not common_dir.is_absolute():
        common_dir = (start / common_dir).resolve()
    return toplevel, common_dir.parent


def _project_root_by_marker(start: Path) -> Path | None:
    """Nearest ancestor of ``start`` holding ``pyproject.toml``.

    Fallback for a source tree unpacked outside a git checkout, where no
    amount of ``git rev-parse`` helps.
    """
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return None


def env_candidates(start: Path) -> list[Path]:
    """``.env`` locations to try, most specific first.

    ``start`` is any directory inside the SDK tree — typically the caller's
    own directory. Depth does not matter: the roots are discovered, never
    counted off in ``.parent`` hops, which is what made the three copies of
    this lookup diverge in the first place.
    """
    override = os.getenv(ENV_FILE_OVERRIDE)
    if override:
        return [Path(override).expanduser()]

    candidates: list[Path] = []
    for root in (*_git_roots(start), _project_root_by_marker(start)):
        if root is None:
            continue
        env_path = root / ".env"
        if env_path not in candidates:
            candidates.append(env_path)
    return candidates


def load_env(start: Path) -> str:
    """Load the first ``.env`` that exists. Returns a description for the summary.

    The description is the load-bearing half: printed on every run, it turns a
    failed lookup into one visible line instead of a wall of skips.
    """
    candidates = env_candidates(start)
    for path in candidates:
        if path.is_file():
            load_dotenv(path)
            return str(path)
    return "none found (tried: " + ", ".join(str(p) for p in candidates) + ")"


def _skip_reason(report: pytest.TestReport | pytest.CollectReport) -> str:
    longrepr = report.longrepr
    if isinstance(longrepr, tuple) and len(longrepr) == 3:
        reason = str(longrepr[2])
    else:
        reason = str(longrepr)
    return reason.removeprefix("Skipped: ").strip() or "(no reason given)"


def write_coverage_summary(
    terminalreporter: pytest.TerminalReporter,
    *,
    label: str,
    env_source: str,
    credential_vars: tuple[str, ...],
) -> None:
    """Report what actually ran, to the terminal and to ``$GITHUB_STEP_SUMMARY``.

    ``label`` names the suite ("integration", "smoke"); ``credential_vars``
    are the variables whose absence means *missing credentials* rather than a
    deliberate opt-out, so the two kinds of skip can be told apart. See #295.
    """
    stats = terminalreporter.stats
    skipped = stats.get("skipped", [])
    ran = sum(
        len(stats.get(key, [])) for key in ("passed", "failed", "error", "xfailed", "xpassed")
    )
    collected = ran + len(skipped)

    def is_credential_skip(reason: str) -> bool:
        return any(var in reason for var in credential_vars)

    reasons: dict[str, int] = {}
    for report in skipped:
        reason = _skip_reason(report)
        reasons[reason] = reasons.get(reason, 0) + 1
    credential_skips = sum(n for reason, n in reasons.items() if is_credential_skip(reason))

    lines = [
        f".env source: {env_source}",
        f"ran {ran} of {collected} collected"
        + (f" ({100 * ran // collected}%)" if collected else "")
        + f" — passed {len(stats.get('passed', []))}, "
        f"failed {len(stats.get('failed', []))}, errors {len(stats.get('error', []))}",
    ]
    if reasons:
        lines.append(f"skipped {len(skipped)}, by reason:")
        for reason, count in sorted(reasons.items(), key=lambda kv: (-kv[1], kv[0])):
            flag = "  <- CREDENTIALS MISSING" if is_credential_skip(reason) else ""
            lines.append(f"  {count:>4}  {reason}{flag}")
    if credential_skips:
        pct = 100 * credential_skips // collected if collected else 0
        lines.append(
            f"WARNING: {credential_skips} tests ({pct}% of collected) never ran because "
            f"credentials were missing. This result is NOT evidence of {label} coverage."
        )

    terminalreporter.write_sep("=", f"{label} coverage summary")
    for line in lines:
        terminalreporter.write_line(line)

    step_summary = os.getenv("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a", encoding="utf-8") as fh:
            fh.write(f"### {label.capitalize()} coverage summary\n\n```\n")
            fh.write("\n".join(lines))
            fh.write("\n```\n")
