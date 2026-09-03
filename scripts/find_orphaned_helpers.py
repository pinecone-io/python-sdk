#!/usr/bin/env python3
"""Find pinecone/_internal/ helper functions with unit tests but no live caller.

Re-derives the call graph from the current tree on every run (liveness is a
fixed point seeded from every file outside pinecone/_internal/, then
propagated through helper functions that are themselves live), so this
keeps working as the tree changes rather than encoding today's findings.

Scope is module-level def/async def statements directly inside
pinecone/_internal/*.py (not the adapters/ subpackage, whose class methods
are consumed by construction and are a different shape than this
name-based call graph can reason about).

Usage:

    uv run python scripts/find_orphaned_helpers.py
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _top_level_functions(tree: ast.Module) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not node.name.startswith("__")
    }


def _non_candidate_statements(tree: ast.Module) -> list[ast.stmt]:
    return [
        node for node in tree.body if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def _referenced_names(nodes: ast.AST | list[ast.stmt]) -> set[str]:
    roots = [nodes] if isinstance(nodes, ast.AST) else nodes
    names: set[str] = set()
    for root in roots:
        for sub in ast.walk(root):
            if isinstance(sub, ast.Name):
                names.add(sub.id)
            elif isinstance(sub, ast.Attribute):
                names.add(sub.attr)
    return names


def _iter_py_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


@dataclass(frozen=True)
class OrphanReport:
    orphaned: dict[str, Path] = field(default_factory=dict)
    unreferenced: dict[str, Path] = field(default_factory=dict)


def find_orphaned_helpers(repo_root: Path = REPO_ROOT) -> OrphanReport:
    pinecone_root = repo_root / "pinecone"
    tests_root = repo_root / "tests" / "unit"

    helper_files = sorted(
        p for p in (pinecone_root / "_internal").glob("*.py") if p.name != "__init__.py"
    )
    helper_file_set = set(helper_files)

    candidates: dict[str, Path] = {}
    helper_trees: dict[Path, ast.Module] = {}
    for f in helper_files:
        tree = _parse(f)
        helper_trees[f] = tree
        for name in _top_level_functions(tree):
            candidates[name] = f

    outside_files = [p for p in _iter_py_files(pinecone_root) if p not in helper_file_set]
    outside_refs: set[str] = set()
    for f in outside_files:
        outside_refs |= _referenced_names(_parse(f))

    call_graph: dict[str, set[str]] = {}
    module_level_refs: set[str] = set()
    for tree in helper_trees.values():
        for name, node in _top_level_functions(tree).items():
            call_graph[name] = _referenced_names(node.body)
        module_level_refs |= _referenced_names(_non_candidate_statements(tree))

    live: set[str] = {
        name for name in candidates if name in outside_refs or name in module_level_refs
    }
    changed = True
    while changed:
        changed = False
        for name, callees in call_graph.items():
            if name not in live:
                continue
            for callee in callees:
                if callee in candidates and callee not in live:
                    live.add(callee)
                    changed = True

    test_refs: set[str] = set()
    for f in _iter_py_files(tests_root):
        test_refs |= _referenced_names(_parse(f))

    orphaned: dict[str, Path] = {}
    unreferenced: dict[str, Path] = {}
    for name, path in candidates.items():
        if name in live:
            continue
        if name in test_refs:
            orphaned[name] = path
        else:
            unreferenced[name] = path

    return OrphanReport(orphaned=orphaned, unreferenced=unreferenced)


def main() -> int:
    report = find_orphaned_helpers()
    if not report.orphaned and not report.unreferenced:
        print("No orphaned helpers found under pinecone/_internal/.")
        return 0

    if report.orphaned:
        print("Tested but no production caller:")
        for name, path in sorted(report.orphaned.items()):
            print(f"  {name}  ({path.relative_to(REPO_ROOT)})")

    if report.unreferenced:
        print("Referenced nowhere at all:")
        for name, path in sorted(report.unreferenced.items()):
            print(f"  {name}  ({path.relative_to(REPO_ROOT)})")

    return 1


if __name__ == "__main__":
    sys.exit(main())
