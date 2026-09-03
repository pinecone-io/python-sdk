"""Guards every module under ``tests/unit/`` against reaching ``load_env()`` (#426).

``tests/unit/test_legacy_index_helper.py`` already pins one instance of this:
``tests/integration/legacy_index.py`` must not import ``tests.integration.conftest``,
because that module calls ``load_env()`` at import time and would inject a real
``PINECONE_API_KEY`` into ``os.environ`` mid-unit-session (#363). That guard only
looks at one file. #426 is the general form: any module reachable from
``tests/unit/`` — through a chain of ordinary or lazy imports, at any nesting
depth — that resolves to something calling ``load_env()`` at its own module
scope has the same defect, whichever file it lives in and however many hops
away it is.

This walks the *source*, not the runtime import machinery: it parses every
module under ``tests/unit/`` for import statements (``ast.walk`` reaches
imports nested inside function bodies too, so a lazy import that only
executes when its containing function is called still counts as "can reach"),
follows edges into other first-party ``tests.*`` modules, and flags any module
in the closure whose own top-level body calls a function named ``load_env``.
The set of risky modules is discovered from the current tree rather than
hard-coded, so a new module written the same way is caught without anyone
updating a list.
"""

from __future__ import annotations

import ast
import pathlib

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_UNIT_ROOT = _REPO_ROOT / "tests" / "unit"
_TESTS_PACKAGE = "tests"


def _parse(path: pathlib.Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def _imported_module_names(tree: ast.Module) -> set[str]:
    """Every module dotted name this file imports, at any nesting depth.

    ``from tests.integration import conftest`` names the package
    (``tests.integration``) and the imported attribute (``conftest``)
    separately in the AST, and that attribute is exactly as likely to be a
    submodule as a name defined in the package's own ``__init__.py`` — the
    #363 shape imports a conftest module this way. Both ``tests.integration``
    and ``tests.integration.conftest`` go in the set; resolution below drops
    whichever one is not an actual file.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None and node.level == 0:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def _resolve_first_party_module(name: str) -> pathlib.Path | None:
    """The file a ``tests.*`` dotted name resolves to, or ``None`` off-tree."""
    if name != _TESTS_PACKAGE and not name.startswith(_TESTS_PACKAGE + "."):
        return None
    rel = pathlib.Path(*name.split("."))
    as_module = _REPO_ROOT / rel.with_suffix(".py")
    if as_module.is_file():
        return as_module
    as_package = _REPO_ROOT / rel / "__init__.py"
    if as_package.is_file():
        return as_package
    return None


def _module_scope_calls(node: ast.AST) -> list[ast.Call]:
    """Every ``Call`` that runs as part of ordinary module execution.

    Does not descend into ``def``/``async def``/``lambda`` bodies, since those
    only run when something calls them — a different, already-covered hazard
    (the function becoming reachable and later invoked). Everything else —
    ``if``, ``try``, ``with``, ``for``, ``while``, and a ``class`` body, which
    executes immediately when the class statement is reached — runs as part
    of top-to-bottom module execution, so a call hidden inside one of those
    still counts as module scope. Bugbot flagged the earlier version of this
    function for missing exactly this: it only looked at bare top-level
    ``Assign``/``Expr`` statements.
    """
    calls: list[ast.Call] = []
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        if isinstance(child, ast.Call):
            calls.append(child)
        calls.extend(_module_scope_calls(child))
    return calls


def _calls_load_env_at_module_scope(tree: ast.Module) -> bool:
    """Whether importing this module runs ``load_env(...)`` as a side effect."""
    for call in _module_scope_calls(tree):
        func = call.func
        called_name = (
            func.id
            if isinstance(func, ast.Name)
            else func.attr
            if isinstance(func, ast.Attribute)
            else None
        )
        if called_name == "load_env":
            return True
    return False


def _neutralizes_env_file_override(tree: ast.Module) -> bool:
    """Whether this module uses the established override-and-restore dance.

    ``test_legacy_index_helper.py`` and ``test_live_suite_cleanup.py`` both
    import a load-env-reaching module deliberately, and both make it safe the
    same way: point ``ENV_FILE_OVERRIDE`` (from ``tests.live_suite``) at a
    nonexistent path in a module-level ``try``, do the import inside that
    ``try``, and restore the variable in a ``finally``. That neutralizes
    ``load_env()``'s side effect before it ever calls ``load_dotenv`` — the
    established, audited way to touch this surface from a unit test, distinct
    from the #363 hazard of an *unguarded* reach. This only recognizes the
    pattern at module scope: it does not chase whether every reachable import
    is actually inside such a ``try``, so a module using the dance is treated
    as exempt as a whole.
    """
    imports_override_name = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "tests.live_suite"
        and any(alias.name == "ENV_FILE_OVERRIDE" for alias in node.names)
        for node in ast.walk(tree)
    )
    if not imports_override_name:
        return False
    for index, node in enumerate(tree.body):
        if not (isinstance(node, ast.Try) and node.finalbody):
            continue
        preceding = tree.body[:index]
        if any(
            isinstance(prior, ast.Assign)
            and any(
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Attribute)
                and target.value.attr == "environ"
                for target in prior.targets
            )
            for prior in preceding
        ):
            return True
    return False


def _find_load_env_chain(start: pathlib.Path) -> list[pathlib.Path] | None:
    """A chain of imports from ``start`` to a module calling ``load_env()``, if any."""
    if _neutralizes_env_file_override(_parse(start)):
        return None

    visited: set[pathlib.Path] = set()

    def dfs(path: pathlib.Path) -> list[pathlib.Path] | None:
        if path in visited:
            return None
        visited.add(path)
        tree = _parse(path)
        if _calls_load_env_at_module_scope(tree):
            return [path]
        for module_name in _imported_module_names(tree):
            target = _resolve_first_party_module(module_name)
            if target is None or target == path:
                continue
            found = dfs(target)
            if found is not None:
                return [path, *found]
        return None

    return dfs(start)


def test_no_unit_test_module_can_reach_load_env() -> None:
    offenders: dict[pathlib.Path, list[pathlib.Path]] = {}
    for path in sorted(_UNIT_ROOT.rglob("*.py")):
        chain = _find_load_env_chain(path)
        if chain is not None and len(chain) > 1:
            offenders[path] = chain

    assert not offenders, "\n".join(
        f"{path.relative_to(_REPO_ROOT)} reaches load_env() via: "
        f"{' -> '.join(str(p.relative_to(_REPO_ROOT)) for p in chain)}"
        for path, chain in offenders.items()
    )


def test_the_known_load_env_call_sites_are_still_the_only_ones() -> None:
    """Sanity check that the walk itself finds real hits, not nothing at all.

    If this list ever shrinks to empty, the traversal has gone vacuous —
    ``load_env`` moved or was renamed and the guard above would pass for the
    wrong reason. If it grows, a new call site needs the same lazy-import
    scrutiny #363 gave the first one.
    """
    hits = {
        path.relative_to(_REPO_ROOT)
        for path in sorted((_REPO_ROOT / "tests").rglob("*.py"))
        if _calls_load_env_at_module_scope(_parse(path))
    }
    assert hits == {
        pathlib.Path("tests/integration/conftest.py"),
        pathlib.Path("tests/smoke/conftest.py"),
    }
