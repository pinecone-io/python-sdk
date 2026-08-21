"""Every module ``pinecone/`` imports eagerly must be a declared runtime dependency.

An eager import — one that runs at module scope, outside ``TYPE_CHECKING`` and
outside a ``try``/``except ImportError`` — is load-bearing: the module cannot be
imported at all without it. If such a module is absent from
``[project].dependencies`` the package still installs, and still works whenever
some *other* dependency drags the module in as a transitive. That is a resolver
accident, and it expires without warning: ``typing_extensions`` reached the SDK
only through ``anyio``'s ``python_version < "3.13"`` marker, so a wheel-only
install of the SDK raised ``ModuleNotFoundError`` on Python 3.13 and 3.14 (#411).

Two tests, because the failure has two independent halves and each half is
invisible to the other:

* :func:`test_eager_imports_are_declared_dependencies` is a static scan. It is
  general — it catches the next undeclared module, not just this one.
* :func:`test_public_surface_imports_without_undeclared_typing_extensions`
  exercises the code path in an environment missing the module. ``import
  pinecone`` is lazy and reaches none of it, which is why the whole 3.10-3.14
  install-verify matrix stayed green while the package was broken.

The scan is over the AST, never the file text: an import mentioned in a comment,
a docstring or a string literal must not satisfy it.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
import textwrap
from importlib.metadata import Distribution, distributions, packages_distributions
from pathlib import Path, PurePosixPath

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
PYPROJECT = REPO_ROOT / "pyproject.toml"
PACKAGE_DIR = REPO_ROOT / "pinecone"

FIRST_PARTY_ROOTS = frozenset({"pinecone"})

_REQUIREMENT_NAME = re.compile(r"^[A-Za-z0-9._-]+")


def _normalize(name: str) -> str:
    """PEP 503 normalized form, so ``typing_extensions`` matches ``typing-extensions``."""
    return re.sub(r"[-_.]+", "-", name).lower()


def declared_runtime_dependencies() -> frozenset[str]:
    """Normalized distribution names from ``[project].dependencies``.

    Read by regex rather than ``tomllib``: the unit suite is a CI gate on
    Python 3.10, which has no ``tomllib``, and the repo declares no TOML
    parser as a runtime dependency. Same approach as
    ``test_smoke_ci_plugin_deps.py``.
    """
    text = PYPROJECT.read_text(encoding="utf-8")
    match = re.search(r"^dependencies\s*=\s*\[(.*?)^\]", text, re.MULTILINE | re.DOTALL)
    assert match, "could not locate [project].dependencies in pyproject.toml"
    names = set()
    for raw in re.findall(r'"([^"]+)"', match.group(1)):
        stem = raw.split("[")[0].split(";")[0].strip()
        name = _REQUIREMENT_NAME.match(stem)
        assert name, f"unparsable requirement {raw!r}"
        names.add(_normalize(name.group(0)))
    assert names, "[project].dependencies parsed as empty"
    return frozenset(names)


def _is_type_checking_guard(test: ast.expr) -> bool:
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    if isinstance(test, ast.Attribute):
        return test.attr == "TYPE_CHECKING"
    return False


def _catches_import_error(handlers: list[ast.ExceptHandler]) -> bool:
    caught = {"ImportError", "ModuleNotFoundError", "Exception", "BaseException"}
    for handler in handlers:
        node = handler.type
        if node is None:
            return True
        candidates: list[ast.expr] = list(node.elts) if isinstance(node, ast.Tuple) else [node]
        for candidate in candidates:
            if isinstance(candidate, ast.Name) and candidate.id in caught:
                return True
            if isinstance(candidate, ast.Attribute) and candidate.attr in caught:
                return True
    return False


def eager_import_roots(body: list[ast.stmt]) -> set[str]:
    """Top-level module names imported unconditionally when *body*'s module loads.

    Descends through ``if``/``try``/``with``/loop bodies, which all execute at
    import time, but stops at function and class bodies (deferred to call time),
    at ``if TYPE_CHECKING:`` (never executed) and at a ``try`` whose handler
    absorbs ``ImportError`` (an optional dependency, by construction).
    """
    roots: set[str] = set()
    for node in body:
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                roots.add(node.module.split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        elif isinstance(node, ast.If):
            if not _is_type_checking_guard(node.test):
                roots |= eager_import_roots(node.body)
            roots |= eager_import_roots(node.orelse)
        elif isinstance(node, ast.Try):
            if not _catches_import_error(node.handlers):
                roots |= eager_import_roots(node.body)
            roots |= eager_import_roots(node.orelse)
            roots |= eager_import_roots(node.finalbody)
        elif isinstance(node, (ast.With, ast.AsyncWith, ast.For, ast.AsyncFor, ast.While)):
            roots |= eager_import_roots(node.body)
            roots |= eager_import_roots(getattr(node, "orelse", []))
    return roots


def _scan_package() -> dict[str, set[Path]]:
    """Map each eagerly-imported third-party root to the files that import it."""
    by_root: dict[str, set[Path]] = {}
    for path in sorted(PACKAGE_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for root in eager_import_roots(tree.body):
            if root in FIRST_PARTY_ROOTS or root in sys.stdlib_module_names:
                continue
            by_root.setdefault(root, set()).add(path.relative_to(REPO_ROOT))
    return by_root


def _top_level_names(dist: Distribution) -> set[str]:
    """Import names *dist* installs, from ``top_level.txt`` or its file list."""
    declared = dist.read_text("top_level.txt")
    if declared:
        return {line.strip() for line in declared.split() if line.strip()}
    names: set[str] = set()
    for entry in dist.files or ():
        parts = PurePosixPath(str(entry)).parts
        if not parts or parts[0] in ("..", "."):
            continue
        head = parts[0]
        if head.endswith((".dist-info", ".egg-info", ".data")):
            continue
        names.add(head if len(parts) > 1 else head.split(".")[0])
    return names


def module_to_distributions() -> dict[str, set[str]]:
    """Map each importable top-level name to the distributions providing it.

    ``importlib.metadata.packages_distributions()`` alone is not enough: on
    Python 3.10 it consults only ``top_level.txt``, which modern wheels
    (``httpx``, ``orjson``, ``typing_extensions``) do not ship, so it returns
    nothing for them and a declared dependency reads as undeclared. Inferring
    from the file list — what 3.12+ does internally — resolves them on every
    supported version. Both sources are unioned rather than one chosen, since
    neither is complete for editable and legacy installs.
    """
    mapping: dict[str, set[str]] = {}
    for dist in distributions():
        name = dist.metadata["Name"]
        if not name:
            continue
        normalized = _normalize(name)
        for top in _top_level_names(dist):
            mapping.setdefault(top, set()).add(normalized)
    for module, dists in packages_distributions().items():
        mapping.setdefault(module, set()).update(_normalize(d) for d in dists)
    return mapping


def test_scanner_sees_the_real_imports() -> None:
    """Guard against the scan going quietly blind and passing vacuously.

    The load-bearing assertion below is of the form "nothing undeclared was
    found", so a scan that found *nothing* would pass it. These are the imports
    we know are there.
    """
    by_root = _scan_package()
    assert {"httpx", "msgspec", "orjson"} <= set(by_root)
    assert len(list(PACKAGE_DIR.rglob("*.py"))) > 100


def test_scanner_flags_an_undeclared_eager_import() -> None:
    """And guard against the scan being unable to detect anything at all."""
    source = "import os\nimport totally_undeclared_xyz\nfrom httpx import Client\n"
    roots = eager_import_roots(ast.parse(source).body)
    assert "totally_undeclared_xyz" in roots
    assert "httpx" in roots


def test_scanner_ignores_a_textual_mention_of_an_import() -> None:
    """A comment or string that looks like an import must not register as one."""
    source = '"""import numpy"""\n# import numpy\nx = "import numpy"\n'
    assert eager_import_roots(ast.parse(source).body) == set()


@pytest.mark.parametrize(
    ("source", "expected_absent"),
    [
        ("from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import pandas\n", "pandas"),
        ("try:\n    import tqdm\nexcept ImportError:\n    tqdm = None\n", "tqdm"),
        ("def f():\n    import pandas\n", "pandas"),
        ("class C:\n    import pandas\n", "pandas"),
    ],
)
def test_scanner_ignores_deferred_and_guarded_imports(source: str, expected_absent: str) -> None:
    """An optional dependency need not be declared, and these are the four shapes.

    ``pandas`` and ``tqdm`` are genuinely optional in this package: both are
    reached only after a caller has opted in, and their absence degrades a
    feature rather than breaking the import.
    """
    assert expected_absent not in eager_import_roots(ast.parse(source).body)


def test_distribution_lookup_resolves_the_declared_dependencies() -> None:
    """The resolver must find the deps we know are declared and installed.

    Without this, a resolver that silently stops resolving anything turns
    :func:`test_eager_imports_are_declared_dependencies` into a wall of false
    "undeclared" reports — which is exactly what
    ``packages_distributions()`` did on Python 3.10 for every wheel that ships
    no ``top_level.txt``.
    """
    mapping = module_to_distributions()
    declared = declared_runtime_dependencies()
    for module in ("httpx", "msgspec", "orjson", "anyio"):
        assert mapping.get(module), module
        assert mapping[module] & declared, (module, mapping[module], declared)


def test_eager_imports_are_declared_dependencies() -> None:
    declared = declared_runtime_dependencies()
    mapping = module_to_distributions()
    undeclared: dict[str, tuple[set[str], set[Path]]] = {}
    for root, files in sorted(_scan_package().items()):
        providers = mapping.get(root, set())
        if not providers & declared:
            undeclared[root] = (providers, files)

    if undeclared:
        lines = [
            f"  {root}: imported by {sorted(str(f) for f in files)[:3]}; "
            f"provided by {sorted(providers) or 'nothing installed'}"
            for root, (providers, files) in undeclared.items()
        ]
        pytest.fail(
            "these modules are imported at module scope by pinecone/ but are not in "
            "[project].dependencies, so they reach users only if some other "
            "dependency happens to pull them in:\n" + "\n".join(lines)
        )


@pytest.mark.timeout(60)
def test_public_surface_imports_without_undeclared_typing_extensions() -> None:
    """Every lazy export must import with ``typing_extensions`` unavailable.

    A meta-path finder that refuses ``typing_extensions`` reproduces a 3.13+
    environment on any interpreter, because nothing in the declared dependency
    set requires it there. Run out-of-process: the finder has to be in place
    before ``pinecone`` is first imported, and the unit session has already
    imported it.

    ``import pinecone`` alone is not the assertion — it succeeded throughout
    #411. Each lazy attribute is touched, which is what pulls in the modules
    that do the importing.
    """
    program = textwrap.dedent(
        """
        import sys

        class _Refuse:
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".")[0] == "typing_extensions":
                    raise ModuleNotFoundError(
                        f"No module named {fullname!r}", name=fullname
                    )
                return None

        sys.meta_path.insert(0, _Refuse())
        sys.modules.pop("typing_extensions", None)

        import pinecone

        names = sorted(pinecone._LAZY_IMPORTS)
        if len(names) < 100:
            print(f"FAIL only {len(names)} lazy exports found", flush=True)
            raise SystemExit(1)

        broken = []
        for name in names:
            try:
                getattr(pinecone, name)
            except ModuleNotFoundError as exc:
                broken.append(f"{name}: missing {exc.name}")
        if broken:
            print("FAIL " + "; ".join(broken[:8]), flush=True)
            raise SystemExit(1)
        print(f"OK {len(names)}", flush=True)
        """
    )
    result = subprocess.run(  # noqa: S603 — fixed argv, program is a literal above
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, (
        f"the public surface needs typing_extensions at runtime:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr[-3000:]}"
    )
    assert result.stdout.startswith("OK ")
