"""Tests that enforce preview/stable isolation rules.

These act as CI guardrails so future changes do not accidentally violate
the preview/stable boundary defined in docs/conventions/preview-channel.md
§ "What NOT to do".
"""

from __future__ import annotations

import ast
from pathlib import Path


def test_no_preview_symbols_in_top_level_all() -> None:
    """Preview symbols must not be re-exported from pinecone/__init__.py."""
    import pinecone

    preview_symbols = [
        name for name in pinecone.__all__ if "Preview" in name or "preview" in name.lower()
    ]
    assert preview_symbols == [], (
        f"Preview symbols found in pinecone.__all__: {preview_symbols}. "
        "Preview symbols must not be re-exported from the top level. "
        "See docs/conventions/preview-channel.md § What NOT to do."
    )


def _is_type_checking_guard(node: ast.If) -> bool:
    """Return True if *node* is an ``if TYPE_CHECKING:`` guard."""
    test = node.test
    # Plain name: `if TYPE_CHECKING:`
    if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
        return True
    # Attribute: `if typing.TYPE_CHECKING:`
    return isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"


def test_no_stable_code_imports_preview_at_module_level() -> None:
    """Stable code must not import from pinecone.preview at module level.

    Allowed exceptions:
    (a) Any file under pinecone/preview/.
    (b) TYPE_CHECKING-gated imports (only evaluated by type checkers, not at
        runtime).
    (c) Deferred imports inside function/method bodies (e.g. the preview
        property getters in _client.py and async_client/pinecone.py).
    """
    sdk_root = Path(__file__).parent.parent.parent
    pinecone_root = sdk_root / "pinecone"
    preview_root = pinecone_root / "preview"

    violations: list[tuple[str, int, str]] = []

    for py_file in sorted(pinecone_root.rglob("*.py")):
        # (a) Skip files under pinecone/preview/
        try:
            py_file.relative_to(preview_root)
            continue
        except ValueError:
            pass

        source = py_file.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=str(py_file))
        except SyntaxError:
            continue

        for stmt in tree.body:
            if isinstance(stmt, ast.ImportFrom):
                module = stmt.module or ""
                if module == "pinecone.preview" or module.startswith("pinecone.preview."):
                    violations.append(
                        (
                            str(py_file.relative_to(sdk_root)),
                            stmt.lineno,
                            ast.unparse(stmt),
                        )
                    )
            elif isinstance(stmt, ast.If) and _is_type_checking_guard(stmt):
                # (b) TYPE_CHECKING-gated imports are allowed — skip the block
                pass
            # FunctionDef / AsyncFunctionDef / ClassDef: do not recurse.
            # Deferred imports inside bodies (c) are explicitly allowed.

    assert violations == [], (
        "Stable code imports from pinecone.preview at module level (not allowed).\n"
        "Allowed exceptions: TYPE_CHECKING guards and deferred imports inside"
        " function/method bodies.\n"
        "Violations:\n"
        + "\n".join(f"  {path}:{lineno}: {import_stmt}" for path, lineno, import_stmt in violations)
        + "\nSee docs/conventions/preview-channel.md § What NOT to do."
    )


def test_preview_init_exports_only_namespace_classes() -> None:
    """pinecone.preview.__all__ must only contain approved symbols.

    Approved: Preview, AsyncPreview (namespace classes), and SchemaBuilder,
    an intentional entry-point export per spec/preview.md §12 (callers do
    ``from pinecone.preview import SchemaBuilder``; the name aliases the
    graduated ``pinecone.schema_builder.SchemaBuilder``).
    """
    from pinecone import preview

    if hasattr(preview, "__all__"):
        allowed = {"Preview", "AsyncPreview", "SchemaBuilder"}
        extra = set(preview.__all__) - allowed
        assert extra == set(), (
            f"Unexpected exports in pinecone.preview.__all__: {extra}. "
            "Only namespace classes and approved entry-point aliases should be exported. "
            "Area-specific symbols belong in their own submodules."
        )
