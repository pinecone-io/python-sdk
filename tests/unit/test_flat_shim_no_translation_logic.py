"""Structural guard: the flat backcompat shims must stay thin delegates.

`create_index`, `configure_index`, `list_indexes`, and `create_index_for_model`
on `Pinecone`/`AsyncPinecone` exist only to translate a legacy call into a
call on the corresponding `Indexes`/`AsyncIndexes` method. All the real
mapping (spec-to-deployment, pod-scaling, schema field construction, enum
resolution) lives in `pinecone/_internal/legacy_index_translation.py` and
`pinecone/_internal/index_migration.py`. This test parses each shim's source
and fails if it grows an `if`/`try`, a dict literal, or manual `.value`
enum unwrapping — the shapes that regressed these methods before.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from collections.abc import Callable
from typing import Any

import pytest

from pinecone import AsyncPinecone, Pinecone

SHIM_METHOD_NAMES = ["create_index", "configure_index", "list_indexes", "create_index_for_model"]


def _shim_methods() -> list[tuple[str, Callable[..., Any]]]:
    methods = []
    for cls in (Pinecone, AsyncPinecone):
        for name in SHIM_METHOD_NAMES:
            methods.append((f"{cls.__name__}.{name}", inspect.unwrap(getattr(cls, name))))
    return methods


def _non_docstring_body(func: Callable[..., Any]) -> list[ast.stmt]:
    src = textwrap.dedent(inspect.getsource(func))
    tree = ast.parse(src)
    func_def = tree.body[0]
    assert isinstance(func_def, (ast.FunctionDef, ast.AsyncFunctionDef))
    body = func_def.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]
    return body


@pytest.mark.parametrize(
    "qualname,func", _shim_methods(), ids=lambda p: p if isinstance(p, str) else ""
)
def test_shim_has_no_branching(qualname: str, func: Callable[..., Any]) -> None:
    body = _non_docstring_body(func)
    for stmt in body:
        for node in ast.walk(stmt):
            assert not isinstance(node, (ast.If, ast.Try, ast.IfExp)), (
                f"{qualname} contains branching ({type(node).__name__}); "
                "the shim must delegate, not reimplement translation logic"
            )


@pytest.mark.parametrize(
    "qualname,func", _shim_methods(), ids=lambda p: p if isinstance(p, str) else ""
)
def test_shim_builds_no_dict_literals(qualname: str, func: Callable[..., Any]) -> None:
    body = _non_docstring_body(func)
    for stmt in body:
        for node in ast.walk(stmt):
            assert not isinstance(node, ast.Dict), (
                f"{qualname} builds a dict literal; request-shape construction "
                "belongs in Indexes.create()/configure(), not the shim"
            )


@pytest.mark.parametrize(
    "qualname,func", _shim_methods(), ids=lambda p: p if isinstance(p, str) else ""
)
def test_shim_does_no_manual_enum_resolution(qualname: str, func: Callable[..., Any]) -> None:
    body = _non_docstring_body(func)
    for stmt in body:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Attribute) and node.attr == "value":
                pytest.fail(f"{qualname} appears to manually resolve an enum via .value")


@pytest.mark.parametrize(
    "qualname,func", _shim_methods(), ids=lambda p: p if isinstance(p, str) else ""
)
def test_shim_body_is_at_most_a_guard_plus_one_delegating_return(
    qualname: str, func: Callable[..., Any]
) -> None:
    body = _non_docstring_body(func)
    assert 1 <= len(body) <= 2, f"{qualname} body has {len(body)} statements, expected 1 or 2"
    assert isinstance(body[-1], ast.Return), f"{qualname}'s last statement must be a return"
    if len(body) == 2:
        guard = body[0]
        assert isinstance(guard, ast.Expr) and isinstance(guard.value, ast.Call), (
            f"{qualname}'s leading statement must be a bare guard call, got {ast.dump(guard)}"
        )
