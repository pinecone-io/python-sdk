"""Turn positional misuse of keyword-only methods into a clear error.

Public data-plane and control-plane methods are keyword-only (defined with a
bare ``*,`` marker). Calling one positionally otherwise raises Python's raw
``TypeError: query() takes 1 positional argument but 2 were given`` — which
never says the method is keyword-only, doesn't name the parameter the value was
meant for, and refers to ``self`` as "1 positional argument".

The :func:`keyword_only_methods` class decorator wraps every public
keyword-only method on a class so the same misuse raises an actionable
:class:`PineconeValueError` instead. The wrapper preserves the wrapped
function's signature via :func:`functools.wraps` (which sets ``__wrapped__``),
so :func:`inspect.signature`, IDE autocomplete, type checkers, and the
generated docs continue to see the real signature.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import Any, TypeVar

from pinecone.errors.exceptions import PineconeValueError

_ClassT = TypeVar("_ClassT", bound=type)

_MAX_EXAMPLE_PARAMS = 3


def _keyword_only_names(func: Callable[..., Any]) -> list[str]:
    return [
        name
        for name, param in inspect.signature(func).parameters.items()
        if param.kind is inspect.Parameter.KEYWORD_ONLY
    ]


def _positional_misuse_message(method: str, kw_names: list[str], owner: str, count: int) -> str:
    plural = "argument" if count == 1 else "arguments"
    example = ", ".join(f"{name}=..." for name in kw_names[:_MAX_EXAMPLE_PARAMS])
    return (
        f"{owner}.{method}() is a keyword-only method and does not accept "
        f"positional arguments. You passed {count} positional {plural}. "
        f"Pass every argument by keyword instead, e.g. {method}({example}). "
        f"Accepted keyword arguments: {', '.join(kw_names)}."
    )


def _guard(func: Callable[..., Any]) -> Callable[..., Any]:
    method = func.__name__
    kw_names = _keyword_only_names(func)

    if inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            if args:
                raise PineconeValueError(
                    _positional_misuse_message(method, kw_names, type(self).__name__, len(args))
                )
            return await func(self, **kwargs)

        return async_wrapper

    @functools.wraps(func)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        if args:
            raise PineconeValueError(
                _positional_misuse_message(method, kw_names, type(self).__name__, len(args))
            )
        return func(self, **kwargs)

    return wrapper


def _is_keyword_only_method(func: Callable[..., Any]) -> bool:
    try:
        params = list(inspect.signature(func).parameters.values())
    except (TypeError, ValueError):
        return False
    saw_keyword_only = False
    for param in params:
        if param.name == "self":
            continue
        if param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.VAR_POSITIONAL,
        ):
            return False
        if param.kind is inspect.Parameter.KEYWORD_ONLY:
            saw_keyword_only = True
    return saw_keyword_only


def keyword_only_methods(cls: _ClassT) -> _ClassT:
    """Guard every public keyword-only method on *cls* against positional misuse.

    A method qualifies when it takes at least one keyword-only parameter and no
    positional parameters other than ``self``. Private and dunder methods,
    properties, and methods that accept positional arguments are left untouched.
    """
    for name, member in list(vars(cls).items()):
        if name.startswith("_") or not inspect.isfunction(member):
            continue
        if _is_keyword_only_method(member):
            setattr(cls, name, _guard(member))
    return cls
