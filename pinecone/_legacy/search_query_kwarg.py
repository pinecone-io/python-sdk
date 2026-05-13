"""Backcompat shim for the legacy v8 ``query=SearchQuery(...)`` kwarg on
``Index.search`` / ``AsyncIndex.search`` / ``GrpcIndex.search`` (and their
``search_records`` aliases).

In v8 the search methods took a single ``query: SearchQuery | dict`` wrapper
parameter; v9 flattened that into top-level kwargs (``top_k``, ``inputs``,
``vector``, ``id``, ``filter``, ``match_terms``). This helper restores the
wrapper form for migrating callers, with a ``DeprecationWarning``.

:meta private:
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Any

from pinecone.models.vectors.search import SearchQuery

_LEGACY_QUERY_FIELDS = ("inputs", "top_k", "vector", "id", "filter", "match_terms")


def unpack_legacy_query(
    *,
    method_name: str,
    query: SearchQuery | Mapping[str, Any] | None,
    top_k: int | None,
    inputs: Any,
    vector: Sequence[float] | Mapping[str, Any] | None,
    id: str | None,
    filter: Mapping[str, Any] | None,
    match_terms: Mapping[str, Any] | None,
) -> tuple[
    int | None,
    Any,
    Sequence[float] | Mapping[str, Any] | None,
    str | None,
    Mapping[str, Any] | None,
    Mapping[str, Any] | None,
]:
    """If *query* is provided, unpack it into the flat kwargs and warn.

    Returns the (possibly rewritten) tuple
    ``(top_k, inputs, vector, id, filter, match_terms)``.

    Raises:
        TypeError: if *query* is provided together with any of the flat
            kwargs (ambiguous call).
    """
    if query is None:
        return top_k, inputs, vector, id, filter, match_terms

    already_set = [
        name
        for name, value in (
            ("top_k", top_k),
            ("inputs", inputs),
            ("vector", vector),
            ("id", id),
            ("filter", filter),
            ("match_terms", match_terms),
        )
        if value is not None
    ]
    if already_set:
        raise TypeError(
            f"{method_name}() received both 'query=' and "
            f"{already_set!r}. Pass either the legacy 'query=SearchQuery(...)' "
            "form OR the new flat keyword arguments, not both."
        )

    warnings.warn(
        f"Passing 'query=SearchQuery(...)' to {method_name}() is a v8 "
        "compatibility shim. Pass top_k, inputs, vector, id, filter, and "
        "match_terms as separate keyword arguments instead.",
        DeprecationWarning,
        stacklevel=3,
    )

    if isinstance(query, SearchQuery):
        unpacked = {name: getattr(query, name) for name in _LEGACY_QUERY_FIELDS}
    elif isinstance(query, Mapping):
        unpacked = {name: query.get(name) for name in _LEGACY_QUERY_FIELDS}
    else:
        raise TypeError(
            f"{method_name}() 'query=' must be a SearchQuery or Mapping, got {type(query).__name__}"
        )

    return (
        unpacked["top_k"],
        unpacked["inputs"],
        unpacked["vector"],
        unpacked["id"],
        unpacked["filter"],
        unpacked["match_terms"],
    )
