"""Programmatic filter builder for metadata queries.

Produces plain dicts compatible with the ``filter`` parameter on
query, delete, update, and search methods.

Usage::

    from pinecone import Field

    f = (Field("genre") == "drama") & (Field("year").gte(2020))
    results = index.query(vector=[...], filter=f.to_dict())
"""

from __future__ import annotations

from typing import Any

# Value types accepted by equality / set operators.
ScalarValue = str | int | float | bool

# Value types accepted by ordering (numeric-only) operators.
NumericValue = int | float


class Condition:
    """A composable metadata filter condition.

    Wraps an internal filter dict and supports ``&`` (AND) and ``|`` (OR)
    combination with automatic flattening of nested same-type logical
    operators.
    """

    __slots__ = ("_filter",)

    def __init__(self, filter_dict: dict[str, Any]) -> None:
        self._filter = filter_dict

    # -- logical combinators --------------------------------------------------

    def __and__(self, other: Condition) -> Condition:
        left = list(self._filter["$and"]) if "$and" in self._filter else [self._filter]
        right = list(other._filter["$and"]) if "$and" in other._filter else [other._filter]
        return Condition({"$and": [*left, *right]})

    def __or__(self, other: Condition) -> Condition:
        left = list(self._filter["$or"]) if "$or" in self._filter else [self._filter]
        right = list(other._filter["$or"]) if "$or" in other._filter else [other._filter]
        return Condition({"$or": [*left, *right]})

    # -- serialisation --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return the filter as a plain dict.

        Raises:
            ValueError: If the condition contains no operators.
        """
        if not self._filter:
            raise ValueError("Cannot convert an empty condition to a filter dict")
        return self._filter

    def __repr__(self) -> str:
        return f"Condition({self._filter!r})"


class Field:
    """Represents a metadata field name for building filter expressions.

    Usage::

        Field("genre") == "drama"       # {"genre": {"$eq": "drama"}}
        Field("score").gt(0.5)           # {"score": {"$gt": 0.5}}
    """

    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        self._name = name

    # -- comparison operators (numeric only) ----------------------------------

    def _require_numeric(self, op: str, value: Any) -> None:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(
                f"{op} requires a numeric value (int or float), got {type(value).__name__}"
            )

    def gt(self, value: int | float) -> Condition:
        """``$gt`` — greater than (*numeric only*)."""
        self._require_numeric("gt", value)
        return Condition({self._name: {"$gt": value}})

    def gte(self, value: int | float) -> Condition:
        """``$gte`` — greater than or equal (*numeric only*)."""
        self._require_numeric("gte", value)
        return Condition({self._name: {"$gte": value}})

    def lt(self, value: int | float) -> Condition:
        """``$lt`` — less than (*numeric only*)."""
        self._require_numeric("lt", value)
        return Condition({self._name: {"$lt": value}})

    def lte(self, value: int | float) -> Condition:
        """``$lte`` — less than or equal (*numeric only*)."""
        self._require_numeric("lte", value)
        return Condition({self._name: {"$lte": value}})

    # -- equality operators ---------------------------------------------------

    def __eq__(self, value: object) -> Condition:  # type: ignore[override]
        """``$eq`` — equal to."""
        return Condition({self._name: {"$eq": value}})

    def __ne__(self, value: object) -> Condition:  # type: ignore[override]
        """``$ne`` — not equal to."""
        return Condition({self._name: {"$ne": value}})

    # -- set operators --------------------------------------------------------

    def _require_list(self, op: str, values: Any) -> None:
        if not isinstance(values, (list, tuple)):
            raise TypeError(
                f"{op} requires a list of values, got {type(values).__name__}"
            )

    def is_in(self, values: list[str | int | float | bool]) -> Condition:
        """``$in`` — value is in the given list."""
        self._require_list("is_in", values)
        return Condition({self._name: {"$in": values}})

    def not_in(self, values: list[str | int | float | bool]) -> Condition:
        """``$nin`` — value is not in the given list."""
        self._require_list("not_in", values)
        return Condition({self._name: {"$nin": values}})

    # -- exists operator ------------------------------------------------------

    def exists(self) -> Condition:
        """``$exists`` — field exists."""
        return Condition({self._name: {"$exists": True}})

    def __repr__(self) -> str:
        return f"Field({self._name!r})"


# Backcompat alias, :meta private:
FilterBuilder = Field
