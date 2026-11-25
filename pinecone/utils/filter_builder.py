"""Optional builder for metadata filters.

A ``filter`` argument is a plain dict, and writing that dict out by hand is
fully supported: ``{"genre": {"$eq": "drama"}}`` is what every method
accepts, and what :meth:`Condition.to_dict` produces. This module is for
filters your code assembles rather than writes: :class:`Field` exposes a
method for each operator (comparison, set, existence and text match), so
the operator names are checked by your editor instead of typed into a
string, and ``&`` / ``|`` compose conditions built in
different places.

Either route produces the same filter, so mixing them is fine.
"""

from __future__ import annotations

from typing import Any

from pinecone.errors.exceptions import PineconeTypeError, PineconeValueError

# Value types accepted by equality / set operators.
ScalarValue = str | int | float | bool

# Value types accepted by ordering (numeric-only) operators.
NumericValue = int | float


class Condition:
    """One filter clause, or several combined.

    Returned by every :class:`Field` operator; you never construct one
    directly. Combine conditions with ``&`` for ``$and`` and ``|`` for
    ``$or``, then call :meth:`to_dict` to get the value to pass as
    ``filter``.

    Combining flattens same-operator nesting, so chaining three ``&`` gives
    one ``$and`` of three clauses rather than nested pairs. Mixing the two
    operators nests as Python's precedence dictates, which is why the
    operands want parentheses.

    Examples:
        >>> from pinecone import Field
        >>> ((Field("genre") == "drama") & Field("year").gte(2020)).to_dict()
        {'$and': [{'genre': {'$eq': 'drama'}}, {'year': {'$gte': 2020}}]}
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
        """Return the condition as the dict to pass as ``filter``.

        The dict is the builder's own state, not a copy, so mutating the
        result mutates the condition.

        Returns:
            The filter, e.g. ``{"year": {"$gte": 2020}}``.

        Examples:
            >>> from pinecone import Field
            >>> Field("year").gte(2020).to_dict()
            {'year': {'$gte': 2020}}
        """
        if not self._filter:
            raise PineconeValueError("Cannot convert an empty condition to a filter dict")
        return self._filter

    def __repr__(self) -> str:
        return f"Condition({self._filter!r})"


class Field:
    """One metadata field, ready to be compared.

    Reach it with ``from pinecone import Field``. Naming a field produces no
    filter on its own; applying an operator to it returns a
    :class:`Condition`, and :meth:`Condition.to_dict` turns that into the
    ``filter`` value.

    The two equality operators are Python's own: ``Field("genre") == "drama"``
    builds ``$eq`` and ``!=`` builds ``$ne``. Because they build a filter
    rather than answer a question, comparing a ``Field`` tests nothing: the
    returned :class:`Condition` is always truthy, so
    ``Field("genre") in [Field("year")]`` is ``True``. A ``Field`` is also
    unhashable, so it cannot be a dict key or set member.

    The ordering operators :meth:`gt`, :meth:`gte`, :meth:`lt` and :meth:`lte`
    are numeric only. :meth:`is_in` and :meth:`not_in` take a list,
    :meth:`exists` takes nothing. The text-match operators
    :meth:`match_phrase`, :meth:`match_all` and :meth:`match_any` take a string
    and apply only to a full-text string field in documents search and fetch;
    documents delete and update reject them.

    Examples:
        >>> from pinecone import Field
        >>> (Field("genre") == "drama").to_dict()
        {'genre': {'$eq': 'drama'}}
        >>> (Field("genre") != "documentary").to_dict()
        {'genre': {'$ne': 'documentary'}}

    .. seealso::
       :class:`Condition`, for combining these with ``&`` and ``|``.
    """

    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        self._name = name

    # -- comparison operators (numeric only) ----------------------------------

    def _require_numeric(self, op: str, value: Any) -> None:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise PineconeTypeError(
                f"{op} requires a numeric value (int or float), got {type(value).__name__}"
            )

    def gt(self, value: int | float) -> Condition:
        """``$gt``: the field is greater than *value* (numeric only).

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeTypeError`: If *value* is
                not an :class:`int` or :class:`float`. A :class:`bool` is rejected
                as well, though Python counts it as an ``int``; compare a boolean
                field with ``==`` instead. It derives from :class:`TypeError`, so
                an existing ``except TypeError`` still catches it.

        Examples:
            >>> from pinecone import Field
            >>> Field("rating").gt(4.5).to_dict()
            {'rating': {'$gt': 4.5}}
        """
        self._require_numeric("gt", value)
        return Condition({self._name: {"$gt": value}})

    def gte(self, value: int | float) -> Condition:
        """``$gte``: the field is greater than or equal to *value* (numeric only).

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeTypeError`: If *value* is
                not an :class:`int` or :class:`float`. A :class:`bool` is rejected
                as well, though Python counts it as an ``int``; compare a boolean
                field with ``==`` instead. It derives from :class:`TypeError`, so
                an existing ``except TypeError`` still catches it.

        Examples:
            >>> from pinecone import Field
            >>> Field("year").gte(2020).to_dict()
            {'year': {'$gte': 2020}}
        """
        self._require_numeric("gte", value)
        return Condition({self._name: {"$gte": value}})

    def lt(self, value: int | float) -> Condition:
        """``$lt``: the field is less than *value* (numeric only).

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeTypeError`: If *value* is
                not an :class:`int` or :class:`float`. A :class:`bool` is rejected
                as well, though Python counts it as an ``int``; compare a boolean
                field with ``==`` instead. It derives from :class:`TypeError`, so
                an existing ``except TypeError`` still catches it.

        Examples:
            >>> from pinecone import Field
            >>> Field("price_usd").lt(25).to_dict()
            {'price_usd': {'$lt': 25}}
        """
        self._require_numeric("lt", value)
        return Condition({self._name: {"$lt": value}})

    def lte(self, value: int | float) -> Condition:
        """``$lte``: the field is less than or equal to *value* (numeric only).

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeTypeError`: If *value* is
                not an :class:`int` or :class:`float`. A :class:`bool` is rejected
                as well, though Python counts it as an ``int``; compare a boolean
                field with ``==`` instead. It derives from :class:`TypeError`, so
                an existing ``except TypeError`` still catches it.

        Examples:
            >>> from pinecone import Field
            >>> Field("duration_minutes").lte(120).to_dict()
            {'duration_minutes': {'$lte': 120}}
        """
        self._require_numeric("lte", value)
        return Condition({self._name: {"$lte": value}})

    # -- equality operators ---------------------------------------------------

    def __eq__(self, value: object) -> Condition:  # type: ignore[override]
        """``$eq``: equal to."""
        return Condition({self._name: {"$eq": value}})

    def __ne__(self, value: object) -> Condition:  # type: ignore[override]
        """``$ne``: not equal to."""
        return Condition({self._name: {"$ne": value}})

    # -- set operators --------------------------------------------------------

    def _require_list(self, op: str, values: Any) -> None:
        if not isinstance(values, (list, tuple)):
            raise TypeError(f"{op} requires a list of values, got {type(values).__name__}")

    def is_in(self, values: list[str | int | float | bool]) -> Condition:
        """``$in``: the field's value is one of *values*.

        Examples:
            >>> from pinecone import Field
            >>> Field("genre").is_in(["drama", "thriller"]).to_dict()
            {'genre': {'$in': ['drama', 'thriller']}}
        """
        self._require_list("is_in", values)
        return Condition({self._name: {"$in": values}})

    def not_in(self, values: list[str | int | float | bool]) -> Condition:
        """``$nin``: the field's value is none of *values*.

        Examples:
            >>> from pinecone import Field
            >>> Field("genre").not_in(["horror"]).to_dict()
            {'genre': {'$nin': ['horror']}}
        """
        self._require_list("not_in", values)
        return Condition({self._name: {"$nin": values}})

    # -- exists operator ------------------------------------------------------

    def exists(self) -> Condition:
        """``$exists``: the field is present on the record, whatever its value.

        Examples:
            >>> from pinecone import Field
            >>> Field("release_year").exists().to_dict()
            {'release_year': {'$exists': True}}
        """
        return Condition({self._name: {"$exists": True}})

    # -- text-match operators -------------------------------------------------

    def _text_match(self, op: str, text: Any) -> Condition:
        if not isinstance(text, str):
            raise PineconeTypeError(
                f"{op} takes one string of words, got {type(text).__name__}; "
                "join a list of words into one string"
            )
        return Condition({self._name: {f"${op}": text}})

    def match_phrase(self, text: str) -> Condition:
        """``$match_phrase``: the field's text contains *text* as a contiguous, ordered phrase.

        Applies to a string field with stored, searchable text (one declared
        with ``full_text_search``, or with ``embed`` or ``sparse_embed`` and
        the default ``store_text``). Accepted in the ``filter`` of documents
        ``search`` and ``fetch``. Documents ``delete`` and ``update`` reject it
        with a ``400``, as does a filter that applies it to any other field.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeTypeError`: If *text* is not a :class:`str`.

        Examples:
            >>> from pinecone import Field
            >>> Field("title").match_phrase("black hole").to_dict()
            {'title': {'$match_phrase': 'black hole'}}
        """
        return self._text_match("match_phrase", text)

    def match_all(self, text: str) -> Condition:
        """``$match_all``: the field's text contains every word of *text*, in any order.

        Applies to a string field with stored, searchable text (one declared
        with ``full_text_search``, or with ``embed`` or ``sparse_embed`` and
        the default ``store_text``). Accepted in the ``filter`` of documents
        ``search`` and ``fetch``. Documents ``delete`` and ``update`` reject it
        with a ``400``, as does a filter that applies it to any other field.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeTypeError`: If *text* is not a :class:`str`.

        Examples:
            >>> from pinecone import Field
            >>> Field("title").match_all("black hole").to_dict()
            {'title': {'$match_all': 'black hole'}}
        """
        return self._text_match("match_all", text)

    def match_any(self, text: str) -> Condition:
        """``$match_any``: the field's text contains at least one word of *text*.

        Applies to a string field with stored, searchable text (one declared
        with ``full_text_search``, or with ``embed`` or ``sparse_embed`` and
        the default ``store_text``). Accepted in the ``filter`` of documents
        ``search`` and ``fetch``. Documents ``delete`` and ``update`` reject it
        with a ``400``, as does a filter that applies it to any other field.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeTypeError`: If *text* is not a :class:`str`.

        Examples:
            >>> from pinecone import Field
            >>> Field("title").match_any("black hole").to_dict()
            {'title': {'$match_any': 'black hole'}}
        """
        return self._text_match("match_any", text)

    def __repr__(self) -> str:
        return f"Field({self._name!r})"


# Backcompat alias, :meta private:
FilterBuilder = Field
