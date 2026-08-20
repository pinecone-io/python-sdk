"""Document record models (2026-07 API).

Documents are open-schema: beyond the reserved ``_id`` (and ``_score`` on
search matches), a document carries arbitrary user-defined fields. These
models therefore wrap a plain ``dict`` rather than declaring closed
``msgspec.Struct`` field sets, so unknown fields always survive a
decode -> ``to_dict()`` round-trip.
"""

from __future__ import annotations

import re
from typing import Any

import orjson

__all__ = ["Document", "DocumentRecord", "UpdateDocumentRecord"]

_MAX_ID_LENGTH = 512
_ID_PATTERN = re.compile(r"^[\x01-\x7F]+$")


def _validate_document_id(value: Any) -> str:
    """Validate a document ``_id`` against the 2026-07 constraints.

    IDs must be strings of 1-512 ASCII characters matching
    ``^[\\x01-\\x7F]+$`` (no NUL byte, no non-ASCII characters).

    Raises:
        ValueError: If the ID violates any constraint. The message names
            the constraint violated.
    """
    if not isinstance(value, str):
        raise ValueError(
            "Document '_id' is required and must be a string of 1-512 ASCII characters."
        )
    if len(value) == 0:
        raise ValueError("Document '_id' must not be empty (1-512 ASCII characters required).")
    if len(value) > _MAX_ID_LENGTH:
        raise ValueError(
            f"Document '_id' exceeds the maximum length of {_MAX_ID_LENGTH} characters "
            f"(got {len(value)})."
        )
    if not _ID_PATTERN.fullmatch(value):
        raise ValueError(
            f"Document '_id' {value!r} is invalid: IDs must contain only ASCII "
            "characters in the range \\x01-\\x7F (no NUL byte, no non-ASCII characters)."
        )
    return value


class Document:
    """A document returned from a search or fetch operation.

    Provides typed access to ``_id`` and ``_score`` fields, plus attribute-style
    and dict-style access to arbitrary document fields.

    The ``id``, ``_id``, and ``score`` typed properties always take precedence
    over document fields with the same names — a document field named ``"_score"``
    is only reachable via ``.get("_score")`` or ``.to_dict()["_score"]``.

    Attributes:
        id: The document's unique identifier.
        _id: Alias for ``id``.
        score: Relevance score, or ``None`` when not present in the response
            (fetched documents carry no score).
        _score: Alias for ``score``.

    Examples:
        >>> doc = Document({"_id": "article-42", "_score": 0.891, "title": "Rome"})
        >>> doc.id
        'article-42'
        >>> doc.score
        0.891
        >>> doc.title
        'Rome'
        >>> doc.get("missing_field", "n/a")
        'n/a'
    """

    __slots__ = ("_data",)
    _data: dict[str, Any]

    def __init__(self, data: dict[str, Any]) -> None:
        object.__setattr__(self, "_data", data)

    @property
    def id(self) -> str:
        value = self._data.get("_id")
        if not value:
            raise AttributeError("Document has no '_id' field")
        return str(value)

    @property
    def _id(self) -> str:
        return self.id

    @property
    def score(self) -> float | None:
        raw = self._data.get("_score")
        if raw is None:
            return None
        return float(raw)

    @property
    def _score(self) -> float | None:
        return self.score

    def get(self, key: str, default: Any = None) -> Any:
        """Return the value for *key* from the document, or *default*.

        Equivalent to :meth:`dict.get` on the underlying document data.
        Reserved fields (``_id``, ``_score``) are reachable via ``.get()``
        alongside any custom fields returned by the operation.

        Args:
            key (str): Name of the document field to retrieve.
            default (Any): Value to return when *key* is absent
                (default: ``None``).

        Returns:
            The field value, or *default* if the field is not present.
        """
        return self._data.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        """Return the document as a plain dictionary.

        Returns a shallow copy of the underlying document data, including
        ``_id``, ``_score``, and all custom fields from the operation.
        Mutating the returned dict does not affect the document.
        """
        return dict(self._data)

    def to_json(self) -> str:
        """Return the document as a compact JSON string (decoded UTF-8)."""
        return orjson.dumps(self._data).decode()

    def __getattr__(self, name: str) -> Any:
        # __slots__ attributes are resolved before __getattr__, so _data is safe.
        # Properties (id, _id, score) are descriptors on the class and normally
        # take precedence over __getattr__. But when a property raises
        # AttributeError, Python falls back to __getattr__; block those reserved
        # names here so a user-defined `id`/`_id`/`score` field cannot leak
        # through the typed property's failure path.
        if name in ("id", "_id", "score"):
            raise AttributeError(f"{type(self).__name__!r} has no attribute {name!r}")
        data = object.__getattribute__(self, "_data")
        if name in data:
            return data[name]
        raise AttributeError(f"{type(self).__name__!r} has no attribute {name!r}")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Document):
            return NotImplemented
        return self._data == other._data

    def __repr__(self) -> str:
        _id = self._data.get("_id", "")
        score = self._data.get("_score")
        extras = {k: v for k, v in self._data.items() if k not in ("_id", "_score")}
        if extras:
            return f"Document(_id={_id!r}, score={score!r}, ...)"
        return f"Document(_id={_id!r}, score={score!r})"


class DocumentRecord:
    """A document to upsert, validated client-side before any HTTP request.

    Wraps the open-schema wire shape: a required ``_id`` plus arbitrary
    user-defined field values. Scalar fields carry the same types as
    metadata (string, number, boolean, or list of strings); a field
    declared in the index schema as ``dense_vector`` carries a list of
    floats, and one declared ``sparse_vector`` carries sparse values.
    Field values are validated server-side against the index schema.

    The ``_id`` is validated on construction: 1-512 characters, ASCII only
    (``^[\\x01-\\x7F]+$``). Violations raise :class:`ValueError` before the
    record could reach the wire.

    Examples:
        >>> DocumentRecord({"_id": "doc-1", "title": "Rome"})
        DocumentRecord(_id='doc-1', fields=1)
        >>> DocumentRecord(_id="doc-1", title="Rome").to_dict()
        {'_id': 'doc-1', 'title': 'Rome'}
    """

    __slots__ = ("_data",)
    _data: dict[str, Any]

    def __init__(self, data: dict[str, Any] | None = None, /, **fields: Any) -> None:
        merged: dict[str, Any] = {**(data or {}), **fields}
        _validate_document_id(merged.get("_id"))
        object.__setattr__(self, "_data", merged)

    @property
    def id(self) -> str:
        """The document's unique identifier (the ``_id`` field)."""
        value: str = self._data["_id"]
        return value

    @property
    def _id(self) -> str:
        return self.id

    def get(self, key: str, default: Any = None) -> Any:
        """Return the value for *key* from the record, or *default*."""
        return self._data.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a plain dictionary (shallow copy), in wire shape."""
        return dict(self._data)

    def to_json(self) -> str:
        """Return the record as a compact JSON string (decoded UTF-8)."""
        return orjson.dumps(self._data).decode()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DocumentRecord):
            return NotImplemented
        return self._data == other._data

    def __repr__(self) -> str:
        extras = sum(1 for k in self._data if k != "_id")
        return f"DocumentRecord(_id={self._data['_id']!r}, fields={extras})"


class UpdateDocumentRecord:
    """A partial update to a document, identified by ``_id``.

    Any field other than ``_id`` and ``_remove_fields`` sets a new value
    for that field. Fields named in ``_remove_fields`` are removed from
    the document. A field cannot be both set and removed in the same
    record, and the ``_id`` is validated with the same rules as
    :class:`DocumentRecord`.

    Examples:
        >>> UpdateDocumentRecord({"_id": "doc-1", "title": "New", "_remove_fields": ["old"]})
        UpdateDocumentRecord(_id='doc-1', set=1, remove=1)
    """

    __slots__ = ("_data",)
    _data: dict[str, Any]

    def __init__(self, data: dict[str, Any] | None = None, /, **fields: Any) -> None:
        merged: dict[str, Any] = {**(data or {}), **fields}
        _validate_document_id(merged.get("_id"))
        remove_fields = merged.get("_remove_fields")
        if remove_fields is not None:
            if not isinstance(remove_fields, list) or not all(
                isinstance(name, str) for name in remove_fields
            ):
                raise ValueError("'_remove_fields' must be a list of field names (strings).")
            overlap = sorted(
                set(remove_fields) & {k for k in merged if k not in ("_id", "_remove_fields")}
            )
            if overlap:
                raise ValueError(
                    f"Fields cannot be both set and removed in the same update: {overlap!r}. "
                    "Remove each name from '_remove_fields' or drop its new value."
                )
        object.__setattr__(self, "_data", merged)

    @property
    def id(self) -> str:
        """The unique identifier of the document to update (the ``_id`` field)."""
        value: str = self._data["_id"]
        return value

    @property
    def _id(self) -> str:
        return self.id

    @property
    def remove_fields(self) -> list[str] | None:
        """The field names to delete from the document, or ``None``."""
        value: list[str] | None = self._data.get("_remove_fields")
        return value

    def get(self, key: str, default: Any = None) -> Any:
        """Return the value for *key* from the update record, or *default*."""
        return self._data.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        """Return the update record as a plain dictionary (shallow copy), in wire shape."""
        return dict(self._data)

    def to_json(self) -> str:
        """Return the update record as a compact JSON string (decoded UTF-8)."""
        return orjson.dumps(self._data).decode()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, UpdateDocumentRecord):
            return NotImplemented
        return self._data == other._data

    def __repr__(self) -> str:
        removes = len(self._data.get("_remove_fields") or [])
        sets = sum(1 for k in self._data if k not in ("_id", "_remove_fields"))
        return f"UpdateDocumentRecord(_id={self._data['_id']!r}, set={sets}, remove={removes})"
