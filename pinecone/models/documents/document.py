"""The document you write, the update you patch with, and the document you read back.

Documents are open-schema: apart from the reserved keys, a document is whatever fields
you put in it. So these models wrap a plain ``dict`` rather than declaring a fixed field
set, and a field the SDK has never heard of survives a decode and a ``to_dict()``
unchanged.

Two conventions run through the whole package, and they are easy to conflate:

* **On the wire and in the dicts you pass in**, the reserved keys are spelled with a
  leading underscore — ``_id``, ``_score``, ``_remove_fields``. They sit in the same dict
  as your own fields but they are directives, not fields: ``_id`` names the document,
  ``_score`` is how well it matched, ``_remove_fields`` lists fields to delete.
* **On the Python objects you read**, use the plain properties — ``doc.id``, ``doc.score``.
  ``doc._id`` and ``doc._score`` exist as aliases so wire-shaped code keeps working, but
  ``doc.id`` is the spelling to write.
"""

from __future__ import annotations

import re
from typing import Any

import orjson

__all__ = ["Document", "DocumentRecord", "UpdateDocumentRecord"]

_MAX_ID_LENGTH = 512
_ID_PATTERN = re.compile(r"^[\x01-\x7F]+$")


def _validate_document_id(value: Any) -> str:
    """Check a document ``_id`` and return it, or raise naming the rule it broke.

    An ID must be a string of 1 to 512 ASCII characters matching
    ``^[\\x01-\\x7F]+$`` — no NUL byte and no non-ASCII characters.

    Raises:
        ValueError: If *value* is not a string, is empty, is too long, or contains a
            character outside that range. The message names the constraint violated.
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
    """One document a search or fetch returned: its ID, its score, and your own fields.

    **Read the identifier as ``doc.id``.** ``doc._id`` returns the same string and is kept
    so that code written against the wire shape keeps working, but ``doc.id`` is the
    canonical spelling in this SDK and the one every example uses. The same holds for
    ``doc.score`` over ``doc._score``. The underscore forms belong to the JSON, not to
    your Python.

    Your own fields are reachable as attributes (``doc.title``) and through
    :meth:`get`; which of them are present depends on the ``include_fields`` the
    operation asked for. An absent field raises :exc:`AttributeError` on attribute
    access, so use :meth:`get` when a field is optional.

    The ``id``, ``_id``, ``score`` and ``_score`` properties always win over a document
    field of the same name. If your data genuinely has a field called ``_score``, reach it
    with ``doc.get("_score")`` or ``doc.to_dict()["_score"]``.

    Attributes:
        id (str): The document's identifier — the value the document was upserted under.
        _id (str): Alias for :attr:`id`, matching the JSON key. Prefer :attr:`id`.
        score (float | None): How well the document matched, or ``None`` when the response
            carried no score. A fetch returns documents without scores, so ``None`` there
            is normal rather than a sign anything went wrong.
        _score (float | None): Alias for :attr:`score`, matching the JSON key. Prefer
            :attr:`score`.

    Examples:
        >>> from pinecone import Document
        >>> doc = Document({"_id": "article-42", "_score": 0.891, "title": "Rome"})
        >>> doc.id, doc.score
        ('article-42', 0.891)
        >>> doc.title
        'Rome'
        >>> doc.get("subtitle", "n/a")
        'n/a'

        A fetched document has no score:

        >>> Document({"_id": "article-42", "title": "Rome"}).score is None
        True
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
        """Read a field without risking :exc:`AttributeError` if it is absent.

        Behaves like :meth:`dict.get` over the document. Reserved keys are readable here
        under their JSON names — ``get("_id")`` and ``get("_score")`` — which is also the
        only way to reach a field of your own that happens to be named ``_score``.

        Args:
            key (str): Field name to read, e.g. ``"title"``.
            default (Any): What to return when the field is absent. Defaults to ``None``.

        Returns:
            The field's value, or *default*.

        Examples:
            >>> from pinecone import Document
            >>> doc = Document({"_id": "article-42", "title": "Rome"})
            >>> doc.get("title"), doc.get("subtitle", "n/a")
            ('Rome', 'n/a')
        """
        return self._data.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        """Return the document as a plain dict, back in its JSON shape.

        A shallow copy, so mutating it does not touch the document. The reserved keys come
        back under their JSON names — ``_id`` and ``_score`` — alongside your own fields,
        which makes this the form to re-serialize or to feed to a
        :class:`DocumentRecord`.

        Returns:
            A ``dict`` of every key the document carries.
        """
        return dict(self._data)

    def to_json(self) -> str:
        """Return the document as a compact JSON string, already decoded to ``str``."""
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
    """A document you are about to upsert, checked before it can reach the wire.

    A record is one required reserved key, ``_id``, plus as many fields of your own as you
    like. ``_id`` sits in the same dict as those fields but it is not one of them: it
    names the document rather than storing anything, and the leading underscore is what
    marks it as reserved. Passing keyword arguments instead of a dict makes the split
    plain, since ``_id=`` reads as the argument it is.

    Your field values are the same shapes metadata takes — a string, a number, a boolean,
    or a list of strings. A field the index schema declares as ``dense_vector`` takes a
    list of floats, and one declared ``sparse_vector`` takes sparse values. Field values
    are checked against the index schema server-side, so a type mismatch surfaces on
    upsert rather than here.

    Only the ``_id`` is validated on construction — a string of 1 to 512 ASCII characters
    — so a bad ID is reported at the line that wrote it.

    Args:
        data (dict[str, Any] | None): The record as a dict, positional-only. Must carry
            ``_id``. Merged with, and overridden by, any keyword fields.
        **fields (Any): Fields given as keyword arguments, including ``_id``.

    Raises:
        ValueError: If ``_id`` is missing, is not a string, is empty, is over 512
            characters, or contains a non-ASCII character or NUL.

    Examples:
        Building from keyword arguments keeps the reserved ``_id`` visually distinct from
        the fields you are storing:

        >>> from pinecone import DocumentRecord
        >>> DocumentRecord(_id="article-42", title="Rome", lang="en").to_dict()
        {'_id': 'article-42', 'title': 'Rome', 'lang': 'en'}

        A dict works too, which is the form to use when your documents already arrive as
        JSON. Here ``_id`` is the reserved key naming the document and ``title`` is a
        field being stored, even though they sit side by side:

        >>> DocumentRecord({"_id": "article-42", "title": "Rome"})
        DocumentRecord(_id='article-42', fields=1)

    .. seealso::
       :class:`UpdateDocumentRecord` — for changing fields on a document that already
       exists, rather than replacing it wholesale.
    """

    __slots__ = ("_data",)
    _data: dict[str, Any]

    def __init__(self, data: dict[str, Any] | None = None, /, **fields: Any) -> None:
        merged: dict[str, Any] = {**(data or {}), **fields}
        _validate_document_id(merged.get("_id"))
        object.__setattr__(self, "_data", merged)

    @property
    def id(self) -> str:
        """The identifier this record will be stored under. The spelling to prefer."""
        value: str = self._data["_id"]
        return value

    @property
    def _id(self) -> str:
        """Alias for :attr:`id`, matching the JSON key. Prefer :attr:`id`."""
        return self.id

    def get(self, key: str, default: Any = None) -> Any:
        """Read a field, or the reserved ``_id``, without raising when it is absent.

        Args:
            key (str): Field name to read, e.g. ``"title"``.
            default (Any): What to return when the field is absent. Defaults to ``None``.

        Returns:
            The field's value, or *default*.
        """
        return self._data.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a plain dict in JSON shape, with ``_id`` among the fields.

        A shallow copy, so mutating it does not touch the record.
        """
        return dict(self._data)

    def to_json(self) -> str:
        """Return the record as a compact JSON string, already decoded to ``str``."""
        return orjson.dumps(self._data).decode()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DocumentRecord):
            return NotImplemented
        return self._data == other._data

    def __repr__(self) -> str:
        extras = sum(1 for k in self._data if k != "_id")
        return f"DocumentRecord(_id={self._data['_id']!r}, fields={extras})"


class UpdateDocumentRecord:
    """A patch to one existing document: the fields to change, and the fields to drop.

    This is a partial update, so fields you do not mention are left exactly as they are —
    that is the difference from :class:`DocumentRecord`, which replaces the document.

    Two reserved keys shape the patch, and neither one stores a value: ``_id`` says which
    document to patch, and ``_remove_fields`` lists field names to delete from it. Every
    *other* key is a field being set to a new value. So in a patch dict the underscore
    keys are instructions and the plain keys are data, even though they appear in the same
    dict at the same level. Keyword arguments make that split easier to read.

    A field cannot be both set and removed in one patch; asking for both is rejected here
    rather than resolved silently in one direction.

    Args:
        data (dict[str, Any] | None): The patch as a dict, positional-only. Must carry
            ``_id``. Merged with, and overridden by, any keyword fields.
        **fields (Any): Fields to set, given as keyword arguments; ``_id`` and
            ``_remove_fields`` may be passed this way too.

    Raises:
        ValueError: If ``_id`` is missing or invalid, if ``_remove_fields`` is not a list
            of strings, or if a field appears both as a new value and in
            ``_remove_fields`` — the message names the overlapping fields.

    Examples:
        Set two fields and leave the rest of the document alone:

        >>> from pinecone import UpdateDocumentRecord
        >>> UpdateDocumentRecord(_id="article-42", title="Rome", lang="en").remove_fields is None
        True

        Set one field and delete another in the same patch. ``_id`` and ``_remove_fields``
        are the two reserved keys here; ``title`` is the only field being written:

        >>> UpdateDocumentRecord(
        ...     {"_id": "article-42", "title": "Rome", "_remove_fields": ["draft_notes"]}
        ... )
        UpdateDocumentRecord(_id='article-42', set=1, remove=1)

    .. seealso::
       :class:`DocumentRecord` — for writing a whole document, where unmentioned fields do
       not survive.
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
        """The identifier of the document this patch applies to. The spelling to prefer."""
        value: str = self._data["_id"]
        return value

    @property
    def _id(self) -> str:
        """Alias for :attr:`id`, matching the JSON key. Prefer :attr:`id`."""
        return self.id

    @property
    def remove_fields(self) -> list[str] | None:
        """Field names this patch deletes, or ``None`` when it only sets values."""
        value: list[str] | None = self._data.get("_remove_fields")
        return value

    def get(self, key: str, default: Any = None) -> Any:
        """Read one entry of the patch, reserved keys included, without raising.

        Args:
            key (str): Name to read — a field being set, or ``"_id"`` or
                ``"_remove_fields"``.
            default (Any): What to return when it is absent. Defaults to ``None``.

        Returns:
            The value, or *default*.
        """
        return self._data.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        """Return the patch as a plain dict in JSON shape, reserved keys included.

        A shallow copy, so mutating it does not touch the record.
        """
        return dict(self._data)

    def to_json(self) -> str:
        """Return the patch as a compact JSON string, already decoded to ``str``."""
        return orjson.dumps(self._data).decode()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, UpdateDocumentRecord):
            return NotImplemented
        return self._data == other._data

    def __repr__(self) -> str:
        removes = len(self._data.get("_remove_fields") or [])
        sets = sum(1 for k in self._data if k not in ("_id", "_remove_fields"))
        return f"UpdateDocumentRecord(_id={self._data['_id']!r}, set={sets}, remove={removes})"
