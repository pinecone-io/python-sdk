"""Vector input format parsing — normalizes user inputs to canonical Vector objects."""

from __future__ import annotations

import json
from typing import Any

from pinecone.errors.exceptions import PineconeTypeError, PineconeValueError
from pinecone.models.vectors.sparse import SparseValues
from pinecone.models.vectors.vector import Vector

_RECOGNIZED_KEYS = {"id", "values", "sparse_values", "metadata"}

_METADATA_DISPLAY_LENGTH = 16

_METADATA_SCALARS = (str, bool, int, float)

_METADATA_SEQUENCES = (list, tuple, set, frozenset)


def _numbers_as_floats(value: Any) -> Any:
    """Normalize numbers to floats, as protobuf ``Struct`` does before the server renders them.

    ``Struct``'s only numeric kind is ``double``, so an integer reaches the server's error
    message as ``10.0``, not ``10`` (pinecone-db ``pc-utils/src/prost_util.rs:89-94``).
    Reproducing that is what makes the SDK's message byte-identical to the server's rather than
    merely similar.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        return {k: _numbers_as_floats(v) for k, v in value.items()}
    if isinstance(value, _METADATA_SEQUENCES):
        return [_numbers_as_floats(v) for v in value]
    return value


def _render_metadata_value(value: Any) -> str:
    """Render *value* the way the server renders it when it rejects it.

    Compact JSON, truncated to :data:`_METADATA_DISPLAY_LENGTH` UTF-8 *bytes* — the server
    counts bytes — with a ``...`` suffix. pinecone-db
    ``pc-metadata-filtering/src/metadata/compiler.rs:246-254`` over
    ``pc-utils/src/string_util.rs:68``.
    """
    try:
        rendered = json.dumps(
            _numbers_as_floats(value), separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
    except (TypeError, ValueError):
        # Not JSON-encodable, so the server would never see this shape and has no rendering of
        # it to match. The value still has to appear in the message.
        rendered = repr(value)
    encoded = rendered.encode("utf-8")
    if len(encoded) > _METADATA_DISPLAY_LENGTH:
        return encoded[:_METADATA_DISPLAY_LENGTH].decode("utf-8", "ignore") + "..."
    return rendered


def _reject_metadata_value(field: str, value: Any) -> None:
    raise PineconeTypeError(
        f"Metadata value must be a string, number, boolean or list of strings, "
        f"got '{_render_metadata_value(value)}' for field '{field}'"
    )


def _validate_metadata(metadata: dict[str, Any]) -> None:
    """Reject metadata values the server would reject, before the request is sent.

    The server's rule (pinecone-db ``pc-metadata-filtering/src/metadata/compiler.rs:218-281``):
    every value is a string, a number, a boolean, or a list whose every element is a string. A
    nested object, or a list with a non-string in it, is a 400 — and on a batch upsert it is a
    400 for the whole batch, which is what makes the scan worth its cost.

    ``None`` is deliberately accepted. The server *strips* null metadata values on write rather
    than refusing them (``NullHandling::Strip``, ``pc-utils/src/prost_util.rs:75-126``), so a
    caller passing ``{"tag": None}`` has been silently dropping that key all along; raising here
    would break working code.

    ``tuple``, ``set`` and ``frozenset`` are validated alongside ``list`` because msgspec encodes
    all four to a JSON array, so all four reach the server as a list and are judged by the same
    rule.
    """
    for field, value in metadata.items():
        value_class = value.__class__
        if (
            value_class is str
            or value_class is bool
            or value_class is int
            or value_class is float
            or value is None
        ):
            continue
        if value_class is list or isinstance(value, _METADATA_SEQUENCES):
            for element in value:
                if not isinstance(element, str):
                    _reject_metadata_value(field, value)
            continue
        if isinstance(value, _METADATA_SCALARS):
            continue
        _reject_metadata_value(field, value)


def _check_metadata(metadata: Any) -> None:
    """Validate a user-supplied ``metadata`` argument: ``None``, or a dict of legal values."""
    if metadata is None:
        return
    if not isinstance(metadata, dict):
        raise PineconeTypeError(f"metadata must be a dict, got {type(metadata).__name__}")
    _validate_metadata(metadata)


def _truncate(value: Any, limit: int = 80) -> str:
    """Show the offending value without pasting a whole embedding into a message."""
    text = repr(value)
    return text if len(text) <= limit else f"{text[:limit]}... ({len(text)} chars)"


def _as_list(raw: Any) -> list[Any]:
    """Convert array-like *raw* to a list of Python scalars.

    numpy and pyarrow expose ``.tolist()``, which unboxes in C. ``list()`` on a
    numpy float32 array instead allocates one ``np.float32`` object per element —
    2246ms and +248MB per 20k x 384 rows — and every one of those has to be
    unboxed again by whatever consumes the list.
    """
    tolist = getattr(raw, "tolist", None)
    if tolist is None:
        return list(raw)
    converted = tolist()
    # A 0-d array yields a bare scalar; fall through so it raises as before.
    return converted if isinstance(converted, list) else list(raw)


def _from_tuple(item: tuple[Any, ...]) -> Vector:
    length = len(item)
    if length == 2:
        id_, values = item
        if not isinstance(id_, str):
            raise PineconeTypeError(f"Vector ID must be a string, got {type(id_).__name__}")
        if not id_.isascii():
            raise PineconeValueError(f"Vector ID must contain only ASCII characters, got: {id_!r}")
        if "\x00" in id_:
            raise PineconeValueError(f"Vector ID must not contain null characters, got: {id_!r}")
        converted = values if isinstance(values, list) else _as_list(values)
        if not converted:
            raise PineconeValueError(
                "Vector must have at least one of non-empty dense values or sparse values"
            )
        return Vector(id_, converted)
    if length == 3:
        id_, values, metadata = item
        if not isinstance(id_, str):
            raise PineconeTypeError(f"Vector ID must be a string, got {type(id_).__name__}")
        if not id_.isascii():
            raise PineconeValueError(f"Vector ID must contain only ASCII characters, got: {id_!r}")
        if "\x00" in id_:
            raise PineconeValueError(f"Vector ID must not contain null characters, got: {id_!r}")
        _check_metadata(metadata)
        converted = values if isinstance(values, list) else _as_list(values)
        if not converted:
            raise PineconeValueError(
                "Vector must have at least one of non-empty dense values or sparse values"
            )
        return Vector(id_, converted, None, metadata)
    raise PineconeValueError(f"Vector tuple must have 2 or 3 elements, got {length}")


def _from_dict(item: dict[str, Any]) -> Vector:
    try:
        id_ = item["id"]
    except KeyError as err:
        raise PineconeValueError("Vector dict must contain an 'id' key") from err
    if not isinstance(id_, str):
        raise PineconeTypeError(f"Vector ID must be a string, got {type(id_).__name__}")
    if not id_.isascii():
        raise PineconeValueError(f"Vector ID must contain only ASCII characters, got: {id_!r}")
    if "\x00" in id_:
        raise PineconeValueError(f"Vector ID must not contain null characters, got: {id_!r}")

    # Fast path: common 2-key dict {"id": ..., "values": ...}
    item_len = len(item)
    if item_len == 2 and "values" in item:
        raw_values = item["values"]
        values = raw_values if isinstance(raw_values, list) else _as_list(raw_values)
        if not values:
            raise PineconeValueError(
                "Vector must have at least one of non-empty dense values or sparse values"
            )
        return Vector(id_, values)

    # Fast path: 3-key dict {"id": ..., "values": ..., "sparse_values": ...}
    if item_len == 3 and "values" in item and "sparse_values" in item:
        raw_values = item["values"]
        values = raw_values if isinstance(raw_values, list) else _as_list(raw_values)
        sv = _parse_sparse(item["sparse_values"])
        return Vector(id_, values, sv, None)

    # General path: validate keys and extract optional fields
    if not _RECOGNIZED_KEYS.issuperset(item):
        extra = item.keys() - _RECOGNIZED_KEYS
        raise PineconeValueError(f"Vector dict contains unrecognized keys: {sorted(extra)}")

    raw_values = item.get("values")
    values = (
        (raw_values if isinstance(raw_values, list) else _as_list(raw_values))
        if raw_values is not None
        else []
    )

    raw_sparse = item.get("sparse_values")
    sparse: SparseValues | None = None
    if raw_sparse is not None:
        sparse = _parse_sparse(raw_sparse)

    metadata = item.get("metadata")
    _check_metadata(metadata)

    if not values and sparse is None:
        raise PineconeValueError(
            "Vector must have at least one of non-empty dense values or sparse values"
        )

    return Vector(id_, values, sparse, metadata)


def _parse_sparse(raw: Any) -> SparseValues:
    if not isinstance(raw, dict):
        raise PineconeTypeError(f"sparse_values must be a dict, got {type(raw).__name__}")
    if "indices" not in raw or "values" not in raw:
        missing = []
        if "indices" not in raw:
            missing.append("indices")
        if "values" not in raw:
            missing.append("values")
        raise PineconeValueError(f"sparse_values dict is missing required keys: {missing}")

    indices = raw["indices"]
    values = raw["values"]

    if len(indices) != len(values):
        raise PineconeValueError(
            f"sparse_values indices and values must have the same length, "
            f"got {len(indices)} and {len(values)}"
        )

    if indices and not isinstance(indices[0], int):
        raise PineconeTypeError(
            f"sparse_values indices must be integers, got {type(indices[0]).__name__}"
        )
    if values and not isinstance(values[0], (int, float)):
        raise PineconeTypeError(
            f"sparse_values values must be floats, got {type(values[0]).__name__}"
        )

    return SparseValues(
        indices if isinstance(indices, list) else _as_list(indices),
        (
            values
            if isinstance(values, list) and (not values or isinstance(values[0], float))
            else [float(v) for v in values]
        ),
    )


def _has_no_values(raw: Any) -> bool:
    """Whether *raw* holds no dense values, without materializing it to find out.

    ``size`` covers numpy and pyarrow; anything else that reached here supports
    ``len``. Neither iterates, which matters because the caller is deliberately
    avoiding a full pass over the array.
    """
    if getattr(raw, "ndim", None) == 0:
        # A 0-d array is a scalar, not a sequence of values, but its `size` is 1
        # so the check below would call it non-empty. Iterating raises exactly
        # what `list()` raises on the VectorFactory path, keeping the two in step.
        iter(raw)
    size = getattr(raw, "size", None)
    if size is not None:
        return bool(size == 0)
    return len(raw) == 0


def _at(row: int | None, id_: Any) -> str:
    """Locate a bad row for a caller who has to filter a frame from the message alone.

    Row index and vector id are both included because an agent acting on this
    has one or the other to hand and cannot be relied on to have a particular one.
    """
    if row is None:
        return ""
    return f" (row {row}, id={id_!r})"


def validate_vector_dict(item: dict[str, Any], *, row: int | None = None) -> None:
    """Apply :class:`VectorFactory`'s checks to a vector dict, converting nothing.

    The gRPC DataFrame path hands array values straight to the Rust layer, which
    unboxes them in C. Skipping :meth:`VectorFactory.build` skips its validation
    too, so the same checks — and the same exception types, which callers catch
    on — have to happen here instead.

    Args:
        item: The vector dict to validate.
        row: Position in the source DataFrame, included in messages when given.
    """
    try:
        id_ = item["id"]
    except KeyError as err:
        raise PineconeValueError(
            f"Vector dict must contain an 'id' key{_at(row, None)}; "
            f"got keys {sorted(item)}. Add an 'id' column of unique string ids."
        ) from err
    if not isinstance(id_, str):
        raise PineconeTypeError(
            f"Vector ID must be a string, got {type(id_).__name__} {id_!r}{_at(row, id_)}. "
            f"Convert the id column to str, e.g. df['id'] = df['id'].astype(str)."
        )
    if not id_.isascii():
        raise PineconeValueError(
            f"Vector ID must contain only ASCII characters, got: {id_!r}{_at(row, id_)}. "
            f"Re-encode or replace the non-ASCII characters in this id."
        )
    if "\x00" in id_:
        raise PineconeValueError(
            f"Vector ID must not contain null characters, got: {id_!r}{_at(row, id_)}. "
            f"Strip the null bytes from this id."
        )

    if not _RECOGNIZED_KEYS.issuperset(item):
        extra = item.keys() - _RECOGNIZED_KEYS
        raise PineconeValueError(
            f"Vector dict contains unrecognized keys: {sorted(extra)}{_at(row, id_)}. "
            f"Only {sorted(_RECOGNIZED_KEYS)} are accepted; drop the extra column(s) "
            f"or move them into 'metadata'."
        )

    raw_values = item.get("values")
    has_values = raw_values is not None and not _has_no_values(raw_values)

    raw_sparse = item.get("sparse_values")
    if raw_sparse is not None:
        try:
            _parse_sparse(raw_sparse)
        except (PineconeTypeError, PineconeValueError) as exc:
            # _parse_sparse is shared with the conversion path, which has no row
            # to report. Re-raise the same type with the location appended, so a
            # bad sparse cell is as locatable as a bad metadata one.
            raise type(exc)(f"{exc}{_at(row, id_)}") from exc

    metadata = item.get("metadata")
    if metadata is not None:
        if not isinstance(metadata, dict):
            raise PineconeTypeError(
                f"metadata must be a dict, got {type(metadata).__name__} "
                f"{_truncate(metadata)}{_at(row, id_)}. Pass a dict of metadata fields, "
                f"or None where a row has none."
            )
        _validate_metadata(metadata)

    if not has_values and raw_sparse is None:
        raise PineconeValueError(
            f"Vector must have at least one of non-empty dense values or sparse "
            f"values{_at(row, id_)}. Give this row a non-empty 'values' array or a "
            f"'sparse_values' dict."
        )


class VectorFactory:
    """Converts user-provided vector inputs into canonical ``Vector`` objects.

    Accepted formats:

    - ``Vector`` instance (passthrough)
    - ``tuple`` of 2 elements ``(id, values)`` or 3 elements ``(id, values, metadata)``
    - ``dict`` with keys drawn from ``{"id", "values", "sparse_values", "metadata"}``

    Metadata values are checked against the grammar the server enforces — string, number,
    boolean, or list of strings — and a violation raises
    :class:`~pinecone.errors.exceptions.PineconeTypeError` carrying the server's own message.
    A batch upsert is rejected server-side in its entirety, so finding the one bad value here is
    worth more than the isinstance chain costs.
    """

    @staticmethod
    def build(item: Any) -> Vector:
        """Convert a user-provided vector input to a ``Vector`` object."""
        item_type = item.__class__
        if item_type is Vector:
            if not item.values and item.sparse_values is None:
                raise PineconeValueError(
                    "Vector must have at least one of non-empty dense values or sparse values"
                )
            if item.metadata is not None:
                _validate_metadata(item.metadata)
            return item  # type: ignore[no-any-return]
        if item_type is dict:
            item_len = len(item)
            # Inline 2-key happy path to avoid function call overhead
            if item_len == 2:
                try:
                    id_ = item["id"]
                except KeyError:
                    return _from_dict(item)
                if id_.__class__ is str and id_.isascii() and "\x00" not in id_:
                    try:
                        raw_values = item["values"]
                    except KeyError:
                        return _from_dict(item)
                    converted = raw_values if raw_values.__class__ is list else _as_list(raw_values)
                    if converted:
                        return Vector(id_, converted)
            elif item_len == 3:
                # Inline 3-key sparse happy path: {"id", "values", "sparse_values"}
                try:
                    id_ = item["id"]
                except KeyError:
                    return _from_dict(item)
                if id_.__class__ is str and id_.isascii() and "\x00" not in id_:
                    try:
                        raw_values = item["values"]
                        raw_sparse = item["sparse_values"]
                    except KeyError:
                        return _from_dict(item)
                    if raw_sparse.__class__ is dict:
                        try:
                            s_indices = raw_sparse["indices"]
                            s_values = raw_sparse["values"]
                        except KeyError:
                            return _from_dict(item)
                        if (
                            s_indices.__class__ is list
                            and s_values.__class__ is list
                            and len(s_indices) == len(s_values)
                            and (not s_indices or s_indices[0].__class__ is int)
                        ):
                            converted = (
                                raw_values if raw_values.__class__ is list else _as_list(raw_values)
                            )
                            if not s_values or s_values[0].__class__ is float:
                                return Vector(
                                    id_, converted, SparseValues(s_indices, s_values), None
                                )
                            if isinstance(s_values[0], (int, float)):
                                return Vector(
                                    id_,
                                    converted,
                                    SparseValues(s_indices, [float(v) for v in s_values]),
                                    None,
                                )
                            # else: non-numeric type → fall through to _from_dict
                            # for proper PineconeTypeError via _parse_sparse
            return _from_dict(item)
        if item_type is tuple:
            # Inline 2-element happy path to avoid function call overhead
            if len(item) == 2:
                id_, values = item
                if id_.__class__ is str and id_.isascii() and "\x00" not in id_:
                    converted = values if values.__class__ is list else _as_list(values)
                    if converted:
                        return Vector(id_, converted)
            return _from_tuple(item)
        # Subclass fallback
        if isinstance(item, Vector):
            if not item.values and item.sparse_values is None:
                raise PineconeValueError(
                    "Vector must have at least one of non-empty dense values or sparse values"
                )
            if item.metadata is not None:
                _validate_metadata(item.metadata)
            return item
        if isinstance(item, tuple):
            return _from_tuple(item)
        if isinstance(item, dict):
            return _from_dict(item)
        raise PineconeTypeError(f"Expected Vector, tuple, or dict, got {type(item).__name__}")
