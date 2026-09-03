"""Input validation utilities."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, overload

from pinecone.errors.exceptions import ValidationError


@overload
def require_non_empty(name: str, value: str) -> None: ...


@overload
def require_non_empty(name: str, value: list[Any]) -> None: ...


def require_non_empty(name: str, value: str | list[Any]) -> None:
    """Raise ValidationError if value is empty, whitespace-only, or an empty list."""
    if isinstance(value, list):
        if not value:
            raise ValidationError(f"{name} must be a non-empty list")
    else:
        if not value or not value.strip():
            raise ValidationError(f"{name} must be a non-empty string")


def require_positive(name: str, value: int) -> None:
    """Raise ValidationError if value is not a positive integer."""
    if value <= 0:
        raise ValidationError(f"{name} must be a positive integer, got {value}")


def require_in_range(name: str, value: int, min_val: int, max_val: int) -> None:
    """Raise ValidationError if value is not in [min_val, max_val] inclusive."""
    if value < min_val or value > max_val:
        raise ValidationError(f"{name} must be between {min_val} and {max_val}, got {value}")


def require_max_length(name: str, value: str, max_length: int) -> None:
    """Raise ValidationError if value exceeds max_length characters."""
    if len(value) > max_length:
        raise ValidationError(f"{name} is too long (max {max_length} characters)")


def require_one_of(name: str, value: str, allowed: Sequence[str]) -> None:
    """Raise ValidationError if *value* is not in the *allowed* set."""
    if value not in allowed:
        opts = ", ".join(repr(a) for a in allowed)
        raise ValidationError(f"{name} must be one of {opts}, got {value!r}")


def require_rerank_top_n(value: int | None) -> None:
    """Raise ValidationError unless *value* is a legal ``rerank`` result count.

    ``None`` asks for every document back, and any value from 1 up is legal —
    including values above the document count, which return every document.
    """
    if value is not None and value < 1:
        raise ValidationError("top_n must be >= 1")


NAMESPACE_NAME_MAX_LEN = 512
"""Longest namespace name the data plane accepts."""

NAMESPACE_LIST_LIMIT_MAX = 100
"""Largest ``limit`` ``listNamespaces`` accepts."""

RESERVED_DEFAULT_NAMESPACE = "__default__"
"""Alias for the namespace requests address when they omit one.

Legal wherever a namespace is *named* — describe, delete, reads, writes — and
rejected only by ``createNamespace``, since the namespace it aliases always
exists and so cannot be created.
"""


def _echo(value: str, limit: int = 64) -> str:
    """``repr`` of *value*, elided in the middle when too long to read."""
    if len(value) <= limit:
        return repr(value)
    half = limit // 2
    return f"{value[:half]!r}...{value[-half:]!r} ({len(value)} characters)"


def _require_namespace_charset(param: str, value: str) -> None:
    if not value.isascii():
        raise ValidationError(
            f"{param} must contain only ASCII characters (code points 1-127), got {_echo(value)}"
        )
    if "\x00" in value:
        raise ValidationError(f"{param} must not contain the NUL character, got {_echo(value)}")


def require_valid_namespace_name(param: str, value: str) -> None:
    """Raise ValidationError unless *value* is a namespace name the API accepts.

    Names must be ASCII, must not contain the NUL character, and must be
    1-512 characters long. ``__default__`` satisfies these rules; use
    :func:`require_creatable_namespace_name` where it is also reserved.
    """
    if not isinstance(value, str):
        raise ValidationError(f"{param} must be a string, got {type(value).__name__}")
    if not value:
        raise ValidationError(
            f"{param} must be a non-empty string; namespace names must be "
            f"1-{NAMESPACE_NAME_MAX_LEN} characters, got ''"
        )
    if len(value) > NAMESPACE_NAME_MAX_LEN:
        raise ValidationError(
            f"{param} must be 1-{NAMESPACE_NAME_MAX_LEN} characters, "
            f"got {len(value)}: {_echo(value)}"
        )
    _require_namespace_charset(param, value)


def require_creatable_namespace_name(param: str, value: str) -> None:
    """Raise ValidationError unless *value* is a namespace name that can be created.

    Adds the ``createNamespace``-only rule to
    :func:`require_valid_namespace_name`: ``__default__`` names the namespace
    requests address when they omit a namespace, so it always exists and cannot
    be created explicitly.
    """
    require_valid_namespace_name(param, value)
    if value == RESERVED_DEFAULT_NAMESPACE:
        raise ValidationError(
            f"{param}={value!r} is reserved and cannot be created: it names the "
            "namespace requests address when they omit a namespace, so it always "
            "exists. Pass any other name, or omit the namespace to address it."
        )


def require_valid_namespace_prefix(param: str, value: str) -> None:
    """Raise ValidationError unless *value* is a namespace-listing prefix the API accepts.

    Same charset and length rule as the names it filters, except that the empty
    prefix is allowed and matches every namespace.
    """
    if not isinstance(value, str):
        raise ValidationError(f"{param} must be a string, got {type(value).__name__}")
    if len(value) > NAMESPACE_NAME_MAX_LEN:
        raise ValidationError(
            f"{param} must be at most {NAMESPACE_NAME_MAX_LEN} characters, "
            f"got {len(value)}: {_echo(value)}"
        )
    _require_namespace_charset(param, value)


def require_valid_namespace_limit(param: str, value: int) -> None:
    """Raise ValidationError unless *value* is a legal ``listNamespaces`` page size."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{param} must be an integer, got {type(value).__name__}")
    require_in_range(param, value, 1, NAMESPACE_LIST_LIMIT_MAX)


def require_valid_namespace_schema(param: str, value: Any) -> None:
    """Raise ValidationError unless *value* is a metadata schema the API accepts.

    ``filterable`` is required on every field and must be literally ``True``;
    the server rejects anything else with the message quoted in the error.
    Leaving a field unindexed is spelled by omitting it from ``fields``.
    """
    if not isinstance(value, dict):
        raise ValidationError(f"{param} must be a dict, got {type(value).__name__}")
    unknown = sorted(k for k in value if k != "fields")
    if unknown:
        raise ValidationError(
            f"{param} contains unsupported keys {unknown}; the only supported key is 'fields'"
        )
    fields = value.get("fields")
    if fields is None:
        raise ValidationError(f"{param} must contain a 'fields' key, got keys {sorted(value)}")
    if not isinstance(fields, dict):
        raise ValidationError(
            f"{param}['fields'] must be a dict of field name to configuration, "
            f"got {type(fields).__name__}"
        )
    for field, config in fields.items():
        where = f"{param}['fields'][{field!r}]"
        if not isinstance(config, dict):
            raise ValidationError(f"{where} must be a dict, got {type(config).__name__}")
        filterable = config.get("filterable")
        if filterable is not True:
            got = "omitted" if "filterable" not in config else repr(filterable)
            raise ValidationError(
                f"{where}['filterable'] must be True, got {got}: "
                f"Field '{field}' is set to filterable: false. Only filterable: true "
                "is supported. To avoid indexing the field, omit it from the list of fields."
            )


VECTOR_ID_MAX_LEN = 512
"""Longest vector ID the data plane accepts."""

ID_PREFIX_MAX_LEN = 512
"""Longest ``listVectors`` prefix the data plane accepts; the same rule as vector IDs."""

LIST_LIMIT_MAX = 100
"""Largest ``limit`` ``listVectors`` accepts."""

QUERY_TOP_K_MAX = 10_000
"""Largest ``top_k`` ``query`` accepts.

Enforced here so an out-of-range value is reported against the argument that
carried it. The server applies its own ceiling as well, and its error carries
the specifics when the two differ.
"""

FETCH_BY_METADATA_LIMIT_MAX = 10_000
"""Largest ``limit`` ``fetch_by_metadata`` accepts."""

DELETE_EMPTY_FILTER_MESSAGE = "Delete with empty metadata filter is not allowed"
"""``deleteVectors``' own wording for an empty filter, quoted in the client-side error."""

UPDATE_EMPTY_FILTER_MESSAGE = "Update with empty metadata filter is not allowed"
"""``updateVector``' own wording for an empty filter, quoted in the client-side error."""

FETCH_BY_METADATA_EMPTY_FILTER_MESSAGE = "Empty filter provided for fetch by metadata request"
"""``fetch_by_metadata``' own wording for an empty filter, quoted in the client-side error."""


def require_valid_vector_id(param: str, value: Any) -> None:
    """Raise ValidationError unless *value* is a vector ID the API accepts.

    IDs must be ASCII, must not contain the NUL character, and must be
    1-512 characters long — the same rule ``upsert`` applies, worded the same
    way, so one bad ID reads identically whichever operation carries it.
    """
    if not isinstance(value, str):
        raise ValidationError(f"{param} must be a string, got {type(value).__name__}")
    if not value:
        raise ValidationError(
            f"{param} must not be empty; vector IDs are 1-{VECTOR_ID_MAX_LEN} characters"
        )
    if len(value) > VECTOR_ID_MAX_LEN:
        raise ValidationError(
            f"{param} exceeds the maximum length of {VECTOR_ID_MAX_LEN} characters, "
            f"got {len(value)}: {_echo(value)}"
        )
    if not value.isascii():
        raise ValidationError(f"{param} must contain only ASCII characters, got: {_echo(value)}")
    if "\x00" in value:
        raise ValidationError(f"{param} must not contain null characters, got: {_echo(value)}")


def require_valid_vector_ids(param: str, values: Sequence[str]) -> None:
    """Raise ValidationError unless *values* is a non-empty list of legal vector IDs."""
    if not values:
        raise ValidationError(f"{param} must be a non-empty list")
    for position, value in enumerate(values):
        require_valid_vector_id(f"{param}[{position}]", value)


def require_valid_id_prefix(param: str, value: Any) -> None:
    """Raise ValidationError unless *value* is an ID prefix ``listVectors`` accepts.

    Same charset and length rule as the IDs it filters, except that the empty
    prefix is allowed and matches every ID.
    """
    if not isinstance(value, str):
        raise ValidationError(f"{param} must be a string, got {type(value).__name__}")
    if len(value) > ID_PREFIX_MAX_LEN:
        raise ValidationError(
            f"{param} must be at most {ID_PREFIX_MAX_LEN} characters, "
            f"got {len(value)}: {_echo(value)}"
        )
    if not value.isascii():
        raise ValidationError(f"{param} must contain only ASCII characters, got: {_echo(value)}")
    if "\x00" in value:
        raise ValidationError(f"{param} must not contain null characters, got: {_echo(value)}")


def require_valid_list_limit(param: str, value: Any) -> None:
    """Raise ValidationError unless *value* is a legal ``listVectors`` page size."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{param} must be an integer, got {type(value).__name__}")
    require_in_range(param, value, 1, LIST_LIMIT_MAX)


def require_valid_fetch_by_metadata_limit(param: str, value: Any) -> None:
    """Raise ValidationError unless *value* is a legal ``fetch_by_metadata`` page size."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{param} must be an integer, got {type(value).__name__}")
    require_in_range(param, value, 1, FETCH_BY_METADATA_LIMIT_MAX)


def require_non_empty_filter(param: str, value: Any, *, server_message: str) -> None:
    """Raise ValidationError unless *value* carries at least one filter condition.

    An empty filter reads as "match nothing" but the server rejects it outright,
    so *server_message* lets the caller see the same bytes either way.
    """
    if not isinstance(value, Mapping):
        raise ValidationError(f"{param} must be a dict, got {type(value).__name__}")
    if not value:
        raise ValidationError(
            f"{param} must contain at least one condition, got {{}}. {server_message}"
        )


def require_query_selectors(*, vector: Any, id: Any, sparse_vector: Any) -> None:
    """Raise ValidationError unless the query names a legal selector set.

    2026-07 admits ``vector``, ``sparse_vector``, both together (hybrid), or
    ``id`` alone. ``id`` names a vector the index already holds, so pairing it
    with literal vector data of either form is contradictory.
    """
    has_vector = vector is not None
    has_id = id is not None
    has_sparse = sparse_vector is not None

    if has_id and has_vector:
        raise ValidationError(
            "id is mutually exclusive with vector — a query uses a stored vector's id "
            "OR literal vector data, not both. "
            "Pass id alone to query by stored vector, or vector alone to query by value. "
            "Cannot provide both 'ID' and 'vector' at the same time"
        )
    if has_id and has_sparse:
        raise ValidationError(
            "id is mutually exclusive with sparse_vector — a query uses a stored vector's id "
            "OR literal vector data, not both. "
            "Pass id alone to query by stored vector, or sparse_vector alone to query by value. "
            "Cannot provide both 'ID' and 'sparse_vector' at the same time"
        )
    if not (has_vector or has_id or has_sparse):
        raise ValidationError("At least one of vector, id, or sparse_vector must be provided")


def require_delete_selectors(*, ids: Any, delete_all: bool, filter: Any) -> None:
    """Raise ValidationError unless the delete names exactly one selector.

    Deliberately stricter than the 2026-07 ``anyOf``, which admits ``ids``
    alongside ``filter``: the server never inspects ``ids`` once a filter is
    present, so such a request deletes everything the filter matches rather
    than the intersection the caller almost certainly meant.
    """
    has_ids = ids is not None
    has_filter = filter is not None
    if sum([has_ids, delete_all, has_filter]) == 0:
        raise ValidationError("Must specify one of ids, delete_all, or filter")
    if delete_all and has_ids:
        raise ValidationError(
            "Cannot combine ids and delete_all — specify exactly one. "
            "delete_all=True already covers every record in the namespace. "
            "No explicit IDs allowed when delete_all=true"
        )
    if delete_all and has_filter:
        raise ValidationError(
            "Cannot combine delete_all and filter — specify exactly one. "
            "delete_all=True already covers every record in the namespace. "
            "No filter allowed when delete_all=true"
        )
    if has_ids and has_filter:
        raise ValidationError(
            "Cannot combine ids and filter — specify exactly one. "
            "The server silently ignores ids when a filter is present, so a request "
            "carrying both would delete every record the filter matches, not the "
            "intersection. To delete the intersection, query with the filter first "
            "and delete the returned ids."
        )


def require_update_selectors(*, id: Any, filter: Any, values: Any, sparse_values: Any) -> None:
    """Raise ValidationError unless the update names one target, consistently.

    2026-07 admits ``id`` or ``filter``, and forbids pairing ``filter`` with
    either form of vector values: a by-filter update spans every record the
    filter matches, so it can only set metadata, never vector data belonging
    to one record.
    """
    has_id = id is not None
    has_filter = filter is not None
    if has_id and has_filter:
        raise ValidationError("Exactly one of id or filter must be provided, not both")
    if not has_id and not has_filter:
        raise ValidationError("Exactly one of id or filter must be provided, got neither")
    if has_filter and values is not None:
        raise ValidationError(
            "filter is mutually exclusive with values — a by-filter update is "
            "metadata-only, because it spans every record the filter matches. "
            "Pass set_metadata to update metadata by filter, or id to update one "
            "record's vector values. "
            "Update by metadata request does not support updating vector values."
        )
    if has_filter and sparse_values is not None:
        raise ValidationError(
            "filter is mutually exclusive with sparse_values — a by-filter update is "
            "metadata-only, because it spans every record the filter matches. "
            "Pass set_metadata to update metadata by filter, or id to update one "
            "record's sparse values. "
            "Update by metadata request does not support updating vector values."
        )


_RESOURCE_NAME_MAX_LEN = 45
_RESOURCE_NAME_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789-")


def require_valid_resource_name(name: str, value: str) -> None:
    """Raise ValidationError if value is not a valid Pinecone resource name.

    Valid names are non-empty, at most 45 characters, consist only of lowercase
    alphanumeric characters and hyphens, and must not start or end with a hyphen.
    """
    if not value or not value.strip():
        raise ValidationError(f"{name} must be a non-empty string")
    if len(value) > _RESOURCE_NAME_MAX_LEN:
        raise ValidationError(
            f"{name} is too long (max {_RESOURCE_NAME_MAX_LEN} characters, got {len(value)})"
        )
    if value[0] == "-":
        raise ValidationError(f"{name} must not start with a hyphen")
    if value[-1] == "-":
        raise ValidationError(f"{name} must not end with a hyphen")
    if not all(c in _RESOURCE_NAME_CHARS for c in value):
        raise ValidationError(
            f"{name} contains invalid characters; must be lowercase alphanumeric and hyphens only"
        )


_TAG_KEY_RE = re.compile(r"^[a-zA-Z0-9_-]+$")
_MAX_TAGS = 20
_MAX_TAG_KEY_LEN = 80
_MAX_TAG_VAL_LEN = 120


def validate_index_tags(tags: Mapping[str, str] | None) -> None:
    """Validate index tags against the 2026-07 IndexTags contract.

    Keys: 1-80 ASCII alphanumerics, ``_`` or ``-``. Values: 0-120 printable
    ASCII characters. At most 20 tags. Raises
    :class:`~pinecone.errors.exceptions.PineconeValueError` naming the
    offending key and the limit.
    """
    if tags is None:
        return
    if len(tags) > _MAX_TAGS:
        raise ValidationError(f"tags exceeded the maximum of {_MAX_TAGS}. Got {len(tags)} tags.")
    for key, value in tags.items():
        if not key:
            raise ValidationError("tags contains an empty key; tag keys must be 1-80 characters.")
        if len(key) > _MAX_TAG_KEY_LEN:
            raise ValidationError(
                f"Tag key {key!r} exceeds the {_MAX_TAG_KEY_LEN}-character limit."
            )
        if not _TAG_KEY_RE.match(key):
            raise ValidationError(
                f"Tag key {key!r} has invalid characters. Must be alphanumeric or '_', '-'."
            )
        if len(value) > _MAX_TAG_VAL_LEN:
            raise ValidationError(
                f"Tag value for key {key!r} exceeds the {_MAX_TAG_VAL_LEN}-character limit."
            )
        if not value.isascii() or not value.isprintable():
            raise ValidationError(
                f"Tag value for key {key!r} contains invalid characters. "
                "Only printable ASCII characters are allowed."
            )
