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


NAMESPACE_NAME_MAX_LEN = 512
"""Longest namespace name the data plane accepts (``max_namespace_length``)."""

NAMESPACE_LIST_LIMIT_MAX = 100
"""Largest ``limit`` ``listNamespaces`` accepts (``max_list_limit``)."""

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
