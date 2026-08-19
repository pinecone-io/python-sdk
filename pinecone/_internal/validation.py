"""Input validation utilities."""

from __future__ import annotations

from collections.abc import Sequence
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
        raise ValidationError(
            f"{name} must be between {min_val} and {max_val}, got {value}. "
            f"Pass a {name} within that range."
        )


def require_max_length(name: str, value: str, max_length: int) -> None:
    """Raise ValidationError if value exceeds max_length characters."""
    if len(value) > max_length:
        raise ValidationError(f"{name} is too long (max {max_length} characters)")


def require_one_of(name: str, value: str, allowed: Sequence[str]) -> None:
    """Raise ValidationError if *value* is not in the *allowed* set."""
    if value not in allowed:
        opts = ", ".join(repr(a) for a in allowed)
        raise ValidationError(f"{name} must be one of {opts}, got {value!r}")


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
