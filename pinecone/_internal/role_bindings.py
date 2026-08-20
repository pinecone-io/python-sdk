"""Shared normalization for the ``role_bindings`` array on principal-creating requests.

The Admin API's ``RoleBindingInput`` schema is reused verbatim by every
operation that creates a principal with initial roles — ``create_invite`` and
``create_service_account`` both embed the same object — so the validation and
wire rendering live here rather than in either namespace.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pinecone.errors.exceptions import ValidationError
from pinecone.models.admin.role_binding import RoleBindingInput

_REQUIRED_BINDING_KEYS = ("resource_type", "role")
_KNOWN_BINDING_KEYS = frozenset({"resource_type", "resource_id", "role"})


def binding_to_payload(binding: RoleBindingInput) -> dict[str, str]:
    """Render one validated binding as the wire object the spec declares.

    ``resource_id`` is omitted rather than sent as ``null`` when unset, so an
    ``organization``-scoped binding matches the spec's "omit for organization
    scope" exactly, and so a binding built from an enum and the same binding
    built from plain strings produce byte-identical bodies.
    """
    payload = {"resource_type": binding.resource_type, "role": binding.role}
    if binding.resource_id is not None:
        payload["resource_id"] = binding.resource_id
    return payload


def normalize_role_bindings(
    role_bindings: Sequence[RoleBindingInput | Mapping[str, Any]],
) -> list[dict[str, str]]:
    """Validate every entry and render the ``role_bindings`` array for the wire.

    Accepts :class:`~pinecone.models.admin.role_binding.RoleBindingInput`
    instances and plain dicts interchangeably. Every failure names the index of
    the offending entry, because the server's own 400 cannot say which one it
    tripped over.
    """
    normalized: list[dict[str, str]] = []
    for index, entry in enumerate(role_bindings):
        if isinstance(entry, RoleBindingInput):
            normalized.append(binding_to_payload(entry))
            continue
        if not isinstance(entry, Mapping):
            raise ValidationError(
                f"role_bindings[{index}] must be a RoleBindingInput or a dict with "
                f"'resource_type' and 'role' keys, got {type(entry).__name__}"
            )
        for key in _REQUIRED_BINDING_KEYS:
            value = entry.get(key)
            if value is None or (isinstance(value, str) and not value.strip()):
                raise ValidationError(
                    f"role_bindings[{index}] missing required key {key!r}; every entry "
                    "needs 'resource_type' ('organization' or 'project') and 'role', "
                    "plus 'resource_id' for 'project' scope"
                )
        unknown = sorted(set(entry) - _KNOWN_BINDING_KEYS)
        if unknown:
            opts = ", ".join(repr(k) for k in unknown)
            raise ValidationError(
                f"role_bindings[{index}] has unrecognized key(s) {opts}; allowed keys are "
                "'resource_type', 'role', and 'resource_id'"
            )
        try:
            binding = RoleBindingInput(
                resource_type=entry["resource_type"],
                role=entry["role"],
                resource_id=entry.get("resource_id"),
            )
        except ValidationError as exc:
            raise ValidationError(f"role_bindings[{index}]: {exc}") from exc
        normalized.append(binding_to_payload(binding))
    return normalized
