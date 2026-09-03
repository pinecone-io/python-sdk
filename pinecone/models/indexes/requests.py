"""The typed request bodies behind index create and configure.

You do not build these yourself — you pass keyword arguments to
:meth:`~pinecone.client.indexes.Indexes.create` or
:meth:`~pinecone.client.indexes.Indexes.configure` and the SDK assembles the
request through them, which is where client-side validation happens. Both omit
their unset optional fields, so a configure sends only what you asked to
change.
"""

from __future__ import annotations

from typing import Any

from msgspec import Struct

from pinecone.errors.exceptions import PineconeValueError
from pinecone.models.indexes.deployment import IndexDeployment
from pinecone.models.indexes.schema import IndexSchema

__all__ = ["ConfigureIndexRequest", "CreateIndexRequest"]

_ALLOWED_DEPLOYMENT_TYPES = ("managed", "pod", "byoc")
_MAX_FIELD_NAME_LENGTH = 64


def validate_schema_field_name(name: str) -> None:
    """Check one schema field name for length before the request is sent.

    Names must be 1-64 characters. Which names are reserved (``_id``,
    ``_values``, ``_sparse_values``) or otherwise special (a leading ``$``) is
    the server's call rather than the SDK's — a client-side copy of that list
    would drift the moment the server's rules changed — so only length is
    checked here.

    Raises:
        PineconeValueError: If the name is empty or too long. The message
            names the field, the rule it broke, and the fix.
    """
    if not name:
        raise PineconeValueError(
            "Invalid schema field name '': field names must be 1-64 characters. "
            "Provide a non-empty field name."
        )
    if len(name) > _MAX_FIELD_NAME_LENGTH:
        raise PineconeValueError(
            f"Invalid schema field name {name!r}: {len(name)} characters exceeds "
            f"the maximum length of {_MAX_FIELD_NAME_LENGTH}. Shorten the field name."
        )


def _validate_schema(schema: dict[str, Any] | IndexSchema) -> None:
    fields = schema.get("fields") if isinstance(schema, dict) else schema.fields
    if not isinstance(fields, dict):
        return
    for field_name in fields:
        if not isinstance(field_name, str):
            raise PineconeValueError(
                f"Invalid schema field name {field_name!r}: expected a str key, "
                f"got {type(field_name).__name__}. Schema field names must be strings."
            )
        validate_schema_field_name(field_name)


def _validate_deployment(deployment: dict[str, Any] | IndexDeployment | None) -> None:
    if not isinstance(deployment, dict):
        return
    deployment_type = deployment.get("deployment_type")
    if deployment_type is not None and deployment_type not in _ALLOWED_DEPLOYMENT_TYPES:
        allowed = " | ".join(_ALLOWED_DEPLOYMENT_TYPES)
        raise PineconeValueError(
            f"Invalid deployment_type {deployment_type!r}: expected one of {allowed}. "
            "Set deployment={'deployment_type': 'managed', 'cloud': ..., 'region': ...} "
            "for a serverless index."
        )


class CreateIndexRequest(Struct, kw_only=True, omit_defaults=True):
    """The body :meth:`~pinecone.client.indexes.Indexes.create` sends.

    Assembled from that method's keyword arguments; the field descriptions
    there are the ones to read. ``schema`` is the only required member.

    Attributes:
        schema: What the index will hold, as an
            :class:`~pinecone.models.indexes.schema.IndexSchema` or the
            equivalent dict. Only searchable fields are declared —
            ``dense_vector``, ``sparse_vector``, or ``string`` with a
            ``full_text_search`` config.
        name: Name for the index; the server assigns one when omitted.
        deployment: Where the index runs, discriminated on
            ``deployment_type`` (``managed``, ``pod`` or ``byoc``). Omitted
            means a managed index on AWS ``us-east-1``.
        read_capacity: Read capacity for a managed or BYOC index.
        deletion_protection: ``"enabled"`` or ``"disabled"``.
        tags: Key-value tags to attach to the index.
        source_collection: Name of a collection to seed the index from.
        source_backup_id: ID of a backup to restore the index from.
        cmek_id: Customer-managed encryption key to encrypt the index with.
            Accepted for managed and BYOC indexes with no full-text search
            field.

    Raises:
        PineconeValueError: If ``deployment`` names a ``deployment_type``
            outside the three above. The comparison is case-sensitive, so
            ``"MANAGED"`` is rejected as well as a genuine typo.
    """

    schema: dict[str, Any] | IndexSchema
    name: str | None = None
    deployment: dict[str, Any] | IndexDeployment | None = None
    read_capacity: dict[str, Any] | None = None
    deletion_protection: str | None = None
    tags: dict[str, str] | None = None
    source_collection: str | None = None
    source_backup_id: str | None = None
    cmek_id: str | None = None

    def __post_init__(self) -> None:
        _validate_schema(self.schema)
        _validate_deployment(self.deployment)


class ConfigureIndexRequest(Struct, kw_only=True, omit_defaults=True):
    """The body :meth:`~pinecone.client.indexes.Indexes.configure` sends.

    Assembled from that method's keyword arguments; the field descriptions
    there are the ones to read. Every member is optional and anything left
    unset stays out of the request, so a configure changes only what you
    named.

    Attributes:
        schema: Schema changes. Only a ``semantic_text`` field's parameters
            can be changed; fields cannot be added or removed.
        deployment: Deployment changes, for pod-based indexes only —
            ``replicas`` and ``pod_type``, with no ``deployment_type`` key.
        read_capacity: Replacement read capacity configuration.
        deletion_protection: ``"enabled"`` or ``"disabled"``.
        tags: Tags to merge into the existing ones. Setting a key to ``""``
            deletes it.
    """

    schema: dict[str, Any] | IndexSchema | None = None
    deployment: dict[str, Any] | None = None
    read_capacity: dict[str, Any] | None = None
    deletion_protection: str | None = None
    tags: dict[str, str] | None = None
