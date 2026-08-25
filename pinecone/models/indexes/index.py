"""Index and IndexStatus response models (2026-07 API)."""

from __future__ import annotations

from typing import Any

import msgspec
from msgspec import Struct

from pinecone._internal.config import normalize_host
from pinecone.models._display import render_table
from pinecone.models._mixin import StructDictMixin
from pinecone.models.indexes.deployment import IndexDeployment
from pinecone.models.indexes.read_capacity import ReadCapacityResponse
from pinecone.models.indexes.schema import (
    DenseVectorField,
    IndexSchema,
    SparseVectorField,
    _strip_untyped_tags,
)

__all__ = ["IndexModel", "IndexStatus", "IndexTags"]


class IndexStatus(StructDictMixin, Struct, kw_only=True):
    """Status of an index.

    Attributes:
        ready: Whether the index is ready to accept requests.
        state: Current state of the index. Possible values:
            ``"Initializing"``, ``"InitializationFailed"``, ``"ScalingUp"``,
            ``"ScalingDown"``, ``"ScalingUpPodSize"``, ``"ScalingDownPodSize"``,
            ``"Terminating"``, ``"Ready"``, or ``"Disabled"``.
    """

    ready: bool
    state: str


class IndexTags(dict):  # type: ignore[type-arg]
    """A dict subclass for index tags that adds a ``to_dict()`` helper."""

    def to_dict(self) -> dict[str, str]:
        return dict(self)


_REMOVED_FIELD_HINTS: dict[str, str] = {
    "dimension": (
        "read it from the schema's dense_vector field instead, e.g. "
        "next(f.dimension for f in index.schema.fields.values() "
        "if type(f).__name__ == 'DenseVectorField')"
    ),
    "metric": (
        "read it from the schema's vector field instead, e.g. "
        "index.schema.fields['<field-name>'].metric"
    ),
    "vector_type": (
        "inspect the schema's field types instead: a DenseVectorField in "
        "index.schema.fields means dense, a SparseVectorField means sparse"
    ),
    "spec": (
        "use index.deployment instead — a ManagedDeployment, PodDeployment, "
        "or ByocDeployment tagged on deployment_type; read_capacity is now "
        "top-level at index.read_capacity"
    ),
    "embed": (
        "integrated-embedding configuration now appears as a SemanticTextField "
        "in index.schema.fields"
    ),
    "created_at": "the 2026-07 API does not return a creation timestamp",
}


class IndexModel(Struct, kw_only=True):
    """Response model for a Pinecone index (2026-07 API).

    Attributes:
        name: The name of the index.
        host: The hostname where this index is served, or ``None`` if the
            index is still initializing and has not yet been assigned a host.
        private_host: The private-endpoint hostname for this index when the
            project has Private Endpoints configured, or ``None`` otherwise.
            Clients inside a VPC should connect to this host instead of
            ``host``.
        status: Current status of the index.
        schema: Field-level schema definition (vector, text, and metadata
            fields), keyed by field name.
        deployment: Deployment configuration — a
            :class:`~pinecone.models.indexes.deployment.ManagedDeployment`,
            :class:`~pinecone.models.indexes.deployment.PodDeployment`, or
            :class:`~pinecone.models.indexes.deployment.ByocDeployment`,
            discriminated on ``deployment_type``.
        deletion_protection: Whether deletion protection is enabled
            (``"enabled"`` or ``"disabled"``).
        read_capacity: Read capacity configuration and status, or ``None``
            if the server response omits it.
        tags: User-defined key-value tags attached to the index, or ``None``
            if no tags are set (the API returns ``"tags": null`` rather
            than ``{}``).
        source_collection: Name of the collection this index was created
            from, or ``None``.
        source_backup_id: ID of the backup this index was restored from,
            or ``None``.
        cmek_id: ID of the customer-managed encryption key protecting this
            index, or ``None`` if CMEK is not configured.
    """

    name: str
    status: IndexStatus
    schema: IndexSchema
    deployment: IndexDeployment
    deletion_protection: str
    host: str | None = None
    read_capacity: ReadCapacityResponse | None = None
    tags: dict[str, str] | None = None
    private_host: str | None = None
    source_collection: str | None = None
    source_backup_id: str | None = None
    cmek_id: str | None = None

    def __post_init__(self) -> None:
        """Normalize hosts to include the https:// scheme; wrap tags in IndexTags."""
        if self.host is not None:
            self.host = normalize_host(self.host)
        if self.private_host is not None:
            self.private_host = normalize_host(self.private_host)
        if isinstance(self.tags, dict) and not isinstance(self.tags, IndexTags):
            self.tags = IndexTags(self.tags)

    def __getattr__(self, name: str) -> Any:
        if name in ("dimension", "metric", "vector_type"):
            raise self._legacy_vector_accessor_error(name)
        if name in _REMOVED_FIELD_HINTS:
            raise AttributeError(
                f"IndexModel.{name} was removed in the 2026-07 Pinecone API: "
                f"{_REMOVED_FIELD_HINTS[name]}. See docs/migration/v10-2026-07-index-model.md."
            )
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def _dense_fields(self) -> list[DenseVectorField]:
        return [f for f in self.schema.fields.values() if isinstance(f, DenseVectorField)]

    def _sparse_fields(self) -> list[SparseVectorField]:
        return [f for f in self.schema.fields.values() if isinstance(f, SparseVectorField)]

    def _legacy_vector_accessor_error(self, name: str) -> AttributeError:
        dense = self._dense_fields()
        if len(dense) > 1:
            names = ", ".join(
                sorted(k for k, f in self.schema.fields.items() if isinstance(f, DenseVectorField))
            )
            return AttributeError(
                f"IndexModel.{name} is ambiguous: the schema has {len(dense)} dense "
                f"vector fields ({names}); there is no single field to resolve this "
                "deprecated accessor to. Read the specific field directly, e.g. "
                "index.schema.fields['<field-name>']."
            )
        if name == "vector_type":
            sparse = self._sparse_fields()
            if len(sparse) > 1:
                names = ", ".join(
                    sorted(
                        k for k, f in self.schema.fields.items() if isinstance(f, SparseVectorField)
                    )
                )
                return AttributeError(
                    f"IndexModel.vector_type is ambiguous: the schema has {len(sparse)} "
                    f"sparse vector fields ({names}) and no dense vector field; there "
                    "is no single field to resolve this deprecated accessor to."
                )
        return AttributeError(
            f"IndexModel.{name} was removed in the 2026-07 Pinecone API: "
            f"{_REMOVED_FIELD_HINTS[name]}. See docs/migration/v10-2026-07-index-model.md."
        )

    @property
    def dimension(self) -> int:
        """**Deprecated.** Use ``index.schema.fields["<field-name>"].dimension`` instead."""
        dense = self._dense_fields()
        if len(dense) != 1:
            raise AttributeError("dimension")
        return dense[0].dimension

    @property
    def metric(self) -> str:
        """**Deprecated.** Use ``index.schema.fields["<field-name>"].metric`` instead."""
        dense = self._dense_fields()
        if len(dense) != 1:
            raise AttributeError("metric")
        return dense[0].metric

    @property
    def vector_type(self) -> str:
        """**Deprecated.** Inspect ``index.schema.fields`` field types instead."""
        dense = self._dense_fields()
        if len(dense) > 1:
            raise AttributeError("vector_type")
        if dense:
            return "dense"
        sparse = self._sparse_fields()
        if len(sparse) > 1:
            raise AttributeError("vector_type")
        if sparse:
            return "sparse"
        raise AttributeError("vector_type")

    def __getitem__(self, key: str) -> Any:
        if key not in self.__struct_fields__:
            raise KeyError(key)
        return getattr(self, key)

    def __contains__(self, key: object) -> bool:
        return key in self.__struct_fields__

    def __dir__(self) -> list[str]:
        attrs = set(super().__dir__())
        public = {name for name in attrs if not name.startswith("_")}
        return sorted(public)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict representation, recursively converting nested fields.

        Nested structs (``status``, ``schema``, ``deployment``,
        ``read_capacity``) become plain dicts. Tagged-union members include
        their discriminator key (``deployment_type``, ``mode``, ``type``);
        legacy untyped schema fields are emitted without a ``type`` key,
        matching the wire format. Optional fields that are ``None`` are
        included with their ``None`` values.
        """
        result: dict[str, Any] = {
            field: _to_builtins_stripped(getattr(self, field)) for field in self.__struct_fields__
        }
        return result

    def __repr__(self) -> str:
        dep_name = type(self.deployment).__name__.replace("Deployment", "")
        parts = [
            f"name={self.name!r}",
            f"status={self.status.state!r}",
            f"host={self.host!r}",
            f"deployment={dep_name!r}",
            f"deletion_protection={self.deletion_protection!r}",
        ]
        if self.schema.fields:
            parts.append(f"schema_fields={len(self.schema.fields)}")
        if self.tags:
            parts.append(f"tags={len(self.tags)} items")
        if self.private_host is not None:
            parts.append(f"private_host={self.private_host!r}")
        return f"IndexModel({', '.join(parts)})"

    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        """Pretty-printer support for IPython."""
        if cycle:
            p.text("IndexModel(...)")
            return
        p.text("IndexModel(")
        with p.group(2, "", ")"):
            p.breakable()
            p.text(f"name={self.name!r},")
            p.breakable()
            p.text(f"status={self.status.state!r},")
            p.breakable()
            p.text(f"host={self.host!r},")
            p.breakable()
            p.text(f"deployment={self.deployment!r},")
            p.breakable()
            p.text(f"deletion_protection={self.deletion_protection!r},")
            p.breakable()
            p.text(f"schema=IndexSchema(fields={len(self.schema.fields)} fields),")
            if self.read_capacity is not None:
                p.breakable()
                p.text(f"read_capacity={self.read_capacity!r},")
            if self.tags:
                p.breakable()
                p.text(f"tags={self.tags!r},")
            if self.private_host is not None:
                p.breakable()
                p.text(f"private_host={self.private_host!r},")

    def _repr_html_(self) -> str:
        """Jupyter notebook HTML representation."""
        dep_name = type(self.deployment).__name__.replace("Deployment", "")
        dep_detail = ""
        if hasattr(self.deployment, "cloud") and hasattr(self.deployment, "region"):
            cloud = getattr(self.deployment, "cloud", "")
            region = getattr(self.deployment, "region", "")
            dep_detail = f" ({cloud}/{region})"
        elif hasattr(self.deployment, "environment"):
            dep_detail = f" ({getattr(self.deployment, 'environment', '')})"

        rows: list[tuple[str, str | int]] = [
            ("Name:", self.name),
            ("Status:", self.status.state),
            ("Ready:", "Yes" if self.status.ready else "No"),
            ("Deployment:", f"{dep_name}{dep_detail}"),
            ("Host:", self.host if self.host is not None else "not yet assigned"),
            ("Deletion Protection:", self.deletion_protection),
            ("Schema fields:", len(self.schema.fields)),
        ]
        if self.read_capacity is not None:
            rows.append(
                ("Read capacity:", getattr(self.read_capacity, "mode", str(self.read_capacity)))
            )
        if self.tags:
            tags_str = ", ".join(f"{k}={v}" for k, v in self.tags.items())
            rows.append(("Tags:", tags_str))
        return render_table("IndexModel", rows)


def _to_builtins_stripped(value: Any) -> Any:
    if isinstance(value, Struct):
        return _strip_untyped_tags(msgspec.to_builtins(value))
    if isinstance(value, dict):
        return dict(value)
    return value
