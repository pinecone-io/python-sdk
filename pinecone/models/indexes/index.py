"""The model an index describe returns, and its status."""

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
    """Whether an index can serve requests yet, and what it is busy doing.

    Branch on :attr:`ready`; read :attr:`state` when you want to say *why* an
    index is not ready, or to distinguish a scaling operation from a failed
    initialization.

    Attributes:
        ready: Whether the index accepts requests. This is the field to poll,
            and the one :meth:`create <pinecone.client.indexes.Indexes.create>`
            waits on for you unless you passed ``timeout=-1``.
        state: What the index is doing, as a readable label —
            ``"Initializing"``, ``"InitializationFailed"``, ``"ScalingUp"``,
            ``"ScalingDown"``, ``"ScalingUpPodSize"``, ``"ScalingDownPodSize"``,
            ``"Terminating"``, ``"Ready"``, or ``"Disabled"``. An index can
            report ``ready`` while a scaling state is in progress, so the two
            answer different questions.
    """

    ready: bool
    state: str


class IndexTags(dict):  # type: ignore[type-arg]
    """An index's tags: an ordinary dict, plus ``to_dict()`` for symmetry.

    ``IndexModel`` wraps whatever tags come back in this so that every nested
    model on the response answers ``to_dict()``.
    """

    def to_dict(self) -> dict[str, str]:
        return dict(self)


#: Names that resolve through a deprecated computed property rather than a
#: struct field. Attribute, item, ``in`` and ``to_dict()`` access all read it.
_LEGACY_VECTOR_ACCESSORS: tuple[str, ...] = ("dimension", "metric", "vector_type")

_MIGRATION_GUIDE = "https://sdk.pinecone.io/python/migration/v10-migration.html"

_REMOVED_FIELD_HINTS: dict[str, str] = {
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


def _removed_field_message(name: str) -> str:
    """Build the guided message for a field the 2026-07 API no longer returns.

    Shared by the attribute and mapping paths so ``index.created_at`` and
    ``index["created_at"]`` explain the removal in the same words.
    """
    return (
        f"IndexModel.{name} was removed in the 2026-07 Pinecone API: "
        f"{_REMOVED_FIELD_HINTS[name]}. See {_MIGRATION_GUIDE}."
    )


class IndexModel(Struct, kw_only=True):
    """Everything the control plane knows about one index.

    What :meth:`describe <pinecone.client.indexes.Indexes.describe>`,
    :meth:`create <pinecone.client.indexes.Indexes.create>` and
    :meth:`configure <pinecone.client.indexes.Indexes.configure>` return, and
    what iterating :meth:`list <pinecone.client.indexes.Indexes.list>` yields.
    Two fields carry most of the traffic: ``status.ready`` is what you poll
    to know the index can serve requests, and :attr:`host` is what
    you hand to :meth:`Pinecone.index() <pinecone.Pinecone.index>` to get a
    data-plane client. Everything about the index's shape — dimension, metric,
    which fields are searchable — is in :attr:`schema`.

    Attributes:
        name: The name of the index.
        host: Where the index is served. Pass it to
            :meth:`Pinecone.index() <pinecone.Pinecone.index>` to open a
            data-plane client. ``None`` while the index is still initializing
            and has not been assigned one.
        private_host: The private-endpoint hostname for this index when the
            project has Private Endpoints configured, or ``None`` otherwise.
            Clients inside a VPC should connect to this host instead of
            ``host``.
        status: An :class:`IndexStatus`; ``status.ready`` is the field to poll.
        schema: An :class:`~pinecone.models.indexes.schema.IndexSchema` naming
            every field in the index and what each can do — where dimension,
            metric and vector type live.
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

    Examples:
        >>> idx = pc.indexes.describe("my-index")
        >>> idx.status.ready, idx.deployment.cloud, idx.deployment.region
        (True, 'aws', 'us-east-1')
        >>> index = pc.index(host=idx.host)

    ``IndexModel`` also reads like a mapping: ``index["host"]`` and
    ``"host" in index`` work for every attribute above, and
    :meth:`to_dict` returns the same key set.

    .. versionchanged:: 10.0
       ``dimension``, ``metric``, ``vector_type``, ``spec``, ``embed`` and
       ``created_at`` are no longer plain attributes.

       ``dimension``, ``metric`` and ``vector_type`` survive as deprecated
       properties that resolve when the schema has exactly one vector field,
       and every spelling agrees: ``index.metric``, ``index["metric"]``,
       ``"metric" in index`` and the ``"metric"`` key of :meth:`to_dict` all
       answer from the same schema lookup. On a schema where the accessor is
       ambiguous — two dense fields, say — the attribute raises
       :exc:`AttributeError`, the item access raises :exc:`KeyError` carrying
       that same explanation, ``in`` is ``False``, and :meth:`to_dict` omits
       the key.

       ``spec``, ``embed`` and ``created_at`` are gone. Reading one raises
       :exc:`AttributeError`, and ``index["spec"]`` a :exc:`KeyError`, with
       the replacement named in the message.
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
        if name in _LEGACY_VECTOR_ACCESSORS:
            raise self._legacy_vector_accessor_error(name)
        if name in _REMOVED_FIELD_HINTS:
            raise AttributeError(_removed_field_message(name))
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
        if name in ("vector_type", "metric"):
            sparse = self._sparse_fields()
            if len(sparse) > 1:
                names = ", ".join(
                    sorted(
                        k for k, f in self.schema.fields.items() if isinstance(f, SparseVectorField)
                    )
                )
                return AttributeError(
                    f"IndexModel.{name} is ambiguous: the schema has {len(sparse)} "
                    f"sparse vector fields ({names}) and no dense vector field; there "
                    "is no single field to resolve this deprecated accessor to."
                )
        return AttributeError(
            f"IndexModel.{name} could not be determined: the schema has no "
            "dense or sparse vector fields to infer it from. Inspect "
            "index.schema.fields directly to see what fields are defined."
        )

    @property
    def dimension(self) -> int | None:
        """Width of the schema's sole dense vector field.

        ``None`` for a sparse-only schema, since sparse vectors have no fixed
        dimension. Raises :exc:`AttributeError` when the schema has more than
        one dense field, or no vector field at all: there is no single field to
        resolve to, and the message says which fields it found.

        Also readable as ``index["dimension"]``, testable with
        ``"dimension" in index``, and present in :meth:`to_dict` — all three
        resolve exactly when this property does.

        .. deprecated:: 10.0
           Read ``index.schema.fields["<field-name>"].dimension`` instead.
        """
        dense = self._dense_fields()
        if len(dense) == 1:
            return dense[0].dimension
        if not dense and self._sparse_fields():
            return None
        raise AttributeError("dimension")

    @property
    def metric(self) -> str:
        """Metric of the schema's sole dense vector field.

        Resolves to ``"dotproduct"`` for a schema whose only vector field is
        sparse, since sparse scoring is always dot product. Raises
        :exc:`AttributeError` when more than one field could answer.

        Also readable as ``index["metric"]``, testable with
        ``"metric" in index``, and present in :meth:`to_dict` — all three
        resolve exactly when this property does.

        .. deprecated:: 10.0
           Read ``index.schema.fields["<field-name>"].metric`` instead.
        """
        dense = self._dense_fields()
        if len(dense) == 1:
            return dense[0].metric
        if not dense:
            sparse = self._sparse_fields()
            if len(sparse) == 1:
                return "dotproduct"
        raise AttributeError("metric")

    @property
    def vector_type(self) -> str:
        """``"dense"`` or ``"sparse"``, for a schema with one vector field.

        Raises :exc:`AttributeError` when the schema has several fields of one
        kind — a hybrid schema has no single vector type to report.

        Also readable as ``index["vector_type"]``, testable with
        ``"vector_type" in index``, and present in :meth:`to_dict` — all
        three resolve exactly when this property does.

        .. deprecated:: 10.0
           Inspect the field types in ``index.schema.fields`` instead.
        """
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

    def _legacy_accessor_resolves(self, key: str) -> bool:
        try:
            getattr(self, key)
        except AttributeError:
            return False
        return True

    def __getitem__(self, key: str) -> Any:
        """Return the value for *key*, including the deprecated accessors.

        ``index["dimension"]``, ``index["metric"]`` and
        ``index["vector_type"]`` answer whatever the like-named property
        answers. When the property cannot resolve — an ambiguous schema, or
        one with no vector field — the :exc:`KeyError` carries that same
        explanation rather than a bare key name, as does a key removed in
        10.0 such as ``"created_at"``.
        """
        if key in self.__struct_fields__:
            return getattr(self, key)
        if key in _REMOVED_FIELD_HINTS:
            raise KeyError(_removed_field_message(key))
        if key in _LEGACY_VECTOR_ACCESSORS:
            try:
                return getattr(self, key)
            except AttributeError as exc:
                raise KeyError(str(exc)) from None
        raise KeyError(key)

    def __contains__(self, key: object) -> bool:
        """Return whether *key* is readable through :meth:`__getitem__`.

        ``True`` for every struct field, and for a deprecated accessor that
        resolves against this index's schema. ``False`` for one that would
        raise, and for a key removed in 10.0.
        """
        if key in self.__struct_fields__:
            return True
        if key in _LEGACY_VECTOR_ACCESSORS:
            return self._legacy_accessor_resolves(str(key))
        return False

    def __dir__(self) -> list[str]:
        attrs = set(super().__dir__())
        public = {name for name in attrs if not name.startswith("_")}
        return sorted(public)

    def to_dict(self) -> dict[str, Any]:
        """Return the whole model as nested plain dicts, for logging or JSON.

        ``status``, ``schema``, ``deployment`` and ``read_capacity`` become
        dicts too, each keeping the key that identifies which variant it is
        (``deployment_type``, ``mode``, ``type``). A
        :class:`~pinecone.models.indexes.schema.LegacyMetadataField` is
        emitted without a ``type``, matching the wire format. Optional fields
        that are ``None`` are present with a ``None`` value rather than
        omitted, so the key set is the same for every index.

        The deprecated ``dimension``, ``metric`` and ``vector_type`` keys are
        included whenever the like-named property resolves, which is what 9.x
        emitted. An index whose schema makes one of them ambiguous omits that
        key rather than guessing. ``spec``, ``embed`` and ``created_at`` are
        not emitted at all.

        The result is still accepted by ``msgspec.convert(d, IndexModel)``,
        which ignores the three derived keys. It is not constructor input:
        ``IndexModel(**d)`` rejects them, and never built a usable model
        anyway, since the nested values are dicts rather than the structs the
        fields are typed as.
        """
        result: dict[str, Any] = {
            field: _to_builtins_stripped(getattr(self, field)) for field in self.__struct_fields__
        }
        for key in _LEGACY_VECTOR_ACCESSORS:
            try:
                result[key] = getattr(self, key)
            except AttributeError:
                continue
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
