"""The ``spec=`` objects that 9.x-style index creation takes.

:class:`ServerlessSpec`, :class:`PodSpec` and :class:`ByocSpec` are deprecated
sugar: the SDK translates each into the ``deployment=`` (and where present the
``read_capacity=``) that :meth:`~pinecone.client.indexes.Indexes.create` now
takes directly. :class:`EmbedConfig` is not deprecated — it is one of the
shapes :meth:`~pinecone.client.indexes.Indexes.create_for_model` accepts for
``embed=``.
"""

from __future__ import annotations

from typing import Any

from msgspec import Struct

from pinecone.models._mixin import StructDictMixin


class EmbedConfig(Struct, frozen=True, kw_only=True):
    """Which model embeds your text, and which field it reads.

    One of the shapes :meth:`~pinecone.client.indexes.Indexes.create_for_model`
    accepts for ``embed=`` — a plain dict with the same keys works too. The
    field it names comes back on the created index as a
    :class:`~pinecone.models.indexes.schema.SemanticTextField`, and the model
    cannot be changed afterwards.

    Attributes:
        model: Embedding model to use, e.g. ``"multilingual-e5-large"``. See
            :class:`~pinecone.models.enums.EmbedModel`.
        field_map: Which document field holds the text to embed, as
            ``{"text": "<your field name>"}`` — e.g.
            ``{"text": "chunk_text"}``.
        dimension: Output width to ask the model for, when it supports more
            than one. ``None`` takes the model's own dimension. Note that
            :meth:`to_dict` omits this field; ``create_for_model`` reads the
            attribute directly, so the create path is unaffected, but a dict
            you build with :meth:`to_dict` loses it.
        metric: How similarity is scored, or ``None`` for the model default.
        read_parameters: Extra arguments passed to the model when embedding a
            query, e.g. ``{"input_type": "query"}``.
        write_parameters: Extra arguments passed to the model when embedding
            an upsert, e.g. ``{"input_type": "passage"}``.
    """

    model: str
    field_map: dict[str, str]
    dimension: int | None = None
    metric: str | None = None
    read_parameters: dict[str, Any] | None = None
    write_parameters: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict of ``model``, ``field_map`` and metric.

        ``read_parameters`` and ``write_parameters`` come out as empty dicts
        rather than being omitted when they were never set, and ``dimension``
        is left out entirely — pass the :class:`EmbedConfig` itself to
        ``create_for_model``, which reads the attribute, rather than the
        output of this method.
        """
        result: dict[str, Any] = {
            "model": self.model.value if hasattr(self.model, "value") else self.model,
            "field_map": self.field_map,
        }
        if self.metric is not None:
            result["metric"] = self.metric.value if hasattr(self.metric, "value") else self.metric
        result["read_parameters"] = self.read_parameters if self.read_parameters is not None else {}
        result["write_parameters"] = (
            self.write_parameters if self.write_parameters is not None else {}
        )
        return result


class IntegratedSpec(StructDictMixin, Struct, frozen=True, kw_only=True):  # type: ignore[misc]
    """Cloud, region and embedding config bundled into one 9.x-style spec.

    Unlike its sibling specs this one has no ``deployment=`` translation, so
    passing it as ``spec=`` to :meth:`~pinecone.client.indexes.Indexes.create`
    raises :exc:`~pinecone.errors.exceptions.PineconeTypeError` rather than being rewritten. Call
    :meth:`~pinecone.client.indexes.Indexes.create_for_model` with ``cloud``,
    ``region`` and ``embed`` instead — the same three values, as arguments.

    Attributes:
        cloud: Public cloud to run in, e.g. ``"aws"``.
        region: Region within that cloud, e.g. ``"us-east-1"``.
        embed: An :class:`EmbedConfig`.

    .. deprecated:: 10.0
       Pass ``cloud=``, ``region=`` and ``embed=`` to ``create_for_model()``.
    """

    cloud: str
    region: str
    embed: EmbedConfig


class ServerlessSpec(StructDictMixin, Struct, frozen=True, kw_only=True, omit_defaults=True):  # type: ignore[misc]
    """A serverless index, described the 9.x way.

    Deprecated sugar for :meth:`~pinecone.client.indexes.Indexes.create`'s
    ``spec=``: the SDK turns it into a managed ``deployment=``, lifting any
    ``read_capacity`` out to the top level as it goes. ``spec=`` and
    ``deployment=`` are mutually exclusive.

    Attributes:
        cloud: Public cloud to run in, e.g. ``"aws"``.
        region: Region within that cloud, e.g. ``"us-east-1"``.
        read_capacity: Read capacity configuration, or ``None`` for the
            default.
        schema: Not translated. A schema set here does not reach the create
            request, and a ``create()`` call that relied on it fails with
            :exc:`~pinecone.errors.exceptions.PineconeValueError` saying
            ``schema is required`` — which reads as though you passed none.
            Pass ``schema=`` to ``create()`` directly.

    .. deprecated:: 10.0
       Pass ``deployment={"deployment_type": "managed", "cloud": ...,
       "region": ...}`` instead.
    """

    cloud: str
    region: str
    read_capacity: dict[str, Any] | None = None
    schema: dict[str, Any] | None = None

    def asdict(self) -> dict[str, Any]:
        """Return the 9.x request shape, ``{"serverless": {...}}``."""
        body: dict[str, Any] = {"cloud": self.cloud, "region": self.region}
        if self.read_capacity is not None:
            body["read_capacity"] = self.read_capacity
        if self.schema is not None:
            body["schema"] = self.schema
        return {"serverless": body}


class PodSpec(StructDictMixin, Struct, frozen=True, kw_only=True):  # type: ignore[misc]
    """A pod-based index, described the 9.x way.

    Deprecated sugar for :meth:`~pinecone.client.indexes.Indexes.create`'s
    ``spec=``, translated into a pod ``deployment=``. ``spec=`` and
    ``deployment=`` are mutually exclusive. The struct still carries every
    9.x field so an old spec object survives a round trip, but two of them
    have nowhere to go in a create request and are rejected rather than
    dropped.

    Attributes:
        environment: The environment hosting the index, e.g.
            ``"us-east-1-aws"``.
        pod_type: Hardware family and size. Defaults to ``"p1.x1"``.
        replicas: How many copies of the index to run. Defaults to 1.
        shards: How many pods to split the data across. Defaults to 1.
        pods: Total pod count, kept only for 9.x compatibility. Leave it at
            its default of 1 or set it to exactly ``replicas * shards``;
            anything else raises
            :exc:`~pinecone.errors.exceptions.PineconeValueError`, because
            there is no independent pod count to translate it into.
        metadata_config: Rejected with
            :exc:`~pinecone.errors.exceptions.PineconeTypeError` when set —
            metadata fields are indexed automatically at upsert, so there is
            nothing to declare at create time.
        source_collection: Rejected with
            :exc:`~pinecone.errors.exceptions.PineconeTypeError` when set.
            Use :meth:`Pinecone.create_index_from_backup
            <pinecone.Pinecone.create_index_from_backup>` to restore a backup
            instead.

    .. deprecated:: 10.0
       Pass ``deployment={"deployment_type": "pod", "environment": ...,
       "pod_type": ..., "replicas": ..., "shards": ...}`` instead.
    """

    environment: str
    pod_type: str = "p1.x1"
    replicas: int = 1
    shards: int = 1
    pods: int = 1
    metadata_config: dict[str, Any] | None = None
    source_collection: str | None = None

    def asdict(self) -> dict[str, Any]:
        """Return the 9.x request shape, ``{"pod": {...}}``."""
        body: dict[str, Any] = {
            "environment": self.environment,
            "pod_type": self.pod_type,
            "replicas": self.replicas,
            "shards": self.shards,
            "pods": self.pods,
        }
        if self.metadata_config is not None:
            body["metadata_config"] = self.metadata_config
        if self.source_collection is not None:
            body["source_collection"] = self.source_collection
        return {"pod": body}


class ByocSpec(StructDictMixin, Struct, frozen=True, kw_only=True, omit_defaults=True):  # type: ignore[misc]
    """A BYOC index, described the 9.x way.

    Deprecated sugar for :meth:`~pinecone.client.indexes.Indexes.create`'s
    ``spec=``, translated into a BYOC ``deployment=`` with any
    ``read_capacity`` lifted to the top level. ``spec=`` and ``deployment=``
    are mutually exclusive.

    Attributes:
        environment: The BYOC environment to run in, e.g.
            ``"aws-us-east-1-b921"``.
        read_capacity: Read capacity configuration, or ``None`` for the
            default.
        schema: Not translated, exactly as on :class:`ServerlessSpec`. Pass
            ``schema=`` to ``create()`` directly.

    .. deprecated:: 10.0
       Pass ``deployment={"deployment_type": "byoc", "environment": ...}``
       instead.
    """

    environment: str
    read_capacity: dict[str, Any] | None = None
    schema: dict[str, Any] | None = None

    def asdict(self) -> dict[str, Any]:
        """Return the 9.x request shape, ``{"byoc": {...}}``."""
        body: dict[str, Any] = {"environment": self.environment}
        if self.read_capacity is not None:
            body["read_capacity"] = self.read_capacity
        if self.schema is not None:
            body["schema"] = self.schema
        return {"byoc": body}
