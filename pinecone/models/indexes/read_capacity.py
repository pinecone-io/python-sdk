"""How much read capacity an index has, and how it is provisioned.

``IndexModel.read_capacity`` is either on-demand or dedicated, told apart by
the ``mode`` key — see :data:`ReadCapacityResponse`.
"""

from __future__ import annotations

from msgspec import Struct

__all__ = [
    "ReadCapacityDedicatedConfig",
    "ReadCapacityDedicatedResponse",
    "ReadCapacityOnDemandResponse",
    "ReadCapacityResponse",
    "ReadCapacityStatus",
    "ScalingConfigManual",
]


class ScalingConfigManual(Struct, kw_only=True):
    """The shard and replica counts you chose for dedicated read capacity.

    Present when ``scaling`` is ``"Manual"`` — you are sizing the read side
    yourself rather than letting Pinecone size it.

    Attributes:
        shards: How many shards to split reads across, which is what decides
            how much data the read tier holds.
        replicas: How many copies of each shard to run, which is what decides
            read throughput. ``0`` is legal and stops the index serving reads
            entirely — a way to pause the cost of an index you are not
            querying without deleting it.
    """

    shards: int
    replicas: int


class ReadCapacityStatus(Struct, kw_only=True):
    """Whether an index's read capacity is provisioned and serving.

    Separate from :class:`~pinecone.models.indexes.index.IndexStatus`: an
    index can be ready while its read tier is still scaling into place.

    Attributes:
        state: Where provisioning is — ``"Ready"`` most of the time,
            ``"Scaling"`` after a recent replica or shard change,
            ``"Migrating"`` while moving to a new node type, or
            ``"Error"``, in which case read ``error_message``.
        current_shards: Current number of active shards. ``None`` for an
            index with on-demand read capacity, which has no fixed shard
            count.
        current_replicas: Current number of active replicas. ``None`` for
            an index with on-demand read capacity, which has no fixed
            replica count.
        error_message: Message describing a read-capacity configuration
            issue; ``None`` unless ``state`` is ``"Error"``.
    """

    state: str
    current_shards: int | None = None
    current_replicas: int | None = None
    error_message: str | None = None


class ReadCapacityDedicatedConfig(Struct, kw_only=True):
    """What the dedicated read tier is made of.

    Attributes:
        node_type: Machine class the read tier runs on — ``"b1"``, or
            ``"t1"`` for more processing power and memory per node.
        scaling: How the shard and replica counts are decided, e.g.
            ``"Manual"``.
        manual: The counts themselves, as a :class:`ScalingConfigManual`.
            Present when ``scaling`` is ``"Manual"``.
    """

    node_type: str
    scaling: str
    manual: ScalingConfigManual | None = None


class ReadCapacityOnDemandResponse(Struct, tag="OnDemand", tag_field="mode", kw_only=True):
    """Read capacity that scales with traffic and bills per read.

    The default, and the one with nothing to size: reads bill per operation,
    so there is no configuration to read back — only a status. Its ``mode`` is
    ``"OnDemand"``. Reach for :class:`ReadCapacityDedicatedResponse` when you
    want to control the shards and replicas serving reads instead; see
    :doc:`/guides/concepts`.

    Attributes:
        status: A :class:`ReadCapacityStatus`.
    """

    status: ReadCapacityStatus


class ReadCapacityDedicatedResponse(Struct, tag="Dedicated", tag_field="mode", kw_only=True):
    """Read capacity served by nodes provisioned for this index alone.

    Its ``mode`` is ``"Dedicated"``, and unlike on-demand it reports the
    hardware behind it, because you chose it. Changing the counts puts
    ``status.state`` into ``"Scaling"`` until the new shape is in place.

    Attributes:
        dedicated: A :class:`ReadCapacityDedicatedConfig` — node type and
            shard/replica counts.
        status: A :class:`ReadCapacityStatus`.
    """

    dedicated: ReadCapacityDedicatedConfig
    status: ReadCapacityStatus


#: The two read-capacity variants, told apart by their ``mode``. Narrow an
#: ``IndexModel.read_capacity`` with ``isinstance`` before reading
#: ``dedicated``, which only one variant has.
ReadCapacityResponse = ReadCapacityOnDemandResponse | ReadCapacityDedicatedResponse
