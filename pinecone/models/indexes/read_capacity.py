"""Read-capacity response models (2026-07 API)."""

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
    """Manual scaling configuration for dedicated read capacity.

    Attributes:
        shards: Number of shards. Each shard provides 250 GB of storage.
        replicas: Number of replicas. Setting replicas to 0 disables the
            index but can be used to reduce costs while usage is paused.
    """

    shards: int
    replicas: int


class ReadCapacityStatus(Struct, kw_only=True):
    """Read capacity provisioning status.

    Attributes:
        state: Current provisioning state — ``"Ready"`` most of the time,
            ``"Scaling"`` after a recent replica/shard change,
            ``"Migrating"`` while moving to a new node type, or
            ``"Error"`` (see ``error_message``).
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
    """Dedicated read-capacity configuration details.

    Attributes:
        node_type: The type of machines to use — ``"b1"`` or ``"t1"``
            (``t1`` includes increased processing power and memory).
        scaling: Scaling strategy (e.g. ``"Manual"``).
        manual: Manual scaling configuration, present when
            ``scaling="Manual"``.
    """

    node_type: str
    scaling: str
    manual: ScalingConfigManual | None = None


class ReadCapacityOnDemandResponse(Struct, tag="OnDemand", tag_field="mode", kw_only=True):
    """On-demand read capacity in API responses.

    Attributes:
        status: Current provisioning status.

    Note:
        The ``mode`` field is automatically set to ``"OnDemand"`` by
        msgspec's tagged-union system.
    """

    status: ReadCapacityStatus


class ReadCapacityDedicatedResponse(Struct, tag="Dedicated", tag_field="mode", kw_only=True):
    """Dedicated read capacity in API responses.

    Attributes:
        dedicated: Dedicated capacity configuration details.
        status: Current provisioning status.

    Note:
        The ``mode`` field is automatically set to ``"Dedicated"`` by
        msgspec's tagged-union system.
    """

    dedicated: ReadCapacityDedicatedConfig
    status: ReadCapacityStatus


#: Union of read-capacity response variants, dispatched on the ``mode`` field.
ReadCapacityResponse = ReadCapacityOnDemandResponse | ReadCapacityDedicatedResponse
