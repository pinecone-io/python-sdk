"""Index deployment response models (2026-07 API)."""

from __future__ import annotations

from msgspec import Struct

__all__ = [
    "ByocDeployment",
    "IndexDeployment",
    "ManagedDeployment",
    "PodDeployment",
]


class ManagedDeployment(Struct, tag="managed", tag_field="deployment_type", kw_only=True):
    """Managed (serverless) deployment configuration.

    Serverless indexes scale automatically and are billed per usage.  This
    deployment type also covers full-text search indexes.

    Attributes:
        cloud: Cloud provider — ``"aws"``, ``"gcp"``, or ``"azure"``.
        region: Cloud region (e.g. ``"us-east-1"``).
        environment: The internal environment (cell) hosting the index,
            derived from ``cloud`` and ``region``. Response-only and
            informational — it cannot be set on create and is not stable
            API surface.

    Note:
        The ``deployment_type`` field is automatically set to ``"managed"``
        by msgspec's tagged union system.
    """

    cloud: str
    region: str
    environment: str | None = None


class PodDeployment(Struct, tag="pod", tag_field="deployment_type", kw_only=True):
    """Pod-based deployment configuration.

    All properties are required on create — omitting ``replicas`` or
    ``shards`` is rejected with a ``422``.  Responses always carry all of
    them as well.

    Attributes:
        environment: Environment where the index is hosted
            (e.g. ``"us-east1-gcp"``).
        pod_type: Pod type — one of ``s1``, ``p1``, or ``p2`` appended
            with ``.`` and one of ``x1``, ``x2``, ``x4``, or ``x8``
            (e.g. ``"p1.x1"``).
        replicas: Number of replicas. Replicas duplicate the index for
            higher availability and throughput.
        shards: Number of shards. Shards split data across multiple pods
            to fit more data into an index.

    Note:
        The ``deployment_type`` field is automatically set to ``"pod"``
        by msgspec's tagged union system.
    """

    environment: str
    pod_type: str
    replicas: int
    shards: int


class ByocDeployment(Struct, tag="byoc", tag_field="deployment_type", kw_only=True):
    """Bring-your-own-compute (BYOC) deployment configuration.

    BYOC indexes run in customer-managed infrastructure.

    Attributes:
        environment: BYOC environment identifier
            (e.g. ``"aws-us-east-1-b921"``).

    Note:
        The ``deployment_type`` field is automatically set to ``"byoc"``
        by msgspec's tagged union system.
    """

    environment: str


#: Union of all deployment variants, dispatched on the ``deployment_type`` field.
IndexDeployment = ManagedDeployment | PodDeployment | ByocDeployment
