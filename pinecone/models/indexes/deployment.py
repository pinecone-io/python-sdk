"""Where and how an index runs.

An index's ``deployment`` says which of three models it uses, and the
``deployment_type`` key is what tells the variants apart on the wire and in
:data:`IndexDeployment`.
"""

from __future__ import annotations

from msgspec import Struct

__all__ = [
    "ByocDeployment",
    "IndexDeployment",
    "ManagedDeployment",
    "PodDeployment",
]


class ManagedDeployment(Struct, tag="managed", tag_field="deployment_type", kw_only=True):
    """A serverless index: Pinecone picks the capacity, you pick the region.

    The default deployment, and what to reach for unless you have a reason
    not to — no replicas or shards to size, and full-text search indexes run
    here too. Its ``deployment_type`` is ``"managed"``.

    Attributes:
        cloud: Public cloud to run in — ``"aws"``, ``"gcp"``, or ``"azure"``.
            See :class:`~pinecone.models.enums.CloudProvider`.
        region: Region within that cloud, e.g. ``"us-east-1"``.
        environment: The internal cell hosting the index, derived from
            ``cloud`` and ``region``. Response-only and informational; you
            cannot set it, and it is not something to build on.

    Examples:
        The ``deployment=`` argument that asks for one:

        .. code-block:: python

            {"deployment_type": "managed", "cloud": "aws", "region": "us-east-1"}
    """

    cloud: str
    region: str
    environment: str | None = None


class PodDeployment(Struct, tag="pod", tag_field="deployment_type", kw_only=True):
    """A pod-based index: you size the hardware yourself.

    The older deployment model, where capacity is something you choose and pay
    for rather than something that scales. Its ``deployment_type`` is
    ``"pod"``. Every attribute below is required on create — leaving out
    ``replicas`` or ``shards`` is rejected — and every one comes back on a
    describe.

    Attributes:
        environment: The environment hosting the index, which stands in for a
            cloud and region pair, e.g. ``"us-east1-gcp"``. See
            :class:`~pinecone.models.enums.PodIndexEnvironment`.
        pod_type: Hardware family and size, e.g. ``"p1.x1"``. See
            :class:`~pinecone.models.enums.PodType`.
        replicas: How many copies of the index to run. More replicas mean more
            query throughput and more availability, at proportional cost.
            One of the two things :meth:`configure
            <pinecone.client.indexes.Indexes.configure>` can change later,
            along with ``pod_type``.
        shards: How many pods to split the data across, which is what decides
            how much data fits. Fixed once the index exists.
    """

    environment: str
    pod_type: str
    replicas: int
    shards: int


class ByocDeployment(Struct, tag="byoc", tag_field="deployment_type", kw_only=True):
    """A BYOC index: Pinecone's data plane, running in your own account.

    Bring-your-own-compute indexes run in infrastructure you operate, so the
    only thing to name is the environment Pinecone provisioned there. Its
    ``deployment_type`` is ``"byoc"``.

    Attributes:
        environment: The BYOC environment to run in, e.g.
            ``"aws-us-east-1-b921"``. Pinecone gives you this identifier when
            the environment is set up.
    """

    environment: str


#: The three deployment variants, told apart by their ``deployment_type``.
#: Narrow an ``IndexModel.deployment`` with ``isinstance`` before reading
#: fields only one variant has.
IndexDeployment = ManagedDeployment | PodDeployment | ByocDeployment
