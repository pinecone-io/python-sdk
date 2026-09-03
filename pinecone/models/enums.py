"""Enumerations for the string-valued options SDK methods accept.

Every member is a ``str``, so an enum member and its plain-string value are
interchangeable at every call site. The enums exist for autocomplete and typo
protection, not to constrain what the API accepts.
"""

from __future__ import annotations

from enum import Enum


class CloudProvider(str, Enum):
    """Public cloud a managed index runs in.

    Goes in the ``cloud`` key of a managed ``deployment``, and in
    :meth:`~pinecone.client.indexes.Indexes.create_for_model`'s ``cloud``
    argument. Pair it with a region enum for the same provider —
    :class:`AwsRegion`, :class:`GcpRegion`, or :class:`AzureRegion`.
    """

    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"


class Metric(str, Enum):
    """How similarity between two dense vectors is scored.

    Set on the dense vector field in an index's ``schema`` and fixed for the
    life of that field. ``COSINE`` compares direction and ignores magnitude,
    which is what most text embedding models are trained for and the right
    default when in doubt. ``DOTPRODUCT`` takes magnitude into account, and is
    the metric sparse fields always use. ``EUCLIDEAN`` scores straight-line
    distance, so a smaller score is a closer match.
    """

    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOTPRODUCT = "dotproduct"


class DeletionProtection(str, Enum):
    """Whether an index refuses to be deleted.

    While ``ENABLED``, :meth:`~pinecone.client.indexes.Indexes.delete` on the
    index fails with :exc:`~pinecone.errors.exceptions.ForbiddenError`, and you
    have to ``configure`` it back to ``DISABLED`` first. New indexes are
    ``DISABLED``.
    """

    ENABLED = "enabled"
    DISABLED = "disabled"


class VectorType(str, Enum):
    """Dense or sparse, for the deprecated single-vector index shape.

    Reaches the API only through the deprecated ``vector_type=`` argument to
    :meth:`~pinecone.client.indexes.Indexes.create`. A current ``schema``
    names a ``dense_vector`` or ``sparse_vector`` field type instead, which is
    what lets one index hold both.
    """

    DENSE = "dense"
    SPARSE = "sparse"


class EmbedModel(str, Enum):
    """Known embedding models for integrated indexes.

    A convenience enum rather than an exhaustive list: ``model`` is also accepted
    as a plain string, so a model added after this SDK release can still be used.
    Call :meth:`~pinecone.client.inference.Inference.list_models` for the models
    currently available.
    """

    Multilingual_E5_Large = "multilingual-e5-large"
    Pinecone_Sparse_English_V0 = "pinecone-sparse-english-v0"
    Llama_Text_Embed_V2 = "llama-text-embed-v2"
    Pinecone_Sparse_Multilingual_V0 = "pinecone-sparse-multilingual-v0"


class RerankModel(str, Enum):
    """Known reranking models.

    Like :class:`EmbedModel`, a convenience enum rather than an exhaustive list.

    .. note::
       ``Pinecone_Rerank_V0`` is deprecated and most projects can no longer use
       it: a request naming it is rejected with a permission error whose message
       points to a current model. Prefer another member of this enum.
    """

    Bge_Reranker_V2_M3 = "bge-reranker-v2-m3"
    Cohere_Rerank_3_5 = "cohere-rerank-3.5"
    Pinecone_Rerank_V0 = "pinecone-rerank-v0"


class PodType(str, Enum):
    """Pod hardware family and size, for the ``pod_type`` of a pod deployment.

    The family before the dot picks what the pod is optimized for — ``s1`` for
    storage, ``p1`` for balanced performance, ``p2`` for query throughput —
    and the ``xN`` after it is the size multiplier. See
    :doc:`/how-to/indexes/pod` for what each family trades away. Pod-based
    indexes predate managed ones; reach for a managed deployment unless you
    have a reason not to.
    """

    P1_X1 = "p1.x1"
    P1_X2 = "p1.x2"
    P1_X4 = "p1.x4"
    P1_X8 = "p1.x8"
    S1_X1 = "s1.x1"
    S1_X2 = "s1.x2"
    S1_X4 = "s1.x4"
    S1_X8 = "s1.x8"
    P2_X1 = "p2.x1"
    P2_X2 = "p2.x2"
    P2_X4 = "p2.x4"
    P2_X8 = "p2.x8"


class AwsRegion(str, Enum):
    """AWS regions for the ``region`` of a managed index on ``cloud="aws"``.

    A convenience enum rather than an exhaustive list, like
    :class:`~pinecone.models.enums.EmbedModel`: ``region`` also accepts a
    plain string, so a region added after this SDK release still works.
    """

    US_EAST_1 = "us-east-1"
    US_WEST_2 = "us-west-2"
    EU_WEST_1 = "eu-west-1"
    EU_CENTRAL_1 = "eu-central-1"
    AP_SOUTHEAST_1 = "ap-southeast-1"


class GcpRegion(str, Enum):
    """GCP regions for the ``region`` of a managed index on ``cloud="gcp"``.

    A convenience enum rather than an exhaustive list, like
    :class:`~pinecone.models.enums.EmbedModel`: ``region`` also accepts a
    plain string, so a region added after this SDK release still works.
    """

    US_CENTRAL1 = "us-central1"
    EUROPE_WEST4 = "europe-west4"


class AzureRegion(str, Enum):
    """Azure regions for the ``region`` of a managed index on ``cloud="azure"``.

    A convenience enum rather than an exhaustive list, like
    :class:`~pinecone.models.enums.EmbedModel`: ``region`` also accepts a
    plain string, so a region added after this SDK release still works.
    """

    EASTUS2 = "eastus2"
    GERMANYWESTCENTRAL = "germanywestcentral"


class PodIndexEnvironment(str, Enum):
    """Environments for the ``environment`` of a pod deployment.

    A pod-based index names one environment instead of a cloud and region
    pair; the member names encode both. Also a convenience enum rather than
    an exhaustive list.
    """

    US_WEST1_GCP = "us-west1-gcp"
    US_CENTRAL1_GCP = "us-central1-gcp"
    US_WEST4_GCP = "us-west4-gcp"
    US_EAST4_GCP = "us-east4-gcp"
    NORTHAMERICA_NORTHEAST1_GCP = "northamerica-northeast1-gcp"
    ASIA_NORTHEAST1_GCP = "asia-northeast1-gcp"
    ASIA_SOUTHEAST1_GCP = "asia-southeast1-gcp"
    US_EAST1_GCP = "us-east1-gcp"
    EU_WEST1_GCP = "eu-west1-gcp"
    EU_WEST4_GCP = "eu-west4-gcp"
    US_EAST1_AWS = "us-east-1-aws"
    EASTUS_AZURE = "eastus-azure"
