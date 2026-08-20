"""Data models for preview areas.

Each preview area adds its own model module under this package.
Preview models are isolated from stable models — see
docs/conventions/preview-channel.md § Type isolation.
"""

from __future__ import annotations

from pinecone.preview.models.backups import PreviewBackupModel, PreviewCreateBackupRequest
from pinecone.preview.models.deployment import (
    PreviewByocDeployment,
    PreviewDeployment,
    PreviewManagedDeployment,
    PreviewPodDeployment,
)
from pinecone.preview.models.documents import (
    PreviewDocument,
    PreviewDocumentFetchResponse,
    PreviewDocumentSearchResponse,
    PreviewDocumentUpsertResponse,
    PreviewUsage,
)
from pinecone.preview.models.indexes import PreviewIndexModel
from pinecone.preview.models.read_capacity import (
    PreviewReadCapacity,
    PreviewReadCapacityDedicatedInner,
    PreviewReadCapacityDedicatedResponse,
    PreviewReadCapacityManualScaling,
    PreviewReadCapacityOnDemandResponse,
    PreviewReadCapacityStatus,
)
from pinecone.preview.models.requests import (
    PreviewConfigureIndexRequest,
    PreviewCreateIndexRequest,
)
from pinecone.preview.models.schema import (
    PreviewBooleanField,
    PreviewDenseVectorField,
    PreviewIntegerField,
    PreviewLegacyIntegerField,
    PreviewSchema,
    PreviewSchemaField,
    PreviewSemanticTextField,
    PreviewSparseVectorField,
    PreviewStringField,
)
from pinecone.preview.models.score_by import (
    PreviewDenseVectorQuery,
    PreviewQueryStringQuery,
    PreviewScoreByQuery,
    PreviewSparseVectorQuery,
    PreviewTextQuery,
)
from pinecone.preview.models.sparse import PreviewSparseValues
from pinecone.preview.models.status import PreviewIndexStatus

__all__ = [
    "PreviewBackupModel",
    "PreviewBooleanField",
    "PreviewByocDeployment",
    "PreviewConfigureIndexRequest",
    "PreviewCreateBackupRequest",
    "PreviewCreateIndexRequest",
    "PreviewDenseVectorField",
    "PreviewDenseVectorQuery",
    "PreviewDeployment",
    "PreviewDocument",
    "PreviewDocumentFetchResponse",
    "PreviewDocumentSearchResponse",
    "PreviewDocumentUpsertResponse",
    "PreviewIndexModel",
    "PreviewIndexStatus",
    "PreviewIntegerField",
    "PreviewLegacyIntegerField",
    "PreviewManagedDeployment",
    "PreviewPodDeployment",
    "PreviewQueryStringQuery",
    "PreviewReadCapacity",
    "PreviewReadCapacityDedicatedInner",
    "PreviewReadCapacityDedicatedResponse",
    "PreviewReadCapacityManualScaling",
    "PreviewReadCapacityOnDemandResponse",
    "PreviewReadCapacityStatus",
    "PreviewSchema",
    "PreviewSchemaField",
    "PreviewScoreByQuery",
    "PreviewSemanticTextField",
    "PreviewSparseValues",
    "PreviewSparseVectorField",
    "PreviewSparseVectorQuery",
    "PreviewStringField",
    "PreviewTextQuery",
    "PreviewUsage",
]
