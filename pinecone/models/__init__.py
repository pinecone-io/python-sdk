"""Every model type SDK methods return, re-exported and lazily imported.

Names resolve on first access, so importing ``pinecone`` does not pay for
every model module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pinecone.models.admin.api_key import (  # noqa: F401
        APIKeyList,
        APIKeyModel,
        APIKeyRole,
        APIKeyWithSecret,
    )
    from pinecone.models.admin.invite import (  # noqa: F401
        InviteList,
        InviteModel,
        InviteStatus,
    )
    from pinecone.models.admin.organization import (  # noqa: F401
        OrganizationList,
        OrganizationModel,
    )
    from pinecone.models.admin.pagination import PaginationResponse  # noqa: F401
    from pinecone.models.admin.project import ProjectList, ProjectModel  # noqa: F401
    from pinecone.models.admin.role_binding import (  # noqa: F401
        PrincipalType,
        ResourceType,
        RoleBindingInput,
        RoleBindingList,
        RoleBindingModel,
        RoleName,
    )
    from pinecone.models.admin.service_account import (  # noqa: F401
        ServiceAccountList,
        ServiceAccountModel,
        ServiceAccountWithSecret,
    )
    from pinecone.models.admin.token import TokenResponse  # noqa: F401
    from pinecone.models.admin.user import UserList, UserModel  # noqa: F401
    from pinecone.models.assistant.chat import (  # noqa: F401
        ChatCitation,
        ChatCompletionChoice,
        ChatCompletionResponse,
        ChatHighlight,
        ChatMessage,
        ChatReference,
        ChatResponse,
        ChatUsage,
    )
    from pinecone.models.assistant.context import (  # noqa: F401
        ContextContentBlock,
        ContextImageBlock,
        ContextImageData,
        ContextReference,
        ContextResponse,
        ContextSnippet,
        ContextTextBlock,
        FileReference,
        MultimodalSnippet,
        PageReference,
        TextSnippet,
    )
    from pinecone.models.assistant.evaluation import (  # noqa: F401
        AlignmentResult,
        AlignmentScores,
        EntailmentResult,
    )
    from pinecone.models.assistant.file_model import AssistantFileModel  # noqa: F401
    from pinecone.models.assistant.list import (  # noqa: F401
        ListAssistantsResponse,
        ListFilesResponse,
        ListOperationsResponse,
    )
    from pinecone.models.assistant.message import Message  # noqa: F401
    from pinecone.models.assistant.model import AssistantModel  # noqa: F401
    from pinecone.models.assistant.operation import OperationModel  # noqa: F401
    from pinecone.models.assistant.options import ContextOptions  # noqa: F401
    from pinecone.models.assistant.streaming import (  # noqa: F401
        ChatCompletionStreamChoice,
        ChatCompletionStreamChunk,
        ChatCompletionStreamDelta,
        ChatStreamChunk,
        StreamCitationChunk,
        StreamContentChunk,
        StreamContentDelta,
        StreamMessageEnd,
        StreamMessageStart,
    )
    from pinecone.models.backups.list import (  # noqa: F401
        BackupList,
        BackupScheduleHistoryList,
        BackupScheduleList,
        RestoreJobList,
    )
    from pinecone.models.backups.model import (  # noqa: F401
        BackupModel,
        CreateIndexFromBackupRequest,
        CreateIndexFromBackupResponse,
        RestoreJobModel,
    )
    from pinecone.models.backups.schedules import (  # noqa: F401
        BackupScheduleHistoryItem,
        BackupScheduleModel,
        CreateBackupScheduleRequest,
        UpdateBackupScheduleRequest,
    )
    from pinecone.models.batch import BatchError, BatchResult  # noqa: F401
    from pinecone.models.collections.list import CollectionList  # noqa: F401
    from pinecone.models.collections.model import CollectionModel  # noqa: F401
    from pinecone.models.documents.document import (  # noqa: F401
        Document,
        DocumentRecord,
        UpdateDocumentRecord,
    )
    from pinecone.models.documents.requests import (  # noqa: F401
        DeleteDocumentsRequest,
        FetchDocumentsRequest,
        ListDocumentsRequest,
        SearchDocumentsRequest,
        UpdateDocumentsRequest,
        UpsertDocumentsRequest,
    )
    from pinecone.models.documents.responses import (  # noqa: F401
        DeleteDocumentsResponse,
        DocumentFetchUsage,
        DocumentListUsage,
        DocumentSearchUsage,
        FetchDocumentsResponse,
        ListDocumentsResponse,
        ListedDocumentRecord,
        SearchDocumentsResponse,
        UpdateDocumentsResponse,
        UpsertDocumentsResponse,
    )
    from pinecone.models.documents.score_by import (  # noqa: F401
        DenseVectorQuery,
        DocumentScoringMethod,
        QueryStringQuery,
        SparseVectorQuery,
        TextQuery,
    )
    from pinecone.models.enums import (  # noqa: F401
        CloudProvider,
        DeletionProtection,
        EmbedModel,
        Metric,
        PodType,
        RerankModel,
        VectorType,
    )
    from pinecone.models.imports.list import ImportList  # noqa: F401
    from pinecone.models.imports.model import ImportModel, StartImportResponse  # noqa: F401
    from pinecone.models.indexes.deployment import (  # noqa: F401
        ByocDeployment,
        IndexDeployment,
        ManagedDeployment,
        PodDeployment,
    )
    from pinecone.models.indexes.index import (  # noqa: F401
        ByocSpecInfo,
        IndexModel,
        IndexSpec,
        IndexStatus,
        IndexTags,
        ModelIndexEmbed,
        PodSpecInfo,
        ServerlessSpecInfo,
    )
    from pinecone.models.indexes.list import IndexList  # noqa: F401
    from pinecone.models.indexes.read_capacity import (  # noqa: F401
        ReadCapacityDedicatedConfig,
        ReadCapacityDedicatedResponse,
        ReadCapacityOnDemandResponse,
        ReadCapacityResponse,
        ReadCapacityStatus,
        ScalingConfigManual,
    )
    from pinecone.models.indexes.requests import (  # noqa: F401
        ConfigureIndexRequest,
        CreateIndexRequest,
    )
    from pinecone.models.indexes.schema import (  # noqa: F401
        BooleanField,
        DenseVectorField,
        FloatField,
        FullTextSearchConfig,
        IndexSchema,
        IndexSchemaField,
        IntegerField,
        LegacyMetadataField,
        NgramConfig,
        SemanticTextField,
        SparseVectorField,
        StringField,
        StringListField,
    )
    from pinecone.models.indexes.specs import (  # noqa: F401
        ByocSpec,
        EmbedConfig,
        IntegratedSpec,
        PodSpec,
        ServerlessSpec,
    )
    from pinecone.models.inference.embed import (  # noqa: F401
        DenseEmbedding,
        Embedding,
        EmbeddingsList,
        EmbedUsage,
        SparseEmbedding,
    )
    from pinecone.models.inference.model_list import ModelInfoList  # noqa: F401
    from pinecone.models.inference.models import (  # noqa: F401
        ModelInfo,
        ModelInfoSupportedParameter,
    )
    from pinecone.models.inference.rerank import (  # noqa: F401
        RankedDocument,
        RerankResult,
        RerankUsage,
    )
    from pinecone.models.namespaces.models import (  # noqa: F401
        IndexedFields,
        ListNamespacesResponse,
        NamespaceDescription,
        NamespaceFieldConfig,
        NamespaceSchema,
    )
    from pinecone.models.response_info import BatchResponseInfo  # noqa: F401
    from pinecone.models.vectors.query_aggregator import QueryNamespacesResults  # noqa: F401
    from pinecone.models.vectors.responses import (  # noqa: F401
        DescribeIndexStatsResponse,
        FetchByMetadataResponse,
        FetchResponse,
        ListItem,
        ListResponse,
        NamespaceSummary,
        Pagination,
        QueryResponse,
        ResponseInfo,
        UpdateResponse,
        UpsertRecordsResponse,
        UpsertResponse,
    )
    from pinecone.models.vectors.search import (  # noqa: F401
        Hit,
        SearchRecordsResponse,
        SearchResult,
        SearchUsage,
    )
    from pinecone.models.vectors.sparse import SparseValues  # noqa: F401
    from pinecone.models.vectors.usage import Usage  # noqa: F401
    from pinecone.models.vectors.vector import ScoredVector, Vector  # noqa: F401

_LAZY_IMPORTS: dict[str, str] = {
    # Batch
    "BatchError": "pinecone.models.batch",
    "BatchResponseInfo": "pinecone.models.response_info",
    "BatchResult": "pinecone.models.batch",
    # Admin
    "APIKeyList": "pinecone.models.admin.api_key",
    "APIKeyModel": "pinecone.models.admin.api_key",
    "APIKeyRole": "pinecone.models.admin.api_key",
    "APIKeyWithSecret": "pinecone.models.admin.api_key",
    "InviteList": "pinecone.models.admin.invite",
    "InviteModel": "pinecone.models.admin.invite",
    "InviteStatus": "pinecone.models.admin.invite",
    "OrganizationList": "pinecone.models.admin.organization",
    "OrganizationModel": "pinecone.models.admin.organization",
    "PaginationResponse": "pinecone.models.admin.pagination",
    "PrincipalType": "pinecone.models.admin.role_binding",
    "ProjectList": "pinecone.models.admin.project",
    "ProjectModel": "pinecone.models.admin.project",
    "ResourceType": "pinecone.models.admin.role_binding",
    "RoleBindingInput": "pinecone.models.admin.role_binding",
    "RoleBindingList": "pinecone.models.admin.role_binding",
    "RoleBindingModel": "pinecone.models.admin.role_binding",
    "RoleName": "pinecone.models.admin.role_binding",
    "ServiceAccountList": "pinecone.models.admin.service_account",
    "ServiceAccountModel": "pinecone.models.admin.service_account",
    "ServiceAccountWithSecret": "pinecone.models.admin.service_account",
    "TokenResponse": "pinecone.models.admin.token",
    "UserList": "pinecone.models.admin.user",
    "UserModel": "pinecone.models.admin.user",
    # Assistant — chat
    "ChatCitation": "pinecone.models.assistant.chat",
    "ChatCompletionChoice": "pinecone.models.assistant.chat",
    "ChatCompletionResponse": "pinecone.models.assistant.chat",
    "ChatHighlight": "pinecone.models.assistant.chat",
    "ChatMessage": "pinecone.models.assistant.chat",
    "ChatReference": "pinecone.models.assistant.chat",
    "ChatResponse": "pinecone.models.assistant.chat",
    "ChatUsage": "pinecone.models.assistant.chat",
    # Assistant — context
    "ContextContentBlock": "pinecone.models.assistant.context",
    "ContextImageBlock": "pinecone.models.assistant.context",
    "ContextImageData": "pinecone.models.assistant.context",
    "ContextReference": "pinecone.models.assistant.context",
    "ContextResponse": "pinecone.models.assistant.context",
    "ContextSnippet": "pinecone.models.assistant.context",
    "ContextTextBlock": "pinecone.models.assistant.context",
    "FileReference": "pinecone.models.assistant.context",
    "MultimodalSnippet": "pinecone.models.assistant.context",
    "PageReference": "pinecone.models.assistant.context",
    "TextSnippet": "pinecone.models.assistant.context",
    # Assistant — evaluation
    "AlignmentResult": "pinecone.models.assistant.evaluation",
    "AlignmentScores": "pinecone.models.assistant.evaluation",
    "EntailmentResult": "pinecone.models.assistant.evaluation",
    # Assistant — misc
    "AssistantFileModel": "pinecone.models.assistant.file_model",
    "ListAssistantsResponse": "pinecone.models.assistant.list",
    "ListFilesResponse": "pinecone.models.assistant.list",
    "ListOperationsResponse": "pinecone.models.assistant.list",
    "OperationModel": "pinecone.models.assistant.operation",
    "Message": "pinecone.models.assistant.message",
    "AssistantModel": "pinecone.models.assistant.model",
    "ContextOptions": "pinecone.models.assistant.options",
    # Assistant — streaming
    "ChatCompletionStreamChoice": "pinecone.models.assistant.streaming",
    "ChatCompletionStreamChunk": "pinecone.models.assistant.streaming",
    "ChatCompletionStreamDelta": "pinecone.models.assistant.streaming",
    "ChatStreamChunk": "pinecone.models.assistant.streaming",
    "StreamCitationChunk": "pinecone.models.assistant.streaming",
    "StreamContentChunk": "pinecone.models.assistant.streaming",
    "StreamContentDelta": "pinecone.models.assistant.streaming",
    "StreamMessageEnd": "pinecone.models.assistant.streaming",
    "StreamMessageStart": "pinecone.models.assistant.streaming",
    # Backups
    "BackupList": "pinecone.models.backups.list",
    "RestoreJobList": "pinecone.models.backups.list",
    "BackupModel": "pinecone.models.backups.model",
    "CreateIndexFromBackupRequest": "pinecone.models.backups.model",
    "CreateIndexFromBackupResponse": "pinecone.models.backups.model",
    "RestoreJobModel": "pinecone.models.backups.model",
    # Backup schedules
    "BackupScheduleHistoryList": "pinecone.models.backups.list",
    "BackupScheduleList": "pinecone.models.backups.list",
    "BackupScheduleHistoryItem": "pinecone.models.backups.schedules",
    "BackupScheduleModel": "pinecone.models.backups.schedules",
    "CreateBackupScheduleRequest": "pinecone.models.backups.schedules",
    "UpdateBackupScheduleRequest": "pinecone.models.backups.schedules",
    # Collections
    "CollectionList": "pinecone.models.collections.list",
    "CollectionModel": "pinecone.models.collections.model",
    # Enums
    "CloudProvider": "pinecone.models.enums",
    "DeletionProtection": "pinecone.models.enums",
    "EmbedModel": "pinecone.models.enums",
    "Metric": "pinecone.models.enums",
    "PodType": "pinecone.models.enums",
    "RerankModel": "pinecone.models.enums",
    "VectorType": "pinecone.models.enums",
    # Documents
    "Document": "pinecone.models.documents.document",
    "DocumentRecord": "pinecone.models.documents.document",
    "UpdateDocumentRecord": "pinecone.models.documents.document",
    "DenseVectorQuery": "pinecone.models.documents.score_by",
    "DocumentScoringMethod": "pinecone.models.documents.score_by",
    "QueryStringQuery": "pinecone.models.documents.score_by",
    "SparseVectorQuery": "pinecone.models.documents.score_by",
    "TextQuery": "pinecone.models.documents.score_by",
    "DeleteDocumentsRequest": "pinecone.models.documents.requests",
    "FetchDocumentsRequest": "pinecone.models.documents.requests",
    "ListDocumentsRequest": "pinecone.models.documents.requests",
    "SearchDocumentsRequest": "pinecone.models.documents.requests",
    "UpdateDocumentsRequest": "pinecone.models.documents.requests",
    "UpsertDocumentsRequest": "pinecone.models.documents.requests",
    "DeleteDocumentsResponse": "pinecone.models.documents.responses",
    "DocumentFetchUsage": "pinecone.models.documents.responses",
    "DocumentListUsage": "pinecone.models.documents.responses",
    "DocumentSearchUsage": "pinecone.models.documents.responses",
    "FetchDocumentsResponse": "pinecone.models.documents.responses",
    "ListDocumentsResponse": "pinecone.models.documents.responses",
    "ListedDocumentRecord": "pinecone.models.documents.responses",
    "SearchDocumentsResponse": "pinecone.models.documents.responses",
    "UpdateDocumentsResponse": "pinecone.models.documents.responses",
    "UpsertDocumentsResponse": "pinecone.models.documents.responses",
    # Imports
    "ImportList": "pinecone.models.imports.list",
    "ImportModel": "pinecone.models.imports.model",
    "StartImportResponse": "pinecone.models.imports.model",
    # Indexes
    "IndexModel": "pinecone.models.indexes.index",
    "IndexStatus": "pinecone.models.indexes.index",
    "IndexTags": "pinecone.models.indexes.index",
    "IndexSpec": "pinecone.models.indexes.index",
    "ServerlessSpecInfo": "pinecone.models.indexes.index",
    "PodSpecInfo": "pinecone.models.indexes.index",
    "ByocSpecInfo": "pinecone.models.indexes.index",
    "ModelIndexEmbed": "pinecone.models.indexes.index",
    "IndexList": "pinecone.models.indexes.list",
    "ByocDeployment": "pinecone.models.indexes.deployment",
    "IndexDeployment": "pinecone.models.indexes.deployment",
    "ManagedDeployment": "pinecone.models.indexes.deployment",
    "PodDeployment": "pinecone.models.indexes.deployment",
    "ReadCapacityDedicatedConfig": "pinecone.models.indexes.read_capacity",
    "ReadCapacityDedicatedResponse": "pinecone.models.indexes.read_capacity",
    "ReadCapacityOnDemandResponse": "pinecone.models.indexes.read_capacity",
    "ReadCapacityResponse": "pinecone.models.indexes.read_capacity",
    "ReadCapacityStatus": "pinecone.models.indexes.read_capacity",
    "ScalingConfigManual": "pinecone.models.indexes.read_capacity",
    "ConfigureIndexRequest": "pinecone.models.indexes.requests",
    "CreateIndexRequest": "pinecone.models.indexes.requests",
    "BooleanField": "pinecone.models.indexes.schema",
    "DenseVectorField": "pinecone.models.indexes.schema",
    "FloatField": "pinecone.models.indexes.schema",
    "FullTextSearchConfig": "pinecone.models.indexes.schema",
    "IndexSchema": "pinecone.models.indexes.schema",
    "IndexSchemaField": "pinecone.models.indexes.schema",
    "IntegerField": "pinecone.models.indexes.schema",
    "LegacyMetadataField": "pinecone.models.indexes.schema",
    "NgramConfig": "pinecone.models.indexes.schema",
    "SemanticTextField": "pinecone.models.indexes.schema",
    "SparseVectorField": "pinecone.models.indexes.schema",
    "StringField": "pinecone.models.indexes.schema",
    "StringListField": "pinecone.models.indexes.schema",
    "ByocSpec": "pinecone.models.indexes.specs",
    "EmbedConfig": "pinecone.models.indexes.specs",
    "IntegratedSpec": "pinecone.models.indexes.specs",
    "PodSpec": "pinecone.models.indexes.specs",
    "ServerlessSpec": "pinecone.models.indexes.specs",
    # Inference
    "DenseEmbedding": "pinecone.models.inference.embed",
    "Embedding": "pinecone.models.inference.embed",
    "EmbeddingsList": "pinecone.models.inference.embed",
    "EmbedUsage": "pinecone.models.inference.embed",
    "SparseEmbedding": "pinecone.models.inference.embed",
    "ModelInfoList": "pinecone.models.inference.model_list",
    "ModelInfo": "pinecone.models.inference.models",
    "ModelInfoSupportedParameter": "pinecone.models.inference.models",
    "RankedDocument": "pinecone.models.inference.rerank",
    "RerankResult": "pinecone.models.inference.rerank",
    "RerankUsage": "pinecone.models.inference.rerank",
    # Namespaces
    "IndexedFields": "pinecone.models.namespaces.models",
    "ListNamespacesResponse": "pinecone.models.namespaces.models",
    "NamespaceDescription": "pinecone.models.namespaces.models",
    "NamespaceFieldConfig": "pinecone.models.namespaces.models",
    "NamespaceSchema": "pinecone.models.namespaces.models",
    # Vectors
    "QueryNamespacesResults": "pinecone.models.vectors.query_aggregator",
    "DescribeIndexStatsResponse": "pinecone.models.vectors.responses",
    "FetchByMetadataResponse": "pinecone.models.vectors.responses",
    "FetchResponse": "pinecone.models.vectors.responses",
    "ListItem": "pinecone.models.vectors.responses",
    "ListResponse": "pinecone.models.vectors.responses",
    "NamespaceSummary": "pinecone.models.vectors.responses",
    "Pagination": "pinecone.models.vectors.responses",
    "QueryResponse": "pinecone.models.vectors.responses",
    "ResponseInfo": "pinecone.models.response_info",
    "UpdateResponse": "pinecone.models.vectors.responses",
    "UpsertRecordsResponse": "pinecone.models.vectors.responses",
    "UpsertResponse": "pinecone.models.vectors.responses",
    "Hit": "pinecone.models.vectors.search",
    "SearchRecordsResponse": "pinecone.models.vectors.search",
    "SearchResult": "pinecone.models.vectors.search",
    "SearchUsage": "pinecone.models.vectors.search",
    "SparseValues": "pinecone.models.vectors.sparse",
    "Usage": "pinecone.models.vectors.usage",
    "ScoredVector": "pinecone.models.vectors.vector",
    "Vector": "pinecone.models.vectors.vector",
}

__all__ = list(_LAZY_IMPORTS.keys())


def __getattr__(name: str) -> Any:
    """Lazy-load models on first access."""
    if name in _LAZY_IMPORTS:
        from importlib import import_module

        module = import_module(_LAZY_IMPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    import builtins

    return builtins.list({*globals(), *__all__, *_LAZY_IMPORTS})
