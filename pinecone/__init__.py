"""Pinecone Python SDK — vector database for similarity search.

:class:`Pinecone` is the control plane: it creates, inspects, configures, and
deletes indexes. :class:`Index` is the data plane: it reads and writes the
records inside one index, against that index's own host. ``pc.index(name)`` is
the bridge between the two.

Start by declaring the fields the index searches. Pass ``api_key``, or omit it
and set ``PINECONE_API_KEY``::

    from pinecone import Pinecone

    pc = Pinecone(api_key="your-api-key")

    pc.indexes.create(
        name="movie-recommendations",
        schema={"fields": {"embedding": {
            "type": "dense_vector", "dimension": 1536, "metric": "cosine"}}},
        deployment={"deployment_type": "managed", "cloud": "aws", "region": "us-east-1"},
    )

Then write and read documents through the handle. The three floats below stand
in for a full 1536-dimensional embedding::

    from pinecone import DenseVectorQuery

    index = pc.index("movie-recommendations")
    index.documents.upsert(
        namespace="movies-en",
        documents=[{"_id": "movie-42", "embedding": [0.012, -0.087, 0.153]}],
    )
    results = index.documents.search(
        namespace="movies-en",
        top_k=5,
        score_by=[DenseVectorQuery(field="embedding", values=[0.012, -0.087, 0.153])],
    )

How the index was created decides which data-plane interface it answers on. An
index declaring a ``schema`` of your own field names stores documents, read and
written through ``index.documents``. The vector methods :meth:`Index.upsert` and
:meth:`Index.query` serve indexes created with top-level ``dimension`` and
``metric``, and the server rejects them on a schema-based index. An index built
by ``pc.indexes.create_for_model()`` embeds text server-side, so you search it
with text rather than with a vector::

    index = pc.index("semantic-search")
    results = index.search(
        namespace="articles-en",
        top_k=5,
        inputs={"text": "how does vector search work?"},
    )

:class:`AsyncPinecone` is the same control plane for ``asyncio``. Its
``index()`` is a coroutine, and the handle it returns is a context manager::

    from pinecone import AsyncPinecone, DenseVectorQuery

    async with AsyncPinecone(api_key="your-api-key") as pc:
        index_names = [idx.name async for idx in pc.indexes.list()]

        index = await pc.index("movie-recommendations")
        async with index:
            results = await index.documents.search(
                namespace="movies-en",
                top_k=5,
                score_by=[DenseVectorQuery(field="embedding",
                                           values=[0.012, -0.087, 0.153])],
            )

Organizations, projects, and API keys are managed through :class:`Admin`, which
authenticates with OAuth2 client credentials rather than an API key. Every
error the SDK raises derives from :class:`PineconeError`; see
:doc:`/guides/error-handling` for which call produces which.

Upgrading from 9.x? ``create`` and ``configure`` moved from
``spec=``/``dimension=`` to ``schema=``/``deployment=``; see
:doc:`/migration/v10-migration` for the field-by-field mapping and
before/after code for each flow.
"""

from __future__ import annotations

import os as _os

# Avoid importing typing at runtime — its transitive deps (re, enum,
# collections, contextlib, functools, warnings) add ~28ms to cold import.
# All annotations use PEP 563 (from __future__ import annotations), so
# typing.Any is a string at runtime and never evaluated.
# mypy recognises a module-level `TYPE_CHECKING = False` as a type-checking
# guard, so the if-block below is analysed by type checkers but skipped at
# runtime.
TYPE_CHECKING = False

if TYPE_CHECKING:
    from typing import Any

    from pinecone._client import Pinecone
    from pinecone._internal.config import PineconeConfig, RetryConfig
    from pinecone.admin import Admin
    from pinecone.async_client.async_index import AsyncIndex
    from pinecone.async_client.pinecone import AsyncPinecone
    from pinecone.async_client.pinecone import AsyncPinecone as PineconeAsyncio
    from pinecone.db_control.enums.clouds import AwsRegion, AzureRegion, GcpRegion
    from pinecone.db_control.models.collection_description import CollectionDescription
    from pinecone.db_data.dataclasses.search_query import SearchQuery
    from pinecone.db_data.dataclasses.search_rerank import SearchRerank
    from pinecone.errors.exceptions import (
        ApiError,
        ConflictError,
        FailedPreconditionError,
        ForbiddenError,
        ForbiddenException,
        IndexInitFailedError,
        IndexTerminatedError,
        ListConversionException,
        NotFoundError,
        NotFoundException,
        PaymentRequiredError,
        PineconeApiAttributeError,
        PineconeApiException,
        PineconeApiKeyError,
        PineconeApiTypeError,
        PineconeApiValueError,
        PineconeConfigurationError,
        PineconeConnectionError,
        PineconeError,
        PineconeException,
        PineconeProtocolError,
        PineconeTimeoutError,
        PineconeTypeError,
        PineconeValueError,
        RateLimitError,
        RateLimitException,
        ResponseParsingError,
        ServiceError,
        ServiceException,
        UnauthorizedError,
        UnauthorizedException,
    )
    from pinecone.grpc import GrpcIndex
    from pinecone.grpc.future import PineconeFuture
    from pinecone.index import Index
    from pinecone.inference.models.index_embed import IndexEmbed
    from pinecone.models.admin.api_key import APIKeyList, APIKeyModel, APIKeyRole, APIKeyWithSecret
    from pinecone.models.admin.invite import InviteList, InviteModel, InviteStatus
    from pinecone.models.admin.organization import OrganizationList, OrganizationModel
    from pinecone.models.admin.pagination import PaginationResponse
    from pinecone.models.admin.project import ProjectList, ProjectModel
    from pinecone.models.admin.role_binding import (
        PrincipalType,
        ResourceType,
        RoleBindingInput,
        RoleBindingList,
        RoleBindingModel,
        RoleName,
    )
    from pinecone.models.admin.service_account import (
        ServiceAccountList,
        ServiceAccountModel,
        ServiceAccountWithSecret,
    )
    from pinecone.models.admin.token import TokenResponse
    from pinecone.models.admin.user import UserList, UserModel
    from pinecone.models.assistant.chat import (
        ChatCompletionMessage,
        ChatCompletionResponse,
        ChatResponse,
    )
    from pinecone.models.assistant.context import ContextResponse
    from pinecone.models.assistant.evaluation import AlignmentResult
    from pinecone.models.assistant.file_model import AssistantFileModel
    from pinecone.models.assistant.list import (
        ListAssistantsResponse,
        ListFilesResponse,
        ListOperationsResponse,
    )
    from pinecone.models.assistant.message import Message
    from pinecone.models.assistant.model import AssistantModel
    from pinecone.models.assistant.operation import OperationModel
    from pinecone.models.assistant.options import ContextOptions
    from pinecone.models.assistant.streaming import (
        AsyncChatCompletionStream,
        AsyncChatStream,
        ChatCompletionStream,
        ChatCompletionStreamChunk,
        ChatStream,
        ChatStreamChunk,
        StreamCitationChunk,
        StreamContentChunk,
        StreamMessageEnd,
        StreamMessageStart,
    )
    from pinecone.models.backups.list import (
        BackupList,
        BackupScheduleHistoryList,
        BackupScheduleList,
        RestoreJobList,
    )
    from pinecone.models.backups.model import (
        BackupModel,
        CreateIndexFromBackupRequest,
        CreateIndexFromBackupResponse,
        RestoreJobModel,
    )
    from pinecone.models.backups.schedules import (
        BackupScheduleHistoryItem,
        BackupScheduleModel,
        CreateBackupScheduleRequest,
        UpdateBackupScheduleRequest,
    )
    from pinecone.models.collections.list import CollectionList
    from pinecone.models.collections.model import CollectionModel
    from pinecone.models.documents.document import (
        Document,
        DocumentRecord,
        UpdateDocumentRecord,
    )
    from pinecone.models.documents.requests import (
        DeleteDocumentsRequest,
        FetchDocumentsRequest,
        ListDocumentsRequest,
        SearchDocumentsRequest,
        UpdateDocumentsRequest,
        UpsertDocumentsRequest,
    )
    from pinecone.models.documents.responses import (
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
    from pinecone.models.documents.score_by import (
        DenseVectorQuery,
        DocumentScoringMethod,
        QueryStringQuery,
        SparseVectorQuery,
        TextQuery,
    )
    from pinecone.models.enums import (
        CloudProvider,
        DeletionProtection,
        EmbedModel,
        Metric,
        PodIndexEnvironment,
        PodType,
        RerankModel,
        VectorType,
    )
    from pinecone.models.imports.error_mode import ImportErrorMode
    from pinecone.models.imports.list import ImportList
    from pinecone.models.imports.model import ImportModel, StartImportResponse
    from pinecone.models.indexes.deployment import (
        ByocDeployment,
        IndexDeployment,
        ManagedDeployment,
        PodDeployment,
    )
    from pinecone.models.indexes.index import (
        ByocSpecInfo,
        IndexModel,
        IndexSpec,
        IndexStatus,
        IndexTags,
        ModelIndexEmbed,
        PodSpecInfo,
        ServerlessSpecInfo,
    )
    from pinecone.models.indexes.list import IndexList
    from pinecone.models.indexes.read_capacity import (
        ReadCapacityDedicatedConfig,
        ReadCapacityDedicatedResponse,
        ReadCapacityOnDemandResponse,
        ReadCapacityResponse,
        ReadCapacityStatus,
        ScalingConfigManual,
    )
    from pinecone.models.indexes.requests import (
        ConfigureIndexRequest,
        CreateIndexRequest,
    )
    from pinecone.models.indexes.schema import (
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
    from pinecone.models.indexes.specs import (
        ByocSpec,
        EmbedConfig,
        IntegratedSpec,
        PodSpec,
        ServerlessSpec,
    )
    from pinecone.models.inference.embed import DenseEmbedding, EmbeddingsList, SparseEmbedding
    from pinecone.models.inference.model_list import ModelInfoList
    from pinecone.models.inference.models import ModelInfo
    from pinecone.models.inference.rerank import RankedDocument, RerankResult
    from pinecone.models.namespaces.models import (
        IndexedFields,
        ListNamespacesResponse,
        NamespaceDescription,
        NamespaceFieldConfig,
        NamespaceSchema,
    )
    from pinecone.models.pagination import AsyncPaginator, Page, Paginator
    from pinecone.models.response_info import BatchResponseInfo
    from pinecone.models.vectors.query_aggregator import (
        QueryNamespacesResults,
        QueryResultsAggregator,
    )
    from pinecone.models.vectors.responses import (
        DescribeIndexStatsResponse,
        FetchByMetadataResponse,
        FetchResponse,
        ListResponse,
        QueryResponse,
        ResponseInfo,
        UpdateResponse,
        UpsertRecordsResponse,
        UpsertResponse,
    )
    from pinecone.models.vectors.search import (
        Hit,
        RerankConfig,
        SearchInputs,
        SearchRecordsResponse,
        SearchResult,
        SearchUsage,
    )
    from pinecone.models.vectors.sparse import SparseValues
    from pinecone.models.vectors.vector import ScoredVector, Vector
    from pinecone.schema_builder import SchemaBuilder
    from pinecone.utils.filter_builder import Field, FilterBuilder

__version__ = "10.0.0"

if _os.environ.get("PINECONE_DEBUG"):
    import logging as _logging

    _logging.getLogger("pinecone").setLevel(_logging.DEBUG)

__all__ = [
    "APIKeyList",
    "APIKeyModel",
    "APIKeyRole",
    "APIKeyWithSecret",
    "Admin",
    "AlignmentResult",
    "ApiError",
    "AssistantFileModel",
    "AssistantModel",
    "AsyncChatCompletionStream",
    "AsyncChatStream",
    "AsyncIndex",
    "AsyncPaginator",
    "AsyncPinecone",
    "AwsRegion",
    "AzureRegion",
    "BackupList",
    "BackupModel",
    "BackupScheduleHistoryItem",
    "BackupScheduleHistoryList",
    "BackupScheduleList",
    "BackupScheduleModel",
    "BatchResponseInfo",
    "BooleanField",
    "ByocDeployment",
    "ByocSpec",
    "ByocSpecInfo",
    "ChatCompletionMessage",
    "ChatCompletionResponse",
    "ChatCompletionStream",
    "ChatCompletionStreamChunk",
    "ChatResponse",
    "ChatStream",
    "ChatStreamChunk",
    "CloudProvider",
    "CollectionDescription",
    "CollectionList",
    "CollectionModel",
    "ConfigureIndexRequest",
    "ConflictError",
    "ContextOptions",
    "ContextResponse",
    "CreateBackupScheduleRequest",
    "CreateIndexFromBackupRequest",
    "CreateIndexFromBackupResponse",
    "CreateIndexRequest",
    "DeleteDocumentsRequest",
    "DeleteDocumentsResponse",
    "DeletionProtection",
    "DenseEmbedding",
    "DenseVectorField",
    "DenseVectorQuery",
    "DescribeIndexStatsResponse",
    "Document",
    "DocumentFetchUsage",
    "DocumentListUsage",
    "DocumentRecord",
    "DocumentScoringMethod",
    "DocumentSearchUsage",
    "EmbedConfig",
    "EmbedModel",
    "EmbeddingsList",
    "FailedPreconditionError",
    "FetchByMetadataResponse",
    "FetchDocumentsRequest",
    "FetchDocumentsResponse",
    "FetchResponse",
    "Field",
    "FilterBuilder",
    "FloatField",
    "ForbiddenError",
    "ForbiddenException",
    "FullTextSearchConfig",
    "GcpRegion",
    "GrpcIndex",
    "Hit",
    "ImportErrorMode",
    "ImportList",
    "ImportModel",
    "Index",
    "IndexDeployment",
    "IndexEmbed",
    "IndexInitFailedError",
    "IndexList",
    "IndexModel",
    "IndexSchema",
    "IndexSchemaField",
    "IndexSpec",
    "IndexStatus",
    "IndexTags",
    "IndexTerminatedError",
    "IndexedFields",
    "IntegerField",
    "IntegratedSpec",
    "InviteList",
    "InviteModel",
    "InviteStatus",
    "LegacyMetadataField",
    "ListAssistantsResponse",
    "ListConversionException",
    "ListDocumentsRequest",
    "ListDocumentsResponse",
    "ListFilesResponse",
    "ListNamespacesResponse",
    "ListOperationsResponse",
    "ListResponse",
    "ListedDocumentRecord",
    "ManagedDeployment",
    "Message",
    "Metric",
    "ModelIndexEmbed",
    "ModelInfo",
    "ModelInfoList",
    "NamespaceDescription",
    "NamespaceFieldConfig",
    "NamespaceSchema",
    "NgramConfig",
    "NotFoundError",
    "NotFoundException",
    "OperationModel",
    "OrganizationList",
    "OrganizationModel",
    "Page",
    "PaginationResponse",
    "Paginator",
    "PaymentRequiredError",
    "Pinecone",
    "PineconeApiAttributeError",
    "PineconeApiException",
    "PineconeApiKeyError",
    "PineconeApiTypeError",
    "PineconeApiValueError",
    "PineconeAsyncio",  # legacy alias for AsyncPinecone
    "PineconeConfig",
    "PineconeConfigurationError",
    "PineconeConnectionError",
    "PineconeError",
    "PineconeException",
    "PineconeFuture",
    "PineconeProtocolError",
    "PineconeTimeoutError",
    "PineconeTypeError",
    "PineconeValueError",
    "PodDeployment",
    "PodIndexEnvironment",
    "PodSpec",
    "PodSpecInfo",
    "PodType",
    "PrincipalType",
    "ProjectList",
    "ProjectModel",
    "QueryNamespacesResults",
    "QueryResponse",
    "QueryResultsAggregator",
    "QueryStringQuery",
    "RankedDocument",
    "RateLimitError",
    "RateLimitException",
    "ReadCapacityDedicatedConfig",
    "ReadCapacityDedicatedResponse",
    "ReadCapacityOnDemandResponse",
    "ReadCapacityResponse",
    "ReadCapacityStatus",
    "RerankConfig",
    "RerankModel",
    "RerankResult",
    "ResourceType",
    "ResponseInfo",
    "ResponseParsingError",
    "RestoreJobList",
    "RestoreJobModel",
    "RetryConfig",
    "RoleBindingInput",
    "RoleBindingList",
    "RoleBindingModel",
    "RoleName",
    "ScalingConfigManual",
    "SchemaBuilder",
    "ScoredVector",
    "SearchDocumentsRequest",
    "SearchDocumentsResponse",
    "SearchInputs",
    "SearchQuery",
    "SearchRecordsResponse",
    "SearchRerank",
    "SearchResult",
    "SearchUsage",
    "SemanticTextField",
    "ServerlessSpec",
    "ServerlessSpecInfo",
    "ServiceAccountList",
    "ServiceAccountModel",
    "ServiceAccountWithSecret",
    "ServiceError",
    "ServiceException",
    "SparseEmbedding",
    "SparseValues",
    "SparseVectorField",
    "SparseVectorQuery",
    "StartImportResponse",
    "StreamCitationChunk",
    "StreamContentChunk",
    "StreamMessageEnd",
    "StreamMessageStart",
    "StringField",
    "StringListField",
    "TextQuery",
    "TokenResponse",
    "UnauthorizedError",
    "UnauthorizedException",
    "UpdateBackupScheduleRequest",
    "UpdateDocumentRecord",
    "UpdateDocumentsRequest",
    "UpdateDocumentsResponse",
    "UpdateResponse",
    "UpsertDocumentsRequest",
    "UpsertDocumentsResponse",
    "UpsertRecordsResponse",
    "UpsertResponse",
    "UserList",
    "UserModel",
    "Vector",
    "VectorType",
    "__version__",
]

# Lazy-load heavy classes to keep cold import under 10ms.
# Importing Pinecone/AsyncPinecone/Index eagerly pulls in httpx (~120ms).
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "APIKeyList": ("pinecone.models.admin.api_key", "APIKeyList"),
    "APIKeyModel": ("pinecone.models.admin.api_key", "APIKeyModel"),
    "APIKeyRole": ("pinecone.models.admin.api_key", "APIKeyRole"),
    "APIKeyWithSecret": ("pinecone.models.admin.api_key", "APIKeyWithSecret"),
    "Admin": ("pinecone.admin", "Admin"),
    "AlignmentResult": ("pinecone.models.assistant.evaluation", "AlignmentResult"),
    "ApiError": ("pinecone.errors.exceptions", "ApiError"),
    "AssistantFileModel": ("pinecone.models.assistant.file_model", "AssistantFileModel"),
    "AssistantModel": ("pinecone.models.assistant.model", "AssistantModel"),
    "AsyncChatCompletionStream": (
        "pinecone.models.assistant.streaming",
        "AsyncChatCompletionStream",
    ),
    "AsyncChatStream": ("pinecone.models.assistant.streaming", "AsyncChatStream"),
    "AsyncIndex": ("pinecone.async_client.async_index", "AsyncIndex"),
    "AsyncPaginator": ("pinecone.models.pagination", "AsyncPaginator"),
    "AsyncPinecone": ("pinecone.async_client.pinecone", "AsyncPinecone"),
    "AwsRegion": ("pinecone.db_control.enums.clouds", "AwsRegion"),
    "AzureRegion": ("pinecone.db_control.enums.clouds", "AzureRegion"),
    "BackupList": ("pinecone.models.backups.list", "BackupList"),
    "BackupModel": ("pinecone.models.backups.model", "BackupModel"),
    "BackupScheduleHistoryItem": (
        "pinecone.models.backups.schedules",
        "BackupScheduleHistoryItem",
    ),
    "BackupScheduleHistoryList": (
        "pinecone.models.backups.list",
        "BackupScheduleHistoryList",
    ),
    "BackupScheduleList": ("pinecone.models.backups.list", "BackupScheduleList"),
    "BackupScheduleModel": ("pinecone.models.backups.schedules", "BackupScheduleModel"),
    "BatchResponseInfo": ("pinecone.models.response_info", "BatchResponseInfo"),
    "BooleanField": ("pinecone.models.indexes.schema", "BooleanField"),
    "ByocDeployment": ("pinecone.models.indexes.deployment", "ByocDeployment"),
    "ByocSpec": ("pinecone.models.indexes.specs", "ByocSpec"),
    "ByocSpecInfo": ("pinecone.models.indexes.index", "ByocSpecInfo"),
    "ChatCompletionMessage": ("pinecone.models.assistant.chat", "ChatCompletionMessage"),
    "ChatCompletionResponse": ("pinecone.models.assistant.chat", "ChatCompletionResponse"),
    "ChatCompletionStream": ("pinecone.models.assistant.streaming", "ChatCompletionStream"),
    "ChatCompletionStreamChunk": (
        "pinecone.models.assistant.streaming",
        "ChatCompletionStreamChunk",
    ),
    "ChatResponse": ("pinecone.models.assistant.chat", "ChatResponse"),
    "ChatStream": ("pinecone.models.assistant.streaming", "ChatStream"),
    "ChatStreamChunk": ("pinecone.models.assistant.streaming", "ChatStreamChunk"),
    "CloudProvider": ("pinecone.models.enums", "CloudProvider"),
    "CollectionDescription": (
        "pinecone.db_control.models.collection_description",
        "CollectionDescription",
    ),
    "CollectionList": ("pinecone.models.collections.list", "CollectionList"),
    "CollectionModel": ("pinecone.models.collections.model", "CollectionModel"),
    "ConfigureIndexRequest": ("pinecone.models.indexes.requests", "ConfigureIndexRequest"),
    "ConflictError": ("pinecone.errors.exceptions", "ConflictError"),
    "ContextOptions": ("pinecone.models.assistant.options", "ContextOptions"),
    "ContextResponse": ("pinecone.models.assistant.context", "ContextResponse"),
    "CreateBackupScheduleRequest": (
        "pinecone.models.backups.schedules",
        "CreateBackupScheduleRequest",
    ),
    "CreateIndexFromBackupRequest": (
        "pinecone.models.backups.model",
        "CreateIndexFromBackupRequest",
    ),
    "CreateIndexFromBackupResponse": (
        "pinecone.models.backups.model",
        "CreateIndexFromBackupResponse",
    ),
    "CreateIndexRequest": ("pinecone.models.indexes.requests", "CreateIndexRequest"),
    "DeleteDocumentsRequest": ("pinecone.models.documents.requests", "DeleteDocumentsRequest"),
    "DeleteDocumentsResponse": ("pinecone.models.documents.responses", "DeleteDocumentsResponse"),
    "DeletionProtection": ("pinecone.models.enums", "DeletionProtection"),
    "DenseEmbedding": ("pinecone.models.inference.embed", "DenseEmbedding"),
    "DenseVectorField": ("pinecone.models.indexes.schema", "DenseVectorField"),
    "DenseVectorQuery": ("pinecone.models.documents.score_by", "DenseVectorQuery"),
    "DescribeIndexStatsResponse": (
        "pinecone.models.vectors.responses",
        "DescribeIndexStatsResponse",
    ),
    "Document": ("pinecone.models.documents.document", "Document"),
    "DocumentFetchUsage": ("pinecone.models.documents.responses", "DocumentFetchUsage"),
    "DocumentListUsage": ("pinecone.models.documents.responses", "DocumentListUsage"),
    "DocumentRecord": ("pinecone.models.documents.document", "DocumentRecord"),
    "DocumentScoringMethod": ("pinecone.models.documents.score_by", "DocumentScoringMethod"),
    "DocumentSearchUsage": ("pinecone.models.documents.responses", "DocumentSearchUsage"),
    "EmbedConfig": ("pinecone.models.indexes.specs", "EmbedConfig"),
    "EmbedModel": ("pinecone.models.enums", "EmbedModel"),
    "EmbeddingsList": ("pinecone.models.inference.embed", "EmbeddingsList"),
    "FailedPreconditionError": ("pinecone.errors.exceptions", "FailedPreconditionError"),
    "FetchByMetadataResponse": ("pinecone.models.vectors.responses", "FetchByMetadataResponse"),
    "FetchDocumentsRequest": ("pinecone.models.documents.requests", "FetchDocumentsRequest"),
    "FetchDocumentsResponse": ("pinecone.models.documents.responses", "FetchDocumentsResponse"),
    "FetchResponse": ("pinecone.models.vectors.responses", "FetchResponse"),
    "Field": ("pinecone.utils.filter_builder", "Field"),
    "FilterBuilder": ("pinecone.utils.filter_builder", "FilterBuilder"),
    "FloatField": ("pinecone.models.indexes.schema", "FloatField"),
    "ForbiddenError": ("pinecone.errors.exceptions", "ForbiddenError"),
    "ForbiddenException": ("pinecone.errors.exceptions", "ForbiddenException"),
    "FullTextSearchConfig": ("pinecone.models.indexes.schema", "FullTextSearchConfig"),
    "GcpRegion": ("pinecone.db_control.enums.clouds", "GcpRegion"),
    "GrpcIndex": ("pinecone.grpc", "GrpcIndex"),
    "Hit": ("pinecone.models.vectors.search", "Hit"),
    "ImportErrorMode": ("pinecone.models.imports.error_mode", "ImportErrorMode"),
    "ImportList": ("pinecone.models.imports.list", "ImportList"),
    "ImportModel": ("pinecone.models.imports.model", "ImportModel"),
    "Index": ("pinecone.index", "Index"),
    "IndexDeployment": ("pinecone.models.indexes.deployment", "IndexDeployment"),
    "IndexEmbed": ("pinecone.inference.models.index_embed", "IndexEmbed"),
    "IndexInitFailedError": ("pinecone.errors.exceptions", "IndexInitFailedError"),
    "IndexList": ("pinecone.models.indexes.list", "IndexList"),
    "IndexModel": ("pinecone.models.indexes.index", "IndexModel"),
    "IndexSchema": ("pinecone.models.indexes.schema", "IndexSchema"),
    "IndexSchemaField": ("pinecone.models.indexes.schema", "IndexSchemaField"),
    "IndexSpec": ("pinecone.models.indexes.index", "IndexSpec"),
    "IndexStatus": ("pinecone.models.indexes.index", "IndexStatus"),
    "IndexTags": ("pinecone.models.indexes.index", "IndexTags"),
    "IndexTerminatedError": ("pinecone.errors.exceptions", "IndexTerminatedError"),
    "IndexedFields": ("pinecone.models.namespaces.models", "IndexedFields"),
    "IntegerField": ("pinecone.models.indexes.schema", "IntegerField"),
    "IntegratedSpec": ("pinecone.models.indexes.specs", "IntegratedSpec"),
    "InviteList": ("pinecone.models.admin.invite", "InviteList"),
    "InviteModel": ("pinecone.models.admin.invite", "InviteModel"),
    "InviteStatus": ("pinecone.models.admin.invite", "InviteStatus"),
    "LegacyMetadataField": ("pinecone.models.indexes.schema", "LegacyMetadataField"),
    "ListAssistantsResponse": ("pinecone.models.assistant.list", "ListAssistantsResponse"),
    "ListConversionException": ("pinecone.errors.exceptions", "ListConversionException"),
    "ListDocumentsRequest": ("pinecone.models.documents.requests", "ListDocumentsRequest"),
    "ListDocumentsResponse": ("pinecone.models.documents.responses", "ListDocumentsResponse"),
    "ListFilesResponse": ("pinecone.models.assistant.list", "ListFilesResponse"),
    "ListNamespacesResponse": ("pinecone.models.namespaces.models", "ListNamespacesResponse"),
    "ListOperationsResponse": ("pinecone.models.assistant.list", "ListOperationsResponse"),
    "ListResponse": ("pinecone.models.vectors.responses", "ListResponse"),
    "ListedDocumentRecord": ("pinecone.models.documents.responses", "ListedDocumentRecord"),
    "ManagedDeployment": ("pinecone.models.indexes.deployment", "ManagedDeployment"),
    "Message": ("pinecone.models.assistant.message", "Message"),
    "Metric": ("pinecone.models.enums", "Metric"),
    "ModelIndexEmbed": ("pinecone.models.indexes.index", "ModelIndexEmbed"),
    "ModelInfo": ("pinecone.models.inference.models", "ModelInfo"),
    "ModelInfoList": ("pinecone.models.inference.model_list", "ModelInfoList"),
    "NamespaceDescription": ("pinecone.models.namespaces.models", "NamespaceDescription"),
    "NamespaceFieldConfig": ("pinecone.models.namespaces.models", "NamespaceFieldConfig"),
    "NamespaceSchema": ("pinecone.models.namespaces.models", "NamespaceSchema"),
    "NgramConfig": ("pinecone.models.indexes.schema", "NgramConfig"),
    "NotFoundError": ("pinecone.errors.exceptions", "NotFoundError"),
    "NotFoundException": ("pinecone.errors.exceptions", "NotFoundException"),
    "OperationModel": ("pinecone.models.assistant.operation", "OperationModel"),
    "OrganizationList": ("pinecone.models.admin.organization", "OrganizationList"),
    "OrganizationModel": ("pinecone.models.admin.organization", "OrganizationModel"),
    "Page": ("pinecone.models.pagination", "Page"),
    "PaginationResponse": ("pinecone.models.admin.pagination", "PaginationResponse"),
    "Paginator": ("pinecone.models.pagination", "Paginator"),
    "PaymentRequiredError": ("pinecone.errors.exceptions", "PaymentRequiredError"),
    "Pinecone": ("pinecone._client", "Pinecone"),
    "PineconeApiAttributeError": ("pinecone.errors.exceptions", "PineconeApiAttributeError"),
    "PineconeApiException": ("pinecone.errors.exceptions", "PineconeApiException"),
    "PineconeApiKeyError": ("pinecone.errors.exceptions", "PineconeApiKeyError"),
    "PineconeApiTypeError": ("pinecone.errors.exceptions", "PineconeApiTypeError"),
    "PineconeApiValueError": ("pinecone.errors.exceptions", "PineconeApiValueError"),
    "PineconeAsyncio": ("pinecone.async_client.pinecone", "AsyncPinecone"),
    "PineconeConfig": ("pinecone._internal.config", "PineconeConfig"),
    "PineconeConfigurationError": ("pinecone.errors.exceptions", "PineconeConfigurationError"),
    "PineconeConnectionError": ("pinecone.errors.exceptions", "PineconeConnectionError"),
    "PineconeError": ("pinecone.errors.exceptions", "PineconeError"),
    "PineconeException": ("pinecone.errors.exceptions", "PineconeException"),
    "PineconeFuture": ("pinecone.grpc.future", "PineconeFuture"),
    "PineconeProtocolError": ("pinecone.errors.exceptions", "PineconeProtocolError"),
    "PineconeTimeoutError": ("pinecone.errors.exceptions", "PineconeTimeoutError"),
    "PineconeTypeError": ("pinecone.errors.exceptions", "PineconeTypeError"),
    "PineconeValueError": ("pinecone.errors.exceptions", "PineconeValueError"),
    "PodDeployment": ("pinecone.models.indexes.deployment", "PodDeployment"),
    "PodIndexEnvironment": ("pinecone.models.enums", "PodIndexEnvironment"),
    "PodSpec": ("pinecone.models.indexes.specs", "PodSpec"),
    "PodSpecInfo": ("pinecone.models.indexes.index", "PodSpecInfo"),
    "PodType": ("pinecone.models.enums", "PodType"),
    "PrincipalType": ("pinecone.models.admin.role_binding", "PrincipalType"),
    "ProjectList": ("pinecone.models.admin.project", "ProjectList"),
    "ProjectModel": ("pinecone.models.admin.project", "ProjectModel"),
    "QueryNamespacesResults": (
        "pinecone.models.vectors.query_aggregator",
        "QueryNamespacesResults",
    ),
    "QueryResponse": ("pinecone.models.vectors.responses", "QueryResponse"),
    "QueryResultsAggregator": (
        "pinecone.models.vectors.query_aggregator",
        "QueryResultsAggregator",
    ),
    "QueryStringQuery": ("pinecone.models.documents.score_by", "QueryStringQuery"),
    "RankedDocument": ("pinecone.models.inference.rerank", "RankedDocument"),
    "RateLimitError": ("pinecone.errors.exceptions", "RateLimitError"),
    "RateLimitException": ("pinecone.errors.exceptions", "RateLimitException"),
    "ReadCapacityDedicatedConfig": (
        "pinecone.models.indexes.read_capacity",
        "ReadCapacityDedicatedConfig",
    ),
    "ReadCapacityDedicatedResponse": (
        "pinecone.models.indexes.read_capacity",
        "ReadCapacityDedicatedResponse",
    ),
    "ReadCapacityOnDemandResponse": (
        "pinecone.models.indexes.read_capacity",
        "ReadCapacityOnDemandResponse",
    ),
    "ReadCapacityResponse": ("pinecone.models.indexes.read_capacity", "ReadCapacityResponse"),
    "ReadCapacityStatus": ("pinecone.models.indexes.read_capacity", "ReadCapacityStatus"),
    "RerankConfig": ("pinecone.models.vectors.search", "RerankConfig"),
    "RerankModel": ("pinecone.models.enums", "RerankModel"),
    "RerankResult": ("pinecone.models.inference.rerank", "RerankResult"),
    "ResourceType": ("pinecone.models.admin.role_binding", "ResourceType"),
    "ResponseInfo": ("pinecone.models.response_info", "ResponseInfo"),
    "ResponseParsingError": ("pinecone.errors.exceptions", "ResponseParsingError"),
    "RestoreJobList": ("pinecone.models.backups.list", "RestoreJobList"),
    "RestoreJobModel": ("pinecone.models.backups.model", "RestoreJobModel"),
    "RetryConfig": ("pinecone._internal.config", "RetryConfig"),
    "RoleBindingInput": ("pinecone.models.admin.role_binding", "RoleBindingInput"),
    "RoleBindingList": ("pinecone.models.admin.role_binding", "RoleBindingList"),
    "RoleBindingModel": ("pinecone.models.admin.role_binding", "RoleBindingModel"),
    "RoleName": ("pinecone.models.admin.role_binding", "RoleName"),
    "ScalingConfigManual": ("pinecone.models.indexes.read_capacity", "ScalingConfigManual"),
    "SchemaBuilder": ("pinecone.schema_builder", "SchemaBuilder"),
    "ScoredVector": ("pinecone.models.vectors.vector", "ScoredVector"),
    "SearchDocumentsRequest": ("pinecone.models.documents.requests", "SearchDocumentsRequest"),
    "SearchDocumentsResponse": ("pinecone.models.documents.responses", "SearchDocumentsResponse"),
    "SearchInputs": ("pinecone.models.vectors.search", "SearchInputs"),
    "SearchQuery": ("pinecone.db_data.dataclasses.search_query", "SearchQuery"),
    "SearchRecordsResponse": ("pinecone.models.vectors.search", "SearchRecordsResponse"),
    "SearchRerank": ("pinecone.db_data.dataclasses.search_rerank", "SearchRerank"),
    "SearchResult": ("pinecone.models.vectors.search", "SearchResult"),
    "SearchUsage": ("pinecone.models.vectors.search", "SearchUsage"),
    "SemanticTextField": ("pinecone.models.indexes.schema", "SemanticTextField"),
    "ServerlessSpec": ("pinecone.models.indexes.specs", "ServerlessSpec"),
    "ServerlessSpecInfo": ("pinecone.models.indexes.index", "ServerlessSpecInfo"),
    "ServiceAccountList": ("pinecone.models.admin.service_account", "ServiceAccountList"),
    "ServiceAccountModel": ("pinecone.models.admin.service_account", "ServiceAccountModel"),
    "ServiceAccountWithSecret": (
        "pinecone.models.admin.service_account",
        "ServiceAccountWithSecret",
    ),
    "ServiceError": ("pinecone.errors.exceptions", "ServiceError"),
    "ServiceException": ("pinecone.errors.exceptions", "ServiceException"),
    "SparseEmbedding": ("pinecone.models.inference.embed", "SparseEmbedding"),
    "SparseValues": ("pinecone.models.vectors.sparse", "SparseValues"),
    "SparseVectorField": ("pinecone.models.indexes.schema", "SparseVectorField"),
    "SparseVectorQuery": ("pinecone.models.documents.score_by", "SparseVectorQuery"),
    "StartImportResponse": ("pinecone.models.imports.model", "StartImportResponse"),
    "StreamCitationChunk": ("pinecone.models.assistant.streaming", "StreamCitationChunk"),
    "StreamContentChunk": ("pinecone.models.assistant.streaming", "StreamContentChunk"),
    "StreamMessageEnd": ("pinecone.models.assistant.streaming", "StreamMessageEnd"),
    "StreamMessageStart": ("pinecone.models.assistant.streaming", "StreamMessageStart"),
    "StringField": ("pinecone.models.indexes.schema", "StringField"),
    "StringListField": ("pinecone.models.indexes.schema", "StringListField"),
    "TextQuery": ("pinecone.models.documents.score_by", "TextQuery"),
    "TokenResponse": ("pinecone.models.admin.token", "TokenResponse"),
    "UnauthorizedError": ("pinecone.errors.exceptions", "UnauthorizedError"),
    "UnauthorizedException": ("pinecone.errors.exceptions", "UnauthorizedException"),
    "UpdateBackupScheduleRequest": (
        "pinecone.models.backups.schedules",
        "UpdateBackupScheduleRequest",
    ),
    "UpdateDocumentRecord": ("pinecone.models.documents.document", "UpdateDocumentRecord"),
    "UpdateDocumentsRequest": ("pinecone.models.documents.requests", "UpdateDocumentsRequest"),
    "UpdateDocumentsResponse": ("pinecone.models.documents.responses", "UpdateDocumentsResponse"),
    "UpdateResponse": ("pinecone.models.vectors.responses", "UpdateResponse"),
    "UpsertDocumentsRequest": ("pinecone.models.documents.requests", "UpsertDocumentsRequest"),
    "UpsertDocumentsResponse": ("pinecone.models.documents.responses", "UpsertDocumentsResponse"),
    "UpsertRecordsResponse": ("pinecone.models.vectors.responses", "UpsertRecordsResponse"),
    "UpsertResponse": ("pinecone.models.vectors.responses", "UpsertResponse"),
    "UserList": ("pinecone.models.admin.user", "UserList"),
    "UserModel": ("pinecone.models.admin.user", "UserModel"),
    "Vector": ("pinecone.models.vectors.vector", "Vector"),
    "VectorType": ("pinecone.models.enums", "VectorType"),
}


_REMOVED_TOPLEVEL_FUNCTIONS: tuple[str, ...] = (
    "init",
    "create_index",
    "delete_index",
    "list_indexes",
    "describe_index",
    "configure_index",
    "scale_index",
    "create_collection",
    "delete_collection",
    "describe_collection",
    "list_collections",
)

_REMOVED_FUNCTION_EXAMPLES: dict[str, str] = {
    "init": """
    import os
    from pinecone import Pinecone, ServerlessSpec

    pc = Pinecone(
        api_key=os.environ.get("PINECONE_API_KEY")
    )

    # Now do stuff
    if 'my_index' not in pc.list_indexes().names():
        pc.create_index(
            name='my_index',
            dimension=1536,
            metric='euclidean',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-west-2'
            )
        )
""",
    "list_indexes": """
    from pinecone import Pinecone

    pc = Pinecone(api_key='YOUR_API_KEY')

    index_name = "quickstart" # or your index name

    if index_name not in pc.list_indexes().names():
        # do something
""",
    "describe_index": """
    from pinecone import Pinecone

    pc = Pinecone(api_key='YOUR_API_KEY')
    pc.describe_index('my_index')
""",
    "create_index": """
    from pinecone import Pinecone, ServerlessSpec

    pc = Pinecone(api_key='YOUR_API_KEY')
    pc.create_index(
        name='my-index',
        dimension=1536,
        metric='euclidean',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-west-2'
        )
    )
""",
    "delete_index": """
    from pinecone import Pinecone

    pc = Pinecone(api_key='YOUR_API_KEY')
    pc.delete_index('my_index')
""",
    "scale_index": """
    from pinecone import Pinecone

    pc = Pinecone(api_key='YOUR_API_KEY')
    pc.configure_index('my_index', replicas=2)
""",
    "create_collection": """
    from pinecone import Pinecone

    pc = Pinecone(api_key='YOUR_API_KEY')
    pc.create_collection(name='my_collection', source='my_index')
""",
    "list_collections": """
    from pinecone import Pinecone

    pc = Pinecone(api_key='YOUR_API_KEY')
    pc.list_collections()
""",
    "delete_collection": """
    from pinecone import Pinecone

    pc = Pinecone(api_key='YOUR_API_KEY')
    pc.delete_collection('my_collection')
""",
    "describe_collection": """
    from pinecone import Pinecone

    pc = Pinecone(api_key='YOUR_API_KEY')
    pc.describe_collection('my_collection')
""",
    "configure_index": """
    from pinecone import Pinecone

    pc = Pinecone(api_key='YOUR_API_KEY')
    pc.configure_index('my_index', replicas=2)
""",
}


def _removed_function_message(name: str) -> str:
    example = _REMOVED_FUNCTION_EXAMPLES[name]
    if name == "init":
        return (
            "init is no longer a top-level attribute of the pinecone package.\n\n"
            "Please create an instance of the Pinecone class instead.\n\n"
            f"Example:\n{example}\n"
        )
    if name == "scale_index":
        return (
            "scale_index is no longer a top-level attribute of the pinecone package.\n\n"
            "Please create a client instance and call the configure_index method instead.\n\n"
            f"Example:\n{example}\n"
        )
    return (
        f"{name} is no longer a top-level attribute of the pinecone package.\n\n"
        f"To use {name}, please create a client instance and call the method there instead.\n\n"
        f"Example:\n{example}\n"
    )


def __getattr__(name: str) -> Any:
    """Lazily resolve top-level names on first access.

    Most names are imported from their defining submodule the first time
    they're accessed; this is also how the legacy name ``PineconeAsyncio``
    resolves to :class:`AsyncPinecone`. ``ValidationError`` is a legacy
    alias for :class:`PineconeValueError`; accessing it emits a
    :class:`DeprecationWarning`.
    """
    if name == "ValidationError":
        import warnings

        warnings.warn(
            "ValidationError is deprecated; use PineconeValueError instead",
            DeprecationWarning,
            stacklevel=2,
        )
        from pinecone.errors.exceptions import ValidationError

        globals()["ValidationError"] = ValidationError
        return ValidationError
    if name in _REMOVED_TOPLEVEL_FUNCTIONS:
        raise AttributeError(_removed_function_message(name))
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib

        mod = importlib.import_module(module_path)
        value = getattr(mod, attr)
        # Cache on the module so subsequent accesses skip __getattr__
        globals()[name] = value
        return value
    raise AttributeError(f"module 'pinecone' has no attribute {name!r}")


def __dir__() -> list[str]:
    import builtins

    return builtins.list({*globals(), *__all__, *_LAZY_IMPORTS})
