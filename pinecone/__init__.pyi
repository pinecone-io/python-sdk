"""Auto-generated stub for pinecone/__init__.py — do not edit manually.

Regenerate after changes to _LAZY_IMPORTS or __all__ in pinecone/__init__.py:
    uv run python scripts/generate_init_stub.py
"""
from pinecone._client import Pinecone as Pinecone
from pinecone._internal.config import PineconeConfig as PineconeConfig, RetryConfig as RetryConfig
from pinecone.admin import Admin as Admin
from pinecone.async_client.async_index import AsyncIndex as AsyncIndex
from pinecone.async_client.pinecone import AsyncPinecone as AsyncPinecone, AsyncPinecone as PineconeAsyncio
from pinecone.db_control.enums.clouds import AwsRegion as AwsRegion, AzureRegion as AzureRegion, GcpRegion as GcpRegion
from pinecone.db_control.models.collection_description import CollectionDescription as CollectionDescription
from pinecone.db_data.dataclasses.search_query import SearchQuery as SearchQuery
from pinecone.db_data.dataclasses.search_rerank import SearchRerank as SearchRerank
from pinecone.errors.exceptions import ApiError as ApiError, ConflictError as ConflictError, FailedPreconditionError as FailedPreconditionError, ForbiddenError as ForbiddenError, ForbiddenException as ForbiddenException, IndexInitFailedError as IndexInitFailedError, IndexTerminatedError as IndexTerminatedError, ListConversionException as ListConversionException, NotFoundError as NotFoundError, NotFoundException as NotFoundException, PaymentRequiredError as PaymentRequiredError, PineconeApiAttributeError as PineconeApiAttributeError, PineconeApiException as PineconeApiException, PineconeApiKeyError as PineconeApiKeyError, PineconeApiTypeError as PineconeApiTypeError, PineconeApiValueError as PineconeApiValueError, PineconeConfigurationError as PineconeConfigurationError, PineconeConnectionError as PineconeConnectionError, PineconeError as PineconeError, PineconeException as PineconeException, PineconeProtocolError as PineconeProtocolError, PineconeTimeoutError as PineconeTimeoutError, PineconeTypeError as PineconeTypeError, PineconeValueError as PineconeValueError, RateLimitError as RateLimitError, RateLimitException as RateLimitException, ResponseParsingError as ResponseParsingError, ServiceError as ServiceError, ServiceException as ServiceException, UnauthorizedError as UnauthorizedError, UnauthorizedException as UnauthorizedException
from pinecone.grpc import GrpcIndex as GrpcIndex
from pinecone.grpc.future import PineconeFuture as PineconeFuture
from pinecone.index import Index as Index
from pinecone.inference.models.index_embed import IndexEmbed as IndexEmbed
from pinecone.models.admin.api_key import APIKeyList as APIKeyList, APIKeyModel as APIKeyModel, APIKeyRole as APIKeyRole, APIKeyWithSecret as APIKeyWithSecret
from pinecone.models.admin.invite import InviteList as InviteList, InviteModel as InviteModel, InviteStatus as InviteStatus
from pinecone.models.admin.organization import OrganizationList as OrganizationList, OrganizationModel as OrganizationModel
from pinecone.models.admin.pagination import PaginationResponse as PaginationResponse
from pinecone.models.admin.project import ProjectList as ProjectList, ProjectModel as ProjectModel
from pinecone.models.admin.role_binding import PrincipalType as PrincipalType, ResourceType as ResourceType, RoleBindingInput as RoleBindingInput, RoleBindingList as RoleBindingList, RoleBindingModel as RoleBindingModel, RoleName as RoleName
from pinecone.models.admin.service_account import ServiceAccountList as ServiceAccountList, ServiceAccountModel as ServiceAccountModel, ServiceAccountWithSecret as ServiceAccountWithSecret
from pinecone.models.admin.token import TokenResponse as TokenResponse
from pinecone.models.admin.user import UserList as UserList, UserModel as UserModel
from pinecone.models.assistant.chat import ChatCompletionMessage as ChatCompletionMessage, ChatCompletionResponse as ChatCompletionResponse, ChatResponse as ChatResponse
from pinecone.models.assistant.context import ContextResponse as ContextResponse
from pinecone.models.assistant.evaluation import AlignmentResult as AlignmentResult
from pinecone.models.assistant.file_model import AssistantFileModel as AssistantFileModel
from pinecone.models.assistant.list import ListAssistantsResponse as ListAssistantsResponse, ListFilesResponse as ListFilesResponse, ListOperationsResponse as ListOperationsResponse
from pinecone.models.assistant.message import Message as Message
from pinecone.models.assistant.model import AssistantModel as AssistantModel
from pinecone.models.assistant.operation import OperationModel as OperationModel
from pinecone.models.assistant.options import ContextOptions as ContextOptions
from pinecone.models.assistant.streaming import AsyncChatCompletionStream as AsyncChatCompletionStream, AsyncChatStream as AsyncChatStream, ChatCompletionStream as ChatCompletionStream, ChatCompletionStreamChunk as ChatCompletionStreamChunk, ChatStream as ChatStream, ChatStreamChunk as ChatStreamChunk, StreamCitationChunk as StreamCitationChunk, StreamContentChunk as StreamContentChunk, StreamMessageEnd as StreamMessageEnd, StreamMessageStart as StreamMessageStart
from pinecone.models.backups.list import BackupList as BackupList, BackupScheduleHistoryList as BackupScheduleHistoryList, BackupScheduleList as BackupScheduleList, RestoreJobList as RestoreJobList
from pinecone.models.backups.model import BackupModel as BackupModel, CreateIndexFromBackupRequest as CreateIndexFromBackupRequest, CreateIndexFromBackupResponse as CreateIndexFromBackupResponse, RestoreJobModel as RestoreJobModel
from pinecone.models.backups.schedules import BackupScheduleHistoryItem as BackupScheduleHistoryItem, BackupScheduleModel as BackupScheduleModel, CreateBackupScheduleRequest as CreateBackupScheduleRequest, UpdateBackupScheduleRequest as UpdateBackupScheduleRequest
from pinecone.models.collections.list import CollectionList as CollectionList
from pinecone.models.collections.model import CollectionModel as CollectionModel
from pinecone.models.documents.document import Document as Document, DocumentRecord as DocumentRecord, UpdateDocumentRecord as UpdateDocumentRecord
from pinecone.models.documents.requests import DeleteDocumentsRequest as DeleteDocumentsRequest, FetchDocumentsRequest as FetchDocumentsRequest, ListDocumentsRequest as ListDocumentsRequest, SearchDocumentsRequest as SearchDocumentsRequest, UpdateDocumentsRequest as UpdateDocumentsRequest, UpsertDocumentsRequest as UpsertDocumentsRequest
from pinecone.models.documents.responses import DeleteDocumentsResponse as DeleteDocumentsResponse, DocumentFetchUsage as DocumentFetchUsage, DocumentListUsage as DocumentListUsage, DocumentSearchUsage as DocumentSearchUsage, FetchDocumentsResponse as FetchDocumentsResponse, ListDocumentsResponse as ListDocumentsResponse, ListedDocumentRecord as ListedDocumentRecord, SearchDocumentsResponse as SearchDocumentsResponse, UpdateDocumentsResponse as UpdateDocumentsResponse, UpsertDocumentsResponse as UpsertDocumentsResponse
from pinecone.models.documents.score_by import DenseVectorQuery as DenseVectorQuery, DocumentScoringMethod as DocumentScoringMethod, QueryStringQuery as QueryStringQuery, SparseVectorQuery as SparseVectorQuery, TextQuery as TextQuery
from pinecone.models.enums import CloudProvider as CloudProvider, DeletionProtection as DeletionProtection, EmbedModel as EmbedModel, Metric as Metric, PodIndexEnvironment as PodIndexEnvironment, PodType as PodType, RerankModel as RerankModel, VectorType as VectorType
from pinecone.models.imports.error_mode import ImportErrorMode as ImportErrorMode
from pinecone.models.imports.list import ImportList as ImportList
from pinecone.models.imports.model import ImportModel as ImportModel, StartImportResponse as StartImportResponse
from pinecone.models.indexes.deployment import ByocDeployment as ByocDeployment, IndexDeployment as IndexDeployment, ManagedDeployment as ManagedDeployment, PodDeployment as PodDeployment
from pinecone.models.indexes.index import ByocSpecInfo as ByocSpecInfo, IndexModel as IndexModel, IndexSpec as IndexSpec, IndexStatus as IndexStatus, IndexTags as IndexTags, ModelIndexEmbed as ModelIndexEmbed, PodSpecInfo as PodSpecInfo, ServerlessSpecInfo as ServerlessSpecInfo
from pinecone.models.indexes.list import IndexList as IndexList
from pinecone.models.indexes.read_capacity import ReadCapacityDedicatedConfig as ReadCapacityDedicatedConfig, ReadCapacityDedicatedResponse as ReadCapacityDedicatedResponse, ReadCapacityOnDemandResponse as ReadCapacityOnDemandResponse, ReadCapacityResponse as ReadCapacityResponse, ReadCapacityStatus as ReadCapacityStatus, ScalingConfigManual as ScalingConfigManual
from pinecone.models.indexes.requests import ConfigureIndexRequest as ConfigureIndexRequest, CreateIndexRequest as CreateIndexRequest
from pinecone.models.indexes.schema import BooleanField as BooleanField, DenseVectorField as DenseVectorField, FloatField as FloatField, FullTextSearchConfig as FullTextSearchConfig, IndexSchema as IndexSchema, IndexSchemaField as IndexSchemaField, IntegerField as IntegerField, LegacyMetadataField as LegacyMetadataField, NgramConfig as NgramConfig, SemanticTextField as SemanticTextField, SparseVectorField as SparseVectorField, StringField as StringField, StringListField as StringListField
from pinecone.models.indexes.specs import ByocSpec as ByocSpec, EmbedConfig as EmbedConfig, IntegratedSpec as IntegratedSpec, PodSpec as PodSpec, ServerlessSpec as ServerlessSpec
from pinecone.models.inference.embed import DenseEmbedding as DenseEmbedding, EmbeddingsList as EmbeddingsList, SparseEmbedding as SparseEmbedding
from pinecone.models.inference.model_list import ModelInfoList as ModelInfoList
from pinecone.models.inference.models import ModelInfo as ModelInfo
from pinecone.models.inference.rerank import RankedDocument as RankedDocument, RerankResult as RerankResult
from pinecone.models.namespaces.models import IndexedFields as IndexedFields, ListNamespacesResponse as ListNamespacesResponse, NamespaceDescription as NamespaceDescription, NamespaceFieldConfig as NamespaceFieldConfig, NamespaceSchema as NamespaceSchema
from pinecone.models.pagination import AsyncPaginator as AsyncPaginator, Page as Page, Paginator as Paginator
from pinecone.models.response_info import BatchResponseInfo as BatchResponseInfo, ResponseInfo as ResponseInfo
from pinecone.models.vectors.query_aggregator import QueryNamespacesResults as QueryNamespacesResults, QueryResultsAggregator as QueryResultsAggregator
from pinecone.models.vectors.responses import DescribeIndexStatsResponse as DescribeIndexStatsResponse, FetchByMetadataResponse as FetchByMetadataResponse, FetchResponse as FetchResponse, ListResponse as ListResponse, QueryResponse as QueryResponse, UpdateResponse as UpdateResponse, UpsertRecordsResponse as UpsertRecordsResponse, UpsertResponse as UpsertResponse
from pinecone.models.vectors.search import Hit as Hit, RerankConfig as RerankConfig, SearchInputs as SearchInputs, SearchRecordsResponse as SearchRecordsResponse, SearchResult as SearchResult, SearchUsage as SearchUsage
from pinecone.models.vectors.sparse import SparseValues as SparseValues
from pinecone.models.vectors.vector import ScoredVector as ScoredVector, Vector as Vector
from pinecone.schema_builder import SchemaBuilder as SchemaBuilder
from pinecone.utils.filter_builder import Field as Field, FilterBuilder as FilterBuilder

__version__: str

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
    "PineconeAsyncio",
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
