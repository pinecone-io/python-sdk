"""Pinecone Python SDK — vector database for similarity search.

Quick Start::

    from pinecone import Pinecone, ServerlessSpec

    pc = Pinecone(api_key="your-api-key")  # or set PINECONE_API_KEY env var

    # Control plane: manage indexes
    pc.indexes.create(
        name="movie-recommendations",
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

    # Data plane: operate on vectors
    index = pc.index("movie-recommendations")
    index.upsert(vectors=[("movie-42", [0.012, -0.087, 0.153])])  # 1536-dim vector
    results = index.query(vector=[0.012, -0.087, 0.153], top_k=5)  # 1536-dim vector

    # Integrated inference: search with text (server-side embedding)
    index = pc.index("my-integrated-index")
    results = index.search(namespace="default", top_k=5, inputs={"text": "search query"})

The :class:`Pinecone` client manages indexes (control plane). Call
``pc.index(name)`` to get an :class:`Index` for vector operations (data plane).

Async Quick Start::

    from pinecone import AsyncPinecone, ServerlessSpec

    async with AsyncPinecone(api_key="your-api-key") as pc:
        # Control plane: manage indexes
        indexes = await pc.indexes.list()

        # Data plane: resolve host first, then create index client
        desc = await pc.indexes.describe("my-index")
        index = pc.index(host=desc.host)

        async with index:
            results = await index.query(vector=[0.012, -0.087, 0.153], top_k=5)

For async usage, see :class:`AsyncPinecone`. For admin/org management,
see :class:`Admin`.
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
        ForbiddenError,
        ForbiddenException,
        IndexInitFailedError,
        ListConversionException,
        NotFoundError,
        NotFoundException,
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
    from pinecone.models.admin.organization import OrganizationList, OrganizationModel
    from pinecone.models.admin.project import ProjectList, ProjectModel
    from pinecone.models.assistant.chat import (
        ChatCompletionMessage,
        ChatCompletionResponse,
        ChatResponse,
    )
    from pinecone.models.assistant.context import ContextResponse
    from pinecone.models.assistant.evaluation import AlignmentResult
    from pinecone.models.assistant.file_model import AssistantFileModel
    from pinecone.models.assistant.list import ListAssistantsResponse, ListFilesResponse
    from pinecone.models.assistant.message import Message
    from pinecone.models.assistant.model import AssistantModel
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
    from pinecone.models.backups.list import BackupList, RestoreJobList
    from pinecone.models.backups.model import (
        BackupModel,
        CreateIndexFromBackupResponse,
        RestoreJobModel,
    )
    from pinecone.models.collections.list import CollectionList
    from pinecone.models.collections.model import CollectionModel
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
    from pinecone.models.indexes.index import (
        ByocSpecInfo,
        IndexModel,
        IndexSpec,
        IndexTags,
        ModelIndexEmbed,
        PodSpecInfo,
        ServerlessSpecInfo,
    )
    from pinecone.models.indexes.list import IndexList
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
        ListNamespacesResponse,
        NamespaceDescription,
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
    from pinecone.utils.filter_builder import Field, FilterBuilder

__version__ = "9.1.0"

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
    "BatchResponseInfo",
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
    "ConflictError",
    "ContextOptions",
    "ContextResponse",
    "CreateIndexFromBackupResponse",
    "DeletionProtection",
    "DenseEmbedding",
    "DescribeIndexStatsResponse",
    "EmbedConfig",
    "EmbedModel",
    "EmbeddingsList",
    "FetchByMetadataResponse",
    "FetchResponse",
    "Field",
    "FilterBuilder",
    "ForbiddenError",
    "ForbiddenException",
    "GcpRegion",
    "GrpcIndex",
    "Hit",
    "ImportErrorMode",
    "ImportList",
    "ImportModel",
    "Index",
    "IndexEmbed",
    "IndexInitFailedError",
    "IndexList",
    "IndexModel",
    "IndexSpec",
    "IndexTags",
    "IntegratedSpec",
    "ListAssistantsResponse",
    "ListConversionException",
    "ListFilesResponse",
    "ListNamespacesResponse",
    "ListResponse",
    "Message",
    "Metric",
    "ModelIndexEmbed",
    "ModelInfo",
    "ModelInfoList",
    "NamespaceDescription",
    "NotFoundError",
    "NotFoundException",
    "OrganizationList",
    "OrganizationModel",
    "Page",
    "Paginator",
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
    "PodIndexEnvironment",
    "PodSpec",
    "PodSpecInfo",
    "PodType",
    "ProjectList",
    "ProjectModel",
    "QueryNamespacesResults",
    "QueryResponse",
    "QueryResultsAggregator",
    "RankedDocument",
    "RateLimitError",
    "RateLimitException",
    "RerankConfig",
    "RerankModel",
    "RerankResult",
    "ResponseInfo",
    "ResponseParsingError",
    "RestoreJobList",
    "RestoreJobModel",
    "RetryConfig",
    "ScoredVector",
    "SearchInputs",
    "SearchQuery",
    "SearchRecordsResponse",
    "SearchRerank",
    "SearchResult",
    "SearchUsage",
    "ServerlessSpec",
    "ServerlessSpecInfo",
    "ServiceError",
    "ServiceException",
    "SparseEmbedding",
    "SparseValues",
    "StartImportResponse",
    "StreamCitationChunk",
    "StreamContentChunk",
    "StreamMessageEnd",
    "StreamMessageStart",
    "UnauthorizedError",
    "UnauthorizedException",
    "UpdateResponse",
    "UpsertRecordsResponse",
    "UpsertResponse",
    "Vector",
    "VectorType",
    "__version__",
]

# Lazy-load heavy classes to keep cold import under 10ms.
# Importing Pinecone/AsyncPinecone/Index eagerly pulls in httpx (~120ms).
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "ApiError": ("pinecone.errors.exceptions", "ApiError"),
    "ConflictError": ("pinecone.errors.exceptions", "ConflictError"),
    "ForbiddenError": ("pinecone.errors.exceptions", "ForbiddenError"),
    "ForbiddenException": ("pinecone.errors.exceptions", "ForbiddenException"),
    "IndexInitFailedError": ("pinecone.errors.exceptions", "IndexInitFailedError"),
    "ListConversionException": ("pinecone.errors.exceptions", "ListConversionException"),
    "NotFoundError": ("pinecone.errors.exceptions", "NotFoundError"),
    "NotFoundException": ("pinecone.errors.exceptions", "NotFoundException"),
    "PineconeApiAttributeError": ("pinecone.errors.exceptions", "PineconeApiAttributeError"),
    "PineconeApiException": ("pinecone.errors.exceptions", "PineconeApiException"),
    "PineconeApiKeyError": ("pinecone.errors.exceptions", "PineconeApiKeyError"),
    "PineconeApiTypeError": ("pinecone.errors.exceptions", "PineconeApiTypeError"),
    "PineconeApiValueError": ("pinecone.errors.exceptions", "PineconeApiValueError"),
    "PineconeConfigurationError": ("pinecone.errors.exceptions", "PineconeConfigurationError"),
    "PineconeConnectionError": ("pinecone.errors.exceptions", "PineconeConnectionError"),
    "PineconeError": ("pinecone.errors.exceptions", "PineconeError"),
    "PineconeException": ("pinecone.errors.exceptions", "PineconeException"),
    "PineconeProtocolError": ("pinecone.errors.exceptions", "PineconeProtocolError"),
    "PineconeTimeoutError": ("pinecone.errors.exceptions", "PineconeTimeoutError"),
    "PineconeTypeError": ("pinecone.errors.exceptions", "PineconeTypeError"),
    "PineconeValueError": ("pinecone.errors.exceptions", "PineconeValueError"),
    "RateLimitError": ("pinecone.errors.exceptions", "RateLimitError"),
    "RateLimitException": ("pinecone.errors.exceptions", "RateLimitException"),
    "ResponseParsingError": ("pinecone.errors.exceptions", "ResponseParsingError"),
    "ServiceError": ("pinecone.errors.exceptions", "ServiceError"),
    "ServiceException": ("pinecone.errors.exceptions", "ServiceException"),
    "UnauthorizedError": ("pinecone.errors.exceptions", "UnauthorizedError"),
    "UnauthorizedException": ("pinecone.errors.exceptions", "UnauthorizedException"),
    "Admin": ("pinecone.admin", "Admin"),
    "APIKeyRole": ("pinecone.models.admin.api_key", "APIKeyRole"),
    "APIKeyList": ("pinecone.models.admin.api_key", "APIKeyList"),
    "APIKeyModel": ("pinecone.models.admin.api_key", "APIKeyModel"),
    "APIKeyWithSecret": ("pinecone.models.admin.api_key", "APIKeyWithSecret"),
    "AlignmentResult": ("pinecone.models.assistant.evaluation", "AlignmentResult"),
    "AsyncPaginator": ("pinecone.models.pagination", "AsyncPaginator"),
    "AssistantFileModel": ("pinecone.models.assistant.file_model", "AssistantFileModel"),
    "AssistantModel": ("pinecone.models.assistant.model", "AssistantModel"),
    "AsyncChatCompletionStream": (
        "pinecone.models.assistant.streaming",
        "AsyncChatCompletionStream",
    ),
    "AsyncChatStream": ("pinecone.models.assistant.streaming", "AsyncChatStream"),
    "AsyncIndex": ("pinecone.async_client.async_index", "AsyncIndex"),
    "AsyncPinecone": ("pinecone.async_client.pinecone", "AsyncPinecone"),
    "PineconeAsyncio": ("pinecone.async_client.pinecone", "AsyncPinecone"),
    "BackupList": ("pinecone.models.backups.list", "BackupList"),
    "BackupModel": ("pinecone.models.backups.model", "BackupModel"),
    "ByocSpec": ("pinecone.models.indexes.specs", "ByocSpec"),
    "ByocSpecInfo": ("pinecone.models.indexes.index", "ByocSpecInfo"),
    "CloudProvider": ("pinecone.models.enums", "CloudProvider"),
    "ChatCompletionMessage": ("pinecone.models.assistant.chat", "ChatCompletionMessage"),
    "ChatCompletionResponse": ("pinecone.models.assistant.chat", "ChatCompletionResponse"),
    "ChatCompletionStream": (
        "pinecone.models.assistant.streaming",
        "ChatCompletionStream",
    ),
    "ChatCompletionStreamChunk": (
        "pinecone.models.assistant.streaming",
        "ChatCompletionStreamChunk",
    ),
    "ChatResponse": ("pinecone.models.assistant.chat", "ChatResponse"),
    "ChatStream": ("pinecone.models.assistant.streaming", "ChatStream"),
    "ChatStreamChunk": ("pinecone.models.assistant.streaming", "ChatStreamChunk"),
    "AwsRegion": ("pinecone.db_control.enums.clouds", "AwsRegion"),
    "AzureRegion": ("pinecone.db_control.enums.clouds", "AzureRegion"),
    "CollectionDescription": (
        "pinecone.db_control.models.collection_description",
        "CollectionDescription",
    ),
    "CollectionList": ("pinecone.models.collections.list", "CollectionList"),
    "CollectionModel": ("pinecone.models.collections.model", "CollectionModel"),
    "ContextOptions": ("pinecone.models.assistant.options", "ContextOptions"),
    "ContextResponse": ("pinecone.models.assistant.context", "ContextResponse"),
    "CreateIndexFromBackupResponse": (
        "pinecone.models.backups.model",
        "CreateIndexFromBackupResponse",
    ),
    "DeletionProtection": ("pinecone.models.enums", "DeletionProtection"),
    "DenseEmbedding": ("pinecone.models.inference.embed", "DenseEmbedding"),
    "DescribeIndexStatsResponse": (
        "pinecone.models.vectors.responses",
        "DescribeIndexStatsResponse",
    ),
    "EmbedConfig": ("pinecone.models.indexes.specs", "EmbedConfig"),
    "EmbedModel": ("pinecone.models.enums", "EmbedModel"),
    "EmbeddingsList": ("pinecone.models.inference.embed", "EmbeddingsList"),
    "FetchByMetadataResponse": (
        "pinecone.models.vectors.responses",
        "FetchByMetadataResponse",
    ),
    "FetchResponse": ("pinecone.models.vectors.responses", "FetchResponse"),
    "Field": ("pinecone.utils.filter_builder", "Field"),
    "FilterBuilder": ("pinecone.utils.filter_builder", "FilterBuilder"),
    "GcpRegion": ("pinecone.db_control.enums.clouds", "GcpRegion"),
    "GrpcIndex": ("pinecone.grpc", "GrpcIndex"),
    "Hit": ("pinecone.models.vectors.search", "Hit"),
    "PineconeFuture": ("pinecone.grpc.future", "PineconeFuture"),
    "ImportErrorMode": ("pinecone.models.imports.error_mode", "ImportErrorMode"),
    "ImportList": ("pinecone.models.imports.list", "ImportList"),
    "ImportModel": ("pinecone.models.imports.model", "ImportModel"),
    "Index": ("pinecone.index", "Index"),
    "IndexEmbed": ("pinecone.inference.models.index_embed", "IndexEmbed"),
    "IndexList": ("pinecone.models.indexes.list", "IndexList"),
    "IndexModel": ("pinecone.models.indexes.index", "IndexModel"),
    "IndexSpec": ("pinecone.models.indexes.index", "IndexSpec"),
    "IndexTags": ("pinecone.models.indexes.index", "IndexTags"),
    "IntegratedSpec": ("pinecone.models.indexes.specs", "IntegratedSpec"),
    "ListNamespacesResponse": (
        "pinecone.models.namespaces.models",
        "ListNamespacesResponse",
    ),
    "ListAssistantsResponse": (
        "pinecone.models.assistant.list",
        "ListAssistantsResponse",
    ),
    "ListFilesResponse": ("pinecone.models.assistant.list", "ListFilesResponse"),
    "ListResponse": ("pinecone.models.vectors.responses", "ListResponse"),
    "Message": ("pinecone.models.assistant.message", "Message"),
    "Metric": ("pinecone.models.enums", "Metric"),
    "ModelInfo": ("pinecone.models.inference.models", "ModelInfo"),
    "ModelIndexEmbed": ("pinecone.models.indexes.index", "ModelIndexEmbed"),
    "ModelInfoList": ("pinecone.models.inference.model_list", "ModelInfoList"),
    "NamespaceDescription": (
        "pinecone.models.namespaces.models",
        "NamespaceDescription",
    ),
    "OrganizationList": ("pinecone.models.admin.organization", "OrganizationList"),
    "OrganizationModel": ("pinecone.models.admin.organization", "OrganizationModel"),
    "Page": ("pinecone.models.pagination", "Page"),
    "Paginator": ("pinecone.models.pagination", "Paginator"),
    "Pinecone": ("pinecone._client", "Pinecone"),
    "PineconeConfig": ("pinecone._internal.config", "PineconeConfig"),
    "PodIndexEnvironment": ("pinecone.models.enums", "PodIndexEnvironment"),
    "PodSpec": ("pinecone.models.indexes.specs", "PodSpec"),
    "PodType": ("pinecone.models.enums", "PodType"),
    "ProjectList": ("pinecone.models.admin.project", "ProjectList"),
    "ProjectModel": ("pinecone.models.admin.project", "ProjectModel"),
    "PodSpecInfo": ("pinecone.models.indexes.index", "PodSpecInfo"),
    "QueryNamespacesResults": (
        "pinecone.models.vectors.query_aggregator",
        "QueryNamespacesResults",
    ),
    "QueryResponse": ("pinecone.models.vectors.responses", "QueryResponse"),
    "QueryResultsAggregator": (
        "pinecone.models.vectors.query_aggregator",
        "QueryResultsAggregator",
    ),
    "RerankConfig": ("pinecone.models.vectors.search", "RerankConfig"),
    "RankedDocument": ("pinecone.models.inference.rerank", "RankedDocument"),
    "RerankModel": ("pinecone.models.enums", "RerankModel"),
    "RerankResult": ("pinecone.models.inference.rerank", "RerankResult"),
    "BatchResponseInfo": ("pinecone.models.response_info", "BatchResponseInfo"),
    "ResponseInfo": ("pinecone.models.response_info", "ResponseInfo"),
    "RestoreJobList": ("pinecone.models.backups.list", "RestoreJobList"),
    "RestoreJobModel": ("pinecone.models.backups.model", "RestoreJobModel"),
    "RetryConfig": ("pinecone._internal.config", "RetryConfig"),
    "SearchInputs": ("pinecone.models.vectors.search", "SearchInputs"),
    "SearchQuery": ("pinecone.db_data.dataclasses.search_query", "SearchQuery"),
    "SearchRecordsResponse": (
        "pinecone.models.vectors.search",
        "SearchRecordsResponse",
    ),
    "SearchRerank": ("pinecone.db_data.dataclasses.search_rerank", "SearchRerank"),
    "SearchResult": ("pinecone.models.vectors.search", "SearchResult"),
    "SearchUsage": ("pinecone.models.vectors.search", "SearchUsage"),
    "ScoredVector": ("pinecone.models.vectors.vector", "ScoredVector"),
    "ServerlessSpec": ("pinecone.models.indexes.specs", "ServerlessSpec"),
    "ServerlessSpecInfo": ("pinecone.models.indexes.index", "ServerlessSpecInfo"),
    "SparseEmbedding": ("pinecone.models.inference.embed", "SparseEmbedding"),
    "SparseValues": ("pinecone.models.vectors.sparse", "SparseValues"),
    "StartImportResponse": (
        "pinecone.models.imports.model",
        "StartImportResponse",
    ),
    "StreamCitationChunk": (
        "pinecone.models.assistant.streaming",
        "StreamCitationChunk",
    ),
    "StreamContentChunk": (
        "pinecone.models.assistant.streaming",
        "StreamContentChunk",
    ),
    "StreamMessageEnd": ("pinecone.models.assistant.streaming", "StreamMessageEnd"),
    "StreamMessageStart": ("pinecone.models.assistant.streaming", "StreamMessageStart"),
    "UpdateResponse": ("pinecone.models.vectors.responses", "UpdateResponse"),
    "UpsertRecordsResponse": (
        "pinecone.models.vectors.responses",
        "UpsertRecordsResponse",
    ),
    "UpsertResponse": ("pinecone.models.vectors.responses", "UpsertResponse"),
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
