Models
======

All public model types returned by SDK methods.  Every model is an immutable
:class:`msgspec.Struct` subclass — fields are accessed as plain attributes
(e.g. ``idx.name``).

.. contents:: Sections
   :local:
   :depth: 1

Index Models
------------

.. autoclass:: pinecone.models.indexes.index.IndexModel
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.indexes.list.IndexList
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.indexes.index.IndexStatus
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.indexes.index.IndexTags
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.indexes.specs.ServerlessSpec
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.indexes.specs.PodSpec
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.indexes.specs.ByocSpec
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.indexes.specs.IntegratedSpec
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.indexes.specs.EmbedConfig
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.indexes.requests.CreateIndexRequest
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.indexes.requests.ConfigureIndexRequest
   :members:
   :show-inheritance:

Index Schema Models
-------------------

``IndexModel.schema`` describes every field in the index.  These types replace
the removed ``IndexModel.dimension``, ``.metric``, ``.vector_type`` and
``.embed`` attributes.

.. autoclass:: pinecone.models.indexes.schema.IndexSchema
   :members:
   :show-inheritance:

.. autodata:: pinecone.models.indexes.schema.IndexSchemaField

.. autoclass:: pinecone.models.indexes.schema.DenseVectorField
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.indexes.schema.SparseVectorField
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.indexes.schema.SemanticTextField
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.indexes.schema.StringField
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.indexes.schema.StringListField
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.indexes.schema.BooleanField
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.indexes.schema.IntegerField
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.indexes.schema.FloatField
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.indexes.schema.LegacyMetadataField
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.indexes.schema.FullTextSearchConfig
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.indexes.schema.NgramConfig
   :members:
   :show-inheritance:

Index Deployment Models
-----------------------

``IndexModel.deployment`` describes where and how the index runs.  These types
replace the removed ``IndexSpec``, ``ServerlessSpecInfo``, ``PodSpecInfo`` and
``ByocSpecInfo``.

.. autodata:: pinecone.models.indexes.deployment.IndexDeployment

.. autoclass:: pinecone.models.indexes.deployment.ManagedDeployment
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.indexes.deployment.PodDeployment
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.indexes.deployment.ByocDeployment
   :members:
   :show-inheritance:

Read Capacity Models
--------------------

``IndexModel.read_capacity`` replaces the removed
``IndexSpec.serverless.read_capacity``.

.. autodata:: pinecone.models.indexes.read_capacity.ReadCapacityResponse

.. autoclass:: pinecone.models.indexes.read_capacity.ReadCapacityOnDemandResponse
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.indexes.read_capacity.ReadCapacityDedicatedResponse
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.indexes.read_capacity.ReadCapacityDedicatedConfig
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.indexes.read_capacity.ReadCapacityStatus
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.indexes.read_capacity.ScalingConfigManual
   :members:
   :show-inheritance:

Vector Models
-------------

.. autoclass:: pinecone.models.vectors.vector.Vector
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.vectors.vector.ScoredVector
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.vectors.sparse.SparseValues
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.vectors.usage.Usage
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.vectors.responses.QueryResponse
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.vectors.responses.FetchResponse
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.vectors.responses.FetchByMetadataResponse
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.vectors.responses.UpsertResponse
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.vectors.responses.UpdateResponse
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.vectors.responses.ListResponse
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.vectors.responses.ListItem
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.vectors.responses.Pagination
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.vectors.responses.DescribeIndexStatsResponse
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.vectors.responses.NamespaceSummary
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.vectors.responses.UpsertRecordsResponse
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.response_info.BatchResponseInfo
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.response_info.ResponseInfo
   :members:
   :show-inheritance:

Batch Models
------------

``index.documents.batch_upsert`` returns a :class:`~pinecone.models.batch.BatchResult`,
which collects per-batch failures instead of raising on the first one.

.. autoclass:: pinecone.models.batch.BatchResult
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.batch.BatchError
   :members:
   :show-inheritance:

Search Models
-------------

.. autoclass:: pinecone.models.vectors.search.Hit
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.vectors.search.SearchResult
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.vectors.search.SearchRecordsResponse
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.vectors.search.SearchInputs
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.vectors.search.SearchUsage
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.vectors.search.RerankConfig
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.vectors.search.SearchQuery
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.vectors.search.SearchQueryVector
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.vectors.search.SearchRerank
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.vectors.query_aggregator.QueryNamespacesResults
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.vectors.query_aggregator.QueryResultsAggregator
   :members:
   :show-inheritance:

Document Models
---------------

The document interface — ``index.documents`` on either client — works in whole
records rather than raw vectors.  You write :class:`~pinecone.models.documents.document.DocumentRecord`
and read back :class:`~pinecone.models.documents.document.Document`.

.. autoclass:: pinecone.models.documents.document.Document
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.documents.document.DocumentRecord
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.documents.document.UpdateDocumentRecord
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.documents.responses.ListedDocumentRecord
   :members:
   :show-inheritance:

Document Scoring
~~~~~~~~~~~~~~~~

``score_by`` picks how a search ranks candidates.  Pass one of these query types.

.. autodata:: pinecone.models.documents.score_by.DocumentScoringMethod

.. autoclass:: pinecone.models.documents.score_by.TextQuery
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.documents.score_by.QueryStringQuery
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.documents.score_by.DenseVectorQuery
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.documents.score_by.SparseVectorQuery
   :members:
   :show-inheritance:

Document Responses
~~~~~~~~~~~~~~~~~~

.. autoclass:: pinecone.models.documents.responses.UpsertDocumentsResponse
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.documents.responses.SearchDocumentsResponse
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.documents.responses.FetchDocumentsResponse
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.documents.responses.ListDocumentsResponse
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.documents.responses.UpdateDocumentsResponse
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.documents.responses.DeleteDocumentsResponse
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.documents.responses.DocumentSearchUsage
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.documents.responses.DocumentFetchUsage
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.documents.responses.DocumentListUsage
   :members:
   :show-inheritance:

Document Requests
~~~~~~~~~~~~~~~~~

Document methods are keyword-only; you never build one of these yourself.  They
document the request body each method sends, which is what you are reading when
a validation error names a field.

.. autoclass:: pinecone.models.documents.requests.UpsertDocumentsRequest
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.documents.requests.SearchDocumentsRequest
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.documents.requests.FetchDocumentsRequest
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.documents.requests.ListDocumentsRequest
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.documents.requests.UpdateDocumentsRequest
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.documents.requests.DeleteDocumentsRequest
   :members:
   :show-inheritance:

Inference Models
----------------

.. autoclass:: pinecone.models.inference.embed.DenseEmbedding
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.inference.embed.SparseEmbedding
   :members:
   :show-inheritance:

.. autodata:: pinecone.models.inference.embed.Embedding

.. autoclass:: pinecone.models.inference.embed.EmbeddingsList
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.inference.embed.EmbedUsage
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.inference.rerank.RerankResult
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.inference.rerank.RankedDocument
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.inference.rerank.RerankUsage
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.inference.models.ModelInfo
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.inference.models.ModelInfoSupportedParameter
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.inference.model_list.ModelInfoList
   :members:
   :show-inheritance:

.. autoclass:: pinecone.inference.models.index_embed.IndexEmbed
   :members:
   :show-inheritance:

Import Models
-------------

.. autoclass:: pinecone.models.imports.model.ImportModel
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.imports.list.ImportList
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.imports.model.StartImportResponse
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.imports.error_mode.ImportErrorMode
   :members:
   :show-inheritance:

Collection Models
-----------------

.. autoclass:: pinecone.models.collections.model.CollectionModel
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.collections.list.CollectionList
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.collections.description.CollectionDescription
   :members:
   :show-inheritance:

Backup and Restore Models
--------------------------

.. autoclass:: pinecone.models.backups.model.BackupModel
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.backups.list.BackupList
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.backups.model.RestoreJobModel
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.backups.list.RestoreJobList
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.backups.model.CreateIndexFromBackupRequest
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.backups.model.CreateIndexFromBackupResponse
   :members:
   :show-inheritance:

Backup Schedule Models
-----------------------

.. autoclass:: pinecone.models.backups.schedules.BackupScheduleModel
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.backups.list.BackupScheduleList
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.backups.schedules.BackupScheduleHistoryItem
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.backups.list.BackupScheduleHistoryList
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.backups.schedules.CreateBackupScheduleRequest
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.backups.schedules.UpdateBackupScheduleRequest
   :members:
   :show-inheritance:

Namespace Models
----------------

.. autoclass:: pinecone.models.namespaces.models.NamespaceDescription
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.namespaces.models.ListNamespacesResponse
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.namespaces.models.NamespaceSchema
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.namespaces.models.NamespaceFieldConfig
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.namespaces.models.IndexedFields
   :members:
   :show-inheritance:

Pagination Models
-----------------

.. autoclass:: pinecone.models.pagination.Page
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.pagination.Paginator
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.pagination.AsyncPaginator
   :members:
   :show-inheritance:

Enums
-----

.. autoclass:: pinecone.models.enums.CloudProvider
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.enums.Metric
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.enums.VectorType
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.enums.DeletionProtection
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.enums.EmbedModel
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.enums.RerankModel
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.enums.PodType
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.enums.PodIndexEnvironment
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.enums.AwsRegion
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.enums.AzureRegion
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.enums.GcpRegion
   :members:
   :show-inheritance:

Admin Models
------------

.. autoclass:: pinecone.models.admin.api_key.APIKeyModel
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.admin.api_key.APIKeyList
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.admin.api_key.APIKeyWithSecret
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.admin.api_key.APIKeyRole
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.admin.organization.OrganizationModel
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.admin.organization.OrganizationList
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.admin.project.ProjectModel
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.admin.project.ProjectList
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.admin.token.TokenResponse
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.admin.pagination.PaginationResponse
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.admin.user.UserModel
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.admin.user.UserList
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.admin.invite.InviteModel
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.admin.invite.InviteList
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.admin.invite.InviteStatus
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.admin.service_account.ServiceAccountModel
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.admin.service_account.ServiceAccountList
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.admin.service_account.ServiceAccountWithSecret
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.admin.role_binding.RoleBindingModel
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.admin.role_binding.RoleBindingList
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.admin.role_binding.RoleBindingInput
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.admin.role_binding.RoleName
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.admin.role_binding.PrincipalType
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.admin.role_binding.ResourceType
   :members:
   :show-inheritance:

Assistant Models
----------------

.. autoclass:: pinecone.models.assistant.model.AssistantModel
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.assistant.file_model.AssistantFileModel
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.assistant.list.ListAssistantsResponse
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.assistant.list.ListFilesResponse
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.assistant.operation.OperationModel
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.assistant.list.ListOperationsResponse
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.assistant.message.Message
   :members:
   :show-inheritance:

Assistant Chat Models
~~~~~~~~~~~~~~~~~~~~~

:meth:`pc.assistants.chat() <pinecone.client.assistants.Assistants.chat>` returns a
:class:`~pinecone.models.assistant.chat.ChatResponse`;
:meth:`pc.assistants.chat_completions() <pinecone.client.assistants.Assistants.chat_completions>`
returns the OpenAI-shaped :class:`~pinecone.models.assistant.chat.ChatCompletionResponse`.

.. autoclass:: pinecone.models.assistant.chat.ChatResponse
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.assistant.chat.ChatMessage
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.assistant.chat.ChatCitation
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.assistant.chat.ChatReference
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.assistant.chat.ChatHighlight
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.assistant.chat.ChatUsage
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.assistant.chat.ChatCompletionResponse
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.assistant.chat.ChatCompletionChoice
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.assistant.chat.ChatCompletionMessage
   :members:
   :show-inheritance:

Assistant Context Models
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pinecone.models.assistant.context.ContextResponse
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.assistant.options.ContextOptions
   :members:
   :show-inheritance:

.. autodata:: pinecone.models.assistant.context.ContextSnippet

.. autoclass:: pinecone.models.assistant.context.TextSnippet
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.assistant.context.MultimodalSnippet
   :members:
   :show-inheritance:

.. autodata:: pinecone.models.assistant.context.ContextContentBlock

.. autoclass:: pinecone.models.assistant.context.ContextTextBlock
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.assistant.context.ContextImageBlock
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.assistant.context.ContextImageData
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.assistant.context.ContextReference
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.assistant.context.FileReference
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.assistant.context.PageReference
   :members:
   :show-inheritance:

Assistant Evaluation Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pinecone.models.assistant.evaluation.AlignmentResult
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.assistant.evaluation.AlignmentScores
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.assistant.evaluation.EntailmentResult
   :members:
   :show-inheritance:

Assistant Streaming Models
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pinecone.models.assistant.streaming.ChatStream
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.assistant.streaming.ChatStreamChunk
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.assistant.streaming.ChatCompletionStream
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.assistant.streaming.ChatCompletionStreamChunk
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.assistant.streaming.ChatCompletionStreamChoice
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.assistant.streaming.ChatCompletionStreamDelta
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.assistant.streaming.StreamContentDelta
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.assistant.streaming.StreamMessageStart
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.assistant.streaming.StreamMessageEnd
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.assistant.streaming.StreamContentChunk
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.assistant.streaming.StreamCitationChunk
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.assistant.streaming.AsyncChatStream
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.assistant.streaming.AsyncChatCompletionStream
   :members:
   :show-inheritance:

Filter Builder
--------------

:class:`~pinecone.utils.filter_builder.Field` builds metadata filter expressions
with Python operators instead of nested dicts.  Comparisons return a
:class:`~pinecone.utils.filter_builder.Condition`, which combines with ``&`` and
``|`` and converts to the wire form with ``to_dict()``.

.. autoclass:: pinecone.utils.filter_builder.Field
   :members:
   :special-members: __init__, __eq__, __ne__
   :show-inheritance:

.. autoclass:: pinecone.utils.filter_builder.Condition
   :members:
   :special-members: __init__, __and__, __or__
   :show-inheritance:
