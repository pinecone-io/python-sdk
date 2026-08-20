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

.. autoclass:: pinecone.models.vectors.sparse.SparseValues
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

.. autoclass:: pinecone.models.vectors.responses.DescribeIndexStatsResponse
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

.. autoclass:: pinecone.models.vectors.query_aggregator.QueryNamespacesResults
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.vectors.query_aggregator.QueryResultsAggregator
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

.. autoclass:: pinecone.models.inference.embed.EmbeddingsList
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.inference.rerank.RerankResult
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.inference.rerank.RankedDocument
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.inference.models.ModelInfo
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

.. autoclass:: pinecone.db_control.models.collection_description.CollectionDescription
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

Namespace Models
----------------

.. autoclass:: pinecone.models.namespaces.models.NamespaceDescription
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.namespaces.models.ListNamespacesResponse
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

.. autoclass:: pinecone.db_control.enums.clouds.AwsRegion
   :members:
   :show-inheritance:

.. autoclass:: pinecone.db_control.enums.clouds.AzureRegion
   :members:
   :show-inheritance:

.. autoclass:: pinecone.db_control.enums.clouds.GcpRegion
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

.. autoclass:: pinecone.models.assistant.chat.ChatResponse
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.assistant.chat.ChatCompletionResponse
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.assistant.context.ContextResponse
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.assistant.options.ContextOptions
   :members:
   :show-inheritance:

.. autoclass:: pinecone.models.assistant.evaluation.AlignmentResult
   :members:
   :show-inheritance:

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

.. autoclass:: pinecone.utils.filter_builder.Field
   :members:
   :show-inheritance:

.. autoclass:: pinecone.utils.filter_builder.FilterBuilder
   :members:
   :show-inheritance:
