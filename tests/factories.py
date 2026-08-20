"""Response factories that return realistic dicts matching the Pinecone API specs.

Each factory returns a dict suitable for ``httpx.Response(200, json=factory())``.
Pass ``**overrides`` to customise individual fields.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Control-plane factories
# ---------------------------------------------------------------------------


def make_index_response(**overrides: Any) -> dict[str, Any]:
    """Return a single IndexModel dict (db_control 2026-07 ``GET /indexes/{name}``)."""
    base: dict[str, Any] = {
        "name": "test-index",
        "host": "test-index-abc1234.svc.us-east1-gcp.pinecone.io",
        "deletion_protection": "disabled",
        "tags": None,
        "deployment": {
            "deployment_type": "managed",
            "cloud": "aws",
            "region": "us-east-1",
        },
        "schema": {
            "fields": {
                "embedding": {
                    "type": "dense_vector",
                    "dimension": 1536,
                    "metric": "cosine",
                },
            },
        },
        "read_capacity": {
            "mode": "OnDemand",
            "status": {"state": "Ready", "current_shards": None, "current_replicas": None},
        },
        "status": {"ready": True, "state": "Ready"},
    }
    base.update(overrides)
    return base


def make_index_list_response(**overrides: Any) -> dict[str, Any]:
    """Return an IndexList dict (db_control ``GET /indexes``)."""
    base: dict[str, Any] = {
        "indexes": [make_index_response()],
    }
    base.update(overrides)
    return base


def make_collection_response(**overrides: Any) -> dict[str, Any]:
    """Return a single CollectionModel dict (db_control ``GET /collections/{name}``)."""
    base: dict[str, Any] = {
        "name": "test-collection",
        "size": 10_000_000,
        "status": "Ready",
        "dimension": 1536,
        "vector_count": 120_000,
        "environment": "us-east1-gcp",
    }
    base.update(overrides)
    return base


def make_backup_response(**overrides: Any) -> dict[str, Any]:
    """Return a single BackupModel dict (db_control ``GET /indexes/{name}/backups/{id}``)."""
    base: dict[str, Any] = {
        "backup_id": "670e8400-e29b-41d4-a716-446655440001",
        "source_index_name": "test-index",
        "source_index_id": "670e8400-e29b-41d4-a716-446655440000",
        "name": "backup-2025-02-04",
        "description": "Backup before bulk update.",
        "status": "Ready",
        "cloud": "aws",
        "region": "us-east-1",
        "dimension": 1536,
        "metric": "cosine",
        "record_count": 120_000,
        "namespace_count": 3,
        "size_bytes": 10_000_000,
        "tags": {},
        "created_at": "2025-02-04T12:00:00Z",
    }
    base.update(overrides)
    return base


def make_restore_job_response(**overrides: Any) -> dict[str, Any]:
    """Return a single RestoreJobModel dict (db_control ``GET /restore-jobs/{id}``)."""
    base: dict[str, Any] = {
        "restore_job_id": "rj-670e8400-e29b-41d4-a716-446655440001",
        "backup_id": "670e8400-e29b-41d4-a716-446655440001",
        "target_index_name": "restored-index",
        "target_index_id": "idx-670e8400-e29b-41d4-a716-446655440002",
        "status": "Completed",
        "created_at": "2025-02-04T12:00:00Z",
        "completed_at": "2025-02-04T12:15:00Z",
        "percent_complete": 100.0,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Data-plane factories
# ---------------------------------------------------------------------------


def make_upsert_response(**overrides: Any) -> dict[str, Any]:
    """Return an UpsertResponse dict (db_data ``POST /vectors/upsert``)."""
    base: dict[str, Any] = {
        "upsertedCount": 10,
    }
    base.update(overrides)
    return base


def make_query_response(**overrides: Any) -> dict[str, Any]:
    """Return a QueryResponse dict (db_data ``POST /query``)."""
    base: dict[str, Any] = {
        "matches": [
            {
                "id": "vec-1",
                "score": 0.95,
                "values": [0.1, 0.2, 0.3],
                "metadata": {"genre": "documentary"},
            },
            {
                "id": "vec-2",
                "score": 0.80,
                "values": [0.4, 0.5, 0.6],
                "metadata": {"genre": "comedy"},
            },
        ],
        "namespace": "test-namespace",
        "usage": {"readUnits": 5},
    }
    base.update(overrides)
    return base


def make_fetch_response(**overrides: Any) -> dict[str, Any]:
    """Return a FetchResponse dict (db_data ``GET /vectors/fetch``)."""
    base: dict[str, Any] = {
        "vectors": {
            "id-1": {"id": "id-1", "values": [1.0, 1.5]},
            "id-2": {"id": "id-2", "values": [2.0, 1.0]},
        },
        "namespace": "test-namespace",
        "usage": {"readUnits": 1},
    }
    base.update(overrides)
    return base


def make_describe_index_stats_response(**overrides: Any) -> dict[str, Any]:
    """Return a DescribeIndexStatsResponse dict (db_data ``POST /describe_index_stats``)."""
    base: dict[str, Any] = {
        "namespaces": {
            "ns1": {"vectorCount": 100},
            "ns2": {"vectorCount": 200},
        },
        "dimension": 128,
        "indexFullness": 0.5,
        "totalVectorCount": 300,
    }
    base.update(overrides)
    return base


def make_namespace_description_response(**overrides: Any) -> dict[str, Any]:
    """Return a NamespaceDescription dict (db_data 2026-07 ``GET /namespaces/{ns}``)."""
    base: dict[str, Any] = {
        "name": "test-namespace",
        "record_count": 42,
        "schema": {"fields": {"genre": {"filterable": True}}},
        "indexed_fields": {"fields": ["genre", "year"]},
        "size_bytes": 1048576,
    }
    base.update(overrides)
    return base


def make_namespace_description_grpc_dict(**overrides: Any) -> dict[str, Any]:
    """Return the same namespace as :func:`make_namespace_description_response`, in
    the shape ``GrpcChannel`` hands to ``_dict_to_namespace_description``.

    The two shapes are not interchangeable: ``namespace_description_to_py_dict``
    (rust/src/transport.rs) emits ``indexed_fields`` as a bare list of names,
    where the REST JSON nests it under a ``fields`` key. Anything decoding one
    shape with the other's reader silently loses the indexed fields, so the
    parity tests compare the resulting models rather than the input dicts.
    """
    base: dict[str, Any] = {
        "name": "test-namespace",
        "record_count": 42,
        "schema": {"fields": {"genre": {"filterable": True}}},
        "indexed_fields": ["genre", "year"],
        "size_bytes": 1048576,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Inference factories
# ---------------------------------------------------------------------------


def make_embed_response(**overrides: Any) -> dict[str, Any]:
    """Return an EmbeddingsList dict (inference ``POST /embed``)."""
    base: dict[str, Any] = {
        "model": "multilingual-e5-large",
        "vector_type": "dense",
        "data": [
            {"values": [0.1, 0.2, 0.3], "vector_type": "dense"},
        ],
        "usage": {"total_tokens": 205},
    }
    base.update(overrides)
    return base


def make_rerank_response(**overrides: Any) -> dict[str, Any]:
    """Return a RerankResult dict (inference ``POST /rerank``)."""
    base: dict[str, Any] = {
        "model": "bge-reranker-v2-m3",
        "data": [
            {
                "index": 0,
                "score": 0.95,
                "document": {"id": "1", "text": "Paris is the capital of France."},
            },
            {
                "index": 1,
                "score": 0.45,
                "document": {"id": "2", "text": "Berlin is the capital of Germany."},
            },
        ],
        "usage": {"rerank_units": 1},
    }
    base.update(overrides)
    return base


def make_model_info(**overrides: Any) -> dict[str, Any]:
    """Return a ModelInfo dict (inference ``GET /models/{model_name}``)."""
    base: dict[str, Any] = {
        "model": "multilingual-e5-large",
        "short_description": "A multilingual embedding model.",
        "type": "embed",
        "vector_type": "dense",
        "default_dimension": 1024,
        "supported_dimensions": [256, 512, 1024],
        "modality": "text",
        "max_sequence_length": 512,
        "max_batch_size": 96,
        "provider_name": "Pinecone",
        "supported_metrics": ["cosine", "euclidean"],
        "supported_parameters": [
            {
                "parameter": "input_type",
                "type": "one_of",
                "value_type": "string",
                "required": True,
                "allowed_values": ["passage", "query"],
            },
            {
                "parameter": "truncate",
                "type": "one_of",
                "value_type": "string",
                "required": False,
                "allowed_values": ["END", "NONE"],
                "default": "END",
            },
        ],
    }
    base.update(overrides)
    return base


def make_model_list_response(**overrides: Any) -> dict[str, Any]:
    """Return a ModelInfoList dict (inference ``GET /models``)."""
    base: dict[str, Any] = {
        "models": [
            make_model_info(),
            make_model_info(
                model="bge-reranker-v2-m3",
                short_description="A reranking model.",
                type="rerank",
                vector_type=None,
                default_dimension=None,
                supported_dimensions=None,
                supported_metrics=None,
                max_batch_size=100,
                max_sequence_length=1024,
                supported_parameters=[
                    {
                        "parameter": "return_documents",
                        "type": "any",
                        "value_type": "boolean",
                        "required": False,
                        "default": True,
                    },
                ],
            ),
        ],
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Error factory
# ---------------------------------------------------------------------------


def make_error_response(
    status_code: int = 500,
    message: str = "Internal server error",
    **overrides: Any,
) -> dict[str, Any]:
    """Return an ErrorResponse dict matching the Pinecone error envelope."""
    code_map: dict[int, str] = {
        400: "INVALID_ARGUMENT",
        401: "UNAUTHENTICATED",
        403: "FORBIDDEN",
        404: "NOT_FOUND",
        409: "ALREADY_EXISTS",
        429: "QUOTA_EXCEEDED",
        500: "UNKNOWN",
    }
    base: dict[str, Any] = {
        "status": status_code,
        "error": {
            "code": code_map.get(status_code, "UNKNOWN"),
            "message": message,
        },
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Assistant factories
# ---------------------------------------------------------------------------


def make_assistant_response(**overrides: Any) -> dict[str, Any]:
    """Return a single Assistant dict (assistant_control ``GET /assistants/{name}``)."""
    base: dict[str, Any] = {
        "name": "test-assistant",
        "status": "Ready",
        "created_at": "2025-01-15T12:00:00Z",
        "updated_at": "2025-01-15T12:00:00Z",
        "metadata": {},
        "instructions": None,
        "host": "test-assistant-abc123.svc.pinecone.io",
        "region": "us",
    }
    base.update(overrides)
    return base


def make_assistant_file_response(**overrides: Any) -> dict[str, Any]:
    """Return a single AssistantFileModel dict (assistant_data ``POST /files/{name}``).

    The wire key for the file hash is ``crc32c_hash`` (not ``content_hash``).
    The SDK struct maps this to the Python attribute ``content_hash`` via a
    ``rename={"content_hash": "crc32c_hash"}`` directive on ``AssistantFileModel``.

    ``percent_done`` and ``error_message`` are absent: ``2026-07`` does not
    return them (see ``docs/migration/v10-2026-07-assistant-models.md``).
    """
    base: dict[str, Any] = {
        "name": "test-file.pdf",
        "id": "file-abc123",
        "metadata": None,
        "created_on": "2025-01-15T12:00:00Z",
        "updated_on": "2025-01-15T12:00:00Z",
        "status": "Available",
        "size": 1024,
        "multimodal": False,
        "signed_url": None,
        "crc32c_hash": None,
    }
    base.update(overrides)
    return base


def make_operation_response(**overrides: Any) -> dict[str, Any]:
    """Return an OperationModel dict (assistant_data ``PUT /files/{name}/{id}`` 2026-04).

    Uses actual API wire-format field names: ``id`` (not ``operation_id``),
    ``created_on`` (not ``created_at``), and ``error_message`` (not ``error``).
    OperationModel.rename maps these on decode.
    """
    base: dict[str, Any] = {
        "id": "op-abc123",
        "status": "Succeeded",
        "created_on": "2025-01-15T12:00:00Z",
        "error_message": None,
    }
    base.update(overrides)
    return base


def make_context_response(**overrides: Any) -> dict[str, Any]:
    """Return a ContextModel dict (assistant_data ``POST /chat/{name}/context``)."""
    base: dict[str, Any] = {
        "id": "ctx-abc123",
        "snippets": [
            {
                "type": "text",
                "content": "Pinecone is a vector database.",
                "score": 0.95,
                "reference": {
                    "file": make_assistant_file_response(
                        name="pinecone-overview.pdf", id="file-abc123"
                    )
                },
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10},
    }
    base.update(overrides)
    return base


def make_alignment_response(**overrides: Any) -> dict[str, Any]:
    """Return an AlignmentResponse dict (evaluation ``POST /evaluation/metrics/alignment``)."""
    base: dict[str, Any] = {
        "metrics": {
            "correctness": 0.0,
            "completeness": 1.0,
            "alignment": 0.0,
        },
        "reasoning": {
            "evaluated_facts": [
                {
                    "fact": {"content": "Madrid is the capital of Spain."},
                    "entailment": "entailed",
                    "reasoning": "The answer states Barcelona but Madrid is the capital.",
                }
            ]
        },
        "usage": {"prompt_tokens": 120, "completion_tokens": 40, "total_tokens": 160},
    }
    base.update(overrides)
    return base
