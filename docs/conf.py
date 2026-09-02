from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "Pinecone"
author = "Pinecone"
release = "9.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.doctest",
    "sphinx.ext.coverage",
    "sphinx_copybutton",
    "sphinx_tabs.tabs",
    "myst_parser",
]

html_theme = "furo"
html_logo = "_static/pinecone-logo.svg"
html_favicon = "_static/favicon-32x32.png"
html_static_path = ["_static"]
html_title = "Python SDK documentation"

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "pinecone/db_data.py",
    "pinecone/db_data/**",
    "pinecone/db_control/**",
    "pinecone/admin/resources/**",
    "pinecone/config/**",
    "pinecone/utils/response_info.py",
    "pinecone/exceptions.py",
    "README.md",
]

autodoc_mock_imports = ["pinecone._grpc", "pandas"]

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "special-members": "__init__",
}

autodoc_typehints = "description"

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_param = True
napoleon_use_returns = True
napoleon_use_ivar = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

nitpick_ignore = [
    # Private base classes intentionally hidden from public docs
    ("py:class", "pinecone.models._mixin.StructDictMixin"),
    ("py:class", "pinecone.models._mixin.DictLikeStruct"),
    ("py:class", "pinecone._internal.config.PineconeConfig"),
    ("py:class", "pinecone._internal.config.RetryConfig"),
    ("py:class", "pinecone._internal.http_client.HTTPClient"),
    ("py:class", "pinecone._internal.http_client.AsyncHTTPClient"),
    ("py:class", "pinecone.client._assistant_namespace_proxy._AssistantNamespaceProxy"),
]

myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3

copybutton_prompt_text = r">>> |\.\.\. "

suppress_warnings = ["myst.header", "intersphinx", "toc.excluded", "toc.secnum"]

doctest_global_setup = """
import os
import sys
import json
import httpx
from unittest.mock import patch

os.environ.setdefault("PINECONE_API_KEY", "test-key")

# Mock pandas — not installed in the docs environment
import types
_pandas = types.ModuleType("pandas")
class _DataFrame:
    def __init__(self, data=None, **kw):
        self._data = data or []
    def __repr__(self):
        return "DataFrame(...)"
_pandas.DataFrame = _DataFrame
sys.modules["pandas"] = _pandas
sys.modules["pandas.core"] = types.ModuleType("pandas.core")
sys.modules["pandas.core.frame"] = types.ModuleType("pandas.core.frame")

_INDEX_RESPONSE = {
    "name": "my-index",
    "dimension": 1536,
    "metric": "cosine",
    "host": "my-index-abc123.svc.pinecone.io",
    "deletion_protection": "disabled",
    "tags": {},
    "spec": {
        "serverless": {
            "cloud": "aws",
            "region": "us-east-1",
            "read_capacity": {"mode": "OnDemand", "status": {"state": "Ready"}},
        }
    },
    "status": {"ready": True, "state": "Ready"},
    "vector_type": "dense",
    # Extra fields for PreviewIndexModel (ignored by regular IndexModel)
    "schema": {"fields": {}},
    "deployment": {
        "deployment_type": "managed",
        "environment": "us-east-1-aws",
        "cloud": "aws",
        "region": "us-east-1",
    },
}

_SEMANTIC_INDEX_RESPONSE = {
    **_INDEX_RESPONSE,
    "name": "semantic-search",
    "host": "semantic-search-abc123.svc.pinecone.io",
    "schema": {"fields": {"chunk_text": {"type": "semantic_text",
                                         "model": "multilingual-e5-large"}}},
}

_ORG_RESPONSE = {
    "id": "org-abc123",
    "name": "Acme Corp",
    "plan": "Standard",
    "payment_status": "Active",
    "created_at": "2024-01-01T00:00:00Z",
    "support_tier": "Standard",
}

_PROJECT_RESPONSE = {
    "id": "proj-abc123",
    "name": "my-project",
    "max_pods": 10,
    "force_encryption_with_cmek": False,
    "organization_id": "org-abc123",
    "created_at": "2024-01-01T00:00:00Z",
}

_API_KEY_MODEL = {
    "id": "key-abc123",
    "name": "prod-search-key",
    "project_id": "proj-abc123",
    "roles": ["DataPlaneEditor"],
    "description": "Used by the search service",
}

_API_KEY_WITH_SECRET = {
    "key": _API_KEY_MODEL,
    "value": "pcsk_abc123_secretvalue",
}

_BACKUP_RESPONSE = {
    "backup_id": "bk-abc123",
    "source_index_name": "product-search",
    "source_index_id": "idx-abc123",
    "status": "Ready",
    "cloud": "aws",
    "region": "us-east-1",
    "name": "daily-20240115",
    "created_at": "2024-01-15T00:00:00Z",
}

_BACKUP_RESPONSE_2 = {
    "backup_id": "bk-def456", "source_index_name": "product-search",
    "source_index_id": "idx-abc123", "status": "Ready", "cloud": "aws",
    "region": "us-east-1", "name": "daily-20240116",
    "created_at": "2024-01-16T00:00:00Z",
}

_BACKUP_RESPONSE_3 = {
    "backup_id": "bk-ghi789", "source_index_name": "support-tickets",
    "source_index_id": "idx-def456", "status": "Ready", "cloud": "aws",
    "region": "us-east-1", "name": "pre-migration",
    "created_at": "2024-01-17T00:00:00Z",
}

_ORPHANED_BACKUP_RESPONSE = {
    "backup_id": "bk-old111", "source_index_name": "legacy-catalog",
    "source_index_id": "idx-legacy", "status": "Ready", "cloud": "aws",
    "region": "us-east-1", "name": "final-snapshot",
    "created_at": "2023-11-01T00:00:00Z",
    "source_index_deleted_at": "2024-01-02T00:00:00Z",
}

_RESTORE_JOB_RESPONSE = {
    "restore_job_id": "rj-abc123",
    "backup_id": "bkp-abc123",
    "target_index_name": "product-search-restored",
    "target_index_id": "idx-def456",
    "status": "Completed",
    "created_at": "2024-01-15T00:00:00Z",
}

_ASSISTANT_RESPONSE = {
    "name": "acme-support-bot",
    "status": "Ready",
    "created_at": "2024-01-01T00:00:00Z",
    "host": "https://acme-support-bot-abc123.svc.pinecone.io",
}

_ASSISTANT_FILE_RESPONSE = {
    "name": "q3-revenue-review.pdf", "id": "file-abc123", "status": "Available",
    "created_on": "2024-01-15T00:00:00Z", "size": 20480,
    "metadata": {"department": "finance", "quarter": "2024-Q3"},
}

_ASSISTANT_OPERATION_RESPONSE = {
    "id": "op-1234-abcd-5678", "status": "Completed", "operation_type": "upload_file",
    "file_id": "file-abc123", "percent_complete": 100,
    "created_at": "2024-01-15T00:00:00Z", "completed_on": "2024-01-15T00:00:05Z",
    "ingestion_units": 1.0,
}

_ALIGNMENT_RESPONSE = {
    "metrics": {"correctness": 0.0, "completeness": 0.0, "alignment": 0.0},
    "reasoning": {"evaluated_facts": [{
        "fact": {"content": "The capital of Spain is Madrid."},
        "entailment": "contradicted",
        "reasoning": "The answer names Barcelona instead of Madrid.",
    }]},
    "usage": {"prompt_tokens": 26, "completion_tokens": 12, "total_tokens": 38},
}

_MODEL_INFO_RESPONSE = {
    "model": "multilingual-e5-large",
    "short_description": "A multilingual embedding model",
    "type": "embed",
    "supported_parameters": [
        {"parameter": "input_type", "type": "one_of", "value_type": "string",
         "required": False, "allowed_values": ["query", "passage"]},
        {"parameter": "truncate", "type": "one_of", "value_type": "string",
         "required": False, "allowed_values": ["END", "NONE", "START"], "default": "END"},
        {"parameter": "dimension", "type": "one_of", "value_type": "integer",
         "required": False, "allowed_values": [1024]},
    ],
    "vector_type": "dense",
    "default_dimension": 1024,
}

_COLLECTION_RESPONSE = {
    "name": "movie-embeddings-snapshot",
    "status": "Ready",
    "environment": "us-east1-gcp",
    "size": 3126700,
    "dimension": 1024,
    "vector_count": 99,
}

_MODELS = [
    {"model": "multilingual-e5-large", "short_description": "A multilingual embedding model",
     "type": "embed", "supported_parameters": [], "vector_type": "dense",
     "default_dimension": 1024},
    {"model": "pinecone-sparse-english-v0", "short_description": "A sparse embedding model",
     "type": "embed", "supported_parameters": [], "vector_type": "sparse"},
    {"model": "bge-reranker-v2-m3", "short_description": "A reranking model",
     "type": "rerank", "supported_parameters": []},
]


def _embed_response(request):
    _body = json.loads(request.content or b"{}")
    _n = max(1, len(_body.get("inputs", []) or []))
    return {
        "model": _body.get("model", "multilingual-e5-large"),
        "vector_type": "dense",
        "data": [{"values": [0.1, 0.2, 0.3]} for _ in range(_n)],
        "usage": {"total_tokens": 5 * _n},
    }


def _rerank_response(request):
    # A static fixture pins index 0, hiding the reordering rerank exists to do.
    documents = json.loads(request.content or b"{}").get("documents", [])
    if not documents:
        return {"model": "bge-reranker-v2-m3", "data": [], "usage": {"rerank_units": 1}}
    return {
        "model": "bge-reranker-v2-m3",
        "data": [{"index": len(documents) - 1, "score": 0.95, "document": documents[-1]}],
        "usage": {"rerank_units": 1},
    }

_BACKUP_SCHEDULE_RESPONSE = {
    "schedule_id": "e88f7273-42aa-47e9-af73-593827136867",
    "name": "compliance-snapshots",
    "index_id": "8cbf7ba6-4135-438e-a3c3-4a89a3298905",
    "project_id": "71ce31ea-75f7-45d6-a147-ef67f661a1b0",
    "schedule_type": "time-based",
    "frequency": "daily",
    "retention_expire_after_days": 90,
    "enabled": True,
    "next_scheduled_run": "2026-04-03T06:00:00Z",
    "created_at": "2026-04-02T18:22:56Z",
}

_BACKUP_SCHEDULE_HISTORY_SCHEDULED = {
    "backup_id": "b2c3d4e5-f6a7-8901-bcde-f12345678901",
    "source_index_id": "8cbf7ba6-4135-438e-a3c3-4a89a3298905",
    "source_index_name": "product-search",
    "status": "Scheduled",
    "cloud": "aws",
    "region": "us-east-1",
    "created_at": "2026-04-02T18:22:56Z",
    "scheduled_execution_at": "2026-04-03T06:00:00Z",
    "name": "compliance-snapshots-20260403T060000Z",
    "record_count": 0,
    "namespace_count": 1,
    "size_bytes": 0,
}

_BACKUP_SCHEDULE_HISTORY_DONE = {
    "backup_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "source_index_id": "8cbf7ba6-4135-438e-a3c3-4a89a3298905",
    "source_index_name": "product-search",
    "status": "Ready",
    "cloud": "aws",
    "region": "us-east-1",
    "created_at": "2026-04-01T06:00:00Z",
    "scheduled_execution_at": None,
    "name": "compliance-snapshots-20260401T060000Z",
    "record_count": 12000,
    "namespace_count": 3,
    "size_bytes": 4194304,
}

_USER_RESPONSE = {
    "id": "e2e92523-85dc-4142-b8c2-e681be8b78df",
    "email": "alice@example.com",
    "name": "Alice Nakamura",
}

def _route_request(request):
    url = str(request.url)
    path = request.url.path
    method = request.method

    if "oauth/token" in url:
        body = json.dumps({"access_token": "mock-token", "token_type": "Bearer"}).encode()
        return httpx.Response(200, content=body, headers={"content-type": "application/json"})

    if "/admin/organizations" in path:
        if method == "GET" and path.endswith("/admin/organizations"):
            return httpx.Response(200, content=json.dumps({"data": []}).encode(),
                                  headers={"content-type": "application/json"})
        if method == "DELETE":
            return httpx.Response(204, content=b"",
                                  headers={"content-type": "application/json"})
        return httpx.Response(200, content=json.dumps(_ORG_RESPONSE).encode(),
                              headers={"content-type": "application/json"})

    # Check /api-keys BEFORE /admin/projects: create/list routes go through
    # /admin/projects/{id}/api-keys which would otherwise be caught by the projects block.
    if "/admin/api-keys" in path or "/api-keys" in path:
        if method == "GET" and "api-keys" in path and not path.split("api-keys")[-1].lstrip("/"):
            return httpx.Response(200, content=json.dumps({"data": []}).encode(),
                                  headers={"content-type": "application/json"})
        if method == "GET":
            return httpx.Response(200, content=json.dumps(_API_KEY_MODEL).encode(),
                                  headers={"content-type": "application/json"})
        if method == "PATCH":
            return httpx.Response(200, content=json.dumps(_API_KEY_MODEL).encode(),
                                  headers={"content-type": "application/json"})
        if method == "DELETE":
            return httpx.Response(204, content=b"",
                                  headers={"content-type": "application/json"})
        return httpx.Response(201, content=json.dumps(_API_KEY_WITH_SECRET).encode(),
                              headers={"content-type": "application/json"})

    if "/admin/users" in path:
        if method == "DELETE":
            return httpx.Response(204, content=b"",
                                  headers={"content-type": "application/json"})
        if method == "GET" and path.rstrip("/").endswith("/admin/users"):
            return httpx.Response(200, content=json.dumps(
                {"data": [_USER_RESPONSE], "pagination": None}).encode(),
                                  headers={"content-type": "application/json"})
        return httpx.Response(200, content=json.dumps(_USER_RESPONSE).encode(),
                              headers={"content-type": "application/json"})

    if "/admin/projects" in path:
        if "delete_with_cleanup" in path:
            return httpx.Response(204, content=b"",
                                  headers={"content-type": "application/json"})
        if method == "GET" and path.endswith("/admin/projects"):
            return httpx.Response(200, content=json.dumps({"data": []}).encode(),
                                  headers={"content-type": "application/json"})
        if method == "DELETE":
            return httpx.Response(204, content=b"",
                                  headers={"content-type": "application/json"})
        return httpx.Response(200, content=json.dumps(_PROJECT_RESPONSE).encode(),
                              headers={"content-type": "application/json"})

    if "delete_with_cleanup" in path:
        return httpx.Response(204, content=b"",
                              headers={"content-type": "application/json"})

    if "/backup-schedules" in path:
        if method == "DELETE":
            return httpx.Response(204, content=b"",
                                  headers={"content-type": "application/json"})
        if method == "PATCH":
            _patch_body = json.loads(request.read() or b"{}")
            _sched = dict(_BACKUP_SCHEDULE_RESPONSE)
            if "frequency" in _patch_body:
                _sched["frequency"] = _patch_body["frequency"]
            if "retention" in _patch_body:
                _sched["retention_expire_after_days"] = (
                    _patch_body["retention"]["expire_after_days"])
            if _patch_body.get("enabled") is False:
                _sched["enabled"] = False
                _sched["next_scheduled_run"] = None
            return httpx.Response(200, content=json.dumps(_sched).encode(),
                                  headers={"content-type": "application/json"})
        if path.endswith("/history"):
            if request.url.params.get("paginationToken"):
                _page = {"data": [_BACKUP_SCHEDULE_HISTORY_DONE], "pagination": None}
            else:
                _page = {"data": [_BACKUP_SCHEDULE_HISTORY_SCHEDULED],
                         "pagination": {"next": "history-page-2"}}
            return httpx.Response(200, content=json.dumps(_page).encode(),
                                  headers={"content-type": "application/json"})
        if method == "GET" and path.endswith("/backup-schedules"):
            return httpx.Response(200, content=json.dumps(
                {"data": [_BACKUP_SCHEDULE_RESPONSE], "pagination": None}).encode(),
                                  headers={"content-type": "application/json"})
        if method == "POST":
            return httpx.Response(201,
                                  content=json.dumps(_BACKUP_SCHEDULE_RESPONSE).encode(),
                                  headers={"content-type": "application/json"})
        return httpx.Response(200, content=json.dumps(_BACKUP_SCHEDULE_RESPONSE).encode(),
                              headers={"content-type": "application/json"})

    if "/indexes" in path:
        if "create-for-model" in path or path.endswith("/indexes/semantic-search"):
            return httpx.Response(200, content=json.dumps(_SEMANTIC_INDEX_RESPONSE).encode(),
                                  headers={"content-type": "application/json"})
        if "/backups" in path:
            if method == "GET":
                if "/indexes/legacy-catalog/" in path:
                    _rows = [_ORPHANED_BACKUP_RESPONSE]
                else:
                    _rows = [_BACKUP_RESPONSE]
                return httpx.Response(200, content=json.dumps(
                    {"data": _rows, "pagination": None}).encode(),
                                      headers={"content-type": "application/json"})
            # POST create-backup returns a BackupModel
            return httpx.Response(202, content=json.dumps(_BACKUP_RESPONSE).encode(),
                                  headers={"content-type": "application/json"})
        if "create-index" in path:
            return httpx.Response(202, content=json.dumps({
                "restore_job_id": "rj-123", "index_id": "idx-123"}).encode(),
                                  headers={"content-type": "application/json"})
        if method in ("GET", "HEAD") and not path.split("/indexes")[-1].lstrip("/"):
            return httpx.Response(200, content=json.dumps({"indexes": []}).encode(),
                                  headers={"content-type": "application/json"})
        if method == "DELETE":
            return httpx.Response(204, content=b"",
                                  headers={"content-type": "application/json"})
        return httpx.Response(200, content=json.dumps(_INDEX_RESPONSE).encode(),
                              headers={"content-type": "application/json"})

    if "/backups" in path:
        if "create-index" in path:
            return httpx.Response(202, content=json.dumps(
                {"restore_job_id": "rj-abc123", "index_id": "idx-abc123"}).encode(),
                                  headers={"content-type": "application/json"})
        if method == "DELETE":
            return httpx.Response(204, content=b"",
                                  headers={"content-type": "application/json"})
        if method == "GET" and path.rstrip("/").endswith("/backups"):
            if request.url.params.get("paginationToken") is None:
                _body = {"data": [_BACKUP_RESPONSE, _BACKUP_RESPONSE_2],
                         "pagination": {"next": "bk-page-2"}}
            else:
                _body = {"data": [_BACKUP_RESPONSE_3], "pagination": None}
            return httpx.Response(200, content=json.dumps(_body).encode(),
                                  headers={"content-type": "application/json"})
        return httpx.Response(200, content=json.dumps(_BACKUP_RESPONSE).encode(),
                              headers={"content-type": "application/json"})

    if "/restore-jobs" in path:
        if method == "GET" and path.endswith("/restore-jobs"):
            return httpx.Response(200, content=json.dumps({"data": [], "pagination": None}).encode(),
                                  headers={"content-type": "application/json"})
        return httpx.Response(200, content=json.dumps(_RESTORE_JOB_RESPONSE).encode(),
                              headers={"content-type": "application/json"})

    if "/collections" in path:
        if method == "GET" and path.endswith("/collections"):
            return httpx.Response(200, content=json.dumps({"collections": [
                _COLLECTION_RESPONSE,
                {"name": "product-catalog-snapshot", "status": "Initializing",
                 "environment": "us-east1-gcp"}]}).encode(),
                                  headers={"content-type": "application/json"})
        if method == "DELETE":
            return httpx.Response(204, content=b"",
                                  headers={"content-type": "application/json"})
        if method == "POST":
            return httpx.Response(201, content=json.dumps(
                {"name": "movie-embeddings-snapshot", "status": "Initializing",
                 "environment": "us-east1-gcp"}).encode(),
                                  headers={"content-type": "application/json"})
        return httpx.Response(200, content=json.dumps(_COLLECTION_RESPONSE).encode(),
                              headers={"content-type": "application/json"})

    if "/assistant/evaluation/metrics/alignment" in path:
        return httpx.Response(200, content=json.dumps(_ALIGNMENT_RESPONSE).encode(),
                              headers={"content-type": "application/json"})

    if "/assistant/operations/" in path:
        return httpx.Response(200, content=json.dumps(_ASSISTANT_OPERATION_RESPONSE).encode(),
                              headers={"content-type": "application/json"})

    if "/assistant/files/" in path:
        if method == "DELETE":
            return httpx.Response(204, content=b"",
                                  headers={"content-type": "application/json"})
        if method in ("POST", "PUT"):
            return httpx.Response(200, content=json.dumps(_ASSISTANT_OPERATION_RESPONSE).encode(),
                                  headers={"content-type": "application/json"})
        return httpx.Response(200, content=json.dumps(_ASSISTANT_FILE_RESPONSE).encode(),
                              headers={"content-type": "application/json"})

    if "/assistants" in path:
        if method == "GET" and path.endswith("/assistants"):
            return httpx.Response(200, content=json.dumps({"assistants": []}).encode(),
                                  headers={"content-type": "application/json"})
        if method == "DELETE":
            return httpx.Response(204, content=b"",
                                  headers={"content-type": "application/json"})
        return httpx.Response(200, content=json.dumps(_ASSISTANT_RESPONSE).encode(),
                              headers={"content-type": "application/json"})

    if "/embed" in path:
        return httpx.Response(200, content=json.dumps(_embed_response(request)).encode(),
                              headers={"content-type": "application/json"})

    if "/rerank" in path:
        return httpx.Response(200, content=json.dumps(_rerank_response(request)).encode(),
                              headers={"content-type": "application/json"})

    if "/models" in path:
        if path.endswith("/models"):
            wanted_type = request.url.params.get("type")
            wanted_vector_type = request.url.params.get("vector_type")
            models = [
                m for m in _MODELS
                if (wanted_type is None or m["type"] == wanted_type)
                and (wanted_vector_type is None or m.get("vector_type") == wanted_vector_type)
            ]
            return httpx.Response(200, content=json.dumps({"models": models}).encode(),
                                  headers={"content-type": "application/json"})
        # Single model describe
        return httpx.Response(200, content=json.dumps(_MODEL_INFO_RESPONSE).encode(),
                              headers={"content-type": "application/json"})

    if "/bulk/imports" in path:
        if method == "GET" and path.endswith("/imports"):
            return httpx.Response(200, content=json.dumps({"data": [], "pagination": None}).encode(),
                                  headers={"content-type": "application/json"})
        if method == "POST":
            return httpx.Response(200, content=json.dumps({"id": "1", "status": "InProgress"}).encode(),
                                  headers={"content-type": "application/json"})
        return httpx.Response(200, content=json.dumps({"id": "1", "status": "InProgress",
            "percent_complete": 50.0, "records_imported": 0}).encode(),
                              headers={"content-type": "application/json"})

    if "/vectors" in path or "/query" in path or "/upsert" in path or "/fetch" in path:
        return httpx.Response(200, content=json.dumps({}).encode(),
                              headers={"content-type": "application/json"})

    return httpx.Response(200, content=b"{}",
                          headers={"content-type": "application/json"})

def _mock_send(self, request, **kw):
    return _route_request(request)

def _mock_handle_request(self, request):
    return _route_request(request)

_sync_patcher = patch.object(httpx.Client, "send", _mock_send)
_sync_patcher.start()

_transport_patcher = patch.object(httpx.HTTPTransport, "handle_request", _mock_handle_request)
_transport_patcher.start()

import time
_sleep_patcher = patch.object(time, "sleep", lambda seconds: None)
_sleep_patcher.start()

from pinecone import Pinecone, Admin
pc = Pinecone(api_key="test-key")
admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
"""
