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
    "msgspec": ("https://jcristharif.com/msgspec/", None),
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
}

_MODEL_INFO_RESPONSE = {
    "model": "multilingual-e5-large",
    "short_description": "A multilingual embedding model",
    "type": "embed",
    "supported_parameters": [],
    "vector_type": "dense",
    "default_dimension": 1024,
}

_EMBED_RESPONSE = {
    "model": "multilingual-e5-large",
    "vector_type": "dense",
    "data": [{"values": [0.1, 0.2, 0.3]}],
    "usage": {"total_tokens": 5},
}

_RERANK_RESPONSE = {
    "model": "bge-reranker-v2-m3",
    "data": [{"index": 0, "score": 0.9, "document": None}],
    "usage": {"rerank_units": 1},
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

    if "/indexes" in path:
        if "/backups" in path:
            if method == "GET":
                return httpx.Response(200, content=json.dumps({"data": [], "pagination": None}).encode(),
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
            return httpx.Response(200, content=json.dumps({"collections": []}).encode(),
                                  headers={"content-type": "application/json"})
        if method == "DELETE":
            return httpx.Response(204, content=b"",
                                  headers={"content-type": "application/json"})
        return httpx.Response(200, content=json.dumps({"name": "my-collection",
            "size": 1000, "status": "Ready", "dimension": 1536,
            "vector_count": 1000, "environment": "us-east1-gcp"}).encode(),
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
        return httpx.Response(200, content=json.dumps(_EMBED_RESPONSE).encode(),
                              headers={"content-type": "application/json"})

    if "/rerank" in path:
        return httpx.Response(200, content=json.dumps(_RERANK_RESPONSE).encode(),
                              headers={"content-type": "application/json"})

    if "/models" in path:
        if path.endswith("/models"):
            return httpx.Response(200, content=json.dumps({"models": []}).encode(),
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

from pinecone import Pinecone, Admin
pc = Pinecone(api_key="test-key")
admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
"""
