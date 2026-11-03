from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "Pinecone"
author = "Pinecone"
release = "10.3.0"

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
    "sphinx_reredirects",
    "sphinx_sitemap",
]

html_theme = "furo"
html_logo = "_static/pinecone-logo.svg"
html_favicon = "_static/favicon-32x32.png"
html_static_path = ["_static"]
html_title = "Python SDK documentation"

# Published at this URL by .github/workflows/build-and-publish-docs.yaml. Sphinx
# emits <link rel="canonical"> on every page from it and sphinx-sitemap needs it
# to write absolute URLs.
html_baseurl = "https://sdk.pinecone.io/python/"
sitemap_url_scheme = "{link}"
# Viewcode source listings and the generated index pages carry no content of
# their own; keep the sitemap to pages worth a search result.
sitemap_excludes = ["_modules/*", "search.html", "genindex.html", "py-modindex.html"]

# Pages the 10.0 restructure removed. Each becomes a stub that forwards to its
# replacement and carries autodoc anchors across, so search results, bookmarks,
# and blog links written against the flat pre-10.0 layout keep resolving.
redirect_html_template_file = "_templates/redirect.html"
redirects = {
    "rest": "reference/pinecone.html",
    "asyncio": "reference/async-pinecone.html",
    "grpc": "reference/grpc.html",
    "admin": "reference/admin.html",
    "upgrading": "migration/v10-migration.html",
    "client-configuration": "getting-started/authentication.html",
    "working-with-indexes": "how-to/indexes/serverless.html",
    "db_control/collections": "../how-to/collections.html",
    "db_control/serverless-indexes": "../how-to/indexes/serverless.html",
    "db_control/pod-indexes": "../how-to/indexes/pod.html",
    "db_control/shared-index-actions": "../how-to/indexes/serverless.html",
    "db_control/shared-index-configs": "../how-to/indexes/serverless.html",
    "db_data/index-usage-byov": "../how-to/vectors/upsert-and-query.html",
    "inference/inference-api": "../how-to/inference/embeddings.html",
    # Published with the 10.0 release candidates, removed when the preview
    # namespace graduated.
    "reference/preview": "../migration/v10-migration.html#preview-namespace-removed",
    #
    # Pages from the pdoc era (v3.0-v7.0), which named one page per module path.
    # Each goes to the page its symbol lives on now, else to the page for a
    # subject still covered, else to the index -- which means no counterpart
    # exists, rather than a guess.
    # Pre-10.0 URLs. Generated: depth from key, every target asserted
    # present in the published build. Pages whose symbol survives go to
    # that symbol's page; the rest go to index.html rather than guess.
    "maintainers/debugging": "../index.html",  # no v10 counterpart
    "maintainers/releasing": "../index.html",  # no v10 counterpart
    "maintainers/testing-guide": "../index.html",  # no v10 counterpart
    "pinecone": "index.html",  # README landing page
    "pinecone/config": "../getting-started/authentication.html",  # topic still documented
    "pinecone/config/config": "../../getting-started/authentication.html",  # topic still documented
    "pinecone/config/openapi": "../../getting-started/authentication.html",  # topic still documented
    "pinecone/config/pinecone_config": "../../getting-started/authentication.html",  # topic still documented
    "pinecone/control": "../reference/pinecone.html",  # topic still documented
    "pinecone/control/index_host_store": "../../index.html",  # no v10 counterpart
    "pinecone/control/langchain_import_warnings": "../../index.html",  # no v10 counterpart
    "pinecone/control/pinecone": "../../reference/pinecone.html",
    "pinecone/control/pinecone_asyncio": "../../reference/async-pinecone.html",
    "pinecone/control/pinecone_interface": "../../reference/pinecone.html",
    "pinecone/control/pinecone_interface_asyncio": "../../reference/async-pinecone.html",
    "pinecone/control/repr_overrides": "../../index.html",  # no v10 counterpart
    "pinecone/control/request_factory": "../../index.html",  # no v10 counterpart
    "pinecone/control/types": "../../reference/models.html",  # topic still documented
    "pinecone/control/types/create_index_for_model_embed": "../../../index.html",  # no v10 counterpart
    "pinecone/core": "../index.html",  # no v10 counterpart
    "pinecone/core/api_action": "../../index.html",  # no v10 counterpart
    "pinecone/core/api_base": "../../index.html",  # no v10 counterpart
    "pinecone/core/client": "../../index.html",  # no v10 counterpart
    "pinecone/core/client/api": "../../../index.html",  # no v10 counterpart
    "pinecone/core/client/api/index_operations_api": "../../../../index.html",  # no v10 counterpart
    "pinecone/core/client/api/manage_indexes_api": "../../../../index.html",  # no v10 counterpart
    "pinecone/core/client/api/vector_operations_api": "../../../../index.html",  # no v10 counterpart
    "pinecone/core/client/api_client": "../../../index.html",  # no v10 counterpart
    "pinecone/core/client/apis": "../../../index.html",  # no v10 counterpart
    "pinecone/core/client/configuration": "../../../getting-started/authentication.html",  # topic still documented
    "pinecone/core/client/exceptions": "../../../reference/exceptions.html",  # topic still documented
    "pinecone/core/client/model": "../../../index.html",  # no v10 counterpart
    "pinecone/core/client/model/approximated_config": "../../../../index.html",  # no v10 counterpart
    "pinecone/core/client/model/aws_regions": "../../../../index.html",  # no v10 counterpart
    "pinecone/core/client/model/collection_list": "../../../../reference/models.html",
    "pinecone/core/client/model/collection_meta": "../../../../index.html",  # no v10 counterpart
    "pinecone/core/client/model/collection_model": "../../../../reference/models.html",
    "pinecone/core/client/model/configure_index_request": "../../../../reference/models.html",
    "pinecone/core/client/model/configure_index_request_spec": "../../../../index.html",  # no v10 counterpart
    "pinecone/core/client/model/configure_index_request_spec_pod": "../../../../index.html",  # no v10 counterpart
    "pinecone/core/client/model/create_collection_request": "../../../../index.html",  # no v10 counterpart
    "pinecone/core/client/model/create_index_request": "../../../../reference/models.html",
    "pinecone/core/client/model/create_request": "../../../../index.html",  # no v10 counterpart
    "pinecone/core/client/model/delete_request": "../../../../index.html",  # no v10 counterpart
    "pinecone/core/client/model/describe_index_stats_request": "../../../../index.html",  # no v10 counterpart
    "pinecone/core/client/model/describe_index_stats_response": "../../../../reference/models.html",
    "pinecone/core/client/model/error_response": "../../../../index.html",  # no v10 counterpart
    "pinecone/core/client/model/error_response_error": "../../../../index.html",  # no v10 counterpart
    "pinecone/core/client/model/fetch_response": "../../../../reference/models.html",
    "pinecone/core/client/model/gcp_regions": "../../../../index.html",  # no v10 counterpart
    "pinecone/core/client/model/hnsw_config": "../../../../index.html",  # no v10 counterpart
    "pinecone/core/client/model/index_list": "../../../../reference/models.html",
    "pinecone/core/client/model/index_meta": "../../../../index.html",  # no v10 counterpart
    "pinecone/core/client/model/index_meta_database": "../../../../index.html",  # no v10 counterpart
    "pinecone/core/client/model/index_meta_database_status": "../../../../index.html",  # no v10 counterpart
    "pinecone/core/client/model/index_meta_status": "../../../../index.html",  # no v10 counterpart
    "pinecone/core/client/model/index_model": "../../../../reference/models.html",
    "pinecone/core/client/model/index_model_spec": "../../../../index.html",  # no v10 counterpart
    "pinecone/core/client/model/index_model_status": "../../../../index.html",  # no v10 counterpart
    "pinecone/core/client/model/namespace_summary": "../../../../reference/models.html",
    "pinecone/core/client/model/patch_request": "../../../../index.html",  # no v10 counterpart
    "pinecone/core/client/model/pod_spec": "../../../../reference/models.html",
    "pinecone/core/client/model/pod_spec_metadata_config": "../../../../index.html",  # no v10 counterpart
    "pinecone/core/client/model/protobuf_any": "../../../../index.html",  # no v10 counterpart
    "pinecone/core/client/model/protobuf_null_value": "../../../../index.html",  # no v10 counterpart
    "pinecone/core/client/model/query_request": "../../../../index.html",  # no v10 counterpart
    "pinecone/core/client/model/query_response": "../../../../reference/models.html",
    "pinecone/core/client/model/query_vector": "../../../../index.html",  # no v10 counterpart
    "pinecone/core/client/model/rpc_status": "../../../../index.html",  # no v10 counterpart
    "pinecone/core/client/model/scored_vector": "../../../../reference/models.html",
    "pinecone/core/client/model/serverless_spec": "../../../../reference/models.html",
    "pinecone/core/client/model/single_query_results": "../../../../index.html",  # no v10 counterpart
    "pinecone/core/client/model/sparse_values": "../../../../reference/models.html",
    "pinecone/core/client/model/update_request": "../../../../index.html",  # no v10 counterpart
    "pinecone/core/client/model/upsert_request": "../../../../index.html",  # no v10 counterpart
    "pinecone/core/client/model/upsert_response": "../../../../reference/models.html",
    "pinecone/core/client/model/usage": "../../../../reference/models.html",
    "pinecone/core/client/model/vector": "../../../../reference/models.html",
    "pinecone/core/client/model_utils": "../../../index.html",  # no v10 counterpart
    "pinecone/core/client/models": "../../../index.html",  # no v10 counterpart
    "pinecone/core/client/rest": "../../../index.html",  # no v10 counterpart
    "pinecone/core/exceptions": "../../reference/exceptions.html",  # topic still documented
    "pinecone/core/grpc": "../../reference/grpc.html",
    "pinecone/core/grpc/index_grpc": "../../../reference/grpc.html",
    "pinecone/core/grpc/protos": "../../../reference/grpc.html",  # topic still documented
    "pinecone/core/grpc/protos/vector_column_service_pb2": "../../../../reference/grpc.html",  # topic still documented
    "pinecone/core/grpc/protos/vector_column_service_pb2_grpc": "../../../../reference/grpc.html",  # topic still documented
    "pinecone/core/grpc/protos/vector_service_pb2": "../../../../reference/grpc.html",  # topic still documented
    "pinecone/core/grpc/protos/vector_service_pb2_grpc": "../../../../reference/grpc.html",  # topic still documented
    "pinecone/core/grpc/retry": "../../../reference/grpc.html",  # topic still documented
    "pinecone/core/utils": "../../index.html",  # no v10 counterpart
    "pinecone/core/utils/constants": "../../../index.html",  # no v10 counterpart
    "pinecone/core/utils/error_handling": "../../../guides/error-handling.html",  # topic still documented
    "pinecone/data": "../reference/sync-index.html",  # topic still documented
    "pinecone/data/dataclasses": "../../reference/models.html",  # topic still documented
    "pinecone/data/dataclasses/fetch_response": "../../../reference/models.html",
    "pinecone/data/dataclasses/search_query": "../../../reference/models.html",
    "pinecone/data/dataclasses/search_query_vector": "../../../reference/models.html",
    "pinecone/data/dataclasses/search_rerank": "../../../reference/models.html",
    "pinecone/data/dataclasses/sparse_values": "../../../reference/models.html",
    "pinecone/data/dataclasses/utils": "../../../index.html",  # no v10 counterpart
    "pinecone/data/dataclasses/vector": "../../../reference/models.html",
    "pinecone/data/errors": "../../reference/exceptions.html",  # topic still documented
    "pinecone/data/features": "../../index.html",  # no v10 counterpart
    "pinecone/data/features/bulk_import": "../../../how-to/vectors/bulk-import.html",  # topic still documented
    "pinecone/data/features/bulk_import/bulk_import": "../../../../how-to/vectors/bulk-import.html",  # topic still documented
    "pinecone/data/features/bulk_import/bulk_import_asyncio": "../../../../how-to/vectors/bulk-import.html",  # topic still documented
    "pinecone/data/features/bulk_import/bulk_import_request_factory": "../../../../how-to/vectors/bulk-import.html",  # topic still documented
    "pinecone/data/features/inference": "../../../reference/pinecone.html",
    "pinecone/data/features/inference/inference": "../../../../reference/pinecone.html",
    "pinecone/data/features/inference/inference_asyncio": "../../../../reference/async-pinecone.html",
    "pinecone/data/features/inference/inference_request_builder": "../../../../index.html",  # no v10 counterpart
    "pinecone/data/features/inference/models": "../../../../index.html",  # no v10 counterpart
    "pinecone/data/features/inference/models/embedding_list": "../../../../../index.html",  # no v10 counterpart
    "pinecone/data/features/inference/models/rerank_result": "../../../../../reference/models.html",
    "pinecone/data/features/inference/repl_overrides": "../../../../index.html",  # no v10 counterpart
    "pinecone/data/import_error": "../../reference/exceptions.html",  # topic still documented
    "pinecone/data/index": "../../reference/sync-index.html",
    "pinecone/data/index_asyncio": "../../reference/async-index.html",
    "pinecone/data/index_asyncio_interface": "../../reference/async-index.html",
    "pinecone/data/interfaces": "../../reference/sync-index.html",
    "pinecone/data/query_results_aggregator": "../../reference/models.html",
    "pinecone/data/request_factory": "../../index.html",  # no v10 counterpart
    "pinecone/data/sparse_values_factory": "../../index.html",  # no v10 counterpart
    "pinecone/data/sparse_vector_factory": "../../index.html",  # no v10 counterpart
    "pinecone/data/types": "../../reference/models.html",  # topic still documented
    "pinecone/data/types/query_filter": "../../../index.html",  # no v10 counterpart
    "pinecone/data/types/search_query_typed_dict": "../../../index.html",  # no v10 counterpart
    "pinecone/data/types/search_query_vector_typed_dict": "../../../index.html",  # no v10 counterpart
    "pinecone/data/types/search_rerank_typed_dict": "../../../index.html",  # no v10 counterpart
    "pinecone/data/types/sparse_vector_typed_dict": "../../../index.html",  # no v10 counterpart
    "pinecone/data/types/vector_metadata_dict": "../../../index.html",  # no v10 counterpart
    "pinecone/data/types/vector_tuple": "../../../index.html",  # no v10 counterpart
    "pinecone/data/types/vector_typed_dict": "../../../index.html",  # no v10 counterpart
    "pinecone/data/vector_factory": "../../index.html",  # no v10 counterpart
    "pinecone/deprecated_plugins": "../migration/v10-migration.html",  # topic still documented
    "pinecone/deprecation_warnings": "../migration/v10-migration.html",  # topic still documented
    "pinecone/enums": "../reference/models.html",  # topic still documented
    "pinecone/exceptions": "../reference/exceptions.html",  # topic still documented
    "pinecone/grpc": "../reference/grpc.html",
    "pinecone/grpc/base": "../../reference/grpc.html",  # topic still documented
    "pinecone/grpc/config": "../../getting-started/authentication.html",  # topic still documented
    "pinecone/grpc/future": "../../reference/grpc.html",  # topic still documented
    "pinecone/grpc/index_grpc": "../../reference/grpc.html",
    "pinecone/grpc/pinecone": "../../reference/pinecone.html",
    "pinecone/grpc/retry": "../../reference/grpc.html",  # topic still documented
    "pinecone/grpc/sparse_values_factory": "../../reference/grpc.html",  # topic still documented
    "pinecone/grpc/utils": "../../reference/grpc.html",  # topic still documented
    "pinecone/grpc/vector_factory_grpc": "../../reference/grpc.html",  # topic still documented
    "pinecone/index": "../reference/sync-index.html",
    "pinecone/info": "../index.html",  # no v10 counterpart
    "pinecone/manage": "../reference/pinecone.html",
    "pinecone/models": "../reference/models.html",  # topic still documented
    "pinecone/openapi_support": "../reference/pinecone.html",  # topic still documented
    "pinecone/openapi_support/api_client": "../../index.html",  # no v10 counterpart
    "pinecone/openapi_support/api_client_utils": "../../index.html",  # no v10 counterpart
    "pinecone/openapi_support/api_version": "../../index.html",  # no v10 counterpart
    "pinecone/openapi_support/asyncio_api_client": "../../index.html",  # no v10 counterpart
    "pinecone/openapi_support/asyncio_endpoint": "../../index.html",  # no v10 counterpart
    "pinecone/openapi_support/auth_util": "../../getting-started/authentication.html",  # topic still documented
    "pinecone/openapi_support/cached_class_property": "../../index.html",  # no v10 counterpart
    "pinecone/openapi_support/configuration": "../../getting-started/authentication.html",  # topic still documented
    "pinecone/openapi_support/constants": "../../index.html",  # no v10 counterpart
    "pinecone/openapi_support/deserializer": "../../index.html",  # no v10 counterpart
    "pinecone/openapi_support/endpoint": "../../index.html",  # no v10 counterpart
    "pinecone/openapi_support/endpoint_utils": "../../index.html",  # no v10 counterpart
    "pinecone/openapi_support/exceptions": "../../reference/exceptions.html",  # topic still documented
    "pinecone/openapi_support/model_utils": "../../index.html",  # no v10 counterpart
    "pinecone/openapi_support/rest_aiohttp": "../../index.html",  # no v10 counterpart
    "pinecone/openapi_support/rest_urllib3": "../../index.html",  # no v10 counterpart
    "pinecone/openapi_support/rest_utils": "../../index.html",  # no v10 counterpart
    "pinecone/openapi_support/serializer": "../../index.html",  # no v10 counterpart
    "pinecone/openapi_support/types": "../../index.html",  # no v10 counterpart
    "pinecone/utils": "../index.html",  # no v10 counterpart
    "pinecone/utils/check_kwargs": "../../index.html",  # no v10 counterpart
    "pinecone/utils/constants": "../../index.html",  # no v10 counterpart
    "pinecone/utils/convert_to_list": "../../index.html",  # no v10 counterpart
    "pinecone/utils/deprecation_notice": "../../migration/v10-migration.html",  # topic still documented
    "pinecone/utils/error_handling": "../../guides/error-handling.html",  # topic still documented
    "pinecone/utils/fix_tuple_length": "../../index.html",  # no v10 counterpart
    "pinecone/utils/normalize_host": "../../index.html",  # no v10 counterpart
    "pinecone/utils/user_agent": "../../index.html",  # no v10 counterpart
    "pinecone/utils/version": "../../index.html",  # no v10 counterpart
}

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

# Source order, not alphabetical. Members then render in the order the author
# wrote them (create/describe/list/delete rather than create/delete/describe),
# and doctests in one group run in that order too, so an example no longer sees
# a resource a later-alphabetically-named example deleted.
autodoc_member_order = "bysource"

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_param = True
napoleon_use_returns = True
napoleon_use_ivar = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "msgspec": ("https://msgspec.dev/", None),
}

nitpick_ignore = [
    # Internal HTTP plumbing and type variables that appear in signatures but
    # are not part of the public API.
    ("py:class", "HTTPClient"),
    ("py:class", "AsyncHTTPClient"),
    ("py:class", "T"),
    ("py:class", "pinecone.models.pagination.T"),
    ("py:obj", "pinecone.models.pagination.T"),
]

# Any target with a private (underscore-prefixed) segment, such as
# pinecone._internal.http_client.HTTPClient or pinecone.grpc.future._T.
nitpick_ignore_regex = [
    (r"py:(class|obj)", r"(.*\.)?_[A-Za-z]\w*(\..*)?"),
]

# Third-party names as they appear in annotations, mapped to the target the
# intersphinx inventory publishes.
_EXTERNAL_ALIASES = {
    "pd.DataFrame": "pandas.DataFrame",
    "Future": "concurrent.futures.Future",
    "concurrent.futures._base.Future": "concurrent.futures.Future",
}


def _resolve_short_reference(app, env, node, contnode):
    """Link ``PineconeValueError`` or ``pinecone.Admin`` to where autodoc documents it.

    A ``pinecone.``-qualified target follows the imported object to its defining
    module. A bare name links only when exactly one documented object has it;
    ambiguous names stay unresolved rather than guessed.
    """
    if node.get("refdomain") != "py":
        return None
    domain = env.get_domain("py")
    target = node["reftarget"]
    reftype = node["reftype"]
    fromdoc = node.get("refdoc", env.docname)

    if target in _EXTERNAL_ALIASES:
        from sphinx.ext.intersphinx import missing_reference

        node["reftarget"] = _EXTERNAL_ALIASES[target]
        return missing_reference(app, env, node, contnode)

    if target.startswith("pinecone."):
        real = _defining_path(target)
        if real is None or real == target:
            return None
        return domain.resolve_xref(env, fromdoc, app.builder, reftype, real, node, contnode)

    matches = [
        name
        for name, entry in domain.find_obj(env, None, None, target, reftype, 1)
        if not entry.aliased
    ]
    if len(matches) != 1:
        return None
    return domain.resolve_xref(env, fromdoc, app.builder, reftype, matches[0], node, contnode)


def _defining_path(target):
    import importlib

    parts = target.split(".")
    for split in range(len(parts) - 1, 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:split]))
        except ImportError:
            continue
        try:
            for attr in parts[split:]:
                obj = getattr(obj, attr)
        except AttributeError:
            return None
        owner = getattr(obj, "__module__", None)
        qualname = getattr(obj, "__qualname__", None)
        return f"{owner}.{qualname}" if owner and qualname else None
    return None


def setup(app):
    app.connect("missing-reference", _resolve_short_reference)


myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3

copybutton_prompt_text = r">>> |\.\.\. "

suppress_warnings = ["myst.header", "intersphinx", "toc.excluded", "toc.secnum"]

doctest_global_setup = """
import os
import re
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
    "read_capacity": {"mode": "OnDemand", "status": {"state": "Ready"}},
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
    "percent_complete": 100,
    "created_at": "2024-01-15T00:00:00Z",
}

_RESTORE_JOB_RESPONSE_2 = {
    "restore_job_id": "rj-def456", "backup_id": "bk-def456",
    "target_index_name": "support-tickets-restored", "target_index_id": "idx-ghi789",
    "status": "Pending", "created_at": "2024-01-16T00:00:00Z",
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
    "created_on": "2024-01-15T00:00:00Z", "completed_on": "2024-01-15T00:00:05Z",
    "ingestion_units": 1.0,
}

_ASSISTANT_CHAT_RESPONSE = {
    "id": "chat-abc123", "model": "gpt-4o-2024-11-20", "finish_reason": "stop",
    "message": {"role": "assistant",
                "content": "BYOC is available in aws us-east-1."},
    "citations": [{"position": 34, "references": [
        {"file": _ASSISTANT_FILE_RESPONSE, "pages": [3]}]}],
    "context_snippet_count": 2,
    "usage": {"prompt_tokens": 240, "completion_tokens": 18, "total_tokens": 258},
}

_ASSISTANT_CONTEXT_RESPONSE = {
    "id": "ctx-abc123",
    "snippets": [{"type": "text", "score": 0.87,
                  "content": "BYOC is available in aws us-east-1.",
                  "reference": {"type": "pdf", "pages": [3],
                                "file": _ASSISTANT_FILE_RESPONSE}}],
    "usage": {"prompt_tokens": 240, "completion_tokens": 0, "total_tokens": 240},
}

_ASSISTANT_COMPLETION_RESPONSE = {
    "id": "chatcmpl-abc123", "model": "gpt-4o-2024-11-20",
    "choices": [{"index": 0, "finish_reason": "stop",
                 "message": {"role": "assistant",
                             "content": "BYOC is available in aws us-east-1."}}],
    "usage": {"prompt_tokens": 240, "completion_tokens": 18, "total_tokens": 258},
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

_DOCUMENT_MATCHES = [
    {"_id": "article-101", "_score": 0.92, "title": "Intro to vectors",
     "content": "Vectors encode meaning as coordinates.", "category": "tech"},
    {"_id": "article-102", "_score": 0.74, "title": "Advanced retrieval",
     "content": "Hybrid retrieval mixes dense and sparse scores.", "category": "tech"},
]


def _documents_response(request):
    _path = request.url.path
    _body = json.loads(request.content or b"{}")
    if _path.endswith("/documents/upsert"):
        return {"upserted_count": len(_body.get("documents", []))}
    if _path.endswith("/documents/search"):
        _fields = _body.get("include_fields") or []
        _matches = []
        for _match in _DOCUMENT_MATCHES[: _body.get("top_k", len(_DOCUMENT_MATCHES))]:
            if _fields == ["*"]:
                _matches.append(dict(_match))
            else:
                _kept = {"_id": _match["_id"], "_score": _match["_score"]}
                _kept.update({_f: _match[_f] for _f in _fields if _f in _match})
                _matches.append(_kept)
        return {"matches": _matches, "namespace": "published", "usage": {"read_units": 1}}
    if _path.endswith("/documents/fetch"):
        _ids = _body.get("ids") or [_m["_id"] for _m in _DOCUMENT_MATCHES]
        return {
            "documents": {
                _m["_id"]: {_k: _v for _k, _v in _m.items() if _k != "_score"}
                for _m in _DOCUMENT_MATCHES
                if _m["_id"] in _ids
            },
            "namespace": "published",
            "usage": {"read_units": 1},
        }
    if _path.endswith("/documents/list"):
        _prefix = _body.get("prefix") or ""
        return {
            "documents": [
                {"_id": _m["_id"]} for _m in _DOCUMENT_MATCHES
                if _m["_id"].startswith(_prefix)
            ],
            "namespace": "published",
            "usage": {"read_units": 1},
        }
    return {} if _body.get("filter") is None else {"matched_records": 2}


_INVITE_RESPONSE = {
    "id": "9c8e3528-b9c0-4358-84ce-84c28e91b566",
    "email": "newhire@acme.com",
    "status": "pending",
    "expires_at": "2026-05-21T03:00:00Z",
    "created_at": "2026-04-14T20:00:00Z",
}

_SERVICE_ACCOUNT_RESPONSE = {
    "id": "f8a3b2c1-4d5e-6f7a-8b9c-0d1e2f3a4b5c",
    "name": "ci-prod",
    "client_id": "l3Ow0CmFyc4jOONcwiKUCRqQKN0tiCAn",
    "created_at": "2026-04-10T15:23:00Z",
    "updated_at": "2026-04-12T09:11:00Z",
}

_SERVICE_ACCOUNT_WITH_SECRET = {
    "service_account": _SERVICE_ACCOUNT_RESPONSE,
    "client_secret": "8p-kkC23XOWvkCosKq",
}

_ROLE_BINDING_RESPONSE = {
    "id": "9a8e3528-b9c0-4358-84ce-84c28e91b566",
    "principal_type": "user",
    "principal_id": "e2e92523-85dc-4142-b8c2-e681be8b78df",
    "resource_type": "organization",
    "resource_id": "4f6a1e0c-8f2b-4c1a-9d3e-1b2c3d4e5f60",
    "role": "OrgMember",
    "created_at": "2026-04-10T15:23:00Z",
}


_SEARCH_HITS = [
    {"_id": "article-1", "_score": 0.92,
     "fields": {"chunk_text": "Vector databases accelerate AI search.",
                "category": "tech"}},
    {"_id": "article-2", "_score": 0.74,
     "fields": {"chunk_text": "RAG pipelines combine retrieval with generation.",
                "category": "tech"}},
]


def _records_search_response(request):
    _body = json.loads(request.content or b"{}")
    _query = _body.get("query") or {}
    _rerank = _body.get("rerank") or {}
    _limit = _rerank.get("top_n") or _query.get("top_k") or len(_SEARCH_HITS)
    _usage = {"read_units": 1}
    if _query.get("inputs"):
        _usage["embed_total_tokens"] = 8
    if _rerank:
        _usage["rerank_units"] = 1
    return {"result": {"hits": _SEARCH_HITS[:_limit]}, "usage": _usage}


def _vectors_query_response(request):
    _body = json.loads(request.content or b"{}")
    _matches = [
        {"id": "prod-%d" % i, "score": 0.9 - i / 10,
         "metadata": {"description": "Noise-cancelling over-ear headphones %d" % i}}
        for i in range(min(_body.get("topK") or 2, 2))
    ]
    if not _body.get("includeMetadata"):
        for _match in _matches:
            _match.pop("metadata")
    return {"matches": _matches, "namespace": _body.get("namespace", ""),
            "usage": {"readUnits": 1}}


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

    if "/admin/invites" in path:
        if method == "DELETE":
            return httpx.Response(204, content=b"",
                                  headers={"content-type": "application/json"})
        if method == "GET" and path.rstrip("/").endswith("/admin/invites"):
            return httpx.Response(200, content=json.dumps(
                {"data": [_INVITE_RESPONSE], "pagination": None}).encode(),
                                  headers={"content-type": "application/json"})
        return httpx.Response(200, content=json.dumps(_INVITE_RESPONSE).encode(),
                              headers={"content-type": "application/json"})

    if "/admin/service-accounts" in path:
        if method == "DELETE":
            return httpx.Response(204, content=b"",
                                  headers={"content-type": "application/json"})
        if path.endswith("/rotate-secret"):
            return httpx.Response(200,
                                  content=json.dumps(_SERVICE_ACCOUNT_WITH_SECRET).encode(),
                                  headers={"content-type": "application/json"})
        if method == "GET" and path.rstrip("/").endswith("/admin/service-accounts"):
            return httpx.Response(200, content=json.dumps(
                {"data": [_SERVICE_ACCOUNT_RESPONSE], "pagination": None}).encode(),
                                  headers={"content-type": "application/json"})
        if method == "POST":
            return httpx.Response(201,
                                  content=json.dumps(_SERVICE_ACCOUNT_WITH_SECRET).encode(),
                                  headers={"content-type": "application/json"})
        return httpx.Response(200, content=json.dumps(_SERVICE_ACCOUNT_RESPONSE).encode(),
                              headers={"content-type": "application/json"})

    if "/admin/role-bindings" in path:
        if method == "DELETE":
            return httpx.Response(204, content=b"",
                                  headers={"content-type": "application/json"})
        if method == "GET" and path.rstrip("/").endswith("/admin/role-bindings"):
            return httpx.Response(200, content=json.dumps(
                {"data": [_ROLE_BINDING_RESPONSE], "pagination": None}).encode(),
                                  headers={"content-type": "application/json"})
        return httpx.Response(200, content=json.dumps(_ROLE_BINDING_RESPONSE).encode(),
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
        if path.endswith("/indexes/legacy-recommender"):
            return httpx.Response(200, content=json.dumps({
                **_INDEX_RESPONSE, "name": "legacy-recommender",
                "deployment": {"deployment_type": "pod", "environment": "us-east-1-aws",
                               "pod_type": "p1.x1", "replicas": 2, "shards": 1},
            }).encode(), headers={"content-type": "application/json"})
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
            if request.url.params.get("paginationToken") is None:
                _body = {"data": [_RESTORE_JOB_RESPONSE],
                         "pagination": {"next": "rj-page-2"}}
            else:
                _body = {"data": [_RESTORE_JOB_RESPONSE_2], "pagination": None}
            return httpx.Response(200, content=json.dumps(_body).encode(),
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

    # Prefix-ordered: the two specific paths first. Real paths are /chat/{name},
    # /chat/{name}/context and /chat/{name}/chat/completions.
    if "/chat/" in path:
        if path.endswith("/context"):
            return httpx.Response(200,
                content=json.dumps(_ASSISTANT_CONTEXT_RESPONSE).encode(),
                headers={"content-type": "application/json"})
        if path.endswith("/chat/completions"):
            return httpx.Response(200,
                content=json.dumps(_ASSISTANT_COMPLETION_RESPONSE).encode(),
                headers={"content-type": "application/json"})
        return httpx.Response(200,
            content=json.dumps(_ASSISTANT_CHAT_RESPONSE).encode(),
            headers={"content-type": "application/json"})

    # Above the /assistant/* branches: list paths are /files/{name} and
    # /operations/{name}, which those prefixes do not match, so a list decode
    # would otherwise fail for want of its "files"/"operations" key.
    if re.match(r"^/files/[^/]+$", path):
        return httpx.Response(200, content=json.dumps(
            {"files": [_ASSISTANT_FILE_RESPONSE], "pagination": None}).encode(),
            headers={"content-type": "application/json"})

    if re.match(r"^/operations/[^/]+$", path):
        return httpx.Response(200, content=json.dumps(
            {"operations": [_ASSISTANT_OPERATION_RESPONSE], "pagination": None}).encode(),
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
        return httpx.Response(200, content=json.dumps({
            "id": "1", "status": "InProgress",
            "uri": "s3://acme-exports/articles/",
            "createdAt": "2026-01-15T00:00:00Z",
            "percentComplete": 50.0, "recordsImported": 12000}).encode(),
                              headers={"content-type": "application/json"})

    # Above the generic data-plane branch: its "/upsert" and "/fetch" substring
    # tests would otherwise shadow /namespaces/{ns}/documents/upsert and .../fetch.
    if "/documents/" in path:
        return httpx.Response(
            200, content=json.dumps(_documents_response(request)).encode(),
            headers={"content-type": "application/json"})

    if "/records/namespaces/" in path and path.endswith("/search"):
        return httpx.Response(
            200, content=json.dumps(_records_search_response(request)).encode(),
            headers={"content-type": "application/json"})

    if path.endswith("/query"):
        return httpx.Response(
            200, content=json.dumps(_vectors_query_response(request)).encode(),
            headers={"content-type": "application/json"})

    if path.endswith("/vectors/upsert"):
        _upserted = len(json.loads(request.content or b"{}").get("vectors", []))
        return httpx.Response(
            200, content=json.dumps({"upsertedCount": _upserted}).encode(),
            headers={"content-type": "application/json"})

    # Below /documents/ (those paths also end in /upsert) and above the generic
    # data-plane branch. /records/.../upsert stays generic: upsert_records builds
    # its own response locally.
    if "/upsert" in path and "/records/" not in path:
        _n = len(json.loads(request.content or b"{}").get("vectors", []) or [])
        return httpx.Response(200, content=json.dumps({"upsertedCount": _n}).encode(),
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
