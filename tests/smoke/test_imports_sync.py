"""Smoke test — sync ``Index`` imports lifecycle (env-var-gated).

Skipped when ``PINECONE_IMPORT_S3_URI`` is not set, because
``start_import`` requires a real S3 / GCS URI of parquet data and a
configured storage integration on the project.

Punchlist coverage (sync): Index.start_import, describe_import,
list_imports, list_imports_paginated, cancel_import.
"""

from __future__ import annotations

import os

import pytest

from pinecone import Pinecone
from pinecone.index import Index
from tests.integration.index_shapes import MANAGED_AWS, dense_schema
from tests.smoke.conftest import (
    SMOKE_PREFIX,
    ensure_index_deleted,
    unique_name,
)

_IMPORT_URI = os.getenv("PINECONE_IMPORT_S3_URI")
_INTEGRATION_ID = os.getenv("PINECONE_IMPORT_INTEGRATION_ID")

if not _IMPORT_URI:
    pytest.skip(
        "PINECONE_IMPORT_S3_URI not set — imports tests require a "
        "configured storage integration; opt in by setting "
        "PINECONE_IMPORT_S3_URI (and optionally "
        "PINECONE_IMPORT_INTEGRATION_ID).",
        allow_module_level=True,
    )

DIM = 8


@pytest.mark.smoke
def test_imports_lifecycle_sync(client: Pinecone) -> None:
    """Start, describe, list (both forms), and cancel an import."""
    assert _IMPORT_URI is not None
    name = unique_name(f"{SMOKE_PREFIX}-imp")
    try:
        client.indexes.create(
            name=name,
            schema=dense_schema(DIM),
            deployment=MANAGED_AWS,
        )
        raw_idx = client.index(name=name)
        assert isinstance(raw_idx, Index)
        idx: Index = raw_idx
        try:
            start_resp = idx.start_import(
                uri=_IMPORT_URI,
                integration_id=_INTEGRATION_ID,
            )
            import_id = start_resp.id
            desc = idx.describe_import(import_id)
            assert desc.id == import_id

            imports_iter = list(idx.list_imports(limit=5))
            assert any(imp.id == import_id for imp in imports_iter)

            page = idx.list_imports_paginated(limit=5)
            assert page is not None

            idx.cancel_import(import_id)
        finally:
            idx.close()
    finally:
        ensure_index_deleted(client, name)
        client.close()
