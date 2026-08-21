"""Smoke test — async ``AsyncIndex`` imports lifecycle (env-var-gated).

Skipped when ``PINECONE_IMPORT_S3_URI`` is not set, because
``start_import`` requires a real S3 / GCS URI of parquet data and a
configured storage integration on the project.

Punchlist coverage (async): AsyncIndex.start_import, describe_import,
list_imports, list_imports_paginated, cancel_import.
"""

from __future__ import annotations

import os

import pytest

from pinecone import AsyncPinecone
from tests.integration.index_shapes import MANAGED_AWS, dense_schema
from tests.smoke.conftest import (
    SMOKE_PREFIX,
    async_ensure_index_deleted,
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
@pytest.mark.asyncio
async def test_imports_lifecycle_async(api_key: str) -> None:
    """Start, describe, list (both forms), and cancel an import (async)."""
    assert _IMPORT_URI is not None
    pc = AsyncPinecone(api_key=api_key)
    name = unique_name(f"{SMOKE_PREFIX}-imp-async")
    try:
        await pc.indexes.create(
            name=name,
            schema=dense_schema(DIM),
            deployment=MANAGED_AWS,
        )
        idx = await pc.index(name=name)
        try:
            start_resp = await idx.start_import(
                uri=_IMPORT_URI,
                integration_id=_INTEGRATION_ID,
            )
            import_id = start_resp.id
            desc = await idx.describe_import(import_id)
            assert desc.id == import_id

            imports_collected = [imp async for imp in idx.list_imports(limit=5)]
            assert any(imp.id == import_id for imp in imports_collected)

            page = await idx.list_imports_paginated(limit=5)
            assert page is not None

            await idx.cancel_import(import_id)
        finally:
            await idx.close()
    finally:
        await async_ensure_index_deleted(pc, name)
        await pc.close()
