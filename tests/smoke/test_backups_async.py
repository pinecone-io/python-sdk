"""Smoke test — async backup lifecycle against ``AsyncPinecone.backups``.

Gated by ``SMOKE_RUN_BACKUPS=1`` because backups require a configured
data-integration on the project. When the env var is unset, the
whole module is skipped.

Punchlist coverage (async): AsyncBackups.create, list, describe, get, delete.
"""

from __future__ import annotations

import asyncio
import os
import time

import pytest

from pinecone import AsyncPinecone, NotFoundError
from tests.integration.index_shapes import MANAGED_AWS, dense_schema
from tests.smoke.conftest import (
    SMOKE_PREFIX,
    async_ensure_index_deleted,
    unique_name,
)

if os.getenv("SMOKE_RUN_BACKUPS") != "1":
    pytest.skip(
        "SMOKE_RUN_BACKUPS=1 not set — backup tests require a configured "
        "data integration; opt in with SMOKE_RUN_BACKUPS=1 to run.",
        allow_module_level=True,
    )

DIM = 8


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_backups_smoke_async(api_key: str) -> None:
    """End-to-end backup walkthrough on a fresh managed index (async)."""
    pc = AsyncPinecone(api_key=api_key)
    index_name = unique_name(f"{SMOKE_PREFIX}-bkp-async")
    backup_id: str | None = None

    try:
        # Nothing below touches the vectors API, so a 2026-07 document-schema
        # index is a legitimate subject here (#322).
        await pc.indexes.create(
            name=index_name,
            schema=dense_schema(DIM),
            deployment=MANAGED_AWS,
        )

        # ----- backups.create -----
        backup = await pc.backups.create(
            index_name=index_name,
            name=unique_name(f"{SMOKE_PREFIX}-bkp-async-name"),
            description="smoke-test backup created by IT-0027 (async)",
        )
        backup_id = backup.backup_id
        assert backup_id

        # Backups create asynchronously server-side — poll briefly until
        # describe returns Ready (or the test times out).
        deadline = time.monotonic() + 300
        status = ""
        while time.monotonic() < deadline:
            d = await pc.backups.describe(backup_id=backup_id)
            status = getattr(d, "status", "")
            if status == "Ready":
                break
            await asyncio.sleep(5)
        assert status == "Ready", f"Backup {backup_id} never became Ready (last status: {status!r})"

        # ----- backups.list (project-wide) -----
        listing = await pc.backups.list(limit=20)
        assert any(b.backup_id == backup_id for b in listing)

        # ----- backups.list (filtered by index_name) -----
        filtered = await pc.backups.list(index_name=index_name, limit=20)
        assert any(b.backup_id == backup_id for b in filtered)

        # ----- backups.get -----
        got = await pc.backups.get(backup_id=backup_id)
        assert got.backup_id == backup_id

        # ----- backups.delete -----
        await pc.backups.delete(backup_id=backup_id)
        backup_id = None  # mark as cleaned up

        # Confirm post-delete describe raises NotFound (eventually).
        deadline = time.monotonic() + 60
        notfound_raised = False
        while time.monotonic() < deadline:
            try:
                await pc.backups.describe(backup_id=got.backup_id)
                await asyncio.sleep(2)
            except NotFoundError:
                notfound_raised = True
                break
        assert notfound_raised, f"Expected NotFoundError after deleting backup {got.backup_id}"
    finally:
        # Clean up backup if the test left one alive.
        if backup_id is not None:
            try:
                await pc.backups.delete(backup_id=backup_id)
            except Exception as exc:
                print(f"  WARN: failed to delete leaked backup {backup_id}: {exc}")
        await async_ensure_index_deleted(pc, index_name)
        await pc.close()
