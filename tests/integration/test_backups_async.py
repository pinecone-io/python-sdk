"""Integration tests for backup lifecycle (async REST).

Phase 4 area tags: backup-lifecycle, create-index-from-backup
Transport: rest-async

Backups can be created from serverless indexes.
Tests create a small index, create a backup, verify BackupModel fields,
list/describe the backup, then delete both the backup and the source index.

Also tests restoring an index from a backup via create_index_from_backup().
"""

from __future__ import annotations

import pytest

from pinecone import AsyncPinecone
from pinecone.errors.exceptions import NotFoundError
from pinecone.models.backups.list import BackupList, RestoreJobList
from pinecone.models.backups.model import BackupModel, RestoreJobModel
from pinecone.models.indexes.index import IndexModel
from tests.integration.conftest import async_cleanup_resource, async_poll_until, unique_name

# ---------------------------------------------------------------------------
# backup-get-alias — REST async
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_backup_get_alias_and_default_description_async(async_client: AsyncPinecone) -> None:
    """Verify unified-bak-0007 (default description is empty/null when omitted)
    and unified-bak-0012 (get() and describe() are aliases returning identical results).

    Async variant of the sync test above.

    Area tag: backup-get-alias
    Transport: rest-async
    """
    index_name = unique_name("idx")
    backup_id: str | None = None

    try:
        # 1. Create a small serverless index to back up
        await async_client.indexes.create(
            name=index_name,
            dimension=2,
            metric="cosine",
            spec={"serverless": {"cloud": "aws", "region": "us-east-1"}},
            timeout=120,
        )

        # 2. Create a backup WITHOUT specifying description (SDK default is "")
        backup = await async_client.backups.create(
            index_name=index_name,
            name=unique_name("bk"),
            # description omitted on purpose — verifies unified-bak-0007
        )
        assert isinstance(backup, BackupModel)
        backup_id = backup.backup_id

        # unified-bak-0007: description should be empty ("" or None)
        assert backup.description in ("", None), (
            f"Expected description to be '' or None when omitted, got {backup.description!r}"
        )

        # 3. Call describe() — baseline
        described = await async_client.backups.describe(backup_id=backup_id)
        assert isinstance(described, BackupModel)
        assert described.backup_id == backup_id

        # 4. Call get() — unified-bak-0012: must be alias of describe()
        gotten = await async_client.backups.get(backup_id=backup_id)
        assert isinstance(gotten, BackupModel)
        assert gotten.backup_id == described.backup_id
        assert gotten.source_index_name == described.source_index_name
        assert gotten.cloud == described.cloud
        assert gotten.region == described.region
        assert gotten.description == described.description, (
            f"get().description={gotten.description!r} != "
            f"describe().description={described.description!r}"
        )

    finally:
        if backup_id is not None:
            await async_cleanup_resource(
                lambda: async_client.backups.delete(backup_id=backup_id),
                backup_id,
                "backup",
            )
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(index_name),
            index_name,
            "index",
        )


# ---------------------------------------------------------------------------
# backup-lifecycle — REST async
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_backup_lifecycle_async(async_client: AsyncPinecone) -> None:
    """Full backup lifecycle via async REST: create a serverless index,
    create a backup, poll until Ready, verify BackupModel fields, list
    (with and without index_name filter), describe, then delete backup and index.

    Area tag: backup-lifecycle
    Transport: rest-async
    """
    index_name = unique_name("idx")
    backup_id: str | None = None

    try:
        # 1. Create a small serverless index to back up
        await async_client.indexes.create(
            name=index_name,
            dimension=2,
            metric="cosine",
            spec={"serverless": {"cloud": "aws", "region": "us-east-1"}},
            timeout=120,
        )

        # 2. Create a backup from the index
        backup = await async_client.backups.create(
            index_name=index_name,
            name=unique_name("bk"),
            description="integration test backup",
        )
        assert isinstance(backup, BackupModel)
        backup_id = backup.backup_id

        # Verify initial BackupModel fields
        assert backup.backup_id != ""
        assert backup.source_index_name == index_name
        assert backup.status in ("Initializing", "Pending", "Ready")
        assert backup.cloud == "aws"
        assert backup.region == "us-east-1"
        assert backup.dense_dimension in (2, None)
        assert backup.created_at is not None

        # 3. Poll until the backup is Ready
        ready_backup = await async_poll_until(
            query_fn=lambda: async_client.backups.describe(backup_id=backup_id),
            check_fn=lambda b: b.status == "Ready",
            timeout=300,
            interval=10,
            description="backup Ready",
        )
        assert isinstance(ready_backup, BackupModel)
        assert ready_backup.backup_id == backup_id
        assert ready_backup.status == "Ready"
        assert ready_backup.source_index_name == index_name

        # 4. list() without index_name filter — backup appears in full list
        all_backups = await async_client.backups.list()
        assert isinstance(all_backups, BackupList)
        all_ids = [b.backup_id for b in all_backups]
        assert backup_id in all_ids

        # 5. list() with index_name filter — backup appears in filtered list
        idx_backups = await async_client.backups.list(index_name=index_name)
        assert isinstance(idx_backups, BackupList)
        idx_ids = [b.backup_id for b in idx_backups]
        assert backup_id in idx_ids

        # 6. describe() returns correct BackupModel
        desc = await async_client.backups.describe(backup_id=backup_id)
        assert isinstance(desc, BackupModel)
        assert desc.backup_id == backup_id
        assert desc.source_index_name == index_name
        assert desc.status == "Ready"

    finally:
        # Clean up backup first, then the index
        if backup_id is not None:
            await async_cleanup_resource(
                lambda: async_client.backups.delete(backup_id=backup_id),
                backup_id,
                "backup",
            )
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(index_name),
            index_name,
            "index",
        )


# ---------------------------------------------------------------------------
# create-index-from-backup — REST async
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_create_index_from_backup_async(async_client: AsyncPinecone) -> None:
    """Create a serverless index, create a backup, then restore a new index
    from the backup via async REST.  Verify the restored IndexModel has the
    same dimension/metric as the source, and the index handle is queryable.

    Note on vector data: backups from freshly created serverless indexes
    consistently report record_count=0 (data is snapshotted from durable
    storage which lags behind the query-visible layer).  This test focuses
    on the structural properties of the restore operation.

    Area tag: create-index-from-backup
    Transport: rest-async
    """
    source_index_name = unique_name("idx")
    restore_index_name = unique_name("idx")
    backup_id: str | None = None

    try:
        # 1. Create a small source index
        await async_client.indexes.create(
            name=source_index_name,
            dimension=4,
            metric="dotproduct",
            spec={"serverless": {"cloud": "aws", "region": "us-east-1"}},
            timeout=120,
        )

        # 2. Create a backup from the source index
        backup = await async_client.backups.create(
            index_name=source_index_name,
            name=unique_name("bk"),
            description="create-index-from-backup async test",
        )
        assert isinstance(backup, BackupModel)
        backup_id = backup.backup_id

        # Initial BackupModel should reflect the source index schema
        assert backup.source_index_name == source_index_name
        assert backup.cloud == "aws"
        assert backup.region == "us-east-1"

        # 3. Poll until the backup is Ready (up to 5 minutes)
        ready_backup = await async_poll_until(
            query_fn=lambda: async_client.backups.describe(backup_id=backup_id),
            check_fn=lambda b: b.status == "Ready",
            timeout=300,
            interval=10,
            description="backup Ready",
        )
        assert isinstance(ready_backup, BackupModel)
        assert ready_backup.status == "Ready"
        assert ready_backup.dense_dimension in (4, None)

        # 4. Restore a new index from the backup (SDK polls until index is ready)
        restored = await async_client.create_index_from_backup(
            backup_id=backup_id,
            name=restore_index_name,
            timeout=600,
        )

        # 5. Verify the restored IndexModel has the same dimension and metric
        assert isinstance(restored, IndexModel)
        assert restored.name == restore_index_name
        assert restored.dimension == 4
        assert restored.metric == "dotproduct"
        assert restored.status.ready is True
        # Serverless spec should be preserved
        assert restored.spec.serverless is not None
        assert restored.spec.serverless.cloud == "aws"
        assert restored.spec.serverless.region == "us-east-1"

        # 6. Get an Index handle — index should be reachable and queryable
        restore_desc = await async_client.indexes.describe(restore_index_name)
        restore_index = await async_client.index(host=restore_desc.host)
        stats = await restore_index.describe_index_stats()
        assert stats.dimension == 4
        # total_vector_count may be 0 for a freshly created backup (durable
        # storage snapshot lag), which is acceptable
        assert stats.total_vector_count >= 0

    finally:
        # Clean up: restored index, then backup, then source index
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(restore_index_name),
            restore_index_name,
            "index (restored)",
        )
        if backup_id is not None:
            await async_cleanup_resource(
                lambda: async_client.backups.delete(backup_id=backup_id),
                backup_id,
                "backup",
            )
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(source_index_name),
            source_index_name,
            "index (source)",
        )


# ---------------------------------------------------------------------------
# backup-error-paths — REST async
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_backup_and_restore_job_error_paths_async(async_client: AsyncPinecone) -> None:
    """Keyword-only enforcement and not-found errors for backups and restore jobs (async).

    Verifies:
    - unified-bak-0013: All async backup and restore_job method parameters must be
      passed as keyword arguments; positional args raise TypeError (enforced by *).
    - unified-bak-0017: Describing a nonexistent backup raises NotFoundError.
    - unified-bak-0018: Describing a nonexistent restore job raises NotFoundError.

    No resources are created — keyword-only checks are pure Python, and the
    not-found errors are live API responses to known-bogus identifiers.

    Area tag: backup-error-paths
    Transport: rest-async
    """
    # --- unified-bak-0013: keyword-only enforcement (async variants) ---
    with pytest.raises(TypeError):
        await async_client.backups.describe("definitely-not-a-keyword-arg")  # type: ignore[call-arg]

    with pytest.raises(TypeError):
        await async_client.backups.delete("definitely-not-a-keyword-arg")  # type: ignore[call-arg]

    with pytest.raises(TypeError):
        await async_client.backups.create("some-index-name")  # type: ignore[call-arg]

    with pytest.raises(TypeError):
        await async_client.restore_jobs.describe("definitely-not-a-keyword-arg")  # type: ignore[call-arg]

    # --- unified-bak-0017: nonexistent backup raises NotFoundError ---
    with pytest.raises(NotFoundError):
        await async_client.backups.describe(backup_id="nonexistent-backup-id-xyz-000")

    # --- unified-bak-0018: nonexistent restore job raises NotFoundError ---
    with pytest.raises(NotFoundError):
        await async_client.restore_jobs.describe(job_id="nonexistent-restore-job-id-xyz-000")


# ---------------------------------------------------------------------------
# restore-jobs — REST async
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_restore_jobs_list_and_describe_async(async_client: AsyncPinecone) -> None:
    """Verify async pc.restore_jobs.list() and pc.restore_jobs.describe() structure.

    Async variant of test_restore_jobs_list_and_describe_rest.

    Verifies:
    - unified-bak-0005: Can describe a restore job by identifier.
    - unified-bak-0006: Can list all restore jobs in the project.

    Area tag: restore-jobs
    Transport: rest-async
    """
    source_index_name = unique_name("idx")
    restore_index_name = unique_name("idx")
    backup_id: str | None = None

    try:
        # 1. Create a small source index
        await async_client.indexes.create(
            name=source_index_name,
            dimension=2,
            metric="cosine",
            spec={"serverless": {"cloud": "aws", "region": "us-east-1"}},
            timeout=120,
        )

        # 2. Create a backup
        backup = await async_client.backups.create(
            index_name=source_index_name,
            name=unique_name("bk"),
        )
        assert isinstance(backup, BackupModel)
        backup_id = backup.backup_id

        # 3. Poll until backup is Ready (up to 5 minutes)
        await async_poll_until(
            query_fn=lambda: async_client.backups.describe(backup_id=backup_id),
            check_fn=lambda b: b.status == "Ready",
            timeout=300,
            interval=10,
            description="backup Ready",
        )

        # 4. Start restore with timeout=-1 — triggers restore job without blocking
        await async_client.create_index_from_backup(
            backup_id=backup_id,
            name=restore_index_name,
            timeout=-1,
        )

        # 5. Poll restore_jobs.list() until the restore job for our index appears
        job_list = await async_poll_until(
            query_fn=lambda: async_client.restore_jobs.list(),
            check_fn=lambda lst: any(j.target_index_name == restore_index_name for j in lst),
            timeout=60,
            interval=5,
            description=f"restore job for {restore_index_name!r} visible in list",
        )

        # 6. Verify RestoreJobList container structure
        assert isinstance(job_list, RestoreJobList)
        assert len(job_list) >= 1

        # 7. Find and inspect our specific restore job in the list
        our_job = next(j for j in job_list if j.target_index_name == restore_index_name)
        assert isinstance(our_job, RestoreJobModel)
        assert isinstance(our_job.restore_job_id, str) and our_job.restore_job_id
        assert isinstance(our_job.backup_id, str) and our_job.backup_id == backup_id
        assert isinstance(our_job.target_index_name, str)
        assert our_job.target_index_name == restore_index_name
        assert isinstance(our_job.target_index_id, str) and our_job.target_index_id
        assert our_job.status in ("Pending", "InProgress", "Completed", "Failed")
        assert isinstance(our_job.created_at, str) and our_job.created_at
        assert our_job.completed_at is None or isinstance(our_job.completed_at, str)
        assert our_job.percent_complete is None or isinstance(our_job.percent_complete, float)
        # Bracket access
        assert our_job["restore_job_id"] == our_job.restore_job_id
        assert our_job["backup_id"] == our_job.backup_id
        assert "restore_job_id" in our_job
        assert "nonexistent_field" not in our_job

        # 8. Call restore_jobs.describe() — verify same job by ID
        described = await async_client.restore_jobs.describe(job_id=our_job.restore_job_id)
        assert isinstance(described, RestoreJobModel)
        assert described.restore_job_id == our_job.restore_job_id
        assert described.backup_id == backup_id
        assert described.target_index_name == restore_index_name
        assert isinstance(described.target_index_id, str) and described.target_index_id
        assert described.status in ("Pending", "InProgress", "Completed", "Failed")
        assert isinstance(described.created_at, str) and described.created_at

    finally:
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(restore_index_name),
            restore_index_name,
            "index (restored)",
        )
        if backup_id is not None:
            await async_cleanup_resource(
                lambda: async_client.backups.delete(backup_id=backup_id),
                backup_id,
                "backup",
            )
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(source_index_name),
            source_index_name,
            "index (source)",
        )
