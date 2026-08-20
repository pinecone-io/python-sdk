# Backups and Restore

Backups are point-in-time snapshots of an index. Use them to protect against data loss,
create copies of an index, or restore a previous state.

## Create a backup

Pass the name of the index you want to back up:

```python
from pinecone import Pinecone

pc = Pinecone(api_key="your-api-key")

backup = pc.backups.create(index_name="product-search")
print(backup.backup_id)   # e.g. "bk-abc123"
print(backup.status)      # e.g. "Initializing"
```

Add a name and description for easier identification:

```python
backup = pc.backups.create(
    index_name="product-search",
    name="pre-reindex-snapshot",
    description="Backup before schema migration on 2025-03-01",
)
```

The backup transitions through ``Initializing`` → ``Ready`` when complete.

The same operation is available index-first on the indexes namespace, if that
reads better in an index-centric flow:

```python
backup = pc.indexes.create_backup("product-search", name="pre-reindex-snapshot")
```


## List backups

List all backups in the project:

```python
for backup in pc.backups.list():
    print(backup.backup_id, backup.name, backup.status)
```

Filter by index:

```python
for backup in pc.backups.list(index_name="product-search"):
    print(backup.backup_id, backup.created_at)
```

`list` returns a {class}`~pinecone.models.backups.list.BackupList` with cursor-based
pagination. Pass `limit` to control page size and `pagination_token` to advance pages:

```python
page = pc.backups.list(limit=5)
if page.pagination and page.pagination.next:
    next_page = pc.backups.list(limit=5, pagination_token=page.pagination.next)
```

`pagination` is `None` on the final page.

`pc.indexes.list_backups` is the index-scoped equivalent, returning a
{class}`~pinecone.models.pagination.Paginator` that walks the pages for you:

```python
for backup in pc.indexes.list_backups("product-search"):
    print(backup.backup_id, backup.status)
```


### Backups of a deleted index

A backup outlives its source index, so an index-scoped listing has to say what
it means by the index name. **By default it means the active index.** If every
index that used the name has been deleted, the API answers **404 — not an empty
list.**

Pass `include_deleted=True` to widen the listing to every index that has ever
used the name:

```python
orphaned = pc.backups.list(index_name="product-search", include_deleted=True)

for backup in orphaned:
    if backup.source_index_deleted_at:
        print(backup.backup_id, "orphaned at", backup.source_index_deleted_at)
```

So a 404 from `pc.backups.list(index_name=...)` is not proof the name was never
used — retry with `include_deleted=True` before concluding that. A 404 *with*
`include_deleted=True` does mean the name has never existed in this project.

`include_deleted` applies only to index-scoped listings. The project-wide
`pc.backups.list()` already returns backups whose source index is gone, and
passing `include_deleted` there raises `PineconeValueError` rather than being
silently ignored. Omitting the argument leaves the parameter off the request
entirely, so the server's default applies.


## Describe a backup

```python
backup = pc.backups.describe(backup_id="bk-abc123")
print(backup.source_index_name)
print(backup.status)
print(backup.dense_dimension)
print(backup.schema.fields["embedding"].metric)
print(backup.record_count)
print(backup.size_bytes)
print(backup.source_index_deleted_at)
```

`schema` is an {class}`~pinecone.models.indexes.schema.IndexSchema`, the same
typed model returned by `pc.indexes.describe(...).schema`, and is `None` when
the server returns no schema for the backup.

`pc.indexes.describe_backup("bk-abc123")` is the same call under the indexes
namespace.


## Restore a backup to a new index

Use `create_index_from_backup` on the top-level client to restore a backup into a new
index:

```python
from pinecone import Pinecone

pc = Pinecone(api_key="your-api-key")

index = pc.create_index_from_backup(
    name="product-search-restored",
    backup_id="bk-abc123",
)
print(index.name)
print(index.status.state)
```

`create_index_from_backup` polls until the new index is ready. Pass `timeout=-1` to
return immediately:

```python
index = pc.create_index_from_backup(
    name="product-search-restored",
    backup_id="bk-abc123",
    timeout=-1,
)
```

Enable deletion protection or add tags to the restored index:

```python
index = pc.create_index_from_backup(
    name="product-search-restored",
    backup_id="bk-abc123",
    deletion_protection="enabled",
    tags={"env": "production", "team": "search"},
)
```

A restore defaults to on-demand read capacity. Pass `read_capacity` to land the
restored index straight onto dedicated read nodes instead, skipping a
`configure` round trip after the restore:

```python
index = pc.create_index_from_backup(
    name="product-search-restored",
    backup_id="bk-abc123",
    read_capacity={
        "mode": "Dedicated",
        "dedicated": {
            "node_type": "t1",
            "scaling": "Manual",
            "manual": {"shards": 2, "replicas": 2},
        },
    },
)
```

Dedicated capacity is a serverless-backup feature, and the server rejects a
configuration too small to hold the backup.

`create_index_from_backup` is the only supported way to restore a backup —
`pc.create_index(source_backup_id=...)` raises a `PineconeTypeError` pointing
here.


## Monitor restore jobs

Each call to `create_index_from_backup` starts a restore job. List all restore jobs:

```python
for job in pc.restore_jobs.list():
    print(job.restore_job_id, job.status, job.percent_complete)
```

Describe a specific job:

```python
job = pc.restore_jobs.describe(job_id="rj-xyz789")
print(job.restore_job_id)
print(job.backup_id)
print(job.target_index_name)
print(job.status)         # e.g. "Running", "Completed"
print(job.percent_complete)
print(job.completed_at)
```

`describe` returns a {class}`~pinecone.models.backups.model.RestoreJobModel`.


## Delete a backup

```python
pc.backups.delete(backup_id="bk-abc123")
```

Deleting a backup does not affect the source index or any indexes restored from it.
The call returns `None` — the API answers `202 Accepted` with no body.

A backup with a restore job still in flight cannot be deleted; the API returns
`412` and the SDK raises an `ApiError` naming the pending job ids. Wait for the
restore to finish, then delete.


## See also

- {class}`~pinecone.models.backups.model.BackupModel` — backup response model
- {class}`~pinecone.models.backups.list.BackupList` — backup list response
- {class}`~pinecone.models.backups.model.RestoreJobModel` — restore job model
- {class}`~pinecone.models.backups.list.RestoreJobList` — restore job list response
- {doc}`/how-to/indexes/serverless` — serverless index management
- {doc}`/how-to/indexes/pod` — pod-based index management
