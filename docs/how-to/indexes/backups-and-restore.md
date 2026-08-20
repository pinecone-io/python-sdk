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


## See also

- {class}`~pinecone.models.backups.model.BackupModel` — backup response model
- {class}`~pinecone.models.backups.list.BackupList` — backup list response
- {class}`~pinecone.models.backups.model.RestoreJobModel` — restore job model
- {class}`~pinecone.models.backups.list.RestoreJobList` — restore job list response
- {doc}`/how-to/indexes/serverless` — serverless index management
- {doc}`/how-to/indexes/pod` — pod-based index management
