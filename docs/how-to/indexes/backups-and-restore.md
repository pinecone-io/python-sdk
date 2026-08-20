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


## Schedule automatic backups

Everything above takes a backup when *you* ask for one. A **backup schedule**
attaches a recurring cadence to an index, so backups keep happening without a
caller. Schedules live on `pc.backup_schedules`.

Scheduled backups are a plan entitlement. Where the API enforces it, the check
runs before the index is even looked up, so a project without the entitlement
gets `403` rather than a `404` for a missing index. The SDK appends a
clarification to that `403` and keeps the backend's own message as the prefix.

```python
schedule = pc.backup_schedules.create(
    index_name="product-search",
    name="daily-compliance-backup",
    frequency="daily",       # daily | weekly | monthly
    retention_days=90,
)
print(schedule.schedule_id)
print(schedule.next_scheduled_run)   # a datetime, not a string
```

There is **no cron support** anywhere in this API. `frequency` accepts exactly
`daily`, `weekly`, or `monthly`, and the SDK rejects anything else before
sending a request. The run time is chosen server-side and reported through
`next_scheduled_run`; there is no way to pick an hour or a timezone.

`retention_days` must be at least 1. Its upper bound is your project's
`max_backup_retention_days` (365 by default), which the SDK does not know
per-project, so a too-large value is rejected by the server with a message
naming the real limit.

Only **one enabled schedule per index** is allowed. Creating a second one fails
with `409` and a message telling you to disable or delete the first. Pod-based
indexes cannot be scheduled at all and are rejected with `400`.

```{important}
Keep the schedule `name` to **28 characters or fewer.** Each run names its
backup `"{name}-{run timestamp}"`, and that timestamp suffix is a fixed 17
characters (`-YYYYMMDDTHHMMSSZ`) against a 45-character resource name limit.
Nothing validates this — the API declares no length limit on a schedule name,
and the create-schedule path does not check the derived backup name either — so
a longer name is accepted here and then produces backup names the backup
endpoints themselves would have rejected. The failure surfaces later, on the
runs, rather than on `create`. The SDK does not enforce the limit, because
doing so would reject names the API accepts.
```

### List the schedules on an index

Schedules are always listed per index — there is no project-wide schedule
listing. Disabled schedules are included.

```python
for schedule in pc.backup_schedules.iter_schedules(index_name="product-search"):
    print(schedule.schedule_id, schedule.frequency, schedule.enabled)
```

`iter_schedules` walks every page. `list` returns a single page plus a
`pagination` token if you would rather drive pagination yourself:

```python
page = pc.backup_schedules.list(index_name="product-search", limit=10)
print(page.names())
print([s.schedule_id for s in page.enabled_schedules()])
print(page.pagination)   # None on the final page
```

### Describe, update, and delete a schedule

```python
schedule = pc.backup_schedules.describe(schedule_id="sched-abc123")
```

`update` is a sparse PATCH: only the arguments you pass are sent, so anything
you omit is left unchanged rather than reset.

```python
paused = pc.backup_schedules.update(schedule_id="sched-abc123", enabled=False)
assert paused.next_scheduled_run is None

pc.backup_schedules.update(
    schedule_id="sched-abc123", frequency="weekly", retention_days=30
)
```

```{warning}
Re-enabling a disabled schedule with `enabled=True` **immediately enqueues a
backup run** — it is not a free toggle. It also recomputes
`next_scheduled_run` from the moment of the update rather than resuming the old
slot, so a disable/re-enable cycle shifts the cadence. And because only one
schedule per index may be enabled, re-enabling fails with `409` when another
one already is.
```

```python
pc.backup_schedules.delete(schedule_id="sched-abc123")
```

Deleting a schedule stops future runs. Backups it already produced are **not**
deleted; they age out on their own retention window.

```{important}
`delete` is not safe to retry blindly. Success answers `204` with no body, and
a second attempt on the same `schedule_id` answers `404` — so a retry after a
dropped response looks identical to deleting something that was never there.
Treat a `404` following a delete attempt as success.
```

### Inspect what a schedule has produced

```python
for run in pc.backup_schedules.iter_history(schedule_id="sched-abc123"):
    print(run.backup_id, run.status, run.record_count)
```

History rows describe backup *snapshots*, not the schedule. A row appears as
soon as a run is planned, so the listing mixes completed runs with ones that
have not started; `run.is_scheduled` and `history.scheduled()` pick out the
latter. As with the schedule listing, `history` returns one page and
`iter_history` walks all of them — prefer the iterator here, because a daily
schedule with a 90-day retention window has far more rows than one page holds.

```{note}
Against today's backend, schedule history is served by the shared backup
handler, which never reports the `Scheduled` status and does not send
`scheduled_execution_at` at all. Those fields are typed and will populate when
the backend graduates; until then `scheduled_execution_at` reads as `None` and
`name` / `record_count` / `namespace_count` / `size_bytes` can come back
`None` on freshly created rows.
```


## See also

- {class}`~pinecone.models.backups.model.BackupModel` — backup response model
- {class}`~pinecone.models.backups.list.BackupList` — backup list response
- {class}`~pinecone.models.backups.model.RestoreJobModel` — restore job model
- {class}`~pinecone.models.backups.list.RestoreJobList` — restore job list response
- {class}`~pinecone.models.backups.schedules.BackupScheduleModel` — schedule response model
- {class}`~pinecone.models.backups.list.BackupScheduleList` — schedule list response
- {class}`~pinecone.models.backups.schedules.BackupScheduleHistoryItem` — one run produced by a schedule
- {class}`~pinecone.models.backups.list.BackupScheduleHistoryList` — schedule history response
- {doc}`/how-to/indexes/serverless` — serverless index management
- {doc}`/how-to/indexes/pod` — pod-based index management
