# 2026-07: BackupModel and backup endpoint changes

The backup response models now follow the Pinecone `2026-07` API shapes.
This is a breaking change to `BackupModel`. The operations themselves change
additively — see [Operations](#operations) below.

Release-notes source of truth for the **backup and backup-schedule** half of
the 2026-07 db_control surface. The index create/configure half is in
[the deployment/schema break](v10-2026-07-db-control.md); response-model changes
to `IndexModel` are in [IndexModel changes](v10-2026-07-index-model.md).

Every `python` block below is executed against a stubbed control plane in
`tests/unit/test_docs_migration_backup_notes_339.py`, which reads the blocks
**out of this file** rather than from a transcription, so an example here cannot
drift from what runs. Sync/async twins are checked twice: the async source must
be its sync neighbour word-for-word modulo `await`, and both must put
byte-identical requests on the wire.

## What changed

`BackupModel` no longer has `.dimension` or `.metric`. Accessing either
raises an `AttributeError` that names the replacement. `.schema` changed
type: it was a plain `dict` shaped like the old metadata schema
(`{"fields": {"<name>": {"filterable": bool}}}`) and is now a typed
`IndexSchema` — the same class returned by `index.schema`.

| Removed | Replacement |
| --- | --- |
| `backup.dimension` | `backup.schema.fields["<field>"].dimension` on the `DenseVectorField`, or `backup.dense_dimension` |
| `backup.metric` | `backup.schema.fields["<field>"].metric` on the vector field |
| `backup.schema["fields"]["<name>"]["filterable"]` | `backup.schema.fields["<name>"].filterable` |

```python
# 9.x
dim = backup.dimension
metric = backup.metric

# 10.x
dim = backup.dense_dimension
metric = backup.schema.fields["embedding"].metric
```

`dense_dimension` is a convenience for the common one-vector-field case.
It returns `None` when the schema is absent, declares no `dense_vector`
field, or declares more than one — read the field you want out of
`backup.schema.fields` in that case.

New field on `BackupModel`: `source_index_deleted_at` — the deletion
timestamp of the source index, or `None` when that index is still active.
Only `list_index_backups(include_deleted=True)` populates it.

## New exports

`CreateIndexFromBackupRequest` is exported from `pinecone`,
`pinecone.models`, and `pinecone.models.backups`. It carries the additive
`read_capacity` field, which the 2026-07 OAS declares on the restore request
— but the backend does not read it, so **it has no effect today**; see
[`read_capacity` on restore](#read-capacity-on-restore) below. Unset optional
fields stay off the wire, so a request built with only `name` serialises to
`{"name": ...}`.

## Removed exports

`PreviewBackupModel` and `PreviewCreateBackupRequest` are removed from
`pinecone.preview.models`, along with the
`pinecone.preview.models.backups` module. Preview backup operations
(`pc.preview.indexes.create_backup`, `list_backups`, `describe_backup`)
now return the single top-level `BackupModel`, so `pc.backups.*` and
`pc.preview.indexes.*_backup` no longer hand back two different backup
types. Callers of the preview surface that read `.dimension` must move to
`.dense_dimension` or `.schema.fields`.

## Operations

No backup method lost an argument. What changed:

| Method | Change |
| --- | --- |
| `pc.backups.list` | new keyword-only `include_deleted: bool \| None = None` |
| `pc.create_index_from_backup` | new keyword-only `read_capacity: dict \| None = None` |
| `pc.indexes.create_backup` / `list_backups` / `describe_backup` | graduated out of `pc.preview.indexes` |

### `include_deleted` and what a 404 means

`pc.backups.list(index_name=...)` (and `pc.indexes.list_backups(...)`) resolve
the name against **active** indexes by default. If every index that used the
name has been deleted, the API returns **404 rather than an empty list** — so a
404 is not proof the name was never used:

What the two settings do with each state of the name
(`db_control_2026-07.oas.yaml:828-829,849` @ apis `5f808858`):

| State of `index_name` in the project | omitted / `False` | `True` |
| --- | --- | --- |
| An active index has the name | its backups | backups of **every** index that has held the name, active and deleted |
| Only deleted indexes have held the name | **404** — not an empty list | those backups, each with `source_index_deleted_at` set |
| The name has never existed | **404** | **404** |
| The name exists but has no backups | empty list | empty list |

So a 404 is only conclusive when you sent `include_deleted=True`: a 404 there
means the name has never been used in this project. A 404 without it proves
nothing about whether the name was ever used — retry with `include_deleted=True`
before concluding anything.

```python
# 10.x — recover the backups of an index you already deleted
page = pc.backups.list(index_name="product-search", include_deleted=True)
orphaned = [b for b in page if b.source_index_deleted_at]
```

`source_index_deleted_at` is populated **only** on the `include_deleted=True`
listing, so it is the field that separates a deleted index's backups from an
active one's in a mixed response.

Omitting the argument keeps the parameter off the request entirely, so the
server's default (`false`) applies. `include_deleted` on the project-wide
`pc.backups.list()` raises `PineconeValueError` instead of being silently
dropped, because that operation does not accept it — the project-wide listing
already returns backups whose source index was deleted.

(read-capacity-on-restore)=
### `read_capacity` on restore

```{warning}
**`read_capacity=` on a restore is accepted and silently ignored.** The
2026-07 OAS declares it on `CreateIndexFromBackupRequest`
(`db_control_2026-07.oas.yaml:3656-3673`, with a worked `dedicated` example at
`:1957-1968`), but the backend's request struct has only
`{name, tags, deletion_protection}` — `.../base/backups.rs:77-87` @ pinecone-db
`cbee5a67fe` — and `create_index_from_backup` never reads a read-capacity
value. It hardcodes on both restore branches:

* managed/serverless → `ReadCapacityRequest::default()` (on-demand),
  `backups.rs:468` and `:522`, both carrying
  `// TODO(@damargulis): allow users to set here too`
* BYOC → `Provisioned { tier: B1, shards: 1, replicas: 1 }`, `backups.rs:481-485`
  and `:533-537` — so a BYOC restore additionally **loses the source index's
  provisioned tier**

There is no error and no warning: the field goes on the wire, the restore
succeeds, and the index lands on on-demand capacity. Tracked as
[#333](https://github.com/pinecone-io/python-sdk-internal/issues/333).
```

The SDK keeps the spec-shaped parameter rather than rejecting it client-side,
so passing it type-checks and serialises. Until the backend lands the TODO, a
restore onto dedicated read nodes is **two calls** — restore, then configure:

:::::{tabs}
::::{tab} Sync

```python
job = pc.create_index_from_backup(
    name="product-search-restored",
    backup_id="bk-abc123",
    timeout=-1,
)
index = pc.indexes.configure(
    "product-search-restored",
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

::::
::::{tab} Async

```python
job = await pc.create_index_from_backup(
    name="product-search-restored",
    backup_id="bk-abc123",
    timeout=-1,
)
index = await pc.indexes.configure(
    "product-search-restored",
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

::::
:::::

`read_capacity` changes apply asynchronously, so poll
`index.read_capacity.status` rather than assuming the returned model is
settled. Omitted, the key stays off the wire.
`create_index_from_backup` remains the only supported restore path —
`pc.create_index(source_backup_id=...)` raises a `PineconeTypeError` pointing
at it.

```{note}
**Do not poll a restore job for an in-progress state.** `restore_job.status`
is one of `Pending`, `Failed`, `Completed`, `Cancelled` and nothing else
(`pc-reconciler/src/lib.rs:12-22`), and `percent_complete` is `100` on
`Completed` and `None` otherwise — never an intermediate percentage
(`.../base/restore_jobs.rs:180-184`). A loop that waits for a `"Running"`
status, or for `percent_complete` to climb, never terminates. Wait on
`status == "Completed"` and treat `Failed`/`Cancelled` as terminal.
```

### Graduated index-scoped methods

`pc.preview.indexes.create_backup` / `list_backups` / `describe_backup` now
exist on `pc.indexes` with the same signatures, returning the top-level
`BackupModel`. `pc.indexes.list_backups` returns a `Paginator[BackupModel]`
and additionally accepts `include_deleted`. `pc.backups.*` is unchanged apart
from the new `include_deleted` keyword, and remains the only surface for the
project-wide listing and for `delete`.

### Asyncio lane

`AsyncPinecone.backups` mirrors `pc.backups` one-for-one: same method names,
same keyword-only arguments, same validation messages, and the same request on
the wire — only `await` differs. Construct the client with
`async with AsyncPinecone(api_key="...") as pc:` as usual.

:::::{tabs}
::::{tab} Sync

```python
page = pc.backups.list(index_name="product-search", include_deleted=True)
```

::::
::::{tab} Async

```python
page = await pc.backups.list(index_name="product-search", include_deleted=True)
```

::::
:::::

The legacy top-level shims are mirrored too: `await pc.list_backups(...)`
forwards `include_deleted`, and `await pc.create_index_from_backup(...)` accepts
`read_capacity` — both landed with the async index control-plane work in
[#131](https://github.com/pinecone-io/python-sdk-internal/issues/131), and both
put byte-identical requests on the wire as their sync twins. (`read_capacity`
serialises but has no effect on either lane — see the warning above.)

## Backup schedules — new in 2026-07

`pc.backup_schedules` is **additive**: there is no 9.x equivalent and nothing
to migrate. Six operations — `create`, `list`, `describe`, `update`, `delete`,
`history` — plus the lazy `iter_schedules` / `iter_history` walkers. The
docstrings carry the full signatures; this section covers only the three
behaviours that surprise callers, all verified against pinecone-db
`cbee5a67fe`.

### Plan gating is asymmetric between `create` and `update`

Backups are a plan entitlement: `validate_backups_allowed()` returns
`PermissionDenied` → **403** on exactly the Free and Builder plans, and `Ok` on
Standard, Enterprise (both including their trial variants), Dedicated and
Internal (`pc-cps/src/core/organization_store/mod.rs:116-131`;
`ErrorCode::PermissionDenied → 403` at `pc-error/src/server.rs:51`).

Where it is enforced is the part worth knowing:

| Operation | Entitlement checked? |
| --- | --- |
| `create` | **always**, before the index is looked up — `.../base/backup_schedules.rs:97-100` |
| `update` toggling disabled → enabled | **yes** — `backup_schedules.rs:304-310` |
| `update` disabling, or changing `frequency` / `retention_days` | **no** — deliberately left open |
| `update` setting `enabled=True` on an already-enabled schedule | **no** — not a transition, so not a re-enable |
| `list`, `describe`, `delete`, `history` | **no** |

Two consequences:

* Because `create`'s check runs **before** the index lookup, a project without
  the entitlement gets **403 for a nonexistent index name too** — the 403 is
  about the plan, never about the index. (Request-body validation of
  `retention_days` also precedes the lookup, so a bad retention on a
  nonexistent index is a 400.)
* A downgraded organisation can still **disable, retune and delete** an
  existing schedule; it just cannot create one or switch one back on. That is
  intentional, so a downgrade does not trap callers with a schedule they cannot
  turn off.

```{warning}
On-demand backups are gated by the **same** entitlement —
`pc.backups.create()` calls `validate_backups_allowed()` before its own index
lookup (`.../base/backups.rs:183-186`). A 403 from a schedule operation is not
a reason to fall back to `pc.backups.create()`; that call answers 403 for the
same projects.
```

### One enabled schedule per index (409)

An index may hold at most one *enabled* schedule. Disabled schedules do not
count, so you can keep several and enable one at a time. The conflict surfaces
as **409** (`ErrorCode::AlreadyExists → 409`, `pc-error/src/server.rs:50`;
`StoreError::Conflict → 409`, `pc-error/src/store.rs:56`) from three distinct
places:

| Path | Where | Message |
| --- | --- | --- |
| `create` while one is enabled | in-transaction `SELECT ... enabled = true` pre-check, `pc-cps/src/core/backup_schedule_store/postgres.rs:41-57` | "This index already has an enabled backup schedule. Disable or delete it first." |
| `update` re-enabling while another is enabled | explicit pre-check, `backup_schedules.rs:312-327` | "…Disable it first before re-enabling this one." |
| the same re-enable, losing a race | Postgres unique violation `23505`, `postgres.rs:233-240` | "…(unique constraint). Disable it first before re-enabling." |

The three messages differ, so do not match on text — match on the 409.

### Re-enabling runs a backup immediately, and shifts the cadence

`update(enabled=True)` on a **disabled** schedule is not just a flag flip. The
backend, in one request:

1. recomputes `next_scheduled_run` **from the moment of the update** rather than
   from the schedule's original anchor — `calculate_next_run_from_now(...,
   Utc::now())`, `backup_schedules.rs:344-354`; and
2. enqueues a backup operation for that run — `backup_schedules.rs:394-417`.

So a disable/re-enable cycle **permanently shifts the schedule's cadence** to
the re-enable time, and costs one extra backup. A daily schedule anchored at
06:00 UTC, disabled and re-enabled at 15:20, runs daily at 15:20 from then on.
There is no way to re-enable without this side effect.

`enabled=True` on an **already-enabled** schedule is a no-op for both: the
backend guards every step on `!existing.enabled`, so it neither recomputes
`next_scheduled_run` nor enqueues a duplicate operation. Re-sending it is
therefore safe.

A re-enable also re-checks that the index still supports backups —
`ensure_backup_supported` rejects a CMEK-encrypted index with **400** *before*
the schedule change is persisted (`backups.rs:166-174`), so a refused re-enable
leaves the schedule disabled rather than enabled-but-idle.

:::::{tabs}
::::{tab} Sync

```python
schedule = pc.backup_schedules.create(
    index_name="product-search",
    name="daily-compliance-backup",
    frequency="daily",
    retention_days=90,
)
paused = pc.backup_schedules.update(schedule_id=schedule.schedule_id, enabled=False)
resumed = pc.backup_schedules.update(schedule_id=schedule.schedule_id, enabled=True)
```

::::
::::{tab} Async

```python
schedule = await pc.backup_schedules.create(
    index_name="product-search",
    name="daily-compliance-backup",
    frequency="daily",
    retention_days=90,
)
paused = await pc.backup_schedules.update(schedule_id=schedule.schedule_id, enabled=False)
resumed = await pc.backup_schedules.update(schedule_id=schedule.schedule_id, enabled=True)
```

::::
:::::

`resumed.next_scheduled_run` is the recomputed timestamp; the backup enqueued
alongside it shows up in `pc.backup_schedules.history(...)`.

### Paginating the schedule listings

`list` and `history` return one page each. `limit` is **dropped whenever
`pagination_token` is given** — the token is a base64 `{limit, offset}` pair
and the SDK omits `limit` rather than letting it override the token's page size
at the token's offset, which would skip or repeat rows
(`pinecone/_internal/backups_helpers.py:58-65`). The same rule holds for
`pc.backups.list` and `pc.restore_jobs.list`, which share that helper. Send
`limit` on the first call only, or use `iter_schedules` / `iter_history`, which
follow the tokens for you.

## Behavior notes

- `status` is documented as `Initializing`, `Ready`, or `Failed`. The
  backend currently returns `InitializationFailed` in place of `Failed`
  for backups; the model keeps `status` a plain `str`, so both decode.
- `schema` is `None` when the server omits it or returns `"schema": null`
  — for example a schedule-produced backup of an index that declared no
  schema.
- Backups captured before typed schemas existed arrive with no `type` key
  on each field and decode to `LegacyMetadataField`, matching
  `IndexModel`. `to_dict()` emits those fields without a `type` key, so
  the projection still matches the wire format.
- `created_at` stays optional (`str | None`) so payloads from older API
  versions still decode.
- `tags` is `None` when the source index had no tags — the API returns
  `"tags": null` rather than `{}` — and tolerates non-string values.
- `BackupList.pagination` is `None` on the final page; the API returns
  `"pagination": null` there.
