# 2026-07: BackupModel and backup endpoint changes

The backup response models now follow the Pinecone `2026-07` API shapes.
This is a breaking change to `BackupModel`. The operations themselves change
additively — see [Operations](#operations) below.

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
`pinecone.models`, and `pinecone.models.backups`. It carries the
additive `read_capacity` field, which lets a restore land directly on
dedicated read nodes instead of defaulting to on-demand capacity. Unset
optional fields stay off the wire, so a request built with only `name`
serialises to `{"name": ...}`.

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

```python
# 10.x — recover the backups of an index you already deleted
orphaned = pc.backups.list(index_name="product-search", include_deleted=True)
for backup in orphaned:
    print(backup.backup_id, backup.source_index_deleted_at)
```

A 404 *with* `include_deleted=True` does mean the name has never existed in the
project. Omitting the argument keeps the parameter off the request entirely, so
the server's default (`false`) applies; `include_deleted` on the project-wide
`pc.backups.list()` raises `PineconeValueError` instead of being silently
dropped, because that operation does not accept it.

### `read_capacity` on restore

A restore used to always land on on-demand capacity, needing a follow-up
`configure` to move it onto dedicated read nodes. `read_capacity` does it in
one call:

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

Omitted, the key stays off the wire. `create_index_from_backup` remains the
only supported restore path — `pc.create_index(source_backup_id=...)` raises a
`PineconeTypeError` pointing at it.

### Graduated index-scoped methods

`pc.preview.indexes.create_backup` / `list_backups` / `describe_backup` now
exist on `pc.indexes` with the same signatures, returning the top-level
`BackupModel`. `pc.indexes.list_backups` returns a `Paginator[BackupModel]`
and additionally accepts `include_deleted`. `pc.backups.*` is unchanged apart
from the new `include_deleted` keyword, and remains the only surface for the
project-wide listing and for `delete`.

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
