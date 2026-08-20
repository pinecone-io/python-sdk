# 2026-07: BackupModel and backup request changes

The backup response models now follow the Pinecone `2026-07` API shapes.
This is a breaking change to `BackupModel`.

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
