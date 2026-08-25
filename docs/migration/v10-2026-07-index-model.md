# 2026-07: IndexModel and index sub-model changes

The index response models now follow the Pinecone `2026-07` API shapes.
This is a breaking change to `IndexModel` and its exports.

For the **request** side — the field-by-field `create_index` /
`configure_index` / `create_index_for_model` tables, the `read_capacity`
keyword warning, and executed before/after code for each create and
configure flow — see
[db_control create/configure](v10-2026-07-db-control.md).

## Control plane now negotiates 2026-07

Every control-plane request (indexes, collections, backups, backup
schedules, restore jobs) now sends `X-Pinecone-Api-Version: 2026-07` — both
`Pinecone` and `AsyncPinecone` read the same constant. This is the atomic
switch that makes all of the `2026-07` request and response shapes in this
document live on the wire at once; there is no per-endpoint version pinning.
One behavioral consequence: `include_deleted=True` on index-scoped backup
listings is honored by `2026-07` servers, whereas the `2025-10` handler
silently ignored the parameter.

## What changed

`IndexModel` no longer has `.spec`, `.embed`, or `.created_at`. Accessing any
of them raises an `AttributeError` that names the replacement.

| Removed | Replacement |
| --- | --- |
| `index.spec.serverless` / `.pod` / `.byoc` | `index.deployment` — a `ManagedDeployment`, `PodDeployment`, or `ByocDeployment` discriminated on `deployment_type` |
| `index.spec.serverless.read_capacity` | `index.read_capacity` (top level) |
| `index.embed` | a `SemanticTextField` in `index.schema.fields` |
| `index.created_at` | not returned by the `2026-07` API |

`.dimension`, `.metric`, and `.vector_type` are deprecated but still work —
they're computed read-only properties resolved from `schema.fields` at
access time, so a `describe_index()` round-trip that used to hard-stop at
the first read of one of these keeps working through the upgrade. They
raise `AttributeError` when the schema doesn't have a resolvable field
(e.g. no dense vector field for `.dimension`), or when it has more than one
candidate field. They will be removed in a later major version.

| Deprecated, still works | Replacement |
| --- | --- |
| `index.dimension` | `index.schema.fields["<field>"].dimension` on the `DenseVectorField` |
| `index.metric` | `index.schema.fields["<field>"].metric` on the vector field |
| `index.vector_type` | field types in `index.schema.fields` (`DenseVectorField` = dense, `SparseVectorField` = sparse) |

New fields on `IndexModel`: `schema` (typed field union), `deployment`
(tagged union), `read_capacity`, `source_collection`, `source_backup_id`,
`cmek_id`.

## Removed exports

`ServerlessSpecInfo`, `PodSpecInfo`, `ByocSpecInfo`, `IndexSpec`, and
`ModelIndexEmbed` are removed from `pinecone`, `pinecone.models`, and
`pinecone.models.indexes`.

## New exports

`IndexSchema`, `IndexSchemaField`, `DenseVectorField`, `SparseVectorField`,
`SemanticTextField`, `StringField`, `StringListField`, `BooleanField`,
`FloatField`, `IntegerField`, `LegacyMetadataField`, `FullTextSearchConfig`,
`NgramConfig`, `IndexDeployment`, `ManagedDeployment`, `PodDeployment`,
`ByocDeployment`, `ReadCapacityResponse`, `ReadCapacityOnDemandResponse`,
`ReadCapacityDedicatedResponse`, `ReadCapacityDedicatedConfig`,
`ReadCapacityStatus`, `ScalingConfigManual`, `CreateIndexRequest`,
`ConfigureIndexRequest`, `IndexStatus`.

## Behavior notes

- `tags` is `None` when an index has no tags — the API returns
  `"tags": null` rather than `{}`.
- Schema-field `description` values are always present in responses and
  `null` when no description was given.
- `ReadCapacityStatus.current_shards` / `.current_replicas` are always
  present and `null` for on-demand read capacity.
- Legacy metadata fields (from indexes that pre-date typed schemas) arrive
  with no `type` key and decode to `LegacyMetadataField`.
- Indexes whose schema uses a field type unknown to this SDK version are
  skipped by `list()` with a warning naming the index; `describe()` raises
  `ResponseParsingError`.

## Operations: create / configure / list (sync client)

The sync index operations now speak the `2026-07` control-plane contract.
Legacy keyword arguments are not silently translated — passing one raises a
`PineconeTypeError` whose message shows the equivalent `2026-07` call with
your own values filled in wherever a faithful translation exists.

### create_index / indexes.create

```python
# 9.x
pc.create_index(
    name="movies",
    dimension=1536,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)

# 10.x
pc.indexes.create(
    name="movies",
    schema={"fields": {"embedding": {
        "type": "dense_vector", "dimension": 1536, "metric": "cosine"}}},
    deployment={"deployment_type": "managed", "cloud": "aws", "region": "us-east-1"},
)
```

- `spec=`, `dimension=`, `metric=`, `vector_type=`, `pods=`, and
  `metadata_config=` are removed. Vector shape lives inside a **named**
  schema field — pick the field name your upsert/query code will address;
  there is no automatic translation because the `2026-07` data plane
  addresses vectors by field name.
- `name` is now optional; the server assigns one if omitted.
- The old metadata `schema=` kwarg changed meaning entirely: the new
  `schema=` declares **searched** fields (`dense_vector`, `sparse_vector`,
  `string` with `full_text_search`). Metadata-only fields are no longer
  declared at create time — they are indexed automatically at upsert, and
  the server rejects them with a 400 if declared.
- `read_capacity` moved to the top level; `cmek_id` is new.
- `source_collection=` / `source_backup_id=` are not exposed: the `2026-07`
  backend rejects both with 400 (`Creating an index from collection or
  backup is not yet supported`). Use `pc.create_index_from_backup(...)` to
  restore a backup.
- A **hybrid** index must declare its `sparse_vector` field. `metric="dotproduct"`
  on the dense field no longer implies sparse support, and the field cannot be
  added later — see [sparse writes now require a declared `sparse_vector`
  field](#sparse-writes).
- Readiness polling stays the default (`timeout=-1` opts out), matching the
  9.x top-level behavior.
- Pod deployments must include all of `environment`, `pod_type`,
  `replicas`, and `shards` — the server rejects omissions with 422.

### create_index_for_model / indexes.create_for_model

Integrated-embedding creation moved from `create(spec=IntegratedSpec(...))`
to a dedicated `pc.indexes.create_for_model(name=..., cloud=..., region=...,
embed={"model": ..., "field_map": {"text": ...}})`. The wire shape is
unchanged from 9.x (`cloud`/`region`/`embed`); the top-level
`pc.create_index_for_model(...)` shim keeps its signature. The embedding
configuration surfaces as a `semantic_text` field in the returned
`index.schema`, named after the `field_map` text entry.

### configure_index / indexes.configure

```python
# 9.x
pc.configure_index("movies", replicas=4, pod_type="p1.x2")

# 10.x
pc.indexes.configure("movies", deployment={"replicas": 4, "pod_type": "p1.x2"})
```

- `replicas=` / `pod_type=` nest under `deployment=` (no `deployment_type`
  key — deployment type, cloud/region, and environment cannot change).
- `embed=` is removed entirely; the 2025-10 convert-to-integrated flow no
  longer exists, and the `2026-07` server rejects unknown PATCH fields.
- `serverless_read_capacity=` and the BYOC-only `read_capacity=` collapsed
  into **one** top-level `read_capacity=` for managed and BYOC indexes.
  Note the `read_capacity` kwarg still exists but its meaning widened.
- `configure()` now returns the updated `IndexModel` (9.x returned `None`).
- Tags stay merge-patch: set a value to `""` to delete that tag key.
- The client no longer restricts `schema=` updates to `semantic_text`
  fields; the server enforces that policy and its error is surfaced
  verbatim.

### list_indexes / indexes.list

`list()` returns a `Paginator[IndexModel]` instead of an `IndexList`.
Iteration keeps working; replace `.names()` with a comprehension:

```python
names = [idx.name for idx in pc.indexes.list()]
```

### exists / has_index

An empty index name now raises `PineconeValueError` instead of returning
`False`.

## Operations: async client (AsyncPinecone.indexes)

`AsyncPinecone.indexes` mirrors every change above one-for-one — same
keyword arguments, same guided `PineconeTypeError` messages for legacy
kwargs, same poll-until-ready default (`timeout=-1` opts out, polling awaits
`asyncio.sleep` so the event loop is never blocked). Async-visible deltas:

- `list()` (and the `pc.list_indexes()` shim) returns an
  `AsyncPaginator[IndexModel]` and is **no longer a coroutine**. Replace
  `(await pc.indexes.list()).names()` with
  `[idx.name async for idx in pc.indexes.list()]`.
- `configure()` returns the updated `IndexModel` (previously `None`).
- `exists("")` now raises `PineconeValueError`; the old async client
  returned `False` for an empty name (the sync client's 2025-10 behavior
  already raised, so the two lanes now agree).
- `create_for_model()` is new on the async namespace, replacing
  `create(spec=IntegratedSpec(...))`.
- The index-scoped backup methods graduated from the preview namespace:
  `create_backup()` and `describe_backup()` are coroutines, and
  `list_backups()` returns an `AsyncPaginator[BackupModel]` (not a
  coroutine), with the same `include_deleted` semantics as the sync lane.
- `await pc.create_index_from_backup(...)` gained `read_capacity`, and the
  legacy `await pc.list_backups(...)` shim gained `include_deleted`,
  matching the sync top-level methods.

## Deprecated request-side spec classes

`ServerlessSpec`, `PodSpec`, `ByocSpec`, `IntegratedSpec`, and `EmbedConfig`
remain importable for one major release so the guided errors can translate
real values, but no create/configure code path accepts them any more.
