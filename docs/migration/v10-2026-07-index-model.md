# 2026-07: IndexModel and index sub-model changes

The index response models now follow the Pinecone `2026-07` API shapes.
This is a breaking change to `IndexModel` and its exports.

## What changed

`IndexModel` no longer has `.dimension`, `.metric`, `.vector_type`, `.spec`,
`.embed`, or `.created_at`. Accessing any of them raises an `AttributeError`
that names the replacement. The model now carries:

| Removed | Replacement |
| --- | --- |
| `index.dimension` | `index.schema.fields["<field>"].dimension` on the `DenseVectorField` |
| `index.metric` | `index.schema.fields["<field>"].metric` on the vector field |
| `index.vector_type` | field types in `index.schema.fields` (`DenseVectorField` = dense, `SparseVectorField` = sparse) |
| `index.spec.serverless` / `.pod` / `.byoc` | `index.deployment` — a `ManagedDeployment`, `PodDeployment`, or `ByocDeployment` discriminated on `deployment_type` |
| `index.spec.serverless.read_capacity` | `index.read_capacity` (top level) |
| `index.embed` | a `SemanticTextField` in `index.schema.fields` |
| `index.created_at` | not returned by the `2026-07` API |

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

The request-side spec classes (`ServerlessSpec`, `PodSpec`, `ByocSpec`,
`IntegratedSpec`, `EmbedConfig`) are unchanged for now; the create/configure
call-shape migration is tracked separately.
