# 2026-07: db_control create/configure — the deployment/schema break

Release-notes source of truth for the **request side** of the 2026-07 index
control plane: `create_index`, `configure_index`, `create_index_for_model`.
Every removed or renamed field maps to its replacement below, followed by
executed before/after code for the five flows that carry the break.

Response-model changes are not repeated here — see
[IndexModel changes](v10-2026-07-index-model.md),
[BackupModel and backup endpoints](v10-2026-07-backup-models.md), and
[preview namespace removed](v10-2026-07-preview-graduation.md). Backup and
backup-schedule release notes (plan gating, the single-enabled-schedule
conflict, the re-enable side effect, `include_deleted` 404 semantics, and
restoring onto dedicated read nodes) are in
[BackupModel and backup endpoints](v10-2026-07-backup-models.md).

Every 2026-07 block below is executed against a stubbed control plane in
`tests/unit/test_docs_migration_db_control_137.py`, which asserts the method,
the path, and the exact request body. Each example also runs on
`AsyncPinecone` by prefixing `await`: the same test asserts the two lanes put
**byte-identical** bodies on the wire, so every claim here applies to both.
Readiness polling is on by default; `timeout=-1` returns as soon as the
create is accepted.

## create_index / `pc.indexes.create`

`CreateIndexRequest` is now `{schema, deployment, ...}` with
`additionalProperties: false`. The vector shape moved out of top-level
scalars into a **named** schema field, because the 2026-07 data plane
addresses vectors by field name.

| 2025-10 request field | 2026-07 replacement |
| --- | --- |
| `dimension` | `schema.fields.<name>.dimension` on a `dense_vector` field |
| `metric` | `schema.fields.<name>.metric` on the vector field |
| `vector_type: "dense"` / `"sparse"` | a `dense_vector` / `sparse_vector` field in `schema.fields` |
| `spec.serverless.{cloud,region}` | `deployment: {deployment_type: "managed", cloud, region}` |
| `spec.pod.{environment,pod_type,replicas,shards}` | `deployment: {deployment_type: "pod", ...}` — all five keys required, omissions are a `422` |
| `spec.byoc.environment` | `deployment: {deployment_type: "byoc", environment}` |
| `spec.serverless.read_capacity` | `read_capacity` (top level) |
| `spec.serverless.schema` (metadata `{"fields": {"x": {"filterable": true}}}`) | **none** — metadata-only fields are indexed automatically at upsert and are a `400` if declared |
| `spec.pod.metadata_config` | **none** |
| `spec.serverless.source_collection` | **none** — declared in the OAS, rejected unconditionally with `400 Creating an index from collection or backup is not yet supported` |
| `name` (required) | `name` (optional — the server generates one when omitted) |
| — | `cmek_id` (new) |

Nothing is silently translated. Every removed keyword is intercepted before
the request is built and raises a `PineconeTypeError` that interpolates your
own values into the equivalent 2026-07 call
(`pinecone/_internal/index_migration.py`). `source_collection` and
`source_backup_id` are intercepted too rather than forwarded to a guaranteed
`400` — see
[#144](https://github.com/pinecone-io/python-sdk-internal/issues/144). Use
`pc.create_index_from_backup(...)` to restore a backup.

## configure_index / `pc.indexes.configure`

`ConfigureIndexRequest` is `additionalProperties: false` too, so an unknown
PATCH field is rejected rather than ignored.

| 2025-10 request field | 2026-07 replacement |
| --- | --- |
| `spec.pod.{replicas,pod_type}` | `deployment: {replicas, pod_type}` — **no** `deployment_type` key; type, cloud/region and environment cannot change |
| `spec.serverless.read_capacity` | `read_capacity` (top level) |
| `spec.byoc.read_capacity` | `read_capacity` (top level — the same field) |
| `embed` | **none** — the 2025-10 convert-to-integrated flow is gone; embedding is set at create time via `create_for_model` |
| `tags` | `tags` (unchanged; still merge-patch, `""` deletes a key) |
| `deletion_protection` | `deletion_protection` (unchanged) |
| — | `schema` (new — `semantic_text` read/write parameters only) |

`configure()` also **returns the updated `IndexModel`**; the 9.x method
returned `None`.

```{warning}
**`read_capacity=` is the one legacy keyword that did not start raising.**

2025-10 had two read-capacity keywords: `serverless_read_capacity=` for
managed indexes and `read_capacity=` for BYOC only. In 2026-07 they collapse
into a single top-level `read_capacity=` covering managed **and** BYOC.

`serverless_read_capacity=` now raises a `PineconeTypeError` naming the
replacement, so that half is loud. `read_capacity=` still exists and still
type-checks — its *meaning widened*. Code that passed it intending "BYOC
only" now also takes effect on managed indexes, with no error and no
deprecation warning. Audit call sites that pass it before upgrading.
```

## create_index_for_model / `pc.indexes.create_for_model`

**No request-side break.** The wire shape stays the 9.x
`{name, cloud, region, embed: {model, field_map, ...}}` form: 2026-07 routes
this one operation to the v202604 handler, whose request struct is
`#[serde(deny_unknown_fields)]` over exactly those keys
(`.../global/v202604/indexes.rs:565-576`, routed from
`v202607/indexes.rs:43,229`, @ pinecone-db `cbee5a67fe`). The published
`_build` 2026-07 OAS declares a flat `{deployment, field, model, ...}` shape
here; that build is stale and the backend accepts only the legacy form, so
the SDK sends the legacy form —
[#206](https://github.com/pinecone-io/python-sdk-internal/issues/206).

The **response** does change: the embedding configuration comes back as a
`semantic_text` field in `index.schema`, named after the `field_map` text
entry, instead of `index.embed`.

## Flow 1 — dense serverless create

```python
# 9.x
pc.create_index(
    name="movies",
    dimension=1536,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)
```

Pick the field name your upsert and query code will address — there is no
default and the SDK will not invent one.

:::::{tabs}
::::{tab} Sync

```python
pc.indexes.create(
    name="movies",
    schema={
        "fields": {
            "embedding": {"type": "dense_vector", "dimension": 1536, "metric": "cosine"}
        }
    },
    deployment={"deployment_type": "managed", "cloud": "aws", "region": "us-east-1"},
)
```

::::
::::{tab} Async

```python
await pc.indexes.create(
    name="movies",
    schema={
        "fields": {
            "embedding": {"type": "dense_vector", "dimension": 1536, "metric": "cosine"}
        }
    },
    deployment={"deployment_type": "managed", "cloud": "aws", "region": "us-east-1"},
)
```

::::
:::::

## Flow 2 — sparse create

```python
# 9.x
pc.create_index(
    name="keywords",
    metric="dotproduct",
    vector_type="sparse",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)
```

A `sparse_vector` field takes no `dimension` and no `metric` — both were
implied by `vector_type="sparse"` in 9.x and are not accepted here.

```python
pc.indexes.create(
    name="keywords",
    schema={"fields": {"sparse_terms": {"type": "sparse_vector"}}},
    deployment={"deployment_type": "managed", "cloud": "aws", "region": "us-east-1"},
)
```

```{warning}
**Hybrid indexes must now declare a sparse field.** In 2025-10 a dense index
with `metric="dotproduct"` implicitly accepted sparse values on upsert. In
2026-07 the backend gates sparse writes on the schema actually declaring a
`sparse_vector` field — `supports_sparse_vector_writes()` returns
`schema.has_sparse_vector_field()` whenever a schema is present
(`pc-types/src/cps.rs:682-687` @ pinecone-db `cbee5a67fe`) — and 2026-07
`POST /indexes` always persists a schema
(`.../base/index_v2/indexes.rs:553-557`). So `metric="dotproduct"` alone no
longer buys sparse writes; declare both fields:

    schema={"fields": {
        "embedding": {"type": "dense_vector", "dimension": 1536, "metric": "dotproduct"},
        "sparse_terms": {"type": "sparse_vector"},
    }}

`SchemaBuilder` spells the same thing as
`.add_dense_vector_field("embedding", dimension=1536, metric="dotproduct")
.add_sparse_vector_field("sparse_terms")` — but use the dict form for now:
`add_sparse_vector_field()` emits a `metric` key the server rejects
([#350](https://github.com/pinecone-io/python-sdk-internal/issues/350)).

This fails *silently at create time* — the index is created and only the
sparse upserts are refused later. Full write-up, including which error a
caller actually sees today: [sparse writes now require a declared
`sparse_vector` field](#sparse-writes).
```

## Flow 3 — full-text-search create

No 9.x equivalent: full-text search has no shape in the 2025-10 or 2026-04
API. It existed only on the `2026-01.alpha` preview surface, whose callers
should read the
[preview-graduation guide](v10-2026-07-preview-graduation.md).

A `string` field is accepted only **with** a `full_text_search` object; an
empty object selects the defaults (`language: "en"`, no stemming).

```python
pc.indexes.create(
    name="articles",
    schema={
        "fields": {"body": {"type": "string", "full_text_search": {"language": "en", "stemming": True}}}
    },
    deployment={"deployment_type": "managed", "cloud": "aws", "region": "us-east-1"},
)
```

`stop_words: true` requires `stemming: true`, and `ngram` cannot be combined
with either — each is a separate `400`. `cmek_id` is incompatible with any
`full_text_search` field.

## Flow 4 — configure replicas (pod-based)

```python
# 9.x
pc.configure_index("movies", replicas=4, pod_type="p1.x2")
```

Both keys nest under `deployment=`, which must **not** carry a
`deployment_type` key — the SDK rejects one before the request is sent.

```python
index = pc.indexes.configure("movies", deployment={"replicas": 4, "pod_type": "p1.x2"})
```

## Flow 5 — configure read capacity

```python
# 9.x — managed index
pc.configure_index("movies", serverless_read_capacity={"mode": "OnDemand"})
```

One top-level `read_capacity=` now covers managed and BYOC indexes; read the
warning above before upgrading code that already passes `read_capacity=`.
Read capacity does not apply to pod-based indexes, and changes apply
asynchronously — poll `index.read_capacity.status` rather than assuming the
returned model is settled.

:::::{tabs}
::::{tab} Sync

```python
index = pc.indexes.configure(
    "movies",
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
index = await pc.indexes.configure(
    "movies",
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

## Data-plane note

The flows above create indexes and deliberately stop there rather than
showing an upsert or query against the new index. On 2026-07 every
`POST /indexes` result carries a persisted schema, and how the vector API
behaves against such an index is an **open question** —
[#322](https://github.com/pinecone-io/python-sdk-internal/issues/322) is
tracking it with live observations on both REST and gRPC. Do not read the
absence of a data-plane example as a statement either way: the vector-API
gate runs ahead of every other write check
(`pc-validation/src/data_plane/mod.rs:495` @ pinecone-db `cbee5a67fe`), so
whatever #322 concludes determines what a freshly created index accepts.
