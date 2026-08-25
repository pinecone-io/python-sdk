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

## Deprecated arguments

The 2025-10 arguments below are deprecated, keyword-only sugar rather than
a hard break — each still works, translates into the 2026-07 call it's
paired with below, and is expected to be removed in a later major version.
Everything else that changed on the request side (the tables further down)
has no faithful translation and still raises.

| Argument | Surface | Translates into | Notes |
| --- | --- | --- | --- |
| `dimension=` | `create_index` | `schema.fields._values.dimension` | paired with `metric=` |
| `metric=` | `create_index` | `schema.fields.<field>.metric` | dropped for `vector_type="sparse"` |
| `vector_type=` | `create_index` | a `dense_vector` (`_values`) or `sparse_vector` (`_sparse_values`) field | addresses the reserved field name, not one you choose |
| `spec=` (`ServerlessSpec`/`PodSpec`/`ByocSpec`) | `create_index` | `deployment={...}` (+ `read_capacity=` for `ServerlessSpec`/`ByocSpec`) | `spec=IntegratedSpec(...)` is **not** in this list — see below |
| `replicas=` | `configure_index` | `deployment={"replicas": ...}` | mutually exclusive with `deployment=` |
| `pod_type=` | `configure_index` | `deployment={"pod_type": ...}` | mutually exclusive with `deployment=` |
| `serverless_read_capacity=` | `configure_index` | `read_capacity={...}` | now also covers BYOC — see the widened-meaning warning below |

`spec=IntegratedSpec(...)`, `pods=`, `metadata_config=`, `source_collection=`,
`source_backup_id=`, and `configure_index`'s `embed=`/`spec=` have no
faithful translation and still raise a `PineconeTypeError` naming the
equivalent `2026-07` call (`pinecone/_internal/index_migration.py`).

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

`spec=`, `dimension=`, `metric=`, and `vector_type=` are deprecated,
keyword-only sugar: they still work, translating into the
equivalent `schema=`/`deployment=`/`read_capacity=` shown above, addressing
the vector field by the reserved name `_values` (dense) or `_sparse_values`
(sparse) since the SDK cannot invent the field name your data-plane code
will use. `pods=`, `metadata_config=`, `source_collection=`,
`source_backup_id=`, and `spec=IntegratedSpec(...)` have no faithful
translation and still raise a `PineconeTypeError` that interpolates your own
values into the equivalent 2026-07 call
(`pinecone/_internal/index_migration.py`); for `IntegratedSpec` that call is
`create_for_model`. `source_collection` and `source_backup_id` are
intercepted rather than forwarded to a guaranteed `400` — see
[#144](https://github.com/pinecone-io/python-sdk-internal/issues/144). Use
`pc.create_index_from_backup(...)` to restore a backup.

## configure_index / `pc.indexes.configure`

`ConfigureIndexRequest` is `additionalProperties: false` too, so an unknown
PATCH field is rejected rather than ignored.

| 2025-10 request field | 2026-07 replacement |
| --- | --- |
| `spec.pod.{replicas,pod_type}` | `deployment: {replicas, pod_type}` — **no** `deployment_type` key; type, cloud/region and environment cannot change. `replicas=`/`pod_type=` also still work directly, as deprecated sugar translated into `deployment=`. |
| `spec.serverless.read_capacity` | `read_capacity` (top level). `serverless_read_capacity=` also still works, as deprecated sugar translated into `read_capacity=`. |
| `spec.byoc.read_capacity` | `read_capacity` (top level — the same field) |
| `embed` | **none** — the 2025-10 convert-to-integrated flow is gone; embedding is set at create time via `create_for_model` |
| `tags` | `tags` (unchanged; still merge-patch, `""` deletes a key) |
| `deletion_protection` | `deletion_protection` (unchanged) |
| — | `schema` (new — `semantic_text` read/write parameters only) |

`configure()` also **returns the updated `IndexModel`**; the 9.x method
returned `None`. The flat `pc.configure_index()` shim, which delegates to
`configure()`, returns the same `IndexModel` — a deliberate, additive
behavior change from 9.x, not a bug.

```{warning}
**`replicas=`, `pod_type=`, `serverless_read_capacity=`, and `read_capacity=`
are the legacy keywords that did not start raising.**

`embed=` and `spec=` have no 2026-07 PATCH-body destination and raise a
`PineconeTypeError` naming the equivalent 2026-07 call. `replicas=`/
`pod_type=` and `serverless_read_capacity=` do have one, so they remain
available as deprecated keyword-only arguments that translate into
`deployment=`/`read_capacity=` rather than being sent as-is. Passing both
the deprecated keyword and the 2026-07 argument it translates to (e.g.
`replicas=4` together with `deployment=...`) raises a `PineconeValueError`
naming both.

Separately, `read_capacity=` itself *widened its meaning*: 2025-10 had two
read-capacity keywords, `serverless_read_capacity=` for managed indexes and
`read_capacity=` for BYOC only. In 2026-07 a single top-level `read_capacity=`
covers managed **and** BYOC. Code that passed `read_capacity=` intending
"BYOC only" now also takes effect on managed indexes, with no error and no
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
# Deprecated sugar — still works, and still sends the reserved `_values`
# field name (see below), rather than one you choose.
pc.create_index(
    name="movies",
    dimension=1536,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)
```

Pick the field name your upsert and query code will address — there is no
default and the deprecated form above cannot invent one for you.

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
# Deprecated sugar — still works, and still sends the reserved
# `_sparse_values` field name, rather than one you choose.
pc.create_index(
    name="keywords",
    metric="dotproduct",
    vector_type="sparse",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)
```

A `sparse_vector` field takes no `dimension` and no `metric` — both were
implied by `vector_type="sparse"` in 9.x and are dropped rather than
forwarded by the deprecated form above.

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
.add_sparse_vector_field("sparse_terms")`, and puts identical bytes on the
wire.

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
# still works, deprecated — nests automatically under deployment=
pc.configure_index("movies", replicas=4, pod_type="p1.x2")
```

Both keys nest under `deployment=`, which must **not** carry a
`deployment_type` key — the SDK rejects one before the request is sent.
`deployment=` and `replicas=`/`pod_type=` are mutually exclusive; passing both
raises a `PineconeValueError`.

```python
index = pc.indexes.configure("movies", deployment={"replicas": 4, "pod_type": "p1.x2"})
```

## Flow 5 — configure read capacity

```python
# still works, deprecated — managed index
pc.configure_index("movies", serverless_read_capacity={"mode": "OnDemand"})
```

One top-level `read_capacity=` now covers managed and BYOC indexes; read the
warning above before upgrading code that already passes `read_capacity=`.
`read_capacity=` and `serverless_read_capacity=` are mutually exclusive;
passing both raises a `PineconeValueError`. Read capacity does not apply to
pod-based indexes, and changes apply asynchronously — poll
`index.read_capacity.status` rather than assuming the returned model is
settled.

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

(create-limits)=

## Create-time limits with no client-side equivalent

Nothing below is checked before the request is sent, so each arrives as an
`ApiError` carrying the server's message verbatim. Backend citations are
`pinecone-db @ cbee5a67fe`.

### Schema field limits

A field `description` is capped at **256 bytes of UTF-8**, not 256 characters
(`MAX_SCHEMA_FIELD_DESCRIPTION_BYTES` at
`svc-global-apis/src/control_plane/http/handler/global/base/index/validate.rs:23`,
applied to every field type at `:295-333` as `d.len() > 256` on a Rust
`String`). A schema may declare at most **100** `full_text_search` fields
(`validate.rs:227-246`; the bound is a setting whose default is `100` at
`svc-global-apis/src/control_plane/http/mod.rs:134`), and that message names
both counts. The server walks `fields` as an unordered `HashMap`
(`.../base/index_v2/mod.rs:392-394`) and validates each field's name before
its description, so a schema with two offending fields reports an arbitrary
one of them — fix them all rather than resubmitting one at a time.

```python
pc.indexes.create(
    name="movies",
    schema={
        "fields": {
            "embedding": {
                "type": "dense_vector",
                "dimension": 1536,
                "metric": "cosine",
                # The cap is on len(description.encode("utf-8")), so emoji and
                # CJK text reach 256 at a fraction of their character count.
                "description": "Dense embedding of the movie synopsis",
            }
        }
    },
    deployment={"deployment_type": "managed", "cloud": "aws", "region": "us-east-1"},
)
```

### `language` accepts 18 values; `stop_words` is gated to 13 of them

`full_text_search.language` parses against 18 values — `ar da de el en es fi
fr hu it nl no pt ro ru sv ta tr`, each accepting the two-letter code or the
English name, defaulting to `en` (`pc-types/src/index_schema_def.rs:433-489`).
`stop_words: true` additionally requires the language to be in a sealed
13-value set that excludes **`ar`, `el`, `ro`, `ta` and `tr`**
(`.../global/v202607/indexes.rs:60-74`, resolved per API version at
`.../global/mod.rs:34-52`, where a later version may publish a superset but
never a subset). So `language="tr"` on its own is fine, while
`language="tr", stemming=True, stop_words=True` is a `400` — and that message
names the **English** name, not the code you sent: `stop_words is not
supported for language 'turkish'`.

`ngram` does not reject a `language`, it replaces it. The stored config is
built as `FullTextSearchConfig { ngram: Some(..), ..Default::default() }`
(`.../base/index_v2/mod.rs:322-329`) and that default is `Some(en)`
(`index_schema_def.rs:512-518`), so a `language` sent alongside `ngram` is
accepted and the created index reports `en`. An *unparseable* language is
still a `400`, because the parse runs ahead of the `ngram` branch (`:269-281`);
`stemming` or `stop_words` alongside `ngram` is rejected outright (`:292-297`).

```python
pc.indexes.create(
    name="articles",
    schema={
        "fields": {
            # Valid on its own: `tr` is one of the 18 `language` values.
            "body": {"type": "string", "full_text_search": {"language": "tr"}},
            # Accepted, then stored as `en` — `ngram` discards the language.
            "title": {
                "type": "string",
                "full_text_search": {"ngram": {"min_gram": 2, "max_gram": 4}, "language": "tr"},
            },
        }
    },
    deployment={"deployment_type": "managed", "cloud": "aws", "region": "us-east-1"},
)
```

### CMEK is two rules with two status codes

`cmek_id` has a per-request incompatibility *and* the project has a separate
precondition, and they report differently:

- **`400`, per request.** `cmek_id` with a pod deployment, or `cmek_id`
  alongside any `full_text_search` field, is rejected by `validate_cmek_id`
  (`.../base/index/validate.rs:80-100`). This is what `create`'s `cmek_id`
  docstring has always described.
- **`412`, per project.** If the project enforces CMEK encryption, a pod
  deployment or any `full_text_search` field is rejected by
  `validate_encryption_restrictions` (`validate.rs:195-222`) **whether or not
  the request carries a `cmek_id`** — `Index creation failed. Pod indexes are
  not supported in CMEK-encrypted projects` and `Index creation failed.
  Indexes with full_text_search fields are not supported in CMEK-encrypted
  projects`.

The per-request check runs first (`.../base/index_v2/indexes.rs:485-486`), so a
pod request that also carries `cmek_id` reports the `400`; drop the `cmek_id`
and the identical request reports the `412` instead. Neither block above passes
a `cmek_id`, and on a CMEK-encrypted project the `full_text_search` one is a
`412` exactly as written.

### Validation order, because the first failure is the only failure

`validate_create_request` (`.../base/index_v2/indexes.rs:172-192`) runs: name →
`cmek_id` → capacity mode → read capacity → source data → schema field names
and descriptions → `full_text_search` field count → metadata-field
declarations. Only then does `validate_org_and_project_state` (`:486`) reach
the project-level `412`s above.

The `full_text_search` *analyzer* rules — the `ngram` bounds, the
`ngram`-with-`stemming`/`stop_words` rejection, `stop_words` requiring
`stemming`, and the `stop_words` language gate — are **not** part of that
sweep. They run last, while the stored schema is built at `:553`, after the
project `412`s and after the tag merge. A request that is both a pod create on
a CMEK-encrypted project and asks for `stop_words` in Turkish reports the
`412`, never the `400`.

### Tags: `""` deletes a key, and the 20-tag cap counts the merge

`process_tags` (`svc-global-apis/src/commons/tags.rs:80-118`) is the same code
path on create and on configure — create simply merges into an empty map
(`.../base/index_v2/indexes.rs:527` passes `None` as the existing tags).
Three consequences:

- A `""` value means *delete this key*, on create as well as on configure
  (`process_tag`, `:63-76`, maps both `null` and `""` to a removal). On create
  there is nothing to delete, so the key is simply not stored.
- `MAX_TAGS = 20` is checked on the **merged** total, not on the request
  (`:107-112`). Five new tags against an index already carrying 18 is a `400`
  reading `Maximum tags exceeded. 23 tags total requested, maximum of 20
  allowed`, even though the request held five.
- When the merge leaves no tags the index stores no tag map at all rather than
  an empty one (`:113-117`).

The SDK's own checks are narrower and run first: `tags={}` is a
`PineconeValueError` before any request (pass `None` to send no tags), keys
must match `[a-zA-Z0-9_-]{1,80}` and values must be printable ASCII within 120
characters. Because that ASCII check runs client-side, the server's byte-based
key and value limits (`tags.rs:18-59`) can never differ from a character count
for tags this SDK accepts — unlike the schema `description` cap above, which
has no client-side check at all.

```python
# `env` is stored; `owner` is sent and then deleted by the merge, so the new
# index carries exactly one tag.
index = pc.indexes.create(
    name="movies",
    schema={"fields": {"embedding": {"type": "dense_vector", "dimension": 1536, "metric": "cosine"}}},
    deployment={"deployment_type": "managed", "cloud": "aws", "region": "us-east-1"},
    tags={"env": "prod", "owner": ""},
)
```

The same rule on configure, where there *is* something to delete:

:::::{tabs}
::::{tab} Sync

```python
index = pc.indexes.configure("movies", tags={"team": "search", "owner": ""})
```

::::
::::{tab} Async

```python
index = await pc.indexes.configure("movies", tags={"team": "search", "owner": ""})
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
