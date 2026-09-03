# Migrating to V10

The Pinecone `2026-07` API is what the `10.x` Python SDK speaks. This guide
covers what changed for SDK users moving from `9.x`, organized by what you're
doing rather than by which surface happened to change.

**Where to start:** if you were on `pc.preview.*`, start with
[Preview namespace removed](#preview-graduation): the preview surface is gone
outright, with no shim and no deprecation window. If you were on the stable
`9.x` surface (`pc.create_index(dimension=..., metric=..., spec=...)`), skip
that section and start with [Index creation and configuration](#index-model).

## Contents

- [Import paths](#import-paths)
- [Documents API](#documents-api)
  - [Preview namespace removed](#preview-graduation)
  - [Index creation and configuration](#index-model)
- [Vector data (db_data)](#vector-data)
  - [Model and wire-format changes](#vector-models)
  - [Breaking changes and migration table](#db-data-breaking-changes)
- [Bulk ingest: concurrency and deadlines](#bulk-ingest)
  - [`max_concurrency` defaults to 8](#bulk-max-concurrency)
  - [The admission gate can refuse batches](#bulk-backpressure)
  - [`total_timeout` bounds admission, not requests](#bulk-total-timeout)
- [Backups and backup schedules](#backup-models)
- [Assistant](#assistant-models)
  - [File uploads and deletes are operations](#assistant-files)
- [Inference](#inference-model-enums)
- [Admin and OAuth](#admin-oauth)
- [TLS/SSL configuration](#ssl-config)
  - [The gRPC endpoint scheme](#grpc-scheme)

(import-paths)=
## Import paths

Almost nothing here needs editing, so it's worth settling first. The
top-level `pinecone` package is the canonical import surface, and every name
this guide mentions is reachable from it — models, specs, `SchemaBuilder`,
`Admin`, and the whole exception hierarchy included. `pinecone.models` also
carries every model name.

The module paths that pre-date the package layout still resolve, so a `9.x`
import line is not by itself something to fix:

| Import | Names it still resolves |
| --- | --- |
| `from pinecone.pinecone import Pinecone` | `Pinecone` |
| `from pinecone.exceptions import ...` | the `*Exception` aliases (`PineconeException`, `NotFoundException`, `PineconeApiException`, and the rest) |
| `from pinecone.control import ...` | the control-plane models and specs (`ServerlessSpec`, `PodSpec`, `IndexModel`, `IndexList`, `BackupModel`, the enums) |
| `from pinecone.data import ...` | `Index`, `AsyncIndex`, `Vector`, `SparseValues`, `QueryResponse`, `UpsertResponse`, `FetchResponse`, `DescribeIndexStatsResponse`, `ImportErrorMode`, `SearchQuery`, `SearchRerank` |

Three paths are the exception. `pinecone.config`, `pinecone.db_control`, and
`pinecone.db_data` import as modules but export nothing usable, so a name
read out of any of them raises:

```python
from pinecone.config import Config          # ImportError
from pinecone.db_control import ServerlessSpec   # ImportError
from pinecone.db_data import Index          # ImportError
```

Import those three from `pinecone` instead: `ServerlessSpec`, `Index`, and
`PineconeConfig` are all top-level. Apart from the `Preview*` names covered
next, that is the whole of the import work the upgrade asks of you.

(documents-api)=
## Documents API

Preview graduated into this release, and every 2026-07 index (dense, sparse, hybrid, or full-text-search) is created with a schema and served through the documents API. This section covers migrating off preview and what changed about creating and configuring an index.

(preview-graduation)=
### Preview namespace removed

The `2026-01.alpha` preview surface graduated. Everything that lived under
`pc.preview` is now a first-class part of the SDK, and the `pinecone/preview/`
package is deleted outright. There's no shim and no deprecation window.
Preview was never covered by SemVer, and its docstrings said so.

Every stale preview import or attribute access raises immediately, at the
point of use, so the migration is mechanical to find:

| Stale code | Now raises |
| --- | --- |
| `import pinecone.preview` | `ModuleNotFoundError` |
| `from pinecone.preview.models import PreviewIndexModel` | `ModuleNotFoundError` |
| `pc.preview` on `Pinecone` or `AsyncPinecone` | `AttributeError` |
| `from pinecone import PreviewIndexModel` (or any `Preview*` name) | `ImportError` |

#### Entry points

The graduated surface hangs off the client directly. `pc.preview` was one
extra hop; delete it and the rest of the expression is almost unchanged.

| Removed | Replacement |
| --- | --- |
| `pc.preview.indexes` | `pc.indexes` |
| `pc.preview.index(name=...)` / `(host=...)` | `pc.index(name=...)` / `(host=...)`, or positionally `pc.index("my-index")` |
| `pc.preview.index(...).documents.upsert(...)` | `pc.index(...).documents.upsert(...)` |
| `pc.preview.close()` | `pc.close()` |
| `from pinecone.preview import SchemaBuilder` | `from pinecone import SchemaBuilder` |

```python
from pinecone import Pinecone

pc = Pinecone(api_key="your-api-key")

# was: pc.preview.indexes.describe("articles-en-preview")
info = pc.indexes.describe("articles-en")

# was: index = pc.preview.index(name="articles-en-preview")
with pc.index(name="articles-en") as index:
    result = index.documents.fetch(namespace="articles-en", ids=["doc-1"])
```

##### `await pc.index(...)` is now a coroutine

`AsyncPreview.index()` was synchronous: it handed back an index immediately
and resolved the host lazily, on the first data-plane call. `AsyncPinecone.index()`
resolves eagerly and must be awaited.

```python
import asyncio
from pinecone import AsyncPinecone


async def main() -> None:
    pc = AsyncPinecone(api_key="your-api-key")
    # was: index = pc.preview.index(name="articles-en-preview")
    index = await pc.index(name="articles-en")
    async with index:
        await index.documents.fetch(namespace="articles-en", ids=["doc-1"])
    await pc.close()


asyncio.run(main())
```

Two consequences follow from that:

A bad index name now fails at `pc.index(...)`, not on the first data-plane
call. If you were catching `NotFoundError` around your first `search`/`fetch`,
move the `try` up to the `await pc.index(...)`. And `pc.index(...)` can't be
called outside a running event loop, so module-level `index = pc.preview.index(name=...)`
no longer has an equivalent.

Targeting by `host=` still skips the control-plane round trip, so
`await pc.index(host=...)` never raises `NotFoundError`. The `await` is cheap
there, but still required.

#### Model imports

Every `Preview*` model dropped its prefix and moved out of
`pinecone.preview.models`. All of the replacements are importable from
`pinecone` and from `pinecone.models`.

##### Straight renames

Find and replace:

| Removed | Replacement |
| --- | --- |
| `PreviewIndexModel` | `IndexModel` |
| `PreviewIndexStatus` | `IndexStatus` |
| `PreviewSchema` | `IndexSchema` |
| `PreviewSchemaField` | `IndexSchemaField` |
| `PreviewDenseVectorField` | `DenseVectorField` |
| `PreviewStringField` | `StringField` |
| `PreviewStringListField` | `StringListField` |
| `PreviewBooleanField` | `BooleanField` |
| `PreviewDeployment` | `IndexDeployment` |
| `PreviewManagedDeployment` | `ManagedDeployment` |
| `PreviewReadCapacity` | `ReadCapacityResponse` |
| `PreviewReadCapacityStatus` | `ReadCapacityStatus` |
| `PreviewReadCapacityOnDemandResponse` | `ReadCapacityOnDemandResponse` |
| `PreviewReadCapacityDedicatedResponse` | `ReadCapacityDedicatedResponse` |
| `PreviewReadCapacityManualScaling` | `ScalingConfigManual` |
| `PreviewCreateIndexRequest` | `CreateIndexRequest` |
| `PreviewConfigureIndexRequest` | `ConfigureIndexRequest` |
| `PreviewSparseValues` | `SparseValues` |
| `PreviewDocument` | `Document` |
| `PreviewDocumentUpsertResponse` | `UpsertDocumentsResponse` |
| `PreviewTextQuery` | `TextQuery` |
| `PreviewDenseVectorQuery` | `DenseVectorQuery` |
| `PreviewSparseVectorQuery` | `SparseVectorQuery` |
| `PreviewScoreByQuery` | `DocumentScoringMethod` |

`PreviewSparseValues` → `SparseValues` and `PreviewIndexStatus` → `IndexStatus`
each gained dict-style access (`status["ready"]` alongside `status.ready`).
Nothing was taken away.

```python
from pinecone import (
    BooleanField,
    DenseVectorField,
    DenseVectorQuery,
    Document,
    DocumentScoringMethod,
    IndexDeployment,
    IndexModel,
    IndexSchema,
    IndexSchemaField,
    IndexStatus,
    ManagedDeployment,
    ReadCapacityResponse,
    ScalingConfigManual,
    SparseValues,
    SparseVectorQuery,
    StringField,
    StringListField,
    TextQuery,
    UpsertDocumentsResponse,
)
```

##### `PreviewIntegerField` and `PreviewLegacyIntegerField` cross over

The two preview classes carried each other's wire tags. `PreviewIntegerField`
was tagged `"float"` and `PreviewLegacyIntegerField` was tagged `"integer"`.
The graduated names are correct, which means a blind find-and-replace gives
you the wrong type:

| Removed | Wire `type` | Replacement |
| --- | --- | --- |
| `PreviewIntegerField` | `"float"` | `FloatField` |
| `PreviewLegacyIntegerField` | `"integer"` | `IntegerField` |

If you wrote `isinstance(field, PreviewIntegerField)` to find whole-number
fields in a schema, that check was already matching floats. It needs to
become `isinstance(field, IntegerField)`.

##### Renames where the shape also changed

| Removed → Replacement | What changed |
| --- | --- |
| `PreviewDenseVectorField` → `DenseVectorField` | `dimension` and `metric` are required, no longer defaulted to `None` |
| `PreviewSparseVectorField` → `SparseVectorField` | `metric` removed; sparse fields have no configurable metric |
| `PreviewSemanticTextField` → `SemanticTextField` | `model` is required; `dimension` removed. Can't be declared at create time on `2026-07`, it appears in responses only |
| `PreviewFullTextSearchConfig` → `FullTextSearchConfig` | `lowercase` and `max_term_len` removed; gained `ngram: NgramConfig \| None` |
| `PreviewPodDeployment` → `PodDeployment` | `replicas` and `shards` are required; `pods` removed. The type still describes existing pod indexes, but `2026-07` won't create one — see [Pod deployments, and what that means for collections](#pod-collections) |
| `PreviewByocDeployment` → `ByocDeployment` | `cloud` and `region` removed, only `environment` remains |
| `PreviewQueryStringQuery` → `QueryStringQuery` | gained `field` and `fields`, so a query-string clause can be scoped to named fields |
| `PreviewUsage` → three types | split by operation, see below |
| `PreviewDocumentSearchResponse` → `SearchDocumentsResponse` | `usage` is now `DocumentSearchUsage` |
| `PreviewDocumentFetchResponse` → `FetchDocumentsResponse` | `usage` is now `DocumentFetchUsage`; gained `pagination` |

`PreviewUsage` was one struct shared by the search and fetch responses. It's
now three, one per operation:

| Response | Usage type |
| --- | --- |
| `SearchDocumentsResponse` | `DocumentSearchUsage` |
| `FetchDocumentsResponse` | `DocumentFetchUsage` |
| `ListDocumentsResponse` | `DocumentListUsage` |

All three carry the same single `read_units: int` field, so
`response.usage.read_units` is unchanged. Only a type annotation or an
`isinstance` check needs updating.

New model names with no preview predecessor: `FloatField`, `LegacyMetadataField`,
`NgramConfig`, `FullTextSearchConfig` (as a top-level export), `DocumentRecord`,
`UpdateDocumentRecord`, `ListedDocumentRecord`, `DeleteDocumentsResponse`,
`UpdateDocumentsResponse`, `ListDocumentsResponse`, `DocumentSearchUsage`,
`DocumentFetchUsage`, `DocumentListUsage`, and the six `*DocumentsRequest` structs.
`PreviewBackupModel` and `PreviewCreateBackupRequest` also disappear with the
package; see [backup models](#backup-models) for that mapping.

#### The `.documents` namespace

`pc.preview.index(...)` returned a wrapper whose only job was to hold a
`.documents` proxy. Graduating out of preview retired the wrapper, and the
`.documents` namespace carries over as the surface for document operations:
`index.documents` is a lazily-instantiated property, same as every other
resource namespace on this SDK (`pc.indexes`, `pc.inference`, and so on).

| Preview | Now |
| --- | --- |
| `index.documents.upsert(...)` | `index.documents.upsert(...)` |
| — | `index.documents.batch_upsert(...)` (new) |
| `index.documents.search(...)` | `index.documents.search(...)` |
| `index.documents.fetch(...)` | `index.documents.fetch(...)` |
| `index.documents.delete(...)` | `index.documents.delete(...)` |
| — | `index.documents.update(...)` (new) |
| — | `index.documents.list(...)` (new) |

Every keyword argument preview accepted is still accepted under the same
name, so for most calls only the access path changes. `AsyncIndex.documents`
mirrors all seven; `documents.list(...)` isn't a coroutine on either lane, it
returns a paginator.

```python
from pinecone import Pinecone

pc = Pinecone(api_key="your-api-key")
with pc.index(name="articles-en") as index:
    index.documents.upsert(
        namespace="articles-en",
        documents=[{"_id": "doc-1", "title": "Rome", "views": 12}],
    )

    hits = index.documents.search(
        namespace="articles-en",
        top_k=10,
        score_by=[{"type": "text", "query": "roman aqueducts", "field": "title"}],
        include_fields=["title"],
    )
    for match in hits.matches:
        print(match.id, match.title)
```

```python
import asyncio
from pinecone import AsyncPinecone


async def main() -> None:
    pc = AsyncPinecone(api_key="your-api-key")
    index = await pc.index(name="articles-en")
    async with index:
        await index.documents.upsert(
            namespace="articles-en",
            documents=[{"_id": "doc-1", "title": "Rome", "views": 12}],
        )
        hits = await index.documents.search(
            namespace="articles-en",
            top_k=10,
            score_by=[{"type": "text", "query": "roman aqueducts", "field": "title"}],
            include_fields=["title"],
        )
        for match in hits.matches:
            print(match.id, match.title)
    await pc.close()


asyncio.run(main())
```

##### `documents.delete` returns a response object

Preview's `index.documents.delete(...)` returned `None`. The graduated
version returns a `DeleteDocumentsResponse`, and it accepts a `filter`, which
is why there's now something to return.

```python
# preview returned None:
#   index.documents.delete(namespace="articles-en", ids=["doc-1"])
response = index.documents.delete(namespace="articles-en", ids=["doc-1"])

# new: delete by filter, with a count of what matched
response = index.documents.delete(
    namespace="articles-en",
    filter={"views": {"$lt": 5}},
)
print(response.matched_records)
```

`matched_records` is `None` for an ID-list or `delete_all` delete. For a
filtered delete it's the point-in-time count when the server accepted the
request. The delete applies asynchronously behind a `202`, so it isn't a
promise about how many documents ultimately disappear. `matched_records` on
`UpdateDocumentsResponse` means the same thing.

Preview's `ids` / `delete_all` mutual exclusion still holds, widened to three
options: exactly one of `ids`, `filter`, or `delete_all` must be given.

##### `documents.fetch` gained a filter and pagination

Preview's `fetch` took `ids` only. `documents.fetch` takes exactly one of
`ids` or `filter`, and a filtered fetch is paginated. The server fixes the
page size, so there's no page-size argument to pass — follow
`response.pagination.next` until it's `None`.

```python
page = index.documents.fetch(namespace="articles-en", filter={"views": {"$gt": 100}})
while True:
    for doc_id, doc in page.documents.items():
        print(doc_id, doc.title)
    if page.pagination is None:
        break
    page = index.documents.fetch(
        namespace="articles-en",
        filter={"views": {"$gt": 100}},
        pagination_token=page.pagination.next,
    )
```

An ID-based fetch is never paginated, so `response.pagination` stays `None`
and existing `fetch` call sites need no loop.

##### Smaller signature deltas

`timeout` is a new keyword on every document method, `None` by default,
matching the previous behavior. `documents.batch_upsert(max_concurrency=...)`
defaults to `None`, which lets the host's admission gate use its own default
of 8; pass a number to set your own ceiling, which the gate still holds
effective concurrency below whenever the backend is pushing back.
`documents.batch_upsert` dropped `**kwargs`, so a typo is now a `TypeError`
at the call site instead of being silently swallowed. `documents=` accepts
`Sequence[Mapping[...]]`, not just `list[dict]`, and also accepts typed
`DocumentRecord` / `UpdateDocumentRecord` objects; lists of dicts keep working.

`include_fields` defaults in opposite directions on the two read operations,
which is worth pinning down before you write either call. On `search`,
omitting it (or passing `[]`) returns only `_id` and `_score`, so a hit
carries none of your own fields and `match.title` raises `AttributeError`.
On `fetch`, omitting it returns every field. Name the fields you're going to
read on a `search`, or pass `["*"]` for all of them.

`GrpcIndex` has no document operations. The `2026-07` documents surface is
REST-only.

#### Index operations

All nine method names carry over unchanged: `create`, `configure`, `describe`,
`list`, `exists`, `delete`, `create_backup`, `list_backups`, `describe_backup`.
Deleting `preview.` from the attribute chain is most of the change.
`AsyncPinecone.indexes` mirrors all nine, with `list` and `list_backups`
non-coroutine on both lanes.

```python
from pinecone import Pinecone, SchemaBuilder

pc = Pinecone(api_key="your-api-key")

schema = (
    SchemaBuilder()
    .add_dense_vector_field("embedding", dimension=1536, metric="cosine")
    .build()
)

# was: pc.preview.indexes.create(schema=..., deployment=...)
index = pc.indexes.create(
    name="articles-en",
    schema=schema,
    deployment={"deployment_type": "managed", "cloud": "aws", "region": "us-east-1"},
)
print(index.status.ready)
```

`SchemaBuilder.build()` already returns the `{"fields": {...}}` envelope, so
pass its result straight to `schema=`.

For a preview caller, four things changed: `create()` now polls until the
index is ready by default (`timeout=-1` restores the old return-immediately
behavior); `create()` no longer accepts `source_collection` or
`source_backup_id` (the backend rejects both with a 400, use
`pc.create_index_from_backup(...)` to restore a backup instead); `create()`
and `configure()` accept a typed `IndexSchema` as well as a plain dict; and
`create_for_model()` is new, replacing the old integrated-embedding path.

Everything else about the `2026-07` create/configure contract, including the
schema rules and field removals, is the same for you as for a `9.x` caller
and is covered in [Index creation and configuration](#index-model) below.

#### Newly importable from `pinecone`

Independent of the preview retirement, this release wired up top-level
exports that were built earlier in the cycle but never reachable from
`pinecone` or `pinecone.models`. If you were reaching into
`pinecone.models.admin.*` or `pinecone.models.assistant.*` directly, those
paths still work too.

| Area | Names |
| --- | --- |
| Assistant operations | `OperationModel`, `ListOperationsResponse` |
| Admin: users | `UserModel`, `UserList` |
| Admin: invites | `InviteModel`, `InviteList`, `InviteStatus` |
| Admin: service accounts | `ServiceAccountModel`, `ServiceAccountList`, `ServiceAccountWithSecret` |
| Admin: role bindings | `RoleBindingModel`, `RoleBindingList`, `RoleBindingInput`, `RoleName`, `PrincipalType`, `ResourceType` |
| Admin: pagination | `PaginationResponse` |

```python
from pinecone import (
    InviteList,
    InviteModel,
    InviteStatus,
    ListOperationsResponse,
    OperationModel,
    PaginationResponse,
    PrincipalType,
    ResourceType,
    RoleBindingInput,
    RoleBindingList,
    RoleBindingModel,
    RoleName,
    SchemaBuilder,
    ServiceAccountList,
    ServiceAccountModel,
    ServiceAccountWithSecret,
    UserList,
    UserModel,
)
```

`OperationModel` pairs with the assistant file lifecycle, see
[Assistant](#assistant-files). The admin RBAC models belong to the separate
`Admin` client — `admin.users` / `.invites` / `.service_accounts` /
`.role_bindings`. `Pinecone` has no `admin` attribute; `Admin` authenticates
with OAuth2 client credentials rather than an API key, so it's constructed on
its own. See [Admin and OAuth](#admin-oauth).

(index-model)=
### Index creation and configuration

The `2026-07` control plane addresses vectors by field name instead of
top-level scalars, and every index carries a persisted schema. This section
covers everything about creating and configuring an index: what the response
model (`IndexModel`) looks like now, what the request shape (`create`,
`configure`) looks like now, and the deprecated `9.x` arguments that still
work as sugar for the new shape.

Every control-plane request (indexes, collections, backups, backup schedules,
restore jobs) now sends `X-Pinecone-Api-Version: 2026-07`.

#### What changed on `IndexModel`

`.spec` and `.embed` are deprecated but still work. Both are computed
properties, rebuilt on each access from `deployment`, `read_capacity`,
`schema`, and `source_collection`, so reads like
`index.spec.serverless.region`, `index.spec.pod.pod_type`, and
`index.embed.model` need no editing. Nothing decodes into those classes; they
are views over fields that live elsewhere now, and will be removed in a later
major version.

| Deprecated, still works | Replacement |
| --- | --- |
| `index.spec.serverless` / `.pod` / `.byoc` | `index.deployment`, a `ManagedDeployment`, `PodDeployment`, or `ByocDeployment` discriminated on `deployment_type` |
| `index.spec.serverless.read_capacity` | `index.read_capacity` (top level) |
| `index.embed` | a `SemanticTextField` in `index.schema.fields` |

Being views, they can only report what `2026-07` carries:

- `spec.pod.metadata_config` is always `None`. Metadata fields are indexed
  automatically at upsert, so there is no such configuration to return.
- `spec.pod.pods` is `replicas * shards`, the same identity the create path
  enforces when translating a `9.x` `pods=`.
- `spec.serverless.schema` and `spec.byoc.schema` hold the typed `2026-07`
  schema as a dict. In `9.x` that key held the metadata-indexing schema and
  defaulted to `None`; the `2026-07` schema declares every field, vector
  fields included.
- `embed.dimension` and `embed.vector_type` are always `None`. A
  `semantic_text` field reports neither the width of the vectors it produces
  nor whether its model is dense or sparse.
- `index.embed` is `None` for an index with no semantic text field, which is
  what `9.x` reported for a non-integrated index. A schema with more than one
  raises an `AttributeError` naming them: as with `metric`, there is no single
  field to resolve to.

`.created_at` is gone, and accessing it raises an `AttributeError` that names
the reason.

| Removed | Replacement |
| --- | --- |
| `index.created_at` | not returned by the `2026-07` API |

`.dimension`, `.metric`, and `.vector_type` are deprecated but still work.
They're computed properties resolved from `schema.fields` at access time, and
will be removed in a later major version.

| Deprecated, still works | Replacement |
| --- | --- |
| `index.dimension` | `index.schema.fields["<field>"].dimension` on the `DenseVectorField` |
| `index.metric` | `index.schema.fields["<field>"].metric` on the vector field |
| `index.vector_type` | field types in `index.schema.fields` (`DenseVectorField` = dense, `SparseVectorField` = sparse) |

These accessors are resolved from whatever `DenseVectorField`/`SparseVectorField`
entries are in `schema.fields`, so the result depends on the schema shape. A
`9.x` index only ever had one vector configuration, so schemas with more than
one dense or sparse field are a `2026-07`-only possibility with no `9.x`
equivalent:

| `schema.fields` shape | `9.x` `dimension` / `metric` / `vector_type` | Current accessors |
| --- | --- | --- |
| 1 dense field | `<int>` / `<str>` / `"dense"` | same — resolved from the dense field |
| 1 dense field + any number of sparse fields | not representable (sparse values needed no schema field) | same as above — the dense field wins, sparse fields are ignored entirely |
| 0 dense, 1 sparse field | `None` / `"dotproduct"` / `"sparse"` | same |
| 0 dense, 2+ sparse fields | not representable | `dimension` is still `None`; `metric` and `vector_type` raise `AttributeError` ("ambiguous") |
| 2+ dense fields (any sparse count) | not representable | all three raise `AttributeError` ("ambiguous"), naming the dense fields |
| 0 dense, 0 sparse fields | not representable (every real index has one) | all three raise `AttributeError` ("no dense or sparse vector fields") |

New fields on `IndexModel`: `schema`, `deployment`, `read_capacity`,
`source_collection`, `source_backup_id`, `cmek_id`. `source_collection` is a
response field only, and no `2026-07` create path populates it: see
[Pod deployments, and what that means for collections](#pod-collections).

Deprecated exports, still importable from `pinecone` — these are the classes
`.spec` and `.embed` build: `ServerlessSpecInfo`, `PodSpecInfo`,
`ByocSpecInfo`, `IndexSpec`, `ModelIndexEmbed`.

New exports: `IndexSchema`, `IndexSchemaField`,
`DenseVectorField`, `SparseVectorField`, `SemanticTextField`, `StringField`,
`StringListField`, `BooleanField`, `FloatField`, `IntegerField`,
`LegacyMetadataField`, `FullTextSearchConfig`, `NgramConfig`,
`IndexDeployment`, `ManagedDeployment`, `PodDeployment`, `ByocDeployment`,
`ReadCapacityResponse`, `ReadCapacityOnDemandResponse`,
`ReadCapacityDedicatedResponse`, `ReadCapacityDedicatedConfig`,
`ReadCapacityStatus`, `ScalingConfigManual`, `CreateIndexRequest`,
`ConfigureIndexRequest`, `IndexStatus`.

A few smaller response behaviors: `tags` is `None` when an index has no tags,
not `{}`. Schema-field `description` is always present and `null` when no
description was given. Legacy metadata fields, from indexes that pre-date
typed schemas, decode to `LegacyMetadataField`. Indexes whose schema uses a
field type unknown to this SDK version are skipped by `list()` with a
warning; `describe()` raises `ResponseParsingError`.

`list()` also returns a `Paginator[IndexModel]` instead of an `IndexList`
now. Iteration keeps working; replace `.names()` with a comprehension:

```python
names = [idx.name for idx in pc.indexes.list()]
```

An empty index name now raises `PineconeValueError` from `exists()` /
`has_index()` instead of returning `False`.

(db-control)=
#### create and configure

`dimension=`, `metric=`, `vector_type=`, and `spec=` are deprecated,
keyword-only sugar. They still work, on both `pc.create_index()` and
`pc.indexes.create()`: they translate into the `schema=`/`deployment=` call
below, addressing the vector by the reserved `_values` (dense) or
`_sparse_values` (sparse) field name, since the SDK can't invent the field
name your own data-plane code will use. `replicas=`, `pod_type=`, and
`serverless_read_capacity=` work the same way on both `pc.configure_index()`
and `pc.indexes.configure()`. Everything else with no faithful translation
raises a `PineconeTypeError` whose message shows the equivalent `2026-07`
call with your own values filled in.

The flow runs one way only. `pc.create_index()` is the `9.x`-shaped shim and
accepts the deprecated keywords alone: passing `schema=`, `deployment=`,
`read_capacity=`, or `cmek_id=` to it is a `PineconeTypeError` naming
`pc.indexes.create()` as the place those belong. On `pc.indexes.create()`
itself the two vocabularies are mutually exclusive — `schema=` together with
any of `dimension=`/`metric=`/`vector_type=`, or `deployment=` together with
`spec=`, raises `PineconeValueError` naming both.

```python
from pinecone import ServerlessSpec

# Deprecated sugar, produces a classic vector index served by the vectors
# API, addressing the vector by the reserved `_values` field rather than
# one you choose.
pc.create_index(
    name="movies",
    dimension=1536,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)

# Schema form: what new code should use; pick your own field name.
pc.indexes.create(
    name="movies",
    schema={"fields": {"embedding": {
        "type": "dense_vector", "dimension": 1536, "metric": "cosine"}}},
    deployment={"deployment_type": "managed", "cloud": "aws", "region": "us-east-1"},
)
```

The old metadata `schema=` kwarg changed meaning entirely. The new `schema=`
declares searched fields (`dense_vector`, `sparse_vector`, `string` with
`full_text_search`). Metadata-only fields are no longer declared at create
time, they're indexed automatically at upsert, and the server rejects a 400
if you declare one. `read_capacity` moved to the top level; `cmek_id` is new.
`name` is now optional, the server assigns one if omitted. `pods=` and
`metadata_config=` have no faithful translation and raise a
`PineconeTypeError` naming the equivalent call.

`source_collection=` / `source_backup_id=` aren't exposed: the `2026-07`
backend rejects both with `400 Creating an index from collection or backup
is not yet supported`. Use `pc.create_index_from_backup(...)` to restore a
backup.

(pod-collections)=

##### Pod deployments, and what that means for collections

`2026-07` does not create pod-backed indexes. This covers both spellings —
`deployment={"deployment_type": "pod", ...}` and the deprecated
`spec=PodSpec(...)` — and the refusal comes from the server, not from
client-side validation:

```
[400 INVALID_ARGUMENT] deployment_type 'pod' is not supported on this API
version. Set deployment_type to 'managed' to create a serverless index, or
set the X-Pinecone-API-Version header to an earlier version.
```

This is a property of the API version, not the SDK dropping a type.
`PodDeployment` and `PodSpec` are still exported and still describe pod
indexes that already exist: `describe()` decodes `index.deployment` as a
`PodDeployment` (with its `environment`, `pod_type`, `replicas`, and
`shards`), `list()` returns those indexes, `configure()` still accepts
`replicas=` and `pod_type=` against them, and `delete()` still removes them.
Creation is the one operation `2026-07` refuses.

New workloads take a managed (serverless) deployment, the same `deployment=`
shape used throughout this section:

```python
pc.indexes.create(
    name="movies",
    schema={"fields": {"embedding": {
        "type": "dense_vector", "dimension": 1536, "metric": "cosine"}}},
    deployment={"deployment_type": "managed", "cloud": "aws", "region": "us-east-1"},
)
```

Collections follow from that. A collection is a snapshot of a pod index, and
`pc.collections.create(source=...)` rejects a serverless source with a 400,
so on `2026-07` there is no index you can point it at. The `/collections`
routes are still served and `pc.collections` / `AsyncPinecone.collections`
are still on the client, so `list()`, `describe()`, and `delete()` keep
working against collections that already exist — enough to inventory and
clean them up, including before an `admin.projects.delete()` that a leftover
collection would otherwise block with a 412. `create()` has no reachable
source. Build snapshot-and-restore workflows on
[backups](#backup-models) instead: those work on managed indexes, and
`pc.create_index_from_backup(...)` is the restore path.

If you genuinely need pod indexes, the server's message names the only other
option: send an earlier `X-Pinecone-Api-Version`. `Pinecone` and
`AsyncPinecone` take an `additional_headers=` mapping that is merged last, so
an entry keyed exactly `"X-Pinecone-Api-Version"` replaces the header the SDK
would otherwise send — matching is case-sensitive, and any other spelling is
sent alongside the SDK's header instead of replacing it. The pin then applies
to every request that client makes, and `10.x` models decode `2026-07`
response shapes, so keep a pinned client scoped to the pod and collection
calls that need it rather than using it as a general downgrade.

Integrated-embedding creation moved from `create(spec=IntegratedSpec(...))`
to a dedicated `pc.indexes.create_for_model(name=..., cloud=..., region=...,
embed={"model": ..., "field_map": {"text": ...}})`. The embedding
configuration now surfaces as a `semantic_text` field in the returned
`index.schema`, named after the `field_map` text entry; `index.embed` is a
deprecated view rebuilt from that field.

`ConfigureIndexRequest` rejects an unknown PATCH field rather than ignoring
it.

```python
# Deprecated sugar, translates into deployment= below.
pc.configure_index("movies", replicas=4, pod_type="p1.x2")

# What new code should use.
pc.indexes.configure("movies", deployment={"replicas": 4, "pod_type": "p1.x2"})
```

`embed=` is removed entirely: the `9.x` convert-to-integrated flow no longer
exists, and the server rejects unknown PATCH fields. `serverless_read_capacity=`
and the old BYOC-only `read_capacity=` collapsed into one top-level
`read_capacity=` that covers managed and BYOC indexes. `configure()` now
returns the updated `IndexModel` (it returned `None` in `9.x`). Tags stay
merge-patch, set a value to `""` to delete that key.

```{warning}
`replicas=`, `pod_type=`, and `serverless_read_capacity=` still work as
deprecated sugar, translated into `deployment=`/`read_capacity=`. `embed=`
and `spec=` have no `2026-07` destination and raise a `PineconeTypeError`
naming the equivalent call. Passing both a deprecated keyword and the
`2026-07` argument it translates to (`replicas=4` together with
`deployment=...`) raises a `PineconeValueError` naming both.

Separately, `read_capacity=` widened its meaning. `9.x` had two read-capacity
keywords: `serverless_read_capacity=` for managed indexes, and
`read_capacity=` for BYOC only. In `2026-07` a single top-level
`read_capacity=` covers both. Code that passed `read_capacity=` intending
"BYOC only" now also takes effect on managed indexes, with no error and no
warning. Audit call sites that pass it before upgrading.
```

`AsyncPinecone.indexes` mirrors all of this one-for-one, with these
async-visible deltas: `list()` returns an `AsyncPaginator[IndexModel]` and
isn't a coroutine, so replace `(await pc.indexes.list()).names()` with
`[idx.name async for idx in pc.indexes.list()]`. The `pc.list_indexes()` shim
is unaffected: it stays a coroutine and still hands back an `IndexList`, so
`(await pc.list_indexes()).names()` keeps working. `exists("")` now raises
`PineconeValueError` where the old async client returned `False`.
`create_for_model()` is new on the async namespace. The index-scoped backup
methods graduated too: `create_backup()` and `describe_backup()` are
coroutines, and `list_backups()` returns an `AsyncPaginator[BackupModel]`.

`ServerlessSpec`, `PodSpec`, and `ByocSpec` remain importable and, passed as
the deprecated `spec=` argument, are the sugar translated into
`deployment=`/`schema=` above. `IntegratedSpec` and `EmbedConfig` remain
importable so the guided error can translate real values, but no
create/configure path accepts them anymore.

#### Dense, sparse, and full-text-search examples

Pick the field name your upsert and query code will address. There's no
default, and the deprecated `dimension=`/`spec=` form can't invent one for
you.

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

A `sparse_vector` field takes no `dimension` and no `metric`, both of which
were implied by `vector_type="sparse"` in `9.x`:

```python
pc.indexes.create(
    name="keywords",
    schema={"fields": {"sparse_terms": {"type": "sparse_vector"}}},
    deployment={"deployment_type": "managed", "cloud": "aws", "region": "us-east-1"},
)
```

```{warning}
A hybrid index must declare a `sparse_vector` field explicitly.
`metric="dotproduct"` on the dense field no longer implies sparse support on
its own, and the field can't be added later. This fails silently at create
time: the index is created and only the sparse upserts are refused later. See
[Sparse writes require a declared field](#sparse-writes) for the full
explanation and what error you'll actually see.
```

Full-text search has no `9.x` equivalent; it existed only on the preview
surface. A `string` field is accepted only with a `full_text_search` object.
An empty object selects the defaults (`language: "en"`, no stemming):

```python
pc.indexes.create(
    name="articles",
    schema={
        "fields": {"body": {"type": "string", "full_text_search": {"language": "en", "stemming": True}}}
    },
    deployment={"deployment_type": "managed", "cloud": "aws", "region": "us-east-1"},
)
```

`stop_words: true` requires `stemming: true`, and `ngram` can't be combined
with either, each is a separate 400. `cmek_id` is incompatible with any
`full_text_search` field.

#### Configuring replicas and read capacity

```python
# still works, deprecated, nests automatically under deployment=
pc.configure_index("movies", replicas=4, pod_type="p1.x2")
```

Both keys nest under `deployment=`, which must not carry a `deployment_type`
key. `deployment=` and `replicas=`/`pod_type=` are mutually exclusive;
passing both raises a `PineconeValueError`.

```python
index = pc.indexes.configure("movies", deployment={"replicas": 4, "pod_type": "p1.x2"})
```

```python
# still works, deprecated, managed index
pc.configure_index("movies", serverless_read_capacity={"mode": "OnDemand"})
```

One top-level `read_capacity=` now covers managed and BYOC indexes; read the
warning above before upgrading code that already passes `read_capacity=`.
`read_capacity=` and `serverless_read_capacity=` are mutually exclusive.
Read capacity doesn't apply to pod-based indexes, and changes apply
asynchronously, so poll `index.read_capacity.status` rather than assuming the
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

(create-limits)=

#### Create-time limits worth knowing about

Which of these you find out about locally depends on how you build the
schema. `SchemaBuilder` checks what it can before the request is sent and
raises `PineconeValueError`; a raw `schema=` dict is forwarded as written, so
the same mistake comes back as an `ApiError` carrying the server's message
verbatim.

A field `description` is capped at 256 bytes of UTF-8, not 256 characters, so
emoji and CJK text reach the cap at a fraction of their character count — 64
emoji fit, 65 don't. A field *name* is capped the same way, at 64 bytes.
`SchemaBuilder` enforces both and names the byte count it measured, so
`add_dense_vector_field("embedding", ..., description="🙂" * 65)` raises
before any request:

```text
PineconeValueError: Field 'embedding' description is too long: 260 bytes (max 256)
```

A schema is also capped on how many `full_text_search` fields it may declare.
That one is the server's, so it arrives as an `ApiError` naming the limit.

```python
pc.indexes.create(
    name="movies",
    schema={
        "fields": {
            "embedding": {
                "type": "dense_vector",
                "dimension": 1536,
                "metric": "cosine",
                "description": "Dense embedding of the movie synopsis",
            }
        }
    },
    deployment={"deployment_type": "managed", "cloud": "aws", "region": "us-east-1"},
)
```

`full_text_search.language` accepts a fixed set of language codes, and
`stop_words` is not supported for every language in that set. `language="tr"`
on its own is fine; `language="tr", stemming=True, stop_words=True` is a 400,
and the server's message names the unsupported language by its English name
rather than the code you sent. `ngram` doesn't
reject a `language`, it replaces it: a `language` sent alongside `ngram` is
accepted and the created index reports `en` regardless of what you sent.

```python
pc.indexes.create(
    name="articles",
    schema={
        "fields": {
            # Valid on its own: `tr` is one of the 18 `language` values.
            "body": {"type": "string", "full_text_search": {"language": "tr"}},
            # Accepted, then stored as `en`: ngram discards the language.
            "title": {
                "type": "string",
                "full_text_search": {"ngram": {"min_gram": 2, "max_gram": 4}, "language": "tr"},
            },
        }
    },
    deployment={"deployment_type": "managed", "cloud": "aws", "region": "us-east-1"},
)
```

`cmek_id` has two separate checks that report differently. A `cmek_id` with a
pod deployment, or alongside any `full_text_search` field, is a 400 per
request. If the project enforces CMEK encryption, a pod deployment or any
`full_text_search` field is a 412 per project, regardless of whether the
request carries a `cmek_id`. The per-request check runs first, so a pod
request that also carries `cmek_id` reports the 400; drop the `cmek_id` and
the same request reports the 412 instead.

Tags stay merge-patch on create as well as configure: an empty string value
means delete that key, and the 20-tag cap is checked on the merged total, not
on the request alone. The SDK's own checks run first and are narrower:
`tags={}` is a `PineconeValueError` before any request (pass `None` to send
no tags), keys must match `[a-zA-Z0-9_-]{1,80}`, and values must be printable
ASCII within 120 characters.

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

The same rule on configure, where there's already something to delete:

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

(vector-data)=
## Vector data (db_data)

This covers the classic `upsert`/`query`/`fetch` vector operations, which
keep serving indexes created before `2026-07` unchanged. Nothing here is
about the documents API above.

Every snippet in this section takes `idx` from one call, unchanged from
`9.x`:

```python
idx = pc.index("movies")
```

(vector-models)=
### Model and wire-format changes

The `db_data` vector models now follow the `2026-07` API shapes. Most of this
is invisible to SDK users.

`QueryRequest.queries` and the `QueryVector` schema are gone from the API,
but the SDK never exposed either name, so there's nothing to change in code
that uses `Index.query()`. Pass your query vector as `vector=` (dense),
`sparse_vector=` (sparse), or both. If you bypassed the typed surface and
sent a raw request body containing `queries`, the server now returns a 400;
issue separate queries instead, or use `query_namespaces()` when the same
query fans out across namespaces.

The fullness fields on `describe_index_stats()` are documented in camelCase
now (`memoryFullness`, `storageFullness`); the SDK already decoded the wire
spelling correctly, so `response.memory_fullness` and
`response.storage_fullness` keep working unchanged.

Metadata values are validated before the request is sent. A metadata value
must be a string, number, boolean, or list of strings:

```python
idx.upsert([("id-1", [0.1, 0.2], {
    "genre": "documentary",     # string
    "year": 2019,               # number
    "featured": True,           # boolean
    "tags": ["short", "indie"], # list of strings
})])
```

The server has always enforced this; what's new is that the SDK checks it
too, so a bad value raises locally instead of failing the whole batch
server-side:

```python
idx.upsert([("id-1", [0.1, 0.2], {"price": {"usd": 10}})])
# PineconeTypeError: Metadata value must be a string, number, boolean or list of
# strings, got '{"usd":10.0}' for field 'price'
```

If you were relying on a server 400 to find bad rows, you now get a
`PineconeTypeError` from `upsert()` instead. Catch `PineconeError` to cover
both. The check runs on every input form `upsert()` accepts, including the
gRPC path.

A `None` value is accepted, not rejected. The server strips null metadata
values on write rather than refusing them, so this has always silently
dropped the key rather than storing it:

```python
idx.upsert([("id-1", [0.1, 0.2], {"tag": None})])  # no error
```

This now holds on both transports. The gRPC transport used to encode a
`None` metadata value in a way the server refused with a 400, so the same
upsert succeeded over REST and failed over gRPC; now both sides silently drop
the key. `None` inside a filter is still sent through on both transports and
rejected by the server. An empty list and an empty string are also accepted.

Metadata keys may not begin with `$`, which is reserved for filter operators.
The server rejects a `$`-prefixed key; the SDK doesn't check key names
client-side.

(sparse-writes)=

#### Sparse writes require a declared field

This changes no model and no method signature, which is exactly why it's
worth reading: it's a breaking change that produces no error at the point
where you have to fix it.

In `9.x`, `metric="dotproduct"` on a dense index was the whole hybrid
declaration. Nothing else had to be said, and sparse values worked. In
`2026-07` that index shape doesn't exist. Sparse traffic is gated on the
schema actually declaring a `sparse_vector` field, so a hybrid index must
declare one explicitly, and the metric buys you nothing on its own anymore.

The `2026-07` equivalent names both vector fields explicitly. Pick names your
upsert and query code will address; there's no default. `dotproduct` stays on
the dense field, and the sparse field takes neither `dimension` nor `metric`.

```python
# Deprecated sugar (dimension=/metric=/spec=) still accepts this 9.x-style
# call, and that is exactly the trap: it creates a dense-only schema (the
# reserved `_values` field) with no sparse_vector field, so sparse writes
# still fail silently, with no error at create time.
from pinecone import ServerlessSpec

pc.create_index(
    name="hybrid",
    dimension=1536,
    metric="dotproduct",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)
```

:::::{tabs}
::::{tab} Sync

```python
pc.indexes.create(
    name="hybrid",
    schema={
        "fields": {
            "embedding": {"type": "dense_vector", "dimension": 1536, "metric": "dotproduct"},
            "sparse_terms": {"type": "sparse_vector"},
        }
    },
    deployment={"deployment_type": "managed", "cloud": "aws", "region": "us-east-1"},
)
```

::::
::::{tab} Async

```python
await pc.indexes.create(
    name="hybrid",
    schema={
        "fields": {
            "embedding": {"type": "dense_vector", "dimension": 1536, "metric": "dotproduct"},
            "sparse_terms": {"type": "sparse_vector"},
        }
    },
    deployment={"deployment_type": "managed", "cloud": "aws", "region": "us-east-1"},
)
```

::::
:::::

You cannot add the sparse field afterwards. An index created without one has
to be recreated, which is why this belongs in your upgrade plan rather than
in a later fix.

There is no signal at create time. A create with a dense `dotproduct` field
and no sparse field succeeds, returns a healthy `IndexModel`, and serves
dense traffic normally. The first symptom is a refused sparse write, often
from a different service and long after the upgrade. Audit for
`metric="dotproduct"` before you upgrade; in `9.x` that keyword was the
hybrid declaration, so every occurrence is a candidate for a missing sparse
field.

What actually fails, and with which error, is not what you'd predict. On any
index you create with `2026-07`, a vectors-API upsert is refused before the
sparse check is ever reached, with a message that begins "This index has a
document schema, so writes must go through the documents API." Declaring the
sparse field doesn't satisfy that gate on its own; a hybrid schema is routed
to the documents API rather than the vectors API. If you're debugging a
hybrid upsert against a freshly created `2026-07` index, that document-schema
message is the string to search for. On the documents API, where a
`2026-07` index is routed today, the same requirement shows up at search
time instead, with a clearer message: `Scoring field '<name>' not found in
index schema`.

```{warning}
The server's own sparse error text is out of date. `SparseNotSupported`
still reads "Index configuration does not support sparse values - only
indexes that are sparse or using dotproduct are supported." The "or using
dotproduct" clause describes the `9.x` behavior this change removed, so
following that message's advice and setting `metric="dotproduct"` won't fix
anything. Read it as "only indexes whose schema declares a `sparse_vector`
field" instead.
```

`SchemaBuilder` makes the requirement discoverable:
`add_sparse_vector_field()` sits directly beside `add_dense_vector_field()`,
so the hybrid pair reads as a single chain and you never have to know the
wire spelling.

```python
from pinecone import SchemaBuilder

schema = (
    SchemaBuilder()
    .add_dense_vector_field("embedding", dimension=1536, metric="dotproduct")
    .add_sparse_vector_field("sparse_terms")
    .build()
)
```

The chain and the dict above put identical bytes on the wire.

`add_sparse_vector_field()` no longer emits `metric`. Through `9.x` the
builder put `{"type": "sparse_vector", "metric": "dotproduct"}` on the wire;
a `2026-07` sparse field has no `metric` at all, so the key never configured
anything. It's gone now:

    # before
    {"type": "sparse_vector", "metric": "dotproduct"}
    # now
    {"type": "sparse_vector"}

If you assert on `build()`'s output, drop `metric` from the expected sparse
field. If you pass `metric=` or `dimension=` to `add_sparse_vector_field()`,
the call now raises `PineconeValueError` naming the field and the key,
instead of silently discarding them:

```python
SchemaBuilder().add_sparse_vector_field("sparse_terms", metric="dotproduct")
# PineconeValueError: Field 'sparse_terms' cannot declare 'metric': a sparse
# vector field has no metric — sparse scoring is not configurable. Remove the
# argument — a sparse vector field accepts only a description.
```

Sparse vectors are variable-length and their scoring has no knob to turn.

(db-data-breaking-changes)=
### Breaking changes and migration table

`upsert`, `query`, `fetch`, `update`, `delete`, `list`, and
`describe_index_stats` serve indexes created under earlier API versions, and
they're meant to. None of them is deprecated, none is scheduled for removal,
and none of them takes a different argument or returns a different shape than
it did on `9.x`. If you have a workload upserting and querying an index you
created before `2026-07`, upgrading the SDK doesn't change what those calls
mean. It does change how fast a batched `upsert` paces itself, which is its
own section: [Bulk ingest: concurrency and deadlines](#bulk-ingest).

What changed is index creation. A `2026-07` `pc.indexes.create()` always
persists a document schema, and a document-schema index is addressed through
the document operations, not the vector operations. So pick the family by
the index you're addressing, not by which one looks newer:

| The index you're addressing | The operations that serve it |
| --- | --- |
| Created under an API version earlier than `2026-07` | `upsert`, `query`, `fetch`, `fetch_by_metadata`, `update`, `delete`, `list`, `describe_index_stats` |
| Created with `2026-07` (document schema) | `documents.upsert`, `documents.batch_upsert`, `documents.search`, `documents.fetch`, `documents.update`, `documents.delete`, `documents.list` |

A vector-API write aimed at a document-schema index is refused, and the
server's message names the endpoint to use instead: "This index has a
document schema, so writes must go through the documents API." Read that as
"wrong operation family for this index," not a deprecation notice. See
[The .documents namespace](#preview-graduation) for the document operations
themselves.

#### The migration table

| # | What changed | Before | Now |
| --- | --- | --- | --- |
| 1 | `query(top_k=...)` upper bound | REST and asyncio accepted any `top_k >= 1`; only gRPC bounded it | all three lanes accept `1`-`10000` and raise `PineconeValueError` outside it |
| 2 | `describe_index_stats(filter=...)` | documented as returning counts for matching vectors only | documented as rejected for every index type; there is no filtered-count operation |
| 3 | `SchemaBuilder.add_boolean_field` / `add_float_field` / `add_string_list_field` | `filterable` was left out of the field whenever it was `False` | `filterable` is always emitted, `True` or `False` |
| 4 | `SchemaBuilder.add_sparse_vector_field` | emitted `{"type": "sparse_vector", "metric": "dotproduct"}` | emits `{"type": "sparse_vector"}`; `metric=` and `dimension=` raise |
| 5 | Declaring a hybrid index | `metric="dotproduct"` on the dense field implied sparse support | the schema must declare a `sparse_vector` field |
| 6 | `start_import(error_mode=...)` omitted | documented as `"continue"` in two of three lanes | documented as `"abort"`, which is what the server has always done |
| 7 | `create_namespace(schema=...)` omitted | read as "index every field" | inherits the index's own metadata-index configuration |
| 8 | Enum members on the wire | mangled in inference request bodies and query strings | resolved to their values; no `db_data` call was affected |
| 9 | `query(id=..., sparse_vector=...)` | only `id` alongside `vector` was rejected client-side; `id` alongside `sparse_vector` was forwarded | `id` is rejected alongside either `vector` or `sparse_vector` |
| 10 | `update(filter=..., values=...)` / `update(filter=..., sparse_values=...)` | forwarded to the server | raises `PineconeValueError` before any request; a by-filter update cannot carry vector values |
| 11 | Empty `filter={}` on `delete`, `update`, `fetch_by_metadata` | forwarded to the server, which rejected it | raises `PineconeValueError` locally, with the server's own wording |
| 12 | `fetch(ids=...)`, `fetch_by_metadata(limit=...)`, `list_paginated(prefix=..., limit=...)` | `fetch` checked only that `ids` was non-empty; `fetch_by_metadata` checked only that `limit` was positive; `list_paginated` validated neither argument | all three validate the same ID/prefix shape and limit range every other vector operation already used |
| 13 | `GrpcIndex.upsert_from_dataframe` partial failures | raised on the first failed batch, discarding the count of what had landed | returns an `UpsertResponse` carrying `upserted_count`, `failed_item_count`, `errors` and `failed_items`; `on_error="raise"` restores the raise, now with that partial response attached to the exception. See [gRPC `upsert_from_dataframe` reports partial failures](v10-grpc-partial-failures.md) |

Rows 1, 3, and 4 change the bytes the SDK puts on the wire. Rows 2, 6, and 7
change documentation that was wrong, not behavior. Row 5 changes what a
working index declaration looks like, covered above under
[Sparse writes require a declared field](#sparse-writes). Row 8 needed no fix
on this surface. Rows 9-12 add client-side checks for requests the server was
already going to refuse. Row 13 changes what a gRPC ingest hands back when
some batches fail, and has a page of its own.

#### 1. `query(top_k=...)` is bounded at both ends on every lane

`query` has always had an upper bound on `top_k`. Until this release only
`GrpcIndex` enforced it; `Index` and `AsyncIndex` forwarded anything larger
and it came back as a server error. All three lanes now share one range
check:

```python
idx.query(vector=[0.1, 0.2], top_k=20000, namespace="movies-en")
# PineconeValueError: top_k must be between 1 and 10000, got 20000
```

The bound is `1`-`10000` on `Index`, `AsyncIndex`, and `GrpcIndex` alike, and
it's checked before any request is made. A call the server was going to
reject now fails locally instead, with nothing to un-send. The ceiling is a
deployment setting, not a constant of the API: a deployment configured lower
than the client's ceiling still rejects values this check lets through, and
that arrives as an `ApiError`. Catch `PineconeError` to handle both, or clamp
before you call.

#### 2. `describe_index_stats(filter=...)` never returns a filtered count

The docstrings and how-to guide described a working metadata filter. Both
were wrong: a non-empty `filter` is rejected for every index type, so a
filtered stats call fails rather than returning a subset count. There's no
operation anywhere on this surface that counts only the records matching a
metadata filter, so if you built a count on this argument, it was never
working, and there's nothing to migrate it to.

```python
# This has never worked. The call fails; it does not return a subset count.
stats = idx.describe_index_stats(filter={"genre": {"$eq": "action"}})
```

Drop the argument. The statistics you get back describe the whole index:

```python
stats = idx.describe_index_stats()
print(stats.total_vector_count, stats.dimension)
```

No behavior changed here, all three lanes still forward the filter
unvalidated and the server still rejects it. What changed is that the
documentation now says so. To count a subset, query for it or maintain the
count yourself.

#### 3. `SchemaBuilder` metadata fields always emit `filterable`

`add_boolean_field()`, `add_float_field()`, and `add_string_list_field()`
left `filterable` out of the emitted field whenever it was `False`, the
default, so the shortest documented call produced a create the backend
rejected outright. The key is now always present:

```python
from pinecone import SchemaBuilder

schema = (
    SchemaBuilder()
    .add_boolean_field("is_published")
    .add_float_field("year")
    .add_string_list_field("tags")
    .build()
)
schema["fields"]["is_published"]
# before: {'type': 'boolean'}
# now:    {'type': 'boolean', 'filterable': False}
```

If you assert on `build()`'s output, add `filterable` to the expected
boolean, float, and string-list fields. `filterable=True` was unaffected.
`add_string_field()` is deliberately unchanged; its wire shape is different
enough that the same edit would break it, and its intended shape is still an
open question.

```{warning}
A string field can be declared for metadata filtering or for full-text
search, and the full-text-search spelling doesn't currently reach the
backend as full-text search: the field is created as filter-only metadata,
with no error and no warning, so the omission is invisible until a search
returns nothing. Don't take a full-text-search string field from any example
on this page as working yet. Boolean, float, and string-list fields are
unaffected.
```

Passing `metric=` or `dimension=` to `add_sparse_vector_field()` now raises
instead of silently forwarding a key that does nothing. See
[Sparse writes require a declared field](#sparse-writes) for the full
write-up.

#### 6. `start_import(error_mode=...)` defaults to `"abort"`

The docstrings and the bulk-import how-to said the default was
`"continue"`, meaning a record the import can't read is skipped and the rest
still import. The default is actually `"abort"`: the import ends at the
first record it can't read. The server has always behaved this way; only the
documentation was wrong, so no running import changes, but if you omitted
`error_mode` on the strength of that sentence, you've been getting the
opposite of what you read.

```python
# Ends the whole import at the first unreadable record (the default).
idx.start_import(uri="s3://my-bucket/vectors/")

# Opt in to skipping unreadable records and importing the rest.
idx.start_import(uri="s3://my-bucket/vectors/", error_mode="continue")
```

#### 7. `create_namespace(schema=...)` omitted means inherit, not index everything

Omitting `schema` doesn't create a namespace with every field indexed; the
namespace inherits the index's own metadata-index configuration.

```python
# Inherits the index's metadata-index configuration, whatever that is.
ns = idx.create_namespace(name="movies-en")

# Overrides it for this namespace alone: exactly these fields are indexed.
ns = idx.create_namespace(
    name="movies-en",
    schema={"fields": {"genre": {"filterable": True}}},
)
```

Supplying `schema` overrides the inherited configuration for that namespace
only. Each field listed must set `filterable: True`; to leave a field
unindexed, omit it entirely. Behavior didn't change here, this was already
how the server worked. If you relied on the omitted form to index everything
on a restricted index, supply `schema` explicitly now.

#### 9. `query` rejects `id` alongside `sparse_vector`, not just `vector`

`query` accepts a stored vector's `id`, literal `vector`/`sparse_vector`
data, or both vector forms together for a hybrid query, but never `id`
together with either. Before this release only the `id`+`vector` combination
was checked client-side; `id`+`sparse_vector` was forwarded and refused by
the server instead. Both now raise before any request is sent:

```python
idx.query(id="article-101", sparse_vector={"indices": [0, 1], "values": [0.5, 0.5]}, top_k=10)
# PineconeValueError: id is mutually exclusive with sparse_vector — a query uses
# a stored vector's id OR literal vector data, not both. Pass id alone to query
# by stored vector, or sparse_vector alone to query by value. Cannot provide
# both 'ID' and 'sparse_vector' at the same time
```

A hybrid query, `vector` and `sparse_vector` together with no `id`, is
unaffected.

#### 10. A by-filter `update` cannot carry `values` or `sparse_values`

A by-filter update spans every record the filter matches, so it can only set
metadata. The SDK now catches this before the request leaves the process:

```python
idx.update(filter={"genre": {"$eq": "drama"}}, values=[0.1, 0.2, 0.3])
# PineconeValueError: filter is mutually exclusive with values — a by-filter
# update is metadata-only, because it spans every record the filter matches.
# Pass set_metadata to update metadata by filter, or id to update one record's
# vector values. Update by metadata request does not support updating vector
# values.
```

Update by `id` to change one record's vector values; update by `filter` to
change metadata on every matching record, via `set_metadata`. `id` and
`filter` together were already rejected before this release, and still are,
that check didn't change.

#### 11. An empty `filter={}` now raises locally

`delete`, `update`, and `fetch_by_metadata` have always rejected a metadata
filter with no conditions. Before this release the empty dict was forwarded
and refused by the server; now it raises before any request is sent, with
the server's own wording:

```python
idx.delete(filter={})
# PineconeValueError: filter must contain at least one condition, got {}.
# Delete with empty metadata filter is not allowed
```

`delete` is the one operation here with a true match-everything mode, spell
it with `delete_all=True`, not an empty filter.

#### 12. `fetch`, `fetch_by_metadata`, and `list_paginated` validate up front

`fetch`'s `ids`, `list_paginated`'s `prefix`, and every vector `id` share one
rule: 1-512 ASCII characters, no NUL. Before this release, `fetch` checked
only that `ids` was non-empty and `list_paginated` validated neither
argument at all. Both now raise locally:

```python
idx.fetch(ids=["a" * 600])
# PineconeValueError: ids[0] exceeds the maximum length of 512 characters, got 600: 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'...'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa' (600 characters)
```

Page sizes follow the same pattern. `list_paginated`'s `limit` is now
bounded 1-100, and `fetch_by_metadata`'s `limit` is now also bounded above,
at 10000:

```python
idx.list_paginated(limit=500)
# PineconeValueError: limit must be between 1 and 100, got 500

idx.fetch_by_metadata(filter={"genre": {"$eq": "comedy"}}, limit=20000)
# PineconeValueError: limit must be between 1 and 10000, got 20000
```

As with row 1's `top_k` bound, these are deployment settings the server
enforces, not constants of the API.

#### What you don't need to change

Nothing in the table above changes a method signature. Any working `query` call
keeps working, row 1 only rejects values the server was going to reject
anyway. Vector operations against a legacy index are unaffected, rows 3, 4,
and 5 are about creating an index, so none of them reaches one you already
have. `describe_index_stats` without a filter is unchanged. And nothing
about `error_mode` or `create_namespace(schema=...)` that you already had
right needs to change; rows 6 and 7 correct documentation, not behavior.

(bulk-ingest)=
## Bulk ingest: concurrency and deadlines

Batched writes pace themselves differently on `10.x`. This covers
`upsert(batch_size=...)` and `upsert_from_dataframe` on `Index`, `AsyncIndex`
and `GrpcIndex`, and `documents.batch_upsert` on both document surfaces. The
argument you pass to set concurrency is the same one; its default is not, and
there is now a per-host admission gate between your call and the wire that
can pace a batch job down or refuse the rest of it.

None of this applies to an unbatched write. An `upsert` without `batch_size`
is a single request: `max_concurrency` and `total_timeout` are both ignored
on that path, and `max_concurrency` isn't even range-checked.

(bulk-max-concurrency)=
### `max_concurrency` defaults to 8

`upsert` took `max_concurrency=4` on `Index`, `AsyncIndex` and `GrpcIndex`
alike. It now takes `8` on all three, from a single constant. The accepted
range is unchanged at `1`-`64`, and a value outside it still raises
`PineconeValueError` before anything is sent. What the argument buys is also
unchanged: it caps how many of your batches are in flight at once — a thread
pool that wide on the two sync lanes, that many concurrent coroutines on the
asyncio lane.

If you sized a workload around four in-flight batches, say so:

```python
vectors = [(f"vec-{i}", [0.1, 0.2, 0.3]) for i in range(400)]

# 10.x default: up to 8 batches in flight
idx.upsert(vectors=vectors, namespace="movies-en", batch_size=100)

# the 9.x pacing, stated explicitly
idx.upsert(vectors=vectors, namespace="movies-en", batch_size=100, max_concurrency=4)
```

Each in-flight batch still carries its own retry budget on top of that
(`max_retries=3` by default, so up to four attempts), so doubling the default
doubles the number of concurrent conversations a single ingest opens with one
host, not the attempts within any one of them.

`upsert_from_dataframe` arrives at the same `8` by a different route, because
`9.x` gave it no concurrency argument at all — just `df`, `namespace`,
`batch_size` and `show_progress`. On `10.x` all three lanes — `Index`,
`AsyncIndex` and `GrpcIndex` — take a keyword-only `max_concurrency` that
defaults to `None` and resolves to `8`. pandas is not an SDK dependency; it's
imported when you call this method, and its absence is the one thing that
raises here.
`documents.batch_upsert` resolves `None` the same way, and was `4` under
`pc.preview`; see [Smaller signature deltas](#preview-graduation) for the
rest of that signature's changes.

(bulk-backpressure)=
### The admission gate can refuse batches

Under `9.x` every `Pinecone` client owned its own adaptive limiter, keyed by
host, ceilinged at whatever `max_concurrency` you passed. It halved on a
throttled response and healed back up on success streaks, and that was the
whole of it: it could only make an ingest slower.

`10.x` replaces it with a per-host admission gate that every batch has to be
admitted through. Three differences matter when you upgrade:

- **The gate is process-global, not per-client.** Two `Pinecone` instances,
  or a `Pinecone` and an `AsyncPinecone`, hitting one host in one process
  share one gate and one in-flight count. The key is the bare lowercase
  hostname, so scheme and port variants land on the same gate. Concurrent
  ingests against one host therefore share a budget rather than each getting
  their own. (A forked child starts with fresh gates rather than inheriting
  the parent's counts.)
- **Your `max_concurrency` is a ceiling, not a floor.** Effective
  concurrency is the lower of your bound and the gate's current limit. The
  gate starts each host at its own ceiling of `64`, so on a healthy host your
  bound is the one that binds; once a throttled response halves the gate, the
  gate's limit is. It halves at most once per in-flight generation, so one
  batch burning its retries can't halve the limit six times over, and it adds
  one back after a limit-sized run of successes — counted only while callers
  were actually pressing against the limit, so unused headroom doesn't
  accumulate while the host is idle. A `Retry-After` (or
  `grpc-retry-pushback-ms`) hint blocks admission outright until it elapses.
- **The gate can stop an ingest instead of slowing it.** When its limit is
  at the floor of one and four consecutive batches settle with no success at
  all, it reads the host as unavailable and refuses admission for a 30-second
  cool-down. The batches that were still queued are abandoned without being
  sent, and come back as errors carrying `disposition="abandoned"` and a
  message that says so, rather than as a timeout. `9.x` had nothing that did
  this; a dead host produced batch after batch of exhausted retries.

The gate's state is reported back on `documents.batch_upsert`'s
`BatchResult`: `throttle_event_count`, `final_limit` (a value far below your
`max_concurrency` means the host was pushing back), `peak_inflight`, and
`stalled`. Because the gate is shared per host, those counters are
host-level, not call-level. `UpsertResponse` on the vector paths carries none
of them — there, `errors` and `failed_items` are the whole signal.

(bulk-total-timeout)=
### `total_timeout` bounds admission, not requests

`10.x` adds a `total_timeout` keyword to every batched ingest entry point:
`upsert` and `upsert_from_dataframe` on all three lanes, and
`documents.batch_upsert` on both the sync and asyncio document surfaces.
`9.x` had no equivalent — `timeout` was the only deadline available, and it
bounds one attempt of one batch, so a large ingest had no whole-job bound at
all. It defaults to `None` on every one of those methods, so nothing changes
until you pass it.

What it bounds is admission. When the deadline expires, no further batch is
submitted; batches already in flight are awaited and never cancelled, because
cancelling client-side wouldn't un-send what the server may already be
applying, and the caller would be told less landed than did. So a call can
outlive its own `total_timeout` — by up to one batch's `timeout` budget,
retries included. Waiting on the admission gate counts against the budget
too, which means a throttled host can consume it without a request going out.

The batches that were never submitted are reported, not silently dropped.
Each becomes an error carrying `disposition="unsent"` and a
`PineconeTimeoutError` that names the budget and the batch — `total_timeout
of 5.0s expired before this batch was submitted; 100 items in batch 3 were
not sent` — and its items are in `failed_items`, ready to be passed straight
back in:

```python
vectors = [(f"vec-{i}", [0.1, 0.2, 0.3]) for i in range(400)]

response = idx.upsert(
    vectors=vectors,
    namespace="movies-en",
    batch_size=100,
    total_timeout=30.0,
)
unsent = [error for error in response.errors if error.disposition == "unsent"]
if unsent:
    retry = idx.upsert(
        vectors=response.failed_items,
        namespace="movies-en",
        batch_size=100,
    )
```

A partial ingest still doesn't raise on `upsert` — the deadline is reported
like any other per-batch failure. `upsert_from_dataframe` is where you get a
choice: `on_error="collect"` (the default) hands back the same response, and
`on_error="raise"` re-raises the lowest-indexed failure once every batch has
settled, which on `Index` and `AsyncIndex` means the `PineconeTimeoutError`
for the first unsent batch. `GrpcIndex.upsert_from_dataframe` is the one
variation: for an expired deadline it raises a `PineconeTimeoutError`
summarizing how many vectors landed rather than re-raising one batch's error,
and under `on_error="collect"` it logs that same summary as a warning. Either
way the partial response is on the exception's `response` attribute, so
nothing about the retry is lost.

`documents.batch_upsert` adds one thing the vector paths don't have:
`result.timed_out`. It's `True` only when work was actually left unsent. A
deadline that elapses while the last batches are in flight, all of which then
land, doesn't set it — there'd be nothing to retry.

(backup-models)=
## Backups and backup schedules

The backup response models now follow the `2026-07` API shapes, which is a
breaking change to `BackupModel`. The operations themselves change
additively.

### What changed on `BackupModel`

`BackupModel` no longer has `.dimension` or `.metric`. `.schema` changed
type too: it was a plain dict shaped like the old metadata schema, and is
now a typed `IndexSchema`, the same class returned by `index.schema`.

| Removed | Replacement |
| --- | --- |
| `backup.dimension` | `backup.schema.fields["<field>"].dimension`, or `backup.dense_dimension` |
| `backup.metric` | `backup.schema.fields["<field>"].metric` |
| `backup.schema["fields"]["<name>"]["filterable"]` | `backup.schema.fields["<name>"].filterable` |

Below, `backup` is a `BackupModel` from `pc.backups.describe(backup_id=...)`.
The `9.x` lines are the ones that now raise:

```python
# 9.x
dim = backup.dimension
metric = backup.metric

# 10.x
dim = backup.dense_dimension
metric = backup.schema.fields["embedding"].metric
```

`dense_dimension` is a convenience for the common one-vector-field case. It
returns `None` when the schema is absent, declares no `dense_vector` field,
or declares more than one, read the field you want out of
`backup.schema.fields` in that case.

New field on `BackupModel`: `source_index_deleted_at`, the deletion
timestamp of the source index, or `None` when that index is still active.
Only `list_index_backups(include_deleted=True)` populates it.

`CreateIndexFromBackupRequest` is exported from `pinecone`, `pinecone.models`,
and `pinecone.models.backups`. `PreviewBackupModel` and
`PreviewCreateBackupRequest` are removed along with the preview package;
preview backup operations now return the same top-level `BackupModel` as
everything else.

### Operations

No backup method lost an argument. `pc.backups.list` gained a keyword-only
`include_deleted: bool | None = None`. `pc.create_index_from_backup` gained
a keyword-only `read_capacity: dict | None = None`. And
`pc.indexes.create_backup` / `list_backups` / `describe_backup` graduated out
of the preview namespace onto `pc.indexes` directly.

#### The two backup namespaces are not interchangeable

`pc.indexes.*` and `pc.backups.*` reach the same backups, and it's tempting
to treat one as an alias of the other. Two differences will catch a
find-and-replace between them:

| `pc.indexes` | `pc.backups` |
| --- | --- |
| `create_backup(index_name, *, name=...)` | `create(*, index_name=..., name=...)` |
| `describe_backup(backup_id)` | `describe(*, backup_id=...)` |
| `list_backups(index_name, ...)` → `Paginator[BackupModel]` | `list(*, index_name=..., ...)` → `BackupList` |

The identifier is keyword-only on `pc.backups` and positional-or-keyword on
`pc.indexes`, so the conversion only breaks in one direction:
`pc.backups.describe("bk-abc123")` is a `TypeError`, while both spellings
work on `pc.indexes.describe_backup`.

The listing types differ too, and that one is silent: `list_backups` hands
back a lazy `Paginator` you iterate, `list` hands back a materialized
`BackupList` you can index and take `len()` of. Swapping the call without
changing what you do with the result is where this bites.

`pc.backups` is also the only place `delete` and the project-wide listing
live, since neither belongs to any one index.

#### `include_deleted` and what a 404 means

`pc.backups.list(index_name=...)` and `pc.indexes.list_backups(...)`
both resolve the name against active indexes by default. If every index that
used the name has been deleted, the API returns 404 rather than an empty
list, so a 404 alone is not proof the name was never used:

| State of `index_name` in the project | omitted / `False` | `True` |
| --- | --- | --- |
| An active index has the name | its backups | backups of every index that has held the name, active and deleted |
| Only deleted indexes have held the name | 404, not an empty list | those backups, each with `source_index_deleted_at` set |
| The name has never existed | 404 | 404 |
| The name exists but has no backups | empty list | empty list |

A 404 is only conclusive when you sent `include_deleted=True`. A 404
without it proves nothing about whether the name was ever used, retry with
`include_deleted=True` before concluding anything.

```python
# 10.x: recover the backups of an index you already deleted
page = pc.backups.list(index_name="product-search", include_deleted=True)
orphaned = [b for b in page if b.source_index_deleted_at]
```

`include_deleted` on the project-wide `pc.backups.list()` raises
`PineconeValueError`, because that operation doesn't accept it, the
project-wide listing already returns backups whose source index was
deleted.

#### `read_capacity` on restore

`read_capacity=` is applied to the restored index, so restoring onto
dedicated read nodes is one call. In `9.x` there was no way to ask for it on
the restore; a restored index always came up on on-demand capacity.

The server rejects a dedicated configuration too small to hold the backup, so
an undersized request fails the restore rather than producing an index that
cannot serve the data.

:::::{tabs}
::::{tab} Sync

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

::::
::::{tab} Async

```python
index = await pc.create_index_from_backup(
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

::::
:::::

`create_index_from_backup` remains the only supported restore path;
`pc.create_index(source_backup_id=...)` raises a `PineconeTypeError` pointing
at it.

```{note}
Don't poll a restore job for an in-progress state. `restore_job.status` is
one of `Pending`, `Failed`, `Completed`, `Cancelled` and nothing else, and
`percent_complete` is `100` on `Completed` and `None` otherwise, never an
intermediate percentage. A loop that waits for a `"Running"` status, or for
`percent_complete` to climb, never terminates. Wait on `status ==
"Completed"` and treat `Failed`/`Cancelled` as terminal.
```

`AsyncPinecone.backups` mirrors `pc.backups` one-for-one, only `await`
differs:

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

### Backup schedules

`pc.backup_schedules` is additive, there's no `9.x` equivalent. Six
operations: `create`, `list`, `describe`, `update`, `delete`, `history`, plus
the lazy `iter_schedules` / `iter_history` walkers. Three behaviors are
worth knowing before you rely on this.

**Plan gating is asymmetric between `create` and `update`.** Backups are a
plan entitlement, gated on exactly the Free and Builder plans. `create`
always checks the entitlement, before the index is even looked up, so a
project without the entitlement gets a 403 for a nonexistent index name too;
the 403 is about the plan, never about the index. Toggling `update` from
disabled to enabled checks the entitlement as well, but disabling a schedule,
or changing its `frequency` or `retention_days`, does not. That's
intentional, so a downgraded organization can still turn off or retune a
schedule it can no longer create.

```{warning}
On-demand backups are gated by the same entitlement. `pc.backups.create()`
checks it before its own index lookup, so a 403 from a schedule operation is
not a reason to fall back to `pc.backups.create()`, that call answers 403 for
the same projects.
```

**An index may hold at most one enabled schedule.** Disabled schedules don't
count, so you can keep several and enable one at a time. The conflict
surfaces as a 409 whether it comes from `create` while one is already
enabled, or from `update` re-enabling while another is enabled.

**Re-enabling a disabled schedule runs a backup immediately and shifts the
cadence.** `update(enabled=True)` on a disabled schedule recomputes
`next_scheduled_run` from the moment of the update, not from the schedule's
original anchor, and enqueues a backup for that new time. A daily schedule
anchored at 06:00 UTC, disabled and re-enabled at 15:20, runs daily at 15:20
from then on; there's no way to re-enable without this side effect. Sending
`enabled=True` on an already-enabled schedule is a safe no-op.

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

`list` and `history` return one page each, and `limit` is dropped whenever
`pagination_token` is given (the token already encodes the page size). Send
`limit` on the first call only, or use `iter_schedules` / `iter_history`,
which follow the tokens for you. The same rule holds for `pc.backups.list`
and `pc.restore_jobs.list`.

A few smaller behavior notes: backup `status` is documented as
`Initializing`, `Ready`, or `Failed`, and the backend currently returns
`InitializationFailed` in place of `Failed`; the model keeps `status` a
plain string, so both decode. `schema` is `None` when the server omits it.
`tags` is `None` when the source index had no tags, and `BackupList.pagination`
is `None` on the final page.

(assistant-models)=
## Assistant

### AssistantFileModel: removed fields

`AssistantFileModel` no longer has `.percent_done` or `.error_message`. File
processing progress and failure detail moved to the operations API:

| Removed | Replacement |
| --- | --- |
| `file.percent_done` | `describe_operation(...)` and read `OperationModel.status` |
| `file.error_message` | `describe_operation(...)` and read `OperationModel.error` |

Accessing either attribute raises an `AttributeError` naming
`describe_operation` as the replacement:

```python
file = pc.assistants.describe_file(assistant_name="my-assistant", file_id="f-1")
file.error_message
# AttributeError: AssistantFileModel.error_message was removed in the 2026-07
# Pinecone API: processing failure detail is reported by the operations API
# instead — call describe_operation() and read OperationModel.error. ...
```

Dict-style access follows suit: `file["percent_done"]` raises `KeyError`, and
neither name appears in `file.keys()` or `file.to_dict()`. Responses that
still carry the old keys, from a `2025-10` server or a recorded fixture,
continue to decode; the extra keys are ignored.

`upload_file()` and `delete_file()` used to quote `error_message` in the
error they raised on a failed poll. With the field gone, those messages now
name the file state and point to `describe_operation()` for the reason. Code
matching on the old message text needs updating.

### File IDs are no longer UUIDs

`2026-07` documents `id` as a plain string, because a file ID may be one the
caller supplied. Code that parses a file ID as a UUID breaks:

```python
import uuid

uuid.UUID(file.id)  # no longer safe
```

`size` (bytes, `int64`) is part of the documented shape now, populated on
upload, describe, and list responses.

### `AssistantModel.region`

`AssistantModel` gains `region`, `"us"` or `"eu"`, the region the assistant
is deployed in. `create` takes `region` too, defaulting to `"us"`. It's fixed
at creation time, there's no move-an-assistant operation, so an `eu`-resident
assistant has to be created that way:

```python
assistant = pc.assistants.create(name="eu-assistant", region="eu")
assistant.region  # 'eu'
```

Not every deployment can serve `eu`; where it can't, the request is refused
with a message saying so.

### Model names and finish reasons

The `2026-07` data API changes which model names it documents on `chat()`
and which `finish_reason` values it returns. The SDK doesn't validate
`model` client-side, the backend is authoritative, so these surface as
backend rejections and as different strings, not SDK type errors.

`claude-3-5-sonnet` and `claude-3-7-sonnet` are no longer documented. Migrate
to `claude-sonnet-4-5`:

```python
response = pc.assistants.chat(
    assistant_name="my-assistant",
    messages=[{"content": "What is Pinecone?"}],
    model="claude-sonnet-4-5",  # was "claude-3-5-sonnet"
)
```

The two old names are still accepted as deprecated aliases and silently
remapped to `claude-sonnet-4-5`, so existing code keeps working, but the
responses come from a different model than the name suggests. `gpt-4o`
remains the SDK default. An unrecognized model name is rejected with a 400
whose message lists the values the backend accepts.

Wherever the API reports why generation stopped, the value `function_call`
has been replaced by `tool_calls`. The SDK types these as plain strings, so
nothing in the SDK breaks, but code matching on the string does:

```python
if response.finish_reason == "function_call":  # never true on 2026-07
    ...
if response.finish_reason == "tool_calls":  # replacement
    ...
```

New, additive fields: `ChatResponse` gains `context_snippet_count` and
`content_filter_results`; `StreamMessageStart` gains `context_snippet_count`;
`StreamMessageEnd` gains `finish_reason`; context-snippet references gain a
`type` field (`text`, `json`, `markdown`, `pdf`, or `doc_x`) naming the kind
of document a snippet came from.

(assistant-files)=
### File uploads and deletes are operations

On `2026-07` the assistant file endpoints stopped answering with the file
and started answering with an operation. `upload_file` and `delete_file`
keep their signatures and return types, the SDK performs the new handshake
internally, but the wire contract and what `delete_file` guarantees when it
returns both changed. This applies to `AsyncPinecone().assistants` identically.

`upload_file` reads the operation, polls it until it reports `Completed`,
and then calls `describe_file`, so this keeps working unchanged:

```python
file = pc.assistants.upload_file(
    assistant_name="my-assistant",
    file_path="/data/report.pdf",
)
file.status  # "Available"
```

Progress now comes from the operation rather than the file's `status`.
`timeout=-1` still skips polling and returns one `describe_file`
immediately, but on `2026-07` that means the file may still exist mid-upload
when the call returns, since it means "request accepted" rather than "file
processed."

Metadata moved from a query parameter into the multipart body. The backend
rejects the old form outright rather than ignoring it. Through the SDK
nothing changes, keep passing a dict:

```python
pc.assistants.upload_file(
    assistant_name="my-assistant",
    file_path="/data/report.pdf",
    metadata={"tags": ["report", "Q4"], "published": "2025-10-01"},
    multimodal=True,
)
```

Upload failures now quote the server. With `error_message` gone from
`AssistantFileModel`, the SDK raises `PineconeError` naming the file, the
operation, and the server's message verbatim:

```python
from pinecone import PineconeError

try:
    pc.assistants.upload_file(assistant_name="my-assistant", file_path="/data/logo.gif")
except PineconeError as exc:
    print(exc)
    # Upload of file 'ae79e447-…' failed (operation_id='op-1234-abcd-5678'):
    # Uploaded file can only currently be either a pdf or txt file
```

Previously a failure surfaced as `File processing failed for '<id>'` with no
reason attached. Code matching on that old text needs updating.

`delete_file` is genuinely asynchronous now. It answers either a `202` with
an operation (deletion pending, polled every 5s), or a `204` with no body
(the file was removed at once). Both are success; the old implementation
polled `describe_file` until it 404'd, so a failed deletion now raises with
the server's reason instead of just timing out:

```python
pc.assistants.delete_file(assistant_name="my-assistant", file_id="file-abc123")
# returns once the deletion operation has completed
```

`timeout=-1` returns as soon as the request is accepted, which on `2026-07`
means the file may still exist when the call returns. Use the default
(poll indefinitely) or a positive deadline if you need the deletion done.

`AssistantFileModel.id` also dropped its UUID format constraint, since the
upsert endpoint lets you choose the identifier. Don't parse a file id as a
UUID.

### The operations API

`list_operations` and `describe_operation` are new methods on both clients.
They report on the long-running work the file endpoints now start:
`upload_file`, a metadata update, and `delete_file` each create an operation
server-side. You don't need them for the default flow, `upload_file` and
`delete_file` poll for you and return only once the work is done. They're
for the cases where you deliberately didn't wait.

If you passed `timeout=-1` to `upload_file`, follow the work by finding the
operation for the file it belongs to:

```python
operations = pc.assistants.list_operations(
    assistant_name="my-assistant",
    operation_type="upload_file",
    status="Processing",
)
mine = [op for op in operations if op.file_id == file.id]
```

`describe_operation` is the progress check, and where the failure reason
lives now:

```python
operation = pc.assistants.describe_operation(
    assistant_name="my-assistant",
    operation_id="op-1234-abcd-5678",
)
if operation.status == "Failed":
    print(operation.error)
```

Read `error` only when `status` is `"Failed"`; a retried operation keeps
the previous attempt's text, so a non-`None` `error` isn't by itself
evidence of failure. The async form is the same call with `await`:

```python
operation = await pc.assistants.describe_operation(
    assistant_name="my-assistant",
    operation_id="op-1234-abcd-5678",
)
```

`list_operations` is a lazy paginator over both in-flight and finished
operations, so it doubles as an audit log, a failed upload is discoverable
after the fact now:

```python
failures = pc.assistants.list_operations(
    assistant_name="my-assistant",
    status="Failed",
).to_list()
```

### Rate limits

The assistant error-code enum gained `TOO_MANY_REQUESTS`. The SDK maps a 429
to `RateLimitError`, which carries `retry_after` when the server sends a
`Retry-After` header, and 429 is in the SDK's default retry set, so the
common case is handled before you see it. Catch it when you want to back off
on your own schedule:

```python
from pinecone import RateLimitError

try:
    pc.assistants.list()
except RateLimitError as exc:
    print(exc.retry_after, exc.error_code)
```

### Checklist

Nothing in this section is caught by a type checker, so grep for it:

- `claude-3-5-sonnet`, `claude-3-7-sonnet` in `model=` arguments
- `"function_call"` compared against a `finish_reason`
- `.error_message` and `.percent_done` on a file object
- `uuid.UUID(...)` applied to a file id
- `timeout=-1` on `upload_file` or `delete_file`, and whether the code after
  it assumes the work finished

(inference-model-enums)=
## Inference

`embed`, `rerank`, `list_models`, and `get_model` keep their signatures,
arguments, and return types. The API version header moved to `2026-07`, and
two long-standing serialization bugs were fixed. Both changed what the SDK
puts on the wire without changing what you write.

`rerank()`'s documented exceptions also gained `NotFoundError`, which is
what an unknown model name has always raised. Code that only caught
`ForbiddenError` around `rerank` was catching the wrong one:

```python
from pinecone import ForbiddenError, NotFoundError

try:
    pc.inference.rerank(model="bge-reranker-v2-m3-typo", query="q", documents=["d"])
except NotFoundError:
    pass  # no such model, usually a typo
except ForbiddenError:
    pass  # the model exists; this project may not use it
```

### Enum members now serialize correctly

`EmbedModel`, `RerankModel`, and `VectorType` are `(str, Enum)` mixins.
Passing a member used to send the wrong value on the wire, because the old
code paths serialized with `str(member)`, which returns the member's
repr-style name (`EmbedModel.Multilingual_E5_Large`) rather than its value
(`multilingual-e5-large`). Passing the plain string always worked, and
that's unaffected here.

If every call you make passes `model=` or `vector_type=` as a string
literal, nothing changes for you. If any call passes an enum member, that
call was failing before and works now: it failed loudly, not silently, since
no such model or vector type exists, so the server rejected the request
rather than serving it wrong. Passing `.value` was the documented workaround
while these bugs were open, and it's still correct; `.value` is exactly what
the SDK now extracts for you, so there's nothing to clean up if you already
adopted it.

The SDK doesn't repair a mangled name. Passing the literal string
`"EmbedModel.Multilingual_E5_Large"` sends it verbatim and still raises
`NotFoundError`, because that's a model id the server doesn't have.

#### Request bodies: `embed()` and `rerank()`

```python
from pinecone import EmbedModel

str(EmbedModel.Multilingual_E5_Large)   # 'EmbedModel.Multilingual_E5_Large'
EmbedModel.Multilingual_E5_Large.value  # 'multilingual-e5-large'
```

| `model=` argument | `"model"` sent before | `"model"` sent now |
| --- | --- | --- |
| `"multilingual-e5-large"` | `multilingual-e5-large` | `multilingual-e5-large` |
| `EmbedModel.Multilingual_E5_Large` | `EmbedModel.Multilingual_E5_Large` | `multilingual-e5-large` |
| `RerankModel.Bge_Reranker_V2_M3` | `RerankModel.Bge_Reranker_V2_M3` | `bge-reranker-v2-m3` |

```text
Model 'EmbedModel.Multilingual_E5_Large' not found
```

```python
from pinecone import EmbedModel

pc.inference.embed(model=EmbedModel.Multilingual_E5_Large.value, inputs=["hello"])
```

A model id the installed SDK has no member for, one released after it, is
still accepted as a plain string and sent through unchanged:

```python
pc.inference.embed(model="some-newer-embedding-model", inputs=["hello"])
```

```python
pc.inference.embed(model="EmbedModel.Multilingual_E5_Large", inputs=["hello"])
# NotFoundError: Model 'EmbedModel.Multilingual_E5_Large' not found
```

(query-param-enums)=
#### Query strings: `list_models()`

`inference.list_models()` accepts `vector_type` as a plain string or as a
`VectorType` member; the fix applies there and to the
`pc.inference.model.list()` facade, on both the sync and the async client.

```python
from pinecone import VectorType

str(VectorType.DENSE)   # 'VectorType.DENSE'
VectorType.DENSE.value  # 'dense'
```

| `vector_type=` argument | query sent before | query sent now |
| --- | --- | --- |
| `"dense"` | `vector_type=dense` | `vector_type=dense` |
| `VectorType.DENSE` | `vector_type=VectorType.DENSE` | `vector_type=dense` |
| `VectorType.SPARSE` | `vector_type=VectorType.SPARSE` | `vector_type=sparse` |

```text
Invalid vector_type, expected one of [dense, sparse]
```

```python
from pinecone import VectorType

pc.inference.list_models(type="embed", vector_type=VectorType.DENSE.value)
```

The fix lives at the encoder that every request passes through, so no query
parameter on any surface, including one added later, can carry a mangled
member. Passing the literal string `"VectorType.DENSE"` is rejected before a
request is made, it isn't one of the values the parameter accepts:

```python
pc.inference.list_models(vector_type="VectorType.DENSE")
# PineconeValueError: vector_type must be one of 'dense', 'sparse', got 'VectorType.DENSE'
```

(admin-oauth)=
## Admin and OAuth

Both surfaces are additive at `2026-07`: no field on any pre-existing
request or response changed, was renamed, or was removed. Existing `Admin`
code compiles and behaves as it did before. What's new is four namespaces,
users, invites, service accounts, and role bindings, which together make
organization membership and RBAC manageable from the SDK for the first time.

`Admin` is synchronous only. There's no async form, admin calls are
infrequent control-plane operations.

`ADMIN_API_VERSION` is now `2026-07`, sent on every admin request and on the
OAuth token exchange. Nothing else about authentication changed, the OAuth
surface is a pure version bump.

### New: users, invites, service accounts, role bindings

| Resource | SDK methods |
| --- | --- |
| Users | `admin.users.list()`, `.describe()`, `.delete()` |
| Invites | `admin.invites.list()`, `.create()`, `.describe()`, `.delete()`, `.resend()` |
| Service accounts | `admin.service_accounts.list()`, `.create()`, `.describe()`, `.update()`, `.delete()`, `.rotate_secret()` |
| Role bindings | `admin.role_bindings.list()`, `.create()`, `.describe()`, `.delete()` |

New models, all additive: `UserModel`, `UserList`; `InviteModel`,
`InviteList`, `InviteStatus`; `ServiceAccountModel`, `ServiceAccountList`,
`ServiceAccountWithSecret`; `RoleBindingModel`, `RoleBindingList`,
`RoleBindingInput`, `PrincipalType`, `ResourceType`, `RoleName`. Everywhere a
role binding is accepted, a plain dict works too. The four listing
operations above return a lazy `Paginator`, unlike the older
`projects`/`organizations`/`api_keys` listings, which return eager `*List`
objects.

The three new resources compose, this is the workflow they exist for:

```python
from pinecone import Admin
from pinecone.models.admin import PrincipalType, ResourceType, RoleName

admin = Admin(client_id="...", client_secret="...")
project = admin.projects.create(name="search-prod")

# 1. A service account for CI, with no permissions yet.
created = admin.service_accounts.create(name="ci-search-prod")
store_secret(created.client_secret)  # returned exactly once

# 2. Grant it ownership of that one project. Bindings are what confer access;
#    a service account with none can get a token but do nothing with it.
admin.role_bindings.create(
    principal_type=PrincipalType.SERVICE_ACCOUNT,
    principal_id=created.service_account.id,
    resource_type=ResourceType.PROJECT,
    resource_id=project.id,
    role=RoleName.PROJECT_OWNER,
)

# 3. Invite a human, with their initial roles in the same call. At least one
#    organization-scoped membership role is required.
invite = admin.invites.create(
    email="teammate@example.com",
    role_bindings=[
        {"resource_type": "organization", "role": "OrgMember"},
        {"resource_type": "project", "role": "ProjectViewer",
         "resource_id": project.id},
    ],
)

# 4. Read any principal's access back through role_bindings, not through the
#    principal's own model, no other namespace carries bindings.
for binding in admin.role_bindings.list(
    principal_type="service_account",
    principal_id=created.service_account.id,
):
    print(binding.role, binding.resource_type, binding.resource_id)
```

Two things worth knowing about that shape. Role bindings aren't part of any
principal's own representation: `UserModel`, `ServiceAccountModel`, and
`InviteModel` don't carry them, so `admin.role_bindings.list()` filtered by
principal is the only way to read them. And bindings are immutable, there's
no update. Changing a role is `create` for the new one then `delete` for the
old one, in that order, deleting first can strip a principal's last
organization-membership binding, which the server refuses with a 409.

### Invites: `list` never shows accepted invites

`InviteStatus` has three values: `pending`, `expired`, and `processed`.
`admin.invites.list()` returns only the first two. An accepted invite
disappears from the listing without being deleted, and
`admin.invites.describe(invite_id=...)` still returns it with
`status == InviteStatus.PROCESSED`. Don't treat absence from `list()` as
proof an invite never existed, or as license to send a duplicate, a second
`create` for an address that already belongs to a member is a 409.

```python
ids = {i.id for i in admin.invites.list()}
"9c8e3528-..." in ids                      # False, could mean accepted
admin.invites.describe(invite_id="9c8e3528-...").status  # 'processed'
```

Once an invite is accepted, manage the invitee through `admin.users` and
`admin.role_bindings`, not `admin.invites`. `delete` and `resend` on a
processed invite are both a 409.

### Pointing `Admin` at a non-production host

Previously `Admin` ignored `PINECONE_CONTROLLER_HOST` and always talked to
production. It now applies the same host resolution as `Pinecone`: the
`host` keyword first, then `PINECONE_CONTROLLER_HOST`, then the default.
This is a behavior change for anyone who had that variable set in an
environment where `Admin` also runs, even though no API surface changed,
admin traffic that used to go to production will now follow the variable.

```python
admin = Admin(client_id="...", client_secret="...", host="http://localhost:5080")
```

`oauth_url` is a second new keyword, pointing the token exchange somewhere
other than production. It takes the full URL including the path, and has no
environment-variable fallback:

```python
admin = Admin(
    client_id="...", client_secret="...",
    host="http://localhost:5080",
    oauth_url="http://localhost:5080/oauth/token",
)
```

Both are keyword-only, and intended for local simulators and private
deployments. Leave both unset against production.

### Token refresh

`Admin` now keeps its own bearer token current: it re-mints ahead of the
stated expiry, and retries a request once against a fresh token if one
still comes back 401. A long-lived `Admin` no longer starts returning bare
401s after its first token lapses, and threads sharing one `Admin` cost a
single token exchange between them. Passing your own `Authorization` entry
in `additional_headers` opts out of refresh entirely, the token is then
yours to manage.

### Project deletion now clears assistants too

`admin.projects.delete()` requires an empty project, and indexes,
collections, assistants, and backups each block it with a 412 naming what's
left. API keys aren't a blocker, they're deleted with the project.
`admin.projects.delete_with_cleanup()` clears all four, assistants included.
Earlier releases left assistants behind, so a project holding one still
failed the final delete after a nominally successful cleanup; that gap is
closed.

(ssl-config)=
## TLS/SSL configuration

`Pinecone`, `AsyncPinecone`, `Index`, and `AsyncIndex` accept `ssl_ca_certs`
and `ssl_verify`, and `Admin` accepts `ssl_verify`. Until this release none
of them did anything: both settings were resolved from config and handed to
the underlying HTTP client, but the SDK's own transport (added for
connection retries and TCP keep-alive tuning) was built with the library's
defaults regardless of what you passed. Every connection used the default
trust store no matter what you configured. The signatures didn't change, so
there's no code to edit, but the requests your client makes may now be
verified differently than before.

| keyword arguments | TLS before | TLS now |
| --- | --- | --- |
| `{}` | default trust store, hostname checked | default trust store, hostname checked |
| `{"ssl_verify": False}` | default trust store, hostname checked | verification off, hostname not checked |
| `{"ssl_ca_certs": "bundle.pem"}` | default trust store, hostname checked | only that bundle trusted, hostname checked |
| `{"ssl_ca_certs": "ca-dir"}` | default trust store, hostname checked | only that directory trusted, hostname checked |
| `{"ssl_ca_certs": "missing.pem"}` | default trust store, hostname checked | `FileNotFoundError` when the client is built |

`ssl_ca_certs` continues to win over `ssl_verify` when both are given, as it
always has: supplying a bundle means you want that bundle trusted.

If you pass neither argument, nothing changes, this is the overwhelmingly
common case. If you pass `ssl_ca_certs` because you sit behind a
TLS-inspecting proxy, your bundle is now the one that's actually trusted,
which is what you asked for originally; if your connections were succeeding
before on the default trust store, they'll now succeed or fail on your
bundle instead. If you pass `ssl_verify=False`, verification and hostname
checking are now genuinely off. Traffic is still encrypted, but the SDK no
longer confirms it's talking to the host it dialled, so only use it against
an endpoint you control.

If you pass an `ssl_ca_certs` path that doesn't exist, building the client
now raises `FileNotFoundError` instead of silently ignoring the setting. A
path that exists but holds no readable certificate raises `ssl.SSLError`.
`Pinecone` and `Index` raise at construction; `AsyncPinecone` and
`AsyncIndex` build their connection pool on first use, so they raise at the
first request instead.

`Admin(ssl_verify=False)` now applies to the OAuth token exchange as well as
the Admin API requests that follow it, the token exchange uses its own
client, so before this release the setting was ignored on both.

`GrpcIndex` has no `ssl_ca_certs` or `ssl_verify` of its own, but its
`secure=False` is forwarded to the REST client that backs `upsert_records`
and `search`, where it means the same as `ssl_verify=False` above. Those two
operations are unverified under `secure=False` where before they were
verified. The gRPC channel itself is unaffected by that change; which scheme
it dials is a separate setting, below.

(grpc-scheme)=
### The gRPC endpoint scheme

`GrpcIndex` dials `https`. A data plane reached over something else — a
plaintext gateway, an egress proxy fronting a private endpoint, or a local
simulator — has to say so, with `grpc_scheme`:

```python
from pinecone import Pinecone

pc = Pinecone(api_key="your-api-key", grpc_scheme="http")
idx = pc.index(host="10.0.0.7:50051", grpc=True)
```

`GrpcIndex` takes the same keyword argument when you construct one yourself:

```python
from pinecone.grpc import GrpcIndex

idx = GrpcIndex(
    host="http://10.0.0.7:50051",
    api_key="your-api-key",
    grpc_scheme="http",
)
```

`PINECONE_GRPC_SCHEME` sets it for a process that configures its hosts
through the environment; an explicit keyword argument wins over the variable.

The scheme is what decides whether the wire carries TLS. `secure` supplies
the material for the handshake — system root certificates for the channel,
certificate verification for the REST calls made alongside it — so
`grpc_scheme="http"` is a plaintext channel whatever `secure` says. The one
combination that cannot connect, `grpc_scheme="https"` with `secure=False`,
raises `PineconeValueError` when the index is constructed rather than
failing on the first call.

Leaving `grpc_scheme` unset keeps the scheme following `secure`: `https`
when it is `True`, `http` when it is `False`. Code that reaches a plaintext
data plane today by passing `secure=False` therefore keeps working
unchanged. Naming the scheme is the narrower way to do it, since it leaves
`upsert_records` and `search` verifying certificates.

Because a plaintext channel sends the API key in the clear, an `http` scheme
resolved against a host outside loopback and the RFC 1918 private ranges
emits a `RuntimeWarning` once per process, naming the host. A private
endpoint, a `192.168.x.x` gateway, or a local simulator is silent.
