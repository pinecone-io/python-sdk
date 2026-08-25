# 2026-07: preview namespace removed — surfaces graduated

The `2026-01.alpha` preview surface graduated. Everything that lived under
`pc.preview` is now a first-class part of the SDK, and the
`pinecone/preview/` package is **deleted outright** — there is no shim and no
deprecation window. Preview was never covered by SemVer, and its docstrings
said so; this is the graduation that disclaimer was reserving room for.

Nothing about this release is a silent behavior change. Every stale preview
import or attribute access raises, immediately, at the point of use:

| Stale code | Now raises |
| --- | --- |
| `import pinecone.preview` | `ModuleNotFoundError` |
| `from pinecone.preview.models import PreviewIndexModel` | `ModuleNotFoundError` |
| `from pinecone.preview._internal.constants import INDEXES_API_VERSION` | `ModuleNotFoundError` |
| `pc.preview` on `Pinecone` or `AsyncPinecone` | `AttributeError` |
| `from pinecone import PreviewIndexModel` (or any `Preview*` name) | `ImportError` |

So the migration is mechanical to *find*. The work is in the handful of
places where a name moved **and changed shape** — those are marked ⚠ below,
and they are the only parts of this document that need reading rather than
skimming.

## Who this guide is for

You were on `pc.preview.*`. If you were on the stable `9.x` surface
(`pc.create_index(dimension=..., metric=..., spec=...)`), you never touched
the preview namespace and nothing here applies to you — read
[the IndexModel guide](v10-2026-07-index-model.md) instead, which carries the
`9.x` → `10.x` call-shape table for the same graduated methods.

## 1. Entry points

The graduated surface hangs off the client directly. `pc.preview` was one
extra hop; delete it and the rest of the expression is almost unchanged.

| Removed | Replacement |
| --- | --- |
| `pc.preview.indexes` | `pc.indexes` — the graduated `pinecone.client.indexes.Indexes` |
| `pc.preview.index(name=...)` / `(host=...)` | `pc.index(name=...)` / `(host=...)`, or positionally `pc.index("my-index")` |
| `pc.preview.index(...).documents.upsert(...)` | `pc.index(...).documents.upsert(...)` — same shape, see §3 |
| `pc.preview.close()` | `pc.close()` — the client closes every namespace it opened |
| `from pinecone.preview import SchemaBuilder` | `from pinecone import SchemaBuilder` (the direct path `from pinecone.schema_builder import SchemaBuilder` is unchanged) |
| `pinecone.preview.Preview` / `AsyncPreview` | no replacement — the router classes have no successor; there is nothing left to route to |
| `pinecone.preview.indexes.PreviewIndexes` | `pinecone.client.indexes.Indexes` |
| `pinecone.preview.async_indexes.AsyncPreviewIndexes` | `pinecone.async_client.indexes.AsyncIndexes` |
| `pinecone.preview.index.PreviewIndex` | `pinecone.index.Index` |
| `pinecone.preview.async_index.AsyncPreviewIndex` | `pinecone.async_client.async_index.AsyncIndex` |
| `pinecone.preview.documents.PreviewDocuments` | `pinecone.client.documents.Documents`, via `index.documents` |
| `pinecone.preview.async_documents.AsyncPreviewDocuments` | `pinecone.async_client.documents.AsyncDocuments`, via `index.documents` |

```python
from pinecone import Pinecone

pc = Pinecone(api_key="your-api-key")

# was: pc.preview.indexes.describe("articles-en-preview")
info = pc.indexes.describe("articles-en")

# was: index = pc.preview.index(name="articles-en-preview")
with pc.index(name="articles-en") as index:
    result = index.documents.fetch(namespace="articles-en", ids=["doc-1"])
```

### ⚠ `await pc.index(...)` is now a coroutine

`AsyncPreview.index()` was deliberately synchronous: it handed back an
`AsyncPreviewIndex` immediately and resolved the host lazily, on the first
data-plane call. `AsyncPinecone.index()` resolves eagerly and **must be
awaited**.

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

Two consequences worth knowing before you add the `await`:

- **A bad index name now fails at `pc.index(...)`**, not on the first
  data-plane call. If you were catching `NotFoundError` around your first
  `search`/`fetch`, move the `try` up to the `await pc.index(...)`.
- **`pc.index(...)` cannot be called outside a running event loop.**
  Module-level `index = pc.preview.index(name=...)` was legal; `await
  pc.index(...)` is not.

Targeting by `host=` still skips the control-plane round trip, so
`await pc.index(host=...)` never raises `NotFoundError` — the `await` is
cheap there, but still required.

The host cache moved from the `Preview` router onto the client, so repeated
`pc.index(name=...)` calls still resolve the host once per client. Both lanes
behave the same way.

## 2. Model imports

Every `Preview*` model dropped its prefix and moved out of
`pinecone.preview.models`. All of the replacements are importable from
`pinecone` and from `pinecone.models`.

### Straight renames — safe find-and-replace

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

`PreviewStringListField` and `PreviewBooleanField` are the same struct under
a new name. `PreviewSparseValues` → `SparseValues` and `PreviewIndexStatus` →
`IndexStatus` each *gained* dict-style access (`status["ready"]` alongside
`status.ready`); nothing was taken away.

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

### ⚠ `PreviewIntegerField` and `PreviewLegacyIntegerField` cross over

This is the one rename where matching on the name gives you the wrong type.
The two preview classes carried **each other's** wire tags:
`PreviewIntegerField` was tagged `"float"` and `PreviewLegacyIntegerField`
was tagged `"integer"`. The graduated names are correct, which makes the
mapping inverted relative to what the names suggest:

| Removed | Wire `type` | Replacement |
| --- | --- | --- |
| `PreviewIntegerField` | `"float"` | **`FloatField`** |
| `PreviewLegacyIntegerField` | `"integer"` | **`IntegerField`** |

A blind `PreviewIntegerField` → `IntegerField` substitution compiles,
type-checks, and then decodes the wrong field type. If you wrote
`isinstance(field, PreviewIntegerField)` to find the whole-number fields in a
schema, that check was already matching floats; it now needs to be
`isinstance(field, IntegerField)`, and it will finally mean what you intended.

`LegacyMetadataField` is new and unrelated to the name it appears to inherit:
it decodes fields from indexes that pre-date typed schemas, which arrive with
**no** `type` key at all, and it exposes only `filterable`. Preview had no
equivalent — such a field failed to decode.

### ⚠ Renames where the shape also changed

A moved import is a find-and-replace. These are real work.

| Removed → Replacement | What changed |
| --- | --- |
| `PreviewDenseVectorField` → `DenseVectorField` | `dimension` and `metric` are **required**, no longer `None`-defaulted |
| `PreviewSparseVectorField` → `SparseVectorField` | **`metric` removed** — sparse fields have no configurable metric |
| `PreviewSemanticTextField` → `SemanticTextField` | `model` is **required**; **`dimension` removed**. Cannot be declared at create time on `2026-07` — it appears in responses only |
| `PreviewFullTextSearchConfig` → `FullTextSearchConfig` | **`lowercase` and `max_term_len` removed**; gained `ngram: NgramConfig \| None` |
| `PreviewPodDeployment` → `PodDeployment` | `replicas` and `shards` are **required**; **`pods` removed** |
| `PreviewByocDeployment` → `ByocDeployment` | **`cloud` and `region` removed** — only `environment` remains |
| `PreviewReadCapacityDedicatedInner` → `ReadCapacityDedicatedConfig` | **`auto` removed**, as well as the rename |
| `PreviewQueryStringQuery` → `QueryStringQuery` | gained `field` and `fields` — a query-string clause can now be scoped to named fields |
| `PreviewUsage` → three types | split, see below |
| `PreviewDocumentSearchResponse` → `SearchDocumentsResponse` | `usage` is now `DocumentSearchUsage` |
| `PreviewDocumentFetchResponse` → `FetchDocumentsResponse` | `usage` is now `DocumentFetchUsage`; gained `pagination` |

`PreviewFullTextSearchConfig` was never in `pinecone.preview.models.__all__` —
it was reachable only as `from pinecone.preview.models.schema import
PreviewFullTextSearchConfig`. `FullTextSearchConfig` is a normal top-level
export, as is the new `NgramConfig`.

`PreviewUsage` was one struct shared by the search and fetch responses. It is
now three, one per operation, so a signature can name the usage type it
actually receives:

| Response | Usage type |
| --- | --- |
| `SearchDocumentsResponse` | `DocumentSearchUsage` |
| `FetchDocumentsResponse` | `DocumentFetchUsage` |
| `ListDocumentsResponse` | `DocumentListUsage` |

All three carry the same single `read_units: int` field, so
`response.usage.read_units` is unchanged. Only a type annotation or an
`isinstance` check needs updating.

### New model names with no preview predecessor

`FloatField`, `LegacyMetadataField`, `NgramConfig`, `FullTextSearchConfig` (as
a top-level export), `DocumentRecord`, `UpdateDocumentRecord`,
`ListedDocumentRecord`, `DeleteDocumentsResponse`, `UpdateDocumentsResponse`,
`ListDocumentsResponse`, `DocumentSearchUsage`, `DocumentFetchUsage`,
`DocumentListUsage`, and the six `*DocumentsRequest` structs.

`PreviewBackupModel` and `PreviewCreateBackupRequest` also disappear with the
package; see [the backup-models guide](v10-2026-07-backup-models.md), which
owns that mapping.

## 3. Document operations: the `.documents` namespace is back

`pc.preview.index(...)` returned a `PreviewIndex` whose only job was to hold a
`.documents` proxy. Graduating out of preview retired that wrapper — correctly,
it existed only to host the proxy — but an earlier revision of this guide also
retired the `.documents` **namespace** along with it, flattening every
operation into a `*_documents`-suffixed method directly on `Index`
(`index.upsert_documents(...)`, `index.search_documents(...)`, etc.). That was
never a considered design decision; it was a side effect of deleting the
wrapper, and it made `Index` harder to navigate for no benefit. The namespace
is restored, and it is the *only* surface: `index.documents` is a
lazily-instantiated property, exactly like every other resource namespace on
this SDK (`pc.indexes`, `pc.inference`, and so on) — the implementation
module isn't imported until you touch it.

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
mirrors all seven; `documents.list(...)` is not a coroutine on either lane —
it returns a paginator.

The `*_documents`-suffixed methods directly on `Index`/`AsyncIndex` never
shipped outside this flattening detour, so there is no deprecated alias to
carry forward and no second call shape to migrate off of: `.documents` is the
one way to reach these operations.

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
        )
        for match in hits.matches:
            print(match.id, match.title)
    await pc.close()


asyncio.run(main())
```

### ⚠ `documents.delete` returns a response object

Preview's `index.documents.delete(...)` returned `None`. The graduated
`index.documents.delete(...)` returns a `DeleteDocumentsResponse` — and it
accepts a `filter`, which is why there is now something to return.

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
filtered delete it is the **point-in-time count when the server accepted the
request** — the delete is applied asynchronously behind a `202`, so it is not
a promise about how many documents ultimately disappear. `matched_records` on
`UpdateDocumentsResponse` means the same thing.

Preview's `ids` / `delete_all` mutual exclusion still holds, widened to three
options: exactly one of `ids`, `filter`, or `delete_all` must be given.

### ⚠ `documents.fetch` gained a filter and pagination

Preview's `fetch` took `ids` only. `documents.fetch` takes exactly one of
`ids` or `filter`, and a filtered fetch is paginated — a page holds up to
10000 documents and the page size is fixed.

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

### Smaller signature deltas

- **`timeout` on every document method.** New keyword, `None` by default,
  which is the previous behavior. Purely additive.
- **`documents.batch_upsert(max_concurrency=...)` no longer accepts `None`.**
  It is `int = 4`; preview's `int | None = None` picked a default internally.
  Passing `None` now fails type checking — pass the number, or omit it.
- **`documents.batch_upsert` dropped `**kwargs`.** Preview silently swallowed
  unknown keywords; a typo is now a `TypeError` at the call site.
- **`documents=` accepts `Sequence[Mapping[...]]`**, not just `list[dict]`, and
  also accepts typed `DocumentRecord` / `UpdateDocumentRecord` objects. Lists
  of dicts keep working.

### gRPC does not serve documents

`pc.index(name=..., grpc=True)` returns a `GrpcIndex`, which has **no**
document operations — the `2026-07` documents surface is REST-only. Preview
never offered a gRPC path either, so nothing regressed; it is worth stating
because `grpc=True` is now reachable from the same factory call you use to get
at documents.

## 4. Index operations: `pc.preview.indexes` → `pc.indexes`

All nine method names carry over unchanged — `create`, `configure`,
`describe`, `list`, `exists`, `delete`, `create_backup`, `list_backups`,
`describe_backup` — so deleting `preview.` from the attribute chain is the
bulk of the change. `AsyncPinecone.indexes` mirrors all nine, with `list` and
`list_backups` non-coroutine on both lanes, exactly as in preview.

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

Four deltas for a preview caller:

- **⚠ `create()` now polls until the index is ready.** Preview returned as soon
  as the server accepted the create. The graduated `create()` polls every 5
  seconds and returns a ready index; `timeout=-1` restores the old
  return-immediately behavior, and a positive int sets a deadline after which
  `PineconeTimeoutError` is raised. If your code ran its own
  `while not pc.preview.indexes.describe(name).status.ready` loop, either drop
  the loop or pass `timeout=-1`.
- **⚠ `create()` no longer accepts `source_collection` or `source_backup_id`.**
  Preview exposed both keywords; the `2026-07` backend rejects both with
  `400 Creating an index from collection or backup is not yet supported`. Use
  `pc.create_index_from_backup(...)` to restore a backup.
- **`create(schema=...)` is optional in the signature** (it was required in
  preview) so the legacy-keyword error path can report a useful message. The
  server still requires it for a real create.
- **`create()` and `configure()` accept a typed `IndexSchema`** as well as a
  plain dict, and `create()` gained `timeout`. `configure()` is otherwise
  identical. `create_for_model()` is new — it has no preview predecessor.

Everything else about the `2026-07` create/configure contract — the schema
rules, the pod required-field set, the `read_capacity` collapse, the
`IndexModel` field removals — is the same for you as for a `9.x` caller and is
documented once, in [the IndexModel guide](v10-2026-07-index-model.md). The
index-scoped backup methods, including the new `include_deleted` on
`list_backups`, are in [the backup-models guide](v10-2026-07-backup-models.md).

## 5. Newly importable from `pinecone`

Independent of the preview retirement, this release wired up top-level
exports that were built earlier in the cycle but never reachable from
`pinecone` or `pinecone.models`. Nothing moved and nothing broke — these names
gained a shorter import path. If you were reaching into
`pinecone.models.admin.*` or `pinecone.models.assistant.*` directly, those
paths still work.

`SchemaBuilder` (also still at `pinecone.schema_builder`), plus the following,
from both `pinecone` and `pinecone.models`:

| Area | Names |
| --- | --- |
| Assistant operations | `OperationModel`, `ListOperationsResponse` |
| Admin — users | `UserModel`, `UserList` |
| Admin — invites | `InviteModel`, `InviteList`, `InviteStatus` |
| Admin — service accounts | `ServiceAccountModel`, `ServiceAccountList`, `ServiceAccountWithSecret` |
| Admin — role bindings | `RoleBindingModel`, `RoleBindingList`, `RoleBindingInput`, `RoleName`, `PrincipalType`, `ResourceType` |
| Admin — pagination | `PaginationResponse` |

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

The operations models pair with the asynchronous assistant file lifecycle —
see [the assistant files guide](v10-2026-07-assistant-files.md) for how
`OperationModel` is used. The admin RBAC models belong to the
`pc.admin.users` / `.invites` / `.service_accounts` / `.role_bindings`
resources.

## 6. API version

The preview constant `INDEXES_API_VERSION = "2026-01.alpha"`
(`pinecone/preview/_internal/constants.py`) is deleted with the package. The
graduated surfaces negotiate `X-Pinecone-Api-Version: 2026-07` via
`CONTROL_PLANE_API_VERSION` / `DATA_PLANE_API_VERSION` in
`pinecone/_internal/constants.py`. Nothing in the SDK sends `2026-01.alpha`
any more, and there is no way to pin a single call back to it — the version
constants are per-surface, not per-endpoint.

If you were reading `INDEXES_API_VERSION` to log or assert the negotiated
version, note that its replacements are internal and not part of the public
API; prefer asserting on the request header your transport actually sent.

## 7. Test and CI fallout

- The `preview_integration` pytest marker is removed from `pyproject.toml`. A
  test suite that selects or deselects it (`-m preview_integration`,
  `-m "not preview_integration"`) now errors on an unknown marker under
  `--strict-markers`.
- If you built fixtures against `PreviewIndexes` or `PreviewDocuments`, they
  need the graduated classes from §1 — `Documents`/`AsyncDocuments` for
  `PreviewDocuments`/`AsyncPreviewDocuments`, accessed via `index.documents`.

## Where the rest of the 2026-07 release is documented

This guide covers only the preview-to-default move. The reshaped
request/response contracts underneath it are documented per surface:

- [IndexModel and index sub-models](v10-2026-07-index-model.md) — `create`,
  `configure`, `list`, and every `IndexModel` field change
- [Backup models](v10-2026-07-backup-models.md) — `BackupModel`,
  `include_deleted`, restore
- [Vector models](v10-2026-07-vector-models.md) — the `db_data` vector surface
- [Assistant models](v10-2026-07-assistant-models.md),
  [chat](v10-2026-07-assistant-chat.md), and
  [files](v10-2026-07-assistant-files.md)
