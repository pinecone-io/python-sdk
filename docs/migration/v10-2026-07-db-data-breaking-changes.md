# 2026-07: db_data breaking changes and migration table

Release-notes source of truth for the **`db_data` surface** — the vector
operations, the document operations, namespaces, and bulk imports on `Index`,
`AsyncIndex` and `GrpcIndex`. Every field-level break on this release appears in
the table below, followed by one section per change with before/after code.

Three changes on this surface are written up in full elsewhere. This page states
the consequence and links them rather than repeating them, so that each fact has
exactly one home.

Every `2026-07` code block below is executed against a stubbed data plane in
`tests/unit/test_docs_migration_db_data_138.py`, which asserts the request the
SDK really sends and the exact text of every error message printed here.

## Read this first: which indexes the vector operations serve

`upsert`, `query`, `fetch`, `update`, `delete`, `list` and
`describe_index_stats` **serve indexes created under earlier API versions, and
they are meant to.** None of them is deprecated, none is scheduled for removal,
and none changed meaning on this release. If you have a workload upserting and
querying an index you created before `2026-07`, upgrading the SDK does not
change what those calls do — preserving exactly that is why they are still here.

What changed is index creation. A `2026-07` `pc.indexes.create()` always
persists a document schema, and a document-schema index is addressed through the
**document** operations. `2026-07` deliberately provides no way to create an
index for the vector operations. Whether index creation for the vector data
plane returns in some form is an open internal question and is not part of this
release.

So pick the family by the index you are addressing, not by which one looks
newer:

| The index you are addressing | The operations that serve it |
| --- | --- |
| Created under an API version earlier than `2026-07` | `upsert`, `query`, `fetch`, `fetch_by_metadata`, `update`, `delete`, `list`, `describe_index_stats` |
| Created with `2026-07` (document schema) | `upsert_documents`, `search_documents`, `fetch_documents`, `update_documents`, `delete_documents`, `list_documents` |

A vector-API **write** aimed at a document-schema index is refused, and the
server's message names the endpoint to use instead — it begins *"This index has
a document schema, so writes must go through the documents API."* Read that as
"wrong operation family for this index", not as a deprecation notice.

The document operations are REST-only; `GrpcIndex` has no document methods. The
full document-surface write-up, including the preview-to-GA renames, is
[preview namespace removed](v10-2026-07-preview-graduation.md).

## The migration table

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

Rows 1, 3 and 4 change the bytes the SDK puts on the wire. Rows 2, 6 and 7
change documentation that was wrong, not behavior. Row 5 changes what a working
index declaration looks like. Row 8 is here so you can rule it out.

## 1. `query(top_k=...)` is bounded at both ends on every lane

`query` has always had an upper bound on `top_k`. Until this release only
`GrpcIndex` enforced it: `Index` and `AsyncIndex` checked `top_k >= 1` and
forwarded anything larger, so an oversized value cost a round trip and came
back as a server error. All three lanes now share one range check and one
message.

```python
idx.query(vector=[0.1, 0.2], top_k=20000, namespace="movies-en")
# PineconeValueError: top_k must be between 1 and 10000, got 20000
```

The bound is `1`-`10000` on `Index`, `AsyncIndex` and `GrpcIndex` alike, and it
is checked before any request is made. `top_k=1` and `top_k=10000` are both
accepted.

Two things to know before you treat this as a new restriction:

- **A call the server was going to reject now fails locally instead.** The
  request never left the process, so there is nothing to un-send and no partial
  work to reason about.
- **The ceiling is a deployment setting, not a constant of the API.** A
  deployment configured lower than the client's ceiling still rejects values
  this check lets through, and that rejection arrives as an `ApiError` carrying
  the server's own wording. Handle both if you page near the top of the range.

If you were catching a server error to detect an oversized `top_k`, catch
`PineconeError` — `PineconeValueError` and `ApiError` both derive from it — or
clamp before you call.

## 2. `describe_index_stats(filter=...)` never returns a filtered count

The three `describe_index_stats` docstrings and the upsert-and-query how-to
described a working metadata filter, and the sync and async pages scoped the
restriction to particular index types. Both were wrong. **A non-empty `filter`
is rejected for every index type**, so a filtered stats call fails rather than
returning a subset count.

There is **no operation anywhere on this surface that counts only the records
matching a metadata filter.** That is the part worth acting on: if you built a
count on this argument, it was never returning a filtered number, and there is
nothing to migrate it to.

```python
# This has never worked. The call fails; it does not return a subset count.
stats = idx.describe_index_stats(filter={"genre": {"$eq": "action"}})
```

Drop the argument. The statistics you get back describe the whole index:

```python
stats = idx.describe_index_stats()
print(stats.total_vector_count, stats.dimension)
```

**No behavior changed here** — all three lanes still forward the filter
unvalidated, exactly as before, and the server still rejects it. What changed is
that the documentation now says so. To count a subset, query for it or maintain
the count yourself.

## 3. `SchemaBuilder` metadata fields always emit `filterable`

`add_boolean_field()`, `add_float_field()` and `add_string_list_field()` left
`filterable` out of the emitted field whenever it was `False` — which is the
default, so the shortest documented call omitted it. The server requires the key
on all three field types and refused a body without it, so those calls produced
a create the backend could not accept at all.

The key is now always present:

```python
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

Six emitted bodies went from rejected to accepted, covering each of the three
methods at its default, a field carrying a `description`, all three fields in
one schema, and a metadata-only schema with no vector field. `filterable=True`
was unaffected: it always emitted the key and always worked.

Two consequences:

- **If you assert on `build()`'s output**, add `filterable` to the expected
  boolean, float and string-list fields. That is the whole diff — the dense
  vector field is untouched.
- **`add_string_field()` is deliberately unchanged.** Its wire shape is
  different enough that the same edit would break it, and its intended shape is
  an open question tracked as
  [#391](https://github.com/pinecone-io/python-sdk-internal/issues/391).

```{warning}
**String fields are not settled on this release, and this page does not show
one.** A string field can be declared for metadata filtering or for full-text
search, and the full-text-search spelling does not currently reach the backend
as full-text search — the field is created as filter-only metadata, with no
error and no warning, so the omission is invisible until a search returns
nothing.

This affects `StringField` / `IndexSchema` — tracked as
[#414](https://github.com/pinecone-io/python-sdk-internal/issues/414) — and it
is why the schema examples on this page declare only vector and non-string
metadata fields. **Do not take a full-text-search string field from any
example as working until #414 closes.** Boolean, float and string-list fields,
which are what row 3 is about, are unaffected.
```

## 4. `add_sparse_vector_field()` no longer emits `metric`

Through 9.x the builder put `{"type": "sparse_vector", "metric": "dotproduct"}`
on the wire. A `2026-07` sparse field has no `metric` — sparse scoring is not
configurable — so the key was **accepted and silently dropped**. It configured
nothing and `describe` never echoed it back.

**No create was ever failing because of this key**, and this row is not a fix
for a broken call. What it fixes is a key that read like configuration while
having no effect. The break is in the SDK's output:

```python
schema = (
    SchemaBuilder()
    .add_dense_vector_field("embedding", dimension=1536, metric="dotproduct")
    .add_sparse_vector_field("sparse_terms")
    .build()
)
schema["fields"]["sparse_terms"]
# before: {'type': 'sparse_vector', 'metric': 'dotproduct'}
# now:    {'type': 'sparse_vector'}
```

Passing `metric=` or `dimension=` — the two things `vector_type="sparse"`
implied in 9.x — now raises instead of forwarding a key that does nothing:

```python
SchemaBuilder().add_sparse_vector_field("sparse_terms", metric="dotproduct")
# PineconeValueError: Field 'sparse_terms' cannot declare 'metric': a sparse
# vector field has no metric — sparse scoring is not configurable. Remove the
# argument — a sparse vector field accepts only a description.
```

Delete the argument; sparse vectors are variable-length and their scoring has no
knob. The full write-up is the sparse-writes section of
[vector model changes](v10-2026-07-vector-models.md).

## 5. A hybrid index must declare a `sparse_vector` field

In `2025-10`, `metric="dotproduct"` on a dense index was the entire hybrid
declaration. In `2026-07` sparse traffic is gated on the schema actually
declaring a `sparse_vector` field, and the metric buys nothing on either the
read or the write side.

This is the change on this surface most likely to reach production unnoticed,
because **there is no signal at create time**: a create with a dense
`dotproduct` field and no sparse field succeeds and serves dense traffic
normally. The first symptom is a refused sparse write, often from a different
service and long after the upgrade. **Audit for `metric="dotproduct"` before you
upgrade** — in 9.x that keyword *was* the declaration, so every occurrence is a
candidate for a missing sparse field.

You cannot add the field to an existing index; an index created without one has
to be recreated, which is why this belongs in the upgrade plan rather than in a
later fix. Legacy indexes keep their prior behavior — this row is about indexes
you create with `2026-07`.

The full write-up — the create call, the error you actually see, and the stale
server message not to take literally — is the sparse-writes section of
[vector model changes](v10-2026-07-vector-models.md).

## 6. `start_import(error_mode=...)` defaults to `"abort"`

Two of the three lanes' docstrings, plus the bulk-import how-to, said the
default was `"continue"` — meaning a record the import cannot read is skipped
and the rest still import. **The default is `"abort"`:** the import ends at the
first record it cannot read, so nothing is dropped without telling you.

The server has always behaved this way. Only the documentation was wrong, so no
running import changes — but if you omitted `error_mode` on the strength of that
sentence, you have been getting the opposite of what you read.

```python
# Ends the whole import at the first unreadable record (the default).
idx.start_import(uri="s3://my-bucket/vectors/")

# Opt in to skipping unreadable records and importing the rest.
idx.start_import(uri="s3://my-bucket/vectors/", error_mode="continue")
```

`ImportErrorMode.CONTINUE` and `ImportErrorMode.ABORT` are equivalent to the
strings and always have been, on every lane.

## 7. `create_namespace(schema=...)` omitted means *inherit*, not *index everything*

Omitting `schema` does not create a namespace with every field indexed. The
namespace **inherits the index's own metadata-index configuration**, so an index
that restricts which fields are indexed passes that restriction on to the new
namespace.

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
only. Each field listed must set `filterable: True`; `filterable: False` is not
accepted, and to leave a field unindexed you omit it from `fields` entirely.

Behavior did not change — this was already how the server worked. If you relied
on the omitted form to index everything on a restricted index, supply `schema`
explicitly.

## 8. Enum members on the wire: no `db_data` call was affected

Two enum-mangling defects were fixed on this release: request bodies carried
`"EmbedModel.X"` instead of the model id, and query strings carried
`"VectorType.DENSE"` instead of `dense`. Both are written up on their own pages
— [inference model enums](v10-2026-07-inference-model-enums.md) for bodies and
[query-parameter enums](v10-2026-07-query-param-enums.md) for query strings.

**They are recorded here so you can rule this surface out.** `db_data` exposes
one enum-valued argument, `start_import(error_mode=...)`, and it was never
affected: the value is lower-cased on the way into the request body, and
lower-casing a `(str, Enum)` member yields the member's value rather than its
name. No `db_data` operation takes an enum-valued query parameter.

The query-string fix landed at the encoder every request passes through, so a
`db_data` query parameter added after this release cannot reintroduce the
problem either.

## What you do **not** need to change

- **Your call shapes.** Nothing on this page changes a method signature. If you
  are coming from `pinecone.preview`, the document methods were renamed and some
  of their parameters did change — that move is
  [preview namespace removed](v10-2026-07-preview-graduation.md), not this page.
- **Any working `query` call.** Row 1 only rejects values the server was going
  to reject anyway.
- **Vector operations against a legacy index.** Rows 3, 4 and 5 are about
  creating an index, so none of them reaches one you already have; a legacy
  index keeps the behavior it had. Rows 1, 2, 6, 7 and 8 are about calls, so
  they apply wherever you make them.
- **`describe_index_stats` without a filter.** Unchanged, on all three lanes.
- **Batched query bodies you never wrote.** The API's removal of the deprecated
  multi-query request field is invisible from the SDK, which never exposed it —
  see [vector model changes](v10-2026-07-vector-models.md) if you also send raw
  HTTP.
- **`memory_fullness` / `storage_fullness`.** The OAS corrected their spelling;
  the SDK's attribute names are unchanged. Also in
  [vector model changes](v10-2026-07-vector-models.md).
- **Anything about `error_mode` or `create_namespace(schema=...)` that you had
  already got right.** Rows 6 and 7 correct documentation, not behavior.

## Where each change is documented in full

| Change | Full write-up |
| --- | --- |
| Hybrid `sparse_vector` requirement, `add_sparse_vector_field` | [vector model changes](v10-2026-07-vector-models.md) |
| Removed multi-query request field, fullness field spelling, metadata value validation | [vector model changes](v10-2026-07-vector-models.md) |
| Document operations, preview-to-GA renames, gRPC has no documents | [preview namespace removed](v10-2026-07-preview-graduation.md) |
| Index creation with `schema=` and `deployment=` | [db_control create/configure](v10-2026-07-db-control.md) |
| Enum members in request bodies | [inference model enums](v10-2026-07-inference-model-enums.md) |
| Enum members in query strings | [query-parameter enums](v10-2026-07-query-param-enums.md) |
