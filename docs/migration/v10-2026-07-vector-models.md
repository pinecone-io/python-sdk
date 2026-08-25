# 2026-07: vector model changes

The `db_data` vector models now follow the Pinecone `2026-07` API shapes. Three
things changed there, and only the third one changes SDK behavior. A fourth
section covers a `db_data` **behavior** change with no model change at all —
[sparse writes now require a declared `sparse_vector` field](#sparse-writes) —
which is the one on this page most likely to reach production unnoticed.

## 1. `QueryRequest.queries` and `QueryVector` are gone

The `2026-07` API deletes the deprecated `queries` field from the query request
and removes the `QueryVector` schema entirely.

The SDK never exposed either name, so **there is nothing to change in code that
uses `Index.query()`**. Pass your query vector as `vector=` (dense),
`sparse_vector=` (sparse), or both.

If you bypassed the typed surface and sent a raw request body containing
`queries`, the server now returns `400`. Batching several queries in one request
is no longer expressible; issue them separately, or use `query_namespaces()` when
the same query fans out across namespaces.

## 2. The fullness fields are spelled in camelCase

`describe_index_stats()`'s response carries two optional fullness fields for
dedicated indexes. The `2025-10` OAS documented them as `memory_fullness` and
`storage_fullness`; the wire has always sent `memoryFullness` and
`storageFullness`, and the `2026-07` OAS corrects the spelling to match.

**No change for Python SDK users.** `DescribeIndexStatsResponse` already decoded
the camelCase spelling, so `response.memory_fullness` and
`response.storage_fullness` keep working and keep their snake_case attribute
names. The attribute names are the SDK's convention and are unaffected by the
wire spelling.

This matters only if you also consume a third-party or generated client pinned to
the `2025-10` OAS: that client was reading a field name the server never sent, and
would have been reporting `None` all along.

## 3. Metadata values are validated before the request is sent

`2026-07` formalizes what a metadata value may be — a **string, a number, a
boolean, or a list of strings**:

```python
index.upsert([("id-1", [0.1, 0.2], {
    "genre": "documentary",     # string
    "year": 2019,               # number
    "featured": True,           # boolean
    "tags": ["short", "indie"], # list of strings
})])
```

The server has always enforced this. What is new is that the SDK now checks it
too, so a bad value raises locally instead of failing the whole batch server-side:

```python
index.upsert([("id-1", [0.1, 0.2], {"price": {"usd": 10}})])
# PineconeTypeError: Metadata value must be a string, number, boolean or list of
# strings, got '{"usd":10.0}' for field 'price'
```

The message — including the rendered value and the offending field name — is the
server's own message verbatim, so an error you already handle by string does not
change shape.

### What now raises

| Metadata value | Before | Now |
| --- | --- | --- |
| `{"nested": "object"}` | server `400` for the whole batch | `PineconeTypeError` before any HTTP call |
| `[1, 2]` (list of numbers) | server `400` | `PineconeTypeError` |
| `["a", 1]` (mixed list) | server `400` | `PineconeTypeError` |
| `["a", None]` | server `400` | `PineconeTypeError` |
| `b"bytes"`, or any other type | server `400` or an encoding error | `PineconeTypeError` |

The check runs on every input form `upsert()` accepts — `Vector` objects,
`(id, values, metadata)` tuples, and dicts — and on the gRPC DataFrame path.

**If you were relying on a server `400` to find bad rows,** you now get a
`PineconeTypeError` from `upsert()` instead. Catch `PineconeError` to cover both,
or `TypeError`, which `PineconeTypeError` also subclasses.

### What does *not* raise

A `None` value is **accepted**, not rejected:

```python
index.upsert([("id-1", [0.1, 0.2], {"tag": None})])  # no error
```

The server strips null metadata values on write rather than refusing them, so
this has always silently dropped the `tag` key rather than storing it. The SDK
does not raise here, because doing so would break code that works today. If you
meant to store an absent value, omit the key; if you meant to remove a stored
field, use the update operation's field-removal spelling.

This now holds on **both transports**. The gRPC transport used to encode a
`None` metadata value as a protobuf `NullValue` and the server refused it with
a 400 — the same upsert succeeded over REST and failed over gRPC. As of this
release the gRPC transport strips `None`-valued metadata keys before encoding,
exactly as the server's own JSON path does, so the two transports agree: the
key is silently dropped. `None` inside a *filter* is still sent through on
both transports and rejected by the server.

An empty list (`[]`) and an empty string (`""`) are also accepted, matching the
server.

### Metadata keys

Keys may not begin with `$`, which is reserved for metadata filter operators.
Every other key is accepted, including empty and non-ASCII keys. The SDK does not
check key names; the server rejects a `$`-prefixed key with
`Metadata field '<name>' cannot start with '$'`.

(sparse-writes)=

## 4. Sparse writes now require a declared `sparse_vector` field

This one changes no model and no method signature, which is exactly why it is
worth reading: **it is a breaking change that produces no error at the point
where you have to fix it.**

In `2025-10`, `metric="dotproduct"` on a dense index was the whole hybrid
declaration: nothing else had to be said, and sparse values worked. In
`2026-07` that index shape does not exist. Sparse traffic is gated on the
schema actually declaring a `sparse_vector` field, so a hybrid index must
declare one explicitly.

Worth being precise about what `dotproduct` used to buy, because it is not
symmetric. The `2025-10` **write** path accepted sparse values on any
schemaless index whatever its metric; `dotproduct` is what made the **query**
side accept them, which is why it became the de facto hybrid declaration. In
`2026-07` both sides are gated on the declared field, so the metric buys
neither.

The request-side mechanics of the new create call are in
[db_control create/configure](v10-2026-07-db-control.md); this section is the
full write-up that guide's hybrid warning defers to.

### The rule

`supports_sparse_vector_writes()` returns `schema.has_sparse_vector_field()`
whenever the index has a schema, and `true` only when it has none —
`pc-types/src/cps.rs:682-687` @ pinecone-db `cbee5a67`.

What removes the old behavior is not that rule by itself but the fact that
`2026-07` `POST /indexes` **always** persists a schema
(`.../global/base/index_v2/indexes.rs:553-557`), so the permissive schemaless
branch is unreachable for anything you create on this API version. Whether the
dense field's metric is `dotproduct` is no longer part of the question.

Sparse **reads** are gated the same way, through `supports_sparse_vectors()`
(`cps.rs:667-679`), which keeps the dense-plus-`dotproduct` fallback only for
schemaless indexes.

### Before and after

```python
# Deprecated sugar (dimension=/metric=/spec=) still accepts this 9.x-style
# call, and that is exactly the trap: it creates a dense-only schema (the
# reserved `_values` field) with no sparse_vector field, so sparse writes
# still fail — silently, with no error at create time.
pc.create_index(
    name="hybrid",
    dimension=1536,
    metric="dotproduct",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)
```

The `2026-07` equivalent names both vector fields explicitly. Pick names your
upsert and query code will address — there is no default, and the deprecated
form above cannot invent one for you. `dotproduct` stays on the dense field;
the sparse field takes neither `dimension` nor `metric`, both of which
`vector_type="sparse"` used to imply.

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

**You cannot add the sparse field afterwards.** `configure()`'s `schema=` reaches
a server-side patch type with exactly one variant, `semantic_text`, under
`deny_unknown_fields` (`.../global/base/index_v2/mod.rs:507-524`), so a
`sparse_vector` field cannot be introduced by a `PATCH`. An index created
without one has to be recreated — which is why this belongs in your upgrade
plan rather than in a later fix.

### What actually fails, and with which error

Not the error you would predict, and the answer depends on
[#322](https://github.com/pinecone-io/python-sdk-internal/issues/322), which is
open.

On any index you create with `2026-07`, a vectors-API upsert is refused **before
the sparse check is ever reached**. `validate_vectors_api_write_allowed` is the
first statement of upsert validation
(`pc-validation/src/data_plane/mod.rs:495`), ahead of the namespace check and
ahead of the per-vector sparse check at `:229-235`. Its message:

> This index has a document schema, so writes must go through the documents
> API. Use `POST /namespaces/{namespace}/documents/upsert`, `/update` or
> `/delete` with the `X-Pinecone-API-Version` header set to `2026-07` or later.
> The vectors and records write endpoints do not support document schemas.

(`pc-validation/src/error.rs:278-285`.) Declaring the sparse field does not
satisfy that gate. `data_plane_api()` routes a `semantic_text`-only schema to
the records API and every other schema that declares data fields — a hybrid
pair included — to the documents API (`cps.rs:953-982`). The one schema shape
that keeps the vectors API is a schema holding nothing beyond the reserved names
`_values` / `_sparse_values` (`index_schema_def.rs:11-12`, `:155-162`), and
`validate_field_name` rejects every `_`-prefixed name at create time
(`index_schema_def.rs:52-54`) — so a schema you declare can never qualify.

**So `SparseNotSupported` is not the `400` you will see today.** If you are
debugging a hybrid upsert against a freshly created `2026-07` index, the
document-schema message above is the string to search for.

[#322](https://github.com/pinecone-io/python-sdk-internal/issues/322) is
tracking that gate with live observations on both REST and gRPC, and this
section deliberately does not pre-empt it. The relationship is one-directional:
the create-time requirement in this section is decided and already shipped, and
it is what governs **if and when** #322 resolves in a way that lets vectors-API
writes reach a document-schema index. At that point an index with no declared
`sparse_vector` field starts refusing sparse upserts with `SparseNotSupported`,
and the shape above is what avoids it. If #322 resolves the other way, the
wording here about which error surfaces may need revising; the requirement to
declare the field does not.

On the documents API — where a `2026-07` index is routed today — the same
requirement shows up at **search** time, and more legibly. A sparse scoring
field that is not in the schema fails with `Scoring field '<name>' not found in
index schema` (`svc-docs-api/src/core/documents/validate/mod.rs:361-365`), which
names the actual problem. Same lesson: the schema, not the metric, is what
decides.

```{warning}
**The server's own sparse error text is out of date — do not take it literally.**

`SparseNotSupported` still reads *"Index configuration does not support sparse
values - only indexes that are sparse or using dotproduct are supported"*
(`pc-validation/src/error.rs:101-104`). The "or using dotproduct" clause
describes precisely the `2025-10` behavior this change removed, so an operator
who follows the message's advice will set `metric="dotproduct"` and see no
improvement.

Read it as *"only indexes whose schema declares a `sparse_vector` field"*. The
documents-API variant, *"Sparse vectors are not supported for index with metric
'<metric>'"* (`.../documents/validate/mod.rs:405-415`), misattributes the cause
the same way.

This is a backend message-text problem, not something the SDK rewrites — the SDK
surfaces server messages verbatim on purpose. It is filed for relay to the
backend team as
[#355](https://github.com/pinecone-io/python-sdk-internal/issues/355) rather than
paraphrased into something the server does not say.
```

### Why it fails silently

There is no signal at create time. A create with a dense `dotproduct` field and
no sparse field succeeds, returns a healthy `IndexModel`, and serves dense
traffic normally. The gate lives on the write path, so the first symptom is a
refused upsert in whatever code sends sparse values — frequently a different
service, and long after the upgrade.

Audit for `metric="dotproduct"` before upgrading. In 9.x that keyword *was* the
hybrid declaration, so every occurrence is a candidate for a missing
`sparse_vector` field.

### Declaring the field with `SchemaBuilder`

`SchemaBuilder` does make the requirement discoverable:
`add_sparse_vector_field()` sits directly beside `add_dense_vector_field()`, so
the hybrid pair reads as a single chain and you never have to know the wire
spelling.

```python
schema = (
    SchemaBuilder()
    .add_dense_vector_field("embedding", dimension=1536, metric="dotproduct")
    .add_sparse_vector_field("sparse_terms")
    .build()
)
```

The chain and the dict above put identical bytes on the wire, so either
spelling is a working call.

#### ⚠ `add_sparse_vector_field()` no longer emits `metric`

Through 9.x the builder put `{"type": "sparse_vector", "metric": "dotproduct"}`
on the wire. The `2026-07` create schema has no `metric` on a sparse field —
`SparseVectorField` declares `type` and `description` and nothing else — so the
key configured nothing and was never echoed back by describe. It is gone:

    # before
    {"type": "sparse_vector", "metric": "dotproduct"}
    # now
    {"type": "sparse_vector"}

Two consequences for existing code:

- **If you assert on `build()`'s output**, drop `metric` from the expected
  sparse field. This is the only shape change; `add_dense_vector_field()` is
  untouched and still emits `dimension` and `metric`.
- **If you pass `metric=` or `dimension=` to `add_sparse_vector_field()`** —
  the two things `vector_type="sparse"` implied in 9.x — the call now raises
  `PineconeValueError` naming the field and the key. Both used to travel to the
  server through the builder's forward-compatibility keywords and be discarded
  there, which read like configuration that had taken effect. Delete the
  argument: a sparse field's scoring is not configurable, and sparse vectors
  are variable-length.
