# 2026-07: vector model changes

The `db_data` vector models now follow the Pinecone `2026-07` API shapes. Three
things changed, and only the third one changes SDK behavior.

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
