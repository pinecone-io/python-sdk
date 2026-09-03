# Migrating from v8.x to v9.x

v9 is a ground-up rewrite focused on simplicity, performance, and type safety. This guide
covers the breaking changes and shows you the v9 equivalent for each v8 pattern.

## Key changes

### 1. Namespace pattern for control-plane operations

In v8, control-plane methods lived directly on the `Pinecone` client:

```python
# v8
pc.create_index(
    name="my-index", dimension=1536, metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)
indexes = pc.list_indexes()
pc.delete_index("my-index")
```

In v9, they are grouped under namespace properties:

```python
# v9
pc.indexes.create(
    name="my-index", dimension=1536, metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)
indexes = pc.indexes.list()
pc.indexes.delete("my-index")
```

The same pattern applies to collections, backups, and inference:

```python
pc.collections.create(name="catalog-snapshot", source="my-index")
pc.backups.list()
pc.inference.embed(model="multilingual-e5-large", inputs=["hello"])
```

**You are not forced to move.** `pc.create_index`, `pc.describe_index`,
`pc.list_indexes`, `pc.configure_index`, `pc.delete_index`,
`pc.create_collection`, `pc.list_collections` and `pc.delete_collection` all
still exist on the client, forwarding to the namespace method. They are hidden
from the reference docs rather than removed, so v8 call sites keep working and
the migration can be incremental.

One thing does change even on the shims: they are keyword-only.
`pc.create_index("my-index", ...)` raises `PineconeValueError` naming the
keywords it accepts; pass `name=` instead. `pc.describe_index("my-index")`,
`pc.delete_index("my-index")` and `pc.configure_index("my-index", ...)` still
take the name positionally.

`create` and `delete` both block until the index reaches its end state, polling
with no upper bound unless you pass `timeout`. Pass `timeout=-1` to return as
soon as the request is accepted.

### 2. Async client rename

`PineconeAsyncio` is renamed to `AsyncPinecone`. The old name is still importable, as a
plain alias of the new class — same object, so `isinstance` checks are unaffected.

```python
# v8
from pinecone import PineconeAsyncio
async with PineconeAsyncio(api_key="...") as pc:
    ...

# v9
from pinecone import AsyncPinecone
async with AsyncPinecone(api_key="...") as pc:
    ...
```

### 3. Response models

v8 returned a mix of plain dicts, dataclass models, and bespoke objects. v9 returns
`msgspec.Struct` instances. Attribute access is unchanged — `idx.name`, `idx.host` — and
so is subscripting a declared field, `idx["name"]`. What no longer works is handing the
model to something that expects a real mapping: `dict(idx)` raises `KeyError`.

```python
import msgspec

idx = pc.indexes.describe("my-index")
print(idx.name)                     # works
print(idx["name"])                  # works — declared fields only
print(idx.to_dict())                # nested plain dicts, JSON-ready
print(msgspec.structs.asdict(idx))  # shallow: nested models stay as objects
print(dict(idx))                    # raises KeyError — a Struct is not a mapping
```

Reach for `to_dict()` when you want JSON, and `msgspec.structs.asdict()` when you want
the declared fields with the nested models intact.

### 4. HTTP transport: httpx replaces urllib3

The SDK uses `httpx` instead of `urllib3`. Retry behavior is now configured
with `RetryConfig` passed at client construction:

```python
# v8 — retry parameters were keyword args on the client
pc = Pinecone(api_key="...", retries=3)

# v9
from pinecone import Pinecone, RetryConfig
pc = Pinecone(
    api_key="...",
    retry_config=RetryConfig(max_retries=3, backoff_factor=1.5),
)
```

Unlike the control-plane methods above, this one is not shimmed: an unrecognised keyword
raises `TypeError: Pinecone() got unexpected keyword arguments: ['retries']`. See
[Retries and Resilience](../guides/retries.md) for what `retry_config` reaches and what
it does not.

### 5. gRPC: Rust extension replaces grpcio

`GrpcIndex` is now backed by a compiled Rust extension instead of the Python `grpcio`
package. There is no `grpc` extra to install and no `grpcio` or `grpcio-tools`
dependency to add — the transport ships in the base `pinecone` package. The data-plane
interface — `upsert`, `query`, `fetch`, `delete` — is unchanged.

```python
# v9 — interface is the same; no grpcio dependency required
index = pc.index("my-index", grpc=True)
index.upsert(vectors=[("product-42", [0.012, -0.087, 0.153])])
```

`from pinecone.grpc import PineconeGRPC` still works too, and
`PineconeGRPC(...).Index(name=...)` still returns a `GrpcIndex` — it is a thin subclass
of `Pinecone` kept for v8 call sites. [Using the gRPC Client](../guides/grpc.md) covers
the transport, including what `GrpcIndex` does not carry.

### 6. Import paths

Most public classes are still importable directly from `pinecone`:

```python
from pinecone import Pinecone, AsyncPinecone, Index, GrpcIndex
from pinecone import ServerlessSpec, PodSpec
from pinecone import ConflictError, NotFoundError, ForbiddenError
```

Several pre-rewrite module paths survive as shims that re-export from the new
locations, so imports from them keep working: `pinecone.pinecone`,
`pinecone.pinecone_asyncio`, `pinecone.exceptions`, `pinecone.control` and
`pinecone.data`. They are hidden from the reference docs; new code should import from
`pinecone` directly.

Three others exist but re-export nothing you can import by name — `pinecone.config`,
`pinecone.db_control` and `pinecone.db_data`. `from pinecone.db_data import Index`
raises `ImportError`; import `Index` from `pinecone`.

The generated OpenAPI tree is gone: `from pinecone.core.client.api...` raises
`ModuleNotFoundError`. Use the top-level package instead.

### 7. Python version requirement

Python 3.9 support is dropped. The minimum supported version is Python 3.10.

### 8. Removal of the `pinecone_plugins.assistant` import path

In v8, the assistant SDK shipped as a separate plugin package
(`pinecone-plugin-assistant`) installed alongside `pinecone`. Code
imported model classes from `pinecone_plugins.assistant.*`:

```python
# v8
from pinecone_plugins.assistant.models import (
    AssistantModel, ContextOptions, Message, FileModel,
)
from pinecone_plugins.assistant.models.chat import ChatResponse
```

In v9 the assistant API is built into the main `pinecone` package
and the `pinecone_plugins` import tree has been removed. **All
classes are now reachable from `pinecone.models.assistant`** under
either the canonical name or a legacy alias.

Replace each legacy import with the canonical path:

| v8 import path | v9 import path |
|---|---|
| `from pinecone_plugins.assistant.models import AssistantModel` | `from pinecone.models.assistant import AssistantModel` |
| `from pinecone_plugins.assistant.models import ContextOptions` | `from pinecone.models.assistant import ContextOptions` |
| `from pinecone_plugins.assistant.models import Message` | `from pinecone.models.assistant import Message` |
| `from pinecone_plugins.assistant.models import FileModel` | `from pinecone.models.assistant import FileModel` *(deprecated alias for `AssistantFileModel`)* |
| `from pinecone_plugins.assistant.models.chat import ChatResponse` | `from pinecone.models.assistant import ChatResponse` |
| `from pinecone_plugins.assistant.models.chat import Citation, Reference, Highlight` | `from pinecone.models.assistant import Citation, Reference, Highlight` *(deprecated aliases for `ChatCitation` etc.)* |
| `from pinecone_plugins.assistant.models.chat import StreamChatResponseMessageStart, StreamChatResponseContentDelta, StreamChatResponseCitation, StreamChatResponseMessageEnd` | `from pinecone.models.assistant import StreamChatResponseMessageStart, StreamChatResponseContentDelta, StreamChatResponseCitation, StreamChatResponseMessageEnd` *(deprecated aliases for `StreamMessageStart` etc.)* |
| `from pinecone_plugins.assistant.models.chat_completion import ChatCompletionResponse, StreamingChatCompletionChunk` | `from pinecone.models.assistant import ChatCompletionResponse, StreamingChatCompletionChunk` |
| `from pinecone_plugins.assistant.models.context_responses import ContextResponse, TextSnippet, MultimodalSnippet` | `from pinecone.models.assistant import ContextResponse, TextSnippet, MultimodalSnippet` |
| `from pinecone_plugins.assistant.models.context_responses import TextBlock, ImageBlock, Image` | `from pinecone.models.assistant import TextBlock, ImageBlock, Image` *(deprecated aliases for `ContextTextBlock`, `ContextImageBlock`, `ContextImageData`)* |
| `from pinecone_plugins.assistant.models.context_responses import PdfReference, TextReference, JsonReference, MarkdownReference, DocxReference` | `from pinecone.models.assistant import PdfReference, TextReference, JsonReference, MarkdownReference, DocxReference` *(all five alias the consolidated `FileReference`)* |
| `from pinecone_plugins.assistant.models.evaluation_responses import AlignmentResponse, Metrics, EvaluatedFact` | `from pinecone.models.assistant import AlignmentResponse, Metrics, EvaluatedFact` *(deprecated aliases for `AlignmentResult`, `AlignmentScores`, `EntailmentResult`)* |
| `from pinecone_plugins.assistant.models.list_files_response import ListFilesResponse` | `from pinecone.models.assistant import ListFilesResponse` |
| `from pinecone_plugins.assistant.models.list_assistants_response import ListAssistantsResponse` | `from pinecone.models.assistant import ListAssistantsResponse` |
| `from pinecone_plugins.assistant.models.shared import Message, Usage, TokenCounts` | `from pinecone.models.assistant import Message, Usage, TokenCounts` *(`Usage` and `TokenCounts` are deprecated aliases for `ChatUsage`)* |
| `from pinecone_plugins.assistant.assistant.assistant import Assistant` | No replacement — see note below. |

**What does *not* change.** Method-call backcompat is preserved:

```python
# Both v8 and v9
pc = Pinecone(api_key="...")
pc.assistant.create_assistant("my-assistant")        # works in v9
pc.assistant.list_assistants()                       # works in v9
assistant = pc.assistant.describe_assistant("my-assistant")
assistant.upload_file(file_path="report.pdf")        # works in v9
assistant.chat(messages=[...])                       # works in v9
```

The `pc.assistant` namespace is preserved and singular/plural
forms (`pc.assistant` and `pc.assistants`) are interchangeable.
Legacy method names like `create_assistant`, `delete_assistant`,
`list_assistants_paginated`, etc. continue to work alongside the
canonical `pc.assistants.create`, `.delete`, `.list_page`.

**The legacy plugin class is removed.** Code that manually
instantiated the plugin (`Assistant(config, client_builder)` from
`pinecone_plugins.assistant.assistant.assistant`) has no v9
equivalent — the plugin discovery system was retired and
`pc.assistant` is now a property on the `Pinecone` client. Such
code must be rewritten to use `pc.assistants` directly.


### 9. Partial-success contract for batched upsert

`index.upsert(...)` keeps v8's `batch_size` and `show_progress`
and adds `max_concurrency` — so a v8 call site still compiles.
The failure behaviour underneath it changed.

**v8 behaviour.** `idx.upsert(vectors=[...], batch_size=N)`
raised on the first batch failure. Successful batches were
lost; subsequent batches were not attempted.

**v9 behaviour.** Batches are submitted concurrently, with
`max_concurrency` capping how many are in flight at once
(accepted range 1–64; the default changed after v9 — see
[the next migration guide](v10-migration.md)).
Per-batch HTTP retries happen automatically via
{class}`~pinecone.RetryConfig`. Failures that exceed the
retry budget are captured on the returned
{class}`~pinecone.models.UpsertResponse` rather than raised.
The response now carries `total_item_count`,
`failed_item_count`, `total_batch_count`,
`successful_batch_count`, `failed_batch_count`,
`errors: list[BatchError]`, plus convenience properties
`has_errors`, `error_count`, `success_count`, and
`failed_items`.

Code that wraps a batched upsert in `try/except` will silently
undercount in v9 unless updated:

```python
# v8 — relied on exception to roll back
try:
    idx.upsert(vectors=batch, batch_size=100)
except Exception:
    rollback()

# v9 — inspect response.has_errors instead
response = idx.upsert(vectors=batch, batch_size=100)
if response.has_errors:
    # Optionally retry only the failures:
    idx.upsert(vectors=response.failed_items, batch_size=100)
    # …or roll back if any failure is unacceptable
```

Single-request upsert (`batch_size=None`, the default) keeps
its v8 raise-on-failure semantics. The contract change applies
only when `batch_size` is set.

The same partial-success contract applies to
`AsyncIndex.upsert(...)` and `GrpcIndex.upsert(...)`.

---

## v8 → v9 migration table

| Operation | v8 | v9 |
|---|---|---|
| Create index | `pc.create_index(name=..., dimension=..., spec=...)` | `pc.indexes.create(name=..., dimension=..., spec=...)` |
| List indexes | `pc.list_indexes()` | `pc.indexes.list()` |
| Describe index | `pc.describe_index("name")` | `pc.indexes.describe("name")` |
| Delete index | `pc.delete_index("name")` | `pc.indexes.delete("name")` |
| Configure index | `pc.configure_index("name", ...)` | `pc.indexes.configure("name", ...)` |
| Check index exists | `pc.describe_index("name")` + try/except | `pc.indexes.exists("name")` |
| Get data-plane index | `pc.Index("name")` | `pc.index("name")` *(lowercase in v9; `pc.Index()` is a deprecated shim)* |
| Get gRPC index | `PineconeGRPC(...).Index("name")` | `pc.index("name", grpc=True)` *(`PineconeGRPC` remains as a deprecated shim)* |
| Create collection | `pc.create_collection(name=..., source=...)` | `pc.collections.create(name=..., source=...)` |
| List collections | `pc.list_collections()` | `pc.collections.list()` |
| Delete collection | `pc.delete_collection("name")` | `pc.collections.delete("name")` |
| Upsert vectors | `index.upsert(vectors=[...])` | `index.upsert(vectors=[...])` *(unchanged for single-request; batched form gains `max_concurrency` + partial-success contract — see §9)* |
| Query vectors | `index.query(vector=[...], top_k=10)` | `index.query(vector=[...], top_k=10)` *(unchanged)* |
| Fetch vectors | `index.fetch(ids=[...])` | `index.fetch(ids=[...])` *(unchanged)* |
| Delete vectors | `index.delete(ids=[...])` | `index.delete(ids=[...])` *(unchanged)* |
| Async client | `PineconeAsyncio(api_key=...)` | `AsyncPinecone(api_key=...)` |
| Retry config | `Pinecone(retries=3)` | `Pinecone(retry_config=RetryConfig(max_retries=3))` |
| Convert response to dict | `dict(idx)` or `idx.to_dict()` | `idx.to_dict()` *(still present; `dict(idx)` raises — see §3)* |
| Embed text | `pc.inference.embed(...)` | `pc.inference.embed(...)` *(unchanged)* |

---

## Legacy aliases

The following aliases remain importable from `pinecone` but are deprecated:

| Deprecated name | Canonical name |
|---|---|
| `PineconeAsyncio` | `AsyncPinecone` |
| `ValidationError` | `PineconeValueError` *(the one alias that emits a `DeprecationWarning` on access)* |
| `ForbiddenException` | `ForbiddenError` *(`ForbiddenException` still works as a deprecated alias)* |
| `NotFoundException` | `NotFoundError` *(`NotFoundException` still works as a deprecated alias)* |
| `pinecone_plugins.assistant.*` (any submodule) | `pinecone.models.assistant`, `pinecone.client.assistants.Assistants` (via `pc.assistant` / `pc.assistants`) |

Each is a plain alias of its canonical name, so `isinstance` and `except` clauses
behave identically. Update your code to the canonical names; nothing in the SDK forces
the issue today.
