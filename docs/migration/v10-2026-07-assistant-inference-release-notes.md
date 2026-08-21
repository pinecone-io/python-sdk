# 2026-07 release notes: assistant and inference

What changed for callers of `pc.assistants` and `pc.inference` when the SDK
moved to the `2026-07` Pinecone API, and what to do about each item. Everything
here applies to `AsyncPinecone` identically, differing only in `await`.

The assistant surface has breaking changes. The inference surface does not — its
API version header moved to `2026-07` and two long-standing serialization bugs
were fixed, which changes what goes on the wire but not what you write.

## Breaking changes at a glance

| What broke | What to do instead | Detail |
| --- | --- | --- |
| `chat(model=...)` no longer documents `claude-3-5-sonnet` / `claude-3-7-sonnet` | pass `claude-sonnet-4-5` | [assistant chat](v10-2026-07-assistant-chat.md) |
| `finish_reason` value `function_call` | match on `tool_calls` | [assistant chat](v10-2026-07-assistant-chat.md) |
| `AssistantFileModel.error_message` | `describe_operation(...).error` | [assistant models](v10-2026-07-assistant-models.md) |
| `AssistantFileModel.percent_done` | `describe_operation(...).percent_complete` | [assistant models](v10-2026-07-assistant-models.md) |
| `upload_file` / `delete_file` are asynchronous server-side | nothing, unless you pass `timeout=-1` | [assistant files](v10-2026-07-assistant-files.md) |
| `upload_file(timeout=-1)` no longer means "file processed" | it means "request accepted"; see below | [assistant files](v10-2026-07-assistant-files.md) |
| file ids are no longer guaranteed to be UUIDs | stop parsing them as UUIDs | [assistant models](v10-2026-07-assistant-models.md) |
| upload metadata travels as a multipart form field | nothing through the SDK | [assistant files](v10-2026-07-assistant-files.md) |

The two `claude-3-*` names are still accepted as deprecated aliases rather than
rejected — the backend remaps them — so that row breaks the documented surface,
not your running code. The rest are visible at runtime.

Nothing in the list changes a method signature or a return type. The compiler
and your type checker will not find these for you; the checklist at the bottom
is the substitute.

## New: the operations API

`list_operations` and `describe_operation` are new public methods on both
clients. They report on the long-running work that the file endpoints now start:
`upload_file`, the upsert form of `upload_file`, a metadata update, and
`delete_file` each create an operation server-side.

You do not need them for the default flow. `upload_file` and `delete_file` poll
for you and return only once the work is done, so the operations API is for the
cases where you deliberately did not wait.

### Following a fire-and-forget upload

`timeout=-1` returns as soon as the request is accepted. On `2026-07` that is a
real change in meaning: previously it returned a file that had been processed,
now it returns a file whose processing has only been *started*.

```python
file = pc.assistants.upload_file(
    assistant_name="my-assistant",
    file_path="/data/report.pdf",
    timeout=-1,
)
file.status  # whatever the server reports right now, not a finished state
```

`upload_file` does not hand back the operation id on this path, so to follow the
work you find the operation by the file it belongs to:

```python
operations = pc.assistants.list_operations(
    assistant_name="my-assistant",
    operation_type="upload_file",
    status="Processing",
)
mine = [op for op in operations if op.file_id == file.id]
```

From there `describe_operation` is the progress check, and it is also where the
failure reason lives now that `AssistantFileModel.error_message` is gone:

```python
operation = pc.assistants.describe_operation(
    assistant_name="my-assistant",
    operation_id="op-1234-abcd-5678",
)
if operation.status == "Failed":
    print(operation.error)
```

Read `error` only when `status` is `"Failed"`. A retried operation keeps the
previous attempt's text, so a non-`None` `error` is not by itself evidence of
failure.

The async form is the same call with `await`:

```python
operation = await pc.assistants.describe_operation(
    assistant_name="my-assistant",
    operation_id="op-1234-abcd-5678",
)
```

### Auditing what an assistant has been asked to do

`list_operations` is a lazy paginator over both in-flight and finished
operations, so it doubles as an audit log for a window of recent history — a
failed upload is discoverable after the fact, which it was not before.

```python
failures = pc.assistants.list_operations(
    assistant_name="my-assistant",
    status="Failed",
).to_list()
```

`list_operations_page` is the explicit-pagination form if you want to drive the
continuation token yourself.

## New: `region` on assistant creation

`create` takes `region`, which is `"us"` or `"eu"` and defaults to `"us"`. It is
fixed at creation time — there is no move-an-assistant operation — so an
`eu`-resident assistant has to be created that way:

```python
assistant = pc.assistants.create(name="eu-assistant", region="eu")
assistant.region  # 'eu'
```

`AssistantModel.region` reports it back. Not every deployment can serve `eu`;
where it cannot, the request is refused with a message saying so.

## New: `TOO_MANY_REQUESTS` in the error-code enum

The assistant error-code enum gained `TOO_MANY_REQUESTS`. The SDK maps a `429`
to `RateLimitError`, which carries `retry_after` when the server sends a
`Retry-After` header, and `429` is in the SDK's default retry set — so the
common case is handled before you see it. Catch it when you want to back off on
your own schedule:

```python
from pinecone.errors.exceptions import RateLimitError

try:
    pc.assistants.list()
except RateLimitError as exc:
    print(exc.retry_after, exc.error_code)
```

## Inference: no API changes

`embed`, `rerank`, `list_models` and `get_model` keep their signatures, their
arguments and their return types. The API version header moved to `2026-07`;
nothing in the request or response shapes changed with it.

Two serialization bugs were fixed in the same release. Both changed what the SDK
puts on the wire without changing what you write, so they are worth reading even
though there is nothing to migrate:

- Enum members passed as `model=` reached the wire as `EmbedModel.X` rather than
  the model id — see [inference model enums](v10-2026-07-inference-model-enums.md).
- Enum members passed as *query parameters* were mangled the same way — see
  [query parameter enums](v10-2026-07-query-param-enums.md).

`rerank()`'s documented exceptions also gained `NotFoundError`, which is what an
unknown model name has always raised. Code that only caught `ForbiddenError`
around `rerank` was catching the wrong one:

```python
from pinecone.errors.exceptions import ForbiddenError, NotFoundError

try:
    pc.inference.rerank(model="bge-reranker-v2-m3-typo", query="q", documents=["d"])
except NotFoundError:
    pass  # no such model — usually a typo
except ForbiddenError:
    pass  # the model exists; this project may not use it
```

## Checklist

Nothing here is caught by a type checker, so grep for it:

- `claude-3-5-sonnet`, `claude-3-7-sonnet` in `model=` arguments
- `"function_call"` compared against a `finish_reason`
- `.error_message` and `.percent_done` on a file object
- `uuid.UUID(...)` applied to a file id
- `timeout=-1` on `upload_file` or `delete_file`, and whether the code after it
  assumes the work finished
- `except ForbiddenError` around a `rerank` call
