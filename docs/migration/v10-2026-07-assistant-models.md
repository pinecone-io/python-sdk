# 2026-07: assistant model changes

The assistant response models now follow the Pinecone `2026-07` API shapes.
Removing fields from `AssistantFileModel` is a breaking change.

## AssistantFileModel: removed fields

`AssistantFileModel` no longer has `.percent_done` or `.error_message`. The
`2026-07` API does not return them: file processing progress and failure
detail moved to the operations API.

| Removed | Replacement |
| --- | --- |
| `file.percent_done` | `describe_operation(...)` and read `OperationModel.status` |
| `file.error_message` | `describe_operation(...)` and read `OperationModel.error` |

Accessing either attribute raises an `AttributeError` naming
`describe_operation` as the replacement, so the migration path is visible at
the point of failure:

```python
file = pc.assistant.describe_file(assistant_name="my-assistant", file_id="f-1")
file.error_message
# AttributeError: AssistantFileModel.error_message was removed in the 2026-07
# Pinecone API: processing failure detail is reported by the operations API
# instead — call describe_operation() and read OperationModel.error. ...
```

Dict-style access follows suit: `file["percent_done"]` raises `KeyError`,
`"percent_done" in file` is `False`, and neither name appears in `file.keys()`
or `file.to_dict()`. The `repr` and the notebook HTML rendering no longer show
either value; a `ProcessingFailed` file renders an error block that points at
`describe_operation()` instead of a message.

Responses that still carry the old keys (a `2025-10` server, or a recorded
fixture) continue to decode — the extra keys are ignored rather than
rejected.

## AssistantFileModel: file IDs are no longer UUIDs

`2026-07` documents `id` as a plain string, because a file ID may be one the
caller supplied. Code that parses a file ID as a UUID breaks:

```python
uuid.UUID(file.id)  # no longer safe
```

`size` (bytes, `int64`) is part of the documented `2026-07` shape and is
populated on upload, describe, and list responses.

## AssistantModel: new fields

`AssistantModel` gains `region` — `"us"` or `"eu"`, the region the assistant
is deployed in — alongside the existing `created_at` and `updated_at`
timestamps. All three are optional and are `None` when the API does not
return them, so older recorded responses still parse. `region` now appears in
the `repr` and in the notebook HTML rendering:

```python
assistant = pc.assistant.describe(name="my-assistant")
assistant.region  # 'eu'
```

## Error messages on failed file operations

`upload_file()` and `delete_file()` used to quote
`AssistantFileModel.error_message` in the `PineconeError` they raise on a
failed poll. With the field gone, those messages now name the file state and
direct you to `describe_operation()` for the reason. Code matching on the old
message text needs updating.
