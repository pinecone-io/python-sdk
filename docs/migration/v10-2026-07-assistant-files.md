# 2026-07: assistant file uploads and deletes are operations

On `2026-07` the assistant file endpoints stopped answering with the file and
started answering with an *operation*. `upload_file` and `delete_file` keep
their signatures and their return types — the SDK performs the new handshake
internally — but the wire contract, the failure messages, and what `delete_file`
guarantees when it returns all changed.

Everything below applies to `AsyncPinecone().assistants` identically: the same
requests, the same statuses, the same polling cadence and the same error text,
differing only in `await`.

## `upload_file` still returns an `AssistantFileModel`

`POST /files/{assistant_name}` and `PUT /files/{assistant_name}/{file_id}` now
answer `202 Accepted` with an `OperationModel` and a `Location` header pointing
at the operation, instead of `200` with the file. The SDK reads the operation,
takes the file id from `OperationModel.file_id` (or from your `file_id=` on the
upsert path), polls the operation until it reports `Completed`, and then calls
`describe_file` — so this keeps working unchanged:

```python
file = pc.assistants.upload_file(
    assistant_name="my-assistant",
    file_path="/data/report.pdf",
)
file.status  # "Available"
```

Two consequences worth knowing:

- Progress now comes from the operation, which reports `percent_complete`,
  rather than from the file's `status`. `timeout=-1` still skips polling and
  returns one `describe_file` immediately.
- If you were reading the raw HTTP response through some other client, a
  `200`-with-a-file body is no longer what you get.

## Metadata moves from the query string into the multipart body

This is the change most likely to break a caller that talks to the API directly.
`2026-07` removed the `metadata` query parameter and expects a `metadata`
multipart form field holding the same JSON (still capped at 16KB). The backend
does not ignore the old form — it rejects it:

```
400 INVALID_ARGUMENT: metadata query parameter is not supported in this API
version; include metadata as a multipart form field instead
```

`multimodal` is unaffected and remains a query parameter. Through the SDK
nothing changes: keep passing a dict, and it lands in the right place.

```python
pc.assistants.upload_file(
    assistant_name="my-assistant",
    file_path="/data/report.pdf",
    metadata={"tags": ["report", "Q4"], "published": "2025-10-01"},
    multimodal=True,
)
```

## Upload failures now quote the server

`AssistantFileModel.error_message` and `percent_done` are gone (see
[the assistant models note](v10-2026-07-assistant-models.md)), so the failure
reason comes from the operation record instead. The SDK raises `PineconeError`
naming the file, the operation, and the server's message verbatim:

```python
try:
    pc.assistants.upload_file(assistant_name="my-assistant", file_path="/data/logo.gif")
except PineconeError as exc:
    print(exc)
    # Upload of file 'ae79e447-…' failed (operation_id='op-1234-abcd-5678'):
    # Uploaded file can only currently be either a pdf or txt file
```

Previously a failure surfaced as `File processing failed for '<id>'` with no
reason attached. Code matching on that old text needs updating; code that reads
`exc` as a whole gets strictly more information.

If the server accepts an upload but does not say which file it created, the SDK
raises rather than guessing, and points at `describe_operation()` with the
operation id.

## `delete_file` is genuinely asynchronous

`DELETE /files/{assistant_name}/{assistant_file_id}` answers either:

| Status | Meaning | What the SDK does |
| --- | --- | --- |
| `202` + `OperationModel` | deletion is pending | polls the operation every 5s until it finishes |
| `204`, no body | the file was removed at once (it had previously failed processing, or was already being deleted) | returns immediately |

Both are success. The old implementation ignored the response body and instead
polled `describe_file` until it 404'd; it now polls the operation, which means a
failed deletion raises with the server's reason:

```python
pc.assistants.delete_file(assistant_name="my-assistant", file_id="file-abc123")
# returns once the deletion operation has completed
```

`timeout=-1` returns as soon as the request is accepted — on `2026-07` that
means **the file may still exist when the call returns**. Use the default
(`None`, poll indefinitely) or a positive deadline if you need the deletion to
be done. `PineconeTimeoutError` now names the operation id and its
`percent_complete` instead of saying the file "still exists".

## File ids are no longer guaranteed to be UUIDs

`AssistantFileModel.id` dropped its `format: uuid` constraint, because the
upsert endpoint lets you choose the identifier (1-128 characters of
`[A-Za-z0-9_-]`). Do not parse a file id as a UUID.

## One data-plane client per assistant

Internal, but visible in request counts: `list_files`/`list_files_page` and the
upsert path used to build a fresh, uncached HTTP client per call, each costing
an extra control-plane `describe`. They now share the one cached data-plane
client the other file methods use, so a listing loop no longer re-describes the
assistant on every page.
