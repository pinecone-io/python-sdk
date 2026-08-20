# Working with the Assistant

The assistant client lets you create and manage AI assistants that can answer questions
over your uploaded documents.

## Create an assistant

```python
from pinecone import Pinecone

pc = Pinecone(api_key="your-api-key")

assistant = pc.assistants.create(
    name="my-assistant",
    instructions="Answer questions based on the uploaded documents.",
)
print(assistant.name)    # "my-assistant"
print(assistant.status)  # "Ready"
```

By default, ``create()`` polls until the assistant reaches ``"Ready"`` status before
returning. To return immediately without waiting (for example to kick off creation
asynchronously), pass ``timeout=-1`` — the returned assistant will be in
``"Initializing"`` status and you can check readiness later via ``describe()``.

## List and describe assistants

``list`` returns all assistants in the project:

```python
for asst in pc.assistants.list():
    print(asst.name, asst.status)
```

``describe`` returns details for a single assistant:

```python
asst = pc.assistants.describe(name="my-assistant")
print(asst.name)         # "my-assistant"
print(asst.status)       # "Ready"
print(asst.instructions) # the instruction string
```

## Upload a file

Pass a local file path to upload context documents for the assistant to read:

```python
file = pc.assistants.upload_file(
    assistant_name="my-assistant",
    file_path="data.pdf",
)
print(file.id)     # file ID used for later operations
print(file.name)   # "data.pdf"
print(file.status) # "Processing" → "Available"
```

To upload bytes you already hold, pass `file_stream` — plus a `file_name` that
carries the extension. The server types an uploaded file by its extension alone
(`.txt`, `.pdf`, `.json`, `.md`, `.docx`) and never inspects the bytes, so a
stream without a usable filename raises
{exc}`~pinecone.errors.exceptions.PineconeValueError` before anything is sent:

```python
import io

file = pc.assistants.upload_file(
    assistant_name="my-assistant",
    file_stream=io.BytesIO(pdf_bytes),
    file_name="data.pdf",
)
```

## Track long-running operations

File writes are asynchronous server-side. `upload_file` and `delete_file` poll for
you and only return once the work is done, so most callers never need this. But
`timeout=-1` makes them return as soon as the request is accepted — and these two
methods are how you follow what was started.

`describe_operation` reports one operation:

```python
operation = pc.assistants.describe_operation(
    assistant_name="my-assistant",
    operation_id="op-1234-abcd-5678",
)
print(operation.status)           # "Processing" | "Completed" | "Failed"
print(operation.percent_complete) # 0-100
print(operation.file_id)          # the file this operation is about
print(operation.error)            # the reason, when status is "Failed"
```

`list_operations` returns a lazy paginator over everything in flight and
everything that recently finished — both successes and failures are kept for 30
days:

```python
for op in pc.assistants.list_operations(assistant_name="my-assistant"):
    print(op.operation_id, op.operation_type, op.status)
```

Filter with `operation_type` (`"upload_file"`, `"upsert_file"`,
`"update_file_metadata"`, `"delete_file"`) and `status` (`"Processing"`,
`"Completed"`, `"Failed"` — case-sensitive). An unrecognized value raises
{exc}`~pinecone.errors.exceptions.PineconeValueError` listing the ones that work,
before anything is sent:

```python
stuck = pc.assistants.list_operations(
    assistant_name="my-assistant",
    operation_type="upload_file",
    status="Processing",
).to_list()
```

`limit` caps how many operations the paginator yields in total. For explicit
page-at-a-time control use `list_operations_page`, which takes `page_size` (1-100,
default 50) and returns a
{class}`~pinecone.models.assistant.list.ListOperationsResponse` whose `next` is
the token for the following page.

## Chat

Send a conversation and receive a response:

```python
response = pc.assistants.chat(
    assistant_name="my-assistant",
    messages=[{"role": "user", "content": "What is the main topic of the document?"}],
)
print(response.message.content)
```

## Streaming chat

Pass ``stream=True`` to receive tokens incrementally as text fragments.  Use
``stream.text()`` — the idiomatic text-only accessor — to iterate over plain
strings.  Iterating ``stream`` directly instead yields typed chunk objects
(``StreamMessageStart``, ``StreamContentChunk``, ``StreamCitationChunk``,
``StreamMessageEnd``), which is useful when you need full metadata but would
print their ``repr`` rather than the assistant's text.

```python
stream = pc.assistants.chat(
    assistant_name="my-assistant",
    messages=[{"role": "user", "content": "Summarize the document."}],
    stream=True,
)
for text in stream.text():
    print(text, end="", flush=True)
```

## Delete a file

Remove an uploaded file from an assistant:

```python
pc.assistants.delete_file(
    assistant_name="my-assistant",
    file_id="file-id-here",
)
```

Raises {exc}`~pinecone.errors.exceptions.NotFoundError` if the file does not exist.

## Delete an assistant

```python
pc.assistants.delete(name="my-assistant")
```

Raises {exc}`~pinecone.errors.exceptions.NotFoundError` if the assistant does not exist.

`delete` polls until the assistant is gone, indefinitely by default. A delete that
fails server-side is not retried, so if the assistant reports a terminal failure
status while being deleted, polling stops with
{exc}`~pinecone.errors.exceptions.PineconeError` instead of waiting forever. Pass
`timeout=-1` to return as soon as the request is accepted.
