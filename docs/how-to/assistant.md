# Working with the assistant

A Pinecone assistant is a managed question-answering service grounded in documents you
upload to it. Create the assistant, upload files, then ask questions and get answers with
citations back to the files that supported each claim. Pinecone does both the retrieval
and the generation, which is what separates an assistant from an index — an index hands
you records to feed to a model of your own.

Everything is reached through `pc.assistants`. Methods that talk to one assistant's data
plane — uploading, chatting, retrieving context, listing files and operations — need that
assistant's host, so the first such call for a given name does a `describe` behind the
scenes and caches the host for the life of the client. So the first data-plane call on an
assistant costs one extra round trip, and a name that does not exist surfaces as
{exc}`~pinecone.errors.exceptions.NotFoundError` from that lookup rather than from the
method you actually called.

`AsyncPinecone` exposes the same names with `await`; see
[Sync vs async](../guides/sync-vs-async.md). For which exceptions these methods raise and
which are worth retrying, see [Error handling](../guides/error-handling.md).

## Create an assistant

```python
from pinecone import Pinecone

pc = Pinecone(api_key="your-api-key")

assistant = pc.assistants.create(
    name="research-assistant",
    instructions="Answer questions based on the uploaded documents.",
)
print(assistant.name)    # "research-assistant"
print(assistant.status)  # "Ready"
```

`create` polls until the assistant reaches `"Ready"` before returning, so the assistant
is usable the moment the call comes back. `instructions` and `metadata` are optional, and
`region` is `"us"` or `"eu"` — anything else raises
{exc}`~pinecone.errors.exceptions.PineconeValueError` before a request is sent. Region
cannot be changed afterwards; an assistant in the wrong one has to be recreated.

Pass `timeout=-1` to skip the polling and return as soon as the create request is
accepted. The assistant is not ready yet at that point, so check it later with
`describe`:

```python
assistant = pc.assistants.create(name="research-assistant", timeout=-1)
print(assistant.status)  # not yet "Ready"

assistant = pc.assistants.describe(name="research-assistant")
```

A positive `timeout` polls with a deadline and raises
{exc}`~pinecone.errors.exceptions.PineconeTimeoutError` if the assistant is still not
ready. If it lands in a terminal state instead — `"Failed"`, `"InitializationFailed"`,
`"Terminating"` or `"Terminated"` — polling stops with
{exc}`~pinecone.errors.exceptions.PineconeError` rather than waiting for a status that
will never arrive.

## List and describe assistants

`list` returns a paginator over every assistant in the project:

```python
for asst in pc.assistants.list():
    print(asst.name, asst.status)
```

`describe` returns one assistant, including the `host` its data-plane calls go to:

```python
asst = pc.assistants.describe(name="research-assistant")
print(asst.name, asst.status, asst.region)
print(asst.instructions)
print(asst.host)
```

## Update an assistant

`update` patches `instructions`, `metadata`, or both. At least one is required —
with neither, the patch body would be empty, so the SDK raises
{exc}`~pinecone.errors.exceptions.PineconeValueError` instead of sending it.

```python
assistant = pc.assistants.update(
    name="research-assistant",
    instructions="Always cite the source document.",
)
```

`None` means "leave this field alone", so the call above does not touch metadata. A
`metadata` you do pass replaces the whole dictionary rather than merging into it. To clear
either field, send its empty value — `instructions=""` or `metadata={}`.

## Upload a file

Pass a local file path to upload a document for the assistant to read:

```python
file = pc.assistants.upload_file(
    assistant_name="research-assistant",
    file_path="/data/q3-revenue-review.pdf",
    metadata={"department": "finance", "quarter": "2024-Q3"},
)
print(file.id)      # "file-abc123", the id later calls take
print(file.name)    # "q3-revenue-review.pdf"
print(file.status)  # "Available"
```

`upload_file` waits for server-side processing to finish, so the file it returns is
already `"Available"` — it is queryable as soon as the call returns. Pass `timeout=-1` to
return right after the upload is accepted instead, in which case the status is whatever
processing has reached so far and you follow the rest through the operations API below.

To upload bytes you already hold, pass `file_stream` with a `file_name` that carries the
extension. The server types an uploaded file by its extension alone (`.txt`, `.pdf`,
`.json`, `.md`, `.docx`) and never inspects the bytes, so a stream without a usable
filename raises {exc}`~pinecone.errors.exceptions.PineconeValueError` before anything is
sent:

```python
import io

pdf_bytes = b"%PDF-1.4 ..."

file = pc.assistants.upload_file(
    assistant_name="research-assistant",
    file_stream=io.BytesIO(pdf_bytes),
    file_name="q3-revenue-review.pdf",
)
```

`file_path` and `file_stream` are alternatives; passing both, or neither, raises
{exc}`~pinecone.errors.exceptions.PineconeValueError`. Passing `file_id` replaces the file
already stored under that id rather than adding a second one.

To see what an assistant currently holds, `list_files` paginates over its files and
`describe_file` fetches one by id:

```python
for file in pc.assistants.list_files(assistant_name="research-assistant"):
    print(file.id, file.name, file.status)

file = pc.assistants.describe_file(
    assistant_name="research-assistant",
    file_id="file-abc123",
    include_url=True,
)
print(file.signed_url)  # temporary download URL, only when include_url=True
```

The listing drops a `"ProcessingFailed"` file once it is old enough; `describe_file`
still returns it by id.

## Track long-running operations

File writes are asynchronous server-side. `upload_file` and `delete_file` poll for you and
return only once the work is done, so most callers never need this section. But
`timeout=-1` makes them return as soon as the request is accepted, and these are the
methods that let you follow what was started.

`describe_operation` reports one operation:

```python
operation = pc.assistants.describe_operation(
    assistant_name="research-assistant",
    operation_id="op-1234-abcd-5678",
)
print(operation.status)           # "Processing" | "Completed" | "Failed"
print(operation.percent_complete) # 0-100, or None if not reported
print(operation.file_id)          # the file this operation is about
print(operation.created_at, operation.completed_on)
```

`completed_on` is `None` while `status` is still `"Processing"`, so the pair brackets how
long the work took once it is not.

Read `operation.error` only when `status` is `"Failed"`. The server never clears that
field once it is set, so a retried operation that is back to `"Processing"` — or that
eventually succeeded — still carries the earlier attempt's message.

`list_operations` returns a lazy paginator over everything in flight and everything that
finished recently enough to still be in the API's retention window:

```python
for op in pc.assistants.list_operations(assistant_name="research-assistant"):
    print(op.operation_id, op.operation_type, op.status)
```

Filter with `operation_type` (`"upload_file"`, `"upsert_file"`,
`"update_file_metadata"`, `"delete_file"`) and `status` (`"Processing"`, `"Completed"`,
`"Failed"` — case-sensitive). An unrecognized value raises
{exc}`~pinecone.errors.exceptions.PineconeValueError` listing the ones that work, before
anything is sent:

```python
stuck = pc.assistants.list_operations(
    assistant_name="research-assistant",
    operation_type="upload_file",
    status="Processing",
).to_list()
```

`limit` caps how many operations the paginator yields in total. For page-at-a-time
control use `list_operations_page`, which takes a `page_size` and returns a
{class}`~pinecone.models.assistant.list.ListOperationsResponse` whose `next` is the token
for the following page. See [Pagination](../guides/pagination.md).

## Ask a question

`chat` is the method to reach for. It returns the answer at `response.message.content`
and the sources as structured objects, which is what you need to render source links:

```python
response = pc.assistants.chat(
    assistant_name="research-assistant",
    messages=[{"role": "user", "content": "How did Q3 revenue change?"}],
)
print(response.message.content)
print(response.model)           # the model that answered
print(response.finish_reason)   # "stop" when the model finished on its own
```

Messages can be dicts or {class}`~pinecone.models.assistant.message.Message` objects. In
a dict, `role` defaults to `"user"` when absent. The backend accepts only the exact
strings `"user"` and `"assistant"`, compared case-sensitively, and rejects blank content
— neither is checked client-side, so a bad role comes back as an API error.

The assistant answers only from files you uploaded to it. An assistant with nothing in
`"Available"` status yet raises {exc}`~pinecone.errors.exceptions.ApiError` rather than
replying from general knowledge, so check `list_files` before reading that as a transport
failure.

### Render the citations

Citations are keyed to character positions in the answer, so you can insert footnote
markers into `response.message.content` as you render it. Each citation carries the
references behind it, and each reference names the source file:

```python
for citation in response.citations:
    for reference in citation.references:
        print(citation.position, reference.file.name, reference.pages)
```

`reference.file` is an {class}`~pinecone.models.assistant.file_model.AssistantFileModel`,
so `file.name` gives you a label, `file.id` refetches the document, and `file.metadata`
returns whatever you attached at upload. `reference.pages` is the page numbers for
paginated sources such as PDFs, and `None` for sources that have no pages.

`response.citations` is empty when the answer drew on no file. When that surprises you,
read `response.context_snippet_count`: `0` means retrieval found nothing relevant for the
query, which explains the missing citations.

Pass `include_highlights=True` to also get `reference.highlight.content` — the passage of
the source document the citation drew on, so you can show it without fetching the file.

### `chat`, `chat_completions`, or `context`?

Three methods take the same conversation and differ in what they give back. Pick by what
you are rendering.

{meth}`~pinecone.client.assistants.Assistants.chat`
: The current, fullest surface, and the default choice. Structured `citations` keyed to
  positions in the answer, plus `include_highlights`, `context_options` and
  `json_response`.

{meth}`~pinecone.client.assistants.Assistants.chat_completions`
: The same conversation in OpenAI's response shape — read
  `response.choices[0].message.content`. Use it to drop an assistant into code already
  written against that shape. There is no separate `citations` list: sources arrive woven
  into the message text, and `include_highlights`, `context_options` and `json_response`
  are not accepted.

{meth}`~pinecone.client.assistants.Assistants.context`
: Retrieval with no generation — the snippets the assistant would have been given, for a
  prompt you assemble yourself. Takes exactly one of `query` or `messages`.

```python
response = pc.assistants.chat_completions(
    assistant_name="research-assistant",
    messages=[{"content": "How did Q3 revenue change?"}],
)
print(response.choices[0].message.content)

context = pc.assistants.context(
    assistant_name="research-assistant",
    query="Q3 revenue",
    top_k=5,
)
for snippet in context.snippets:
    print(snippet.score, snippet.content, snippet.reference.file.name)
```

`snippet.content` is a string on the ordinary text snippet. A request with
`multimodal=True` can instead return a `MultimodalSnippet`, whose `content` is a list of
blocks rather than a string — branch with `isinstance` before reading it if you enable
that.

There is also a legacy set of methods on the assistant object itself —
`assistant.chat(...)`, `assistant.upload_file(...)`, and so on, on a model returned by
`create`, `describe` or `list`. They have been deprecated since 9.0.0, are sync-only
(a model from `AsyncPinecone` raises {exc}`TypeError` if you call them), and each just
forwards to the `pc.assistants` method with `assistant_name=self.name`. Call the
namespace methods instead. `pc.assistant` is likewise an alias of `pc.assistants`; the
plural is canonical.

### Choosing a model

`chat` and `chat_completions` take a `model` name and default to `"gpt-4o"`. The name is
not validated client-side. An unrecognized one comes back as an API error enumerating
what the endpoint accepts — but a name the endpoint no longer lists may instead be served
by a successor model rather than rejected. So the response's `model` field, not the
argument you passed, tells you which model answered.

## Stream a chat response

Pass `stream=True` for a {class}`~pinecone.models.assistant.streaming.ChatStream`. When
you only want the text, `text()` yields the fragments as they arrive:

```python
stream = pc.assistants.chat(
    assistant_name="research-assistant",
    messages=[{"content": "Summarize the revenue review."}],
    stream=True,
)
for fragment in stream.text():
    print(fragment, end="", flush=True)
```

`collect()` is the same fragments already joined into one string, for when you do not need
to render as they arrive. A stream is single-pass: iterating it, `text()` and `collect()`
all drain the same underlying iterator, so use exactly one.

Citations and token usage are not reachable through `text()` — it discards every chunk
that is not response text. To get them, iterate the stream itself and branch on the four
chunk types:

```python
from pinecone import (
    StreamCitationChunk,
    StreamContentChunk,
    StreamMessageEnd,
    StreamMessageStart,
)

stream = pc.assistants.chat(
    assistant_name="research-assistant",
    messages=[{"content": "Summarize the revenue review."}],
    stream=True,
)

answer: list[str] = []
sources: list[tuple[int, str]] = []
for chunk in stream:
    if isinstance(chunk, StreamMessageStart):
        if chunk.context_snippet_count == 0:
            print("nothing relevant was retrieved")
    elif isinstance(chunk, StreamContentChunk):
        answer.append(chunk.delta.content)
        print(chunk.delta.content, end="", flush=True)
    elif isinstance(chunk, StreamCitationChunk):
        for reference in chunk.citation.references:
            sources.append((chunk.citation.position, reference.file.name))
    elif isinstance(chunk, StreamMessageEnd):
        print(f"\nfinish_reason={chunk.finish_reason}")
```

What each chunk is for:

{class}`~pinecone.models.assistant.streaming.StreamMessageStart`
: Arrives once, first, with no response text. `context_snippet_count` of `0` tells you
  retrieval found nothing before the answer starts arriving.

{class}`~pinecone.models.assistant.streaming.StreamContentChunk`
: Arrives many times, and is the **only** chunk carrying response text, at
  `chunk.delta.content`. Concatenate the fragments in arrival order. A loop that forgets
  this type renders nothing.

{class}`~pinecone.models.assistant.streaming.StreamCitationChunk`
: Arrives zero or more times, interleaved with the content chunks, and holds the same
  `position` and `references` as a non-streaming citation.

{class}`~pinecone.models.assistant.streaming.StreamMessageEnd`
: Arrives once, last, with no response text. Read `finish_reason` here to tell a complete
  answer from one the model cut short, and `usage` for token counts.

Each chunk also exposes its wire tag as `chunk.type` (`"message_start"`,
`"content_chunk"`, `"citation"`, `"message_end"`), so you can dispatch on the string
instead of on `isinstance`.

`stream=True` cannot be combined with `json_response=True`; that raises
{exc}`~pinecone.errors.exceptions.PineconeValueError`. On a streaming request the
`timeout` argument bounds the gap between chunks rather than the whole response, and a
client-level timeout is raised to a floor for streaming — raised only, never lowered — so
that a model thinking for a while is not mistaken for a dead connection. Pass `timeout`
explicitly to override that, including to a shorter value. A stream that exceeds its
timeout raises {exc}`~pinecone.errors.exceptions.PineconeTimeoutError` partway through
iteration, after earlier chunks have already been yielded.

`chat_completions(..., stream=True)` returns a
{class}`~pinecone.models.assistant.streaming.ChatCompletionStream` instead, whose text
arrives at `chunk.choices[0].delta.content` — `None` on the chunks that carry no text,
such as the opening role chunk and the final one.

## Evaluate an answer against a ground truth

`evaluate_alignment` scores a generated answer against an answer you know to be correct.
It does not involve an assistant, so it takes no `assistant_name` — you can score output
from anywhere.

```python
result = pc.assistants.evaluate_alignment(
    question="What is the capital of Spain?",
    answer="Barcelona.",
    ground_truth_answer="Madrid.",
)
print(result.scores.correctness)   # precision: how much of the answer holds up
print(result.scores.completeness)  # recall: how much of the ground truth it covered
print(result.scores.alignment)     # harmonic mean of the two
```

Because `alignment` is a harmonic mean, a low score on either input drags it down, so
read all three rather than tracking `alignment` alone. The aggregate scores tell you an
answer is wrong; `result.facts` tells you where. Each entry is one fact with a judgment
of `"entailed"`, `"contradicted"` or `"neutral"`, and the reasoning behind it:

```python
for fact in result.facts:
    print(fact.entailment, fact.fact)
    print(fact.reasoning)

contradictions = [f.fact for f in result.facts if f.entailment == "contradicted"]
```

`fact.reasoning` is `""` rather than `None` when the API returned none, so test it for
truthiness. `result.usage` counts the tokens the evaluation itself spent, not the tokens
of the answer being evaluated.

## Delete a file

```python
pc.assistants.delete_file(
    assistant_name="research-assistant",
    file_id="file-abc123",
)
```

Deletion sometimes completes immediately and sometimes runs as a pending operation; when
it is pending, `delete_file` polls until it finishes. Deleting an id that is already gone
raises {exc}`~pinecone.errors.exceptions.NotFoundError` rather than returning silently.
Pass `timeout=-1` to return as soon as the request is accepted — the file may still exist
when that returns.

## Delete an assistant

```python
pc.assistants.delete(name="research-assistant")
```

Raises {exc}`~pinecone.errors.exceptions.NotFoundError` if the assistant does not exist.

`delete` polls until the assistant is confirmed gone, indefinitely by default. A delete
that fails server-side is not retried, so if the assistant reports a terminal failure
status while being deleted, polling stops with
{exc}`~pinecone.errors.exceptions.PineconeError` instead of waiting forever. Pass
`timeout=-1` to return as soon as the request is accepted.
