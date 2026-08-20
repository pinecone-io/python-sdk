# 2026-07: assistant chat model names and finish reasons

The `2026-07` assistant data API changes which model names it documents and
which `finish_reason` values it can return. The SDK does not validate `model`
client-side — the backend is authoritative and gains models between SDK
releases — so these changes surface as backend rejections and as different
strings in `finish_reason`, not as SDK type errors.

## Model names: `claude-3-5-sonnet` and `claude-3-7-sonnet` are gone

`2026-07` documents these models on `chat()`:

| Endpoint | Documented models |
| --- | --- |
| `pc.assistants.chat(...)` | `gpt-4o`, `gpt-4.1`, `gpt-5`, `o4-mini`, `claude-sonnet-4-5`, `gemini-2.5-pro` |
| `pc.assistants.chat_completions(...)` | the same list minus `gpt-5` |

`claude-3-5-sonnet` and `claude-3-7-sonnet` are no longer in either list.
Migrate to `claude-sonnet-4-5`:

```python
response = pc.assistants.chat(
    assistant_name="my-assistant",
    messages=[{"content": "What is Pinecone?"}],
    model="claude-sonnet-4-5",  # was "claude-3-5-sonnet"
)
```

The two old names are still accepted as deprecated aliases and are silently
remapped to `claude-sonnet-4-5`, so existing code keeps working for now — but
the responses come from a different model than the name suggests, and the
aliases are not part of the documented surface. `gpt-4o` remains the SDK
default.

An unrecognised model name is rejected with a `400 INVALID_ARGUMENT` whose
message lists the values the backend accepts. That message reaches you
verbatim on the raised `ApiError`:

```python
try:
    pc.assistants.chat(assistant_name="my-assistant", messages=msgs, model="gpt-9")
except ApiError as exc:
    print(exc.message)
    # Invalid model `gpt-9`. Expected one of: gpt-4o, gpt-4.1, o4-mini, gpt-5,
    # claude-sonnet-4-5, gemini-2.5-pro.
```

## `finish_reason`: `function_call` became `tool_calls`

Wherever the API reports why generation stopped — `ChatResponse.finish_reason`,
`ChatCompletionChoice.finish_reason`, `ChatCompletionStreamChoice.finish_reason`,
and now `StreamMessageEnd.finish_reason` — the value `function_call` has been
replaced by `tool_calls`. The full set is `stop`, `length`, `content_filter`,
`tool_calls`.

The SDK types these as plain `str`, so nothing in the SDK breaks. Code that
matches on the string does:

```python
if response.finish_reason == "function_call":  # never true on 2026-07
    ...
if response.finish_reason == "tool_calls":  # replacement
    ...
```

## New fields

Additive, so no code breaks — `None` when the server does not report them:

| Model | New field |
| --- | --- |
| `ChatResponse` | `context_snippet_count`, `content_filter_results` |
| `StreamMessageStart` | `context_snippet_count` |
| `StreamMessageEnd` | `finish_reason` |
| `FileReference` (context snippets) | `type` |

`context_snippet_count` is how many retrieved snippets the model was given;
`0` means no relevant context was found, which arrives on `message_start`
before any content so a streaming caller can react early.
`content_filter_results` is left as a plain dict because its `results` shape is
defined by the provider named in its `spec` field. `FileReference.type` is the
kind of document a context snippet came from — `text`, `json`, `markdown`,
`pdf`, or `doc_x` — which the API always sends and the SDK previously dropped.
