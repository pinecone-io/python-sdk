# 2026-07: `EmbedModel` and `RerankModel` members reach the wire correctly

`inference.embed()` and `inference.rerank()` accept the `model` argument either
as a plain string or as an `EmbedModel` / `RerankModel` member. Until this
release the member form sent the **wrong value**, and this release changes what
goes on the wire. Nothing about the signatures changed, so the fix is invisible
in your code — which is why it is worth reading.

## What was sent before, and what is sent now

Both enums are `(str, Enum)` mixins, and the serialization went through
`str(model)`. For that kind of enum `str()` returns the member's *repr-style
name*, not its value:

```python
from pinecone import EmbedModel

str(EmbedModel.Multilingual_E5_Large)   # 'EmbedModel.Multilingual_E5_Large'
EmbedModel.Multilingual_E5_Large.value  # 'multilingual-e5-large'
```

So the request body differed depending on which of the two accepted spellings
you used:

| `model=` argument | `"model"` sent before | `"model"` sent now |
| --- | --- | --- |
| `"multilingual-e5-large"` | `multilingual-e5-large` | `multilingual-e5-large` |
| `EmbedModel.Multilingual_E5_Large` | `EmbedModel.Multilingual_E5_Large` | `multilingual-e5-large` |
| `RerankModel.Bge_Reranker_V2_M3` | `RerankModel.Bge_Reranker_V2_M3` | `bge-reranker-v2-m3` |

The change applies to `embed()` and `rerank()` on both the sync and the async
client.

## Was my code affected?

If every `embed()` and `rerank()` call passes `model=` as a **string literal**,
nothing changes for you.

If any call passes an **enum member**, that call was failing before and works
now. It failed loudly rather than silently: no such model exists, so the server
rejected the request rather than serving it with the wrong model. Both `embed()`
and `rerank()` raised `NotFoundError`, with a message quoting the mangled name —
which is the string to search your logs for:

```text
Model 'EmbedModel.Multilingual_E5_Large' not found
```

A third spelling was the documented workaround while the bug was open, and it
is still correct — `.value` is exactly what the SDK now extracts for you:

```python
pc.inference.embed(model=EmbedModel.Multilingual_E5_Large.value, inputs=["hello"])
```

Passing `.value` and passing the member now produce byte-identical requests, so
there is nothing to clean up if you adopted it. New code can drop the `.value`.

## Plain strings still work

The enums are a convenience, not an allowlist. A model id the installed SDK has
no member for — one released after it — is still accepted as a string and sent
through unchanged:

```python
pc.inference.embed(model="some-newer-embedding-model", inputs=["hello"])
```

The SDK does not repair a mangled name either. Passing the literal string
`"EmbedModel.Multilingual_E5_Large"` sends it verbatim and still raises
`NotFoundError`, because that is a model id the server does not have.

```python
pc.inference.embed(model="EmbedModel.Multilingual_E5_Large", inputs=["hello"])
# NotFoundError: Model 'EmbedModel.Multilingual_E5_Large' not found
```
