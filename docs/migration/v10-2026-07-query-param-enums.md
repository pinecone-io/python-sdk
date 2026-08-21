# 2026-07: enum members reach the query string correctly

`inference.list_models()` accepts `vector_type` either as a plain string or as
a `VectorType` member. Until this release the member form put the **wrong value**
in the query string, and this release changes what goes on the wire. The
signatures did not change, so the fix is invisible in your code.

This is the query-string half of the story
[the model-enum guide](v10-2026-07-inference-model-enums.md) tells for request
bodies. The two had the same cause and needed different fixes, because bodies and
query strings are encoded by different machinery.

## What was sent before, and what is sent now

`VectorType` is a `(str, Enum)` mixin, and query values are encoded with
`str()`. For that kind of enum `str()` returns the member's *repr-style name*,
not its value:

```python
from pinecone import VectorType

str(VectorType.DENSE)   # 'VectorType.DENSE'
VectorType.DENSE.value  # 'dense'
```

So the request URL differed depending on which of the two accepted spellings you
used:

| `vector_type=` argument | query sent before | query sent now |
| --- | --- | --- |
| `"dense"` | `vector_type=dense` | `vector_type=dense` |
| `VectorType.DENSE` | `vector_type=VectorType.DENSE` | `vector_type=dense` |
| `VectorType.SPARSE` | `vector_type=VectorType.SPARSE` | `vector_type=sparse` |

The change applies to `list_models()` and to the `pc.inference.model.list()`
facade that delegates to it, on both the sync and the async client.

## Was my code affected?

If every `list_models()` call passes `vector_type=` as a **string literal**,
nothing changes for you.

If any call passes a **member**, that call was failing before and works now. It
failed loudly rather than silently: the server does not recognise the mangled
name, so it rejected the request rather than answering it with an unfiltered or
wrongly filtered list. The error was an `ApiError` carrying the server's own
wording, which is the string to search your logs for:

```text
Invalid vector_type, expected one of [dense, sparse]
```

Passing `.value` was the workaround while the bug was open, and it is still
correct — `.value` is exactly what the SDK now extracts for you:

```python
pc.inference.list_models(type="embed", vector_type=VectorType.DENSE.value)
```

Passing `.value` and passing the member now produce identical requests, so there
is nothing to clean up if you adopted it. New code can drop the `.value`.

## The fix is at the encoder, not at this one parameter

The resolution happens where query parameters are built for every request the
client makes, so no query parameter on any surface can carry a mangled member —
including parameters added after this release. `vector_type` was the only one
reachable with an enum member when the bug was found.

## The SDK does not repair a mangled name

Passing the literal string `"VectorType.DENSE"` is not the same as passing the
member. It is rejected before a request is made, because it is not one of the
values the parameter accepts:

```python
pc.inference.list_models(vector_type="VectorType.DENSE")
# PineconeValueError: vector_type must be one of 'dense', 'sparse', got 'VectorType.DENSE'
```
