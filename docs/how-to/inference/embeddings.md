# Generating embeddings

Pinecone hosts embedding models, so you can turn text into vectors without running
embedding infrastructure of your own. `pc.inference.embed` takes the text and hands the
vectors back to you — to upsert, to query with, or to keep somewhere else entirely.

If you would rather never handle a vector, that is an index with integrated inference,
where Pinecone embeds on both write and read. See
[Integrated records](../integrated-records.md).

## Basic usage

```python
from pinecone import Pinecone

pc = Pinecone(api_key="your-api-key")

embeddings = pc.inference.embed(
    model="multilingual-e5-large",
    inputs=["The quick brown fox", "A second piece of text"],
    parameters={"input_type": "passage"},
)

for embedding in embeddings:
    print(embedding.values[:3])
```

One embedding comes back per input, in input order. A bare string is wrapped for you, so
`inputs="one string"` still returns a one-item result rather than a lone embedding.

## `input_type`: the mismatch that costs you relevance

Many hosted models are asymmetric — they embed stored text and search queries
differently, and `input_type` is how you say which side you are on. Embed your corpus as
`"passage"` and your queries as `"query"`:

```python
passages = pc.inference.embed(
    model="multilingual-e5-large",
    inputs=["Vector databases index embeddings for similarity search."],
    parameters={"input_type": "passage"},
)

query = pc.inference.embed(
    model="multilingual-e5-large",
    inputs="How does similarity search work?",
    parameters={"input_type": "query"},
)
```

Get this backwards and nothing fails: both calls succeed, both return vectors of the
right width, and your search quality quietly degrades. Nothing in the SDK can catch it
for you, so decide per call site which side you are embedding.

## Discover what a model accepts

`parameters` is model-specific. Read the keys off the model rather than hardcoding a list
that will go stale:

```python
info = pc.inference.model.get("multilingual-e5-large")

for parameter in info.supported_parameters:
    print(parameter.parameter, parameter.allowed_values, parameter.default)
```

Each entry carries `allowed_values` when the set is fixed, `min`/`max` when the value is
numeric, and `default` for what the model uses when you omit the key. `required` tells
you whether it must be sent at all.

The same {class}`~pinecone.models.inference.models.ModelInfo` answers the other questions
you would otherwise guess at:

```python
print(info.type)                # 'embed' or 'rerank'
print(info.vector_type)         # 'dense' or 'sparse'
print(info.default_dimension)   # output width when no dimension is requested
print(info.max_batch_size)      # most inputs one call may carry
print(info.max_sequence_length) # longest input accepted
```

Those fields are `None` on a reranking model, so read `info.type` before relying on them.

## Response: EmbeddingsList

`embed` returns an {class}`~pinecone.models.inference.embed.EmbeddingsList`. Iterating it
yields the embeddings, and `len()` and integer indexing reach the same items:

- `.data` — the same list the iteration walks, one entry per input.
- `.vector_type` — `"dense"` or `"sparse"`, and so which fields the entries carry.
- `.model` — the model that served the request. Pinecone may substitute a different model
  from the one you asked for, so read it here rather than assuming it echoes your
  argument.
- `.usage.total_tokens` — tokens counted for the call.

## Dense and sparse embeddings

The two shapes carry the vector in different fields. A
{class}`~pinecone.models.inference.embed.DenseEmbedding` has `values`; a
{class}`~pinecone.models.inference.embed.SparseEmbedding` has `sparse_values` and
`sparse_indices` paired position by position, and no `values` at all. Branch on
`vector_type` when the model is not fixed in advance:

```python
embeddings = pc.inference.embed(
    model="multilingual-e5-large",
    inputs=["machine learning frameworks"],
    parameters={"input_type": "passage"},
)

first = embeddings.data[0]
if embeddings.vector_type == "sparse":
    print(first.sparse_indices, first.sparse_values)
else:
    print(first.values[:3])
```

Reading `.values` on a sparse embedding does not raise — it hands back a dict-view method
instead of a vector, and your code carries on with the wrong object. The `vector_type`
check is what protects you.

One `embed` call returns either all-dense or all-sparse results, never a mix. To build a
hybrid index, call `embed` once per model and pair the two result sets yourself.

## Using the EmbedModel enum

{class}`~pinecone.models.enums.EmbedModel` gives you tab-completion and typo safety in
place of a bare string:

```python
from pinecone import Pinecone, EmbedModel

pc = Pinecone(api_key="your-api-key")

embeddings = pc.inference.embed(
    model=EmbedModel.Multilingual_E5_Large,
    inputs=["search query"],
    parameters={"input_type": "query"},
)
print(embeddings.model)
```

A model that is not an enum member is still fine as a string — the enum lags new models
by a release, and `embed` accepts either.

## Batch size

Sending several inputs in one call amortizes the round trip. Each model caps how many
inputs it accepts per call, and the cap is reported as `max_batch_size` on the model info
— read it from [the model](#discover-what-a-model-accepts) rather than guessing, and keep
your chunk size at or below it. Exceeding it is a server-side rejection, not something
the SDK splits for you:

```python
texts = [f"document number {n}" for n in range(250)]

batch_size = 50
all_embeddings = []
for start in range(0, len(texts), batch_size):
    batch = pc.inference.embed(
        model="multilingual-e5-large",
        inputs=texts[start : start + batch_size],
        parameters={"input_type": "passage"},
    )
    all_embeddings.extend(batch.data)

print(len(all_embeddings))
```

Overlong individual inputs are a separate concern from batch size: the `truncate`
parameter decides whether the model trims them or rejects the request. Its accepted
values are in `supported_parameters`, above.

## Storing embeddings in an index

Read the values off each embedding and upsert them like any other vector:

```python
index = pc.index("product-search")

embeddings = pc.inference.embed(
    model="multilingual-e5-large",
    inputs=["The quick brown fox", "A second piece of text"],
    parameters={"input_type": "passage"},
)

index.upsert(
    vectors=[(f"doc-{i}", embedding.values) for i, embedding in enumerate(embeddings)]
)
```

The index's dense vector field must have been created at the model's output dimension —
`info.default_dimension` above — or the server rejects the upsert.

At query time, embed the query text with `input_type="query"` and pass the result to
{meth}`~pinecone.index.Index.query`. See
[Upserting and querying vectors](../vectors/upsert-and-query.md).

## List available models

```python
models = pc.inference.model.list(type="embed")
print(models.names())
```

`type` narrows the listing to `"embed"` or `"rerank"`; omit it for both. For embedding
models, `vector_type="dense"` or `"sparse"` narrows it further — pairing `vector_type`
with `type="rerank"` is rejected client-side rather than ignored.

```python
sparse_models = pc.inference.model.list(type="embed", vector_type="sparse")
print(sparse_models.names())
```
