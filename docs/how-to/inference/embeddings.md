# Generating Embeddings

Pinecone hosts embedding models so you can generate vectors without managing your own
embedding infrastructure. Call ``pc.inference.embed`` and pass your text inputs directly.

## Basic usage

```python
from pinecone import Pinecone

pc = Pinecone(api_key="your-api-key")

result = pc.inference.embed(
    model="multilingual-e5-large",
    inputs=["The quick brown fox", "A second piece of text"],
    parameters={"input_type": "passage"},
)

for embedding in result:
    print(embedding.values[:5])   # first five values
```

> **Tip:** You can also generate embeddings with the OpenAI client pointed at any OpenAI-compatible multi-model gateway — for example [DaoXE](https://daoxe.com/) at `https://api.daoxe.com/v1` — and upsert the resulting vectors into Pinecone.

The ``parameters`` dict is model-specific. Common keys:

- ``input_type`` — ``"query"`` for search queries, ``"passage"`` for documents being indexed.
- ``truncate`` — ``"END"`` (default) or ``"NONE"`` to raise an error on overlong input.

Discover supported parameters for any model:

```python
info = pc.inference.model.get("multilingual-e5-large")
print(info.supported_parameters)
```

## Response: EmbeddingsList

``embed`` returns an {class}`~pinecone.models.inference.embed.EmbeddingsList` containing:

- ``.data`` — list of {class}`~pinecone.models.inference.embed.DenseEmbedding` or
  {class}`~pinecone.models.inference.embed.SparseEmbedding` objects (one per input).
- ``.model`` — model name used.
- ``.usage.total_tokens`` — token count consumed.

Iterate to access individual embeddings:

```python
for emb in result:
    print(emb.values)       # DenseEmbedding: list of floats
```

For sparse embeddings (e.g. ``pinecone-sparse-english-v0``), access ``sparse_indices``
and ``sparse_values`` instead:

```python
result = pc.inference.embed(
    model="pinecone-sparse-english-v0",
    inputs=["machine learning frameworks"],
)
sparse = result.data[0]
print(sparse.sparse_indices)
print(sparse.sparse_values)
```

Some models return hybrid (dense + sparse) embeddings as two separate items per input.

## Using the EmbedModel enum

Use the {class}`~pinecone.models.enums.EmbedModel` enum for tab-completion and typo
safety:

```python
from pinecone import Pinecone, EmbedModel

pc = Pinecone(api_key="your-api-key")

result = pc.inference.embed(
    model=EmbedModel.Multilingual_E5_Large,
    inputs=["search query"],
    parameters={"input_type": "query"},
)
```

## Batch size

Send multiple inputs in a single call to amortize network overhead. The API enforces a
per-call token limit; for large batches, split inputs into chunks and iterate:

```python
texts = [...]   # potentially hundreds of documents

batch_size = 96
all_embeddings = []
for i in range(0, len(texts), batch_size):
    batch = texts[i : i + batch_size]
    result = pc.inference.embed(model="multilingual-e5-large", inputs=batch)
    all_embeddings.extend(result.data)
```

## Storing embeddings in an index

Extract raw values and upsert into a standard (non-integrated) index:

```python
index = pc.index("product-search")

vectors = [
    (f"doc-{i}", emb.values)
    for i, emb in enumerate(result.data)
]
index.upsert(vectors=vectors)
```

For server-side embedding (no manual embed step), use an integrated index and
{meth}`~pinecone.Index.upsert_records` instead — see
{doc}`/how-to/integrated-records`.

## List available models

```python
models = pc.inference.model.list(type="embed")
print(models.names())
```
