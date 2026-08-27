# Reranking results

Reranking reorders a set of candidate documents by relevance to a query. Use it after
an initial retrieval step (vector search, keyword search, or a combined approach) to
surface the most relevant results at the top.

## Basic usage

```python
from pinecone import Pinecone

pc = Pinecone(api_key="your-api-key")

documents = [
    {"text": "Apple is a technology company."},
    {"text": "The apple is a popular fruit."},
    {"text": "Pinecone is a vector database."},
]

result = pc.inference.rerank(
    model="bge-reranker-v2-m3",
    query="Tell me about tech companies",
    documents=documents,
)

for doc in result.data:
    print(doc.index, doc.score, doc.document)
```

Pass plain strings and they are automatically wrapped as ``{"text": ...}``:

```python
result = pc.inference.rerank(
    model="bge-reranker-v2-m3",
    query="machine learning",
    documents=["Neural networks are a type of ML model.", "Python is a programming language."],
)
```

## Response: RerankResult

``rerank`` returns a {class}`~pinecone.models.inference.rerank.RerankResult` containing:

- ``.data``: list of {class}`~pinecone.models.inference.rerank.RankedDocument`, ordered
  by descending score.
- ``.model``: model name used.
- ``.usage.rerank_units``: rerank units consumed.

Each {class}`~pinecone.models.inference.rerank.RankedDocument` has:

- ``.index``: the original position of the document in the input list.
- ``.score``: relevance score (higher is more relevant).
- ``.document``: the original document dict (``None`` when ``return_documents=False``).

## top_n: return only the best results

Pass ``top_n`` to receive only the top N documents after reranking:

```python
result = pc.inference.rerank(
    model="bge-reranker-v2-m3",
    query="search query",
    documents=documents,
    top_n=3,
)
assert len(result.data) == 3
```

## rank_fields: choose which field to score on

By default the reranker scores the ``"text"`` field. Override with ``rank_fields``:

```python
result = pc.inference.rerank(
    model="bge-reranker-v2-m3",
    query="quarterly earnings",
    documents=[
        {"title": "Apple Q4 2024 results", "body": "Revenue grew 6% year over year."},
        {"title": "Banana prices rise", "body": "Fruit prices hit new highs."},
    ],
    rank_fields=["title", "body"],
)
```

## Using the RerankModel enum

Use the {class}`~pinecone.models.enums.RerankModel` enum for tab-completion and typo
safety:

```python
from pinecone import Pinecone, RerankModel

pc = Pinecone(api_key="your-api-key")

result = pc.inference.rerank(
    model=RerankModel.Bge_Reranker_V2_M3,
    query="machine learning",
    documents=documents,
)
```

## Reranking in a pipeline

A typical two-stage retrieval pipeline: fetch candidates from Pinecone, then rerank.

```python
from pinecone import Pinecone

pc = Pinecone(api_key="your-api-key")
index = pc.index("product-search")

# Stage 1: vector retrieval
query_embedding = pc.inference.embed(
    model="multilingual-e5-large",
    inputs=["best noise-cancelling headphones"],
    parameters={"input_type": "query"},
).data[0].values

matches = index.query(vector=query_embedding, top_k=20, include_metadata=True)
candidates = [
    {"text": (m.metadata or {}).get("description", ""), "id": m.id}
    for m in matches.matches
]

# Stage 2: rerank
result = pc.inference.rerank(
    model="bge-reranker-v2-m3",
    query="best noise-cancelling headphones",
    documents=candidates,
    rank_fields=["text"],
    top_n=5,
)

for doc in result.data:
    print(doc.score, doc.document)
```

For integrated indexes, pass ``rerank`` directly inside {meth}`~pinecone.Index.search`.
See {doc}`/how-to/integrated-records`.

## List available reranking models

```python
models = pc.inference.model.list(type="rerank")
print(models.names())
```
