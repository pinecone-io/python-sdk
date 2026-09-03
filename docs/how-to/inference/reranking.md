# Reranking results

Reranking reorders a set of candidate documents by relevance to a query. It runs a
second, slower model over candidates something else already found — vector search,
keyword search, or another system entirely — and usually buys precision at the top of the
list at the cost of latency.

## Basic usage

```python
from pinecone import Pinecone

pc = Pinecone(api_key="your-api-key")

documents = [
    {"text": "Apple is a technology company."},
    {"text": "The apple is a popular fruit."},
    {"text": "Acme Inc. revolutionized enterprise software."},
]

result = pc.inference.rerank(
    model="bge-reranker-v2-m3",
    query="Tell me about tech companies",
    documents=documents,
)

for ranked in result.data:
    print(ranked.index, ranked.score, ranked.document)
```

Pass plain strings and each is wrapped as `{"text": ...}`, which is what the default
`rank_fields` scores on:

```python
result = pc.inference.rerank(
    model="bge-reranker-v2-m3",
    query="machine learning",
    documents=["Neural networks are a type of ML model.", "Python is a programming language."],
)
print(result.data[0].index)
```

## `index` maps a result back onto your list

`result.data` is ordered by descending score, **not** by the order you passed the
documents in. Each {class}`~pinecone.models.inference.rerank.RankedDocument` carries the
position it held in your `documents` argument as `.index`, and that is the whole point of
the API: it is how you find the record, the ID, or the row that the score belongs to.

In the first example above, the third document is the most relevant, so the first result
has `.index == 2` — not `0`:

```python
top = result.data[0]
print(top.index)      # position in the documents you sent
print(top.score)      # relevance, higher is more relevant
print(top.document)   # the document as sent, or None with return_documents=False
```

Reading `result.data` positionally and ignoring `.index` silently pairs your scores with
the wrong documents.

The usual way to avoid the bookkeeping is to keep your own identifier on each document.
Every key that is not named in `rank_fields` rides along untouched and comes back in
`.document`:

```python
result = pc.inference.rerank(
    model="bge-reranker-v2-m3",
    query="Tell me about tech companies",
    documents=[
        {"id": "doc-1", "summary": "Apple is a fruit."},
        {"id": "doc-2", "summary": "Acme Inc. revolutionized tech."},
    ],
    rank_fields=["summary"],
)
print(result.data[0].document["id"])
```

## Response: RerankResult

`rerank` returns a {class}`~pinecone.models.inference.rerank.RerankResult`:

- `.data` — the {class}`~pinecone.models.inference.rerank.RankedDocument` list, ordered
  by descending `score`.
- `.model` — the model that served the request. Pinecone may substitute a different model
  from the one you asked for, so read it here rather than assuming it echoes your
  argument.
- `.usage.rerank_units` — rerank units counted for the call.

## `top_n`: return only the best results

By default every document you send comes back with a score. Pass `top_n` to keep only the
best `n`:

```python
result = pc.inference.rerank(
    model="bge-reranker-v2-m3",
    query="search query",
    documents=documents,
    top_n=2,
)
print(len(result.data))
```

`top_n` below `1` is rejected client-side, before any request is made.

## `rank_fields`: choose which field to score on

The reranker scores the `"text"` field unless you say otherwise. Name your own fields
with `rank_fields`:

```python
result = pc.inference.rerank(
    model="bge-reranker-v2-m3",
    query="quarterly earnings",
    documents=[
        {"title": "Acme Q4 results", "body": "Revenue grew year over year."},
        {"title": "Banana prices rise", "body": "Fruit prices hit new highs."},
    ],
    rank_fields=["title", "body"],
)
print(result.data[0].document)
```

Not every model accepts more than one rank field. `supported_parameters` on the model
info says what a given model will take — see
[Discover what a model accepts](embeddings.md#discover-what-a-model-accepts).

## Skip the round trip on the documents

Set `return_documents=False` when you already hold the documents in memory and want only
the ordering. `.document` is then `None` and you rely on `.index`:

```python
result = pc.inference.rerank(
    model="bge-reranker-v2-m3",
    query="search query",
    documents=documents,
    return_documents=False,
)
print(result.data[0].index, result.data[0].document)
```

## Using the RerankModel enum

{class}`~pinecone.models.enums.RerankModel` gives you tab-completion and typo safety in
place of a bare string:

```python
from pinecone import Pinecone, RerankModel

pc = Pinecone(api_key="your-api-key")

result = pc.inference.rerank(
    model=RerankModel.Bge_Reranker_V2_M3,
    query="machine learning",
    documents=["Neural networks are a type of ML model."],
)
print(result.model)
```

A model that is not an enum member is still fine as a string. A name the API does not
serve raises {exc}`~pinecone.errors.exceptions.NotFoundError`, and a model your project
is not authorized to use — including a deprecated one — raises
{exc}`~pinecone.errors.exceptions.ForbiddenError`. Check the name before suspecting the
request body.

## Reranking in a pipeline

Two-stage retrieval: fetch a wide candidate set from Pinecone, then rerank it down.

```python
from pinecone import Pinecone

pc = Pinecone(api_key="your-api-key")
index = pc.index("product-search")

# Stage 1: vector retrieval. Embed the query as a query, not a passage.
query_vector = pc.inference.embed(
    model="multilingual-e5-large",
    inputs="best noise-cancelling headphones",
    parameters={"input_type": "query"},
).data[0].values

response = index.query(vector=query_vector, top_k=20, include_metadata=True)

# Stage 2: rerank. Carry the vector ID through so the scores stay attributable.
candidates = [
    {"id": match.id, "text": (match.metadata or {}).get("description", "")}
    for match in response.matches
]

reranked = pc.inference.rerank(
    model="bge-reranker-v2-m3",
    query="best noise-cancelling headphones",
    documents=candidates,
    rank_fields=["text"],
    top_n=5,
)

for ranked in reranked.data:
    print(ranked.document["id"], ranked.score)
```

`top_k` decides how many candidates stage one retrieves; `top_n` decides how many survive
stage two. Retrieving more than you intend to keep is the point — reranking can only
reorder what stage one found.

An empty `documents` list is rejected client-side, so guard the second stage when stage
one can return nothing.

For an index with integrated inference, `rerank` is an argument to
{meth}`~pinecone.index.Index.search` and both stages happen in one request. See
[Integrated records](../integrated-records.md#rerank-in-a-single-search-call). Reach for
`pc.inference.rerank` when the candidates came from somewhere other than that index.

## List available reranking models

```python
models = pc.inference.model.list(type="rerank")
print(models.names())
```
