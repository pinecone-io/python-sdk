# Integrated records (server-side embedding)

Integrated indexes store text records and embed them server-side using a hosted model.
You write text; Pinecone handles the embedding. No separate ``embed`` step required.

## Create an integrated index

Call ``pc.indexes.create_for_model()`` with an {class}`~pinecone.EmbedConfig` that maps
a document field to the embedding input:

```python
from pinecone import Pinecone
from pinecone.models.indexes.specs import EmbedConfig

pc = Pinecone(api_key="your-api-key")

pc.indexes.create_for_model(
    name="articles",
    cloud="aws",
    region="us-east-1",
    embed=EmbedConfig(
        model="multilingual-e5-large",
        field_map={"text": "body"},   # embed the "body" field of each record
    ),
)
```

``field_map`` maps the model's input field name (``"text"`` for most models) to the
field in your records that holds the content to embed.

Get a handle to the index:

```python
index = pc.index("articles")
```

## Upsert records

Call {meth}`~pinecone.Index.upsert_records` with a list of record dicts. Each record
must have an ``_id`` (or ``id``) field. Include any fields you configured in
``field_map``:

```python
response = index.upsert_records(
    namespace="en",
    records=[
        {"_id": "article-1", "body": "Vector databases accelerate AI search."},
        {"_id": "article-2", "body": "RAG pipelines combine retrieval with generation."},
        {"_id": "article-3", "body": "Pinecone scales to billions of vectors."},
    ],
)
print(response.record_count)   # number of records submitted
```

Records are sent as newline-delimited JSON (NDJSON). Embeddings are generated
asynchronously by Pinecone; allow a moment before searching.

## Search records

Call {meth}`~pinecone.Index.search` with ``inputs`` containing the query text. Pinecone
embeds the query server-side and returns the nearest records:

```python
results = index.search(
    namespace="en",
    top_k=5,
    inputs={"text": "AI and machine learning"},
)

for hit in results.result.hits:
    print(hit.id, hit.score, hit.fields)
```

### Response: SearchRecordsResponse

``search`` returns a {class}`~pinecone.models.vectors.search.SearchRecordsResponse`:

- ``.result.hits``: list of {class}`~pinecone.models.vectors.search.Hit` objects, ordered
  by descending score.
- ``.usage.read_units``: read units consumed.
- ``.usage.embed_total_tokens``: tokens used for embedding the query.

Each {class}`~pinecone.models.vectors.search.Hit` exposes:

- ``.id``: the record identifier.
- ``.score``: similarity score, higher is more relevant.
- ``.fields``: dict of record fields returned in the result.

### Use SearchInputs for IDE support

```python
from pinecone.models.vectors.search import SearchInputs

results = index.search(
    namespace="en",
    top_k=5,
    inputs=SearchInputs(text="AI and machine learning"),
)
```

## Rerank in a single search call

Pass a ``rerank`` config to retrieve and rerank in one request:

```python
from pinecone.models.vectors.search import RerankConfig

results = index.search(
    namespace="en",
    top_k=10,
    inputs={"text": "best practices for vector search"},
    rerank=RerankConfig(
        model="bge-reranker-v2-m3",
        rank_fields=["body"],
        top_n=3,
    ),
)

for hit in results.result.hits:
    print(hit.id, hit.score)
```

``top_k`` controls how many candidates are retrieved; ``top_n`` controls how many
survive reranking. The response contains at most ``top_n`` hits.

You can also pass a plain dict for ``rerank``:

```python
results = index.search(
    namespace="en",
    top_k=10,
    inputs={"text": "best practices"},
    rerank={"model": "bge-reranker-v2-m3", "rank_fields": ["body"], "top_n": 3},
)
```

## Filter by metadata

Pass a ``filter`` dict to restrict results to records matching metadata conditions:

```python
results = index.search(
    namespace="en",
    top_k=5,
    inputs={"text": "quantum computing"},
    filter={"category": {"$eq": "science"}},
)
```

## Select returned fields

By default the server returns all record fields. Use ``fields`` to restrict the response:

```python
results = index.search(
    namespace="en",
    top_k=5,
    inputs={"text": "AI research"},
    fields=["body", "author"],
)
```

## See also

- {doc}`/how-to/inference/embeddings`: generate embeddings manually for non-integrated indexes.
- {doc}`/how-to/inference/reranking`: rerank results from any source.
