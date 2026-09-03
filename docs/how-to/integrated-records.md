# Integrated records (server-side embedding)

An index with integrated inference stores text records and embeds them server-side with a
hosted model — the same model on write and on read. You upsert text and you search with
text; no vector ever passes through your code.

The alternative is to embed the text yourself and upsert vectors. See
[Generating embeddings](inference/embeddings.md) for that path, and
[Upserting and querying vectors](vectors/upsert-and-query.md) for the read side of it.

## Create an index with integrated inference

{meth}`~pinecone.client.indexes.Indexes.create_for_model` is how these indexes are
created. `embed` needs a `model` and a `field_map`:

```python
from pinecone import Pinecone

pc = Pinecone(api_key="your-api-key")

index_model = pc.indexes.create_for_model(
    name="semantic-search",
    cloud="aws",
    region="us-east-1",
    embed={"model": "multilingual-e5-large", "field_map": {"text": "chunk_text"}},
)
print(index_model.schema.fields["chunk_text"])
```

In `field_map`, the **key** is the model's own input field — `"text"` for the text
models — and the **value** is the field in your records that holds the content to embed.
The call blocks until the index is ready unless you pass `timeout=-1`.

{class}`~pinecone.models.indexes.specs.EmbedConfig` is the typed equivalent of that
dict, and adds `read_parameters` and `write_parameters` for the model arguments applied
when embedding a query and an upsert respectively:

```python
from pinecone import EmbedConfig

index_model = pc.indexes.create_for_model(
    name="semantic-search",
    cloud="aws",
    region="us-east-1",
    embed=EmbedConfig(
        model="multilingual-e5-large",
        field_map={"text": "chunk_text"},
        write_parameters={"input_type": "passage"},
        read_parameters={"input_type": "query"},
    ),
)
```

The model cannot be changed after creation.

### How this appears in the index schema

The embedding configuration comes back as a
{class}`~pinecone.models.indexes.schema.SemanticTextField` in the index's schema, named
after the `field_map` text entry:

```python
index_model = pc.indexes.describe("semantic-search")
print({name: type(f).__name__ for name, f in index_model.schema.fields.items()})
# {'chunk_text': 'SemanticTextField'}
print(index_model.schema.fields["chunk_text"].model)
```

`semantic_text` is read-only in a schema. You cannot declare one in a `schema=` you pass
to {meth}`~pinecone.client.indexes.Indexes.create` — sending it is rejected, and a single
rejected field fails the whole schema. `create_for_model` is what produces the field.
`create(schema=...)` is for the index shapes where you supply the vectors yourself:
`dense_vector`, `sparse_vector`, and `string` with full-text search. See
[Working with serverless indexes](indexes/serverless.md).

Passing `spec=IntegratedSpec(...)` to `create()` raises
{exc}`~pinecone.errors.exceptions.PineconeTypeError`; the message prints the equivalent
`create_for_model` call, filled in from the spec you passed.

## Get a handle to the index

```python
index = pc.index("semantic-search")
```

## Upsert records

{meth}`~pinecone.index.Index.upsert_records` takes record dicts, each with an `_id` (or
`id`) plus the field named in `field_map`. Records are sent as newline-delimited JSON
and embedded server-side:

```python
response = index.upsert_records(
    namespace="en",
    records=[
        {"_id": "article-1", "chunk_text": "Vector databases accelerate AI search."},
        {"_id": "article-2", "chunk_text": "RAG pipelines combine retrieval with generation."},
        {"_id": "article-3", "chunk_text": "Pinecone scales to billions of vectors."},
    ],
)
print(response.record_count)
```

`namespace` is required and must be non-empty. Unlike
{meth}`~pinecone.index.Index.upsert`, this API has no default namespace to fall back on,
and the SDK rejects the call before sending anything.

Any other keys on the record are stored as metadata and indexed for filtering
automatically — they need no schema declaration. A record giving both `_id` and `id`
keeps `_id`. A missing or non-string `_id` raises
{exc}`~pinecone.errors.exceptions.PineconeValueError`, naming the offending record's
position, before any HTTP request.

`record_count` is the number of records you submitted, counted client-side. Writes are not
readable the instant this returns, so allow for freshness lag before searching for a
record you just wrote.

## Search records

{meth}`~pinecone.index.Index.search` with `inputs` sends text, and the index's own model
embeds it as the query vector:

```python
results = index.search(
    namespace="en",
    top_k=5,
    inputs={"text": "AI and machine learning"},
)

for hit in results.result.hits:
    print(hit.id, hit.score, hit.fields)
```

The `"text"` key here is the model's input field, the same key as on the left of
`field_map` — not the name of your record field.

`search` also accepts `vector=` or `id=` in place of `inputs=`, for when you already hold
a vector or want an existing record to serve as the query. Exactly one query source is
required.

### Response: hits, not matches

`search` returns a {class}`~pinecone.models.vectors.search.SearchRecordsResponse`, and
its shape is **not** {meth}`~pinecone.index.Index.query`'s:

| | `search` | `query` |
|---|---|---|
| Where the results live | `response.result.hits` | `response.matches` |
| One result | {class}`~pinecone.models.vectors.search.Hit` | `ScoredVector` |
| Your record's data | `hit.fields`, one dict | split across `.values` and `.metadata` |

The extra `result` step is the response envelope, and forgetting it is the usual first
stumble. A search that matched nothing returns an empty `hits` list rather than raising.

Each {class}`~pinecone.models.vectors.search.Hit` exposes:

- `.id` — the record identifier.
- `.score` — how well the record matched, higher being better. After reranking the scale
  is the reranker's, not the index's, so compare scores only within one response.
- `.fields` — your record's own fields, keyed by field name, subject to the `fields`
  argument below.

`.usage` breaks the cost out by stage: `read_units` always, `embed_total_tokens` only when
the index embedded your text, and `rerank_units` only when you passed `rerank`. A stage
that did not run reports `None`.

For the vector-supplied side of the comparison, see [`query` or `search`: pick
one](vectors/upsert-and-query.md#query-or-search-pick-one).

### Typed dicts for `inputs` and `rerank`

{class}`~pinecone.models.vectors.search.SearchInputs` and
{class}`~pinecone.models.vectors.search.RerankConfig` are
{class}`~typing.TypedDict` definitions, so there is nothing to instantiate — annotate a
plain dict and your editor and type checker will check the keys for you:

```python
from pinecone import SearchInputs

inputs: SearchInputs = {"text": "AI and machine learning"}

results = index.search(namespace="en", top_k=5, inputs=inputs)
```

## Rerank in a single search call

A `rerank` config retrieves and reranks in one request. `model` and `rank_fields` are
both required, and the SDK rejects the call before sending if either is missing:

```python
results = index.search(
    namespace="en",
    top_k=50,
    inputs={"text": "best practices for vector search"},
    rerank={
        "model": "bge-reranker-v2-m3",
        "rank_fields": ["chunk_text"],
        "top_n": 5,
    },
)

for hit in results.result.hits:
    print(hit.id, hit.score)
```

`top_k` is how many candidates the search retrieves; `top_n` is how many survive
reranking, defaulting to `top_k`. Set `top_n` lower to have the reranker narrow a wider
candidate set — that is the whole reason to retrieve more than you intend to show.

`rank_fields` names record fields, so they must be fields the search returns. Add
`query` to the rerank config when the text to rerank against should differ from the search
text.

For reranking candidates that came from somewhere other than this index, use
`pc.inference.rerank`. See [Reranking results](inference/reranking.md).

## Filter by metadata

```python
results = index.search(
    namespace="en",
    top_k=5,
    inputs={"text": "quantum computing"},
    filter={"category": {"$eq": "science"}},
)
print(len(results.result.hits))
```

The filter grammar is the same one {meth}`~pinecone.index.Index.query` uses — see
[Filter by metadata](vectors/upsert-and-query.md#filter-by-metadata).

## Select returned fields

Omitting `fields` returns every field each record has. Narrow it when you only need one
or two:

```python
results = index.search(
    namespace="en",
    top_k=5,
    inputs={"text": "AI research"},
    fields=["chunk_text", "category"],
)
```

## See also

- [Generating embeddings](inference/embeddings.md) — embed text yourself, for an index
  you supply vectors to.
- [Reranking results](inference/reranking.md) — rerank candidates from any source.
- [Upserting and querying vectors](vectors/upsert-and-query.md) — the vector-in,
  vector-out surface.
