# Declare a schema and upsert documents

A document index is an index created with a `schema` that names its searchable fields.
You write and read it through `index.documents`, and each record is a JSON document: a
reserved `_id` plus your own fields. This page covers choosing fields, creating the
index, and loading data.

If your index was created another way, it uses a different interface. For an index
created with top-level `dimension=`/`metric=`, use {meth}`~pinecone.index.Index.upsert`
(see [Upserting and querying vectors](../vectors/upsert-and-query.md)). For one created
with {meth}`~pinecone.client.indexes.Indexes.create_for_model`, use the legacy Records API
(see [Legacy integrated indexes](../integrated-records.md)).

## Choose your fields

Declare only the fields you search. The table shows the kind of search each field type
supports and the `score_by` clause that runs it:

| Kind of search | `score_by` clause | Field type | Your documents carry |
|---|---|---|---|
| [Semantic](search.md#semantic-search) | {class}`~pinecone.models.documents.score_by.DenseVectorQuery` | `dense_vector` | a list of floats from your embedding model |
| [Semantic](search.md#semantic-search) | {class}`~pinecone.models.documents.score_by.EmbedQuery` | `string` with `embed` | text; Pinecone embeds it with a hosted dense model |
| [Sparse-vector](search.md#sparse-vector-search) | {class}`~pinecone.models.documents.score_by.SparseVectorQuery` | `sparse_vector` | a sparse vector, `{"indices": [...], "values": [...]}` |
| [Sparse-vector](search.md#sparse-vector-search) | {class}`~pinecone.models.documents.score_by.SparseEmbedQuery` | `string` with `sparse_embed` | text; Pinecone embeds it with a hosted sparse model |
| [Full-text](search.md#full-text-search) | {class}`~pinecone.models.documents.score_by.TextQuery` or {class}`~pinecone.models.documents.score_by.QueryStringQuery` | `string` with full-text search | text |

An index holds at most one dense field (a `dense_vector`, or a `string` with `embed`) and
at most one sparse field (a `sparse_vector`, or a `string` with `sparse_embed`). Fields
you only use for filtering, such as a category or a year, don't need to be specified in
the schema. Include them in your documents and they're automatically indexed for filtering
when first written.

Currently, the schema cannot change after the index is created, so declare every searched
field up front.

## Create the index

{class}`~pinecone.schema_builder.SchemaBuilder` lays out the field types and their
options. This index stores dense and sparse vectors that your embedding model computes
from `body`, and supports full-text search on `title` and `body`:

```python
from pinecone import Pinecone, SchemaBuilder

pc = Pinecone(api_key="your-api-key")

schema = (
    SchemaBuilder()
    .add_dense_vector_field("body_vector", dimension=1536, metric="cosine")
    .add_sparse_vector_field("body_sparse_vector")
    .add_string_field("title", full_text_search=True)
    .add_string_field("body", full_text_search={"language": "en", "stemming": True})
    .build()
)
pc.indexes.create(name="article-search", schema=schema)
```

`dimension` and `metric` must match the embedding model you use. `create` blocks until
the index is ready; pass `timeout=-1` to return immediately. Without `deployment=`, the
index is serverless on AWS `us-east-1`; see
[Working with serverless indexes](../indexes/serverless.md) to choose another cloud or
region.

`support-articles` is the same kind of index, except that instead of vectors it has a
`body` field configured for integrated embedding, so Pinecone embeds the text for you.
Give the `string` field `embed` for a dense embedding, here with `text-embedding-3-small`,
and `sparse_embed` for a sparse one; `True` takes the default model. As in
`article-search`, `body` is also declared full-text, so the index has two full-text
fields, `title` and `body`.

```python
schema = (
    SchemaBuilder()
    .add_string_field("title", full_text_search=True)
    .add_string_field("body",
        embed={"model": "text-embedding-3-small"},
        sparse_embed=True,
        full_text_search={"language": "en", "stemming": True},
    )
    .build()
)
pc.indexes.create(name="support-articles", schema=schema)
```

`full_text_search=True` gives a field the default full-text settings, as `title` has in
both indexes. The stored text of an `embed` field is full-text searchable with those
defaults even without `full_text_search`; declaring a settings dict, as `body` does here,
sets the language and stemming. When `embed` or `sparse_embed` is configured, a field can
set `store_text=False` to skip storing its text, but note that this also disables
full-text search on it. See
[Integrated embedding](../../guides/concepts.md#integrated-embedding) for choosing a model.

`build()` returns a plain `{"fields": {...}}` dict, and writing that dict yourself is
equally supported. The builder does not decide what the server accepts: an invalid
combination is sent as written, and `create` raises
{exc}`~pinecone.errors.exceptions.ApiError` with the server's message.

## Upsert documents

{meth}`~pinecone.client.documents.Documents.upsert` writes a list of documents to a
namespace. A document whose `_id` already exists is replaced.

```python
index = pc.index("article-search")

response = index.documents.upsert(
    namespace="articles-en",
    documents=[
        {
            "_id": "article-101",
            "body_vector": [0.012, -0.087, 0.153],  # 1536 floats in practice
            "body_sparse_vector": {"indices": [10, 42], "values": [0.5, 0.3]},
            "title": "How a Roman aqueduct carried water",
            "body": "Roman aqueducts carried water by gravity",
            "category": "history",
            "year": 2024,
        },
        {
            "_id": "article-102",
            "body_vector": [0.045, 0.021, -0.064],
            "title": "Gravity-fed irrigation today",
            "body": "Farms still move water downhill without pumps",
            "category": "engineering",
            "year": 2023,
        },
    ],
)
print(response.upserted_count)
```

Each document must carry every `dense_vector` field and the text of every `embed` field.
Sparse fields, full-text-only fields, and metadata such as `category` are optional. The
server rejects a document that is missing a required field with a `400` naming the
document and the field.

The `support-articles` index above takes the same documents without the vectors, since
Pinecone embeds `body` for you:

```python
support = pc.index("support-articles")

response = support.documents.upsert(
    namespace="articles-en",
    documents=[
        {
            "_id": "article-101",
            "title": "How a Roman aqueduct carried water",
            "body": "Roman aqueducts carried water by gravity",
            "category": "history",
            "year": 2024,
        },
        {
            "_id": "article-102",
            "title": "Gravity-fed irrigation today",
            "body": "Farms still move water downhill without pumps",
            "category": "engineering",
            "year": 2023,
        },
    ],
)
print(response.upserted_count)
```

Documents can also be built as {class}`~pinecone.models.documents.document.DocumentRecord`
objects, which check the `_id` where you construct them.

### Load large datasets

`upsert` sends one request, and the server limits a request by document count and by
size. For more documents than that, or documents of unknown size, use
{meth}`~pinecone.client.documents.Documents.batch_upsert`. It splits the list into
batches, sends them in parallel, and halves any batch the server rejects as too large.
Here `documents` is a list of document dicts like the ones above:

```python
result = index.documents.batch_upsert(namespace="articles-en", documents=documents)

for error in result.errors:
    print(error.batch_index, error.retryable, error.error_message)
```

A failed batch is recorded in the returned {class}`~pinecone.models.batch.BatchResult`
instead of raised, so one bad batch does not stop the rest. Resend `result.failed_items`
after fixing whatever `result.errors` reports. On an index with an `embed` or
`sparse_embed` field, the server accepts fewer documents per request; the default
`batch_size` fits, so raise it only on other indexes. See
[How bulk ingest behaves](../../guides/bulk-ingest.md) for tuning `batch_size` and
`max_concurrency`.

## Wait for documents to be visible

Upserts apply asynchronously: `upsert` returns once the server accepts the documents, and
a search or fetch made right afterward may not see them yet. Code that reads its own
writes, such as a test or a demo script, should poll until the documents appear:

```python
import time

ids = ["article-101", "article-102"]
while len(index.documents.fetch(namespace="articles-en", ids=ids).documents) < len(ids):
    time.sleep(1)
```

## Async client

{class}`~pinecone.async_client.documents.AsyncDocuments` on
{class}`~pinecone.async_client.async_index.AsyncIndex` has the same methods with the same
arguments; await each call:

```python
from pinecone import AsyncPinecone

async with AsyncPinecone(api_key="your-api-key") as pc:
    index = await pc.index(name="article-search")
    async with index:
        await index.documents.upsert(
            namespace="articles-en",
            documents=[
                {
                    "_id": "article-102",
                    "body_vector": [0.045, 0.021, -0.064],  # 1536 floats in practice
                    "title": "Gravity-fed irrigation today",
                    "body": "Farms still move water downhill without pumps",
                },
            ],
        )
```
