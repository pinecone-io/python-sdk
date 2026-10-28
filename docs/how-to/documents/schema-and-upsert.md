# Declare a schema and upsert documents

A document index is an index created with a `schema` that names its searchable fields.
You write and read it through `index.documents`, and each record is a JSON document: a
reserved `_id` plus your own fields. This page covers choosing fields, creating the
index, and loading data.

If your index was created another way, it uses a different interface: top-level
`dimension=`/`metric=` indexes take {meth}`~pinecone.index.Index.upsert` (see
[Upserting and querying vectors](../vectors/upsert-and-query.md)), and indexes from
{meth}`~pinecone.client.indexes.Indexes.create_for_model` take the legacy Records API (see
[Legacy integrated indexes](../integrated-records.md)).

## Choose your fields

Declare only the fields you search. Each kind of field is searched with its own
`score_by` clause:

| Field | Your documents carry | Search it with |
|---|---|---|
| `dense_vector` | a list of floats from your embedding model | {class}`~pinecone.models.documents.score_by.DenseVectorQuery` |
| `sparse_vector` | `{"indices": [...], "values": [...]}` | {class}`~pinecone.models.documents.score_by.SparseVectorQuery` |
| `string` with full-text search | text | {class}`~pinecone.models.documents.score_by.TextQuery`, {class}`~pinecone.models.documents.score_by.QueryStringQuery` |
| `string` with `embed` | text; Pinecone embeds it with a hosted dense model | {class}`~pinecone.models.documents.score_by.EmbedQuery` |
| `string` with `sparse_embed` | text; Pinecone embeds it with a hosted sparse model | {class}`~pinecone.models.documents.score_by.SparseEmbedQuery` |

An index holds at most one dense field (a `dense_vector`, or a `string` with `embed`) and
at most one sparse field (a `sparse_vector`, or a `string` with `sparse_embed`). Values
you only filter on, such as a category or a year, stay out of the schema: put them on
the documents and they are indexed for filtering when first written.

The schema cannot change after the index is created, so declare every searched field up
front.

## Create the index

{class}`~pinecone.schema_builder.SchemaBuilder` lays out the field types and their
options. This index stores your own embeddings and also supports keyword search on the
title:

```python
from pinecone import Pinecone, SchemaBuilder

pc = Pinecone(api_key="your-api-key")

schema = (
    SchemaBuilder()
    .add_dense_vector_field("your_vector", dimension=1536, metric="cosine")
    .add_string_field("title", full_text_search={"language": "en", "stemming": True})
    .build()
)
pc.indexes.create(name="article-search", schema=schema)
```

`dimension` and `metric` must match the embedding model you use. `create` blocks until
the index is ready; pass `timeout=-1` to return immediately. Without `deployment=`, the
index is serverless on AWS `us-east-1`; see
[Working with serverless indexes](../indexes/serverless.md) to choose another cloud or
region.

To have Pinecone embed text for you, give a `string` field `embed`, `sparse_embed`, or
both. `True` takes the default model:

```python
schema = SchemaBuilder().add_string_field("body", embed=True, sparse_embed=True).build()
pc.indexes.create(name="support-articles", schema=schema)
```

The text of an embed field is stored and full-text searchable unless you pass
`store_text=False`. See
[Integrated embedding](../../guides/concepts.md#integrated-embedding) for choosing a model
and what `store_text=False` gives up.

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
            "your_vector": [0.012, -0.087, 0.153],  # 1536 floats in practice
            "title": "How Roman aqueducts carried water",
            "category": "history",
            "year": 2024,
        },
        {
            "_id": "article-102",
            "your_vector": [0.045, 0.021, -0.064],
            "title": "Gravity-fed irrigation today",
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

For the `support-articles` index above, a document is just its `_id`, its text, and any
metadata: `{"_id": "article-101", "body": "Roman aqueducts carried water by gravity"}`.

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
        await index.documents.upsert(namespace="articles-en", documents=documents)
```
