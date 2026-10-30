# Search documents

{meth}`~pinecone.client.documents.Documents.search` ranks the documents in one namespace
by the clauses in `score_by` and returns the `top_k` best. The examples continue the
`article-search` index from [Declare a schema and upsert documents](schema-and-upsert.md):
a `dense_vector` field `body_vector` and a `sparse_vector` field `body_sparse_vector`,
computed from `body` by your embedding model; full-text `title` and `body` fields; and
`category` and `year` metadata. Searches that have Pinecone embed
the query use `support-articles`, the same kind of index with Pinecone embedding `body`.

```python
from pinecone import DenseVectorQuery, Pinecone

pc = Pinecone(api_key="your-api-key")
index = pc.index("article-search")

response = index.documents.search(
    namespace="articles-en",
    top_k=10,
    score_by=[DenseVectorQuery(field="body_vector", values=[0.012, -0.087, 0.153])],
    include_fields=["title", "year"],
)
for doc in response.matches:
    print(doc.id, doc.score, doc.title)
```

## Choose a scoring clause

Each clause scores against fields of one kind, so the schema decides which clauses an
index can use. The clauses fall into three kinds of search. Semantic and sparse search
each come as a pair: one clause takes query text and Pinecone embeds it with the field's
model, the other takes a query vector you computed yourself.

| Kind of search | Pass query text | Pass a query vector you computed |
|---|---|---|
| [Semantic](#semantic-search): by meaning | {class}`~pinecone.models.documents.score_by.EmbedQuery`, on a `string` field with `embed` | {class}`~pinecone.models.documents.score_by.DenseVectorQuery`, on a `dense_vector` field |
| [Sparse](#sparse-search): by weighted term overlap | {class}`~pinecone.models.documents.score_by.SparseEmbedQuery`, on a `string` field with `sparse_embed` | {class}`~pinecone.models.documents.score_by.SparseVectorQuery`, on a `sparse_vector` field |
| [Keyword](#keyword-search): by the words themselves | {class}`~pinecone.models.documents.score_by.TextQuery` (BM25 relevance) or {class}`~pinecone.models.documents.score_by.QueryStringQuery` (a boolean expression), on full-text `string` fields | |

`score_by` also accepts each clause as a plain dict with a `type` key, such as
`{"type": "text", "query": "roman aqueduct", "fields": ["title"]}`.

### Semantic search

Semantic search finds documents by meaning, so a query can match documents that share
none of its words. It scores against the index's dense field, which is either a `string`
field with `embed` or a `dense_vector` field.

On a `string` field with `embed`, use `EmbedQuery` and pass the query as text. Pinecone
embeds it with the field's own model, so you never handle a vector. `article-search` has
no such field; this example searches the `support-articles` index from
[Declare a schema and upsert documents](schema-and-upsert.md), whose `body` field embeds
with `text-embedding-3-small`:

```python
from pinecone import EmbedQuery

support = pc.index("support-articles")
response = support.documents.search(
    namespace="articles-en",
    top_k=5,
    score_by=[EmbedQuery(field="body", query="how did Rome move water")],
)
```

On a `dense_vector` field, use `DenseVectorQuery` and pass a query embedding you
computed. It must come from the same model, with the same dimension, as the document
embeddings. The `body_vector` field of `article-search` holds them, computed from `body`
by your embedding model:

```python
from pinecone import DenseVectorQuery

response = index.documents.search(
    namespace="articles-en",
    top_k=10,
    score_by=[DenseVectorQuery(field="body_vector", values=[0.012, -0.087, 0.153])],
)
```

#### Optional: set the task type for `EmbedQuery`

`EmbedQuery` embeds the query with the field's `read_parameters`. To override them for one
search, pass `parameters`. Only `task_type` can be set, and only on a field whose model
accepts it; {meth}`pc.inference.list_models() <pinecone.client.inference.Inference.list_models>`
lists the models. See {class}`~pinecone.models.documents.score_by.EmbedQuery` for the
parameter reference.

`text-embedding-3-small` takes no `task_type`, so this example uses its own `fact-checks`
index, whose `body` field embeds with `gemini-embedding-2` for retrieval:

```python
from pinecone import EmbedQuery, SchemaBuilder

schema = (
    SchemaBuilder()
    .add_string_field(
        "body",
        embed={
            "model": "gemini-embedding-2",
            "write_parameters": {"task_type": "retrieval_document"},
        },
    )
    .build()
)
pc.indexes.create(name="fact-checks", schema=schema)

facts = pc.index("fact-checks")
facts.documents.upsert(
    namespace="claims-en",
    documents=[{"_id": "fact-1", "body": "The Eiffel Tower is 330 m tall."}],
)

response = facts.documents.search(
    namespace="claims-en",
    top_k=5,
    score_by=[
        EmbedQuery(
            field="body",
            query="Is the Eiffel Tower taller than 300 m?",
            parameters={"task_type": "fact_verification"},
        )
    ],
)
```

### Sparse search

Sparse search scores documents by the terms they share with the query, each weighted by
how much it matters, as a sparse embedding model or a keyword-weighting scheme encodes
them. It scores against the index's sparse field, which is either a `string` field with
`sparse_embed` or a `sparse_vector` field.

On a `string` field with `sparse_embed`, `SparseEmbedQuery` works like `EmbedQuery`: pass
the query as text, and Pinecone encodes it with the field's sparse model. It takes no
`parameters`.

On a `sparse_vector` field, use `SparseVectorQuery` and pass a
{class}`~pinecone.models.vectors.sparse.SparseValues` you encoded yourself, the same way
as the document vectors, so its indices name the same terms. The `body_sparse_vector`
field of `article-search` holds them, computed from `body` by your sparse embedding model:

```python
from pinecone import SparseValues, SparseVectorQuery

response = index.documents.search(
    namespace="articles-en",
    top_k=10,
    score_by=[
        SparseVectorQuery(
            field="body_sparse_vector",
            sparse_values=SparseValues(indices=[10, 42], values=[0.4, 0.9]),
        )
    ],
)
```

### Keyword search

Keyword search matches the words themselves, which suits exact terms, names, and codes.
It scores against full-text `string` fields. `title` and `body` are full-text in both
indexes; see
[Create the index](schema-and-upsert.md#create-the-index) for how full-text fields are
declared.

`TextQuery` matches the words literally across the fields you list and scores them by
BM25 relevance. `QueryStringQuery` takes a Lucene expression with the fields named inside
it, for when the query needs `AND`, `OR`, or `NOT`:

```python
from pinecone import QueryStringQuery, TextQuery

keyword = index.documents.search(
    namespace="articles-en",
    top_k=10,
    score_by=[TextQuery(query="roman aqueduct", fields=["title"])],
)
structured = index.documents.search(
    namespace="articles-en",
    top_k=10,
    score_by=[QueryStringQuery(query="title:(aqueduct AND roman) NOT draft")],
)
```

To weight terms or combine several keyword clauses, see [Combine clauses](#combine-clauses).

## Combine clauses

`text` and `query_string` clauses can share one `score_by` list. They are OR'd: a
document that matches any clause is returned, and one that matches several scores the
sum of its scores. Repeating a clause does not up-weight it; to weight a term, give it a
`^N` boost inside a `QueryStringQuery`, such as `title:aqueduct^3`. To require terms
together, use `AND` inside one `QueryStringQuery`.

A `dense_vector`, `sparse_vector`, `embed`, or `sparse_embed` clause must be the only
clause in `score_by`, and the client raises
{exc}`~pinecone.errors.exceptions.PineconeValueError` before sending a search that mixes
one with anything else. To rank by both meaning and keywords, run one search per clause
and merge the results in your code.

## Choose what comes back

`response.matches` is ordered best first; an empty list means nothing matched. Each match
is a {class}`~pinecone.models.documents.document.Document`: read `doc.id` and
`doc.score`, then your own fields as attributes.

By default a match carries only its ID and score. Name the fields you will read in
`include_fields`, or pass `["*"]` for every stored field. Use `doc.get("year")` for a
field some documents lack, since attribute access raises `AttributeError` when the field
is absent. A string field declared with `store_text=False` has no stored text to return,
and naming it in `include_fields` fails with a `400`.

## Filter the search

`filter` restricts which documents are scored. Which operators a predicate takes depends
on the field it names:

- Metadata fields, the ones the schema does not declare for search (`category`, `year`),
  take comparison, set, and existence operators such as `$eq`, `$gte`, `$in`, and
  `$exists`.
- Full-text `string` fields, including `embed` fields that store their text, take the text-match operators `$match_phrase` (the words in
  order), `$match_all` (every word), and `$match_any` (at least one word). They reject
  comparison operators.

Combine the two kinds under `$and`:

```python
response = index.documents.search(
    namespace="articles-en",
    top_k=10,
    score_by=[DenseVectorQuery(field="body_vector", values=[0.012, -0.087, 0.153])],
    filter={
        "$and": [
            {"year": {"$gte": 2020}},
            {"category": {"$in": ["history", "engineering"]}},
            {"title": {"$match_any": "aqueduct irrigation"}},
        ]
    },
)
```

To assemble a filter in code, {class}`~pinecone.utils.filter_builder.Field` builds the
same dict; pass its `to_dict()` result as `filter`:

```python
from pinecone import Field

condition = Field("year").gte(2020) & Field("title").match_any("aqueduct irrigation")
response = index.documents.search(
    namespace="articles-en",
    top_k=10,
    score_by=[DenseVectorQuery(field="body_vector", values=[0.012, -0.087, 0.153])],
    filter=condition.to_dict(),
)
```

A search on an index created another way uses a different method:
{meth}`~pinecone.index.Index.query` for an index created with `dimension=`, and
{meth}`~pinecone.index.Index.search` for a legacy integrated index created with
{meth}`~pinecone.client.indexes.Indexes.create_for_model`.
