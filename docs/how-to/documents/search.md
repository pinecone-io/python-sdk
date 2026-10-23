# Search documents

{meth}`~pinecone.client.documents.Documents.search` ranks the documents in one namespace
by the clauses in `score_by` and returns the `top_k` best. The examples continue the
`article-search` index from [Declare a schema and upsert documents](schema-and-upsert.md):
a `dense_vector` field `embedding`, a full-text `title`, and `category` and `year`
metadata.

```python
from pinecone import DenseVectorQuery, Pinecone

pc = Pinecone(api_key="your-api-key")
index = pc.index("article-search")

response = index.documents.search(
    namespace="articles-en",
    top_k=10,
    score_by=[DenseVectorQuery(field="embedding", values=[0.012, -0.087, 0.153])],
    include_fields=["title", "year"],
)
for doc in response.matches:
    print(doc.id, doc.score, doc.title)
```

## Choose a scoring clause

Each clause scores against fields of one kind, so the schema decides which clauses an
index can use:

| To find documents by | Use | Against |
|---|---|---|
| meaning, with a query embedding you computed | {class}`~pinecone.models.documents.score_by.DenseVectorQuery` | a `dense_vector` field |
| meaning, from query text Pinecone embeds | {class}`~pinecone.models.documents.score_by.EmbedQuery` | a `string` field with `embed` |
| the words in the query (BM25 keyword relevance) | {class}`~pinecone.models.documents.score_by.TextQuery` | full-text `string` fields |
| a boolean expression: required, excluded, or weighted terms | {class}`~pinecone.models.documents.score_by.QueryStringQuery` | full-text `string` fields |
| term overlap, with a sparse query vector you computed | {class}`~pinecone.models.documents.score_by.SparseVectorQuery` | a `sparse_vector` field |
| term overlap, from query text Pinecone encodes | {class}`~pinecone.models.documents.score_by.SparseEmbedQuery` | a `string` field with `sparse_embed` |

A dense query embedding must come from the same model, with the same dimension, as the
document embeddings. With `EmbedQuery` and `SparseEmbedQuery`, Pinecone uses the field's
own model, so you never handle a vector:

```python
from pinecone import EmbedQuery

support = pc.index("support-articles")
response = support.documents.search(
    namespace="articles-en",
    top_k=5,
    score_by=[EmbedQuery(field="body", query="how did Rome move water")],
)
```

`EmbedQuery` embeds the query with the field's `read_parameters`. To override them for one
search, pass `parameters`. Only `task_type` can be set, and `GET /models` lists the models
that accept it, such as `gemini-embedding-2`. `SparseEmbedQuery` takes no `parameters`.

```python
response = support.documents.search(
    namespace="articles-en",
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

`TextQuery` matches the words literally across the fields you list. `QueryStringQuery`
takes a Lucene expression with the fields named inside it:

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

`score_by` also accepts each clause as a plain dict with a `type` key, such as
`{"type": "text", "query": "roman aqueduct", "fields": ["title"]}`.

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
    score_by=[DenseVectorQuery(field="embedding", values=[0.012, -0.087, 0.153])],
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
    score_by=[DenseVectorQuery(field="embedding", values=[0.012, -0.087, 0.153])],
    filter=condition.to_dict(),
)
```

A search on an index created another way uses a different method:
{meth}`~pinecone.index.Index.query` for an index created with `dimension=`, and
{meth}`~pinecone.index.Index.search` for a legacy integrated index created with
{meth}`~pinecone.client.indexes.Indexes.create_for_model`.
