# Working with namespaces

Namespaces are logical partitions within a Pinecone index. Vectors in different
namespaces are completely isolated: a query in one namespace never returns results from
another, and there is no operation that searches every namespace at once.

Common uses include separating data by customer, language, environment (staging vs.
production), or data version.

Every operation that accepts a `namespace` defaults it to `""`, the empty string, which
addresses the index's default namespace. That default is where the isolation rule bites:

:::{warning}
Writing to a namespace and then reading without naming it is the most common way to
"lose" data in Pinecone. The read succeeds, searches the default namespace `""`, and
returns nothing.

```python
index.upsert(vectors=[("article-101", [0.012, -0.087, 0.153])], namespace="articles-en")

index.query(vector=[0.012, -0.087, 0.153], top_k=10)
# -> matches == [], because this queried the default namespace ""

index.query(vector=[0.012, -0.087, 0.153], top_k=10, namespace="articles-en")
# -> the vector that was written
```

Nothing is raised and nothing is logged, so if a query comes back empty against an index
you know holds data, check the `namespace` on the write and the read before anything
else. {meth}`~pinecone.index.Index.describe_index_stats` lists every namespace with its vector
count, which settles the question in one call.
:::

Vectors on this page are written as three floats so they fit the page. Pass your index's
full `dimension`.


## Upsert into a namespace

Pass `namespace` to {meth}`~pinecone.index.Index.upsert` to write vectors into a specific partition:

```python
from pinecone import Pinecone, Vector

pc = Pinecone(api_key="your-api-key")
index = pc.index("product-search")

index.upsert(
    vectors=[
        Vector(id="product-001", values=[0.012, -0.087, 0.153]),
        Vector(id="product-002", values=[0.045, 0.021, -0.064]),
    ],
    namespace="catalog-us",
)
```

Vectors upserted without a `namespace` go into the default namespace `""`. The namespace
does not have to exist first — the write creates it.


## Query within a namespace

Pass `namespace` to {meth}`~pinecone.index.Index.query` to restrict the search to a single partition:

```python
response = index.query(
    vector=[0.012, -0.087, 0.153],
    top_k=10,
    namespace="catalog-us",
)
for match in response.matches:
    print(match.id, match.score)
```

`response.namespace` names the namespace that was searched, and reads `""` for the
default one — useful when the namespace came from a variable and you want the response
to confirm it.

### Query across multiple namespaces

There is no "search everything" flag, but {meth}`~pinecone.index.Index.query_namespaces` runs
one query per namespace in parallel and merges the results into a single ranking. It
needs `metric` because merging scores from separate namespaces means knowing whether
higher or lower is closer:

```python
results = index.query_namespaces(
    vector=[0.012, -0.087, 0.153],
    namespaces=["catalog-us", "catalog-eu", "catalog-ap"],
    metric="cosine",
    top_k=10,
)
for match in results.matches:
    print(match.id, match.score)
```


## List namespaces

{meth}`~pinecone.index.Index.list_namespaces` yields one {class}`~pinecone.models.namespaces.models.ListNamespacesResponse`
per page, following pagination automatically:

```python
for page in index.list_namespaces():
    for ns in page.namespaces:
        print(ns.name, ns.record_count)
```

Each {class}`~pinecone.models.namespaces.models.NamespaceDescription` carries `name`, `record_count`, and
`size_bytes`. When the namespace restricts which metadata fields are indexed, it also
carries `schema` and `indexed_fields`.

`size_bytes` is an approximation, not an exact byte count: data written before size
tracking was enabled reads as `0`, and recently deleted data may still be counted until
compaction converges the value. A response that omits the field also reads as `0`, so
treat a `0` as "no size reported" rather than "the namespace is empty".

Filter by prefix to list a subset of namespaces:

```python
for page in index.list_namespaces(prefix="catalog-"):
    for ns in page.namespaces:
        print(ns.name)
```

For a single page without automatic pagination, use
{meth}`~pinecone.index.Index.list_namespaces_paginated`:

```python
page = index.list_namespaces_paginated(limit=50)
for ns in page.namespaces:
    print(ns.name, ns.record_count)

# Fetch the next page manually
if page.pagination and page.pagination.next:
    next_page = index.list_namespaces_paginated(
        limit=50,
        pagination_token=page.pagination.next,
    )
```


## Delete all vectors in a namespace

{meth}`~pinecone.index.Index.delete` with `delete_all=True` removes every vector in a namespace
without deleting the namespace itself:

```python
index.delete(delete_all=True, namespace="catalog-staging")
```

Alternatively, {meth}`~pinecone.index.Index.delete_namespace` removes the namespace and all its
vectors:

```python
index.delete_namespace(name="catalog-staging")
```


## Describe a namespace

{meth}`~pinecone.index.Index.describe_namespace` returns metadata for a single namespace:

```python
ns = index.describe_namespace(name="catalog-us")
print(ns.name)
print(ns.record_count)
print(ns.size_bytes)
```

Pass `__default__` to describe the namespace that requests address when they omit one:

```python
ns = index.describe_namespace(name="__default__")
```

This operation is rate limited per index, independently of the other namespace
operations. To describe more than one namespace, use `list_namespaces()` instead. It
returns the same information for every namespace in a single request and is not subject
to that limit. Fanning out `describe_namespace` calls will raise
{exc}`~pinecone.errors.exceptions.RateLimitError`.


## Create a namespace

Creating a namespace explicitly is **optional**: upserting into a name that does not
exist yet creates it. {meth}`~pinecone.index.Index.create_namespace` is for the case where you
want the namespace to exist before any data lands in it, or where you want to configure
which metadata fields it indexes up front. It raises
{exc}`~pinecone.errors.exceptions.ConflictError` if the namespace already exists, so it
is not an idempotent "ensure exists".

```python
ns = index.create_namespace(
    name="catalog-us",
    schema={"fields": {"category": {"filterable": True}}},
)
print(ns.name, ns.record_count)
```

Every field listed in `schema["fields"]` must set `filterable: True`; `filterable: False`
is not supported. To leave a field unindexed, omit it from `fields` entirely.

Omitting `schema` altogether is not the same as indexing every field. A namespace created
without one inherits the index's own metadata-index configuration, so if the index
restricts which fields are indexed, the new namespace carries that restriction too.
Supplying `schema` overrides the inherited configuration for that namespace alone.

### Name rules

Namespace names must be ASCII, must not contain the NUL character, and must be 1-512
characters long. `__default__` is reserved, since it names the namespace requests address
when they omit a namespace, so it always exists and `create_namespace` rejects it. Names that
break these rules raise {exc}`~pinecone.errors.exceptions.PineconeValueError` before any
request is sent, so the offending value is reported back to you rather than to the server.


## See also

- {doc}`/how-to/vectors/upsert-and-query`: upsert and query operations
- {doc}`/how-to/vectors/bulk-import`: loading a namespace from cloud storage
- {class}`~pinecone.index.Index`: full data plane client reference
- {class}`~pinecone.models.namespaces.models.ListNamespacesResponse`: list namespaces response model
- {class}`~pinecone.models.namespaces.models.NamespaceDescription`: namespace metadata model
