# Working with Namespaces

Namespaces are logical partitions within a Pinecone index. Vectors in different namespaces
are completely isolated — a query in one namespace never returns results from another.

Common uses include separating data by customer, language, environment (staging vs.
production), or data version.

The default namespace is the empty string `""`. All operations that accept a `namespace`
parameter default to `""` when `namespace` is omitted.


## Upsert into a namespace

Pass `namespace` to {meth}`~pinecone.Index.upsert` to write vectors into a specific partition:

```python
from pinecone import Pinecone, Vector

pc = Pinecone(api_key="your-api-key")
index = pc.index("product-search")

index.upsert(
    vectors=[
        Vector(id="product-001", values=[0.012, -0.087, 0.153, ...]),
        Vector(id="product-002", values=[0.045,  0.021, -0.064, ...]),
    ],
    namespace="catalog-us",
)
```

Vectors upserted without a `namespace` go into the default namespace `""`.


## Query within a namespace

Pass `namespace` to {meth}`~pinecone.Index.query` to restrict the search to a single partition:

```python
response = index.query(
    vector=[0.012, -0.087, 0.153, ...],
    top_k=10,
    namespace="catalog-us",
)
for match in response.matches:
    print(match.id, match.score)
```

Queries return `response.namespace` indicating which namespace was searched.

### Query across multiple namespaces

{meth}`~pinecone.Index.query_namespaces` fans out queries in parallel and returns merged
top results:

```python
results = index.query_namespaces(
    vector=[0.012, -0.087, 0.153, ...],
    namespaces=["catalog-us", "catalog-eu", "catalog-ap"],
    metric="cosine",
    top_k=10,
)
for match in results.matches:
    print(match.id, match.score)
```


## List namespaces

{meth}`~pinecone.Index.list_namespaces` yields one {class}`~pinecone.models.ListNamespacesResponse`
per page, following pagination automatically:

```python
for page in index.list_namespaces():
    for ns in page.namespaces:
        print(ns.name, ns.record_count)
```

Each {class}`~pinecone.models.NamespaceDescription` carries `name`, `record_count`,
`size_bytes`, and — when the namespace restricts which metadata fields are indexed —
`schema` and `indexed_fields`.

`size_bytes` is approximate: data written before size tracking reads as `0`, and recently
deleted data may still be counted; compaction converges the value. `0` is also what the
field reads as against a server older than 2026-07, so treat it as "no size reported"
rather than "the namespace is empty".

Filter by prefix to list a subset of namespaces:

```python
for page in index.list_namespaces(prefix="catalog-"):
    for ns in page.namespaces:
        print(ns.name)
```

For a single page without automatic pagination, use
{meth}`~pinecone.Index.list_namespaces_paginated`:

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

{meth}`~pinecone.Index.delete` with `delete_all=True` removes every vector in a namespace
without deleting the namespace itself:

```python
index.delete(delete_all=True, namespace="catalog-staging")
```

Alternatively, {meth}`~pinecone.Index.delete_namespace` removes the namespace and all its
vectors:

```python
index.delete_namespace(name="catalog-staging")
```


## Describe a namespace

{meth}`~pinecone.Index.describe_namespace` returns metadata for a single namespace:

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
operations. To describe more than one namespace, use `list_namespaces()` instead — it
returns the same information for every namespace in a single request and is not subject
to that limit. Fanning out `describe_namespace` calls will raise
{exc}`~pinecone.errors.RateLimitError`.


## Create a namespace

Namespaces are created automatically when you first upsert into them. Use
{meth}`~pinecone.Index.create_namespace` when you need to pre-create one with a custom
schema or when you want to configure indexed metadata fields up front:

```python
ns = index.create_namespace(
    name="catalog-us",
    schema={"fields": {"category": {"filterable": True}}},
)
print(ns.name, ns.record_count)
```

Every field listed in `schema["fields"]` must set `filterable: True`; `filterable: False`
is not supported. To leave a field unindexed, omit it from `fields` entirely.

### Name rules

Namespace names must be ASCII, must not contain the NUL character, and must be 1-512
characters long. `__default__` is reserved — it names the namespace requests address when
they omit a namespace, so it always exists and `create_namespace` rejects it. Names that
break these rules raise {exc}`~pinecone.errors.ValidationError` before any request is
sent, so the offending value is reported back to you rather than to the server.


## See also

- {doc}`/how-to/vectors/upsert-and-query` — upsert and query operations
- {class}`~pinecone.Index` — full data plane client reference
- {class}`~pinecone.models.ListNamespacesResponse` — list namespaces response model
- {class}`~pinecone.models.NamespaceDescription` — namespace metadata model
