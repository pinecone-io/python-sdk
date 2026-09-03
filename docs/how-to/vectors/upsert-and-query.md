# Upserting and querying vectors

Use the {class}`~pinecone.Index` client to insert and retrieve vectors from a Pinecone index.
Get an index client via {meth}`~pinecone.Pinecone.index`:

```python
from pinecone import Pinecone

pc = Pinecone(api_key="your-api-key")
index = pc.index("movie-recommendations")
```

:::{note}
The methods on this page serve indexes created with top-level `dimension` and `metric`.
An index created with a `schema` naming your own vector field stores documents instead,
and the server rejects `upsert` and `query` on it — use `index.documents` and see the
[quickstart](../../getting-started/quickstart.md#choosing-an-interface) for how the two
interfaces divide.
:::


## Upsert vectors

{meth}`~pinecone.Index.upsert` inserts vectors or overwrites existing ones with the same ID.

Pass a list of tuples `(id, values)` or `(id, values, metadata)`:

```python
index.upsert(
    vectors=[
        ("movie-001", [0.012, -0.087, 0.153, ...]),
        ("movie-002", [0.045,  0.021, -0.064, ...]),
    ]
)
```

### Using Vector objects

{class}`~pinecone.Vector` objects support metadata and sparse values:

```python
from pinecone import Vector

response = index.upsert(
    vectors=[
        Vector(id="movie-001", values=[0.012, -0.087, 0.153, ...]),
        Vector(
            id="movie-002",
            values=[0.045, 0.021, -0.064, ...],
            metadata={"genre": "comedy", "year": 2022},
        ),
    ]
)
print(response.upserted_count)  # 2
```

`upsert` returns an {class}`~pinecone.models.UpsertResponse`. For a
single-request upsert, only `upserted_count` is meaningful; for a
batched upsert (see "Large datasets" below), the response also
carries per-batch counters and a `failed_items` list for retry.

### Upsert into a namespace

Pass `namespace` to target a specific partition:

```python
index.upsert(
    vectors=[("movie-001", [0.012, -0.087, 0.153, ...])],
    namespace="movies-en",
)
```

The default namespace is `""`.

### Large datasets

A single upsert request is capped both on the number of vectors it carries
and on its encoded size, and with wide vectors or heavy metadata the size
cap is usually the one reached first, so a vector count that worked for one
dataset can be rejected for another. Pass `batch_size` to
split the upload into chunks that stay under both; lower it and retry if a
request comes back rejected for size. Batches are sent **in parallel**
via a `ThreadPoolExecutor` (sync) or `asyncio.Semaphore`
(async) of `max_concurrency` workers. HTTP-level retries
happen automatically per batch via the configured
{class}`~pinecone.RetryConfig`.

```python
response = index.upsert(
    vectors=large_list,
    batch_size=200,        # vectors per request
    max_concurrency=8,     # parallel in-flight requests (1–64)
    show_progress=True,    # tqdm progress bar (auto-skipped if tqdm not installed)
)
print(response.upserted_count)         # successful items
print(response.total_item_count)       # total submitted
print(response.successful_batch_count) # batches that succeeded
```

Defaults: `batch_size=None` keeps the single-request behaviour
(no batching). When `batch_size` is set, `max_concurrency`
defaults to `4` and `show_progress` defaults to `True`.

For DataFrame input, {meth}`~pinecone.Index.upsert_from_dataframe`
provides the same parallel batching with column extraction.
For millions of vectors, consider
{meth}`~pinecone.Index.start_import` to load from cloud storage.

### Handling partial failures

Unlike a single-request upsert (which raises on failure), a
batched upsert never raises for per-batch errors. Instead, the
returned {class}`~pinecone.models.UpsertResponse` carries each
failed batch's exception and items, so you can retry only the
failures.

```python
response = index.upsert(vectors=huge_list, batch_size=200)

if response.has_errors:
    print(f"{response.failed_item_count} of {response.total_item_count} items failed")
    for err in response.errors:
        print(f"  batch {err.batch_index}: {err.error_message}")

    # Retry only the failures:
    retry = index.upsert(
        vectors=response.failed_items,
        batch_size=200,
    )
```

`response.failed_items` is a flat `list[dict]` of every item
from every failed batch, in original order. Pass it directly
back to `upsert(...)` for retry.

#### Inspect errors before retrying

Before retrying `failed_items`, look at why batches failed:

```python
if response.has_errors:
    first = response.errors[0]
    print(f"first failure: {first.error_message}")
```

If every error has the same HTTP status, especially a 4xx
like 400 (Bad Request), 401 (Unauthorized), 403 (Forbidden),
or 422 (Unprocessable Entity), the failures are about your
data or your credentials, not transient infrastructure.
Retrying with the same input will fail the same way. Fix the
data or the credentials and retry the corrected items, or
stop.

#### Why surfaced errors are usually persistent

The HTTP transport retries `{408, 429, 500, 502, 503, 504}`
automatically up to three times (four total attempts) with decorrelated jitter
(see {class}`~pinecone.RetryConfig`). That layer absorbs nearly
all transient infrastructure issues. By the time an error
reaches `response.errors`, it has either:

- exhausted the retry budget (sustained 5xx, persistent 429), or
- wasn't retryable in the first place (4xx: bad input, auth,
  validation).

Either way, naive retries usually re-create the same problem.
Treat each entry in `response.errors` as a real signal worth
reading.

#### Batches fail atomically

Any per-batch error fails the **entire batch**, even if only
one of its 200 vectors was the actual problem. So
`response.failed_items` may contain 199 items that would have
succeeded on their own, plus the one bad row that triggered
the rejection. The server doesn't surface per-item rejection
details on the upsert path.

To isolate the bad row, re-batch the failures with a smaller
`batch_size` (down to `batch_size=1` if needed). Successful
single-item batches narrow the problem to the rejected ones:

```python
if response.has_errors:
    narrow = index.upsert(vectors=response.failed_items, batch_size=1)
    # narrow.failed_items now contains only the actually-bad rows
```


## Query for nearest neighbors

{meth}`~pinecone.Index.query` returns the `top_k` closest vectors to a query vector:

```python
response = index.query(
    vector=[0.012, -0.087, 0.153, ...],
    top_k=10,
)
for match in response.matches:
    print(match.id, match.score)
```

Each element of `response.matches` is a {class}`~pinecone.models.ScoredVector` with
`id`, `score`, `values`, `metadata`, and `sparse_values` fields. Results are ordered from
most similar to least similar.

### Include values or metadata in results

By default, `values` and `metadata` are omitted from matches to reduce payload size.
Enable them explicitly:

```python
response = index.query(
    vector=[0.012, -0.087, 0.153, ...],
    top_k=10,
    include_values=True,
    include_metadata=True,
)
for match in response.matches:
    print(match.id, match.score, match.metadata)
```

### Filter by metadata

Pass a `filter` expression to restrict results to vectors whose metadata satisfies the condition:

```python
response = index.query(
    vector=[0.012, -0.087, 0.153, ...],
    top_k=5,
    filter={"genre": {"$eq": "action"}, "year": {"$gte": 2020}},
    include_metadata=True,
)
```

### Using the Field filter builder

{class}`~pinecone.Field` provides a Python-native API for building filter expressions.
The `==`, `!=`, `&`, and `|` operators and `.gt()` / `.gte()` / `.lt()` / `.lte()` /
`.is_in()` / `.not_in()` methods return a {class}`~pinecone.utils.filter_builder.Condition`
object. Pass it to `filter` via `.to_dict()`:

```python
from pinecone import Field

condition = (Field("genre") == "action") & Field("year").gte(2020)

response = index.query(
    vector=[0.012, -0.087, 0.153, ...],
    top_k=5,
    filter=condition.to_dict(),
    include_metadata=True,
)
```

`FilterBuilder` is an alias for `Field` and can be used interchangeably.


## Fetch vectors by ID

{meth}`~pinecone.Index.fetch` retrieves stored vectors by their IDs:

```python
response = index.fetch(ids=["movie-001", "movie-002"])
for vid, vec in response.vectors.items():
    print(vid, vec.values[:3])
```

`response.vectors` is a `dict[str, Vector]`. IDs that do not exist are omitted rather than
raising an error.


## Update a vector

{meth}`~pinecone.Index.update` replaces a vector's dense values, sparse values, or metadata.

Update dense values by ID:

```python
index.update(id="movie-001", values=[0.099, -0.045, 0.210, ...])
```

Update metadata without changing values:

```python
index.update(id="movie-001", set_metadata={"rating": 4.5, "genre": "thriller"})
```

Bulk-update metadata for all vectors matching a filter:

```python
index.update(
    filter={"genre": {"$eq": "drama"}},
    set_metadata={"category": "classic"},
)
```


## Delete vectors

{meth}`~pinecone.Index.delete` removes vectors from a namespace. Specify exactly one of
`ids`, `delete_all`, or `filter`.

Delete by ID:

```python
index.delete(ids=["movie-001", "movie-002"])
```

Delete all vectors in a namespace:

```python
index.delete(delete_all=True, namespace="movies-deprecated")
```

Delete by metadata filter:

```python
index.delete(filter={"year": {"$lte": 2000}})
```


## Inspect index stats

{meth}`~pinecone.Index.describe_index_stats` returns aggregate counts and
per-namespace summaries:

```python
stats = index.describe_index_stats()
print(stats.total_vector_count)
print(stats.dimension)
print(stats.index_fullness)     # fraction 0.0–1.0

for namespace, summary in stats.namespaces.items():
    print(namespace, summary.vector_count)
```

```{note}
These counts always cover the whole index. `describe_index_stats` accepts a
`filter` argument for API compatibility, but a non-empty filter is rejected for
every index type, so a filtered stats call fails rather than returning a subset
count. There is no operation that counts only the vectors matching a metadata
filter.
```


## See also

- {doc}`/how-to/vectors/namespaces`: working with namespaces
- {doc}`/how-to/vectors/bulk-import`: bulk importing from cloud storage
- {class}`~pinecone.Index`: full data plane client reference
- {class}`~pinecone.models.QueryResponse`: query response model
- {class}`~pinecone.models.ScoredVector`: individual match in query results
