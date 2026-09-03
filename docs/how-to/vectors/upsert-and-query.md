# Upserting and querying vectors

Use the {class}`~pinecone.index.Index` client to insert and retrieve vectors from a Pinecone index.
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

{meth}`~pinecone.index.Index.upsert` inserts vectors or overwrites existing ones with the same ID.

Every argument is keyword-only, so the vectors go in as `vectors=`.

:::{note}
Vectors on this page are written as three floats so they fit the page. Pass your
index's full `dimension` — a length mismatch is rejected by the server.
:::

### The three vector shapes

`vectors` accepts a tuple, a dict, or a {class}`~pinecone.models.vectors.vector.Vector`, and the three are
interchangeable within one call:

```python
from pinecone import Vector

response = index.upsert(
    vectors=[
        # (id, values) or (id, values, metadata)
        ("movie-001", [0.012, -0.087, 0.153]),
        ("movie-002", [0.045, 0.021, -0.064], {"genre": "comedy", "year": 2022}),
        # a dict, keys drawn from id / values / sparse_values / metadata
        {"id": "movie-003", "values": [0.091, -0.032, 0.178]},
        {
            "id": "movie-004",
            "values": [0.020, 0.030, 0.040],
            "sparse_values": {"indices": [7, 21], "values": [0.4, 0.6]},
            "metadata": {"genre": "drama"},
        },
        # a Vector object, which your editor can check
        Vector(id="movie-005", values=[0.063, 0.011, -0.022]),
    ]
)
print(response.upserted_count)
```

A dict with a key outside that set is rejected before any request is sent — move
your own fields into `metadata`. Metadata values must be a string, a number, a
boolean, or a list of strings; anything else raises
{exc}`~pinecone.errors.exceptions.PineconeTypeError` locally rather than failing the
whole batch at the server.

The dict form is also the shape `upsert` hands back in `failed_items`, which is why a
retry can pass that list straight back in (see [Handling partial
failures](#handling-partial-failures)).

`upsert` returns an {class}`~pinecone.models.vectors.responses.UpsertResponse`. Without `batch_size` the
client sends one request, so `upserted_count` is the whole answer and every batch
counter reads `0`. With `batch_size` set (see [Large datasets](#large-datasets)) the
counters and `failed_items` describe a partial success.

### Upsert into a namespace

Pass `namespace` to target a specific partition:

```python
index.upsert(
    vectors=[("movie-001", [0.012, -0.087, 0.153])],
    namespace="movies-en",
)
```

Omitting `namespace` writes to the default namespace, which is spelled `""`. Namespaces
are isolated, so a vector written to `"movies-en"` is invisible to a query that does not
name it — see {doc}`/how-to/vectors/namespaces`.

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
`RetryConfig`.

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
defaults to `8` and `show_progress` defaults to `True`.
`total_timeout` bounds the whole batched call in wall-clock
seconds and has no default; `timeout` bounds one attempt of one
batch, so it is not a substitute.

For DataFrame input, {meth}`~pinecone.index.Index.upsert_from_dataframe`
provides the same parallel batching with column extraction.
For millions of vectors, consider
{meth}`~pinecone.index.Index.start_import` to load from cloud storage.

### Handling partial failures

Unlike a single-request upsert (which raises on failure), a
batched upsert never raises for per-batch errors. Instead, the
returned {class}`~pinecone.models.vectors.responses.UpsertResponse` carries each
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
    print(first.error_message, first.retryable, first.disposition)
```

`retryable` is the SDK's own verdict, and cheaper to act on than reading statuses:
`False` marks a deterministic rejection — a validation error, a 4xx — that will fail
identically however many times you send it. Filter on it before any retry loop.

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
(see `RetryConfig`). That layer absorbs nearly
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

### `query` or `search`: pick one

Two search methods exist, and which one your index supports is fixed when the index is
created.

- {meth}`~pinecone.index.Index.query` — **you supply the vector.** Nothing is embedded for
  you. This is the method for a standard index created with a top-level `dimension` and
  `metric`, and it is what the rest of this page uses.
- {meth}`~pinecone.index.Index.search` — **you supply text**, and the index's own
  embedding model turns it into the query vector server-side. This needs an index with
  integrated inference, which is created with
  {meth}`~pinecone.client.indexes.Indexes.create_for_model` and no other way, and
  `search` takes a required `namespace`. See {doc}`/how-to/integrated-records`.

Both accept the same `filter` dict, so nothing in [Filter by
metadata](#filter-by-metadata) below is specific to `query`. If you pass text to
`query`, or a raw vector to an index that has no dense vector field, the server rejects
the call rather than guessing.

### Running a query

`query` returns the `top_k` closest vectors to the vector you give it. `top_k` is
required, and every argument is keyword-only:

```python
response = index.query(
    vector=[0.012, -0.087, 0.153],
    top_k=10,
)
for match in response.matches:
    print(match.id, match.score)
```

Each element of `response.matches` is a `ScoredVector`, ordered
most similar first. A query that matched nothing returns an empty `matches` rather than
raising, so check the length instead of catching an exception.

You can also query by the ID of a vector the index already holds — `index.query(id=...)`
— but an `id` cannot be combined with `vector` or `sparse_vector`, since it is a
reference to stored data rather than a vector of your own.

### Include values or metadata in results

Only `id` and `score` are always populated. `values` and `metadata` are left out of the
response unless you ask for them, which keeps the payload small but is also the single
most common surprise in this SDK:

:::{warning}
Without `include_values=True`, `match.values` is an **empty list**, not the stored
vector. Without `include_metadata=True`, `match.metadata` is **`None`**, so
`match.metadata["genre"]` raises `TypeError` even for a vector that has metadata
stored — and even when you filtered on that very field. An empty `values` or a `None`
`metadata` is far more often an unset flag than an empty record.
:::

```python
response = index.query(
    vector=[0.012, -0.087, 0.153],
    top_k=10,
    include_values=True,
    include_metadata=True,
)
for match in response.matches:
    print(match.id, match.score, match.values[:3], match.metadata)
```

Run the same query both ways once against your own index and the difference is obvious:

```python
without = index.query(vector=[0.012, -0.087, 0.153], top_k=1)
with_meta = index.query(vector=[0.012, -0.087, 0.153], top_k=1, include_metadata=True)

print(without.matches[0].metadata)    # None
print(with_meta.matches[0].metadata)  # the stored fields
```

### Filter by metadata

A `filter` is a plain dict, and writing it by hand is the primary form — no builder or
helper class is needed:

```python
response = index.query(
    vector=[0.012, -0.087, 0.153],
    top_k=5,
    filter={"genre": {"$eq": "action"}, "year": {"$gte": 2020}},
    include_metadata=True,
)
```

The operators are:

| Operator | Takes | Means |
|---|---|---|
| `$eq` / `$ne` | a string, number, or boolean | equal / not equal |
| `$gt` / `$gte` / `$lt` / `$lte` | a number | ordering comparison |
| `$in` / `$nin` | a list | is / is not one of |
| `$exists` | a boolean | the field is present |
| `$and` / `$or` | a list of clauses | combine clauses |

Naming a field once with several operators, as in `{"year": {"$gte": 2020, "$lte":
2024}}`, is an implicit `$and`. The same `filter` argument selects records for
{meth}`~pinecone.index.Index.update` and {meth}`~pinecone.index.Index.delete`.

### Building filters in code

Reach for {class}`~pinecone.utils.filter_builder.Field` when your code *assembles* a filter rather than
writing one out: it gives one method per operator, so your editor checks the operator
names instead of you typing them into a string, and `&` / `|` compose clauses built in
different places. `==` and `!=` build `$eq` and `$ne`; `.gt()` / `.gte()` / `.lt()` /
`.lte()` are numeric only; `.is_in()` / `.not_in()` take a list; `.exists()` takes
nothing. Each returns a {class}`~pinecone.utils.filter_builder.Condition`, and
`.to_dict()` produces the dict to pass as `filter`:

```python
from pinecone import Field

condition = (Field("genre") == "action") & Field("year").gte(2020)
print(condition.to_dict())
# {'$and': [{'genre': {'$eq': 'action'}}, {'year': {'$gte': 2020}}]}

response = index.query(
    vector=[0.012, -0.087, 0.153],
    top_k=5,
    filter=condition.to_dict(),
    include_metadata=True,
)
```

Both routes produce the same filter, so mixing them is fine. Because `==` is overloaded
to build a filter rather than answer a question, a `Field` never compares equal to
anything and cannot be used as a dict key or a set member.


## Fetch vectors by ID

{meth}`~pinecone.index.Index.fetch` retrieves stored vectors by their IDs:

```python
response = index.fetch(ids=["movie-001", "movie-002"])
for vid, vec in response.vectors.items():
    print(vid, vec.values[:3])
```

`response.vectors` is a `dict[str, Vector]`. This is a lookup, not a search: nothing is
ranked and no score comes back. An ID the namespace does not hold is silently absent
from the result rather than raising, so compare the keys you got back against the ones
you asked for.


## Update a vector

{meth}`~pinecone.index.Index.update` patches one vector's dense values, sparse values, or
metadata. It is a partial update: fields you do not name keep the values they had, so
`set_metadata={"year": 2021}` leaves every other metadata key in place. Give exactly one
selector, `id` or `filter`, and an update by `filter` is metadata-only, since values
belong to a single record.

Update dense values by ID:

```python
index.update(id="movie-001", values=[0.099, -0.045, 0.210])
```

Update metadata without changing values:

```python
index.update(id="movie-001", set_metadata={"rating": 4.5, "genre": "thriller"})
```

Bulk-update metadata for every vector matching a filter:

```python
index.update(
    filter={"genre": {"$eq": "drama"}},
    set_metadata={"category": "classic"},
)
```

An update applies asynchronously, so a read straight afterwards can still see the old
value. Pass `dry_run=True` to have the server report how many records a `filter` would
match without changing any of them.


## Delete vectors

{meth}`~pinecone.index.Index.delete` removes vectors from a namespace. Specify exactly one of
`ids`, `delete_all`, or `filter`. Deletes are irreversible, and IDs the namespace does
not hold are ignored rather than reported — a successful call is not evidence anything
was deleted.

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

{meth}`~pinecone.index.Index.describe_index_stats` returns aggregate counts and
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
- {class}`~pinecone.index.Index`: full data plane client reference
- {doc}`/how-to/integrated-records`: `search` on an index with integrated inference
- {doc}`/guides/performance`: choosing a `batch_size` and a `max_concurrency`
- {class}`~pinecone.models.vectors.responses.QueryResponse`: query response model
- `ScoredVector`: individual match in query results
