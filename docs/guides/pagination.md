# Pagination

Listings that can grow without bound come back a page at a time. Which page you
get is decided by an opaque **pagination token**, and the SDK offers two ways to
work with it: a paginator that follows the token for you, or a single-page call
that hands you the token to drive yourself.

## The shapes

Every paged listing in the SDK is one of these.

| Shape | Returns | You do |
|---|---|---|
| **Auto-paginating** | `Paginator[T]` / `AsyncPaginator[T]` | Iterate. Pages are fetched lazily as you cross them. |
| **Single page** | A response object with a `.pagination` envelope | Read the page, then pass `.pagination.next` back as `pagination_token`. |
| **Unpaged** | A plain list-like model with no token | Iterate once. There is nothing to follow. |

Several listings offer both of the first two, under names that differ by
namespace — `iter_*` here, `*_page` there, `*_paginated` on the index:

| Listing | Auto-paginating | Single page |
|---|---|---|
| Indexes | `pc.indexes.list()` | — (one page always) |
| Backups of one index | `pc.indexes.list_backups(name)` | — |
| Backups, project-wide | — | `pc.backups.list()` |
| Restore jobs | — | `pc.restore_jobs.list()` |
| Backup schedules | `pc.backup_schedules.iter_schedules(index_name=…)` | `pc.backup_schedules.list(index_name=…)` |
| Backup schedule history | `pc.backup_schedules.iter_history(schedule_id=…)` | `pc.backup_schedules.history(schedule_id=…)` |
| Assistants | `pc.assistants.list()` | `pc.assistants.list_page()` |
| Assistant files | `pc.assistants.list_files(…)` | `pc.assistants.list_files_page(…)` |
| Assistant operations | `pc.assistants.list_operations(…)` | `pc.assistants.list_operations_page(…)` |
| Documents | `idx.documents.list(namespace=…)` | — |
| Vector IDs | `idx.list()` | `idx.list_paginated()` |
| Namespaces | `idx.list_namespaces()` | `idx.list_namespaces_paginated()` |
| Bulk imports | `idx.list_imports()` | `idx.list_imports_paginated()` |
| Admin users, invites, service accounts, role bindings | `admin.<ns>.list()` | — |

Unpaged listings — `pc.collections.list()`, `pc.inference.list_models()`,
`admin.api_keys.list()`, `admin.organizations.list()`, `admin.projects.list()` —
return every row in one response and carry no `pagination` attribute to check.

Reach for the single-page form when you need the token itself: to checkpoint a
long walk to durable storage, to hand the next page to a different worker, or to
resume after a crash. Otherwise iterate the paginator.

One paged surface is not a listing at all: `idx.documents.fetch(filter=...)` is
a read whose filter can match unboundedly many documents, so it pages too. It
takes a token but no page size — see
[When the first call takes no token](#when-the-first-call-takes-no-token).

## Paginator and AsyncPaginator

{class}`Paginator <pinecone.models.pagination.Paginator>` and
{class}`AsyncPaginator <pinecone.models.pagination.AsyncPaginator>` are lazy: no
request is made until you start iterating, and each page is fetched only when
iteration reaches it. They share one interface.

| Member | What it gives you |
|---|---|
| `__iter__` / `__aiter__` | Individual items, pages hidden |
| `.pages()` | {class}`Page <pinecone.models.pagination.Page>` objects, one per request |
| `.to_list()` | Every item, in one list |
| `.pagination_token` | Where iteration got to; `None` once the last page has been fetched |

A paginator is not a list and not a snapshot. It has no `len()`, and iterating it
twice sends the requests twice.

### Iterating items

The common case. The loop crosses page boundaries without you noticing.

::::{tabs}
:::{tab} Sync
```python
from pinecone import Pinecone

pc = Pinecone()

for backup in pc.indexes.list_backups("product-search"):
    print(backup.backup_id, backup.status)
```
:::
:::{tab} Async
```python
import asyncio
from pinecone import AsyncPinecone

async def main() -> None:
    async with AsyncPinecone() as pc:
        async for backup in pc.indexes.list_backups("product-search"):
            print(backup.backup_id, backup.status)

asyncio.run(main())
```
:::
::::

### Iterating pages

Use `.pages()` when you want the token each page carries, or when a page is the
natural unit of work — one batch write per page, one checkpoint per page. The
generator stops on its own when a page arrives without a token; you do not need
to break on `has_more`.

::::{tabs}
:::{tab} Sync
```python
from pinecone import Pinecone

pc = Pinecone()

for page in pc.backup_schedules.iter_history(schedule_id="sched-abc123").pages():
    for run in page.items:
        print(run.backup_id, run.status)
    print("next page:", page.pagination_token)  # None on the last page
```
:::
:::{tab} Async
```python
from pinecone import AsyncPinecone

async with AsyncPinecone() as pc:
    history = pc.backup_schedules.iter_history(schedule_id="sched-abc123")
    async for page in history.pages():
        for run in page.items:
            print(run.backup_id, run.status)
        print("next page:", page.pagination_token)
```
:::
::::

Each {class}`Page <pinecone.models.pagination.Page>` carries:

| Attribute | Type | Meaning |
|---|---|---|
| `items` | `list[T]` | The rows on this page, in server order |
| `pagination_token` | `str \| None` | The page *after* this one; `None` on the last page |
| `has_more` | `bool` | `pagination_token is not None` |

### Collecting everything

`to_list()` walks every page and returns one list. Only reach for it when you
know the listing is small — it holds the whole result in memory and cannot start
work before the last page arrives.

::::{tabs}
:::{tab} Sync
```python
from pinecone import Pinecone

pc = Pinecone()

backups = pc.indexes.list_backups("product-search").to_list()
print(len(backups))
```
:::
:::{tab} Async
```python
from pinecone import AsyncPinecone

async with AsyncPinecone() as pc:
    backups = await pc.indexes.list_backups("product-search").to_list()
    print(len(backups))
```
:::
::::

### Resuming

`Page.pagination_token` and `Paginator.pagination_token` are both accepted back
as the `pagination_token` argument of the same list method. Store one, and a
later process picks up where the earlier one stopped.

::::{tabs}
:::{tab} Sync
```python
from pinecone import Pinecone

pc = Pinecone()

pages = pc.backup_schedules.iter_history(schedule_id="sched-abc123").pages()
first = next(pages)
for run in first.items:
    print(run.backup_id)

token = first.pagination_token  # save this somewhere durable

# Later, in another process:
for run in pc.backup_schedules.iter_history(
    schedule_id="sched-abc123", pagination_token=token
):
    print(run.backup_id)
```
:::
:::{tab} Async
```python
from pinecone import AsyncPinecone

async with AsyncPinecone() as pc:
    pages = pc.backup_schedules.iter_history(schedule_id="sched-abc123").pages()
    first = await anext(pages)
    for run in first.items:
        print(run.backup_id)

    token = first.pagination_token

    async for run in pc.backup_schedules.iter_history(
        schedule_id="sched-abc123", pagination_token=token
    ):
        print(run.backup_id)
```
:::
::::

Tokens are opaque. They are not IDs, offsets, or timestamps; do not parse one,
build one, or edit one. A token also encodes the page size it was minted with,
which is why methods that accept both send only the token when you supply one.

Paging walks a live result set rather than a fixed snapshot, so rows created or
deleted between requests can shift later pages. De-duplicate on the row's own
identifier rather than relying on page order.

## What `limit` means

`limit` is the page size the SDK asks the server for. On *some* paginators it
also caps the total number of items iteration will yield:

- **Page size and total cap:** `pc.indexes.list`, `pc.indexes.list_backups`,
  `pc.backup_schedules.iter_schedules`, `pc.backup_schedules.iter_history`,
  `pc.assistants.list`, `pc.assistants.list_files`,
  `pc.assistants.list_operations`.
- **Page size only** — the paginator still walks every page:
  `idx.documents.list`, `admin.users.list`, `admin.invites.list`,
  `admin.service_accounts.list`, `admin.role_bindings.list`, and the
  `Index` listings (`idx.list`, `idx.list_namespaces`, `idx.list_imports`).

Check the method's own `limit` documentation before relying on it to stop
iteration. To stop early on any listing, `break` out of the loop or wrap it in
{func}`itertools.islice` — that works everywhere and never depends on which
group the method falls into.

```python
# Stops after 50 items on any listing, regardless of limit semantics.
import itertools

from pinecone import Pinecone

pc = Pinecone()

for backup in itertools.islice(pc.indexes.list_backups("product-search"), 50):
    print(backup.backup_id)
```

When a `limit` truncates a page, that page reports `pagination_token=None` even
though the server had more. Resume from `Paginator.pagination_token` instead,
which still holds the server's token.

## Driving the token yourself

Single-page methods return the page plus a `pagination` envelope. `pagination is
None`, or `pagination.next is None`, both mean the page you are holding is the
last one — that is the loop's exit condition, not an error.

The loop shape below consumes each page before asking for the next one, so it is
correct on both the first and the last iteration:

```python
from pinecone import Pinecone

pc = Pinecone()

page = pc.backups.list(limit=100)
backups = list(page)

while page.pagination is not None and page.pagination.next is not None:
    page = pc.backups.list(pagination_token=page.pagination.next)
    backups.extend(page)

print(len(backups))
```

Do not send `limit` alongside `pagination_token`: the token already carries the
page size it was minted with, and a different one beside it can skip or repeat
rows. The example above passes `limit` on the first request only.

The async form is the same loop with `await` on each request:

```python
from pinecone import AsyncPinecone

async with AsyncPinecone() as pc:
    page = await pc.backups.list(limit=100)
    backups = list(page)

    while page.pagination is not None and page.pagination.next is not None:
        page = await pc.backups.list(pagination_token=page.pagination.next)
        backups.extend(page)

    print(len(backups))
```

### When the first call takes no token

Not every paged surface is a listing. `idx.documents.fetch(filter=...)` is a
paged *read*: the filter can match an unbounded number of documents, so the
response carries a `pagination` envelope, but there is no page-size argument and
no first-page call that differs from the rest. Write it as a `while True` with
the token starting at `None`, and read each page's documents before asking for
the next one:

```python
from pinecone import Pinecone

idx = Pinecone().index(name="articles-en")

pagination_token = None
while True:
    response = idx.documents.fetch(
        namespace="published",
        filter={"category": {"$eq": "tech"}},
        pagination_token=pagination_token,
    )
    for doc_id, doc in response.documents.items():
        print(doc_id, doc.title)
    if response.pagination is None:
        break
    pagination_token = response.pagination.next
```

Fetching by `ids=` instead of `filter=` is not paged at all — you asked for a
bounded set, and you get it in one response. Only the filtered form pages, and
`pagination_token` is rejected without a `filter`.

## Listings on the index

The data-plane listings on `Index` predate `Paginator` and keep their own shape.
The auto-following form is a plain generator, not a paginator, so `.pages()`,
`.to_list()`, and `.pagination_token` are not available on it.

`idx.list()` and `idx.list_namespaces()` yield **one response object per page**,
not one row at a time — the loop body has to iterate the page as well:

```python
from pinecone import Pinecone

pc = Pinecone()
idx = pc.index(name="product-search")

for page in idx.list(prefix="product#", limit=100):
    for item in page.vectors:
        print(item.id)

for page in idx.list_namespaces(prefix="prod-"):
    for ns in page.namespaces:
        print(ns.name, ns.record_count, ns.size_bytes)
```

`idx.list_imports()` is the exception: it yields the import records themselves.

```python
from pinecone import Pinecone

idx = Pinecone().index(name="product-search")

for imp in idx.list_imports():
    print(imp.id, imp.status)
```

Each has a `*_paginated` twin that fetches exactly one page and leaves the token
to you:

```python
from pinecone import Pinecone

idx = Pinecone().index(name="product-search")

page = idx.list_paginated(prefix="product#", limit=100)
ids = [item.id for item in page.vectors]

while page.pagination is not None and page.pagination.next is not None:
    page = idx.list_paginated(
        prefix="product#",
        pagination_token=page.pagination.next,
    )
    ids.extend(item.id for item in page.vectors)

print(len(ids))
```

`prefix` and `limit` on these methods are validated by the SDK before any
request goes out, so an out-of-range `limit` or a non-ASCII `prefix` raises
{exc}`~pinecone.errors.exceptions.PineconeValueError` rather than reaching the
server. `idx.list_namespaces()` is also the operation to reach for over repeated
`describe_namespace()` calls: it describes every namespace in one request per
page, and `describe_namespace` is rate limited per index.

## See also

- {doc}`/guides/error-handling` — what a failed page request raises.
- {doc}`/guides/sync-vs-async` — choosing between `Pinecone` and `AsyncPinecone`.
- {doc}`/guides/retries` — a page request retries like any other request.
