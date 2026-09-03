# Sync vs Async Clients

The SDK ships two client pairs over the same API, plus a sync-only gRPC data-plane client.

| | Sync | Async |
|---|---|---|
| Control plane | `Pinecone` | `AsyncPinecone` |
| Data plane | `Index` (HTTP), `GrpcIndex` (gRPC) | `AsyncIndex` (HTTP) |
| Transport | httpx `Client` | httpx `AsyncClient` |
| Getting a data-plane client | `pc.index(...)` | `await pc.index(...)` |
| Closing | `with ... as x:` / `x.close()` | `async with ... as x:` / `await x.close()` |


## Which one to use

Use the async client when your code is already inside an `async def` — most often because
you are writing handlers for an async web framework such as FastAPI, Starlette, or
Litestar. There, a blocking call either stalls the event loop for every concurrent request
or forces you to offload it to a thread; `await` avoids both.

The sync client is the right default everywhere else: scripts, CLI tools, notebooks, batch
jobs, and any application that is not already running an event loop. It is not the slower
choice — for bulk upsert the two land within a second of each other on the reference
benchmark. See [Async Concurrency](performance.md#async-concurrency) for the numbers.

There is no async gRPC client. `GrpcIndex` is sync-only, so async code uses `AsyncIndex`
over HTTP. See [Using the gRPC Client](grpc.md).


## The one shape difference

The two clients carry the same methods with the same names and arguments. The one
difference in shape is `index()`.

`Pinecone.index()` is a plain call. `AsyncPinecone.index()` is a coroutine and must be
awaited. Both resolve a host the same way: an explicit `host` is used as-is, a name is
served from the client's host cache, and a name that misses the cache costs one describe
request. Awaiting is what makes that request non-blocking.

::::{tabs}
:::{tab} Sync
```python
from pinecone import Pinecone

pc = Pinecone()  # reads PINECONE_API_KEY from the environment

with pc.index(name="product-search") as index:
    print(index.describe_index_stats().total_vector_count)
```
:::
:::{tab} Async
```python
import asyncio
from pinecone import AsyncPinecone


async def main() -> None:
    async with AsyncPinecone() as pc:
        index = await pc.index(name="product-search")
        async with index:
            stats = await index.describe_index_stats()
            print(stats.total_vector_count)


asyncio.run(main())
```
:::
::::

Pass `host=` when you already have it and the lookup is skipped entirely:

```python
import asyncio
from pinecone import AsyncPinecone


async def main() -> None:
    async with AsyncPinecone() as pc:
        desc = await pc.indexes.describe("product-search")
        index = await pc.index(host=desc.host)
        async with index:
            results = await index.query(vector=[0.12, 0.34, 0.56], top_k=5)
            for match in results.matches:
                print(match.id, match.score)


asyncio.run(main())
```

Resolving a name can fail on either lane: an index that is still initializing has no host
assigned yet, and `index(name=...)` raises `PineconeValueError` rather than handing back a
client that cannot connect. Wait for the index status to be `Ready`.

`AsyncPinecone.index()` also takes no `grpc=` argument, gRPC being sync-only, and no
`pool_threads=`, which exists on the sync client only to support the legacy
`async_req=True` execution model.


## Not every async method is a coroutine

Awaiting everything is a common wrong guess. The list-shaped methods return a paginator
rather than a coroutine, so you iterate them with `async for` instead of awaiting them:

```python
import asyncio
from pinecone import AsyncPinecone


async def main() -> None:
    async with AsyncPinecone() as pc:
        # Right: list() hands back an AsyncPaginator
        async for index in pc.indexes.list():
            print(index.name, index.status.state)

        # Also right: awaiting a method that really is a coroutine
        desc = await pc.indexes.describe("product-search")
        print(desc.host)


asyncio.run(main())
```

`await pc.indexes.list()` raises `TypeError`. The same holds for `pc.backups.list()`,
`pc.indexes.list_backups()`, `index.documents.list()`, and the other paginator-returning
methods. See [Pagination](pagination.md).


## Closing clients

Every client here — `Pinecone`, `Index`, `GrpcIndex`, `AsyncPinecone`, `AsyncIndex` —
holds its own connection pool, and every one is a context manager.

**Entering the control-plane client is not enough.** A data-plane client returned by
`pc.index()` is independent: closing the client it came from does not close it. This is
true on both lanes, and it is the mistake most worth avoiding — an index client that is
never closed holds its connections open for the life of the process.

::::{tabs}
:::{tab} Sync
```python
from pinecone import Pinecone

with Pinecone() as pc:
    with pc.index(name="product-search") as index:
        print(index.describe_index_stats().total_vector_count)
    # index closed here; pc closed at the outer exit
```
:::
:::{tab} Async
```python
import asyncio
from pinecone import AsyncPinecone


async def main() -> None:
    async with AsyncPinecone() as pc:
        index = await pc.index(name="product-search")
        async with index:
            stats = await index.describe_index_stats()
            print(stats.total_vector_count)
        # index closed here; pc closed at the outer exit


asyncio.run(main())
```
:::
::::

The two async steps can be collapsed into one line —
`async with await pc.index(name="product-search") as index:` is valid — but the two-step
form above reads better.

When a client has to outlive a single block, close it yourself in a `finally`:

```python
import asyncio
from pinecone import AsyncPinecone


async def main() -> None:
    pc = AsyncPinecone()
    try:
        print(await pc.indexes.exists("product-search"))
    finally:
        await pc.close()


asyncio.run(main())
```

A client carries your API key, resolved host, and connection pool, so build one and reuse
it rather than constructing one per call. See
[Connection Pooling](performance.md#connection-pooling).


## Other differences

- `proxy_headers=` is not supported on `AsyncPinecone`; a non-empty mapping raises
  `NotImplementedError` at construction. Use `Pinecone` when your proxy needs headers of
  its own.
- `ssl_ca_certs` pointing at a path that does not exist raises `FileNotFoundError` when
  `Pinecone` is constructed, but not until the first request on `AsyncPinecone`, whose
  connection pool is built lazily.
- Retry behavior is the same on both lanes. See [Retries](retries.md).


## See also

- [How Pinecone Works](concepts.md) — the control-plane/data-plane split and the rest of
  the vocabulary used above.
- {doc}`AsyncPinecone reference </reference/async-pinecone>` and
  {doc}`AsyncIndex reference </reference/async-index>`.
- [Performance](performance.md) — throughput measurements for both lanes.
