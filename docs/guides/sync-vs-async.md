# Sync vs Async Clients

The SDK ships two client pairs:

| | Sync | Async |
|---|---|---|
| Control plane | `Pinecone` | `AsyncPinecone` |
| Data plane | `Index` | `AsyncIndex` |
| Transport | httpx (sync) | httpx (async) |
| Context manager | `with Pinecone() as pc:` | `async with AsyncPinecone() as pc:` |

Use the sync client for scripts, CLI tools, and simple integrations. Use the async client
inside async frameworks such as FastAPI, Starlette, or aiohttp, or any code that drives
many concurrent operations.


## Sync Client

`Pinecone` and `Index` are blocking. Each method call completes before returning.

```python
from pinecone import Pinecone

pc = Pinecone()  # reads PINECONE_API_KEY from environment

desc = pc.indexes.describe("product-search")
index = pc.index(host=desc.host)

index.upsert(vectors=[("product-42", [0.1, 0.2, ...])])
results = index.query(vector=[0.1, 0.2, ...], top_k=5)
```

`Pinecone` supports the context manager protocol to ensure the underlying HTTP client
is closed when you are finished:

```python
with Pinecone() as pc:
    results = pc.indexes.list()
```

## Async Client

`AsyncPinecone` and `AsyncIndex` are non-blocking. Every method is a coroutine, and you
must `await` it inside an `async` function.

`AsyncPinecone` is an async context manager. Always use it with `async with` so the
HTTP client is closed properly:

```python
import asyncio
from pinecone import AsyncPinecone

async def main() -> None:
    async with AsyncPinecone() as pc:
        desc = await pc.indexes.describe("product-search")
        index = await pc.index(host=desc.host)

        await index.upsert(vectors=[("product-42", [0.1, 0.2, ...])])
        results = await index.query(vector=[0.1, 0.2, ...], top_k=5)

asyncio.run(main())
```

`pc.index()` on `AsyncPinecone` is itself a coroutine. `await` it to get back an
`AsyncIndex`. The returned `AsyncIndex` manages its own HTTP session; close it
with `async with index:` or `await index.close()`.

```python
async with AsyncPinecone() as pc:
    # Preferred: pass the host directly to skip a describe call
    index = await pc.index(host="product-search-abc123.svc.pinecone.io")
    async with index:
        results = await index.query(vector=[0.1, 0.2, ...], top_k=5)
```


## Same Operation, Two Styles

The example below shows the same upsert-and-query flow in both styles.

::::{tabs}
:::{tab} Sync
```python
from pinecone import Pinecone

pc = Pinecone()

with pc.index(name="movie-recommendations") as index:
    index.upsert(vectors=[
        ("movie-42", [0.012, -0.087, 0.153, ...]),
        ("movie-99", [0.045,  0.021, -0.064, ...]),
    ])
    results = index.query(
        vector=[0.012, -0.087, 0.153, ...],
        top_k=5,
        filter={"genre": "comedy"},
    )

for match in results.matches:
    print(match.id, match.score)
```
:::
:::{tab} Async
```python
import asyncio
from pinecone import AsyncPinecone

async def main() -> None:
    async with AsyncPinecone() as pc:
        # Resolve the host first, then create the index client
        desc = await pc.indexes.describe("movie-recommendations")
        index = await pc.index(host=desc.host)
        async with index:
            await index.upsert(vectors=[
                ("movie-42", [0.012, -0.087, 0.153, ...]),
                ("movie-99", [0.045,  0.021, -0.064, ...]),
            ])
            results = await index.query(
                vector=[0.012, -0.087, 0.153, ...],
                top_k=5,
                filter={"genre": "comedy"},
            )

    for match in results.matches:
        print(match.id, match.score)

asyncio.run(main())
```
:::
::::


## When to Use gRPC

Pass `grpc=True` to `pc.index()` to use a `GrpcIndex` instead of the default HTTP
`Index`. gRPC uses HTTP/2 multiplexing and binary serialization, which can improve
throughput for bulk upsert workloads.

```python
from pinecone import Pinecone

pc = Pinecone()
index = pc.index(name="product-search", grpc=True)

index.upsert(vectors=[("product-42", [0.1, 0.2, ...])])
```

`GrpcIndex` is only available on the sync client. For high-throughput async workloads,
use `AsyncIndex` with concurrent tasks instead:

```python
import asyncio
from pinecone import AsyncPinecone

async def upsert_batch(pc: AsyncPinecone, batch: list[tuple[str, list[float]]]) -> None:
    index = await pc.index(host="product-search-abc123.svc.pinecone.io")
    async with index:
        await index.upsert(vectors=batch)

async def main() -> None:
    batches = [...]  # split your vectors into chunks
    async with AsyncPinecone() as pc:
        await asyncio.gather(*[upsert_batch(pc, b) for b in batches])

asyncio.run(main())
```


## Connection Management

Both clients use [httpx](https://www.python-httpx.org/) under the hood:

- The sync `Pinecone` and `Index` each manage a synchronous `httpx.Client`.
- `AsyncPinecone` and `AsyncIndex` each manage an `httpx.AsyncClient`.

Close the client when you are done to release connections. The context manager protocol
handles this automatically:

```python
# Explicit close
pc = Pinecone()
# ... use pc ...
pc.close()

# Preferred: context manager
with Pinecone() as pc:
    # ... use pc ...
    pass  # close() called on exit
```

The `AsyncIndex` manages its own HTTP session independently of `AsyncPinecone`. Closing
`AsyncPinecone` does **not** close any `AsyncIndex` objects you created from it. Close
each `AsyncIndex` separately:

```python
async with AsyncPinecone() as pc:
    index = await pc.index(host="...")
    async with index:
        await index.query(...)
    # index is closed here; pc is closed at the outer block exit
```
