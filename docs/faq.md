# FAQ

### Why is my import slow?

`import pinecone` on its own is nearly free — the top-level package binds its exports
lazily, so nothing underneath it loads until you name something. What costs time is
which names you then pull through it. Import just the ones you use:

```python
from pinecone import Pinecone   # pulls in the client and its serialization layer
```

`from pinecone import *` resolves every export in the package and is the slowest
form by a wide margin. Avoid it in startup paths you care about.

Of the SDK's own dependencies, `httpx` and `orjson` stay unloaded until a request is
actually made; `msgspec` comes in with the client class itself, because the response
models are `msgspec.Struct` subclasses defined at import time.

### Why does `pc.indexes.list()` yield only a single page?

Because the server sends every index in one page today. The returned `Paginator` (or
`AsyncPaginator`) exposes the paginator interface anyway, so a call site written against
it keeps working if that ever changes. Iterate it rather than assuming one page:

```python
names = [index.name for index in pc.indexes.list()]
```

Note that `pc.list_indexes()`, the `9.x`-shaped shim, is not a paginator — it returns an
`IndexList`, which still has `.names()`.

### Can I use the async client with FastAPI?

Yes. Use `AsyncPinecone` as a FastAPI dependency or inside a lifespan context manager:

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pinecone import AsyncPinecone

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with AsyncPinecone(api_key="...") as pc:
        app.state.pc = pc
        yield

app = FastAPI(lifespan=lifespan)
```

`AsyncPinecone` shares one `httpx.AsyncClient` connection pool for the life of the
context, so requests reuse connections instead of opening new ones each time.

### What is the difference between `Index` and `GrpcIndex`?

The transport, and nothing else about how you call them. `Index` speaks REST over
HTTP; `GrpcIndex` speaks gRPC, which pays off on high-throughput bulk upserts. Both
come with the base `pinecone` package, so it's a keyword argument rather than an
install:

```python
# REST: general purpose
index = pc.index("my-index")

# gRPC: high-throughput upserts
index = pc.index("my-index", grpc=True)
```

`GrpcIndex` carries no document operations — the `2026-07` documents API is REST-only.
See [Using the gRPC client](guides/grpc.md) for the full comparison and when the
switch is worth making.

### How do I handle a `ConflictError` when creating an index that already exists?

Catch `ConflictError` from the top-level `pinecone` package:

```python
from pinecone import Pinecone, ConflictError

pc = Pinecone(api_key="...")
try:
    pc.indexes.create(
        name="my-index",
        schema={"fields": {"embedding": {"type": "dense_vector", "dimension": 1536, "metric": "cosine"}}},
        deployment={"deployment_type": "managed", "cloud": "aws", "region": "us-east-1"},
    )
except ConflictError:
    pass  # index already exists, so continue
```

### Can I modify a response object?

Response objects are `msgspec.Struct` instances. They are **not** frozen (i.e., fields
can be reassigned), but mutating them directly is not recommended because subsequent SDK
calls may return new instances that replace the object. If you need a plain, mutable dict,
use the `.to_dict()` method available on most response structs:

```python
idx = pc.indexes.describe("my-index")
d = idx.to_dict()   # returns a plain dict you can modify
```

Alternatively, `msgspec.structs.asdict(idx)` works for any `msgspec.Struct` but does
not recursively convert nested structs the way `.to_dict()` does.
