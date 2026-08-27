# FAQ

### Why is my import slow?

Cold imports of large packages can take tens of milliseconds. The SDK uses lazy imports
so the heavy modules (`httpx`, `msgspec`, `orjson`) load only when you first use them.
The fastest way to initialize is to import just what you need:

```python
from pinecone import Pinecone   # imports only the Pinecone class
```

Avoid wildcard imports (`from pinecone import *`) in performance-sensitive startup paths.

### Why does `pc.indexes.list()` yield only a single page?

Serverless index listings return at most a few hundred entries, which fits comfortably
in a single response. The returned `Paginator` (or `AsyncPaginator`) exists for
interface consistency with other list methods; the server sends everything in one page.

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

`Index` uses the REST/HTTP API. `GrpcIndex` uses gRPC, which has lower per-request
overhead and is better suited to high-throughput bulk operations such as large upsert
batches. For typical read-heavy or mixed workloads, `Index` is simpler to operate.

```python
# REST: general purpose
index = pc.index("my-index")

# gRPC: high-throughput upserts
index = pc.index("my-index", grpc=True)
```

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
