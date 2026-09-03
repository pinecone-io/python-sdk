# Pinecone Python SDK

The Pinecone Python SDK provides a client for the [Pinecone](https://www.pinecone.io/) vector database. Use it to create and manage indexes, upsert and query records, and run inference operations from Python.

Requires Python 3.10+.

Upgrading from 9.x? `create` and `configure` moved from `spec=`/`dimension=` to `schema=`/`deployment=`; see the [v10 migration guide](https://sdk.pinecone.io/python/migration/v10-migration.html) for the field-by-field mapping.

## Installation

```bash
pip install pinecone
```

## Quick start

An index declares its fields as a schema. Declaring a schema makes it a document
index: you read and write it through `index.documents`, and each record is a JSON
document whose fields you named yourself.

```python
from pinecone import DenseVectorQuery, Pinecone

pc = Pinecone(api_key="your-api-key")  # or omit and set PINECONE_API_KEY

# Create an index. This blocks until the index is ready.
pc.indexes.create(
    name="movie-recommendations",
    schema={"fields": {"embedding": {"type": "dense_vector", "dimension": 3, "metric": "cosine"}}},
    deployment={"deployment_type": "managed", "cloud": "aws", "region": "us-east-1"},
)

# Get a data-plane handle for that index
index = pc.index("movie-recommendations")

# Upsert documents. Each one needs an `_id`; every other key is either a field
# you declared in the schema or arbitrary metadata.
index.documents.upsert(
    namespace="movies-en",
    documents=[
        {"_id": "movie-001", "embedding": [0.1, 0.2, 0.3], "title": "Arrival"},
        {"_id": "movie-002", "embedding": [0.4, 0.5, 0.6], "title": "Interstellar"},
    ],
)

# Search. `score_by` names the field to compare against.
results = index.documents.search(
    namespace="movies-en",
    top_k=5,
    score_by=[DenseVectorQuery(field="embedding", values=[0.1, 0.2, 0.3])],
    include_fields=["title"],
)
for doc in results.matches:
    print(doc.id, doc.score)
```

Upserts apply asynchronously, so a document may not be visible to the next search
immediately.

Two other data-plane interfaces exist, and the way the index was created decides
which one applies: an index created with the deprecated top-level vector
arguments answers on `index.upsert` / `index.query`, and one created with
`pc.indexes.create_for_model(...)` embeds text server-side and answers on
`index.upsert_records` / `index.search`. See the
[quickstart](https://sdk.pinecone.io/python/getting-started/quickstart.html) and
the rest of the [documentation](https://sdk.pinecone.io/python/) for the full
picture.

## Async usage

The SDK provides an async client for use with `asyncio`. Its `index()` is a
coroutine, and the handle it returns is a context manager:

```python
import asyncio

from pinecone import AsyncPinecone, DenseVectorQuery


async def main():
    async with AsyncPinecone(api_key="your-api-key") as pc:
        index = await pc.index("movie-recommendations")
        async with index:
            results = await index.documents.search(
                namespace="movies-en",
                top_k=5,
                score_by=[DenseVectorQuery(field="embedding", values=[0.1, 0.2, 0.3])],
                include_fields=["title"],
            )
            for doc in results.matches:
                print(doc.id, doc.score)


asyncio.run(main())
```

## Configuration

### API key

Pass the API key directly or set the `PINECONE_API_KEY` environment variable:

```python
from pinecone import Pinecone

# Explicit API key
pc = Pinecone(api_key="your-api-key")

# From environment variable (PINECONE_API_KEY)
pc = Pinecone()
```

### Custom host

Connect to a specific control plane host:

```python
pc = Pinecone(api_key="your-api-key", host="https://api.pinecone.io")
```

### Timeout

Configure request timeouts in seconds:

```python
pc = Pinecone(api_key="your-api-key", timeout=30)
```

### Debug logging

Enable debug logging by setting the `PINECONE_DEBUG` environment variable:

```bash
export PINECONE_DEBUG=1
```

## Development

Clone the repository and install the dev dependency group with
[uv](https://docs.astral.sh/uv/), which is what CI does:

```bash
uv sync --group dev
```

```bash
uv run pytest tests/unit/ -x -v      # tests
uv run mypy --strict pinecone/       # type checking
uv run ruff check --fix              # linting
uv run ruff format                   # formatting
```

The unit suite must leave the working tree clean: `git status` is expected to report
no changes after a bare `uv run pytest tests/unit`. Use `tmp_path` /
`tmp_path_factory` if a test genuinely needs a file on disk.

Some suites are opt-in because they hit a real backend and cost money. The live
retry/throttle smoke tests in `tests/integration/test_retry_smoke.py` need
`PINECONE_API_KEY` plus `PINECONE_RETRY_SMOKE=1`; run them before any release that
touches retry logic, HTTP transport, the AIMD adaptive-concurrency limiter, or the
batch-upsert path. Each module documents its own gate and cost.

## License

Apache-2.0. See [LICENSE](LICENSE) for details.
