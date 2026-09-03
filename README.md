# Pinecone Python SDK

The Pinecone Python SDK provides a client for the [Pinecone](https://www.pinecone.io/) vector database. Use it to create and manage indexes, upsert and query vectors, and run inference operations from Python.

Requires Python 3.10+.

## Installation

```bash
pip install pinecone
```

For development dependencies (testing, type checking, linting):

```bash
pip install pinecone[dev]
```

## Quick start

```python
from pinecone import Pinecone, ServerlessSpec

# Initialize the client
pc = Pinecone(api_key="your-api-key")

# Create a serverless index
pc.indexes.create(
    name="movie-recommendations",
    dimension=1536,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)

# Connect to the index
index = pc.index("movie-recommendations")

# Upsert vectors
index.upsert(
    vectors=[
        ("movie-42", [0.012, -0.087, 0.153]),  # 1536-dim embedding
        ("movie-87", [0.045, 0.021, -0.064]),  # 1536-dim embedding
    ],
    namespace="movies-en",
    batch_size=100,  # split larger inputs into parallel batches automatically
)

# Query for similar vectors
results = index.query(
    vector=[0.012, -0.087, 0.153],  # 1536-dim embedding
    top_k=10,
    namespace="movies-en",
)

for match in results.matches:
    print(f"{match.id}: {match.score:.4f}")
```

## Async usage

The SDK provides an async client for use with `asyncio`:

```python
import asyncio
from pinecone import AsyncPinecone

async def main():
    async with AsyncPinecone(api_key="your-api-key") as pc:
        desc = await pc.indexes.describe("movie-recommendations")
        index = await pc.index(host=desc.host)
        async with index:
            results = await index.query(
                vector=[0.012, -0.087, 0.153],  # 1536-dim vector
                top_k=10,
                namespace="movies-en",
            )
            for match in results.matches:
                print(f"{match.id}: {match.score:.4f}")

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

### Setup

Clone the repository and install dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

### Tests

```bash
uv run pytest tests/unit/ -x -v
```

The unit suite must leave the working tree clean: `git status` is expected to report no
changes after a bare `uv run pytest tests/unit`. Tests must not write into the checkout —
use `tmp_path` / `tmp_path_factory` if a test genuinely needs a file on disk. A test that
writes a tracked file both dirties every contributor's tree and, if anything compares
against that file, quietly rewrites the thing it is comparing against.

#### Cross-transport storm parity

`tests/unit/_internal/test_storm_parity.py` checks that the sync, async, and gRPC retry
paths disperse a thundering herd comparably — dispersion widths within 2x and request
amplifications within 1.5x of each other. It runs all three canonical storm scenarios
itself, in-process, via `tests/unit/_internal/_storm_parity_scenarios.py`; there are no
recorded metric files and nothing is a checked-in baseline.

It used to work differently: each transport's storm test wrote a
`_storm_parity_metrics_*.json` file into `tests/unit/_internal/`, and the parity test read
all three back. Those files were tracked, so every unit run dirtied them, and because the
gRPC producer collects *after* the parity consumer, the gRPC comparison always read the
*previous* run's value — a stale-value failure that the same run then overwrote, so it
disappeared on retry. Keep the metrics in-process; don't reintroduce the file handoff.

#### Retry/throttle smoke tests (opt-in)

A suite of live-API smoke tests verifies that the retry stack and AIMD adaptive concurrency
hold up against real Pinecone rate limits. These are **not** run in normal CI because they
require real credentials, create a live serverless index, and take 1–3 minutes per run.

**Required environment variables:**

| Variable | Description |
|---|---|
| `PINECONE_API_KEY` | A valid Pinecone API key |
| `PINECONE_RETRY_SMOKE` | Set to `1` to enable the smoke tests |

**Running the smoke tests:**

```bash
PINECONE_API_KEY=your-api-key PINECONE_RETRY_SMOKE=1 \
  uv run pytest tests/integration/test_retry_smoke.py -x -v -s
```

**Cost:** Each run creates three serverless indexes, upserts ~100K vectors per index, then
deletes all indexes. Total cost is under $3 per run.

**When to run:** Before any release that touches retry logic, HTTP transport, the AIMD
adaptive-concurrency limiter (`pinecone._internal.adaptive`), or the batch-upsert path.
The unit tests mock HTTP responses; this test catches divergence between the synthetic
model and real API behavior (e.g., 503 instead of 429).

### Type checking

```bash
uv run mypy --strict pinecone/
```

### Linting and formatting

```bash
uv run ruff check --fix
uv run ruff format
```

## License

Apache-2.0. See [LICENSE](LICENSE) for details.
