# Quickstart

Get from install to your first similarity search in five minutes.

## 1. Initialize the client

```python
from pinecone import Pinecone

# Option A: read API key from the PINECONE_API_KEY environment variable
pc = Pinecone()

# Option B: pass it explicitly
pc = Pinecone(api_key="your-api-key")
```

## 2. Create a serverless index

An index's fields are declared as a schema. This one has a single dense vector field,
`embedding`:

```python
pc.indexes.create(
    name="quickstart",
    schema={"fields": {"embedding": {"type": "dense_vector", "dimension": 3, "metric": "cosine"}}},
    deployment={"deployment_type": "managed", "cloud": "aws", "region": "us-east-1"},
)
```

## 3. Wait for the index to be ready

`create` polls until the index is ready by default. If you passed `timeout=-1` to
return immediately, check readiness yourself:

```python
import time

while True:
    desc = pc.indexes.describe("quickstart")
    if desc.status.ready:
        break
    time.sleep(1)
```

## 4. Get an Index client

```python
index = pc.index("quickstart")
```

## 5. Upsert vectors

```python
index.upsert(vectors=[
    ("id1", [0.1, 0.2, 0.3]),
    ("id2", [0.4, 0.5, 0.6]),
    ("id3", [0.7, 0.8, 0.9]),
])
```

## 6. Query

```python
results = index.query(vector=[0.1, 0.2, 0.3], top_k=3)
for match in results.matches:
    print(match.id, match.score)
```

## 7. Clean up

```python
pc.indexes.delete("quickstart")
```

## Complete example

```python
from pinecone import Pinecone

pc = Pinecone()  # reads PINECONE_API_KEY

pc.indexes.create(
    name="quickstart",
    schema={"fields": {"embedding": {"type": "dense_vector", "dimension": 3, "metric": "cosine"}}},
    deployment={"deployment_type": "managed", "cloud": "aws", "region": "us-east-1"},
)

index = pc.index("quickstart")

index.upsert(vectors=[
    ("id1", [0.1, 0.2, 0.3]),
    ("id2", [0.4, 0.5, 0.6]),
    ("id3", [0.7, 0.8, 0.9]),
])

results = index.query(vector=[0.1, 0.2, 0.3], top_k=3)
for match in results.matches:
    print(match.id, match.score)

pc.indexes.delete("quickstart")
```
