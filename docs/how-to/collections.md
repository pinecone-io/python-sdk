# Working with collections

Collections are read-only snapshots of pod indexes. Create one to preserve a pod index's
data before deleting or reconfiguring it. Restoring a collection into a new index is not
currently supported by the API (see [below](#create-an-index-from-a-collection)).
Collections are only supported for pod-based indexes; serverless indexes use backups
instead.

## Create a collection

Pass the name of the pod index you want to snapshot:

```python
from pinecone import Pinecone

pc = Pinecone(api_key="your-api-key")

collection = pc.collections.create(name="snap-2025-01", source="my-pod-index")
print(collection.status)   # "Initializing" immediately after creation
```

The collection transitions through ``Initializing`` → ``Ready`` when the snapshot is
complete. ``create`` returns immediately without polling; check status with ``describe``.

## List collections

``list`` returns a {class}`~pinecone.models.collections.list.CollectionList` you can
iterate or call ``.names()`` on:

```python
for col in pc.collections.list():
    print(col.name, col.status)
```

```python
names = pc.collections.list().names()
print(names)   # e.g. ["snap-2025-01", "archive-q3"]
```

## Describe a collection

``describe`` returns a {class}`~pinecone.models.collections.model.CollectionModel` with
detailed information:

```python
col = pc.collections.describe("snap-2025-01")
print(col.name)          # "snap-2025-01"
print(col.status)        # "Ready"
print(col.dimension)     # vector dimension
print(col.vector_count)  # number of vectors stored
print(col.size)          # size in bytes
print(col.environment)   # cloud environment
```

Poll until ready after creation:

```python
import time

while True:
    col = pc.collections.describe("snap-2025-01")
    if col.status == "Ready":
        break
    time.sleep(5)
```

## Delete a collection

```python
pc.collections.delete("snap-2025-01")
```

``delete`` raises {exc}`~pinecone.errors.exceptions.NotFoundError` if the collection does not
exist.

## Create an index from a collection

Restoring a collection into a new index is not currently supported by the API.
``pc.indexes.create(source_collection=...)`` raises {exc}`~pinecone.errors.exceptions.PineconeTypeError`,
and passing ``source_collection`` inside a {class}`~pinecone.PodSpec` (via the deprecated
``spec=`` argument) is silently dropped instead of restoring data, so the index comes back
empty. See {doc}`/how-to/indexes/backups-and-restore` for the supported restore path;
it covers serverless and BYOC indexes only, since pod indexes can't be backed up either.
