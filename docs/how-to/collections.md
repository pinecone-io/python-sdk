# Working with collections

Collections are read-only snapshots of pod indexes. Collections are only supported for
pod-based indexes; serverless indexes use backups instead. Restoring a collection into a new
index is not currently supported by the API (see
[below](#create-an-index-from-a-collection)).

````{important}
API version `2026-07`, the version this SDK targets, does not create pod-based indexes:

```
[400 INVALID_ARGUMENT] deployment_type 'pod' is not supported on this API
version. Set deployment_type to 'managed' to create a serverless index, or
set the X-Pinecone-API-Version header to an earlier version.
```

A collection is a snapshot of a pod index, so with no pod index to point at, there is no
source `pc.collections.create()` can accept here. Everything else on this page still works:
the `/collections` routes are served, `pc.collections` and `AsyncPinecone.collections` are
still on the client, and `list()`, `describe()`, and `delete()` operate normally on
collections that already exist. See {ref}`pod-collections` in the v10 migration guide.
````

## Create a collection

`create()` takes the name of a pod index to snapshot, and `2026-07` has no reachable source
to give it: a pod index cannot be created, and a serverless source is rejected with a 400.

For snapshot and restore on managed (serverless) indexes, use backups instead — see
{doc}`/how-to/indexes/backups-and-restore`, and `pc.create_index_from_backup(...)` for the
restore path.

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

This is the way to inventory the collections a project already holds, including before an
``admin.projects.delete()``, which a leftover collection blocks with a 412.

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
