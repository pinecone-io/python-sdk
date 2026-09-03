# Working with collections

A collection is a read-only, static snapshot of a **pod-based** index's vector data, held
outside the index. Collections and backups are two different mechanisms for two different
deployment types, and they are not interchangeable:

| | Snapshot mechanism | Taken with | Restored with |
|---|---|---|---|
| Pod-based index | Collection | `pc.collections.create()` | nothing — see [below](#create-an-index-from-a-collection) |
| Managed (serverless) or BYOC index | Backup | `pc.backups.create()` | `pc.create_index_from_backup()` |

For the backup side, see [backups and restore](indexes/backups-and-restore.md).

```{important}
API version `2026-07`, the version this SDK targets, does not create pod-based indexes, so
there is no source `pc.collections.create()` can point at. Everything else on this page
still works: the `/collections` routes are served, `pc.collections` and
`AsyncPinecone.collections` are still on the client, and `list()`, `describe()`, and
`delete()` operate normally on collections that already exist. See
[pod-based indexes](indexes/pod.md) for the refusal itself, and {ref}`pod-collections` in
the v10 migration guide for the API-version pin that reaches an earlier version.
```

## Create a collection

`create()` takes a name for the snapshot and the name of the pod index to copy:

```python
col = pc.collections.create(name="movie-embeddings-snapshot", source="movie-recommendations")
print(col.status)   # "Initializing"
```

The source has to be a pod index that is already ready; a managed source is rejected. The
call returns as soon as creation starts, and there is no `timeout=` argument to wait on —
poll `describe` until the status leaves `"Initializing"`.

Because `2026-07` creates no pod index, this call has no reachable source there. For
snapshot and restore on managed indexes, use backups instead — see
[backups and restore](indexes/backups-and-restore.md), and `pc.create_index_from_backup(...)`
for the restore path.

## List collections

`list` returns a {class}`~pinecone.models.collections.list.CollectionList` you can
iterate or call `.names()` on. There is no filtering, sorting, or pagination — every
collection in the project comes back at once:

```python
for col in pc.collections.list():
    print(col.name, col.status)
```

```python
names = pc.collections.list().names()
print(names)   # ['movie-embeddings-snapshot', 'product-catalog-snapshot']
```

This is the way to inventory the collections a project already holds, including before an
`admin.projects.delete()`, which a leftover collection blocks with
{exc}`~pinecone.errors.exceptions.FailedPreconditionError` naming what is still there.

## Describe a collection

`describe` returns a {class}`~pinecone.models.collections.model.CollectionModel`:

```python
col = pc.collections.describe("movie-embeddings-snapshot")
print(col.name)          # 'movie-embeddings-snapshot'
print(col.status)        # 'Ready'
print(col.dimension)     # 1024
print(col.vector_count)  # 99
print(col.size)          # 3126700
print(col.environment)   # 'us-east1-gcp'
```

`size` is how much space the snapshot occupies **in bytes**, not a vector count — that is
`vector_count`. `size`, `dimension`, and `vector_count` are all `None` until the
collection finishes initializing, so read `status` before trusting them.

## Delete a collection

```python
pc.collections.delete("movie-embeddings-snapshot")
```

Deletion is asynchronous: the call returns as soon as the request is accepted, and the
collection can still appear in `list` for a short time afterwards. The source index cannot
be deleted until the collection is really gone.

`delete` raises {exc}`~pinecone.errors.exceptions.NotFoundError` if the collection does not
exist.

## Create an index from a collection

There is no path from a collection back to an index. `2026-07` rejects index creation from
a collection, and the SDK refuses both spellings client-side rather than sending a call it
knows will fail:

- `pc.indexes.create(source_collection=...)` raises
  {exc}`~pinecone.errors.exceptions.PineconeTypeError`.
- `source_collection` set on a {class}`~pinecone.models.indexes.specs.PodSpec` passed to the deprecated
  `spec=` argument raises the same error — the field has no destination in the current
  create request, and dropping it silently would send a different call than the one you
  wrote.

Both messages name `pc.create_index_from_backup(backup_id=..., name=...)` as the
supported restore path. See [backups and restore](indexes/backups-and-restore.md); it
covers managed (serverless) and BYOC indexes, since pod indexes cannot be backed up
either.
