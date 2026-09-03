# Bulk importing vectors

Bulk import loads vectors from cloud storage (Amazon S3, Google Cloud Storage, or Azure
Blob Storage) into a Pinecone index. The import runs server-side, so it handles millions
of vectors without keeping a long-lived client connection open — you start it, then poll
for progress.

Reach for it when the data is already in cloud storage and there is far too much of it
to stream through a client. For vectors you hold in memory or in a DataFrame,
{meth}`~pinecone.index.Index.upsert` and
{meth}`~pinecone.index.Index.upsert_from_dataframe` are the right paths; see
{doc}`/guides/bulk-ingest` for how that choice is made and how the client-side batching
behaves.

The source must be a directory of Parquet files formatted to the
[Pinecone-required schema](https://docs.pinecone.io/guides/data/understanding-imports).

The `uri` names that directory prefix, never an individual file, and takes one of three
forms: `s3://` for Amazon S3, `gs://` for Google Cloud Storage, or an `https://` URL
naming an Azure Blob Storage container. An `s3://` source additionally requires that the
index itself be hosted on AWS. The same bucket is rejected for an index on another
cloud. Anything else fails the call, as does an S3 directory bucket, which imports do
not support.


## Start an import

{meth}`~pinecone.index.Index.start_import` initiates the operation and returns immediately with an
operation ID:

```python
from pinecone import Pinecone

pc = Pinecone(api_key="your-api-key")
index = pc.index("product-search")

response = index.start_import(uri="s3://my-bucket/embeddings/")
import_id = response.id
print(import_id)  # e.g. "1"
```


## Handle errors during import

`error_mode` decides what happens to a record the import cannot read. It defaults to
`"abort"`, which ends the whole import at the first such record, so nothing is silently
dropped. Pass `error_mode="continue"` to skip unreadable records and import the rest:

```python
response = index.start_import(
    uri="s3://my-bucket/embeddings/",
    error_mode="continue",
)
```

You can also use the {class}`~pinecone.models.imports.error_mode.ImportErrorMode` enum:

```python
from pinecone.models.imports.error_mode import ImportErrorMode

response = index.start_import(
    uri="s3://my-bucket/embeddings/",
    error_mode=ImportErrorMode.CONTINUE,
)
```


## Check import status

{meth}`~pinecone.index.Index.describe_import` returns an {class}`~pinecone.models.imports.model.ImportModel`
with the current state:

```python
import_op = index.describe_import(import_id)
print(import_op.status)           # e.g. "InProgress"
print(import_op.percent_complete) # e.g. 42.0
print(import_op.records_imported) # e.g. 150000
```

`status` is one of `"Pending"`, `"InProgress"`, `"Completed"`, `"Failed"`, or
`"Cancelled"`. `percent_complete`, `records_imported`, `finished_at`, and `error` are all
optional and read `None` until the server reports them, so guard on `status` rather than
on those fields being set.


## Poll until complete

Nothing in the SDK waits for an import, so polling is yours to write. Give the loop a
deadline: `"Pending"` and `"InProgress"` are not guaranteed to advance, and an unbounded
`while` against a stuck import hangs forever.

```python
import time

TERMINAL = ("Completed", "Failed", "Cancelled")
deadline = time.monotonic() + 6 * 60 * 60   # give the import six hours

import_op = index.describe_import(import_id)
while import_op.status not in TERMINAL:
    if time.monotonic() > deadline:
        raise TimeoutError(
            f"import {import_op.id} still {import_op.status} "
            f"at {import_op.percent_complete}%"
        )
    time.sleep(10)
    import_op = index.describe_import(import_id)

if import_op.status == "Completed":
    print(f"Imported {import_op.records_imported} records")
else:
    print(f"Import ended with status: {import_op.status}")
    if import_op.error:
        print(import_op.error)
```

Poll on a human timescale — the import is measured in minutes to hours, so a ten-second
interval is already generous.


## List imports

{meth}`~pinecone.index.Index.list_imports` yields one
{class}`~pinecone.models.imports.model.ImportModel` per import operation, following the
pagination tokens itself. Note that it yields **individual operations**, not pages —
unlike {meth}`~pinecone.index.Index.list_namespaces`, which yields a page at a time. Both
are plain generators, so nothing is requested until you iterate:

```python
for imp in index.list_imports():
    print(imp.id, imp.status, imp.percent_complete)
```

Pass `limit` to control the page size:

```python
for imp in index.list_imports(limit=20):
    print(imp.id, imp.status)
```

To hold the pagination token yourself instead, use
{meth}`~pinecone.index.Index.list_imports_paginated`, which returns one page as an
{class}`~pinecone.models.imports.list.ImportList`:

```python
page = index.list_imports_paginated(limit=10)
for imp in page:
    print(imp.id, imp.status)
```


## Cancel an import

{meth}`~pinecone.index.Index.cancel_import` stops an in-progress import. Already-imported
records are not rolled back, so a cancelled import leaves the index holding whatever
landed before the cancellation — `describe_import` reports that count in
`records_imported`.

```python
index.cancel_import(import_id)
```

`describe_import`, `cancel_import`, and `start_import` all take the ID positionally, and
an integer ID is converted to a string for you.


## See also

- {doc}`/guides/bulk-ingest`: choosing between the import and upsert paths
- {doc}`/how-to/vectors/upsert-and-query`: upsert vectors directly in batches
- {class}`~pinecone.index.Index`: full data plane client reference
- {class}`~pinecone.models.imports.model.ImportModel`: import operation model
- {class}`~pinecone.models.imports.list.ImportList`: one page of import operations
