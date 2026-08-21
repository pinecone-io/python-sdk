# Bulk Importing Vectors

Bulk import loads vectors from cloud storage (Amazon S3, Google Cloud Storage, or Azure Blob
Storage) into a Pinecone index. The import runs server-side, so it handles millions of vectors
without keeping a long-lived client connection open.

The source must be a directory of Parquet files formatted to the
`Pinecone-required schema <https://docs.pinecone.io/guides/data/understanding-imports>`_.

The `uri` names that directory prefix, never an individual file, and takes one of three
forms: `s3://` for Amazon S3, `gs://` for Google Cloud Storage, or an `https://` URL
naming an Azure Blob Storage container. An `s3://` source additionally requires that the
index itself be hosted on AWS — the same bucket is rejected for an index on another
cloud. Anything else fails the call, as does an S3 directory bucket, which imports do
not support.


## Start an import

{meth}`~pinecone.Index.start_import` initiates the operation and returns immediately with an
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

{meth}`~pinecone.Index.describe_import` returns an {class}`~pinecone.models.ImportModel`
with the current state:

```python
import_op = index.describe_import(import_id)
print(import_op.status)           # e.g. "InProgress"
print(import_op.percent_complete) # e.g. 42.0
print(import_op.records_imported) # e.g. 150000
```

`status` is one of: `"Pending"`, `"InProgress"`, `"Completed"`, `"Failed"`, `"Cancelled"`.


## Poll until complete

```python
import time

import_op = index.describe_import(import_id)
while import_op.status not in ("Completed", "Failed", "Cancelled"):
    time.sleep(10)
    import_op = index.describe_import(import_id)

if import_op.status == "Completed":
    print(f"Imported {import_op.records_imported} records")
else:
    print(f"Import ended with status: {import_op.status}")
    if import_op.error:
        print(import_op.error)
```


## List imports

{meth}`~pinecone.Index.list_imports` yields {class}`~pinecone.models.ImportModel` objects
for all imports on the index, following pagination automatically:

```python
for imp in index.list_imports():
    print(imp.id, imp.status, imp.percent_complete)
```

Pass `limit` to control the page size:

```python
for imp in index.list_imports(limit=20):
    print(imp.id, imp.status)
```


## Cancel an import

{meth}`~pinecone.Index.cancel_import` stops an in-progress import. Already-imported records
are not rolled back.

```python
index.cancel_import(import_id)
```


## See also

- {doc}`/how-to/vectors/upsert-and-query` — upsert vectors directly in batches
- {class}`~pinecone.Index` — full data plane client reference
- {class}`~pinecone.models.ImportModel` — import operation model
