# Bulk Importing Vectors

Bulk import loads vectors from cloud storage (Amazon S3, Google Cloud Storage, or Azure Blob
Storage) into a Pinecone index. The import runs server-side, so it handles millions of vectors
without keeping a long-lived client connection open.

The source must be a directory of Parquet files formatted to the
`Pinecone-required schema <https://docs.pinecone.io/guides/data/understanding-imports>`_.


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

`uri` points to a directory prefix, not an individual file.


## Handle errors during import

By default, `error_mode` is `"continue"` — the server skips records that fail to parse and
continues importing the rest. Pass `error_mode="abort"` to stop the entire import on the
first error:

```python
response = index.start_import(
    uri="s3://my-bucket/embeddings/",
    error_mode="abort",
)
```

You can also use the {class}`~pinecone.models.ImportErrorMode` enum:

```python
from pinecone.models.imports.error_mode import ImportErrorMode

response = index.start_import(
    uri="s3://my-bucket/embeddings/",
    error_mode=ImportErrorMode.ABORT,
)
```


## Use a storage integration

If your project uses a configured storage integration for the import source, pass the
integration ID with `integration_id`:

```python
response = index.start_import(
    uri="s3://my-bucket/embeddings/",
    integration_id="integration-id",
)
```

You can combine integration_id with error_mode when needed.


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

Pass `limit` to control the page size (max 100):

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
