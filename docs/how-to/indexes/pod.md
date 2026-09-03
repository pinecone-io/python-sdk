# Working with pod-based indexes

Pod-based indexes run on dedicated infrastructure pods. A pod type and pod count determine
the index's throughput, latency, and storage capacity.

````{important}
API version `2026-07`, the version this SDK targets, does not create pod-based indexes.
A `create()` call carrying a pod deployment is refused by the server:

```
[400 INVALID_ARGUMENT] deployment_type 'pod' is not supported on this API
version. Set deployment_type to 'managed' to create a serverless index, or
set the X-Pinecone-API-Version header to an earlier version.
```

This is a property of the API version, not a type the SDK dropped. `PodDeployment` and
`PodSpec` are still exported, `describe()` and `list()` still decode pod indexes that
already exist, and `configure()` and `delete()` still work against them. Creation is the one
operation `2026-07` refuses, so the rest of this page applies to pod indexes you already
have. See {ref}`pod-collections` in the v10 migration guide for the full picture, including
the header pin that reaches an earlier API version.
````

## Create a pod-based index

There is no way to create one at `2026-07`. Both spellings are refused with the 400 above:
`deployment={"deployment_type": "pod", ...}` and the deprecated `spec=PodSpec(...)`. The
refusal comes from the server, not from client-side validation, so it arrives as an
`ApiError` on the request.

New indexes take a managed (serverless) deployment instead:

```python
from pinecone import Pinecone

pc = Pinecone(api_key="your-api-key")

pc.indexes.create(
    name="product-search",
    schema={"fields": {"embedding": {"type": "dense_vector", "dimension": 1536, "metric": "cosine"}}},
    deployment={"deployment_type": "managed", "cloud": "aws", "region": "us-east-1"},
)
```

`create` polls until the index is ready by default. Pass `timeout=-1` to return immediately
without waiting. See {doc}`/how-to/indexes/serverless` for the managed deployment options.

A pod index's `schema` can declare metadata-only fields (`boolean`, `float`, `string_list`,
or `string` without `full_text_search`), which is unique to pod indexes. Managed and BYOC
indexes reject those field types, since metadata there is indexed automatically at upsert.

### Pod types

`describe()` reports an index's pod type, and `configure()` accepts a new one. Use the
{class}`~pinecone.models.enums.PodType` enum for tab-completion and typo safety.

| Pod type | Description |
|---|---|
| ``s1.x1``, ``s1.x2``, ``s1.x4``, ``s1.x8`` | Storage-optimized, lower query throughput |
| ``p1.x1``, ``p1.x2``, ``p1.x4``, ``p1.x8`` | Performance-optimized, balanced storage |
| ``p2.x1``, ``p2.x2``, ``p2.x4``, ``p2.x8`` | High-throughput, lower storage capacity |

The `x1`/`x2`/`x4`/`x8` suffix controls the number of compute units per pod.

### Environments

A pod index lives in one environment, fixed when the index was created and reported as
`deployment.environment`. The {class}`~pinecone.models.enums.PodIndexEnvironment` enum names
the known values; common ones are ``us-east1-gcp``, ``us-west1-gcp``, ``us-east-1-aws``,
``eu-west1-gcp``, and ``eastus-azure``.

### Replicas and shards

Replicas duplicate the index for higher availability and query throughput. Shards split the
index's data across multiple pods to fit more data. The total pod count is replicas times
shards. `describe()` reports both, and both can be changed on a running index with
`configure`.


## Scale replicas

Increase or decrease replicas on a running index with `configure`:

```python
pc.indexes.configure("product-search", deployment={"replicas": 4})
```

Scaling takes effect within a few minutes. The index remains available during the change.

### Change pod type

Upgrade to a larger pod size in-place:

```python
from pinecone.models.enums import PodType

pc.indexes.configure("product-search", deployment={"pod_type": PodType.P1_X2})
```


## Create a collection

A collection is a static snapshot of a pod index's vector data. Since `2026-07` creates no
pod index, `pc.collections.create()` has no source it can accept, so there is no snapshot to
take here. Collections that already exist remain listable, describable, and deletable — see
{doc}`/how-to/collections`.

For snapshot and restore on managed indexes, use backups instead: see
{doc}`/how-to/indexes/backups-and-restore`. Restoring a collection into a new index is not
supported either — `pc.indexes.create(source_collection=...)` raises
{exc}`~pinecone.errors.exceptions.PineconeTypeError` naming
`pc.create_index_from_backup(...)` as the replacement.


## Describe a pod index

The `deployment` field contains pod-specific details:

```python
idx = pc.indexes.describe("product-search")
print(idx.deployment.environment)
print(idx.deployment.pod_type)
print(idx.deployment.replicas)
print(idx.deployment.shards)
```


## Delete a pod index

```python
pc.indexes.delete("product-search")
```

If deletion protection is enabled, disable it first:

```python
pc.indexes.configure("product-search", deletion_protection="disabled")
pc.indexes.delete("product-search")
```


## See also

- {class}`~pinecone.models.IndexModel`: full index response model
- {class}`~pinecone.models.indexes.specs.PodSpec`: deprecated `create()` sugar for `deployment=`
- {class}`~pinecone.models.indexes.deployment.PodDeployment`: response-side pod deployment
- {doc}`/how-to/indexes/serverless`: serverless index management
- {doc}`/how-to/indexes/backups-and-restore`: create and restore backups
