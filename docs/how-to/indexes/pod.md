# Working with pod-based indexes

Pod-based indexes run on dedicated infrastructure pods. You choose a pod type and size
based on your throughput and latency requirements.

## Create a pod-based index

An index's fields are declared as a `schema`; `deployment` picks the pod environment and type:

```python
from pinecone import Pinecone

pc = Pinecone(api_key="your-api-key")

pc.indexes.create(
    name="product-search",
    schema={"fields": {"embedding": {"type": "dense_vector", "dimension": 1536, "metric": "cosine"}}},
    deployment={
        "deployment_type": "pod",
        "environment": "us-east1-gcp",
        "pod_type": "p1.x1",
    },
)
```

`create` polls until the index is ready by default. Pass `timeout=-1` to return immediately
without waiting.

Pod indexes are the only index type where `schema=` can also declare metadata-only fields
(`boolean`, `float`, `string_list`, or `string` without `full_text_search`). Managed and BYOC
indexes reject those field types, since metadata there is indexed automatically at upsert.

### Supported pod types

Use the {class}`~pinecone.models.enums.PodType` enum for tab-completion and typo safety:

```python
from pinecone import Pinecone
from pinecone.models.enums import PodType

pc = Pinecone(api_key="your-api-key")

pc.indexes.create(
    name="product-search",
    schema={"fields": {"embedding": {"type": "dense_vector", "dimension": 1536, "metric": "cosine"}}},
    deployment={
        "deployment_type": "pod",
        "environment": "us-east1-gcp",
        "pod_type": PodType.P1_X1,
    },
)
```

| Pod type | Description |
|---|---|
| ``s1.x1``, ``s1.x2``, ``s1.x4``, ``s1.x8`` | Storage-optimized, lower query throughput |
| ``p1.x1``, ``p1.x2``, ``p1.x4``, ``p1.x8`` | Performance-optimized, balanced storage |
| ``p2.x1``, ``p2.x2``, ``p2.x4``, ``p2.x8`` | High-throughput, lower storage capacity |

The `x1`/`x2`/`x4`/`x8` suffix controls the number of compute units per pod.

### Supported environments

Use the {class}`~pinecone.models.enums.PodIndexEnvironment` enum:

```python
from pinecone.models.enums import PodIndexEnvironment

deployment = {
    "deployment_type": "pod",
    "environment": PodIndexEnvironment.US_EAST1_GCP,
    "pod_type": "p1.x1",
}
```

Common environments: ``us-east1-gcp``, ``us-west1-gcp``, ``us-east-1-aws``,
``eu-west1-gcp``, ``eastus-azure``.

### Replicas and shards

Replicas duplicate the index for higher availability and query throughput. Shards split the
index's data across multiple pods to fit more data. The total pod count is replicas times
shards:

```python
from pinecone import Pinecone

pc = Pinecone(api_key="your-api-key")

pc.indexes.create(
    name="product-search-ha",
    schema={"fields": {"embedding": {"type": "dense_vector", "dimension": 1536, "metric": "cosine"}}},
    deployment={
        "deployment_type": "pod",
        "environment": "us-east1-gcp",
        "pod_type": "p1.x1",
        "replicas": 2,
        "shards": 2,
    },
)
```


## Scale replicas

Increase or decrease replicas on a running index with `configure`:

```python
pc.indexes.configure("product-search", deployment={"replicas": 4})
```

Scaling takes effect within a few minutes. The index remains available during the change.

### Change pod type

Upgrade to a larger pod size in-place:

```python
pc.indexes.configure("product-search", deployment={"pod_type": "p1.x2"})
```


## Create a collection

A collection is a static snapshot of a pod index's vector data. Create one to preserve an
index's contents, for example before deleting the index or changing its pod configuration:

```python
from pinecone import Pinecone

pc = Pinecone(api_key="your-api-key")

pc.collections.create(name="product-search-snapshot", source="product-search")
```

Restoring a collection into a new index is not currently supported. `pc.indexes.create()`
rejects `source_collection` with a 400 error ("Creating an index from collection or backup is
not yet supported"). See {doc}`/how-to/indexes/backups-and-restore` for creating backups and
restoring serverless indexes; pod indexes can't be backed up either.


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
