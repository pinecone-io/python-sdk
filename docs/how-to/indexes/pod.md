# Working with Pod-Based Indexes

Pod-based indexes run on dedicated infrastructure pods. You choose a pod type and size
based on your throughput and latency requirements.

## Create a pod-based index

Pass a {class}`~pinecone.PodSpec` with the environment and pod type:

```python
from pinecone import Pinecone, PodSpec

pc = Pinecone(api_key="your-api-key")

pc.indexes.create(
    name="product-search",
    dimension=1536,
    metric="cosine",
    spec=PodSpec(
        environment="us-east1-gcp",
        pod_type="p1.x1",
        pods=1,
    ),
)
```

`create` polls until the index is ready by default. Pass `timeout=-1` to return immediately
without waiting.

### Supported pod types

Use the {class}`~pinecone.models.enums.PodType` enum for tab-completion and typo safety:

```python
from pinecone import Pinecone, PodSpec
from pinecone.models.enums import PodType

pc = Pinecone(api_key="your-api-key")

pc.indexes.create(
    name="product-search",
    dimension=1536,
    metric="cosine",
    spec=PodSpec(
        environment="us-east1-gcp",
        pod_type=PodType.P1_X1,
    ),
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

spec = PodSpec(
    environment=PodIndexEnvironment.US_EAST1_GCP,
    pod_type="p1.x1",
)
```

Common environments: ``us-east1-gcp``, ``us-west1-gcp``, ``us-east-1-aws``,
``eu-west1-gcp``, ``eastus-azure``.

### Multiple replicas and pods

Replicas increase availability and query throughput. Pods control total storage capacity:

```python
from pinecone import Pinecone, PodSpec

pc = Pinecone(api_key="your-api-key")

pc.indexes.create(
    name="product-search-ha",
    dimension=1536,
    metric="cosine",
    spec=PodSpec(
        environment="us-east1-gcp",
        pod_type="p1.x1",
        pods=2,
        replicas=2,
    ),
)
```


## Scale replicas

Increase or decrease replicas on a running index with `configure`:

```python
pc.indexes.configure("product-search", replicas=4)
```

Scaling takes effect within a few minutes. The index remains available during the change.

### Change pod type

Upgrade to a larger pod size in-place:

```python
pc.indexes.configure("product-search", pod_type="p1.x2")
```


## Create an index from a collection

A collection is a static snapshot of a pod index. You can create a new index from one
to restore a point-in-time state or change pod configuration:

```python
from pinecone import Pinecone, PodSpec

pc = Pinecone(api_key="your-api-key")

pc.indexes.create(
    name="product-search-restored",
    dimension=1536,
    metric="cosine",
    spec=PodSpec(
        environment="us-east1-gcp",
        pod_type="p1.x1",
        source_collection="product-search-snapshot",
    ),
)
```

See {doc}`/how-to/indexes/backups-and-restore` for creating backups and restoring
serverless indexes.


## Describe a pod index

The `spec.pod` field contains pod-specific details:

```python
idx = pc.indexes.describe("product-search")
print(idx.spec.pod.environment)
print(idx.spec.pod.pod_type)
print(idx.spec.pod.replicas)
print(idx.spec.pod.pods)
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

- {class}`~pinecone.models.IndexModel` — full index response model
- {class}`~pinecone.models.indexes.specs.PodSpec` — request-side pod spec
- {class}`~pinecone.models.indexes.deployment.PodDeployment` — response-side pod deployment
- {doc}`/how-to/indexes/serverless` — serverless index management
- {doc}`/how-to/indexes/backups-and-restore` — create and restore backups
