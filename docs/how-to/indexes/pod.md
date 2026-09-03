# Working with pod-based indexes

Pod-based indexes run on dedicated infrastructure pods. A pod type and pod count determine
the index's throughput, latency, and storage capacity. They predate serverless indexes;
reach for a [serverless index](serverless.md) unless you have a reason not to.

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
without waiting. See [serverless indexes](serverless.md) for the managed deployment options.

A pod index's `schema` is not a way around the field types `2026-07` refuses to declare.
`float`, `boolean`, `string_list`, and `string` without a `full_text_search` config are
read back on a describe but rejected on create, on every deployment type this API version
creates — and since it creates no pod index at all, pods are not an exception. Leave
metadata out of the schema and put it on the records you upsert; it is indexed for
filtering automatically.

### Pod types

`describe()` reports an index's pod type, and `configure()` accepts a new one. Use the
{class}`~pinecone.models.enums.PodType` enum for tab-completion and typo safety.

| Pod type | Optimized for |
|---|---|
| ``s1.x1``, ``s1.x2``, ``s1.x4``, ``s1.x8`` | Storage, at lower query throughput |
| ``p1.x1``, ``p1.x2``, ``p1.x4``, ``p1.x8`` | Balanced performance |
| ``p2.x1``, ``p2.x2``, ``p2.x4``, ``p2.x8`` | Query throughput, at lower storage capacity |

The family before the dot picks what the pod is optimized for; the ``xN`` after it is the
size multiplier.

### Environments

A pod index lives in one environment, fixed when the index was created and reported as
`deployment.environment`. `pinecone.models.enums.PodIndexEnvironment` names the values the
SDK ships with — a convenience enum rather than an exhaustive list; common ones are
``us-east1-gcp``, ``us-west1-gcp``, ``us-east-1-aws``, ``eu-west1-gcp``, and
``eastus-azure``.

### Replicas and shards

Replicas duplicate the index for higher availability and query throughput. Shards split the
index's data across multiple pods, which is what decides how much data fits. The total pod
count is replicas times shards, and `describe()` reports both.

Only `replicas` and `pod_type` can be changed on a running index. **`shards` is fixed once
the index exists**, so an index that has outgrown its shard count has to be rebuilt. Note
that `configure`'s `deployment` dict is forwarded verbatim, so a stray `{"shards": 4}` is
not caught locally — it reaches an API that has no field for it.


## Scale replicas

Increase or decrease replicas on a running index with `configure`:

```python
pc.indexes.configure("product-search", deployment={"replicas": 4})
```

Pod scaling is applied in the background, so the call returns while the change is still in
flight. Read `status` on the returned {class}`~pinecone.models.indexes.index.IndexModel` to see how far
it has got rather than assuming it landed. The index stays available during the change.

`deployment` here carries only scaling keys. It must not include `deployment_type`:
deployment type, cloud/region, and environment cannot be changed after creation, and
passing it raises {exc}`~pinecone.errors.exceptions.PineconeValueError`.

### Change pod type

Move to a different pod family or size in place:

```python
from pinecone.models.enums import PodType

pc.indexes.configure("product-search", deployment={"pod_type": PodType.P1_X2})
```

Both keys can go in one call: `deployment={"replicas": 4, "pod_type": "p1.x2"}`.

The deprecated `replicas=` and `pod_type=` keyword arguments still work and are translated
into `deployment=` for you, but cannot be combined with it — passing both raises
{exc}`~pinecone.errors.exceptions.PineconeValueError`.


## Collections

A collection is a static snapshot of a pod index's vector data, and `2026-07` creates no
pod index for `pc.collections.create()` to point at. Collections that already exist remain
listable, describable, and deletable — see [collections](../collections.md).

For snapshot and restore on managed indexes, use backups instead: see
[backups and restore](backups-and-restore.md).


## Describe a pod index

The `deployment` field decodes as a
{class}`~pinecone.models.indexes.deployment.PodDeployment`, carrying the pod-specific
details:

```python
idx = pc.indexes.describe("legacy-recommender")
print(idx.deployment.environment)
print(idx.deployment.pod_type)
print(idx.deployment.replicas)
print(idx.deployment.shards)
```

`read_capacity` is not a pod concept: it is how managed and BYOC indexes size reads,
while a pod index sizes them with `pod_type`, `replicas`, and `shards`.


## Delete a pod index

```python
pc.indexes.delete("legacy-recommender")
```

`delete` polls until the index is gone. Pass `timeout=-1` to return as soon as the request
is accepted.

If deletion protection is enabled, the delete is refused with
{exc}`~pinecone.errors.exceptions.ForbiddenError`. Clear it first:

```python
pc.indexes.configure("legacy-recommender", deletion_protection="disabled")
pc.indexes.delete("legacy-recommender")
```

A pod index that still has a collection taken from it cannot be deleted until the
collection is really gone.


## See also

- {class}`~pinecone.models.indexes.index.IndexModel`: full index response model
- {class}`~pinecone.models.indexes.deployment.PodDeployment`: the pod deployment on a describe
- {class}`~pinecone.models.indexes.specs.PodSpec`: deprecated `create()` sugar for `deployment=`
- [Serverless indexes](serverless.md): the current deployment type
- [Collections](../collections.md): snapshots of a pod index
- [Backups and restore](backups-and-restore.md): the serverless snapshot mechanism
