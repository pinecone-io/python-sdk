# Working with Serverless Indexes

Serverless indexes scale automatically: you pay for storage and queries without managing
infrastructure. Pinecone handles capacity, replication, and availability.

## Create a serverless index

An index's fields are declared as a `schema`; `deployment` picks the cloud and region:

```python
from pinecone import Pinecone

pc = Pinecone(api_key="your-api-key")

pc.indexes.create(
    name="product-search",
    schema={"fields": {"embedding": {"type": "dense_vector", "dimension": 1536, "metric": "cosine"}}},
    deployment={"deployment_type": "managed", "cloud": "aws", "region": "us-east-1"},
)
```

`"deployment_type": "managed"` is what makes an index serverless. Omit `deployment`
entirely and you get a managed index on AWS `us-east-1`.

Only fields that get *searched* go in the schema: `dense_vector`, `sparse_vector`, or
`string` carrying a `full_text_search` config. Metadata you merely filter on stays out —
put it on the records you upsert and it is indexed for filtering automatically. Declaring
a metadata-only field here is rejected server-side, and one rejected field fails the whole
schema. A hybrid index has to declare its `sparse_vector` field up front, because
`configure` cannot add one later.

`create` polls until the index is ready by default, with no upper time bound. Pass a
positive `timeout` to bound the wait — {exc}`~pinecone.errors.exceptions.PineconeTimeoutError`
if the index is still not ready when it elapses — or `timeout=-1` to return as soon as the
create request is accepted.

`dimension=`, `metric=`, `vector_type=`, and `spec=` remain as deprecated keyword-only
sugar, translated into `schema=`/`deployment=` before the request is sent. Each one is
mutually exclusive with the argument it translates to, so combining them raises
{exc}`~pinecone.errors.exceptions.PineconeValueError`. Write new code against `schema=`
and `deployment=`; see the [v10 migration guide](../../migration/v10-migration.md) for the
before/after.

### Clouds and regions

{class}`~pinecone.models.enums.CloudProvider` and the per-cloud region enums
`AwsRegion`, `GcpRegion`, and `AzureRegion` — all in `pinecone.models.enums`,
and all re-exported from `pinecone` — give you tab-completion and typo safety:

```python
from pinecone import Pinecone
from pinecone.models.enums import AwsRegion, CloudProvider

pc = Pinecone(api_key="your-api-key")

pc.indexes.create(
    name="product-search",
    schema={"fields": {"embedding": {"type": "dense_vector", "dimension": 1536, "metric": "cosine"}}},
    deployment={
        "deployment_type": "managed",
        "cloud": CloudProvider.AWS,
        "region": AwsRegion.US_EAST_1,
    },
)
```

The values those enums carry:

**AWS:** ``us-east-1``, ``us-west-2``, ``eu-west-1``, ``eu-central-1``, ``ap-southeast-1``

**GCP:** ``us-central1``, ``europe-west4``

**Azure:** ``eastus2``, ``germanywestcentral``

`cloud` and `region` are sent as plain strings and the SDK does not check them against the
enums, so a region Pinecone adds later works as a string literal before an enum member
names it. Nothing checks the value before the request goes out, so a typo comes back as a
server error rather than a local one.

### Enable deletion protection

Add `deletion_protection="enabled"` to prevent accidental deletes:

```python
from pinecone import Pinecone
from pinecone.models.enums import DeletionProtection

pc = Pinecone(api_key="your-api-key")

pc.indexes.create(
    name="product-search",
    schema={"fields": {"embedding": {"type": "dense_vector", "dimension": 1536, "metric": "cosine"}}},
    deployment={"deployment_type": "managed", "cloud": "aws", "region": "us-east-1"},
    deletion_protection=DeletionProtection.ENABLED,
)
```


## Choose read capacity

`read_capacity` decides how a managed index serves reads and how they are billed. Omit it
and the index comes up on on-demand capacity, billed per operation:

```python
pc.indexes.create(
    name="product-search",
    schema={"fields": {"embedding": {"type": "dense_vector", "dimension": 1536, "metric": "cosine"}}},
    deployment={"deployment_type": "managed", "cloud": "aws", "region": "us-east-1"},
    read_capacity={"mode": "OnDemand"},
)
```

Dedicated capacity provisions read nodes you size yourself:

```python
pc.indexes.create(
    name="product-search",
    schema={"fields": {"embedding": {"type": "dense_vector", "dimension": 1536, "metric": "cosine"}}},
    deployment={"deployment_type": "managed", "cloud": "aws", "region": "us-east-1"},
    read_capacity={
        "mode": "Dedicated",
        "dedicated": {
            "node_type": "t1",
            "scaling": "Manual",
            "manual": {"shards": 2, "replicas": 2},
        },
    },
)
```

`node_type` is ``"b1"`` or ``"t1"``, where `t1` carries more processing power and memory.
`scaling` is ``"Manual"``, and `manual` names the `shards` and `replicas` to provision.
Setting `replicas` to `0` disables the index, which is a way to cut cost while a workload
is paused without dropping the data.

The same argument moves an existing index between modes. Read capacity applies
asynchronously, so `configure` returns while the change is still in flight:

```python
pc.indexes.configure("product-search", read_capacity={"mode": "OnDemand"})
```

Read it back on the index model. It decodes as
{class}`~pinecone.models.indexes.read_capacity.ReadCapacityOnDemandResponse` or
{class}`~pinecone.models.indexes.read_capacity.ReadCapacityDedicatedResponse`, and is
`None` when the response carries no read capacity. There is **no `mode` attribute** on
either — `mode` is the wire discriminator that picked the class, so branch on the type
instead:

```python
from pinecone import ReadCapacityDedicatedResponse

rc = pc.indexes.describe("product-search").read_capacity
if rc is not None:
    print(type(rc).__name__)   # 'ReadCapacityOnDemandResponse'
    print(rc.status.state)     # 'Ready'
    if isinstance(rc, ReadCapacityDedicatedResponse):
        print(rc.dedicated.node_type, rc.dedicated.manual.shards)
```

Reaching for `rc.mode` raises `AttributeError`; the mode is the class you got back.

`status.state` is where an asynchronous change shows up: ``"Ready"`` when settled,
``"Scaling"`` after a replica/shard change, ``"Migrating"`` while moving to another node
type, and ``"Error"`` with the reason in `status.error_message`. `status.current_shards`
and `status.current_replicas` report what is actually running, and are both `None` on
on-demand capacity, which has no fixed counts.

Read capacity applies to managed and BYOC indexes. Pod-based indexes size reads with
`pod_type`, `replicas`, and `shards` instead — see
[pod-based indexes](pod.md).


## Check index status

`describe` returns an {class}`~pinecone.models.indexes.index.IndexModel` with the current state:

```python
desc = pc.indexes.describe("product-search")
print(desc.status.state)   # e.g. "Ready"
print(desc.status.ready)   # True when ready to accept requests
```

Poll manually when you passed `timeout=-1` to `create`:

```python
import time

while not pc.indexes.describe("product-search").status.ready:
    time.sleep(5)
```


## List indexes

`list` returns a {class}`~pinecone.models.pagination.Paginator` you can iterate:

```python
for idx in pc.indexes.list():
    print(idx.name, idx.status.state)

# Just the names
names = [idx.name for idx in pc.indexes.list()]
```

The server returns every index in one page today, so the paginator yields once and stops.
It exposes the paginator interface anyway, so a call site written against it keeps working
if that changes.


## Describe an index

`schema.fields` maps each field name to a typed field model, so the dimension and metric
live on the field rather than on the index:

```python
idx = pc.indexes.describe("product-search")
print(idx.name)
print(idx.host)
for field_name, field in idx.schema.fields.items():
    print(field_name, field)
print(idx.deployment.cloud)
print(idx.deployment.region)
```

Reach a field you declared by name — `idx.schema.fields["embedding"].dimension` and
`.metric` on a `dense_vector` field. `idx.dimension` and `idx.metric` are gone: reading
either raises `AttributeError` naming the field access to use instead, because an
index with several vector fields has no single dimension to report.


## Delete an index

```python
pc.indexes.delete("product-search")
```

`delete` polls until the index is gone, with no upper time bound. Pass a positive
`timeout` to bound the wait, or `timeout=-1` to return as soon as the request is accepted
— the index is still being torn down when the call returns.

If deletion protection is enabled, the delete is refused with
{exc}`~pinecone.errors.exceptions.ForbiddenError`. Clear it first:

```python
pc.indexes.configure("product-search", deletion_protection="disabled")
pc.indexes.delete("product-search")
```


## See also

- {class}`~pinecone.models.indexes.index.IndexModel`: full index response model
- {class}`~pinecone.models.pagination.Paginator`: `list` response wrapper
- [Pod-based indexes](pod.md): the legacy deployment type
- [Backups and restore](backups-and-restore.md): snapshot and restore a serverless index
- [Concepts](../../guides/concepts.md): how deployments, schemas, and namespaces fit together
