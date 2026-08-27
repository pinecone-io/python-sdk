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

`create` polls until the index is ready by default. Pass `timeout=-1` to return immediately
without waiting.

### Supported clouds and regions

Use the {class}`~pinecone.CloudProvider`, {class}`~pinecone.AwsRegion`,
{class}`~pinecone.GcpRegion`, and {class}`~pinecone.AzureRegion` enums for
tab-completion and typo safety:

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

**AWS:** ``us-east-1``, ``us-west-2``, ``eu-west-1``, ``eu-central-1``, ``ap-southeast-1``

**GCP:** ``us-central1``, ``europe-west4``

**Azure:** ``eastus2``, ``germanywestcentral``

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


## Check index status

`describe` returns an {class}`~pinecone.models.IndexModel` with the current state:

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


## Describe an index

```python
idx = pc.indexes.describe("product-search")
print(idx.name)
print(idx.schema.fields["embedding"].dimension)
print(idx.schema.fields["embedding"].metric)
print(idx.deployment.cloud)
print(idx.deployment.region)
```


## Delete an index

```python
pc.indexes.delete("product-search")
```

`delete` polls until the index is gone. Pass `timeout=-1` to return immediately.

If deletion protection is enabled, disable it first:

```python
pc.indexes.configure("product-search", deletion_protection="disabled")
pc.indexes.delete("product-search")
```


## See also

- {class}`~pinecone.models.IndexModel`: full index response model
- {class}`~pinecone.models.pagination.Paginator`: `list` response wrapper
- {doc}`/how-to/indexes/pod`: pod-based index management
- {doc}`/how-to/indexes/backups-and-restore`: create and restore backups
