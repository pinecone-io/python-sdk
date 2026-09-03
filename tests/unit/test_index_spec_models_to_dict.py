"""Unit tests for to_dict() on index sub-models and request specs."""

from __future__ import annotations

from pinecone.models.indexes.index import IndexStatus
from pinecone.models.indexes.read_capacity import (
    ReadCapacityDedicatedConfig,
    ReadCapacityDedicatedResponse,
    ReadCapacityStatus,
    ScalingConfigManual,
)
from pinecone.models.indexes.schema import (
    DenseVectorField,
    FullTextSearchConfig,
    IndexSchema,
    StringField,
)
from pinecone.models.indexes.specs import (
    EmbedConfig,
    IntegratedSpec,
    PodSpec,
    ServerlessSpec,
)


def test_index_status_to_dict() -> None:
    result = IndexStatus(ready=True, state="Ready").to_dict()
    assert result == {"ready": True, "state": "Ready"}


def test_index_schema_to_dict_includes_type_tags() -> None:
    schema = IndexSchema(
        fields={
            "embedding": DenseVectorField(dimension=1536, metric="cosine"),
            "title": StringField(full_text_search=FullTextSearchConfig(language="en")),
        }
    )
    result = schema.to_dict()
    assert result["fields"]["embedding"]["type"] == "dense_vector"
    assert result["fields"]["embedding"]["dimension"] == 1536
    assert result["fields"]["title"]["type"] == "string"
    assert result["fields"]["title"]["full_text_search"]["language"] == "en"


def test_dedicated_read_capacity_to_builtins_shape() -> None:
    import msgspec

    rc = ReadCapacityDedicatedResponse(
        dedicated=ReadCapacityDedicatedConfig(
            node_type="t1",
            scaling="Manual",
            manual=ScalingConfigManual(shards=2, replicas=3),
        ),
        status=ReadCapacityStatus(state="Ready", current_shards=2, current_replicas=3),
    )
    result = msgspec.to_builtins(rc)
    assert result["mode"] == "Dedicated"
    assert result["dedicated"]["node_type"] == "t1"
    assert result["dedicated"]["manual"] == {"shards": 2, "replicas": 3}
    assert result["status"]["current_shards"] == 2


def test_serverless_spec_to_dict() -> None:
    result = ServerlessSpec(cloud="aws", region="us-east-1").to_dict()
    assert result["cloud"] == "aws"
    assert result["region"] == "us-east-1"
    assert "serverless" not in result


def test_serverless_spec_asdict_still_works() -> None:
    result = ServerlessSpec(cloud="aws", region="us-east-1").asdict()
    assert result == {"serverless": {"cloud": "aws", "region": "us-east-1"}}


def test_pod_spec_to_dict() -> None:
    result = PodSpec(environment="us-east-1-gcp").to_dict()
    assert result["environment"] == "us-east-1-gcp"
    assert result["pod_type"] == "p1.x1"
    assert result["replicas"] == 1
    assert result["shards"] == 1
    assert result["pods"] == 1
    assert result["metadata_config"] is None
    assert result["source_collection"] is None


def test_integrated_spec_to_dict_nested_embed() -> None:
    spec = IntegratedSpec(
        cloud="aws",
        region="us-east-1",
        embed=EmbedConfig(model="multilingual-e5-large", field_map={"text": "my_text"}),
    )
    result = spec.to_dict()
    assert isinstance(result["embed"], dict)
    assert result["embed"]["model"] == "multilingual-e5-large"
    assert result["cloud"] == "aws"


def test_to_dict_is_pure_read() -> None:
    schema = IndexSchema(fields={"embedding": DenseVectorField(dimension=3, metric="cosine")})
    first = schema.to_dict()
    first["fields"]["embedding"]["dimension"] = 999
    second = schema.to_dict()
    assert second["fields"]["embedding"]["dimension"] == 3
