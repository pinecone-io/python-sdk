"""Unit tests for IndexModel.to_dict() recursive conversion."""

from __future__ import annotations

from msgspec import Struct

from pinecone.models.indexes.deployment import ManagedDeployment
from pinecone.models.indexes.index import IndexModel, IndexStatus
from pinecone.models.indexes.read_capacity import ReadCapacityOnDemandResponse, ReadCapacityStatus
from pinecone.models.indexes.schema import (
    DenseVectorField,
    IndexSchema,
    LegacyMetadataField,
    SemanticTextField,
)


def _make_serverless_index(**kwargs: object) -> IndexModel:
    return IndexModel(
        name="idx",
        host="localhost",
        status=IndexStatus(ready=True, state="Ready"),
        schema=IndexSchema(fields={"embedding": DenseVectorField(dimension=3, metric="cosine")}),
        deployment=ManagedDeployment(cloud="aws", region="us-east-1"),
        deletion_protection="disabled",
        **kwargs,  # type: ignore[arg-type]
    )


def test_to_dict_required_fields_only() -> None:
    model = _make_serverless_index()
    result = model.to_dict()

    assert isinstance(result["status"], dict)
    assert "ready" in result["status"]
    assert "state" in result["status"]

    assert isinstance(result["deployment"], dict)
    assert result["deployment"]["deployment_type"] == "managed"

    assert isinstance(result["schema"], dict)
    assert result["schema"]["fields"]["embedding"]["type"] == "dense_vector"


def test_to_dict_nested_struct_recursive() -> None:
    model = _make_serverless_index()
    result = model.to_dict()

    assert not isinstance(result["status"], Struct)
    assert not isinstance(result["deployment"], Struct)
    assert not isinstance(result["schema"], Struct)
    assert not isinstance(result["schema"]["fields"]["embedding"], Struct)


def test_to_dict_read_capacity_present() -> None:
    model = _make_serverless_index(
        read_capacity=ReadCapacityOnDemandResponse(status=ReadCapacityStatus(state="Ready"))
    )
    result = model.to_dict()

    assert isinstance(result["read_capacity"], dict)
    assert result["read_capacity"]["mode"] == "OnDemand"
    assert result["read_capacity"]["status"]["state"] == "Ready"


def test_to_dict_read_capacity_none() -> None:
    model = _make_serverless_index()
    result = model.to_dict()

    assert result["read_capacity"] is None


def test_to_dict_semantic_text_field() -> None:
    model = _make_serverless_index()
    model.schema.fields["content"] = SemanticTextField(model="multilingual-e5-large")
    result = model.to_dict()

    assert result["schema"]["fields"]["content"]["type"] == "semantic_text"
    assert result["schema"]["fields"]["content"]["model"] == "multilingual-e5-large"


def test_to_dict_legacy_untyped_field_has_no_type_key() -> None:
    model = _make_serverless_index()
    model.schema.fields["old_meta"] = LegacyMetadataField(filterable=True)
    result = model.to_dict()

    assert result["schema"]["fields"]["old_meta"] == {"filterable": True}


def test_to_dict_tags_preserved() -> None:
    model = _make_serverless_index(tags={"env": "prod"})
    result = model.to_dict()

    assert result["tags"] == {"env": "prod"}


def test_to_dict_is_pure_read() -> None:
    model = _make_serverless_index()
    d = model.to_dict()
    d["name"] = "mutated"
    second = model.to_dict()

    assert second["name"] == "idx"
