"""Unit tests for CreateIndexRequest and ConfigureIndexRequest (2026-07)."""

from __future__ import annotations

import msgspec
import pytest

from pinecone.errors import PineconeError, PineconeValueError
from pinecone.models.indexes.requests import (
    ConfigureIndexRequest,
    CreateIndexRequest,
)
from pinecone.models.indexes.schema import (
    DenseVectorField,
    FullTextSearchConfig,
    IndexSchema,
    StringField,
)

_SCHEMA = {"fields": {"vec": {"type": "dense_vector", "dimension": 1536, "metric": "cosine"}}}


def test_create_request_with_schema_only() -> None:
    req = CreateIndexRequest(schema=_SCHEMA)
    assert req.name is None
    assert req.deployment is None
    assert req.read_capacity is None
    assert req.deletion_protection is None
    assert req.tags is None


def test_create_request_full() -> None:
    req = CreateIndexRequest(
        schema=_SCHEMA,
        name="my-index",
        deployment={"deployment_type": "managed", "cloud": "aws", "region": "us-east-1"},
        read_capacity={"mode": "OnDemand"},
        deletion_protection="enabled",
        tags={"env": "prod"},
        cmek_id="arn:aws:kms:us-east-1:123456789012:key/mrk-abc123",
    )
    result = msgspec.to_builtins(req)
    assert result["name"] == "my-index"
    assert result["deletion_protection"] == "enabled"
    assert result["tags"] == {"env": "prod"}
    assert result["deployment"]["deployment_type"] == "managed"
    assert result["deployment"]["cloud"] == "aws"
    assert result["deployment"]["region"] == "us-east-1"
    assert result["read_capacity"] == {"mode": "OnDemand"}
    assert result["cmek_id"] == "arn:aws:kms:us-east-1:123456789012:key/mrk-abc123"


def test_create_request_serialization_omits_none() -> None:
    req = CreateIndexRequest(schema=_SCHEMA, name="idx")
    result = msgspec.to_builtins(req)
    assert set(result.keys()) == {"schema", "name"}


def test_create_request_accepts_typed_schema() -> None:
    schema = IndexSchema(
        fields={
            "vec": DenseVectorField(dimension=1536, metric="cosine"),
            "text": StringField(full_text_search=FullTextSearchConfig()),
        }
    )
    req = CreateIndexRequest(schema=schema)
    result = msgspec.to_builtins(req)
    assert result["schema"]["fields"]["vec"]["type"] == "dense_vector"
    assert result["schema"]["fields"]["text"]["type"] == "string"


def test_create_request_source_fields_serialized() -> None:
    req = CreateIndexRequest(
        schema=_SCHEMA,
        source_collection="movie-embeddings",
    )
    result = msgspec.to_builtins(req)
    assert result["source_collection"] == "movie-embeddings"
    assert "source_backup_id" not in result


def test_configure_request_all_optional() -> None:
    req = ConfigureIndexRequest()
    result = msgspec.to_builtins(req)
    assert result == {}


def test_configure_request_partial_tags_only() -> None:
    req = ConfigureIndexRequest(tags={"env": "staging"})
    result = msgspec.to_builtins(req)
    assert result == {"tags": {"env": "staging"}}


def test_configure_request_empty_string_tag_passthrough_for_deletion() -> None:
    req = ConfigureIndexRequest(tags={"env": "", "team": "ml"})
    result = msgspec.to_builtins(req)
    assert result == {"tags": {"env": "", "team": "ml"}}
    assert msgspec.json.encode(req) == b'{"tags":{"env":"","team":"ml"}}'


def test_configure_request_deployment_partial_update() -> None:
    req = ConfigureIndexRequest(deployment={"replicas": 3, "pod_type": "p2.x1"})
    result = msgspec.to_builtins(req)
    assert set(result.keys()) == {"deployment"}
    assert result["deployment"] == {"replicas": 3, "pod_type": "p2.x1"}


def test_configure_request_read_capacity_only_stays_sparse() -> None:
    req = ConfigureIndexRequest(
        read_capacity={
            "mode": "Dedicated",
            "dedicated": {"node_type": "t1", "scaling": "Manual", "manual": {"shards": 2}},
        }
    )
    result = msgspec.to_builtins(req)
    assert set(result.keys()) == {"read_capacity"}


class TestCreateRequestValidation:
    def test_empty_field_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="1-64 characters"):
            CreateIndexRequest(schema={"fields": {"": {"type": "dense_vector"}}})

    def test_leading_underscore_rejected(self) -> None:
        with pytest.raises(ValueError, match="reserved for internal use"):
            CreateIndexRequest(schema={"fields": {"_id": {"type": "dense_vector"}}})

    def test_leading_dollar_rejected(self) -> None:
        with pytest.raises(ValueError, match="filter operator"):
            CreateIndexRequest(schema={"fields": {"$and": {"type": "dense_vector"}}})

    def test_over_64_chars_rejected(self) -> None:
        long_name = "f" * 65
        with pytest.raises(ValueError, match="exceeds"):
            CreateIndexRequest(schema={"fields": {long_name: {"type": "dense_vector"}}})

    def test_64_chars_accepted(self) -> None:
        ok_name = "f" * 64
        req = CreateIndexRequest(
            schema={"fields": {ok_name: {"type": "dense_vector", "dimension": 3}}}
        )
        assert ok_name in req.schema["fields"]  # type: ignore[union-attr, index, operator]

    def test_error_message_names_field_and_value(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            CreateIndexRequest(schema={"fields": {"_secret": {"type": "dense_vector"}}})
        message = str(exc_info.value)
        assert "_secret" in message
        assert "Rename" in message

    def test_unknown_deployment_type_lists_allowed_values(self) -> None:
        with pytest.raises(PineconeValueError) as exc_info:
            CreateIndexRequest(
                schema=_SCHEMA,
                deployment={"deployment_type": "serverless", "cloud": "aws"},
            )
        message = str(exc_info.value)
        assert "serverless" in message
        assert "managed | pod | byoc" in message

    @pytest.mark.parametrize("deployment_type", ["MANAGED", "Pod", "serverless", "managed "])
    def test_bad_deployment_type_raises_pinecone_value_error(self, deployment_type: str) -> None:
        """PineconeValueError subclasses ValueError, so pytest.raises(ValueError)
        cannot distinguish the two; the exact-type assert is what fails pre-fix.
        """
        with pytest.raises(PineconeValueError) as exc_info:
            CreateIndexRequest(
                schema=_SCHEMA,
                deployment={"deployment_type": deployment_type, "cloud": "aws"},
            )
        assert type(exc_info.value) is PineconeValueError
        assert deployment_type in str(exc_info.value)

    def test_bad_deployment_type_is_catchable_as_pinecone_error(self) -> None:
        with pytest.raises(PineconeError):
            CreateIndexRequest(schema=_SCHEMA, deployment={"deployment_type": "MANAGED"})

    def test_valid_deployment_types_accepted(self) -> None:
        for deployment_type in ("managed", "pod", "byoc"):
            CreateIndexRequest(schema=_SCHEMA, deployment={"deployment_type": deployment_type})

    def test_dotted_and_unicode_names_accepted(self) -> None:
        CreateIndexRequest(
            schema={
                "fields": {
                    "my.field": {"type": "dense_vector", "dimension": 3},
                    "résumé": {"type": "string", "full_text_search": {}},
                }
            }
        )
