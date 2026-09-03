"""Unit tests for index response models."""

from __future__ import annotations

from typing import Any

import msgspec
import pytest

from pinecone.models.indexes.deployment import (
    ByocDeployment,
    ManagedDeployment,
    PodDeployment,
)
from pinecone.models.indexes.index import IndexModel, IndexStatus
from pinecone.models.indexes.list import IndexList
from pinecone.models.indexes.schema import DenseVectorField
from pinecone.models.indexes.specs import ByocSpec, PodSpec, ServerlessSpec
from tests.factories import make_index_list_response, make_index_response


class TestIndexStatus:
    def test_construct(self) -> None:
        status = IndexStatus(ready=True, state="Ready")
        assert status.ready is True
        assert status.state == "Ready"

    def test_not_ready(self) -> None:
        status = IndexStatus(ready=False, state="Initializing")
        assert status.ready is False
        assert status.state == "Initializing"


class TestIndexModel:
    def test_from_factory_dict(self) -> None:
        data = make_index_response()
        model = msgspec.convert(data, IndexModel)
        assert model.name == "test-index"
        assert model.host == "https://test-index-abc1234.svc.us-east1-gcp.pinecone.io"
        assert model.deletion_protection == "disabled"
        assert model.status.ready is True
        assert model.status.state == "Ready"
        assert isinstance(model.deployment, ManagedDeployment)
        assert model.deployment.cloud == "aws"
        assert model.deployment.region == "us-east-1"
        embedding = model.schema.fields["embedding"]
        assert isinstance(embedding, DenseVectorField)
        assert embedding.dimension == 1536
        assert embedding.metric == "cosine"
        assert model.tags is None

    def test_bracket_access(self) -> None:
        data = make_index_response()
        model = msgspec.convert(data, IndexModel)
        assert model["name"] == "test-index"
        assert model["host"] == model.host
        assert model["status"].ready is True
        assert model["deployment"] is model.deployment

    def test_bracket_access_missing_key(self) -> None:
        data = make_index_response()
        model = msgspec.convert(data, IndexModel)
        with pytest.raises(KeyError, match="nonexistent"):
            model["nonexistent"]

    def test_optional_tags_none(self) -> None:
        data = make_index_response()
        del data["tags"]
        model = msgspec.convert(data, IndexModel)
        assert model.tags is None

    def test_tags_null_decodes_to_none(self) -> None:
        data = make_index_response(tags=None)
        model = msgspec.convert(data, IndexModel)
        assert model.tags is None

    def test_tags_populated(self) -> None:
        data = make_index_response(tags={"env": "prod"})
        model = msgspec.convert(data, IndexModel)
        assert model.tags == {"env": "prod"}
        assert model.tags.to_dict() == {"env": "prod"}  # type: ignore[union-attr]

    def test_pod_deployment(self) -> None:
        data = make_index_response(
            deployment={
                "deployment_type": "pod",
                "environment": "us-east1-gcp",
                "pod_type": "p1.x1",
                "replicas": 2,
                "shards": 1,
            }
        )
        model = msgspec.convert(data, IndexModel)
        assert isinstance(model.deployment, PodDeployment)
        assert model.deployment.environment == "us-east1-gcp"
        assert model.deployment.pod_type == "p1.x1"
        assert model.deployment.replicas == 2
        assert model.deployment.shards == 1

    def test_byoc_deployment(self) -> None:
        data = make_index_response(
            deployment={"deployment_type": "byoc", "environment": "aws-us-east-1-b921"}
        )
        model = msgspec.convert(data, IndexModel)
        assert isinstance(model.deployment, ByocDeployment)
        assert model.deployment.environment == "aws-us-east-1-b921"

    def test_managed_deployment_environment(self) -> None:
        data = make_index_response(
            deployment={
                "deployment_type": "managed",
                "cloud": "aws",
                "region": "us-east-1",
                "environment": "aped-4627-b74a",
            }
        )
        model = msgspec.convert(data, IndexModel)
        assert isinstance(model.deployment, ManagedDeployment)
        assert model.deployment.environment == "aped-4627-b74a"

    def test_read_capacity_absent(self) -> None:
        data = make_index_response()
        del data["read_capacity"]
        model = msgspec.convert(data, IndexModel)
        assert model.read_capacity is None

    def test_host_bare_gets_https_prefix(self) -> None:
        """IndexModel normalizes bare hostname to https:// on construction."""
        data = make_index_response(host="my-index-abc.svc.pinecone.io")
        model = msgspec.convert(data, IndexModel)
        assert model.host == "https://my-index-abc.svc.pinecone.io"

    def test_host_with_https_unchanged(self) -> None:
        """IndexModel preserves an already-prefixed https:// host."""
        data = make_index_response(host="https://my-index-abc.svc.pinecone.io")
        model = msgspec.convert(data, IndexModel)
        assert model.host == "https://my-index-abc.svc.pinecone.io"

    def test_index_model_null_host(self) -> None:
        """IndexModel must decode null host from backend without raising."""
        data = make_index_response(host=None)
        model = msgspec.convert(data, IndexModel)
        assert model.host is None
        assert model.name == "test-index"

    def test_index_model_missing_host(self) -> None:
        """IndexModel must decode when host field is absent from backend response."""
        data = make_index_response()
        del data["host"]
        model = msgspec.convert(data, IndexModel)
        assert model.host is None

    def test_index_model_private_host_decoded(self) -> None:
        """IndexModel must expose private_host when returned by backend."""
        data = make_index_response(private_host="test.svc.private.pinecone.io")
        model = msgspec.convert(data, IndexModel)
        assert model.private_host == "https://test.svc.private.pinecone.io"

    def test_index_model_private_host_absent(self) -> None:
        """IndexModel.private_host is None when backend omits the field."""
        data = make_index_response()
        model = msgspec.convert(data, IndexModel)
        assert model.private_host is None

    def test_source_fields_decoded(self) -> None:
        data = make_index_response(
            source_collection="movie-embeddings",
            source_backup_id="670e8400-e29b-41d4-a716-446655440000",
            cmek_id="arn:aws:kms:us-east-1:123456789012:key/mrk-abc123",
        )
        model = msgspec.convert(data, IndexModel)
        assert model.source_collection == "movie-embeddings"
        assert model.source_backup_id == "670e8400-e29b-41d4-a716-446655440000"
        assert model.cmek_id == "arn:aws:kms:us-east-1:123456789012:key/mrk-abc123"

    @pytest.mark.parametrize("removed", ["spec", "embed", "created_at"])
    def test_removed_attribute_names_replacement(self, removed: str) -> None:
        model = msgspec.convert(make_index_response(), IndexModel)
        with pytest.raises(AttributeError, match="removed in the 2026-07"):
            getattr(model, removed)

    def test_removed_spec_hint_names_deployment(self) -> None:
        model = msgspec.convert(make_index_response(), IndexModel)
        with pytest.raises(AttributeError, match="deployment"):
            model.spec

    def test_unknown_attribute_plain_error(self) -> None:
        model = msgspec.convert(make_index_response(), IndexModel)
        with pytest.raises(AttributeError, match="no attribute"):
            model.bogus_attribute

    def test_dimension_metric_vector_type_resolved_from_dense_schema(self) -> None:
        model = msgspec.convert(make_index_response(), IndexModel)
        assert model.dimension == 1536
        assert model.metric == "cosine"
        assert model.vector_type == "dense"

    def test_dimension_raises_guided_error_with_no_vector_field(self) -> None:
        data = make_index_response(schema={"fields": {}})
        model = msgspec.convert(data, IndexModel)
        with pytest.raises(AttributeError, match="no dense or sparse vector fields"):
            model.dimension

    def test_metric_raises_guided_error_with_no_vector_field(self) -> None:
        data = make_index_response(schema={"fields": {}})
        model = msgspec.convert(data, IndexModel)
        with pytest.raises(AttributeError, match="no dense or sparse vector fields"):
            model.metric

    def test_vector_type_raises_guided_error_with_no_vector_field(self) -> None:
        data = make_index_response(schema={"fields": {}})
        model = msgspec.convert(data, IndexModel)
        with pytest.raises(AttributeError, match="no dense or sparse vector fields"):
            model.vector_type

    def test_dimension_metric_vector_type_ignore_non_vector_fields(self) -> None:
        """A schema with only metadata/text fields is treated the same as an empty one."""
        data = make_index_response(
            schema={"fields": {"category": {"type": "string", "filterable": True}}}
        )
        model = msgspec.convert(data, IndexModel)
        with pytest.raises(AttributeError, match="no dense or sparse vector fields"):
            model.dimension
        with pytest.raises(AttributeError, match="no dense or sparse vector fields"):
            model.metric
        with pytest.raises(AttributeError, match="no dense or sparse vector fields"):
            model.vector_type

    def test_vector_type_sparse_only(self) -> None:
        data = make_index_response(schema={"fields": {"sv": {"type": "sparse_vector"}}})
        model = msgspec.convert(data, IndexModel)
        assert model.vector_type == "sparse"

    def test_dimension_sparse_only_resolves_to_none(self) -> None:
        """Sparse vectors have no fixed dimension, so this is None rather than an error."""
        data = make_index_response(schema={"fields": {"sv": {"type": "sparse_vector"}}})
        model = msgspec.convert(data, IndexModel)
        assert model.dimension is None

    def test_dimension_multiple_sparse_fields_still_resolves_to_none(self) -> None:
        """Unlike metric/vector_type, multiple sparse fields aren't ambiguous for dimension —
        none of them have one, regardless of count."""
        data = make_index_response(
            schema={
                "fields": {
                    "a": {"type": "sparse_vector"},
                    "b": {"type": "sparse_vector"},
                }
            }
        )
        model = msgspec.convert(data, IndexModel)
        assert model.dimension is None

    def test_metric_sparse_only_resolves_to_dotproduct(self) -> None:
        data = make_index_response(schema={"fields": {"sv": {"type": "sparse_vector"}}})
        model = msgspec.convert(data, IndexModel)
        assert model.metric == "dotproduct"

    def test_metric_and_vector_type_raise_when_multiple_sparse_fields_and_no_dense(self) -> None:
        data = make_index_response(
            schema={
                "fields": {
                    "a": {"type": "sparse_vector"},
                    "b": {"type": "sparse_vector"},
                }
            }
        )
        model = msgspec.convert(data, IndexModel)
        with pytest.raises(AttributeError, match="ambiguous"):
            model.metric
        with pytest.raises(AttributeError, match="ambiguous"):
            model.vector_type

    def test_dense_field_takes_precedence_over_sparse_for_all_three_accessors(self) -> None:
        data = make_index_response(
            schema={
                "fields": {
                    "dv": {"type": "dense_vector", "dimension": 8, "metric": "euclidean"},
                    "sv": {"type": "sparse_vector"},
                }
            }
        )
        model = msgspec.convert(data, IndexModel)
        assert model.dimension == 8
        assert model.metric == "euclidean"
        assert model.vector_type == "dense"

    def test_vector_type_dense_with_reserved_field_name(self) -> None:
        data = make_index_response(
            schema={
                "fields": {
                    "_values": {"type": "dense_vector", "dimension": 8, "metric": "dotproduct"}
                }
            }
        )
        model = msgspec.convert(data, IndexModel)
        assert model.dimension == 8
        assert model.metric == "dotproduct"
        assert model.vector_type == "dense"

    def test_vector_type_dense_despite_synthesized_sparse_field(self) -> None:
        data = make_index_response(
            schema={
                "fields": {
                    "_values": {"type": "dense_vector", "dimension": 8, "metric": "euclidean"},
                    "_sparse_values": {"type": "sparse_vector"},
                }
            }
        )
        model = msgspec.convert(data, IndexModel)
        assert model.vector_type == "dense"
        assert model.dimension == 8
        assert model.metric == "euclidean"

    def test_dimension_raises_when_multiple_dense_fields(self) -> None:
        data = make_index_response(
            schema={
                "fields": {
                    "a": {"type": "dense_vector", "dimension": 4, "metric": "cosine"},
                    "b": {"type": "dense_vector", "dimension": 8, "metric": "cosine"},
                }
            }
        )
        model = msgspec.convert(data, IndexModel)
        with pytest.raises(AttributeError, match="ambiguous"):
            model.dimension
        with pytest.raises(AttributeError, match="ambiguous"):
            model.metric
        with pytest.raises(AttributeError, match="ambiguous"):
            model.vector_type

    def test_dense_field_takes_precedence_regardless_of_sparse_field_count(self) -> None:
        data = make_index_response(
            schema={
                "fields": {
                    "dv": {"type": "dense_vector", "dimension": 8, "metric": "euclidean"},
                    "a": {"type": "sparse_vector"},
                    "b": {"type": "sparse_vector"},
                }
            }
        )
        model = msgspec.convert(data, IndexModel)
        assert model.dimension == 8
        assert model.metric == "euclidean"
        assert model.vector_type == "dense"

    def test_multiple_dense_fields_ambiguous_even_with_sparse_fields_present(self) -> None:
        """The dense-ambiguity check runs before any sparse check, so the error names the
        dense fields regardless of how many sparse fields also exist."""
        data = make_index_response(
            schema={
                "fields": {
                    "a": {"type": "dense_vector", "dimension": 4, "metric": "cosine"},
                    "b": {"type": "dense_vector", "dimension": 8, "metric": "cosine"},
                    "sv": {"type": "sparse_vector"},
                }
            }
        )
        model = msgspec.convert(data, IndexModel)
        for accessor in ("dimension", "metric", "vector_type"):
            with pytest.raises(AttributeError, match=r"ambiguous.*dense vector fields \(a, b\)"):
                getattr(model, accessor)

    def test_index_model_dir_lists_deprecated_vector_accessors(self) -> None:
        model = msgspec.convert(make_index_response(), IndexModel)
        listing = dir(model)
        assert "dimension" in listing
        assert "metric" in listing
        assert "vector_type" in listing
        assert hasattr(model, "dimension")
        assert hasattr(model, "metric")
        assert hasattr(model, "vector_type")


_ONE_DENSE = {
    "fields": {"embedding": {"type": "dense_vector", "dimension": 1536, "metric": "cosine"}}
}
_DENSE_AND_SPARSE = {
    "fields": {
        "dv": {"type": "dense_vector", "dimension": 8, "metric": "euclidean"},
        "sv": {"type": "sparse_vector"},
    }
}
_SPARSE_ONLY = {"fields": {"sv": {"type": "sparse_vector"}}}
_TWO_DENSE = {
    "fields": {
        "a": {"type": "dense_vector", "dimension": 4, "metric": "cosine"},
        "b": {"type": "dense_vector", "dimension": 8, "metric": "cosine"},
    }
}
_TWO_SPARSE = {"fields": {"a": {"type": "sparse_vector"}, "b": {"type": "sparse_vector"}}}
_NO_VECTORS: dict[str, Any] = {"fields": {}}

_AMBIGUOUS = "ambiguous"
_UNDETERMINED = "no dense or sparse vector fields"

#: (schema, accessor, resolved value) for every shape where the accessor answers.
_RESOLVING_CASES = [
    (_ONE_DENSE, "dimension", 1536),
    (_ONE_DENSE, "metric", "cosine"),
    (_ONE_DENSE, "vector_type", "dense"),
    (_DENSE_AND_SPARSE, "dimension", 8),
    (_DENSE_AND_SPARSE, "metric", "euclidean"),
    (_DENSE_AND_SPARSE, "vector_type", "dense"),
    (_SPARSE_ONLY, "dimension", None),
    (_SPARSE_ONLY, "metric", "dotproduct"),
    (_SPARSE_ONLY, "vector_type", "sparse"),
    (_TWO_SPARSE, "dimension", None),
]

#: (schema, accessor, expected message fragment) for every shape where it raises.
_FAILING_CASES = [
    (_TWO_DENSE, "dimension", _AMBIGUOUS),
    (_TWO_DENSE, "metric", _AMBIGUOUS),
    (_TWO_DENSE, "vector_type", _AMBIGUOUS),
    (_TWO_SPARSE, "metric", _AMBIGUOUS),
    (_TWO_SPARSE, "vector_type", _AMBIGUOUS),
    (_NO_VECTORS, "dimension", _UNDETERMINED),
    (_NO_VECTORS, "metric", _UNDETERMINED),
    (_NO_VECTORS, "vector_type", _UNDETERMINED),
]


def _model_with(schema: dict[str, Any]) -> IndexModel:
    return msgspec.convert(make_index_response(schema=schema), IndexModel)


class TestLegacyVectorAccessorMappingParity:
    """Every spelling of a deprecated accessor agrees with the property.

    9.x carried ``dimension``/``metric``/``vector_type`` as struct fields, so
    ``index.metric``, ``index["metric"]``, ``"metric" in index`` and
    ``to_dict()["metric"]`` all worked. 10.0 made them computed properties;
    these tests pin that the other three spellings kept up.
    """

    @pytest.mark.parametrize("schema,accessor,expected", _RESOLVING_CASES)
    def test_attribute_resolves(self, schema: dict[str, Any], accessor: str, expected: Any) -> None:
        assert getattr(_model_with(schema), accessor) == expected

    @pytest.mark.parametrize("schema,accessor,expected", _RESOLVING_CASES)
    def test_getitem_matches_attribute(
        self, schema: dict[str, Any], accessor: str, expected: Any
    ) -> None:
        assert _model_with(schema)[accessor] == expected

    @pytest.mark.parametrize("schema,accessor,expected", _RESOLVING_CASES)
    def test_contains_is_true(self, schema: dict[str, Any], accessor: str, expected: Any) -> None:
        assert accessor in _model_with(schema)

    @pytest.mark.parametrize("schema,accessor,expected", _RESOLVING_CASES)
    def test_to_dict_carries_key(
        self, schema: dict[str, Any], accessor: str, expected: Any
    ) -> None:
        assert _model_with(schema).to_dict()[accessor] == expected

    @pytest.mark.parametrize("schema,accessor,message", _FAILING_CASES)
    def test_attribute_raises_attribute_error(
        self, schema: dict[str, Any], accessor: str, message: str
    ) -> None:
        with pytest.raises(AttributeError, match=message):
            getattr(_model_with(schema), accessor)

    @pytest.mark.parametrize("schema,accessor,message", _FAILING_CASES)
    def test_getitem_raises_key_error_carrying_the_same_text(
        self, schema: dict[str, Any], accessor: str, message: str
    ) -> None:
        with pytest.raises(KeyError, match=message):
            _model_with(schema)[accessor]

    @pytest.mark.parametrize("schema,accessor,message", _FAILING_CASES)
    def test_contains_is_false(self, schema: dict[str, Any], accessor: str, message: str) -> None:
        assert accessor not in _model_with(schema)

    @pytest.mark.parametrize("schema,accessor,message", _FAILING_CASES)
    def test_to_dict_omits_key(self, schema: dict[str, Any], accessor: str, message: str) -> None:
        assert accessor not in _model_with(schema).to_dict()

    def test_getitem_key_error_text_equals_attribute_error_text(self) -> None:
        model = _model_with(_TWO_DENSE)
        try:
            model.metric
        except AttributeError as exc:
            attribute_message = str(exc)
        with pytest.raises(KeyError) as excinfo:
            model["metric"]
        assert excinfo.value.args[0] == attribute_message


class TestRemovedKeyMappingParity:
    """``index["spec"]`` explains the removal instead of raising a bare KeyError."""

    @pytest.mark.parametrize("removed", ["spec", "embed", "created_at"])
    def test_getitem_raises_guided_key_error(self, removed: str) -> None:
        model = msgspec.convert(make_index_response(), IndexModel)
        with pytest.raises(KeyError, match="removed in the 2026-07"):
            model[removed]

    @pytest.mark.parametrize("removed", ["spec", "embed", "created_at"])
    def test_key_error_text_equals_attribute_error_text(self, removed: str) -> None:
        model = msgspec.convert(make_index_response(), IndexModel)
        try:
            getattr(model, removed)
        except AttributeError as exc:
            attribute_message = str(exc)
        with pytest.raises(KeyError) as excinfo:
            model[removed]
        assert excinfo.value.args[0] == attribute_message

    @pytest.mark.parametrize("removed", ["spec", "embed", "created_at"])
    def test_contains_is_false(self, removed: str) -> None:
        model = msgspec.convert(make_index_response(), IndexModel)
        assert removed not in model

    @pytest.mark.parametrize("removed", ["spec", "embed", "created_at"])
    def test_to_dict_omits_key(self, removed: str) -> None:
        model = msgspec.convert(make_index_response(), IndexModel)
        assert removed not in model.to_dict()

    def test_unknown_key_still_raises_bare_key_error(self) -> None:
        model = msgspec.convert(make_index_response(), IndexModel)
        with pytest.raises(KeyError) as excinfo:
            model["bogus"]
        assert excinfo.value.args[0] == "bogus"

    def test_dunder_key_rejected(self) -> None:
        model = msgspec.convert(make_index_response(), IndexModel)
        assert "__class__" not in model
        with pytest.raises(KeyError):
            model["__class__"]

    def test_non_string_key_not_contained(self) -> None:
        model = msgspec.convert(make_index_response(), IndexModel)
        assert 3 not in model


class TestToDictRoundTrip:
    def test_msgspec_convert_accepts_to_dict_output(self) -> None:
        """The derived keys must not make to_dict() output undecodable."""
        model = msgspec.convert(make_index_response(), IndexModel)
        payload = model.to_dict()
        assert {"dimension", "metric", "vector_type"} <= set(payload)

        restored = msgspec.convert(payload, IndexModel)
        assert restored.name == model.name
        assert restored.dimension == model.dimension
        assert restored.metric == model.metric

    def test_struct_field_keys_still_all_present(self) -> None:
        model = msgspec.convert(make_index_response(), IndexModel)
        assert set(IndexModel.__struct_fields__) <= set(model.to_dict())


class TestIndexList:
    def _make_list(self) -> IndexList:
        data = make_index_list_response(
            indexes=[
                make_index_response(name="index-a"),
                make_index_response(name="index-b"),
                make_index_response(name="index-c"),
            ]
        )
        indexes = [msgspec.convert(idx, IndexModel) for idx in data["indexes"]]
        return IndexList(indexes)

    def test_iteration(self) -> None:
        index_list = self._make_list()
        names = [idx.name for idx in index_list]
        assert names == ["index-a", "index-b", "index-c"]

    def test_len(self) -> None:
        index_list = self._make_list()
        assert len(index_list) == 3

    def test_getitem(self) -> None:
        index_list = self._make_list()
        assert index_list[0].name == "index-a"
        assert index_list[2].name == "index-c"

    def test_getitem_negative(self) -> None:
        index_list = self._make_list()
        assert index_list[-1].name == "index-c"

    def test_names(self) -> None:
        index_list = self._make_list()
        assert index_list.names() == ["index-a", "index-b", "index-c"]

    def test_empty_list(self) -> None:
        index_list = IndexList([])
        assert len(index_list) == 0
        assert index_list.names() == []
        assert list(index_list) == []


class TestServerlessSpec:
    def test_construct_and_encode(self) -> None:
        spec = ServerlessSpec(cloud="aws", region="us-east-1")
        encoded = msgspec.json.encode(spec)
        decoded: dict[str, Any] = msgspec.json.decode(encoded)
        assert decoded == {"cloud": "aws", "region": "us-east-1"}

    def test_asdict_minimal(self) -> None:
        spec = ServerlessSpec(cloud="aws", region="us-east-1")
        result = spec.asdict()
        assert result == {"serverless": {"cloud": "aws", "region": "us-east-1"}}

    def test_asdict_with_read_capacity(self) -> None:
        spec = ServerlessSpec(cloud="aws", region="us-east-1", read_capacity={"mode": "OnDemand"})
        result = spec.asdict()
        assert result["serverless"]["read_capacity"] == {"mode": "OnDemand"}
        assert result["serverless"]["cloud"] == "aws"
        assert result["serverless"]["region"] == "us-east-1"

    def test_asdict_with_schema(self) -> None:
        spec = ServerlessSpec(
            cloud="aws",
            region="us-east-1",
            schema={"fields": {"genre": {"type": "string"}}},
        )
        result = spec.asdict()
        assert result["serverless"]["schema"] == {"fields": {"genre": {"type": "string"}}}
        assert "read_capacity" not in result["serverless"]

    def test_asdict_with_all_optional(self) -> None:
        spec = ServerlessSpec(
            cloud="aws",
            region="us-east-1",
            read_capacity={"mode": "OnDemand"},
            schema={"fields": {"genre": {"type": "string"}}},
        )
        result = spec.asdict()
        assert result["serverless"]["read_capacity"] == {"mode": "OnDemand"}
        assert result["serverless"]["schema"] == {"fields": {"genre": {"type": "string"}}}


class TestPodSpec:
    def test_asdict_defaults(self) -> None:
        spec = PodSpec(environment="us-east1-gcp")
        result = spec.asdict()
        assert "pod" in result
        pod = result["pod"]
        assert pod["environment"] == "us-east1-gcp"
        assert pod["pod_type"] == "p1.x1"
        assert pod["replicas"] == 1
        assert pod["shards"] == 1
        assert pod["pods"] == 1
        assert "metadata_config" not in pod
        assert "source_collection" not in pod

    def test_asdict_with_metadata_config(self) -> None:
        spec = PodSpec(environment="us-east1-gcp", metadata_config={"indexed": ["genre"]})
        result = spec.asdict()
        assert result["pod"]["metadata_config"] == {"indexed": ["genre"]}
        assert "source_collection" not in result["pod"]

    def test_asdict_with_source_collection(self) -> None:
        spec = PodSpec(environment="us-east1-gcp", source_collection="my-coll")
        result = spec.asdict()
        assert result["pod"]["source_collection"] == "my-coll"
        assert "metadata_config" not in result["pod"]

    def test_asdict_with_all_optional(self) -> None:
        spec = PodSpec(
            environment="us-east1-gcp",
            metadata_config={"indexed": ["genre"]},
            source_collection="my-coll",
        )
        result = spec.asdict()
        assert result["pod"]["metadata_config"] == {"indexed": ["genre"]}
        assert result["pod"]["source_collection"] == "my-coll"

    def test_construct_with_defaults(self) -> None:
        spec = PodSpec(environment="us-east1-gcp")
        assert spec.pod_type == "p1.x1"
        assert spec.replicas == 1
        assert spec.shards == 1
        assert spec.pods == 1
        assert spec.metadata_config is None
        assert spec.source_collection is None

    def test_construct_with_overrides(self) -> None:
        spec = PodSpec(
            environment="us-east1-gcp",
            pod_type="p2.x1",
            replicas=2,
            shards=2,
            pods=4,
            metadata_config={"indexed": ["genre"]},
            source_collection="my-collection",
        )
        assert spec.pod_type == "p2.x1"
        assert spec.replicas == 2
        assert spec.pods == 4
        assert spec.metadata_config == {"indexed": ["genre"]}
        assert spec.source_collection == "my-collection"

    def test_encode(self) -> None:
        spec = PodSpec(environment="us-east1-gcp")
        encoded = msgspec.json.encode(spec)
        decoded: dict[str, Any] = msgspec.json.decode(encoded)
        assert decoded["environment"] == "us-east1-gcp"
        assert decoded["pod_type"] == "p1.x1"


class TestByocSpec:
    def test_asdict_minimal(self) -> None:
        spec = ByocSpec(environment="aws-us-east-1-b921")
        result = spec.asdict()
        assert result == {"byoc": {"environment": "aws-us-east-1-b921"}}

    def test_asdict_with_read_capacity(self) -> None:
        spec = ByocSpec(environment="aws-us-east-1-b921", read_capacity={"mode": "OnDemand"})
        result = spec.asdict()
        assert result["byoc"]["read_capacity"] == {"mode": "OnDemand"}
        assert result["byoc"]["environment"] == "aws-us-east-1-b921"

    def test_asdict_with_schema(self) -> None:
        spec = ByocSpec(environment="aws-us-east-1-b921", schema={"fields": {}})
        result = spec.asdict()
        assert result["byoc"]["schema"] == {"fields": {}}
        assert "read_capacity" not in result["byoc"]

    def test_asdict_with_all_optional(self) -> None:
        spec = ByocSpec(
            environment="aws-us-east-1-b921",
            read_capacity={"mode": "OnDemand"},
            schema={"fields": {}},
        )
        result = spec.asdict()
        assert result["byoc"]["read_capacity"] == {"mode": "OnDemand"}
        assert result["byoc"]["schema"] == {"fields": {}}

    def test_construct_and_encode(self) -> None:
        spec = ByocSpec(environment="aws-us-east-1-b921")
        encoded = msgspec.json.encode(spec)
        decoded: dict[str, Any] = msgspec.json.decode(encoded)
        assert decoded == {"environment": "aws-us-east-1-b921"}
        assert spec.environment == "aws-us-east-1-b921"

    def test_byoc_spec_with_read_capacity_on_demand(self) -> None:
        spec = ByocSpec(environment="aws-us-east-1-b921", read_capacity={"mode": "OnDemand"})
        assert spec.read_capacity == {"mode": "OnDemand"}

    def test_byoc_spec_with_read_capacity_dedicated(self) -> None:
        spec = ByocSpec(
            environment="aws-us-east-1-b921",
            read_capacity={
                "mode": "Dedicated",
                "dedicated": {
                    "node_type": "t1",
                    "scaling": "Manual",
                    "manual": {"replicas": 2, "shards": 1},
                },
            },
        )
        assert spec.read_capacity is not None
        assert spec.read_capacity["mode"] == "Dedicated"
        assert spec.read_capacity["dedicated"]["node_type"] == "t1"

    def test_byoc_spec_defaults_no_read_capacity(self) -> None:
        spec = ByocSpec(environment="aws-us-east-1-b921")
        assert spec.read_capacity is None


class TestReExports:
    """Verify models are importable from the top-level models package."""

    def test_import_from_models(self) -> None:
        from pinecone.models import (
            ByocDeployment,
            ByocSpec,
            IndexDeployment,
            IndexList,
            IndexModel,
            IndexSchema,
            IndexStatus,
            ManagedDeployment,
            PodDeployment,
            PodSpec,
            ReadCapacityResponse,
            ServerlessSpec,
        )

        for symbol in (
            ByocDeployment,
            ByocSpec,
            IndexDeployment,
            IndexList,
            IndexModel,
            IndexSchema,
            IndexStatus,
            ManagedDeployment,
            PodDeployment,
            PodSpec,
            ReadCapacityResponse,
            ServerlessSpec,
        ):
            assert symbol is not None

    def test_removed_names_gone_from_models(self) -> None:
        import pinecone.models as models

        for removed in (
            "ByocSpecInfo",
            "IndexSpec",
            "PodSpecInfo",
            "ServerlessSpecInfo",
        ):
            assert removed not in models.__all__
            with pytest.raises(AttributeError):
                getattr(models, removed)
