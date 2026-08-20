"""Tests for __contains__ and __getitem__ on model Structs.

Verifies that:
- ``"field_name" in model`` returns True for valid fields
- ``"nonexistent" in model`` returns False
- ``model["__class__"]`` raises KeyError (no leaking internals)
- ``"field_name" in model`` works without TypeError crash
"""

from __future__ import annotations

import pytest

from pinecone.models.admin.api_key import APIKeyModel, APIKeyWithSecret
from pinecone.models.admin.organization import OrganizationModel
from pinecone.models.admin.project import ProjectModel
from pinecone.models.backups.model import (
    BackupModel,
)
from pinecone.models.collections.model import CollectionModel
from pinecone.models.imports.model import ImportModel, StartImportResponse
from pinecone.models.indexes.index import IndexModel, IndexStatus
from pinecone.models.inference.embed import (
    DenseEmbedding,
    EmbeddingsList,
    EmbedUsage,
    SparseEmbedding,
)
from pinecone.models.inference.models import ModelInfo, ModelInfoSupportedParameter
from pinecone.models.inference.rerank import RankedDocument, RerankResult, RerankUsage
from pinecone.models.namespaces.models import ListNamespacesResponse, NamespaceDescription
from pinecone.models.vectors.query_aggregator import QueryNamespacesResults
from pinecone.models.vectors.responses import (
    DescribeIndexStatsResponse,
    FetchByMetadataResponse,
    FetchResponse,
    ListResponse,
    QueryResponse,
    UpdateResponse,
    UpsertRecordsResponse,
    UpsertResponse,
)
from pinecone.models.vectors.search import Hit, SearchRecordsResponse, SearchResult, SearchUsage


class TestCollectionModelDictAccess:
    def test_contains_valid_field(self) -> None:
        model = CollectionModel(name="c1", status="Ready", environment="us-east-1")
        assert "name" in model
        assert "status" in model
        assert "size" in model  # optional field still in struct_fields

    def test_contains_invalid_field(self) -> None:
        model = CollectionModel(name="c1", status="Ready", environment="us-east-1")
        assert "nonexistent" not in model

    def test_getitem_rejects_dunder(self) -> None:
        model = CollectionModel(name="c1", status="Ready", environment="us-east-1")
        with pytest.raises(KeyError, match="__class__"):
            model["__class__"]

    def test_getitem_valid_field(self) -> None:
        model = CollectionModel(name="c1", status="Ready", environment="us-east-1")
        assert model["name"] == "c1"

    def test_getitem_invalid_field(self) -> None:
        model = CollectionModel(name="c1", status="Ready", environment="us-east-1")
        with pytest.raises(KeyError):
            model["nonexistent"]


class TestImportModelDictAccess:
    def test_contains_valid_field(self) -> None:
        model = ImportModel(id="imp1", uri="s3://bucket", status="Pending", created_at="2024-01-01")
        assert "id" in model
        assert "status" in model
        assert "error" in model  # optional field

    def test_contains_invalid_field(self) -> None:
        model = ImportModel(id="imp1", uri="s3://bucket", status="Pending", created_at="2024-01-01")
        assert "nonexistent" not in model

    def test_getitem_rejects_dunder(self) -> None:
        model = ImportModel(id="imp1", uri="s3://bucket", status="Pending", created_at="2024-01-01")
        with pytest.raises(KeyError, match="__class__"):
            model["__class__"]

    def test_start_import_response_contains(self) -> None:
        resp = StartImportResponse(id="imp1")
        assert "id" in resp
        assert "nonexistent" not in resp


class TestIndexModelDictAccess:
    def test_contains_and_getitem(self) -> None:
        from pinecone.models.indexes.deployment import ManagedDeployment
        from pinecone.models.indexes.schema import DenseVectorField, IndexSchema

        model = IndexModel(
            name="idx",
            host="host.example.com",
            status=IndexStatus(ready=True, state="Ready"),
            schema=IndexSchema(
                fields={"embedding": DenseVectorField(dimension=3, metric="cosine")}
            ),
            deployment=ManagedDeployment(cloud="aws", region="us-east-1"),
            deletion_protection="disabled",
        )
        assert "name" in model
        assert "deployment" in model
        assert "nonexistent" not in model
        assert model["name"] == "idx"
        with pytest.raises(KeyError):
            model["__class__"]


class TestHitDictAccess:
    """Hit has aliases: 'id' -> id_, 'score' -> score_."""

    def test_contains_alias_fields(self) -> None:
        hit = Hit(id_="v1", score_=0.9)
        assert "id" in hit
        assert "score" in hit
        assert "fields" in hit
        assert "id_" in hit  # internal field name also works
        assert "score_" in hit

    def test_contains_invalid(self) -> None:
        hit = Hit(id_="v1", score_=0.9)
        assert "nonexistent" not in hit
        assert "__class__" not in hit

    def test_getitem_aliases(self) -> None:
        hit = Hit(id_="v1", score_=0.9)
        assert hit["id"] == "v1"
        assert hit["score"] == 0.9

    def test_getitem_rejects_dunder(self) -> None:
        hit = Hit(id_="v1", score_=0.9)
        with pytest.raises(KeyError):
            hit["__class__"]


class TestModelInfoDictAccess:
    """ModelInfo has aliases: 'name' -> model, 'description' -> short_description."""

    def test_contains_alias_fields(self) -> None:
        info = ModelInfo(
            model="embed-v1",
            short_description="test",
            type="embed",
            supported_parameters=[],
        )
        assert "name" in info  # alias for 'model'
        assert "description" in info  # alias for 'short_description'
        assert "model" in info
        assert "short_description" in info

    def test_contains_invalid(self) -> None:
        info = ModelInfo(
            model="embed-v1",
            short_description="test",
            type="embed",
            supported_parameters=[],
        )
        assert "nonexistent" not in info
        assert "__class__" not in info

    def test_getitem_aliases(self) -> None:
        info = ModelInfo(
            model="embed-v1",
            short_description="A test model",
            type="embed",
            supported_parameters=[],
        )
        assert info["name"] == "embed-v1"
        assert info["description"] == "A test model"

    def test_getitem_rejects_dunder(self) -> None:
        info = ModelInfo(
            model="embed-v1",
            short_description="test",
            type="embed",
            supported_parameters=[],
        )
        with pytest.raises(KeyError):
            info["__class__"]


class TestEmbeddingsListDictAccess:
    def test_contains_field(self) -> None:
        embeddings = EmbeddingsList(
            model="embed-v1",
            vector_type="dense",
            data=[DenseEmbedding(values=[0.1, 0.2])],
            usage=EmbedUsage(total_tokens=10),
        )
        assert "model" in embeddings
        assert "data" in embeddings
        assert "nonexistent" not in embeddings
        assert "__class__" not in embeddings

    def test_getitem_rejects_dunder(self) -> None:
        embeddings = EmbeddingsList(
            model="embed-v1",
            vector_type="dense",
            data=[DenseEmbedding(values=[0.1, 0.2])],
            usage=EmbedUsage(total_tokens=10),
        )
        with pytest.raises(KeyError):
            embeddings["__class__"]


class TestSearchRecordsResponseDictAccess:
    def test_contains_and_getitem(self) -> None:
        resp = SearchRecordsResponse(
            result=SearchResult(hits=[]),
            usage=SearchUsage(read_units=1),
        )
        assert "result" in resp
        assert "usage" in resp
        assert "nonexistent" not in resp
        with pytest.raises(KeyError):
            resp["__class__"]


class TestBackupModelDictAccess:
    def test_contains_and_getitem(self) -> None:
        model = BackupModel(
            backup_id="b1",
            source_index_name="idx",
            source_index_id="id1",
            status="Ready",
            cloud="aws",
            region="us-east-1",
        )
        assert "backup_id" in model
        assert "name" in model  # optional field
        assert "nonexistent" not in model
        with pytest.raises(KeyError):
            model["__class__"]


class TestQueryNamespacesResultsDictAccess:
    def test_contains_and_getitem(self) -> None:
        result = QueryNamespacesResults()
        assert "matches" in result
        assert "usage" in result
        assert "nonexistent" not in result
        with pytest.raises(KeyError):
            result["__class__"]


class TestResponseModelsDictAccess:
    """Test __contains__ on various response models."""

    def test_upsert_response(self) -> None:
        resp = UpsertResponse(upserted_count=5)
        assert "upserted_count" in resp
        assert "nonexistent" not in resp

    def test_query_response(self) -> None:
        resp = QueryResponse()
        assert "matches" in resp
        assert "namespace" in resp
        assert "nonexistent" not in resp

    def test_fetch_response(self) -> None:
        resp = FetchResponse()
        assert "vectors" in resp
        assert "nonexistent" not in resp

    def test_upsert_records_response(self) -> None:
        resp = UpsertRecordsResponse(record_count=3)
        assert "record_count" in resp
        assert "nonexistent" not in resp

    def test_update_response(self) -> None:
        resp = UpdateResponse()
        assert "matched_records" in resp
        assert "nonexistent" not in resp

    def test_describe_index_stats_response(self) -> None:
        resp = DescribeIndexStatsResponse()
        assert "dimension" in resp
        assert "nonexistent" not in resp

    def test_list_response(self) -> None:
        resp = ListResponse()
        assert "vectors" in resp
        assert "nonexistent" not in resp

    def test_fetch_by_metadata_response(self) -> None:
        resp = FetchByMetadataResponse()
        assert "vectors" in resp
        assert "nonexistent" not in resp


class TestRerankModelsDictAccess:
    def test_rerank_usage(self) -> None:
        usage = RerankUsage(rerank_units=5)
        assert "rerank_units" in usage
        assert "nonexistent" not in usage

    def test_ranked_document(self) -> None:
        doc = RankedDocument(index=0, score=0.9)
        assert "score" in doc
        assert "nonexistent" not in doc

    def test_rerank_result(self) -> None:
        result = RerankResult(
            model="rerank-v1",
            data=[],
            usage=RerankUsage(rerank_units=1),
        )
        assert "model" in result
        assert "nonexistent" not in result
        with pytest.raises(KeyError):
            result["__class__"]


class TestAdminModelsDictAccess:
    def test_project_model(self) -> None:
        model = ProjectModel(
            id="p1",
            name="proj",
            max_pods=10,
            force_encryption_with_cmek=False,
            organization_id="org1",
        )
        assert "name" in model
        assert "nonexistent" not in model

    def test_organization_model(self) -> None:
        model = OrganizationModel(
            id="o1",
            name="org",
            plan="Free",
            payment_status="active",
            created_at="2024-01-01",
            support_tier="basic",
        )
        assert "name" in model
        assert "nonexistent" not in model

    def test_api_key_model(self) -> None:
        model = APIKeyModel(id="k1", name="key", project_id="p1", roles=["admin"])
        assert "name" in model
        assert "nonexistent" not in model

    def test_api_key_with_secret(self) -> None:
        key = APIKeyModel(id="k1", name="key", project_id="p1", roles=["admin"])
        model = APIKeyWithSecret(key=key, value="secret")
        assert "value" in model
        assert "nonexistent" not in model


class TestNamespaceModelsDictAccess:
    def test_namespace_description(self) -> None:
        ns = NamespaceDescription(name="ns1", record_count=100)
        assert "name" in ns
        assert "nonexistent" not in ns

    def test_list_namespaces_response(self) -> None:
        resp = ListNamespacesResponse()
        assert "namespaces" in resp
        assert "nonexistent" not in resp


class TestEmbedSubModelsDictAccess:
    def test_dense_embedding(self) -> None:
        emb = DenseEmbedding(values=[0.1, 0.2])
        assert "values" in emb
        assert "nonexistent" not in emb

    def test_sparse_embedding(self) -> None:
        emb = SparseEmbedding(sparse_values=[0.1], sparse_indices=[0])
        assert "sparse_values" in emb
        assert "nonexistent" not in emb

    def test_embed_usage(self) -> None:
        usage = EmbedUsage(total_tokens=10)
        assert "total_tokens" in usage
        assert "nonexistent" not in usage


class TestModelInfoSupportedParameterDictAccess:
    def test_contains(self) -> None:
        param = ModelInfoSupportedParameter(
            parameter="input_type",
            type="one_of",
            value_type="string",
            required=False,
        )
        assert "parameter" in param
        assert "nonexistent" not in param
        with pytest.raises(KeyError):
            param["__class__"]
