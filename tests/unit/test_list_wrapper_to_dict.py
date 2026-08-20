"""Tests for to_dict() on list wrapper classes."""

from __future__ import annotations

from pinecone.models.admin.api_key import APIKeyList, APIKeyModel, APIKeyRole
from pinecone.models.admin.organization import OrganizationList, OrganizationModel
from pinecone.models.admin.project import ProjectList, ProjectModel
from pinecone.models.backups.list import BackupList, RestoreJobList
from pinecone.models.backups.model import BackupModel, RestoreJobModel
from pinecone.models.collections.list import CollectionList
from pinecone.models.collections.model import CollectionModel
from pinecone.models.imports.list import ImportList
from pinecone.models.imports.model import ImportModel
from pinecone.models.indexes.deployment import ManagedDeployment
from pinecone.models.indexes.index import IndexModel, IndexStatus
from pinecone.models.indexes.list import IndexList
from pinecone.models.indexes.schema import DenseVectorField, IndexSchema
from pinecone.models.vectors.responses import Pagination


def _make_backup() -> BackupModel:
    return BackupModel(
        backup_id="b1",
        source_index_name="idx",
        source_index_id="idx-id",
        status="Ready",
        cloud="aws",
        region="us-east-1",
    )


def _make_restore_job() -> RestoreJobModel:
    return RestoreJobModel(
        restore_job_id="rj1",
        backup_id="b1",
        target_index_name="restored",
        target_index_id="idx-2",
        status="Completed",
        created_at="2025-01-01T00:00:00Z",
    )


def _make_index() -> IndexModel:
    return IndexModel(
        name="test-index",
        host="test.svc.pinecone.io",
        status=IndexStatus(ready=True, state="Ready"),
        schema=IndexSchema(fields={"embedding": DenseVectorField(dimension=3, metric="cosine")}),
        deployment=ManagedDeployment(cloud="aws", region="us-east-1"),
        deletion_protection="disabled",
    )


class TestBackupListToDict:
    def test_backup_list_to_dict_empty(self) -> None:
        result = BackupList([]).to_dict()
        assert result == {"data": []}

    def test_backup_list_to_dict_items(self) -> None:
        b1 = _make_backup()
        b2 = _make_backup()
        b2 = BackupModel(
            backup_id="b2",
            source_index_name="idx2",
            source_index_id="idx-id-2",
            status="Pending",
            cloud="gcp",
            region="us-central1",
        )
        result = BackupList([b1, b2]).to_dict()
        assert "data" in result
        assert len(result["data"]) == 2
        assert isinstance(result["data"][0], dict)
        assert isinstance(result["data"][1], dict)
        assert result["data"][0]["backup_id"] == "b1"
        assert result["data"][1]["backup_id"] == "b2"

    def test_backup_list_to_dict_with_pagination(self) -> None:
        backup = _make_backup()
        result = BackupList([backup], pagination=Pagination(next="tok123")).to_dict()
        assert "pagination" in result
        assert isinstance(result["pagination"], dict)
        assert result["pagination"]["next"] == "tok123"

    def test_backup_list_to_dict_no_pagination(self) -> None:
        result = BackupList([_make_backup()]).to_dict()
        assert "pagination" not in result


class TestRestoreJobListToDict:
    def test_restore_job_list_to_dict_empty(self) -> None:
        result = RestoreJobList([]).to_dict()
        assert result == {"data": []}

    def test_restore_job_list_to_dict(self) -> None:
        j1 = _make_restore_job()
        j2 = RestoreJobModel(
            restore_job_id="rj2",
            backup_id="b2",
            target_index_name="restored2",
            target_index_id="idx-3",
            status="Pending",
            created_at="2025-02-01T00:00:00Z",
        )
        result = RestoreJobList([j1, j2]).to_dict()
        assert len(result["data"]) == 2
        assert isinstance(result["data"][0], dict)
        assert result["data"][0]["restore_job_id"] == "rj1"
        assert result["data"][1]["restore_job_id"] == "rj2"

    def test_restore_job_list_to_dict_with_pagination(self) -> None:
        result = RestoreJobList(
            [_make_restore_job()], pagination=Pagination(next="page2")
        ).to_dict()
        assert isinstance(result["pagination"], dict)
        assert result["pagination"]["next"] == "page2"


class TestCollectionListToDict:
    def test_collection_list_to_dict_empty(self) -> None:
        result = CollectionList([]).to_dict()
        assert result == {"data": []}

    def test_collection_list_to_dict(self) -> None:
        c1 = CollectionModel(name="col1", status="Ready", environment="us-east1-gcp")
        c2 = CollectionModel(name="col2", status="Initializing", environment="us-west1-gcp")
        result = CollectionList([c1, c2]).to_dict()
        assert len(result["data"]) == 2
        assert isinstance(result["data"][0], dict)
        assert result["data"][0]["name"] == "col1"
        assert result["data"][1]["name"] == "col2"


class TestIndexListToDict:
    def test_index_list_to_dict_empty(self) -> None:
        result = IndexList([]).to_dict()
        assert result == {"data": []}

    def test_index_list_to_dict(self) -> None:
        idx = _make_index()
        result = IndexList([idx]).to_dict()
        assert len(result["data"]) == 1
        item = result["data"][0]
        assert isinstance(item, dict)
        assert item["name"] == "test-index"
        assert isinstance(item["status"], dict)
        assert item["status"]["ready"] is True
        assert isinstance(item["deployment"], dict)
        assert item["deployment"]["deployment_type"] == "managed"


class TestImportListToDict:
    def test_import_list_to_dict_empty(self) -> None:
        result = ImportList([]).to_dict()
        assert result == {"data": []}

    def test_import_list_to_dict(self) -> None:
        imp = ImportModel(
            id="op1",
            uri="s3://bucket/data",
            status="Completed",
            created_at="2025-01-01T00:00:00Z",
        )
        result = ImportList([imp]).to_dict()
        assert len(result["data"]) == 1
        assert isinstance(result["data"][0], dict)
        assert result["data"][0]["id"] == "op1"

    def test_import_list_to_dict_with_pagination(self) -> None:
        imp = ImportModel(
            id="op2",
            uri="s3://bucket/data2",
            status="Pending",
            created_at="2025-02-01T00:00:00Z",
        )
        result = ImportList([imp], pagination=Pagination(next="next-page")).to_dict()
        assert isinstance(result["pagination"], dict)
        assert result["pagination"]["next"] == "next-page"


class TestAPIKeyListToDict:
    def test_api_key_list_to_dict_empty(self) -> None:
        result = APIKeyList([]).to_dict()
        assert result == {"data": []}

    def test_api_key_list_to_dict(self) -> None:
        k1 = APIKeyModel(
            id="k1",
            name="prod-key",
            project_id="p1",
            roles=[APIKeyRole.DATA_PLANE_EDITOR],
        )
        k2 = APIKeyModel(
            id="k2",
            name="readonly-key",
            project_id="p1",
            roles=[APIKeyRole.DATA_PLANE_VIEWER],
        )
        result = APIKeyList([k1, k2]).to_dict()
        assert len(result["data"]) == 2
        assert isinstance(result["data"][0], dict)
        assert result["data"][0]["id"] == "k1"
        roles = result["data"][0]["roles"]
        assert isinstance(roles, list)
        assert all(isinstance(r, str) for r in roles)


class TestOrganizationListToDict:
    def test_organization_list_to_dict_empty(self) -> None:
        result = OrganizationList([]).to_dict()
        assert result == {"data": []}

    def test_organization_list_to_dict(self) -> None:
        o1 = OrganizationModel(
            id="org1",
            name="acme",
            plan="Enterprise",
            payment_status="active",
            created_at="2024-01-01T00:00:00Z",
            support_tier="premium",
        )
        o2 = OrganizationModel(
            id="org2",
            name="research",
            plan="Free",
            payment_status="active",
            created_at="2024-06-01T00:00:00Z",
            support_tier="basic",
        )
        result = OrganizationList([o1, o2]).to_dict()
        assert len(result["data"]) == 2
        assert isinstance(result["data"][0], dict)
        assert result["data"][0]["name"] == "acme"
        assert result["data"][1]["name"] == "research"


class TestProjectListToDict:
    def test_project_list_to_dict_empty(self) -> None:
        result = ProjectList([]).to_dict()
        assert result == {"data": []}

    def test_project_list_to_dict(self) -> None:
        p1 = ProjectModel(
            id="proj1",
            name="prod",
            max_pods=5,
            force_encryption_with_cmek=False,
            organization_id="org1",
        )
        p2 = ProjectModel(
            id="proj2",
            name="staging",
            max_pods=2,
            force_encryption_with_cmek=True,
            organization_id="org1",
        )
        result = ProjectList([p1, p2]).to_dict()
        assert len(result["data"]) == 2
        assert isinstance(result["data"][0], dict)
        assert result["data"][0]["name"] == "prod"
        assert result["data"][1]["name"] == "staging"
