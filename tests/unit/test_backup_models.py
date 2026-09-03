"""Tests for backup and restore models (2026-07 API).

Payloads marked "spec example" are copied verbatim from
``apis/src/release/db/control/resources`` (``DescribeBackup.yaml`` and
``indexes/ListBackups.yaml``) so a spec change breaks a test here.
"""

from __future__ import annotations

from typing import Any

import msgspec
import orjson
import pytest

from pinecone._internal.adapters.backups_adapter import BackupsAdapter
from pinecone.models.backups.list import BackupList, RestoreJobList
from pinecone.models.backups.model import (
    BackupModel,
    CreateIndexFromBackupRequest,
    CreateIndexFromBackupResponse,
    RestoreJobModel,
)
from pinecone.models.indexes.schema import (
    DenseVectorField,
    IndexSchema,
    LegacyMetadataField,
    SparseVectorField,
    StringField,
)

_DENSE_SCHEMA: dict[str, Any] = {
    "fields": {"embedding": {"type": "dense_vector", "dimension": 1536, "metric": "cosine"}}
}

SPEC_DESCRIBE_BACKUP: dict[str, Any] = {
    "backup_id": "670e8400-e29b-41d4-a716-446655440000",
    "source_index_name": "my-index",
    "source_index_id": "670e8400-e29b-41d4-a716-446655440001",
    "name": "backup_2025_03_15",
    "description": "Monthly backup of production index",
    "status": "Ready",
    "cloud": "aws",
    "region": "us-east-1",
    "schema": _DENSE_SCHEMA,
    "record_count": 120000,
    "namespace_count": 3,
    "size_bytes": 10000000,
    "tags": {"environment": "production", "type": "monthly"},
    "created_at": "2025-03-15T10:30:00Z",
}

SPEC_DELETED_SOURCE_BACKUP: dict[str, Any] = {
    "backup_id": "bkp_oldidx",
    "source_index_name": "my-index",
    "source_index_id": "idx_legacy",
    "name": "backup_before_delete",
    "description": "Backup from a deleted index that used the same name",
    "status": "Ready",
    "cloud": "aws",
    "region": "us-east-1",
    "schema": _DENSE_SCHEMA,
    "record_count": 100000,
    "namespace_count": 2,
    "size_bytes": 9000000,
    "tags": {"environment": "production"},
    "created_at": "2025-03-01T09:00:00Z",
    "source_index_deleted_at": "2025-03-05T12:00:00Z",
}


def _make_backup(**overrides: object) -> BackupModel:
    defaults: dict[str, object] = {
        "backup_id": "bkp-123",
        "source_index_name": "my-index",
        "source_index_id": "idx-456",
        "status": "Ready",
        "cloud": "aws",
        "region": "us-east-1",
    }
    defaults.update(overrides)
    return BackupModel(**defaults)  # type: ignore[arg-type]


def _make_restore_job(**overrides: object) -> RestoreJobModel:
    defaults: dict[str, object] = {
        "restore_job_id": "rj-001",
        "backup_id": "bkp-123",
        "target_index_name": "restored-index",
        "target_index_id": "idx-789",
        "status": "Running",
        "created_at": "2025-01-01T00:00:00Z",
    }
    defaults.update(overrides)
    return RestoreJobModel(**defaults)  # type: ignore[arg-type]


class TestBackupModelRequiredFields:
    def test_backup_model_required_fields(self) -> None:
        backup = _make_backup()
        assert backup.backup_id == "bkp-123"
        assert backup.source_index_name == "my-index"
        assert backup.source_index_id == "idx-456"
        assert backup.status == "Ready"
        assert backup.cloud == "aws"
        assert backup.region == "us-east-1"
        assert backup.name is None
        assert backup.description is None
        assert backup.source_index_deleted_at is None
        assert backup.schema is None
        assert backup.record_count is None
        assert backup.namespace_count is None
        assert backup.size_bytes is None
        assert backup.tags is None
        assert backup.created_at is None

    @pytest.mark.parametrize("status", ["Initializing", "Ready", "Failed"])
    def test_spec_statuses_decode(self, status: str) -> None:
        backup = msgspec.convert({**SPEC_DESCRIBE_BACKUP, "status": status}, BackupModel)
        assert backup.status == status

    def test_backend_initialization_failed_status_decodes(self) -> None:
        """pinecone-db maps a failed backup to ``InitializationFailed``, not ``Failed``."""
        backup = msgspec.convert(
            {**SPEC_DESCRIBE_BACKUP, "status": "InitializationFailed"}, BackupModel
        )
        assert backup.status == "InitializationFailed"


class TestBackupModelSpecExamples:
    def test_describe_backup_spec_example_decodes(self) -> None:
        backup = msgspec.json.decode(orjson.dumps(SPEC_DESCRIBE_BACKUP), type=BackupModel)
        assert backup.backup_id == "670e8400-e29b-41d4-a716-446655440000"
        assert backup.name == "backup_2025_03_15"
        assert backup.record_count == 120000
        assert backup.namespace_count == 3
        assert backup.size_bytes == 10000000
        assert backup.tags == {"environment": "production", "type": "monthly"}
        assert backup.created_at == "2025-03-15T10:30:00Z"
        assert backup.source_index_deleted_at is None

        assert isinstance(backup.schema, IndexSchema)
        field = backup.schema.fields["embedding"]
        assert isinstance(field, DenseVectorField)
        assert field.dimension == 1536
        assert field.metric == "cosine"

    def test_include_deleted_spec_example_carries_deletion_timestamp(self) -> None:
        backup = msgspec.json.decode(orjson.dumps(SPEC_DELETED_SOURCE_BACKUP), type=BackupModel)
        assert backup.source_index_deleted_at == "2025-03-05T12:00:00Z"
        assert backup.source_index_id == "idx_legacy"

    def test_source_index_deleted_at_absent_is_none(self) -> None:
        payload = {
            k: v for k, v in SPEC_DELETED_SOURCE_BACKUP.items() if k != "source_index_deleted_at"
        }
        backup = msgspec.json.decode(orjson.dumps(payload), type=BackupModel)
        assert backup.source_index_deleted_at is None

    def test_list_spec_example_decodes_through_adapter(self) -> None:
        payload = {
            "data": [SPEC_DESCRIBE_BACKUP, SPEC_DELETED_SOURCE_BACKUP],
            "pagination": {"next": "dXNlcl9pZD11c2VyXzE="},
        }
        result = BackupsAdapter.to_backup_list(orjson.dumps(payload))
        assert len(result) == 2
        assert result[1].source_index_deleted_at == "2025-03-05T12:00:00Z"
        assert result.pagination is not None
        assert result.pagination.next == "dXNlcl9pZD11c2VyXzE="

    def test_list_final_page_null_pagination_envelope(self) -> None:
        payload = {"data": [SPEC_DESCRIBE_BACKUP], "pagination": None}
        result = BackupsAdapter.to_backup_list(orjson.dumps(payload))
        assert len(result) == 1
        assert result.pagination is None


class TestBackupModelSchema:
    def test_typed_schema_field_variants_decode(self) -> None:
        payload = {
            **SPEC_DESCRIBE_BACKUP,
            "schema": {
                "fields": {
                    "embedding": {"type": "dense_vector", "dimension": 8, "metric": "cosine"},
                    "sparse": {"type": "sparse_vector"},
                    "title": {"type": "string", "full_text_search": {"language": "en"}},
                }
            },
        }
        backup = msgspec.convert(payload, BackupModel)
        assert backup.schema is not None
        assert isinstance(backup.schema.fields["embedding"], DenseVectorField)
        assert isinstance(backup.schema.fields["sparse"], SparseVectorField)
        title = backup.schema.fields["title"]
        assert isinstance(title, StringField)
        assert title.full_text_search is not None
        assert title.full_text_search.language == "en"

    def test_schema_null_decodes_to_none(self) -> None:
        """Schedule-produced backups of a schema-less index return ``schema: null``."""
        payload = {**SPEC_DESCRIBE_BACKUP, "schema": None}
        backup = msgspec.convert(payload, BackupModel)
        assert backup.schema is None

    def test_schema_absent_decodes_to_none(self) -> None:
        payload = {k: v for k, v in SPEC_DESCRIBE_BACKUP.items() if k != "schema"}
        backup = msgspec.convert(payload, BackupModel)
        assert backup.schema is None

    def test_legacy_metadata_schema_decodes_through_adapter(self) -> None:
        """pinecone-db still returns the untyped ``{filterable}`` schema for backups."""
        payload = {
            **SPEC_DESCRIBE_BACKUP,
            "schema": {"fields": {"genre": {"filterable": True}}},
        }
        backup = BackupsAdapter.to_backup(orjson.dumps(payload))
        assert backup.schema is not None
        genre = backup.schema.fields["genre"]
        assert isinstance(genre, LegacyMetadataField)
        assert genre.filterable is True
        assert "type" not in backup.to_dict()["schema"]["fields"]["genre"]

    def test_unparseable_body_raises_the_original_error(self) -> None:
        from pinecone.errors.exceptions import ResponseParsingError

        with pytest.raises(ResponseParsingError):
            BackupsAdapter.to_backup(b"not json")
        with pytest.raises(ResponseParsingError):
            BackupsAdapter.to_backup_list(b"not json")

    def test_unrecognised_schema_field_type_still_raises(self) -> None:
        payload = {**SPEC_DESCRIBE_BACKUP, "schema": {"fields": {"f": {"type": "from_the_future"}}}}
        from pinecone.errors.exceptions import ResponseParsingError

        with pytest.raises(ResponseParsingError):
            BackupsAdapter.to_backup(orjson.dumps(payload))

    def test_legacy_metadata_schema_decodes_in_lists(self) -> None:
        item = {**SPEC_DESCRIBE_BACKUP, "schema": {"fields": {"genre": {"filterable": False}}}}
        result = BackupsAdapter.to_backup_list(orjson.dumps({"data": [item, item]}))
        assert len(result) == 2
        for backup in result:
            assert backup.schema is not None
            assert isinstance(backup.schema.fields["genre"], LegacyMetadataField)


class TestBackupModelRemovedFields:
    @pytest.mark.parametrize("removed", ["dimension", "metric"])
    def test_removed_fields_raise_attribute_error_naming_schema_fields(self, removed: str) -> None:
        backup = msgspec.convert(SPEC_DESCRIBE_BACKUP, BackupModel)
        with pytest.raises(AttributeError) as excinfo:
            getattr(backup, removed)
        message = str(excinfo.value)
        assert f"BackupModel.{removed} was removed in the 2026-07 Pinecone API" in message
        assert "schema.fields" in message
        assert "https://sdk.pinecone.io/python/migration/v10-migration.html" in message

    @pytest.mark.parametrize("removed", ["dimension", "metric"])
    def test_removed_fields_are_not_struct_fields(self, removed: str) -> None:
        assert removed not in BackupModel.__struct_fields__
        backup = msgspec.convert(SPEC_DESCRIBE_BACKUP, BackupModel)
        assert removed not in backup
        with pytest.raises(KeyError):
            backup[removed]

    def test_wire_dimension_and_metric_are_ignored_not_fatal(self) -> None:
        """The 2026-07 backend still emits ``dimension``; decoding must not fail."""
        payload = {**SPEC_DESCRIBE_BACKUP, "dimension": 1536, "metric": "cosine"}
        backup = msgspec.convert(payload, BackupModel)
        assert backup.backup_id == SPEC_DESCRIBE_BACKUP["backup_id"]
        assert "dimension" not in backup.to_dict()

    def test_unknown_future_field_is_ignored(self) -> None:
        backup = msgspec.convert({**SPEC_DESCRIBE_BACKUP, "unknown_future": 42}, BackupModel)
        assert backup.backup_id == SPEC_DESCRIBE_BACKUP["backup_id"]

    def test_unrelated_missing_attribute_keeps_plain_message(self) -> None:
        backup = _make_backup()
        with pytest.raises(AttributeError, match="has no attribute 'nope'"):
            backup.nope


class TestBackupModelDenseDimension:
    def test_single_dense_field(self) -> None:
        backup = msgspec.convert(SPEC_DESCRIBE_BACKUP, BackupModel)
        assert backup.dense_dimension == 1536

    def test_none_when_schema_absent(self) -> None:
        assert _make_backup().dense_dimension is None

    def test_none_when_no_dense_field(self) -> None:
        payload = {**SPEC_DESCRIBE_BACKUP, "schema": {"fields": {"s": {"type": "sparse_vector"}}}}
        assert msgspec.convert(payload, BackupModel).dense_dimension is None

    def test_none_when_multiple_dense_fields(self) -> None:
        payload = {
            **SPEC_DESCRIBE_BACKUP,
            "schema": {
                "fields": {
                    "a": {"type": "dense_vector", "dimension": 8, "metric": "cosine"},
                    "b": {"type": "dense_vector", "dimension": 16, "metric": "cosine"},
                }
            },
        }
        assert msgspec.convert(payload, BackupModel).dense_dimension is None


class TestBackupModelDictAccess:
    def test_backup_model_bracket_access(self) -> None:
        backup = _make_backup()
        assert backup["backup_id"] == "bkp-123"
        assert backup["status"] == "Ready"

    def test_backup_model_bracket_access_invalid_key(self) -> None:
        backup = _make_backup()
        with pytest.raises(KeyError, match="nonexistent"):
            backup["nonexistent"]

    def test_contains_covers_new_field(self) -> None:
        assert "source_index_deleted_at" in _make_backup()

    def test_to_dict_includes_absent_optionals_as_none(self) -> None:
        d = _make_backup().to_dict()
        assert d["source_index_deleted_at"] is None
        assert d["schema"] is None
        assert d["created_at"] is None

    def test_to_dict_renders_schema_as_plain_dict(self) -> None:
        backup = msgspec.convert(SPEC_DESCRIBE_BACKUP, BackupModel)
        d = backup.to_dict()
        assert d["schema"] == {
            "fields": {
                "embedding": {
                    "type": "dense_vector",
                    "dimension": 1536,
                    "metric": "cosine",
                    "description": None,
                }
            }
        }

    def test_dir_exposes_public_names(self) -> None:
        names = dir(_make_backup())
        assert "backup_id" in names
        assert "dense_dimension" in names
        assert not any(n.startswith("_") for n in names)


class TestBackupModelReprs:
    def test_repr_includes_key_fields(self) -> None:
        backup = msgspec.convert(SPEC_DELETED_SOURCE_BACKUP, BackupModel)
        text = repr(backup)
        assert text.startswith("BackupModel(")
        assert "backup_id='bkp_oldidx'" in text
        assert "source_index_deleted_at='2025-03-05T12:00:00Z'" in text
        assert "schema_fields=1" in text
        assert text.endswith(")")

    def test_repr_html_lists_deletion_timestamp(self) -> None:
        backup = msgspec.convert(SPEC_DELETED_SOURCE_BACKUP, BackupModel)
        html = backup._repr_html_()
        assert "Source Index Deleted:" in html
        assert "2025-03-05T12:00:00Z" in html

    def test_repr_pretty_emits_text(self) -> None:
        class _Printer:
            def __init__(self) -> None:
                self.chunks: list[str] = []

            def text(self, value: str) -> None:
                self.chunks.append(value)

            def breakable(self) -> None:
                self.chunks.append(" ")

            def group(self, *args: object) -> _Printer:
                return self

            def __enter__(self) -> _Printer:
                return self

            def __exit__(self, *args: object) -> None:
                return None

        printer = _Printer()
        msgspec.convert(SPEC_DELETED_SOURCE_BACKUP, BackupModel)._repr_pretty_(printer, False)
        rendered = "".join(printer.chunks)
        assert "backup_id='bkp_oldidx'" in rendered

        cyclic = _Printer()
        _make_backup()._repr_pretty_(cyclic, True)
        assert "".join(cyclic.chunks) == "BackupModel(...)"


class TestBackupModelTags:
    def test_backup_model_non_string_tag_values_decode_without_error(self) -> None:
        """Backend tags are Option<serde_json::Value> — non-string values must not raise."""
        payload = {
            **SPEC_DESCRIBE_BACKUP,
            "tags": {"version": 3, "enabled": True, "ratio": 1.5, "nested": {"k": "v"}},
        }
        backup = msgspec.json.decode(orjson.dumps(payload), type=BackupModel)
        assert isinstance(backup.tags, dict)
        assert backup.tags["version"] == 3
        assert backup.tags["enabled"] is True
        assert backup.tags["ratio"] == 1.5
        assert backup.tags["nested"] == {"k": "v"}

    def test_backup_model_null_tags_decode_without_error(self) -> None:
        payload = {**SPEC_DESCRIBE_BACKUP, "tags": None}
        backup = msgspec.json.decode(orjson.dumps(payload), type=BackupModel)
        assert backup.tags is None


class TestCreateIndexFromBackupRequest:
    def test_on_demand_spec_example_body(self) -> None:
        """Spec example ``on-demand``: only ``name`` on the wire."""
        request = CreateIndexFromBackupRequest(name="restored-index")
        assert orjson.loads(msgspec.json.encode(request)) == {"name": "restored-index"}

    def test_dedicated_spec_example_body(self) -> None:
        """Spec example ``dedicated``: read_capacity reproduced verbatim."""
        read_capacity = {
            "mode": "Dedicated",
            "dedicated": {
                "node_type": "t1",
                "scaling": "Manual",
                "manual": {"shards": 2, "replicas": 2},
            },
        }
        request = CreateIndexFromBackupRequest(
            name="restored-drn-index", read_capacity=read_capacity
        )
        assert orjson.loads(msgspec.json.encode(request)) == {
            "name": "restored-drn-index",
            "read_capacity": read_capacity,
        }

    def test_read_capacity_omitted_when_not_provided(self) -> None:
        body = orjson.loads(msgspec.json.encode(CreateIndexFromBackupRequest(name="idx")))
        assert "read_capacity" not in body

    def test_optional_fields_serialize_when_provided(self) -> None:
        request = CreateIndexFromBackupRequest(
            name="idx", tags={"env": "prod"}, deletion_protection="enabled"
        )
        assert orjson.loads(msgspec.json.encode(request)) == {
            "name": "idx",
            "tags": {"env": "prod"},
            "deletion_protection": "enabled",
        }

    def test_name_is_required(self) -> None:
        with pytest.raises(TypeError):
            CreateIndexFromBackupRequest()  # type: ignore[call-arg]


class TestRestoreJobModelRequiredFields:
    def test_restore_job_model_required_fields(self) -> None:
        job = _make_restore_job()
        assert job.restore_job_id == "rj-001"
        assert job.backup_id == "bkp-123"
        assert job.target_index_name == "restored-index"
        assert job.target_index_id == "idx-789"
        assert job.status == "Running"
        assert job.created_at == "2025-01-01T00:00:00Z"
        assert job.completed_at is None
        assert job.percent_complete is None


class TestRestoreJobModelCompleted:
    def test_restore_job_model_completed(self) -> None:
        job = _make_restore_job(
            status="Completed",
            completed_at="2025-01-01T01:00:00Z",
            percent_complete=100.0,
        )
        assert job.status == "Completed"
        assert job.completed_at == "2025-01-01T01:00:00Z"
        assert job.percent_complete == 100.0


class TestBackupListIteration:
    def test_backup_list_iteration(self) -> None:
        b1 = _make_backup(backup_id="bkp-1", name="first")
        b2 = _make_backup(backup_id="bkp-2", name=None)
        bl = BackupList([b1, b2])

        assert len(bl) == 2
        assert bl[0] is b1
        assert bl[1] is b2
        assert list(bl) == [b1, b2]
        assert bl.names() == ["first", "bkp-2"]


class TestRestoreJobListIteration:
    def test_restore_job_list_iteration(self) -> None:
        j1 = _make_restore_job(restore_job_id="rj-1")
        j2 = _make_restore_job(restore_job_id="rj-2")
        jl = RestoreJobList([j1, j2])

        assert len(jl) == 2
        assert list(jl) == [j1, j2]
        assert jl[0] is j1


class TestCreateFromBackupResponse:
    def test_create_from_backup_response(self) -> None:
        resp = CreateIndexFromBackupResponse(
            restore_job_id="rj-100",
            index_id="idx-new",
        )
        assert resp.restore_job_id == "rj-100"
        assert resp.index_id == "idx-new"
        assert resp["restore_job_id"] == "rj-100"
        assert resp["index_id"] == "idx-new"


class TestNoDuplicateBackupSymbols:
    def test_single_backup_model_class_across_namespaces(self) -> None:
        import pinecone
        import pinecone.models
        import pinecone.models.backups

        assert pinecone.BackupModel is BackupModel
        assert pinecone.models.BackupModel is BackupModel
        assert pinecone.models.backups.BackupModel is BackupModel

    def test_create_index_from_backup_request_is_exported(self) -> None:
        import pinecone
        import pinecone.models

        assert pinecone.CreateIndexFromBackupRequest is CreateIndexFromBackupRequest
        assert pinecone.models.CreateIndexFromBackupRequest is CreateIndexFromBackupRequest
        assert "CreateIndexFromBackupRequest" in pinecone.__all__
