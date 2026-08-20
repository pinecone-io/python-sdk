"""Integration tests for the graduated 2026-07 backup endpoints (#113).

Runs against whatever PINECONE_API_KEY / PINECONE_CONTROLLER_HOST point to —
a real project or a local minicone (`PINECONE_API_KEY=mockkey
PINECONE_CONTROLLER_HOST=http://127.0.0.1:5080`). Until #112 flips
CONTROL_PLANE_API_VERSION to 2026-07, version-dispatching servers will
reject these requests; that skew is the planned intermediate state.

Migrated from tests/integration/preview/test_backups.py onto the graduated
names; the spec-shape migration of tests/integration/test_backups.py belongs
to #174.
"""

from __future__ import annotations

import contextlib
import uuid
from collections.abc import Generator
from typing import Any

import pytest

from pinecone import Pinecone
from pinecone.errors import NotFoundError, PineconeError, PineconeValueError
from pinecone.models.backups.list import BackupList
from pinecone.models.backups.model import BackupModel
from tests.integration.conftest import poll_until

pytestmark = [pytest.mark.integration]

_DENSE_SCHEMA: dict[str, Any] = {
    "fields": {"embedding": {"type": "dense_vector", "dimension": 4, "metric": "cosine"}}
}


@pytest.fixture
def index_name(client: Pinecone) -> Generator[str, None, None]:
    name = f"g113-{uuid.uuid4().hex[:10]}"
    yield name
    with contextlib.suppress(NotFoundError, PineconeError):
        client.indexes.delete(name, timeout=120)


def _ready_backup(client: Pinecone, backup_id: str) -> BackupModel:
    result = poll_until(
        query_fn=lambda: client.backups.describe(backup_id=backup_id),
        check_fn=lambda b: b.status == "Ready",
        timeout=300,
        interval=10,
        description="backup Ready",
    )
    assert isinstance(result, BackupModel)
    return result


def test_index_scoped_backup_lifecycle(client: Pinecone, index_name: str) -> None:
    client.indexes.create(name=index_name, schema=_DENSE_SCHEMA, timeout=300)

    created = client.indexes.create_backup(index_name, name=f"{index_name}-bk")
    assert isinstance(created, BackupModel)
    assert created.source_index_name == index_name

    backup_id = created.backup_id
    try:
        ready = _ready_backup(client, backup_id)
        assert ready.schema is not None
        assert ready.dense_dimension in (4, None)
        assert ready.source_index_deleted_at is None

        via_indexes = client.indexes.describe_backup(backup_id)
        assert via_indexes.backup_id == backup_id

        listed = client.indexes.list_backups(index_name).to_list()
        assert backup_id in [b.backup_id for b in listed]

        project = client.backups.list()
        assert isinstance(project, BackupList)
        assert backup_id in [b.backup_id for b in project]
    finally:
        with contextlib.suppress(NotFoundError, PineconeError):
            client.backups.delete(backup_id=backup_id)


def test_include_deleted_recovers_backups_of_a_deleted_index(
    client: Pinecone, index_name: str
) -> None:
    client.indexes.create(name=index_name, schema=_DENSE_SCHEMA, timeout=300)
    backup_id = client.backups.create(index_name=index_name).backup_id

    try:
        _ready_backup(client, backup_id)
        client.indexes.delete(index_name, timeout=120)

        with pytest.raises(NotFoundError):
            client.backups.list(index_name=index_name)

        widened = client.backups.list(index_name=index_name, include_deleted=True)
        orphan = next(b for b in widened if b.backup_id == backup_id)
        assert orphan.source_index_deleted_at is not None
    finally:
        with contextlib.suppress(NotFoundError, PineconeError):
            client.backups.delete(backup_id=backup_id)


def test_include_deleted_is_rejected_on_the_project_wide_listing(client: Pinecone) -> None:
    with pytest.raises(PineconeValueError):
        client.backups.list(include_deleted=True)


def test_restore_onto_dedicated_read_capacity(client: Pinecone, index_name: str) -> None:
    client.indexes.create(name=index_name, schema=_DENSE_SCHEMA, timeout=300)
    backup_id = client.backups.create(index_name=index_name).backup_id
    restored = f"{index_name}-r"

    try:
        _ready_backup(client, backup_id)
        result = client.create_index_from_backup(
            name=restored,
            backup_id=backup_id,
            read_capacity={
                "mode": "Dedicated",
                "dedicated": {
                    "node_type": "t1",
                    "scaling": "Manual",
                    "manual": {"shards": 2, "replicas": 2},
                },
            },
            timeout=-1,
        )
        assert result.restore_job_id
        assert result.index_id
    finally:
        with contextlib.suppress(NotFoundError, PineconeError):
            client.indexes.delete(restored, timeout=120)
        with contextlib.suppress(NotFoundError, PineconeError):
            client.backups.delete(backup_id=backup_id)
