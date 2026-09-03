"""Unit tests for Pinecone.create_index_from_backup."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import httpx
import orjson
import pytest
import respx

from pinecone._client import Pinecone
from pinecone.errors.exceptions import PineconeTypeError, ValidationError
from pinecone.models.backups.model import CreateIndexFromBackupResponse
from pinecone.models.indexes.index import IndexModel
from tests.factories import make_index_response

BASE_URL = "https://api.pinecone.io"


@pytest.fixture
def pc() -> Pinecone:
    return Pinecone(api_key="test-key")


# ---------------------------------------------------------------------------
# Basic success
# ---------------------------------------------------------------------------


@respx.mock
def test_create_index_from_backup_basic(pc: Pinecone) -> None:
    """POST creates the index, then polling returns a ready IndexModel."""
    respx.post(f"{BASE_URL}/backups/bk-123/create-index").mock(
        return_value=httpx.Response(
            202,
            json={"restore_job_id": "rj-1", "index_id": "idx-1"},
        ),
    )
    respx.get(f"{BASE_URL}/indexes/restored-index").mock(
        return_value=httpx.Response(200, json=make_index_response(name="restored-index")),
    )

    result = pc.create_index_from_backup(name="restored-index", backup_id="bk-123")

    assert isinstance(result, IndexModel)
    assert result.name == "restored-index"


# ---------------------------------------------------------------------------
# Tags and deletion protection
# ---------------------------------------------------------------------------


@respx.mock
def test_create_index_from_backup_with_tags_and_protection(pc: Pinecone) -> None:
    """Tags and deletion_protection appear in the request body."""
    route = respx.post(f"{BASE_URL}/backups/bk-456/create-index").mock(
        return_value=httpx.Response(
            202,
            json={"restore_job_id": "rj-2", "index_id": "idx-2"},
        ),
    )
    respx.get(f"{BASE_URL}/indexes/my-restored").mock(
        return_value=httpx.Response(200, json=make_index_response(name="my-restored")),
    )

    pc.create_index_from_backup(
        name="my-restored",
        backup_id="bk-456",
        deletion_protection="enabled",
        tags={"env": "prod"},
    )

    request = route.calls[0].request
    import orjson

    body = orjson.loads(request.content)
    assert body["name"] == "my-restored"
    assert body["deletion_protection"] == "enabled"
    assert body["tags"] == {"env": "prod"}


# ---------------------------------------------------------------------------
# No-poll (timeout=-1)
# ---------------------------------------------------------------------------


@respx.mock
def test_create_index_from_backup_no_poll(pc: Pinecone) -> None:
    """When timeout=-1, returns CreateIndexFromBackupResponse immediately without polling."""
    respx.post(f"{BASE_URL}/backups/bk-789/create-index").mock(
        return_value=httpx.Response(
            202,
            json={"restore_job_id": "rj-3", "index_id": "idx-3"},
        ),
    )

    result = pc.create_index_from_backup(name="quick-restore", backup_id="bk-789", timeout=-1)

    assert isinstance(result, CreateIndexFromBackupResponse)
    assert result.restore_job_id == "rj-3"
    assert result.index_id == "idx-3"


@respx.mock
def test_create_index_from_backup_no_wait_returns_restore_job_id(pc: Pinecone) -> None:
    """timeout=-1 gives callers access to restore_job_id without polling."""
    respx.post(f"{BASE_URL}/backups/bk-nwt/create-index").mock(
        return_value=httpx.Response(
            202,
            json={"restore_job_id": "rj-nowait", "index_id": "idx-nowait"},
        ),
    )

    result = pc.create_index_from_backup(
        name="test-restore-nowait",
        backup_id="bk-nwt",
        timeout=-1,
    )

    assert isinstance(result, CreateIndexFromBackupResponse)
    assert result.restore_job_id
    assert result.index_id


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_create_index_from_backup_empty_name_raises(pc: Pinecone) -> None:
    with pytest.raises(ValidationError) as exc_info:
        pc.create_index_from_backup(name="", backup_id="bk-123")

    assert "name" in str(exc_info.value)
    assert "non-empty" in str(exc_info.value)


def test_create_index_from_backup_empty_backup_id_raises(pc: Pinecone) -> None:
    with pytest.raises(ValidationError) as exc_info:
        pc.create_index_from_backup(name="my-index", backup_id="")

    assert "backup_id" in str(exc_info.value)
    assert "non-empty" in str(exc_info.value)


def test_create_index_from_backup_empty_read_capacity_raises(pc: Pinecone) -> None:
    with pytest.raises(ValidationError) as exc_info:
        pc.create_index_from_backup(name="my-index", backup_id="bk-1", read_capacity={})

    assert "read_capacity" in str(exc_info.value)


# ---------------------------------------------------------------------------
# read_capacity (2026-07)
# ---------------------------------------------------------------------------

DEDICATED_READ_CAPACITY: dict[str, Any] = {
    "mode": "Dedicated",
    "dedicated": {
        "node_type": "t1",
        "scaling": "Manual",
        "manual": {"shards": 2, "replicas": 2},
    },
}


@respx.mock
def test_create_index_from_backup_emits_the_spec_dedicated_example(pc: Pinecone) -> None:
    """The body matches the OAS 'dedicated' example for create_index_from_backup_operation.

    Spec: apis/_build/2026-07/db_control_2026-07.oas.yaml:1955-1968 @ 5f808858.
    """
    route = respx.post(f"{BASE_URL}/backups/bk-drn/create-index").mock(
        return_value=httpx.Response(202, json={"restore_job_id": "rj-drn", "index_id": "idx-drn"}),
    )

    pc.create_index_from_backup(
        name="restored-drn-index",
        backup_id="bk-drn",
        read_capacity=DEDICATED_READ_CAPACITY,
        timeout=-1,
    )

    assert orjson.loads(route.calls.last.request.content) == {
        "name": "restored-drn-index",
        "read_capacity": DEDICATED_READ_CAPACITY,
    }


@respx.mock
def test_create_index_from_backup_omits_read_capacity_when_not_passed(pc: Pinecone) -> None:
    route = respx.post(f"{BASE_URL}/backups/bk-ondemand/create-index").mock(
        return_value=httpx.Response(202, json={"restore_job_id": "rj-1", "index_id": "idx-1"}),
    )

    pc.create_index_from_backup(name="restored-index", backup_id="bk-ondemand", timeout=-1)

    assert orjson.loads(route.calls.last.request.content) == {"name": "restored-index"}


@respx.mock
def test_create_index_from_backup_body_carries_no_null_keys(pc: Pinecone) -> None:
    route = respx.post(f"{BASE_URL}/backups/bk-nulls/create-index").mock(
        return_value=httpx.Response(202, json={"restore_job_id": "rj-1", "index_id": "idx-1"}),
    )

    pc.create_index_from_backup(name="restored-index", backup_id="bk-nulls", timeout=-1)

    body = orjson.loads(route.calls.last.request.content)
    assert set(body) == {"name"}
    assert None not in body.values()


@respx.mock
def test_create_index_from_backup_sends_json_content_type(pc: Pinecone) -> None:
    route = respx.post(f"{BASE_URL}/backups/bk-ct/create-index").mock(
        return_value=httpx.Response(202, json={"restore_job_id": "rj-1", "index_id": "idx-1"}),
    )

    pc.create_index_from_backup(name="restored-index", backup_id="bk-ct", timeout=-1)

    assert route.calls.last.request.headers["Content-Type"] == "application/json"


@respx.mock
def test_create_index_from_backup_all_optionals_together(pc: Pinecone) -> None:
    route = respx.post(f"{BASE_URL}/backups/bk-all/create-index").mock(
        return_value=httpx.Response(202, json={"restore_job_id": "rj-1", "index_id": "idx-1"}),
    )

    pc.create_index_from_backup(
        name="restored-index",
        backup_id="bk-all",
        deletion_protection="enabled",
        tags={"env": "prod"},
        read_capacity={"mode": "OnDemand"},
        timeout=-1,
    )

    assert orjson.loads(route.calls.last.request.content) == {
        "name": "restored-index",
        "tags": {"env": "prod"},
        "deletion_protection": "enabled",
        "read_capacity": {"mode": "OnDemand"},
    }


# ---------------------------------------------------------------------------
# create_index steers callers here (#144 stance, shipped by #131)
# ---------------------------------------------------------------------------


def test_create_index_source_backup_id_points_at_this_method(pc: Pinecone) -> None:
    with pytest.raises(PineconeTypeError) as exc_info:
        pc.create_index(name="restored", source_backup_id="bk-123")

    assert "create_index_from_backup" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Polling
# ---------------------------------------------------------------------------


@patch("pinecone._internal.indexes_helpers.time.sleep")
@respx.mock
def test_create_index_from_backup_polls_until_ready(mock_sleep: object, pc: Pinecone) -> None:
    """Describe is called multiple times until the index becomes ready."""
    respx.post(f"{BASE_URL}/backups/bk-poll/create-index").mock(
        return_value=httpx.Response(
            202,
            json={"restore_job_id": "rj-4", "index_id": "idx-4"},
        ),
    )
    not_ready = make_index_response(
        name="poll-index",
        status={"ready": False, "state": "Initializing"},
    )
    ready = make_index_response(
        name="poll-index",
        status={"ready": True, "state": "Ready"},
    )
    describe_route = respx.get(f"{BASE_URL}/indexes/poll-index").mock(
        side_effect=[
            httpx.Response(200, json=not_ready),
            httpx.Response(200, json=ready),
        ]
    )

    result = pc.create_index_from_backup(name="poll-index", backup_id="bk-poll", timeout=60)

    assert isinstance(result, IndexModel)
    assert result.name == "poll-index"
    # Describe should be called at least 2 times (first not-ready, then ready)
    assert describe_route.call_count >= 2
