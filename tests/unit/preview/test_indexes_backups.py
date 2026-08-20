"""Unit tests for PreviewIndexes.create_backup() and list_backups()."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from pinecone._internal.config import PineconeConfig
from pinecone.errors.exceptions import PineconeValueError
from pinecone.models.backups.model import BackupModel
from pinecone.models.pagination import Paginator
from pinecone.preview._internal.constants import INDEXES_API_VERSION
from pinecone.preview.indexes import PreviewIndexes

BASE_URL = "https://api.test.pinecone.io"

_BACKUP_1: dict = {
    "backup_id": "bkp-001",
    "source_index_id": "idx-001",
    "source_index_name": "my-index",
    "status": "Ready",
    "cloud": "aws",
    "region": "us-east-1",
    "created_at": "2026-01-01T00:00:00Z",
    "name": "nightly",
    "description": "Daily backup",
    "record_count": 1000,
    "namespace_count": 2,
    "size_bytes": 512000,
}

_BACKUP_2: dict = {
    "backup_id": "bkp-002",
    "source_index_id": "idx-001",
    "source_index_name": "my-index",
    "status": "Initializing",
    "cloud": "aws",
    "region": "us-east-1",
    "created_at": "2026-01-02T00:00:00Z",
}


@pytest.fixture
def indexes() -> PreviewIndexes:
    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    return PreviewIndexes(config=config)


# ── create_backup tests ────────────────────────────────────────────────────────


@respx.mock
def test_create_backup_sends_post_with_empty_body_when_no_kwargs(
    indexes: PreviewIndexes,
) -> None:
    route = respx.post(f"{BASE_URL}/indexes/my/backups").mock(
        return_value=httpx.Response(200, json=_BACKUP_1)
    )
    indexes.create_backup("my")
    assert route.called
    request = route.calls.last.request
    assert request.content == b"{}"
    assert request.headers.get("X-Pinecone-Api-Version") == INDEXES_API_VERSION


@respx.mock
def test_create_backup_sends_name_and_description(indexes: PreviewIndexes) -> None:
    route = respx.post(f"{BASE_URL}/indexes/my/backups").mock(
        return_value=httpx.Response(200, json=_BACKUP_1)
    )
    indexes.create_backup("my", name="nightly", description="Daily backup")
    request = route.calls.last.request
    body = json.loads(request.content)
    assert body == {"name": "nightly", "description": "Daily backup"}


@respx.mock
def test_create_backup_sends_name_only(indexes: PreviewIndexes) -> None:
    route = respx.post(f"{BASE_URL}/indexes/my/backups").mock(
        return_value=httpx.Response(200, json=_BACKUP_1)
    )
    indexes.create_backup("my", name="nightly")
    request = route.calls.last.request
    body = json.loads(request.content)
    assert body == {"name": "nightly"}
    assert "description" not in body


@respx.mock
def test_create_backup_returns_preview_backup_model(indexes: PreviewIndexes) -> None:
    respx.post(f"{BASE_URL}/indexes/my/backups").mock(
        return_value=httpx.Response(200, json=_BACKUP_1)
    )
    result = indexes.create_backup("my")
    assert isinstance(result, BackupModel)
    assert result.backup_id == "bkp-001"
    assert result.source_index_name == "my-index"
    assert result.status == "Ready"
    assert result.name == "nightly"


@respx.mock
def test_create_backup_empty_index_name_raises(indexes: PreviewIndexes) -> None:
    with pytest.raises(PineconeValueError):
        indexes.create_backup("")
    # No HTTP call should have been made
    assert not respx.calls


# ── list_backups tests ─────────────────────────────────────────────────────────


@respx.mock
def test_list_backups_yields_backups(indexes: PreviewIndexes) -> None:
    respx.get(f"{BASE_URL}/indexes/my/backups").mock(
        return_value=httpx.Response(
            200,
            json={"data": [_BACKUP_1, _BACKUP_2], "pagination": {"next": None}},
        )
    )
    result = indexes.list_backups("my")
    assert isinstance(result, Paginator)
    items = list(result)
    assert len(items) == 2
    assert all(isinstance(b, BackupModel) for b in items)
    assert items[0].backup_id == "bkp-001"
    assert items[1].backup_id == "bkp-002"


@respx.mock
def test_list_backups_paginates(indexes: PreviewIndexes) -> None:
    first_route = respx.get(f"{BASE_URL}/indexes/my/backups").mock(
        side_effect=[
            httpx.Response(
                200,
                json={"data": [_BACKUP_1], "pagination": {"next": "t2"}},
            ),
            httpx.Response(
                200,
                json={"data": [_BACKUP_2], "pagination": {"next": None}},
            ),
        ]
    )
    items = list(indexes.list_backups("my"))
    assert len(items) == 2
    assert items[0].backup_id == "bkp-001"
    assert items[1].backup_id == "bkp-002"

    assert first_route.call_count == 2
    second_request = first_route.calls[1].request
    assert "paginationToken=t2" in str(second_request.url)


@respx.mock
def test_list_backups_respects_limit(indexes: PreviewIndexes) -> None:
    respx.get(f"{BASE_URL}/indexes/my/backups").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [_BACKUP_1, _BACKUP_2, _BACKUP_1, _BACKUP_2, _BACKUP_1],
                "pagination": {"next": None},
            },
        )
    )
    items = list(indexes.list_backups("my", limit=2))
    assert len(items) == 2


@respx.mock
def test_list_backups_empty(indexes: PreviewIndexes) -> None:
    respx.get(f"{BASE_URL}/indexes/my/backups").mock(
        return_value=httpx.Response(200, json={"data": [], "pagination": {"next": None}})
    )
    assert indexes.list_backups("my").to_list() == []


@respx.mock
def test_list_backups_empty_index_name_raises(indexes: PreviewIndexes) -> None:
    with pytest.raises(PineconeValueError):
        indexes.list_backups("")
    assert not respx.calls


@respx.mock
def test_list_backups_rejects_non_positive_limit(indexes: PreviewIndexes) -> None:
    with pytest.raises(PineconeValueError):
        indexes.list_backups("my", limit=0)
    with pytest.raises(PineconeValueError):
        indexes.list_backups("my", limit=-1)
    assert not respx.calls
