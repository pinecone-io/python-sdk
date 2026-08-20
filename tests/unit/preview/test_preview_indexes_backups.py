"""Unit tests for PreviewIndexes / AsyncPreviewIndexes backup methods.

Covers create_backup and list_backups for both sync and async variants.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from pinecone._internal.config import PineconeConfig
from pinecone.errors.exceptions import PineconeValueError
from pinecone.models.backups.model import BackupModel
from pinecone.models.pagination import AsyncPaginator, Paginator
from pinecone.preview._internal.constants import INDEXES_API_VERSION
from pinecone.preview.async_indexes import AsyncPreviewIndexes
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


@pytest.fixture
def async_indexes() -> AsyncPreviewIndexes:
    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    return AsyncPreviewIndexes(config=config)


# ── sync create_backup ────────────────────────────────────────────────────────


@respx.mock
def test_create_backup_serializes_body(indexes: PreviewIndexes) -> None:
    route = respx.post(f"{BASE_URL}/indexes/my-index/backups").mock(
        return_value=httpx.Response(200, json=_BACKUP_1)
    )
    indexes.create_backup("my-index", name="nightly", description="Daily backup")
    body = json.loads(route.calls.last.request.content)
    assert body == {"name": "nightly", "description": "Daily backup"}
    assert route.calls.last.request.headers["X-Pinecone-Api-Version"] == INDEXES_API_VERSION


@respx.mock
def test_create_backup_omits_none_fields(indexes: PreviewIndexes) -> None:
    route = respx.post(f"{BASE_URL}/indexes/my-index/backups").mock(
        return_value=httpx.Response(200, json=_BACKUP_1)
    )
    indexes.create_backup("my-index")
    body = json.loads(route.calls.last.request.content)
    assert body == {}


@respx.mock
def test_create_backup_decodes_to_preview_backup_model(indexes: PreviewIndexes) -> None:
    respx.post(f"{BASE_URL}/indexes/my-index/backups").mock(
        return_value=httpx.Response(200, json=_BACKUP_1)
    )
    result = indexes.create_backup("my-index")
    assert isinstance(result, BackupModel)
    assert result.backup_id == "bkp-001"
    assert result.source_index_name == "my-index"
    assert result.status == "Ready"
    assert result.name == "nightly"


@respx.mock
def test_create_backup_empty_index_name_raises(indexes: PreviewIndexes) -> None:
    with pytest.raises(PineconeValueError):
        indexes.create_backup("")
    assert not respx.calls


# ── sync list_backups ─────────────────────────────────────────────────────────


@respx.mock
def test_list_backups_returns_paginator(indexes: PreviewIndexes) -> None:
    respx.get(f"{BASE_URL}/indexes/my-index/backups").mock(
        return_value=httpx.Response(200, json={"data": [_BACKUP_1], "pagination": {"next": None}})
    )
    assert isinstance(indexes.list_backups("my-index"), Paginator)


@respx.mock
def test_list_backups_iterates_across_pages(indexes: PreviewIndexes) -> None:
    route = respx.get(f"{BASE_URL}/indexes/my-index/backups").mock(
        side_effect=[
            httpx.Response(200, json={"data": [_BACKUP_1], "pagination": {"next": "tok2"}}),
            httpx.Response(200, json={"data": [_BACKUP_2], "pagination": {"next": None}}),
        ]
    )
    items = list(indexes.list_backups("my-index"))
    assert len(items) == 2
    assert all(isinstance(b, BackupModel) for b in items)
    assert route.call_count == 2
    assert "paginationToken=tok2" in str(route.calls[1].request.url)


@respx.mock
def test_list_backups_respects_limit(indexes: PreviewIndexes) -> None:
    respx.get(f"{BASE_URL}/indexes/my-index/backups").mock(
        return_value=httpx.Response(
            200, json={"data": [_BACKUP_1, _BACKUP_2], "pagination": {"next": None}}
        )
    )
    assert len(list(indexes.list_backups("my-index", limit=1))) == 1


@respx.mock
def test_list_backups_empty_index_name_raises(indexes: PreviewIndexes) -> None:
    with pytest.raises(PineconeValueError):
        indexes.list_backups("")
    assert not respx.calls


@respx.mock
def test_list_backups_forwards_limit_to_server(indexes: PreviewIndexes) -> None:
    route = respx.get(f"{BASE_URL}/indexes/my-index/backups").mock(
        return_value=httpx.Response(200, json={"data": [_BACKUP_1], "pagination": {"next": None}})
    )
    list(indexes.list_backups("my-index", limit=2))
    assert "limit=2" in str(route.calls.last.request.url)


@respx.mock
def test_list_backups_no_limit_omits_limit_param(indexes: PreviewIndexes) -> None:
    route = respx.get(f"{BASE_URL}/indexes/my-index/backups").mock(
        return_value=httpx.Response(200, json={"data": [_BACKUP_1], "pagination": {"next": None}})
    )
    list(indexes.list_backups("my-index"))
    assert "limit" not in str(route.calls.last.request.url)


# ── async create_backup ───────────────────────────────────────────────────────


@respx.mock
async def test_async_create_backup_serializes_body(
    async_indexes: AsyncPreviewIndexes,
) -> None:
    route = respx.post(f"{BASE_URL}/indexes/my-index/backups").mock(
        return_value=httpx.Response(200, json=_BACKUP_1)
    )
    await async_indexes.create_backup("my-index", name="nightly", description="Daily backup")
    body = json.loads(route.calls.last.request.content)
    assert body == {"name": "nightly", "description": "Daily backup"}
    assert route.calls.last.request.headers["X-Pinecone-Api-Version"] == INDEXES_API_VERSION


@respx.mock
async def test_async_create_backup_decodes_to_preview_backup_model(
    async_indexes: AsyncPreviewIndexes,
) -> None:
    respx.post(f"{BASE_URL}/indexes/my-index/backups").mock(
        return_value=httpx.Response(200, json=_BACKUP_1)
    )
    result = await async_indexes.create_backup("my-index")
    assert isinstance(result, BackupModel)
    assert result.backup_id == "bkp-001"
    assert result.status == "Ready"


@respx.mock
async def test_async_create_backup_empty_index_name_raises(
    async_indexes: AsyncPreviewIndexes,
) -> None:
    with pytest.raises(PineconeValueError):
        await async_indexes.create_backup("")
    assert not respx.calls


# ── async list_backups ────────────────────────────────────────────────────────


@respx.mock
async def test_async_list_backups_returns_async_paginator(
    async_indexes: AsyncPreviewIndexes,
) -> None:
    respx.get(f"{BASE_URL}/indexes/my-index/backups").mock(
        return_value=httpx.Response(200, json={"data": [_BACKUP_1], "pagination": {"next": None}})
    )
    assert isinstance(async_indexes.list_backups("my-index"), AsyncPaginator)


@respx.mock
async def test_async_list_backups_iterates_across_pages(
    async_indexes: AsyncPreviewIndexes,
) -> None:
    route = respx.get(f"{BASE_URL}/indexes/my-index/backups").mock(
        side_effect=[
            httpx.Response(200, json={"data": [_BACKUP_1], "pagination": {"next": "tok2"}}),
            httpx.Response(200, json={"data": [_BACKUP_2], "pagination": {"next": None}}),
        ]
    )
    paginator = async_indexes.list_backups("my-index")
    items = [b async for b in paginator]
    assert len(items) == 2
    assert all(isinstance(b, BackupModel) for b in items)
    assert route.call_count == 2
    assert "paginationToken=tok2" in str(route.calls[1].request.url)


@respx.mock
async def test_async_list_backups_empty_index_name_raises(
    async_indexes: AsyncPreviewIndexes,
) -> None:
    with pytest.raises(PineconeValueError):
        async_indexes.list_backups("")
    assert not respx.calls


@respx.mock
async def test_async_list_backups_forwards_limit_to_server(
    async_indexes: AsyncPreviewIndexes,
) -> None:
    route = respx.get(f"{BASE_URL}/indexes/my-index/backups").mock(
        return_value=httpx.Response(200, json={"data": [_BACKUP_1], "pagination": {"next": None}})
    )
    items = [b async for b in async_indexes.list_backups("my-index", limit=2)]
    assert len(items) == 1
    assert "limit=2" in str(route.calls.last.request.url)


@respx.mock
async def test_async_list_backups_no_limit_omits_limit_param(
    async_indexes: AsyncPreviewIndexes,
) -> None:
    route = respx.get(f"{BASE_URL}/indexes/my-index/backups").mock(
        return_value=httpx.Response(200, json={"data": [_BACKUP_1], "pagination": {"next": None}})
    )
    items = [b async for b in async_indexes.list_backups("my-index")]
    assert len(items) == 1
    assert "limit" not in str(route.calls.last.request.url)


# ── sync describe_backup ──────────────────────────────────────────────────────


@respx.mock
def test_describe_backup_returns_model(indexes: PreviewIndexes) -> None:
    respx.get(f"{BASE_URL}/backups/bkp-001").mock(return_value=httpx.Response(200, json=_BACKUP_1))
    result = indexes.describe_backup("bkp-001")
    assert isinstance(result, BackupModel)
    assert result.backup_id == "bkp-001"
    assert result.status == "Ready"
    assert result.source_index_name == "my-index"


@respx.mock
def test_describe_backup_sends_correct_api_version(indexes: PreviewIndexes) -> None:
    route = respx.get(f"{BASE_URL}/backups/bkp-001").mock(
        return_value=httpx.Response(200, json=_BACKUP_1)
    )
    indexes.describe_backup("bkp-001")
    assert route.calls.last.request.headers["X-Pinecone-Api-Version"] == INDEXES_API_VERSION


@respx.mock
def test_describe_backup_empty_id_raises(indexes: PreviewIndexes) -> None:
    with pytest.raises(PineconeValueError):
        indexes.describe_backup("")
    assert not respx.calls


@respx.mock
def test_describe_backup_minimal_response(indexes: PreviewIndexes) -> None:
    respx.get(f"{BASE_URL}/backups/bkp-002").mock(return_value=httpx.Response(200, json=_BACKUP_2))
    result = indexes.describe_backup("bkp-002")
    assert isinstance(result, BackupModel)
    assert result.backup_id == "bkp-002"
    assert result.name is None
    assert result.description is None


# ── async describe_backup ─────────────────────────────────────────────────────


@respx.mock
async def test_async_describe_backup_returns_model(
    async_indexes: AsyncPreviewIndexes,
) -> None:
    respx.get(f"{BASE_URL}/backups/bkp-001").mock(return_value=httpx.Response(200, json=_BACKUP_1))
    result = await async_indexes.describe_backup("bkp-001")
    assert isinstance(result, BackupModel)
    assert result.backup_id == "bkp-001"
    assert result.status == "Ready"


@respx.mock
async def test_async_describe_backup_sends_correct_api_version(
    async_indexes: AsyncPreviewIndexes,
) -> None:
    route = respx.get(f"{BASE_URL}/backups/bkp-001").mock(
        return_value=httpx.Response(200, json=_BACKUP_1)
    )
    await async_indexes.describe_backup("bkp-001")
    assert route.calls.last.request.headers["X-Pinecone-Api-Version"] == INDEXES_API_VERSION


@respx.mock
async def test_async_describe_backup_empty_id_raises(
    async_indexes: AsyncPreviewIndexes,
) -> None:
    with pytest.raises(PineconeValueError):
        await async_indexes.describe_backup("")
    assert not respx.calls


@respx.mock
async def test_async_describe_backup_minimal_response(
    async_indexes: AsyncPreviewIndexes,
) -> None:
    respx.get(f"{BASE_URL}/backups/bkp-002").mock(return_value=httpx.Response(200, json=_BACKUP_2))
    result = await async_indexes.describe_backup("bkp-002")
    assert isinstance(result, BackupModel)
    assert result.backup_id == "bkp-002"
    assert result.name is None
