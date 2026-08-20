"""Unit tests for AsyncPreviewIndexes.create_backup() and list_backups()."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
import respx

from pinecone._internal.config import PineconeConfig
from pinecone.errors.exceptions import PineconeValueError
from pinecone.models.backups.model import BackupModel
from pinecone.models.pagination import AsyncPaginator
from pinecone.preview._internal.constants import INDEXES_API_VERSION
from pinecone.preview.async_indexes import AsyncPreviewIndexes

BASE_URL = "https://api.test.pinecone.io"

_BACKUP_1: dict[str, object] = {
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

_BACKUP_2: dict[str, object] = {
    "backup_id": "bkp-002",
    "source_index_id": "idx-001",
    "source_index_name": "my-index",
    "status": "Initializing",
    "cloud": "aws",
    "region": "us-east-1",
    "created_at": "2026-01-02T00:00:00Z",
}

_BACKUP_3: dict[str, object] = {
    "backup_id": "bkp-003",
    "source_index_id": "idx-001",
    "source_index_name": "my-index",
    "status": "Ready",
    "cloud": "aws",
    "region": "us-east-1",
    "created_at": "2026-01-03T00:00:00Z",
}


@pytest.fixture
def indexes() -> AsyncPreviewIndexes:
    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    return AsyncPreviewIndexes(config=config)


# ── create_backup tests ────────────────────────────────────────────────────────


@respx.mock
async def test_async_create_backup_sends_post_with_api_version_header(
    indexes: AsyncPreviewIndexes,
) -> None:
    route = respx.post(f"{BASE_URL}/indexes/foo/backups").mock(
        return_value=httpx.Response(200, json=_BACKUP_1)
    )
    await indexes.create_backup("foo")
    assert route.called
    request = route.calls.last.request
    assert str(request.url) == f"{BASE_URL}/indexes/foo/backups"
    assert request.method == "POST"
    assert request.headers.get("X-Pinecone-Api-Version") == INDEXES_API_VERSION


@respx.mock
async def test_async_create_backup_empty_body_when_no_args(
    indexes: AsyncPreviewIndexes,
) -> None:
    route = respx.post(f"{BASE_URL}/indexes/foo/backups").mock(
        return_value=httpx.Response(200, json=_BACKUP_1)
    )
    await indexes.create_backup("foo")
    request = route.calls.last.request
    assert request.content == b"{}"


@respx.mock
async def test_async_create_backup_serializes_name_and_description(
    indexes: AsyncPreviewIndexes,
) -> None:
    route = respx.post(f"{BASE_URL}/indexes/foo/backups").mock(
        return_value=httpx.Response(200, json=_BACKUP_1)
    )
    await indexes.create_backup("foo", name="nightly", description="Daily backup")
    request = route.calls.last.request
    body = json.loads(request.content)
    assert "name" in body
    assert "description" in body
    assert body["name"] == "nightly"
    assert body["description"] == "Daily backup"
    assert len(body) == 2


@respx.mock
async def test_async_create_backup_returns_preview_backup_model(
    indexes: AsyncPreviewIndexes,
) -> None:
    respx.post(f"{BASE_URL}/indexes/foo/backups").mock(
        return_value=httpx.Response(200, json=_BACKUP_1)
    )
    result = await indexes.create_backup("foo")
    assert isinstance(result, BackupModel)
    assert result.backup_id == "bkp-001"
    assert result.name == "nightly"


@respx.mock
async def test_async_create_backup_rejects_empty_index_name(
    indexes: AsyncPreviewIndexes,
) -> None:
    with pytest.raises(PineconeValueError):
        await indexes.create_backup("")
    assert not respx.calls


# ── list_backups tests ─────────────────────────────────────────────────────────


@respx.mock
async def test_async_list_backups_returns_async_paginator(
    indexes: AsyncPreviewIndexes,
) -> None:
    respx.get(f"{BASE_URL}/indexes/foo/backups").mock(
        return_value=httpx.Response(200, json={"data": [_BACKUP_1], "pagination": {"next": None}})
    )
    paginator = indexes.list_backups("foo")
    assert isinstance(paginator, AsyncPaginator)


@respx.mock
async def test_async_list_backups_iterates_items(
    indexes: AsyncPreviewIndexes,
) -> None:
    respx.get(f"{BASE_URL}/indexes/foo/backups").mock(
        return_value=httpx.Response(
            200,
            json={"data": [_BACKUP_1, _BACKUP_2, _BACKUP_3], "pagination": {"next": None}},
        )
    )
    paginator = indexes.list_backups("foo")
    items = [backup async for backup in paginator]
    assert len(items) == 3
    assert all(isinstance(b, BackupModel) for b in items)


@respx.mock
async def test_async_list_backups_sends_pagination_token_query_param(
    indexes: AsyncPreviewIndexes,
) -> None:
    route = respx.get(f"{BASE_URL}/indexes/foo/backups").mock(
        return_value=httpx.Response(200, json={"data": [_BACKUP_1], "pagination": {"next": None}})
    )
    paginator = indexes.list_backups("foo", pagination_token="tok123")
    async for _ in paginator:
        pass
    request = route.calls.last.request
    assert "paginationToken=tok123" in str(request.url)


@respx.mock
async def test_async_list_backups_terminates_on_null_next(
    indexes: AsyncPreviewIndexes,
) -> None:
    route = respx.get(f"{BASE_URL}/indexes/foo/backups").mock(
        return_value=httpx.Response(
            200,
            json={"data": [_BACKUP_1, _BACKUP_2], "pagination": {"next": None}},
        )
    )
    paginator = indexes.list_backups("foo")
    async for _ in paginator:
        pass
    assert route.call_count == 1


@respx.mock
async def test_async_list_backups_follows_next_token(
    indexes: AsyncPreviewIndexes,
) -> None:
    route = respx.get(f"{BASE_URL}/indexes/foo/backups").mock(
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
    paginator = indexes.list_backups("foo")
    items = [backup async for backup in paginator]
    assert len(items) == 2
    assert route.call_count == 2
    second_request = route.calls[1].request
    assert "paginationToken=t2" in str(second_request.url)


@respx.mock
async def test_async_list_backups_rejects_empty_index_name(
    indexes: AsyncPreviewIndexes,
) -> None:
    with pytest.raises(PineconeValueError):
        indexes.list_backups("")
    assert not respx.calls


@respx.mock
async def test_async_list_backups_rejects_non_positive_limit(
    indexes: AsyncPreviewIndexes,
) -> None:
    with pytest.raises(PineconeValueError):
        indexes.list_backups("foo", limit=0)
    assert not respx.calls


@respx.mock
async def test_async_list_backups_respects_limit(
    indexes: AsyncPreviewIndexes,
) -> None:
    """list_backups(limit=2) yields at most 2 items even when the server returns more."""
    respx.get(f"{BASE_URL}/indexes/foo/backups").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [_BACKUP_1, _BACKUP_2, _BACKUP_3, _BACKUP_1, _BACKUP_2],
                "pagination": {"next": None},
            },
        )
    )
    paginator = indexes.list_backups("foo", limit=2)
    items = [backup async for backup in paginator]
    assert len(items) == 2


def test_async_create_backup_is_coroutine() -> None:
    assert asyncio.iscoroutinefunction(AsyncPreviewIndexes.create_backup)


def test_async_list_backups_is_not_coroutine(indexes: AsyncPreviewIndexes) -> None:
    assert not asyncio.iscoroutinefunction(AsyncPreviewIndexes.list_backups)
    assert isinstance(indexes.list_backups("foo"), AsyncPaginator)
