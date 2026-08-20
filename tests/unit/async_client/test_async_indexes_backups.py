"""Unit tests for the index-scoped backup methods on the async ``pc.indexes``.

Async mirror of tests/unit/client/test_indexes_backups.py (#113's graduated
sync methods); #114 explicitly left these async twins to #133.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any

import httpx
import pytest
import respx

from pinecone._internal.config import PineconeConfig
from pinecone._internal.constants import CONTROL_PLANE_API_VERSION
from pinecone._internal.http_client import AsyncHTTPClient
from pinecone.async_client.indexes import AsyncIndexes
from pinecone.errors.exceptions import PineconeValueError
from pinecone.models.backups.model import BackupModel
from pinecone.models.pagination import AsyncPaginator

BASE_URL = "https://api.test.pinecone.io"


def _backup(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "backup_id": "bkp_123abc",
        "source_index_name": "my-index",
        "source_index_id": "idx_456",
        "name": "backup_2025_03_15",
        "status": "Ready",
        "cloud": "aws",
        "region": "us-east-1",
        "schema": {
            "fields": {"embedding": {"type": "dense_vector", "dimension": 1536, "metric": "cosine"}}
        },
        "created_at": "2025-03-15T10:30:00Z",
    }
    base.update(overrides)
    return base


@pytest.fixture
async def async_http_client() -> AsyncGenerator[AsyncHTTPClient]:
    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    client = AsyncHTTPClient(config, CONTROL_PLANE_API_VERSION)
    yield client
    await client.close()


@pytest.fixture
def indexes(async_http_client: AsyncHTTPClient) -> AsyncIndexes:
    return AsyncIndexes(http=async_http_client)


class TestCreateBackup:
    @respx.mock
    async def test_posts_to_the_index_scoped_path(self, indexes: AsyncIndexes) -> None:
        route = respx.post(f"{BASE_URL}/indexes/my-index/backups").mock(
            return_value=httpx.Response(201, json=_backup(status="Initializing"))
        )

        result = await indexes.create_backup("my-index")

        assert isinstance(result, BackupModel)
        assert result.status == "Initializing"
        assert route.calls.last.request.url.path == "/indexes/my-index/backups"

    @respx.mock
    async def test_omits_unset_name_and_description(self, indexes: AsyncIndexes) -> None:
        route = respx.post(f"{BASE_URL}/indexes/my-index/backups").mock(
            return_value=httpx.Response(201, json=_backup())
        )

        await indexes.create_backup("my-index")

        assert json.loads(route.calls.last.request.content) == {}

    @respx.mock
    async def test_sends_name_and_description(self, indexes: AsyncIndexes) -> None:
        route = respx.post(f"{BASE_URL}/indexes/my-index/backups").mock(
            return_value=httpx.Response(201, json=_backup(name="nightly"))
        )

        await indexes.create_backup("my-index", name="nightly", description="Daily backup")

        assert json.loads(route.calls.last.request.content) == {
            "name": "nightly",
            "description": "Daily backup",
        }

    async def test_empty_index_name_raises_before_any_http(self, indexes: AsyncIndexes) -> None:
        with pytest.raises(PineconeValueError) as exc:
            await indexes.create_backup("")

        assert "index_name" in str(exc.value)


class TestListBackups:
    @respx.mock
    async def test_returns_an_async_paginator(self, indexes: AsyncIndexes) -> None:
        respx.get(f"{BASE_URL}/indexes/my-index/backups").mock(
            return_value=httpx.Response(200, json={"data": [_backup()]})
        )

        result = indexes.list_backups("my-index")

        assert isinstance(result, AsyncPaginator)
        assert [b.backup_id async for b in result] == ["bkp_123abc"]

    @respx.mock
    async def test_terminates_on_absent_pagination_envelope(self, indexes: AsyncIndexes) -> None:
        route = respx.get(f"{BASE_URL}/indexes/my-index/backups").mock(
            return_value=httpx.Response(200, json={"data": [_backup()]})
        )

        assert len(await indexes.list_backups("my-index").to_list()) == 1
        assert route.call_count == 1

    @respx.mock
    async def test_terminates_on_null_pagination_envelope(self, indexes: AsyncIndexes) -> None:
        route = respx.get(f"{BASE_URL}/indexes/my-index/backups").mock(
            return_value=httpx.Response(200, json={"data": [_backup()], "pagination": None})
        )

        assert len(await indexes.list_backups("my-index").to_list()) == 1
        assert route.call_count == 1

    @respx.mock
    async def test_follows_the_pagination_token_across_pages(self, indexes: AsyncIndexes) -> None:
        route = respx.get(f"{BASE_URL}/indexes/my-index/backups").mock(
            side_effect=[
                httpx.Response(200, json={"data": [_backup()], "pagination": {"next": "tok-2"}}),
                httpx.Response(200, json={"data": [_backup(backup_id="bkp_2")]}),
            ]
        )

        assert [b.backup_id async for b in indexes.list_backups("my-index")] == [
            "bkp_123abc",
            "bkp_2",
        ]
        assert route.call_count == 2
        assert route.calls[1].request.url.params["paginationToken"] == "tok-2"

    @respx.mock
    async def test_include_deleted_forwarded_only_when_set(self, indexes: AsyncIndexes) -> None:
        route = respx.get(f"{BASE_URL}/indexes/my-index/backups").mock(
            return_value=httpx.Response(200, json={"data": []})
        )

        await indexes.list_backups("my-index").to_list()
        assert "include_deleted" not in route.calls.last.request.url.params

        await indexes.list_backups("my-index", include_deleted=True).to_list()
        assert route.calls.last.request.url.params["include_deleted"] == "true"

        await indexes.list_backups("my-index", include_deleted=False).to_list()
        assert route.calls.last.request.url.params["include_deleted"] == "false"

    @respx.mock
    async def test_deleted_source_backups_carry_the_deletion_timestamp(
        self, indexes: AsyncIndexes
    ) -> None:
        respx.get(f"{BASE_URL}/indexes/my-index/backups").mock(
            return_value=httpx.Response(
                200,
                json={
                    "data": [
                        _backup(),
                        _backup(
                            backup_id="bkp_oldidx",
                            source_index_deleted_at="2025-03-05T12:00:00Z",
                        ),
                    ]
                },
            )
        )

        result = await indexes.list_backups("my-index", include_deleted=True).to_list()

        assert [b.source_index_deleted_at for b in result] == [None, "2025-03-05T12:00:00Z"]

    async def test_empty_index_name_raises_before_any_http(self, indexes: AsyncIndexes) -> None:
        with pytest.raises(PineconeValueError) as exc:
            indexes.list_backups("")

        assert "index_name" in str(exc.value)

    @pytest.mark.parametrize("limit", [0, -1])
    async def test_non_positive_limit_raises_before_any_http(
        self, indexes: AsyncIndexes, limit: int
    ) -> None:
        with pytest.raises(PineconeValueError) as exc:
            indexes.list_backups("my-index", limit=limit)

        assert "limit" in str(exc.value)


class TestDescribeBackup:
    @respx.mock
    async def test_gets_the_backup_scoped_path(self, indexes: AsyncIndexes) -> None:
        route = respx.get(f"{BASE_URL}/backups/bkp_123abc").mock(
            return_value=httpx.Response(200, json=_backup())
        )

        result = await indexes.describe_backup("bkp_123abc")

        assert isinstance(result, BackupModel)
        assert route.calls.last.request.url.path == "/backups/bkp_123abc"

    async def test_empty_backup_id_raises_before_any_http(self, indexes: AsyncIndexes) -> None:
        with pytest.raises(PineconeValueError) as exc:
            await indexes.describe_backup("")

        assert "backup_id" in str(exc.value)


class TestSurfaceParity:
    @respx.mock
    async def test_index_scoped_and_project_surfaces_share_one_model(
        self, indexes: AsyncIndexes
    ) -> None:
        respx.get(f"{BASE_URL}/backups/bkp_123abc").mock(
            return_value=httpx.Response(200, json=_backup())
        )
        respx.get(f"{BASE_URL}/indexes/my-index/backups").mock(
            return_value=httpx.Response(200, json={"data": [_backup()]})
        )

        described = await indexes.describe_backup("bkp_123abc")
        listed = (await indexes.list_backups("my-index").to_list())[0]

        assert type(described) is type(listed) is BackupModel
        assert described.to_dict() == listed.to_dict()
