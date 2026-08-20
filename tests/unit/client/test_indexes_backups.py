"""Unit tests for the index-scoped backup methods on ``pc.indexes``.

Migrated from ``tests/unit/preview/test_indexes_backups.py`` and
``tests/unit/preview/test_preview_indexes_backups.py``; the preview copies
stay with the preview code until it is retired (#140).
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx

from pinecone._internal.config import PineconeConfig
from pinecone._internal.constants import CONTROL_PLANE_API_VERSION
from pinecone._internal.http_client import HTTPClient
from pinecone.client.indexes import Indexes
from pinecone.errors.exceptions import PineconeValueError
from pinecone.models.backups.model import BackupModel
from pinecone.models.pagination import Paginator

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
def indexes() -> Indexes:
    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    return Indexes(http=HTTPClient(config, CONTROL_PLANE_API_VERSION))


class TestCreateBackup:
    @respx.mock
    def test_posts_to_the_index_scoped_path(self, indexes: Indexes) -> None:
        route = respx.post(f"{BASE_URL}/indexes/my-index/backups").mock(
            return_value=httpx.Response(201, json=_backup(status="Initializing"))
        )

        result = indexes.create_backup("my-index")

        assert isinstance(result, BackupModel)
        assert result.status == "Initializing"
        assert route.calls.last.request.url.path == "/indexes/my-index/backups"

    @respx.mock
    def test_omits_unset_name_and_description(self, indexes: Indexes) -> None:
        route = respx.post(f"{BASE_URL}/indexes/my-index/backups").mock(
            return_value=httpx.Response(201, json=_backup())
        )

        indexes.create_backup("my-index")

        assert json.loads(route.calls.last.request.content) == {}

    @respx.mock
    def test_sends_name_and_description(self, indexes: Indexes) -> None:
        route = respx.post(f"{BASE_URL}/indexes/my-index/backups").mock(
            return_value=httpx.Response(201, json=_backup(name="nightly"))
        )

        indexes.create_backup("my-index", name="nightly", description="Daily backup")

        assert json.loads(route.calls.last.request.content) == {
            "name": "nightly",
            "description": "Daily backup",
        }

    def test_empty_index_name_raises_before_any_http(self, indexes: Indexes) -> None:
        with pytest.raises(PineconeValueError) as exc:
            indexes.create_backup("")

        assert "index_name" in str(exc.value)


class TestListBackups:
    @respx.mock
    def test_returns_a_paginator(self, indexes: Indexes) -> None:
        respx.get(f"{BASE_URL}/indexes/my-index/backups").mock(
            return_value=httpx.Response(200, json={"data": [_backup()]})
        )

        result = indexes.list_backups("my-index")

        assert isinstance(result, Paginator)
        assert [b.backup_id for b in result] == ["bkp_123abc"]

    @respx.mock
    def test_terminates_on_absent_pagination_envelope(self, indexes: Indexes) -> None:
        route = respx.get(f"{BASE_URL}/indexes/my-index/backups").mock(
            return_value=httpx.Response(200, json={"data": [_backup()]})
        )

        assert len(indexes.list_backups("my-index").to_list()) == 1
        assert route.call_count == 1

    @respx.mock
    def test_terminates_on_null_pagination_envelope(self, indexes: Indexes) -> None:
        route = respx.get(f"{BASE_URL}/indexes/my-index/backups").mock(
            return_value=httpx.Response(200, json={"data": [_backup()], "pagination": None})
        )

        assert len(indexes.list_backups("my-index").to_list()) == 1
        assert route.call_count == 1

    @respx.mock
    def test_follows_the_pagination_token_across_pages(self, indexes: Indexes) -> None:
        route = respx.get(f"{BASE_URL}/indexes/my-index/backups").mock(
            side_effect=[
                httpx.Response(200, json={"data": [_backup()], "pagination": {"next": "tok-2"}}),
                httpx.Response(200, json={"data": [_backup(backup_id="bkp_2")]}),
            ]
        )

        assert [b.backup_id for b in indexes.list_backups("my-index")] == ["bkp_123abc", "bkp_2"]
        assert route.call_count == 2
        assert route.calls[1].request.url.params["paginationToken"] == "tok-2"

    @respx.mock
    def test_include_deleted_forwarded_only_when_set(self, indexes: Indexes) -> None:
        route = respx.get(f"{BASE_URL}/indexes/my-index/backups").mock(
            return_value=httpx.Response(200, json={"data": []})
        )

        indexes.list_backups("my-index").to_list()
        assert "include_deleted" not in route.calls.last.request.url.params

        indexes.list_backups("my-index", include_deleted=True).to_list()
        assert route.calls.last.request.url.params["include_deleted"] == "true"

        indexes.list_backups("my-index", include_deleted=False).to_list()
        assert route.calls.last.request.url.params["include_deleted"] == "false"

    @respx.mock
    def test_deleted_source_backups_carry_the_deletion_timestamp(self, indexes: Indexes) -> None:
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

        result = indexes.list_backups("my-index", include_deleted=True).to_list()

        assert [b.source_index_deleted_at for b in result] == [None, "2025-03-05T12:00:00Z"]

    def test_empty_index_name_raises_before_any_http(self, indexes: Indexes) -> None:
        with pytest.raises(PineconeValueError) as exc:
            indexes.list_backups("")

        assert "index_name" in str(exc.value)

    @pytest.mark.parametrize("limit", [0, -1])
    def test_non_positive_limit_raises_before_any_http(self, indexes: Indexes, limit: int) -> None:
        with pytest.raises(PineconeValueError) as exc:
            indexes.list_backups("my-index", limit=limit)

        assert "limit" in str(exc.value)


class TestDescribeBackup:
    @respx.mock
    def test_gets_the_backup_scoped_path(self, indexes: Indexes) -> None:
        route = respx.get(f"{BASE_URL}/backups/bkp_123abc").mock(
            return_value=httpx.Response(200, json=_backup())
        )

        result = indexes.describe_backup("bkp_123abc")

        assert isinstance(result, BackupModel)
        assert route.calls.last.request.url.path == "/backups/bkp_123abc"

    def test_empty_backup_id_raises_before_any_http(self, indexes: Indexes) -> None:
        with pytest.raises(PineconeValueError) as exc:
            indexes.describe_backup("")

        assert "backup_id" in str(exc.value)


class TestSurfaceParity:
    @respx.mock
    def test_index_scoped_and_project_surfaces_share_one_model(self, indexes: Indexes) -> None:
        respx.get(f"{BASE_URL}/backups/bkp_123abc").mock(
            return_value=httpx.Response(200, json=_backup())
        )
        respx.get(f"{BASE_URL}/indexes/my-index/backups").mock(
            return_value=httpx.Response(200, json={"data": [_backup()]})
        )

        described = indexes.describe_backup("bkp_123abc")
        listed = indexes.list_backups("my-index").to_list()[0]

        assert type(described) is type(listed) is BackupModel
        assert described.to_dict() == listed.to_dict()
