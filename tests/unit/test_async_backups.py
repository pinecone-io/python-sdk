"""Unit tests for AsyncBackups namespace — create, list, describe, get, delete."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator

import httpx
import pytest
import respx

from pinecone._internal.config import PineconeConfig
from pinecone._internal.constants import CONTROL_PLANE_API_VERSION
from pinecone._internal.http_client import AsyncHTTPClient
from pinecone.async_client.backups import AsyncBackups
from pinecone.errors.exceptions import ValidationError
from pinecone.models.backups.list import BackupList
from pinecone.models.backups.model import BackupModel
from tests.factories import make_backup_response

BASE_URL = "https://api.test.pinecone.io"


@pytest.fixture
async def async_http_client() -> AsyncGenerator[AsyncHTTPClient]:
    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    client = AsyncHTTPClient(config, CONTROL_PLANE_API_VERSION)
    yield client
    await client.close()


@pytest.fixture
def async_backups(async_http_client: AsyncHTTPClient) -> AsyncBackups:
    return AsyncBackups(http=async_http_client)


# ---------------------------------------------------------------------------
# create()
# ---------------------------------------------------------------------------


@respx.mock
async def test_async_create_backup(async_backups: AsyncBackups) -> None:
    route = respx.post(f"{BASE_URL}/indexes/my-index/backups").mock(
        return_value=httpx.Response(201, json=make_backup_response()),
    )

    result = await async_backups.create(index_name="my-index")

    assert isinstance(result, BackupModel)
    assert result.backup_id == "670e8400-e29b-41d4-a716-446655440001"

    # Verify description is absent from body when not provided
    request = route.calls[0].request
    body = json.loads(request.content)
    assert "description" not in body


@respx.mock
async def test_async_create_backup_with_name_and_description(
    async_backups: AsyncBackups,
) -> None:
    route = respx.post(f"{BASE_URL}/indexes/my-index/backups").mock(
        return_value=httpx.Response(
            201,
            json=make_backup_response(name="daily-backup", description="A daily backup"),
        ),
    )

    result = await async_backups.create(
        index_name="my-index",
        name="daily-backup",
        description="A daily backup",
    )

    assert isinstance(result, BackupModel)
    assert result.name == "daily-backup"
    assert result.description == "A daily backup"

    # Verify name and description in request body
    request = route.calls[0].request
    expected_body = httpx.Request(
        "POST", "/", json={"name": "daily-backup", "description": "A daily backup"}
    )
    assert request.content == expected_body.content


async def test_async_create_empty_index_name_raises(async_backups: AsyncBackups) -> None:
    with pytest.raises(ValidationError) as exc_info:
        await async_backups.create(index_name="")

    assert "index_name" in str(exc_info.value)
    assert "non-empty" in str(exc_info.value)


# ---------------------------------------------------------------------------
# list()
# ---------------------------------------------------------------------------


@respx.mock
async def test_async_list_backups_for_index(async_backups: AsyncBackups) -> None:
    respx.get(f"{BASE_URL}/indexes/my-index/backups").mock(
        return_value=httpx.Response(
            200,
            json={"data": [make_backup_response()]},
        ),
    )

    result = await async_backups.list(index_name="my-index")

    assert isinstance(result, BackupList)
    assert len(result) == 1
    assert result[0].backup_id == "670e8400-e29b-41d4-a716-446655440001"


@respx.mock
async def test_async_list_backups_all(async_backups: AsyncBackups) -> None:
    route = respx.get(f"{BASE_URL}/backups").mock(
        return_value=httpx.Response(
            200,
            json={"data": [make_backup_response(), make_backup_response(backup_id="second")]},
        ),
    )

    result = await async_backups.list()

    assert isinstance(result, BackupList)
    assert len(result) == 2
    assert route.called


@respx.mock
async def test_async_list_backups_no_limit_param_when_default(async_backups: AsyncBackups) -> None:
    route = respx.get(f"{BASE_URL}/backups").mock(
        return_value=httpx.Response(200, json={"data": []}),
    )

    await async_backups.list()

    request = route.calls[0].request
    assert "limit" not in request.url.params


@respx.mock
async def test_async_list_backups_pagination_params(async_backups: AsyncBackups) -> None:
    route = respx.get(f"{BASE_URL}/backups").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [make_backup_response()],
                "pagination": {"next": "token-abc"},
            },
        ),
    )

    result = await async_backups.list(limit=5, pagination_token="token-xyz")

    assert isinstance(result, BackupList)
    assert len(result) == 1

    request = route.calls[0].request
    assert request.url.params["paginationToken"] == "token-xyz"
    assert "limit" not in request.url.params


# ---------------------------------------------------------------------------
# describe()
# ---------------------------------------------------------------------------


@respx.mock
async def test_async_describe_backup(async_backups: AsyncBackups) -> None:
    backup_id = "670e8400-e29b-41d4-a716-446655440001"
    respx.get(f"{BASE_URL}/backups/{backup_id}").mock(
        return_value=httpx.Response(200, json=make_backup_response()),
    )

    result = await async_backups.describe(backup_id=backup_id)

    assert isinstance(result, BackupModel)
    assert result.backup_id == backup_id
    assert result.status == "Ready"
    assert result.source_index_name == "test-index"


async def test_async_describe_empty_id_raises(async_backups: AsyncBackups) -> None:
    with pytest.raises(ValidationError) as exc_info:
        await async_backups.describe(backup_id="")

    assert "backup_id" in str(exc_info.value)
    assert "non-empty" in str(exc_info.value)


# ---------------------------------------------------------------------------
# get() — alias for describe()
# ---------------------------------------------------------------------------


@respx.mock
async def test_async_get_is_alias_for_describe(async_backups: AsyncBackups) -> None:
    backup_id = "670e8400-e29b-41d4-a716-446655440001"
    respx.get(f"{BASE_URL}/backups/{backup_id}").mock(
        return_value=httpx.Response(200, json=make_backup_response()),
    )

    result = await async_backups.get(backup_id=backup_id)

    assert isinstance(result, BackupModel)
    assert result.backup_id == backup_id


# ---------------------------------------------------------------------------
# delete()
# ---------------------------------------------------------------------------


@respx.mock
async def test_async_delete_backup(async_backups: AsyncBackups) -> None:
    backup_id = "670e8400-e29b-41d4-a716-446655440001"
    respx.delete(f"{BASE_URL}/backups/{backup_id}").mock(
        return_value=httpx.Response(202),
    )

    await async_backups.delete(backup_id=backup_id)

    # delete returns None (202 Accepted), no assertion needed


async def test_async_delete_empty_id_raises(async_backups: AsyncBackups) -> None:
    with pytest.raises(ValidationError) as exc_info:
        await async_backups.delete(backup_id="")

    assert "backup_id" in str(exc_info.value)
    assert "non-empty" in str(exc_info.value)


# ---------------------------------------------------------------------------
# repr()
# ---------------------------------------------------------------------------


def test_repr() -> None:
    from unittest.mock import MagicMock

    backups = AsyncBackups(http=MagicMock())
    assert repr(backups) == "AsyncBackups()"


# ---------------------------------------------------------------------------
# AsyncPinecone.backups property
# ---------------------------------------------------------------------------


def test_async_pinecone_backups_property() -> None:
    from pinecone.async_client.pinecone import AsyncPinecone

    pc = AsyncPinecone(api_key="test-key")
    backups = pc.backups
    assert isinstance(backups, AsyncBackups)
    # Verify lazy caching — same instance returned
    assert pc.backups is backups
