"""Unit tests for Backups namespace — create, list, describe, get, delete."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from pinecone._internal.config import PineconeConfig
from pinecone._internal.constants import CONTROL_PLANE_API_VERSION
from pinecone._internal.http_client import HTTPClient
from pinecone.client.backups import Backups
from pinecone.errors.exceptions import ValidationError
from pinecone.models.backups.list import BackupList
from pinecone.models.backups.model import BackupModel
from tests.factories import make_backup_response

BASE_URL = "https://api.test.pinecone.io"


@pytest.fixture
def http_client() -> HTTPClient:
    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    return HTTPClient(config, CONTROL_PLANE_API_VERSION)


@pytest.fixture
def backups(http_client: HTTPClient) -> Backups:
    return Backups(http=http_client)


# ---------------------------------------------------------------------------
# create()
# ---------------------------------------------------------------------------


@respx.mock
def test_create_backup_minimal(backups: Backups) -> None:
    route = respx.post(f"{BASE_URL}/indexes/my-index/backups").mock(
        return_value=httpx.Response(201, json=make_backup_response()),
    )

    result = backups.create(index_name="my-index")

    assert isinstance(result, BackupModel)
    assert result.backup_id == "670e8400-e29b-41d4-a716-446655440001"

    # Verify description is absent from body when not provided
    request = route.calls[0].request
    body = json.loads(request.content)
    assert "description" not in body


@respx.mock
def test_create_backup_with_name_and_description(backups: Backups) -> None:
    route = respx.post(f"{BASE_URL}/indexes/my-index/backups").mock(
        return_value=httpx.Response(
            201,
            json=make_backup_response(name="daily-backup", description="A daily backup"),
        ),
    )

    result = backups.create(
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


def test_create_backup_empty_index_name_raises(backups: Backups) -> None:
    with pytest.raises(ValidationError) as exc_info:
        backups.create(index_name="")

    assert "index_name" in str(exc_info.value)
    assert "non-empty" in str(exc_info.value)


# ---------------------------------------------------------------------------
# list()
# ---------------------------------------------------------------------------


@respx.mock
def test_list_backups_for_index(backups: Backups) -> None:
    respx.get(f"{BASE_URL}/indexes/my-index/backups").mock(
        return_value=httpx.Response(
            200,
            json={"data": [make_backup_response()]},
        ),
    )

    result = backups.list(index_name="my-index")

    assert isinstance(result, BackupList)
    assert len(result) == 1
    assert result[0].backup_id == "670e8400-e29b-41d4-a716-446655440001"


@respx.mock
def test_list_backups_all(backups: Backups) -> None:
    route = respx.get(f"{BASE_URL}/backups").mock(
        return_value=httpx.Response(
            200,
            json={"data": [make_backup_response(), make_backup_response(backup_id="second")]},
        ),
    )

    result = backups.list()

    assert isinstance(result, BackupList)
    assert len(result) == 2
    assert route.called


@respx.mock
def test_list_backups_pagination_params(backups: Backups) -> None:
    route = respx.get(f"{BASE_URL}/backups").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [make_backup_response()],
                "pagination": {"next": "token-abc"},
            },
        ),
    )

    result = backups.list(limit=5, pagination_token="token-xyz")

    assert isinstance(result, BackupList)
    assert len(result) == 1

    request = route.calls[0].request
    assert request.url.params["paginationToken"] == "token-xyz"
    assert "limit" not in request.url.params


@respx.mock
def test_list_backups_pagination_token_preserved(backups: Backups) -> None:
    respx.get(f"{BASE_URL}/backups").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [make_backup_response()],
                "pagination": {"next": "token-abc"},
            },
        ),
    )

    result = backups.list()

    assert isinstance(result, BackupList)
    assert result.pagination is not None
    assert result.pagination.next == "token-abc"


@respx.mock
def test_list_backups_no_pagination(backups: Backups) -> None:
    respx.get(f"{BASE_URL}/backups").mock(
        return_value=httpx.Response(
            200,
            json={"data": [make_backup_response()]},
        ),
    )

    result = backups.list()

    assert isinstance(result, BackupList)
    assert result.pagination is None


@respx.mock
def test_list_backups_no_limit_param_when_default(backups: Backups) -> None:
    route = respx.get(f"{BASE_URL}/backups").mock(
        return_value=httpx.Response(200, json={"data": []}),
    )

    backups.list()

    request = route.calls[0].request
    assert "limit" not in request.url.params


@respx.mock
def test_list_backups_for_index_no_limit_param_when_default(backups: Backups) -> None:
    route = respx.get(f"{BASE_URL}/indexes/my-index/backups").mock(
        return_value=httpx.Response(200, json={"data": []}),
    )

    backups.list(index_name="my-index")

    request = route.calls[0].request
    assert "limit" not in request.url.params


@respx.mock
def test_list_backups_explicit_limit_sent(backups: Backups) -> None:
    route = respx.get(f"{BASE_URL}/backups").mock(
        return_value=httpx.Response(200, json={"data": []}),
    )

    backups.list(limit=25)

    request = route.calls[0].request
    assert request.url.params["limit"] == "25"


# ---------------------------------------------------------------------------
# describe()
# ---------------------------------------------------------------------------


@respx.mock
def test_describe_backup(backups: Backups) -> None:
    backup_id = "670e8400-e29b-41d4-a716-446655440001"
    respx.get(f"{BASE_URL}/backups/{backup_id}").mock(
        return_value=httpx.Response(200, json=make_backup_response()),
    )

    result = backups.describe(backup_id=backup_id)

    assert isinstance(result, BackupModel)
    assert result.backup_id == backup_id
    assert result.status == "Ready"
    assert result.source_index_name == "test-index"


def test_describe_empty_id_raises(backups: Backups) -> None:
    with pytest.raises(ValidationError) as exc_info:
        backups.describe(backup_id="")

    assert "backup_id" in str(exc_info.value)
    assert "non-empty" in str(exc_info.value)


# ---------------------------------------------------------------------------
# get() — alias for describe()
# ---------------------------------------------------------------------------


@respx.mock
def test_get_is_alias_for_describe(backups: Backups) -> None:
    backup_id = "670e8400-e29b-41d4-a716-446655440001"
    respx.get(f"{BASE_URL}/backups/{backup_id}").mock(
        return_value=httpx.Response(200, json=make_backup_response()),
    )

    result = backups.get(backup_id=backup_id)

    assert isinstance(result, BackupModel)
    assert result.backup_id == backup_id


# ---------------------------------------------------------------------------
# delete()
# ---------------------------------------------------------------------------


@respx.mock
def test_delete_backup(backups: Backups) -> None:
    backup_id = "670e8400-e29b-41d4-a716-446655440001"
    respx.delete(f"{BASE_URL}/backups/{backup_id}").mock(
        return_value=httpx.Response(202),
    )

    backups.delete(backup_id=backup_id)

    # delete returns None (202 Accepted), no assertion needed
