"""2026-07 conformance for the three admin Users operations.

New in 2026-07: ``list_users``, ``fetch_user``, ``delete_user``. The client
under test is a real :class:`Admin`, so the version header on the wire comes
from the SDK's own constant rather than from the test.

``list_users`` is the first admin operation with cursor pagination, so its test
also pins the query-parameter spelling the spec declares — ``email``, ``limit``,
``paginationToken`` — and that the cursor is echoed back verbatim. The response
fixture populates every property ``UserList`` and ``User`` declare so the
round-trip leg has something to lose.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import httpx
import pytest
import respx

from pinecone._internal.constants import DEFAULT_BASE_URL
from pinecone.admin.admin import _OAUTH_URL, Admin
from pinecone.models.admin.user import UserList, UserModel
from tests.unit.conformance import api_op

BASE_URL = DEFAULT_BASE_URL

USER: dict[str, Any] = {
    "id": "e2e92523-85dc-4142-b8c2-e681be8b78df",
    "email": "alice@example.com",
    "name": "Alice Example",
}

USER_ID: str = USER["id"]

CURSOR = "eyJsYXN0X2lkIjoiZTJlOTI1MjMifQ=="

USER_LIST: dict[str, Any] = {"data": [USER], "pagination": {"next": CURSOR}}

TOKEN: dict[str, Any] = {
    "access_token": "conformance-access-token",
    "token_type": "Bearer",
    "expires_in": 1800,
}


@pytest.fixture
def admin(respx_mock: respx.MockRouter) -> Iterator[Admin]:
    respx_mock.post(_OAUTH_URL).mock(return_value=httpx.Response(200, json=TOKEN))
    client = Admin(client_id="conformance-id", client_secret="conformance-secret")
    yield client
    client.close()


@api_op("admin:list_users")
def test_list_users(claim: Any, admin: Admin, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.get(f"{BASE_URL}/admin/users").mock(
        side_effect=[
            httpx.Response(200, json=USER_LIST),
            httpx.Response(200, json={"data": [], "pagination": None}),
        ]
    )

    result = admin.users.list(email="alice@example.com", limit=50).to_list()
    assert [u.email for u in result] == ["alice@example.com"]
    assert isinstance(result[0], UserModel)

    first = route.calls[0].request
    assert first.url.params["email"] == "alice@example.com"
    assert first.url.params["limit"] == "50"
    assert "paginationToken" not in first.url.params
    assert route.calls[1].request.url.params["paginationToken"] == CURSOR

    claim.assert_request(first)
    claim.assert_api_version(first)
    claim.assert_roundtrip(UserList, USER_LIST, optional_absent=["pagination"])


@api_op("admin:fetch_user")
def test_fetch_user(claim: Any, admin: Admin, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.get(f"{BASE_URL}/admin/users/{USER_ID}").mock(
        return_value=httpx.Response(200, json=USER)
    )

    result = admin.users.describe(user_id=USER_ID)
    assert result.email == "alice@example.com"

    request = route.calls.last.request
    claim.assert_request(request)
    claim.assert_api_version(request)
    claim.assert_roundtrip(UserModel, USER, optional_absent=["name"])


@api_op("admin:delete_user")
def test_delete_user(claim: Any, admin: Admin, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.delete(f"{BASE_URL}/admin/users/{USER_ID}").mock(
        return_value=httpx.Response(202)
    )

    returned = admin.users.delete(user_id=USER_ID)

    request = route.calls.last.request
    claim.assert_request(request)
    claim.assert_api_version(request)
    claim.assert_no_response_body(returned)
