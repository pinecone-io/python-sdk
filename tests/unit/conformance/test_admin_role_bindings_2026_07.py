"""2026-07 conformance for the four admin RoleBindings operations.

New in 2026-07: ``list_role_bindings``, ``create_role_binding``,
``fetch_role_binding``, ``delete_role_binding``. The client under test is a real
:class:`Admin`, so the version header on the wire comes from the SDK's own
constant rather than from the test.

``list_role_bindings`` is the only admin list operation with real filters, so its
test pins all five of them onto the query string the manifest checks against the
spec. ``delete_role_binding`` answers ``202`` with no body, and ``create``
answers ``200`` rather than ``201``, so both spellings are pinned here.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import httpx
import pytest
import respx

from pinecone._internal.constants import DEFAULT_BASE_URL
from pinecone.admin.admin import _OAUTH_URL, Admin
from pinecone.models.admin.role_binding import RoleBindingList, RoleBindingModel
from tests.unit.conformance import api_op

BASE_URL = DEFAULT_BASE_URL

ROLE_BINDING: dict[str, Any] = {
    "id": "9a8e3528-b9c0-4358-84ce-84c28e91b566",
    "principal_type": "service_account",
    "principal_id": "f8a3b2c1-4d5e-6f7a-8b9c-0d1e2f3a4b5c",
    "resource_type": "project",
    "resource_id": "a2f7dddb-1597-4eff-9f71-535fde243f58",
    "role": "DataPlaneEditor",
    "created_at": "2026-04-10T15:23:00Z",
}

ROLE_BINDING_ID: str = ROLE_BINDING["id"]

CURSOR = "eyJsYXN0X2lkIjoiOWE4ZTM1MjgifQ=="

ROLE_BINDING_LIST: dict[str, Any] = {"data": [ROLE_BINDING], "pagination": {"next": CURSOR}}

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


@api_op("admin:list_role_bindings")
def test_list_role_bindings(claim: Any, admin: Admin, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.get(f"{BASE_URL}/admin/role-bindings").mock(
        side_effect=[
            httpx.Response(200, json=ROLE_BINDING_LIST),
            httpx.Response(200, json={"data": [], "pagination": None}),
        ]
    )

    result = admin.role_bindings.list(
        principal_type="service_account",
        principal_id=ROLE_BINDING["principal_id"],
        resource_type="project",
        resource_id=ROLE_BINDING["resource_id"],
        role="DataPlaneEditor",
        limit=50,
    ).to_list()
    assert [b.role for b in result] == ["DataPlaneEditor"]
    assert isinstance(result[0], RoleBindingModel)

    first = route.calls[0].request
    assert first.url.params["principal_type"] == "service_account"
    assert first.url.params["principal_id"] == ROLE_BINDING["principal_id"]
    assert first.url.params["resource_type"] == "project"
    assert first.url.params["resource_id"] == ROLE_BINDING["resource_id"]
    assert first.url.params["role"] == "DataPlaneEditor"
    assert first.url.params["limit"] == "50"
    assert "paginationToken" not in first.url.params
    assert route.calls[1].request.url.params["paginationToken"] == CURSOR

    claim.assert_request(first)
    claim.assert_api_version(first)
    claim.assert_roundtrip(RoleBindingList, ROLE_BINDING_LIST, optional_absent=["pagination"])


@api_op("admin:create_role_binding")
def test_create_role_binding(claim: Any, admin: Admin, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.post(f"{BASE_URL}/admin/role-bindings").mock(
        return_value=httpx.Response(200, json=ROLE_BINDING)
    )

    result = admin.role_bindings.create(
        principal_type="service_account",
        principal_id=ROLE_BINDING["principal_id"],
        resource_type="project",
        resource_id=ROLE_BINDING["resource_id"],
        role="DataPlaneEditor",
    )
    assert result.id == ROLE_BINDING_ID
    assert result.role == "DataPlaneEditor"

    request = route.calls.last.request
    claim.assert_request(request)
    claim.assert_api_version(request)
    claim.assert_roundtrip(RoleBindingModel, ROLE_BINDING, optional_absent=[])


@api_op("admin:fetch_role_binding")
def test_fetch_role_binding(claim: Any, admin: Admin, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.get(f"{BASE_URL}/admin/role-bindings/{ROLE_BINDING_ID}").mock(
        return_value=httpx.Response(200, json=ROLE_BINDING)
    )

    result = admin.role_bindings.describe(role_binding_id=ROLE_BINDING_ID)
    assert result.principal_id == ROLE_BINDING["principal_id"]

    request = route.calls.last.request
    claim.assert_request(request)
    claim.assert_api_version(request)
    claim.assert_roundtrip(RoleBindingModel, ROLE_BINDING, optional_absent=[])


@api_op("admin:delete_role_binding")
def test_delete_role_binding(claim: Any, admin: Admin, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.delete(f"{BASE_URL}/admin/role-bindings/{ROLE_BINDING_ID}").mock(
        return_value=httpx.Response(202)
    )

    returned = admin.role_bindings.delete(role_binding_id=ROLE_BINDING_ID)

    request = route.calls.last.request
    claim.assert_request(request)
    claim.assert_api_version(request)
    claim.assert_no_response_body(returned)
