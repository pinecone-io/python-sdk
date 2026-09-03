"""2026-07 conformance for the six admin ServiceAccounts operations.

New in 2026-07: ``list_service_accounts``, ``create_service_account``,
``fetch_service_account``, ``update_service_account``,
``delete_service_account``, ``rotate_service_account_secret``. The client under
test is a real :class:`Admin`, so the version header on the wire comes from the
SDK's own constant rather than from the test.

Two operations return ``ServiceAccountWithSecret`` and the other four must not.
``create`` answers ``201`` rather than ``200``, and ``rotate-secret`` is a POST
with no request body to a sub-resource path, so both spellings are pinned here
where the manifest can check them against the spec.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import httpx
import pytest
import respx

from pinecone._internal.constants import DEFAULT_BASE_URL
from pinecone.admin.admin import _OAUTH_URL, Admin
from pinecone.models.admin.service_account import (
    ServiceAccountList,
    ServiceAccountModel,
    ServiceAccountWithSecret,
)
from tests.unit.conformance import api_op

BASE_URL = DEFAULT_BASE_URL

SERVICE_ACCOUNT: dict[str, Any] = {
    "id": "f8a3b2c1-4d5e-6f7a-8b9c-0d1e2f3a4b5c",
    "name": "My Service Account",
    "client_id": "l3Ow0CmFyc4jOONcwiKUCRqQKN0tiCAn",
    "created_at": "2026-04-10T15:23:00Z",
    "updated_at": "2026-04-12T09:11:00Z",
}

SERVICE_ACCOUNT_ID: str = SERVICE_ACCOUNT["id"]

CURSOR = "eyJsYXN0X2lkIjoiZDI0MTc3YTAifQ=="

SERVICE_ACCOUNT_LIST: dict[str, Any] = {
    "data": [SERVICE_ACCOUNT],
    "pagination": {"next": CURSOR},
}

WITH_SECRET: dict[str, Any] = {
    "service_account": SERVICE_ACCOUNT,
    "client_secret": "8p-kkC23XOWvkCosKq-BOn3G74qp__rBcDMxc82iB4gfzRvuhSCRBKM7C5Q7TAzj",
}

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


@api_op("admin:list_service_accounts")
def test_list_service_accounts(claim: Any, admin: Admin, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.get(f"{BASE_URL}/admin/service-accounts").mock(
        side_effect=[
            httpx.Response(200, json=SERVICE_ACCOUNT_LIST),
            httpx.Response(200, json={"data": [], "pagination": None}),
        ]
    )

    result = admin.service_accounts.list(limit=50).to_list()
    assert [a.name for a in result] == ["My Service Account"]
    assert isinstance(result[0], ServiceAccountModel)

    first = route.calls[0].request
    assert first.url.params["limit"] == "50"
    assert "paginationToken" not in first.url.params
    assert route.calls[1].request.url.params["paginationToken"] == CURSOR

    claim.assert_request(first)
    claim.assert_api_version(first)
    claim.assert_roundtrip(ServiceAccountList, SERVICE_ACCOUNT_LIST, optional_absent=["pagination"])


@api_op("admin:create_service_account")
def test_create_service_account(claim: Any, admin: Admin, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.post(f"{BASE_URL}/admin/service-accounts").mock(
        return_value=httpx.Response(201, json=WITH_SECRET)
    )

    result = admin.service_accounts.create(
        name="My Service Account",
        role_bindings=[{"resource_type": "organization", "role": "OrgMember"}],
    )
    assert result.service_account.name == "My Service Account"
    assert result.client_secret == WITH_SECRET["client_secret"]

    request = route.calls.last.request
    claim.assert_request(request)
    claim.assert_api_version(request)
    claim.assert_roundtrip(ServiceAccountWithSecret, WITH_SECRET, optional_absent=[])


@api_op("admin:fetch_service_account")
def test_fetch_service_account(claim: Any, admin: Admin, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.get(f"{BASE_URL}/admin/service-accounts/{SERVICE_ACCOUNT_ID}").mock(
        return_value=httpx.Response(200, json=SERVICE_ACCOUNT)
    )

    result = admin.service_accounts.describe(service_account_id=SERVICE_ACCOUNT_ID)
    assert result.client_id == SERVICE_ACCOUNT["client_id"]

    request = route.calls.last.request
    claim.assert_request(request)
    claim.assert_api_version(request)
    claim.assert_roundtrip(ServiceAccountModel, SERVICE_ACCOUNT, optional_absent=[])


@api_op("admin:update_service_account")
def test_update_service_account(claim: Any, admin: Admin, respx_mock: respx.MockRouter) -> None:
    renamed = {**SERVICE_ACCOUNT, "name": "ci-prod-renamed"}
    route = respx_mock.patch(f"{BASE_URL}/admin/service-accounts/{SERVICE_ACCOUNT_ID}").mock(
        return_value=httpx.Response(200, json=renamed)
    )

    result = admin.service_accounts.update(
        service_account_id=SERVICE_ACCOUNT_ID, name="ci-prod-renamed"
    )
    assert result.name == "ci-prod-renamed"

    request = route.calls.last.request
    claim.assert_request(request)
    claim.assert_api_version(request)
    claim.assert_roundtrip(ServiceAccountModel, renamed, optional_absent=[])


@api_op("admin:delete_service_account")
def test_delete_service_account(claim: Any, admin: Admin, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.delete(f"{BASE_URL}/admin/service-accounts/{SERVICE_ACCOUNT_ID}").mock(
        return_value=httpx.Response(202)
    )

    returned = admin.service_accounts.delete(service_account_id=SERVICE_ACCOUNT_ID)

    request = route.calls.last.request
    claim.assert_request(request)
    claim.assert_api_version(request)
    claim.assert_no_response_body(returned)


@api_op("admin:rotate_service_account_secret")
def test_rotate_service_account_secret(
    claim: Any, admin: Admin, respx_mock: respx.MockRouter
) -> None:
    rotated = {**WITH_SECRET, "client_secret": "rotated-secret-value-0000000000000000"}
    route = respx_mock.post(
        f"{BASE_URL}/admin/service-accounts/{SERVICE_ACCOUNT_ID}/rotate-secret"
    ).mock(return_value=httpx.Response(200, json=rotated))

    result = admin.service_accounts.rotate_secret(service_account_id=SERVICE_ACCOUNT_ID)
    assert result.client_secret == "rotated-secret-value-0000000000000000"
    assert result.service_account.client_id == SERVICE_ACCOUNT["client_id"]

    request = route.calls.last.request
    assert not request.content

    claim.assert_request(request)
    claim.assert_api_version(request)
    claim.assert_roundtrip(ServiceAccountWithSecret, rotated, optional_absent=[])
