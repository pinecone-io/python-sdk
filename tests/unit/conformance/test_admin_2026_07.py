"""2026-07 conformance for the 14 admin operations carried over from 2025-10.

All 14 are structurally identical to their 2025-10 definitions — the only
wire-visible change is the ``X-Pinecone-Api-Version`` value — so these tests
exist to pin method, path, that header, and the response schemas against
``admin_2026-07.oas.yaml``.

Payloads populate every property the spec's ``Project`` / ``Organization`` /
``APIKey`` / ``APIKeyWithSecret`` schemas declare, so the round-trip leg has
something to lose. The client under test is a real :class:`Admin`, so the
version header on the wire comes from the SDK's own constant.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import httpx
import orjson
import pytest
import respx

from pinecone._internal.adapters.admin_adapter import (
    _APIKeyListEnvelope,
    _OrganizationListEnvelope,
    _ProjectListEnvelope,
)
from pinecone._internal.constants import DEFAULT_BASE_URL
from pinecone.admin.admin import _OAUTH_URL, Admin
from pinecone.models.admin.api_key import APIKeyModel, APIKeyRole, APIKeyWithSecret
from pinecone.models.admin.organization import OrganizationModel
from pinecone.models.admin.project import ProjectModel
from tests.unit.conformance import api_op

BASE_URL = DEFAULT_BASE_URL

PROJECT: dict[str, Any] = {
    "id": "5c8b1a1e-4e6b-4c1a-9a1e-9c8b1a1e4e6b",
    "name": "chatbot-prod",
    "max_pods": 5,
    "force_encryption_with_cmek": True,
    "organization_id": "org-abc123",
    "created_at": "2026-07-01T12:00:00Z",
}

ORGANIZATION: dict[str, Any] = {
    "id": "org-abc123",
    "name": "acme-corp",
    "plan": "Enterprise",
    "payment_status": "Active",
    "created_at": "2026-07-01T12:00:00Z",
    "support_tier": "Pro",
}

API_KEY: dict[str, Any] = {
    "id": "7a2c1d3e-8f4b-4a5c-9d6e-0f1a2b3c4d5e",
    "name": "devkey",
    "project_id": "5c8b1a1e-4e6b-4c1a-9a1e-9c8b1a1e4e6b",
    "roles": ["ProjectEditor", "DataPlaneViewer"],
}

API_KEY_WITH_SECRET: dict[str, Any] = {"key": API_KEY, "value": "pckey_devkey_abc123"}

TOKEN: dict[str, Any] = {
    "access_token": "conformance-access-token",
    "token_type": "Bearer",
    "expires_in": 1800,
}

PROJECT_ID: str = PROJECT["id"]
ORGANIZATION_ID: str = ORGANIZATION["id"]
API_KEY_ID: str = API_KEY["id"]


@pytest.fixture
def admin(respx_mock: respx.MockRouter) -> Iterator[Admin]:
    respx_mock.post(_OAUTH_URL).mock(return_value=httpx.Response(200, json=TOKEN))
    client = Admin(client_id="conformance-id", client_secret="conformance-secret")
    yield client
    client.close()


@api_op("admin:list_projects")
def test_list_projects(claim: Any, admin: Admin, respx_mock: respx.MockRouter) -> None:
    payload = {"data": [PROJECT]}
    route = respx_mock.get(f"{BASE_URL}/admin/projects").mock(
        return_value=httpx.Response(200, json=payload)
    )

    result = admin.projects.list()
    assert result.names() == ["chatbot-prod"]

    request = route.calls.last.request
    claim.assert_request(request)
    claim.assert_api_version(request)
    claim.assert_roundtrip(_ProjectListEnvelope, payload, optional_absent=["data"])


@api_op("admin:create_project")
def test_create_project(claim: Any, admin: Admin, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.post(f"{BASE_URL}/admin/projects").mock(
        return_value=httpx.Response(201, json=PROJECT)
    )

    result = admin.projects.create(name="chatbot-prod", max_pods=5, force_encryption_with_cmek=True)
    assert result.id == PROJECT_ID

    request = route.calls.last.request
    assert orjson.loads(request.content) == {
        "name": "chatbot-prod",
        "max_pods": 5,
        "force_encryption_with_cmek": True,
    }
    claim.assert_request(request)
    claim.assert_api_version(request)
    claim.assert_roundtrip(ProjectModel, PROJECT, optional_absent=["created_at"])


@api_op("admin:fetch_project")
def test_fetch_project(claim: Any, admin: Admin, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.get(f"{BASE_URL}/admin/projects/{PROJECT_ID}").mock(
        return_value=httpx.Response(200, json=PROJECT)
    )

    result = admin.projects.describe(project_id=PROJECT_ID)
    assert result.organization_id == "org-abc123"

    request = route.calls.last.request
    claim.assert_request(request)
    claim.assert_api_version(request)
    claim.assert_roundtrip(ProjectModel, PROJECT, optional_absent=["created_at"])


@api_op("admin:update_project")
def test_update_project(claim: Any, admin: Admin, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.patch(f"{BASE_URL}/admin/projects/{PROJECT_ID}").mock(
        return_value=httpx.Response(200, json=PROJECT)
    )

    result = admin.projects.update(project_id=PROJECT_ID, name="chatbot-prod", max_pods=5)
    assert result.max_pods == 5

    request = route.calls.last.request
    assert orjson.loads(request.content) == {"name": "chatbot-prod", "max_pods": 5}
    claim.assert_request(request)
    claim.assert_api_version(request)
    claim.assert_roundtrip(ProjectModel, PROJECT, optional_absent=["created_at"])


@api_op("admin:delete_project")
def test_delete_project(claim: Any, admin: Admin, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.delete(f"{BASE_URL}/admin/projects/{PROJECT_ID}").mock(
        return_value=httpx.Response(202)
    )

    returned = admin.projects.delete(project_id=PROJECT_ID)

    request = route.calls.last.request
    claim.assert_request(request)
    claim.assert_api_version(request)
    claim.assert_no_response_body(returned)


@api_op("admin:list_organizations")
def test_list_organizations(claim: Any, admin: Admin, respx_mock: respx.MockRouter) -> None:
    payload = {"data": [ORGANIZATION]}
    route = respx_mock.get(f"{BASE_URL}/admin/organizations").mock(
        return_value=httpx.Response(200, json=payload)
    )

    result = admin.organizations.list()
    assert result.names() == ["acme-corp"]

    request = route.calls.last.request
    claim.assert_request(request)
    claim.assert_api_version(request)
    claim.assert_roundtrip(_OrganizationListEnvelope, payload, optional_absent=["data"])


@api_op("admin:fetch_organization")
def test_fetch_organization(claim: Any, admin: Admin, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.get(f"{BASE_URL}/admin/organizations/{ORGANIZATION_ID}").mock(
        return_value=httpx.Response(200, json=ORGANIZATION)
    )

    result = admin.organizations.describe(organization_id=ORGANIZATION_ID)
    assert result.plan == "Enterprise"

    request = route.calls.last.request
    claim.assert_request(request)
    claim.assert_api_version(request)
    claim.assert_roundtrip(OrganizationModel, ORGANIZATION, optional_absent=[])


@api_op("admin:update_organization")
def test_update_organization(claim: Any, admin: Admin, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.patch(f"{BASE_URL}/admin/organizations/{ORGANIZATION_ID}").mock(
        return_value=httpx.Response(200, json=ORGANIZATION)
    )

    result = admin.organizations.update(organization_id=ORGANIZATION_ID, name="acme-corp")
    assert result.name == "acme-corp"

    request = route.calls.last.request
    assert orjson.loads(request.content) == {"name": "acme-corp"}
    claim.assert_request(request)
    claim.assert_api_version(request)
    claim.assert_roundtrip(OrganizationModel, ORGANIZATION, optional_absent=[])


@api_op("admin:delete_organization")
def test_delete_organization(claim: Any, admin: Admin, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.delete(f"{BASE_URL}/admin/organizations/{ORGANIZATION_ID}").mock(
        return_value=httpx.Response(202)
    )

    returned = admin.organizations.delete(organization_id=ORGANIZATION_ID)

    request = route.calls.last.request
    claim.assert_request(request)
    claim.assert_api_version(request)
    claim.assert_no_response_body(returned)


@api_op("admin:list_project_api_keys")
def test_list_project_api_keys(claim: Any, admin: Admin, respx_mock: respx.MockRouter) -> None:
    payload = {"data": [API_KEY]}
    route = respx_mock.get(f"{BASE_URL}/admin/projects/{PROJECT_ID}/api-keys").mock(
        return_value=httpx.Response(200, json=payload)
    )

    result = admin.api_keys.list(project_id=PROJECT_ID)
    assert result.names() == ["devkey"]

    request = route.calls.last.request
    claim.assert_request(request)
    claim.assert_api_version(request)
    claim.assert_roundtrip(_APIKeyListEnvelope, payload, optional_absent=["data"])


@api_op("admin:create_api_key")
def test_create_api_key(claim: Any, admin: Admin, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.post(f"{BASE_URL}/admin/projects/{PROJECT_ID}/api-keys").mock(
        return_value=httpx.Response(201, json=API_KEY_WITH_SECRET)
    )

    result = admin.api_keys.create(
        project_id=PROJECT_ID,
        name="devkey",
        roles=[APIKeyRole.PROJECT_EDITOR, "DataPlaneViewer"],
    )
    assert result.value == "pckey_devkey_abc123"

    request = route.calls.last.request
    assert orjson.loads(request.content) == {
        "name": "devkey",
        "roles": ["ProjectEditor", "DataPlaneViewer"],
    }
    claim.assert_request(request)
    claim.assert_api_version(request)
    claim.assert_roundtrip(APIKeyWithSecret, API_KEY_WITH_SECRET, optional_absent=[])


@api_op("admin:fetch_api_key")
def test_fetch_api_key(claim: Any, admin: Admin, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.get(f"{BASE_URL}/admin/api-keys/{API_KEY_ID}").mock(
        return_value=httpx.Response(200, json=API_KEY)
    )

    result = admin.api_keys.describe(api_key_id=API_KEY_ID)
    assert result.roles == [APIKeyRole.PROJECT_EDITOR, APIKeyRole.DATA_PLANE_VIEWER]

    request = route.calls.last.request
    claim.assert_request(request)
    claim.assert_api_version(request)
    claim.assert_roundtrip(APIKeyModel, API_KEY, optional_absent=["name"])


@api_op("admin:update_api_key")
def test_update_api_key(claim: Any, admin: Admin, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.patch(f"{BASE_URL}/admin/api-keys/{API_KEY_ID}").mock(
        return_value=httpx.Response(200, json=API_KEY)
    )

    result = admin.api_keys.update(
        api_key_id=API_KEY_ID, name="devkey", roles=["ProjectEditor", "DataPlaneViewer"]
    )
    assert result.name == "devkey"

    request = route.calls.last.request
    assert orjson.loads(request.content) == {
        "name": "devkey",
        "roles": ["ProjectEditor", "DataPlaneViewer"],
    }
    claim.assert_request(request)
    claim.assert_api_version(request)
    claim.assert_roundtrip(APIKeyModel, API_KEY, optional_absent=["name"])


@api_op("admin:delete_api_key")
def test_delete_api_key(claim: Any, admin: Admin, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.delete(f"{BASE_URL}/admin/api-keys/{API_KEY_ID}").mock(
        return_value=httpx.Response(202)
    )

    returned = admin.api_keys.delete(api_key_id=API_KEY_ID)

    request = route.calls.last.request
    claim.assert_request(request)
    claim.assert_api_version(request)
    claim.assert_no_response_body(returned)
