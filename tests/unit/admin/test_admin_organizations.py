"""Unit tests for the Admin Organizations namespace."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
import respx

from pinecone._internal.config import PineconeConfig
from pinecone._internal.constants import ADMIN_API_VERSION
from pinecone._internal.http_client import HTTPClient
from pinecone.admin.organizations import Organizations
from pinecone.errors.exceptions import ValidationError
from pinecone.models.admin.organization import OrganizationList, OrganizationModel

BASE_URL = "https://api.test.pinecone.io"


def _org_response(
    *,
    id: str = "org-abc123",
    name: str = "Acme",
    plan: str = "Enterprise",
    payment_status: str = "Active",
    created_at: str = "2025-01-01T00:00:00Z",
    support_tier: str = "Basic",
) -> dict[str, Any]:
    return {
        "id": id,
        "name": name,
        "plan": plan,
        "payment_status": payment_status,
        "created_at": created_at,
        "support_tier": support_tier,
    }


@pytest.fixture
def http_client() -> HTTPClient:
    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    return HTTPClient(config, ADMIN_API_VERSION)


@pytest.fixture
def organizations(http_client: HTTPClient) -> Organizations:
    return Organizations(http=http_client)


# ---------------------------------------------------------------------------
# list()
# ---------------------------------------------------------------------------


@respx.mock
def test_list_organizations(organizations: Organizations) -> None:
    respx.get(f"{BASE_URL}/admin/organizations").mock(
        return_value=httpx.Response(
            200,
            json={"data": [_org_response()]},
        ),
    )

    result = organizations.list()

    assert isinstance(result, OrganizationList)
    assert len(result) == 1
    org = result[0]
    assert isinstance(org, OrganizationModel)
    assert org.id == "org-abc123"
    assert org.name == "Acme"
    assert org.plan == "Enterprise"
    assert org.payment_status == "Active"
    assert org.created_at == "2025-01-01T00:00:00Z"


@respx.mock
def test_list_organizations_empty(organizations: Organizations) -> None:
    respx.get(f"{BASE_URL}/admin/organizations").mock(
        return_value=httpx.Response(200, json={"data": []}),
    )

    result = organizations.list()

    assert isinstance(result, OrganizationList)
    assert len(result) == 0


# ---------------------------------------------------------------------------
# describe()
# ---------------------------------------------------------------------------


@respx.mock
def test_describe_organization(organizations: Organizations) -> None:
    respx.get(f"{BASE_URL}/admin/organizations/org-abc123").mock(
        return_value=httpx.Response(200, json=_org_response()),
    )

    result = organizations.describe(organization_id="org-abc123")

    assert isinstance(result, OrganizationModel)
    assert result.id == "org-abc123"
    assert result.name == "Acme"


def test_describe_requires_organization_id(organizations: Organizations) -> None:
    with pytest.raises(ValidationError, match="organization_id"):
        organizations.describe(organization_id="")


def test_describe_rejects_whitespace_organization_id(
    organizations: Organizations,
) -> None:
    with pytest.raises(ValidationError, match="organization_id"):
        organizations.describe(organization_id="   ")


# ---------------------------------------------------------------------------
# update()
# ---------------------------------------------------------------------------


@respx.mock
def test_update_organization(organizations: Organizations) -> None:
    route = respx.patch(f"{BASE_URL}/admin/organizations/org-abc123").mock(
        return_value=httpx.Response(200, json=_org_response(name="New Name")),
    )

    result = organizations.update(organization_id="org-abc123", name="New Name")

    assert isinstance(result, OrganizationModel)
    assert result.name == "New Name"

    # Verify request body
    request = route.calls[0].request
    expected_body = httpx.Request("PATCH", "/", json={"name": "New Name"})
    assert request.content == expected_body.content


def test_update_requires_organization_id(organizations: Organizations) -> None:
    with pytest.raises(ValidationError, match="organization_id"):
        organizations.update(organization_id="", name="New Name")


@respx.mock
def test_update_sends_empty_name_verbatim(organizations: Organizations) -> None:
    route = respx.patch(f"{BASE_URL}/admin/organizations/org-abc123").mock(
        return_value=httpx.Response(200, json=_org_response(name="")),
    )

    result = organizations.update(organization_id="org-abc123", name="")

    assert result.name == ""
    request = route.calls[0].request
    expected_body = httpx.Request("PATCH", "/", json={"name": ""})
    assert request.content == expected_body.content


# ---------------------------------------------------------------------------
# delete()
# ---------------------------------------------------------------------------


@respx.mock
def test_delete_organization(organizations: Organizations) -> None:
    respx.delete(f"{BASE_URL}/admin/organizations/org-abc123").mock(
        return_value=httpx.Response(202),
    )

    result = organizations.delete(organization_id="org-abc123")

    assert result is None


def test_delete_requires_organization_id(organizations: Organizations) -> None:
    with pytest.raises(ValidationError, match="organization_id"):
        organizations.delete(organization_id="")


# ---------------------------------------------------------------------------
# keyword-only enforcement
# ---------------------------------------------------------------------------


def test_all_methods_keyword_only(organizations: Organizations) -> None:
    with pytest.raises(TypeError):
        organizations.describe("org-abc123")  # type: ignore[misc]

    with pytest.raises(TypeError):
        organizations.update("org-abc123", "New Name")  # type: ignore[misc]

    with pytest.raises(TypeError):
        organizations.delete("org-abc123")  # type: ignore[misc]


# ---------------------------------------------------------------------------
# repr
# ---------------------------------------------------------------------------


def test_repr(organizations: Organizations) -> None:
    assert repr(organizations) == "Organizations()"
