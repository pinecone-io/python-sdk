"""2026-07 conformance for the five admin Invites operations.

New in 2026-07: ``list_invites``, ``create_invite``, ``fetch_invite``,
``delete_invite``, ``resend_invite``. The client under test is a real
:class:`Admin`, so the version header on the wire comes from the SDK's own
constant rather than from the test.

``list_invites`` takes only ``limit`` and ``paginationToken`` — there is no
``email`` filter on this collection, unlike ``list_users`` — so its test pins
that spelling and that the cursor is echoed back verbatim. The response
fixtures populate every property ``Invite`` declares, including the nullable
``expires_at``/``processed_at`` pair, so the round-trip leg has something to
lose.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import httpx
import pytest
import respx

from pinecone._internal.constants import DEFAULT_BASE_URL
from pinecone.admin.admin import _OAUTH_URL, Admin
from pinecone.models.admin.invite import InviteList, InviteModel
from tests.unit.conformance import api_op

BASE_URL = DEFAULT_BASE_URL

INVITE: dict[str, Any] = {
    "id": "9c8e3528-b9c0-4358-84ce-84c28e91b566",
    "email": "newhire@acme.com",
    "status": "pending",
    "expires_at": "2026-05-21T03:00:00Z",
    "processed_at": None,
    "created_at": "2026-04-14T20:00:00Z",
}

INVITE_ID: str = INVITE["id"]

CURSOR = "eyJsYXN0X2lkIjoiOWM4ZTM1MjgifQ=="

INVITE_LIST: dict[str, Any] = {"data": [INVITE], "pagination": {"next": CURSOR}}

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


@api_op("admin:list_invites")
def test_list_invites(claim: Any, admin: Admin, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.get(f"{BASE_URL}/admin/invites").mock(
        side_effect=[
            httpx.Response(200, json=INVITE_LIST),
            httpx.Response(200, json={"data": [], "pagination": None}),
        ]
    )

    result = admin.invites.list(limit=50).to_list()
    assert [i.email for i in result] == ["newhire@acme.com"]
    assert isinstance(result[0], InviteModel)

    first = route.calls[0].request
    assert first.url.params["limit"] == "50"
    assert "paginationToken" not in first.url.params
    assert route.calls[1].request.url.params["paginationToken"] == CURSOR

    claim.assert_request(first)
    claim.assert_api_version(first)
    claim.assert_roundtrip(InviteList, INVITE_LIST, optional_absent=["pagination"])


@api_op("admin:create_invite")
def test_create_invite(claim: Any, admin: Admin, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.post(f"{BASE_URL}/admin/invites").mock(
        return_value=httpx.Response(200, json=INVITE)
    )

    result = admin.invites.create(
        email="newhire@acme.com",
        role_bindings=[{"resource_type": "organization", "role": "OrgMember"}],
    )
    assert result.email == "newhire@acme.com"

    request = route.calls.last.request
    claim.assert_request(request)
    claim.assert_api_version(request)
    claim.assert_roundtrip(InviteModel, INVITE, optional_absent=["expires_at", "processed_at"])


@api_op("admin:fetch_invite")
def test_fetch_invite(claim: Any, admin: Admin, respx_mock: respx.MockRouter) -> None:
    processed = {**INVITE, "status": "processed", "processed_at": "2026-04-15T08:30:00Z"}
    route = respx_mock.get(f"{BASE_URL}/admin/invites/{INVITE_ID}").mock(
        return_value=httpx.Response(200, json=processed)
    )

    result = admin.invites.describe(invite_id=INVITE_ID)
    assert result.status == "processed"

    request = route.calls.last.request
    claim.assert_request(request)
    claim.assert_api_version(request)
    claim.assert_roundtrip(InviteModel, processed, optional_absent=["expires_at", "processed_at"])


@api_op("admin:delete_invite")
def test_delete_invite(claim: Any, admin: Admin, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.delete(f"{BASE_URL}/admin/invites/{INVITE_ID}").mock(
        return_value=httpx.Response(202)
    )

    returned = admin.invites.delete(invite_id=INVITE_ID)

    request = route.calls.last.request
    claim.assert_request(request)
    claim.assert_api_version(request)
    claim.assert_no_response_body(returned)


@api_op("admin:resend_invite")
def test_resend_invite(claim: Any, admin: Admin, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.post(f"{BASE_URL}/admin/invites/{INVITE_ID}/resend").mock(
        return_value=httpx.Response(200, json=INVITE)
    )

    result = admin.invites.resend(invite_id=INVITE_ID)
    assert result.status == "pending"

    request = route.calls.last.request
    claim.assert_request(request)
    claim.assert_api_version(request)
    claim.assert_roundtrip(InviteModel, INVITE, optional_absent=["expires_at", "processed_at"])
