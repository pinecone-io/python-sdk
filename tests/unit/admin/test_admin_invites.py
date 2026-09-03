"""Unit tests for the Admin Invites namespace.

Two things carry most of the weight here. The first is ``create()``'s
client-side validation: the server's 400 cannot say *which* ``role_bindings``
entry it choked on, so every rejection here has to name the index, and these
tests pin that no request is sent when it does. The second is the cursor walk —
``list()`` returns a lazy paginator, so the tests pin that nothing is requested
until iteration, that the cursor goes back to the server byte-for-byte, and
that a page whose ``pagination`` is absent or ``null`` ends the walk rather
than looping forever. The simulator does not paginate this collection
(minicone#50), so the multi-page walk is exercised against mocks only.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import httpx
import orjson
import pytest
import respx
from hypothesis import given, settings
from hypothesis import strategies as st

from pinecone._internal.config import PineconeConfig
from pinecone._internal.constants import ADMIN_API_VERSION
from pinecone._internal.http_client import HTTPClient
from pinecone.admin.invites import Invites
from pinecone.errors.exceptions import (
    ApiError,
    ConflictError,
    NotFoundError,
    RateLimitError,
    ValidationError,
)
from pinecone.models.admin.invite import InviteModel, InviteStatus
from pinecone.models.admin.role_binding import ResourceType, RoleBindingInput, RoleName
from pinecone.models.pagination import Paginator

BASE_URL = "https://api.test.pinecone.io"

INVITE_ID = "9c8e3528-b9c0-4358-84ce-84c28e91b566"
PROJECT_ID = "a2f7dddb-1597-4eff-9f71-535fde243f58"

ORG_BINDING: dict[str, Any] = {"resource_type": "organization", "role": "OrgMember"}


def _invite(
    *,
    id: str = INVITE_ID,
    email: str = "newhire@acme.com",
    status: str = "pending",
    expires_at: str | None = "2026-05-21T03:00:00Z",
    processed_at: str | None = None,
    created_at: str = "2026-04-14T20:00:00Z",
) -> dict[str, Any]:
    return {
        "id": id,
        "email": email,
        "status": status,
        "expires_at": expires_at,
        "processed_at": processed_at,
        "created_at": created_at,
    }


def _page(invites: list[dict[str, Any]], *, next_token: str | None = None) -> dict[str, Any]:
    return {"data": invites, "pagination": {"next": next_token} if next_token else None}


@pytest.fixture
def http_client() -> HTTPClient:
    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    return HTTPClient(config, ADMIN_API_VERSION)


@pytest.fixture
def invites(http_client: HTTPClient) -> Invites:
    return Invites(http=http_client)


def test_repr(invites: Invites) -> None:
    assert repr(invites) == "Invites()"


# ---------------------------------------------------------------------------
# list()
# ---------------------------------------------------------------------------


@respx.mock
def test_list_invites(invites: Invites) -> None:
    route = respx.get(f"{BASE_URL}/admin/invites").mock(
        return_value=httpx.Response(200, json=_page([_invite()]))
    )

    result = invites.list()

    assert isinstance(result, Paginator)
    items = result.to_list()
    assert len(items) == 1
    assert isinstance(items[0], InviteModel)
    assert items[0].id == INVITE_ID
    assert items[0].email == "newhire@acme.com"
    assert items[0].status == InviteStatus.PENDING
    assert items[0].expires_at == "2026-05-21T03:00:00Z"
    assert items[0].processed_at is None
    assert route.calls.last.request.url.path == "/admin/invites"


@respx.mock
def test_list_sends_api_version_header(invites: Invites) -> None:
    route = respx.get(f"{BASE_URL}/admin/invites").mock(
        return_value=httpx.Response(200, json=_page([_invite()]))
    )

    invites.list().to_list()

    assert route.calls.last.request.headers["X-Pinecone-Api-Version"] == ADMIN_API_VERSION


@respx.mock
def test_list_is_lazy(invites: Invites) -> None:
    route = respx.get(f"{BASE_URL}/admin/invites").mock(
        return_value=httpx.Response(200, json=_page([_invite()]))
    )

    paginator = invites.list()

    assert route.call_count == 0
    paginator.to_list()
    assert route.call_count == 1


@respx.mock
def test_list_omits_unset_query_params(invites: Invites) -> None:
    route = respx.get(f"{BASE_URL}/admin/invites").mock(
        return_value=httpx.Response(200, json=_page([]))
    )

    invites.list().to_list()

    assert dict(route.calls.last.request.url.params) == {}


@respx.mock
def test_list_sends_limit(invites: Invites) -> None:
    route = respx.get(f"{BASE_URL}/admin/invites").mock(
        return_value=httpx.Response(200, json=_page([_invite()]))
    )

    invites.list(limit=25).to_list()

    assert route.calls.last.request.url.params["limit"] == "25"


@respx.mock
def test_list_sends_initial_pagination_token(invites: Invites) -> None:
    route = respx.get(f"{BASE_URL}/admin/invites").mock(
        return_value=httpx.Response(200, json=_page([_invite()]))
    )

    invites.list(pagination_token="cursor-abc").to_list()

    assert route.calls.last.request.url.params["paginationToken"] == "cursor-abc"


@respx.mock
def test_list_empty_page(invites: Invites) -> None:
    respx.get(f"{BASE_URL}/admin/invites").mock(return_value=httpx.Response(200, json=_page([])))

    assert invites.list().to_list() == []


@respx.mock
def test_list_tolerates_absent_pagination_key(invites: Invites) -> None:
    route = respx.get(f"{BASE_URL}/admin/invites").mock(
        return_value=httpx.Response(200, json={"data": [_invite()]})
    )

    items = invites.list().to_list()

    assert len(items) == 1
    assert route.call_count == 1


@respx.mock
def test_list_tolerates_absent_nullable_timestamps(invites: Invites) -> None:
    respx.get(f"{BASE_URL}/admin/invites").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": INVITE_ID,
                        "email": "newhire@acme.com",
                        "status": "expired",
                        "created_at": "2026-04-14T20:00:00Z",
                    }
                ]
            },
        )
    )

    invite = invites.list().to_list()[0]

    assert invite.expires_at is None
    assert invite.processed_at is None
    assert invite.status == InviteStatus.EXPIRED


@respx.mock
def test_list_surfaces_unknown_status_verbatim(invites: Invites) -> None:
    respx.get(f"{BASE_URL}/admin/invites").mock(
        return_value=httpx.Response(200, json=_page([_invite(status="revoked")]))
    )

    assert invites.list().to_list()[0].status == "revoked"


@respx.mock
def test_list_follows_pagination_cursor_verbatim(invites: Invites) -> None:
    cursor = "eyJsYXN0X2lkIjoiOWM4ZTM1MjgifQ=="
    route = respx.get(f"{BASE_URL}/admin/invites").mock(
        side_effect=[
            httpx.Response(200, json=_page([_invite(id="i1")], next_token=cursor)),
            httpx.Response(200, json=_page([_invite(id="i2")])),
        ]
    )

    items = invites.list().to_list()

    assert [i.id for i in items] == ["i1", "i2"]
    assert route.call_count == 2
    assert "paginationToken" not in route.calls[0].request.url.params
    assert route.calls[1].request.url.params["paginationToken"] == cursor


@respx.mock
def test_list_carries_limit_onto_later_pages(invites: Invites) -> None:
    route = respx.get(f"{BASE_URL}/admin/invites").mock(
        side_effect=[
            httpx.Response(200, json=_page([_invite(id="i1")], next_token="c1")),
            httpx.Response(200, json=_page([_invite(id="i2")])),
        ]
    )

    invites.list(limit=1).to_list()

    second = route.calls[1].request.url.params
    assert second["limit"] == "1"
    assert second["paginationToken"] == "c1"


@respx.mock
def test_list_stops_on_null_pagination(invites: Invites) -> None:
    route = respx.get(f"{BASE_URL}/admin/invites").mock(
        side_effect=[
            httpx.Response(200, json=_page([_invite(id="i1")], next_token="c1")),
            httpx.Response(200, json={"data": [_invite(id="i2")], "pagination": None}),
            httpx.Response(200, json=_page([_invite(id="i3")])),
        ]
    )

    items = invites.list().to_list()

    assert [i.id for i in items] == ["i1", "i2"]
    assert route.call_count == 2


@respx.mock
def test_list_stops_on_pagination_with_null_next(invites: Invites) -> None:
    route = respx.get(f"{BASE_URL}/admin/invites").mock(
        side_effect=[
            httpx.Response(200, json={"data": [_invite(id="i1")], "pagination": {"next": None}}),
            httpx.Response(200, json=_page([_invite(id="i2")])),
        ]
    )

    assert [i.id for i in invites.list().to_list()] == ["i1"]
    assert route.call_count == 1


@respx.mock
def test_list_pages_exposes_page_level_access(invites: Invites) -> None:
    respx.get(f"{BASE_URL}/admin/invites").mock(
        side_effect=[
            httpx.Response(200, json=_page([_invite(id="i1")], next_token="c1")),
            httpx.Response(200, json=_page([_invite(id="i2")])),
        ]
    )

    pages = list(invites.list().pages())

    assert [p.pagination_token for p in pages] == ["c1", None]
    assert [[i.id for i in p.items] for p in pages] == [["i1"], ["i2"]]
    assert pages[0].has_more is True
    assert pages[1].has_more is False


@respx.mock
def test_list_paginator_token_supports_resumption(invites: Invites) -> None:
    respx.get(f"{BASE_URL}/admin/invites").mock(
        side_effect=[
            httpx.Response(200, json=_page([_invite(id="i1")], next_token="c1")),
            httpx.Response(200, json=_page([_invite(id="i2")])),
        ]
    )

    paginator = invites.list()
    pages = paginator.pages()
    next(pages)

    assert paginator.pagination_token == "c1"


@pytest.mark.parametrize("bad_limit", [0, -1, 101, 1000])
def test_list_rejects_out_of_range_limit_before_network(invites: Invites, bad_limit: int) -> None:
    with respx.mock:
        route = respx.get(f"{BASE_URL}/admin/invites").mock(
            return_value=httpx.Response(200, json=_page([]))
        )
        with pytest.raises(ValidationError) as exc:
            invites.list(limit=bad_limit)
        assert route.call_count == 0

    message = str(exc.value)
    assert "limit" in message
    assert str(bad_limit) in message
    assert "1" in message and "100" in message


@respx.mock
@pytest.mark.parametrize("good_limit", [1, 50, 100])
def test_list_accepts_boundary_limits(invites: Invites, good_limit: int) -> None:
    route = respx.get(f"{BASE_URL}/admin/invites").mock(
        return_value=httpx.Response(200, json=_page([]))
    )

    invites.list(limit=good_limit).to_list()

    assert route.calls.last.request.url.params["limit"] == str(good_limit)


@respx.mock
def test_list_surfaces_api_error(invites: Invites) -> None:
    respx.get(f"{BASE_URL}/admin/invites").mock(
        return_value=httpx.Response(
            400,
            json={
                "error": {"code": "INVALID_ARGUMENT", "message": "paginationToken is invalid"},
                "status": 400,
            },
        )
    )

    with pytest.raises(ApiError) as exc:
        invites.list(pagination_token="garbage").to_list()

    assert exc.value.error_code == "INVALID_ARGUMENT"
    assert "paginationToken is invalid" in str(exc.value)


# ---------------------------------------------------------------------------
# create()
# ---------------------------------------------------------------------------


@respx.mock
def test_create_invite(invites: Invites) -> None:
    route = respx.post(f"{BASE_URL}/admin/invites").mock(
        return_value=httpx.Response(200, json=_invite())
    )

    result = invites.create(email="newhire@acme.com", role_bindings=[ORG_BINDING])

    assert isinstance(result, InviteModel)
    assert result.id == INVITE_ID
    assert result.status == InviteStatus.PENDING

    request = route.calls.last.request
    assert request.headers["X-Pinecone-Api-Version"] == ADMIN_API_VERSION
    assert orjson.loads(request.content) == {
        "email": "newhire@acme.com",
        "role_bindings": [{"resource_type": "organization", "role": "OrgMember"}],
    }


@respx.mock
def test_create_omits_resource_id_for_organization_scope(invites: Invites) -> None:
    route = respx.post(f"{BASE_URL}/admin/invites").mock(
        return_value=httpx.Response(200, json=_invite())
    )

    invites.create(
        email="newhire@acme.com",
        role_bindings=[{"resource_type": "organization", "role": "OrgMember"}],
    )

    body = orjson.loads(route.calls.last.request.content)
    assert "resource_id" not in body["role_bindings"][0]


@respx.mock
def test_create_sends_project_scoped_resource_id(invites: Invites) -> None:
    route = respx.post(f"{BASE_URL}/admin/invites").mock(
        return_value=httpx.Response(200, json=_invite())
    )

    invites.create(
        email="newhire@acme.com",
        role_bindings=[
            ORG_BINDING,
            {"resource_type": "project", "role": "ProjectViewer", "resource_id": PROJECT_ID},
        ],
    )

    body = orjson.loads(route.calls.last.request.content)
    assert body["role_bindings"][1] == {
        "resource_type": "project",
        "role": "ProjectViewer",
        "resource_id": PROJECT_ID,
    }


@respx.mock
def test_create_accepts_role_binding_input_models(invites: Invites) -> None:
    route = respx.post(f"{BASE_URL}/admin/invites").mock(
        return_value=httpx.Response(200, json=_invite())
    )

    invites.create(
        email="newhire@acme.com",
        role_bindings=[
            RoleBindingInput(resource_type=ResourceType.ORGANIZATION, role=RoleName.ORG_MEMBER),
            RoleBindingInput(
                resource_type=ResourceType.PROJECT,
                role=RoleName.PROJECT_VIEWER,
                resource_id=PROJECT_ID,
            ),
        ],
    )

    body = orjson.loads(route.calls.last.request.content)
    assert body["role_bindings"] == [
        {"resource_type": "organization", "role": "OrgMember"},
        {"resource_type": "project", "role": "ProjectViewer", "resource_id": PROJECT_ID},
    ]


@respx.mock
def test_create_accepts_mixed_models_and_dicts(invites: Invites) -> None:
    route = respx.post(f"{BASE_URL}/admin/invites").mock(
        return_value=httpx.Response(200, json=_invite())
    )

    invites.create(
        email="newhire@acme.com",
        role_bindings=[
            RoleBindingInput(resource_type="organization", role="OrgMember"),
            {"resource_type": "project", "role": "ProjectEditor", "resource_id": PROJECT_ID},
        ],
    )

    body = orjson.loads(route.calls.last.request.content)
    assert [b["role"] for b in body["role_bindings"]] == ["OrgMember", "ProjectEditor"]


@respx.mock
def test_create_preserves_role_binding_order(invites: Invites) -> None:
    route = respx.post(f"{BASE_URL}/admin/invites").mock(
        return_value=httpx.Response(200, json=_invite())
    )

    invites.create(
        email="newhire@acme.com",
        role_bindings=[
            {"resource_type": "project", "role": "ProjectViewer", "resource_id": PROJECT_ID},
            ORG_BINDING,
            {"resource_type": "project", "role": "DataPlaneEditor", "resource_id": PROJECT_ID},
        ],
    )

    body = orjson.loads(route.calls.last.request.content)
    assert [b["role"] for b in body["role_bindings"]] == [
        "ProjectViewer",
        "OrgMember",
        "DataPlaneEditor",
    ]


@pytest.mark.parametrize("bad_email", ["", "   "])
def test_create_rejects_empty_email_before_network(invites: Invites, bad_email: str) -> None:
    with respx.mock:
        route = respx.post(f"{BASE_URL}/admin/invites").mock(
            return_value=httpx.Response(200, json=_invite())
        )
        with pytest.raises(ValidationError) as exc:
            invites.create(email=bad_email, role_bindings=[ORG_BINDING])
        assert route.call_count == 0

    message = str(exc.value)
    assert "email" in message
    assert "non-empty" in message


def test_create_forwards_unvalidated_email_to_server(invites: Invites) -> None:
    with respx.mock:
        route = respx.post(f"{BASE_URL}/admin/invites").mock(
            return_value=httpx.Response(200, json=_invite(email="not-an-email"))
        )

        invites.create(email="not-an-email", role_bindings=[ORG_BINDING])

        assert orjson.loads(route.calls.last.request.content)["email"] == "not-an-email"


def test_create_rejects_empty_role_bindings_before_network(invites: Invites) -> None:
    with respx.mock:
        route = respx.post(f"{BASE_URL}/admin/invites").mock(
            return_value=httpx.Response(200, json=_invite())
        )
        with pytest.raises(ValidationError) as exc:
            invites.create(email="newhire@acme.com", role_bindings=[])
        assert route.call_count == 0

    message = str(exc.value)
    assert "role_bindings" in message
    assert "non-empty" in message


@pytest.mark.parametrize(
    ("entry", "missing_key"),
    [
        ({"role": "OrgMember"}, "resource_type"),
        ({"resource_type": "organization"}, "role"),
        ({"resource_type": "organization", "role": None}, "role"),
        ({"resource_type": "", "role": "OrgMember"}, "resource_type"),
        ({"resource_type": "organization", "role": "   "}, "role"),
    ],
)
def test_create_names_index_of_entry_missing_a_required_key(
    invites: Invites, entry: dict[str, Any], missing_key: str
) -> None:
    with respx.mock:
        route = respx.post(f"{BASE_URL}/admin/invites").mock(
            return_value=httpx.Response(200, json=_invite())
        )
        with pytest.raises(ValidationError) as exc:
            invites.create(email="newhire@acme.com", role_bindings=[ORG_BINDING, entry])
        assert route.call_count == 0

    message = str(exc.value)
    assert "role_bindings[1]" in message
    assert repr(missing_key) in message


def test_create_names_index_of_entry_with_unknown_resource_type(invites: Invites) -> None:
    with respx.mock:
        route = respx.post(f"{BASE_URL}/admin/invites").mock(
            return_value=httpx.Response(200, json=_invite())
        )
        with pytest.raises(ValidationError) as exc:
            invites.create(
                email="newhire@acme.com",
                role_bindings=[{"resource_type": "workspace", "role": "OrgMember"}],
            )
        assert route.call_count == 0

    message = str(exc.value)
    assert "role_bindings[0]" in message
    assert "resource_type" in message
    assert "workspace" in message


def test_create_names_index_of_entry_with_unknown_role(invites: Invites) -> None:
    with respx.mock:
        route = respx.post(f"{BASE_URL}/admin/invites").mock(
            return_value=httpx.Response(200, json=_invite())
        )
        with pytest.raises(ValidationError) as exc:
            invites.create(
                email="newhire@acme.com",
                role_bindings=[ORG_BINDING, {"resource_type": "organization", "role": "OrgGod"}],
            )
        assert route.call_count == 0

    message = str(exc.value)
    assert "role_bindings[1]" in message
    assert "OrgGod" in message


def test_create_names_index_of_project_entry_missing_resource_id(invites: Invites) -> None:
    with respx.mock:
        route = respx.post(f"{BASE_URL}/admin/invites").mock(
            return_value=httpx.Response(200, json=_invite())
        )
        with pytest.raises(ValidationError) as exc:
            invites.create(
                email="newhire@acme.com",
                role_bindings=[
                    ORG_BINDING,
                    {"resource_type": "project", "role": "ProjectViewer"},
                ],
            )
        assert route.call_count == 0

    message = str(exc.value)
    assert "role_bindings[1]" in message
    assert "resource_id" in message
    assert "project" in message


def test_create_names_index_of_entry_with_unrecognized_key(invites: Invites) -> None:
    with respx.mock:
        route = respx.post(f"{BASE_URL}/admin/invites").mock(
            return_value=httpx.Response(200, json=_invite())
        )
        with pytest.raises(ValidationError) as exc:
            invites.create(
                email="newhire@acme.com",
                role_bindings=[{**ORG_BINDING, "project_id": PROJECT_ID}],
            )
        assert route.call_count == 0

    message = str(exc.value)
    assert "role_bindings[0]" in message
    assert "project_id" in message


def test_create_names_index_of_entry_of_the_wrong_type(invites: Invites) -> None:
    with respx.mock:
        route = respx.post(f"{BASE_URL}/admin/invites").mock(
            return_value=httpx.Response(200, json=_invite())
        )
        with pytest.raises(ValidationError) as exc:
            invites.create(
                email="newhire@acme.com",
                role_bindings=[ORG_BINDING, "OrgMember"],  # type: ignore[list-item]
            )
        assert route.call_count == 0

    message = str(exc.value)
    assert "role_bindings[1]" in message
    assert "str" in message


@respx.mock
def test_create_409_surfaces_code_and_message(invites: Invites) -> None:
    message = "A pending invite already exists for this email; its ID is 9c8e3528."
    respx.post(f"{BASE_URL}/admin/invites").mock(
        return_value=httpx.Response(
            409, json={"error": {"code": "ALREADY_EXISTS", "message": message}, "status": 409}
        )
    )

    with pytest.raises(ConflictError) as exc:
        invites.create(email="newhire@acme.com", role_bindings=[ORG_BINDING])

    assert exc.value.error_code == "ALREADY_EXISTS"
    assert exc.value.status_code == 409
    assert message in str(exc.value)


@respx.mock
def test_create_400_surfaces_server_email_validation(invites: Invites) -> None:
    respx.post(f"{BASE_URL}/admin/invites").mock(
        return_value=httpx.Response(
            400,
            json={
                "error": {
                    "code": "INVALID_ARGUMENT",
                    "message": 'Invalid email address: "not-an-email"',
                },
                "status": 400,
            },
        )
    )

    with pytest.raises(ApiError) as exc:
        invites.create(email="not-an-email", role_bindings=[ORG_BINDING])

    assert exc.value.error_code == "INVALID_ARGUMENT"
    assert "not-an-email" in str(exc.value)


# ---------------------------------------------------------------------------
# describe()
# ---------------------------------------------------------------------------


@respx.mock
def test_describe_invite(invites: Invites) -> None:
    route = respx.get(f"{BASE_URL}/admin/invites/{INVITE_ID}").mock(
        return_value=httpx.Response(200, json=_invite())
    )

    result = invites.describe(invite_id=INVITE_ID)

    assert isinstance(result, InviteModel)
    assert result.id == INVITE_ID
    assert result.email == "newhire@acme.com"
    assert route.calls.last.request.headers["X-Pinecone-Api-Version"] == ADMIN_API_VERSION


@respx.mock
def test_describe_returns_processed_invites(invites: Invites) -> None:
    respx.get(f"{BASE_URL}/admin/invites/{INVITE_ID}").mock(
        return_value=httpx.Response(
            200,
            json=_invite(
                status="processed",
                expires_at=None,
                processed_at="2026-04-15T08:30:00Z",
            ),
        )
    )

    result = invites.describe(invite_id=INVITE_ID)

    assert result.status == InviteStatus.PROCESSED
    assert result.processed_at == "2026-04-15T08:30:00Z"
    assert result.expires_at is None


@pytest.mark.parametrize("bad_id", ["", "   "])
def test_describe_rejects_empty_invite_id(invites: Invites, bad_id: str) -> None:
    with pytest.raises(ValidationError, match="invite_id"):
        invites.describe(invite_id=bad_id)


@respx.mock
def test_describe_404_surfaces_code_and_message(invites: Invites) -> None:
    respx.get(f"{BASE_URL}/admin/invites/{INVITE_ID}").mock(
        return_value=httpx.Response(
            404,
            json={"error": {"code": "NOT_FOUND", "message": "invite not found"}, "status": 404},
        )
    )

    with pytest.raises(NotFoundError) as exc:
        invites.describe(invite_id=INVITE_ID)

    assert exc.value.error_code == "NOT_FOUND"
    assert exc.value.status_code == 404
    assert "invite not found" in str(exc.value)


# ---------------------------------------------------------------------------
# delete()
# ---------------------------------------------------------------------------


@respx.mock
def test_delete_invite(invites: Invites) -> None:
    route = respx.delete(f"{BASE_URL}/admin/invites/{INVITE_ID}").mock(
        return_value=httpx.Response(202)
    )

    assert invites.delete(invite_id=INVITE_ID) is None
    assert route.call_count == 1
    assert route.calls.last.request.headers["X-Pinecone-Api-Version"] == ADMIN_API_VERSION


@pytest.mark.parametrize("bad_id", ["", "   "])
def test_delete_rejects_empty_invite_id(invites: Invites, bad_id: str) -> None:
    with pytest.raises(ValidationError, match="invite_id"):
        invites.delete(invite_id=bad_id)


@respx.mock
def test_delete_404_surfaces_code_and_message(invites: Invites) -> None:
    respx.delete(f"{BASE_URL}/admin/invites/{INVITE_ID}").mock(
        return_value=httpx.Response(
            404,
            json={"error": {"code": "NOT_FOUND", "message": "invite not found"}, "status": 404},
        )
    )

    with pytest.raises(NotFoundError) as exc:
        invites.delete(invite_id=INVITE_ID)

    assert exc.value.error_code == "NOT_FOUND"
    assert "invite not found" in str(exc.value)


@respx.mock
def test_delete_409_on_processed_invite(invites: Invites) -> None:
    message = "Invite 9c8e3528 has already been processed"
    respx.delete(f"{BASE_URL}/admin/invites/{INVITE_ID}").mock(
        return_value=httpx.Response(
            409, json={"error": {"code": "ABORTED", "message": message}, "status": 409}
        )
    )

    with pytest.raises(ConflictError) as exc:
        invites.delete(invite_id=INVITE_ID)

    assert exc.value.error_code == "ABORTED"
    assert exc.value.status_code == 409
    assert message in str(exc.value)


# ---------------------------------------------------------------------------
# resend()
# ---------------------------------------------------------------------------


@respx.mock
def test_resend_invite(invites: Invites) -> None:
    route = respx.post(f"{BASE_URL}/admin/invites/{INVITE_ID}/resend").mock(
        return_value=httpx.Response(200, json=_invite(expires_at="2026-05-28T20:14:00Z"))
    )

    result = invites.resend(invite_id=INVITE_ID)

    assert isinstance(result, InviteModel)
    assert result.status == InviteStatus.PENDING
    assert result.expires_at == "2026-05-28T20:14:00Z"

    request = route.calls.last.request
    assert request.url.path == f"/admin/invites/{INVITE_ID}/resend"
    assert request.headers["X-Pinecone-Api-Version"] == ADMIN_API_VERSION


@respx.mock
def test_resend_sends_no_request_body(invites: Invites) -> None:
    route = respx.post(f"{BASE_URL}/admin/invites/{INVITE_ID}/resend").mock(
        return_value=httpx.Response(200, json=_invite())
    )

    invites.resend(invite_id=INVITE_ID)

    assert route.calls.last.request.content == b""


@pytest.mark.parametrize("bad_id", ["", "   "])
def test_resend_rejects_empty_invite_id(invites: Invites, bad_id: str) -> None:
    with pytest.raises(ValidationError, match="invite_id"):
        invites.resend(invite_id=bad_id)


@respx.mock
def test_resend_404_surfaces_code_and_message(invites: Invites) -> None:
    respx.post(f"{BASE_URL}/admin/invites/{INVITE_ID}/resend").mock(
        return_value=httpx.Response(
            404,
            json={"error": {"code": "NOT_FOUND", "message": "invite not found"}, "status": 404},
        )
    )

    with pytest.raises(NotFoundError) as exc:
        invites.resend(invite_id=INVITE_ID)

    assert exc.value.error_code == "NOT_FOUND"
    assert "invite not found" in str(exc.value)


@respx.mock
def test_resend_409_on_accepted_invite(invites: Invites) -> None:
    message = "Invite has already been accepted and cannot be resent."
    respx.post(f"{BASE_URL}/admin/invites/{INVITE_ID}/resend").mock(
        return_value=httpx.Response(
            409, json={"error": {"code": "ALREADY_EXISTS", "message": message}, "status": 409}
        )
    )

    with pytest.raises(ConflictError) as exc:
        invites.resend(invite_id=INVITE_ID)

    assert exc.value.error_code == "ALREADY_EXISTS"
    assert exc.value.status_code == 409
    assert message in str(exc.value)


@respx.mock
def test_resend_429_is_a_distinct_retry_hinting_error(invites: Invites) -> None:
    """A rate-limited resend must be distinguishable from a permanent 409.

    An agent that cannot tell them apart either hammers a conflict forever or
    gives up on a limit that clears. ``RateLimitError`` is its own subclass, the
    message carries the 429, and ``retry_after`` carries the server's hint.
    """
    message = "Too many invite emails sent; limit is 100 per hour per organization."
    respx.post(f"{BASE_URL}/admin/invites/{INVITE_ID}/resend").mock(
        return_value=httpx.Response(
            429,
            headers={"Retry-After": "1800"},
            json={
                "error": {"code": "RESOURCE_EXHAUSTED", "message": message},
                "status": 429,
            },
        )
    )

    with pytest.raises(RateLimitError) as exc:
        invites.resend(invite_id=INVITE_ID)

    assert not isinstance(exc.value, ConflictError)
    assert exc.value.status_code == 429
    assert exc.value.retry_after == 1800.0
    assert "429" in str(exc.value)
    assert message in str(exc.value)


@respx.mock
def test_resend_429_without_retry_after_header(invites: Invites) -> None:
    respx.post(f"{BASE_URL}/admin/invites/{INVITE_ID}/resend").mock(
        return_value=httpx.Response(
            429,
            json={
                "error": {"code": "RESOURCE_EXHAUSTED", "message": "rate limited"},
                "status": 429,
            },
        )
    )

    with pytest.raises(RateLimitError) as exc:
        invites.resend(invite_id=INVITE_ID)

    assert exc.value.retry_after is None
    assert "429" in str(exc.value)


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def property_invites(hermetic_pinecone_env_module: None) -> Iterator[Invites]:
    """One namespace reused by every example of the property tests below.

    Constructing an ``HTTPClient`` builds a fresh ``ssl.SSLContext`` and parses
    the whole CA bundle into it, which dominated these tests and is the part
    that amplifies on slower CI runners (#345). None of it is exercised: respx
    intercepts at the transport, so no example opens a socket. The examples
    still register their own routes and assert on their own request, so this
    only moves setup out of the loop.
    """
    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    http = HTTPClient(config, ADMIN_API_VERSION)
    yield Invites(http=http)
    http.close()


@settings(max_examples=200, deadline=None)
@given(email=st.text(min_size=1, max_size=254).filter(lambda s: bool(s.strip())))
def test_email_survives_json_body_encoding_unmodified(
    email: str, property_invites: Invites
) -> None:
    """Whatever string the caller passes is the string the server receives.

    The SDK deliberately does not validate address format — the server owns
    that — so it must not mangle one either. Quotes, backslashes, newlines,
    unicode, and surrogate-adjacent codepoints all have to survive JSON
    encoding byte-for-byte, or a rejected address would come back with a
    message about a string the caller never sent.
    """
    with respx.mock:
        route = respx.post(f"{BASE_URL}/admin/invites").mock(
            return_value=httpx.Response(200, json=_invite())
        )
        property_invites.create(email=email, role_bindings=[ORG_BINDING])

        body = orjson.loads(route.calls.last.request.content)

    assert body["email"] == email


@settings(max_examples=100, deadline=None)
@given(
    resource_type=st.sampled_from(list(ResourceType)),
    role=st.sampled_from(list(RoleName)),
)
def test_enum_and_str_role_bindings_produce_identical_payloads(
    resource_type: ResourceType, role: RoleName
) -> None:
    """Enum members, raw strings, and models are three spellings of one binding.

    Callers reach for whichever is at hand, and an agent mixes them within a
    single call. If the three spellings serialized differently, an invite would
    grant different permissions depending on how it was typed.
    """
    resource_id = PROJECT_ID if resource_type is ResourceType.PROJECT else None

    enum_dict: dict[str, Any] = {"resource_type": resource_type, "role": role}
    str_dict: dict[str, Any] = {"resource_type": resource_type.value, "role": role.value}
    if resource_id is not None:
        enum_dict["resource_id"] = resource_id
        str_dict["resource_id"] = resource_id

    model = RoleBindingInput(resource_type=resource_type, role=role, resource_id=resource_id)

    bodies = []
    for binding in (enum_dict, str_dict, model):
        config = PineconeConfig(api_key="test-key", host=BASE_URL)
        namespace = Invites(http=HTTPClient(config, ADMIN_API_VERSION))
        with respx.mock:
            route = respx.post(f"{BASE_URL}/admin/invites").mock(
                return_value=httpx.Response(200, json=_invite())
            )
            namespace.create(email="newhire@acme.com", role_bindings=[binding])
            bodies.append(route.calls.last.request.content)

    assert bodies[0] == bodies[1] == bodies[2]
