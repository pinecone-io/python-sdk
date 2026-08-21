"""Unit tests for the Admin RoleBindings namespace.

Three things carry most of the weight here. The first is the filter
co-requirements: ``principal_id`` without ``principal_type`` and ``resource_id``
without ``resource_type`` are both server-side ``400``s, and the SDK refuses them
before the wire with a message naming both halves — a round trip to learn about a
typo is the worst possible way to find out. The second is the query string:
``list()`` is the only admin list with real filters, so the property test below
pins that every combination of supplied filters produces exactly those parameters
and no others, in either direction. The third is that the server, not the SDK,
owns whether a grant is legal — role-vs-scope, api-key role restrictions, and
plan gating are all 403s whose messages must reach the caller verbatim, so the
tests assert pass-through rather than local rules.

A fourth concern arrived with #408. This namespace accepts ``PrincipalType``,
``ResourceType``, and ``RoleName`` members interchangeably with plain strings, and
it no longer converts them itself: query parameters are resolved at the
``_prepare_params`` boundary (#371) and bodies by the JSON encoder. Both are
structural rather than local, so the tests that pin them assert on
``request.url`` and ``request.content`` and are parametrized over the enum classes
themselves — a member added later is covered without editing them — and a premise
test asserts the mangling those two mechanisms prevent is still real, so the
byte-identity assertions cannot silently become tautologies.

The simulator does not paginate this collection (minicone#50), so the multi-page
cursor walk is exercised against mocks only.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from enum import Enum
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
from pinecone.admin.role_bindings import RoleBindings
from pinecone.errors.exceptions import (
    ApiError,
    ConflictError,
    ForbiddenError,
    NotFoundError,
    ValidationError,
)
from pinecone.models.admin.role_binding import (
    PrincipalType,
    ResourceType,
    RoleBindingModel,
    RoleName,
)
from pinecone.models.pagination import Paginator

BASE_URL = "https://api.test.pinecone.io"

BINDING_ID = "9a8e3528-b9c0-4358-84ce-84c28e91b566"
PRINCIPAL_ID = "f8a3b2c1-4d5e-6f7a-8b9c-0d1e2f3a4b5c"
PROJECT_ID = "a2f7dddb-1597-4eff-9f71-535fde243f58"
ORG_ID = "-ExampleOrgId0000000"

_FILTER_PARAMS = ("principal_type", "principal_id", "resource_type", "resource_id", "role")

_FILTER_ENUM_MEMBERS = [
    *(("principal_type", m) for m in PrincipalType),
    *(("resource_type", m) for m in ResourceType),
    *(("role", m) for m in RoleName),
]


def _enum_member_id(value: object) -> str:
    if isinstance(value, Enum):
        return f"{type(value).__name__}.{value.name}"
    return str(value)


def _binding(
    *,
    id: str = BINDING_ID,
    principal_type: str = "service_account",
    principal_id: str = PRINCIPAL_ID,
    resource_type: str = "project",
    resource_id: str = PROJECT_ID,
    role: str = "DataPlaneEditor",
    created_at: str = "2026-04-10T15:23:00Z",
) -> dict[str, Any]:
    return {
        "id": id,
        "principal_type": principal_type,
        "principal_id": principal_id,
        "resource_type": resource_type,
        "resource_id": resource_id,
        "role": role,
        "created_at": created_at,
    }


def _page(bindings: list[dict[str, Any]], *, next_token: str | None = None) -> dict[str, Any]:
    return {"data": bindings, "pagination": {"next": next_token} if next_token else None}


def _error(code: str, message: str, status: int) -> dict[str, Any]:
    return {"error": {"code": code, "message": message}, "status": status}


@pytest.fixture
def http_client() -> HTTPClient:
    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    return HTTPClient(config, ADMIN_API_VERSION)


@pytest.fixture
def role_bindings(http_client: HTTPClient) -> RoleBindings:
    return RoleBindings(http=http_client)


def test_repr(role_bindings: RoleBindings) -> None:
    assert repr(role_bindings) == "RoleBindings()"


# ---------------------------------------------------------------------------
# list()
# ---------------------------------------------------------------------------


@respx.mock
def test_list_role_bindings(role_bindings: RoleBindings) -> None:
    route = respx.get(f"{BASE_URL}/admin/role-bindings").mock(
        return_value=httpx.Response(200, json=_page([_binding()]))
    )

    result = role_bindings.list()

    assert isinstance(result, Paginator)
    items = result.to_list()
    assert len(items) == 1
    assert isinstance(items[0], RoleBindingModel)
    assert items[0].id == BINDING_ID
    assert items[0].principal_type == "service_account"
    assert items[0].role == "DataPlaneEditor"
    assert route.calls.last.request.url.path == "/admin/role-bindings"


@respx.mock
def test_list_sends_api_version_header(role_bindings: RoleBindings) -> None:
    route = respx.get(f"{BASE_URL}/admin/role-bindings").mock(
        return_value=httpx.Response(200, json=_page([_binding()]))
    )

    role_bindings.list().to_list()

    assert route.calls.last.request.headers["X-Pinecone-Api-Version"] == ADMIN_API_VERSION


@respx.mock
def test_list_is_lazy(role_bindings: RoleBindings) -> None:
    route = respx.get(f"{BASE_URL}/admin/role-bindings").mock(
        return_value=httpx.Response(200, json=_page([_binding()]))
    )

    paginator = role_bindings.list()

    assert route.call_count == 0
    paginator.to_list()
    assert route.call_count == 1


@respx.mock
def test_list_omits_unset_query_params(role_bindings: RoleBindings) -> None:
    route = respx.get(f"{BASE_URL}/admin/role-bindings").mock(
        return_value=httpx.Response(200, json=_page([]))
    )

    role_bindings.list().to_list()

    params = route.calls.last.request.url.params
    for name in (*_FILTER_PARAMS, "limit", "paginationToken"):
        assert name not in params


@respx.mock
def test_list_sends_all_five_filters(role_bindings: RoleBindings) -> None:
    route = respx.get(f"{BASE_URL}/admin/role-bindings").mock(
        return_value=httpx.Response(200, json=_page([_binding()]))
    )

    role_bindings.list(
        principal_type="user",
        principal_id=PRINCIPAL_ID,
        resource_type="organization",
        resource_id=ORG_ID,
        role="OrgMember",
    ).to_list()

    params = route.calls.last.request.url.params
    assert params["principal_type"] == "user"
    assert params["principal_id"] == PRINCIPAL_ID
    assert params["resource_type"] == "organization"
    assert params["resource_id"] == ORG_ID
    assert params["role"] == "OrgMember"


@respx.mock
def test_list_accepts_enum_members_for_filters(role_bindings: RoleBindings) -> None:
    route = respx.get(f"{BASE_URL}/admin/role-bindings").mock(
        return_value=httpx.Response(200, json=_page([]))
    )

    role_bindings.list(
        principal_type=PrincipalType.SERVICE_ACCOUNT,
        resource_type=ResourceType.PROJECT,
        role=RoleName.DATA_PLANE_EDITOR,
    ).to_list()

    params = route.calls.last.request.url.params
    assert params["principal_type"] == "service_account"
    assert params["resource_type"] == "project"
    assert params["role"] == "DataPlaneEditor"


@pytest.mark.parametrize(("param", "member"), _FILTER_ENUM_MEMBERS, ids=_enum_member_id)
@respx.mock
def test_list_query_string_is_identical_for_member_and_string(
    role_bindings: RoleBindings, param: str, member: Enum
) -> None:
    """A member and its own ``.value`` produce the same query string, for every member.

    Parametrized over the enum classes themselves so a member added to any of the
    three is covered without editing this test, and asserted on the real
    ``request.url`` so it cannot pass while the request carries something else.
    """
    route = respx.get(f"{BASE_URL}/admin/role-bindings").mock(
        return_value=httpx.Response(200, json=_page([]))
    )

    role_bindings.list(**{param: member}).to_list()
    role_bindings.list(**{param: member.value}).to_list()

    assert str(route.calls[0].request.url) == str(route.calls[1].request.url)
    assert route.calls[0].request.url.params[param] == member.value
    assert type(member).__name__ not in str(route.calls[0].request.url)


@respx.mock
def test_list_principal_type_alone_needs_no_principal_id(role_bindings: RoleBindings) -> None:
    route = respx.get(f"{BASE_URL}/admin/role-bindings").mock(
        return_value=httpx.Response(200, json=_page([]))
    )

    role_bindings.list(principal_type="invite").to_list()

    params = route.calls.last.request.url.params
    assert params["principal_type"] == "invite"
    assert "principal_id" not in params


@respx.mock
def test_list_principal_id_without_principal_type_raises_before_network(
    role_bindings: RoleBindings,
) -> None:
    route = respx.get(f"{BASE_URL}/admin/role-bindings").mock(
        return_value=httpx.Response(200, json=_page([]))
    )

    with pytest.raises(ValidationError) as exc:
        role_bindings.list(principal_id=PRINCIPAL_ID)

    message = str(exc.value)
    assert "principal_id" in message
    assert "principal_type" in message
    assert "'service_account'" in message
    assert route.call_count == 0


@respx.mock
def test_list_resource_id_without_resource_type_raises_before_network(
    role_bindings: RoleBindings,
) -> None:
    route = respx.get(f"{BASE_URL}/admin/role-bindings").mock(
        return_value=httpx.Response(200, json=_page([]))
    )

    with pytest.raises(ValidationError) as exc:
        role_bindings.list(resource_id=PROJECT_ID)

    message = str(exc.value)
    assert "resource_id" in message
    assert "resource_type" in message
    assert "'project'" in message
    assert route.call_count == 0


@respx.mock
def test_list_rejects_unknown_principal_type(role_bindings: RoleBindings) -> None:
    route = respx.get(f"{BASE_URL}/admin/role-bindings").mock(
        return_value=httpx.Response(200, json=_page([]))
    )

    with pytest.raises(ValidationError) as exc:
        role_bindings.list(principal_type="robot")

    message = str(exc.value)
    assert "principal_type" in message
    assert "'robot'" in message
    assert "'api_key'" in message
    assert route.call_count == 0


@respx.mock
def test_list_rejects_unknown_resource_type(role_bindings: RoleBindings) -> None:
    with pytest.raises(ValidationError) as exc:
        role_bindings.list(resource_type="index")

    assert "resource_type" in str(exc.value)
    assert "'index'" in str(exc.value)


@respx.mock
def test_list_rejects_unknown_role(role_bindings: RoleBindings) -> None:
    route = respx.get(f"{BASE_URL}/admin/role-bindings").mock(
        return_value=httpx.Response(200, json=_page([]))
    )

    with pytest.raises(ValidationError) as exc:
        role_bindings.list(role="Sysadmin")

    message = str(exc.value)
    assert "role" in message
    assert "'Sysadmin'" in message
    assert "'DataPlaneEditor'" in message
    assert route.call_count == 0


@respx.mock
def test_list_sends_limit(role_bindings: RoleBindings) -> None:
    route = respx.get(f"{BASE_URL}/admin/role-bindings").mock(
        return_value=httpx.Response(200, json=_page([]))
    )

    role_bindings.list(limit=25).to_list()

    assert route.calls.last.request.url.params["limit"] == "25"


@respx.mock
@pytest.mark.parametrize("bad_limit", [0, -1, 101, 1000])
def test_list_rejects_out_of_range_limit_before_network(
    role_bindings: RoleBindings, bad_limit: int
) -> None:
    route = respx.get(f"{BASE_URL}/admin/role-bindings").mock(
        return_value=httpx.Response(200, json=_page([]))
    )

    with pytest.raises(ValidationError):
        role_bindings.list(limit=bad_limit)

    assert route.call_count == 0


@respx.mock
@pytest.mark.parametrize("good_limit", [1, 100])
def test_list_accepts_boundary_limits(role_bindings: RoleBindings, good_limit: int) -> None:
    route = respx.get(f"{BASE_URL}/admin/role-bindings").mock(
        return_value=httpx.Response(200, json=_page([]))
    )

    role_bindings.list(limit=good_limit).to_list()

    assert route.calls.last.request.url.params["limit"] == str(good_limit)


@respx.mock
def test_list_sends_initial_pagination_token(role_bindings: RoleBindings) -> None:
    route = respx.get(f"{BASE_URL}/admin/role-bindings").mock(
        return_value=httpx.Response(200, json=_page([]))
    )

    role_bindings.list(pagination_token="resume-here").to_list()

    assert route.calls.last.request.url.params["paginationToken"] == "resume-here"


@respx.mock
def test_list_empty_page(role_bindings: RoleBindings) -> None:
    respx.get(f"{BASE_URL}/admin/role-bindings").mock(
        return_value=httpx.Response(200, json=_page([]))
    )

    assert role_bindings.list().to_list() == []


@respx.mock
def test_list_tolerates_absent_pagination_key(role_bindings: RoleBindings) -> None:
    respx.get(f"{BASE_URL}/admin/role-bindings").mock(
        return_value=httpx.Response(200, json={"data": [_binding()]})
    )

    assert len(role_bindings.list().to_list()) == 1


@respx.mock
def test_list_follows_pagination_cursor_verbatim(role_bindings: RoleBindings) -> None:
    cursor = "eyJsYXN0X2lkIjoiOWE4ZTM1MjgifQ=="
    route = respx.get(f"{BASE_URL}/admin/role-bindings").mock(
        side_effect=[
            httpx.Response(200, json=_page([_binding(id="one")], next_token=cursor)),
            httpx.Response(200, json=_page([_binding(id="two")])),
        ]
    )

    items = role_bindings.list().to_list()

    assert [b.id for b in items] == ["one", "two"]
    assert "paginationToken" not in route.calls[0].request.url.params
    assert route.calls[1].request.url.params["paginationToken"] == cursor


@respx.mock
def test_list_carries_filters_and_limit_onto_later_pages(role_bindings: RoleBindings) -> None:
    route = respx.get(f"{BASE_URL}/admin/role-bindings").mock(
        side_effect=[
            httpx.Response(200, json=_page([_binding()], next_token="next-1")),
            httpx.Response(200, json=_page([])),
        ]
    )

    role_bindings.list(principal_type="user", role="OrgOwner", limit=7).to_list()

    second = route.calls[1].request.url.params
    assert second["principal_type"] == "user"
    assert second["role"] == "OrgOwner"
    assert second["limit"] == "7"
    assert second["paginationToken"] == "next-1"


@respx.mock
def test_list_stops_on_pagination_with_null_next(role_bindings: RoleBindings) -> None:
    route = respx.get(f"{BASE_URL}/admin/role-bindings").mock(
        return_value=httpx.Response(200, json={"data": [_binding()], "pagination": {"next": None}})
    )

    assert len(role_bindings.list().to_list()) == 1
    assert route.call_count == 1


@respx.mock
def test_list_pages_exposes_page_level_access(role_bindings: RoleBindings) -> None:
    respx.get(f"{BASE_URL}/admin/role-bindings").mock(
        side_effect=[
            httpx.Response(200, json=_page([_binding(id="one")], next_token="cursor-1")),
            httpx.Response(200, json=_page([_binding(id="two")])),
        ]
    )

    pages = list(role_bindings.list().pages())

    assert [p.pagination_token for p in pages] == ["cursor-1", None]
    assert [len(p.items) for p in pages] == [1, 1]


@respx.mock
def test_list_surfaces_server_400_for_expired_cursor(role_bindings: RoleBindings) -> None:
    respx.get(f"{BASE_URL}/admin/role-bindings").mock(
        return_value=httpx.Response(
            400, json=_error("INVALID_ARGUMENT", "Invalid pagination token", 400)
        )
    )

    with pytest.raises(ApiError) as exc:
        role_bindings.list(pagination_token="stale").to_list()

    assert "Invalid pagination token" in str(exc.value)


@respx.mock
def test_list_yields_unknown_role_from_server_as_raw_string(role_bindings: RoleBindings) -> None:
    """A role added after this SDK release must read back, not raise.

    ``role`` is validated on the way out and deliberately untyped on the way in:
    a filter typo should fail fast, but a response carrying a role this release
    has never heard of has to decode, or the SDK breaks on a server upgrade.
    """
    respx.get(f"{BASE_URL}/admin/role-bindings").mock(
        return_value=httpx.Response(200, json=_page([_binding(role="FutureRole")]))
    )

    assert role_bindings.list().to_list()[0].role == "FutureRole"


# ---------------------------------------------------------------------------
# create()
# ---------------------------------------------------------------------------


@respx.mock
def test_create_role_binding(role_bindings: RoleBindings) -> None:
    route = respx.post(f"{BASE_URL}/admin/role-bindings").mock(
        return_value=httpx.Response(200, json=_binding())
    )

    result = role_bindings.create(
        principal_type="service_account",
        principal_id=PRINCIPAL_ID,
        resource_type="project",
        resource_id=PROJECT_ID,
        role="DataPlaneEditor",
    )

    assert isinstance(result, RoleBindingModel)
    assert result.id == BINDING_ID
    assert orjson.loads(route.calls.last.request.content) == {
        "principal_type": "service_account",
        "principal_id": PRINCIPAL_ID,
        "resource_type": "project",
        "resource_id": PROJECT_ID,
        "role": "DataPlaneEditor",
    }


@respx.mock
def test_create_omits_resource_id_for_organization_scope(role_bindings: RoleBindings) -> None:
    route = respx.post(f"{BASE_URL}/admin/role-bindings").mock(
        return_value=httpx.Response(
            200, json=_binding(resource_type="organization", resource_id=ORG_ID, role="OrgMember")
        )
    )

    role_bindings.create(
        principal_type="user",
        principal_id=PRINCIPAL_ID,
        resource_type="organization",
        role="OrgMember",
    )

    body = orjson.loads(route.calls.last.request.content)
    assert "resource_id" not in body
    assert body["resource_type"] == "organization"


@respx.mock
def test_create_forwards_explicit_resource_id_for_organization_scope(
    role_bindings: RoleBindings,
) -> None:
    """An explicit org id is passed through, not dropped, so the server can 404 it.

    The backend accepts ``resource_id`` at organization scope only when it names
    the caller's own organization and answers ``404`` otherwise. Swallowing the
    value would turn that check into a silent success against the wrong org.
    """
    route = respx.post(f"{BASE_URL}/admin/role-bindings").mock(
        return_value=httpx.Response(
            200, json=_binding(resource_type="organization", resource_id=ORG_ID, role="OrgMember")
        )
    )

    role_bindings.create(
        principal_type="user",
        principal_id=PRINCIPAL_ID,
        resource_type="organization",
        resource_id=ORG_ID,
        role="OrgMember",
    )

    assert orjson.loads(route.calls.last.request.content)["resource_id"] == ORG_ID


@respx.mock
def test_create_accepts_enum_members(role_bindings: RoleBindings) -> None:
    route = respx.post(f"{BASE_URL}/admin/role-bindings").mock(
        return_value=httpx.Response(200, json=_binding())
    )

    role_bindings.create(
        principal_type=PrincipalType.SERVICE_ACCOUNT,
        principal_id=PRINCIPAL_ID,
        resource_type=ResourceType.PROJECT,
        resource_id=PROJECT_ID,
        role=RoleName.DATA_PLANE_EDITOR,
    )

    assert orjson.loads(route.calls.last.request.content) == {
        "principal_type": "service_account",
        "principal_id": PRINCIPAL_ID,
        "resource_type": "project",
        "resource_id": PROJECT_ID,
        "role": "DataPlaneEditor",
    }


@respx.mock
def test_create_enum_and_string_bodies_are_identical(role_bindings: RoleBindings) -> None:
    route = respx.post(f"{BASE_URL}/admin/role-bindings").mock(
        return_value=httpx.Response(200, json=_binding())
    )

    role_bindings.create(
        principal_type=PrincipalType.SERVICE_ACCOUNT,
        principal_id=PRINCIPAL_ID,
        resource_type=ResourceType.PROJECT,
        resource_id=PROJECT_ID,
        role=RoleName.DATA_PLANE_EDITOR,
    )
    role_bindings.create(
        principal_type="service_account",
        principal_id=PRINCIPAL_ID,
        resource_type="project",
        resource_id=PROJECT_ID,
        role="DataPlaneEditor",
    )

    assert route.calls[0].request.content == route.calls[1].request.content


@pytest.mark.parametrize("member", list(PrincipalType), ids=_enum_member_id)
@respx.mock
def test_create_body_bytes_are_identical_for_member_and_string(
    role_bindings: RoleBindings, member: PrincipalType
) -> None:
    """``principal_type`` is the one enum-valued input on this surface that feeds a BODY.

    Query parameters are normalized structurally at ``_prepare_params`` (#371);
    bodies are not, and are instead safe because the JSON encoder resolves a
    member to its value. This pins that for every member, on the real
    ``request.content``, so the mechanism cannot quietly stop holding (#408).
    """
    route = respx.post(f"{BASE_URL}/admin/role-bindings").mock(
        return_value=httpx.Response(200, json=_binding())
    )

    role_bindings.create(
        principal_type=member,
        principal_id=PRINCIPAL_ID,
        resource_type="organization",
        role="OrgMember",
    )
    role_bindings.create(
        principal_type=member.value,
        principal_id=PRINCIPAL_ID,
        resource_type="organization",
        role="OrgMember",
    )

    assert route.calls[0].request.content == route.calls[1].request.content
    assert orjson.loads(route.calls[0].request.content)["principal_type"] == member.value
    assert b"PrincipalType" not in route.calls[0].request.content


def test_enum_members_would_be_mangled_without_the_encoder_and_the_boundary() -> None:
    """The premise both preceding tests rest on: ``str()`` of these members is wrong.

    ``role_bindings`` normalizes nothing itself (#408 removed its ``_as_str``), so
    if this ever stops being true the two byte-identity tests above become
    tautologies and the guarantee they describe needs restating.
    """
    assert len(_FILTER_ENUM_MEMBERS) == len(PrincipalType) + len(ResourceType) + len(RoleName)
    for _, member in _FILTER_ENUM_MEMBERS:
        assert str(member) != member.value
        assert str(httpx.QueryParams({"k": member})) == f"k={type(member).__name__}.{member.name}"
        assert orjson.loads(orjson.dumps({"k": member}))["k"] == member.value


@respx.mock
def test_create_rejects_unknown_principal_type_before_network(
    role_bindings: RoleBindings,
) -> None:
    route = respx.post(f"{BASE_URL}/admin/role-bindings").mock(
        return_value=httpx.Response(200, json=_binding())
    )

    with pytest.raises(ValidationError) as exc:
        role_bindings.create(
            principal_type="robot",
            principal_id=PRINCIPAL_ID,
            resource_type="organization",
            role="OrgMember",
        )

    message = str(exc.value)
    assert "principal_type" in message
    assert "'robot'" in message
    for allowed in ("'user'", "'service_account'", "'api_key'", "'invite'"):
        assert allowed in message
    assert route.call_count == 0


@respx.mock
def test_create_rejects_unknown_resource_type_before_network(role_bindings: RoleBindings) -> None:
    route = respx.post(f"{BASE_URL}/admin/role-bindings").mock(
        return_value=httpx.Response(200, json=_binding())
    )

    with pytest.raises(ValidationError) as exc:
        role_bindings.create(
            principal_type="user",
            principal_id=PRINCIPAL_ID,
            resource_type="index",
            role="OrgMember",
        )

    message = str(exc.value)
    assert "resource_type" in message
    assert "'index'" in message
    assert "'organization'" in message
    assert "'project'" in message
    assert route.call_count == 0


@respx.mock
def test_create_rejects_unknown_role_naming_every_allowed_value(
    role_bindings: RoleBindings,
) -> None:
    route = respx.post(f"{BASE_URL}/admin/role-bindings").mock(
        return_value=httpx.Response(200, json=_binding())
    )

    with pytest.raises(ValidationError) as exc:
        role_bindings.create(
            principal_type="user",
            principal_id=PRINCIPAL_ID,
            resource_type="organization",
            role="Sysadmin",
        )

    message = str(exc.value)
    assert "role" in message
    assert "'Sysadmin'" in message
    for allowed in [r.value for r in RoleName]:
        assert repr(allowed) in message
    assert route.call_count == 0


@respx.mock
@pytest.mark.parametrize("bad_id", ["", "   "])
def test_create_rejects_empty_principal_id(role_bindings: RoleBindings, bad_id: str) -> None:
    route = respx.post(f"{BASE_URL}/admin/role-bindings").mock(
        return_value=httpx.Response(200, json=_binding())
    )

    with pytest.raises(ValidationError) as exc:
        role_bindings.create(
            principal_type="user",
            principal_id=bad_id,
            resource_type="organization",
            role="OrgMember",
        )

    assert "principal_id" in str(exc.value)
    assert route.call_count == 0


@respx.mock
def test_create_project_scope_requires_resource_id_before_network(
    role_bindings: RoleBindings,
) -> None:
    route = respx.post(f"{BASE_URL}/admin/role-bindings").mock(
        return_value=httpx.Response(200, json=_binding())
    )

    with pytest.raises(ValidationError) as exc:
        role_bindings.create(
            principal_type="service_account",
            principal_id=PRINCIPAL_ID,
            resource_type="project",
            role="DataPlaneEditor",
        )

    message = str(exc.value)
    assert "resource_id" in message
    assert "project" in message
    assert route.call_count == 0


@respx.mock
def test_create_does_not_enforce_role_scope_compatibility_locally(
    role_bindings: RoleBindings,
) -> None:
    """An org role at project scope reaches the server, which owns the rule.

    ``check_valid_role_binding`` gates role-vs-scope, api-key role restrictions,
    and plan availability, and the answers vary by organization plan. Replicating
    them here would produce confidently wrong refusals, so the SDK sends the
    request and lets the 403 explain itself.
    """
    route = respx.post(f"{BASE_URL}/admin/role-bindings").mock(
        return_value=httpx.Response(
            403,
            json=_error(
                "PERMISSION_DENIED",
                "The OrgOwner role cannot be bound for resource_type=project",
                403,
            ),
        )
    )

    with pytest.raises(ForbiddenError) as exc:
        role_bindings.create(
            principal_type="user",
            principal_id=PRINCIPAL_ID,
            resource_type="project",
            resource_id=PROJECT_ID,
            role="OrgOwner",
        )

    assert "cannot be bound for resource_type=project" in str(exc.value)
    assert orjson.loads(route.calls.last.request.content)["role"] == "OrgOwner"


@respx.mock
def test_create_403_plan_restriction_message_passes_through(role_bindings: RoleBindings) -> None:
    respx.post(f"{BASE_URL}/admin/role-bindings").mock(
        return_value=httpx.Response(
            403,
            json=_error(
                "PERMISSION_DENIED",
                "The ControlPlaneEditor role is not available on the starter plan. "
                "Upgrade to a standard or enterprise plan to use this feature.",
                403,
            ),
        )
    )

    with pytest.raises(ForbiddenError) as exc:
        role_bindings.create(
            principal_type="api_key",
            principal_id=PRINCIPAL_ID,
            resource_type="project",
            resource_id=PROJECT_ID,
            role="ControlPlaneEditor",
        )

    assert "not available on the starter plan" in str(exc.value)
    assert "Upgrade to a standard or enterprise plan" in str(exc.value)


@respx.mock
def test_create_404_for_unknown_principal(role_bindings: RoleBindings) -> None:
    respx.post(f"{BASE_URL}/admin/role-bindings").mock(
        return_value=httpx.Response(404, json=_error("NOT_FOUND", "User nope not found", 404))
    )

    with pytest.raises(NotFoundError) as exc:
        role_bindings.create(
            principal_type="user",
            principal_id="nope",
            resource_type="organization",
            role="OrgMember",
        )

    assert "not found" in str(exc.value)


@respx.mock
def test_create_409_on_duplicate_binding(role_bindings: RoleBindings) -> None:
    respx.post(f"{BASE_URL}/admin/role-bindings").mock(
        return_value=httpx.Response(
            409,
            json=_error(
                "ALREADY_EXISTS", "A role binding with these attributes already exists.", 409
            ),
        )
    )

    with pytest.raises(ConflictError) as exc:
        role_bindings.create(
            principal_type="user",
            principal_id=PRINCIPAL_ID,
            resource_type="organization",
            role="OrgMember",
        )

    assert "already exists" in str(exc.value)


@respx.mock
def test_create_409_on_binding_to_accepted_invite(role_bindings: RoleBindings) -> None:
    respx.post(f"{BASE_URL}/admin/role-bindings").mock(
        return_value=httpx.Response(
            409,
            json=_error(
                "ALREADY_EXISTS",
                "The invite has already been accepted; manage role bindings on the "
                "resulting user instead.",
                409,
            ),
        )
    )

    with pytest.raises(ConflictError) as exc:
        role_bindings.create(
            principal_type="invite",
            principal_id=PRINCIPAL_ID,
            resource_type="organization",
            role="OrgMember",
        )

    assert "manage role bindings on the resulting user instead" in str(exc.value)


@respx.mock
def test_create_400_passes_server_message_through(role_bindings: RoleBindings) -> None:
    respx.post(f"{BASE_URL}/admin/role-bindings").mock(
        return_value=httpx.Response(
            400,
            json=_error(
                "INVALID_ARGUMENT", "Missing resource_id for project-scoped role binding", 400
            ),
        )
    )

    with pytest.raises(ApiError) as exc:
        role_bindings.create(
            principal_type="user",
            principal_id=PRINCIPAL_ID,
            resource_type="organization",
            role="OrgMember",
        )

    assert "Missing resource_id for project-scoped role binding" in str(exc.value)


# ---------------------------------------------------------------------------
# describe()
# ---------------------------------------------------------------------------


@respx.mock
def test_describe_role_binding(role_bindings: RoleBindings) -> None:
    route = respx.get(f"{BASE_URL}/admin/role-bindings/{BINDING_ID}").mock(
        return_value=httpx.Response(200, json=_binding())
    )

    result = role_bindings.describe(role_binding_id=BINDING_ID)

    assert isinstance(result, RoleBindingModel)
    assert result.principal_id == PRINCIPAL_ID
    assert result.created_at == "2026-04-10T15:23:00Z"
    assert route.calls.last.request.url.path == f"/admin/role-bindings/{BINDING_ID}"


@respx.mock
@pytest.mark.parametrize("bad_id", ["", "   "])
def test_describe_rejects_empty_id(role_bindings: RoleBindings, bad_id: str) -> None:
    with pytest.raises(ValidationError) as exc:
        role_bindings.describe(role_binding_id=bad_id)

    assert "role_binding_id" in str(exc.value)


@respx.mock
def test_describe_404_surfaces_code_and_message(role_bindings: RoleBindings) -> None:
    respx.get(f"{BASE_URL}/admin/role-bindings/{BINDING_ID}").mock(
        return_value=httpx.Response(
            404, json=_error("NOT_FOUND", f"Role binding {BINDING_ID} not found", 404)
        )
    )

    with pytest.raises(NotFoundError) as exc:
        role_bindings.describe(role_binding_id=BINDING_ID)

    assert "not found" in str(exc.value)


# ---------------------------------------------------------------------------
# delete()
# ---------------------------------------------------------------------------


@respx.mock
def test_delete_role_binding(role_bindings: RoleBindings) -> None:
    route = respx.delete(f"{BASE_URL}/admin/role-bindings/{BINDING_ID}").mock(
        return_value=httpx.Response(202)
    )

    assert role_bindings.delete(role_binding_id=BINDING_ID) is None
    assert route.calls.last.request.url.path == f"/admin/role-bindings/{BINDING_ID}"


@respx.mock
def test_delete_sends_no_request_body(role_bindings: RoleBindings) -> None:
    route = respx.delete(f"{BASE_URL}/admin/role-bindings/{BINDING_ID}").mock(
        return_value=httpx.Response(202)
    )

    role_bindings.delete(role_binding_id=BINDING_ID)

    assert not route.calls.last.request.content


@respx.mock
@pytest.mark.parametrize("bad_id", ["", "   "])
def test_delete_rejects_empty_id(role_bindings: RoleBindings, bad_id: str) -> None:
    with pytest.raises(ValidationError) as exc:
        role_bindings.delete(role_binding_id=bad_id)

    assert "role_binding_id" in str(exc.value)


@respx.mock
def test_delete_repeat_is_404_not_a_second_success(role_bindings: RoleBindings) -> None:
    route = respx.delete(f"{BASE_URL}/admin/role-bindings/{BINDING_ID}").mock(
        side_effect=[
            httpx.Response(202),
            httpx.Response(404, json=_error("NOT_FOUND", "Role binding not found", 404)),
        ]
    )

    assert role_bindings.delete(role_binding_id=BINDING_ID) is None
    with pytest.raises(NotFoundError):
        role_bindings.delete(role_binding_id=BINDING_ID)

    assert route.call_count == 2


@respx.mock
def test_delete_409_on_last_org_owner(role_bindings: RoleBindings) -> None:
    respx.delete(f"{BASE_URL}/admin/role-bindings/{BINDING_ID}").mock(
        return_value=httpx.Response(
            409,
            json=_error(
                "ABORTED",
                "Deleting this role binding would leave the organization without any owners. "
                "Assign OrgOwner to another user in the organization first.",
                409,
            ),
        )
    )

    with pytest.raises(ConflictError) as exc:
        role_bindings.delete(role_binding_id=BINDING_ID)

    assert "without any owners" in str(exc.value)


@respx.mock
def test_delete_409_on_last_org_membership_binding(role_bindings: RoleBindings) -> None:
    respx.delete(f"{BASE_URL}/admin/role-bindings/{BINDING_ID}").mock(
        return_value=httpx.Response(
            409,
            json=_error(
                "ABORTED",
                "Deleting this role binding would remove user u1 from the organization. "
                "Remove all other role bindings for the user first.",
                409,
            ),
        )
    )

    with pytest.raises(ConflictError) as exc:
        role_bindings.delete(role_binding_id=BINDING_ID)

    assert "would remove user u1 from the organization" in str(exc.value)


@respx.mock
def test_delete_409_when_user_management_is_federated(role_bindings: RoleBindings) -> None:
    respx.delete(f"{BASE_URL}/admin/role-bindings/{BINDING_ID}").mock(
        return_value=httpx.Response(
            409,
            json=_error(
                "ABORTED",
                "User and role management is controlled by your identity provider (scim). "
                "Switch user management back to manual to modify users, invites, or role "
                "bindings via the API.",
                409,
            ),
        )
    )

    with pytest.raises(ConflictError) as exc:
        role_bindings.delete(role_binding_id=BINDING_ID)

    assert "controlled by your identity provider" in str(exc.value)


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

_PRINCIPAL_TYPES = [p.value for p in PrincipalType]
_RESOURCE_TYPES = [r.value for r in ResourceType]
_ROLE_NAMES = [r.value for r in RoleName]


@pytest.fixture(scope="module")
def property_role_bindings(hermetic_pinecone_env_module: None) -> Iterator[RoleBindings]:
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
    yield RoleBindings(http=http)
    http.close()


@settings(max_examples=300, deadline=None)
@given(
    principal_type=st.one_of(st.none(), st.sampled_from(_PRINCIPAL_TYPES)),
    principal_id=st.one_of(st.none(), st.text(min_size=1, max_size=40)),
    resource_type=st.one_of(st.none(), st.sampled_from(_RESOURCE_TYPES)),
    resource_id=st.one_of(st.none(), st.text(min_size=1, max_size=40)),
    role=st.one_of(st.none(), st.sampled_from(_ROLE_NAMES)),
    limit=st.one_of(st.none(), st.integers(min_value=1, max_value=100)),
)
def test_list_query_params_match_supplied_filters_exactly(
    principal_type: str | None,
    principal_id: str | None,
    resource_type: str | None,
    resource_id: str | None,
    role: str | None,
    limit: int | None,
    property_role_bindings: RoleBindings,
) -> None:
    """The query string is exactly the non-``None`` filters — nothing dropped or invented.

    Filters are AND-combined server-side, so a dropped parameter silently widens
    the result set and a spurious one silently narrows it. Both failures look like
    a correct answer to the caller, which is why this is asserted in both
    directions over every combination rather than one example at a time.
    """
    supplied = {
        "principal_type": principal_type,
        "principal_id": principal_id,
        "resource_type": resource_type,
        "resource_id": resource_id,
        "role": role,
    }
    expected = {k: v for k, v in supplied.items() if v is not None}

    with respx.mock:
        route = respx.get(f"{BASE_URL}/admin/role-bindings").mock(
            return_value=httpx.Response(200, json=_page([]))
        )
        try:
            property_role_bindings.list(
                principal_type=principal_type,
                principal_id=principal_id,
                resource_type=resource_type,
                resource_id=resource_id,
                role=role,
                limit=limit,
            ).to_list()
        except ValidationError:
            assert (principal_id is not None and principal_type is None) or (
                resource_id is not None and resource_type is None
            )
            assert route.call_count == 0
            return

        params = route.calls.last.request.url.params

    on_the_wire = {k: params[k] for k in _FILTER_PARAMS if k in params}
    assert on_the_wire == expected
    assert ("limit" in params) == (limit is not None)
    if limit is not None:
        assert params["limit"] == str(limit)


@settings(max_examples=200, deadline=None)
@given(
    filters=st.sets(st.sampled_from(_FILTER_PARAMS), min_size=0, max_size=5),
    role=st.sampled_from(_ROLE_NAMES),
)
def test_list_never_sends_a_filter_the_caller_omitted(
    filters: set[str], role: str, property_role_bindings: RoleBindings
) -> None:
    """Omitted filters are absent from the URL, not sent empty.

    An empty-string query value is not the same as an absent one: the server
    treats ``?resource_id=`` as absent today, but relying on that would make the
    SDK's meaning depend on a server-side convention it does not own.
    """
    values = {
        "principal_type": "user",
        "principal_id": PRINCIPAL_ID,
        "resource_type": "project",
        "resource_id": PROJECT_ID,
        "role": role,
    }
    kwargs: dict[str, Any] = {k: values[k] for k in filters}
    if "principal_id" in kwargs:
        kwargs["principal_type"] = values["principal_type"]
    if "resource_id" in kwargs:
        kwargs["resource_type"] = values["resource_type"]

    with respx.mock:
        route = respx.get(f"{BASE_URL}/admin/role-bindings").mock(
            return_value=httpx.Response(200, json=_page([]))
        )
        property_role_bindings.list(**kwargs).to_list()
        params = route.calls.last.request.url.params

    for name in _FILTER_PARAMS:
        if name in kwargs:
            assert params[name] == kwargs[name]
        else:
            assert name not in params


@settings(max_examples=200, deadline=None)
@given(
    principal_type=st.sampled_from(_PRINCIPAL_TYPES),
    resource_type=st.sampled_from(_RESOURCE_TYPES),
    role=st.sampled_from(_ROLE_NAMES),
)
def test_create_body_carries_every_known_combination_unmodified(
    principal_type: str, resource_type: str, role: str
) -> None:
    """Every enum triple the SDK accepts reaches the wire verbatim.

    The SDK checks membership and nothing else, so no combination may be
    reshaped, reordered into a different key set, or rejected locally — the
    server owns which triples are legal, and it can only say so if it receives
    them.
    """
    resource_id = PROJECT_ID if resource_type == "project" else None

    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    namespace = RoleBindings(http=HTTPClient(config, ADMIN_API_VERSION))

    with respx.mock:
        route = respx.post(f"{BASE_URL}/admin/role-bindings").mock(
            return_value=httpx.Response(200, json=_binding())
        )
        namespace.create(
            principal_type=principal_type,
            principal_id=PRINCIPAL_ID,
            resource_type=resource_type,
            resource_id=resource_id,
            role=role,
        )
        body = orjson.loads(route.calls.last.request.content)

    expected = {
        "principal_type": principal_type,
        "principal_id": PRINCIPAL_ID,
        "resource_type": resource_type,
        "role": role,
    }
    if resource_id is not None:
        expected["resource_id"] = resource_id
    assert body == expected


def test_every_role_name_and_principal_type_pair_is_accepted_locally() -> None:
    """No enum pair is refused before the wire.

    Role-vs-principal-type legality is the server's call — an ``api_key``
    principal accepts only a subset of roles — so an exhaustive sweep pins that
    the SDK's own validation never anticipates it.
    """
    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    namespace = RoleBindings(http=HTTPClient(config, ADMIN_API_VERSION))

    with respx.mock:
        respx.post(f"{BASE_URL}/admin/role-bindings").mock(
            return_value=httpx.Response(200, json=_binding())
        )
        for principal_type, role in itertools.product(_PRINCIPAL_TYPES, _ROLE_NAMES):
            namespace.create(
                principal_type=principal_type,
                principal_id=PRINCIPAL_ID,
                resource_type="organization",
                role=role,
            )
