"""Unit tests for the Admin ServiceAccounts namespace.

Three things carry most of the weight here. The first is secret hygiene: the
``client_secret`` returned by ``create()`` and ``rotate_secret()`` is the only
copy that will ever exist, so it must reach the caller intact while never
appearing in a log record or a ``repr``. The second is ``update()``'s
fieldless-patch guard — the server answers a fieldless PATCH with a no-op 200
that bumps ``updated_at``, so a caller who misspelled the keyword gets an
apparent success; these tests pin that the SDK refuses before the wire instead.
The third is the cursor walk: ``list()`` returns a lazy paginator, so nothing is
requested until iteration, the cursor goes back byte-for-byte, and a page with
absent or null ``pagination`` ends the walk rather than looping. The simulator
does not paginate this collection (minicone#50), so the multi-page walk is
exercised against mocks only.
"""

from __future__ import annotations

import logging
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
from pinecone.admin.service_accounts import ServiceAccounts
from pinecone.errors.exceptions import (
    ApiError,
    ForbiddenError,
    NotFoundError,
    ValidationError,
)
from pinecone.models.admin.role_binding import ResourceType, RoleBindingInput, RoleName
from pinecone.models.admin.service_account import ServiceAccountModel, ServiceAccountWithSecret
from pinecone.models.pagination import Paginator

BASE_URL = "https://api.test.pinecone.io"

ACCOUNT_ID = "f8a3b2c1-4d5e-6f7a-8b9c-0d1e2f3a4b5c"
PROJECT_ID = "a2f7dddb-1597-4eff-9f71-535fde243f58"
CLIENT_ID = "l3Ow0CmFyc4jOONcwiKUCRqQKN0tiCAn"
SECRET = "8p-kkC23XOWvkCosKq-BOn3G74qp__rBcDMxc82iB4gfzRvuhSCRBKM7C5Q7TAzj"

ORG_BINDING: dict[str, Any] = {"resource_type": "organization", "role": "OrgMember"}


def _account(
    *,
    id: str = ACCOUNT_ID,
    name: str = "ci-prod",
    client_id: str = CLIENT_ID,
    created_at: str = "2026-04-10T15:23:00Z",
    updated_at: str = "2026-04-12T09:11:00Z",
) -> dict[str, Any]:
    return {
        "id": id,
        "name": name,
        "client_id": client_id,
        "created_at": created_at,
        "updated_at": updated_at,
    }


def _with_secret(*, secret: str = SECRET, **kwargs: Any) -> dict[str, Any]:
    return {"service_account": _account(**kwargs), "client_secret": secret}


def _page(accounts: list[dict[str, Any]], *, next_token: str | None = None) -> dict[str, Any]:
    return {"data": accounts, "pagination": {"next": next_token} if next_token else None}


@pytest.fixture
def http_client() -> HTTPClient:
    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    return HTTPClient(config, ADMIN_API_VERSION)


@pytest.fixture
def service_accounts(http_client: HTTPClient) -> ServiceAccounts:
    return ServiceAccounts(http=http_client)


def test_repr(service_accounts: ServiceAccounts) -> None:
    assert repr(service_accounts) == "ServiceAccounts()"


# ---------------------------------------------------------------------------
# list()
# ---------------------------------------------------------------------------


@respx.mock
def test_list_service_accounts(service_accounts: ServiceAccounts) -> None:
    route = respx.get(f"{BASE_URL}/admin/service-accounts").mock(
        return_value=httpx.Response(200, json=_page([_account()]))
    )

    result = service_accounts.list()

    assert isinstance(result, Paginator)
    items = result.to_list()
    assert len(items) == 1
    assert isinstance(items[0], ServiceAccountModel)
    assert items[0].id == ACCOUNT_ID
    assert items[0].name == "ci-prod"
    assert items[0].client_id == CLIENT_ID
    assert route.calls.last.request.url.path == "/admin/service-accounts"


@respx.mock
def test_list_sends_api_version_header(service_accounts: ServiceAccounts) -> None:
    route = respx.get(f"{BASE_URL}/admin/service-accounts").mock(
        return_value=httpx.Response(200, json=_page([_account()]))
    )

    service_accounts.list().to_list()

    assert route.calls.last.request.headers["X-Pinecone-Api-Version"] == ADMIN_API_VERSION


@respx.mock
def test_list_is_lazy(service_accounts: ServiceAccounts) -> None:
    route = respx.get(f"{BASE_URL}/admin/service-accounts").mock(
        return_value=httpx.Response(200, json=_page([_account()]))
    )

    paginator = service_accounts.list()

    assert route.call_count == 0
    paginator.to_list()
    assert route.call_count == 1


@respx.mock
def test_list_omits_unset_query_params(service_accounts: ServiceAccounts) -> None:
    route = respx.get(f"{BASE_URL}/admin/service-accounts").mock(
        return_value=httpx.Response(200, json=_page([]))
    )

    service_accounts.list().to_list()

    params = route.calls.last.request.url.params
    assert "limit" not in params
    assert "paginationToken" not in params


@respx.mock
def test_list_sends_limit(service_accounts: ServiceAccounts) -> None:
    route = respx.get(f"{BASE_URL}/admin/service-accounts").mock(
        return_value=httpx.Response(200, json=_page([]))
    )

    service_accounts.list(limit=25).to_list()

    assert route.calls.last.request.url.params["limit"] == "25"


@respx.mock
def test_list_sends_initial_pagination_token(service_accounts: ServiceAccounts) -> None:
    route = respx.get(f"{BASE_URL}/admin/service-accounts").mock(
        return_value=httpx.Response(200, json=_page([]))
    )

    service_accounts.list(pagination_token="resume-here").to_list()

    assert route.calls.last.request.url.params["paginationToken"] == "resume-here"


@respx.mock
def test_list_empty_page(service_accounts: ServiceAccounts) -> None:
    respx.get(f"{BASE_URL}/admin/service-accounts").mock(
        return_value=httpx.Response(200, json=_page([]))
    )

    assert service_accounts.list().to_list() == []


@respx.mock
def test_list_tolerates_absent_pagination_key(service_accounts: ServiceAccounts) -> None:
    respx.get(f"{BASE_URL}/admin/service-accounts").mock(
        return_value=httpx.Response(200, json={"data": [_account()]})
    )

    items = service_accounts.list().to_list()

    assert len(items) == 1


@respx.mock
def test_list_follows_pagination_cursor_verbatim(service_accounts: ServiceAccounts) -> None:
    cursor = "eyJsYXN0X2lkIjoiZDI0MTc3YTAifQ=="
    route = respx.get(f"{BASE_URL}/admin/service-accounts").mock(
        side_effect=[
            httpx.Response(200, json=_page([_account(id="one")], next_token=cursor)),
            httpx.Response(200, json=_page([_account(id="two")])),
        ]
    )

    items = service_accounts.list().to_list()

    assert [a.id for a in items] == ["one", "two"]
    assert "paginationToken" not in route.calls[0].request.url.params
    assert route.calls[1].request.url.params["paginationToken"] == cursor


@respx.mock
def test_list_carries_limit_onto_later_pages(service_accounts: ServiceAccounts) -> None:
    route = respx.get(f"{BASE_URL}/admin/service-accounts").mock(
        side_effect=[
            httpx.Response(200, json=_page([_account()], next_token="next-1")),
            httpx.Response(200, json=_page([])),
        ]
    )

    service_accounts.list(limit=7).to_list()

    assert route.calls[1].request.url.params["limit"] == "7"


@respx.mock
def test_list_stops_on_pagination_with_null_next(service_accounts: ServiceAccounts) -> None:
    route = respx.get(f"{BASE_URL}/admin/service-accounts").mock(
        return_value=httpx.Response(200, json={"data": [_account()], "pagination": {"next": None}})
    )

    items = service_accounts.list().to_list()

    assert len(items) == 1
    assert route.call_count == 1


@respx.mock
def test_list_pages_exposes_page_level_access(service_accounts: ServiceAccounts) -> None:
    respx.get(f"{BASE_URL}/admin/service-accounts").mock(
        side_effect=[
            httpx.Response(200, json=_page([_account(id="one")], next_token="cursor-1")),
            httpx.Response(200, json=_page([_account(id="two")])),
        ]
    )

    pages = list(service_accounts.list().pages())

    assert [p.pagination_token for p in pages] == ["cursor-1", None]
    assert [len(p.items) for p in pages] == [1, 1]


@respx.mock
@pytest.mark.parametrize("bad_limit", [0, -1, 101, 1000])
def test_list_rejects_out_of_range_limit_before_network(
    service_accounts: ServiceAccounts, bad_limit: int
) -> None:
    route = respx.get(f"{BASE_URL}/admin/service-accounts").mock(
        return_value=httpx.Response(200, json=_page([]))
    )

    with pytest.raises(ValidationError):
        service_accounts.list(limit=bad_limit)

    assert route.call_count == 0


@respx.mock
@pytest.mark.parametrize("good_limit", [1, 100])
def test_list_accepts_boundary_limits(service_accounts: ServiceAccounts, good_limit: int) -> None:
    route = respx.get(f"{BASE_URL}/admin/service-accounts").mock(
        return_value=httpx.Response(200, json=_page([]))
    )

    service_accounts.list(limit=good_limit).to_list()

    assert route.calls.last.request.url.params["limit"] == str(good_limit)


@respx.mock
def test_list_surfaces_api_error(service_accounts: ServiceAccounts) -> None:
    respx.get(f"{BASE_URL}/admin/service-accounts").mock(
        return_value=httpx.Response(
            400,
            json={
                "error": {"code": "OUT_OF_RANGE", "message": "limit out of range"},
                "status": 400,
            },
        )
    )

    with pytest.raises(ApiError) as exc:
        service_accounts.list().to_list()

    assert "OUT_OF_RANGE" in str(exc.value)


# ---------------------------------------------------------------------------
# create()
# ---------------------------------------------------------------------------


@respx.mock
def test_create_service_account(service_accounts: ServiceAccounts) -> None:
    route = respx.post(f"{BASE_URL}/admin/service-accounts").mock(
        return_value=httpx.Response(201, json=_with_secret())
    )

    result = service_accounts.create(name="ci-prod")

    assert isinstance(result, ServiceAccountWithSecret)
    assert result.service_account.id == ACCOUNT_ID
    assert result.service_account.client_id == CLIENT_ID
    assert result.client_secret == SECRET
    assert orjson.loads(route.calls.last.request.content) == {"name": "ci-prod"}


@respx.mock
def test_create_accepts_201_created(service_accounts: ServiceAccounts) -> None:
    respx.post(f"{BASE_URL}/admin/service-accounts").mock(
        return_value=httpx.Response(201, json=_with_secret())
    )

    assert service_accounts.create(name="ci-prod").client_secret == SECRET


@respx.mock
def test_create_omits_role_bindings_when_not_given(service_accounts: ServiceAccounts) -> None:
    route = respx.post(f"{BASE_URL}/admin/service-accounts").mock(
        return_value=httpx.Response(201, json=_with_secret())
    )

    service_accounts.create(name="ci-prod")

    assert "role_bindings" not in orjson.loads(route.calls.last.request.content)


@respx.mock
def test_create_sends_explicit_empty_role_bindings(service_accounts: ServiceAccounts) -> None:
    route = respx.post(f"{BASE_URL}/admin/service-accounts").mock(
        return_value=httpx.Response(201, json=_with_secret())
    )

    service_accounts.create(name="ci-prod", role_bindings=[])

    assert orjson.loads(route.calls.last.request.content)["role_bindings"] == []


@respx.mock
def test_create_sends_project_scoped_binding(service_accounts: ServiceAccounts) -> None:
    route = respx.post(f"{BASE_URL}/admin/service-accounts").mock(
        return_value=httpx.Response(201, json=_with_secret())
    )

    service_accounts.create(
        name="ci-prod",
        role_bindings=[
            {
                "resource_type": "project",
                "role": "DataPlaneEditor",
                "resource_id": PROJECT_ID,
            }
        ],
    )

    body = orjson.loads(route.calls.last.request.content)
    assert body["role_bindings"] == [
        {"resource_type": "project", "role": "DataPlaneEditor", "resource_id": PROJECT_ID}
    ]


@respx.mock
def test_create_omits_resource_id_for_organization_scope(
    service_accounts: ServiceAccounts,
) -> None:
    route = respx.post(f"{BASE_URL}/admin/service-accounts").mock(
        return_value=httpx.Response(201, json=_with_secret())
    )

    service_accounts.create(name="ci-prod", role_bindings=[ORG_BINDING])

    binding = orjson.loads(route.calls.last.request.content)["role_bindings"][0]
    assert "resource_id" not in binding


@respx.mock
def test_create_accepts_mixed_models_and_dicts(service_accounts: ServiceAccounts) -> None:
    route = respx.post(f"{BASE_URL}/admin/service-accounts").mock(
        return_value=httpx.Response(201, json=_with_secret())
    )

    service_accounts.create(
        name="ci-prod",
        role_bindings=[
            RoleBindingInput(resource_type=ResourceType.ORGANIZATION, role=RoleName.ORG_MEMBER),
            {"resource_type": "project", "role": "ProjectViewer", "resource_id": PROJECT_ID},
        ],
    )

    bindings = orjson.loads(route.calls.last.request.content)["role_bindings"]
    assert [b["resource_type"] for b in bindings] == ["organization", "project"]


@respx.mock
@pytest.mark.parametrize("bad_name", ["", "   "])
def test_create_rejects_empty_name_before_network(
    service_accounts: ServiceAccounts, bad_name: str
) -> None:
    route = respx.post(f"{BASE_URL}/admin/service-accounts").mock(
        return_value=httpx.Response(201, json=_with_secret())
    )

    with pytest.raises(ValidationError) as exc:
        service_accounts.create(name=bad_name)

    assert "name" in str(exc.value)
    assert route.call_count == 0


@respx.mock
def test_create_forwards_unvalidated_long_name_to_server(
    service_accounts: ServiceAccounts,
) -> None:
    route = respx.post(f"{BASE_URL}/admin/service-accounts").mock(
        return_value=httpx.Response(
            400,
            json={
                "error": {"code": "INVALID_ARGUMENT", "message": "Name is too long"},
                "status": 400,
            },
        )
    )

    with pytest.raises(ApiError):
        service_accounts.create(name="x" * 500)

    assert orjson.loads(route.calls.last.request.content)["name"] == "x" * 500


@respx.mock
def test_create_names_index_of_entry_missing_a_required_key(
    service_accounts: ServiceAccounts,
) -> None:
    route = respx.post(f"{BASE_URL}/admin/service-accounts").mock(
        return_value=httpx.Response(201, json=_with_secret())
    )

    with pytest.raises(ValidationError) as exc:
        service_accounts.create(
            name="ci-prod", role_bindings=[ORG_BINDING, {"resource_type": "project"}]
        )

    assert "role_bindings[1]" in str(exc.value)
    assert route.call_count == 0


@respx.mock
def test_create_names_index_of_entry_with_unrecognized_key(
    service_accounts: ServiceAccounts,
) -> None:
    route = respx.post(f"{BASE_URL}/admin/service-accounts").mock(
        return_value=httpx.Response(201, json=_with_secret())
    )

    with pytest.raises(ValidationError) as exc:
        service_accounts.create(
            name="ci-prod",
            role_bindings=[{**ORG_BINDING, "principal_type": "service_account"}],
        )

    assert "role_bindings[0]" in str(exc.value)
    assert "principal_type" in str(exc.value)
    assert route.call_count == 0


@respx.mock
def test_create_names_index_of_entry_with_unknown_role(
    service_accounts: ServiceAccounts,
) -> None:
    route = respx.post(f"{BASE_URL}/admin/service-accounts").mock(
        return_value=httpx.Response(201, json=_with_secret())
    )

    with pytest.raises(ValidationError) as exc:
        service_accounts.create(
            name="ci-prod",
            role_bindings=[{"resource_type": "organization", "role": "Sysadmin"}],
        )

    assert "role_bindings[0]" in str(exc.value)
    assert route.call_count == 0


@respx.mock
def test_create_403_when_plan_excludes_service_accounts(
    service_accounts: ServiceAccounts,
) -> None:
    respx.post(f"{BASE_URL}/admin/service-accounts").mock(
        return_value=httpx.Response(
            403,
            json={
                "error": {
                    "code": "FORBIDDEN",
                    "message": "Service accounts require an Enterprise plan",
                },
                "status": 403,
            },
        )
    )

    with pytest.raises(ForbiddenError) as exc:
        service_accounts.create(name="ci-prod")

    assert "Enterprise" in str(exc.value)


# ---------------------------------------------------------------------------
# describe()
# ---------------------------------------------------------------------------


@respx.mock
def test_describe_service_account(service_accounts: ServiceAccounts) -> None:
    route = respx.get(f"{BASE_URL}/admin/service-accounts/{ACCOUNT_ID}").mock(
        return_value=httpx.Response(200, json=_account())
    )

    result = service_accounts.describe(service_account_id=ACCOUNT_ID)

    assert isinstance(result, ServiceAccountModel)
    assert result.client_id == CLIENT_ID
    assert route.calls.last.request.url.path == f"/admin/service-accounts/{ACCOUNT_ID}"


@respx.mock
@pytest.mark.parametrize("bad_id", ["", "   "])
def test_describe_rejects_empty_id(service_accounts: ServiceAccounts, bad_id: str) -> None:
    with pytest.raises(ValidationError):
        service_accounts.describe(service_account_id=bad_id)


@respx.mock
def test_describe_404_surfaces_code_and_message(service_accounts: ServiceAccounts) -> None:
    respx.get(f"{BASE_URL}/admin/service-accounts/{ACCOUNT_ID}").mock(
        return_value=httpx.Response(
            404,
            json={
                "error": {
                    "code": "NOT_FOUND",
                    "message": f"Service Account {ACCOUNT_ID} not found",
                },
                "status": 404,
            },
        )
    )

    with pytest.raises(NotFoundError) as exc:
        service_accounts.describe(service_account_id=ACCOUNT_ID)

    assert "not found" in str(exc.value)


# ---------------------------------------------------------------------------
# update()
# ---------------------------------------------------------------------------


@respx.mock
def test_update_service_account(service_accounts: ServiceAccounts) -> None:
    route = respx.patch(f"{BASE_URL}/admin/service-accounts/{ACCOUNT_ID}").mock(
        return_value=httpx.Response(200, json=_account(name="renamed"))
    )

    result = service_accounts.update(service_account_id=ACCOUNT_ID, name="renamed")

    assert result.name == "renamed"
    assert orjson.loads(route.calls.last.request.content) == {"name": "renamed"}


@respx.mock
def test_update_without_fields_raises_before_network(service_accounts: ServiceAccounts) -> None:
    route = respx.patch(f"{BASE_URL}/admin/service-accounts/{ACCOUNT_ID}").mock(
        return_value=httpx.Response(200, json=_account())
    )

    with pytest.raises(ValidationError) as exc:
        service_accounts.update(service_account_id=ACCOUNT_ID)

    assert "provide at least one of: name" in str(exc.value)
    assert route.call_count == 0


@respx.mock
def test_update_returns_no_secret(service_accounts: ServiceAccounts) -> None:
    respx.patch(f"{BASE_URL}/admin/service-accounts/{ACCOUNT_ID}").mock(
        return_value=httpx.Response(200, json=_account(name="renamed"))
    )

    result = service_accounts.update(service_account_id=ACCOUNT_ID, name="renamed")

    assert not hasattr(result, "client_secret")


@respx.mock
@pytest.mark.parametrize("bad_id", ["", "   "])
def test_update_rejects_empty_id(service_accounts: ServiceAccounts, bad_id: str) -> None:
    with pytest.raises(ValidationError):
        service_accounts.update(service_account_id=bad_id, name="renamed")


@respx.mock
def test_update_400_surfaces_server_name_validation(service_accounts: ServiceAccounts) -> None:
    respx.patch(f"{BASE_URL}/admin/service-accounts/{ACCOUNT_ID}").mock(
        return_value=httpx.Response(
            400,
            json={
                "error": {
                    "code": "INVALID_ARGUMENT",
                    "message": "Name is too long. Maximum is 80 characters",
                },
                "status": 400,
            },
        )
    )

    with pytest.raises(ApiError) as exc:
        service_accounts.update(service_account_id=ACCOUNT_ID, name="x" * 200)

    assert "too long" in str(exc.value)


# ---------------------------------------------------------------------------
# delete()
# ---------------------------------------------------------------------------


@respx.mock
def test_delete_service_account(service_accounts: ServiceAccounts) -> None:
    route = respx.delete(f"{BASE_URL}/admin/service-accounts/{ACCOUNT_ID}").mock(
        return_value=httpx.Response(202)
    )

    assert service_accounts.delete(service_account_id=ACCOUNT_ID) is None
    assert route.calls.last.request.url.path == f"/admin/service-accounts/{ACCOUNT_ID}"


@respx.mock
@pytest.mark.parametrize("bad_id", ["", "   "])
def test_delete_rejects_empty_id(service_accounts: ServiceAccounts, bad_id: str) -> None:
    with pytest.raises(ValidationError):
        service_accounts.delete(service_account_id=bad_id)


@respx.mock
def test_delete_404_surfaces_code_and_message(service_accounts: ServiceAccounts) -> None:
    respx.delete(f"{BASE_URL}/admin/service-accounts/{ACCOUNT_ID}").mock(
        return_value=httpx.Response(
            404,
            json={"error": {"code": "NOT_FOUND", "message": "not found"}, "status": 404},
        )
    )

    with pytest.raises(NotFoundError):
        service_accounts.delete(service_account_id=ACCOUNT_ID)


# ---------------------------------------------------------------------------
# rotate_secret()
# ---------------------------------------------------------------------------


@respx.mock
def test_rotate_secret(service_accounts: ServiceAccounts) -> None:
    route = respx.post(f"{BASE_URL}/admin/service-accounts/{ACCOUNT_ID}/rotate-secret").mock(
        return_value=httpx.Response(200, json=_with_secret(secret="brand-new-secret"))
    )

    result = service_accounts.rotate_secret(service_account_id=ACCOUNT_ID)

    assert isinstance(result, ServiceAccountWithSecret)
    assert result.client_secret == "brand-new-secret"
    assert (
        route.calls.last.request.url.path == f"/admin/service-accounts/{ACCOUNT_ID}/rotate-secret"
    )


@respx.mock
def test_rotate_secret_sends_no_request_body(service_accounts: ServiceAccounts) -> None:
    route = respx.post(f"{BASE_URL}/admin/service-accounts/{ACCOUNT_ID}/rotate-secret").mock(
        return_value=httpx.Response(200, json=_with_secret())
    )

    service_accounts.rotate_secret(service_account_id=ACCOUNT_ID)

    assert not route.calls.last.request.content


@respx.mock
def test_rotate_secret_preserves_client_id(service_accounts: ServiceAccounts) -> None:
    respx.post(f"{BASE_URL}/admin/service-accounts/{ACCOUNT_ID}/rotate-secret").mock(
        return_value=httpx.Response(200, json=_with_secret(secret="new-one"))
    )

    result = service_accounts.rotate_secret(service_account_id=ACCOUNT_ID)

    assert result.service_account.client_id == CLIENT_ID
    assert result.service_account.id == ACCOUNT_ID


@respx.mock
@pytest.mark.parametrize("bad_id", ["", "   "])
def test_rotate_secret_rejects_empty_id(service_accounts: ServiceAccounts, bad_id: str) -> None:
    with pytest.raises(ValidationError):
        service_accounts.rotate_secret(service_account_id=bad_id)


@respx.mock
def test_rotate_secret_404_surfaces_code_and_message(
    service_accounts: ServiceAccounts,
) -> None:
    respx.post(f"{BASE_URL}/admin/service-accounts/{ACCOUNT_ID}/rotate-secret").mock(
        return_value=httpx.Response(
            404,
            json={"error": {"code": "NOT_FOUND", "message": "not found"}, "status": 404},
        )
    )

    with pytest.raises(NotFoundError):
        service_accounts.rotate_secret(service_account_id=ACCOUNT_ID)


# ---------------------------------------------------------------------------
# Secret hygiene
# ---------------------------------------------------------------------------


@respx.mock
def test_created_secret_never_appears_in_logs(
    service_accounts: ServiceAccounts, caplog: pytest.LogCaptureFixture
) -> None:
    respx.post(f"{BASE_URL}/admin/service-accounts").mock(
        return_value=httpx.Response(201, json=_with_secret())
    )

    with caplog.at_level(logging.DEBUG):
        result = service_accounts.create(name="ci-prod")

    assert result.client_secret == SECRET
    assert caplog.records
    for record in caplog.records:
        assert SECRET not in record.getMessage()
        assert SECRET not in str(record.args)


@respx.mock
def test_rotated_secret_never_appears_in_logs(
    service_accounts: ServiceAccounts, caplog: pytest.LogCaptureFixture
) -> None:
    respx.post(f"{BASE_URL}/admin/service-accounts/{ACCOUNT_ID}/rotate-secret").mock(
        return_value=httpx.Response(200, json=_with_secret(secret="rotated-abcdef"))
    )

    with caplog.at_level(logging.DEBUG):
        result = service_accounts.rotate_secret(service_account_id=ACCOUNT_ID)

    assert result.client_secret == "rotated-abcdef"
    assert caplog.records
    for record in caplog.records:
        assert "rotated-abcdef" not in record.getMessage()


@respx.mock
def test_secret_is_masked_in_repr_and_str(service_accounts: ServiceAccounts) -> None:
    respx.post(f"{BASE_URL}/admin/service-accounts").mock(
        return_value=httpx.Response(201, json=_with_secret())
    )

    result = service_accounts.create(name="ci-prod")

    assert SECRET not in repr(result)
    assert SECRET not in str(result)
    assert SECRET not in f"{result}"
    assert result.client_secret == SECRET


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


@settings(max_examples=200, deadline=None)
@given(name=st.text(min_size=1, max_size=80).filter(lambda s: bool(s.strip())))
def test_name_survives_json_body_encoding_unmodified(name: str) -> None:
    """Whatever string the caller passes is the string the server receives.

    The SDK deliberately does not validate the name — the server owns length
    (in UTF-8 bytes) and content — so it must not mangle one either. Quotes,
    backslashes, newlines, unicode, and surrogate-adjacent codepoints all have
    to survive JSON encoding byte-for-byte, or a rejected name would come back
    with a message about a string the caller never sent.
    """
    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    namespace = ServiceAccounts(http=HTTPClient(config, ADMIN_API_VERSION))

    with respx.mock:
        route = respx.post(f"{BASE_URL}/admin/service-accounts").mock(
            return_value=httpx.Response(201, json=_with_secret())
        )
        namespace.create(name=name)

        body = orjson.loads(route.calls.last.request.content)

    assert body["name"] == name


@settings(max_examples=200, deadline=None)
@given(
    old_secret=st.text(min_size=1, max_size=64).filter(lambda s: bool(s.strip())),
    new_secret=st.text(min_size=1, max_size=64).filter(lambda s: bool(s.strip())),
)
def test_rotate_secret_parsing_never_conflates_old_and_new(
    old_secret: str, new_secret: str
) -> None:
    """Parsing a rotation response yields the payload's secret and nothing else.

    A rotation is only useful if the caller can tell the new secret from the
    one it replaces. The adapter holds no state between calls, so parsing a
    response that carries ``new_secret`` must never surface ``old_secret``, in
    either order and whatever the two strings look like.
    """
    from pinecone._internal.adapters.admin_adapter import AdminAdapter

    adapter = AdminAdapter()

    before = adapter.to_service_account_with_secret(orjson.dumps(_with_secret(secret=old_secret)))
    after = adapter.to_service_account_with_secret(orjson.dumps(_with_secret(secret=new_secret)))

    assert before.client_secret == old_secret
    assert after.client_secret == new_secret
    if old_secret != new_secret:
        assert after.client_secret != old_secret
        assert before.client_secret != new_secret
    assert after.service_account.client_id == before.service_account.client_id
