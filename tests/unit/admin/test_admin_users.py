"""Unit tests for the Admin Users namespace.

The pagination assertions are the load-bearing ones: ``list()`` returns a lazy
:class:`Paginator`, so these tests pin that nothing is requested until
iteration, that the cursor is echoed back to the server byte-for-byte, and that
a page whose ``pagination`` is absent or ``null`` ends the walk rather than
looping forever.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
import respx
from hypothesis import given, settings
from hypothesis import strategies as st

from pinecone._internal.config import PineconeConfig
from pinecone._internal.constants import ADMIN_API_VERSION
from pinecone._internal.http_client import HTTPClient
from pinecone.admin.users import Users
from pinecone.errors.exceptions import (
    ApiError,
    ConflictError,
    NotFoundError,
    ValidationError,
)
from pinecone.models.admin.user import UserModel
from pinecone.models.pagination import Paginator

BASE_URL = "https://api.test.pinecone.io"

USER_ID = "e2e92523-85dc-4142-b8c2-e681be8b78df"


def _user(
    *,
    id: str = USER_ID,
    email: str = "alice@example.com",
    name: str | None = "Alice Example",
) -> dict[str, Any]:
    payload: dict[str, Any] = {"id": id, "email": email}
    if name is not None:
        payload["name"] = name
    return payload


def _page(users: list[dict[str, Any]], *, next_token: str | None = None) -> dict[str, Any]:
    return {"data": users, "pagination": {"next": next_token} if next_token else None}


@pytest.fixture
def http_client() -> HTTPClient:
    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    return HTTPClient(config, ADMIN_API_VERSION)


@pytest.fixture
def users(http_client: HTTPClient) -> Users:
    return Users(http=http_client)


def test_repr(users: Users) -> None:
    assert repr(users) == "Users()"


# ---------------------------------------------------------------------------
# list()
# ---------------------------------------------------------------------------


@respx.mock
def test_list_users(users: Users) -> None:
    route = respx.get(f"{BASE_URL}/admin/users").mock(
        return_value=httpx.Response(200, json=_page([_user()]))
    )

    result = users.list()

    assert isinstance(result, Paginator)
    items = result.to_list()
    assert len(items) == 1
    assert isinstance(items[0], UserModel)
    assert items[0].id == USER_ID
    assert items[0].email == "alice@example.com"
    assert items[0].name == "Alice Example"
    assert route.calls.last.request.url.path == "/admin/users"


@respx.mock
def test_list_sends_api_version_header(users: Users) -> None:
    route = respx.get(f"{BASE_URL}/admin/users").mock(
        return_value=httpx.Response(200, json=_page([_user()]))
    )

    users.list().to_list()

    assert route.calls.last.request.headers["X-Pinecone-Api-Version"] == ADMIN_API_VERSION


@respx.mock
def test_list_is_lazy(users: Users) -> None:
    route = respx.get(f"{BASE_URL}/admin/users").mock(
        return_value=httpx.Response(200, json=_page([_user()]))
    )

    paginator = users.list()

    assert route.call_count == 0
    paginator.to_list()
    assert route.call_count == 1


@respx.mock
def test_list_omits_unset_query_params(users: Users) -> None:
    route = respx.get(f"{BASE_URL}/admin/users").mock(
        return_value=httpx.Response(200, json=_page([]))
    )

    users.list().to_list()

    assert dict(route.calls.last.request.url.params) == {}


@respx.mock
def test_list_sends_email_filter(users: Users) -> None:
    route = respx.get(f"{BASE_URL}/admin/users").mock(
        return_value=httpx.Response(200, json=_page([_user()]))
    )

    users.list(email="alice@example.com").to_list()

    assert route.calls.last.request.url.params["email"] == "alice@example.com"


@respx.mock
def test_list_sends_limit(users: Users) -> None:
    route = respx.get(f"{BASE_URL}/admin/users").mock(
        return_value=httpx.Response(200, json=_page([_user()]))
    )

    users.list(limit=25).to_list()

    assert route.calls.last.request.url.params["limit"] == "25"


@respx.mock
def test_list_sends_initial_pagination_token(users: Users) -> None:
    route = respx.get(f"{BASE_URL}/admin/users").mock(
        return_value=httpx.Response(200, json=_page([_user()]))
    )

    users.list(pagination_token="cursor-abc").to_list()

    assert route.calls.last.request.url.params["paginationToken"] == "cursor-abc"


@respx.mock
def test_list_empty_page(users: Users) -> None:
    respx.get(f"{BASE_URL}/admin/users").mock(return_value=httpx.Response(200, json=_page([])))

    assert users.list().to_list() == []


@respx.mock
def test_list_tolerates_absent_pagination_key(users: Users) -> None:
    route = respx.get(f"{BASE_URL}/admin/users").mock(
        return_value=httpx.Response(200, json={"data": [_user()]})
    )

    items = users.list().to_list()

    assert len(items) == 1
    assert route.call_count == 1


@respx.mock
def test_list_tolerates_absent_name(users: Users) -> None:
    respx.get(f"{BASE_URL}/admin/users").mock(
        return_value=httpx.Response(200, json=_page([_user(name=None)]))
    )

    assert users.list().to_list()[0].name is None


@respx.mock
def test_list_follows_pagination_cursor_verbatim(users: Users) -> None:
    cursor = "eyJsYXN0X2lkIjoiZTJlOTI1MjMifQ=="
    route = respx.get(f"{BASE_URL}/admin/users").mock(
        side_effect=[
            httpx.Response(200, json=_page([_user(id="u1", email="a@x.com")], next_token=cursor)),
            httpx.Response(200, json=_page([_user(id="u2", email="b@x.com")])),
        ]
    )

    items = users.list().to_list()

    assert [u.id for u in items] == ["u1", "u2"]
    assert route.call_count == 2
    assert "paginationToken" not in route.calls[0].request.url.params
    assert route.calls[1].request.url.params["paginationToken"] == cursor


@respx.mock
def test_list_carries_filters_onto_later_pages(users: Users) -> None:
    route = respx.get(f"{BASE_URL}/admin/users").mock(
        side_effect=[
            httpx.Response(200, json=_page([_user(id="u1")], next_token="c1")),
            httpx.Response(200, json=_page([_user(id="u2")])),
        ]
    )

    users.list(email="alice@example.com", limit=1).to_list()

    second = route.calls[1].request.url.params
    assert second["email"] == "alice@example.com"
    assert second["limit"] == "1"
    assert second["paginationToken"] == "c1"


@respx.mock
def test_list_stops_on_null_pagination(users: Users) -> None:
    route = respx.get(f"{BASE_URL}/admin/users").mock(
        side_effect=[
            httpx.Response(200, json=_page([_user(id="u1")], next_token="c1")),
            httpx.Response(200, json={"data": [_user(id="u2")], "pagination": None}),
            httpx.Response(200, json=_page([_user(id="u3")])),
        ]
    )

    items = users.list().to_list()

    assert [u.id for u in items] == ["u1", "u2"]
    assert route.call_count == 2


@respx.mock
def test_list_stops_on_pagination_with_null_next(users: Users) -> None:
    route = respx.get(f"{BASE_URL}/admin/users").mock(
        side_effect=[
            httpx.Response(200, json={"data": [_user(id="u1")], "pagination": {"next": None}}),
            httpx.Response(200, json=_page([_user(id="u2")])),
        ]
    )

    assert [u.id for u in users.list().to_list()] == ["u1"]
    assert route.call_count == 1


@respx.mock
def test_list_pages_exposes_page_level_access(users: Users) -> None:
    respx.get(f"{BASE_URL}/admin/users").mock(
        side_effect=[
            httpx.Response(200, json=_page([_user(id="u1")], next_token="c1")),
            httpx.Response(200, json=_page([_user(id="u2")])),
        ]
    )

    pages = list(users.list().pages())

    assert [p.pagination_token for p in pages] == ["c1", None]
    assert [[u.id for u in p.items] for p in pages] == [["u1"], ["u2"]]
    assert pages[0].has_more is True
    assert pages[1].has_more is False


@respx.mock
def test_list_paginator_token_supports_resumption(users: Users) -> None:
    respx.get(f"{BASE_URL}/admin/users").mock(
        side_effect=[
            httpx.Response(200, json=_page([_user(id="u1")], next_token="c1")),
            httpx.Response(200, json=_page([_user(id="u2")])),
        ]
    )

    paginator = users.list()
    pages = paginator.pages()
    next(pages)

    assert paginator.pagination_token == "c1"


@pytest.mark.parametrize("bad_limit", [0, -1, 101, 1000])
def test_list_rejects_out_of_range_limit_before_network(users: Users, bad_limit: int) -> None:
    with respx.mock:
        route = respx.get(f"{BASE_URL}/admin/users").mock(
            return_value=httpx.Response(200, json=_page([]))
        )
        with pytest.raises(ValidationError) as exc:
            users.list(limit=bad_limit)
        assert route.call_count == 0

    message = str(exc.value)
    assert "limit" in message
    assert str(bad_limit) in message
    assert "1" in message and "100" in message


@respx.mock
@pytest.mark.parametrize("good_limit", [1, 50, 100])
def test_list_accepts_boundary_limits(users: Users, good_limit: int) -> None:
    route = respx.get(f"{BASE_URL}/admin/users").mock(
        return_value=httpx.Response(200, json=_page([]))
    )

    users.list(limit=good_limit).to_list()

    assert route.calls.last.request.url.params["limit"] == str(good_limit)


@respx.mock
def test_list_surfaces_api_error(users: Users) -> None:
    respx.get(f"{BASE_URL}/admin/users").mock(
        return_value=httpx.Response(
            400,
            json={
                "error": {"code": "INVALID_ARGUMENT", "message": "email is invalid"},
                "status": 400,
            },
        )
    )

    with pytest.raises(ApiError) as exc:
        users.list(email="not-an-email").to_list()

    assert exc.value.error_code == "INVALID_ARGUMENT"
    assert "email is invalid" in str(exc.value)


# ---------------------------------------------------------------------------
# describe()
# ---------------------------------------------------------------------------


@respx.mock
def test_describe_user(users: Users) -> None:
    route = respx.get(f"{BASE_URL}/admin/users/{USER_ID}").mock(
        return_value=httpx.Response(200, json=_user())
    )

    result = users.describe(user_id=USER_ID)

    assert isinstance(result, UserModel)
    assert result.id == USER_ID
    assert result.email == "alice@example.com"
    assert result.name == "Alice Example"
    assert route.calls.last.request.headers["X-Pinecone-Api-Version"] == ADMIN_API_VERSION


@respx.mock
def test_describe_user_without_name(users: Users) -> None:
    respx.get(f"{BASE_URL}/admin/users/{USER_ID}").mock(
        return_value=httpx.Response(200, json=_user(name=None))
    )

    assert users.describe(user_id=USER_ID).name is None


@pytest.mark.parametrize("bad_id", ["", "   "])
def test_describe_rejects_empty_user_id(users: Users, bad_id: str) -> None:
    with pytest.raises(ValidationError, match="user_id"):
        users.describe(user_id=bad_id)


@respx.mock
def test_describe_404_surfaces_code_and_message(users: Users) -> None:
    respx.get(f"{BASE_URL}/admin/users/{USER_ID}").mock(
        return_value=httpx.Response(
            404,
            json={"error": {"code": "NOT_FOUND", "message": "user not found"}, "status": 404},
        )
    )

    with pytest.raises(NotFoundError) as exc:
        users.describe(user_id=USER_ID)

    assert exc.value.error_code == "NOT_FOUND"
    assert exc.value.status_code == 404
    assert "user not found" in str(exc.value)
    assert "NOT_FOUND" in str(exc.value)


# ---------------------------------------------------------------------------
# delete()
# ---------------------------------------------------------------------------


@respx.mock
def test_delete_user(users: Users) -> None:
    route = respx.delete(f"{BASE_URL}/admin/users/{USER_ID}").mock(return_value=httpx.Response(202))

    assert users.delete(user_id=USER_ID) is None
    assert route.call_count == 1
    assert route.calls.last.request.headers["X-Pinecone-Api-Version"] == ADMIN_API_VERSION


@pytest.mark.parametrize("bad_id", ["", "   "])
def test_delete_rejects_empty_user_id(users: Users, bad_id: str) -> None:
    with pytest.raises(ValidationError, match="user_id"):
        users.delete(user_id=bad_id)


@respx.mock
def test_delete_404_surfaces_code_and_message(users: Users) -> None:
    respx.delete(f"{BASE_URL}/admin/users/{USER_ID}").mock(
        return_value=httpx.Response(
            404,
            json={"error": {"code": "NOT_FOUND", "message": "user not found"}, "status": 404},
        )
    )

    with pytest.raises(NotFoundError) as exc:
        users.delete(user_id=USER_ID)

    assert exc.value.error_code == "NOT_FOUND"
    assert "user not found" in str(exc.value)


@respx.mock
def test_delete_409_surfaces_code_and_message(users: Users) -> None:
    message = "Cannot delete the last OrgOwner role binding for this organization."
    respx.delete(f"{BASE_URL}/admin/users/{USER_ID}").mock(
        return_value=httpx.Response(
            409, json={"error": {"code": "ABORTED", "message": message}, "status": 409}
        )
    )

    with pytest.raises(ConflictError) as exc:
        users.delete(user_id=USER_ID)

    assert exc.value.error_code == "ABORTED"
    assert exc.value.status_code == 409
    assert message in str(exc.value)
    assert "ABORTED" in str(exc.value)


# ---------------------------------------------------------------------------
# Property: the email filter reaches the server unmangled
# ---------------------------------------------------------------------------


@settings(max_examples=200, deadline=None)
@given(email=st.text(min_size=0, max_size=254))
def test_email_filter_survives_url_encoding(email: str) -> None:
    """Whatever string the caller passes is the string the server receives.

    ``email`` is a server-validated filter, so the SDK forwards it verbatim
    rather than pre-judging it. Reserved URL characters (``&``, ``=``, ``#``,
    ``?``, ``%``, ``+``), unicode, and whitespace must therefore survive query
    encoding intact — a mangled filter would silently return the wrong users
    instead of erroring.
    """
    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    namespace = Users(http=HTTPClient(config, ADMIN_API_VERSION))

    with respx.mock:
        route = respx.get(f"{BASE_URL}/admin/users").mock(
            return_value=httpx.Response(200, json=_page([]))
        )
        namespace.list(email=email).to_list()

        request = route.calls.last.request

    assert request.url.params["email"] == email
    assert request.url.path == "/admin/users"
    assert set(request.url.params.keys()) == {"email"}
