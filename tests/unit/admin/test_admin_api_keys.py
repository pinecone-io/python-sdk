"""Unit tests for the Admin ApiKeys namespace."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
import respx

from pinecone._internal.config import PineconeConfig
from pinecone._internal.constants import ADMIN_API_VERSION
from pinecone._internal.http_client import HTTPClient
from pinecone.admin.api_keys import ApiKeys
from pinecone.errors.exceptions import ValidationError
from pinecone.models.admin.api_key import APIKeyList, APIKeyModel, APIKeyRole, APIKeyWithSecret

BASE_URL = "https://api.test.pinecone.io"


def _api_key_response(
    *,
    id: str = "key-abc123",
    name: str = "my-key",
    project_id: str = "proj-abc123",
    roles: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "id": id,
        "name": name,
        "project_id": project_id,
        "roles": roles or ["ProjectEditor"],
    }


def _api_key_with_secret_response(
    *,
    key: dict[str, Any] | None = None,
    value: str = "pckey_abc_123",
) -> dict[str, Any]:
    return {
        "key": key or _api_key_response(),
        "value": value,
    }


@pytest.fixture
def http_client() -> HTTPClient:
    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    return HTTPClient(config, ADMIN_API_VERSION)


@pytest.fixture
def api_keys(http_client: HTTPClient) -> ApiKeys:
    return ApiKeys(http=http_client)


# ---------------------------------------------------------------------------
# list()
# ---------------------------------------------------------------------------


@respx.mock
def test_list_api_keys(api_keys: ApiKeys) -> None:
    respx.get(f"{BASE_URL}/admin/projects/p1/api-keys").mock(
        return_value=httpx.Response(
            200,
            json={"data": [_api_key_response(id="k1", name="key1", project_id="p1")]},
        ),
    )

    result = api_keys.list(project_id="p1")

    assert isinstance(result, APIKeyList)
    assert len(result) == 1
    key = result[0]
    assert isinstance(key, APIKeyModel)
    assert key.id == "k1"
    assert key.name == "key1"
    assert key.project_id == "p1"
    assert key.roles == ["ProjectEditor"]


@respx.mock
def test_list_api_keys_empty(api_keys: ApiKeys) -> None:
    respx.get(f"{BASE_URL}/admin/projects/p1/api-keys").mock(
        return_value=httpx.Response(200, json={"data": []}),
    )

    result = api_keys.list(project_id="p1")

    assert isinstance(result, APIKeyList)
    assert len(result) == 0


# ---------------------------------------------------------------------------
# create()
# ---------------------------------------------------------------------------


@respx.mock
def test_create_api_key(api_keys: ApiKeys) -> None:
    route = respx.post(f"{BASE_URL}/admin/projects/p1/api-keys").mock(
        return_value=httpx.Response(
            201,
            json=_api_key_with_secret_response(
                key=_api_key_response(id="k1", name="mykey", project_id="p1"),
                value="pckey_abc_123",
            ),
        ),
    )

    result = api_keys.create(project_id="p1", name="mykey")

    assert isinstance(result, APIKeyWithSecret)
    assert result.key.id == "k1"
    assert result.key.name == "mykey"
    assert result.key.project_id == "p1"
    assert result.value == "pckey_abc_123"

    # Verify request body contains only name
    request = route.calls[0].request
    expected_body = httpx.Request("POST", "/", json={"name": "mykey"})
    assert request.content == expected_body.content


@respx.mock
def test_create_api_key_with_roles(api_keys: ApiKeys) -> None:
    route = respx.post(f"{BASE_URL}/admin/projects/p1/api-keys").mock(
        return_value=httpx.Response(
            201,
            json=_api_key_with_secret_response(
                key=_api_key_response(
                    id="k1",
                    name="mykey",
                    project_id="p1",
                    roles=["DataPlaneEditor", "DataPlaneViewer"],
                ),
            ),
        ),
    )

    result = api_keys.create(
        project_id="p1",
        name="mykey",
        roles=["DataPlaneEditor", "DataPlaneViewer"],
    )

    assert isinstance(result, APIKeyWithSecret)
    assert result.key.roles == ["DataPlaneEditor", "DataPlaneViewer"]

    # Verify roles are included in the POST body
    request = route.calls[0].request
    expected_body = httpx.Request(
        "POST",
        "/",
        json={"name": "mykey", "roles": ["DataPlaneEditor", "DataPlaneViewer"]},
    )
    assert request.content == expected_body.content


@respx.mock
def test_api_key_list_name_optional(api_keys: ApiKeys) -> None:
    respx.get(f"{BASE_URL}/admin/projects/p1/api-keys").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    _api_key_response(id="k1", name="has-name", project_id="p1"),
                    {
                        "id": "k2",
                        "name": None,
                        "project_id": "p1",
                        "roles": ["ProjectEditor"],
                    },
                ]
            },
        ),
    )

    result = api_keys.list(project_id="p1")

    assert len(result) == 2
    assert result[0].name == "has-name"
    assert result[1].name is None
    for key in result:
        assert isinstance(key.name, (str, type(None)))


@respx.mock
def test_create_without_description(api_keys: ApiKeys) -> None:
    route = respx.post(f"{BASE_URL}/admin/projects/p1/api-keys").mock(
        return_value=httpx.Response(
            201,
            json=_api_key_with_secret_response(
                key=_api_key_response(id="k1", name="key", project_id="p1"),
            ),
        ),
    )

    api_keys.create(project_id="p1", name="key")

    request = route.calls[0].request
    expected_body = httpx.Request("POST", "/", json={"name": "key"})
    assert request.content == expected_body.content


def test_api_key_create_description_not_accepted(api_keys: ApiKeys) -> None:
    with pytest.raises(TypeError, match="description"):
        api_keys.create(project_id="p1", name="key", description="My key")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# describe()
# ---------------------------------------------------------------------------


@respx.mock
def test_describe_api_key(api_keys: ApiKeys) -> None:
    respx.get(f"{BASE_URL}/admin/api-keys/k1").mock(
        return_value=httpx.Response(
            200,
            json=_api_key_response(id="k1", name="my-key"),
        ),
    )

    result = api_keys.describe(api_key_id="k1")

    assert isinstance(result, APIKeyModel)
    assert result.id == "k1"
    assert result.name == "my-key"


# ---------------------------------------------------------------------------
# update()
# ---------------------------------------------------------------------------


@respx.mock
def test_update_api_key_name(api_keys: ApiKeys) -> None:
    route = respx.patch(f"{BASE_URL}/admin/api-keys/k1").mock(
        return_value=httpx.Response(
            200,
            json=_api_key_response(id="k1", name="newname"),
        ),
    )

    result = api_keys.update(api_key_id="k1", name="newname")

    assert isinstance(result, APIKeyModel)
    assert result.name == "newname"

    # Verify PATCH body contains only name
    request = route.calls[0].request
    expected_body = httpx.Request("PATCH", "/", json={"name": "newname"})
    assert request.content == expected_body.content


@respx.mock
def test_update_api_key_roles(api_keys: ApiKeys) -> None:
    route = respx.patch(f"{BASE_URL}/admin/api-keys/k1").mock(
        return_value=httpx.Response(
            200,
            json=_api_key_response(id="k1", roles=["DataPlaneViewer"]),
        ),
    )

    result = api_keys.update(api_key_id="k1", roles=["DataPlaneViewer"])

    assert isinstance(result, APIKeyModel)
    assert result.roles == ["DataPlaneViewer"]

    # Verify PATCH body contains only roles — replaces entire role set
    request = route.calls[0].request
    expected_body = httpx.Request("PATCH", "/", json={"roles": ["DataPlaneViewer"]})
    assert request.content == expected_body.content


# ---------------------------------------------------------------------------
# delete()
# ---------------------------------------------------------------------------


@respx.mock
def test_delete_api_key(api_keys: ApiKeys) -> None:
    respx.delete(f"{BASE_URL}/admin/api-keys/k1").mock(
        return_value=httpx.Response(202),
    )

    result = api_keys.delete(api_key_id="k1")

    assert result is None


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def test_list_requires_project_id(api_keys: ApiKeys) -> None:
    with pytest.raises(ValidationError, match="project_id"):
        api_keys.list(project_id="")


def test_create_requires_name(api_keys: ApiKeys) -> None:
    with pytest.raises(ValidationError, match="name"):
        api_keys.create(project_id="p1", name="")


def test_create_requires_project_id(api_keys: ApiKeys) -> None:
    with pytest.raises(ValidationError, match="project_id"):
        api_keys.create(project_id="", name="mykey")


def test_api_key_create_name_too_long(api_keys: ApiKeys) -> None:
    with pytest.raises(ValidationError, match="name"):
        api_keys.create(project_id="p1", name="x" * 81)


def test_describe_requires_api_key_id(api_keys: ApiKeys) -> None:
    with pytest.raises(ValidationError, match="api_key_id"):
        api_keys.describe(api_key_id="")


def test_update_requires_api_key_id(api_keys: ApiKeys) -> None:
    with pytest.raises(ValidationError, match="api_key_id"):
        api_keys.update(api_key_id="")


def test_delete_requires_api_key_id(api_keys: ApiKeys) -> None:
    with pytest.raises(ValidationError, match="api_key_id"):
        api_keys.delete(api_key_id="")


# ---------------------------------------------------------------------------
# keyword-only enforcement
# ---------------------------------------------------------------------------


def test_keyword_only_args(api_keys: ApiKeys) -> None:
    with pytest.raises(TypeError):
        api_keys.list("p1")  # type: ignore[misc]

    with pytest.raises(TypeError):
        api_keys.create("p1", "mykey")  # type: ignore[misc]

    with pytest.raises(TypeError):
        api_keys.describe("k1")  # type: ignore[misc]

    with pytest.raises(TypeError):
        api_keys.update("k1")  # type: ignore[misc]

    with pytest.raises(TypeError):
        api_keys.delete("k1")  # type: ignore[misc]


# ---------------------------------------------------------------------------
# repr
# ---------------------------------------------------------------------------


def test_repr(api_keys: ApiKeys) -> None:
    assert repr(api_keys) == "ApiKeys()"


# ---------------------------------------------------------------------------
# APIKeyRole enum
# ---------------------------------------------------------------------------


def test_api_key_role_values() -> None:
    assert APIKeyRole.PROJECT_EDITOR == "ProjectEditor"
    assert APIKeyRole.PROJECT_VIEWER == "ProjectViewer"
    assert APIKeyRole.CONTROL_PLANE_EDITOR == "ControlPlaneEditor"
    assert APIKeyRole.CONTROL_PLANE_VIEWER == "ControlPlaneViewer"
    assert APIKeyRole.DATA_PLANE_EDITOR == "DataPlaneEditor"
    assert APIKeyRole.DATA_PLANE_VIEWER == "DataPlaneViewer"
    assert len(APIKeyRole) == 6


def test_api_key_role_is_str() -> None:
    assert isinstance(APIKeyRole.PROJECT_EDITOR, str)
    assert APIKeyRole.PROJECT_EDITOR.value == "ProjectEditor"


@respx.mock
def test_create_with_enum_roles(api_keys: ApiKeys) -> None:
    route = respx.post(f"{BASE_URL}/admin/projects/p1/api-keys").mock(
        return_value=httpx.Response(
            201,
            json=_api_key_with_secret_response(
                key=_api_key_response(
                    id="k1", name="mykey", project_id="p1", roles=["ProjectViewer"]
                ),
            ),
        ),
    )

    result = api_keys.create(
        project_id="p1",
        name="mykey",
        roles=[APIKeyRole.PROJECT_VIEWER],
    )

    assert result.key.roles == ["ProjectViewer"]

    # Enum value serialized as plain string on the wire
    request = route.calls[0].request
    expected_body = httpx.Request("POST", "/", json={"name": "mykey", "roles": ["ProjectViewer"]})
    assert request.content == expected_body.content


@respx.mock
def test_update_with_enum_roles(api_keys: ApiKeys) -> None:
    route = respx.patch(f"{BASE_URL}/admin/api-keys/k1").mock(
        return_value=httpx.Response(
            200,
            json=_api_key_response(id="k1", roles=["ControlPlaneViewer"]),
        ),
    )

    result = api_keys.update(api_key_id="k1", roles=[APIKeyRole.CONTROL_PLANE_VIEWER])

    assert result.roles == ["ControlPlaneViewer"]

    request = route.calls[0].request
    expected_body = httpx.Request("PATCH", "/", json={"roles": ["ControlPlaneViewer"]})
    assert request.content == expected_body.content


def test_create_invalid_role_raises(api_keys: ApiKeys) -> None:
    with pytest.raises(ValidationError, match="Invalid role"):
        api_keys.create(project_id="p1", name="mykey", roles=["NotARealRole"])


def test_update_invalid_role_raises(api_keys: ApiKeys) -> None:
    with pytest.raises(ValidationError, match="Invalid role"):
        api_keys.update(api_key_id="k1", roles=["NotARealRole"])


def test_invalid_role_message_names_parameter_value_and_all_allowed_values(
    api_keys: ApiKeys,
) -> None:
    with pytest.raises(ValidationError) as exc_info:
        api_keys.create(
            project_id="p1", name="mykey", roles=[APIKeyRole.PROJECT_EDITOR, "NotARealRole"]
        )

    message = str(exc_info.value)
    assert "roles[1]" in message
    assert "'NotARealRole'" in message
    for role in APIKeyRole:
        assert repr(role.value) in message


@respx.mock
def test_decoded_roles_are_api_key_role_instances(api_keys: ApiKeys) -> None:
    respx.get(f"{BASE_URL}/admin/api-keys/k1").mock(
        return_value=httpx.Response(
            200,
            json=_api_key_response(id="k1", roles=["ProjectEditor"]),
        ),
    )

    result = api_keys.describe(api_key_id="k1")

    assert isinstance(result.roles[0], APIKeyRole)


@respx.mock
def test_decoded_roles_enum_comparison(api_keys: ApiKeys) -> None:
    respx.get(f"{BASE_URL}/admin/api-keys/k1").mock(
        return_value=httpx.Response(
            200,
            json=_api_key_response(id="k1", roles=["ProjectEditor"]),
        ),
    )

    result = api_keys.describe(api_key_id="k1")

    assert result.roles[0] == APIKeyRole.PROJECT_EDITOR
    assert result.roles[0] == "ProjectEditor"


@respx.mock
def test_decoded_role_property_is_api_key_role(api_keys: ApiKeys) -> None:
    respx.get(f"{BASE_URL}/admin/api-keys/k1").mock(
        return_value=httpx.Response(
            200,
            json=_api_key_response(id="k1", roles=["ProjectEditor"]),
        ),
    )

    result = api_keys.describe(api_key_id="k1")

    assert isinstance(result.role, APIKeyRole)
    assert result.role == APIKeyRole.PROJECT_EDITOR


@respx.mock
def test_create_mixed_enum_and_str_roles(api_keys: ApiKeys) -> None:
    route = respx.post(f"{BASE_URL}/admin/projects/p1/api-keys").mock(
        return_value=httpx.Response(
            201,
            json=_api_key_with_secret_response(
                key=_api_key_response(
                    id="k1",
                    name="mykey",
                    project_id="p1",
                    roles=["ProjectEditor", "DataPlaneViewer"],
                ),
            ),
        ),
    )

    result = api_keys.create(
        project_id="p1",
        name="mykey",
        roles=[APIKeyRole.PROJECT_EDITOR, "DataPlaneViewer"],
    )

    assert result.key.roles == ["ProjectEditor", "DataPlaneViewer"]

    request = route.calls[0].request
    expected_body = httpx.Request(
        "POST",
        "/",
        json={"name": "mykey", "roles": ["ProjectEditor", "DataPlaneViewer"]},
    )
    assert request.content == expected_body.content
