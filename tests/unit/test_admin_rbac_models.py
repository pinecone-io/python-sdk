"""Tests for the 2026-07 admin user / invite / service account / role binding models (#96).

Three things are worth pinning here, and they pull in opposite directions.

*The spec's own example payloads must decode exactly.* Each ``_EXAMPLE_*`` constant
below is transcribed verbatim from ``admin_2026-07.oas.yaml``, so a schema change
upstream shows up as a failure here rather than as a runtime surprise. Round-trip is
checked as ``decode -> encode -> decode`` struct equality, which is the only form that
catches a dropped field: comparing ``to_dict()`` against the payload would not, because
an absent optional field legitimately becomes an explicit ``None``.

*Responses stay permissive, inputs stay strict.* The server may add an enum value
between SDK releases, so ``status`` / ``role`` / ``principal_type`` / ``resource_type``
are plain ``str`` on the response models and a made-up value must decode rather than
raise. ``RoleBindingInput`` is the one model the SDK sends, so it validates on
construction and the message has to name the field, echo the bad value, and list the
accepted set.

*Cursors are opaque.* A pagination token is base64 today, but the SDK must never parse
one, so the property test pushes arbitrary printable text through the envelope and
demands the exact bytes back.
"""

from __future__ import annotations

from typing import Any

import msgspec
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pinecone.errors.exceptions import PineconeValueError
from pinecone.models.admin.invite import InviteList, InviteModel, InviteStatus
from pinecone.models.admin.pagination import PaginationResponse
from pinecone.models.admin.role_binding import (
    PrincipalType,
    ResourceType,
    RoleBindingInput,
    RoleBindingList,
    RoleBindingModel,
    RoleName,
)
from pinecone.models.admin.service_account import (
    ServiceAccountList,
    ServiceAccountModel,
    ServiceAccountWithSecret,
)
from pinecone.models.admin.user import UserList, UserModel

_EXAMPLE_USER: dict[str, Any] = {
    "email": "alice@example.com",
    "id": "e2e92523-85dc-4142-b8c2-e681be8b78df",
    "name": "Alice Example",
}

_EXAMPLE_USER_LIST: dict[str, Any] = {
    "data": [_EXAMPLE_USER],
    "pagination": {"next": "eyJsYXN0X2lkIjoiZTJlOTI1MjMifQ=="},
}

_EXAMPLE_INVITE: dict[str, Any] = {
    "created_at": "2026-04-14T20:00:00Z",
    "email": "newhire@acme.com",
    "expires_at": "2026-05-21T03:00:00Z",
    "id": "9c8e3528-b9c0-4358-84ce-84c28e91b566",
    "processed_at": None,
    "status": "pending",
}

_EXAMPLE_INVITE_LIST: dict[str, Any] = {
    "data": [_EXAMPLE_INVITE],
    "pagination": {"next": "eyJsYXN0X2lkIjoiOWM4ZTM1MjgifQ=="},
}

_EXAMPLE_SERVICE_ACCOUNT: dict[str, Any] = {
    "client_id": "l3Ow0CmFyc4jOONcwiKUCRqQKN0tiCAn",
    "created_at": "2026-04-10T15:23:00Z",
    "id": "f8a3b2c1-4d5e-6f7a-8b9c-0d1e2f3a4b5c",
    "name": "My Service Account",
    "updated_at": "2026-04-12T09:11:00Z",
}

_EXAMPLE_SERVICE_ACCOUNT_LIST: dict[str, Any] = {
    "data": [_EXAMPLE_SERVICE_ACCOUNT],
    "pagination": {"next": "eyJsYXN0X2lkIjoiZDI0MTc3YTAifQ=="},
}

_EXAMPLE_SERVICE_ACCOUNT_WITH_SECRET: dict[str, Any] = {
    "client_secret": "8p-kkC23XOWvkCosKq-BOn3G74qp__rBcDMxc82iB4gfzRvuhSCRBKM7C5Q7TAzj",
    "service_account": {
        "client_id": "l3Ow0CmFyc4jOONcwiKUCRqQKN0tiCAn",
        "created_at": "2026-04-10T15:23:00Z",
        "id": "f8a3b2c1-4d5e-6f7a-8b9c-0d1e2f3a4b5c",
        "name": "My Service Account",
        "updated_at": "2026-04-10T15:23:00Z",
    },
}

_EXAMPLE_ROLE_BINDING: dict[str, Any] = {
    "created_at": "2026-04-10T15:23:00Z",
    "id": "9a8e3528-b9c0-4358-84ce-84c28e91b566",
    "principal_id": "f8a3b2c1-4d5e-6f7a-8b9c-0d1e2f3a4b5c",
    "principal_type": "service_account",
    "resource_id": "a2f7dddb-1597-4eff-9f71-535fde243f58",
    "resource_type": "project",
    "role": "DataPlaneEditor",
}

_EXAMPLE_ROLE_BINDING_LIST: dict[str, Any] = {
    "data": [_EXAMPLE_ROLE_BINDING],
    "pagination": {"next": "eyJsYXN0X2lkIjoiOWE4ZTM1MjgifQ=="},
}

_SPEC_EXAMPLES: list[tuple[str, type, dict[str, Any]]] = [
    ("User", UserModel, _EXAMPLE_USER),
    ("UserList", UserList, _EXAMPLE_USER_LIST),
    ("Invite", InviteModel, _EXAMPLE_INVITE),
    ("InviteList", InviteList, _EXAMPLE_INVITE_LIST),
    ("ServiceAccount", ServiceAccountModel, _EXAMPLE_SERVICE_ACCOUNT),
    ("ServiceAccountList", ServiceAccountList, _EXAMPLE_SERVICE_ACCOUNT_LIST),
    (
        "ServiceAccountWithSecret",
        ServiceAccountWithSecret,
        _EXAMPLE_SERVICE_ACCOUNT_WITH_SECRET,
    ),
    ("RoleBinding", RoleBindingModel, _EXAMPLE_ROLE_BINDING),
    ("RoleBindingList", RoleBindingList, _EXAMPLE_ROLE_BINDING_LIST),
]


def _round_trip(model_type: type, payload: dict[str, Any]) -> Any:
    decoded = msgspec.json.decode(msgspec.json.encode(payload), type=model_type)
    reencoded = msgspec.json.decode(msgspec.json.encode(decoded), type=model_type)
    assert reencoded == decoded
    return decoded


class TestSpecExamples:
    @pytest.mark.parametrize(
        ("schema", "model_type", "payload"),
        [(s, t, p) for s, t, p in _SPEC_EXAMPLES],
        ids=[s for s, _, _ in _SPEC_EXAMPLES],
    )
    def test_spec_example_round_trips(
        self, schema: str, model_type: type, payload: dict[str, Any]
    ) -> None:
        _round_trip(model_type, payload)

    def test_user_example_fields(self) -> None:
        user = _round_trip(UserModel, _EXAMPLE_USER)
        assert user.id == "e2e92523-85dc-4142-b8c2-e681be8b78df"
        assert user.email == "alice@example.com"
        assert user.name == "Alice Example"

    def test_user_list_example_fields(self) -> None:
        users = _round_trip(UserList, _EXAMPLE_USER_LIST)
        assert len(users) == 1
        assert users[0].email == "alice@example.com"
        assert users.emails() == ["alice@example.com"]
        assert users.pagination_token == "eyJsYXN0X2lkIjoiZTJlOTI1MjMifQ=="
        assert users.has_more is True

    def test_invite_example_fields(self) -> None:
        invite = _round_trip(InviteModel, _EXAMPLE_INVITE)
        assert invite.status == InviteStatus.PENDING
        assert invite.expires_at == "2026-05-21T03:00:00Z"
        assert invite.processed_at is None
        assert invite.created_at == "2026-04-14T20:00:00Z"

    def test_invite_list_example_fields(self) -> None:
        invites = _round_trip(InviteList, _EXAMPLE_INVITE_LIST)
        assert invites.emails() == ["newhire@acme.com"]
        assert invites.pagination_token == "eyJsYXN0X2lkIjoiOWM4ZTM1MjgifQ=="

    def test_service_account_example_fields(self) -> None:
        account = _round_trip(ServiceAccountModel, _EXAMPLE_SERVICE_ACCOUNT)
        assert account.id == "f8a3b2c1-4d5e-6f7a-8b9c-0d1e2f3a4b5c"
        assert account.name == "My Service Account"
        assert account.client_id == "l3Ow0CmFyc4jOONcwiKUCRqQKN0tiCAn"
        assert account.created_at == "2026-04-10T15:23:00Z"
        assert account.updated_at == "2026-04-12T09:11:00Z"

    def test_service_account_list_example_fields(self) -> None:
        accounts = _round_trip(ServiceAccountList, _EXAMPLE_SERVICE_ACCOUNT_LIST)
        assert accounts.names() == ["My Service Account"]
        assert accounts.pagination_token == "eyJsYXN0X2lkIjoiZDI0MTc3YTAifQ=="

    def test_service_account_with_secret_example_fields(self) -> None:
        created = _round_trip(ServiceAccountWithSecret, _EXAMPLE_SERVICE_ACCOUNT_WITH_SECRET)
        assert created.client_secret == _EXAMPLE_SERVICE_ACCOUNT_WITH_SECRET["client_secret"]
        assert created.service_account.name == "My Service Account"
        assert created.service_account.updated_at == "2026-04-10T15:23:00Z"

    def test_role_binding_example_fields(self) -> None:
        binding = _round_trip(RoleBindingModel, _EXAMPLE_ROLE_BINDING)
        assert binding.principal_type == PrincipalType.SERVICE_ACCOUNT
        assert binding.principal_id == "f8a3b2c1-4d5e-6f7a-8b9c-0d1e2f3a4b5c"
        assert binding.resource_type == ResourceType.PROJECT
        assert binding.resource_id == "a2f7dddb-1597-4eff-9f71-535fde243f58"
        assert binding.role == RoleName.DATA_PLANE_EDITOR
        assert binding.created_at == "2026-04-10T15:23:00Z"

    def test_role_binding_list_example_fields(self) -> None:
        bindings = _round_trip(RoleBindingList, _EXAMPLE_ROLE_BINDING_LIST)
        assert bindings.roles() == ["DataPlaneEditor"]
        assert bindings.pagination_token == "eyJsYXN0X2lkIjoiOWE4ZTM1MjgifQ=="


class TestOptionalFieldsAbsent:
    def test_user_without_name(self) -> None:
        user = _round_trip(UserModel, {"id": "u1", "email": "alice@example.com"})
        assert user.name is None
        assert user.to_dict() == {"id": "u1", "email": "alice@example.com", "name": None}

    def test_invite_without_expires_or_processed(self) -> None:
        invite = _round_trip(
            InviteModel,
            {
                "id": "i1",
                "email": "newhire@acme.com",
                "status": "expired",
                "created_at": "2026-04-14T20:00:00Z",
            },
        )
        assert invite.expires_at is None
        assert invite.processed_at is None

    def test_invite_with_null_expires_at(self) -> None:
        invite = _round_trip(
            InviteModel,
            {
                "id": "i1",
                "email": "newhire@acme.com",
                "status": "pending",
                "expires_at": None,
                "processed_at": None,
                "created_at": "2026-04-14T20:00:00Z",
            },
        )
        assert invite.expires_at is None

    def test_processed_invite_carries_processed_at(self) -> None:
        invite = _round_trip(
            InviteModel,
            {
                "id": "i1",
                "email": "newhire@acme.com",
                "status": "processed",
                "expires_at": None,
                "processed_at": "2026-04-15T10:00:00Z",
                "created_at": "2026-04-14T20:00:00Z",
            },
        )
        assert invite.status == InviteStatus.PROCESSED
        assert invite.processed_at == "2026-04-15T10:00:00Z"

    @pytest.mark.parametrize(
        "model_type", [UserList, InviteList, ServiceAccountList, RoleBindingList]
    )
    def test_list_with_pagination_absent(self, model_type: type) -> None:
        page = _round_trip(model_type, {"data": []})
        assert page.pagination is None
        assert page.pagination_token is None
        assert page.has_more is False
        assert len(page) == 0
        assert list(page) == []

    @pytest.mark.parametrize(
        "model_type", [UserList, InviteList, ServiceAccountList, RoleBindingList]
    )
    def test_list_with_pagination_null(self, model_type: type) -> None:
        page = _round_trip(model_type, {"data": [], "pagination": None})
        assert page.pagination is None
        assert page.has_more is False

    @pytest.mark.parametrize(
        "model_type", [UserList, InviteList, ServiceAccountList, RoleBindingList]
    )
    def test_list_to_dict_shape(self, model_type: type) -> None:
        page = _round_trip(model_type, {"data": []})
        assert page.to_dict() == {"data": [], "pagination": None}

    def test_pagination_envelope_without_next(self) -> None:
        envelope = _round_trip(PaginationResponse, {})
        assert envelope.next is None

    def test_role_binding_input_without_resource_id(self) -> None:
        binding = RoleBindingInput(resource_type="organization", role="OrgMember")
        assert binding.resource_id is None
        assert msgspec.json.decode(msgspec.json.encode(binding)) == {
            "resource_type": "organization",
            "role": "OrgMember",
        }


class TestForwardCompatibleEnums:
    def test_unknown_role_decodes_as_string(self) -> None:
        payload = {**_EXAMPLE_ROLE_BINDING, "role": "QuantumOverseer"}
        binding = _round_trip(RoleBindingModel, payload)
        assert binding.role == "QuantumOverseer"
        assert binding.role not in {r.value for r in RoleName}

    def test_unknown_principal_type_decodes_as_string(self) -> None:
        payload = {**_EXAMPLE_ROLE_BINDING, "principal_type": "workload_identity"}
        binding = _round_trip(RoleBindingModel, payload)
        assert binding.principal_type == "workload_identity"

    def test_unknown_resource_type_decodes_as_string(self) -> None:
        payload = {**_EXAMPLE_ROLE_BINDING, "resource_type": "namespace"}
        binding = _round_trip(RoleBindingModel, payload)
        assert binding.resource_type == "namespace"

    def test_unknown_invite_status_decodes_as_string(self) -> None:
        payload = {**_EXAMPLE_INVITE, "status": "revoked"}
        invite = _round_trip(InviteModel, payload)
        assert invite.status == "revoked"
        assert invite.status not in {s.value for s in InviteStatus}

    def test_unknown_role_survives_a_list_page(self) -> None:
        payload = {
            "data": [{**_EXAMPLE_ROLE_BINDING, "role": "QuantumOverseer"}],
            "pagination": None,
        }
        bindings = _round_trip(RoleBindingList, payload)
        assert bindings.roles() == ["QuantumOverseer"]


class TestRoleBindingInputValidation:
    def test_invalid_resource_type_names_field_value_and_options(self) -> None:
        with pytest.raises(PineconeValueError) as exc_info:
            RoleBindingInput(resource_type="org", role="OrgMember")
        message = str(exc_info.value)
        assert "resource_type" in message
        assert "'org'" in message
        assert "'organization'" in message
        assert "'project'" in message

    def test_invalid_role_names_field_value_and_options(self) -> None:
        with pytest.raises(PineconeValueError) as exc_info:
            RoleBindingInput(resource_type="organization", role="Overlord")
        message = str(exc_info.value)
        assert "role" in message
        assert "'Overlord'" in message
        assert "'OrgOwner'" in message

    def test_validation_error_is_a_value_error(self) -> None:
        with pytest.raises(ValueError):
            RoleBindingInput(resource_type="org", role="OrgMember")

    def test_project_scope_requires_resource_id(self) -> None:
        with pytest.raises(PineconeValueError, match="resource_id is required"):
            RoleBindingInput(resource_type="project", role="ProjectViewer")

    def test_project_scope_rejects_empty_resource_id(self) -> None:
        with pytest.raises(PineconeValueError, match="resource_id is required"):
            RoleBindingInput(resource_type="project", role="ProjectViewer", resource_id="")

    def test_enum_and_string_are_interchangeable(self) -> None:
        from_enum = RoleBindingInput(
            resource_type=ResourceType.PROJECT,
            role=RoleName.PROJECT_VIEWER,
            resource_id="p1",
        )
        from_string = RoleBindingInput(
            resource_type="project", role="ProjectViewer", resource_id="p1"
        )
        assert from_enum == from_string
        assert msgspec.json.encode(from_enum) == msgspec.json.encode(from_string)

    def test_enum_inputs_normalize_to_plain_strings(self) -> None:
        binding = RoleBindingInput(
            resource_type=ResourceType.ORGANIZATION, role=RoleName.ORG_BILLING_ADMIN
        )
        assert type(binding.resource_type) is str
        assert type(binding.role) is str


class TestSecretMasking:
    def test_repr_masks_the_client_secret(self) -> None:
        created = msgspec.json.decode(
            msgspec.json.encode(_EXAMPLE_SERVICE_ACCOUNT_WITH_SECRET),
            type=ServiceAccountWithSecret,
        )
        assert created.client_secret not in repr(created)
        assert "...TAzj" in repr(created)
        assert str(created) == repr(created)

    def test_to_dict_still_carries_the_secret(self) -> None:
        created = ServiceAccountWithSecret(
            service_account=msgspec.json.decode(
                msgspec.json.encode(_EXAMPLE_SERVICE_ACCOUNT), type=ServiceAccountModel
            ),
            client_secret="super-secret",
        )
        assert created.to_dict()["client_secret"] == "super-secret"
        assert created.to_dict()["service_account"]["name"] == "My Service Account"

    def test_short_secret_is_fully_masked(self) -> None:
        created = ServiceAccountWithSecret(
            service_account=msgspec.json.decode(
                msgspec.json.encode(_EXAMPLE_SERVICE_ACCOUNT), type=ServiceAccountModel
            ),
            client_secret="ab",
        )
        assert "***" in repr(created)


_TIMESTAMPS = st.text(alphabet="0123456789-:TZ", min_size=1, max_size=24)
_IDS = st.text(alphabet="abcdef0123456789-", min_size=1, max_size=36)
_NAMES = st.text(min_size=1, max_size=40)
_EMAILS = st.text(alphabet="abcdefghijklmnopqrstuvwxyz.", min_size=1, max_size=12).map(
    lambda local: f"{local}@example.com"
)
_CURSORS = st.text(
    alphabet=st.characters(min_codepoint=32, max_codepoint=126), min_size=1, max_size=120
)

_USERS = st.builds(UserModel, id=_IDS, email=_EMAILS, name=st.none() | _NAMES)
_INVITES = st.builds(
    InviteModel,
    id=_IDS,
    email=_EMAILS,
    status=st.sampled_from([s.value for s in InviteStatus]) | _NAMES,
    expires_at=st.none() | _TIMESTAMPS,
    processed_at=st.none() | _TIMESTAMPS,
    created_at=_TIMESTAMPS,
)
_SERVICE_ACCOUNTS = st.builds(
    ServiceAccountModel,
    id=_IDS,
    name=_NAMES,
    client_id=_IDS,
    created_at=_TIMESTAMPS,
    updated_at=_TIMESTAMPS,
)
_ROLE_BINDINGS = st.builds(
    RoleBindingModel,
    id=_IDS,
    principal_type=st.sampled_from([p.value for p in PrincipalType]) | _NAMES,
    principal_id=_IDS,
    resource_type=st.sampled_from([r.value for r in ResourceType]) | _NAMES,
    resource_id=_IDS,
    role=st.sampled_from([r.value for r in RoleName]) | _NAMES,
    created_at=_TIMESTAMPS,
)

_PAGINATION = st.none() | st.builds(PaginationResponse, next=st.none() | _CURSORS)


class TestRoundTripProperties:
    @settings(max_examples=100)
    @given(model=_USERS | _INVITES | _SERVICE_ACCOUNTS | _ROLE_BINDINGS)
    def test_entity_models_survive_encode_decode(self, model: Any) -> None:
        assert msgspec.json.decode(msgspec.json.encode(model), type=type(model)) == model

    @settings(max_examples=100)
    @given(model=_USERS | _INVITES | _SERVICE_ACCOUNTS | _ROLE_BINDINGS)
    def test_entity_to_dict_reconstructs_the_model(self, model: Any) -> None:
        as_dict = model.to_dict()
        assert set(as_dict) == set(model.__struct_fields__)
        assert type(model)(**as_dict) == model

    @settings(max_examples=100)
    @given(users=st.lists(_USERS, max_size=5), pagination=_PAGINATION)
    def test_user_list_survives_encode_decode(
        self, users: list[UserModel], pagination: PaginationResponse | None
    ) -> None:
        page = UserList(data=users, pagination=pagination)
        assert msgspec.json.decode(msgspec.json.encode(page), type=UserList) == page

    @settings(max_examples=100)
    @given(bindings=st.lists(_ROLE_BINDINGS, max_size=5), pagination=_PAGINATION)
    def test_role_binding_list_survives_encode_decode(
        self, bindings: list[RoleBindingModel], pagination: PaginationResponse | None
    ) -> None:
        page = RoleBindingList(data=bindings, pagination=pagination)
        assert msgspec.json.decode(msgspec.json.encode(page), type=RoleBindingList) == page

    @settings(max_examples=100)
    @given(
        resource_type=st.sampled_from([r.value for r in ResourceType]),
        role=st.sampled_from([r.value for r in RoleName]),
        resource_id=_IDS,
    )
    def test_role_binding_input_survives_encode_decode(
        self, resource_type: str, role: str, resource_id: str
    ) -> None:
        binding = RoleBindingInput(resource_type=resource_type, role=role, resource_id=resource_id)
        assert msgspec.json.decode(msgspec.json.encode(binding), type=RoleBindingInput) == binding


class TestCursorOpacity:
    @settings(max_examples=200)
    @given(cursor=_CURSORS)
    def test_arbitrary_cursor_passes_through_untouched(self, cursor: str) -> None:
        payload = {"data": [], "pagination": {"next": cursor}}
        page = msgspec.json.decode(msgspec.json.encode(payload), type=UserList)
        assert page.pagination_token == cursor
        assert page.has_more is True
        reencoded = msgspec.json.decode(msgspec.json.encode(page), type=UserList)
        assert reencoded.pagination_token == cursor

    @settings(max_examples=100)
    @given(cursor=_CURSORS)
    def test_cursor_opacity_holds_for_every_list_model(self, cursor: str) -> None:
        for model_type in (UserList, InviteList, ServiceAccountList, RoleBindingList):
            payload = {"data": [], "pagination": {"next": cursor}}
            page = msgspec.json.decode(msgspec.json.encode(payload), type=model_type)
            assert page.pagination_token == cursor
            assert page.to_dict()["pagination"] == {"next": cursor}
