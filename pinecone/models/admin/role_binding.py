"""Role binding models for the Admin API."""

from __future__ import annotations

from collections.abc import Iterator
from enum import Enum
from typing import Any

from msgspec import Struct

from pinecone._internal.validation import require_one_of
from pinecone.errors.exceptions import ValidationError
from pinecone.models._mixin import StructDictMixin, _struct_to_dict_recursive
from pinecone.models.admin.pagination import PaginationResponse


class PrincipalType(str, Enum):
    """The kind of principal that receives permissions from a role binding.

    Possible values: ``user``, ``service_account``, ``api_key``, ``invite``.

    Examples:
        >>> from pinecone.models.admin.role_binding import PrincipalType
        >>> PrincipalType.SERVICE_ACCOUNT == "service_account"
        True
    """

    USER = "user"
    SERVICE_ACCOUNT = "service_account"
    API_KEY = "api_key"
    INVITE = "invite"


class ResourceType(str, Enum):
    """The kind of resource scope a role binding applies to.

    Possible values: ``organization``, ``project``.

    Examples:
        >>> from pinecone.models.admin.role_binding import ResourceType
        >>> ResourceType.PROJECT == "project"
        True
    """

    ORGANIZATION = "organization"
    PROJECT = "project"


class RoleName(str, Enum):
    """A role that can be assigned to a principal at a resource scope.

    Organization-scoped roles: ``OrgOwner``, ``OrgManager``, ``OrgMember``,
    ``OrgBillingAdmin``. Project-scoped roles: ``ProjectOwner``,
    ``ProjectManager``, ``ProjectMember``, ``ProjectEditor``,
    ``ProjectViewer``, ``ControlPlaneEditor``, ``ControlPlaneViewer``,
    ``DataPlaneEditor``, ``DataPlaneViewer``.

    Membership in this enum only means the SDK will forward the value. Which of
    these roles may be bound to which scope and principal type, and which the
    organization's plan includes, is the server's decision and is reported as
    :exc:`~pinecone.errors.exceptions.ForbiddenError` at bind time.

    Examples:
        >>> from pinecone.models.admin.role_binding import RoleName
        >>> RoleName.DATA_PLANE_EDITOR == "DataPlaneEditor"
        True
    """

    ORG_OWNER = "OrgOwner"
    ORG_MANAGER = "OrgManager"
    ORG_MEMBER = "OrgMember"
    ORG_BILLING_ADMIN = "OrgBillingAdmin"
    PROJECT_OWNER = "ProjectOwner"
    PROJECT_MANAGER = "ProjectManager"
    PROJECT_MEMBER = "ProjectMember"
    PROJECT_EDITOR = "ProjectEditor"
    PROJECT_VIEWER = "ProjectViewer"
    CONTROL_PLANE_EDITOR = "ControlPlaneEditor"
    CONTROL_PLANE_VIEWER = "ControlPlaneViewer"
    DATA_PLANE_EDITOR = "DataPlaneEditor"
    DATA_PLANE_VIEWER = "DataPlaneViewer"


_VALID_RESOURCE_TYPES = [r.value for r in ResourceType]
_VALID_ROLE_NAMES = [r.value for r in RoleName]


def _as_str(value: str) -> str:
    return value.value if isinstance(value, Enum) else value


class RoleBindingModel(StructDictMixin, Struct, kw_only=True):
    """Response model for a role binding: a ``role`` granted to a principal at a scope.

    ``principal_type``, ``resource_type``, and ``role`` are typed as
    :class:`str` rather than as enums so values the server adds after this SDK
    release surface as their raw strings instead of raising. Compare against
    :class:`PrincipalType`, :class:`ResourceType`, and :class:`RoleName`
    directly — they are ``str`` values.

    Attributes:
        id (str): Unique identifier (UUID) for the role binding. This is what
            :meth:`RoleBindings.delete()
            <pinecone.admin.role_bindings.RoleBindings.delete>` takes — revoking a role is
            addressed by the binding, never by the principal/scope/role triple.
        principal_type (str): One of the :class:`PrincipalType` values.
        principal_id (str): The principal's UUID.
        resource_type (str): One of the :class:`ResourceType` values.
        resource_id (str): The organization or project the binding is scoped to.
            Always populated, including on organization-scoped bindings whose
            create request omitted it.
        role (str): One of the :class:`RoleName` values.
        created_at (str): RFC 3339 timestamp for when the binding was created.
            There is no updated timestamp: bindings are immutable, so a role
            change is a create plus a delete rather than an edit.

    Examples:
        >>> from pinecone.models.admin.role_binding import RoleBindingModel, RoleName
        >>> binding = RoleBindingModel(
        ...     id="9a8e3528-b9c0-4358-84ce-84c28e91b566",
        ...     principal_type="service_account",
        ...     principal_id="f8a3b2c1-4d5e-6f7a-8b9c-0d1e2f3a4b5c",
        ...     resource_type="project",
        ...     resource_id="a2f7dddb-1597-4eff-9f71-535fde243f58",
        ...     role="DataPlaneEditor",
        ...     created_at="2026-04-10T15:23:00Z",
        ... )
        >>> binding.role == RoleName.DATA_PLANE_EDITOR
        True
    """

    id: str
    principal_type: str
    principal_id: str
    resource_type: str
    resource_id: str
    role: str
    created_at: str


class RoleBindingInput(StructDictMixin, Struct, kw_only=True, omit_defaults=True):
    """A role to grant when creating an invite or a service account.

    Unlike the response models, this is an input the SDK sends, so
    ``resource_type`` and ``role`` are validated on construction against the
    values this SDK release knows about.

    ``resource_type`` selects the binding scope. For ``organization`` scope,
    omit ``resource_id`` — the binding applies to the organization inferred
    from the request context. For ``project`` scope, ``resource_id`` is
    required and must be the project UUID.

    Attributes:
        resource_type (str): One of the :class:`ResourceType` values.
        role (str): One of the :class:`RoleName` values.
        resource_id (str | None): The project UUID for ``project`` scope; leave
            unset for ``organization`` scope.

    Raises:
        :exc:`~pinecone.errors.exceptions.PineconeValueError`: If
            ``resource_type`` or ``role`` is not a recognized value, or if
            ``resource_type`` is ``project`` and ``resource_id`` is missing or
            empty. Raised at construction, so a malformed binding fails before
            the call that would have sent it.

    Examples:
        >>> from pinecone.models.admin.role_binding import (
        ...     ResourceType,
        ...     RoleBindingInput,
        ...     RoleName,
        ... )
        >>> RoleBindingInput(
        ...     resource_type=ResourceType.ORGANIZATION, role=RoleName.ORG_MEMBER
        ... ).to_dict()
        {'resource_type': 'organization', 'role': 'OrgMember', 'resource_id': None}

        Project-scoped bindings need the project UUID:

        >>> RoleBindingInput(
        ...     resource_type="project",
        ...     role="ProjectViewer",
        ...     resource_id="a2f7dddb-1597-4eff-9f71-535fde243f58",
        ... ).resource_id
        'a2f7dddb-1597-4eff-9f71-535fde243f58'

    .. seealso::
       - :class:`RoleBindingModel` — what the server returns. This input type
         names only the scope and the role; the response adds the binding's own
         ``id`` and the principal.
       - :meth:`RoleBindings.create()
         <pinecone.admin.role_bindings.RoleBindings.create>` — grants a role to an
         existing principal, taking the same fields as keyword arguments rather
         than as this struct.
    """

    resource_type: str
    role: str
    resource_id: str | None = None

    def __post_init__(self) -> None:
        self.resource_type = _as_str(self.resource_type)
        self.role = _as_str(self.role)
        require_one_of("resource_type", self.resource_type, _VALID_RESOURCE_TYPES)
        require_one_of("role", self.role, _VALID_ROLE_NAMES)
        if self.resource_type == ResourceType.PROJECT.value and not self.resource_id:
            raise ValidationError(
                "resource_id is required when resource_type is 'project' "
                "(it must be the project UUID)"
            )


class RoleBindingList(Struct, kw_only=True):
    """A page of role bindings, plus the cursor for the next page.

    One raw page of a role-binding listing. Callers who reach bindings through
    :meth:`RoleBindings.list() <pinecone.admin.role_bindings.RoleBindings.list>` get a
    :class:`~pinecone.models.pagination.Paginator` instead, which follows these
    cursors for them.

    Attributes:
        data (list[RoleBindingModel]): The role bindings on this page.
        pagination (PaginationResponse | None): Cursor envelope for the next
            page, or ``None`` on the final page.

    Examples:
        >>> from pinecone.models.admin.role_binding import RoleBindingList, RoleBindingModel
        >>> bindings = RoleBindingList(
        ...     data=[
        ...         RoleBindingModel(
        ...             id="9a8e3528-b9c0-4358-84ce-84c28e91b566",
        ...             principal_type="user",
        ...             principal_id="e2e92523-85dc-4142-b8c2-e681be8b78df",
        ...             resource_type="organization",
        ...             resource_id="4f6a1e0c-8f2b-4c1a-9d3e-1b2c3d4e5f60",
        ...             role="OrgMember",
        ...             created_at="2026-04-10T15:23:00Z",
        ...         )
        ...     ]
        ... )
        >>> bindings.roles()
        ['OrgMember']
    """

    data: list[RoleBindingModel] = []
    pagination: PaginationResponse | None = None

    def __iter__(self) -> Iterator[RoleBindingModel]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> RoleBindingModel:
        return self.data[index]

    @property
    def pagination_token(self) -> str | None:
        """Opaque cursor for the next page, or ``None`` if this is the last page."""
        return self.pagination.next if self.pagination is not None else None

    @property
    def has_more(self) -> bool:
        """``True`` when the server supplied a cursor for a further page."""
        return self.pagination_token is not None

    def roles(self) -> list[str]:
        """Return the role names on this page, in order."""
        return [binding.role for binding in self.data]

    def to_dict(self) -> dict[str, Any]:
        """Return this page as a serializable dict with ``data`` and ``pagination`` keys."""
        return {
            "data": [_struct_to_dict_recursive(binding) for binding in self.data],
            "pagination": _struct_to_dict_recursive(self.pagination),
        }

    def __repr__(self) -> str:
        summaries = ", ".join(
            f"<principal_type={b.principal_type!r}, principal_id={b.principal_id!r}, "
            f"role={b.role!r}>"
            for b in self.data
        )
        return f"RoleBindingList([{summaries}], has_more={self.has_more!r})"
