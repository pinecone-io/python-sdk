"""RoleBindings namespace — list, create, describe, delete."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pinecone._internal.adapters.admin_adapter import AdminAdapter
from pinecone._internal.role_bindings import binding_to_payload
from pinecone._internal.validation import require_in_range, require_non_empty, require_one_of
from pinecone.errors.exceptions import ValidationError
from pinecone.models.admin.role_binding import (
    PrincipalType,
    ResourceType,
    RoleBindingInput,
    RoleBindingModel,
    RoleName,
)
from pinecone.models.pagination import Page, Paginator

if TYPE_CHECKING:
    from pinecone._internal.http_client import HTTPClient

logger = logging.getLogger(__name__)

_LIMIT_MIN = 1
_LIMIT_MAX = 100

_VALID_PRINCIPAL_TYPES = [p.value for p in PrincipalType]
_VALID_RESOURCE_TYPES = [r.value for r in ResourceType]
_VALID_ROLE_NAMES = [r.value for r in RoleName]


def _as_str(value: str | PrincipalType | ResourceType | RoleName) -> str:
    """Accept an enum member or a plain string and return the wire string."""
    return value.value if isinstance(value, (PrincipalType, ResourceType, RoleName)) else value


class RoleBindings:
    """Control-plane operations for the organization's role bindings.

    A role binding is the whole of Pinecone's authorization model: it grants one
    ``role`` to one principal — a user, service account, API key, or pending
    invite — at one scope, either the organization or a single project. Nothing
    else confers permissions, so this namespace is where a principal's access is
    read and changed. The other admin namespaces deliberately do not carry role
    bindings in their models; :meth:`list` with ``principal_type`` and
    ``principal_id`` is how a principal's access is enumerated.

    Bindings are immutable: there is no update. Changing a principal's role means
    :meth:`create` for the new one and :meth:`delete` for the old one, in that
    order — deleting first can strip the principal's last organization-membership
    binding, which the server refuses with ``409``.

    The server owns which role may be bound to which scope and principal type,
    and which roles an organization's plan includes. Those rules vary by plan, so
    the SDK does not replicate them: it checks only that a value is one this
    release knows about and that the filter co-requirements hold, and lets the
    server's own ``403``/``400`` messages — which name the role, the scope, and
    the plan — explain the rest verbatim.

    Args:
        http (HTTPClient): HTTP client for making API requests.

    Examples:
        >>> from pinecone import Admin
        >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
        >>> for binding in admin.role_bindings.list():
        ...     print(binding.principal_id, binding.role, binding.resource_id)
    """

    def __init__(self, *, http: HTTPClient) -> None:
        self._http = http
        self._adapter = AdminAdapter()

    def __repr__(self) -> str:
        """Return developer-friendly representation."""
        return "RoleBindings()"

    def list(
        self,
        *,
        principal_type: str | PrincipalType | None = None,
        principal_id: str | None = None,
        resource_type: str | ResourceType | None = None,
        resource_id: str | None = None,
        role: str | RoleName | None = None,
        limit: int | None = None,
        pagination_token: str | None = None,
    ) -> Paginator[RoleBindingModel]:
        """List the organization's role bindings, with lazy pagination.

        Every supplied filter is combined with **AND**, so
        ``list(principal_type="user", role="OrgOwner")`` returns the bindings
        that are both. With no filters at all it walks every binding the caller
        is allowed to see, which for an org owner is the organization's entire
        authorization state.

        No request is sent until the returned paginator is iterated. Iterating
        past the first page reuses the cursor from the previous response's
        ``pagination.next`` verbatim; iteration stops on the first page that
        comes back without one. The filters and ``limit`` are carried onto every
        later page, because the server requires a cursor to be replayed with the
        query context that produced it.

        Args:
            principal_type (str | PrincipalType | None): Restrict to one kind of
                principal — ``"user"``, ``"service_account"``, ``"api_key"``, or
                ``"invite"``. Required whenever *principal_id* is given, since an
                ID alone is ambiguous across principal kinds. Omitted when
                ``None``.
            principal_id (str | None): Restrict to one principal's bindings — a
                UUID for every principal type. Requires *principal_type*. Sent
                verbatim; the server owns the format and answers an unparseable
                value with ``400``. Omitted when ``None``.
            resource_type (str | ResourceType | None): Restrict to one scope kind
                — ``"organization"`` or ``"project"``. Required whenever
                *resource_id* is given. Omitted when ``None``.
            resource_id (str | None): Restrict to one organization or project.
                Requires *resource_type*. Omitted when ``None``.
            role (str | RoleName | None): Restrict to one role, spelled as the
                wire name (``"ProjectOwner"``, not ``"project_owner"``).
                :class:`~pinecone.models.admin.role_binding.RoleName` members are
                accepted interchangeably. Omitted when ``None``.
            limit (int | None): Number of bindings the server returns **per
                page**, between 1 and 100. It caps each page, not how many
                bindings the paginator yields in total; the paginator keeps
                following cursors until the pages run out. Use
                :func:`itertools.islice` to cap the total. When ``None`` the
                parameter is omitted and the server chooses the page size.
            pagination_token (str | None): Cursor from a prior response's
                ``pagination.next``, to resume where a previous iteration
                stopped. Reuse it with the same filters and ``limit``.

        Returns:
            :class:`~pinecone.models.pagination.Paginator` over
            :class:`~pinecone.models.admin.role_binding.RoleBindingModel`
            objects. Supports ``for`` loops, ``.to_list()``, ``.pages()`` for
            page-level access, and ``.pagination_token`` for resumption.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`:
                If *principal_id* is given without *principal_type*, or
                *resource_id* without *resource_type*; if *principal_type*,
                *resource_type*, or *role* names a value this SDK release does
                not know; or if *limit* is outside 1-100. Raised before any
                network call.
            :exc:`ApiError`: If the API returns an error response.

        Examples:
            .. code-block:: python

                for binding in admin.role_bindings.list():
                    print(binding.id, binding.principal_id, binding.role)

                everything_one_service_account_can_do = admin.role_bindings.list(
                    principal_type="service_account",
                    principal_id="f8a3b2c1-4d5e-6f7a-8b9c-0d1e2f3a4b5c",
                ).to_list()

                project_owners = admin.role_bindings.list(
                    resource_type="project",
                    resource_id="a2f7dddb-1597-4eff-9f71-535fde243f58",
                    role="ProjectOwner",
                ).to_list()

                for page in admin.role_bindings.list(limit=25).pages():
                    print(len(page.items), page.pagination_token)
        """
        if principal_id is not None and principal_type is None:
            raise ValidationError(
                "principal_id requires principal_type: a principal ID is ambiguous on its "
                "own, so the server rejects the pair. Pass principal_type as well, one of "
                f"{', '.join(repr(p) for p in _VALID_PRINCIPAL_TYPES)}."
            )
        if resource_id is not None and resource_type is None:
            raise ValidationError(
                "resource_id requires resource_type: a resource ID is ambiguous on its own, "
                "so the server rejects the pair. Pass resource_type as well, one of "
                f"{', '.join(repr(r) for r in _VALID_RESOURCE_TYPES)}."
            )

        principal_type_value = None if principal_type is None else _as_str(principal_type)
        resource_type_value = None if resource_type is None else _as_str(resource_type)
        role_value = None if role is None else _as_str(role)

        if principal_type_value is not None:
            require_one_of("principal_type", principal_type_value, _VALID_PRINCIPAL_TYPES)
        if resource_type_value is not None:
            require_one_of("resource_type", resource_type_value, _VALID_RESOURCE_TYPES)
        if role_value is not None:
            require_one_of("role", role_value, _VALID_ROLE_NAMES)
        if limit is not None:
            require_in_range("limit", limit, _LIMIT_MIN, _LIMIT_MAX)

        filters: dict[str, str] = {}
        if principal_type_value is not None:
            filters["principal_type"] = principal_type_value
        if principal_id is not None:
            filters["principal_id"] = principal_id
        if resource_type_value is not None:
            filters["resource_type"] = resource_type_value
        if resource_id is not None:
            filters["resource_id"] = resource_id
        if role_value is not None:
            filters["role"] = role_value

        logger.info("Listing role bindings (filters=%r, limit=%r)", sorted(filters), limit)

        def fetch_page(token: str | None) -> Page[RoleBindingModel]:
            params: dict[str, str | int] = dict(filters)
            if limit is not None:
                params["limit"] = limit
            if token is not None:
                params["paginationToken"] = token
            response = self._http.get("/admin/role-bindings", params=params)
            result = self._adapter.to_role_binding_list(response.content)
            logger.debug("Listed %d role bindings (has_more=%s)", len(result), result.has_more)
            return Page(items=result.data, pagination_token=result.pagination_token)

        return Paginator(fetch_page=fetch_page, initial_token=pagination_token)

    def create(
        self,
        *,
        principal_type: str | PrincipalType,
        principal_id: str,
        resource_type: str | ResourceType,
        role: str | RoleName,
        resource_id: str | None = None,
    ) -> RoleBindingModel:
        """Grant a role to a principal at an organization or project scope.

        The binding takes effect immediately and is returned with the ``id``
        :meth:`delete` needs — the only way to revoke it, since bindings cannot
        be edited in place.

        The same scope-and-role pair is accepted as an initial binding by
        :meth:`~pinecone.admin.invites.Invites.create` and
        :meth:`~pinecone.admin.service_accounts.ServiceAccounts.create`, so a
        grant expressed once works in all three places.

        Whether the grant is *allowed* is entirely the server's call, and it
        refuses for several distinct reasons the SDK cannot tell apart in
        advance: a project-scoped binding must name a project-scoped role, an
        ``api_key`` principal accepts only the data/control-plane roles, some
        roles are gated behind the organization's plan, and the caller cannot
        grant a permission it does not itself hold. Each rejection carries a
        message naming the role, the scope, and — for plan gating — the plan
        required, so read the error rather than pre-flighting the rules.

        Args:
            principal_type (str | PrincipalType): The kind of principal receiving
                the role — ``"user"``, ``"service_account"``, ``"api_key"``, or
                ``"invite"``. Binding to an ``invite`` grants the role to whoever
                accepts it; once accepted the server refuses further bindings on
                the invite (``409``) and the roles must be managed on the
                resulting user instead.
            principal_id (str): The principal's UUID. Sent verbatim — the server
                owns the format and answers an unknown or unparseable principal
                with ``404``.
            resource_type (str | ResourceType): The scope — ``"organization"`` or
                ``"project"``.
            role (str | RoleName): The role to grant, spelled as the wire name
                (``"DataPlaneEditor"``).
                :class:`~pinecone.models.admin.role_binding.RoleName` members are
                accepted interchangeably.
            resource_id (str | None): The project UUID. Required when
                *resource_type* is ``"project"``. For ``"organization"`` scope
                leave it unset — the organization is inferred from the
                credentials, and passing any organization other than the caller's
                own is a ``404``.

        Returns:
            The created
            :class:`~pinecone.models.admin.role_binding.RoleBindingModel`, whose
            ``resource_id`` is always populated: an organization-scoped binding
            comes back carrying the organization the credentials resolved to,
            even though the request omitted it.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`:
                If *principal_id* is empty; if *principal_type*, *resource_type*,
                or *role* names a value this SDK release does not know; or if
                *resource_type* is ``"project"`` and *resource_id* is missing.
                Raised before any network call.
            :exc:`~pinecone.errors.exceptions.NotFoundError`:
                If the principal or the resource does not exist in the caller's
                organization (404).
            :exc:`~pinecone.errors.exceptions.ConflictError`:
                If an identical binding already exists, or the principal is an
                invite that has already been accepted (409).
            :exc:`~pinecone.errors.exceptions.ForbiddenError`:
                If the role cannot be bound to that scope or principal type, the
                organization's plan does not include it, or the caller would be
                granting a permission it does not hold (403).
            :exc:`ApiError`: If the API returns an error response.

        Examples:
            >>> from pinecone import Admin
            >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
            >>> binding = admin.role_bindings.create(  # doctest: +SKIP
            ...     principal_type="user",
            ...     principal_id="e2e92523-85dc-4142-b8c2-e681be8b78df",
            ...     resource_type="organization",
            ...     role="OrgMember",
            ... )

            A project-scoped grant, with enums:

            .. code-block:: python

                from pinecone.models.admin import PrincipalType, ResourceType, RoleName

                binding = admin.role_bindings.create(
                    principal_type=PrincipalType.SERVICE_ACCOUNT,
                    principal_id="f8a3b2c1-4d5e-6f7a-8b9c-0d1e2f3a4b5c",
                    resource_type=ResourceType.PROJECT,
                    resource_id="a2f7dddb-1597-4eff-9f71-535fde243f58",
                    role=RoleName.DATA_PLANE_EDITOR,
                )
        """
        principal_type_value = _as_str(principal_type)
        require_one_of("principal_type", principal_type_value, _VALID_PRINCIPAL_TYPES)
        require_non_empty("principal_id", principal_id)

        scope = RoleBindingInput(resource_type=resource_type, role=role, resource_id=resource_id)
        body: dict[str, str] = {
            "principal_type": principal_type_value,
            "principal_id": principal_id,
            **binding_to_payload(scope),
        }

        logger.info(
            "Creating role binding (principal_type=%r, resource_type=%r, role=%r)",
            principal_type_value,
            scope.resource_type,
            scope.role,
        )
        response = self._http.post("/admin/role-bindings", json=body)
        result = self._adapter.to_role_binding(response.content)
        logger.debug("Created role binding %r", result.id)
        return result

    def describe(self, *, role_binding_id: str) -> RoleBindingModel:
        """Get detailed information about one role binding.

        Args:
            role_binding_id (str): The identifier of the role binding.

        Returns:
            A :class:`~pinecone.models.admin.role_binding.RoleBindingModel` with
            the principal, the scope, the role, and when it was granted.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`:
                If *role_binding_id* is empty.
            :exc:`~pinecone.errors.exceptions.NotFoundError`:
                If no such role binding is visible to the caller (404). A binding
                in another organization, and a project binding the caller cannot
                see, both read as 404 rather than 403 — absence and
                inaccessibility are deliberately indistinguishable here.
            :exc:`ApiError`: If the API returns an error response.

        Examples:
            >>> from pinecone import Admin
            >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
            >>> binding = admin.role_bindings.describe(  # doctest: +SKIP
            ...     role_binding_id="9a8e3528-b9c0-4358-84ce-84c28e91b566"
            ... )
        """
        require_non_empty("role_binding_id", role_binding_id)
        logger.info("Describing role binding %r", role_binding_id)
        response = self._http.get(f"/admin/role-bindings/{role_binding_id}")
        result = self._adapter.to_role_binding(response.content)
        logger.debug("Described role binding %r", role_binding_id)
        return result

    def delete(self, *, role_binding_id: str) -> None:
        """Revoke a role binding, by the binding's own ID.

        Deletion is addressed by ``role_binding_id`` rather than by the
        principal/scope/role triple, so revoking a role means finding the binding
        first — usually with :meth:`list` filtered by ``principal_type`` and
        ``principal_id``, or from the :meth:`create` result.

        The server answers ``202`` with no body; the permissions are already
        revoked, after which the binding reads back as ``404`` — including for a
        repeat of this call, so delete is not idempotent in the "second call also
        succeeds" sense.

        Some bindings cannot be deleted at all, and the refusal is a ``409``
        rather than a ``403``: the organization's last ``OrgOwner``, a user's last
        organization-membership binding while they still hold other roles, and a
        pending invite's last organization-membership binding (delete the invite
        instead). Organizations whose users are managed by an identity provider
        refuse user and invite binding changes outright, also with ``409``.

        Args:
            role_binding_id (str): The identifier of the role binding to delete.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`:
                If *role_binding_id* is empty.
            :exc:`~pinecone.errors.exceptions.NotFoundError`:
                If no such role binding is visible to the caller (404), including
                a repeat of a successful delete.
            :exc:`~pinecone.errors.exceptions.ConflictError`:
                If deleting the binding would strip the organization of its last
                owner, remove a principal's last organization membership, or the
                organization's user management is delegated to an identity
                provider (409).
            :exc:`ApiError`: If the API returns an error response.

        Examples:
            >>> from pinecone import Admin
            >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
            >>> admin.role_bindings.delete(  # doctest: +SKIP
            ...     role_binding_id="9a8e3528-b9c0-4358-84ce-84c28e91b566"
            ... )
        """
        require_non_empty("role_binding_id", role_binding_id)
        logger.info("Deleting role binding %r", role_binding_id)
        self._http.delete(f"/admin/role-bindings/{role_binding_id}")
        logger.debug("Deleted role binding %r", role_binding_id)
