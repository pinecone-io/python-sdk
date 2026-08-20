"""Invites namespace — list, create, describe, delete, and resend organization invites."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from pinecone._internal.adapters.admin_adapter import AdminAdapter
from pinecone._internal.validation import require_in_range, require_non_empty
from pinecone.errors.exceptions import ValidationError
from pinecone.models.admin.invite import InviteModel
from pinecone.models.admin.role_binding import RoleBindingInput
from pinecone.models.pagination import Page, Paginator

if TYPE_CHECKING:
    from pinecone._internal.http_client import HTTPClient

logger = logging.getLogger(__name__)

_LIMIT_MIN = 1
_LIMIT_MAX = 100

_REQUIRED_BINDING_KEYS = ("resource_type", "role")
_KNOWN_BINDING_KEYS = frozenset({"resource_type", "resource_id", "role"})


def _binding_to_payload(binding: RoleBindingInput) -> dict[str, str]:
    """Render one validated binding as the wire object the spec declares.

    ``resource_id`` is omitted rather than sent as ``null`` when unset, so an
    ``organization``-scoped binding matches the spec's "omit for organization
    scope" exactly, and so a binding built from an enum and the same binding
    built from plain strings produce byte-identical bodies.
    """
    payload = {"resource_type": binding.resource_type, "role": binding.role}
    if binding.resource_id is not None:
        payload["resource_id"] = binding.resource_id
    return payload


def _normalize_role_bindings(
    role_bindings: Sequence[RoleBindingInput | Mapping[str, Any]],
) -> list[dict[str, str]]:
    """Validate every entry and render the ``role_bindings`` array for the wire.

    Accepts :class:`~pinecone.models.admin.role_binding.RoleBindingInput`
    instances and plain dicts interchangeably. Every failure names the index of
    the offending entry, because the server's own 400 cannot say which one it
    tripped over.
    """
    normalized: list[dict[str, str]] = []
    for index, entry in enumerate(role_bindings):
        if isinstance(entry, RoleBindingInput):
            normalized.append(_binding_to_payload(entry))
            continue
        if not isinstance(entry, Mapping):
            raise ValidationError(
                f"role_bindings[{index}] must be a RoleBindingInput or a dict with "
                f"'resource_type' and 'role' keys, got {type(entry).__name__}"
            )
        for key in _REQUIRED_BINDING_KEYS:
            value = entry.get(key)
            if value is None or (isinstance(value, str) and not value.strip()):
                raise ValidationError(
                    f"role_bindings[{index}] missing required key {key!r}; every entry "
                    "needs 'resource_type' ('organization' or 'project') and 'role', "
                    "plus 'resource_id' for 'project' scope"
                )
        unknown = sorted(set(entry) - _KNOWN_BINDING_KEYS)
        if unknown:
            opts = ", ".join(repr(k) for k in unknown)
            raise ValidationError(
                f"role_bindings[{index}] has unrecognized key(s) {opts}; allowed keys are "
                "'resource_type', 'role', and 'resource_id'"
            )
        try:
            binding = RoleBindingInput(
                resource_type=entry["resource_type"],
                role=entry["role"],
                resource_id=entry.get("resource_id"),
            )
        except ValidationError as exc:
            raise ValidationError(f"role_bindings[{index}]: {exc}") from exc
        normalized.append(_binding_to_payload(binding))
    return normalized


class Invites:
    """Control-plane operations for organization invites.

    Provides methods to list, create, describe, delete, and resend the invites
    of the organization associated with the :class:`~pinecone.Admin` client's
    OAuth credentials.

    An invite's role bindings are first-class objects rather than part of the
    invite's representation: none of the methods here return them. Create sends
    them, and the role-binding operations read or change them afterwards
    (filtering on ``principal_type=invite``).

    Args:
        http (HTTPClient): HTTP client for making API requests.

    Examples:
        >>> from pinecone import Admin
        >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
        >>> for invite in admin.invites.list():
        ...     print(invite.email, invite.status)
    """

    def __init__(self, *, http: HTTPClient) -> None:
        self._http = http
        self._adapter = AdminAdapter()

    def __repr__(self) -> str:
        """Return developer-friendly representation."""
        return "Invites()"

    def list(
        self,
        *,
        limit: int | None = None,
        pagination_token: str | None = None,
    ) -> Paginator[InviteModel]:
        """List the organization's pending and expired invites, with lazy pagination.

        .. warning::
            This endpoint **omits processed invites** — the ones that have
            already been accepted. An invite missing from this listing has not
            necessarily vanished; it may have been accepted, in which case
            :meth:`describe` still returns it with
            ``status == InviteStatus.PROCESSED``, and the accepted invitee is
            now a member reachable through ``admin.users``. Do not treat
            absence here as proof an invite never existed.

        No request is sent until the returned paginator is iterated. Iterating
        past the first page reuses the cursor from the previous response's
        ``pagination.next`` verbatim; iteration stops on the first page that
        comes back without one.

        Args:
            limit (int | None): Number of invites the server returns **per
                page**, between 1 and 100. This is the spec's ``limit`` query
                parameter, not a cap on how many invites the paginator yields
                in total; the paginator keeps following cursors until the pages
                run out. Use :func:`itertools.islice` to cap the total. When
                ``None`` the parameter is omitted and the server defaults
                to 100.
            pagination_token (str | None): Cursor from a prior response's
                ``pagination.next``, to resume where a previous iteration
                stopped. Reuse it with the same ``limit``.

        Returns:
            :class:`~pinecone.models.pagination.Paginator` over
            :class:`~pinecone.models.admin.invite.InviteModel` objects.
            Supports ``for`` loops, ``.to_list()``, ``.pages()`` for page-level
            access, and ``.pagination_token`` for resumption.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`:
                If *limit* is outside 1-100. Raised before any network call.
            :exc:`ApiError`: If the API returns an error response.

        Examples:
            .. code-block:: python

                for invite in admin.invites.list():
                    print(invite.id, invite.email, invite.status)

                for page in admin.invites.list(limit=25).pages():
                    print(len(page.items), page.pagination_token)
        """
        if limit is not None:
            require_in_range("limit", limit, _LIMIT_MIN, _LIMIT_MAX)

        logger.info("Listing invites (limit=%r)", limit)

        def fetch_page(token: str | None) -> Page[InviteModel]:
            params: dict[str, str | int] = {}
            if limit is not None:
                params["limit"] = limit
            if token is not None:
                params["paginationToken"] = token
            response = self._http.get("/admin/invites", params=params)
            result = self._adapter.to_invite_list(response.content)
            logger.debug("Listed %d invites (has_more=%s)", len(result), result.has_more)
            return Page(items=result.data, pagination_token=result.pagination_token)

        return Paginator(fetch_page=fetch_page, initial_token=pagination_token)

    def create(
        self,
        *,
        email: str,
        role_bindings: Sequence[RoleBindingInput | Mapping[str, Any]],
    ) -> InviteModel:
        """Invite a user to the organization and grant their initial role bindings.

        On success the server has already sent the invite email; the returned
        invite is ``pending`` and expires seven days out by default. The
        response does **not** echo the role bindings — read them back through
        the role-binding operations, filtering on ``principal_type=invite``.

        Args:
            email (str): The email address to invite. Sent verbatim in the
                request body — the SDK checks only that it is non-empty and
                leaves address validity to the server, which rejects a
                malformed or over-long address with ``400 INVALID_ARGUMENT``.
            role_bindings (Sequence[RoleBindingInput | Mapping[str, Any]]):
                The roles to grant the invitee, as
                :class:`~pinecone.models.admin.role_binding.RoleBindingInput`
                instances or plain dicts, mixed freely. Each entry needs
                ``resource_type`` (``"organization"`` or ``"project"``) and
                ``role``; ``project`` scope additionally needs ``resource_id``,
                the project UUID. At least one entry is required, and the
                server requires at least one of them to be an
                ``organization``-scoped membership role (``OrgOwner``,
                ``OrgManager``, ``OrgBillingAdmin``, or ``OrgMember``).

        Returns:
            The created :class:`~pinecone.models.admin.invite.InviteModel`.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`:
                If *email* is empty, if *role_bindings* is empty, or if any
                entry is missing ``resource_type``/``role``, carries an
                unrecognized key, or names a value this SDK release does not
                know. The message names the index of the offending entry.
                Raised before any network call.
            :exc:`~pinecone.errors.exceptions.ConflictError`:
                If a pending invite already exists for the address, or the
                address already belongs to an organization member (409).
            :exc:`ApiError`: If the API returns an error response.

        Examples:
            >>> from pinecone import Admin
            >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
            >>> invite = admin.invites.create(  # doctest: +SKIP
            ...     email="newhire@acme.com",
            ...     role_bindings=[{"resource_type": "organization", "role": "OrgMember"}],
            ... )

            Typed inputs and dicts are interchangeable, and may be mixed:

            .. code-block:: python

                from pinecone.models.admin import ResourceType, RoleBindingInput, RoleName

                admin.invites.create(
                    email="newhire@acme.com",
                    role_bindings=[
                        RoleBindingInput(
                            resource_type=ResourceType.ORGANIZATION,
                            role=RoleName.ORG_MEMBER,
                        ),
                        {
                            "resource_type": "project",
                            "role": "ProjectViewer",
                            "resource_id": "a2f7dddb-1597-4eff-9f71-535fde243f58",
                        },
                    ],
                )
        """
        require_non_empty("email", email)
        bindings = list(role_bindings)
        if not bindings:
            raise ValidationError(
                "role_bindings must be a non-empty list; an invite needs at least one "
                "role binding, including an 'organization'-scoped membership role"
            )
        body: dict[str, Any] = {"email": email, "role_bindings": _normalize_role_bindings(bindings)}

        logger.info("Creating invite (bindings=%d)", len(bindings))
        response = self._http.post("/admin/invites", json=body)
        result = self._adapter.to_invite(response.content)
        logger.debug("Created invite %r", result.id)
        return result

    def describe(self, *, invite_id: str) -> InviteModel:
        """Get detailed information about one invite, whatever its status.

        Unlike :meth:`list`, this reaches processed invites too — it is the only
        operation that can return ``status == InviteStatus.PROCESSED``.

        Args:
            invite_id (str): The identifier of the invite.

        Returns:
            An :class:`~pinecone.models.admin.invite.InviteModel` with the
            invite's details.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`: If *invite_id* is empty.
            :exc:`~pinecone.errors.exceptions.NotFoundError`:
                If no such invite exists in the organization (404). A deleted
                invite reads back as 404, not as a status.
            :exc:`ApiError`: If the API returns an error response.

        Examples:
            >>> from pinecone import Admin
            >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
            >>> invite = admin.invites.describe(  # doctest: +SKIP
            ...     invite_id="9c8e3528-b9c0-4358-84ce-84c28e91b566"
            ... )
        """
        require_non_empty("invite_id", invite_id)
        logger.info("Describing invite %r", invite_id)
        response = self._http.get(f"/admin/invites/{invite_id}")
        result = self._adapter.to_invite(response.content)
        logger.debug("Described invite %r", invite_id)
        return result

    def delete(self, *, invite_id: str) -> None:
        """Delete a pending or expired invite, along with its role bindings.

        The server answers ``202`` with no body; the invite and its role
        bindings are already gone, after which the invite reads back as ``404``
        — including for a repeat of this call. An invite that has already been
        accepted cannot be deleted this way: remove the resulting member with
        ``admin.users.delete`` instead.

        Args:
            invite_id (str): The identifier of the invite to delete.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`: If *invite_id* is empty.
            :exc:`~pinecone.errors.exceptions.NotFoundError`:
                If no such invite exists in the organization (404).
            :exc:`~pinecone.errors.exceptions.ConflictError`:
                If the invite has already been processed (409). The server's
                error code and message are carried through verbatim.
            :exc:`ApiError`: If the API returns an error response.

        Examples:
            >>> from pinecone import Admin
            >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
            >>> admin.invites.delete(  # doctest: +SKIP
            ...     invite_id="9c8e3528-b9c0-4358-84ce-84c28e91b566"
            ... )
        """
        require_non_empty("invite_id", invite_id)
        logger.info("Deleting invite %r", invite_id)
        self._http.delete(f"/admin/invites/{invite_id}")
        logger.debug("Deleted invite %r", invite_id)

    def resend(self, *, invite_id: str) -> InviteModel:
        """Resend an invite's email and extend its expiration to seven days out.

        Works on pending and expired invites alike: the returned invite is
        ``pending`` again with a fresh ``expires_at``.

        .. warning::
            Invite emails are **rate limited to 100 per hour per
            organization**. Over that ceiling this raises
            :exc:`~pinecone.errors.exceptions.RateLimitError` (429) — do not
            retry in a tight loop. Honor ``exc.retry_after`` when the server
            supplies a ``Retry-After`` header, and back off well beyond a
            second otherwise; the budget refills over an hour, not
            milliseconds. A ``409`` is not a rate limit and never becomes
            retryable: it means the invite was already accepted, so there is
            nothing left to resend.

        Args:
            invite_id (str): The identifier of the invite to resend.

        Returns:
            The updated :class:`~pinecone.models.admin.invite.InviteModel`,
            with ``status`` back to ``pending`` and ``expires_at`` moved out
            seven days.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`: If *invite_id* is empty.
            :exc:`~pinecone.errors.exceptions.NotFoundError`:
                If no such invite exists in the organization (404).
            :exc:`~pinecone.errors.exceptions.ConflictError`:
                If the invite has already been accepted and so cannot be
                resent (409).
            :exc:`~pinecone.errors.exceptions.RateLimitError`:
                If the organization's invite-email budget is exhausted (429).
                ``retry_after`` carries the server's ``Retry-After`` when it
                sends a numeric one.
            :exc:`ApiError`: If the API returns an error response.

        Examples:
            >>> from pinecone import Admin
            >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
            >>> invite = admin.invites.resend(  # doctest: +SKIP
            ...     invite_id="9c8e3528-b9c0-4358-84ce-84c28e91b566"
            ... )
        """
        require_non_empty("invite_id", invite_id)
        logger.info("Resending invite %r", invite_id)
        response = self._http.post(f"/admin/invites/{invite_id}/resend")
        result = self._adapter.to_invite(response.content)
        logger.debug("Resent invite %r", invite_id)
        return result
