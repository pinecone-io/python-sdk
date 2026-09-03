"""Invites namespace — list, create, describe, delete, and resend organization invites."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from pinecone._internal.adapters.admin_adapter import AdminAdapter
from pinecone._internal.role_bindings import normalize_role_bindings
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


class Invites:
    """Offers of organization membership that have not yet been accepted.

    An invite is an emailed offer for someone to join the organization that the
    :class:`~pinecone.Admin` client's OAuth credentials resolve to. It is a
    principal in its own right — roles can be bound to it before anyone accepts
    — and accepting it turns the recipient into a user. Not constructed directly
    — reach it as ``admin.invites``.

    An invite's role bindings are not part of its representation: :meth:`create`
    sends them, but no method here returns them. Read or change them through
    :class:`~pinecone.admin.role_bindings.RoleBindings` with
    ``principal_type="invite"``.

    See :doc:`/guides/error-handling` for the exceptions every operation here
    can raise.

    Examples:
        >>> from pinecone import Admin
        >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
        >>> for invite in admin.invites.list():
        ...     print(invite.email, invite.status)
        newhire@acme.com pending

    .. seealso::
       - :class:`~pinecone.admin.users.Users` — the same person after they
         accept. An invite and the user it produces are separate records with
         separate IDs, and only one of the two appears in each list.
       - :class:`~pinecone.admin.service_accounts.ServiceAccounts` — machine
         identities, which are created directly and never invited.
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

        Accepted invites are omitted, so absence from this list does not mean an
        invite never existed — see the note below. No request is sent until the
        returned paginator is iterated; see :doc:`/guides/pagination`.

        Args:
            limit (int | None): Number of invites the server returns **per
                page**. It caps each page, not how many invites the paginator
                yields in total; the paginator keeps following cursors until the
                pages run out, so use :func:`itertools.islice` to cap the total.
                When ``None`` the server chooses the page size.
            pagination_token (str | None): Cursor from a previous paginator's
                ``pagination_token``, to resume where that iteration stopped.
                Reuse it with the same ``limit``.

        Returns:
            :class:`~pinecone.models.pagination.Paginator` yielding
            :class:`~pinecone.models.admin.invite.InviteModel` objects.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`:
                If *limit* is outside 1-100. Raised before any network call.

        Examples:
            >>> from pinecone import Admin
            >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
            >>> for invite in admin.invites.list():
            ...     print(invite.email, invite.status)
            newhire@acme.com pending

            Page-level access exposes the cursor, which is ``None`` once there
            is no further page to fetch:

            >>> for page in admin.invites.list(limit=25).pages():
            ...     print(len(page.items), page.pagination_token)
            1 None

        .. note::
            An invite missing from this list may simply have been accepted.
            :meth:`describe` still returns it, with
            ``status == InviteStatus.PROCESSED``, and the accepted invitee is
            now a member reachable through
            :meth:`Users.list() <pinecone.admin.users.Users.list>`. To reconcile who has
            access, read both lists.

        .. seealso::
           - :meth:`Users.list() <pinecone.admin.users.Users.list>` — the members this list's
             invitees become once they accept.
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
        """Invite someone to the organization and grant their initial roles.

        The server has already sent the email by the time this returns; the
        invite comes back ``pending``, with ``expires_at`` set to when it lapses.
        The response does **not** echo the role bindings.

        Args:
            email (str): The address to invite, e.g. ``"newhire@acme.com"``. The
                SDK checks only that it isn't empty; the server validates the
                address itself and rejects a malformed or over-long one.
            role_bindings (Sequence[RoleBindingInput | Mapping[str, Any]]):
                The roles to grant the invitee, as
                :class:`~pinecone.models.admin.role_binding.RoleBindingInput`
                instances or plain dicts, mixed freely. Each entry needs
                ``resource_type`` (``"organization"`` or ``"project"``) and
                ``role``; ``project`` scope additionally needs ``resource_id``,
                the project UUID. At least one entry is required, and the server
                requires at least one of them to be an ``organization``-scoped
                membership role (``OrgOwner``, ``OrgManager``,
                ``OrgBillingAdmin``, or ``OrgMember``) — a project-only invite is
                rejected.

        Returns:
            The created :class:`~pinecone.models.admin.invite.InviteModel`, whose
            ``id`` is what :meth:`describe`, :meth:`resend`, and :meth:`delete`
            take, and what identifies the invite as a ``principal_id`` in
            role-binding queries.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`:
                If *email* is empty, if *role_bindings* is empty, or if any
                entry is missing ``resource_type``/``role``, carries an
                unrecognized key, or names a value this SDK release does not
                know. The message names the index of the offending entry.
                Raised before any network call.
            :exc:`~pinecone.errors.exceptions.ConflictError`:
                If a pending invite already exists for the address, or the
                address already belongs to a member. In the second case there is
                nothing to invite — manage the existing user's roles instead.

        Examples:
            >>> from pinecone import Admin
            >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
            >>> invite = admin.invites.create(
            ...     email="newhire@acme.com",
            ...     role_bindings=[{"resource_type": "organization", "role": "OrgMember"}],
            ... )
            >>> invite.status
            'pending'
            >>> invite.processed_at is None
            True

            The result is a pending principal, not a member: its own ``id`` is
            what :meth:`resend`, :meth:`delete`, and role-binding queries take,
            and the invitee stays absent from
            :meth:`Users.list() <pinecone.admin.users.Users.list>` until they accept.

            Typed inputs and dicts are interchangeable, and may be mixed:

            >>> from pinecone.models.admin import ResourceType, RoleBindingInput, RoleName
            >>> invite = admin.invites.create(
            ...     email="newhire@acme.com",
            ...     role_bindings=[
            ...         RoleBindingInput(
            ...             resource_type=ResourceType.ORGANIZATION,
            ...             role=RoleName.ORG_MEMBER,
            ...         ),
            ...         {
            ...             "resource_type": "project",
            ...             "role": "ProjectViewer",
            ...             "resource_id": "a2f7dddb-1597-4eff-9f71-535fde243f58",
            ...         },
            ...     ],
            ... )
            >>> invite.email
            'newhire@acme.com'

        .. seealso::
           - :meth:`RoleBindings.create()
             <pinecone.admin.role_bindings.RoleBindings.create>` — how to grant
             a further role after the invite exists, and the only way to read
             back the roles this call sent.
           - :meth:`ServiceAccounts.create()
             <pinecone.admin.service_accounts.ServiceAccounts.create>` — the machine
             equivalent, which takes the same binding shape but mints
             credentials instead of emailing anyone.
        """
        require_non_empty("email", email)
        bindings = list(role_bindings)
        if not bindings:
            raise ValidationError(
                "role_bindings must be a non-empty list; an invite needs at least one "
                "role binding, including an 'organization'-scoped membership role"
            )
        body: dict[str, Any] = {"email": email, "role_bindings": normalize_role_bindings(bindings)}

        logger.info("Creating invite (bindings=%d)", len(bindings))
        response = self._http.post("/admin/invites", json=body)
        result = self._adapter.to_invite(response.content)
        logger.debug("Created invite %r", result.id)
        return result

    def describe(self, *, invite_id: str) -> InviteModel:
        """Get one invite's details, whatever its status.

        Unlike :meth:`list`, this reaches accepted invites too — it is the only
        operation that can return ``status == InviteStatus.PROCESSED``, which is
        how you tell an accepted invite from one that never existed.

        Args:
            invite_id (str): The invite's UUID, as carried by ``InviteModel.id``.
                This is not the ID of the user the invite produced on
                acceptance; the two records have separate IDs.

        Returns:
            An :class:`~pinecone.models.admin.invite.InviteModel`. On an accepted
            invite, ``processed_at`` carries when it was accepted.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`: If *invite_id* is empty.
            :exc:`~pinecone.errors.exceptions.NotFoundError`:
                If no such invite exists in the organization. A deleted invite
                reads back as not found rather than as a status value, so
                not-found and accepted are genuinely different answers here.

        Examples:
            >>> from pinecone import Admin
            >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
            >>> invite = admin.invites.describe(
            ...     invite_id="9c8e3528-b9c0-4358-84ce-84c28e91b566"
            ... )
            >>> invite.email, invite.status
            ('newhire@acme.com', 'pending')

        .. seealso::
           - :meth:`Users.describe() <pinecone.admin.users.Users.describe>` — the member record
             created when this invite was accepted, addressed by its own user ID.
        """
        require_non_empty("invite_id", invite_id)
        logger.info("Describing invite %r", invite_id)
        response = self._http.get(f"/admin/invites/{quote(invite_id, safe='')}")
        result = self._adapter.to_invite(response.content)
        logger.debug("Described invite %r", invite_id)
        return result

    def delete(self, *, invite_id: str) -> None:
        """Withdraw a pending or expired invite, along with its role bindings.

        The invite and its role bindings are gone by the time this returns — a
        repeat call, or fetching it by ID afterwards, gets a not-found error.

        Args:
            invite_id (str): The UUID of the invite to withdraw.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`: If *invite_id* is empty.
            :exc:`~pinecone.errors.exceptions.NotFoundError`:
                If no such invite exists in the organization.
            :exc:`~pinecone.errors.exceptions.ConflictError`:
                If the invite has already been accepted. There is no invite left
                to withdraw; remove the resulting member with
                :meth:`Users.delete() <pinecone.admin.users.Users.delete>` instead.

        Examples:
            >>> from pinecone import Admin
            >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
            >>> admin.invites.delete(invite_id="9c8e3528-b9c0-4358-84ce-84c28e91b566")

        .. seealso::
           - :meth:`Users.delete() <pinecone.admin.users.Users.delete>` — the only way to revoke
             access once an invite has been accepted.
        """
        require_non_empty("invite_id", invite_id)
        logger.info("Deleting invite %r", invite_id)
        self._http.delete(f"/admin/invites/{quote(invite_id, safe='')}")
        logger.debug("Deleted invite %r", invite_id)

    def resend(self, *, invite_id: str) -> InviteModel:
        """Resend an invite's email and push its expiration back out.

        Works on pending and expired invites alike: the returned invite is
        ``pending`` again with a fresh ``expires_at``. Invite emails are rate
        limited per organization, so this is not safe to call in a tight loop —
        see the note below.

        Args:
            invite_id (str): The UUID of the invite to resend.

        Returns:
            The updated :class:`~pinecone.models.admin.invite.InviteModel`, with
            ``status`` back to ``pending`` and a later ``expires_at``. Read the
            new expiry from here rather than computing it.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`: If *invite_id* is empty.
            :exc:`~pinecone.errors.exceptions.NotFoundError`:
                If no such invite exists in the organization.
            :exc:`~pinecone.errors.exceptions.ConflictError`:
                If the invite has already been accepted. Never retry this one —
                there is nothing left to resend.
            :exc:`~pinecone.errors.exceptions.RateLimitError`:
                If the organization's invite-email budget is exhausted.
                ``retry_after`` carries the server's cooldown when one is
                supplied.

        Examples:
            >>> from pinecone import Admin
            >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
            >>> invite = admin.invites.resend(
            ...     invite_id="9c8e3528-b9c0-4358-84ce-84c28e91b566"
            ... )
            >>> invite.status
            'pending'
            >>> invite.expires_at
            '2026-05-21T03:00:00Z'

        .. note::
            A :exc:`~pinecone.errors.exceptions.RateLimitError` that reaches you
            has already survived the SDK's own retries, which honor
            ``Retry-After`` (see :doc:`/guides/retries`) — so an immediate retry
            of your own will just fail again. Honor ``exc.retry_after`` when the
            server supplies one, and back off generously otherwise.
        """
        require_non_empty("invite_id", invite_id)
        logger.info("Resending invite %r", invite_id)
        response = self._http.post(f"/admin/invites/{quote(invite_id, safe='')}/resend")
        result = self._adapter.to_invite(response.content)
        logger.debug("Resent invite %r", invite_id)
        return result
