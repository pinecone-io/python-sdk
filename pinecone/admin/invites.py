"""Invites namespace — list, create, describe, delete, and resend organization invites."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

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
    """Operations on organization invites.

    An invite is an offer, sent by email, for someone to join the
    organization; accepting it turns the recipient into a member. This
    namespace lists, creates, describes, deletes, and resends invites for the
    organization associated with the :class:`~pinecone.Admin` client's OAuth
    credentials.

    An invite's role bindings are not part of its representation: ``create``
    sends them, but no method here returns them. Read or change them
    afterwards through the role-binding operations, filtering on
    ``principal_type=invite``.

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
            This omits invites that have already been accepted. An invite
            missing from this list has not necessarily vanished — it may have
            been accepted, in which case :meth:`describe` still returns it
            with ``status == InviteStatus.PROCESSED``, and the accepted
            invitee is now a member reachable through ``admin.users``. Don't
            treat absence here as proof an invite never existed.

        No request is sent until the returned paginator is iterated. Iterating
        past the first page reuses the cursor returned with the previous page;
        iteration stops once a page comes back without one.

        Args:
            limit (int | None): Number of invites returned per page, between
                1 and 100. It caps page size, not how many invites the
                paginator yields in total — the paginator keeps following
                cursors until the pages run out. Use :func:`itertools.islice`
                to cap the total. ``None`` lets the server choose the page size.
            pagination_token (str | None): Cursor to resume iteration from a
                prior call's ``.pagination_token``. Reuse it with the same
                ``limit``.

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
        invite is ``pending``, and its ``expires_at`` is when it lapses. The
        response does **not** echo the role bindings — read them back through
        the role-binding operations, filtering on ``principal_type=invite``.

        Args:
            email (str): The address to invite, e.g. ``"newhire@acme.com"``.
                The SDK checks only that it isn't empty; the server validates
                the address itself and rejects a malformed or over-long one.
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
                address already belongs to an organization member.
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
        body: dict[str, Any] = {"email": email, "role_bindings": normalize_role_bindings(bindings)}

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
                If no such invite exists in the organization. A deleted
                invite reads back as not found rather than as a status value.
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

        By the time this call returns, the invite and its role bindings are
        gone — a repeat call, or fetching it by ID afterwards, gets a
        not-found error. An invite that has already been accepted can't be
        deleted this way: remove the resulting member with
        ``admin.users.delete`` instead.

        Args:
            invite_id (str): The identifier of the invite to delete.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`: If *invite_id* is empty.
            :exc:`~pinecone.errors.exceptions.NotFoundError`:
                If no such invite exists in the organization.
            :exc:`~pinecone.errors.exceptions.ConflictError`:
                If the invite has already been processed.
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
        """Resend an invite's email and push its expiration back out.

        Works on pending and expired invites alike: the returned invite is
        ``pending`` again with a fresh ``expires_at``.

        .. warning::
            Invite emails are rate limited per organization. Past that limit
            this raises :exc:`~pinecone.errors.exceptions.RateLimitError` —
            don't retry in a tight loop. Honor ``exc.retry_after`` when the
            server supplies one, and back off generously otherwise; the
            budget refills slowly enough that a sub-second retry will just
            fail again. An already-accepted invite raises
            :exc:`~pinecone.errors.exceptions.ConflictError` instead, which is
            never a signal to retry: there is nothing left to resend.

        Args:
            invite_id (str): The identifier of the invite to resend.

        Returns:
            The updated :class:`~pinecone.models.admin.invite.InviteModel`,
            with ``status`` back to ``pending`` and a later ``expires_at``.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`: If *invite_id* is empty.
            :exc:`~pinecone.errors.exceptions.NotFoundError`:
                If no such invite exists in the organization.
            :exc:`~pinecone.errors.exceptions.ConflictError`:
                If the invite has already been accepted and so cannot be
                resent.
            :exc:`~pinecone.errors.exceptions.RateLimitError`:
                If the organization's invite-email budget is exhausted.
                ``retry_after`` carries the server's cooldown period when
                one is supplied.
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
