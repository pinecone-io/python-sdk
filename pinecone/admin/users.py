"""Users namespace — list, describe, and delete organization members."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from urllib.parse import quote

from pinecone._internal.adapters.admin_adapter import AdminAdapter
from pinecone._internal.validation import require_in_range, require_non_empty
from pinecone.models.admin.user import UserModel
from pinecone.models.pagination import Page, Paginator

if TYPE_CHECKING:
    from pinecone._internal.http_client import HTTPClient

logger = logging.getLogger(__name__)

_LIMIT_MIN = 1
_LIMIT_MAX = 100


class Users:
    """The human members of a Pinecone organization.

    A user is a person who has accepted an invitation and now belongs to the
    organization that the :class:`~pinecone.Admin` client's OAuth credentials
    resolve to. Not constructed directly — reach it as ``admin.users``.

    What a user is allowed to do is not part of this model. Permissions come
    only from role bindings, so :class:`~pinecone.admin.role_bindings.RoleBindings`
    is where a user's access is read and changed.

    See :doc:`/guides/error-handling` for the exceptions every operation here
    can raise.

    Examples:
        >>> from pinecone import Admin
        >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
        >>> for user in admin.users.list():
        ...     print(user.email)
        alice@example.com

    .. seealso::
       - :class:`~pinecone.admin.invites.Invites` — the same person before they
         accept. An invitee is not yet a user and is not listed here.
       - :class:`~pinecone.admin.service_accounts.ServiceAccounts` — the machine
         equivalent, for programmatic access rather than a person.
    """

    def __init__(self, *, http: HTTPClient) -> None:
        self._http = http
        self._adapter = AdminAdapter()

    def __repr__(self) -> str:
        """Return developer-friendly representation."""
        return "Users()"

    def list(
        self,
        *,
        email: str | None = None,
        limit: int | None = None,
        pagination_token: str | None = None,
    ) -> Paginator[UserModel]:
        """List the users in the organization, with lazy pagination.

        No request is sent until the returned paginator is iterated; see
        :doc:`/guides/pagination`.

        Args:
            email (str | None): Filter on the user's email address, e.g.
                ``"alice@example.com"``. Forwarded verbatim — the SDK does not
                validate or normalize it, so the server decides what matches
                and rejects a malformed address. Omit to list every user.
            limit (int | None): Number of users the server returns **per page**.
                It caps each page, not how many users the paginator yields in
                total; the paginator keeps following cursors until the pages run
                out, so use :func:`itertools.islice` to cap the total. When
                ``None`` the server chooses the page size.
            pagination_token (str | None): Cursor from a previous paginator's
                ``pagination_token``, to resume where that iteration stopped.
                Reuse it with the same ``email`` and ``limit``.

        Returns:
            :class:`~pinecone.models.pagination.Paginator` yielding
            :class:`~pinecone.models.admin.user.UserModel` objects.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`:
                If *limit* is outside 1-100. Raised before any network call.

        Examples:
            >>> from pinecone import Admin
            >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
            >>> for user in admin.users.list():
            ...     print(user.id, user.email)
            e2e92523-85dc-4142-b8c2-e681be8b78df alice@example.com

            Filtering by address still returns a paginator, not a single user:

            >>> admin.users.list(email="alice@example.com").to_list()[0].name
            'Alice Nakamura'

        .. seealso::
           - :meth:`Invites.list() <pinecone.admin.invites.Invites.list>` — invitees who have
             not accepted yet, and so are absent from this list.
           - :meth:`RoleBindings.list() <pinecone.admin.role_bindings.RoleBindings.list>` — with
             ``principal_type="user"``, what each user can do.
        """
        if limit is not None:
            require_in_range("limit", limit, _LIMIT_MIN, _LIMIT_MAX)

        logger.info("Listing users (email_filter=%s, limit=%r)", email is not None, limit)

        def fetch_page(token: str | None) -> Page[UserModel]:
            params: dict[str, str | int] = {}
            if email is not None:
                params["email"] = email
            if limit is not None:
                params["limit"] = limit
            if token is not None:
                params["paginationToken"] = token
            response = self._http.get("/admin/users", params=params)
            result = self._adapter.to_user_list(response.content)
            logger.debug("Listed %d users (has_more=%s)", len(result), result.has_more)
            return Page(items=result.data, pagination_token=result.pagination_token)

        return Paginator(fetch_page=fetch_page, initial_token=pagination_token)

    def describe(self, *, user_id: str) -> UserModel:
        """Get one user's details by their user ID.

        Args:
            user_id (str): The user's UUID, as carried by ``UserModel.id`` — not
                their email address, and not the ID of the invite they accepted.

        Returns:
            :class:`~pinecone.models.admin.user.UserModel` with ``id``,
            ``email``, and ``name`` (``None`` when the user has not set one).

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`: If *user_id* is empty.
            :exc:`~pinecone.errors.exceptions.NotFoundError`:
                If no such user is a member of the organization. A person who
                was invited but has not accepted reads back as not found here —
                look for them under :meth:`Invites.list() <pinecone.admin.invites.Invites.list>`
                instead.

        Examples:
            >>> from pinecone import Admin
            >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
            >>> user = admin.users.describe(user_id="e2e92523-85dc-4142-b8c2-e681be8b78df")
            >>> user.email
            'alice@example.com'

        .. seealso::
           - :meth:`Invites.describe() <pinecone.admin.invites.Invites.describe>` — the invite a
             user accepted, which keeps its own ID and status after acceptance.
        """
        require_non_empty("user_id", user_id)
        logger.info("Describing user %r", user_id)
        response = self._http.get(f"/admin/users/{quote(user_id, safe='')}")
        result = self._adapter.to_user(response.content)
        logger.debug("Described user %r", user_id)
        return result

    def delete(self, *, user_id: str) -> None:
        """Remove a user from the organization.

        The user's role bindings are revoked immediately; their Pinecone account
        itself is not deleted. This call is not repeatable — once it succeeds, a
        second call with the same *user_id* raises
        :exc:`~pinecone.errors.exceptions.NotFoundError`, as does
        :meth:`describe` for that user.

        Args:
            user_id (str): The UUID of the user to remove.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`: If *user_id* is empty.
            :exc:`~pinecone.errors.exceptions.NotFoundError`:
                If no such user is a member of the organization.
            :exc:`~pinecone.errors.exceptions.ConflictError`:
                If removal would violate an organization invariant, such as
                dropping the last ``OrgOwner``. Resolve what the server's
                message names — usually by granting that role to someone else —
                then retry.

        Examples:
            >>> from pinecone import Admin
            >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
            >>> admin.users.delete(user_id="e2e92523-85dc-4142-b8c2-e681be8b78df")

        .. seealso::
           - :meth:`Invites.delete() <pinecone.admin.invites.Invites.delete>` — how to withdraw
             access from someone who never accepted; this method cannot reach them.
        """
        require_non_empty("user_id", user_id)
        logger.info("Deleting user %r", user_id)
        self._http.delete(f"/admin/users/{quote(user_id, safe='')}")
        logger.debug("Deleted user %r", user_id)
