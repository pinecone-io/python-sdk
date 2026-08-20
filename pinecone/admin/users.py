"""Users namespace — list, describe, and delete organization members."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

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
    """Control-plane operations for the users in an organization.

    Provides methods to list, describe, and remove the members of the
    organization associated with the :class:`~pinecone.Admin` client's OAuth
    credentials.

    Role bindings are not part of a user's representation. Use the role-binding
    operations to see or change what a user can do.

    Args:
        http (HTTPClient): HTTP client for making API requests.

    Examples:
        >>> from pinecone import Admin
        >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
        >>> for user in admin.users.list():
        ...     print(user.email)
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
        """List the users in the organization, with transparent lazy pagination.

        No request is sent until the returned paginator is iterated. Iterating
        past the first page reuses the cursor from the previous response's
        ``pagination.next`` verbatim; iteration stops on the first page that
        comes back without one.

        Args:
            email (str | None): Case-insensitive filter on the user's email
                address. Sent verbatim as a query parameter — the SDK does not
                validate or normalize it, so a malformed address is rejected by
                the server with ``400 INVALID_ARGUMENT``. Omitted when ``None``.
            limit (int | None): Number of users the server returns **per page**,
                between 1 and 100. This is the spec's ``limit`` query parameter,
                not a cap on how many users the paginator yields in total; the
                paginator keeps following cursors until the pages run out. Use
                :func:`itertools.islice` to cap the total. When ``None`` the
                parameter is omitted and the server defaults to 100.
            pagination_token (str | None): Cursor from a prior response's
                ``pagination.next``, to resume where a previous iteration
                stopped. Reuse it with the same ``email`` and ``limit``.

        Returns:
            :class:`~pinecone.models.pagination.Paginator` over
            :class:`~pinecone.models.admin.user.UserModel` objects. Supports
            ``for`` loops, ``.to_list()``, ``.pages()`` for page-level access,
            and ``.pagination_token`` for resumption.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`:
                If *limit* is outside 1-100. Raised before any network call.
            :exc:`ApiError`: If the API returns an error response.

        Examples:
            .. code-block:: python

                for user in admin.users.list():
                    print(user.id, user.email)

                matches = admin.users.list(email="alice@example.com").to_list()

                for page in admin.users.list(limit=25).pages():
                    print(len(page.items), page.pagination_token)
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
        """Get detailed information about a user in the organization.

        Args:
            user_id (str): The identifier of the user.

        Returns:
            A :class:`~pinecone.models.admin.user.UserModel` with the user's details.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`: If *user_id* is empty.
            :exc:`~pinecone.errors.exceptions.NotFoundError`:
                If no such user is a member of the organization (404).
            :exc:`ApiError`: If the API returns an error response.

        Examples:
            >>> from pinecone import Admin
            >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
            >>> user = admin.users.describe(user_id="e2e92523-85dc-4142-b8c2-e681be8b78df")
            >>> user.email  # doctest: +SKIP
            'alice@example.com'
        """
        require_non_empty("user_id", user_id)
        logger.info("Describing user %r", user_id)
        response = self._http.get(f"/admin/users/{user_id}")
        result = self._adapter.to_user(response.content)
        logger.debug("Described user %r", user_id)
        return result

    def delete(self, *, user_id: str) -> None:
        """Remove a user from the organization.

        The user's role bindings are revoked immediately; their Pinecone
        account itself is not deleted. The server answers ``202`` with no body,
        after which the user reads back as ``404`` — including for a repeat of
        this call.

        Args:
            user_id (str): The identifier of the user to remove.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`: If *user_id* is empty.
            :exc:`~pinecone.errors.exceptions.NotFoundError`:
                If no such user is a member of the organization (404).
            :exc:`~pinecone.errors.exceptions.ConflictError`:
                If removal would violate an organization invariant, such as
                dropping the last ``OrgOwner`` (409). The server's error code
                and message are carried through verbatim.
            :exc:`ApiError`: If the API returns an error response.

        Examples:
            >>> from pinecone import Admin
            >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
            >>> admin.users.delete(user_id="e2e92523-85dc-4142-b8c2-e681be8b78df")
        """
        require_non_empty("user_id", user_id)
        logger.info("Deleting user %r", user_id)
        self._http.delete(f"/admin/users/{user_id}")
        logger.debug("Deleted user %r", user_id)
