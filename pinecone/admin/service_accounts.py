"""ServiceAccounts namespace — list, create, describe, update, delete, rotate secret."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from pinecone._internal.adapters.admin_adapter import AdminAdapter
from pinecone._internal.role_bindings import normalize_role_bindings
from pinecone._internal.validation import require_in_range, require_non_empty
from pinecone.errors.exceptions import ValidationError
from pinecone.models.admin.role_binding import RoleBindingInput
from pinecone.models.admin.service_account import ServiceAccountModel, ServiceAccountWithSecret
from pinecone.models.pagination import Page, Paginator

if TYPE_CHECKING:
    from pinecone._internal.http_client import HTTPClient

logger = logging.getLogger(__name__)

_LIMIT_MIN = 1
_LIMIT_MAX = 100


class ServiceAccounts:
    """Control-plane operations for the organization's service accounts.

    A service account is a non-human, machine identity for programmatic API
    access — distinct from the human members that
    :class:`~pinecone.admin.users.Users` manages. It is also the OAuth
    principal that :class:`~pinecone.Admin` itself authenticates as, so this
    namespace manages the same kind of credential the client is holding. Two
    consequences are worth knowing before calling anything here:

    - :meth:`create` and :meth:`rotate_secret` are the only operations that ever
      return a ``client_secret``, and each returns it exactly once. Nothing can
      retrieve it afterwards.
    - :meth:`rotate_secret` and :meth:`delete` aimed at the account whose
      credentials built this client will break it. See those methods.

    Role bindings are not part of a service account's representation:
    :meth:`create` can send initial ones, but no method here returns them. Use
    the role-binding operations with ``principal_type="service_account"`` and
    the account's ``id`` as ``principal_id`` to read or change them afterwards.

    Args:
        http (HTTPClient): HTTP client for making API requests.

    Examples:
        >>> from pinecone import Admin
        >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
        >>> for account in admin.service_accounts.list():
        ...     print(account.id, account.name)
    """

    def __init__(self, *, http: HTTPClient) -> None:
        self._http = http
        self._adapter = AdminAdapter()

    def __repr__(self) -> str:
        """Return developer-friendly representation."""
        return "ServiceAccounts()"

    def list(
        self,
        *,
        limit: int | None = None,
        pagination_token: str | None = None,
    ) -> Paginator[ServiceAccountModel]:
        """List the organization's service accounts, with lazy pagination.

        No request is sent until the returned paginator is iterated. Iterating
        past the first page reuses the cursor from the previous response's
        ``pagination.next`` verbatim; iteration stops on the first page that
        comes back without one.

        Args:
            limit (int | None): Number of service accounts the server returns
                **per page**, between 1 and 100. It caps each page, not how many
                accounts the paginator yields in total; the paginator keeps
                following cursors until the pages run out. Use
                :func:`itertools.islice` to cap the total. When ``None`` the
                parameter is omitted and the server chooses the page size.
            pagination_token (str | None): Cursor from a prior response's
                ``pagination.next``, to resume where a previous iteration
                stopped. Reuse it with the same ``limit``.

        Returns:
            :class:`~pinecone.models.pagination.Paginator` over
            :class:`~pinecone.models.admin.service_account.ServiceAccountModel`
            objects. Supports ``for`` loops, ``.to_list()``, ``.pages()`` for
            page-level access, and ``.pagination_token`` for resumption. The
            listed accounts carry no ``client_secret`` — that is returned only
            by :meth:`create` and :meth:`rotate_secret`.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`:
                If *limit* is outside 1-100. Raised before any network call.
            :exc:`ApiError`: If the API returns an error response.

        Examples:
            .. code-block:: python

                for account in admin.service_accounts.list():
                    print(account.id, account.name, account.client_id)

                for page in admin.service_accounts.list(limit=25).pages():
                    print(len(page.items), page.pagination_token)
        """
        if limit is not None:
            require_in_range("limit", limit, _LIMIT_MIN, _LIMIT_MAX)

        logger.info("Listing service accounts (limit=%r)", limit)

        def fetch_page(token: str | None) -> Page[ServiceAccountModel]:
            params: dict[str, str | int] = {}
            if limit is not None:
                params["limit"] = limit
            if token is not None:
                params["paginationToken"] = token
            response = self._http.get("/admin/service-accounts", params=params)
            result = self._adapter.to_service_account_list(response.content)
            logger.debug("Listed %d service accounts (has_more=%s)", len(result), result.has_more)
            return Page(items=result.data, pagination_token=result.pagination_token)

        return Paginator(fetch_page=fetch_page, initial_token=pagination_token)

    def create(
        self,
        *,
        name: str,
        role_bindings: Sequence[RoleBindingInput | Mapping[str, Any]] | None = None,
    ) -> ServiceAccountWithSecret:
        """Create a service account and receive its OAuth secret, once.

        .. warning::
            The returned ``client_secret`` is shown **exactly once**. It is not
            stored by the SDK and no later request can retrieve it — not
            :meth:`describe`, not :meth:`list`. Capture it now or the only
            recovery is :meth:`rotate_secret`, which mints a different one.
            Store it as a credential; ``repr()`` of the result masks it, but
            ``to_dict()`` and JSON encoding do not.

        The server does not deduplicate on name: repeating this call creates
        another, separate service account with its own credentials.

        Args:
            name (str): Human-readable label for the account. Sent verbatim —
                the SDK checks only that it is non-empty and leaves length and
                content to the server to validate. The server measures length
                in UTF-8 bytes rather than codepoints, so a name of multi-byte
                characters can be rejected while looking short to Python's
                ``len()``.
            role_bindings (Sequence[RoleBindingInput | Mapping[str, Any]] | None):
                Optional initial roles, as
                :class:`~pinecone.models.admin.role_binding.RoleBindingInput`
                instances or plain dicts, mixed freely. Each entry needs
                ``resource_type`` (``"organization"`` or ``"project"``) and
                ``role``; ``project`` scope additionally needs ``resource_id``,
                the project UUID. ``None`` and ``[]`` both create an account
                with no roles at all — it can obtain a token but do nothing
                with it until roles are granted through the role-binding
                operations. The bindings are **not** echoed in the response.

        Returns:
            A
            :class:`~pinecone.models.admin.service_account.ServiceAccountWithSecret`
            exposing ``.service_account`` (the metadata, including the ``id``
            and the OAuth ``client_id``) and ``.client_secret``.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`:
                If *name* is empty, or if any *role_bindings* entry is missing
                ``resource_type``/``role``, carries an unrecognized key, or
                names a value this SDK release does not know. The message names
                the index of the offending entry. Raised before any network call.
            :exc:`~pinecone.errors.exceptions.ForbiddenError`:
                If the caller lacks permission to create service accounts, or
                the organization's plan does not include them. The two cases
                are distinguishable only by the server's message.
            :exc:`ApiError`: If the API returns an error response.

        Examples:
            >>> from pinecone import Admin
            >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
            >>> created = admin.service_accounts.create(name="ci-prod")  # doctest: +SKIP
            >>> created.service_account.client_id  # doctest: +SKIP
            'l3Ow0CmFyc4jOONcwiKUCRqQKN0tiCAn'

            With initial roles, typed or as dicts:

            .. code-block:: python

                from pinecone.models.admin import ResourceType, RoleBindingInput, RoleName

                created = admin.service_accounts.create(
                    name="ci-prod",
                    role_bindings=[
                        RoleBindingInput(
                            resource_type=ResourceType.PROJECT,
                            role=RoleName.DATA_PLANE_EDITOR,
                            resource_id="a2f7dddb-1597-4eff-9f71-535fde243f58",
                        ),
                        {"resource_type": "organization", "role": "OrgMember"},
                    ],
                )
                store_secret(created.client_secret)
        """
        require_non_empty("name", name)
        body: dict[str, Any] = {"name": name}
        if role_bindings is not None:
            body["role_bindings"] = normalize_role_bindings(list(role_bindings))

        logger.info("Creating service account %r", name)
        response = self._http.post("/admin/service-accounts", json=body)
        result = self._adapter.to_service_account_with_secret(response.content)
        logger.debug("Created service account %r", result.service_account.id)
        return result

    def describe(self, *, service_account_id: str) -> ServiceAccountModel:
        """Get detailed information about one service account.

        The ``client_secret`` is never part of this response — it exists in the
        clear only in the :meth:`create` and :meth:`rotate_secret` results.

        Args:
            service_account_id (str): The identifier of the service account.

        Returns:
            A
            :class:`~pinecone.models.admin.service_account.ServiceAccountModel`
            with the account's metadata.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`:
                If *service_account_id* is empty.
            :exc:`~pinecone.errors.exceptions.NotFoundError`:
                If no such service account exists in the organization.
            :exc:`ApiError`: If the API returns an error response.

        Examples:
            >>> from pinecone import Admin
            >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
            >>> account = admin.service_accounts.describe(  # doctest: +SKIP
            ...     service_account_id="f8a3b2c1-4d5e-6f7a-8b9c-0d1e2f3a4b5c"
            ... )
        """
        require_non_empty("service_account_id", service_account_id)
        logger.info("Describing service account %r", service_account_id)
        response = self._http.get(f"/admin/service-accounts/{service_account_id}")
        result = self._adapter.to_service_account(response.content)
        logger.debug("Described service account %r", service_account_id)
        return result

    def update(
        self,
        *,
        service_account_id: str,
        name: str | None = None,
    ) -> ServiceAccountModel:
        """Rename a service account.

        Only the name is mutable here. Roles are managed through the
        role-binding operations, and the OAuth ``client_id`` and
        ``client_secret`` are not editable at all — rotate the secret with
        :meth:`rotate_secret` instead.

        Args:
            service_account_id (str): The identifier of the service account.
            name (str | None): The new name. Sent verbatim; the server owns the
                length and content rules, and measures length in UTF-8 bytes
                rather than codepoints.

        Returns:
            The updated
            :class:`~pinecone.models.admin.service_account.ServiceAccountModel`,
            with a fresh ``updated_at``. No secret is returned.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`:
                If *service_account_id* is empty, or if no updatable field was
                given. The server accepts a fieldless patch as a no-op success
                that merely bumps ``updated_at``, which hides a caller bug —
                usually a misspelled keyword — behind an apparent success, so
                the SDK names it instead. Raised before any network call.
            :exc:`~pinecone.errors.exceptions.NotFoundError`:
                If no such service account exists in the organization.
            :exc:`~pinecone.errors.exceptions.ForbiddenError`:
                If the caller lacks the update permission, or the
                organization's plan does not include service accounts.
            :exc:`ApiError`: If the API returns an error response.

        Examples:
            >>> from pinecone import Admin
            >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
            >>> account = admin.service_accounts.update(  # doctest: +SKIP
            ...     service_account_id="f8a3b2c1-4d5e-6f7a-8b9c-0d1e2f3a4b5c",
            ...     name="ci-prod-renamed",
            ... )
        """
        require_non_empty("service_account_id", service_account_id)
        if name is None:
            raise ValidationError("update() has nothing to change; provide at least one of: name")
        body: dict[str, Any] = {"name": name}

        logger.info("Updating service account %r", service_account_id)
        response = self._http.patch(f"/admin/service-accounts/{service_account_id}", json=body)
        result = self._adapter.to_service_account(response.content)
        logger.debug("Updated service account %r", service_account_id)
        return result

    def delete(self, *, service_account_id: str) -> None:
        """Delete a service account, its role bindings, and its credentials.

        .. warning::
            Deleting the service account whose ``client_id``/``client_secret``
            built this :class:`~pinecone.Admin` client revokes the credentials
            the client authenticates with. Tokens it already minted stop
            working within seconds and no new one can be obtained.

        The account and its role bindings are gone by the time this call
        returns; a repeat of this call raises
        :exc:`~pinecone.errors.exceptions.NotFoundError`, like any other
        reference to a deleted account.

        Args:
            service_account_id (str): The identifier of the service account to
                delete.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`:
                If *service_account_id* is empty.
            :exc:`~pinecone.errors.exceptions.NotFoundError`:
                If no such service account exists in the organization.
            :exc:`ApiError`: If the API returns an error response.

        Examples:
            >>> from pinecone import Admin
            >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
            >>> admin.service_accounts.delete(  # doctest: +SKIP
            ...     service_account_id="f8a3b2c1-4d5e-6f7a-8b9c-0d1e2f3a4b5c"
            ... )
        """
        require_non_empty("service_account_id", service_account_id)
        logger.info("Deleting service account %r", service_account_id)
        self._http.delete(f"/admin/service-accounts/{service_account_id}")
        logger.debug("Deleted service account %r", service_account_id)

    def rotate_secret(self, *, service_account_id: str) -> ServiceAccountWithSecret:
        """Issue a new OAuth client secret for a service account, revoking the old one.

        .. warning::
            The new ``client_secret`` is shown **exactly once**, in this
            response. It is not stored by the SDK and no later request can
            retrieve it; a rotation whose result is dropped can only be
            recovered by rotating again. ``repr()`` of the result masks it, but
            ``to_dict()`` and JSON encoding do not — never log the raw value.

        .. warning::
            Rotating the secret of the service account whose credentials built
            this :class:`~pinecone.Admin` client invalidates the secret that
            client holds. Its current access token keeps working until it
            expires, but the next token exchange fails until the client is
            rebuilt with the new secret. The previous secret and the tokens it
            minted are revoked within seconds.

        The account's ``id`` and OAuth ``client_id`` are unchanged: only the
        secret is new, so callers replace one value rather than reconfiguring
        the client identity. ``updated_at`` is not touched either — rotation
        leaves no trace in the account metadata, so do not use it to tell
        whether a rotation happened.

        Args:
            service_account_id (str): The identifier of the service account
                whose secret should be rotated.

        Returns:
            A
            :class:`~pinecone.models.admin.service_account.ServiceAccountWithSecret`
            whose ``.client_secret`` is the newly issued secret and whose
            ``.service_account`` carries the unchanged ``id`` and ``client_id``.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`:
                If *service_account_id* is empty.
            :exc:`~pinecone.errors.exceptions.NotFoundError`:
                If no such service account exists in the organization.
            :exc:`~pinecone.errors.exceptions.ForbiddenError`:
                If the caller lacks the rotate permission, or the
                organization's plan does not include service accounts.
            :exc:`ApiError`: If the API returns an error response.

        Examples:
            >>> from pinecone import Admin
            >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
            >>> rotated = admin.service_accounts.rotate_secret(  # doctest: +SKIP
            ...     service_account_id="f8a3b2c1-4d5e-6f7a-8b9c-0d1e2f3a4b5c"
            ... )
            >>> store_secret(rotated.client_secret)  # doctest: +SKIP
        """
        require_non_empty("service_account_id", service_account_id)
        logger.info("Rotating secret for service account %r", service_account_id)
        response = self._http.post(f"/admin/service-accounts/{service_account_id}/rotate-secret")
        result = self._adapter.to_service_account_with_secret(response.content)
        logger.debug("Rotated secret for service account %r", result.service_account.id)
        return result
