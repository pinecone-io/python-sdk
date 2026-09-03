"""ServiceAccounts namespace — list, create, describe, update, delete, rotate secret."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

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
    """The organization's machine identities, and the credentials they authenticate with.

    A service account is a non-human principal for programmatic API access. It
    is also the kind of principal :class:`~pinecone.Admin` itself authenticates
    as, so this namespace manages the same species of credential the client is
    holding. Not constructed directly — reach it as ``admin.service_accounts``.

    Two consequences are worth knowing before calling anything here:

    - :meth:`create` and :meth:`rotate_secret` are the only operations that ever
      return a ``client_secret``, and each returns it exactly once. Capture it,
      or rotate again — nothing can retrieve it afterwards.
    - :meth:`rotate_secret` and :meth:`delete` aimed at the account whose
      credentials built this client will break it. See those methods.

    Role bindings are not part of a service account's representation:
    :meth:`create` can send initial ones, but no method here returns them. Use
    :class:`~pinecone.admin.role_bindings.RoleBindings` with
    ``principal_type="service_account"`` and the account's ``id`` as
    ``principal_id`` to read or change them afterwards.

    See :doc:`/guides/error-handling` for the exceptions every operation here
    can raise.

    Examples:
        >>> from pinecone import Admin
        >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
        >>> for account in admin.service_accounts.list():
        ...     print(account.name, account.client_id)
        ci-prod l3Ow0CmFyc4jOONcwiKUCRqQKN0tiCAn

    .. seealso::
       - :class:`~pinecone.admin.users.Users` — the human members. A person is
         invited and accepts; a service account is created outright and holds
         its own OAuth credentials, so the two are never interchangeable.
       - :class:`~pinecone.admin.api_keys.ApiKeys` — the other machine
         credential. An API key authorizes data-plane and control-plane calls
         within one project; a service account authenticates the Admin API
         across the organization.
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

        No request is sent until the returned paginator is iterated; see
        :doc:`/guides/pagination`.

        Args:
            limit (int | None): Number of service accounts the server returns
                **per page**. It caps each page, not how many accounts the
                paginator yields in total; the paginator keeps following cursors
                until the pages run out, so use :func:`itertools.islice` to cap
                the total. When ``None`` the server chooses the page size.
            pagination_token (str | None): Cursor from a previous paginator's
                ``pagination_token``, to resume where that iteration stopped.
                Reuse it with the same ``limit``.

        Returns:
            :class:`~pinecone.models.pagination.Paginator` yielding
            :class:`~pinecone.models.admin.service_account.ServiceAccountModel`
            objects. The listed accounts carry no ``client_secret`` — that is
            returned only by :meth:`create` and :meth:`rotate_secret`.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`:
                If *limit* is outside 1-100. Raised before any network call.

        Examples:
            >>> from pinecone import Admin
            >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
            >>> for account in admin.service_accounts.list():
            ...     print(account.name, account.client_id)
            ci-prod l3Ow0CmFyc4jOONcwiKUCRqQKN0tiCAn

            Page-level access exposes the cursor, which is ``None`` once there
            is no further page to fetch:

            >>> for page in admin.service_accounts.list(limit=25).pages():
            ...     print(len(page.items), page.pagination_token)
            1 None

        .. seealso::
           - :meth:`Users.list() <pinecone.admin.users.Users.list>` — the human members, which
             this list deliberately excludes.
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

        The returned ``client_secret`` is shown **exactly once**: it is not
        stored by the SDK, and neither :meth:`describe` nor :meth:`list` can
        retrieve it. Capture it here or the only recovery is
        :meth:`rotate_secret`, which mints a different one. The server does not
        deduplicate on name, so repeating this call creates another, separate
        account with its own credentials rather than returning the first.

        Args:
            name (str): Human-readable label for the account, e.g. ``"ci-prod"``.
                Sent verbatim — the SDK checks only that it is non-empty and
                leaves length and content to the server. The server measures
                length in UTF-8 bytes rather than codepoints, so a name of
                multi-byte characters can be rejected while looking short to
                Python's ``len()``.
            role_bindings (Sequence[RoleBindingInput | Mapping[str, Any]] | None):
                Optional initial roles, as
                :class:`~pinecone.models.admin.role_binding.RoleBindingInput`
                instances or plain dicts, mixed freely. Each entry needs
                ``resource_type`` (``"organization"`` or ``"project"``) and
                ``role``; ``project`` scope additionally needs ``resource_id``,
                the project UUID. ``None`` and ``[]`` both create an account with
                no roles at all — it can obtain a token but do nothing with it
                until roles are granted. The bindings are **not** echoed in the
                response.

        Returns:
            A
            :class:`~pinecone.models.admin.service_account.ServiceAccountWithSecret`
            exposing ``.service_account`` (the metadata, including the ``id``
            every other method here takes and the OAuth ``client_id``) and
            ``.client_secret``.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`:
                If *name* is empty, or if any *role_bindings* entry is missing
                ``resource_type``/``role``, carries an unrecognized key, or names
                a value this SDK release does not know. The message names the
                index of the offending entry. Raised before any network call.
            :exc:`~pinecone.errors.exceptions.ForbiddenError`:
                If the caller lacks permission to create service accounts, or the
                organization's plan does not include them. The two cases are
                distinguishable only by the server's message, so read it before
                concluding which one you hit.

        Examples:
            >>> from pinecone import Admin
            >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
            >>> created = admin.service_accounts.create(name="ci-prod")
            >>> created.service_account.name
            'ci-prod'
            >>> bool(created.client_secret)
            True

            That is the only moment ``created.client_secret`` is readable —
            hand it straight to whatever stores your credentials, because
            neither :meth:`describe` nor :meth:`list` will return it and the
            only other way to get a working secret is :meth:`rotate_secret`.

            With initial roles, typed or as dicts:

            >>> from pinecone.models.admin import ResourceType, RoleBindingInput, RoleName
            >>> created = admin.service_accounts.create(
            ...     name="ci-prod",
            ...     role_bindings=[
            ...         RoleBindingInput(
            ...             resource_type=ResourceType.PROJECT,
            ...             role=RoleName.DATA_PLANE_EDITOR,
            ...             resource_id="a2f7dddb-1597-4eff-9f71-535fde243f58",
            ...         ),
            ...         {"resource_type": "organization", "role": "OrgMember"},
            ...     ],
            ... )
            >>> bool(created.client_secret)
            True

            The roles are not echoed back, so read them through
            :meth:`RoleBindings.list() <pinecone.admin.role_bindings.RoleBindings.list>`.

        .. warning::
            Treat ``client_secret`` as a credential. ``repr()`` of the result
            masks it, but ``to_dict()`` and JSON encoding return it in full, so a
            result logged or serialized wholesale leaks the secret.

        .. seealso::
           - :meth:`rotate_secret` — the only way to obtain a working secret for
             an account whose creation result was dropped.
           - :meth:`Invites.create() <pinecone.admin.invites.Invites.create>` — the human
             equivalent, which takes the same binding shape but emails an offer
             instead of minting credentials.
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
        """Get one service account's metadata.

        The ``client_secret`` is never part of this response — it exists in the
        clear only in the :meth:`create` and :meth:`rotate_secret` results.

        Args:
            service_account_id (str): The account's UUID, as carried by
                ``ServiceAccountModel.id``. Not the OAuth ``client_id``, which
                identifies the account only during token exchange.

        Returns:
            A
            :class:`~pinecone.models.admin.service_account.ServiceAccountModel`
            with ``id``, ``name``, ``client_id``, ``created_at``, and
            ``updated_at``.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`:
                If *service_account_id* is empty.
            :exc:`~pinecone.errors.exceptions.NotFoundError`:
                If no such service account exists in the organization. Passing
                the OAuth ``client_id`` instead of the ``id`` lands here too.

        Examples:
            >>> from pinecone import Admin
            >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
            >>> account = admin.service_accounts.describe(
            ...     service_account_id="f8a3b2c1-4d5e-6f7a-8b9c-0d1e2f3a4b5c"
            ... )
            >>> account.name
            'ci-prod'
            >>> "client_secret" in account.to_dict()
            False

        .. seealso::
           - :meth:`Users.describe() <pinecone.admin.users.Users.describe>` — the human
             equivalent, addressed by user ID.
        """
        require_non_empty("service_account_id", service_account_id)
        logger.info("Describing service account %r", service_account_id)
        response = self._http.get(f"/admin/service-accounts/{quote(service_account_id, safe='')}")
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

        Only the name is mutable here. Roles are managed through
        :class:`~pinecone.admin.role_bindings.RoleBindings`, and the OAuth
        ``client_id`` and ``client_secret`` are not editable at all — rotate the
        secret with :meth:`rotate_secret` instead.

        Args:
            service_account_id (str): The account's UUID.
            name (str | None): The new name, e.g. ``"ci-prod-eu"``. Sent
                verbatim; the server owns the length and content rules, and
                measures length in UTF-8 bytes rather than codepoints.

        Returns:
            The updated
            :class:`~pinecone.models.admin.service_account.ServiceAccountModel`,
            with a fresh ``updated_at``. No secret is returned.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`:
                If *service_account_id* is empty, or if no updatable field was
                given. The server would accept a fieldless patch as a success
                that merely bumps ``updated_at``, hiding a caller bug — usually a
                misspelled keyword — behind an apparent success, so the SDK
                rejects it first. Raised before any network call.
            :exc:`~pinecone.errors.exceptions.NotFoundError`:
                If no such service account exists in the organization.
            :exc:`~pinecone.errors.exceptions.ForbiddenError`:
                If the caller lacks the update permission, or the organization's
                plan does not include service accounts.

        Examples:
            >>> from pinecone import Admin
            >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
            >>> account = admin.service_accounts.update(
            ...     service_account_id="f8a3b2c1-4d5e-6f7a-8b9c-0d1e2f3a4b5c",
            ...     name="ci-prod-eu",
            ... )
            >>> account.client_id
            'l3Ow0CmFyc4jOONcwiKUCRqQKN0tiCAn'

            The OAuth ``client_id`` is untouched by a rename, so anything
            already authenticating as this account keeps working.
        """
        require_non_empty("service_account_id", service_account_id)
        if name is None:
            raise ValidationError("update() has nothing to change; provide at least one of: name")
        body: dict[str, Any] = {"name": name}

        logger.info("Updating service account %r", service_account_id)
        response = self._http.patch(
            f"/admin/service-accounts/{quote(service_account_id, safe='')}", json=body
        )
        result = self._adapter.to_service_account(response.content)
        logger.debug("Updated service account %r", service_account_id)
        return result

    def delete(self, *, service_account_id: str) -> None:
        """Delete a service account, its role bindings, and its credentials.

        The account and its role bindings are gone by the time this returns; a
        repeat of this call raises
        :exc:`~pinecone.errors.exceptions.NotFoundError`, like any other
        reference to a deleted account.

        Args:
            service_account_id (str): The UUID of the service account to delete.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`:
                If *service_account_id* is empty.
            :exc:`~pinecone.errors.exceptions.NotFoundError`:
                If no such service account exists in the organization.

        Examples:
            >>> from pinecone import Admin
            >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
            >>> admin.service_accounts.delete(
            ...     service_account_id="f8a3b2c1-4d5e-6f7a-8b9c-0d1e2f3a4b5c"
            ... )

        .. warning::
            Deleting the service account whose ``client_id``/``client_secret``
            built this :class:`~pinecone.Admin` client revokes the credentials
            that client authenticates with. Tokens it already minted stop
            working and no new one can be obtained, so the client cannot undo
            this — recovery needs another account's credentials.

        .. seealso::
           - :meth:`rotate_secret` — replaces the secret without destroying the
             account or its role bindings.
        """
        require_non_empty("service_account_id", service_account_id)
        logger.info("Deleting service account %r", service_account_id)
        self._http.delete(f"/admin/service-accounts/{quote(service_account_id, safe='')}")
        logger.debug("Deleted service account %r", service_account_id)

    def rotate_secret(self, *, service_account_id: str) -> ServiceAccountWithSecret:
        """Issue a new OAuth client secret for a service account, revoking the old one.

        The new ``client_secret`` is shown **exactly once**, in this response: it
        is not stored by the SDK and no later request can retrieve it, so a
        rotation whose result is dropped can only be recovered by rotating
        again. The account's ``id`` and OAuth ``client_id`` are unchanged, so
        callers replace one value rather than reconfiguring the client identity.

        Args:
            service_account_id (str): The UUID of the service account whose
                secret should be rotated.

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
                If the caller lacks the rotate permission, or the organization's
                plan does not include service accounts.

        Examples:
            >>> from pinecone import Admin
            >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
            >>> rotated = admin.service_accounts.rotate_secret(
            ...     service_account_id="f8a3b2c1-4d5e-6f7a-8b9c-0d1e2f3a4b5c"
            ... )
            >>> rotated.service_account.client_id
            'l3Ow0CmFyc4jOONcwiKUCRqQKN0tiCAn'
            >>> bool(rotated.client_secret)
            True

            The ``client_id`` above is the pre-rotation one, unchanged: only
            the secret is new, so a caller replaces one value. Read
            ``rotated.client_secret`` now and store it — this response is the
            only place it exists in the clear.

        .. warning::
            Rotating the secret of the service account whose credentials built
            this :class:`~pinecone.Admin` client invalidates the secret that
            client holds. Its current access token keeps working until it
            expires, but the next token exchange fails until the client is
            rebuilt with the new secret.

        .. warning::
            Treat ``client_secret`` as a credential. ``repr()`` of the result
            masks it, but ``to_dict()`` and JSON encoding return it in full — so
            never log or serialize the result wholesale.

        .. seealso::
           - :meth:`create` — the other operation that returns a
             ``client_secret``, and the only one that returns a new account.
        """
        require_non_empty("service_account_id", service_account_id)
        logger.info("Rotating secret for service account %r", service_account_id)
        response = self._http.post(
            f"/admin/service-accounts/{quote(service_account_id, safe='')}/rotate-secret"
        )
        result = self._adapter.to_service_account_with_secret(response.content)
        logger.debug("Rotated secret for service account %r", result.service_account.id)
        return result
