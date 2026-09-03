"""Service account response models for the Admin API."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from msgspec import Struct

from pinecone.models._mixin import StructDictMixin, _struct_to_dict_recursive
from pinecone.models.admin.pagination import PaginationResponse


class ServiceAccountModel(StructDictMixin, Struct, kw_only=True):
    """Response model for a service account. The OAuth secret is not included.

    What the account is allowed to do is not part of this model. Permissions
    come only from role bindings, so read them through
    :meth:`RoleBindings.list() <pinecone.admin.role_bindings.RoleBindings.list>` with
    ``principal_type="service_account"`` and this ``id`` as ``principal_id``.

    Attributes:
        id (str): Unique identifier (UUID) for the service account. Use this as
            the path parameter on service account operations and as the
            ``principal_id`` when querying or creating role bindings.
        name (str): Short human-readable label set at creation time.
        client_id (str): OAuth client ID the service account uses to obtain
            access tokens. Used only for OAuth token exchange — it is not the
            service account's identifier for role bindings, and passing it where
            ``id`` is expected reads back as not found.
        created_at (str): RFC 3339 timestamp for when the account was created.
        updated_at (str): RFC 3339 timestamp of the most recent metadata update.

    Examples:
        >>> from pinecone.models.admin.service_account import ServiceAccountModel
        >>> account = ServiceAccountModel(
        ...     id="f8a3b2c1-4d5e-6f7a-8b9c-0d1e2f3a4b5c",
        ...     name="ci-prod",
        ...     client_id="l3Ow0CmFyc4jOONcwiKUCRqQKN0tiCAn",
        ...     created_at="2026-04-10T15:23:00Z",
        ...     updated_at="2026-04-12T09:11:00Z",
        ... )
        >>> account.name
        'ci-prod'

    .. seealso::
       - :class:`~pinecone.models.admin.service_account.ServiceAccountWithSecret`
         — what :meth:`ServiceAccounts.create()
         <pinecone.admin.service_accounts.ServiceAccounts.create>`
         and :meth:`ServiceAccounts.rotate_secret()
         <pinecone.admin.service_accounts.ServiceAccounts.rotate_secret>` return instead,
         wrapping this model alongside the one-time secret.
       - :class:`~pinecone.models.admin.user.UserModel` — the human equivalent,
         which has an email address rather than OAuth credentials.
    """

    id: str
    name: str
    client_id: str
    created_at: str
    updated_at: str


class ServiceAccountWithSecret(StructDictMixin, Struct, kw_only=True):
    """Response model for a service account with a newly issued OAuth secret.

    Returned only by :meth:`ServiceAccounts.create()
    <pinecone.admin.service_accounts.ServiceAccounts.create>`
    and :meth:`ServiceAccounts.rotate_secret()
    <pinecone.admin.service_accounts.ServiceAccounts.rotate_secret>`, and
    the secret it carries is obtainable exactly once — nothing can retrieve it
    afterwards, so capture it before the object goes out of scope.

    Attributes:
        service_account (ServiceAccountModel): The service account metadata,
            including the ``id`` every other service-account operation takes.
        client_secret (str): The OAuth client secret. Treat as a credential.

    Examples:
        >>> from pinecone.models.admin.service_account import (
        ...     ServiceAccountModel,
        ...     ServiceAccountWithSecret,
        ... )
        >>> created = ServiceAccountWithSecret(
        ...     service_account=ServiceAccountModel(
        ...         id="f8a3b2c1-4d5e-6f7a-8b9c-0d1e2f3a4b5c",
        ...         name="ci-prod",
        ...         client_id="l3Ow0CmFyc4jOONcwiKUCRqQKN0tiCAn",
        ...         created_at="2026-04-10T15:23:00Z",
        ...         updated_at="2026-04-10T15:23:00Z",
        ...     ),
        ...     client_secret="8p-kkC23XOWvkCosKq",
        ... )
        >>> created.client_secret
        '8p-kkC23XOWvkCosKq'

        ``repr()`` keeps only the last four characters, so an object logged
        whole does not leak the secret:

        >>> repr(created).endswith("client_secret='...osKq')")
        True

    .. warning::
        The masking stops at ``repr()``. ``to_dict()`` and JSON encoding return
        ``client_secret`` in full, so a result serialized wholesale into a log
        line, an error report, or a cache writes the live credential out.
    """

    service_account: ServiceAccountModel
    client_secret: str

    def __repr__(self) -> str:
        masked = f"...{self.client_secret[-4:]}" if len(self.client_secret) >= 4 else "***"
        return (
            f"ServiceAccountWithSecret(service_account={self.service_account!r}, "
            f"client_secret='{masked}')"
        )

    def __str__(self) -> str:
        return repr(self)


class ServiceAccountList(Struct, kw_only=True):
    """A page of service accounts, plus the cursor for the next page.

    One raw page of a service-account listing. Callers who reach accounts
    through :meth:`ServiceAccounts.list() <pinecone.admin.service_accounts.ServiceAccounts.list>`
    get a :class:`~pinecone.models.pagination.Paginator` instead, which follows
    these cursors for them.

    Attributes:
        data (list[ServiceAccountModel]): The service accounts on this page.
            None of them carries a ``client_secret``.
        pagination (PaginationResponse | None): Cursor envelope for the next
            page, or ``None`` on the final page.

    Examples:
        >>> from pinecone.models.admin.service_account import (
        ...     ServiceAccountList,
        ...     ServiceAccountModel,
        ... )
        >>> accounts = ServiceAccountList(
        ...     data=[
        ...         ServiceAccountModel(
        ...             id="f8a3b2c1-4d5e-6f7a-8b9c-0d1e2f3a4b5c",
        ...             name="ci-prod",
        ...             client_id="l3Ow0CmFyc4jOONcwiKUCRqQKN0tiCAn",
        ...             created_at="2026-04-10T15:23:00Z",
        ...             updated_at="2026-04-10T15:23:00Z",
        ...         )
        ...     ]
        ... )
        >>> accounts.names()
        ['ci-prod']
    """

    data: list[ServiceAccountModel] = []
    pagination: PaginationResponse | None = None

    def __iter__(self) -> Iterator[ServiceAccountModel]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> ServiceAccountModel:
        return self.data[index]

    @property
    def pagination_token(self) -> str | None:
        """Opaque cursor for the next page, or ``None`` if this is the last page."""
        return self.pagination.next if self.pagination is not None else None

    @property
    def has_more(self) -> bool:
        """``True`` when the server supplied a cursor for a further page."""
        return self.pagination_token is not None

    def names(self) -> list[str]:
        """Return the service account names on this page, in order."""
        return [account.name for account in self.data]

    def to_dict(self) -> dict[str, Any]:
        """Return this page as a serializable dict with ``data`` and ``pagination`` keys."""
        return {
            "data": [_struct_to_dict_recursive(account) for account in self.data],
            "pagination": _struct_to_dict_recursive(self.pagination),
        }

    def __repr__(self) -> str:
        summaries = ", ".join(f"<id={a.id!r}, name={a.name!r}>" for a in self.data)
        return f"ServiceAccountList([{summaries}], has_more={self.has_more!r})"
