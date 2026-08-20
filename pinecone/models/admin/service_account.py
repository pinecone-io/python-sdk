"""Service account response models for the Admin API."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from msgspec import Struct

from pinecone.models._mixin import StructDictMixin, _struct_to_dict_recursive
from pinecone.models.admin.pagination import PaginationResponse


class ServiceAccountModel(StructDictMixin, Struct, kw_only=True):
    """Response model for a service account. The OAuth secret is not included.

    Role bindings are not included; use the role binding operations with
    ``principal_type="service_account"`` to see what the account can do.

    Attributes:
        id (str): Unique identifier (UUID) for the service account. Use this as
            the path parameter on service account operations and as the
            ``principal_id`` when querying or creating role bindings.
        name (str): Short human-readable label set at creation time.
        client_id (str): OAuth client ID the service account uses to obtain
            access tokens. Used only for OAuth token exchange — it is not the
            service account's identifier for role bindings.
        created_at (str): RFC 3339 timestamp for when the account was created.
        updated_at (str): RFC 3339 timestamp of the most recent metadata update.

    Examples:
        >>> from pinecone.models.admin.service_account import ServiceAccountModel
        >>> account = ServiceAccountModel(
        ...     id="f8a3b2c1-4d5e-6f7a-8b9c-0d1e2f3a4b5c",
        ...     name="My Service Account",
        ...     client_id="l3Ow0CmFyc4jOONcwiKUCRqQKN0tiCAn",
        ...     created_at="2026-04-10T15:23:00Z",
        ...     updated_at="2026-04-12T09:11:00Z",
        ... )
        >>> account.name
        'My Service Account'
    """

    id: str
    name: str
    client_id: str
    created_at: str
    updated_at: str


class ServiceAccountWithSecret(StructDictMixin, Struct, kw_only=True):
    """Response model for a service account with a newly issued OAuth secret.

    The secret is returned exactly once — at creation and on secret rotation —
    and cannot be retrieved later. :meth:`__repr__` masks it so it does not
    leak into logs; ``to_dict()`` and JSON encoding return it in full.

    Attributes:
        service_account (ServiceAccountModel): The service account metadata.
        client_secret (str): The OAuth client secret. Treat as a credential.

    Examples:
        >>> from pinecone.models.admin.service_account import (
        ...     ServiceAccountModel,
        ...     ServiceAccountWithSecret,
        ... )
        >>> created = ServiceAccountWithSecret(
        ...     service_account=ServiceAccountModel(
        ...         id="sa1",
        ...         name="ci-prod",
        ...         client_id="cid",
        ...         created_at="2026-04-10T15:23:00Z",
        ...         updated_at="2026-04-10T15:23:00Z",
        ...     ),
        ...     client_secret="8p-kkC23XOWvkCosKq",
        ... )
        >>> created.client_secret
        '8p-kkC23XOWvkCosKq'
        >>> repr(created).endswith("client_secret='...osKq')")
        True
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

    Attributes:
        data (list[ServiceAccountModel]): The service accounts on this page.
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
        ...             id="sa1",
        ...             name="ci-prod",
        ...             client_id="cid",
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
