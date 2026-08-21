"""Organization response models for the Admin API."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from msgspec import Struct

from pinecone.models._mixin import StructDictMixin


class OrganizationModel(StructDictMixin, Struct, kw_only=True):
    """Response model for a Pinecone organization.

    Attributes:
        id: Unique identifier for the organization.
        name: Name of the organization.
        plan: The organization's plan tier, as the server names it.
        payment_status: Current payment status.
        created_at: Timestamp when the organization was created.
        support_tier: Support tier for the organization.
    """

    id: str
    name: str
    plan: str
    payment_status: str
    created_at: str
    support_tier: str

    def __getitem__(self, key: str) -> Any:
        """Support bracket access (e.g. org['name'])."""
        if key not in self.__struct_fields__:
            raise KeyError(key)
        return getattr(self, key)

    def __contains__(self, key: object) -> bool:
        """Support ``in`` operator (e.g. ``'name' in org``)."""
        return key in self.__struct_fields__


class OrganizationList:
    """Wrapper around a list of OrganizationModel with convenience methods."""

    def __init__(self, organizations: list[OrganizationModel]) -> None:
        """Initialize an OrganizationList.

        Args:
            organizations: List of :class:`OrganizationModel` instances
                representing Pinecone organizations.
        """
        self._organizations = organizations

    def __iter__(self) -> Iterator[OrganizationModel]:
        return iter(self._organizations)

    def __len__(self) -> int:
        return len(self._organizations)

    def __getitem__(self, index: int) -> OrganizationModel:
        return self._organizations[index]

    def to_dict(self) -> dict[str, Any]:
        """Return the list as a serializable dict.

        Returns:
            dict[str, Any]: A dict with a ``"data"`` key containing a list of
            organization dicts, each produced by :meth:`OrganizationModel.to_dict`.

        Examples:
            >>> from pinecone import Admin
            >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
            >>> orgs = admin.organizations.list()
            >>> orgs.to_dict()  # doctest: +SKIP
            {'data': [{'name': 'acme-corp', ...}, {'name': 'research-team', ...}]}
        """
        return {"data": [o.to_dict() for o in self._organizations]}

    def names(self) -> list[str]:
        """Return a list of organization names.

        Returns:
            list[str]: Organization names in the same order as the list.

        Examples:
            >>> from pinecone import Admin
            >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
            >>> orgs = admin.organizations.list()
            >>> orgs.names()  # doctest: +SKIP
            ['acme-corp', 'research-team']
        """
        return [org.name for org in self._organizations]

    def __repr__(self) -> str:
        summaries = ", ".join(f"<name={o.name!r}, plan={o.plan!r}>" for o in self._organizations)
        return f"OrganizationList([{summaries}])"
