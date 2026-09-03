"""Organizations namespace — list, describe, update, and delete operations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from urllib.parse import quote

from pinecone._internal.adapters.admin_adapter import AdminAdapter
from pinecone._internal.validation import require_non_empty
from pinecone.models.admin.organization import OrganizationList, OrganizationModel

if TYPE_CHECKING:
    from pinecone._internal.http_client import HTTPClient

logger = logging.getLogger(__name__)


class Organizations:
    """Operations on Pinecone organizations.

    An organization is the top-level account boundary in Pinecone: it holds
    projects, users, and billing. This namespace lists, describes, updates,
    and deletes organizations.

    Args:
        http (HTTPClient): HTTP client for making API requests.

    Examples:

        >>> from pinecone import Admin
        >>> admin = Admin(client_id="my-id", client_secret="my-secret")
        >>> for org in admin.organizations.list():
        ...     print(org.name)
    """

    def __init__(self, *, http: HTTPClient) -> None:
        self._http = http
        self._adapter = AdminAdapter()

    def __repr__(self) -> str:
        """Return developer-friendly representation."""
        return "Organizations()"

    def list(self) -> OrganizationList:
        """List the organizations your credentials can access.

        Returns:
            An :class:`OrganizationList` supporting iteration, ``len()``, and
            index access.

        Raises:
            :exc:`ApiError`: If the API returns an error response.

        Examples:
            >>> admin = Admin(client_id="my-id", client_secret="my-secret")
            >>> for org in admin.organizations.list():
            ...     print(org.name)
        """
        logger.info("Listing organizations")
        response = self._http.get("/admin/organizations")
        result = self._adapter.to_organization_list(response.content)
        logger.debug("Listed %d organizations", len(result))
        return result

    def describe(self, *, organization_id: str) -> OrganizationModel:
        """Get details for one organization.

        Args:
            organization_id (str): The organization's identifier, e.g. ``"org-abc123"``.

        Returns:
            An :class:`OrganizationModel` with the organization's name, plan,
            payment status, support tier, and creation time.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`: If *organization_id* is empty.
            :exc:`ApiError`: If the API returns an error response.

        Examples:
            >>> org = admin.organizations.describe(organization_id="org-abc123")
            >>> org.name
            'Acme Corp'
        """
        require_non_empty("organization_id", organization_id)
        logger.info("Describing organization %r", organization_id)
        response = self._http.get(f"/admin/organizations/{quote(organization_id, safe='')}")
        result = self._adapter.to_organization(response.content)
        logger.debug("Described organization %r", organization_id)
        return result

    def update(self, *, organization_id: str, name: str) -> OrganizationModel:
        """Rename an organization.

        Args:
            organization_id (str): The organization's identifier, e.g. ``"org-abc123"``.
            name (str): The new name for the organization, e.g. ``"Acme Corp"``.

        Returns:
            An :class:`OrganizationModel` with the updated organization details.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`:
                If *organization_id* is empty.
            :exc:`ApiError`: If the API returns an error response.

        Examples:
            >>> org = admin.organizations.update(
            ...     organization_id="org-abc123", name="New Name"
            ... )
            >>> org.name  # doctest: +SKIP
            'New Name'
        """
        require_non_empty("organization_id", organization_id)
        logger.info("Updating organization %r", organization_id)
        response = self._http.patch(
            f"/admin/organizations/{quote(organization_id, safe='')}",
            json={"name": name},
        )
        result = self._adapter.to_organization(response.content)
        logger.debug("Updated organization %r", organization_id)
        return result

    def delete(self, *, organization_id: str) -> None:
        """Delete an organization.

        An organization must meet three conditions before it can be deleted:

        - It is not on a paid plan (downgrade first).
        - Its payment status is active, with no open invoices.
        - It contains no projects (see
          :meth:`Projects.delete <pinecone.admin.projects.Projects.delete>`).

        All three must hold at once — an organization with no projects can
        still be blocked by its plan or payment status.

        Args:
            organization_id (str): The organization's identifier, e.g. ``"org-abc123"``.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`: If *organization_id* is empty.
            :exc:`~pinecone.errors.exceptions.FailedPreconditionError`: If the
                organization is on a paid plan, its payment status is not active, or it
                still contains projects. The error message names the blocker.
            :exc:`ApiError`: If the API returns an error response.

        Examples:
            >>> admin.organizations.delete(organization_id="org-abc123")
        """
        require_non_empty("organization_id", organization_id)
        logger.info("Deleting organization %r", organization_id)
        self._http.delete(f"/admin/organizations/{quote(organization_id, safe='')}")
        logger.debug("Deleted organization %r", organization_id)
