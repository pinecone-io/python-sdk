"""Organizations namespace — list, describe, update, and delete operations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pinecone._internal.adapters.admin_adapter import AdminAdapter
from pinecone._internal.validation import require_non_empty
from pinecone.models.admin.organization import OrganizationList, OrganizationModel

if TYPE_CHECKING:
    from pinecone._internal.http_client import HTTPClient

logger = logging.getLogger(__name__)


class Organizations:
    """Control-plane operations for Pinecone organizations.

    Provides methods to list, describe, update, and delete organizations.

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
        """List all organizations accessible to the authenticated user.

        Returns:
            An :class:`OrganizationList` supporting iteration, len(), and index access.

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
        """Get detailed information about an organization.

        Args:
            organization_id (str): The identifier of the organization.

        Returns:
            An :class:`OrganizationModel` with full organization details.

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
        response = self._http.get(f"/admin/organizations/{organization_id}")
        result = self._adapter.to_organization(response.content)
        logger.debug("Described organization %r", organization_id)
        return result

    def update(self, *, organization_id: str, name: str) -> OrganizationModel:
        """Update an organization's name.

        Args:
            organization_id (str): The identifier of the organization to update.
            name (str): The new name for the organization.

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
            f"/admin/organizations/{organization_id}",
            json={"name": name},
        )
        result = self._adapter.to_organization(response.content)
        logger.debug("Updated organization %r", organization_id)
        return result

    def delete(self, *, organization_id: str) -> None:
        """Delete an organization.

        Deletion has three preconditions, and each failure is a 412 whose message
        says which one blocked:

        - The organization must not be on a paid plan; a paid plan must be
          downgraded first.
        - Its payment status must be active, with no open invoices.
        - It must contain no projects (see
          :meth:`Projects.delete <pinecone.admin.projects.Projects.delete>`).

        Having no projects is necessary but not sufficient: an organization on a paid
        plan with no projects still cannot be deleted.

        Args:
            organization_id (str): The identifier of the organization to delete.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`: If *organization_id* is empty.
            :exc:`~pinecone.errors.exceptions.FailedPreconditionError`: If the
                organization is on a paid plan, its payment status is not active, or it
                still contains projects (412). The error names the blocker.
            :exc:`ApiError`: If the API returns an error response.

        Examples:
            >>> admin.organizations.delete(organization_id="org-abc123")
        """
        require_non_empty("organization_id", organization_id)
        logger.info("Deleting organization %r", organization_id)
        self._http.delete(f"/admin/organizations/{organization_id}")
        logger.debug("Deleted organization %r", organization_id)
