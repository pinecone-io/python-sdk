"""ApiKeys namespace — list, create, describe, update, and delete operations."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from pinecone._internal.adapters.admin_adapter import AdminAdapter
from pinecone._internal.validation import require_max_length, require_non_empty
from pinecone.errors.exceptions import ValidationError
from pinecone.models.admin.api_key import APIKeyList, APIKeyModel, APIKeyRole, APIKeyWithSecret

if TYPE_CHECKING:
    from pinecone._internal.http_client import HTTPClient

logger = logging.getLogger(__name__)

_VALID_ROLES = {r.value for r in APIKeyRole}


def _validate_roles(roles: Sequence[APIKeyRole | str]) -> list[APIKeyRole]:
    """Validate each role and return typed enum values."""
    result: list[APIKeyRole] = []
    for index, role in enumerate(roles):
        role_str = role.value if isinstance(role, APIKeyRole) else role
        if role_str not in _VALID_ROLES:
            opts = ", ".join(repr(v) for v in sorted(_VALID_ROLES))
            raise ValidationError(
                f"roles[{index}]: Invalid role {role_str!r}. Must be one of {opts}"
            )
        result.append(APIKeyRole(role_str))
    return result


class ApiKeys:
    """Control-plane operations for Pinecone API keys.

    Provides methods to list, create, describe, update, and delete API keys
    scoped to a project.

    Args:
        http (HTTPClient): HTTP client for making API requests.

    Examples:
        >>> from pinecone import Admin
        >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
        >>> for key in admin.api_keys.list(project_id="proj-abc123"):
        ...     print(key.name)
    """

    def __init__(self, *, http: HTTPClient) -> None:
        self._http = http
        self._adapter = AdminAdapter()

    def __repr__(self) -> str:
        """Return developer-friendly representation."""
        return "ApiKeys()"

    def list(self, *, project_id: str) -> APIKeyList:
        """List all API keys for a project.

        Args:
            project_id (str): The identifier of the project.

        Returns:
            An :class:`APIKeyList` supporting iteration, len(), and index access.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`: If *project_id* is empty.
            :exc:`ApiError`: If the API returns an error response.

        Examples:
            >>> from pinecone import Admin
            >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
            >>> for key in admin.api_keys.list(project_id="proj-abc123"):
            ...     print(key.name)
        """
        require_non_empty("project_id", project_id)
        logger.info("Listing API keys for project %r", project_id)
        response = self._http.get(f"/admin/projects/{project_id}/api-keys")
        result = self._adapter.to_api_key_list(response.content)
        logger.debug("Listed %d API keys", len(result))
        return result

    def create(
        self,
        *,
        project_id: str,
        name: str,
        roles: Sequence[APIKeyRole | str] | None = None,
    ) -> APIKeyWithSecret:
        """Create a new API key for a project.

        Args:
            project_id (str): The identifier of the project.
            name (str): Name for the new API key (1-80 characters).
            roles (list[APIKeyRole | str] | None): Roles to assign to the key.
                Valid values are ``"ProjectEditor"``, ``"ProjectViewer"``,
                ``"ControlPlaneEditor"``, ``"ControlPlaneViewer"``,
                ``"DataPlaneEditor"``, and ``"DataPlaneViewer"``.
                Defaults to ``["ProjectEditor"]`` if omitted.

                Which of these a key may actually hold is plan-dependent: on the
                Free and Builder plans the only accepted role is
                ``"ProjectEditor"``, and any other value is refused with a
                :exc:`~pinecone.errors.exceptions.ForbiddenError`. The error
                names the role and the plan it needs.

        Returns:
            An :class:`APIKeyWithSecret` containing the key metadata and secret value.
            The secret value is only available at creation time.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`:
                If *project_id* or *name* is empty, or if *name* exceeds 80 characters.
            :exc:`~pinecone.errors.exceptions.PaymentRequiredError`: If the organization's
                billing state does not permit creating an API key (402).
            :exc:`~pinecone.errors.exceptions.ForbiddenError`: (403) Either the project has
                reached its API-key quota, or *roles* names a role the organization's plan
                does not permit (see *roles* above). The error message distinguishes the
                two. Note that quota exhaustion on the admin API is a **403, not a 429** —
                :exc:`~pinecone.errors.exceptions.RateLimitError` will not catch it.
            :exc:`ApiError`: If the API returns an error response.

        Examples:
            >>> from pinecone import Admin, APIKeyRole
            >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
            >>> result = admin.api_keys.create(
            ...     project_id="proj-abc123", name="prod-search-key",
            ...     roles=[APIKeyRole.PROJECT_EDITOR]
            ... )
            >>> result.value
            'pcsk_abc123_secretvalue'

            >>> result = admin.api_keys.create(
            ...     project_id="proj-abc123", name="ci-pipeline-key", roles=["ProjectViewer"]
            ... )
            >>> result.key.roles  # doctest: +SKIP
            ['ProjectViewer']
        """
        require_non_empty("project_id", project_id)
        require_non_empty("name", name)
        require_max_length("name", name, 80)
        body: dict[str, Any] = {"name": name}
        if roles is not None:
            body["roles"] = _validate_roles(roles)
        logger.info("Creating API key %r in project %r", name, project_id)
        response = self._http.post(f"/admin/projects/{project_id}/api-keys", json=body)
        result = self._adapter.to_api_key_with_secret(response.content)
        logger.debug("Created API key %r", result.key.id)
        return result

    def describe(self, *, api_key_id: str) -> APIKeyModel:
        """Get detailed information about an API key.

        Args:
            api_key_id (str): The identifier of the API key.

        Returns:
            An :class:`APIKeyModel` with full API key details.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`: If *api_key_id* is empty.
            :exc:`ApiError`: If the API returns an error response.

        Examples:
            >>> from pinecone import Admin
            >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
            >>> key = admin.api_keys.describe(api_key_id="key-abc123")
            >>> key.name
            'prod-search-key'
        """
        require_non_empty("api_key_id", api_key_id)
        logger.info("Describing API key %r", api_key_id)
        response = self._http.get(f"/admin/api-keys/{api_key_id}")
        result = self._adapter.to_api_key(response.content)
        logger.debug("Described API key %r", api_key_id)
        return result

    def update(
        self,
        *,
        api_key_id: str,
        name: str | None = None,
        roles: Sequence[APIKeyRole | str] | None = None,
    ) -> APIKeyModel:
        """Update an API key's settings.

        When *roles* is provided, it replaces the entire role set.

        Args:
            api_key_id (str): The identifier of the API key to update.
            name (str | None): New name for the API key. Unlike :meth:`create`, the
                length limit is not checked locally — an over-long name is rejected
                by the server instead.
            roles (list[APIKeyRole | str] | None): New roles for the API key.
                Replaces all existing roles. Subject to the same plan-dependent
                restriction as :meth:`create`.

        Returns:
            An :class:`APIKeyModel` with the updated API key details.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`: If *api_key_id* is empty.
            :exc:`~pinecone.errors.exceptions.ForbiddenError`: (403) If *roles* names a role
                the organization's plan does not permit for API keys. Unlike
                :meth:`create`, no API-key quota check applies here.
            :exc:`ApiError`: If the API returns an error response.

        Examples:
            >>> from pinecone import Admin
            >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
            >>> key = admin.api_keys.update(
            ...     api_key_id="key-abc123", name="new-name"
            ... )
            >>> key.name  # doctest: +SKIP
            'new-name'
        """
        require_non_empty("api_key_id", api_key_id)
        body: dict[str, Any] = {}
        if name is not None:
            body["name"] = name
        if roles is not None:
            body["roles"] = _validate_roles(roles)
        logger.info("Updating API key %r", api_key_id)
        response = self._http.patch(f"/admin/api-keys/{api_key_id}", json=body)
        result = self._adapter.to_api_key(response.content)
        logger.debug("Updated API key %r", api_key_id)
        return result

    def delete(self, *, api_key_id: str) -> None:
        """Delete an API key.

        Args:
            api_key_id (str): The identifier of the API key to delete.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`: If *api_key_id* is empty.
            :exc:`ApiError`: If the API returns an error response.

        Examples:
            >>> from pinecone import Admin
            >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
            >>> admin.api_keys.delete(api_key_id="key-abc123")
        """
        require_non_empty("api_key_id", api_key_id)
        logger.info("Deleting API key %r", api_key_id)
        self._http.delete(f"/admin/api-keys/{api_key_id}")
        logger.debug("Deleted API key %r", api_key_id)
