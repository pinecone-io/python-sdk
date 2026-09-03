"""ApiKeys namespace — list, create, describe, update, and delete operations."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

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
    """Operations on Pinecone API keys.

    An API key is a project-scoped credential: it is the thing you pass to
    :class:`~pinecone.Pinecone` to read and write indexes in one project. Where a service
    account authenticates an :class:`~pinecone.Admin` client against the whole
    organization, an API key reaches exactly one project. Not constructed directly — reach
    it as ``admin.api_keys``.

    :meth:`create` is the only call that returns a key's secret, and it returns it once.

    Examples:
        >>> from pinecone import Admin
        >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
        >>> for key in admin.api_keys.list(project_id="proj-abc123"):
        ...     print(key.name, key.roles)

    .. seealso::
       :class:`~pinecone.admin.service_accounts.ServiceAccounts` — the organization-scoped
       OAuth credentials an :class:`~pinecone.Admin` client itself uses.

       :doc:`/guides/error-handling` — what each exception these calls raise means.
    """

    def __init__(self, *, http: HTTPClient) -> None:
        self._http = http
        self._adapter = AdminAdapter()

    def __repr__(self) -> str:
        """Return developer-friendly representation."""
        return "ApiKeys()"

    def list(self, *, project_id: str) -> APIKeyList:
        """List the API keys belonging to a project.

        Secrets are never returned here; only :meth:`create` carries one.

        Args:
            project_id (str): The project's identifier, e.g. ``"proj-abc123"``.

        Returns:
            An :class:`APIKeyList` of every key in the project, supporting iteration,
            ``len()``, and index access. Returned whole — there is no paging.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`: If *project_id* is empty
                or whitespace-only. Checked before the request is sent.

        Examples:
            >>> for key in admin.api_keys.list(project_id="proj-abc123"):
            ...     print(key.name, key.roles)
        """
        require_non_empty("project_id", project_id)
        logger.info("Listing API keys for project %r", project_id)
        response = self._http.get(f"/admin/projects/{quote(project_id, safe='')}/api-keys")
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
        """Create an API key scoped to one project.

        The response is the only place the key's secret ever appears — ``value`` is
        returned here and nowhere else, and no call recovers it later. Store it before
        doing anything else; if you lose it, delete the key and create another.

        Args:
            project_id (str): The project the key will reach, e.g. ``"proj-abc123"``.
            name (str): Label for the key, e.g. ``"prod-search-key"``; 1-80 characters,
                checked client-side.
            roles (list[APIKeyRole | str] | None): Roles the key holds. Valid values are
                ``"ProjectEditor"``, ``"ProjectViewer"``, ``"ControlPlaneEditor"``,
                ``"ControlPlaneViewer"``, ``"DataPlaneEditor"``, and
                ``"DataPlaneViewer"``, either as strings or as
                :class:`~pinecone.APIKeyRole` members. Defaults to ``["ProjectEditor"]``.
                A role the organization is not entitled to grant is refused even though
                the name is valid; see *Raises*.

        Returns:
            An :class:`APIKeyWithSecret` with ``value`` (the secret, this once only) and
            ``key`` (an :class:`APIKeyModel` carrying the key's ``id``, ``name``, and
            ``roles``). Pass ``value`` to :class:`~pinecone.Pinecone`; keep ``key.id`` to
            reach the key again through :meth:`describe`, :meth:`update`, or
            :meth:`delete`.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`: If *project_id* or
                *name* is empty, if *name* is longer than 80 characters, or if *roles*
                contains a value that is not one of the six role names. All checked
                before the request is sent.
            :exc:`~pinecone.errors.exceptions.PaymentRequiredError`: If the organization's
                billing state does not permit creating an API key.
            :exc:`~pinecone.errors.exceptions.ForbiddenError`: Either the project has
                reached its API-key quota or *roles* names a role the organization cannot
                grant — the error message distinguishes the two. A full quota surfaces
                here rather than as :exc:`~pinecone.errors.exceptions.RateLimitError`, so
                do not retry it.

        Examples:
            >>> from pinecone import APIKeyRole
            >>> result = admin.api_keys.create(
            ...     project_id="proj-abc123", name="prod-search-key",
            ...     roles=[APIKeyRole.DATA_PLANE_EDITOR]
            ... )
            >>> result.value
            'pcsk_abc123_secretvalue'
            >>> result.key.roles
            [<APIKeyRole.DATA_PLANE_EDITOR: 'DataPlaneEditor'>]

            The secret is what the data-plane client authenticates with, so this is where
            an admin workflow hands off to :class:`~pinecone.Pinecone`:

            >>> from pinecone import Pinecone
            >>> pc = Pinecone(api_key=result.value)
            >>> for index in pc.indexes.list():
            ...     print(index.name)
        """
        require_non_empty("project_id", project_id)
        require_non_empty("name", name)
        require_max_length("name", name, 80)
        body: dict[str, Any] = {"name": name}
        if roles is not None:
            body["roles"] = _validate_roles(roles)
        logger.info("Creating API key %r in project %r", name, project_id)
        response = self._http.post(
            f"/admin/projects/{quote(project_id, safe='')}/api-keys", json=body
        )
        result = self._adapter.to_api_key_with_secret(response.content)
        logger.debug("Created API key %r", result.key.id)
        return result

    def describe(self, *, api_key_id: str) -> APIKeyModel:
        """Get one API key's metadata.

        The secret is not part of it; only :meth:`create` ever returns that.

        Args:
            api_key_id (str): The key's identifier — ``key.id`` from :meth:`create` or
                :meth:`list`, e.g. ``"key-abc123"``. This is not the secret.

        Returns:
            An :class:`APIKeyModel` with the key's ``id``, ``name``, ``project_id``, and
            ``roles``.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`: If *api_key_id* is empty
                or whitespace-only. Checked before the request is sent.

        Examples:
            >>> key = admin.api_keys.describe(api_key_id="key-abc123")
            >>> key.name
            'prod-search-key'
            >>> key.roles
            [<APIKeyRole.DATA_PLANE_EDITOR: 'DataPlaneEditor'>]
        """
        require_non_empty("api_key_id", api_key_id)
        logger.info("Describing API key %r", api_key_id)
        response = self._http.get(f"/admin/api-keys/{quote(api_key_id, safe='')}")
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
        """Change an API key's name or roles.

        Omitted arguments are left alone, but *roles* is not merged: passing it replaces
        the whole role set, so include every role the key should keep. The secret does not
        change, so callers holding it keep working under the new roles.

        Args:
            api_key_id (str): The key's identifier, e.g. ``"key-abc123"``. Left unchanged
                by this call.
            name (str | None): New label for the key, e.g. ``"prod-search-key-v2"``. Left
                unchanged if omitted. Unlike :meth:`create`, the length limit is not
                checked client-side — an over-long name is rejected by the server.
            roles (list[APIKeyRole | str] | None): The key's complete new role set, from
                the same six values :meth:`create` accepts. Left unchanged if omitted, and
                subject to the same entitlement restriction.

        Returns:
            An :class:`APIKeyModel` reflecting the stored state after the change.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`: If *api_key_id* is empty
                or whitespace-only, or if *roles* contains a value that is not one of the
                six role names. Both checked before the request is sent.
            :exc:`~pinecone.errors.exceptions.ForbiddenError`: If *roles* names a role the
                organization cannot grant. Unlike :meth:`create`, no API-key quota applies
                here — the key already exists.

        Examples:
            >>> key = admin.api_keys.update(
            ...     api_key_id="key-abc123", roles=["DataPlaneEditor", "DataPlaneViewer"]
            ... )
        """
        require_non_empty("api_key_id", api_key_id)
        body: dict[str, Any] = {}
        if name is not None:
            body["name"] = name
        if roles is not None:
            body["roles"] = _validate_roles(roles)
        logger.info("Updating API key %r", api_key_id)
        response = self._http.patch(f"/admin/api-keys/{quote(api_key_id, safe='')}", json=body)
        result = self._adapter.to_api_key(response.content)
        logger.debug("Updated API key %r", api_key_id)
        return result

    def delete(self, *, api_key_id: str) -> None:
        """Delete an API key permanently.

        Anything still authenticating with the key's secret starts failing, and there is
        no way to restore it — a replacement is a new :meth:`create` with a new secret.

        Args:
            api_key_id (str): The key's identifier, e.g. ``"key-abc123"``, not the secret.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`: If *api_key_id* is empty
                or whitespace-only. Checked before the request is sent.

        Examples:
            >>> admin.api_keys.delete(api_key_id="key-abc123")
        """
        require_non_empty("api_key_id", api_key_id)
        logger.info("Deleting API key %r", api_key_id)
        self._http.delete(f"/admin/api-keys/{quote(api_key_id, safe='')}")
        logger.debug("Deleted API key %r", api_key_id)
