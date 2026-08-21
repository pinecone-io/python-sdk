"""Projects namespace — list, create, describe, update, and delete operations."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from pinecone._internal.adapters.admin_adapter import AdminAdapter
from pinecone._internal.validation import require_non_empty
from pinecone.errors.exceptions import (
    ForbiddenError,
    NotFoundError,
    PineconeError,
    PineconeValueError,
    ValidationError,
)
from pinecone.models.admin.api_key import APIKeyRole
from pinecone.models.admin.project import ProjectList, ProjectModel

if TYPE_CHECKING:
    from pinecone._internal.http_client import HTTPClient
    from pinecone.admin.admin import Admin

logger = logging.getLogger(__name__)


class Projects:
    """Control-plane operations for Pinecone projects.

    Provides methods to list, create, describe, update, and delete projects.

    Args:
        http (HTTPClient): HTTP client for making API requests.

    Examples:
        >>> from pinecone import Admin
        >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
        >>> for project in admin.projects.list():
        ...     print(project.name)
    """

    def __init__(self, *, http: HTTPClient, admin: Admin | None = None) -> None:
        self._http = http
        self._adapter = AdminAdapter()
        self._admin = admin

    def __repr__(self) -> str:
        """Return developer-friendly representation."""
        return "Projects()"

    def list(self) -> ProjectList:
        """List all projects accessible to the authenticated user.

        Returns:
            A :class:`ProjectList` supporting iteration, len(), and index access.

        Raises:
            :exc:`ApiError`: If the API returns an error response.

        Examples:
            >>> from pinecone import Admin
            >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
            >>> for project in admin.projects.list():
            ...     print(project.name)
        """
        logger.info("Listing projects")
        response = self._http.get("/admin/projects")
        result = self._adapter.to_project_list(response.content)
        logger.debug("Listed %d projects", len(result))
        return result

    def create(
        self,
        *,
        name: str,
        max_pods: int | None = None,
        force_encryption_with_cmek: bool | None = None,
    ) -> ProjectModel:
        """Create a new project.

        Args:
            name (str): Name for the new project.
            max_pods (int | None): Maximum number of pods allowed. Omitted if None.
                Pod-based capacity is legacy: unless the organization has
                pre-existing pod access, only ``0`` — the default, meaning
                serverless-only — is accepted, and a non-zero value is rejected with
                a 400 explaining what to do.
            force_encryption_with_cmek (bool | None): Whether to enforce CMEK encryption.
                Omitted if None. Requesting ``True`` requires CMEK to be enabled for
                the organization; otherwise the request is refused with a
                :exc:`~pinecone.errors.exceptions.ForbiddenError`. ``False`` and
                ``None`` are always accepted.

        Returns:
            A :class:`ProjectModel` with the created project details.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`: If *name* is empty.
            :exc:`~pinecone.errors.exceptions.PaymentRequiredError`: If the organization's
                billing state does not permit creating a project (402).
            :exc:`~pinecone.errors.exceptions.ForbiddenError`: (403) Either the organization
                has reached its project quota, or *force_encryption_with_cmek* was
                requested without CMEK enabled. As elsewhere on the admin API, quota
                exhaustion is a **403, not a 429**.
            :exc:`ApiError`: If the API returns an error response — including the 400
                raised for a non-zero *max_pods* without pod access.

        Examples:
            >>> from pinecone import Admin
            >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
            >>> project = admin.projects.create(name="my-project")
            >>> project.name
            'my-project'
        """
        require_non_empty("name", name)
        if len(name) > 512:
            raise PineconeValueError("name cannot be longer than 512 characters")
        if "\x00" in name:
            raise PineconeValueError("name cannot contain null characters")
        if max_pods is not None and max_pods < 0:
            raise ValidationError("max_pods must be a non-negative integer")
        body: dict[str, Any] = {"name": name}
        if max_pods is not None:
            body["max_pods"] = max_pods
        if force_encryption_with_cmek is not None:
            body["force_encryption_with_cmek"] = force_encryption_with_cmek
        logger.info("Creating project %r", name)
        response = self._http.post("/admin/projects", json=body)
        result = self._adapter.to_project(response.content)
        logger.debug("Created project %r", result.id)
        return result

    def describe(self, *, project_id: str) -> ProjectModel:
        """Get detailed information about a project.

        Args:
            project_id (str): The identifier of the project.

        Returns:
            A :class:`ProjectModel` with full project details.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`: If *project_id* is empty.
            :exc:`ApiError`: If the API returns an error response.

        Examples:
            >>> from pinecone import Admin
            >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
            >>> project = admin.projects.describe(project_id="proj-abc123")
            >>> project.name
            'my-project'
        """
        require_non_empty("project_id", project_id)
        logger.info("Describing project %r", project_id)
        response = self._http.get(f"/admin/projects/{project_id}")
        result = self._adapter.to_project(response.content)
        logger.debug("Described project %r", project_id)
        return result

    def describe_by_name(self, *, name: str) -> ProjectModel:
        """Get detailed information about a project by name.

        Lists all projects and filters client-side for an exact name match.

        Args:
            name (str): The name of the project.

        Returns:
            A :class:`ProjectModel` with full project details.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`: If *name* is empty.
            :exc:`NotFoundError`: If no project matches *name*.
            :exc:`PineconeError`: If multiple projects share *name*.

        Examples:
            .. code-block:: python

                from pinecone import Admin
                admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
                project = admin.projects.describe_by_name(name="my-project")
                project.id  # 'proj-abc123'
        """
        require_non_empty("name", name)
        logger.info("Describing project by name %r", name)
        projects = self.list()
        matches = [p for p in projects if p.name == name]
        if len(matches) == 0:
            raise NotFoundError(message=f"No project found with name {name!r}")
        if len(matches) > 1:
            raise PineconeError(
                f"Multiple projects found with name {name!r}; use project_id instead"
            )
        logger.debug("Found project %r by name %r", matches[0].id, name)
        return matches[0]

    def exists(
        self,
        *,
        project_id: str | None = None,
        name: str | None = None,
    ) -> bool:
        """Check whether a project exists.

        Exactly one of *project_id* or *name* must be provided.

        Args:
            project_id (str | None): The identifier of the project.
            name (str | None): The name of the project.

        Returns:
            ``True`` if the project exists, ``False`` otherwise.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`:
                If neither or both arguments are provided.

        Examples:
            >>> from pinecone import Admin
            >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
            >>> admin.projects.exists(project_id="proj-abc123")
            True
            >>> admin.projects.exists(name="nonexistent")
            False
        """
        if (project_id is None) == (name is None):
            raise ValidationError("Exactly one of 'project_id' or 'name' must be provided")
        try:
            if project_id is not None:
                self.describe(project_id=project_id)
            elif name is not None:
                self.describe_by_name(name=name)
        except NotFoundError:
            return False
        except PineconeError:
            # Multiple projects with same name — they exist
            return True
        return True

    def update(
        self,
        *,
        project_id: str,
        name: str | None = None,
        max_pods: int | None = None,
        force_encryption_with_cmek: bool | None = None,
    ) -> ProjectModel:
        """Update a project's settings.

        Args:
            project_id (str): The identifier of the project to update.
            name (str | None): New name for the project.
            max_pods (int | None): New maximum pod count. Subject to the same
                pod-access constraint as :meth:`create`.
            force_encryption_with_cmek (bool | None): New CMEK enforcement setting.
                Enabling it requires the same entitlement as :meth:`create`. CMEK is a
                one-way door: once a project has it enabled it cannot be turned back
                off, and attempting to is rejected with a 400. Passing ``False`` for a
                project that never had it enabled is a no-op.

        Returns:
            A :class:`ProjectModel` with the updated project details.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`: If *project_id* is empty,
                or if *name* is empty, exceeds 512 characters, or contains null bytes.
            :exc:`~pinecone.errors.exceptions.ForbiddenError`: (403) If
                *force_encryption_with_cmek* is ``True`` and CMEK is not enabled for
                the organization.
            :exc:`ApiError`: If the API returns an error response — including the 400s for
                a non-zero *max_pods* without pod access and for attempting to turn CMEK
                back off.

        Examples:
            >>> from pinecone import Admin
            >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
            >>> project = admin.projects.update(
            ...     project_id="proj-abc123", name="new-name"
            ... )
            >>> project.name  # doctest: +SKIP
            'new-name'
        """
        require_non_empty("project_id", project_id)
        if max_pods is not None and max_pods < 0:
            raise PineconeValueError("max_pods must be a non-negative integer")
        body: dict[str, Any] = {}
        if name is not None:
            if len(name) == 0:
                raise PineconeValueError("name cannot be empty")
            if len(name) > 512:
                raise PineconeValueError("name cannot be longer than 512 characters")
            if "\x00" in name:
                raise PineconeValueError("name cannot contain null characters")
            body["name"] = name
        if max_pods is not None:
            body["max_pods"] = max_pods
        if force_encryption_with_cmek is not None:
            body["force_encryption_with_cmek"] = force_encryption_with_cmek
        logger.info("Updating project %r", project_id)
        response = self._http.patch(
            f"/admin/projects/{project_id}",
            json=body,
        )
        result = self._adapter.to_project(response.content)
        logger.debug("Updated project %r", project_id)
        return result

    def _cleanup_project_resources(self, *, api_key: str) -> None:
        """Delete every index, collection, assistant, and backup in the project scoped to *api_key*.

        This is the inner loop of the project-deletion-with-cleanup workflow.

        Each deletion waits for the resource to actually disappear rather than
        just acknowledging the request, because a resource that is still
        winding down continues to block the project delete.

        Each deletion is wrapped in a try/except for :exc:`NotFoundError` to
        handle race conditions where a resource is deleted between the list
        and delete calls.

        Args:
            api_key: A Pinecone API key scoped to the target project.
        """
        from pinecone._client import Pinecone

        pc = Pinecone(api_key=api_key)
        try:
            # Delete all indexes
            for index in pc.indexes.list():
                try:
                    logger.debug("Cleanup: deleting index %r", index.name)
                    pc.indexes.delete(index.name)
                except NotFoundError:
                    logger.debug("Cleanup: index %r already deleted", index.name)

            # Delete all collections
            for collection in pc.collections.list():
                try:
                    logger.debug("Cleanup: deleting collection %r", collection.name)
                    pc.collections.delete(collection.name)
                except NotFoundError:
                    logger.debug("Cleanup: collection %r already deleted", collection.name)

            for assistant in pc.assistants.list():
                try:
                    logger.debug("Cleanup: deleting assistant %r", assistant.name)
                    pc.assistants.delete(name=assistant.name)
                except NotFoundError:
                    logger.debug("Cleanup: assistant %r already deleted", assistant.name)

            # Delete all backups
            for backup in pc.backups.list():
                try:
                    logger.debug("Cleanup: deleting backup %r", backup.backup_id)
                    pc.backups.delete(backup_id=backup.backup_id)
                except NotFoundError:
                    logger.debug("Cleanup: backup %r already deleted", backup.backup_id)
        finally:
            pc.close()

    def delete_with_cleanup(
        self,
        *,
        project_id: str,
        max_attempts: int = 5,
        retry_delay: float = 30.0,
    ) -> None:
        """Delete a project after cleaning up all its resources.

        Creates a temporary API key scoped to the project, uses it to delete
        every index, collection, assistant, and backup, then deletes the
        temporary key and finally deletes the project itself.

        The cleanup is retried up to *max_attempts* times with *retry_delay*
        seconds between attempts to handle transient failures.

        Creating the temporary key is the first thing this method does, so a
        project whose API-key quota is already full cannot be cleaned up: the
        403 is re-raised with the quota named as the blocker and nothing is
        deleted. Free a key slot and call again.

        Cleanup covers every resource that blocks a project delete. It is not
        atomic, though: a resource created in the project while cleanup is
        running can still leave the final delete blocked.

        Args:
            project_id: The identifier of the project to delete.
            max_attempts: Maximum number of cleanup attempts. Defaults to 5.
            retry_delay: Seconds to wait between retry attempts. Defaults to 30.0.

        Raises:
            :exc:`PineconeError`: If no admin back-reference is available.
            :exc:`~pinecone.errors.exceptions.PineconeValueError`: If *project_id* is empty.
            :exc:`~pinecone.errors.exceptions.ForbiddenError`: If the temporary API key
                cannot be created — typically because the project's API-key quota is
                exhausted. No resources are deleted in this case.
            :exc:`~pinecone.errors.exceptions.FailedPreconditionError`: A 412 raised
                when the project is still not empty as the final delete runs, which
                happens when something is created in it after cleanup. The error names
                what is blocking.
            :exc:`ApiError`: If resource cleanup or project deletion fails after all retries.

        Examples:
            .. code-block:: python

                from pinecone import Admin
                admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
                admin.projects.delete_with_cleanup(project_id="proj-abc123")
        """
        if self._admin is None:
            raise PineconeError(
                "delete_with_cleanup requires an Admin back-reference. "
                "Use admin.projects.delete_with_cleanup() instead of "
                "constructing Projects directly."
            )
        require_non_empty("project_id", project_id)

        logger.info("Deleting project %r with cleanup (max_attempts=%d)", project_id, max_attempts)

        try:
            temp_key = self._admin.api_keys.create(
                project_id=project_id,
                name="_cleanup_temp_key",
                roles=[APIKeyRole.PROJECT_EDITOR],
            )
        except ForbiddenError as exc:
            raise ForbiddenError(
                "delete_with_cleanup could not create the temporary API key it needs to "
                f"clean up project_id={project_id!r} (server said: {exc.message}). The "
                "usual blocker is the per-project API-key quota: a project already at its "
                "limit has no free slot for the temporary key. Nothing was deleted — the "
                "project and every resource in it are untouched. To proceed, free a slot "
                f"with admin.api_keys.list(project_id={project_id!r}) followed by "
                "admin.api_keys.delete(api_key_id=...), then call delete_with_cleanup "
                "again. If the quota is not the blocker, the credentials in use lack "
                "permission to create API keys in this project.",
                status_code=exc.status_code,
                body=exc.body,
                reason=exc.reason,
                headers=exc.headers,
                error_code=exc.error_code,
                request_id=exc.request_id,
            ) from exc

        try:
            last_error: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    logger.debug(
                        "Cleanup attempt %d/%d for project %r",
                        attempt,
                        max_attempts,
                        project_id,
                    )
                    self._cleanup_project_resources(api_key=temp_key.value)
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    logger.warning(
                        "Cleanup attempt %d/%d failed for project %r: %s",
                        attempt,
                        max_attempts,
                        project_id,
                        exc,
                    )
                    if attempt < max_attempts:
                        time.sleep(retry_delay)

            if last_error is not None:
                raise last_error
        finally:
            try:
                self._admin.api_keys.delete(api_key_id=temp_key.key.id)
            except Exception:
                logger.warning(
                    "Failed to delete temporary cleanup key %r for project %r; "
                    "delete it manually via admin.api_keys.delete(api_key_id=%r)",
                    temp_key.key.id,
                    project_id,
                    temp_key.key.id,
                )

        self.delete(project_id=project_id)

    def delete(self, *, project_id: str) -> None:
        """Delete a project.

        The project must be empty first. Indexes, collections, assistants, and
        backups all block deletion, and the error names what is still there. API
        keys are *not* a blocker — they are deleted along with the project.

        :meth:`delete_with_cleanup` clears all of them for you.

        Args:
            project_id (str): The identifier of the project to delete.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`: If *project_id* is empty.
            :exc:`~pinecone.errors.exceptions.FailedPreconditionError`: (412) If the project
                still owns indexes, collections, assistants, or backups. The error names
                what is blocking.
            :exc:`ApiError`: If the API returns an error response.

        Examples:
            >>> from pinecone import Admin
            >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
            >>> admin.projects.delete(project_id="proj-abc123")
        """
        require_non_empty("project_id", project_id)
        logger.info("Deleting project %r", project_id)
        self._http.delete(f"/admin/projects/{project_id}")
        logger.debug("Deleted project %r", project_id)
