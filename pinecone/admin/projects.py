"""Projects namespace — list, create, describe, update, and delete operations."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

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
    """Operations on Pinecone projects.

    A project is the quota and credential boundary inside an organization: every index,
    collection, backup, assistant, and API key belongs to exactly one project. Where an
    organization scopes members and billing, a project scopes the resources you actually
    query. Not constructed directly — reach it as ``admin.projects``.

    Examples:
        >>> from pinecone import Admin
        >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
        >>> for project in admin.projects.list():
        ...     print(project.name)

    .. seealso::
       :class:`~pinecone.admin.api_keys.ApiKeys` — the keys that let
       :class:`~pinecone.Pinecone` reach a project's data.

       :doc:`/guides/error-handling` — what each exception these calls raise means.
    """

    def __init__(self, *, http: HTTPClient, admin: Admin | None = None) -> None:
        self._http = http
        self._adapter = AdminAdapter()
        self._admin = admin

    def __repr__(self) -> str:
        """Return developer-friendly representation."""
        return "Projects()"

    def list(self) -> ProjectList:
        """List the projects your credentials can reach.

        Returns:
            A :class:`ProjectList` of every reachable project, supporting iteration,
            ``len()``, and index access. Returned whole — there is no paging.

        Examples:
            >>> for project in admin.projects.list():
            ...     print(project.name, project.id)
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
        """Create a project in the organization your credentials belong to.

        The project starts empty; create an API key in it with
        :meth:`~pinecone.admin.api_keys.ApiKeys.create` before any index work can reach it.

        Args:
            name (str): Name for the project, e.g. ``"product-search"``; 1-512 characters
                and no null bytes, both checked client-side. Names need not be unique
                within an organization, which is why :meth:`describe_by_name` can find
                more than one.
            max_pods (int | None): Pod ceiling for the project. Pod-based capacity is
                legacy and a non-zero value is rejected unless the organization has pod
                access; ``0`` means serverless-only. Omitted from the request if ``None``.
            force_encryption_with_cmek (bool | None): Require customer-managed encryption
                keys for everything in the project. Requesting ``True`` needs CMEK enabled
                for the organization, and it cannot be turned back off later — see
                :meth:`update`. Omitted from the request if ``None``.

        Returns:
            A :class:`ProjectModel` with the new project's ``id`` — the value every other
            project and API-key call takes — plus its ``name``, quotas, and
            ``organization_id``.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`: If *name* is empty,
                longer than 512 characters, or contains a null byte, or if *max_pods* is
                negative. All checked before the request is sent.
            :exc:`~pinecone.errors.exceptions.PaymentRequiredError`: If the organization's
                billing state does not permit creating a project.
            :exc:`~pinecone.errors.exceptions.ForbiddenError`: If the organization has
                reached its project quota, or if *force_encryption_with_cmek* was
                requested without CMEK enabled for the organization.
            :exc:`ApiError`: If a non-zero *max_pods* was requested without pod access.

        Examples:
            >>> project = admin.projects.create(name="product-search")
            >>> project.id
            'proj-abc123'
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
        """Get details for one project.

        Args:
            project_id (str): The project's identifier, e.g. ``"proj-abc123"``.

        Returns:
            A :class:`ProjectModel` with the project's name, quotas, and
            organization.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`: If *project_id* is empty
                or whitespace-only. Checked before the request is sent.

        Examples:
            >>> project = admin.projects.describe(project_id="proj-abc123")
            >>> project.name
            'my-project'
        """
        require_non_empty("project_id", project_id)
        logger.info("Describing project %r", project_id)
        response = self._http.get(f"/admin/projects/{quote(project_id, safe='')}")
        result = self._adapter.to_project(response.content)
        logger.debug("Described project %r", project_id)
        return result

    def describe_by_name(self, *, name: str) -> ProjectModel:
        """Get details for one project by name.

        Project names are not unique, so this is a client-side convenience, not a lookup
        the API offers: it fetches every reachable project and filters for an exact,
        case-sensitive name match. Prefer :meth:`describe` with a ``project_id`` in code
        that runs often or in an organization with many projects.

        Args:
            name (str): The project's name, e.g. ``"my-project"``.

        Returns:
            A :class:`ProjectModel` with the project's name, quotas, and
            organization.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`: If *name* is empty or
                whitespace-only. Checked before the request is sent.
            :exc:`NotFoundError`: If no reachable project has that exact name. Raised by
                the client after the listing comes back, so the message names *name*
                rather than any URL.
            :exc:`PineconeError`: If more than one project shares *name*. Nothing
                disambiguates them here — use :meth:`describe` with the ``project_id`` you
                want.

        Examples:
            .. code-block:: python

                project = admin.projects.describe_by_name(name="product-search")
                print(project.id)
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

        Pass exactly one of *project_id* or *name*. Only a definite "no such project"
        gives ``False``; every other failure the lookup hits — no permission to read the
        project, a connection problem, a server error — is currently reported as ``True``
        as well, so treat a ``True`` as "not known to be absent" and do not use this as an
        authorization check. Several projects sharing *name* also give ``True``, since
        they all exist.

        Args:
            project_id (str | None): The project's identifier, e.g. ``"proj-abc123"``.
            name (str | None): The project's name, e.g. ``"product-search"``. Matched
                exactly and case-sensitively, as in :meth:`describe_by_name`.

        Returns:
            ``True`` if the project exists, ``False`` if nothing matches.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`: If both arguments are
                given, or neither.

        Examples:
            >>> admin.projects.exists(project_id="proj-abc123")
            True
            >>> admin.projects.exists(name="archived-catalog")
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
        """Change a project's name, pod ceiling, or CMEK enforcement.

        Omitted arguments are left alone. Renaming a project does not change its ``id``,
        so API keys and index hosts keep working.

        Args:
            project_id (str): The project to update, e.g. ``"proj-abc123"``.
            name (str | None): New name, e.g. ``"product-search-eu"``; the same 1-512
                characters and no-null-bytes rule :meth:`create` applies. Left unchanged
                if omitted.
            max_pods (int | None): New pod ceiling, under the same pod-access constraint
                as :meth:`create`. Left unchanged if omitted.
            force_encryption_with_cmek (bool | None): New CMEK enforcement setting.
                Enabling it needs the same entitlement as :meth:`create`, and it is a
                one-way door — a project with CMEK on cannot have it turned off, while
                passing ``False`` for a project that never had it on is a no-op. Left
                unchanged if omitted.

        Returns:
            A :class:`ProjectModel` reflecting the stored state after the change.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`: If *project_id* is
                empty, if *name* is given but empty, longer than 512 characters, or
                contains a null byte, or if *max_pods* is negative. All checked before the
                request is sent.
            :exc:`~pinecone.errors.exceptions.ForbiddenError`: If
                *force_encryption_with_cmek* is ``True`` and CMEK is not enabled for the
                organization.
            :exc:`ApiError`: If a non-zero *max_pods* was requested without pod access, or
                if the call tries to turn CMEK back off.

        Examples:
            >>> project = admin.projects.update(
            ...     project_id="proj-abc123", name="product-search-eu"
            ... )
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
            f"/admin/projects/{quote(project_id, safe='')}",
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
        """Empty a project of every resource, then delete it permanently.

        Destructive and unattended: it creates a temporary API key scoped to the project,
        uses that key to delete every index, collection, assistant, and backup in it,
        deletes the temporary key, and finally deletes the project. Nothing is recoverable
        afterwards. It also blocks for as long as the deletions take, retrying failed
        cleanup passes up to *max_attempts* times with *retry_delay* seconds between them.

        Creating that temporary key is the first thing it does, so a project whose API-key
        quota is already full cannot be cleaned up at all — the error names the quota and
        nothing has been deleted. Free a key slot and call again.

        Cleanup is not atomic. It covers every resource kind that blocks a project delete,
        but anything created in the project while cleanup is running can still leave the
        final delete blocked.

        Args:
            project_id (str): The project to empty and delete, e.g. ``"proj-abc123"``.
            max_attempts (int): How many times to retry the whole cleanup pass before
                giving up. Defaults to 5; the default is fine unless the project is large
                enough that resources routinely take longer than the retries allow.
            retry_delay (float): Seconds to wait between cleanup attempts. Defaults to
                30.0, which paces the retries against resources that are still winding
                down.

        Raises:
            :exc:`PineconeError`: If no admin back-reference is available — call this
                through ``admin.projects.delete_with_cleanup(...)`` rather than
                constructing :class:`Projects` directly.
            :exc:`~pinecone.errors.exceptions.PineconeValueError`: If *project_id* is
                empty or whitespace-only. Checked before anything is deleted.
            :exc:`~pinecone.errors.exceptions.ForbiddenError`: If the temporary API key
                cannot be created — usually the project's API-key quota. Nothing is
                deleted in this case, and the message says so and how to clear it.
            :exc:`~pinecone.errors.exceptions.FailedPreconditionError`: If the project is
                still not empty when the final delete runs, which happens when something
                is created in it after cleanup finishes. The error names what is blocking.
            :exc:`ApiError`: If the last cleanup attempt failed; the error from that
                attempt is re-raised, and the project is left partly emptied.

        Examples:
            .. code-block:: python

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
        """Delete an empty project permanently.

        The project must already be empty. Indexes, collections, assistants, and backups
        all block the delete, and the error names what is still there; API keys are *not*
        a blocker — they go with the project. Use :meth:`delete_with_cleanup` to clear the
        blockers first. Nothing here is recoverable.

        Args:
            project_id (str): The project to delete, e.g. ``"proj-abc123"``.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`: If *project_id* is
                empty or whitespace-only. Checked before the request is sent.
            :exc:`~pinecone.errors.exceptions.FailedPreconditionError`: If the project
                still owns indexes, collections, assistants, or backups. The error names
                what is blocking.

        Examples:
            >>> admin.projects.delete(project_id="proj-abc123")

        .. seealso::
           :meth:`delete_with_cleanup` — empties the project first; use that one when you
           do not already know it is empty.
        """
        require_non_empty("project_id", project_id)
        logger.info("Deleting project %r", project_id)
        self._http.delete(f"/admin/projects/{quote(project_id, safe='')}")
        logger.debug("Deleted project %r", project_id)
