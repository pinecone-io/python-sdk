"""Project response models for the Admin API."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from msgspec import Struct

from pinecone.models._mixin import StructDictMixin


class ProjectModel(StructDictMixin, Struct, kw_only=True):
    """Response model for a Pinecone project.

    A project owns indexes and API keys, and is the finer of the two scopes a
    role binding can name. Project names are not unique within an organization,
    so ``id`` is the only safe way to refer to one.

    Attributes:
        id (str): Unique identifier for the project. This is the ``resource_id``
            a project-scoped role binding takes, and the ``project_id`` the
            API-key operations take.
        name (str): Name of the project. Not unique — two projects in the same
            organization can share one.
        max_pods (int): Maximum number of pods allowed in the project. Applies
            to pod-based indexes only; serverless indexes are unaffected.
        force_encryption_with_cmek (bool): Whether CMEK encryption is enforced.
        organization_id (str): Identifier of the parent organization.
        created_at (str | None): Timestamp when the project was created, or
            ``None`` when the server omits it.

    Examples:
        >>> from pinecone import Admin
        >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
        >>> project = admin.projects.describe(project_id="proj-abc123")
        >>> project.name
        'my-project'
        >>> project.organization_id
        'org-abc123'
    """

    id: str
    name: str
    max_pods: int
    force_encryption_with_cmek: bool
    organization_id: str
    created_at: str | None = None

    def __getitem__(self, key: str) -> Any:
        """Support bracket access (e.g. project['name'])."""
        if key not in self.__struct_fields__:
            raise KeyError(key)
        return getattr(self, key)

    def __contains__(self, key: object) -> bool:
        """Support ``in`` operator (e.g. ``'name' in project``)."""
        return key in self.__struct_fields__


class ProjectList:
    """The projects of the organization the credentials resolve to.

    A sequence of :class:`ProjectModel` — iterable, indexable, and sized — with
    :meth:`names` and :meth:`to_dict` on top. Not constructed directly; it is
    what :meth:`Projects.list() <pinecone.admin.projects.Projects.list>` returns.

    This listing is not paginated: the projects arrive in one response, so there
    is no cursor to follow. Because names are not unique, :meth:`names` can
    contain duplicates.

    Examples:
        >>> from pinecone.models.admin.project import ProjectList, ProjectModel
        >>> projects = ProjectList(
        ...     [
        ...         ProjectModel(
        ...             id="proj-abc123",
        ...             name="production-search",
        ...             max_pods=10,
        ...             force_encryption_with_cmek=False,
        ...             organization_id="org-abc123",
        ...         )
        ...     ]
        ... )
        >>> projects.names()
        ['production-search']
    """

    def __init__(self, projects: list[ProjectModel]) -> None:
        """Initialize a ProjectList.

        Args:
            projects: List of :class:`ProjectModel` instances representing
                Pinecone projects.
        """
        self._projects = projects

    def __iter__(self) -> Iterator[ProjectModel]:
        return iter(self._projects)

    def __len__(self) -> int:
        return len(self._projects)

    def __getitem__(self, index: int) -> ProjectModel:
        return self._projects[index]

    def to_dict(self) -> dict[str, Any]:
        """Return the list as a serializable dict.

        Returns:
            dict[str, Any]: A dict with a ``"data"`` key containing a list of
            project dicts, each produced by :meth:`ProjectModel.to_dict`.

        Examples:
            >>> from pinecone import Admin
            >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
            >>> projects = admin.projects.list()
            >>> projects.to_dict()  # doctest: +SKIP
            {'data': [{'name': 'production-search', ...}, {'name': 'staging-recommendations', ...}]}
        """
        return {"data": [p.to_dict() for p in self._projects]}

    def names(self) -> list[str]:
        """Return a list of project names.

        Returns:
            list[str]: Project names in the same order as the list.

        Examples:
            >>> from pinecone import Admin
            >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
            >>> projects = admin.projects.list()
            >>> projects.names()  # doctest: +SKIP
            ['production-search', 'staging-recommendations']
        """
        return [project.name for project in self._projects]

    def __repr__(self) -> str:
        summaries = ", ".join(f"<name={p.name!r}, id={p.id!r}>" for p in self._projects)
        return f"ProjectList([{summaries}])"
