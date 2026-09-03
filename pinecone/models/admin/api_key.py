"""API key response models for the Admin API."""

from __future__ import annotations

from collections.abc import Iterator
from enum import Enum
from typing import Any

from msgspec import Struct

from pinecone.models._mixin import StructDictMixin


class APIKeyRole(str, Enum):
    """Roles that can be assigned to a Pinecone API key.

    Possible values: ``PROJECT_EDITOR``, ``PROJECT_VIEWER``,
    ``CONTROL_PLANE_EDITOR``, ``CONTROL_PLANE_VIEWER``,
    ``DATA_PLANE_EDITOR``, ``DATA_PLANE_VIEWER``.

    Every role here is project-scoped: an API key's authority never reaches
    beyond the project it was created in. This is a ``str`` enum, so the plain
    role names are accepted interchangeably with the members.

    Examples:
        >>> from pinecone import Admin
        >>> from pinecone.models.admin.api_key import APIKeyRole
        >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
        >>> result = admin.api_keys.create(
        ...     project_id="proj-abc123",
        ...     name="search-service-key",
        ...     roles=[APIKeyRole.DATA_PLANE_EDITOR],
        ... )
        >>> result.key.roles
        [<APIKeyRole.DATA_PLANE_EDITOR: 'DataPlaneEditor'>]

    .. seealso::
       - :class:`~pinecone.models.admin.role_binding.RoleName` — the roles used
         for users, service accounts, and invites. That set includes
         organization-scoped roles, which an API key cannot hold.
       - :meth:`ApiKeys.update() <pinecone.admin.api_keys.ApiKeys.update>` — changing a key's
         roles replaces the whole set rather than adding to it.
    """

    PROJECT_EDITOR = "ProjectEditor"
    PROJECT_VIEWER = "ProjectViewer"
    CONTROL_PLANE_EDITOR = "ControlPlaneEditor"
    CONTROL_PLANE_VIEWER = "ControlPlaneViewer"
    DATA_PLANE_EDITOR = "DataPlaneEditor"
    DATA_PLANE_VIEWER = "DataPlaneViewer"


class APIKeyModel(StructDictMixin, Struct, kw_only=True):
    """Response model for a Pinecone API key. The secret is not included.

    Attributes:
        id (str): Unique identifier for the API key. This is what every API-key
            operation takes as ``api_key_id``, and it is not the secret.
        name (str | None): Name of the API key, or ``None`` when the backend
            has no display label set for this key.
        project_id (str): Identifier of the project the key belongs to. A key's
            authority never reaches outside that project.
        roles (list[APIKeyRole]): List of roles assigned to the key
            (see :class:`APIKeyRole`).

    Examples:
        >>> from pinecone import Admin
        >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
        >>> key = admin.api_keys.describe(api_key_id="key-abc123")
        >>> key.id
        'key-abc123'
        >>> key.name
        'prod-search-key'
        >>> key.roles
        [<APIKeyRole.DATA_PLANE_EDITOR: 'DataPlaneEditor'>]

    .. seealso::
       - :class:`APIKeyWithSecret` — what
         :meth:`ApiKeys.create() <pinecone.admin.api_keys.ApiKeys.create>` returns instead,
         wrapping this model alongside the secret it shows only once.
    """

    id: str
    name: str | None = None
    project_id: str
    roles: list[APIKeyRole]

    @property
    def role(self) -> APIKeyRole:
        """Singular alias for ``roles`` when the key has exactly one role.

        Returns:
            :class:`APIKeyRole`: The single role assigned to this key.

        Raises:
            :exc:`ValueError`: If the key has no roles or more than one role.

        Examples:
            >>> from pinecone import Admin
            >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
            >>> key = admin.api_keys.describe(api_key_id="key-abc123")
            >>> key.role
            <APIKeyRole.DATA_PLANE_EDITOR: 'DataPlaneEditor'>

            Keys with two or more roles raise :exc:`ValueError`, so reach for
            this only where a key is known to hold exactly one:

            >>> from pinecone.models.admin.api_key import APIKeyModel, APIKeyRole
            >>> multi_role_key = APIKeyModel(
            ...     id="key-def456",
            ...     name="ci-pipeline-key",
            ...     project_id="proj-abc123",
            ...     roles=[APIKeyRole.CONTROL_PLANE_EDITOR, APIKeyRole.DATA_PLANE_EDITOR],
            ... )
            >>> multi_role_key.role
            Traceback (most recent call last):
                ...
            ValueError: API key has 2 roles; use .roles to access all
        """
        if len(self.roles) == 0:
            raise ValueError("API key has no roles")
        if len(self.roles) > 1:
            raise ValueError(f"API key has {len(self.roles)} roles; use .roles to access all")
        return self.roles[0]

    def __getitem__(self, key: str) -> Any:
        """Support bracket access (e.g. api_key['name'])."""
        if key not in self.__struct_fields__:
            raise KeyError(key)
        return getattr(self, key)

    def __contains__(self, key: object) -> bool:
        """Support ``in`` operator (e.g. ``'name' in api_key``)."""
        return key in self.__struct_fields__


class APIKeyWithSecret(StructDictMixin, Struct, kw_only=True):
    """Response model for an API key together with its secret value.

    Returned only by :meth:`ApiKeys.create() <pinecone.admin.api_keys.ApiKeys.create>`, and the
    secret it carries is obtainable exactly once — no later request returns it,
    and there is no rotation for API keys, so a lost secret means creating a
    replacement key and deleting the old one.

    Attributes:
        key (APIKeyModel): The API key metadata, including the ``id`` every
            other API-key operation takes.
        value (str): The secret API key string — what
            :class:`~pinecone.Pinecone` is constructed with. Treat as a
            credential.

    Examples:
        >>> from pinecone import Admin
        >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
        >>> created = admin.api_keys.create(project_id="proj-abc123", name="prod-search-key")
        >>> created.key.id
        'key-abc123'

        ``repr()`` keeps only the last four characters of the secret, so an
        object logged whole does not leak it:

        >>> repr(created).endswith("value='...alue')")
        True

    .. warning::
        The masking stops at ``repr()``. ``to_dict()`` and JSON encoding return
        ``value`` in full, so a result serialized wholesale into a log line, an
        error report, or a cache writes the live credential out.
    """

    key: APIKeyModel
    value: str

    def __repr__(self) -> str:
        masked = f"...{self.value[-4:]}" if len(self.value) >= 4 else "***"
        return f"APIKeyWithSecret(key={self.key!r}, value='{masked}')"

    def __str__(self) -> str:
        return repr(self)

    def __getitem__(self, key: str) -> Any:
        """Support bracket access (e.g. response['value'])."""
        if key not in self.__struct_fields__:
            raise KeyError(key)
        return getattr(self, key)

    def __contains__(self, key: object) -> bool:
        """Support ``in`` operator (e.g. ``'value' in response``)."""
        return key in self.__struct_fields__


class APIKeyList:
    """The API keys of one project, as returned by a list call.

    A sequence of :class:`APIKeyModel` — iterable, indexable, and sized — with
    :meth:`names` and :meth:`to_dict` on top. Not constructed directly; it is
    what :meth:`ApiKeys.list() <pinecone.admin.api_keys.ApiKeys.list>` returns.

    Unlike the organization-wide admin listings, this is not paginated: a
    project's keys arrive in one response, so there is no cursor to follow.

    Examples:
        >>> from pinecone.models.admin.api_key import APIKeyList, APIKeyModel, APIKeyRole
        >>> keys = APIKeyList(
        ...     [
        ...         APIKeyModel(
        ...             id="key-abc123",
        ...             name="prod-search-key",
        ...             project_id="proj-abc123",
        ...             roles=[APIKeyRole.DATA_PLANE_EDITOR],
        ...         )
        ...     ]
        ... )
        >>> keys.names()
        ['prod-search-key']
    """

    def __init__(self, api_keys: list[APIKeyModel]) -> None:
        """Initialize an APIKeyList.

        Args:
            api_keys: List of :class:`APIKeyModel` instances representing
                Pinecone API keys.
        """
        self._api_keys = api_keys

    def __iter__(self) -> Iterator[APIKeyModel]:
        return iter(self._api_keys)

    def __len__(self) -> int:
        return len(self._api_keys)

    def __getitem__(self, index: int) -> APIKeyModel:
        return self._api_keys[index]

    def to_dict(self) -> dict[str, Any]:
        """Return the list as a serializable dict.

        Returns:
            dict[str, Any]: A dict with a ``"data"`` key containing a list of
            API key dicts, each produced by :meth:`APIKeyModel.to_dict`.

        Examples:
            >>> from pinecone import Admin
            >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
            >>> keys = admin.api_keys.list(project_id="proj-abc123")
            >>> keys.to_dict()  # doctest: +SKIP
            {'data': [{'name': 'prod-search-key', ...}, {'name': 'ci-pipeline-key', ...}]}
        """
        return {"data": [k.to_dict() for k in self._api_keys]}

    def names(self) -> list[str | None]:
        """Return a list of API key names.

        Returns:
            list[str | None]: API key names in the same order as the list.
                Elements are ``None`` for keys whose backend display label is unset.

        Examples:
            >>> from pinecone import Admin
            >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
            >>> keys = admin.api_keys.list(project_id="proj-abc123")
            >>> keys.names()  # doctest: +SKIP
            ['prod-search-key', 'ci-pipeline-key']
        """
        return [api_key.name for api_key in self._api_keys]

    def __repr__(self) -> str:
        summaries = ", ".join(
            f"<name={k.name!r}, project_id={k.project_id!r}>" for k in self._api_keys
        )
        return f"APIKeyList([{summaries}])"
