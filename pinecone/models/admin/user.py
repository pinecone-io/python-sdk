"""User response models for the Admin API."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from msgspec import Struct

from pinecone.models._mixin import StructDictMixin, _struct_to_dict_recursive
from pinecone.models.admin.pagination import PaginationResponse


class UserModel(StructDictMixin, Struct, kw_only=True):
    """Response model for a user who is a member of the organization.

    Role bindings are not included; use the role binding operations with
    ``principal_type="user"`` to see what a user can do.

    Attributes:
        id (str): Unique identifier (UUID) for the user.
        email (str): The user's email address.
        name (str | None): The user's display name, or ``None`` when the user
            has not set one. The server omits the field entirely in that case.

    Examples:
        >>> from pinecone.models.admin.user import UserModel
        >>> user = UserModel(id="e2e92523-85dc-4142-b8c2-e681be8b78df", email="alice@example.com")
        >>> user.email
        'alice@example.com'
        >>> user.name is None
        True
    """

    id: str
    email: str
    name: str | None = None


class UserList(Struct, kw_only=True):
    """A page of users, plus the cursor for the next page.

    Attributes:
        data (list[UserModel]): The users on this page.
        pagination (PaginationResponse | None): Cursor envelope for the next
            page, or ``None`` on the final page.

    Examples:
        >>> from pinecone.models.admin.user import UserList, UserModel
        >>> users = UserList(data=[UserModel(id="u1", email="alice@example.com")])
        >>> len(users)
        1
        >>> users.has_more
        False
        >>> users.emails()
        ['alice@example.com']
    """

    data: list[UserModel] = []
    pagination: PaginationResponse | None = None

    def __iter__(self) -> Iterator[UserModel]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> UserModel:
        return self.data[index]

    @property
    def pagination_token(self) -> str | None:
        """Opaque cursor for the next page, or ``None`` if this is the last page."""
        return self.pagination.next if self.pagination is not None else None

    @property
    def has_more(self) -> bool:
        """``True`` when the server supplied a cursor for a further page."""
        return self.pagination_token is not None

    def emails(self) -> list[str]:
        """Return the email addresses on this page, in order."""
        return [user.email for user in self.data]

    def to_dict(self) -> dict[str, Any]:
        """Return this page as a serializable dict with ``data`` and ``pagination`` keys."""
        return {
            "data": [_struct_to_dict_recursive(user) for user in self.data],
            "pagination": _struct_to_dict_recursive(self.pagination),
        }

    def __repr__(self) -> str:
        summaries = ", ".join(f"<id={u.id!r}, email={u.email!r}>" for u in self.data)
        return f"UserList([{summaries}], has_more={self.has_more!r})"
