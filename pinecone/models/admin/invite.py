"""Invite response models for the Admin API."""

from __future__ import annotations

from collections.abc import Iterator
from enum import Enum
from typing import Any

from msgspec import Struct

from pinecone.models._mixin import StructDictMixin, _struct_to_dict_recursive
from pinecone.models.admin.pagination import PaginationResponse


class InviteStatus(str, Enum):
    """The lifecycle status of an organization invite.

    Possible values: ``pending``, ``expired``, ``processed``.

    List operations return only ``pending`` and ``expired`` invites;
    ``processed`` is returned only when fetching a single invite by ID.

    Examples:
        >>> from pinecone.models.admin.invite import InviteStatus
        >>> InviteStatus.PENDING == "pending"
        True
    """

    PENDING = "pending"
    EXPIRED = "expired"
    PROCESSED = "processed"


class InviteModel(StructDictMixin, Struct, kw_only=True):
    """Response model for an invitation to join the organization.

    ``status`` is typed as :class:`str` rather than :class:`InviteStatus` so a
    status added by the server after this SDK release surfaces as its raw
    string instead of raising. Compare against :class:`InviteStatus` members
    directly — they are ``str`` values.

    Attributes:
        id (str): Unique identifier (UUID) for the invite.
        email (str): The email address the invite was sent to.
        status (str): One of the :class:`InviteStatus` values.
        expires_at (str | None): RFC 3339 timestamp for when the invite expires
            if not accepted, or ``None`` if it does not expire. Resending an
            invite pushes this further out; read the new value from the
            resend response rather than computing it.
        processed_at (str | None): RFC 3339 timestamp for when the invite was
            accepted. ``None`` (or omitted by the server) while the invite is
            still pending or expired.
        created_at (str): RFC 3339 timestamp for when the invite was created.

    Examples:
        >>> from pinecone.models.admin.invite import InviteModel, InviteStatus
        >>> invite = InviteModel(
        ...     id="9c8e3528-b9c0-4358-84ce-84c28e91b566",
        ...     email="newhire@acme.com",
        ...     status="pending",
        ...     expires_at="2026-05-21T03:00:00Z",
        ...     created_at="2026-04-14T20:00:00Z",
        ... )
        >>> invite.status == InviteStatus.PENDING
        True
        >>> invite.processed_at is None
        True

    .. seealso::
       - :class:`~pinecone.models.admin.user.UserModel` — the member record
         created when the invite is accepted. The two carry separate IDs, so the
         invite's ``id`` is not usable as a user ID.
    """

    id: str
    email: str
    status: str
    expires_at: str | None = None
    processed_at: str | None = None
    created_at: str


class InviteList(Struct, kw_only=True):
    """A page of invites, plus the cursor for the next page.

    One raw page of an invite listing. Callers who reach invites through
    :meth:`Invites.list() <pinecone.admin.invites.Invites.list>` get a
    :class:`~pinecone.models.pagination.Paginator` instead, which follows these
    cursors for them.

    Attributes:
        data (list[InviteModel]): The invites on this page. Accepted invites are
            never among them; the listing covers pending and expired only.
        pagination (PaginationResponse | None): Cursor envelope for the next
            page, or ``None`` on the final page.

    Examples:
        >>> from pinecone.models.admin.invite import InviteList, InviteModel
        >>> invites = InviteList(
        ...     data=[
        ...         InviteModel(
        ...             id="9c8e3528-b9c0-4358-84ce-84c28e91b566",
        ...             email="newhire@acme.com",
        ...             status="pending",
        ...             created_at="2026-04-14T20:00:00Z",
        ...         )
        ...     ]
        ... )
        >>> invites.emails()
        ['newhire@acme.com']
    """

    data: list[InviteModel] = []
    pagination: PaginationResponse | None = None

    def __iter__(self) -> Iterator[InviteModel]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> InviteModel:
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
        """Return the invited email addresses on this page, in order."""
        return [invite.email for invite in self.data]

    def to_dict(self) -> dict[str, Any]:
        """Return this page as a serializable dict with ``data`` and ``pagination`` keys."""
        return {
            "data": [_struct_to_dict_recursive(invite) for invite in self.data],
            "pagination": _struct_to_dict_recursive(self.pagination),
        }

    def __repr__(self) -> str:
        summaries = ", ".join(f"<email={i.email!r}, status={i.status!r}>" for i in self.data)
        return f"InviteList([{summaries}], has_more={self.has_more!r})"
