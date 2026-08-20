"""Cursor pagination envelope shared by the paginated Admin API list responses."""

from __future__ import annotations

from msgspec import Struct

from pinecone.models._mixin import StructDictMixin


class PaginationResponse(StructDictMixin, Struct, kw_only=True):
    """Cursor envelope returned by paginated Admin API list responses.

    Attributes:
        next (str | None): Opaque cursor for the next page, or ``None`` when the
            server did not supply one. The value is never parsed or constructed
            by the SDK — pass it back verbatim as the ``pagination_token``
            argument on the following list call.

    Examples:
        >>> from pinecone.models.admin.pagination import PaginationResponse
        >>> page = PaginationResponse(next="eyJsYXN0X2lkIjoiZTJlOTI1MjMifQ==")
        >>> page.next
        'eyJsYXN0X2lkIjoiZTJlOTI1MjMifQ=='
    """

    next: str | None = None
