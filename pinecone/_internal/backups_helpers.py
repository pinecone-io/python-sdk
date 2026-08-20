"""Shared request-building helpers for the backup endpoints (2026-07 API).

Lives outside the sync client so the asyncio twin builds byte-identical
query strings from the same code.
"""

from __future__ import annotations

from pinecone.errors.exceptions import PineconeValueError

INCLUDE_DELETED_IS_INDEX_SCOPED = (
    "include_deleted only applies when listing the backups of one index: "
    "pass index_name= alongside it. The project-wide listing "
    "(GET /backups) already returns backups whose source index was deleted."
)


def backup_list_params(
    *,
    limit: int | None = None,
    pagination_token: str | None = None,
    include_deleted: bool | None = None,
) -> dict[str, str | int]:
    """Build the query params for a list-backups request.

    Each parameter is omitted when ``None`` so the server applies its own
    default rather than the SDK's idea of it.
    """
    params: dict[str, str | int] = {}
    if limit is not None:
        params["limit"] = limit
    if pagination_token is not None:
        params["paginationToken"] = pagination_token
    if include_deleted is not None:
        params["include_deleted"] = "true" if include_deleted else "false"
    return params


def require_index_scope_for_include_deleted(
    index_name: str | None, include_deleted: bool | None
) -> None:
    """Reject ``include_deleted`` on the project-wide listing.

    Only ``list_index_backups`` declares the parameter; the project-wide
    operation would ignore it silently, leaving the caller believing they
    had widened a listing that never narrowed.
    """
    if include_deleted is not None and index_name is None:
        raise PineconeValueError(INCLUDE_DELETED_IS_INDEX_SCOPED)
