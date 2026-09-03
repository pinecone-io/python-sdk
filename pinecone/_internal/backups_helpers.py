"""Shared request-building helpers for the backup endpoints (2026-07 API).

Lives outside the sync client so the asyncio twin builds byte-identical
query strings from the same code.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

from pinecone.errors.exceptions import (
    ForbiddenError,
    PineconeError,
    PineconeValueError,
)

INCLUDE_DELETED_IS_INDEX_SCOPED = (
    "include_deleted only applies when listing the backups of one index: "
    "pass index_name= alongside it. The project-wide listing "
    "(GET /backups) already returns backups whose source index was deleted."
)

LIMIT_AND_PAGINATION_ARE_EXCLUSIVE = (
    "A pagination token already carries the page size it was minted with. "
    "A limit sent alongside it overrides that page size while keeping the "
    "token's position -- so the next page starts where the old page size "
    "said it would and runs for the new length, skipping or repeating rows. "
    "The SDK therefore omits limit whenever a token is present."
)

SCHEDULED_BACKUPS_PLAN_HINT = (
    "Backups are a plan entitlement, so this is about the project's plan, "
    "not a missing resource and not a key permission — an un-entitled "
    "project sees this even for a schedule that does not exist. On-demand "
    "backups via pc.backups.create() are gated on the same entitlement and "
    "fail the same way, so there is no fallback that avoids it."
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

    ``limit`` is additionally omitted whenever *pagination_token* is set --
    see :data:`LIMIT_AND_PAGINATION_ARE_EXCLUSIVE` for why sending both
    corrupts the page boundary. ``include_deleted`` is *not* dropped: it is a
    filter rather than a window, the token does not encode it, and every page
    of one listing has to be asked the same question.
    """
    params: dict[str, str | int] = {}
    if pagination_token is not None:
        params["paginationToken"] = pagination_token
    elif limit is not None:
        params["limit"] = limit
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


def backup_schedule_list_params(
    *, limit: int | None = None, pagination_token: str | None = None
) -> dict[str, str | int]:
    """Build the query params for a list-schedules or list-history request.

    Both schedule listings declare exactly ``limit`` and ``paginationToken``
    and no ``include_deleted``, so this narrows :func:`backup_list_params`
    rather than restating it.
    """
    return backup_list_params(limit=limit, pagination_token=pagination_token)


def restore_job_list_params(
    *, limit: int | None = None, pagination_token: str | None = None
) -> dict[str, str | int]:
    """Build the query params for a list-restore-jobs request.

    ``GET /restore-jobs`` declares the same ``limit``/``paginationToken`` pair
    as the schedule listings and is served by the same offset-token
    machinery, so it delegates rather than hand-rolling the dict -- which is
    how it inherits :data:`LIMIT_AND_PAGINATION_ARE_EXCLUSIVE`.
    """
    return backup_list_params(limit=limit, pagination_token=pagination_token)


@contextlib.contextmanager
def schedule_request_validation() -> Iterator[None]:
    """Surface the schedule request models' ``ValueError`` as a Pinecone error.

    :class:`~pinecone.models.backups.schedules.CreateBackupScheduleRequest`
    and its update twin validate ``frequency`` and ``retention_days`` on
    construction and raise a bare :exc:`ValueError`. Every other client
    method raises a :exc:`PineconeError` subclass for bad input, so this
    widens the type without narrowing it: :class:`PineconeValueError` *is* a
    :exc:`ValueError`, so ``except ValueError`` keeps working and
    ``except PineconeError`` starts working.
    """
    try:
        yield
    except PineconeError:
        raise
    except ValueError as exc:
        raise PineconeValueError(str(exc)) from exc


@contextlib.contextmanager
def scheduled_backups_plan_gate() -> Iterator[None]:
    """Append the plan-upgrade hint to a plan-gated 403 from a schedule op."""
    try:
        yield
    except ForbiddenError as exc:
        annotated = annotate_plan_gated_forbidden(exc)
        if annotated is exc:
            raise
        raise annotated from exc


def annotate_plan_gated_forbidden(exc: ForbiddenError) -> ForbiddenError:
    """Return *exc* with the scheduled-backups plan hint appended, or *exc*.

    The hint is appended *after* the server's own message rather than
    replacing it, so that text stays intact as the prefix.

    A 403 that is not the plan gate (an API key without project permissions,
    say) is returned untouched rather than being given a misleading plan
    hint, which is why this matches on the message rather than on the
    permission-denied error code the two cases share.
    """
    message = exc.message or ""
    if "plan" not in message.lower():
        return exc
    return ForbiddenError(
        message=f"{message} {SCHEDULED_BACKUPS_PLAN_HINT}",
        status_code=exc.status_code,
        body=exc.body,
        reason=exc.reason,
        headers=exc.headers,
        error_code=exc.error_code,
        request_id=exc.request_id,
    )
