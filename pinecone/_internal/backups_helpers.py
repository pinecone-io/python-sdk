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

SCHEDULED_BACKUPS_PLAN_HINT = (
    "Scheduled backups are a plan entitlement, and it is checked before the "
    "index or schedule is looked up — so this is about the project's plan, "
    "not a missing resource and not a key permission. On-demand backups via "
    "pc.backups.create() need no entitlement and remain available."
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


def backup_schedule_list_params(
    *, limit: int | None = None, pagination_token: str | None = None
) -> dict[str, str | int]:
    """Build the query params for a list-schedules or list-history request.

    Both schedule listings declare exactly ``limit`` and ``paginationToken``
    and no ``include_deleted``, so this narrows :func:`backup_list_params`
    rather than restating it.
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

    Every schedule operation answers 403 for one reason: the project's plan
    does not include scheduled backups. The backend's own message
    ("Scheduled backups are not available for your plan") says nothing about
    what to do next, so the hint is appended *after* it -- the backend text
    stays intact as the prefix.

    A 403 that is not the plan gate (an API key without project permissions,
    say) is returned untouched rather than being given a misleading upgrade
    hint, which is why this matches on the message instead of on the
    ``PERMISSION_DENIED`` code the two cases share.
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
