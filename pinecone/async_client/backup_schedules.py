"""Async BackupSchedules namespace — automatic, time-based index backups.

The asyncio twin of :mod:`pinecone.client.backup_schedules`. A backup schedule
attaches a recurring backup cadence to one index, at one of three cadences
(``daily``, ``weekly``, ``monthly``); the run time is chosen server-side and
reported through
:attr:`~pinecone.models.backups.schedules.BackupScheduleModel.next_scheduled_run`.
**There is no cron support anywhere in this API**, so there is no way to ask
for an arbitrary expression or a caller-chosen timezone. The SDK always sends
``"time-based"`` as the schedule type, so :meth:`create` takes no ``type``
argument.

Each of the two listings comes in two shapes. :meth:`AsyncBackupSchedules.list`
and :meth:`AsyncBackupSchedules.history` return one page plus its pagination
token, matching :meth:`~pinecone.async_client.backups.AsyncBackups.list`.
:meth:`AsyncBackupSchedules.iter_schedules` and
:meth:`AsyncBackupSchedules.iter_history` return an
:class:`~pinecone.models.pagination.AsyncPaginator` that walks every page,
matching :meth:`~pinecone.async_client.indexes.AsyncIndexes.list_backups`.
Both iterators await each page rather than blocking, so the event loop stays
free between them. Prefer them unless you are driving the token yourself —
see :doc:`/guides/pagination`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from pinecone._internal.adapters.backup_schedules_adapter import BackupSchedulesAdapter
from pinecone._internal.backups_helpers import (
    backup_schedule_list_params,
    schedule_request_validation,
    scheduled_backups_plan_gate,
)
from pinecone._internal.validation import require_non_empty, require_positive
from pinecone.models.backups.list import BackupScheduleHistoryList, BackupScheduleList
from pinecone.models.backups.schedules import (
    BackupScheduleHistoryItem,
    BackupScheduleModel,
    CreateBackupScheduleRequest,
    UpdateBackupScheduleRequest,
)
from pinecone.models.pagination import AsyncPaginator, Page

if TYPE_CHECKING:
    from pinecone._internal.http_client import AsyncHTTPClient

logger = logging.getLogger(__name__)


class AsyncBackupSchedules:
    """Recurring, time-based backups of a single index.

    A schedule snapshots its index on a fixed cadence and retains each backup
    for a set number of days, so you do not have to call
    :meth:`~pinecone.async_client.backups.AsyncBackups.create` on a timer of
    your own. Reached as ``pc.backup_schedules``; not constructed directly.

    At most one schedule per index can be enabled at a time. The snapshots a
    schedule produces are ordinary backups: read one with
    :meth:`~pinecone.async_client.backups.AsyncBackups.describe`, or list a
    schedule's own runs with :meth:`history`.

    Note:
        Backups are a plan entitlement. A project without it gets a
        :exc:`ForbiddenError` rather than a :exc:`NotFoundError` even for a
        schedule that does not exist, and on-demand backups are gated on the
        same entitlement, so they are not a fallback.

    Examples:

        Create a schedule, then follow the backups it produces. History rows
        appear as runs are planned, so a schedule created moments ago has
        little or nothing in it yet:

        .. code-block:: python

            from pinecone import AsyncPinecone

            async with AsyncPinecone(api_key="your-api-key") as pc:
                schedule = await pc.backup_schedules.create(
                    index_name="product-search",
                    name="compliance-snapshots",
                    frequency="daily",
                    retention_days=90,
                )
                print(schedule.schedule_id, schedule.next_scheduled_run)
                async for run in pc.backup_schedules.iter_history(
                    schedule_id=schedule.schedule_id
                ):
                    print(run.backup_id, run.status)

    .. seealso::
       :doc:`/guides/error-handling` — the exceptions any of these methods
       can raise, and which ones are worth retrying.
    """

    def __init__(self, http: AsyncHTTPClient) -> None:
        self._http = http
        self._adapter = BackupSchedulesAdapter()

    def __repr__(self) -> str:
        """Return developer-friendly representation."""
        return "AsyncBackupSchedules()"

    async def create(
        self,
        *,
        index_name: str,
        name: str,
        frequency: str,
        retention_days: int,
    ) -> BackupScheduleModel:
        """Create a time-based backup schedule for an index.

        A backup schedule runs automatically at a fixed cadence, producing a
        backup of the index on each run. There is no cron support here —
        choose one of the three fixed cadences below. For a single snapshot
        taken now, use
        :meth:`~pinecone.async_client.backups.AsyncBackups.create` instead.

        Args:
            index_name (str): Name of the index to attach the schedule to.
            name (str): Name for the schedule. Backups it produces are named
                ``"{name}-{run timestamp}"``, so keep it short — see the note
                below.
            frequency (str): Cadence for the schedule: ``"daily"``,
                ``"weekly"``, or ``"monthly"``.
            retention_days (int): Number of days to retain each backup this
                schedule produces. Must be at least 1.

        Returns:
            A :class:`~pinecone.models.backups.schedules.BackupScheduleModel`
            describing the new schedule. It is created enabled, so
            ``next_scheduled_run`` is already populated.

        Raises:
            :exc:`PineconeValueError`: If *index_name* or *name* is empty, if
                *frequency* is not a supported cadence, or if *retention_days*
                is less than 1.
            :exc:`ForbiddenError`: If the project's plan does not include
                scheduled backups.
            :exc:`NotFoundError`: If the index does not exist.
            :exc:`ConflictError`: If the index already has an *enabled*
                schedule — only one per index is allowed, so disable or delete
                the existing one first.
            :exc:`ApiError`: If *index_name* names a pod-based index, which
                cannot be scheduled.

        Examples:

            .. code-block:: python

                from pinecone import AsyncPinecone

                async with AsyncPinecone(api_key="your-api-key") as pc:
                    schedule = await pc.backup_schedules.create(
                        index_name="product-search",
                        name="compliance-snapshots",
                        frequency="daily",
                        retention_days=90,
                    )
                    print(schedule.schedule_id, schedule.next_scheduled_run)

            The response spells the retention window
            ``retention_expire_after_days``, mirroring the request body's
            ``retention.expire_after_days`` — the returned schedule has no
            ``retention_days`` attribute.

        .. important::
           Keep the schedule name short. Each run names its backup
           ``"{name}-{run timestamp}"``, and the timestamp consumes a fixed
           share of the limit on resource names, so a long schedule name
           yields backup names past that limit. Neither the SDK nor the
           server rejects a long schedule name at create time; the cost
           surfaces later, at run time.
        """
        require_non_empty("index_name", index_name)
        require_non_empty("name", name)
        with schedule_request_validation():
            request = CreateBackupScheduleRequest(
                name=name, frequency=frequency, retention_days=retention_days
            )

        logger.info("Creating backup schedule %r for index %r", name, index_name)
        with scheduled_backups_plan_gate():
            response = await self._http.post(
                f"/indexes/{quote(index_name, safe='')}/backup-schedules", json=request.to_wire()
            )
        result = self._adapter.to_schedule(response.content)
        logger.debug("Created backup schedule %r", result.schedule_id)
        return result

    async def list(
        self,
        *,
        index_name: str,
        limit: int | None = None,
        pagination_token: str | None = None,
    ) -> BackupScheduleList:
        """List one page of an index's backup schedules.

        Schedules are always listed per index; there is no project-wide
        schedule listing. Disabled schedules are included, so a listing can
        hold several rows even though at most one may be enabled. One call
        returns one page — see :doc:`/guides/pagination`.

        Args:
            index_name (str): Name of the index whose schedules to list.
            limit (int | None): Maximum results per page. Defaults to the
                server's page size when ``None``. Ignored when a pagination
                token is given, since the token already carries the page
                size it was created with.
            pagination_token (str | None): Token naming the next page, taken
                from the previous page's ``pagination.next``. Takes precedence
                over *limit* — see above.

        Returns:
            A :class:`~pinecone.models.backups.list.BackupScheduleList`
            supporting iteration, ``len()``, and index access.
            ``BackupScheduleList.pagination`` is ``None`` on the final page.

        Raises:
            :exc:`PineconeValueError`: If *index_name* is empty or *limit* is
                zero or negative.
            :exc:`ForbiddenError`: If the project's plan does not include
                scheduled backups.
            :exc:`NotFoundError`: If the index does not exist.

        Examples:

            .. code-block:: python

                from pinecone import AsyncPinecone

                async with AsyncPinecone(api_key="your-api-key") as pc:
                    schedules = await pc.backup_schedules.list(index_name="product-search")
                    print(schedules.names())
                    print([s.schedule_id for s in schedules.enabled_schedules()])

            ``names()`` and ``enabled_schedules()`` read the page in hand
            rather than the whole listing, so check ``schedules.pagination``
            before concluding that an index has no enabled schedule.

        .. seealso::
           :meth:`iter_schedules` — the same listing as a paginator that walks
           every page, instead of one page plus a token.
        """
        require_non_empty("index_name", index_name)
        if limit is not None:
            require_positive("limit", limit)
        params: dict[str, Any] = backup_schedule_list_params(
            limit=limit, pagination_token=pagination_token
        )

        logger.info("Listing backup schedules for index %r", index_name)
        with scheduled_backups_plan_gate():
            response = await self._http.get(
                f"/indexes/{quote(index_name, safe='')}/backup-schedules", params=params
            )
        result = self._adapter.to_schedule_list(response.content)
        logger.debug("Listed %d backup schedules", len(result))
        return result

    def iter_schedules(
        self,
        *,
        index_name: str,
        limit: int | None = None,
        pagination_token: str | None = None,
    ) -> AsyncPaginator[BackupScheduleModel]:
        """Iterate every backup schedule on an index, fetching pages on demand.

        The auto-paginating twin of :meth:`list`. Iteration stops when a
        response carries no pagination envelope or a ``null`` one.

        Args:
            index_name (str): Name of the index whose schedules to iterate.
            limit (int | None): Maximum number of schedules to yield across
                all pages. Must be positive. ``None`` yields all of them.
            pagination_token (str | None): Token to resume from a previous
                call. *limit* still caps the total yield.

        Returns:
            A :class:`~pinecone.models.pagination.AsyncPaginator` over
            :class:`~pinecone.models.backups.schedules.BackupScheduleModel`
            instances.

        Raises:
            :exc:`PineconeValueError`: If *index_name* is empty or *limit* is
                zero or negative. Raised as soon as you call this method,
                before the first page is fetched.
            :exc:`ForbiddenError`: If the project's plan does not include
                scheduled backups. Raised while iterating, when a page is
                fetched.
            :exc:`NotFoundError`: If the index does not exist.

        Examples:

            .. code-block:: python

                from pinecone import AsyncPinecone

                async with AsyncPinecone(api_key="your-api-key") as pc:
                    async for s in pc.backup_schedules.iter_schedules(
                        index_name="product-search"
                    ):
                        print(s.schedule_id, s.frequency, s.enabled)

        .. seealso::
           :meth:`list` — one page plus its token, when you are driving
           pagination yourself.
        """
        require_non_empty("index_name", index_name)
        if limit is not None:
            require_positive("limit", limit)

        async def fetch_page(token: str | None) -> Page[BackupScheduleModel]:
            params = backup_schedule_list_params(limit=limit, pagination_token=token)
            logger.info("Listing backup schedules for index %r", index_name)
            with scheduled_backups_plan_gate():
                response = await self._http.get(
                    f"/indexes/{quote(index_name, safe='')}/backup-schedules", params=params
                )
            result = self._adapter.to_schedule_list(response.content)
            next_token = result.pagination.next if result.pagination is not None else None
            return Page(items=list(result), pagination_token=next_token)

        return AsyncPaginator(fetch_page=fetch_page, initial_token=pagination_token, limit=limit)

    async def describe(self, *, schedule_id: str) -> BackupScheduleModel:
        """Get the current configuration of one backup schedule.

        Args:
            schedule_id (str): The identifier of the schedule to describe.
                This is the ``schedule_id`` from :meth:`create` or
                :meth:`list`, not the index name.

        Returns:
            A :class:`~pinecone.models.backups.schedules.BackupScheduleModel`
            carrying the schedule's ``frequency``, its ``enabled`` flag, its
            ``retention_expire_after_days`` window, and
            ``next_scheduled_run`` — which is ``None`` exactly when the
            schedule is disabled.

        Raises:
            :exc:`PineconeValueError`: If *schedule_id* is empty.
            :exc:`ForbiddenError`: If the project's plan does not include
                scheduled backups.
            :exc:`NotFoundError`: If the schedule does not exist.

        Examples:

            .. code-block:: python

                from pinecone import AsyncPinecone

                async with AsyncPinecone(api_key="your-api-key") as pc:
                    schedule = await pc.backup_schedules.describe(
                        schedule_id="e88f7273-42aa-47e9-af73-593827136867"
                    )
                    print(schedule.enabled, schedule.next_scheduled_run)
        """
        require_non_empty("schedule_id", schedule_id)
        logger.info("Describing backup schedule %r", schedule_id)
        with scheduled_backups_plan_gate():
            response = await self._http.get(f"/backup-schedules/{quote(schedule_id, safe='')}")
        result = self._adapter.to_schedule(response.content)
        logger.debug("Described backup schedule %r", schedule_id)
        return result

    async def get(self, *, schedule_id: str) -> BackupScheduleModel:
        """Get detailed information about a schedule (alias for :meth:`describe`).

        Args:
            schedule_id (str): The identifier of the schedule.

        Returns:
            A :class:`~pinecone.models.backups.schedules.BackupScheduleModel`
            carrying the schedule's ``frequency``, its ``enabled`` flag, its
            ``retention_expire_after_days`` window, and
            ``next_scheduled_run`` — which is ``None`` exactly when the
            schedule is disabled.

        Raises:
            :exc:`PineconeValueError`: If *schedule_id* is empty.
            :exc:`ForbiddenError`: If the project's plan does not include
                scheduled backups.
            :exc:`NotFoundError`: If the schedule does not exist.

        Examples:

            .. code-block:: python

                from pinecone import AsyncPinecone

                async with AsyncPinecone(api_key="your-api-key") as pc:
                    schedule = await pc.backup_schedules.get(
                        schedule_id="e88f7273-42aa-47e9-af73-593827136867"
                    )
                    print(schedule.frequency)
        """
        return await self.describe(schedule_id=schedule_id)

    async def update(
        self,
        *,
        schedule_id: str,
        frequency: str | None = None,
        retention_days: int | None = None,
        enabled: bool | None = None,
    ) -> BackupScheduleModel:
        """Update a backup schedule's cadence, retention, or enabled state.

        Only the arguments you pass are sent, so omitted fields are left
        unchanged rather than reset. The schedule's ``name`` and its index
        cannot be changed -- the API exposes no field for either.

        Args:
            schedule_id (str): The identifier of the schedule to update.
            frequency (str | None): New cadence, one of ``"daily"``,
                ``"weekly"``, ``"monthly"``. ``None`` leaves it unchanged.
            retention_days (int | None): New retention window in days, at
                least 1. ``None`` leaves it unchanged. Changing it also
                re-times the pending deletion of backups this schedule has
                already produced.
            enabled (bool | None): ``False`` to disable (clearing
                ``next_scheduled_run``), ``True`` to re-enable -- see the
                warning above. ``None`` leaves it unchanged.

        Returns:
            A :class:`~pinecone.models.backups.schedules.BackupScheduleModel`
            with the updated configuration. After ``enabled=False`` its
            ``next_scheduled_run`` is ``None``.

        Raises:
            :exc:`PineconeValueError`: If *schedule_id* is empty, if
                *frequency* is set to an unsupported cadence, or if
                *retention_days* is set to less than 1.
            :exc:`ForbiddenError`: If the project's plan does not include
                scheduled backups.
            :exc:`NotFoundError`: If the schedule does not exist.
            :exc:`ConflictError`: If ``enabled=True`` and another schedule on
                the same index is already enabled.

        Note:
            Calling this with none of *frequency*, *retention_days*, or
            *enabled* set still issues the ``PATCH``, with an empty body. It
            changes nothing server-side and hands back the schedule as it
            stands, but it is a request rather than a skipped one. Use
            :meth:`describe` to re-read a schedule.

        Examples:

            Only the fields you name are sent. Moving this schedule to a
            weekly cadence with a shorter retention window leaves its
            ``name``, its index, and its enabled state exactly as they were:

            .. code-block:: python

                from pinecone import AsyncPinecone

                async with AsyncPinecone(api_key="your-api-key") as pc:
                    updated = await pc.backup_schedules.update(
                        schedule_id="e88f7273-42aa-47e9-af73-593827136867",
                        frequency="weekly",
                        retention_days=30,
                    )
                    print(updated.frequency, updated.retention_expire_after_days)
                    print(updated.name, updated.enabled)

            Pause the schedule instead, keeping the rest of its configuration:

            .. code-block:: python

                from pinecone import AsyncPinecone

                async with AsyncPinecone(api_key="your-api-key") as pc:
                    paused = await pc.backup_schedules.update(
                        schedule_id="e88f7273-42aa-47e9-af73-593827136867",
                        enabled=False,
                    )
                    print(paused.frequency, paused.next_scheduled_run)

        .. warning::
           Passing ``enabled=True`` on a *disabled* schedule immediately
           enqueues a backup run and recomputes ``next_scheduled_run`` from
           the moment of the update rather than resuming the old slot, so a
           disable/re-enable cycle shifts the cadence rather than pausing it.
           Only one schedule per index can be enabled, so re-enabling raises
           :exc:`ConflictError` if another one already is. On an
           already-enabled schedule, ``enabled=True`` enqueues nothing.
        """
        require_non_empty("schedule_id", schedule_id)
        with schedule_request_validation():
            request = UpdateBackupScheduleRequest(
                frequency=frequency, retention_days=retention_days, enabled=enabled
            )

        logger.info("Updating backup schedule %r", schedule_id)
        with scheduled_backups_plan_gate():
            response = await self._http.patch(
                f"/backup-schedules/{quote(schedule_id, safe='')}", json=request.to_wire()
            )
        result = self._adapter.to_schedule(response.content)
        logger.debug("Updated backup schedule %r", schedule_id)
        return result

    async def delete(self, *, schedule_id: str) -> None:
        """Permanently delete a backup schedule.

        Backups the schedule already produced are **not** deleted; they age
        out on their own retention window. Deleting the schedule only stops
        future runs.

        Args:
            schedule_id (str): The identifier of the schedule to delete.

        Raises:
            :exc:`PineconeValueError`: If *schedule_id* is empty.
            :exc:`ForbiddenError`: If the project's plan does not include
                scheduled backups.
            :exc:`NotFoundError`: If the schedule does not exist -- see the
                retry caveat above.

        Examples:

            .. code-block:: python

                from pinecone import AsyncPinecone

                async with AsyncPinecone(api_key="your-api-key") as pc:
                    await pc.backup_schedules.delete(
                        schedule_id="e88f7273-42aa-47e9-af73-593827136867"
                    )

        .. important::
           This is not safe to retry blindly. A successful delete raises
           nothing, and a second attempt on the same ``schedule_id`` raises
           :exc:`NotFoundError` -- so a retry after a dropped response is
           indistinguishable from deleting something that was never there.
           Treat a :exc:`NotFoundError` following a delete attempt as
           success.
        """
        require_non_empty("schedule_id", schedule_id)
        logger.info("Deleting backup schedule %r", schedule_id)
        with scheduled_backups_plan_gate():
            await self._http.delete(f"/backup-schedules/{quote(schedule_id, safe='')}")
        logger.debug("Deleted backup schedule %r", schedule_id)

    async def history(
        self,
        *,
        schedule_id: str,
        limit: int | None = None,
        pagination_token: str | None = None,
    ) -> BackupScheduleHistoryList:
        """List one page of the backups produced by a schedule.

        Rows describe backup *snapshots*, not the schedule, and a row appears
        as soon as a run is planned -- so the listing mixes runs that have
        already completed with ones that have not started. One call returns
        one page, and a frequent cadence with a long retention window has far
        more rows than one page holds — see :doc:`/guides/pagination`.

        Args:
            schedule_id (str): The identifier of the schedule whose history
                to list.
            limit (int | None): Maximum results per page. Defaults to the
                server's page size when ``None``. Ignored when a pagination
                token is given, since the token already carries the page
                size it was created with.
            pagination_token (str | None): Token naming the next page, taken
                from the previous page's ``pagination.next``. Takes precedence
                over *limit* — see above.

        Returns:
            A :class:`~pinecone.models.backups.list.BackupScheduleHistoryList`
            supporting iteration, ``len()``, and index access.
            ``BackupScheduleHistoryList.pagination`` is ``None`` on the final
            page.

        Raises:
            :exc:`PineconeValueError`: If *schedule_id* is empty or *limit*
                is zero or negative.
            :exc:`ForbiddenError`: If the project's plan does not include
                scheduled backups.
            :exc:`NotFoundError`: If the schedule does not exist.

        Examples:

            Walk the history a page at a time, narrowing each page to the runs
            that have not started yet. ``scheduled()`` filters the page in
            hand, so it belongs inside the loop rather than after it:

            .. code-block:: python

                from pinecone import AsyncPinecone

                async with AsyncPinecone(api_key="your-api-key") as pc:
                    pagination_token = None
                    while True:
                        runs = await pc.backup_schedules.history(
                            schedule_id="e88f7273-42aa-47e9-af73-593827136867",
                            pagination_token=pagination_token,
                        )
                        for run in runs.scheduled():
                            print(run.backup_id, run.scheduled_execution_at)
                        pagination_token = runs.pagination.next if runs.pagination else None
                        if pagination_token is None:
                            break

        .. seealso::
           :meth:`iter_history` — the same listing as a paginator that walks
           every page, instead of one page plus a token.
        """
        require_non_empty("schedule_id", schedule_id)
        if limit is not None:
            require_positive("limit", limit)
        params: dict[str, Any] = backup_schedule_list_params(
            limit=limit, pagination_token=pagination_token
        )

        logger.info("Listing history for backup schedule %r", schedule_id)
        with scheduled_backups_plan_gate():
            response = await self._http.get(
                f"/backup-schedules/{quote(schedule_id, safe='')}/history", params=params
            )
        result = self._adapter.to_history_list(response.content)
        logger.debug("Listed %d backup schedule history rows", len(result))
        return result

    def iter_history(
        self,
        *,
        schedule_id: str,
        limit: int | None = None,
        pagination_token: str | None = None,
    ) -> AsyncPaginator[BackupScheduleHistoryItem]:
        """Iterate every backup a schedule has produced, fetching pages on demand.

        The auto-paginating twin of :meth:`history`. Iteration stops when a
        response carries no pagination envelope or a ``null`` one.

        Args:
            schedule_id (str): The identifier of the schedule whose history
                to iterate.
            limit (int | None): Maximum number of rows to yield across all
                pages. Must be positive. ``None`` yields all of them.
            pagination_token (str | None): Token to resume from a previous
                call. *limit* still caps the total yield.

        Returns:
            A :class:`~pinecone.models.pagination.AsyncPaginator` over
            :class:`~pinecone.models.backups.schedules.BackupScheduleHistoryItem`
            instances.

        Raises:
            :exc:`PineconeValueError`: If *schedule_id* is empty or *limit*
                is zero or negative. Raised as soon as you call this method,
                before the first page is fetched.
            :exc:`ForbiddenError`: If the project's plan does not include
                scheduled backups. Raised while iterating, when a page is
                fetched.
            :exc:`NotFoundError`: If the schedule does not exist.

        Examples:

            .. code-block:: python

                from pinecone import AsyncPinecone

                async with AsyncPinecone(api_key="your-api-key") as pc:
                    async for run in pc.backup_schedules.iter_history(
                        schedule_id="e88f7273-42aa-47e9-af73-593827136867"
                    ):
                        print(run.backup_id, run.status, run.scheduled_execution_at)

        .. seealso::
           :meth:`history` — one page plus its token, when you are driving
           pagination yourself.
        """
        require_non_empty("schedule_id", schedule_id)
        if limit is not None:
            require_positive("limit", limit)

        async def fetch_page(token: str | None) -> Page[BackupScheduleHistoryItem]:
            params = backup_schedule_list_params(limit=limit, pagination_token=token)
            logger.info("Listing history for backup schedule %r", schedule_id)
            with scheduled_backups_plan_gate():
                response = await self._http.get(
                    f"/backup-schedules/{quote(schedule_id, safe='')}/history", params=params
                )
            result = self._adapter.to_history_list(response.content)
            next_token = result.pagination.next if result.pagination is not None else None
            return Page(items=list(result), pagination_token=next_token)

        return AsyncPaginator(fetch_page=fetch_page, initial_token=pagination_token, limit=limit)
