"""Async BackupSchedules namespace — automatic, time-based index backups.

The asyncio twin of :mod:`pinecone.client.backup_schedules`. A backup schedule
attaches a recurring backup cadence to one index, at one of three cadences
(``daily``, ``weekly``, ``monthly``); the run time is chosen server-side and
reported through
:attr:`~pinecone.models.backups.schedules.BackupScheduleModel.next_scheduled_run`.
**There is no cron support anywhere in this API**, so there is no way to ask
for an arbitrary expression or a caller-chosen timezone.

The SDK always sends ``"time-based"`` as the schedule type, so :meth:`create`
takes no ``type`` argument. That is a client-side decision rather than an API
constraint: the server stores the value and echoes it back without validating
it, so
:attr:`~pinecone.models.backups.schedules.BackupScheduleModel.schedule_type`
reports whatever the schedule was created with -- always ``"time-based"`` for
schedules created through this SDK, not guaranteed for one created by another
client. ``frequency`` is the opposite: a real server-side enum, checked here as
well.

Two shapes are offered for each of the two listings.
:meth:`AsyncBackupSchedules.list` and :meth:`AsyncBackupSchedules.history`
return one page plus its pagination token, matching
:meth:`~pinecone.async_client.backups.AsyncBackups.list`.
:meth:`AsyncBackupSchedules.iter_schedules` and
:meth:`AsyncBackupSchedules.iter_history` return an
:class:`~pinecone.models.pagination.AsyncPaginator` that walks every page,
matching :meth:`~pinecone.async_client.indexes.AsyncIndexes.list_backups`.
Prefer the iterators unless you are managing pagination yourself: a daily
schedule with a 90-day retention window accumulates far more history rows than
one page holds. Both iterators await each page rather than blocking, so the
event loop stays free between them.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

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
    """Async control-plane operations for automatic, time-based backup schedules.

    Args:
        http (AsyncHTTPClient): Async HTTP client for making API requests.

    Note:
        Backups are a plan entitlement. A project without it gets a 403 rather
        than a 404 for a schedule that does not exist, and the SDK appends that
        clarification to the 403 while keeping the server's own message as the
        prefix. On-demand backups are gated on the same entitlement, so they
        are not a fallback.

    Examples:

        .. code-block:: python

            from pinecone import AsyncPinecone

            async with AsyncPinecone(api_key="your-api-key") as pc:
                schedule = await pc.backup_schedules.create(
                    index_name="product-search",
                    name="daily-compliance-backup",
                    frequency="daily",
                    retention_days=90,
                )
                async for run in pc.backup_schedules.iter_history(
                    schedule_id=schedule.schedule_id
                ):
                    print(run.backup_id, run.status)
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

        .. important::
           **Keep the schedule name short.** Each run names its backup
           ``"{name}-{run timestamp}"``, so a long schedule name pushes the
           derived backup name past the length limit backup names are held to.
           Nothing checks this at create time: the API declares no length limit
           on a schedule name, and the SDK does not invent one, because that
           would reject names the API accepts. An over-long name is therefore
           accepted here and fails later, on the runs, rather than on this
           call.

        Args:
            index_name (str): Name of the index to attach the schedule to.
            name (str): Name for the schedule. Backups it produces are named
                ``"{name}-{run timestamp}"`` — see the length caveat above.
            frequency (str): Cadence, one of ``"daily"``, ``"weekly"``,
                ``"monthly"``. Validated before any HTTP request; there is no
                cron alternative.
            retention_days (int): Days to retain each backup this schedule
                produces. Must be at least 1, which is checked here. The upper
                bound is a per-project setting the SDK does not know, so a
                too-large value is rejected server-side rather than here.

        Returns:
            A :class:`~pinecone.models.backups.schedules.BackupScheduleModel`
            describing the new schedule. It is created enabled, so
            ``next_scheduled_run`` is already populated.

        Raises:
            :exc:`PineconeValueError`: If *index_name* or *name* is empty, if
                *frequency* is not a supported cadence, or if *retention_days*
                is less than 1. All four are checked before any HTTP request.
            :exc:`ForbiddenError`: If the project's plan does not include
                scheduled backups. Checked before the index is looked up, so
                this wins over a 404 for a missing index.
            :exc:`NotFoundError`: If the index does not exist.
            :exc:`ConflictError`: If the index already has an *enabled*
                schedule — only one per index is allowed, so disable or delete
                the existing one first.
            :exc:`ApiError`: If the API returns another error response, such
                as a 400 for a pod-based index, which cannot be scheduled.

        Examples:

            .. code-block:: python

                from pinecone import AsyncPinecone

                async with AsyncPinecone(api_key="your-api-key") as pc:
                    schedule = await pc.backup_schedules.create(
                        index_name="product-search",
                        name="daily-compliance-backup",
                        frequency="daily",
                        retention_days=90,
                    )
                    print(schedule.frequency)

            An unsupported cadence is rejected before the request is sent, with
            a :exc:`PineconeValueError` naming the three it accepts:

            .. code-block:: python

                async with AsyncPinecone(api_key="your-api-key") as pc:
                    await pc.backup_schedules.create(
                        index_name="product-search",
                        name="hourly",
                        frequency="0 * * * *",
                        retention_days=7,
                    )
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
                f"/indexes/{index_name}/backup-schedules", json=request.to_wire()
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
        hold several rows even though at most one may be enabled.

        .. note::
           This returns a **single page**. Use :meth:`iter_schedules` to walk
           every page instead of managing the token yourself.

        Args:
            index_name (str): Name of the index whose schedules to list.
            limit (int | None): Maximum results per page. When ``None``, the
                parameter is omitted and the server applies its own default.
                **Ignored when a pagination token is given**: the token
                already carries the page size it was minted with, and sending
                a different one alongside it would skip or repeat rows.
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
            :exc:`ApiError`: If the API returns another error response.

        Examples:

            .. code-block:: python

                from pinecone import AsyncPinecone

                async with AsyncPinecone(api_key="your-api-key") as pc:
                    schedules = await pc.backup_schedules.list(index_name="my-index")
                    print(schedules.names())
                    print([s.schedule_id for s in schedules.enabled_schedules()])
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
                f"/indexes/{index_name}/backup-schedules", params=params
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
                all pages. Must be positive. ``None`` yields all of them. It
                also sets the requested page size, but only on a request that
                carries no pagination token: every later page is sized by the
                token, which already encodes it.
            pagination_token (str | None): Token to resume from a previous
                call. *limit* still caps the total yield, but it is not sent
                alongside a token — see above.

        Returns:
            A :class:`~pinecone.models.pagination.AsyncPaginator` over
            :class:`~pinecone.models.backups.schedules.BackupScheduleModel`
            instances.

        Raises:
            :exc:`PineconeValueError`: If *index_name* is empty or *limit* is
                zero or negative. Raised eagerly, before the first page is
                fetched.
            :exc:`ForbiddenError`: If the project's plan does not include
                scheduled backups. Raised when a page is fetched.
            :exc:`NotFoundError`: If the index does not exist.
            :exc:`ApiError`: If the API returns another error response.

        Examples:

            .. code-block:: python

                from pinecone import AsyncPinecone

                async with AsyncPinecone(api_key="your-api-key") as pc:
                    async for s in pc.backup_schedules.iter_schedules(
                        index_name="my-index"
                    ):
                        print(s.schedule_id, s.frequency, s.enabled)
        """
        require_non_empty("index_name", index_name)
        if limit is not None:
            require_positive("limit", limit)

        async def fetch_page(token: str | None) -> Page[BackupScheduleModel]:
            params = backup_schedule_list_params(limit=limit, pagination_token=token)
            logger.info("Listing backup schedules for index %r", index_name)
            with scheduled_backups_plan_gate():
                response = await self._http.get(
                    f"/indexes/{index_name}/backup-schedules", params=params
                )
            result = self._adapter.to_schedule_list(response.content)
            next_token = result.pagination.next if result.pagination is not None else None
            return Page(items=list(result), pagination_token=next_token)

        return AsyncPaginator(fetch_page=fetch_page, initial_token=pagination_token, limit=limit)

    async def describe(self, *, schedule_id: str) -> BackupScheduleModel:
        """Get detailed information about a backup schedule.

        Args:
            schedule_id (str): The identifier of the schedule to describe.
                This is the ``schedule_id`` from :meth:`create` or
                :meth:`list`, not the index name.

        Returns:
            A :class:`~pinecone.models.backups.schedules.BackupScheduleModel`
            with the schedule's current configuration.

        Raises:
            :exc:`PineconeValueError`: If *schedule_id* is empty.
            :exc:`ForbiddenError`: If the project's plan does not include
                scheduled backups.
            :exc:`NotFoundError`: If the schedule does not exist.
            :exc:`ApiError`: If the API returns another error response.

        Examples:

            .. code-block:: python

                from pinecone import AsyncPinecone

                async with AsyncPinecone(api_key="your-api-key") as pc:
                    schedule = await pc.backup_schedules.describe(
                        schedule_id="e88f7273-42aa-47e9-af73-593827136867"
                    )
                    print(schedule.enabled)
        """
        require_non_empty("schedule_id", schedule_id)
        logger.info("Describing backup schedule %r", schedule_id)
        with scheduled_backups_plan_gate():
            response = await self._http.get(f"/backup-schedules/{schedule_id}")
        result = self._adapter.to_schedule(response.content)
        logger.debug("Described backup schedule %r", schedule_id)
        return result

    async def get(self, *, schedule_id: str) -> BackupScheduleModel:
        """Get detailed information about a schedule (alias for :meth:`describe`).

        Args:
            schedule_id (str): The identifier of the schedule.

        Returns:
            A :class:`~pinecone.models.backups.schedules.BackupScheduleModel`
            with the schedule's current configuration.

        Raises:
            :exc:`PineconeValueError`: If *schedule_id* is empty.
            :exc:`NotFoundError`: If the schedule does not exist.
            :exc:`ApiError`: If the API returns another error response.

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

        .. warning::
           Passing ``enabled=True`` on a *disabled* schedule **immediately
           enqueues a backup run**; it is not a free toggle. It also
           recomputes ``next_scheduled_run`` from the moment of the update
           rather than resuming the old slot, so a disable/re-enable cycle
           shifts the cadence. And because only one schedule per index may be
           enabled, re-enabling fails with a 409 when another one already is.
           On an already-enabled schedule, ``enabled=True`` enqueues nothing.

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
                *retention_days* is set to less than 1. All checked before
                any HTTP request.
            :exc:`ForbiddenError`: If the project's plan does not include
                scheduled backups. Re-enabling is gated on the same
                entitlement as :meth:`create`.
            :exc:`NotFoundError`: If the schedule does not exist.
            :exc:`ConflictError`: If ``enabled=True`` and another schedule on
                the same index is already enabled.
            :exc:`ApiError`: If the API returns another error response.

        Note:
            Passing none of *frequency*, *retention_days*, or *enabled* sends
            an empty body, which the API accepts as a no-op and answers with
            the unchanged schedule.

        Examples:

            Pause a schedule without losing its configuration:

            .. code-block:: python

                from pinecone import AsyncPinecone

                async with AsyncPinecone(api_key="your-api-key") as pc:
                    paused = await pc.backup_schedules.update(
                        schedule_id="e88f7273-42aa-47e9-af73-593827136867",
                        enabled=False,
                    )
                    assert paused.next_scheduled_run is None

            Move to a weekly cadence with a shorter retention window:

            .. code-block:: python

                async with AsyncPinecone(api_key="your-api-key") as pc:
                    await pc.backup_schedules.update(
                        schedule_id="e88f7273-42aa-47e9-af73-593827136867",
                        frequency="weekly",
                        retention_days=30,
                    )
        """
        require_non_empty("schedule_id", schedule_id)
        with schedule_request_validation():
            request = UpdateBackupScheduleRequest(
                frequency=frequency, retention_days=retention_days, enabled=enabled
            )

        logger.info("Updating backup schedule %r", schedule_id)
        with scheduled_backups_plan_gate():
            response = await self._http.patch(
                f"/backup-schedules/{schedule_id}", json=request.to_wire()
            )
        result = self._adapter.to_schedule(response.content)
        logger.debug("Updated backup schedule %r", schedule_id)
        return result

    async def delete(self, *, schedule_id: str) -> None:
        """Permanently delete a backup schedule.

        Backups the schedule already produced are **not** deleted; they age
        out on their own retention window. Deleting the schedule only stops
        future runs.

        .. important::
           This is not safe to retry blindly. A successful delete answers 204
           with no body, and a second attempt on the same ``schedule_id``
           answers 404 -- so a retry after a dropped response is
           indistinguishable from deleting something that was never there.
           Treat a 404 following a delete attempt as success.

        Args:
            schedule_id (str): The identifier of the schedule to delete.

        Returns:
            ``None``. The 204 carries no body, and none is parsed.

        Raises:
            :exc:`PineconeValueError`: If *schedule_id* is empty.
            :exc:`ForbiddenError`: If the project's plan does not include
                scheduled backups.
            :exc:`NotFoundError`: If the schedule does not exist -- see the
                retry caveat above.
            :exc:`ApiError`: If the API returns another error response.

        Examples:

            .. code-block:: python

                from pinecone import AsyncPinecone

                async with AsyncPinecone(api_key="your-api-key") as pc:
                    await pc.backup_schedules.delete(
                        schedule_id="e88f7273-42aa-47e9-af73-593827136867"
                    )
        """
        require_non_empty("schedule_id", schedule_id)
        logger.info("Deleting backup schedule %r", schedule_id)
        with scheduled_backups_plan_gate():
            await self._http.delete(f"/backup-schedules/{schedule_id}")
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
        already completed with ones that have not started.

        .. note::
           This returns a **single page**. A daily schedule with a 90-day
           retention window has many more rows than one page holds, so prefer
           :meth:`iter_history` unless you are managing pagination yourself.

        Args:
            schedule_id (str): The identifier of the schedule whose history
                to list.
            limit (int | None): Maximum results per page. When ``None``, the
                parameter is omitted and the server applies its own default.
                **Ignored when a pagination token is given**: the token
                already carries the page size it was minted with, and sending
                a different one alongside it would skip or repeat rows.
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
            :exc:`ApiError`: If the API returns another error response.

        Examples:

            .. code-block:: python

                from pinecone import AsyncPinecone

                async with AsyncPinecone(api_key="your-api-key") as pc:
                    runs = await pc.backup_schedules.history(
                        schedule_id="e88f7273-42aa-47e9-af73-593827136867"
                    )
                    print([r.backup_id for r in runs.scheduled()])
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
                f"/backup-schedules/{schedule_id}/history", params=params
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
                pages. Must be positive. ``None`` yields all of them. It also
                sets the requested page size, but only on a request that
                carries no pagination token: every later page is sized by the
                token, which already encodes it.
            pagination_token (str | None): Token to resume from a previous
                call. *limit* still caps the total yield, but it is not sent
                alongside a token — see above.

        Returns:
            A :class:`~pinecone.models.pagination.AsyncPaginator` over
            :class:`~pinecone.models.backups.schedules.BackupScheduleHistoryItem`
            instances.

        Raises:
            :exc:`PineconeValueError`: If *schedule_id* is empty or *limit*
                is zero or negative. Raised eagerly, before the first page is
                fetched.
            :exc:`ForbiddenError`: If the project's plan does not include
                scheduled backups. Raised when a page is fetched.
            :exc:`NotFoundError`: If the schedule does not exist.
            :exc:`ApiError`: If the API returns another error response.

        Examples:

            .. code-block:: python

                from pinecone import AsyncPinecone

                async with AsyncPinecone(api_key="your-api-key") as pc:
                    async for run in pc.backup_schedules.iter_history(
                        schedule_id="e88f7273-42aa-47e9-af73-593827136867"
                    ):
                        print(run.backup_id, run.status, run.scheduled_execution_at)
        """
        require_non_empty("schedule_id", schedule_id)
        if limit is not None:
            require_positive("limit", limit)

        async def fetch_page(token: str | None) -> Page[BackupScheduleHistoryItem]:
            params = backup_schedule_list_params(limit=limit, pagination_token=token)
            logger.info("Listing history for backup schedule %r", schedule_id)
            with scheduled_backups_plan_gate():
                response = await self._http.get(
                    f"/backup-schedules/{schedule_id}/history", params=params
                )
            result = self._adapter.to_history_list(response.content)
            next_token = result.pagination.next if result.pagination is not None else None
            return Page(items=list(result), pagination_token=next_token)

        return AsyncPaginator(fetch_page=fetch_page, initial_token=pagination_token, limit=limit)
