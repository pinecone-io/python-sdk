"""Async RestoreJobs namespace — list and describe restore job operations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from pinecone._internal.adapters.restore_jobs_adapter import RestoreJobsAdapter
from pinecone._internal.backups_helpers import restore_job_list_params
from pinecone._internal.validation import require_non_empty
from pinecone.models.backups.list import RestoreJobList
from pinecone.models.backups.model import RestoreJobModel

if TYPE_CHECKING:
    from pinecone._internal.http_client import AsyncHTTPClient

logger = logging.getLogger(__name__)


class AsyncRestoreJobs:
    """Progress reports for restores of a backup into a new index.

    :meth:`~pinecone.AsyncPinecone.create_index_from_backup` hands back a
    ``restore_job_id`` and leaves the restore running in the background; this
    namespace is how you follow it to completion. Reached as
    ``pc.restore_jobs``; not constructed directly.

    A restore job is not a backup:
    :class:`~pinecone.async_client.backups.AsyncBackups` manages the snapshots
    themselves, while a job here is a read-only record of one attempt at
    turning a snapshot back into an index.

    Examples:

        .. code-block:: python

            from pinecone import AsyncPinecone

            async with AsyncPinecone(api_key="your-api-key") as pc:
                job = await pc.restore_jobs.describe(job_id="rj-abc123")
                print(job.status, job.target_index_name)

    .. seealso::
       :doc:`/guides/error-handling` — the exceptions any of these methods
       can raise, and which ones are worth retrying.
    """

    def __init__(self, http: AsyncHTTPClient) -> None:
        self._http = http
        self._adapter = RestoreJobsAdapter()

    def __repr__(self) -> str:
        """Return developer-friendly representation."""
        return "AsyncRestoreJobs()"

    async def list(
        self,
        *,
        limit: int | None = None,
        pagination_token: str | None = None,
    ) -> RestoreJobList:
        """List one page of the project's restore jobs.

        One call returns one page: :class:`RestoreJobList` carries a
        ``pagination`` token but never follows it, so iterating the return
        value sees at most one page. Drive the token yourself to walk the
        rest — see :doc:`/guides/pagination`. The result is a best-effort
        sample rather than an inventory; the warning below says why that
        matters.

        Args:
            limit (int | None): Maximum number of results per page. When ``None``,
                the parameter is omitted and the server applies its own
                default. Omitted too when *pagination_token* is given: the
                token already carries the page size it was minted with, and a
                different one sent alongside it would skip or repeat rows.
            pagination_token (str | None): Offset token naming the next page,
                taken from ``RestoreJobList.pagination.next``. A malformed or
                truncated token is rejected with ``400`` (:exc:`ApiError`)
                rather than restarting the listing.

        Returns:
            A :class:`RestoreJobList` supporting iteration, len(), and index access.
            Its ``pagination`` attribute is ``None`` on the final page.

        Examples:
            Walk every page the server will hand out. Because pages can overlap,
            the loop collects into a dict keyed by ``restore_job_id`` rather than
            a list — that is the de-duplication the warning below calls for, and
            it costs nothing on a listing that happens not to repeat:

            .. code-block:: python

                from pinecone import AsyncPinecone

                async with AsyncPinecone(api_key="your-api-key") as pc:
                    by_id = {}
                    page = await pc.restore_jobs.list(limit=100)
                    while True:
                        for job in page:
                            by_id[job.restore_job_id] = job
                        if not (page.pagination and page.pagination.next):
                            break
                        page = await pc.restore_jobs.list(
                            pagination_token=page.pagination.next
                        )

                    for job in by_id.values():
                        print(job.restore_job_id, job.target_index_name, job.status)

            When one page is all you want:

            .. code-block:: python

                async with AsyncPinecone(api_key="your-api-key") as pc:
                    page = await pc.restore_jobs.list(limit=5)
                    print(len(page))

        .. warning::
           **This listing can silently drop restore jobs, stop paginating
           early, and repeat rows across pages.** The token stream can end
           while restore jobs remain, and successive pages can overlap, so
           pages are neither exhaustive nor disjoint; a restore job whose
           target index has been deleted is dropped from the listing
           entirely. Treat the result as a best-effort sample rather than an
           inventory, never conclude a restore job does not exist from its
           absence here, and de-duplicate by ``restore_job_id`` while walking
           pages.

        .. seealso::
           :meth:`describe` — the authoritative read for a single job, by id.
        """
        params: dict[str, Any] = restore_job_list_params(
            limit=limit, pagination_token=pagination_token
        )

        logger.info("Listing restore jobs")
        response = await self._http.get("/restore-jobs", params=params)
        result = self._adapter.to_restore_job_list(response.content)
        logger.debug("Listed %d restore jobs", len(result))
        return result

    async def describe(self, *, job_id: str) -> RestoreJobModel:
        """Get the current state of one restore job.

        Args:
            job_id (str): The identifier of the restore job to describe.

        Returns:
            A :class:`RestoreJobModel` naming the ``backup_id`` restored and
            the ``target_index_name`` it lands in. ``status`` is one of
            ``"Pending"``, ``"Completed"``, ``"Failed"``, or ``"Cancelled"``:
            there is **no in-progress state**, so a restore that is actively
            running reports ``"Pending"`` and polling for a ``"Running"``-style
            value never succeeds. ``percent_complete`` and ``completed_at``
            are populated only once ``status`` is ``"Completed"``, so
            ``percent_complete`` reports completion rather than progress and
            cannot drive a progress bar.

        Raises:
            :exc:`PineconeValueError`: If *job_id* is empty.
            :exc:`NotFoundError`: If the API answers ``404`` — which is **not**
                the same as "the restore job does not exist"; see the warning
                below.

        Examples:
            .. code-block:: python

                from pinecone import AsyncPinecone

                async with AsyncPinecone(api_key="your-api-key") as pc:
                    job = await pc.restore_jobs.describe(job_id="rj-abc123")
                    print(job.status, job.target_index_name)

            To wait for a restore, poll until ``status`` *leaves* ``"Pending"``
            rather than waiting for it to reach a running state — there is no
            running state to reach. Bound the wait with a deadline so a job that
            never lands stops the loop instead of spinning forever; ten minutes
            below is illustrative, not a service guarantee:

            .. code-block:: python

                import asyncio
                import time

                async with AsyncPinecone(api_key="your-api-key") as pc:
                    deadline = time.monotonic() + 600
                    job = await pc.restore_jobs.describe(job_id="rj-abc123")
                    while job.status == "Pending" and time.monotonic() < deadline:
                        await asyncio.sleep(5)
                        job = await pc.restore_jobs.describe(job_id="rj-abc123")

                    print(job.status, job.completed_at)

        .. warning::
           **A ``404`` from this endpoint cannot be trusted to mean "no such
           restore job".** Any failure to read the restore-job store, an
           outage included, is answered with ``404``: what you see is
           :exc:`NotFoundError`, and what it actually means is "could not read
           this job", not "this job does not exist". Control flow keyed on it
           — giving up, deleting local state, reporting the job as gone — can
           each be wrong about what was really a transient failure, so treat
           it as possibly transient unless you have independent evidence the
           id is bad. A restore job whose target index has been deleted also
           answers ``404``, under a different message, so do not match on
           message text either; such a job is dropped from :meth:`list`
           entirely rather than reported.
        """
        require_non_empty("job_id", job_id)
        logger.info("Describing restore job %r", job_id)
        response = await self._http.get(f"/restore-jobs/{quote(job_id, safe='')}")
        result = self._adapter.to_restore_job(response.content)
        logger.debug("Described restore job %r", job_id)
        return result
