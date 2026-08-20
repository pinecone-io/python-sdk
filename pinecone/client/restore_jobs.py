"""RestoreJobs namespace — list and describe restore job operations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pinecone._internal.adapters.restore_jobs_adapter import RestoreJobsAdapter
from pinecone._internal.backups_helpers import restore_job_list_params
from pinecone._internal.validation import require_non_empty
from pinecone.models.backups.list import RestoreJobList
from pinecone.models.backups.model import RestoreJobModel

if TYPE_CHECKING:
    from pinecone._internal.http_client import HTTPClient

logger = logging.getLogger(__name__)


class RestoreJobs:
    """Control-plane operations for Pinecone restore jobs.

    Provides methods to list and describe restore jobs.

    Args:
        http (HTTPClient): HTTP client for making API requests.

    Examples:

        .. code-block:: python

            from pinecone import Pinecone

            pc = Pinecone(api_key="your-api-key")
            ids = [job.restore_job_id for job in pc.restore_jobs.list()]
    """

    def __init__(self, http: HTTPClient) -> None:
        self._http = http
        self._adapter = RestoreJobsAdapter()

    def __repr__(self) -> str:
        """Return developer-friendly representation."""
        return "RestoreJobs()"

    def list(
        self,
        *,
        limit: int | None = None,
        pagination_token: str | None = None,
    ) -> RestoreJobList:
        """List one page of the project's restore jobs.

        Pagination is **offset-based**, not cursor-based: the token is
        base64url-encoded JSON of the shape ``{"limit": N, "offset": M}``. This
        returns a **single page** and does not auto-fetch:
        :class:`RestoreJobList` carries a ``pagination`` token but never follows
        it, so iterating the return value sees at most one page. Drive the token
        yourself to walk the rest — see *Examples*.

        Args:
            limit (int | None): Maximum number of results per page. When ``None``,
                the backend applies its own default (100). Applies **even
                alongside** *pagination_token*, replacing the page size the
                token was minted with while keeping its offset — so send the
                same *limit* for the whole walk, or only on the first call.
            pagination_token (str | None): Offset token naming the next page,
                taken from ``RestoreJobList.pagination.next``. A malformed or
                truncated token is rejected with ``400`` (:exc:`ApiError`)
                rather than restarting the listing.

        Returns:
            A :class:`RestoreJobList` supporting iteration, len(), and index access.
            Its ``pagination`` attribute is ``None`` on the final page.

        Raises:
            :exc:`ApiError`: If the API returns an error response.

        .. warning::
            **Against today's backend this listing can silently drop restore
            jobs, stop paginating early, and repeat rows across pages.** The
            server pages over raw import operations, filters them down to
            restore jobs, and then computes the next-page token from the
            *post-filter* count. A page of 100 raw operations that happens to
            contain 3 restore jobs therefore looks like a short final page:
            pagination ends there, and every restore job behind it is never
            returned. When pagination does continue, the server's offset
            advances by the filtered count rather than the raw count, so a
            later page can re-return jobs you have already seen. Separately, a
            restore job whose target index has been deleted is dropped from the
            listing entirely, which shortens the page and can itself trigger
            the early stop.

            What that means for you: treat the result as a best-effort sample
            rather than an exhaustive inventory, never conclude a restore job
            does not exist from its absence here, and de-duplicate by
            ``restore_job_id`` while walking pages. The SDK offers no
            workaround on purpose — the token stream itself ends early, so no
            client-side code can recover pages the server never points at.
            Tracked upstream in `pinecone-io/python-sdk-internal#250
            <https://github.com/pinecone-io/python-sdk-internal/issues/250>`_.

        Examples:
            Walk every page the server will hand out:

            .. code-block:: python

                from pinecone import Pinecone

                pc = Pinecone(api_key="your-api-key")

                page = pc.restore_jobs.list(limit=100)
                jobs = list(page)
                while page.pagination and page.pagination.next:
                    page = pc.restore_jobs.list(pagination_token=page.pagination.next)
                    jobs.extend(page)

                for job in jobs:
                    print(job.restore_job_id, job.status, job.percent_complete)

            When one page is all you want:

            .. code-block:: python

                page = pc.restore_jobs.list(limit=5)
                print(len(page))
        """
        params: dict[str, Any] = restore_job_list_params(
            limit=limit, pagination_token=pagination_token
        )

        logger.info("Listing restore jobs")
        response = self._http.get("/restore-jobs", params=params)
        result = self._adapter.to_restore_job_list(response.content)
        logger.debug("Listed %d restore jobs", len(result))
        return result

    def describe(self, *, job_id: str) -> RestoreJobModel:
        """Get detailed information about a restore job.

        Args:
            job_id (str): The identifier of the restore job to describe.

        Returns:
            A :class:`RestoreJobModel` with full restore job details.
            ``status`` is one of ``"Pending"``, ``"Completed"``, ``"Failed"``,
            or ``"Cancelled"``. There is **no in-progress state**: a restore
            that is actively running reports ``"Pending"``, so do not poll for
            a ``"Running"``-style value. ``percent_complete`` is ``100`` once
            ``status`` is ``"Completed"`` and ``None`` at every other point —
            it reports completion, not progress, and cannot be used to draw a
            progress bar. ``completed_at`` is populated on the same condition.

        Raises:
            :exc:`PineconeValueError`: If *job_id* is empty.
            :exc:`NotFoundError`: If the API answers ``404`` — which is **not**
                the same as "the restore job does not exist"; see the warning
                below.
            :exc:`ApiError`: If the API returns another error response.

        .. warning::
            **A ``404`` from this endpoint cannot be trusted to mean "no such
            restore job".** Against today's backend every failure to read the
            restore-job store — a store outage included — is flattened into a
            ``404``, so :exc:`NotFoundError` here means "could not produce this
            job", not "this job does not exist". Any retry policy or control
            flow keyed on a ``404`` from ``describe`` is therefore unsafe:
            giving up, deleting local state, or reporting the job as gone can
            each be the wrong call on what was really a transient store
            failure. Treat it as possibly transient unless you have
            independent evidence the id is bad.

            The same flattening means a restore job whose target index has been
            deleted also answers ``404``, carrying a message about index
            metadata rather than "Restore job not found" — so do not match on
            the message text either. Such a job is dropped from :meth:`list`
            entirely rather than reported. Tracked upstream in
            `pinecone-io/python-sdk-internal#250
            <https://github.com/pinecone-io/python-sdk-internal/issues/250>`_.

        Examples:
            .. code-block:: python

                from pinecone import Pinecone

                pc = Pinecone(api_key="your-api-key")
                job = pc.restore_jobs.describe(job_id="rj-restore-20240115")
                print(job.status)
        """
        require_non_empty("job_id", job_id)
        logger.info("Describing restore job %r", job_id)
        response = self._http.get(f"/restore-jobs/{job_id}")
        result = self._adapter.to_restore_job(response.content)
        logger.debug("Described restore job %r", job_id)
        return result
