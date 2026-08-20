"""Async RestoreJobs namespace — list and describe restore job operations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pinecone._internal.adapters.restore_jobs_adapter import RestoreJobsAdapter
from pinecone._internal.backups_helpers import restore_job_list_params
from pinecone._internal.validation import require_non_empty
from pinecone.models.backups.list import RestoreJobList
from pinecone.models.backups.model import RestoreJobModel

if TYPE_CHECKING:
    from pinecone._internal.http_client import AsyncHTTPClient

logger = logging.getLogger(__name__)


class AsyncRestoreJobs:
    """Async control-plane operations for Pinecone restore jobs.

    Provides methods to list and describe restore jobs.

    Args:
        http (AsyncHTTPClient): Async HTTP client for making API requests.

    Examples:

        .. code-block:: python

            from pinecone import AsyncPinecone

            async with AsyncPinecone(api_key="your-api-key") as pc:
                for job in await pc.restore_jobs.list():
                    print(job.restore_job_id)
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
        """List all restore jobs in the project.

        Supports cursor-based pagination.

        Args:
            limit (int | None): Maximum number of results per page. When ``None``,
                the backend applies its own default (100). **Ignored when
                *pagination_token* is given**: the token already carries the
                page size it was minted with, and sending a different one
                alongside it would skip or repeat rows.
            pagination_token (str | None): Token for cursor-based pagination.
                Takes precedence over *limit* — see above.

        Returns:
            A :class:`RestoreJobList` supporting iteration, len(), and index access.

        Raises:
            :exc:`ApiError`: If the API returns an error response.

        Examples:

            .. code-block:: python

                # List all restore jobs
                from pinecone import AsyncPinecone
                async with AsyncPinecone(api_key="your-api-key") as pc:
                    for job in await pc.restore_jobs.list():
                        print(job.restore_job_id, job.status)

            .. code-block:: python

                # List with a page size limit
                async with AsyncPinecone(api_key="your-api-key") as pc:
                    jobs = await pc.restore_jobs.list(limit=5)
                    print(len(jobs))
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
        """Get detailed information about a restore job.

        Args:
            job_id (str): The identifier of the restore job to describe.

        Returns:
            A :class:`RestoreJobModel` with full restore job details.

        Raises:
            :exc:`PineconeValueError`: If *job_id* is empty.
            :exc:`NotFoundError`: If the restore job does not exist.
            :exc:`ApiError`: If the API returns another error response.

        Examples:
            .. code-block:: python

                from pinecone import AsyncPinecone
                async with AsyncPinecone(api_key="your-api-key") as pc:
                    job = await pc.restore_jobs.describe(
                        job_id="rj-restore-20240115",
                    )
                    print(job.status)
        """
        require_non_empty("job_id", job_id)
        logger.info("Describing restore job %r", job_id)
        response = await self._http.get(f"/restore-jobs/{job_id}")
        result = self._adapter.to_restore_job(response.content)
        logger.debug("Described restore job %r", job_id)
        return result
