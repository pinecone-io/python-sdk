"""Shared building blocks for the bulk engines in ``_internal/bulk/``.

Parameter validation, the whole-operation deadline, the progress bar, the
zero-count result, and LSN aggregation. The engines that drive them live in
``_internal/bulk/engine.py`` (sync) and ``_internal/bulk/async_engine.py``
(async); this module holds only the parts both need and neither owns.
"""

from __future__ import annotations

import time
from typing import Any

from pinecone.errors.exceptions import PineconeValueError
from pinecone.models.batch import BatchError, BatchResult
from pinecone.models.response_info import BatchResponseInfo

_MAX_WORKERS = 64


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_batch_params(batch_size: int, concurrency: int) -> None:
    """Raise ``PineconeValueError`` for invalid batch_size or concurrency values."""
    if batch_size < 1:
        raise PineconeValueError(f"batch_size must be >= 1, got {batch_size}")
    if concurrency < 1 or concurrency > _MAX_WORKERS:
        raise PineconeValueError(
            f"concurrency must be between 1 and {_MAX_WORKERS}, got {concurrency}"
        )


# ---------------------------------------------------------------------------
# Progress bar helpers
# ---------------------------------------------------------------------------


class _NoOpProgressBar:
    """Drop-in replacement when tqdm is not installed."""

    def update(self, n: int = 1) -> None:
        pass

    def set_postfix_str(self, s: str) -> None:
        pass

    def close(self) -> None:
        pass

    def __enter__(self) -> _NoOpProgressBar:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


def _create_progress_bar(
    total: int,
    desc: str,
    show: bool,
) -> Any:
    """Return a tqdm bar if available, otherwise a silent no-op."""
    if not show:
        return _NoOpProgressBar()
    try:
        from tqdm.auto import tqdm  # type: ignore[import-untyped]

        return tqdm(total=total, desc=desc, unit="batch")
    except ImportError:
        return _NoOpProgressBar()


class _Deadline:
    """A monotonic budget for a whole batch operation.

    On expiry, batches that have not started are cancelled and batches already
    running are left to settle: dropping a running one client-side would not stop
    the server from applying it, so the caller would be told less landed than did.
    """

    __slots__ = ("_expires_at",)

    def __init__(self, total_timeout: float | None) -> None:
        self._expires_at = None if total_timeout is None else time.monotonic() + total_timeout

    def expired(self) -> bool:
        return self._expires_at is not None and time.monotonic() >= self._expires_at

    def remaining(self) -> float | None:
        if self._expires_at is None:
            return None
        return max(0.0, self._expires_at - time.monotonic())

    def at(self) -> float | None:
        """The absolute monotonic instant this expires, or None if unbounded."""
        return self._expires_at


def _abandoned_error(batch_index: int, batch: list[dict[str, Any]], total_timeout: float) -> Any:
    from pinecone.errors.exceptions import PineconeTimeoutError

    message = (
        f"total_timeout of {total_timeout}s expired before this batch was submitted; "
        f"{len(batch)} items in batch {batch_index} were not sent"
    )
    return BatchError(
        batch_index=batch_index,
        items=batch,
        error=PineconeTimeoutError(message),
        error_message=message,
        disposition="unsent",
    )


def _empty_result() -> BatchResult:
    """Return a zero-count result for empty input."""
    return BatchResult(
        total_item_count=0,
        successful_item_count=0,
        failed_item_count=0,
        total_batch_count=0,
        successful_batch_count=0,
        failed_batch_count=0,
        errors=[],
        response_info=None,
    )


def _collect_lsn(
    batch_result: Any,
    lsn_reconciled_values: list[int],
    lsn_committed_values: list[int],
) -> None:
    response_info = getattr(batch_result, "response_info", None)
    if response_info is None:
        return
    lsn_reconciled = getattr(response_info, "lsn_reconciled", None)
    if lsn_reconciled is not None:
        lsn_reconciled_values.append(lsn_reconciled)
    lsn_committed = getattr(response_info, "lsn_committed", None)
    if lsn_committed is not None:
        lsn_committed_values.append(lsn_committed)


def _build_aggregate(
    lsn_reconciled_values: list[int],
    lsn_committed_values: list[int],
) -> BatchResponseInfo | None:
    if not lsn_reconciled_values and not lsn_committed_values:
        return None
    return BatchResponseInfo(
        lsn_reconciled=max(lsn_reconciled_values) if lsn_reconciled_values else None,
        lsn_committed=max(lsn_committed_values) if lsn_committed_values else None,
    )
