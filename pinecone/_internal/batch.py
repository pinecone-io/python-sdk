"""Generic batch execution engine for parallel bulk operations.

Provides sync (ThreadPoolExecutor) and async (asyncio.Semaphore + gather)
executors that chunk a list of items, run an operation on each chunk in
parallel, collect errors, and optionally display a tqdm progress bar.
"""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as _FuturesTimeoutError
from typing import TYPE_CHECKING, Any, TypeVar

from pinecone._internal.adaptive import _AdaptiveLimiterRegistry
from pinecone.models.batch import BatchError, BatchResult
from pinecone.models.response_info import BatchResponseInfo

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

_MAX_WORKERS = 64

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_batch_params(batch_size: int, concurrency: int) -> None:
    """Raise ``ValueError`` for invalid batch_size or concurrency values."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if concurrency < 1 or concurrency > _MAX_WORKERS:
        raise ValueError(f"concurrency must be between 1 and {_MAX_WORKERS}, got {concurrency}")


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def _chunk(items: list[T], size: int) -> list[list[T]]:
    """Split *items* into sublists of at most *size* elements."""
    return [items[i : i + size] for i in range(0, len(items), size)]


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


# ---------------------------------------------------------------------------
# Sync executor
# ---------------------------------------------------------------------------


def batch_execute(
    *,
    items: list[dict[str, Any]],
    operation: Callable[[list[dict[str, Any]]], Any],
    batch_size: int,
    max_concurrency: int = 4,
    show_progress: bool = True,
    desc: str = "Batches",
    executor: ThreadPoolExecutor | None = None,
    limiter_registry: _AdaptiveLimiterRegistry | None = None,
    host: str | None = None,
    total_timeout: float | None = None,
) -> BatchResult:
    """Execute *operation* on *items* in parallel batches.

    Items are split into chunks of *batch_size* and submitted to a
    ``ThreadPoolExecutor`` with *max_concurrency* threads.  Exceptions raised
    by *operation* are caught per-batch and recorded as ``BatchError``
    entries in the result rather than propagated.

    Args:
        items (list[dict[str, Any]]): Full list of items to process.
        operation (Callable): Callable that accepts a batch (sublist).
        batch_size (int): Maximum items per batch (must be >= 1).
        max_concurrency (int): Thread pool size for concurrent requests
            (1-64, default 4).
        show_progress (bool): Display a tqdm progress bar when installed.
        desc (str): Label shown on the progress bar.
        executor (ThreadPoolExecutor | None): Optional caller-owned executor
            to reuse across calls. When provided, threads are not spawned
            or torn down per call. Caller is responsible for ``shutdown()``.
            When ``None`` (default), a private executor is created and
            shut down at the end of this call.
        limiter_registry (_AdaptiveLimiterRegistry | None): Optional registry
            for adaptive concurrency. SDK-internal; not for user code.
        host (str | None): Host key for the limiter registry lookup.
            SDK-internal; not for user code.
        total_timeout (float | None): Deadline in seconds for the whole
            operation. On expiry no further batches are submitted, batches
            already in flight are awaited, and the un-submitted ones are
            reported as errors so ``failed_items`` is what remains to be sent.
            ``timed_out`` is set only when something was actually left unsent —
            if the in-flight batches were the last ones and all landed, the
            operation succeeded late rather than failing. ``None`` (default)
            means no deadline.

    Returns:
        BatchResult with aggregated success/failure counts.

    Raises:
        ValueError: If *batch_size* or *max_concurrency* is out of range.
    """
    _validate_batch_params(batch_size, max_concurrency)

    if not items:
        return _empty_result()

    batches = _chunk(items, batch_size)
    total_batches = len(batches)
    errors: list[BatchError] = []
    successful_item_count = 0
    lsn_reconciled_values: list[int] = []
    lsn_committed_values: list[int] = []

    if limiter_registry is not None and host is not None:
        limiter = limiter_registry.get(host, max_concurrency)
    else:
        limiter = None

    condition = threading.Condition()
    inflight = 0

    def _acquire(deadline: _Deadline) -> bool:
        """Take an in-flight slot, or report that the budget ran out waiting for one."""
        nonlocal inflight
        if limiter is None:
            return not deadline.expired()
        with condition:
            while inflight >= limiter.current_limit():
                if deadline.expired():
                    return False
                condition.wait(timeout=deadline.remaining())
            if deadline.expired():
                return False
            inflight += 1
        return True

    def _release() -> None:
        nonlocal inflight
        if limiter is None:
            return
        with condition:
            inflight -= 1
            condition.notify_all()

    def _wrapped_op(batch: list[dict[str, Any]]) -> Any:
        try:
            result = operation(batch)
        finally:
            _release()
        # AIMD's increase half. Without this the limiter only ever halves, so one
        # throttle permanently pins a long-lived client to a lower concurrency.
        if limiter is not None:
            limiter.report_success()
        return result

    deadline = _Deadline(total_timeout)

    progress = _create_progress_bar(total_batches, desc, show_progress)

    own_executor = executor is None
    if executor is None:
        executor = ThreadPoolExecutor(max_workers=max_concurrency)

    timed_out = False

    budget = total_timeout if total_timeout is not None else 0.0

    def _collect(future: Future[Any]) -> None:
        nonlocal successful_item_count
        batch_idx, batch = future_to_batch[future]
        try:
            batch_result = future.result()
        except Exception as exc:
            errors.append(
                BatchError(
                    batch_index=batch_idx,
                    items=batch,
                    error=exc,
                    error_message=str(exc),
                )
            )
        else:
            successful_item_count += len(batch)
            _collect_lsn(batch_result, lsn_reconciled_values, lsn_committed_values)
        progress.update(1)

    try:
        future_to_batch: dict[Future[Any], tuple[int, list[dict[str, Any]]]] = {}
        for idx, batch in enumerate(batches):
            # The check lives inside _acquire too: waiting for a limiter slot can
            # itself outlast the budget, and submitting after that would put work
            # in flight the caller has already been told will not happen.
            if not _acquire(deadline):
                timed_out = True
                errors.extend(
                    _abandoned_error(i, b, budget) for i, b in enumerate(batches[idx:], start=idx)
                )
                progress.update(total_batches - idx)
                break
            future_to_batch[executor.submit(_wrapped_op, batch)] = (idx, batch)

        outstanding = set(future_to_batch)
        try:
            for future in as_completed(future_to_batch, timeout=deadline.remaining()):
                outstanding.discard(future)
                _collect(future)
        except _FuturesTimeoutError:
            # cancel() only succeeds for batches that have not started, which is
            # exactly the set that can be dropped without leaving work applied
            # server-side. Whatever is already running is awaited below.
            cancelled = {future for future in outstanding if future.cancel()}
            # The deadline elapsing is not itself a failure. If every outstanding
            # batch was already running and all of them land, the work is done —
            # late, but done — and reporting a timeout would hand the caller an
            # empty set of items to retry.
            timed_out = timed_out or bool(cancelled)
            for future in cancelled:
                _release()
                batch_idx, batch = future_to_batch[future]
                errors.append(_abandoned_error(batch_idx, batch, budget))
                progress.update(1)
            for future in as_completed(outstanding - cancelled):
                _collect(future)
    finally:
        progress.close()
        if own_executor:
            executor.shutdown()

    failed_item_count = sum(len(e.items) for e in errors)
    response_info = _build_aggregate(lsn_reconciled_values, lsn_committed_values)

    return BatchResult(
        total_item_count=len(items),
        successful_item_count=successful_item_count,
        failed_item_count=failed_item_count,
        total_batch_count=total_batches,
        successful_batch_count=total_batches - len(errors),
        failed_batch_count=len(errors),
        errors=sorted(errors, key=lambda err: err.batch_index),
        response_info=response_info,
        timed_out=timed_out,
    )


# ---------------------------------------------------------------------------
# Async executor
# ---------------------------------------------------------------------------


async def async_batch_execute(
    *,
    items: list[dict[str, Any]],
    operation: Callable[[list[dict[str, Any]]], Awaitable[Any]],
    batch_size: int,
    max_concurrency: int = 4,
    show_progress: bool = True,
    desc: str = "Batches",
    limiter_registry: _AdaptiveLimiterRegistry | None = None,
    host: str | None = None,
) -> BatchResult:
    """Async version of :func:`batch_execute`.

    Items are split into chunks of *batch_size* and run concurrently
    behind an ``asyncio.Semaphore`` with *max_concurrency* slots.
    Exceptions raised by *operation* are caught per-batch and recorded
    as ``BatchError`` entries in the result rather than propagated.

    Args:
        items (list[dict[str, Any]]): Full list of items to process.
        operation (Callable): Async callable that accepts a batch (sublist).
        batch_size (int): Maximum items per batch (must be >= 1).
        max_concurrency (int): Maximum concurrent batch requests
            (1-64, default 4).
        show_progress (bool): Display a tqdm progress bar when installed.
        desc (str): Label shown on the progress bar.
        limiter_registry (_AdaptiveLimiterRegistry | None): Optional registry
            for adaptive concurrency. SDK-internal; not for user code.
        host (str | None): Host key for the limiter registry lookup.
            SDK-internal; not for user code.

    Returns:
        BatchResult with aggregated success/failure counts.

    Raises:
        ValueError: If *batch_size* or *max_concurrency* is out of range.
    """
    _validate_batch_params(batch_size, max_concurrency)

    if not items:
        return _empty_result()

    batches = _chunk(items, batch_size)
    total_batches = len(batches)
    errors: list[BatchError] = []
    successful_item_count = 0
    lsn_reconciled_values: list[int] = []
    lsn_committed_values: list[int] = []

    use_limiter = limiter_registry is not None and host is not None
    if limiter_registry is not None and host is not None:
        limiter = limiter_registry.get(host, max_concurrency)
    else:
        limiter = None
    semaphore = asyncio.Semaphore(max_concurrency) if not use_limiter else None
    inflight = 0
    inflight_lock = asyncio.Lock()

    progress = _create_progress_bar(total_batches, desc, show_progress)

    async def _acquire() -> None:
        if semaphore is not None:
            await semaphore.acquire()
            return
        # Limiter path: spin until inflight < current_limit
        if limiter is None:
            return
        while True:
            async with inflight_lock:
                if inflight < limiter.current_limit():
                    return
            await asyncio.sleep(0.05)

    async def _release() -> None:
        if semaphore is not None:
            semaphore.release()

    async def _run_batch(batch_idx: int, batch: list[dict[str, Any]]) -> None:
        # nonlocal is safe: asyncio coroutines run on a single thread,
        # so += and .append() cannot interleave between await points.
        nonlocal successful_item_count, inflight
        await _acquire()
        if use_limiter:
            async with inflight_lock:
                inflight += 1
        try:
            try:
                batch_result = await operation(batch)
            except Exception as exc:
                errors.append(
                    BatchError(
                        batch_index=batch_idx,
                        items=batch,
                        error=exc,
                        error_message=str(exc),
                    )
                )
            else:
                successful_item_count += len(batch)
                _collect_lsn(batch_result, lsn_reconciled_values, lsn_committed_values)
                if limiter is not None:
                    limiter.report_success()
            progress.update(1)
        finally:
            if use_limiter:
                async with inflight_lock:
                    inflight -= 1
            else:
                await _release()

    try:
        tasks = [_run_batch(i, batch) for i, batch in enumerate(batches)]
        await asyncio.gather(*tasks)
    finally:
        progress.close()

    failed_item_count = sum(len(e.items) for e in errors)
    response_info = _build_aggregate(lsn_reconciled_values, lsn_committed_values)

    return BatchResult(
        total_item_count=len(items),
        successful_item_count=successful_item_count,
        failed_item_count=failed_item_count,
        total_batch_count=total_batches,
        successful_batch_count=total_batches - len(errors),
        failed_batch_count=len(errors),
        errors=errors,
        response_info=response_info,
    )
