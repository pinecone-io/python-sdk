"""The sync bulk driver: chunk, admit, dispatch, collect. No decisions live
here — admission belongs to the gate, arithmetic to the helpers it imports.

Slot lifecycle is release-by-construction: a slot is handed to the future's
done-callback the moment ``submit`` succeeds, and released in ``except`` when
``submit`` itself raises — the acquire/release pairing never crosses an
ownership boundary bare. That shape is the fix for the reproduced deadlock
(slot taken, submit raised, release lived in a finally that never ran).

The per-call ``max_concurrency`` bound is a local semaphore: this call simply
never has more than its own bound outstanding, so the gate only ever needs
its host-wide limit and global admission is min(bound, limit) by construction.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as _FuturesTimeoutError
from typing import TYPE_CHECKING, Any

from pinecone._internal.batch import (
    _abandoned_error,
    _build_aggregate,
    _collect_lsn,
    _create_progress_bar,
    _Deadline,
    _empty_result,
    _validate_batch_params,
)
from pinecone._internal.bulk.classify import DISPOSITION_ABANDONED, is_retryable
from pinecone._internal.bulk.core import AcquireOutcome
from pinecone._internal.bulk.registry import GateRegistry, get_registry
from pinecone._internal.constants import DEFAULT_MAX_CONCURRENCY
from pinecone.models.batch import BatchError, BatchResult

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


def _stalled_error(batch_index: int, batch: list[dict[str, Any]]) -> BatchError:
    from pinecone.errors.exceptions import PineconeError

    message = (
        f"backend appears unavailable: the adaptive limiter reached its floor with "
        f"consecutive failures, so {len(batch)} items in batch {batch_index} were "
        f"abandoned without being sent; retry later with response.failed_items"
    )
    return BatchError(
        batch_index=batch_index,
        items=batch,
        error=PineconeError(message),
        error_message=message,
        disposition=DISPOSITION_ABANDONED,
    )


def bulk_execute_sync(
    *,
    items: list[dict[str, Any]],
    operation: Callable[[list[dict[str, Any]]], Any],
    batch_size: int,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    show_progress: bool = True,
    desc: str = "Batches",
    host: str,
    registry: GateRegistry | None = None,
    total_timeout: float | None = None,
) -> BatchResult:
    """Execute *operation* over *items* in gate-admitted parallel batches.

    Result contract: per-batch errors are
    collected (never propagated), ``timed_out`` is set only when something was
    left unsent, errors are ordered by batch index. Batches abandoned by the
    stall detector arrive as errors with an explicit backend-unavailable
    message rather than a timeout.
    """
    _validate_batch_params(batch_size, max_concurrency)
    if not items:
        return _empty_result()

    total_batches = (len(items) + batch_size - 1) // batch_size
    gate = (registry if registry is not None else get_registry()).get(host)
    deadline = _Deadline(total_timeout)
    budget = total_timeout if total_timeout is not None else 0.0

    errors: list[BatchError] = []
    successful_item_count = 0
    lsn_reconciled_values: list[int] = []
    lsn_committed_values: list[int] = []
    timed_out = False
    stalled = False
    peak_inflight = 0
    outstanding_now = 0
    counter_lock = threading.Lock()
    throttle_events_at_start = gate.throttle_events

    progress = _create_progress_bar(total_batches, desc, show_progress)
    call_bound = threading.Semaphore(max_concurrency)
    executor = ThreadPoolExecutor(
        min(max_concurrency, total_batches),
        thread_name_prefix="pinecone-bulk",
    )

    def _wrapped_op(batch: list[dict[str, Any]]) -> Any:
        nonlocal peak_inflight, outstanding_now
        with counter_lock:
            outstanding_now += 1
            peak_inflight = max(peak_inflight, outstanding_now)
        try:
            result = operation(batch)
        except Exception:
            gate.report_failure()
            raise
        finally:
            with counter_lock:
                outstanding_now -= 1
        gate.report_success()
        return result

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
                    retryable=is_retryable(exc),
                )
            )
        else:
            successful_item_count += len(batch)
            _collect_lsn(batch_result, lsn_reconciled_values, lsn_committed_values)
        progress.update(1)

    def _abandon_from(idx: int) -> None:
        nonlocal timed_out
        maker = _stalled_error if stalled else (lambda i, b: _abandoned_error(i, b, budget))
        for i in range(idx, total_batches):
            batch = items[i * batch_size : (i + 1) * batch_size]
            errors.append(maker(i, batch))
        if not stalled:
            timed_out = True
        progress.update(total_batches - idx)

    try:
        future_to_batch: dict[Future[Any], tuple[int, list[dict[str, Any]]]] = {}
        for idx in range(total_batches):
            batch = items[idx * batch_size : (idx + 1) * batch_size]
            if not call_bound.acquire(timeout=deadline.remaining()):
                _abandon_from(idx)
                break
            if deadline.expired():
                call_bound.release()
                _abandon_from(idx)
                break
            outcome, slot = gate.acquire(deadline.at())
            if outcome is not AcquireOutcome.GRANTED or slot is None:
                call_bound.release()
                if outcome is AcquireOutcome.STALLED:
                    stalled = True
                    logger.warning(
                        "bulk %s against %s abandoned %d of %d batches: backend appears "
                        "unavailable (adaptive limiter at floor with consecutive failures)",
                        desc,
                        host,
                        total_batches - idx,
                        total_batches,
                    )
                _abandon_from(idx)
                break
            try:
                future = executor.submit(_wrapped_op, batch)
            except Exception as exc:
                slot.release()
                call_bound.release()
                errors.append(
                    BatchError(batch_index=idx, items=batch, error=exc, error_message=str(exc))
                )
                progress.update(1)
                continue
            future.add_done_callback(slot.release)
            future.add_done_callback(lambda _f: call_bound.release())
            future_to_batch[future] = (idx, batch)

        outstanding = set(future_to_batch)
        try:
            for future in as_completed(future_to_batch, timeout=deadline.remaining()):
                outstanding.discard(future)
                _collect(future)
        except _FuturesTimeoutError:
            cancelled = {future for future in outstanding if future.cancel()}
            timed_out = timed_out or bool(cancelled)
            for future in cancelled:
                batch_idx, batch = future_to_batch[future]
                errors.append(_abandoned_error(batch_idx, batch, budget))
                progress.update(1)
            for future in as_completed(outstanding - cancelled):
                _collect(future)
    finally:
        progress.close()
        executor.shutdown(wait=True, cancel_futures=True)

    failed_item_count = sum(len(e.items) for e in errors)
    return BatchResult(
        total_item_count=len(items),
        successful_item_count=successful_item_count,
        failed_item_count=failed_item_count,
        total_batch_count=total_batches,
        successful_batch_count=total_batches - len(errors),
        failed_batch_count=len(errors),
        errors=sorted(errors, key=lambda err: err.batch_index),
        response_info=_build_aggregate(lsn_reconciled_values, lsn_committed_values),
        timed_out=timed_out,
        throttle_event_count=gate.throttle_events - throttle_events_at_start,
        final_limit=gate.limit,
        peak_inflight=peak_inflight,
        stalled=stalled,
    )
