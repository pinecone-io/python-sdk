"""The async bulk driver: the sync engine's twin, differing only in how
waiting works — coroutines park on the gate's per-waiter futures instead of
threads blocking on its condition. Wire format is not this module's concern;
the operation closure is.

3.10-clean by design, and deliberately not TaskGroup even where available:
TaskGroup's abort-on-first-error semantics are wrong for an engine whose
contract is that operations never raise (exceptions become BatchError rows).
External cancellation re-raises untouched — swallowing CancelledError to
return a partial result would break the caller's own timeout machinery.
Slot release rides task done-callbacks, which fire even for tasks cancelled
before their first step; a slot handed to a task that never runs still
returns.

total_timeout bounds admission only: an expired deadline stops new batches
and waits for in-flight ones, because cancelling client-side does not unsend
what the server may already be applying — identical semantics to the sync
driver, via the same _Deadline.
"""

from __future__ import annotations

import asyncio
import logging
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
from pinecone._internal.bulk.classify import is_retryable
from pinecone._internal.bulk.core import AcquireOutcome
from pinecone._internal.bulk.engine import _stalled_error
from pinecone._internal.bulk.registry import GateRegistry, get_registry
from pinecone._internal.constants import DEFAULT_MAX_CONCURRENCY
from pinecone.models.batch import BatchError, BatchResult

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)


async def bulk_execute_async(
    *,
    items: list[dict[str, Any]],
    operation: Callable[[list[dict[str, Any]]], Awaitable[Any]],
    batch_size: int,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    show_progress: bool = True,
    desc: str = "Batches",
    host: str,
    registry: GateRegistry | None = None,
    total_timeout: float | None = None,
) -> BatchResult:
    """Async twin of ``bulk_execute_sync`` with an identical result contract:
    errors collected and ordered by batch index, ``timed_out`` only when
    something was left unsent, dispositions and counters identical."""
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
    throttle_events_at_start = gate.throttle_events

    progress = _create_progress_bar(total_batches, desc, show_progress)
    call_bound = asyncio.Semaphore(max_concurrency)

    async def _run_batch(batch_idx: int, batch: list[dict[str, Any]], slot: Any) -> None:
        nonlocal successful_item_count, peak_inflight, outstanding_now
        try:
            outstanding_now += 1
            peak_inflight = max(peak_inflight, outstanding_now)
            try:
                batch_result = await operation(batch)
            except Exception as exc:
                gate.report_failure()
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
                gate.report_success()
                successful_item_count += len(batch)
                _collect_lsn(batch_result, lsn_reconciled_values, lsn_committed_values)
            progress.update(1)
        finally:
            outstanding_now -= 1
            call_bound.release()

    def _abandon_from(idx: int) -> None:
        nonlocal timed_out
        maker = _stalled_error if stalled else (lambda i, b: _abandoned_error(i, b, budget))
        for i in range(idx, total_batches):
            batch = items[i * batch_size : (i + 1) * batch_size]
            errors.append(maker(i, batch))
        if not stalled:
            timed_out = True
        progress.update(total_batches - idx)

    tasks: list[asyncio.Task[None]] = []
    try:
        for idx in range(total_batches):
            batch = items[idx * batch_size : (idx + 1) * batch_size]
            try:
                remaining = deadline.remaining()
                if remaining is None:
                    await call_bound.acquire()
                else:
                    await asyncio.wait_for(call_bound.acquire(), timeout=max(remaining, 0.001))
            except asyncio.TimeoutError:
                _abandon_from(idx)
                break
            if deadline.expired():
                call_bound.release()
                _abandon_from(idx)
                break
            outcome, slot = await gate.acquire_async(deadline.at())
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
            task = asyncio.get_running_loop().create_task(_run_batch(idx, batch, slot))
            task.add_done_callback(slot.release)
            tasks.append(task)
        if tasks:
            await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    finally:
        progress.close()

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
