"""What a bulk method reports back about the requests it made."""

from __future__ import annotations

import html as html_mod
from collections import Counter
from typing import Any

import msgspec
from msgspec import Struct

from pinecone.models._display import render_table
from pinecone.models.response_info import BatchResponseInfo


class BatchError(Struct, kw_only=True):
    """One batch that did not land, and everything needed to retry it.

    A bulk method captures per-batch failures instead of raising, so one bad
    batch does not abandon the rest. Every failure arrives as one of these in
    :attr:`BatchResult.errors`, still holding its own items.

    Check :attr:`retryable` before resending anything: a deterministic failure
    resent unchanged fails again, and a retry loop that ignores the flag spins
    forever on it.

    Attributes:
        batch_index: Where this batch sat in the list you passed, counting
            from zero.
        items: The items this batch was carrying, ready to be resent.
        error: The exception that ended the attempt — the thing to log or
            re-raise when you want the underlying cause.
        error_message: The same failure as a readable string, which is what
            :class:`BatchResult` groups and counts by.
        disposition: How far this batch got before failing, which is what
            tells you whether a retry could double-write. ``"rejected"`` means
            the attempt reached the server and came back with an error, so the
            write may have landed anyway; ``"unsent"`` means a deadline
            expired before it was submitted; ``"abandoned"`` means the backend
            looked down and the rest of the operation was dropped without
            sending. Treat the set as open — match the values you care about
            and let the rest fall through, because new ones can appear in a
            minor release.
        retryable: Whether resending these items could plausibly work.
            ``False`` marks a deterministic failure — a validation error, a
            4xx rejection — that would fail identically every time. Filter on
            it before any retry loop.
    """

    batch_index: int
    items: list[dict[str, Any]]
    error: Exception
    error_message: str
    disposition: str = "rejected"
    retryable: bool = True

    def __repr__(self) -> str:
        return (
            f"BatchError(batch_index={self.batch_index}, "
            f"item_count={len(self.items)}, "
            f"error_message={self.error_message!r})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the failure as a plain dict, for logging or JSON.

        ``error`` becomes ``str(error)``, since an exception is not
        serializable; reach for the attribute itself when you need the real
        exception.
        """
        return {
            "batch_index": self.batch_index,
            "items": self.items,
            "error": str(self.error),
            "error_message": self.error_message,
            "disposition": self.disposition,
            "retryable": self.retryable,
        }

    def to_json(self) -> str:
        """Return the failure as a JSON string, with ``error`` stringified."""
        return msgspec.json.encode(self.to_dict()).decode("utf-8")


class BatchResult(Struct, kw_only=True):
    """What a bulk method returns instead of raising.

    A bulk method splits your items into batches and reports the outcome of
    all of them here, so a partial failure is a value you inspect rather than
    an exception that loses the successes. Start with :attr:`has_errors`; if
    it is ``True``, :attr:`errors` holds one
    :class:`BatchError` per failed batch, each carrying its own items back.

    The last five attributes are backpressure telemetry rather than results.
    Read them when a bulk write was slower than expected: they say whether the
    backend was throttling you, and whether the SDK gave up on it.

    Attributes:
        total_item_count: How many items you passed in.
        successful_item_count: How many were in batches that landed.
        failed_item_count: How many were in batches that did not.
        total_batch_count: How many batches the items were split into.
        successful_batch_count: How many of those landed.
        failed_batch_count: How many did not.
        errors: One :class:`BatchError` per failed batch.
        response_info: A :class:`BatchResponseInfo` carrying the log position
            the batch is durable through, or ``None`` when no batch reported
            one. Use it to check that a later read sees these writes.
        timed_out: Whether a ``total_timeout`` expired with work left unsent. The
            batches that were never attempted appear in ``errors``, so
            ``failed_items`` is what remains to be sent. A deadline that elapses
            while the last batches are in flight, all of which then land, does not
            set this — there would be nothing to retry.
        throttle_event_count: Throttle signals the host's adaptive gate heard
            during this operation. Host-level, not call-level: concurrent
            operations against the same host share the gate, so their
            throttles are counted here too.
        final_limit: The adaptive concurrency limit when this operation
            finished, or ``None`` when the operation did not run through the
            gate. A value far below your ``max_concurrency`` means the
            backend was pushing back.
        peak_inflight: The most batches this operation had in flight at once.
        stalled: Whether the host gate's stall detector fired during this
            operation — the adaptive limit was at the floor with consecutive
            all-failed settles, so the remainder was abandoned rather than
            queued against an apparently-dead backend. Abandoned batches
            appear in ``errors`` with ``disposition="abandoned"``; the gate
            itself re-probes after a cool-down.

    Examples:
        >>> index = pc.index(name="product-search")
        >>> documents = [
        ...     {"_id": f"article-{i:05d}", "chunk_text": f"Paragraph {i}"}
        ...     for i in range(1000)
        ... ]
        >>> result = index.documents.batch_upsert(
        ...     namespace="published", documents=documents
        ... )
        >>> result.successful_item_count, result.has_errors
        (1000, False)

        Resend only the batches a retry could actually help, which is not the
        same set as :attr:`failed_items`:

        .. code-block:: python

            worth_retrying = [
                item
                for error in result.errors
                if error.retryable
                for item in error.items
            ]
            if worth_retrying:
                index.documents.batch_upsert(
                    namespace="published", documents=worth_retrying
                )
    """

    total_item_count: int
    successful_item_count: int
    failed_item_count: int
    total_batch_count: int
    successful_batch_count: int
    failed_batch_count: int
    errors: list[BatchError]
    response_info: BatchResponseInfo | None = None
    timed_out: bool = False
    throttle_event_count: int = 0
    final_limit: int | None = None
    peak_inflight: int = 0
    stalled: bool = False

    @property
    def has_errors(self) -> bool:
        """Whether any batch failed — the first thing to check on a result."""
        return len(self.errors) > 0

    @property
    def error_count(self) -> int:
        """Alias for :attr:`failed_item_count`; counts items, not batches."""
        return self.failed_item_count

    @property
    def success_count(self) -> int:
        """Alias for :attr:`successful_item_count`; counts items, not batches."""
        return self.successful_item_count

    @property
    def failed_items(self) -> list[dict[str, Any]]:
        """Every item from every failed batch, flattened into one list.

        Convenient to resend, but it includes batches whose
        :attr:`BatchError.retryable` is ``False`` — those fail identically on
        every attempt. Filter :attr:`errors` on that flag instead when the
        retry is in a loop.

        Returns:
            A flat list of the items that did not land.
        """
        items: list[dict[str, Any]] = []
        for error in self.errors:
            items.extend(error.items)
        return items

    def _error_summary(self) -> list[tuple[str, int]]:
        """Distinct error messages with batch counts, most frequent first."""
        counts: Counter[str] = Counter()
        for err in self.errors:
            counts[err.error_message] += 1
        return counts.most_common()

    def __repr__(self) -> str:
        status = "PARTIAL FAILURE" if self.has_errors else "SUCCESS"
        header = (
            f"BatchResult({status}: "
            f"{self.successful_item_count}/{self.total_item_count} items, "
            f"{self.successful_batch_count}/{self.total_batch_count} batches"
        )
        if not self.has_errors:
            return header + ")"

        summary = self._error_summary()
        lines = []
        for msg, count in summary:
            batch_word = "batch" if count == 1 else "batches"
            lines.append(f"    {msg} ({count} {batch_word})")
        return header + "\n  Errors:\n" + "\n".join(lines) + "\n)"

    def to_dict(self) -> dict[str, Any]:
        """Return the whole result as nested plain dicts, for logging or JSON.

        Each failure's ``error`` becomes ``str(error)``, since an exception is
        not serializable.
        """
        return {
            "total_item_count": self.total_item_count,
            "successful_item_count": self.successful_item_count,
            "failed_item_count": self.failed_item_count,
            "total_batch_count": self.total_batch_count,
            "successful_batch_count": self.successful_batch_count,
            "failed_batch_count": self.failed_batch_count,
            "errors": [error.to_dict() for error in self.errors],
            "response_info": (
                self.response_info.to_dict() if self.response_info is not None else None
            ),
            "timed_out": self.timed_out,
            "throttle_event_count": self.throttle_event_count,
            "final_limit": self.final_limit,
            "peak_inflight": self.peak_inflight,
            "stalled": self.stalled,
        }

    def to_json(self) -> str:
        """Return the whole result as a JSON string, with errors stringified."""
        return msgspec.json.encode(self.to_dict()).decode("utf-8")

    def _repr_html_(self) -> str:
        """Jupyter notebook HTML representation."""
        rows: list[tuple[str, str | int | float]] = [
            ("Total items:", self.total_item_count),
            ("Successful items:", self.successful_item_count),
            ("Failed items:", self.failed_item_count),
            ("Total batches:", self.total_batch_count),
            ("Successful batches:", self.successful_batch_count),
            ("Failed batches:", self.failed_batch_count),
        ]
        table = render_table("BatchResult", rows)

        if not self.has_errors:
            return table

        error_rows = "".join(
            f"""<tr>
                <td style="padding: 4px 8px; color: #666;">{html_mod.escape(msg)}</td>
                <td style="padding: 4px 8px; text-align: right;">{count}</td>
            </tr>"""
            for msg, count in self._error_summary()
        )
        error_section = f"""
        <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                    padding: 12px; border: 1px solid #e8c4c4;
                    border-radius: 6px; background-color: #fdf2f2;
                    max-width: 500px; margin-top: 8px;">
            <div style="font-weight: 600; margin-bottom: 10px; font-size: 14px;
                        color: #991b1b;">Errors</div>
            <table style="border-collapse: collapse; width: 100%;">
                <tr>
                    <th style="padding: 4px 8px; text-align: left; color: #666;
                               font-weight: 500;">Message</th>
                    <th style="padding: 4px 8px; text-align: right; color: #666;
                               font-weight: 500;">Batches</th>
                </tr>
                {error_rows}
            </table>
        </div>
        """
        return table + error_section
