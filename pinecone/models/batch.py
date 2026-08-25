"""Models for batch operations."""

from __future__ import annotations

import html as html_mod
from collections import Counter
from typing import Any

import msgspec
from msgspec import Struct

from pinecone.models._display import render_table
from pinecone.models.response_info import BatchResponseInfo


class BatchError(Struct, kw_only=True):
    """Details about a single failed batch within a batch operation.

    Attributes:
        batch_index: Zero-based position of this batch in the original sequence.
        items: The items that were in this batch (for retry).
        error: The exception that caused the failure.
        error_message: Human-readable description of the error.
    """

    batch_index: int
    items: list[dict[str, Any]]
    error: Exception
    error_message: str

    def __repr__(self) -> str:
        return (
            f"BatchError(batch_index={self.batch_index}, "
            f"item_count={len(self.items)}, "
            f"error_message={self.error_message!r})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary, with exception converted to string.

        Returns:
            Dictionary with all fields, where the error field is converted
            to its string representation for serializability.
        """
        return {
            "batch_index": self.batch_index,
            "items": self.items,
            "error": str(self.error),
            "error_message": self.error_message,
        }

    def to_json(self) -> str:
        """Convert to JSON string, with exception converted to string.

        Returns:
            JSON string with all fields, where the error field is converted
            to its string representation for serializability.
        """
        return msgspec.json.encode(self.to_dict()).decode("utf-8")


class BatchResult(Struct, kw_only=True):
    """Aggregated result of a batch operation.

    Tracks how many items and batches succeeded or failed, and provides
    access to the failed items for easy retry.

    Attributes:
        total_item_count: Total number of items submitted.
        successful_item_count: Number of items in successful batches.
        failed_item_count: Number of items in failed batches.
        total_batch_count: Total number of batches executed.
        successful_batch_count: Number of batches that succeeded.
        failed_batch_count: Number of batches that failed.
        errors: List of ``BatchError`` objects describing each failure.
        response_info: Aggregate durability signal (``lsn_reconciled`` /
            ``lsn_committed``) across successful sub-batches, or ``None``
            when no sub-batch reported these headers. See
            :class:`BatchResponseInfo`.
        timed_out: Whether a ``total_timeout`` expired with work left unsent. The
            batches that were never attempted appear in ``errors``, so
            ``failed_items`` is what remains to be sent. A deadline that elapses
            while the last batches are in flight, all of which then land, does not
            set this — there would be nothing to retry.

    Examples:
        >>> from pinecone import Pinecone
        >>> pc = Pinecone(api_key="your-api-key")
        >>> index = pc.index(name="articles-en")
        >>> documents = [
        ...     {
        ...         "_id": f"article-{i:05d}",
        ...         "content": f"Article {i}",
        ...         "embedding": [0.012, -0.087, 0.153],  # 1536-dim in practice
        ...     }
        ...     for i in range(1000)
        ... ]
        >>> result = index.documents.batch_upsert(
        ...     namespace="articles-en",
        ...     documents=documents,
        ... )
        >>> if result.has_errors:
        ...     print(f"{result.failed_item_count} items failed, retrying...")
        ...     retry = index.documents.batch_upsert(
        ...         namespace="articles-en",
        ...         documents=result.failed_items,
        ...     )
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

    @property
    def has_errors(self) -> bool:
        """Whether any batches failed."""
        return len(self.errors) > 0

    @property
    def error_count(self) -> int:
        """Alias for failed_item_count."""
        return self.failed_item_count

    @property
    def success_count(self) -> int:
        """Alias for successful_item_count."""
        return self.successful_item_count

    @property
    def failed_items(self) -> list[dict[str, Any]]:
        """All items from failed batches, flattened for retry.

        Pass directly back to the same batch method to retry only the
        items that failed on the previous attempt.

        Returns:
            list[dict[str, Any]]: Flat list of all items from failed batches.
        """
        items: list[dict[str, Any]] = []
        for error in self.errors:
            items.extend(error.items)
        return items

    def _error_summary(self) -> list[tuple[str, int]]:
        """Deduplicated error messages with counts, ordered by frequency."""
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
        """Convert to dictionary, with exceptions in errors converted to strings.

        Returns:
            Dictionary with all fields, where error exceptions in the errors
            list are converted to their string representations for serializability.
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
        }

    def to_json(self) -> str:
        """Convert to JSON string, with exceptions in errors converted to strings.

        Returns:
            JSON string with all fields, where error exceptions in the errors
            list are converted to their string representations for serializability.
        """
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
