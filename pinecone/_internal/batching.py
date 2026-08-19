"""Shared chunking and progress-bar utilities for batched upserts."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TypeVar

from pinecone.errors.exceptions import PineconeValueError

T = TypeVar("T")


def validate_batch_size(batch_size: int) -> None:
    """Raise PineconeValueError unless batch_size is a positive int."""
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise PineconeValueError(
            f"batch_size must be a positive integer, got {batch_size!r} "
            f"({type(batch_size).__name__}). Pass a positive int, e.g. batch_size=500."
        )


def chunked(items: Sequence[T], batch_size: int) -> list[list[T]]:
    """Split items into successive ``batch_size``-sized lists.

    Returns ``[]`` for empty input. The final batch may be shorter
    than ``batch_size``. Caller is responsible for validating
    ``batch_size`` (use :func:`validate_batch_size`).
    """
    return [list(items[i : i + batch_size]) for i in range(0, len(items), batch_size)]


def with_progress(
    batches: Iterable[T], *, show_progress: bool, desc: str = "Upserting"
) -> Iterable[T]:
    """If ``show_progress`` is True and tqdm is installed, wrap with a progress bar.

    Falls back silently to the original iterable when tqdm is missing
    or ``show_progress`` is False.
    """
    if not show_progress:
        return batches
    try:
        from tqdm.auto import tqdm  # type: ignore[import-untyped]
    except ImportError:
        return batches
    return tqdm(batches, desc=desc)  # type: ignore[no-any-return]
