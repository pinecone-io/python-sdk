"""Property-based tests for the sync bulk engine's result contract.

The engine's aggregate must hold for ANY failure pattern, not just the
hand-picked ones in test_engine.py: items are conserved (each lands in
exactly one of successful/failed), counts cross-check, errors arrive sorted
by batch index, and dispositions are internally consistent. Hypothesis
drives arbitrary per-batch outcome scripts through the real engine with a
fresh gate per example.
"""

from __future__ import annotations

import threading
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from pinecone._internal.bulk import bulk_execute_sync, get_registry
from pinecone.models.batch import BatchResult

HOST = "engine-properties.example.com"


def _run(
    n_items: int,
    batch_size: int,
    max_concurrency: int,
    failing_batches: set[int],
) -> tuple[BatchResult, int]:
    get_registry()._reset()
    items = [{"id": str(i)} for i in range(n_items)]
    calls_lock = threading.Lock()
    calls = {"n": 0}

    def operation(batch: list[dict[str, Any]]) -> dict[str, int]:
        with calls_lock:
            calls["n"] += 1
        first_index = int(batch[0]["id"]) // batch_size
        if first_index in failing_batches:
            raise RuntimeError(f"scripted failure for batch {first_index}")
        return {"upserted_count": len(batch)}

    result = bulk_execute_sync(
        items=items,
        operation=operation,
        batch_size=batch_size,
        max_concurrency=max_concurrency,
        show_progress=False,
        host=HOST,
    )
    return result, calls["n"]


@st.composite
def _scenario(draw: st.DrawFn) -> tuple[int, int, int, set[int]]:
    n_items = draw(st.integers(min_value=1, max_value=120))
    batch_size = draw(st.integers(min_value=1, max_value=25))
    max_concurrency = draw(st.integers(min_value=1, max_value=8))
    total_batches = -(-n_items // batch_size)
    failing = draw(
        st.sets(st.integers(min_value=0, max_value=total_batches - 1), max_size=total_batches)
    )
    return n_items, batch_size, max_concurrency, failing


@settings(max_examples=60, deadline=None)
@given(_scenario())
def test_items_are_conserved_under_any_failure_pattern(
    scenario: tuple[int, int, int, set[int]],
) -> None:
    n_items, batch_size, max_concurrency, failing = scenario
    result, _ = _run(n_items, batch_size, max_concurrency, failing)

    assert result.total_item_count == n_items
    assert result.successful_item_count + result.failed_item_count == n_items
    failed_ids = [item["id"] for err in result.errors for item in err.items]
    assert len(failed_ids) == len(set(failed_ids)), "an item failed in two batches"
    assert result.failed_item_count == len(failed_ids)


@settings(max_examples=60, deadline=None)
@given(_scenario())
def test_batch_counts_cross_check(scenario: tuple[int, int, int, set[int]]) -> None:
    n_items, batch_size, max_concurrency, failing = scenario
    result, _ = _run(n_items, batch_size, max_concurrency, failing)

    total_batches = -(-n_items // batch_size)
    assert result.total_batch_count == total_batches
    assert result.successful_batch_count + result.failed_batch_count == total_batches
    assert result.failed_batch_count == len(result.errors)
    assert result.has_errors == bool(failing)


@settings(max_examples=60, deadline=None)
@given(_scenario())
def test_errors_sorted_and_dispositions_consistent(
    scenario: tuple[int, int, int, set[int]],
) -> None:
    n_items, batch_size, max_concurrency, failing = scenario
    result, _ = _run(n_items, batch_size, max_concurrency, failing)

    indices = [err.batch_index for err in result.errors]
    assert indices == sorted(indices)
    assert not result.timed_out, "no deadline was set"
    for err in result.errors:
        assert err.disposition in {"rejected", "unsent", "abandoned"}
        assert err.items, "an error without items is unretryable by definition"


@settings(max_examples=40, deadline=None)
@given(_scenario())
def test_no_stall_means_every_batch_was_attempted(
    scenario: tuple[int, int, int, set[int]],
) -> None:
    n_items, batch_size, max_concurrency, failing = scenario
    result, attempted = _run(n_items, batch_size, max_concurrency, failing)

    total_batches = -(-n_items // batch_size)
    if not result.stalled:
        assert attempted == total_batches
    else:
        abandoned = [e for e in result.errors if e.disposition == "abandoned"]
        assert abandoned, "a stalled result must carry abandoned batches"
