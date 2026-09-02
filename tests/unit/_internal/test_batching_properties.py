"""Property-based tests for pinecone._internal.batching.

Complements the example-based tests in test_batching.py: instead of fixed
inputs, Hypothesis generates sequences and batch sizes, and each test asserts
an invariant of ``chunked`` that must hold for every input.
"""

from __future__ import annotations

from hypothesis import example, given
from hypothesis import strategies as st

from pinecone._internal.batching import chunked

_items = st.lists(st.integers(), max_size=200)
_batch_size = st.integers(min_value=1, max_value=50)


@given(items=_items, batch_size=_batch_size)
@example(items=[], batch_size=1)
@example(items=[1], batch_size=1)
@example(items=[1, 2, 3], batch_size=5)
def test_chunked_concatenation_reconstructs_input(items: list[int], batch_size: int) -> None:
    batches = chunked(items, batch_size)
    flattened = [x for batch in batches for x in batch]
    assert flattened == list(items)


@given(items=_items, batch_size=_batch_size)
def test_chunked_no_batch_exceeds_size(items: list[int], batch_size: int) -> None:
    for batch in chunked(items, batch_size):
        assert 1 <= len(batch) <= batch_size


@given(items=_items, batch_size=_batch_size)
def test_chunked_only_last_batch_may_be_short(items: list[int], batch_size: int) -> None:
    for batch in chunked(items, batch_size)[:-1]:
        assert len(batch) == batch_size


@given(items=_items, batch_size=_batch_size)
def test_chunked_batch_count_is_ceil_division(items: list[int], batch_size: int) -> None:
    expected = (len(items) + batch_size - 1) // batch_size
    assert len(chunked(items, batch_size)) == expected


@given(batch_size=_batch_size)
def test_chunked_empty_input_yields_no_batches(batch_size: int) -> None:
    assert chunked([], batch_size) == []
