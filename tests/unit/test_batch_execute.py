"""Unit tests for batch execution helpers."""

from __future__ import annotations

import pytest
from msgspec import Struct

from pinecone._internal.batch import (
    _chunk,
    async_batch_execute,
    batch_execute,
)
from pinecone.errors import PineconeValueError
from pinecone.models.batch import BatchResult
from pinecone.models.response_info import ResponseInfo


class _FakeResp(Struct, kw_only=True):
    response_info: ResponseInfo | None = None


# ---------------------------------------------------------------------------
# Sync: batch_execute
# ---------------------------------------------------------------------------


def test_batch_execute_empty_items() -> None:
    """Empty item list returns zero-count BatchResult."""

    def noop(batch: list[dict]) -> None:  # type: ignore[type-arg]
        pass

    result = batch_execute(items=[], operation=noop, batch_size=10, show_progress=False)

    assert isinstance(result, BatchResult)
    assert result.total_item_count == 0
    assert result.successful_item_count == 0
    assert result.failed_item_count == 0
    assert result.total_batch_count == 0
    assert result.successful_batch_count == 0
    assert result.failed_batch_count == 0
    assert result.errors == []
    assert result.has_errors is False


def test_batch_execute_all_succeed() -> None:
    """10 items with batch_size=3 all succeed — counts match."""
    items = [{"id": str(i)} for i in range(10)]

    def noop(batch: list[dict]) -> None:  # type: ignore[type-arg]
        pass

    result = batch_execute(
        items=items,
        operation=noop,
        batch_size=3,
        max_concurrency=2,
        show_progress=False,
    )

    assert result.successful_item_count == 10
    assert result.failed_item_count == 0
    assert result.has_errors is False
    assert result.total_item_count == 10


def test_batch_execute_partial_failure() -> None:
    """Second batch raises — 5 succeed, 5 fail, 1 error recorded."""
    items = [{"id": str(i)} for i in range(10)]
    call_count = 0

    def op(batch: list[dict]) -> None:  # type: ignore[type-arg]
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("batch 2 failed")

    result = batch_execute(
        items=items,
        operation=op,
        batch_size=5,
        max_concurrency=1,  # serial so call order is deterministic
        show_progress=False,
    )

    assert result.successful_item_count == 5
    assert result.failed_item_count == 5
    assert result.has_errors is True
    assert len(result.errors) == 1
    assert result.errors[0].error_message == "batch 2 failed"


def test_batch_execute_invalid_batch_size() -> None:
    """batch_size=0 raises PineconeValueError (a ValueError subclass)."""
    with pytest.raises(PineconeValueError, match="batch_size must be >= 1") as exc_info:
        batch_execute(
            items=[{"id": "x"}],
            operation=lambda b: None,
            batch_size=0,
            show_progress=False,
        )
    assert type(exc_info.value) is PineconeValueError
    assert isinstance(exc_info.value, ValueError)


def test_batch_execute_invalid_max_concurrency_zero() -> None:
    """max_concurrency=0 raises PineconeValueError (a ValueError subclass)."""
    with pytest.raises(PineconeValueError, match="concurrency must be between") as exc_info:
        batch_execute(
            items=[{"id": "x"}],
            operation=lambda b: None,
            batch_size=1,
            max_concurrency=0,
            show_progress=False,
        )
    assert type(exc_info.value) is PineconeValueError
    assert isinstance(exc_info.value, ValueError)


def test_batch_execute_invalid_max_concurrency_too_high() -> None:
    """max_concurrency=65 raises PineconeValueError (a ValueError subclass)."""
    with pytest.raises(PineconeValueError, match="concurrency must be between") as exc_info:
        batch_execute(
            items=[{"id": "x"}],
            operation=lambda b: None,
            batch_size=1,
            max_concurrency=65,
            show_progress=False,
        )
    assert type(exc_info.value) is PineconeValueError
    assert isinstance(exc_info.value, ValueError)


# ---------------------------------------------------------------------------
# Async: async_batch_execute
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_batch_execute_invalid_batch_size() -> None:
    """Async: batch_size=0 raises PineconeValueError (mirrors the sync path)."""

    async def noop(batch: list[dict]) -> None:  # type: ignore[type-arg]
        pass

    with pytest.raises(PineconeValueError, match="batch_size must be >= 1") as exc_info:
        await async_batch_execute(
            items=[{"id": "x"}],
            operation=noop,
            batch_size=0,
            show_progress=False,
        )
    assert type(exc_info.value) is PineconeValueError
    assert isinstance(exc_info.value, ValueError)


@pytest.mark.asyncio
async def test_async_batch_execute_invalid_max_concurrency_zero() -> None:
    """Async: max_concurrency=0 raises PineconeValueError (mirrors the sync path)."""

    async def noop(batch: list[dict]) -> None:  # type: ignore[type-arg]
        pass

    with pytest.raises(PineconeValueError, match="concurrency must be between") as exc_info:
        await async_batch_execute(
            items=[{"id": "x"}],
            operation=noop,
            batch_size=1,
            max_concurrency=0,
            show_progress=False,
        )
    assert type(exc_info.value) is PineconeValueError
    assert isinstance(exc_info.value, ValueError)


@pytest.mark.asyncio
async def test_async_batch_execute_invalid_max_concurrency_too_high() -> None:
    """Async: max_concurrency=65 raises PineconeValueError (mirrors the sync path)."""

    async def noop(batch: list[dict]) -> None:  # type: ignore[type-arg]
        pass

    with pytest.raises(PineconeValueError, match="concurrency must be between") as exc_info:
        await async_batch_execute(
            items=[{"id": "x"}],
            operation=noop,
            batch_size=1,
            max_concurrency=65,
            show_progress=False,
        )
    assert type(exc_info.value) is PineconeValueError
    assert isinstance(exc_info.value, ValueError)


@pytest.mark.asyncio
async def test_async_batch_execute_empty_items() -> None:
    """Async: empty item list returns zero-count BatchResult."""

    async def noop(batch: list[dict]) -> None:  # type: ignore[type-arg]
        pass

    result = await async_batch_execute(items=[], operation=noop, batch_size=10, show_progress=False)

    assert isinstance(result, BatchResult)
    assert result.total_item_count == 0
    assert result.successful_item_count == 0
    assert result.failed_item_count == 0
    assert result.has_errors is False


@pytest.mark.asyncio
async def test_async_batch_execute_all_succeed() -> None:
    """Async: 10 items all succeed — counts match."""
    items = [{"id": str(i)} for i in range(10)]

    async def noop(batch: list[dict]) -> None:  # type: ignore[type-arg]
        pass

    result = await async_batch_execute(
        items=items,
        operation=noop,
        batch_size=3,
        max_concurrency=2,
        show_progress=False,
    )

    assert result.successful_item_count == 10
    assert result.failed_item_count == 0
    assert result.has_errors is False
    assert result.total_item_count == 10


@pytest.mark.asyncio
async def test_async_batch_execute_partial_failure() -> None:
    """Async: one batch raises — failure recorded, others succeed."""
    items = [{"id": str(i)} for i in range(10)]

    async def op(batch: list[dict]) -> None:  # type: ignore[type-arg]
        # fail the batch containing id "5"
        if any(item["id"] == "5" for item in batch):
            raise RuntimeError("async batch failed")

    result = await async_batch_execute(
        items=items,
        operation=op,
        batch_size=5,
        max_concurrency=1,
        show_progress=False,
    )

    assert result.has_errors is True
    assert len(result.errors) == 1
    assert result.failed_item_count == 5
    assert result.successful_item_count == 5


# ---------------------------------------------------------------------------
# _chunk helper
# ---------------------------------------------------------------------------


def test_chunk_helper() -> None:
    """_chunk splits list into sublists of the given size."""
    result = _chunk([1, 2, 3, 4, 5], 2)
    assert result == [[1, 2], [3, 4], [5]]


# ---------------------------------------------------------------------------
# Sync: response_info aggregation
# ---------------------------------------------------------------------------


def _make_items(n: int) -> list[dict[str, object]]:
    return [{"id": str(i)} for i in range(n)]


class TestBatchExecuteResponseInfo:
    def test_no_response_info_yields_none(self) -> None:
        def op(batch: list[dict[str, object]]) -> _FakeResp:
            return _FakeResp()

        result = batch_execute(
            items=_make_items(10),
            operation=op,
            batch_size=10,
            max_concurrency=1,
            show_progress=False,
        )
        assert result.response_info is None

    def test_aggregates_max_across_successful_batches(self) -> None:
        counter = [0]

        def op(batch: list[dict[str, object]]) -> _FakeResp:
            counter[0] += 1
            i = counter[0]
            return _FakeResp(
                response_info=ResponseInfo(
                    raw_headers={
                        "x-pinecone-lsn-reconciled": str(i * 10),
                        "x-pinecone-lsn-committed": str(i * 5),
                    }
                )
            )

        result = batch_execute(
            items=_make_items(30),
            operation=op,
            batch_size=10,
            max_concurrency=1,
            show_progress=False,
        )
        assert result.response_info is not None
        assert result.response_info.lsn_reconciled == 30
        assert result.response_info.lsn_committed == 15

    def test_failed_batches_excluded(self) -> None:
        counter = [0]

        def op(batch: list[dict[str, object]]) -> _FakeResp:
            counter[0] += 1
            if counter[0] == 2:
                raise RuntimeError("middle batch failed")
            return _FakeResp(
                response_info=ResponseInfo(raw_headers={"x-pinecone-lsn-reconciled": "50"})
            )

        result = batch_execute(
            items=_make_items(30),
            operation=op,
            batch_size=10,
            max_concurrency=1,
            show_progress=False,
        )
        assert result.response_info is not None
        assert result.response_info.lsn_reconciled == 50

    def test_all_failed_yields_none(self) -> None:
        def op(batch: list[dict[str, object]]) -> _FakeResp:
            raise RuntimeError("always fails")

        result = batch_execute(
            items=_make_items(20),
            operation=op,
            batch_size=10,
            max_concurrency=1,
            show_progress=False,
        )
        assert result.response_info is None

    def test_partial_lsn_coverage(self) -> None:
        counter = [0]

        def op(batch: list[dict[str, object]]) -> _FakeResp:
            counter[0] += 1
            if counter[0] == 1:
                return _FakeResp(
                    response_info=ResponseInfo(raw_headers={"x-pinecone-lsn-reconciled": "42"})
                )
            return _FakeResp()

        result = batch_execute(
            items=_make_items(30),
            operation=op,
            batch_size=10,
            max_concurrency=1,
            show_progress=False,
        )
        assert result.response_info is not None
        assert result.response_info.lsn_reconciled == 42

    def test_only_lsn_reconciled_no_committed(self) -> None:
        def op(batch: list[dict[str, object]]) -> _FakeResp:
            return _FakeResp(
                response_info=ResponseInfo(raw_headers={"x-pinecone-lsn-reconciled": "7"})
            )

        result = batch_execute(
            items=_make_items(20),
            operation=op,
            batch_size=10,
            max_concurrency=1,
            show_progress=False,
        )
        assert result.response_info is not None
        assert result.response_info.lsn_reconciled == 7
        assert result.response_info.lsn_committed is None

    def test_operation_returning_none_no_raise(self) -> None:
        def op(batch: list[dict[str, object]]) -> None:
            return None

        result = batch_execute(
            items=_make_items(10),
            operation=op,
            batch_size=10,
            max_concurrency=1,
            show_progress=False,
        )
        assert result.response_info is None

    def test_operation_returning_plain_object_no_raise(self) -> None:
        def op(batch: list[dict[str, object]]) -> object:
            return object()

        result = batch_execute(
            items=_make_items(10),
            operation=op,
            batch_size=10,
            max_concurrency=1,
            show_progress=False,
        )
        assert result.response_info is None

    def test_malformed_response_info_no_raise(self) -> None:
        class _BadResp:
            response_info = object()

        def op(batch: list[dict[str, object]]) -> _BadResp:
            return _BadResp()

        result = batch_execute(
            items=_make_items(10),
            operation=op,
            batch_size=10,
            max_concurrency=1,
            show_progress=False,
        )
        assert result.response_info is None
