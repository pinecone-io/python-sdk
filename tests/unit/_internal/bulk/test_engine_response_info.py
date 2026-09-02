"""LSN aggregation and result-shape tolerance, driven through the gate engines.

``_collect_lsn`` and ``_build_aggregate`` live in ``_internal/batch.py`` and
are imported by both bulk engines, so they are live code — but every test of
their edges used to run through the pre-gate ``batch_execute``. Deleting that
engine (D7) would have taken this coverage with it, and nothing in the rest of
the bulk suite replaces it: the vector surface's batched-upsert tests cover
"aggregated across batches" and "None when no LSN" only, and only on the sync
transport.

Two properties are asserted here that a naive aggregate would get wrong:

- a failed batch contributes nothing, so a run whose highest LSN came from a
  batch that then failed must not report it;
- the operation's return value is untrusted. It may be ``None``, an arbitrary
  object, or carry a ``response_info`` of the wrong type — the engine reads it
  with ``getattr`` and must degrade to ``response_info=None`` rather than
  raise, because an aggregation bug must not turn a successful ingest into an
  exception.

Validation is included for the same reason: ``_validate_batch_params`` is the
engines' guard, and its out-of-range cases were only ever exercised through
the deleted engine at this level.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from msgspec import Struct

from pinecone._internal.bulk.async_engine import bulk_execute_async
from pinecone._internal.bulk.engine import bulk_execute_sync
from pinecone.errors import PineconeValueError
from pinecone.models.batch import BatchResult
from pinecone.models.response_info import ResponseInfo

HOST = "engine-response-info.svc.pinecone.io"


class _FakeResp(Struct, kw_only=True):
    response_info: ResponseInfo | None = None


def _items(n: int) -> list[dict[str, Any]]:
    return [{"id": str(i)} for i in range(n)]


def _run_sync(operation: Any, n: int = 10, batch_size: int = 10) -> BatchResult:
    return bulk_execute_sync(
        items=_items(n),
        operation=operation,
        batch_size=batch_size,
        max_concurrency=1,
        show_progress=False,
        host=HOST,
    )


def _run_async(operation: Any, n: int = 10, batch_size: int = 10) -> BatchResult:
    async def _go() -> BatchResult:
        return await bulk_execute_async(
            items=_items(n),
            operation=operation,
            batch_size=batch_size,
            max_concurrency=1,
            show_progress=False,
            host=HOST,
        )

    return asyncio.run(_go())


def _headers(reconciled: str | None = None, committed: str | None = None) -> _FakeResp:
    raw: dict[str, str] = {}
    if reconciled is not None:
        raw["x-pinecone-lsn-reconciled"] = reconciled
    if committed is not None:
        raw["x-pinecone-lsn-committed"] = committed
    return _FakeResp(response_info=ResponseInfo(raw_headers=raw))


class TestSyncEngineResponseInfo:
    def test_no_response_info_yields_none(self) -> None:
        assert _run_sync(lambda batch: _FakeResp()).response_info is None

    def test_aggregates_max_across_successful_batches(self) -> None:
        counter = [0]

        def op(batch: list[dict[str, Any]]) -> _FakeResp:
            counter[0] += 1
            i = counter[0]
            return _headers(str(i * 10), str(i * 5))

        result = _run_sync(op, n=30)
        assert result.response_info is not None
        assert result.response_info.lsn_reconciled == 30
        assert result.response_info.lsn_committed == 15

    def test_failed_batches_excluded(self) -> None:
        counter = [0]

        def op(batch: list[dict[str, Any]]) -> _FakeResp:
            counter[0] += 1
            if counter[0] == 2:
                raise RuntimeError("middle batch failed")
            return _headers("50")

        result = _run_sync(op, n=30)
        assert result.response_info is not None
        assert result.response_info.lsn_reconciled == 50

    def test_all_failed_yields_none(self) -> None:
        def op(batch: list[dict[str, Any]]) -> _FakeResp:
            raise RuntimeError("always fails")

        result = _run_sync(op, n=20)
        assert result.response_info is None
        assert result.failed_item_count == 20

    def test_partial_lsn_coverage(self) -> None:
        counter = [0]

        def op(batch: list[dict[str, Any]]) -> _FakeResp:
            counter[0] += 1
            return _headers("42") if counter[0] == 1 else _FakeResp()

        result = _run_sync(op, n=30)
        assert result.response_info is not None
        assert result.response_info.lsn_reconciled == 42

    def test_only_lsn_reconciled_no_committed(self) -> None:
        result = _run_sync(lambda batch: _headers("7"), n=20)
        assert result.response_info is not None
        assert result.response_info.lsn_reconciled == 7
        assert result.response_info.lsn_committed is None

    def test_operation_returning_none_no_raise(self) -> None:
        result = _run_sync(lambda batch: None)
        assert result.response_info is None
        assert result.successful_item_count == 10

    def test_operation_returning_plain_object_no_raise(self) -> None:
        result = _run_sync(lambda batch: object())
        assert result.response_info is None
        assert result.successful_item_count == 10

    def test_malformed_response_info_no_raise(self) -> None:
        class _BadResp:
            response_info = object()

        result = _run_sync(lambda batch: _BadResp())
        assert result.response_info is None
        assert result.successful_item_count == 10


class TestAsyncEngineResponseInfo:
    def test_no_response_info_yields_none(self) -> None:
        async def op(batch: list[dict[str, Any]]) -> _FakeResp:
            return _FakeResp()

        assert _run_async(op).response_info is None

    def test_aggregates_max_across_successful_batches(self) -> None:
        counter = [0]

        async def op(batch: list[dict[str, Any]]) -> _FakeResp:
            counter[0] += 1
            i = counter[0]
            return _headers(str(i * 10), str(i * 5))

        result = _run_async(op, n=30)
        assert result.response_info is not None
        assert result.response_info.lsn_reconciled == 30
        assert result.response_info.lsn_committed == 15

    def test_failed_batches_excluded(self) -> None:
        counter = [0]

        async def op(batch: list[dict[str, Any]]) -> _FakeResp:
            counter[0] += 1
            if counter[0] == 2:
                raise RuntimeError("middle batch failed")
            return _headers("50")

        result = _run_async(op, n=30)
        assert result.response_info is not None
        assert result.response_info.lsn_reconciled == 50

    def test_all_failed_yields_none(self) -> None:
        async def op(batch: list[dict[str, Any]]) -> _FakeResp:
            raise RuntimeError("always fails")

        result = _run_async(op, n=20)
        assert result.response_info is None
        assert result.failed_item_count == 20

    def test_partial_lsn_coverage(self) -> None:
        counter = [0]

        async def op(batch: list[dict[str, Any]]) -> _FakeResp:
            counter[0] += 1
            return _headers("42") if counter[0] == 1 else _FakeResp()

        result = _run_async(op, n=30)
        assert result.response_info is not None
        assert result.response_info.lsn_reconciled == 42

    def test_only_lsn_reconciled_no_committed(self) -> None:
        async def op(batch: list[dict[str, Any]]) -> _FakeResp:
            return _headers("7")

        result = _run_async(op, n=20)
        assert result.response_info is not None
        assert result.response_info.lsn_reconciled == 7
        assert result.response_info.lsn_committed is None

    def test_operation_returning_none_no_raise(self) -> None:
        async def op(batch: list[dict[str, Any]]) -> None:
            return None

        result = _run_async(op)
        assert result.response_info is None
        assert result.successful_item_count == 10

    def test_operation_returning_plain_object_no_raise(self) -> None:
        async def op(batch: list[dict[str, Any]]) -> object:
            return object()

        result = _run_async(op)
        assert result.response_info is None
        assert result.successful_item_count == 10

    def test_malformed_response_info_no_raise(self) -> None:
        class _BadResp:
            response_info = object()

        async def op(batch: list[dict[str, Any]]) -> _BadResp:
            return _BadResp()

        result = _run_async(op)
        assert result.response_info is None
        assert result.successful_item_count == 10


class TestEngineParamValidation:
    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"batch_size": 0}, "batch_size must be >= 1"),
            ({"max_concurrency": 0}, "concurrency must be between 1 and 64"),
            ({"max_concurrency": 65}, "concurrency must be between 1 and 64"),
        ],
    )
    def test_sync_rejects_out_of_range(self, kwargs: dict[str, int], message: str) -> None:
        call = {"batch_size": 10, "max_concurrency": 1, **kwargs}
        with pytest.raises(PineconeValueError, match=message):
            bulk_execute_sync(
                items=_items(10),
                operation=lambda batch: {"upserted_count": len(batch)},
                show_progress=False,
                host=HOST,
                **call,
            )

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"batch_size": 0}, "batch_size must be >= 1"),
            ({"max_concurrency": 0}, "concurrency must be between 1 and 64"),
            ({"max_concurrency": 65}, "concurrency must be between 1 and 64"),
        ],
    )
    def test_async_rejects_out_of_range(self, kwargs: dict[str, int], message: str) -> None:
        call = {"batch_size": 10, "max_concurrency": 1, **kwargs}

        async def op(batch: list[dict[str, Any]]) -> dict[str, int]:
            return {"upserted_count": len(batch)}

        async def _go() -> None:
            await bulk_execute_async(
                items=_items(10),
                operation=op,
                show_progress=False,
                host=HOST,
                **call,
            )

        with pytest.raises(PineconeValueError, match=message):
            asyncio.run(_go())

    def test_validation_precedes_the_empty_short_circuit(self) -> None:
        """Order matters: an empty list with a bad batch_size must still raise,
        or a caller only learns their parameters are wrong once they happen to
        pass a non-empty list."""
        with pytest.raises(PineconeValueError):
            bulk_execute_sync(
                items=[],
                operation=lambda batch: None,
                batch_size=0,
                show_progress=False,
                host=HOST,
            )
