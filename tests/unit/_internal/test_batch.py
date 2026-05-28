"""Unit tests for adaptive concurrency in pinecone._internal.batch."""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

import pytest

from pinecone._internal.adaptive import _AdaptiveLimiterRegistry
from pinecone._internal.batch import async_batch_execute, batch_execute


class TestAdaptiveBatchExecute:
    def test_uses_semaphore_when_no_limiter(self) -> None:
        """When no limiter_registry is passed, behavior is identical to baseline."""

        async def run() -> None:
            items = [{"id": str(i), "values": [0.1, 0.2]} for i in range(6)]

            async def op(batch: list[dict[str, Any]]) -> Any:
                return {"upserted_count": len(batch)}

            result = await async_batch_execute(
                items=items,
                operation=op,
                batch_size=2,
                max_concurrency=4,
                show_progress=False,
            )
            assert result.successful_item_count == 6
            assert result.failed_item_count == 0

        asyncio.get_event_loop().run_until_complete(run())

    def test_uses_limiter_when_provided(self) -> None:
        """When a limiter is pre-throttled, observed max concurrency must not exceed it."""

        async def run() -> None:
            registry = _AdaptiveLimiterRegistry()
            limiter = registry.get("test-host", 8)
            # Simulate an upfront throttle that halves the limit before any work runs
            limiter.report_throttled()  # 8 → 4
            assert limiter.current_limit() == 4

            inflight_observed = 0
            max_observed = [0]
            lock = asyncio.Lock()

            async def slow_op(batch: list[dict[str, Any]]) -> Any:
                nonlocal inflight_observed
                async with lock:
                    inflight_observed += 1
                    max_observed[0] = max(max_observed[0], inflight_observed)
                await asyncio.sleep(0.1)
                async with lock:
                    inflight_observed -= 1
                return {"upserted_count": len(batch)}

            items = [{"id": str(i), "values": [0.1, 0.2]} for i in range(20)]
            await async_batch_execute(
                items=items,
                operation=slow_op,
                batch_size=2,  # 10 batches
                max_concurrency=8,
                show_progress=False,
                limiter_registry=registry,
                host="test-host",
            )
            assert max_observed[0] <= 4, (
                f"observed max concurrency {max_observed[0]} exceeds limiter cap"
            )

        asyncio.get_event_loop().run_until_complete(run())

    def test_recovers_when_limiter_increases(self) -> None:
        """After throttle, limiter recovers as successes are reported externally."""

        async def run() -> None:
            registry = _AdaptiveLimiterRegistry()
            limiter = registry.get("test-host", 8)
            # Throttle all the way down to 1
            for _ in range(4):
                limiter.report_throttled()
            assert limiter.current_limit() == 1

            call_count = 0
            max_observed: list[int] = []

            async def counting_op(batch: list[dict[str, Any]]) -> Any:
                nonlocal call_count
                call_count += 1
                # After 3 calls, recover the limiter by reporting successes
                if call_count == 3:
                    for _ in range(10):
                        limiter.report_success()
                max_observed.append(limiter.current_limit())
                return {"upserted_count": len(batch)}

            items = [{"id": str(i), "values": [0.1]} for i in range(20)]
            await async_batch_execute(
                items=items,
                operation=counting_op,
                batch_size=2,
                max_concurrency=8,
                show_progress=False,
                limiter_registry=registry,
                host="test-host",
            )
            # Later batches should have run at a higher concurrency ceiling
            assert max(max_observed) > 1, "limiter should have recovered above 1"

        asyncio.get_event_loop().run_until_complete(run())

    def test_returns_all_items_counted(self) -> None:
        """All items are processed and counted even when limiter is active."""

        async def run() -> None:
            registry = _AdaptiveLimiterRegistry()
            items = [{"id": str(i), "values": [float(i)]} for i in range(15)]

            async def op(batch: list[dict[str, Any]]) -> Any:
                return {"upserted_count": len(batch)}

            result = await async_batch_execute(
                items=items,
                operation=op,
                batch_size=3,
                max_concurrency=4,
                show_progress=False,
                limiter_registry=registry,
                host="test-host",
            )
            assert result.successful_item_count == 15
            assert result.failed_item_count == 0
            assert result.total_item_count == 15

        asyncio.get_event_loop().run_until_complete(run())

    def test_errors_collected_with_limiter(self) -> None:
        """Errors are captured per-batch even when limiter is active."""

        async def run() -> None:
            registry = _AdaptiveLimiterRegistry()
            call_count = 0

            async def sometimes_fails(batch: list[dict[str, Any]]) -> Any:
                nonlocal call_count
                call_count += 1
                if call_count % 2 == 0:
                    raise RuntimeError("simulated failure")
                return {"upserted_count": len(batch)}

            items = [{"id": str(i)} for i in range(8)]
            result = await async_batch_execute(
                items=items,
                operation=sometimes_fails,
                batch_size=2,
                max_concurrency=2,
                show_progress=False,
                limiter_registry=registry,
                host="test-host",
            )
            assert result.failed_batch_count > 0
            assert result.successful_batch_count > 0
            assert result.total_batch_count == 4

        asyncio.get_event_loop().run_until_complete(run())

    @pytest.mark.asyncio
    async def test_uses_limiter_when_provided_async(self) -> None:
        """Async variant using pytest-asyncio marker."""
        registry = _AdaptiveLimiterRegistry()
        limiter = registry.get("host2", 6)
        limiter.report_throttled()  # 6 → 3
        assert limiter.current_limit() == 3

        inflight = 0
        peak = [0]
        lock = asyncio.Lock()

        async def slow(batch: list[dict[str, Any]]) -> Any:
            nonlocal inflight
            async with lock:
                inflight += 1
                peak[0] = max(peak[0], inflight)
            await asyncio.sleep(0.05)
            async with lock:
                inflight -= 1
            return {"upserted_count": len(batch)}

        items = [{"id": str(i)} for i in range(12)]
        await async_batch_execute(
            items=items,
            operation=slow,
            batch_size=1,
            max_concurrency=6,
            show_progress=False,
            limiter_registry=registry,
            host="host2",
        )
        assert peak[0] <= 3


class TestAdaptiveBatchExecuteSync:
    def test_sync_no_limiter_unchanged(self) -> None:
        """Without a limiter_registry, behavior is identical to baseline."""
        items = [{"id": str(i), "values": [0.1, 0.2]} for i in range(6)]

        def op(batch: list[dict[str, Any]]) -> Any:
            return {"upserted_count": len(batch)}

        result = batch_execute(
            items=items,
            operation=op,
            batch_size=2,
            max_concurrency=4,
            show_progress=False,
        )
        assert result.successful_item_count == 6
        assert result.failed_item_count == 0
        assert result.total_item_count == 6

    def test_sync_uses_limiter(self) -> None:
        """With a pre-throttled limiter (4), max observed inflight must not exceed 4."""
        registry = _AdaptiveLimiterRegistry()
        limiter = registry.get("test-host", 8)
        limiter.report_throttled()  # 8 → 4
        assert limiter.current_limit() == 4

        inflight_counter = 0
        max_observed = [0]
        lock = threading.Lock()

        def slow_op(batch: list[dict[str, Any]]) -> Any:
            nonlocal inflight_counter
            with lock:
                inflight_counter += 1
                max_observed[0] = max(max_observed[0], inflight_counter)
            time.sleep(0.05)
            with lock:
                inflight_counter -= 1
            return {"upserted_count": len(batch)}

        items = [{"id": str(i), "values": [0.1, 0.2]} for i in range(20)]
        result = batch_execute(
            items=items,
            operation=slow_op,
            batch_size=2,  # 10 batches
            max_concurrency=8,
            show_progress=False,
            limiter_registry=registry,
            host="test-host",
        )
        assert max_observed[0] <= 4, (
            f"observed max concurrency {max_observed[0]} exceeds limiter cap"
        )
        assert result.successful_item_count == 20
        assert result.failed_item_count == 0

    def test_sync_concurrency_recovers_after_signals(self) -> None:
        """Concurrency recovers as limiter is signalled with successes mid-execution."""
        registry = _AdaptiveLimiterRegistry()
        limiter = registry.get("test-host", 8)
        # Pre-throttle to 1
        for _ in range(3):
            limiter.report_throttled()
        assert limiter.current_limit() == 1

        call_count = 0
        call_lock = threading.Lock()
        limits_observed: list[int] = []

        def op(batch: list[dict[str, Any]]) -> Any:
            nonlocal call_count
            with call_lock:
                call_count += 1
                current = call_count
            # After 3 calls, recover by reporting successes
            if current == 3:
                for _ in range(20):
                    limiter.report_success()
            limits_observed.append(limiter.current_limit())
            return {"upserted_count": len(batch)}

        items = [{"id": str(i)} for i in range(20)]  # 10 batches of 2
        result = batch_execute(
            items=items,
            operation=op,
            batch_size=2,
            max_concurrency=8,
            show_progress=False,
            limiter_registry=registry,
            host="test-host",
        )
        assert result.successful_item_count == 20
        assert result.failed_item_count == 0
        assert max(limits_observed) > 1, "limiter should have recovered above 1"
