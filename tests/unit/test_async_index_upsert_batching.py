"""Unit tests for AsyncIndex.upsert() batch_size, max_concurrency, and show_progress parameters."""

from __future__ import annotations

import asyncio
import sys
from typing import Any
from unittest.mock import patch

import httpx
import pytest
import respx

from pinecone import AsyncIndex
from pinecone.errors.exceptions import PineconeValueError
from pinecone.models.response_info import ResponseInfo
from pinecone.models.vectors.responses import UpsertResponse
from pinecone.models.vectors.vector import Vector

INDEX_HOST = "test-index-abc1234.svc.us-east1-gcp.pinecone.io"
INDEX_HOST_HTTPS = f"https://{INDEX_HOST}"
UPSERT_URL = f"{INDEX_HOST_HTTPS}/vectors/upsert"


def _make_upsert_response(*, upserted_count: int) -> dict[str, object]:
    return {"upsertedCount": upserted_count}


def _make_async_index() -> AsyncIndex:
    return AsyncIndex(host=INDEX_HOST, api_key="test-key")


def _make_vectors(n: int) -> list[Vector]:
    return [Vector(id=f"v{i}", values=[float(i), float(i + 1)]) for i in range(n)]


class TestAsyncUpsertNoBatchSize:
    """batch_size=None (default) sends a single request."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_upsert_no_batch_size_sends_single_request(self) -> None:
        route = respx.post(UPSERT_URL).mock(
            return_value=httpx.Response(200, json=_make_upsert_response(upserted_count=100))
        )
        idx = _make_async_index()
        result = await idx.upsert(vectors=_make_vectors(100), batch_size=None)
        assert len(route.calls) == 1
        assert result.upserted_count == 100

    @respx.mock
    @pytest.mark.asyncio
    async def test_upsert_response_info_preserved_when_not_batched(self) -> None:
        respx.post(UPSERT_URL).mock(
            return_value=httpx.Response(
                200,
                json=_make_upsert_response(upserted_count=5),
                headers={"X-Pinecone-Request-Id": "req-abc123"},
            )
        )
        idx = _make_async_index()
        result = await idx.upsert(vectors=_make_vectors(5))
        assert result.response_info is not None


class TestAsyncUpsertWithBatchSize:
    """batch_size=N splits vectors and sends one request per chunk."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_upsert_with_batch_size_sends_multiple_requests(self) -> None:
        route = respx.post(UPSERT_URL).mock(
            return_value=httpx.Response(200, json=_make_upsert_response(upserted_count=100))
        )
        idx = _make_async_index()
        await idx.upsert(vectors=_make_vectors(250), batch_size=100, show_progress=False)
        assert len(route.calls) == 3

        import orjson

        batch_sizes = sorted(
            len(orjson.loads(call.request.content)["vectors"]) for call in route.calls
        )
        assert batch_sizes == [50, 100, 100]

    @respx.mock
    @pytest.mark.asyncio
    async def test_upsert_with_batch_size_aggregates_response(self) -> None:
        respx.post(UPSERT_URL).mock(
            return_value=httpx.Response(200, json=_make_upsert_response(upserted_count=100))
        )
        idx = _make_async_index()
        result = await idx.upsert(vectors=_make_vectors(250), batch_size=100, show_progress=False)
        assert isinstance(result, UpsertResponse)
        assert result.upserted_count == 250

    @respx.mock
    @pytest.mark.asyncio
    async def test_upsert_response_info_is_none_when_batched(self) -> None:
        respx.post(UPSERT_URL).mock(
            return_value=httpx.Response(200, json=_make_upsert_response(upserted_count=100))
        )
        idx = _make_async_index()
        result = await idx.upsert(vectors=_make_vectors(250), batch_size=100, show_progress=False)
        assert result.response_info is None

    @respx.mock
    @pytest.mark.asyncio
    async def test_upsert_namespace_forwarded_per_batch(self) -> None:
        route = respx.post(UPSERT_URL).mock(
            return_value=httpx.Response(200, json=_make_upsert_response(upserted_count=5))
        )
        idx = _make_async_index()
        await idx.upsert(
            vectors=_make_vectors(10), batch_size=5, namespace="my-ns", show_progress=False
        )

        import orjson

        for call in route.calls:
            body = orjson.loads(call.request.content)
            assert body.get("namespace") == "my-ns"

    @respx.mock
    @pytest.mark.asyncio
    async def test_upsert_timeout_forwarded_per_batch(self) -> None:
        route = respx.post(UPSERT_URL).mock(
            return_value=httpx.Response(200, json=_make_upsert_response(upserted_count=5))
        )
        idx = _make_async_index()
        await idx.upsert(vectors=_make_vectors(10), batch_size=5, timeout=5.0, show_progress=False)
        assert len(route.calls) == 2

    @respx.mock
    @pytest.mark.asyncio
    async def test_upsert_empty_vectors_with_batch_size(self) -> None:
        route = respx.post(UPSERT_URL).mock(
            return_value=httpx.Response(200, json=_make_upsert_response(upserted_count=0))
        )
        idx = _make_async_index()
        result = await idx.upsert(vectors=[], batch_size=100, show_progress=False)
        assert len(route.calls) == 0
        assert result.upserted_count == 0


class TestAsyncUpsertInvalidBatchSize:
    """Invalid batch_size values raise PineconeValueError."""

    @pytest.mark.asyncio
    async def test_upsert_invalid_batch_size_raises(self) -> None:
        idx = _make_async_index()
        with pytest.raises(PineconeValueError):
            await idx.upsert(vectors=_make_vectors(5), batch_size=0)

    @pytest.mark.asyncio
    async def test_upsert_invalid_batch_size_negative(self) -> None:
        idx = _make_async_index()
        with pytest.raises(PineconeValueError):
            await idx.upsert(vectors=_make_vectors(5), batch_size=-1)

    @pytest.mark.asyncio
    async def test_upsert_invalid_batch_size_float(self) -> None:
        idx = _make_async_index()
        with pytest.raises(PineconeValueError):
            await idx.upsert(vectors=_make_vectors(5), batch_size=1.5)  # type: ignore[arg-type]


class TestAsyncUpsertShowProgress:
    """show_progress behavior with and without tqdm."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_upsert_show_progress_false_does_not_import_tqdm(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "tqdm", None)  # type: ignore[arg-type]
        monkeypatch.setitem(sys.modules, "tqdm.auto", None)  # type: ignore[arg-type]
        respx.post(UPSERT_URL).mock(
            return_value=httpx.Response(200, json=_make_upsert_response(upserted_count=5))
        )
        idx = _make_async_index()
        result = await idx.upsert(vectors=_make_vectors(10), batch_size=5, show_progress=False)
        assert result.upserted_count == 10

    @respx.mock
    @pytest.mark.asyncio
    async def test_upsert_show_progress_true_works_without_tqdm(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "tqdm", None)  # type: ignore[arg-type]
        monkeypatch.setitem(sys.modules, "tqdm.auto", None)  # type: ignore[arg-type]
        respx.post(UPSERT_URL).mock(
            return_value=httpx.Response(200, json=_make_upsert_response(upserted_count=5))
        )
        idx = _make_async_index()
        result = await idx.upsert(vectors=_make_vectors(10), batch_size=5, show_progress=True)
        assert result.upserted_count == 10


class TestAsyncUpsertConcurrency:
    """Batches are submitted concurrently, not sequentially."""

    @pytest.mark.asyncio
    async def test_upsert_batches_run_concurrently(self) -> None:
        n_batches = 3
        all_started: asyncio.Event = asyncio.Event()
        gate: asyncio.Event = asyncio.Event()
        start_count = 0

        async def mock_upsert_dict_batch(
            *,
            items: list[dict[str, Any]],
            namespace: str,
            timeout: float | None,
        ) -> UpsertResponse:
            nonlocal start_count
            start_count += 1
            if start_count >= n_batches:
                all_started.set()
            await gate.wait()
            return UpsertResponse(upserted_count=len(items))

        idx = _make_async_index()
        with patch.object(idx, "_upsert_dict_batch", side_effect=mock_upsert_dict_batch):
            task = asyncio.create_task(
                idx.upsert(vectors=_make_vectors(15), batch_size=5, show_progress=False)
            )
            # All 3 batches must start before any returns (proves concurrency)
            await asyncio.wait_for(all_started.wait(), timeout=5.0)
            gate.set()
            result = await asyncio.wait_for(task, timeout=5.0)

        assert result.upserted_count == 15

    @pytest.mark.asyncio
    async def test_upsert_max_concurrency_bounds_in_flight(self) -> None:
        peak = 0
        in_flight = 0

        async def mock_upsert_dict_batch(
            *,
            items: list[dict[str, Any]],
            namespace: str,
            timeout: float | None,
        ) -> UpsertResponse:
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0)  # yield so other waiting batches can attempt to start
            in_flight -= 1
            return UpsertResponse(upserted_count=len(items))

        idx = _make_async_index()
        with patch.object(idx, "_upsert_dict_batch", side_effect=mock_upsert_dict_batch):
            result = await idx.upsert(
                vectors=_make_vectors(25), batch_size=5, max_concurrency=2, show_progress=False
            )

        assert result.upserted_count == 25
        assert peak <= 2

    def test_upsert_max_concurrency_default_is_8(self) -> None:
        import inspect

        idx = _make_async_index()
        sig = inspect.signature(idx.upsert)
        assert sig.parameters["max_concurrency"].default == 8


class TestAsyncUpsertMaxConcurrencyValidation:
    """max_concurrency values outside [1, 64] raise PineconeValueError."""

    @pytest.mark.asyncio
    async def test_upsert_invalid_max_concurrency_raises(self) -> None:
        idx = _make_async_index()
        for invalid in (0, 65, -1):
            with pytest.raises(PineconeValueError):
                await idx.upsert(vectors=_make_vectors(10), batch_size=5, max_concurrency=invalid)


class TestAsyncUpsertPartialFailure:
    """Partial batch failures return a rich UpsertResponse instead of raising."""

    @pytest.mark.asyncio
    async def test_upsert_partial_failure_returns_rich_response(self) -> None:
        call_count = 0

        async def mock_upsert_dict_batch(
            *,
            items: list[dict[str, Any]],
            namespace: str,
            timeout: float | None,
        ) -> UpsertResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("network error on batch 1")
            return UpsertResponse(upserted_count=len(items))

        idx = _make_async_index()
        with patch.object(idx, "_upsert_dict_batch", side_effect=mock_upsert_dict_batch):
            result = await idx.upsert(vectors=_make_vectors(15), batch_size=5, show_progress=False)

        assert result.has_errors is True
        assert result.failed_batch_count == 1
        assert result.failed_item_count == 5
        assert len(result.errors) == 1
        assert isinstance(result.errors[0].error, ValueError)
        assert result.upserted_count == 10  # 2 successful batches

    @pytest.mark.asyncio
    async def test_upsert_failed_items_for_retry(self) -> None:
        call_count = 0

        async def mock_upsert_dict_batch(
            *,
            items: list[dict[str, Any]],
            namespace: str,
            timeout: float | None,
        ) -> UpsertResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("batch failed")
            return UpsertResponse(upserted_count=len(items))

        idx = _make_async_index()
        with patch.object(idx, "_upsert_dict_batch", side_effect=mock_upsert_dict_batch):
            result = await idx.upsert(vectors=_make_vectors(10), batch_size=5, show_progress=False)

        assert result.has_errors
        failed = result.failed_items
        assert len(failed) == 5
        assert all(isinstance(item, dict) for item in failed)

    @pytest.mark.asyncio
    async def test_upsert_response_info_aggregated_across_batches(self) -> None:
        lsns = [10, 20, 15]
        lsn_idx = 0

        async def mock_upsert_dict_batch(
            *,
            items: list[dict[str, Any]],
            namespace: str,
            timeout: float | None,
        ) -> UpsertResponse:
            nonlocal lsn_idx
            lsn = lsns[lsn_idx]
            lsn_idx += 1
            ri = ResponseInfo(raw_headers={"x-pinecone-lsn-committed": str(lsn)})
            return UpsertResponse(upserted_count=len(items), response_info=ri)

        idx = _make_async_index()
        with patch.object(idx, "_upsert_dict_batch", side_effect=mock_upsert_dict_batch):
            result = await idx.upsert(vectors=_make_vectors(15), batch_size=5, show_progress=False)

        assert result.response_info is not None
        assert result.response_info.lsn_committed == 20  # max([10, 20, 15])
