"""Unit tests for Index.upsert() batch_size and show_progress parameters."""

from __future__ import annotations

import sys
from collections import Counter
from unittest.mock import patch

import httpx
import orjson
import pytest
import respx

from pinecone import Index
from pinecone._internal.bulk import bulk_execute_sync
from pinecone.errors.exceptions import PineconeValueError
from pinecone.models.vectors.responses import UpsertResponse
from pinecone.models.vectors.vector import Vector

INDEX_HOST = "test-index-abc1234.svc.us-east1-gcp.pinecone.io"
INDEX_HOST_HTTPS = f"https://{INDEX_HOST}"
UPSERT_URL = f"{INDEX_HOST_HTTPS}/vectors/upsert"


def _make_upsert_response(*, upserted_count: int) -> dict[str, object]:
    return {"upsertedCount": upserted_count}


def _make_index() -> Index:
    return Index(host=INDEX_HOST, api_key="test-key")


def _make_vectors(n: int) -> list[Vector]:
    return [Vector(id=f"v{i}", values=[float(i), float(i + 1)]) for i in range(n)]


class TestUpsertNoBatchSize:
    """batch_size=None (default) sends a single request."""

    @respx.mock
    def test_upsert_no_batch_size_sends_single_request(self) -> None:
        route = respx.post(UPSERT_URL).mock(
            return_value=httpx.Response(200, json=_make_upsert_response(upserted_count=100))
        )
        idx = _make_index()
        result = idx.upsert(vectors=_make_vectors(100), batch_size=None)
        assert len(route.calls) == 1
        assert result.upserted_count == 100

    @respx.mock
    def test_upsert_default_batch_size_sends_single_request(self) -> None:
        route = respx.post(UPSERT_URL).mock(
            return_value=httpx.Response(200, json=_make_upsert_response(upserted_count=50))
        )
        idx = _make_index()
        result = idx.upsert(vectors=_make_vectors(50))
        assert len(route.calls) == 1
        assert result.upserted_count == 50

    @respx.mock
    def test_upsert_response_info_preserved_when_not_batched(self) -> None:
        respx.post(UPSERT_URL).mock(
            return_value=httpx.Response(
                200,
                json=_make_upsert_response(upserted_count=5),
                headers={"X-Pinecone-Request-Id": "req-abc123"},
            )
        )
        idx = _make_index()
        result = idx.upsert(vectors=_make_vectors(5))
        assert result.response_info is not None


class TestUpsertWithBatchSize:
    """batch_size=N splits vectors and sends one request per chunk."""

    @respx.mock
    def test_upsert_with_batch_size_sends_multiple_requests(self) -> None:
        route = respx.post(UPSERT_URL).mock(
            side_effect=lambda req: httpx.Response(
                200,
                json=_make_upsert_response(
                    upserted_count=len(orjson.loads(req.content)["vectors"])
                ),
            )
        )
        idx = _make_index()
        idx.upsert(vectors=_make_vectors(250), batch_size=100, show_progress=False)
        assert len(route.calls) == 3
        sizes = Counter(len(orjson.loads(call.request.content)["vectors"]) for call in route.calls)
        assert sizes == Counter({100: 2, 50: 1})

    @respx.mock
    def test_upsert_with_batch_size_aggregates_response(self) -> None:
        respx.post(UPSERT_URL).mock(
            side_effect=lambda req: httpx.Response(
                200,
                json=_make_upsert_response(
                    upserted_count=len(orjson.loads(req.content)["vectors"])
                ),
            )
        )
        idx = _make_index()
        result = idx.upsert(vectors=_make_vectors(250), batch_size=100, show_progress=False)
        assert isinstance(result, UpsertResponse)
        assert result.upserted_count == 250

    @respx.mock
    def test_upsert_response_info_is_none_when_batched_no_lsn(self) -> None:
        respx.post(UPSERT_URL).mock(
            side_effect=lambda req: httpx.Response(
                200,
                json=_make_upsert_response(
                    upserted_count=len(orjson.loads(req.content)["vectors"])
                ),
            )
        )
        idx = _make_index()
        result = idx.upsert(vectors=_make_vectors(250), batch_size=100, show_progress=False)
        assert result.response_info is None

    @respx.mock
    def test_upsert_namespace_forwarded_per_batch(self) -> None:
        route = respx.post(UPSERT_URL).mock(
            side_effect=lambda req: httpx.Response(
                200,
                json=_make_upsert_response(
                    upserted_count=len(orjson.loads(req.content)["vectors"])
                ),
            )
        )
        idx = _make_index()
        idx.upsert(vectors=_make_vectors(10), batch_size=5, namespace="my-ns", show_progress=False)
        for call in route.calls:
            body = orjson.loads(call.request.content)
            assert body.get("namespace") == "my-ns"

    @respx.mock
    def test_upsert_timeout_forwarded_per_batch(self) -> None:
        route = respx.post(UPSERT_URL).mock(
            side_effect=lambda req: httpx.Response(
                200,
                json=_make_upsert_response(
                    upserted_count=len(orjson.loads(req.content)["vectors"])
                ),
            )
        )
        idx = _make_index()
        idx.upsert(vectors=_make_vectors(10), batch_size=5, timeout=5.0, show_progress=False)
        assert len(route.calls) == 2

    @respx.mock
    def test_upsert_empty_vectors_with_batch_size(self) -> None:
        route = respx.post(UPSERT_URL).mock(
            return_value=httpx.Response(200, json=_make_upsert_response(upserted_count=0))
        )
        idx = _make_index()
        result = idx.upsert(vectors=[], batch_size=100, show_progress=False)
        assert len(route.calls) == 0
        assert result.upserted_count == 0


class TestUpsertInvalidBatchSize:
    """Invalid batch_size values raise PineconeValueError."""

    def test_upsert_invalid_batch_size_zero(self) -> None:
        idx = _make_index()
        with pytest.raises(PineconeValueError):
            idx.upsert(vectors=_make_vectors(5), batch_size=0)

    def test_upsert_invalid_batch_size_negative(self) -> None:
        idx = _make_index()
        with pytest.raises(PineconeValueError):
            idx.upsert(vectors=_make_vectors(5), batch_size=-1)

    def test_upsert_invalid_batch_size_float(self) -> None:
        idx = _make_index()
        with pytest.raises(PineconeValueError):
            idx.upsert(vectors=_make_vectors(5), batch_size=1.5)  # type: ignore[arg-type]


class TestUpsertMaxConcurrency:
    """max_concurrency parameter validation and forwarding."""

    @respx.mock
    def test_upsert_max_concurrency_default_is_8(self) -> None:
        respx.post(UPSERT_URL).mock(
            side_effect=lambda req: httpx.Response(
                200,
                json=_make_upsert_response(
                    upserted_count=len(orjson.loads(req.content)["vectors"])
                ),
            )
        )
        idx = _make_index()
        with patch("pinecone.index.bulk_execute_sync", wraps=bulk_execute_sync) as spy:
            idx.upsert(vectors=_make_vectors(10), batch_size=5, show_progress=False)
        assert spy.call_args.kwargs["max_concurrency"] == 8

    @respx.mock
    def test_upsert_max_concurrency_explicit(self) -> None:
        respx.post(UPSERT_URL).mock(
            side_effect=lambda req: httpx.Response(
                200,
                json=_make_upsert_response(
                    upserted_count=len(orjson.loads(req.content)["vectors"])
                ),
            )
        )
        idx = _make_index()
        with patch("pinecone.index.bulk_execute_sync", wraps=bulk_execute_sync) as spy:
            idx.upsert(
                vectors=_make_vectors(10), batch_size=5, max_concurrency=8, show_progress=False
            )
        assert spy.call_args.kwargs["max_concurrency"] == 8

    def test_upsert_invalid_max_concurrency_zero(self) -> None:
        idx = _make_index()
        with pytest.raises(PineconeValueError):
            idx.upsert(vectors=_make_vectors(5), batch_size=5, max_concurrency=0)

    def test_upsert_invalid_max_concurrency_65(self) -> None:
        idx = _make_index()
        with pytest.raises(PineconeValueError):
            idx.upsert(vectors=_make_vectors(5), batch_size=5, max_concurrency=65)

    def test_upsert_invalid_max_concurrency_negative(self) -> None:
        idx = _make_index()
        with pytest.raises(PineconeValueError):
            idx.upsert(vectors=_make_vectors(5), batch_size=5, max_concurrency=-1)


class TestUpsertPartialFailure:
    """Partial-success contract: per-batch errors captured, method does not raise."""

    @respx.mock
    def test_upsert_partial_failure_returns_rich_response(self) -> None:
        # 300 vectors, batch_size=100 → 3 batches of 100 each.
        # Fail the batch whose first vector id is "v100" (batch 1).
        def _side_effect(req: httpx.Request) -> httpx.Response:
            body = orjson.loads(req.content)
            vectors = body["vectors"]
            if vectors[0]["id"] == "v100":
                return httpx.Response(500, json={"error": {"message": "server error"}})
            return httpx.Response(200, json={"upsertedCount": len(vectors)})

        respx.post(UPSERT_URL).mock(side_effect=_side_effect)
        idx = _make_index()
        result = idx.upsert(vectors=_make_vectors(300), batch_size=100, show_progress=False)

        assert isinstance(result, UpsertResponse)
        assert result.has_errors is True
        assert result.failed_batch_count == 1
        assert result.failed_item_count == 100
        assert result.upserted_count == 200
        assert result.total_item_count == 300
        assert result.total_batch_count == 3
        assert result.successful_batch_count == 2
        assert len(result.errors) == 1

    @respx.mock
    def test_upsert_failed_items_for_retry(self) -> None:
        # The failed batch (v100-v199) should be recoverable via failed_items.
        def _side_effect(req: httpx.Request) -> httpx.Response:
            body = orjson.loads(req.content)
            vectors = body["vectors"]
            if vectors[0]["id"] == "v100":
                return httpx.Response(500, json={"error": {"message": "server error"}})
            return httpx.Response(200, json={"upsertedCount": len(vectors)})

        respx.post(UPSERT_URL).mock(side_effect=_side_effect)
        idx = _make_index()
        result = idx.upsert(vectors=_make_vectors(300), batch_size=100, show_progress=False)

        failed = result.failed_items
        assert len(failed) == 100
        assert all(isinstance(item, dict) for item in failed)
        ids = [item["id"] for item in failed]
        assert ids == [f"v{i}" for i in range(100, 200)]


class TestUpsertConcurrencyBehavior:
    """The per-handle executor cache is gone (it replaced a live pool on any
    max_concurrency change — the pre-A5 bug on its last transport). What
    remains testable is behavior: distinct sizes never interfere."""

    @respx.mock
    def test_changing_max_concurrency_across_calls_is_safe(self) -> None:
        respx.post(UPSERT_URL).mock(
            side_effect=lambda req: httpx.Response(
                200,
                json=_make_upsert_response(
                    upserted_count=len(orjson.loads(req.content)["vectors"])
                ),
            )
        )
        idx = _make_index()
        for concurrency in (1, 4, 8, 3, 16, 4):
            result = idx.upsert(
                vectors=_make_vectors(10),
                batch_size=5,
                max_concurrency=concurrency,
                show_progress=False,
            )
            assert result.upserted_count == 10
            assert not result.has_errors


class TestUpsertResponseInfoAggregated:
    """LSN values from batched responses are aggregated into response_info."""

    @respx.mock
    def test_upsert_response_info_aggregated_across_batches(self) -> None:
        lsn_values = [10, 20, 15]

        def _side_effect(req: httpx.Request) -> httpx.Response:
            body = orjson.loads(req.content)
            vectors = body["vectors"]
            # Assign LSN by batch index based on first vector id.
            first_id = vectors[0]["id"]
            if first_id == "v0":
                lsn = lsn_values[0]
            elif first_id == "v100":
                lsn = lsn_values[1]
            else:
                lsn = lsn_values[2]
            return httpx.Response(
                200,
                json={"upsertedCount": len(vectors)},
                headers={
                    "x-pinecone-lsn-committed": str(lsn),
                    "x-pinecone-lsn-reconciled": str(lsn),
                },
            )

        respx.post(UPSERT_URL).mock(side_effect=_side_effect)
        idx = _make_index()
        result = idx.upsert(vectors=_make_vectors(250), batch_size=100, show_progress=False)

        assert result.response_info is not None
        assert result.response_info.lsn_committed == max(lsn_values)
        assert result.response_info.lsn_reconciled == max(lsn_values)


class TestUpsertShowProgress:
    """show_progress behavior with and without tqdm."""

    @respx.mock
    def test_upsert_show_progress_false_does_not_import_tqdm(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "tqdm", None)  # type: ignore[arg-type]
        monkeypatch.setitem(sys.modules, "tqdm.auto", None)  # type: ignore[arg-type]
        respx.post(UPSERT_URL).mock(
            side_effect=lambda req: httpx.Response(
                200,
                json=_make_upsert_response(
                    upserted_count=len(orjson.loads(req.content)["vectors"])
                ),
            )
        )
        idx = _make_index()
        result = idx.upsert(vectors=_make_vectors(10), batch_size=5, show_progress=False)
        assert result.upserted_count == 10

    @respx.mock
    def test_upsert_show_progress_true_works_without_tqdm(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "tqdm", None)  # type: ignore[arg-type]
        monkeypatch.setitem(sys.modules, "tqdm.auto", None)  # type: ignore[arg-type]
        respx.post(UPSERT_URL).mock(
            side_effect=lambda req: httpx.Response(
                200,
                json=_make_upsert_response(
                    upserted_count=len(orjson.loads(req.content)["vectors"])
                ),
            )
        )
        idx = _make_index()
        result = idx.upsert(vectors=_make_vectors(10), batch_size=5, show_progress=True)
        assert result.upserted_count == 10
