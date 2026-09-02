"""Admission-gate wiring for the documents batch path, sync and async.

The documents surface reaches the same process-global gate registry as the
vector bulk path, so ``batch_upsert`` gets backpressure rather than handing a
struggling host every batch at once. These tests exist because a green
``test_documents_operations.py`` proves aggregation, not admission: the
pre-gate implementation drove a cached ``ThreadPoolExecutor`` directly and
passed every one of those tests while no gate was involved at all.

``BatchResult.final_limit`` is the discriminator. Only the gate-aware engines
populate it; the executor-driven ``batch_execute`` leaves it ``None``. So
``final_limit is not None`` is a statement about which code path ran, not
about a value.
"""

from __future__ import annotations

import asyncio
import inspect
import threading
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx
import orjson
import pytest
import respx

from pinecone._internal.bulk import get_registry
from pinecone._internal.constants import DEFAULT_MAX_CONCURRENCY
from pinecone.async_client.async_index import AsyncIndex
from pinecone.async_client.documents import AsyncDocuments
from pinecone.client.documents import Documents
from pinecone.index import Index

HOST = "documents-gate-idx.svc.us-east-1-aws.pinecone.io"
NS = "articles-en"
UPSERT_URL = f"https://{HOST}/namespaces/{NS}/documents/upsert"


@pytest.fixture(autouse=True)
def _fresh_registry() -> Iterator[None]:
    get_registry()._reset()
    yield
    get_registry()._reset()


@pytest.fixture
def index() -> Iterator[Index]:
    client = Index(host=HOST, api_key="test-key")
    yield client
    client.close()


@pytest.fixture
async def async_index() -> AsyncIterator[AsyncIndex]:
    client = AsyncIndex(host=HOST, api_key="test-key")
    yield client
    await client.close()


def _docs(n: int) -> list[dict[str, Any]]:
    return [{"_id": f"doc-{i}"} for i in range(n)]


def _ok(request: httpx.Request) -> httpx.Response:
    count = len(orjson.loads(request.content)["documents"])
    return httpx.Response(202, json={"upserted_count": count})


def test_max_concurrency_is_gate_owned_on_both_surfaces() -> None:
    """``None`` is the signature default so the gate owns the number; the value
    it resolves to is ``DEFAULT_MAX_CONCURRENCY``, not the retired hard 4."""
    for cls in (Documents, AsyncDocuments):
        param = inspect.signature(cls.batch_upsert).parameters["max_concurrency"]
        assert param.default is None, f"{cls.__name__} still hard-codes a concurrency default"
    assert DEFAULT_MAX_CONCURRENCY == 8


class TestSyncDocumentsGateWiring:
    @respx.mock
    def test_batch_upsert_runs_through_the_gate(self, index: Index) -> None:
        respx.post(UPSERT_URL).mock(side_effect=_ok)
        result = index.documents.batch_upsert(
            namespace=NS, documents=_docs(6), batch_size=2, show_progress=False
        )

        assert result.successful_item_count == 6
        assert result.final_limit is not None, "batch_upsert bypassed the admission gate"
        assert result.final_limit == get_registry().get(HOST).limit
        assert get_registry().get(HOST).quiescent()

    @respx.mock
    def test_429_lowers_the_gate_for_the_documents_host(self, index: Index) -> None:
        """The throttle is heard in the HTTP retry loop, one layer below the
        documents surface; this asserts it reaches the gate keyed on the same
        host the documents call admits through."""
        calls = {"n": 0}

        def flaky(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(429, json={"error": {"message": "slow down"}})
            return _ok(request)

        respx.post(UPSERT_URL).mock(side_effect=flaky)
        gate = get_registry().get(HOST)
        before = gate.limit

        result = index.documents.batch_upsert(
            namespace=NS, documents=_docs(10), batch_size=5, show_progress=False
        )

        assert result.successful_item_count == 10
        assert gate.limit < before, "the 429 never reached the documents gate"
        assert result.throttle_event_count >= 1

    def test_gate_limit_beats_max_concurrency(
        self, index: Index, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The point of the port: the caller's ``max_concurrency`` is a ceiling,
        not a promise. With the host's gate driven to its floor, eight
        requested slots admit one at a time — the pre-gate implementation would
        have opened all eight against a host that just said stop.

        The transport is stubbed rather than mocked through respx because
        respx's sync transport serializes requests, which would make
        ``inflight == 1`` hold whether or not a gate exists.
        """
        started = threading.Semaphore(0)
        release = threading.Event()
        inflight = 0
        peak = 0
        lock = threading.Lock()

        def _post(path: str, **kwargs: Any) -> httpx.Response:
            nonlocal inflight, peak
            with lock:
                inflight += 1
                peak = max(peak, inflight)
            started.release()
            release.wait(5.0)
            with lock:
                inflight -= 1
            return httpx.Response(202, json={"upserted_count": len(kwargs["json"]["documents"])})

        monkeypatch.setattr(index._http, "post", _post)
        gate = get_registry().get(HOST)
        while gate.limit > 1:
            gate.report_throttled()
        assert gate.limit == 1

        collected: list[Any] = []

        def run() -> None:
            collected.append(
                index.documents.batch_upsert(
                    namespace=NS,
                    documents=_docs(8),
                    batch_size=1,
                    max_concurrency=8,
                    show_progress=False,
                )
            )

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        assert started.acquire(timeout=5.0), "no batch was ever admitted"
        assert not started.acquire(timeout=0.5), (
            "a second batch started while the first was still in flight and the "
            "gate limit was 1; max_concurrency was allowed to win"
        )
        with lock:
            assert inflight == 1

        release.set()
        thread.join(timeout=10.0)
        assert not thread.is_alive()
        assert collected[0].successful_item_count == 8
        assert peak < 8, f"peak in-flight {peak} reached the requested ceiling despite the gate"


class TestAsyncDocumentsGateWiring:
    @respx.mock
    async def test_batch_upsert_runs_through_the_gate(self, async_index: AsyncIndex) -> None:
        respx.post(UPSERT_URL).mock(side_effect=_ok)
        result = await async_index.documents.batch_upsert(
            namespace=NS, documents=_docs(6), batch_size=2, show_progress=False
        )

        assert result.successful_item_count == 6
        assert result.final_limit is not None, "batch_upsert bypassed the admission gate"
        assert result.final_limit == get_registry().get(HOST).limit
        assert get_registry().get(HOST).quiescent()

    @respx.mock
    async def test_429_lowers_the_gate_for_the_documents_host(
        self, async_index: AsyncIndex
    ) -> None:
        calls = {"n": 0}

        def flaky(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(429, json={"error": {"message": "slow down"}})
            return _ok(request)

        respx.post(UPSERT_URL).mock(side_effect=flaky)
        gate = get_registry().get(HOST)
        before = gate.limit

        result = await async_index.documents.batch_upsert(
            namespace=NS, documents=_docs(10), batch_size=5, show_progress=False
        )

        assert result.successful_item_count == 10
        assert gate.limit < before, "the 429 never reached the documents gate"
        assert result.throttle_event_count >= 1

    async def test_gate_limit_beats_max_concurrency(
        self, async_index: AsyncIndex, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Async twin: the ``asyncio.Semaphore`` bound is the caller's ceiling,
        the gate's admission is the real limit. The scheduler yields are what
        make ``inflight == 1`` an assertion about admission rather than about
        the event loop not having got round to the other seven yet."""
        first_started = asyncio.Event()
        release = asyncio.Event()
        inflight = 0
        peak = 0

        async def _post(path: str, **kwargs: Any) -> httpx.Response:
            nonlocal inflight, peak
            inflight += 1
            peak = max(peak, inflight)
            first_started.set()
            await release.wait()
            inflight -= 1
            return httpx.Response(202, json={"upserted_count": len(kwargs["json"]["documents"])})

        monkeypatch.setattr(async_index._http, "post", _post)
        gate = get_registry().get(HOST)
        while gate.limit > 1:
            gate.report_throttled()
        assert gate.limit == 1

        task = asyncio.create_task(
            async_index.documents.batch_upsert(
                namespace=NS,
                documents=_docs(8),
                batch_size=1,
                max_concurrency=8,
                show_progress=False,
            )
        )
        await asyncio.wait_for(first_started.wait(), timeout=5.0)
        for _ in range(20):
            await asyncio.sleep(0)
        assert inflight == 1, (
            f"gate limit 1 admitted {inflight} concurrent batches; "
            "max_concurrency was allowed to win"
        )

        release.set()
        result = await asyncio.wait_for(task, timeout=10.0)
        assert result.successful_item_count == 8
        assert peak < 8, f"peak in-flight {peak} reached the requested ceiling despite the gate"
