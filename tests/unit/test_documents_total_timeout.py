"""``total_timeout`` on the documents batch path, sync and async (D8).

The vector ingest paths have taken ``total_timeout`` since v10; the documents
surface did not, which made "backpressure and ``total_timeout`` semantics on
the ingest paths" only half true. These tests hold the documents surface to
the *same* contract the vector paths document, not merely to accepting the
kwarg:

- on expiry no further batches are submitted, and the un-submitted ones come
  back as ``unsent`` errors in ``failed_items``;
- batches already in flight are awaited and never cancelled, because dropping
  one client-side would not stop the host from applying it;
- ``timed_out`` is ``True`` only when something was actually left unsent;
- time spent waiting for the host's **admission gate** counts against the
  budget, so a throttled host can burn it without a request being sent. That
  last one is the interaction the pre-gate implementation could not have had,
  and it is the reason this is not just a plumbing test.

The transport is stubbed rather than mocked through respx: respx's sync
transport serializes requests, which would make the in-flight assertions hold
whether or not a deadline existed.

Mutation-checked: dropping ``total_timeout=total_timeout`` from both
surfaces' ``bulk_execute_*`` calls — accepting the kwarg and ignoring it, the
exact half-done shape this module exists to catch — fails 4 of the 9 tests
here (both expiry tests and both backpressure tests). The remaining five
assert the absence of a timeout or the signature, so they pass under that
mutation by design.
"""

from __future__ import annotations

import asyncio
import inspect
import threading
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx
import pytest

from pinecone._internal.bulk import get_registry
from pinecone.async_client.async_index import AsyncIndex
from pinecone.async_client.documents import AsyncDocuments
from pinecone.client.documents import Documents
from pinecone.index import Index

HOST = "documents-deadline-idx.svc.us-east-1-aws.pinecone.io"
NS = "articles-en"


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


def _accepted(kwargs: dict[str, Any]) -> httpx.Response:
    return httpx.Response(202, json={"upserted_count": len(kwargs["json"]["documents"])})


def _floor_the_gate() -> Any:
    gate = get_registry().get(HOST)
    while gate.limit > 1:
        gate.report_throttled()
    assert gate.limit == 1
    return gate


def test_total_timeout_defaults_to_none_on_both_surfaces() -> None:
    """The parameter exists on both surfaces and is opt-in, so an existing
    caller's behaviour is unchanged."""
    for cls in (Documents, AsyncDocuments):
        params = inspect.signature(cls.batch_upsert).parameters
        assert "total_timeout" in params, f"{cls.__name__}.batch_upsert has no total_timeout"
        assert params["total_timeout"].default is None
        assert params["total_timeout"].kind is inspect.Parameter.KEYWORD_ONLY


class TestSyncDocumentsTotalTimeout:
    def test_expiry_abandons_unsent_and_awaits_inflight(
        self, index: Index, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        completed: list[str] = []
        lock = threading.Lock()

        def _post(path: str, **kwargs: Any) -> httpx.Response:
            threading.Event().wait(0.4)
            with lock:
                completed.append(kwargs["json"]["documents"][0]["_id"])
            return _accepted(kwargs)

        monkeypatch.setattr(index._http, "post", _post)

        result = index.documents.batch_upsert(
            namespace=NS,
            documents=_docs(8),
            batch_size=1,
            max_concurrency=4,
            show_progress=False,
            total_timeout=0.2,
        )

        assert result.timed_out, "total_timeout expired but the result does not say so"
        unsent = [e for e in result.errors if e.disposition == "unsent"]
        assert unsent, "no batch was reported unsent; the deadline was not enforced"
        assert all(e.retryable for e in unsent)
        assert completed, "in-flight batches must be awaited, not cancelled"
        assert len(completed) == result.successful_item_count
        assert result.successful_item_count + result.failed_item_count == 8
        assert len(result.failed_items) == result.failed_item_count
        assert get_registry().get(HOST).quiescent()

    def test_gate_backpressure_consumes_the_budget(
        self, index: Index, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The interaction that only exists once the gate is in the path: with
        the host's gate at its floor, batches wait for *admission* rather than
        for the wire, and that wait is spent from the same budget. So the
        deadline expires with batches never sent at all."""
        release = threading.Event()
        sent = []
        sent_lock = threading.Lock()

        def _post(path: str, **kwargs: Any) -> httpx.Response:
            with sent_lock:
                sent.append(path)
            release.wait(5.0)
            return _accepted(kwargs)

        monkeypatch.setattr(index._http, "post", _post)
        _floor_the_gate()

        collected: list[Any] = []

        def run() -> None:
            collected.append(
                index.documents.batch_upsert(
                    namespace=NS,
                    documents=_docs(4),
                    batch_size=1,
                    max_concurrency=4,
                    show_progress=False,
                    total_timeout=0.25,
                )
            )

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        thread.join(timeout=10.0)
        release.set()
        thread.join(timeout=10.0)
        assert not thread.is_alive()

        result = collected[0]
        assert result.timed_out
        assert result.failed_item_count >= 3, (
            "with the gate at limit 1 only one batch can be in flight; the rest "
            "should have run out the budget waiting for admission"
        )
        assert all(e.disposition == "unsent" for e in result.errors)
        assert len(sent) == 1, "the gate admitted more than its floor allowed"

    def test_none_means_no_deadline(self, index: Index, monkeypatch: pytest.MonkeyPatch) -> None:
        def _post(path: str, **kwargs: Any) -> httpx.Response:
            threading.Event().wait(0.05)
            return _accepted(kwargs)

        monkeypatch.setattr(index._http, "post", _post)

        result = index.documents.batch_upsert(
            namespace=NS,
            documents=_docs(6),
            batch_size=1,
            max_concurrency=2,
            show_progress=False,
        )

        assert not result.timed_out
        assert result.successful_item_count == 6
        assert result.failed_item_count == 0

    def test_generous_budget_does_not_time_out(
        self, index: Index, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``timed_out`` is about work left unsent, not about the clock: a
        deadline that never bites must leave the flag clear."""

        def _post(path: str, **kwargs: Any) -> httpx.Response:
            return _accepted(kwargs)

        monkeypatch.setattr(index._http, "post", _post)

        result = index.documents.batch_upsert(
            namespace=NS,
            documents=_docs(6),
            batch_size=2,
            show_progress=False,
            total_timeout=30.0,
        )

        assert not result.timed_out
        assert result.successful_item_count == 6
        assert not result.errors


class TestAsyncDocumentsTotalTimeout:
    async def test_expiry_abandons_unsent_and_awaits_inflight(
        self, async_index: AsyncIndex, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        completed: list[str] = []

        async def _post(path: str, **kwargs: Any) -> httpx.Response:
            await asyncio.sleep(0.4)
            completed.append(kwargs["json"]["documents"][0]["_id"])
            return _accepted(kwargs)

        monkeypatch.setattr(async_index._http, "post", _post)

        result = await async_index.documents.batch_upsert(
            namespace=NS,
            documents=_docs(8),
            batch_size=1,
            max_concurrency=4,
            show_progress=False,
            total_timeout=0.2,
        )

        assert result.timed_out, "total_timeout expired but the result does not say so"
        unsent = [e for e in result.errors if e.disposition == "unsent"]
        assert unsent, "no batch was reported unsent; the deadline was not enforced"
        assert all(e.retryable for e in unsent)
        assert completed, "in-flight batches must be awaited, not cancelled"
        assert len(completed) == result.successful_item_count
        assert result.successful_item_count + result.failed_item_count == 8
        assert len(result.failed_items) == result.failed_item_count
        assert get_registry().get(HOST).quiescent()

    async def test_gate_backpressure_consumes_the_budget(
        self, async_index: AsyncIndex, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        release = asyncio.Event()
        sent = []

        async def _post(path: str, **kwargs: Any) -> httpx.Response:
            sent.append(path)
            await release.wait()
            return _accepted(kwargs)

        monkeypatch.setattr(async_index._http, "post", _post)
        _floor_the_gate()

        task = asyncio.create_task(
            async_index.documents.batch_upsert(
                namespace=NS,
                documents=_docs(4),
                batch_size=1,
                max_concurrency=4,
                show_progress=False,
                total_timeout=0.25,
            )
        )
        await asyncio.sleep(0.6)
        release.set()
        result = await asyncio.wait_for(task, timeout=10.0)

        assert result.timed_out
        assert result.failed_item_count >= 3, (
            "with the gate at limit 1 only one batch can be in flight; the rest "
            "should have run out the budget waiting for admission"
        )
        assert all(e.disposition == "unsent" for e in result.errors)
        assert len(sent) == 1, "the gate admitted more than its floor allowed"

    async def test_none_means_no_deadline(
        self, async_index: AsyncIndex, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def _post(path: str, **kwargs: Any) -> httpx.Response:
            await asyncio.sleep(0.05)
            return _accepted(kwargs)

        monkeypatch.setattr(async_index._http, "post", _post)

        result = await async_index.documents.batch_upsert(
            namespace=NS,
            documents=_docs(6),
            batch_size=1,
            max_concurrency=2,
            show_progress=False,
        )

        assert not result.timed_out
        assert result.successful_item_count == 6
        assert result.failed_item_count == 0

    async def test_generous_budget_does_not_time_out(
        self, async_index: AsyncIndex, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def _post(path: str, **kwargs: Any) -> httpx.Response:
            return _accepted(kwargs)

        monkeypatch.setattr(async_index._http, "post", _post)

        result = await async_index.documents.batch_upsert(
            namespace=NS,
            documents=_docs(6),
            batch_size=2,
            show_progress=False,
            total_timeout=30.0,
        )

        assert not result.timed_out
        assert result.successful_item_count == 6
        assert not result.errors
