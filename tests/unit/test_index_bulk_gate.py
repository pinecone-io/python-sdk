"""REST sync admission-gate behavior (#71): backpressure and total_timeout on
the REST bulk path, wired through the same process-global registry as gRPC —
throttles heard over one transport lower the limit every transport admits
through."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import httpx
import orjson
import pytest
import respx

from pinecone import Index
from pinecone._internal.bulk import get_registry

HOST = "rest-gate-idx.svc.pinecone.io"
UPSERT_URL = f"https://{HOST}/vectors/upsert"


@pytest.fixture(autouse=True)
def _fresh_registry() -> Iterator[None]:
    get_registry()._reset()
    yield
    get_registry()._reset()


def _make_index() -> Index:
    return Index(host=HOST, api_key="test-key")


def _vectors(n: int) -> list[dict[str, Any]]:
    return [{"id": f"v{i}", "values": [0.1, 0.2]} for i in range(n)]


def _ok(req: httpx.Request) -> httpx.Response:
    count = len(orjson.loads(req.content)["vectors"])
    return httpx.Response(200, json={"upsertedCount": count})


@respx.mock
def test_429_lowers_the_gate_and_bulk_admission_follows() -> None:
    """End-to-end across layers: a throttled response inside the HTTP retry
    loop must move the limit the bulk path admits through — the assertion
    whose absence let #60 ship."""
    calls = {"n": 0}

    def flaky(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, json={"error": {"message": "slow down"}})
        return _ok(req)

    respx.post(UPSERT_URL).mock(side_effect=flaky)
    gate = get_registry().get(HOST)
    before = gate.limit

    idx = _make_index()
    result = idx.upsert(vectors=_vectors(10), batch_size=5, show_progress=False)

    assert result.upserted_count == 10
    assert gate.limit < before, "the 429 never reached the gate"
    assert gate.quiescent()


@respx.mock
def test_retry_after_header_becomes_a_pushback_hold() -> None:
    """The unit conftest no-ops the HTTP retry sleeps, so upsert returns
    in milliseconds and a 30s hold is still comfortably live at assertion
    time — the large value is the margin that keeps this test valid even
    if scheduling adds real delay between throttle and assertion."""
    import time

    def flaky(req: httpx.Request) -> httpx.Response:
        if not hasattr(flaky, "hit"):
            flaky.hit = True  # type: ignore[attr-defined]
            return httpx.Response(
                429, headers={"retry-after": "30"}, json={"error": {"message": "hold"}}
            )
        return _ok(req)

    respx.post(UPSERT_URL).mock(side_effect=flaky)
    gate = get_registry().get(HOST)
    idx = _make_index()
    idx.upsert(vectors=_vectors(2), batch_size=2, show_progress=False)
    remaining = gate._core.hold_remaining(time.monotonic())
    assert remaining is not None and remaining > 20, (
        f"retry-after never became a live gate hold (remaining={remaining})"
    )


@respx.mock
def test_total_timeout_lands_on_rest_sync() -> None:
    """The matrix row this ticket closes: a deadline on the REST bulk path."""
    import threading

    release = threading.Event()

    def slow(req: httpx.Request) -> httpx.Response:
        release.wait(2.0)
        return _ok(req)

    respx.post(UPSERT_URL).mock(side_effect=slow)
    gate = get_registry().get(HOST)
    while gate.limit > 1:
        gate.report_throttled()

    idx = _make_index()
    result = idx.upsert(
        vectors=_vectors(8),
        batch_size=2,
        max_concurrency=4,
        show_progress=False,
        total_timeout=0.3,
    )
    release.set()
    assert result.failed_item_count >= 4, "unsent batches must be reported"
    assert result.failed_items, "failed_items is the retry contract"
    assert gate.quiescent()


@respx.mock
def test_gate_is_shared_between_two_index_handles() -> None:
    """Process-global scope: two handles to one host share one limit — the
    #56 shape (per-object state, per-host intent) cannot recur."""
    respx.post(UPSERT_URL).mock(side_effect=_ok)
    a = _make_index()
    b = Index(host=f"https://{HOST}", api_key="other-key")

    gate = get_registry().get(HOST)
    gate.report_throttled()
    limited = gate.limit

    a.upsert(vectors=_vectors(4), batch_size=2, show_progress=False)
    b.upsert(vectors=_vectors(4), batch_size=2, show_progress=False)

    assert get_registry().get(HOST) is gate
    assert gate.limit >= limited
    assert gate.quiescent()
