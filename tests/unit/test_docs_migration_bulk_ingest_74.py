"""Executes ``docs/migration/v10-migration.md``'s bulk-ingest section (#74, #71).

Same discipline as ``test_docs_migration_db_data_415.py``: every python block in
the section is read out of the published file and executed, never retyped here,
so a transcription cannot drift from what a reader copies. Every number the
prose names -- the ``8`` default, the ``1``-``64`` range, the gate's ``64``
ceiling, its floor of one, four consecutive failed settles, the 30-second
cool-down -- is pinned against the constant it describes, and the
``total_timeout`` message the section quotes is compared with the one the SDK
really builds.

What this file does NOT assert, and why: the section's ``9.x`` column. That
column describes a released version whose code is not importable here (the
``9.x`` default of ``4``, the absent ``max_concurrency`` on
``upsert_from_dataframe``, the per-client limiter). Those were read off the
``v9.1.0`` tag; the citation is in the PR body, not an assertion here.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import orjson
import pytest
import respx

from pinecone import AsyncIndex, Index
from pinecone._internal.batch import _abandoned_error
from pinecone._internal.bulk import get_registry
from pinecone._internal.bulk.classify import DISPOSITION_ABANDONED
from pinecone._internal.bulk.core import (
    GATE_CEILING,
    STALL_CONSECUTIVE_FAILURES,
    STALL_COOLDOWN_SECONDS,
)
from pinecone._internal.bulk.engine import _stalled_error, bulk_execute_sync
from pinecone._internal.constants import DEFAULT_MAX_CONCURRENCY
from pinecone.client.documents import Documents
from pinecone.errors.exceptions import PineconeTimeoutError, PineconeValueError
from pinecone.models.batch import BatchResult
from pinecone.models.vectors.responses import UpsertResponse

GUIDE = Path(__file__).resolve().parents[2] / "docs/migration/v10-migration.md"
SECTION_START = "(bulk-ingest)="
SECTION_END = "(backup-models)="
TEXT = GUIDE.read_text().split(SECTION_START, 1)[1].split(SECTION_END, 1)[0]

INDEX_HOST = "bulk-ingest-doc-abc1234.svc.us-east1-gcp.pinecone.io"
UPSERT_URL = f"https://{INDEX_HOST}/vectors/upsert"


@pytest.fixture(autouse=True)
def _fresh_registry() -> Iterator[None]:
    """Process-global gates outlive a test; a leftover limit would change what
    the next one admits."""
    get_registry()._reset()
    yield
    get_registry()._reset()


def _blocks() -> list[str]:
    sources = [m.group(1) for m in re.finditer(r"```python\n(.*?)```", TEXT, re.DOTALL)]
    assert sources, f"no python blocks found in {GUIDE}"
    return sources


def _flat(text: str) -> str:
    """Whitespace, smart quotes and dashes flattened, so a reflow cannot hide a phrase."""
    quotes = dict.fromkeys("\u2018\u2019", "'") | dict.fromkeys("\u201c\u201d", '"')
    dashes = dict.fromkeys("\u2013\u2014\u2212", "-")
    return re.sub(r"\s+", " ", text.translate(str.maketrans(quotes | dashes)))


def _index() -> Index:
    return Index(host=INDEX_HOST, api_key="test-key")


def _ok(request: httpx.Request) -> httpx.Response:
    count = len(orjson.loads(request.content)["vectors"])
    return httpx.Response(200, json={"upsertedCount": count})


def _vectors(n: int) -> list[dict[str, Any]]:
    return [{"id": f"v{i}", "values": [0.1, 0.2]} for i in range(n)]


@pytest.mark.parametrize("index", range(len(_blocks())))
@respx.mock
def test_every_python_block_in_the_section_runs(index: int) -> None:
    """The whole premise of the section is that a reader can copy these."""
    route = respx.post(UPSERT_URL).mock(side_effect=_ok)

    exec(_blocks()[index], {"idx": _index()})  # noqa: S102

    assert route.calls, "a block that sends nothing is not showing an ingest"


@pytest.mark.parametrize("lane", ["pinecone.index", "pinecone.async_client.async_index"])
def test_the_default_the_section_names_is_the_default_it_documents(lane: str) -> None:
    import inspect

    assert "defaults to 8" in TEXT
    assert DEFAULT_MAX_CONCURRENCY == 8

    module = __import__(lane, fromlist=["__file__"])
    upsert = module.Index.upsert if lane.endswith(".index") else module.AsyncIndex.upsert
    assert inspect.signature(upsert).parameters["max_concurrency"].default == 8


def test_the_grpc_lane_shares_that_default() -> None:
    import inspect

    grpc = pytest.importorskip("pinecone.grpc")
    assert "`GrpcIndex`" in TEXT
    assert inspect.signature(grpc.GrpcIndex.upsert).parameters["max_concurrency"].default == 8


@pytest.mark.parametrize("bad", [0, 65])
@respx.mock
def test_the_range_the_section_names_is_the_range_enforced(bad: int) -> None:
    assert "`1`-`64`" in TEXT
    route = respx.post(UPSERT_URL).mock(side_effect=_ok)

    with pytest.raises(PineconeValueError):
        _index().upsert(vectors=_vectors(4), batch_size=2, max_concurrency=bad)

    assert not route.calls, "the section says this raises before anything is sent"


@pytest.mark.parametrize("good", [1, 64])
@respx.mock
def test_both_ends_of_that_range_are_accepted(good: int) -> None:
    respx.post(UPSERT_URL).mock(side_effect=_ok)
    result = _index().upsert(
        vectors=_vectors(4), batch_size=2, max_concurrency=good, show_progress=False
    )
    assert result.upserted_count == 4


@respx.mock
def test_an_unbatched_upsert_ignores_both_arguments_as_the_section_says() -> None:
    flat = _flat(TEXT)
    assert "`max_concurrency` and `total_timeout` are both ignored" in flat
    assert "isn't even range-checked" in flat
    route = respx.post(UPSERT_URL).mock(side_effect=_ok)

    result = _index().upsert(vectors=_vectors(4), max_concurrency=999, total_timeout=0.0)

    assert result.upserted_count == 4
    assert len(route.calls) == 1, "no batch_size means one request"


@pytest.mark.parametrize(
    "module",
    ["pinecone.index", "pinecone.async_client.async_index", "pinecone.grpc"],
)
def test_none_resolves_to_the_documented_constant_on_every_dataframe_lane(module: str) -> None:
    import inspect

    assert "defaults to `None` and resolves to `8`" in _flat(TEXT)
    lane = pytest.importorskip(module)
    source = Path(lane.__file__ or "").read_text()
    assert "DEFAULT_MAX_CONCURRENCY if max_concurrency is None" in source

    cls = next(
        getattr(lane, name)
        for name in ("Index", "AsyncIndex", "GrpcIndex")
        if hasattr(lane, name) and hasattr(getattr(lane, name), "upsert_from_dataframe")
    )
    parameter = inspect.signature(cls.upsert_from_dataframe).parameters["max_concurrency"]
    assert parameter.default is None
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


@pytest.mark.parametrize("module", ["pinecone.client.documents", "pinecone.async_client.documents"])
def test_batch_upsert_resolves_none_the_same_way(module: str) -> None:
    import inspect

    assert "`documents.batch_upsert` resolves `None` the same way" in _flat(TEXT)
    lane = __import__(module, fromlist=["Documents"])
    documents = getattr(lane, "Documents", None) or lane.AsyncDocuments
    assert inspect.signature(documents.batch_upsert).parameters["max_concurrency"].default is None
    assert (
        "DEFAULT_MAX_CONCURRENCY if max_concurrency is None"
        in Path(lane.__file__ or "").read_text()
    )


def test_total_timeout_is_on_every_entry_point_the_section_lists() -> None:
    """The section's uniformity claim, method by method."""
    import inspect

    flat = _flat(TEXT)
    assert "`upsert` and `upsert_from_dataframe` on all three lanes" in flat
    assert "`documents.batch_upsert` on both the sync and asyncio document surfaces" in flat
    assert "It defaults to `None` on every one of those methods" in flat

    grpc = pytest.importorskip("pinecone.grpc")
    from pinecone.async_client.documents import AsyncDocuments

    entry_points = [
        Index.upsert,
        Index.upsert_from_dataframe,
        AsyncIndex.upsert,
        AsyncIndex.upsert_from_dataframe,
        grpc.GrpcIndex.upsert,
        grpc.GrpcIndex.upsert_from_dataframe,
        Documents.batch_upsert,
        AsyncDocuments.batch_upsert,
    ]
    for entry_point in entry_points:
        parameter = inspect.signature(entry_point).parameters["total_timeout"]
        assert parameter.default is None, entry_point.__qualname__


def test_the_gate_numbers_the_section_names_are_the_gates_own() -> None:
    flat = _flat(TEXT)
    assert "own ceiling of `64`" in flat
    assert "at the floor of one and four consecutive batches" in flat
    assert "30-second cool-down" in flat

    assert GATE_CEILING == 64
    assert STALL_CONSECUTIVE_FAILURES == 4
    assert STALL_COOLDOWN_SECONDS == 30.0
    assert get_registry().get(INDEX_HOST).limit == GATE_CEILING


def test_only_a_throttle_halves_the_limit_as_the_section_says() -> None:
    flat = _flat(TEXT)
    assert "once a throttled response halves the gate" in flat
    assert "at most once per in-flight generation" in flat

    gate = get_registry().get(INDEX_HOST)
    before = gate.limit
    for _ in range(STALL_CONSECUTIVE_FAILURES + 1):
        gate.report_failure()
    assert gate.limit == before, "the section credits the halving to throttling only"

    gate.report_throttled()
    assert gate.limit == before // 2


def test_the_grpc_dataframe_variation_the_section_names_is_real() -> None:
    flat = _flat(TEXT)
    assert "`GrpcIndex.upsert_from_dataframe` is the one variation" in flat
    assert "summarizing how many vectors landed" in flat

    grpc = pytest.importorskip("pinecone.grpc")
    source = Path(grpc.__file__ or "").read_text()
    assert "expired after " in source
    assert "of {batch_result.total_item_count} vectors were " in source
    assert "logger.warning(message)" in source


def test_a_fresh_gate_leaves_your_own_bound_binding() -> None:
    assert "your bound is the one that binds" in _flat(TEXT)
    assert get_registry().get(INDEX_HOST).limit > DEFAULT_MAX_CONCURRENCY


def test_the_gate_is_process_global_and_keyed_by_the_bare_hostname() -> None:
    flat = _flat(TEXT)
    assert "process-global, not per-client" in flat
    assert "bare lowercase hostname" in flat
    assert "scheme and port variants land on the same gate" in flat

    registry = get_registry()
    gate = registry.get(INDEX_HOST)
    for variant in (f"https://{INDEX_HOST}", f"{INDEX_HOST}:443", INDEX_HOST.upper()):
        assert registry.get(variant) is gate


def test_the_unsent_message_the_section_quotes_is_the_one_the_sdk_builds() -> None:
    quoted = re.search(r"`total_timeout\s+of 5\.0s[^`]*`", TEXT)
    assert quoted, "the section stopped quoting the unsent-batch message"
    expected = _flat(quoted.group(0).strip("`"))

    error = _abandoned_error(3, [{"id": f"v{i}"} for i in range(100)], 5.0)
    assert error.error_message == expected
    assert error.disposition == "unsent"
    assert isinstance(error.error, PineconeTimeoutError)


@respx.mock
def test_an_expired_deadline_reports_unsent_batches_and_never_cancels_a_flight() -> None:
    flat = _flat(TEXT)
    assert "no further batch is submitted" in flat
    assert "awaited and never cancelled" in flat
    assert 'carrying `disposition="unsent"`' in flat

    release = threading.Event()

    def slow(request: httpx.Request) -> httpx.Response:
        release.wait(2.0)
        return _ok(request)

    respx.post(UPSERT_URL).mock(side_effect=slow)
    gate = get_registry().get(INDEX_HOST)
    while gate.limit > 1:
        gate.report_throttled()

    result = _index().upsert(
        vectors=_vectors(8),
        batch_size=2,
        max_concurrency=4,
        show_progress=False,
        total_timeout=0.3,
    )
    release.set()

    unsent = [error for error in result.errors if error.disposition == "unsent"]
    assert unsent, "the section promises the unsent batches are reported"
    assert all(isinstance(error.error, PineconeTimeoutError) for error in unsent)
    assert result.failed_items, "failed_items is the retry contract the section names"
    assert gate.quiescent(), "in-flight batches were awaited, not dropped"


def test_a_deadline_that_expires_with_everything_in_flight_is_a_late_success() -> None:
    assert "only when work was actually left unsent" in _flat(TEXT)
    started = threading.Event()

    def slow(_batch: list[dict[str, Any]]) -> dict[str, Any]:
        started.set()
        threading.Event().wait(0.25)
        return {}

    result = bulk_execute_sync(
        items=_vectors(2),
        operation=slow,
        batch_size=1,
        max_concurrency=2,
        show_progress=False,
        host=INDEX_HOST,
        total_timeout=0.15,
    )

    assert started.is_set()
    assert result.successful_item_count == 2
    assert result.timed_out is False, "nothing was left unsent, so nothing to retry"


def test_the_stall_disposition_is_abandoned_and_is_not_a_timeout() -> None:
    flat = _flat(TEXT)
    assert 'carrying `disposition="abandoned"`' in flat
    assert "rather than as a timeout" in flat

    error = _stalled_error(0, [{"id": "v0"}])
    assert error.disposition == DISPOSITION_ABANDONED
    assert not isinstance(error.error, PineconeTimeoutError)


def test_the_result_fields_the_section_names_live_where_it_says_they_do() -> None:
    gate_fields = ("throttle_event_count", "final_limit", "peak_inflight", "stalled")
    for field in (*gate_fields, "timed_out"):
        assert field in TEXT, f"the section stopped naming {field}"
        assert field in BatchResult.__struct_fields__

    assert "`UpsertResponse` on the vector paths carries none" in _flat(TEXT)
    assert "`result.timed_out`" in TEXT
    for field in (*gate_fields, "timed_out"):
        assert field not in UpsertResponse.__struct_fields__
        assert not hasattr(UpsertResponse, field)


def test_the_contents_list_names_the_section_and_every_anchor_resolves() -> None:
    full_text = GUIDE.read_text()
    contents = full_text.split("## Contents", 1)[1].split("(documents-api)=", 1)[0]
    assert "[Bulk ingest: concurrency and deadlines](#bulk-ingest)" in contents

    anchors = ("bulk-ingest", "bulk-max-concurrency", "bulk-backpressure", "bulk-total-timeout")
    for anchor in anchors:
        assert f"({anchor})=" in full_text, f"missing anchor target for #{anchor}"
        assert f"](#{anchor})" in full_text, f"nothing links to #{anchor}"

    for target in set(re.findall(r"\]\(#([\w-]+)\)", TEXT)):
        assert f"({target})=" in full_text, f"dead link: #{target}"
