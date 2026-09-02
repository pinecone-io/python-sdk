"""Integration coverage for retry / bulk-batching behavior NOT exercised by
tests/integration/test_retry_smoke.py.

The smoke suite only asserts that a 100k-vector, max_concurrency=64 ingest
triggers real backend rate-limiting. That precondition is environment
dependent (a given project/backend tier may not throttle that workload), so
this file adds *deterministic* integration coverage that does not depend on
the backend refusing requests:

  * total_timeout on REST / async / gRPC upsert reports unsent work in
    ``failed_items`` (bounded by the wall clock, not by rate-limiting).
  * max_concurrency default is 8, out-of-range ceilings are rejected before
    any network I/O, and a default-concurrency batched upsert is sane.
  * RetryConfig wiring on a live client (max_retries=0 and custom
    retryable_status_codes) leaves a normal upsert working.
  * upsert_from_dataframe partial-failure aggregation: ``on_error="collect"``
    returns failed_items on the response; ``on_error="raise"`` carries the
    partial result on the raised exception's ``.response``.

These tests use modest vector counts (<= ~3000) with a tiny total_timeout so
they expire by the wall clock and never hammer the backend.

The total_timeout arms run ``batch_size=1, max_concurrency=1`` rather than
``batch_size=25, max_concurrency=64``. The deadline has to expire *part way*
through the ingest for ``upserted_count + failed_item_count`` to mean anything,
and 120 wide-fanned batches complete inside 50 ms against a loopback backend —
which reports 3000 upserted, 0 failed, and asserts nothing. One batch at a time
makes the wall clock, not the backend's speed, decide where the ingest stops,
so the split is partial on a local simulator and on the real API alike.

The index comes from :func:`legacy_index_factory`, not from
``pc.indexes.create``: 2026-07 has no way to create an index the vectors API
will serve, and every write here is a vectors-API call. See
:mod:`tests.integration.legacy_index` for the sanctioned pattern.
"""

from __future__ import annotations

import uuid

import pandas as pd
import pytest

from pinecone import AsyncPinecone, Pinecone, PineconeValueError, RetryConfig
from pinecone._internal.constants import DEFAULT_MAX_CONCURRENCY
from tests.integration.conftest import LegacyIndexFactory, poll_until
from tests.integration.legacy_index import assert_serves_vectors_api

pytestmark = pytest.mark.integration

_DIM = 8

# Wide enough that the deadline lands mid-ingest on a fast loopback backend and
# on a slow real one: at 0.5s, minicone gets through a few hundred of the 3000
# single-row batches, the real API through a few dozen.
_PARTIAL_TOTAL_TIMEOUT = 0.5


@pytest.fixture(scope="module")
def gap_index(client: Pinecone, legacy_index_factory: LegacyIndexFactory) -> str:
    """Shared legacy index (dim=8, cosine) for all retry-gap tests."""
    index = legacy_index_factory(dimension=_DIM)
    assert_serves_vectors_api(client, index)
    return index.name


def _ns(tag: str) -> str:
    return f"retry-{tag}-{uuid.uuid4().hex[:8]}"


def _vectors(n: int, prefix: str = "g") -> list[tuple[str, list[float]]]:
    return [(f"{prefix}-{i}", [0.1] * _DIM) for i in range(n)]


# ---------------------------------------------------------------------------
# total_timeout on upsert (REST / async / gRPC)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_rest_upsert_total_timeout_reports_unsent_failed_items(
    client: Pinecone,
    gap_index: str,
) -> None:
    """A tiny total_timeout must leave unsent batches reported as failed_items."""
    index = client.index(name=gap_index)
    response = index.upsert(
        vectors=_vectors(3000, "r-u"),
        namespace=_ns("rest-tt"),
        batch_size=1,
        max_concurrency=1,
        show_progress=False,
        total_timeout=_PARTIAL_TOTAL_TIMEOUT,
    )
    assert response.has_errors, "total_timeout expiry should produce error entries"
    assert response.failed_item_count > 0, "bulk methods must report unsent work"
    assert response.upserted_count > 0, (
        "the deadline must expire part way through, not before the first batch — "
        "otherwise the accounting assertion below holds trivially"
    )
    assert response.failed_item_count == len(response.failed_items)
    assert response.upserted_count + response.failed_item_count == 3000


@pytest.mark.integration
@pytest.mark.anyio
async def test_async_upsert_total_timeout_reports_unsent_failed_items(
    async_client: AsyncPinecone,
    gap_index: str,
) -> None:
    index = await async_client.index(name=gap_index)
    response = await index.upsert(
        vectors=_vectors(3000, "a-u"),
        namespace=_ns("async-tt"),
        batch_size=1,
        max_concurrency=1,
        show_progress=False,
        total_timeout=_PARTIAL_TOTAL_TIMEOUT,
    )
    assert response.has_errors
    assert response.failed_item_count > 0
    assert response.upserted_count > 0, (
        "the deadline must expire part way through, not before the first batch"
    )
    assert response.failed_item_count == len(response.failed_items)
    assert response.upserted_count + response.failed_item_count == 3000


@pytest.mark.integration
def test_grpc_upsert_total_timeout_reports_unsent_failed_items(
    client: Pinecone,
    gap_index: str,
) -> None:
    index = client.index(name=gap_index, grpc=True)
    response = index.upsert(
        vectors=_vectors(3000, "g-u"),
        namespace=_ns("grpc-tt"),
        batch_size=1,
        max_concurrency=1,
        show_progress=False,
        total_timeout=_PARTIAL_TOTAL_TIMEOUT,
    )
    assert response.has_errors
    assert response.failed_item_count > 0
    assert response.failed_item_count == len(response.failed_items)
    assert response.upserted_count + response.failed_item_count == 3000


# ---------------------------------------------------------------------------
# max_concurrency default / validation
# ---------------------------------------------------------------------------


def test_max_concurrency_default_is_eight() -> None:
    assert DEFAULT_MAX_CONCURRENCY == 8


@pytest.mark.integration
def test_max_concurrency_out_of_range_rejected(
    client: Pinecone,
    gap_index: str,
) -> None:
    """max_concurrency outside [1, 64] must be rejected before any DB I/O."""
    index = client.index(name=gap_index)
    with pytest.raises(PineconeValueError):
        index.upsert(
            vectors=_vectors(10), namespace=_ns("badconc"), batch_size=5, max_concurrency=65
        )


@pytest.mark.integration
def test_default_concurrency_upsert_is_sane(
    client: Pinecone,
    gap_index: str,
) -> None:
    """With default concurrency (8) a batched upsert lands everything."""
    ns = _ns("sane")
    index = client.index(name=gap_index)
    response = index.upsert(
        vectors=_vectors(200, "r-sane"),
        namespace=ns,
        batch_size=50,
        show_progress=False,
    )
    assert response.upserted_count == 200
    assert not response.has_errors
    stats = poll_until(
        query_fn=lambda: index.describe_index_stats(),
        check_fn=lambda r: ns in r.namespaces and r.namespaces[ns].vector_count >= 200,
        timeout=120,
        description=f"200 default-concurrency vectors in stats ({ns})",
    )
    assert stats.namespaces[ns].vector_count == 200


# ---------------------------------------------------------------------------
# RetryConfig customization against a live backend
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_retry_config_max_retries_zero_upsert_works(
    api_key: str,
    gap_index: str,
) -> None:
    """max_retries=0 (single attempt, no retry) must still upsert successfully."""
    pc = Pinecone(api_key=api_key, retry_config=RetryConfig(max_retries=0))
    index = pc.index(name=gap_index)
    response = index.upsert(
        vectors=_vectors(150, "r-z"), namespace=_ns("retries0"), batch_size=50, show_progress=False
    )
    assert response.upserted_count == 150
    assert not response.has_errors


@pytest.mark.integration
def test_retry_config_custom_retryable_codes_upsert_works(
    api_key: str,
    gap_index: str,
) -> None:
    """Custom retryable_status_codes must not break a normal upsert."""
    pc = Pinecone(
        api_key=api_key,
        retry_config=RetryConfig(
            max_retries=2,
            backoff_factor=0.1,
            max_wait=2.0,
            retryable_status_codes=frozenset({408, 429, 500, 502, 503, 504}),
        ),
    )
    index = pc.index(name=gap_index)
    response = index.upsert(
        vectors=_vectors(150, "r-c3"), namespace=_ns("codes"), batch_size=50, show_progress=False
    )
    assert response.upserted_count == 150
    assert not response.has_errors


# ---------------------------------------------------------------------------
# upsert_from_dataframe partial-failure aggregation
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_upsert_from_dataframe_partial_failure_collect(
    client: Pinecone,
    gap_index: str,
) -> None:
    """on_error="collect" (default) returns failed_items on the response."""
    index = client.index(name=gap_index)
    df = pd.DataFrame(
        {
            "id": [f"df-c-{i}" for i in range(3000)],
            "values": [[0.2] * _DIM for _ in range(3000)],
        }
    )
    response = index.upsert_from_dataframe(
        df,
        namespace=_ns("df-collect"),
        batch_size=1,
        max_concurrency=1,
        show_progress=False,
        total_timeout=_PARTIAL_TOTAL_TIMEOUT,
    )
    assert response.has_errors
    assert response.failed_item_count > 0
    assert response.upserted_count > 0, (
        "the deadline must expire part way through, not before the first batch — "
        "otherwise the accounting assertion below holds trivially and nothing "
        "partial has been demonstrated"
    )
    assert response.failed_item_count == len(response.failed_items)
    assert response.upserted_count + response.failed_item_count == 3000


@pytest.mark.integration
def test_upsert_from_dataframe_on_error_raise_attaches_partial_response(
    client: Pinecone,
    gap_index: str,
) -> None:
    """on_error="raise" re-raises but carries the partial result on .response."""
    index = client.index(name=gap_index)
    df = pd.DataFrame(
        {
            "id": [f"df-r-{i}" for i in range(3000)],
            "values": [[0.3] * _DIM for _ in range(3000)],
        }
    )
    with pytest.raises(Exception) as excinfo:
        index.upsert_from_dataframe(
            df,
            namespace=_ns("df-raise"),
            batch_size=1,
            max_concurrency=1,
            show_progress=False,
            total_timeout=_PARTIAL_TOTAL_TIMEOUT,
            on_error="raise",
        )
    exc = excinfo.value
    assert getattr(exc, "response", None) is not None
    assert exc.response.has_errors  # type: ignore[attr-defined]
    assert exc.response.failed_item_count > 0  # type: ignore[attr-defined]
    assert exc.response.upserted_count > 0, (  # type: ignore[attr-defined]
        "the deadline must expire part way through, not before the first batch — "
        "otherwise the accounting assertion below holds trivially and the "
        "attached response is not a partial result"
    )
    assert exc.response.upserted_count + exc.response.failed_item_count == 3000  # type: ignore[attr-defined]
