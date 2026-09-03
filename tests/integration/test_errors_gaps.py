"""Gap-filling integration tests for error handling.

The core error surface (typed ApiError subclasses for 4xx/5xx, client-side
validation, gRPC DEADLINE_EXCEEDED -> PineconeTimeoutError) is covered by
test_errors.py / test_errors_async.py. This file closes the remaining gaps:

1. PineconeTimeoutError surfaced end-to-end over REST (sync + async) — the
   existing timeout test only exercises gRPC.
2. PineconeConnectionError from pointing an Index at an unreachable host, over
   all three transports (sync REST, async REST, gRPC) — not covered anywhere.
3. Error-surface parity: the same underlying fault (unreachable host / tiny
   timeout) raises the SAME exception type across sync REST, async REST, and gRPC.

The two timeout tests need an index that answers a real ``query``, so they take
one from :func:`legacy_index_factory` rather than ``pc.indexes.create``:
2026-07 has no way to create an index the vectors API will serve. The
connection-error tests need no index at all — they point a client at a dead
port on purpose.
"""

from __future__ import annotations

import uuid

import pytest

from pinecone import AsyncIndex, AsyncPinecone, GrpcIndex, Index, Pinecone
from pinecone.errors import (
    PineconeConnectionError,
    PineconeError,
    PineconeTimeoutError,
)
from tests.integration.conftest import LegacyIndexFactory
from tests.integration.legacy_index import assert_serves_vectors_api

# An unreachable but format-valid data-plane host. Port 1 has no listener, so
# the connection is refused immediately (fast, deterministic) without touching
# the real API. Contains dots so it passes the SDK's host-format check.
_UNREACHABLE_HOST = "http://127.0.0.1:1"


# ---------------------------------------------------------------------------
# PineconeConnectionError — sync REST Index
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_connection_error_rest_sync() -> None:
    """Pointing a sync REST Index at an unreachable host raises PineconeConnectionError.

    Verifies gap: only ApiError subclasses (HTTP status) were covered before.
    A network-level failure must surface as the typed PineconeConnectionError,
    and it must be catchable as the base PineconeError (unified-err-0001).
    """
    index = Index(host=_UNREACHABLE_HOST, api_key="testkey")
    with pytest.raises(PineconeConnectionError) as exc_info:
        index.describe_index_stats()

    err = exc_info.value
    assert isinstance(err, PineconeError)
    # message is human readable, not empty
    assert str(err)


# ---------------------------------------------------------------------------
# PineconeConnectionError — async REST Index
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_connection_error_rest_async() -> None:
    """Pointing an async REST Index at an unreachable host raises PineconeConnectionError.

    Async parity for test_connection_error_rest_sync.
    """
    index = AsyncIndex(host=_UNREACHABLE_HOST, api_key="testkey")
    try:
        with pytest.raises(PineconeConnectionError) as exc_info:
            await index.describe_index_stats()
        err = exc_info.value
        assert isinstance(err, PineconeError)
        assert str(err)
    finally:
        await index.close()


# ---------------------------------------------------------------------------
# PineconeConnectionError — gRPC Index
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_connection_error_rest_grpc() -> None:
    """Pointing a gRPC Index at an unreachable host raises PineconeConnectionError.

    The Rust-backed gRPC transport must not leak a raw gRPC/channel exception;
    it must map an unroutable connection to the same typed PineconeConnectionError
    as the REST transports.
    """
    index = GrpcIndex(host=_UNREACHABLE_HOST, api_key="testkey")
    try:
        with pytest.raises(PineconeConnectionError) as exc_info:
            index.describe_index_stats()
        err = exc_info.value
        assert isinstance(err, PineconeError)
        assert str(err)
    finally:
        index.close()


# ---------------------------------------------------------------------------
# Connection-error parity across all three transports
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_connection_error_surface_parity() -> None:
    """The same unreachable-host fault raises the SAME type on sync/async/gRPC.

    Verifies gap: error-surface parity across transports for a network-level fault.
    """
    # sync REST
    sync_index = Index(host=_UNREACHABLE_HOST, api_key="testkey")
    with pytest.raises(PineconeConnectionError):
        sync_index.describe_index_stats()
    sync_index.close()

    # async REST
    async_index = AsyncIndex(host=_UNREACHABLE_HOST, api_key="testkey")
    try:
        with pytest.raises(PineconeConnectionError):
            await async_index.describe_index_stats()
    finally:
        await async_index.close()

    # gRPC
    grpc_index = GrpcIndex(host=_UNREACHABLE_HOST, api_key="testkey")
    try:
        with pytest.raises(PineconeConnectionError):
            grpc_index.describe_index_stats()
    finally:
        grpc_index.close()


# ---------------------------------------------------------------------------
# PineconeTimeoutError — REST sync (real index)
# ---------------------------------------------------------------------------

# A per-attempt timeout that is far smaller than any real round-trip; the
# client must raise PineconeTimeoutError instead of a raw httpx error.
_TINY_TIMEOUT = 0.000001
_GENEROUS_TIMEOUT = 30.0


@pytest.fixture(scope="module")
def timeout_index(client: Pinecone, legacy_index_factory: LegacyIndexFactory):
    """Shared legacy index (dim=3, cosine) for the per-call timeout tests."""
    index = legacy_index_factory(dimension=3)
    assert_serves_vectors_api(client, index)
    return index


@pytest.mark.integration
def test_rest_query_too_short_timeout_raises(client: Pinecone, timeout_index) -> None:
    """A sub-millisecond per-call timeout over REST raises PineconeTimeoutError (sync).

    The existing timeout test only covers gRPC. A REST data-plane request whose
    client-side deadline fires must map to PineconeTimeoutError, and a subsequent
    call with a generous timeout must succeed, proving the knob is per-call.
    """
    ns = f"to-sync-{uuid.uuid4().hex[:8]}"
    index = client.index(name=timeout_index.name)
    index.upsert(vectors=[("t1", [0.1, 0.2, 0.3])], namespace=ns)

    with pytest.raises(PineconeTimeoutError) as exc_info:
        index.query(vector=[0.1, 0.2, 0.3], top_k=1, namespace=ns, timeout=_TINY_TIMEOUT)

    err = exc_info.value
    assert isinstance(err, PineconeError)

    # generous timeout proves channel is healthy and the knob is per-call
    result = index.query(vector=[0.1, 0.2, 0.3], top_k=1, namespace=ns, timeout=_GENEROUS_TIMEOUT)
    assert result.matches is not None


# ---------------------------------------------------------------------------
# PineconeTimeoutError — REST async (real index)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_rest_query_too_short_timeout_raises_async(
    async_client: AsyncPinecone, timeout_index
) -> None:
    """A sub-millisecond per-call timeout over REST raises PineconeTimeoutError (async).

    Async parity for test_rest_query_too_short_timeout_raises.
    """
    ns = f"to-async-{uuid.uuid4().hex[:8]}"
    index = await async_client.index(host=timeout_index.host)
    try:
        await index.upsert(vectors=[("t1", [0.1, 0.2, 0.3])], namespace=ns)

        with pytest.raises(PineconeTimeoutError) as exc_info:
            await index.query(vector=[0.1, 0.2, 0.3], top_k=1, namespace=ns, timeout=_TINY_TIMEOUT)

        err = exc_info.value
        assert isinstance(err, PineconeError)

        result = await index.query(
            vector=[0.1, 0.2, 0.3], top_k=1, namespace=ns, timeout=_GENEROUS_TIMEOUT
        )
        assert result.matches is not None
    finally:
        await index.close()
