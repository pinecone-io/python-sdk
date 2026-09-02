"""Smoke test — async ``AsyncIndex.upsert_from_dataframe`` (pandas-gated).

Skipped when pandas is not installed. Run with ``uv run --with pandas``
or by adding pandas to the dev environment.

``upsert_from_dataframe`` is a vectors-API write, so the index comes from the
``legacy_index_dim8`` fixture rather than from ``pc.indexes.create`` — see
``test_serverless_dense_async.py`` for why (#322, #379).

Mirror of ``test_upsert_from_dataframe_sync.py`` against ``AsyncPinecone`` and
``AsyncIndex``, which carry the same single signature as the sync and gRPC
transports (#525). The invalid-``batch_size`` case is validated client-side
before any request is built, so it targets an unreachable ``host=`` rather
than spending the shared fixture index on it.

Punchlist coverage (async): AsyncIndex.upsert_from_dataframe, and its
rejection of a non-positive ``batch_size``.
"""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas", reason="pandas required for upsert_from_dataframe")

from pinecone import AsyncPinecone, Pinecone, PineconeValueError  # noqa: E402
from pinecone.async_client import AsyncIndex  # noqa: E402
from tests.integration.legacy_index import (  # noqa: E402
    LegacyIndex,
    assert_serves_vectors_api,
)
from tests.smoke.conftest import (  # noqa: E402
    SMOKE_PREFIX,
    SMOKE_VECTOR_DIM,
    unique_name,
)
from tests.smoke.helpers import async_wait_for_vector_count  # noqa: E402

DIM = SMOKE_VECTOR_DIM

_UNREACHABLE_HOST = "smoke-unused-udf-async.svc.pinecone.io"
"""Never dialed: ``batch_size`` is rejected before any request is built."""


def _frame(rows: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": f"d{i}",
                "values": [0.30 + i * 0.01 + j * 0.001 for j in range(DIM)],
            }
            for i in range(rows)
        ]
    )


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_upsert_from_dataframe_async(api_key: str, legacy_index_dim8: LegacyIndex) -> None:
    """Upsert a small DataFrame over async REST and confirm the count round-trips."""
    namespace = unique_name(f"{SMOKE_PREFIX}-udf-async")
    with Pinecone(api_key=api_key) as guard:
        assert_serves_vectors_api(guard, legacy_index_dim8)

    pc = AsyncPinecone(api_key=api_key)
    try:
        raw_idx = await pc.index(name=legacy_index_dim8.name)
        assert isinstance(raw_idx, AsyncIndex)
        idx: AsyncIndex = raw_idx
        try:
            resp = await idx.upsert_from_dataframe(
                _frame(5), namespace=namespace, batch_size=2, show_progress=False
            )
            assert resp.upserted_count == 5
            assert resp.failed_item_count == 0
            await async_wait_for_vector_count(idx, namespace, expected=5, timeout=60)

            fetched = await idx.fetch(ids=[f"d{i}" for i in range(5)], namespace=namespace)
            assert set(fetched.vectors) == {f"d{i}" for i in range(5)}
            assert fetched.vectors["d2"].values == pytest.approx(
                [0.30 + 2 * 0.01 + j * 0.001 for j in range(DIM)], abs=1e-5
            )
        finally:
            await idx.close()
    finally:
        await pc.close()


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_upsert_from_dataframe_async_rejects_invalid_batch_size(api_key: str) -> None:
    """A non-positive batch_size raises PineconeValueError before any request is sent."""
    pc = AsyncPinecone(api_key=api_key)
    try:
        idx = await pc.index(host=_UNREACHABLE_HOST)
        try:
            with pytest.raises(PineconeValueError, match="batch_size must be a positive integer"):
                await idx.upsert_from_dataframe(_frame(1), batch_size=0, show_progress=False)
        finally:
            await idx.close()
    finally:
        await pc.close()
