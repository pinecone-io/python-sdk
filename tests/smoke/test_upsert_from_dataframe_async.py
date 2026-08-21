"""Smoke test — async ``AsyncIndex.upsert_from_dataframe`` (pandas-gated).

Skipped when pandas is not installed. Run with ``uv run --with pandas``
or by adding pandas to the dev environment.

``upsert_from_dataframe`` is a vectors-API write, so the index comes from the
``legacy_index_dim8`` fixture rather than from ``pc.indexes.create`` — see
``test_serverless_dense_sync.py`` for why (#322, #379).

**This test currently fails**, and not for a reason #379 owns:
``AsyncIndex.upsert_from_dataframe`` is a deliberate stub that always raises
``NotImplementedError``, and #5 (implement it) was closed unimplemented. The
stale create kwargs used to mask that with a ``PineconeTypeError``; migrating
them makes the real gap the visible failure. Whether the async punchlist line
should assert the refusal instead of waiting for #5 is tracked by #429, so
nothing here is weakened to get green.

Punchlist coverage (async): AsyncIndex.upsert_from_dataframe.
"""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas", reason="pandas required for upsert_from_dataframe")

from pinecone import AsyncPinecone, Pinecone  # noqa: E402
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


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_upsert_from_dataframe_async(api_key: str, legacy_index_dim8: LegacyIndex) -> None:
    """Upsert a small DataFrame via AsyncIndex and confirm the count round-trips."""
    pc = AsyncPinecone(api_key=api_key)
    namespace = unique_name(f"{SMOKE_PREFIX}-udf-async")
    try:
        with Pinecone(api_key=api_key) as guard:
            assert_serves_vectors_api(guard, legacy_index_dim8)

        idx = await pc.index(name=legacy_index_dim8.name)
        try:
            df = pd.DataFrame(
                [
                    {
                        "id": f"d{i}",
                        "values": [0.30 + i * 0.01 + j * 0.001 for j in range(DIM)],
                    }
                    for i in range(5)
                ]
            )
            resp = await idx.upsert_from_dataframe(df, namespace=namespace, show_progress=False)
            assert resp.upserted_count == 5
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
