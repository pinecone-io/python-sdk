"""Smoke test — async ``AsyncIndex.upsert_from_dataframe`` (pandas-gated).

Skipped when pandas is not installed. Run with ``uv run --with pandas``
or by adding pandas to the dev environment.

``AsyncIndex.upsert_from_dataframe`` is a deliberate stub (#5, closed
without implementation): it validates ``batch_size`` and then always
raises ``NotImplementedError``. #429 resolved the punchlist line to
assert that documented refusal rather than wait on #5. The refusal
contract also has CI-gated unit coverage in
``tests/unit/test_upsert_from_dataframe.py``
(``TestAsyncUpsertFromDataframe``), so this module exists to make the
async punchlist line truthful rather than to add net-new coverage. The
call never reaches the network, so no ``legacy_index_dim8`` (or any
other) index fixture is needed — ``host=`` targets an index client
without triggering a describe call.

Punchlist coverage (async): AsyncIndex.upsert_from_dataframe refuses
with ``NotImplementedError``, after validating ``batch_size`` first.
"""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas", reason="pandas required for upsert_from_dataframe")

from pinecone import AsyncPinecone, PineconeValueError  # noqa: E402
from tests.smoke.conftest import SMOKE_VECTOR_DIM  # noqa: E402

DIM = SMOKE_VECTOR_DIM

_UNUSED_HOST = "smoke-unused-udf-async.svc.pinecone.io"
"""Never dialed: the stub raises before any request is built."""


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_upsert_from_dataframe_async_not_implemented(api_key: str) -> None:
    """AsyncIndex.upsert_from_dataframe always raises NotImplementedError."""
    pc = AsyncPinecone(api_key=api_key)
    try:
        idx = await pc.index(host=_UNUSED_HOST)
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
            with pytest.raises(NotImplementedError, match="not supported for async"):
                await idx.upsert_from_dataframe(df, show_progress=False)
        finally:
            await idx.close()
    finally:
        await pc.close()


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_upsert_from_dataframe_async_batch_size_checked_first(api_key: str) -> None:
    """An invalid batch_size raises PineconeValueError, not the unconditional NotImplementedError."""
    pc = AsyncPinecone(api_key=api_key)
    try:
        idx = await pc.index(host=_UNUSED_HOST)
        try:
            df = pd.DataFrame([{"id": "d0", "values": [0.30 + j * 0.001 for j in range(DIM)]}])
            with pytest.raises(PineconeValueError, match="batch_size must be a positive integer"):
                await idx.upsert_from_dataframe(df, batch_size=0, show_progress=False)
        finally:
            await idx.close()
    finally:
        await pc.close()
