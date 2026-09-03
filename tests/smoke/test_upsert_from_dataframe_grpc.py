"""Smoke test — gRPC ``GrpcIndex.upsert_from_dataframe`` (pandas-gated).

Skipped when pandas is not installed. Run with ``uv run --with pandas``
or by adding pandas to the dev environment.

``upsert_from_dataframe`` is a vectors-API write, so the index comes from the
``legacy_index_dim8`` fixture rather than from ``pc.indexes.create`` — see
``test_serverless_dense_sync.py`` for why (#322, #379).

Punchlist coverage (gRPC): GrpcIndex.upsert_from_dataframe.
"""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas", reason="pandas required for upsert_from_dataframe")

from pinecone import GrpcIndex, Pinecone  # noqa: E402
from tests.integration.legacy_index import (  # noqa: E402
    LegacyIndex,
    assert_serves_vectors_api,
)
from tests.smoke.conftest import (  # noqa: E402
    SMOKE_PREFIX,
    SMOKE_VECTOR_DIM,
    unique_name,
)
from tests.smoke.helpers import wait_for_vector_count  # noqa: E402

DIM = SMOKE_VECTOR_DIM


@pytest.mark.smoke
def test_upsert_from_dataframe_grpc(client: Pinecone, legacy_index_dim8: LegacyIndex) -> None:
    """Upsert a small DataFrame via GrpcIndex and confirm the count round-trips."""
    namespace = unique_name(f"{SMOKE_PREFIX}-udf-grpc")
    try:
        assert_serves_vectors_api(client, legacy_index_dim8)

        idx = client.index(name=legacy_index_dim8.name, grpc=True)
        assert isinstance(idx, GrpcIndex)
        try:
            df = pd.DataFrame(
                [
                    {
                        "id": f"d{i}",
                        "values": [0.50 + 0.01 * i + j * 0.001 for j in range(DIM)],
                    }
                    for i in range(5)
                ]
            )
            resp = idx.upsert_from_dataframe(df, namespace=namespace, show_progress=False)
            assert resp.upserted_count == 5
            wait_for_vector_count(idx, namespace, expected=5, timeout=60)

            fetched = idx.fetch(ids=[f"d{i}" for i in range(5)], namespace=namespace)
            assert set(fetched.vectors) == {f"d{i}" for i in range(5)}
            # Looser tolerance than REST: gRPC returns float32.
            assert fetched.vectors["d2"].values == pytest.approx(
                [0.50 + 0.01 * 2 + j * 0.001 for j in range(DIM)], abs=1e-5
            )
        finally:
            idx.close()
    finally:
        client.close()
