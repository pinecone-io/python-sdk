"""Priority-3 smoke test — async serverless dense + namespaces.

Mirror of ``test_serverless_dense_sync.py`` against ``AsyncPinecone`` and
``AsyncIndex``, including its use of the ``legacy_index_dim8`` fixture in
place of ``pc.indexes.create`` — see that module's docstring for why, and for
where ``create`` / ``delete`` are covered instead.

Punchlist coverage (async): AsyncIndexes describe / list / exists / configure,
the full AsyncIndex data plane, and the AsyncPinecone top-level surface.

Note: upsert_from_dataframe is tested in test_upsert_from_dataframe_async.py
(pandas-gated). Imports lifecycle is tested in test_imports_async.py
(PINECONE_IMPORT_S3_URI-gated).
"""

from __future__ import annotations

import pytest

from pinecone import AsyncPinecone, Pinecone, Vector
from tests.integration.legacy_index import LegacyIndex, assert_serves_vectors_api
from tests.smoke.conftest import (
    SMOKE_PREFIX,
    SMOKE_VECTOR_DIM,
    async_poll_until,
    unique_name,
)
from tests.smoke.helpers import async_wait_for_namespace_visible, async_wait_for_vector_count

DIM = SMOKE_VECTOR_DIM


def _vec(i: int, *, category: str | None = None) -> dict[str, object]:
    base = (i + 1) * 0.05
    record: dict[str, object] = {
        "id": f"v{i}",
        "values": [base + j * 0.01 for j in range(DIM)],
    }
    if category is not None:
        record["metadata"] = {"category": category}
    return record


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_serverless_dense_smoke_async(api_key: str, legacy_index_dim8: LegacyIndex) -> None:
    """End-to-end async serverless dense walkthrough."""
    pc = AsyncPinecone(api_key=api_key)
    name = legacy_index_dim8.name
    alpha = unique_name(f"{SMOKE_PREFIX}-alpha-async")
    beta = unique_name(f"{SMOKE_PREFIX}-beta-async")

    try:
        with Pinecone(api_key=api_key) as guard:
            assert_serves_vectors_api(guard, legacy_index_dim8)

        # ----- control plane -----
        described = await pc.indexes.describe(name)
        assert described.name == name

        assert name in {i.name async for i in pc.indexes.list()}

        assert await pc.indexes.exists(name) is True
        assert await pc.indexes.exists(f"{SMOKE_PREFIX}-does-not-exist-async") is False

        assert pc.config.api_key

        await pc.indexes.configure(name, tags={"env": "smoke-async"})

        # ----- data plane -----
        idx = await pc.index(name=name)
        assert idx.host
        try:
            mixed = [
                Vector(id="v0", values=[0.05 + j * 0.01 for j in range(DIM)]),
                ("v1", [0.10 + j * 0.01 for j in range(DIM)]),
                ("v2", [0.15 + j * 0.01 for j in range(DIM)], {"category": "x"}),
                _vec(3, category="x"),
                _vec(4),
                _vec(5, category="y"),
                _vec(6),
                _vec(7),
                _vec(8),
                _vec(9, category="x"),
            ]
            up_resp = await idx.upsert(vectors=mixed, namespace=alpha)
            assert up_resp.upserted_count == 10

            # ----- upsert: populate beta namespace (plain upsert) -----
            beta_records = [
                {
                    "id": f"b{i}",
                    "values": [0.30 + i * 0.01 + j * 0.001 for j in range(DIM)],
                }
                for i in range(5)
            ]
            beta_resp = await idx.upsert(vectors=beta_records, namespace=beta)
            assert beta_resp.upserted_count == 5

            await async_wait_for_vector_count(idx, alpha, expected=10)

            q_resp = await idx.query(
                top_k=3,
                vector=[0.10 + j * 0.01 for j in range(DIM)],
                namespace=alpha,
                include_metadata=True,
            )
            assert len(q_resp.matches) == 3
            # Keyed by id, never by position: the server returns matches
            # unsorted by score (#368), so asserting an order would encode a
            # server bug as our contract and break when it is fixed.
            matched = {m.id for m in q_resp.matches}
            assert matched <= {f"v{i}" for i in range(10)}
            assert "v1" in matched  # the query vector is v1's own values

            # ----- query_namespaces (alpha + beta both populated above) -----
            await async_wait_for_vector_count(idx, beta, expected=5, timeout=60)
            multi = await idx.query_namespaces(
                vector=[0.10 + j * 0.01 for j in range(DIM)],
                namespaces=[alpha, beta],
                metric="cosine",
                top_k=3,
            )
            # Keyed by id for the same reason as above (#368).
            multi_ids = {m.id for m in multi.matches}
            assert multi_ids
            assert multi_ids <= {f"v{i}" for i in range(10)} | {f"b{i}" for i in range(5)}

            fetch_resp = await idx.fetch(ids=["v0", "v1", "v2"], namespace=alpha)
            assert set(fetch_resp.vectors.keys()) == {"v0", "v1", "v2"}
            assert fetch_resp.vectors["v2"].values == pytest.approx(
                [0.15 + j * 0.01 for j in range(DIM)], abs=1e-5
            )
            assert fetch_resp.vectors["v2"].metadata == {"category": "x"}

            fbm = await idx.fetch_by_metadata(
                filter={"category": {"$eq": "x"}},
                namespace=alpha,
                limit=10,
            )
            assert len(fbm.vectors) >= 1
            assert set(fbm.vectors) <= {"v2", "v3", "v9"}

            await idx.update(
                id="v0",
                set_metadata={"tag": "updated-async"},
                namespace=alpha,
            )
            await async_poll_until(
                lambda: idx.fetch(ids=["v0"], namespace=alpha),
                lambda r: r.vectors["v0"].metadata == {"tag": "updated-async"},
                timeout=60,
                description=f"v0 metadata update visible in {alpha}",
            )

            page = await idx.list_paginated(prefix="v", limit=5, namespace=alpha)
            assert len(page.vectors) > 0

            all_ids: list[str] = []
            async for p in idx.list(prefix="v", namespace=alpha):
                all_ids.extend(item.id for item in p.vectors)
            assert set(all_ids) == {f"v{i}" for i in range(10)}

            stats = await idx.describe_index_stats()
            assert alpha in stats.namespaces
            assert stats.namespaces[alpha].vector_count == 10

            ns_name = unique_name(f"{SMOKE_PREFIX}-gamma-async")
            created_ns = await idx.create_namespace(name=ns_name)
            assert created_ns.name == ns_name
            # describe/list briefly return 404 after create_namespace returns
            # 200 — wait for visibility before asserting.
            described_ns = await async_wait_for_namespace_visible(idx, ns_name)
            assert described_ns.name == ns_name
            # The index is shared by every dim-8 smoke module, so the listing
            # carries their namespaces too — page wide enough to reach ours.
            ns_page = await idx.list_namespaces_paginated(limit=100)
            ns_names = {ns.name for ns in ns_page.namespaces}
            assert ns_name in ns_names
            iter_names: list[str] = []
            async for ns_response in idx.list_namespaces():
                iter_names.extend(ns.name for ns in ns_response.namespaces)
            assert ns_name in iter_names
            await idx.delete_namespace(name=ns_name)

            await idx.delete(ids=["v0", "v1"], namespace=alpha)
            await async_poll_until(
                lambda: idx.fetch(ids=["v0", "v1", "v2"], namespace=alpha),
                lambda r: set(r.vectors) == {"v2"},
                timeout=60,
                description=f"v0/v1 deleted from {alpha}",
            )
        finally:
            await idx.close()

        async with await pc.index(name=name) as idx2:
            stats2 = await idx2.describe_index_stats()
            assert stats2.dimension == DIM
    finally:
        await pc.close()
