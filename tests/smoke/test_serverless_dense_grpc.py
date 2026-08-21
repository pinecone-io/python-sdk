"""Priority-3 smoke test — gRPC dense data plane.

Walks every method on :class:`GrpcIndex` against a serverless dense index.
Includes the future-returning ``*_async`` variants — each one is consumed via
``.result(timeout=10)`` to confirm they return the same shape as their
synchronous counterparts.

Punchlist coverage (gRPC):

- upsert / upsert_async
- query / query_async
- fetch / fetch_async
- delete (ids/delete_all/filter) / delete_async
- update / update_async
- list / list_paginated
- describe_index_stats
- close / ``with`` ctx mgr
- host property

GrpcIndex has no namespace ops, no fetch_by_metadata, no query_namespaces,
no imports. Those checkboxes only get ticked by the sync/async variants.
upsert_from_dataframe is tested in test_upsert_from_dataframe_grpc.py
(pandas-gated).
``upsert_records``, ``search``, and ``search_records`` require an integrated
index — covered by the Priority-4 gRPC test.

The index comes from the ``legacy_index_dim8`` fixture rather than from
``pc.indexes.create`` — see ``test_serverless_dense_sync.py`` for why, and for
where ``create`` / ``delete`` are covered instead.
"""

from __future__ import annotations

import pytest

from pinecone import GrpcIndex, Pinecone, Vector
from tests.integration.legacy_index import LegacyIndex, assert_serves_vectors_api
from tests.smoke.conftest import SMOKE_PREFIX, SMOKE_VECTOR_DIM, poll_until, unique_name
from tests.smoke.helpers import wait_for_vector_count

DIM = SMOKE_VECTOR_DIM


@pytest.mark.smoke
def test_serverless_dense_grpc_smoke(client: Pinecone, legacy_index_dim8: LegacyIndex) -> None:
    """End-to-end gRPC dense data-plane walkthrough."""
    name = legacy_index_dim8.name
    alpha = unique_name(f"{SMOKE_PREFIX}-alpha-grpc")
    gamma = unique_name(f"{SMOKE_PREFIX}-gamma-grpc")
    try:
        assert_serves_vectors_api(client, legacy_index_dim8)

        # gRPC handle via Pinecone.index(grpc=True)
        idx = client.index(name=name, grpc=True)
        assert isinstance(idx, GrpcIndex)
        assert idx.host

        try:
            # ----- upsert (sync) -----
            # Two metadata categories so we can later delete by filter.
            base_vectors = [
                Vector(
                    id=f"v{i}",
                    values=[0.05 * (i + 1) + j * 0.01 for j in range(DIM)],
                    metadata={"category": "keep" if i < 3 else "drop"},
                )
                for i in range(5)
            ]
            sync_resp = idx.upsert(vectors=base_vectors, namespace=alpha)
            assert sync_resp.upserted_count == 5

            # ----- upsert_async (returns PineconeFuture) -----
            async_vectors = [
                {
                    "id": f"a{i}",
                    "values": [0.30 + 0.01 * i + j * 0.002 for j in range(DIM)],
                }
                for i in range(5)
            ]
            up_future = idx.upsert_async(vectors=async_vectors, namespace=alpha)
            up_async_resp = up_future.result(timeout=10.0)
            assert up_async_resp.upserted_count == 5

            # ----- vector freshness -----
            wait_for_vector_count(idx, alpha, expected=10, timeout=60)

            # ----- query (sync) -----
            q_resp = idx.query(
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
            assert matched <= {f"v{i}" for i in range(5)} | {f"a{i}" for i in range(5)}
            assert "v1" in matched  # the query vector is v1's own values

            # ----- query_async -----
            # Polled, not single-shot: this returned matches=[] once in a run
            # where the sync query immediately above returned 3 and the count
            # wait had already confirmed 10 vectors — the same read-side
            # inconsistency #322 records, seen here on a legacy index.
            q_async = poll_until(
                lambda: idx.query_async(
                    top_k=2,
                    vector=[0.10 + j * 0.01 for j in range(DIM)],
                    namespace=alpha,
                ).result(timeout=10.0),
                lambda r: len(r.matches) == 2,
                timeout=60,
                description=f"query_async returns 2 matches in {alpha}",
            )
            # Keyed by id for the same reason as above (#368).
            assert {m.id for m in q_async.matches} <= {f"v{i}" for i in range(5)} | {
                f"a{i}" for i in range(5)
            }

            # ----- fetch (sync) -----
            fetch_resp = idx.fetch(ids=["v0", "v1"], namespace=alpha)
            assert set(fetch_resp.vectors.keys()) == {"v0", "v1"}
            # Looser tolerance than REST: gRPC returns float32.
            assert fetch_resp.vectors["v1"].values == pytest.approx(
                [0.10 + j * 0.01 for j in range(DIM)], abs=1e-5
            )
            assert fetch_resp.vectors["v1"].metadata == {"category": "keep"}

            # ----- fetch_async -----
            f_future = idx.fetch_async(ids=["v2", "v3"], namespace=alpha)
            f_async = f_future.result(timeout=10.0)
            assert set(f_async.vectors.keys()) == {"v2", "v3"}
            assert f_async.vectors["v3"].metadata == {"category": "drop"}

            # ----- update (sync) -----
            idx.update(
                id="v0",
                set_metadata={"tag": "grpc-updated"},
                namespace=alpha,
            )
            poll_until(
                lambda: idx.fetch(ids=["v0"], namespace=alpha),
                lambda r: r.vectors["v0"].metadata.get("tag") == "grpc-updated",
                timeout=60,
                description=f"v0 metadata update visible in {alpha}",
            )

            # ----- update_async -----
            u_future = idx.update_async(
                id="v1",
                set_metadata={"tag": "grpc-updated-async"},
                namespace=alpha,
            )
            u_future.result(timeout=10.0)
            poll_until(
                lambda: idx.fetch(ids=["v1"], namespace=alpha),
                lambda r: r.vectors["v1"].metadata.get("tag") == "grpc-updated-async",
                timeout=60,
                description=f"v1 metadata update visible in {alpha}",
            )

            # ----- list_paginated -----
            page = idx.list_paginated(prefix="v", limit=5, namespace=alpha)
            ids_seen = [item.id for item in page.vectors]
            assert len(ids_seen) > 0

            # ----- list (iterator) -----
            all_ids: list[str] = []
            for p in idx.list(prefix="v", namespace=alpha):
                all_ids.extend(item.id for item in p.vectors)
            assert set(all_ids) == {f"v{i}" for i in range(5)}

            # ----- describe_index_stats -----
            stats = idx.describe_index_stats()
            assert alpha in stats.namespaces
            assert stats.namespaces[alpha].vector_count == 10

            # ----- delete by filter -----
            # Removes any vector with category=drop: v3 and v4. set_metadata
            # merges, so the updates above left v0/v1 on category=keep.
            idx.delete(filter={"category": {"$eq": "drop"}}, namespace=alpha)
            poll_until(
                lambda: idx.fetch(ids=["v2", "v3", "v4"], namespace=alpha),
                lambda r: set(r.vectors) == {"v2"},
                timeout=60,
                description=f"category=drop deleted from {alpha}",
            )

            # ----- delete_all on a throwaway namespace -----
            # Populate a separate namespace, then nuke it. Run against gRPC to
            # exercise the delete_all=True wire path specifically.
            throwaway = [
                Vector(id=f"t{i}", values=[0.50 + i * 0.01 + j * 0.001 for j in range(DIM)])
                for i in range(3)
            ]
            throwaway_resp = idx.upsert(vectors=throwaway, namespace=gamma)
            assert throwaway_resp.upserted_count == 3
            wait_for_vector_count(idx, gamma, expected=3, timeout=60)
            idx.delete(delete_all=True, namespace=gamma)
            poll_until(
                lambda: idx.fetch(ids=[f"t{i}" for i in range(3)], namespace=gamma),
                lambda r: not r.vectors,
                timeout=60,
                description=f"delete_all emptied {gamma}",
            )

            # ----- delete (sync) -----
            idx.delete(ids=["v0"], namespace=alpha)

            # ----- delete_async -----
            d_future = idx.delete_async(ids=["v1"], namespace=alpha)
            d_future.result(timeout=10.0)
            poll_until(
                lambda: idx.fetch(ids=["v0", "v1", "v2"], namespace=alpha),
                lambda r: set(r.vectors) == {"v2"},
                timeout=60,
                description=f"v0/v1 deleted from {alpha}",
            )
        finally:
            idx.close()

        # ----- with ctx mgr -----
        with client.index(name=name, grpc=True) as ctx_idx:
            stats2 = ctx_idx.describe_index_stats()
            assert stats2.dimension == DIM
    finally:
        client.close()
