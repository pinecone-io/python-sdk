"""Priority-3 smoke test — sync serverless dense + namespaces.

Walks the full Index data plane on a single dense serverless index, plus the
control-plane Indexes namespace and Pinecone top-level surface.

The index comes from the ``legacy_index_dim8`` fixture rather than from
``pc.indexes.create``: every 2026-07-created index is served by the documents
API, which refuses the entire vectors surface this module exists to walk
(#322, #379). ``pc.indexes.create`` and ``pc.indexes.delete`` are therefore
covered by the modules that legitimately create a 2026-07 index —
``test_serverless_integrated_sync.py``, ``test_backups_sync.py`` and
``test_imports_sync.py`` — and not here.

Punchlist coverage (sync):

- pc.indexes.list / describe / exists / configure
- pc.index (factory), pc.config, pc.close, ``with`` ctx mgr
- Index: upsert, query, query_namespaces, fetch, fetch_by_metadata, delete,
  update, describe_index_stats, create_namespace, describe_namespace,
  delete_namespace, list_namespaces_paginated, list_namespaces,
  list_paginated, list, close, ``with`` ctx mgr, host

Note: upsert_from_dataframe is tested in test_upsert_from_dataframe_sync.py
(pandas-gated). Imports lifecycle is tested in test_imports_sync.py
(PINECONE_IMPORT_S3_URI-gated).
"""

from __future__ import annotations

from typing import Any

import pytest

from pinecone import Pinecone, Vector
from tests.integration.legacy_index import LegacyIndex, assert_serves_vectors_api
from tests.smoke.conftest import (
    SMOKE_PREFIX,
    SMOKE_VECTOR_DIM,
    poll_until,
    unique_name,
)
from tests.smoke.helpers import wait_for_namespace_visible, wait_for_vector_count

DIM = SMOKE_VECTOR_DIM


def _vec(i: int, *, category: str | None = None) -> dict[str, object]:
    """Build a unique deterministic dict-form vector at index ``i``."""
    base = (i + 1) * 0.05
    record: dict[str, object] = {
        "id": f"v{i}",
        "values": [base + j * 0.01 for j in range(DIM)],
    }
    if category is not None:
        record["metadata"] = {"category": category}
    return record


@pytest.mark.smoke
def test_serverless_dense_smoke(client: Pinecone, legacy_index_dim8: LegacyIndex) -> None:
    """End-to-end serverless dense walkthrough."""
    name = legacy_index_dim8.name
    alpha = unique_name(f"{SMOKE_PREFIX}-alpha")
    beta = unique_name(f"{SMOKE_PREFIX}-beta")

    try:
        assert_serves_vectors_api(client, legacy_index_dim8)

        # ----- control plane: describe / list / exists / configure -----
        described = client.indexes.describe(name)
        assert described.name == name
        assert described.host

        assert name in {i.name for i in client.indexes.list()}

        assert client.indexes.exists(name) is True
        assert client.indexes.exists(f"{SMOKE_PREFIX}-does-not-exist") is False

        assert client.config.api_key  # Pinecone.config property

        # configure: tweak tags + deletion_protection (no-op rename essentially)
        client.indexes.configure(name, tags={"env": "smoke"})

        # ----- data plane via pc.index() factory -----
        idx = client.index(name=name)
        assert idx.host  # Index.host property
        try:
            # ----- upsert (mixed forms) -----
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
            up_resp = idx.upsert(vectors=mixed, namespace=alpha)
            assert up_resp.upserted_count == 10

            # ----- upsert: populate beta namespace (plain upsert) -----
            beta_records = [
                {
                    "id": f"b{i}",
                    "values": [0.30 + i * 0.01 + j * 0.001 for j in range(DIM)],
                }
                for i in range(5)
            ]
            beta_resp = idx.upsert(vectors=beta_records, namespace=beta)
            assert beta_resp.upserted_count == 5

            # ----- vector freshness -----
            wait_for_vector_count(idx, alpha, expected=10)

            # ----- query (vector form) -----
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
            assert matched <= {f"v{i}" for i in range(10)}
            assert "v1" in matched  # the query vector is v1's own values

            # ----- query_namespaces (alpha + beta both populated above) -----
            wait_for_vector_count(idx, beta, expected=5, timeout=60)
            multi = idx.query_namespaces(
                vector=[0.10 + j * 0.01 for j in range(DIM)],
                namespaces=[alpha, beta],
                metric="cosine",
                top_k=3,
            )
            # Keyed by id for the same reason as above (#368).
            multi_ids = {m.id for m in multi.matches}
            assert multi_ids
            assert multi_ids <= {f"v{i}" for i in range(10)} | {f"b{i}" for i in range(5)}

            # ----- fetch -----
            fetch_resp = idx.fetch(ids=["v0", "v1", "v2"], namespace=alpha)
            assert set(fetch_resp.vectors.keys()) == {"v0", "v1", "v2"}
            assert fetch_resp.vectors["v2"].values == pytest.approx(
                [0.15 + j * 0.01 for j in range(DIM)], abs=1e-5
            )
            assert fetch_resp.vectors["v2"].metadata == {"category": "x"}

            # ----- fetch_by_metadata -----
            fbm = idx.fetch_by_metadata(
                filter={"category": {"$eq": "x"}},
                namespace=alpha,
                limit=10,
            )
            # We upserted 3 vectors with category=x (v2, v3, v9). Loose assertion
            # because fetch_by_metadata may paginate.
            assert len(fbm.vectors) >= 1
            assert set(fbm.vectors) <= {"v2", "v3", "v9"}

            # ----- update -----
            idx.update(
                id="v0",
                set_metadata={"tag": "updated"},
                namespace=alpha,
            )
            poll_until(
                lambda: idx.fetch(ids=["v0"], namespace=alpha),
                lambda r: r.vectors["v0"].metadata == {"tag": "updated"},
                timeout=60,
                description=f"v0 metadata update visible in {alpha}",
            )

            # ----- list_paginated / list -----
            page = idx.list_paginated(prefix="v", limit=5, namespace=alpha)
            assert len(page.vectors) > 0
            all_ids: list[str] = []
            for p in idx.list(prefix="v", namespace=alpha):
                all_ids.extend(item.id for item in p.vectors)
            assert set(all_ids) == {f"v{i}" for i in range(10)}

            # ----- describe_index_stats -----
            stats = idx.describe_index_stats()
            assert alpha in stats.namespaces
            assert stats.namespaces[alpha].vector_count == 10

            # ----- namespace ops -----
            ns_name = unique_name(f"{SMOKE_PREFIX}-gamma")
            created_ns = idx.create_namespace(name=ns_name)
            assert created_ns.name == ns_name
            # describe/list briefly return 404 after create_namespace returns
            # 200 — wait for visibility before asserting.
            described_ns = wait_for_namespace_visible(idx, ns_name)
            assert described_ns.name == ns_name
            # The index is shared by every dim-8 smoke module, so the listing
            # carries their namespaces too — page wide enough to reach ours.
            ns_page = idx.list_namespaces_paginated(limit=100)
            ns_names = {ns.name for ns in ns_page.namespaces}
            assert ns_name in ns_names
            iter_names: list[str] = []
            for page in idx.list_namespaces():
                iter_names.extend(ns.name for ns in page.namespaces)
            assert ns_name in iter_names
            idx.delete_namespace(name=ns_name)

            # ----- delete (by ids) -----
            idx.delete(ids=["v0", "v1"], namespace=alpha)
            poll_until(
                lambda: idx.fetch(ids=["v0", "v1", "v2"], namespace=alpha),
                lambda r: set(r.vectors) == {"v2"},
                timeout=60,
                description=f"v0/v1 deleted from {alpha}",
            )

            # ----- with-statement context manager on Index -----
        finally:
            idx.close()

        # Re-open via context manager
        with client.index(name=name) as idx2:
            stats2 = idx2.describe_index_stats()
            assert stats2.dimension == DIM

            # async_req=True opt-in walkthrough (legacy execution model)
            from multiprocessing.pool import ApplyResult

            pool_client = Pinecone(api_key=client.config.api_key, pool_threads=2)
            with pool_client.index(name=name) as pool_idx:
                async_upsert: Any = pool_idx.upsert(  # type: ignore[call-arg]
                    vectors=[_vec(100), _vec(101)],
                    namespace=beta,
                    async_req=True,
                )
                assert isinstance(async_upsert, ApplyResult)
                assert async_upsert.get(timeout=60).upserted_count == 2

                async_stats: Any = pool_idx.describe_index_stats(async_req=True)  # type: ignore[call-arg]
                assert isinstance(async_stats, ApplyResult)
                stats = async_stats.get(timeout=60)
                assert stats.dimension == DIM

            # BC-0113: async_req=True works without an explicit pool_threads=
            # opt-in. The default 10-thread pool is installed lazily on first
            # async_req call, no upfront configuration needed.
            default_async_upsert: Any = idx2.upsert(  # type: ignore[call-arg]
                vectors=[_vec(102), _vec(103)],
                namespace=beta,
                async_req=True,
            )
            assert isinstance(default_async_upsert, ApplyResult)
            assert default_async_upsert.get(timeout=60).upserted_count == 2

            wait_for_vector_count(idx2, beta, expected=9)
            default_async_query: Any = idx2.query(  # type: ignore[call-arg]
                vector=[0.30 + j * 0.001 for j in range(DIM)],
                top_k=1,
                namespace=beta,
                async_req=True,
            )
            assert isinstance(default_async_query, ApplyResult)
            async_matches = default_async_query.get(timeout=60).matches
            assert len(async_matches) == 1
            assert async_matches[0].id in {f"b{i}" for i in range(5)} | {
                "v100",
                "v101",
                "v102",
                "v103",
            }

    finally:
        client.close()
