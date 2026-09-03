"""The upgrade path: a 2026-07 SDK against an index created earlier (#363).

Vector operations stay available for indexes created under earlier API
versions, and a production workload upserting and querying one of those must
not break when it upgrades the SDK. Nothing else in the suite covers that:
every other vector test creates its index at the current API version, which
on 2026-07 yields a document-schema index that refuses the whole vectors API.

So these tests get their index from :mod:`tests.integration.legacy_index`,
and they assert **success with real values** — a round trip whose contents
are checked, not merely the absence of an exception. A test that only proved
"no exception" would still pass against a lane returning empty results.

The one deliberate negative, ``test_wrong_dimension_upsert_*``, carries a
correct-size control write in the same test. Without the control it would
also pass against an index that refuses every write, which is exactly how
``test_dimension_mismatch`` went vacuous on #305.

Query assertions here are keyed by id, not by position. The backend returns
``matches`` unsorted by score once ``top_k >= 3`` (#368) — server-side and
present at 2026-04 too, so asserting a list order would fail for a reason
that has nothing to do with this guarantee. Asserting that the self-match
scores highest is the stronger check regardless.

Sync REST and gRPC live here; async REST is in the ``_async`` twin. Both
share the session-scoped index, so every test namespaces itself.
"""

from __future__ import annotations

import uuid

import pytest

from pinecone import GrpcIndex, Index, Pinecone
from pinecone.errors import ApiError
from tests.integration.conftest import poll_until
from tests.integration.legacy_index import LegacyIndex, assert_serves_vectors_api

VECTORS = {
    "lv-a": [0.10, 0.20, 0.30],
    "lv-b": [0.40, 0.50, 0.60],
    "lv-c": [0.70, 0.80, 0.90],
}


def _namespace(label: str) -> str:
    return f"legacy-{label}-{uuid.uuid4().hex[:8]}"


def _seed(index: Index | GrpcIndex, namespace: str) -> None:
    """Upsert :data:`VECTORS` and wait until all three are fetchable."""
    result = index.upsert(
        vectors=[
            {"id": "lv-a", "values": VECTORS["lv-a"], "metadata": {"genre": "drama", "year": 1994}},
            {"id": "lv-b", "values": VECTORS["lv-b"]},
            {"id": "lv-c", "values": VECTORS["lv-c"]},
        ],
        namespace=namespace,
    )
    assert result.upserted_count == 3

    poll_until(
        query_fn=lambda: index.fetch(ids=list(VECTORS), namespace=namespace),
        check_fn=lambda r: len(r.vectors) == len(VECTORS),
        timeout=120,
        description="seeded legacy-index vectors fetchable",
    )


@pytest.mark.integration
def test_legacy_index_serves_vectors_api(client: Pinecone, legacy_index_dim3: LegacyIndex) -> None:
    """Guard against the whole module going vacuous.

    On a document-schema index every vectors-API call is refused, so the
    negative test below would pass for the wrong reason and no other
    assertion here would notice. This one fails loudly instead.
    """
    assert_serves_vectors_api(client, legacy_index_dim3)
    assert legacy_index_dim3.dimension == 3
    assert legacy_index_dim3.metric == "cosine"


@pytest.mark.integration
def test_legacy_upsert_fetch_query_rest(client: Pinecone, legacy_index_dim3: LegacyIndex) -> None:
    """upsert -> fetch -> query over REST, checking values and metadata."""
    ns = _namespace("rest-read")
    index = client.index(host=legacy_index_dim3.host)
    _seed(index, ns)

    fetched = index.fetch(ids=["lv-a", "lv-b"], namespace=ns)
    assert set(fetched.vectors) == {"lv-a", "lv-b"}
    assert fetched.vectors["lv-a"].values == pytest.approx(VECTORS["lv-a"])
    assert fetched.vectors["lv-a"].metadata == {"genre": "drama", "year": 1994}
    assert fetched.vectors["lv-b"].values == pytest.approx(VECTORS["lv-b"])

    result = index.query(
        vector=VECTORS["lv-a"],
        top_k=3,
        namespace=ns,
        include_values=True,
        include_metadata=True,
    )
    # Keyed by id, never by position: the server returns matches unsorted by
    # score (#368), so asserting an order would encode a server bug as our
    # contract and break when it is fixed. The set and the argmax are checked
    # instead, which is what "the right neighbours, scored right" means.
    by_match = {m.id: m for m in result.matches}
    assert set(by_match) == set(VECTORS)
    assert by_match["lv-a"].score == pytest.approx(1.0, abs=1e-4)
    assert by_match["lv-a"].values == pytest.approx(VECTORS["lv-a"])
    assert by_match["lv-a"].metadata == {"genre": "drama", "year": 1994}
    assert max(by_match, key=lambda i: by_match[i].score) == "lv-a"

    by_id = index.query(id="lv-c", top_k=1, namespace=ns, include_values=True)
    assert len(by_id.matches) == 1
    assert by_id.matches[0].id == "lv-c"
    assert by_id.matches[0].values == pytest.approx(VECTORS["lv-c"])


@pytest.mark.integration
def test_legacy_update_list_stats_delete_rest(
    client: Pinecone, legacy_index_dim3: LegacyIndex
) -> None:
    """update -> list -> describe_index_stats -> delete, each verified by its effect."""
    ns = _namespace("rest-write")
    index = client.index(host=legacy_index_dim3.host)
    _seed(index, ns)

    replacement = [0.01, 0.02, 0.03]
    index.update(id="lv-a", values=replacement, namespace=ns)
    updated = poll_until(
        query_fn=lambda: index.fetch(ids=["lv-a"], namespace=ns),
        check_fn=lambda r: r.vectors["lv-a"].values == pytest.approx(replacement),
        timeout=120,
        description="updated values visible",
    )
    assert updated.vectors["lv-a"].values == pytest.approx(replacement)

    listed = poll_until(
        query_fn=lambda: {item.id for page in index.list(namespace=ns) for item in page.vectors},
        check_fn=lambda ids: ids == set(VECTORS),
        timeout=120,
        description="all seeded ids visible to list",
    )
    assert listed == set(VECTORS)

    paginated = index.list_paginated(namespace=ns)
    assert paginated.vectors, "list_paginated returned an empty page"
    assert {item.id for item in paginated.vectors} <= set(VECTORS)

    stats = poll_until(
        query_fn=index.describe_index_stats,
        check_fn=lambda s: (
            s.namespaces.get(ns) is not None and s.namespaces[ns].vector_count == len(VECTORS)
        ),
        timeout=120,
        description="describe_index_stats counts the seeded namespace",
    )
    assert stats.dimension == 3
    assert stats.metric == "cosine"
    assert stats.namespaces[ns].vector_count == len(VECTORS)

    index.delete(ids=["lv-b"], namespace=ns)
    poll_until(
        query_fn=lambda: index.fetch(ids=["lv-b"], namespace=ns),
        check_fn=lambda r: len(r.vectors) == 0,
        timeout=120,
        description="deleted vector gone",
    )
    survivors = index.fetch(ids=["lv-a", "lv-c"], namespace=ns)
    assert set(survivors.vectors) == {"lv-a", "lv-c"}


@pytest.mark.integration
def test_legacy_wrong_dimension_upsert_rejected_rest(
    client: Pinecone, legacy_index_dim3: LegacyIndex
) -> None:
    """A wrong-dimension upsert fails while a correct-size one succeeds.

    The control write is the point: it proves the rejection is about the
    dimension rather than a blanket refusal of the vectors API.
    """
    ns = _namespace("rest-dim")
    index = client.index(host=legacy_index_dim3.host)

    control = index.upsert(vectors=[{"id": "ctl", "values": [0.1, 0.2, 0.3]}], namespace=ns)
    assert control.upserted_count == 1

    with pytest.raises(ApiError) as excinfo:
        index.upsert(vectors=[{"id": "bad", "values": [0.1, 0.2]}], namespace=ns)
    assert "dimension" in str(excinfo.value).lower()


@pytest.mark.integration
def test_legacy_upsert_fetch_query_grpc(client: Pinecone, legacy_index_dim3: LegacyIndex) -> None:
    """The same round trip over gRPC.

    Values arrive as float32, so the tolerance is looser than REST's.
    """
    ns = _namespace("grpc-read")
    index: GrpcIndex = client.index(host=legacy_index_dim3.host, grpc=True)
    _seed(index, ns)

    fetched = index.fetch(ids=["lv-a"], namespace=ns)
    assert fetched.vectors["lv-a"].values == pytest.approx(VECTORS["lv-a"], abs=1e-6)
    assert fetched.vectors["lv-a"].metadata == {"genre": "drama", "year": 1994}

    result = index.query(
        vector=VECTORS["lv-a"],
        top_k=3,
        namespace=ns,
        include_values=True,
        include_metadata=True,
    )
    # Keyed by id, never by position: the server returns matches unsorted by
    # score (#368), so asserting an order would encode a server bug as our
    # contract and break when it is fixed. The set and the argmax are checked
    # instead, which is what "the right neighbours, scored right" means.
    by_match = {m.id: m for m in result.matches}
    assert set(by_match) == set(VECTORS)
    assert by_match["lv-a"].score == pytest.approx(1.0, abs=1e-4)
    assert by_match["lv-a"].values == pytest.approx(VECTORS["lv-a"], abs=1e-6)
    assert max(by_match, key=lambda i: by_match[i].score) == "lv-a"

    stats = poll_until(
        query_fn=index.describe_index_stats,
        check_fn=lambda s: (
            s.namespaces.get(ns) is not None and s.namespaces[ns].vector_count == len(VECTORS)
        ),
        timeout=120,
        description="gRPC describe_index_stats counts the seeded namespace",
    )
    assert stats.dimension == 3
    assert stats.namespaces[ns].vector_count == len(VECTORS)

    listed = poll_until(
        query_fn=lambda: {item.id for page in index.list(namespace=ns) for item in page.vectors},
        check_fn=lambda ids: ids == set(VECTORS),
        timeout=120,
        description="all seeded ids visible to gRPC list",
    )
    assert listed == set(VECTORS)

    index.update(id="lv-c", values=[0.05, 0.06, 0.07], namespace=ns)
    poll_until(
        query_fn=lambda: index.fetch(ids=["lv-c"], namespace=ns),
        check_fn=lambda r: r.vectors["lv-c"].values == pytest.approx([0.05, 0.06, 0.07], abs=1e-6),
        timeout=120,
        description="gRPC update visible",
    )

    index.delete(ids=["lv-b"], namespace=ns)
    poll_until(
        query_fn=lambda: index.fetch(ids=["lv-b"], namespace=ns),
        check_fn=lambda r: len(r.vectors) == 0,
        timeout=120,
        description="gRPC delete visible",
    )


@pytest.mark.integration
def test_legacy_wrong_dimension_upsert_rejected_grpc(
    client: Pinecone, legacy_index_dim3: LegacyIndex
) -> None:
    """gRPC twin of the dimension negative, with the same control write."""
    ns = _namespace("grpc-dim")
    index: GrpcIndex = client.index(host=legacy_index_dim3.host, grpc=True)

    control = index.upsert(vectors=[{"id": "ctl", "values": [0.1, 0.2, 0.3]}], namespace=ns)
    assert control.upserted_count == 1

    with pytest.raises(ApiError) as excinfo:
        index.upsert(vectors=[{"id": "bad", "values": [0.1, 0.2]}], namespace=ns)
    assert "dimension" in str(excinfo.value).lower()
