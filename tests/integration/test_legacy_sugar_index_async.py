"""Async REST counterpart to the sync module in this package, covering the same guarantee and index shapes.

Async client, same index shapes as the sync module. See that module for why
this whole module is skipped, why these indexes come from
``client.indexes.create()`` rather than a bypass fixture, and how
``test_legacy_sugar_index_blocker.py`` keeps that skip from rotting silently.
"""

from __future__ import annotations

import uuid
from collections.abc import Generator

import pytest

from pinecone import AsyncPinecone, IndexModel, Pinecone, ServerlessSpec
from tests.integration.conftest import async_poll_until, ensure_index_deleted, unique_name
from tests.integration.legacy_index import (
    LEGACY_DENSE_FIELD,
    LEGACY_SPARSE_FIELD,
    LegacyIndex,
    assert_serves_vectors_api,
)

pytestmark = [
    pytest.mark.skip(
        reason=(
            "Blocked on minicone implementing pinecone-io/pinecone-db#18066: "
            "the local simulator does not yet route an index whose schema is "
            "exactly the reserved _values / _sparse_values field(s) to the "
            "vectors API, so an index built here through Indexes.create()'s "
            "dimension=/metric=/vector_type=/spec= keywords would still land "
            "on a documents-schema index and every assertion below would "
            "fail. Skipped (not xfail) so the reason is specific; see "
            "python-sdk-internal#504. test_legacy_sugar_index_blocker.py "
            "stays unskipped and fails loudly once this gap closes, so this "
            "mark is never the only signal."
        )
    ),
    pytest.mark.anyio,
]

CLOUD = "aws"
REGION = "us-east-1"

DENSE_VECTORS = {
    "sda-a": [0.10, 0.20, 0.30],
    "sda-b": [0.40, 0.50, 0.60],
    "sda-c": [0.70, 0.80, 0.90],
}

SPARSE_VECTORS = {
    "ssa-a": {"indices": [0, 5, 10], "values": [0.5, 0.8, 0.3]},
    "ssa-b": {"indices": [2, 7], "values": [0.3, 0.9]},
}

HYBRID_DENSE_VALUES = {
    "sha-a": [0.10, 0.20, 0.30, 0.40],
    "sha-b": [0.50, 0.60, 0.70, 0.80],
}
HYBRID_SPARSE_VALUES = {
    "sha-a": {"indices": [0, 5], "values": [0.5, 0.8]},
    "sha-b": {"indices": [2, 7], "values": [0.3, 0.9]},
}


def _namespace(label: str) -> str:
    return f"sugar-async-{label}-{uuid.uuid4().hex[:8]}"


def _probe_view(
    model: IndexModel, *, dimension: int | None, metric: str, vector_type: str
) -> LegacyIndex:
    """Carry a created index's name, host, and requested shape for the shared vacuity guard."""
    assert model.host is not None, f"{model.name}: no host on a supposedly-ready index"
    return LegacyIndex(
        name=model.name,
        host=model.host,
        dimension=dimension,
        metric=metric,
        vector_type=vector_type,
    )


@pytest.fixture(scope="module")
def sugar_dense_index_async(client: Pinecone) -> Generator[IndexModel, None, None]:
    """Dense index created via ``dimension=``/``metric=``/``spec=`` keyword sugar."""
    name = unique_name("sugar-dense-a")
    model = client.indexes.create(
        name=name,
        dimension=3,
        metric="cosine",
        spec=ServerlessSpec(cloud=CLOUD, region=REGION),
    )
    try:
        yield model
    finally:
        ensure_index_deleted(client, name)


@pytest.fixture(scope="module")
def sugar_sparse_index_async(client: Pinecone) -> Generator[IndexModel, None, None]:
    """Sparse index created via ``vector_type="sparse"`` keyword sugar, no ``dimension=``."""
    name = unique_name("sugar-sparse-a")
    model = client.indexes.create(
        name=name,
        vector_type="sparse",
        spec=ServerlessSpec(cloud=CLOUD, region=REGION),
    )
    try:
        yield model
    finally:
        ensure_index_deleted(client, name)


@pytest.fixture(scope="module")
def sugar_hybrid_index_async(client: Pinecone) -> Generator[IndexModel, None, None]:
    """A dense ``metric="dotproduct"`` index whose create call names no sparse field at all."""
    name = unique_name("sugar-hybrid-a")
    model = client.indexes.create(
        name=name,
        dimension=4,
        metric="dotproduct",
        spec=ServerlessSpec(cloud=CLOUD, region=REGION),
    )
    try:
        yield model
    finally:
        ensure_index_deleted(client, name)


async def test_sugar_dense_serves_vectors_api_async(
    client: Pinecone, async_client: AsyncPinecone, sugar_dense_index_async: IndexModel
) -> None:
    """Guards the dense round-trip test against passing on an index that refuses every write."""
    assert_serves_vectors_api(
        client,
        _probe_view(sugar_dense_index_async, dimension=3, metric="cosine", vector_type="dense"),
    )
    described = await async_client.indexes.describe(sugar_dense_index_async.name)
    assert described.schema is not None
    assert set(described.schema.fields) == {LEGACY_DENSE_FIELD}
    assert described.schema.fields[LEGACY_DENSE_FIELD].dimension == 3


async def test_sugar_dense_upsert_query_fetch_delete_async(
    async_client: AsyncPinecone, sugar_dense_index_async: IndexModel
) -> None:
    """upsert -> query -> fetch -> delete on the async client, checking real values."""
    ns = _namespace("dense")
    index = await async_client.index(host=sugar_dense_index_async.host)

    upserted = await index.upsert(
        vectors=[
            {"id": "sda-a", "values": DENSE_VECTORS["sda-a"], "metadata": {"genre": "drama"}},
            {"id": "sda-b", "values": DENSE_VECTORS["sda-b"]},
            {"id": "sda-c", "values": DENSE_VECTORS["sda-c"]},
        ],
        namespace=ns,
    )
    assert upserted.upserted_count == 3

    await async_poll_until(
        query_fn=lambda: index.fetch(ids=list(DENSE_VECTORS), namespace=ns),
        check_fn=lambda r: len(r.vectors) == len(DENSE_VECTORS),
        timeout=120,
        description="sugar dense vectors fetchable (async)",
    )

    result = await index.query(
        vector=DENSE_VECTORS["sda-a"],
        top_k=3,
        namespace=ns,
        include_values=True,
        include_metadata=True,
    )
    by_match = {m.id: m for m in result.matches}
    assert set(by_match) == set(DENSE_VECTORS)
    assert by_match["sda-a"].score == pytest.approx(1.0, abs=1e-4)
    assert by_match["sda-a"].values == pytest.approx(DENSE_VECTORS["sda-a"])
    assert by_match["sda-a"].metadata == {"genre": "drama"}
    assert max(by_match, key=lambda i: by_match[i].score) == "sda-a"

    fetched = await index.fetch(ids=["sda-a", "sda-b"], namespace=ns)
    assert set(fetched.vectors) == {"sda-a", "sda-b"}
    assert fetched.vectors["sda-a"].values == pytest.approx(DENSE_VECTORS["sda-a"])
    assert fetched.vectors["sda-a"].metadata == {"genre": "drama"}

    await index.delete(ids=["sda-b"], namespace=ns)
    await async_poll_until(
        query_fn=lambda: index.fetch(ids=["sda-b"], namespace=ns),
        check_fn=lambda r: len(r.vectors) == 0,
        timeout=120,
        description="sugar dense delete visible (async)",
    )
    survivors = await index.fetch(ids=["sda-a", "sda-c"], namespace=ns)
    assert set(survivors.vectors) == {"sda-a", "sda-c"}


async def test_sugar_sparse_serves_vectors_api_async(
    client: Pinecone, async_client: AsyncPinecone, sugar_sparse_index_async: IndexModel
) -> None:
    """Guards the sparse round-trip test against passing on an index that refuses every write."""
    assert_serves_vectors_api(
        client,
        _probe_view(
            sugar_sparse_index_async, dimension=None, metric="dotproduct", vector_type="sparse"
        ),
    )
    described = await async_client.indexes.describe(sugar_sparse_index_async.name)
    assert described.schema is not None
    assert set(described.schema.fields) == {LEGACY_SPARSE_FIELD}
    assert LEGACY_DENSE_FIELD not in described.schema.fields


async def test_sugar_sparse_upsert_query_delete_async(
    async_client: AsyncPinecone, sugar_sparse_index_async: IndexModel
) -> None:
    """Sparse-only upsert -> fetch -> query -> delete on the async client."""
    ns = _namespace("sparse")
    index = await async_client.index(host=sugar_sparse_index_async.host)

    upserted = await index.upsert(
        vectors=[
            {"id": "ssa-a", "sparse_values": SPARSE_VECTORS["ssa-a"]},
            {"id": "ssa-b", "sparse_values": SPARSE_VECTORS["ssa-b"]},
        ],
        namespace=ns,
    )
    assert upserted.upserted_count == 2

    await async_poll_until(
        query_fn=lambda: index.fetch(ids=list(SPARSE_VECTORS), namespace=ns),
        check_fn=lambda r: len(r.vectors) == len(SPARSE_VECTORS),
        timeout=120,
        description="sugar sparse vectors fetchable (async)",
    )

    fetched = await index.fetch(ids=["ssa-a", "ssa-b"], namespace=ns)
    assert set(fetched.vectors) == {"ssa-a", "ssa-b"}
    assert fetched.vectors["ssa-a"].values == []
    assert fetched.vectors["ssa-a"].sparse_values is not None
    assert fetched.vectors["ssa-a"].sparse_values.indices == SPARSE_VECTORS["ssa-a"]["indices"]
    assert fetched.vectors["ssa-a"].sparse_values.values == pytest.approx(
        SPARSE_VECTORS["ssa-a"]["values"]
    )

    result = await index.query(sparse_vector=SPARSE_VECTORS["ssa-a"], top_k=2, namespace=ns)
    assert {m.id for m in result.matches} == {"ssa-a", "ssa-b"}

    await index.delete(ids=["ssa-b"], namespace=ns)
    await async_poll_until(
        query_fn=lambda: index.fetch(ids=["ssa-b"], namespace=ns),
        check_fn=lambda r: len(r.vectors) == 0,
        timeout=120,
        description="sugar sparse delete visible (async)",
    )


async def test_sugar_hybrid_serves_vectors_api_async(
    client: Pinecone, async_client: AsyncPinecone, sugar_hybrid_index_async: IndexModel
) -> None:
    """Guards the hybrid round-trip test and pins the create-time shape: dense field only, no sparse field."""
    assert_serves_vectors_api(
        client,
        _probe_view(
            sugar_hybrid_index_async, dimension=4, metric="dotproduct", vector_type="dense"
        ),
    )
    described = await async_client.indexes.describe(sugar_hybrid_index_async.name)
    assert described.schema is not None
    assert set(described.schema.fields) == {LEGACY_DENSE_FIELD}
    assert described.schema.fields[LEGACY_DENSE_FIELD].metric == "dotproduct"


async def test_sugar_hybrid_upsert_query_delete_async(
    async_client: AsyncPinecone, sugar_hybrid_index_async: IndexModel
) -> None:
    """dense+sparse upsert -> fetch -> query -> delete on the async client.

    The create call declared only a dense field. Accepting ``sparse_values``
    on the upsert anyway, and returning them on fetch and query, is the
    schema-synthesis behavior pinecone-io/pinecone-db#18066 covers.
    """
    ns = _namespace("hybrid")
    index = await async_client.index(host=sugar_hybrid_index_async.host)

    upserted = await index.upsert(
        vectors=[
            {
                "id": "sha-a",
                "values": HYBRID_DENSE_VALUES["sha-a"],
                "sparse_values": HYBRID_SPARSE_VALUES["sha-a"],
            },
            {
                "id": "sha-b",
                "values": HYBRID_DENSE_VALUES["sha-b"],
                "sparse_values": HYBRID_SPARSE_VALUES["sha-b"],
            },
        ],
        namespace=ns,
    )
    assert upserted.upserted_count == 2

    await async_poll_until(
        query_fn=lambda: index.fetch(ids=["sha-a", "sha-b"], namespace=ns),
        check_fn=lambda r: len(r.vectors) == 2,
        timeout=120,
        description="sugar hybrid vectors fetchable (async)",
    )

    fetched = await index.fetch(ids=["sha-a", "sha-b"], namespace=ns)
    assert fetched.vectors["sha-a"].values == pytest.approx(HYBRID_DENSE_VALUES["sha-a"])
    assert fetched.vectors["sha-a"].sparse_values is not None
    assert (
        fetched.vectors["sha-a"].sparse_values.indices == HYBRID_SPARSE_VALUES["sha-a"]["indices"]
    )
    assert fetched.vectors["sha-a"].sparse_values.values == pytest.approx(
        HYBRID_SPARSE_VALUES["sha-a"]["values"]
    )

    result = await index.query(
        vector=HYBRID_DENSE_VALUES["sha-a"],
        sparse_vector=HYBRID_SPARSE_VALUES["sha-a"],
        top_k=2,
        namespace=ns,
    )
    by_match = {m.id: m for m in result.matches}
    assert set(by_match) == {"sha-a", "sha-b"}
    assert max(by_match, key=lambda i: by_match[i].score) == "sha-a"

    described = await async_client.indexes.describe(sugar_hybrid_index_async.name)
    assert described.schema is not None
    assert LEGACY_SPARSE_FIELD in described.schema.fields, (
        f"sparse upsert succeeded but {sugar_hybrid_index_async.name}'s schema "
        f"still has no {LEGACY_SPARSE_FIELD} field: the backend accepted the "
        "write without actually persisting a synthesized sparse field"
    )

    await index.delete(ids=["sha-b"], namespace=ns)
    await async_poll_until(
        query_fn=lambda: index.fetch(ids=["sha-b"], namespace=ns),
        check_fn=lambda r: len(r.vectors) == 0,
        timeout=120,
        description="sugar hybrid delete visible (async)",
    )
