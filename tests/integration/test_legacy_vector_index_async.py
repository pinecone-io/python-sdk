"""Async twin of ``test_legacy_vector_index.py`` (#363).

Same guarantee, same index shape, async REST client. See the sync module for
why the index cannot come from ``pc.indexes.create``.
"""

from __future__ import annotations

import uuid

import pytest

from pinecone import AsyncPinecone
from pinecone.errors import ApiError
from tests.integration.conftest import async_poll_until
from tests.integration.legacy_index import LegacyIndex

pytestmark = pytest.mark.anyio

VECTORS = {
    "lva-a": [0.10, 0.20, 0.30],
    "lva-b": [0.40, 0.50, 0.60],
    "lva-c": [0.70, 0.80, 0.90],
}


def _namespace(label: str) -> str:
    return f"legacy-async-{label}-{uuid.uuid4().hex[:8]}"


async def _seed(index: object, namespace: str) -> None:
    """Upsert :data:`VECTORS` and wait until all three are fetchable."""
    result = await index.upsert(  # type: ignore[attr-defined]
        vectors=[
            {
                "id": "lva-a",
                "values": VECTORS["lva-a"],
                "metadata": {"genre": "drama", "year": 1994},
            },
            {"id": "lva-b", "values": VECTORS["lva-b"]},
            {"id": "lva-c", "values": VECTORS["lva-c"]},
        ],
        namespace=namespace,
    )
    assert result.upserted_count == 3

    await async_poll_until(
        query_fn=lambda: index.fetch(ids=list(VECTORS), namespace=namespace),  # type: ignore[attr-defined]
        check_fn=lambda r: len(r.vectors) == len(VECTORS),
        timeout=120,
        description="seeded legacy-index vectors fetchable (async)",
    )


@pytest.mark.integration
async def test_legacy_index_describe_async(
    async_client: AsyncPinecone, legacy_index_dim3: LegacyIndex
) -> None:
    """The async client can describe an index it did not create.

    Also the vacuity guard for this module: a schema declaring anything
    beyond the reserved legacy vector fields means the index is served by the
    documents API, and every assertion below would be measuring a blanket
    refusal instead of the vectors API.
    """
    described = await async_client.indexes.describe(legacy_index_dim3.name)
    assert described.schema is not None
    assert set(described.schema.fields) <= {"_values", "_sparse_values"}
    assert described.schema.fields["_values"].dimension == 3


@pytest.mark.integration
async def test_legacy_upsert_fetch_query_async(
    async_client: AsyncPinecone, legacy_index_dim3: LegacyIndex
) -> None:
    """upsert -> fetch -> query on the async client, checking values and metadata."""
    ns = _namespace("read")
    index = await async_client.index(host=legacy_index_dim3.host)
    await _seed(index, ns)

    fetched = await index.fetch(ids=["lva-a", "lva-b"], namespace=ns)
    assert set(fetched.vectors) == {"lva-a", "lva-b"}
    assert fetched.vectors["lva-a"].values == pytest.approx(VECTORS["lva-a"])
    assert fetched.vectors["lva-a"].metadata == {"genre": "drama", "year": 1994}

    result = await index.query(
        vector=VECTORS["lva-a"],
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
    assert by_match["lva-a"].score == pytest.approx(1.0, abs=1e-4)
    assert by_match["lva-a"].values == pytest.approx(VECTORS["lva-a"])
    assert by_match["lva-a"].metadata == {"genre": "drama", "year": 1994}
    assert max(by_match, key=lambda i: by_match[i].score) == "lva-a"


@pytest.mark.integration
async def test_legacy_update_list_stats_delete_async(
    async_client: AsyncPinecone, legacy_index_dim3: LegacyIndex
) -> None:
    """update -> list -> describe_index_stats -> delete on the async client."""
    ns = _namespace("write")
    index = await async_client.index(host=legacy_index_dim3.host)
    await _seed(index, ns)

    replacement = [0.01, 0.02, 0.03]
    await index.update(id="lva-a", values=replacement, namespace=ns)
    await async_poll_until(
        query_fn=lambda: index.fetch(ids=["lva-a"], namespace=ns),
        check_fn=lambda r: r.vectors["lva-a"].values == pytest.approx(replacement),
        timeout=120,
        description="updated values visible (async)",
    )

    async def _listed_ids() -> set[str]:
        return {item.id async for page in index.list(namespace=ns) for item in page.vectors}

    listed = await async_poll_until(
        query_fn=_listed_ids,
        check_fn=lambda ids: ids == set(VECTORS),
        timeout=120,
        description="all seeded ids visible to list (async)",
    )
    assert listed == set(VECTORS)

    paginated = await index.list_paginated(namespace=ns)
    assert paginated.vectors, "list_paginated returned an empty page"
    assert {item.id for item in paginated.vectors} <= set(VECTORS)

    stats = await async_poll_until(
        query_fn=index.describe_index_stats,
        check_fn=lambda s: (
            s.namespaces.get(ns) is not None and s.namespaces[ns].vector_count == len(VECTORS)
        ),
        timeout=120,
        description="describe_index_stats counts the seeded namespace (async)",
    )
    assert stats.dimension == 3
    assert stats.namespaces[ns].vector_count == len(VECTORS)

    await index.delete(ids=["lva-b"], namespace=ns)
    await async_poll_until(
        query_fn=lambda: index.fetch(ids=["lva-b"], namespace=ns),
        check_fn=lambda r: len(r.vectors) == 0,
        timeout=120,
        description="deleted vector gone (async)",
    )
    survivors = await index.fetch(ids=["lva-a", "lva-c"], namespace=ns)
    assert set(survivors.vectors) == {"lva-a", "lva-c"}


@pytest.mark.integration
async def test_legacy_wrong_dimension_upsert_rejected_async(
    async_client: AsyncPinecone, legacy_index_dim3: LegacyIndex
) -> None:
    """Async twin of the dimension negative, with the same control write."""
    ns = _namespace("dim")
    index = await async_client.index(host=legacy_index_dim3.host)

    control = await index.upsert(vectors=[{"id": "ctl", "values": [0.1, 0.2, 0.3]}], namespace=ns)
    assert control.upserted_count == 1

    with pytest.raises(ApiError) as excinfo:
        await index.upsert(vectors=[{"id": "bad", "values": [0.1, 0.2]}], namespace=ns)
    assert "dimension" in str(excinfo.value).lower()
