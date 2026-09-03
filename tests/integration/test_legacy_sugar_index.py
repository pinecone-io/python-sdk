"""Proves that Indexes.create()'s dimension=/metric=/vector_type=/spec= keywords build a real vectors-API index.

A schema-based create can succeed while producing an index whose ``query`` /
``fetch`` / ``describe_index_stats`` calls stay callable and return empty —
only writes are refused — so a read-only test cannot tell that shape apart
from a working one. Every test below upserts real vectors first and asserts
``upserted_count`` before reading anything back, and each index shape is
checked once with the shared vacuity-guard helper before its round-trip
tests run.

The indexes here come from ``client.indexes.create()`` itself, because that
call's keyword handling is what needs proving. That is different from the
rest of this package's vector-operation tests, which get their index from a
bespoke ``httpx`` call against an older API version precisely so the SDK
under test is never also the fixture.

The whole module is skipped: the local API simulator this suite runs
against does not yet implement pinecone-io/pinecone-db#18066, the backend
change that routes a schema containing only the reserved ``_values`` /
``_sparse_values`` field(s) to the vectors API. Without it, every index
built here is a documents-schema index and every assertion below fails for
that reason alone. The skip names the gap explicitly instead of marking the
tests ``xfail``, so a run against a backend that has the fix is unambiguous.
``test_legacy_sugar_index_blocker.py`` stays unskipped and fails loudly the
moment that backend exists.

Sync REST and gRPC live here; async REST is in the ``_async`` twin.
"""

from __future__ import annotations

import uuid
from collections.abc import Generator

import pytest

from pinecone import GrpcIndex, IndexModel, Pinecone, ServerlessSpec
from tests.integration.conftest import ensure_index_deleted, poll_until, unique_name
from tests.integration.legacy_index import (
    LEGACY_DENSE_FIELD,
    LEGACY_SPARSE_FIELD,
    LegacyIndex,
    assert_serves_vectors_api,
)

pytestmark = pytest.mark.skip(
    reason=(
        "Blocked on minicone implementing pinecone-io/pinecone-db#18066: the "
        "local simulator does not yet route an index whose schema is exactly "
        "the reserved _values / _sparse_values field(s) to the vectors API, "
        "so an index built here through Indexes.create()'s dimension=/"
        "metric=/vector_type=/spec= keywords would still land on a "
        "documents-schema index and every assertion below would fail. "
        "Skipped (not xfail) so the reason is specific; see "
        "python-sdk-internal#504. test_legacy_sugar_index_blocker.py stays "
        "unskipped and fails loudly once this gap closes, so this mark is "
        "never the only signal."
    )
)

CLOUD = "aws"
REGION = "us-east-1"

DENSE_VECTORS = {
    "sd-a": [0.10, 0.20, 0.30],
    "sd-b": [0.40, 0.50, 0.60],
    "sd-c": [0.70, 0.80, 0.90],
}

SPARSE_VECTORS = {
    "ss-a": {"indices": [0, 5, 10], "values": [0.5, 0.8, 0.3]},
    "ss-b": {"indices": [2, 7], "values": [0.3, 0.9]},
}

HYBRID_DENSE_VALUES = {
    "sh-a": [0.10, 0.20, 0.30, 0.40],
    "sh-b": [0.50, 0.60, 0.70, 0.80],
}
HYBRID_SPARSE_VALUES = {
    "sh-a": {"indices": [0, 5], "values": [0.5, 0.8]},
    "sh-b": {"indices": [2, 7], "values": [0.3, 0.9]},
}


def _namespace(label: str) -> str:
    return f"sugar-{label}-{uuid.uuid4().hex[:8]}"


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
def sugar_dense_index(client: Pinecone) -> Generator[IndexModel, None, None]:
    """Dense index created via ``dimension=``/``metric=``/``spec=`` keyword sugar."""
    name = unique_name("sugar-dense")
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
def sugar_sparse_index(client: Pinecone) -> Generator[IndexModel, None, None]:
    """Sparse index created via ``vector_type="sparse"`` keyword sugar, no ``dimension=``."""
    name = unique_name("sugar-sparse")
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
def sugar_hybrid_index(client: Pinecone) -> Generator[IndexModel, None, None]:
    """A dense ``metric="dotproduct"`` index whose create call names no sparse field at all.

    Its round-trip tests upsert sparse values into it anyway, which only
    succeeds if the backend synthesizes an implicit sparse field for a
    reserved-field ``dotproduct`` index — the behavior pinecone-io/pinecone-db#18066
    covers — rather than refusing the write the way an index with a named
    2026-07 ``dense_vector`` field and no declared ``sparse_vector`` field
    would.
    """
    name = unique_name("sugar-hybrid")
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


@pytest.mark.integration
def test_sugar_dense_serves_vectors_api(client: Pinecone, sugar_dense_index: IndexModel) -> None:
    """Guards the dense round-trip tests against passing on an index that refuses every write."""
    assert_serves_vectors_api(
        client, _probe_view(sugar_dense_index, dimension=3, metric="cosine", vector_type="dense")
    )
    described = client.indexes.describe(sugar_dense_index.name)
    assert described.schema is not None
    assert set(described.schema.fields) == {LEGACY_DENSE_FIELD}
    assert described.schema.fields[LEGACY_DENSE_FIELD].dimension == 3


@pytest.mark.integration
def test_sugar_dense_upsert_query_fetch_delete_rest(
    client: Pinecone, sugar_dense_index: IndexModel
) -> None:
    """upsert -> query -> fetch -> delete over REST, checking real values."""
    ns = _namespace("dense-rest")
    index = client.index(host=sugar_dense_index.host)

    upserted = index.upsert(
        vectors=[
            {"id": "sd-a", "values": DENSE_VECTORS["sd-a"], "metadata": {"genre": "drama"}},
            {"id": "sd-b", "values": DENSE_VECTORS["sd-b"]},
            {"id": "sd-c", "values": DENSE_VECTORS["sd-c"]},
        ],
        namespace=ns,
    )
    assert upserted.upserted_count == 3

    poll_until(
        query_fn=lambda: index.fetch(ids=list(DENSE_VECTORS), namespace=ns),
        check_fn=lambda r: len(r.vectors) == len(DENSE_VECTORS),
        timeout=120,
        description="sugar dense vectors fetchable (rest)",
    )

    result = index.query(
        vector=DENSE_VECTORS["sd-a"],
        top_k=3,
        namespace=ns,
        include_values=True,
        include_metadata=True,
    )
    by_match = {m.id: m for m in result.matches}
    assert set(by_match) == set(DENSE_VECTORS)
    assert by_match["sd-a"].score == pytest.approx(1.0, abs=1e-4)
    assert by_match["sd-a"].values == pytest.approx(DENSE_VECTORS["sd-a"])
    assert by_match["sd-a"].metadata == {"genre": "drama"}
    assert max(by_match, key=lambda i: by_match[i].score) == "sd-a"

    fetched = index.fetch(ids=["sd-a", "sd-b"], namespace=ns)
    assert set(fetched.vectors) == {"sd-a", "sd-b"}
    assert fetched.vectors["sd-a"].values == pytest.approx(DENSE_VECTORS["sd-a"])
    assert fetched.vectors["sd-a"].metadata == {"genre": "drama"}

    index.delete(ids=["sd-b"], namespace=ns)
    poll_until(
        query_fn=lambda: index.fetch(ids=["sd-b"], namespace=ns),
        check_fn=lambda r: len(r.vectors) == 0,
        timeout=120,
        description="sugar dense delete visible (rest)",
    )
    survivors = index.fetch(ids=["sd-a", "sd-c"], namespace=ns)
    assert set(survivors.vectors) == {"sd-a", "sd-c"}


@pytest.mark.integration
def test_sugar_dense_upsert_query_fetch_delete_grpc(
    client: Pinecone, sugar_dense_index: IndexModel
) -> None:
    """Same round trip over gRPC. Values arrive as float32, so tolerance is looser."""
    ns = _namespace("dense-grpc")
    index: GrpcIndex = client.index(host=sugar_dense_index.host, grpc=True)

    upserted = index.upsert(
        vectors=[
            {"id": "sd-a", "values": DENSE_VECTORS["sd-a"], "metadata": {"genre": "drama"}},
            {"id": "sd-b", "values": DENSE_VECTORS["sd-b"]},
            {"id": "sd-c", "values": DENSE_VECTORS["sd-c"]},
        ],
        namespace=ns,
    )
    assert upserted.upserted_count == 3

    poll_until(
        query_fn=lambda: index.fetch(ids=list(DENSE_VECTORS), namespace=ns),
        check_fn=lambda r: len(r.vectors) == len(DENSE_VECTORS),
        timeout=120,
        description="sugar dense vectors fetchable (grpc)",
    )

    result = index.query(
        vector=DENSE_VECTORS["sd-a"],
        top_k=3,
        namespace=ns,
        include_values=True,
        include_metadata=True,
    )
    by_match = {m.id: m for m in result.matches}
    assert set(by_match) == set(DENSE_VECTORS)
    assert by_match["sd-a"].score == pytest.approx(1.0, abs=1e-4)
    assert by_match["sd-a"].values == pytest.approx(DENSE_VECTORS["sd-a"], abs=1e-6)
    assert max(by_match, key=lambda i: by_match[i].score) == "sd-a"

    fetched = index.fetch(ids=["sd-a"], namespace=ns)
    assert fetched.vectors["sd-a"].values == pytest.approx(DENSE_VECTORS["sd-a"], abs=1e-6)
    assert fetched.vectors["sd-a"].metadata == {"genre": "drama"}

    index.delete(ids=["sd-b"], namespace=ns)
    poll_until(
        query_fn=lambda: index.fetch(ids=["sd-b"], namespace=ns),
        check_fn=lambda r: len(r.vectors) == 0,
        timeout=120,
        description="sugar dense delete visible (grpc)",
    )


@pytest.mark.integration
def test_sugar_sparse_serves_vectors_api(client: Pinecone, sugar_sparse_index: IndexModel) -> None:
    """Guards the sparse round-trip tests against passing on an index that refuses every write."""
    assert_serves_vectors_api(
        client,
        _probe_view(sugar_sparse_index, dimension=None, metric="dotproduct", vector_type="sparse"),
    )
    described = client.indexes.describe(sugar_sparse_index.name)
    assert described.schema is not None
    assert set(described.schema.fields) == {LEGACY_SPARSE_FIELD}
    assert LEGACY_DENSE_FIELD not in described.schema.fields


@pytest.mark.integration
def test_sugar_sparse_upsert_query_delete_rest(
    client: Pinecone, sugar_sparse_index: IndexModel
) -> None:
    """Sparse-only upsert -> fetch -> query -> delete over REST."""
    ns = _namespace("sparse-rest")
    index = client.index(host=sugar_sparse_index.host)

    upserted = index.upsert(
        vectors=[
            {"id": "ss-a", "sparse_values": SPARSE_VECTORS["ss-a"]},
            {"id": "ss-b", "sparse_values": SPARSE_VECTORS["ss-b"]},
        ],
        namespace=ns,
    )
    assert upserted.upserted_count == 2

    poll_until(
        query_fn=lambda: index.fetch(ids=list(SPARSE_VECTORS), namespace=ns),
        check_fn=lambda r: len(r.vectors) == len(SPARSE_VECTORS),
        timeout=120,
        description="sugar sparse vectors fetchable (rest)",
    )

    fetched = index.fetch(ids=["ss-a", "ss-b"], namespace=ns)
    assert set(fetched.vectors) == {"ss-a", "ss-b"}
    assert fetched.vectors["ss-a"].values == []
    assert fetched.vectors["ss-a"].sparse_values is not None
    assert fetched.vectors["ss-a"].sparse_values.indices == SPARSE_VECTORS["ss-a"]["indices"]
    assert fetched.vectors["ss-a"].sparse_values.values == pytest.approx(
        SPARSE_VECTORS["ss-a"]["values"]
    )

    result = index.query(sparse_vector=SPARSE_VECTORS["ss-a"], top_k=2, namespace=ns)
    assert {m.id for m in result.matches} == {"ss-a", "ss-b"}

    index.delete(ids=["ss-b"], namespace=ns)
    poll_until(
        query_fn=lambda: index.fetch(ids=["ss-b"], namespace=ns),
        check_fn=lambda r: len(r.vectors) == 0,
        timeout=120,
        description="sugar sparse delete visible (rest)",
    )


@pytest.mark.integration
def test_sugar_sparse_upsert_query_delete_grpc(
    client: Pinecone, sugar_sparse_index: IndexModel
) -> None:
    """Sparse-only round trip over gRPC."""
    ns = _namespace("sparse-grpc")
    index: GrpcIndex = client.index(host=sugar_sparse_index.host, grpc=True)

    upserted = index.upsert(
        vectors=[
            {"id": "ss-a", "sparse_values": SPARSE_VECTORS["ss-a"]},
            {"id": "ss-b", "sparse_values": SPARSE_VECTORS["ss-b"]},
        ],
        namespace=ns,
    )
    assert upserted.upserted_count == 2

    poll_until(
        query_fn=lambda: index.fetch(ids=list(SPARSE_VECTORS), namespace=ns),
        check_fn=lambda r: len(r.vectors) == len(SPARSE_VECTORS),
        timeout=120,
        description="sugar sparse vectors fetchable (grpc)",
    )

    fetched = index.fetch(ids=["ss-a"], namespace=ns)
    assert fetched.vectors["ss-a"].values == []
    assert fetched.vectors["ss-a"].sparse_values is not None
    assert fetched.vectors["ss-a"].sparse_values.indices == SPARSE_VECTORS["ss-a"]["indices"]

    result = index.query(sparse_vector=SPARSE_VECTORS["ss-a"], top_k=2, namespace=ns)
    assert {m.id for m in result.matches} == {"ss-a", "ss-b"}

    index.delete(ids=["ss-b"], namespace=ns)
    poll_until(
        query_fn=lambda: index.fetch(ids=["ss-b"], namespace=ns),
        check_fn=lambda r: len(r.vectors) == 0,
        timeout=120,
        description="sugar sparse delete visible (grpc)",
    )


@pytest.mark.integration
def test_sugar_hybrid_serves_vectors_api(client: Pinecone, sugar_hybrid_index: IndexModel) -> None:
    """Guards the hybrid round-trip tests and pins the create-time shape: dense field only, no sparse field."""
    assert_serves_vectors_api(
        client,
        _probe_view(sugar_hybrid_index, dimension=4, metric="dotproduct", vector_type="dense"),
    )
    described = client.indexes.describe(sugar_hybrid_index.name)
    assert described.schema is not None
    assert set(described.schema.fields) == {LEGACY_DENSE_FIELD}
    assert described.schema.fields[LEGACY_DENSE_FIELD].metric == "dotproduct"


@pytest.mark.integration
def test_sugar_hybrid_upsert_query_delete_rest(
    client: Pinecone, sugar_hybrid_index: IndexModel
) -> None:
    """dense+sparse upsert -> fetch -> query -> delete over REST.

    The create call declared only a dense field. Accepting ``sparse_values``
    on the upsert anyway, and returning them on fetch and query, is the
    schema-synthesis behavior pinecone-io/pinecone-db#18066 covers.
    """
    ns = _namespace("hybrid-rest")
    index = client.index(host=sugar_hybrid_index.host)

    upserted = index.upsert(
        vectors=[
            {
                "id": "sh-a",
                "values": HYBRID_DENSE_VALUES["sh-a"],
                "sparse_values": HYBRID_SPARSE_VALUES["sh-a"],
            },
            {
                "id": "sh-b",
                "values": HYBRID_DENSE_VALUES["sh-b"],
                "sparse_values": HYBRID_SPARSE_VALUES["sh-b"],
            },
        ],
        namespace=ns,
    )
    assert upserted.upserted_count == 2

    poll_until(
        query_fn=lambda: index.fetch(ids=["sh-a", "sh-b"], namespace=ns),
        check_fn=lambda r: len(r.vectors) == 2,
        timeout=120,
        description="sugar hybrid vectors fetchable (rest)",
    )

    fetched = index.fetch(ids=["sh-a", "sh-b"], namespace=ns)
    assert fetched.vectors["sh-a"].values == pytest.approx(HYBRID_DENSE_VALUES["sh-a"])
    assert fetched.vectors["sh-a"].sparse_values is not None
    assert fetched.vectors["sh-a"].sparse_values.indices == HYBRID_SPARSE_VALUES["sh-a"]["indices"]
    assert fetched.vectors["sh-a"].sparse_values.values == pytest.approx(
        HYBRID_SPARSE_VALUES["sh-a"]["values"]
    )

    result = index.query(
        vector=HYBRID_DENSE_VALUES["sh-a"],
        sparse_vector=HYBRID_SPARSE_VALUES["sh-a"],
        top_k=2,
        namespace=ns,
    )
    by_match = {m.id: m for m in result.matches}
    assert set(by_match) == {"sh-a", "sh-b"}
    assert max(by_match, key=lambda i: by_match[i].score) == "sh-a"

    described = client.indexes.describe(sugar_hybrid_index.name)
    assert described.schema is not None
    assert LEGACY_SPARSE_FIELD in described.schema.fields, (
        f"sparse upsert succeeded but {sugar_hybrid_index.name}'s schema still "
        f"has no {LEGACY_SPARSE_FIELD} field: the backend accepted the write "
        "without actually persisting a synthesized sparse field"
    )

    index.delete(ids=["sh-b"], namespace=ns)
    poll_until(
        query_fn=lambda: index.fetch(ids=["sh-b"], namespace=ns),
        check_fn=lambda r: len(r.vectors) == 0,
        timeout=120,
        description="sugar hybrid delete visible (rest)",
    )


@pytest.mark.integration
def test_sugar_hybrid_upsert_query_delete_grpc(
    client: Pinecone, sugar_hybrid_index: IndexModel
) -> None:
    """Hybrid round trip over gRPC."""
    ns = _namespace("hybrid-grpc")
    index: GrpcIndex = client.index(host=sugar_hybrid_index.host, grpc=True)

    upserted = index.upsert(
        vectors=[
            {
                "id": "sh-a",
                "values": HYBRID_DENSE_VALUES["sh-a"],
                "sparse_values": HYBRID_SPARSE_VALUES["sh-a"],
            },
            {
                "id": "sh-b",
                "values": HYBRID_DENSE_VALUES["sh-b"],
                "sparse_values": HYBRID_SPARSE_VALUES["sh-b"],
            },
        ],
        namespace=ns,
    )
    assert upserted.upserted_count == 2

    poll_until(
        query_fn=lambda: index.fetch(ids=["sh-a", "sh-b"], namespace=ns),
        check_fn=lambda r: len(r.vectors) == 2,
        timeout=120,
        description="sugar hybrid vectors fetchable (grpc)",
    )

    fetched = index.fetch(ids=["sh-a"], namespace=ns)
    assert fetched.vectors["sh-a"].values == pytest.approx(HYBRID_DENSE_VALUES["sh-a"], abs=1e-6)
    assert fetched.vectors["sh-a"].sparse_values is not None
    assert fetched.vectors["sh-a"].sparse_values.indices == HYBRID_SPARSE_VALUES["sh-a"]["indices"]

    result = index.query(
        vector=HYBRID_DENSE_VALUES["sh-a"],
        sparse_vector=HYBRID_SPARSE_VALUES["sh-a"],
        top_k=2,
        namespace=ns,
    )
    assert {m.id for m in result.matches} == {"sh-a", "sh-b"}

    described = client.indexes.describe(sugar_hybrid_index.name)
    assert described.schema is not None
    assert LEGACY_SPARSE_FIELD in described.schema.fields, (
        f"sparse upsert succeeded but {sugar_hybrid_index.name}'s schema still "
        f"has no {LEGACY_SPARSE_FIELD} field: the backend accepted the write "
        "without actually persisting a synthesized sparse field"
    )

    index.delete(ids=["sh-b"], namespace=ns)
    poll_until(
        query_fn=lambda: index.fetch(ids=["sh-b"], namespace=ns),
        check_fn=lambda r: len(r.vectors) == 0,
        timeout=120,
        description="sugar hybrid delete visible (grpc)",
    )
