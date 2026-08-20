"""Integration tests for preview document upsert + search flows (2026-01.alpha).

Covers §13 end-to-end examples from spec/preview.md:
- Full-text search only (SchemaBuilder + batch_upsert + PreviewTextQuery)
- Hybrid: dense vector + full-text search (upsert + PreviewDenseVectorQuery + PreviewTextQuery)
- Boolean query string with filter (PreviewQueryStringQuery + filter dict)
- Sparse vector search (PreviewSparseVectorQuery + PreviewSparseValues)

These tests make real API calls and skip gracefully when the preview endpoint
is unavailable. They do NOT gate CI (preview_integration marker).

Eventual consistency: after upsert, search readiness is eventually consistent.
Each test polls via _wait_for_searchable() with up to 60 s timeout.
"""

from __future__ import annotations

import pytest

from pinecone import Pinecone
from pinecone.preview import SchemaBuilder as PreviewSchemaBuilder
from pinecone.preview.models import (
    PreviewDenseVectorQuery,
    PreviewDocument,
    PreviewDocumentSearchResponse,
    PreviewQueryStringQuery,
    PreviewSparseValues,
    PreviewSparseVectorQuery,
    PreviewTextQuery,
    PreviewUsage,
)
from tests.integration.conftest import poll_until

pytestmark = [pytest.mark.integration, pytest.mark.preview_integration]


# ---------------------------------------------------------------------------
# Local helper
# ---------------------------------------------------------------------------


def _wait_for_searchable(
    idx: object,
    namespace: str,
    expected_ids: set[str],
    score_by: list[object],
    timeout: int = 60,
) -> PreviewDocumentSearchResponse:
    """Poll search until all expected_ids appear in matches. Returns final response."""
    from pinecone.preview.index import PreviewIndex

    assert isinstance(idx, PreviewIndex)

    def _query() -> PreviewDocumentSearchResponse:
        return idx.documents.search(
            namespace=namespace,
            top_k=len(expected_ids),
            score_by=score_by,  # type: ignore[arg-type]
        )

    def _check(r: PreviewDocumentSearchResponse) -> bool:
        found = {d._id for d in r.matches}
        return expected_ids.issubset(found)

    result = poll_until(
        _query,
        _check,
        timeout=timeout,
        interval=3,
        description=f"docs {expected_ids} searchable in namespace {namespace!r}",
    )
    assert isinstance(result, PreviewDocumentSearchResponse)
    return result


# ---------------------------------------------------------------------------
# test_full_text_search_only_flow — §13 "Full-text search only"
# ---------------------------------------------------------------------------


def test_full_text_search_only_flow(
    client: Pinecone,
    preview_index_name: str,
    cleanup_preview_indexes: list[str],
    preview_namespace: str,
    require_preview: None,
) -> None:
    """FTS-only schema: batch_upsert 3 docs, poll until searchable, assert top result."""
    from pinecone.preview.models import PreviewIndexModel

    schema = (
        PreviewSchemaBuilder().add_string_field("text", full_text_search={"language": "en"}).build()
    )
    cleanup_preview_indexes.append(preview_index_name)
    client.preview.indexes.create(name=preview_index_name, schema=schema)

    def _is_ready(m: object) -> bool:
        return isinstance(m, PreviewIndexModel) and m.status.state == "Ready"

    poll_until(
        lambda: client.preview.indexes.describe(preview_index_name),
        _is_ready,
        timeout=300,
        interval=5,
        description=f"index {preview_index_name} ready",
    )

    texts = ["ancient Rome", "medieval Europe", "modern tech"]
    documents = [{"_id": f"doc-{i}", "text": text} for i, text in enumerate(texts)]

    idx = client.preview.index(name=preview_index_name)
    idx.documents.batch_upsert(
        namespace=preview_namespace,
        documents=documents,
        batch_size=2,
        max_concurrency=2,
    )

    score_by: list[object] = [PreviewTextQuery(fields=["text"], query="ancient")]
    results = _wait_for_searchable(idx, preview_namespace, {"doc-0", "doc-1", "doc-2"}, score_by)

    assert len(results.matches) >= 1
    # doc-0 ("ancient Rome") should rank highest for the query "ancient"
    top_match = max(results.matches, key=lambda d: d.score or 0.0)
    assert top_match._id == "doc-0"

    for doc in results.matches:
        assert isinstance(doc, PreviewDocument)
        assert isinstance(doc._id, str)
        assert doc.score is not None
        assert isinstance(doc.score, float)


# ---------------------------------------------------------------------------
# test_hybrid_search_combines_dense_and_text — §13 "Hybrid"
# ---------------------------------------------------------------------------


def test_hybrid_search_combines_dense_and_text(
    client: Pinecone,
    preview_index_name: str,
    cleanup_preview_indexes: list[str],
    preview_namespace: str,
    require_preview: None,
) -> None:
    """Hybrid schema: single upsert, search with dense + text queries, assert doc-1 ranks first."""
    from pinecone.preview.models import PreviewIndexModel

    schema = (
        PreviewSchemaBuilder()
        .add_dense_vector_field("embedding", dimension=4, metric="cosine")
        .add_string_field("chunk", full_text_search={})
        .build()
    )
    cleanup_preview_indexes.append(preview_index_name)
    client.preview.indexes.create(name=preview_index_name, schema=schema)

    def _is_ready(m: object) -> bool:
        return isinstance(m, PreviewIndexModel) and m.status.state == "Ready"

    poll_until(
        lambda: client.preview.indexes.describe(preview_index_name),
        _is_ready,
        timeout=300,
        interval=5,
        description=f"index {preview_index_name} ready",
    )

    idx = client.preview.index(name=preview_index_name)
    idx.documents.upsert(
        namespace=preview_namespace,
        documents=[{"_id": "doc-1", "embedding": [0.1, 0.2, 0.3, 0.4], "chunk": "hello world"}],
    )

    score_by: list[object] = [
        PreviewDenseVectorQuery(field="embedding", values=[0.1, 0.2, 0.3, 0.4]),
        PreviewTextQuery(fields=["chunk"], query="hello"),
    ]
    results = _wait_for_searchable(idx, preview_namespace, {"doc-1"}, score_by)

    assert len(results.matches) >= 1
    assert results.matches[0]._id == "doc-1"


# ---------------------------------------------------------------------------
# test_boolean_query_string_with_filter — §13 "Boolean query string with filter"
# ---------------------------------------------------------------------------


def test_boolean_query_string_with_filter(
    client: Pinecone,
    preview_index_name: str,
    cleanup_preview_indexes: list[str],
    preview_namespace: str,
    require_preview: None,
) -> None:
    """Query string with filter: only sports-category docs matching football (not american)."""
    from pinecone.preview.models import PreviewIndexModel

    schema = (
        PreviewSchemaBuilder()
        .add_string_field("chunk", full_text_search={})
        .add_string_field("category", filterable=True)
        .build()
    )
    cleanup_preview_indexes.append(preview_index_name)
    client.preview.indexes.create(name=preview_index_name, schema=schema)

    def _is_ready(m: object) -> bool:
        return isinstance(m, PreviewIndexModel) and m.status.state == "Ready"

    poll_until(
        lambda: client.preview.indexes.describe(preview_index_name),
        _is_ready,
        timeout=300,
        interval=5,
        description=f"index {preview_index_name} ready",
    )

    idx = client.preview.index(name=preview_index_name)
    idx.documents.upsert(
        namespace=preview_namespace,
        documents=[
            {"_id": "sports-1", "chunk": "football match premier league", "category": "sports"},
            {"_id": "news-1", "chunk": "american football NFL draft", "category": "news"},
            {"_id": "sports-2", "chunk": "football world cup Brazil", "category": "sports"},
        ],
    )

    score_by: list[object] = [PreviewQueryStringQuery(query="chunk:(+football NOT american)")]

    # Poll until at least one sports doc is found (filter applied server-side)
    def _query_with_filter() -> PreviewDocumentSearchResponse:
        return idx.documents.search(
            namespace=preview_namespace,
            top_k=10,
            score_by=score_by,  # type: ignore[arg-type]
            filter={"category": {"$eq": "sports"}},
            include_fields=["chunk", "category"],
        )

    def _has_results(r: PreviewDocumentSearchResponse) -> bool:
        return len(r.matches) > 0

    results = poll_until(
        _query_with_filter,
        _has_results,
        timeout=60,
        interval=3,
        description="filtered sports docs searchable",
    )
    assert isinstance(results, PreviewDocumentSearchResponse)

    for doc in results.matches:
        assert isinstance(doc, PreviewDocument)
        assert doc.get("category") == "sports", f"expected sports, got {doc.get('category')}"
        # Verify include_fields populated both attributes
        assert doc.get("chunk") is not None
        assert doc.get("category") is not None

    match_ids = {d._id for d in results.matches}
    assert "news-1" not in match_ids, "news-1 (american football) should be excluded by filter"


# ---------------------------------------------------------------------------
# test_sparse_vector_search — §13 "Sparse vector search"
# ---------------------------------------------------------------------------


def test_sparse_vector_search(
    client: Pinecone,
    preview_index_name: str,
    cleanup_preview_indexes: list[str],
    preview_namespace: str,
    require_preview: None,
) -> None:
    """Sparse vector schema: upsert 2 docs, search by sparse similarity, assert correct ranking."""
    from pinecone.preview.models import PreviewIndexModel

    schema = (
        PreviewSchemaBuilder()
        .add_sparse_vector_field("sparse_embedding")
        .add_string_field("title", full_text_search={"language": "en"})
        .build()
    )
    cleanup_preview_indexes.append(preview_index_name)
    client.preview.indexes.create(name=preview_index_name, schema=schema)

    def _is_ready(m: object) -> bool:
        return isinstance(m, PreviewIndexModel) and m.status.state == "Ready"

    poll_until(
        lambda: client.preview.indexes.describe(preview_index_name),
        _is_ready,
        timeout=300,
        interval=5,
        description=f"index {preview_index_name} ready",
    )

    idx = client.preview.index(name=preview_index_name)
    idx.documents.upsert(
        namespace=preview_namespace,
        documents=[
            {
                "_id": "doc-1",
                "sparse_embedding": {"indices": [0, 42], "values": [0.8, 0.3]},
                "title": "alpha",
            },
            {
                "_id": "doc-2",
                "sparse_embedding": {"indices": [1, 99], "values": [0.1, 0.1]},
                "title": "beta",
            },
        ],
    )

    score_by: list[object] = [
        PreviewSparseVectorQuery(
            field="sparse_embedding",
            sparse_values=PreviewSparseValues(indices=[0, 42], values=[0.8, 0.3]),
        )
    ]
    results = _wait_for_searchable(idx, preview_namespace, {"doc-1", "doc-2"}, score_by)

    assert len(results.matches) >= 1
    # doc-1 shares the exact same indices/values as the query → should rank first
    assert results.matches[0]._id == "doc-1"


# ---------------------------------------------------------------------------
# test_search_include_fields_variants — §5 search include_fields behavior, §7 PreviewDocument
# ---------------------------------------------------------------------------


def test_search_include_fields_variants(
    client: Pinecone,
    preview_index_name: str,
    cleanup_preview_indexes: list[str],
    preview_namespace: str,
    require_preview: None,
) -> None:
    """Verify include_fields request construction: explicit list, ["*"], and None (→ []) accepted.

    This test validates the SDK's request-level behavior for include_fields.
    It does NOT assert on returned document fields because search indexing is
    eventually consistent and the test avoids polling to keep it fast.
    """
    from pinecone.preview.models import PreviewIndexModel

    schema = (
        PreviewSchemaBuilder()
        .add_dense_vector_field("embedding", dimension=4, metric="cosine")
        .add_string_field("title", filterable=True)
        .add_string_field("category", filterable=True)
        .build()
    )
    cleanup_preview_indexes.append(preview_index_name)
    client.preview.indexes.create(name=preview_index_name, schema=schema)

    def _is_ready(m: object) -> bool:
        return isinstance(m, PreviewIndexModel) and m.status.state == "Ready"

    poll_until(
        lambda: client.preview.indexes.describe(preview_index_name),
        _is_ready,
        timeout=300,
        interval=5,
        description=f"index {preview_index_name} ready",
    )

    idx = client.preview.index(name=preview_index_name)
    idx.documents.upsert(
        namespace=preview_namespace,
        documents=[
            {
                "_id": "doc-1",
                "embedding": [0.1, 0.2, 0.3, 0.4],
                "title": "ancient Rome",
                "category": "history",
            }
        ],
    )

    query_vec = [0.1, 0.2, 0.3, 0.4]
    score_by: list[object] = [PreviewDenseVectorQuery(field="embedding", values=query_vec)]

    # Case 1: include_fields=["*"] — SDK sends field; API accepts (200 OK).
    results_star = idx.documents.search(
        namespace=preview_namespace,
        top_k=5,
        score_by=score_by,  # type: ignore[arg-type]
        include_fields=["*"],
    )
    assert isinstance(results_star, PreviewDocumentSearchResponse)

    # Case 2: include_fields=["title"] — SDK sends field; API accepts (200 OK).
    results_named = idx.documents.search(
        namespace=preview_namespace,
        top_k=5,
        score_by=score_by,  # type: ignore[arg-type]
        include_fields=["title"],
    )
    assert isinstance(results_named, PreviewDocumentSearchResponse)

    # Case 3: include_fields=None (default) — SDK sends include_fields=[] → 200 OK.
    # Per spec, None returns only _id and score (empty list is the wire equivalent).
    results_default = idx.documents.search(
        namespace=preview_namespace,
        top_k=5,
        score_by=score_by,  # type: ignore[arg-type]
    )
    assert isinstance(results_default, PreviewDocumentSearchResponse)


# ---------------------------------------------------------------------------
# test_search_response_namespace_and_usage — §7 PreviewDocumentSearchResponse envelope
# ---------------------------------------------------------------------------


def test_search_response_namespace_and_usage(
    client: Pinecone,
    preview_index_name: str,
    cleanup_preview_indexes: list[str],
    preview_namespace: str,
    require_preview: None,
) -> None:
    """Verify PreviewDocumentSearchResponse.namespace and .usage fields (§7).

    The spec declares that search() returns a PreviewDocumentSearchResponse with:
    - namespace: str  — echoed back from the request parameter
    - usage: PreviewUsage | None — with read_units: int >= 0

    No existing test checks these envelope fields; all existing tests inspect
    only response.matches items. This test targets the response envelope directly.
    Uses include_fields=["*"] to return all stored fields.
    0 matches are acceptable — the namespace and usage fields are always present.
    """
    from pinecone.preview.models import PreviewIndexModel

    schema = (
        PreviewSchemaBuilder()
        .add_dense_vector_field("embedding", dimension=4, metric="cosine")
        .build()
    )
    cleanup_preview_indexes.append(preview_index_name)
    client.preview.indexes.create(name=preview_index_name, schema=schema)

    def _is_ready(m: object) -> bool:
        return isinstance(m, PreviewIndexModel) and m.status.state == "Ready"

    poll_until(
        lambda: client.preview.indexes.describe(preview_index_name),
        _is_ready,
        timeout=300,
        interval=5,
        description=f"index {preview_index_name} ready",
    )

    idx = client.preview.index(name=preview_index_name)
    idx.documents.upsert(
        namespace=preview_namespace,
        documents=[{"_id": "doc-env", "embedding": [0.1, 0.2, 0.3, 0.4]}],
    )

    score_by: list[object] = [
        PreviewDenseVectorQuery(field="embedding", values=[0.1, 0.2, 0.3, 0.4])
    ]
    response = idx.documents.search(
        namespace=preview_namespace,
        top_k=5,
        score_by=score_by,  # type: ignore[arg-type]
        include_fields=["*"],
    )

    assert isinstance(response, PreviewDocumentSearchResponse)
    # §7: namespace must be echoed back from the request.
    assert response.namespace == preview_namespace, (
        f"response.namespace {response.namespace!r} != request namespace {preview_namespace!r}"
    )
    # §7: usage must be present with a non-negative read_units counter.
    assert response.usage is not None, "response.usage must not be None after a successful search"
    assert isinstance(response.usage, PreviewUsage), (
        f"expected PreviewUsage, got {type(response.usage)}"
    )
    assert isinstance(response.usage.read_units, int), (
        f"read_units must be int, got {type(response.usage.read_units)}"
    )
    assert response.usage.read_units >= 0, (
        f"read_units must be >= 0, got {response.usage.read_units}"
    )


# ---------------------------------------------------------------------------
# test_filter_integer_gte_and_operator_accepted — §8 Metadata filtering
# ---------------------------------------------------------------------------


def test_filter_integer_gte_and_operator_accepted(
    client: Pinecone,
    preview_index_name: str,
    cleanup_preview_indexes: list[str],
    preview_namespace: str,
    require_preview: None,
) -> None:
    """Verify §8: filter with integer $gte and $and operator is serialized and accepted (200 OK).

    Spec §8 declares these operators: $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin, $and, $or.
    No existing test verifies that the SDK serializes $gte on an integer field or
    that the $and logical operator is accepted by the API.

    Creates a dense vector schema with filterable integer 'year' and filterable
    string 'category'. Upserts a document. Searches with a filter combining $and
    with $gte on year and $eq on category. Asserts 200 OK regardless of matches
    (OnDemand dense vector indexes are eventually consistent; 0 matches is acceptable).
    """
    from pinecone.preview.models import PreviewIndexModel

    schema = (
        PreviewSchemaBuilder()
        .add_dense_vector_field("embedding", dimension=4, metric="cosine")
        .add_string_field("category", filterable=True)
        .add_float_field("year", filterable=True)
        .build()
    )
    cleanup_preview_indexes.append(preview_index_name)
    client.preview.indexes.create(name=preview_index_name, schema=schema)

    def _is_ready(m: object) -> bool:
        return isinstance(m, PreviewIndexModel) and m.status.state == "Ready"

    poll_until(
        lambda: client.preview.indexes.describe(preview_index_name),
        _is_ready,
        timeout=300,
        interval=5,
        description=f"index {preview_index_name} ready",
    )

    idx = client.preview.index(name=preview_index_name)
    idx.documents.upsert(
        namespace=preview_namespace,
        documents=[
            {"_id": "doc-1", "embedding": [0.1, 0.2, 0.3, 0.4], "category": "tech", "year": 2022},
            {
                "_id": "doc-2",
                "embedding": [0.5, 0.6, 0.7, 0.8],
                "category": "science",
                "year": 2018,
            },
        ],
    )

    score_by: list[object] = [
        PreviewDenseVectorQuery(field="embedding", values=[0.1, 0.2, 0.3, 0.4])
    ]

    # Verify $gte filter on integer field is accepted (200 OK).
    result_gte = idx.documents.search(
        namespace=preview_namespace,
        top_k=5,
        score_by=score_by,  # type: ignore[arg-type]
        filter={"year": {"$gte": 2020}},
        include_fields=["*"],
    )
    assert isinstance(result_gte, PreviewDocumentSearchResponse), (
        "$gte filter on integer field should return 200 OK"
    )

    # Verify $and operator combining $gte + $eq is accepted (200 OK).
    result_and = idx.documents.search(
        namespace=preview_namespace,
        top_k=5,
        score_by=score_by,  # type: ignore[arg-type]
        filter={
            "$and": [
                {"category": {"$eq": "tech"}},
                {"year": {"$gte": 2020}},
            ]
        },
        include_fields=["*"],
    )
    assert isinstance(result_and, PreviewDocumentSearchResponse), (
        "$and filter with $gte + $eq should return 200 OK"
    )


# ---------------------------------------------------------------------------
# test_search_client_side_validation — §7 search() parameter validation
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_search_client_side_validation_rejects_invalid_parameters(
    client: Pinecone,
    require_preview: None,
) -> None:
    """search() raises ValidationError for empty namespace, out-of-range top_k, and empty score_by.

    Spec §7 declares client-side validation:
    - namespace must be a non-empty string
    - top_k must be between 1 and 10000
    - score_by must be a non-empty list

    No API call is made — validation fires synchronously before the HTTP request is sent.
    Uses a dummy host to obtain a PreviewIndex without a real index.
    """
    from pinecone.errors.exceptions import ValidationError

    # Use a dummy host — no HTTP call happens because validation fires first.
    idx = client.preview.index(host="https://dummy-host.pinecone.io")

    valid_score_by: list[object] = [{"type": "dense_vector", "field": "emb", "values": [0.1]}]

    # Empty namespace string must raise ValidationError.
    with pytest.raises(ValidationError, match="namespace"):
        idx.documents.search(namespace="", top_k=5, score_by=valid_score_by)  # type: ignore[arg-type]

    # top_k=0 is below the minimum of 1.
    with pytest.raises(ValidationError, match="top_k"):
        idx.documents.search(namespace="ns", top_k=0, score_by=valid_score_by)  # type: ignore[arg-type]

    # top_k=10001 is above the maximum of 10000.
    with pytest.raises(ValidationError, match="top_k"):
        idx.documents.search(namespace="ns", top_k=10001, score_by=valid_score_by)  # type: ignore[arg-type]

    # Empty score_by list must raise ValidationError.
    with pytest.raises(ValidationError, match="score_by"):
        idx.documents.search(namespace="ns", top_k=5, score_by=[])


# ---------------------------------------------------------------------------
# test_filter_remaining_operators_accepted — §8 Metadata filtering ($ne, $gt, $lt, $lte, $in, $nin, $or)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_filter_remaining_operators_accepted(
    client: Pinecone,
    preview_index_name: str,
    cleanup_preview_indexes: list[str],
    preview_namespace: str,
    require_preview: None,
) -> None:
    """Verify §8: remaining filter operators ($ne, $gt, $lt, $lte, $in, $nin, $or) are each accepted.

    PVT-013 integration-tested $gte and $and. Spec §8 declares 10 operators total;
    this test covers the 7 not yet verified in integration: $ne, $gt, $lt, $lte,
    $in, $nin, and $or.

    Creates a dense vector schema with filterable string 'category' and filterable
    integer 'year'. Upserts two documents. Searches with each operator independently
    and asserts 200 OK (OnDemand indexing is eventually consistent; 0 matches is
    acceptable — the goal is to confirm serialization and API acceptance).
    """
    from pinecone.preview.models import PreviewIndexModel

    schema = (
        PreviewSchemaBuilder()
        .add_dense_vector_field("embedding", dimension=4, metric="cosine")
        .add_string_field("category", filterable=True)
        .add_float_field("year", filterable=True)
        .build()
    )
    cleanup_preview_indexes.append(preview_index_name)
    client.preview.indexes.create(name=preview_index_name, schema=schema)

    def _is_ready(m: object) -> bool:
        return isinstance(m, PreviewIndexModel) and m.status.state == "Ready"

    poll_until(
        lambda: client.preview.indexes.describe(preview_index_name),
        _is_ready,
        timeout=300,
        interval=5,
        description=f"index {preview_index_name} ready",
    )

    idx = client.preview.index(name=preview_index_name)
    idx.documents.upsert(
        namespace=preview_namespace,
        documents=[
            {"_id": "doc-1", "embedding": [0.1, 0.2, 0.3, 0.4], "category": "tech", "year": 2022},
            {
                "_id": "doc-2",
                "embedding": [0.5, 0.6, 0.7, 0.8],
                "category": "science",
                "year": 2018,
            },
        ],
    )

    score_by: list[object] = [
        PreviewDenseVectorQuery(field="embedding", values=[0.1, 0.2, 0.3, 0.4])
    ]

    def _search_with_filter(f: dict[str, object]) -> None:
        result = idx.documents.search(
            namespace=preview_namespace,
            top_k=5,
            score_by=score_by,  # type: ignore[arg-type]
            filter=f,
            include_fields=["*"],
        )
        assert isinstance(result, PreviewDocumentSearchResponse), (
            f"filter {f} should return a PreviewDocumentSearchResponse (200 OK)"
        )

    # $ne — not equals on string field
    _search_with_filter({"category": {"$ne": "finance"}})

    # $gt — greater than on integer field
    _search_with_filter({"year": {"$gt": 2010}})

    # $lt — less than on integer field
    _search_with_filter({"year": {"$lt": 2025}})

    # $lte — less or equal on integer field
    _search_with_filter({"year": {"$lte": 2022}})

    # $in — value in array on string field
    _search_with_filter({"category": {"$in": ["tech", "medicine"]}})

    # $nin — value not in array on string field
    _search_with_filter({"category": {"$nin": ["finance", "sports"]}})

    # $or — logical OR combining two conditions
    _search_with_filter({"$or": [{"category": {"$eq": "tech"}}, {"year": {"$lt": 2020}}]})


# ---------------------------------------------------------------------------
# test_preview_index_model_read_capacity_on_demand — §3 PreviewIndexModel.read_capacity
# ---------------------------------------------------------------------------


def test_preview_index_model_read_capacity_on_demand(
    client: Pinecone,
    preview_index_name: str,
    cleanup_preview_indexes: list[str],
    require_preview: None,
) -> None:
    """Verify PreviewIndexModel.read_capacity is deserialized to PreviewReadCapacityOnDemandResponse (§3).

    §3 defines read_capacity as a discriminated union keyed on the "mode" field:
      - mode="OnDemand"  → PreviewReadCapacityOnDemandResponse (has .status)
      - mode="Dedicated" → PreviewReadCapacityDedicatedResponse (has .dedicated and .status)

    This test creates a default-capacity index (no read_capacity argument → OnDemand),
    verifies create() returns a PreviewReadCapacityOnDemandResponse, then verifies
    describe() returns the same type with a string status.state. No existing test
    checks the read_capacity field or the OnDemand discriminated union dispatch.
    """
    from pinecone.preview.models import (
        PreviewIndexModel,
        PreviewReadCapacity,
        PreviewReadCapacityOnDemandResponse,
        PreviewReadCapacityStatus,
    )

    schema = (
        PreviewSchemaBuilder()
        .add_dense_vector_field("embedding", dimension=4, metric="cosine")
        .build()
    )
    cleanup_preview_indexes.append(preview_index_name)
    created = client.preview.indexes.create(name=preview_index_name, schema=schema)

    assert isinstance(created, PreviewIndexModel)

    # §3: read_capacity is a PreviewReadCapacity (OnDemand or Dedicated union) or None
    assert created.read_capacity is None or isinstance(
        created.read_capacity,
        (PreviewReadCapacityOnDemandResponse,)
        + (PreviewReadCapacity.__args__ if hasattr(PreviewReadCapacity, "__args__") else ()),
    ), (
        f"create() read_capacity expected PreviewReadCapacity or None, got {type(created.read_capacity)}"
    )

    # Verify it is the OnDemand variant (default capacity mode)
    rc = created.read_capacity
    assert rc is not None, "read_capacity should not be None for a default OnDemand index"
    assert isinstance(rc, PreviewReadCapacityOnDemandResponse), (
        f"Expected PreviewReadCapacityOnDemandResponse (mode=OnDemand), got {type(rc)}"
    )

    # §3: OnDemand variant has a .status of type PreviewReadCapacityStatus
    assert isinstance(rc.status, PreviewReadCapacityStatus), (
        f"read_capacity.status expected PreviewReadCapacityStatus, got {type(rc.status)}"
    )
    assert isinstance(rc.status.state, str) and len(rc.status.state) > 0, (
        f"read_capacity.status.state should be a non-empty string, got {rc.status.state!r}"
    )

    # Verify describe() also returns the same read_capacity type
    def _is_ready(m: object) -> bool:
        return isinstance(m, PreviewIndexModel) and m.status.state == "Ready"

    poll_until(
        lambda: client.preview.indexes.describe(preview_index_name),
        _is_ready,
        timeout=300,
        interval=5,
        description=f"index {preview_index_name} ready",
    )

    described = client.preview.indexes.describe(preview_index_name)
    assert isinstance(described, PreviewIndexModel)
    assert described.read_capacity is not None, (
        "describe() read_capacity should not be None for an OnDemand index"
    )
    assert isinstance(described.read_capacity, PreviewReadCapacityOnDemandResponse), (
        f"describe() read_capacity: expected PreviewReadCapacityOnDemandResponse, got {type(described.read_capacity)}"
    )
    assert isinstance(described.read_capacity.status.state, str), (
        "describe() read_capacity.status.state should be a string"
    )


# ---------------------------------------------------------------------------
# test_search_score_by_plain_dict_accepted — §6 "Dict format"
# ---------------------------------------------------------------------------


@pytest.mark.timeout(300)
def test_search_score_by_plain_dict_accepted(
    client: Pinecone,
    preview_index_name: str,
    cleanup_preview_indexes: list[str],
    preview_namespace: str,
    require_preview: None,
) -> None:
    """Verify search() accepts plain dicts in score_by in addition to typed models (§6 "Dict format").

    The spec §6 states: "All query types can also be passed as plain dicts matching
    the wire format. This is useful when the SDK's typed models haven't been updated
    for new score-by types added by the API."

    All existing search tests use typed objects (PreviewDenseVectorQuery, etc.). This test
    verifies the alternative path: plain dict entries are passed through to the API unchanged.
    Uses include_fields=["*"] to return all stored fields. 0 matches are acceptable.
    """
    from pinecone.preview.models import PreviewIndexModel

    schema = (
        PreviewSchemaBuilder()
        .add_dense_vector_field("embedding", dimension=4, metric="cosine")
        .build()
    )
    cleanup_preview_indexes.append(preview_index_name)
    client.preview.indexes.create(name=preview_index_name, schema=schema)

    def _is_ready(m: object) -> bool:
        return isinstance(m, PreviewIndexModel) and m.status.state == "Ready"

    poll_until(
        lambda: client.preview.indexes.describe(preview_index_name),
        _is_ready,
        timeout=300,
        interval=5,
        description=f"index {preview_index_name} ready",
    )

    idx = client.preview.index(name=preview_index_name)
    idx.documents.upsert(
        namespace=preview_namespace,
        documents=[{"_id": "doc-dict", "embedding": [0.1, 0.2, 0.3, 0.4]}],
    )

    # §6: pass score_by as a plain dict (wire format) rather than a typed model
    response = idx.documents.search(
        namespace=preview_namespace,
        top_k=5,
        score_by=[{"type": "dense_vector", "field": "embedding", "values": [0.1, 0.2, 0.3, 0.4]}],
        include_fields=["*"],
    )

    assert isinstance(response, PreviewDocumentSearchResponse), (
        f"search() with plain dict score_by expected PreviewDocumentSearchResponse, got {type(response)}"
    )
    assert response.namespace == preview_namespace, (
        f"response.namespace expected {preview_namespace!r}, got {response.namespace!r}"
    )
    assert response.usage is not None, "response.usage should not be None for a 200 OK response"
    assert isinstance(response.usage, PreviewUsage), (
        f"response.usage expected PreviewUsage, got {type(response.usage)}"
    )
    assert isinstance(response.usage.read_units, int), (
        f"usage.read_units expected int, got {type(response.usage.read_units)}"
    )


# ---------------------------------------------------------------------------
# test_search_response_and_index_model_display_methods — §14 Notebook and REPL rendering
# ---------------------------------------------------------------------------


@pytest.mark.timeout(300)
def test_search_response_and_index_model_display_methods(
    client: Pinecone,
    preview_index_name: str,
    cleanup_preview_indexes: list[str],
    preview_namespace: str,
    require_preview: None,
) -> None:
    """Verify __repr__ and _repr_html_ for PreviewDocumentSearchResponse and PreviewIndexModel (§14).

    Spec §14 "Models that need display methods" lists PreviewIndexModel and
    PreviewDocumentSearchResponse as requiring __repr__ and _repr_html_ implementations.
    PVT-027 verified BatchResult display methods; this test covers the two remaining
    preview-specific response models.

    PreviewIndexModel.__repr__ must start with 'PreviewIndexModel(' and include
    name, status, host, and deletion_protection fields. _repr_html_() must return a
    non-empty HTML string containing 'PreviewIndexModel'.

    PreviewDocumentSearchResponse.__repr__ must start with 'SearchResponse(' and
    include matches count, namespace, and usage fields. _repr_html_() must return a
    non-empty HTML string containing 'SearchResponse'.

    Uses an OnDemand dense vector index — 0 matches are acceptable since display
    methods operate on the envelope (namespace, usage, match count) not results.
    Uses include_fields=["*"] to return all stored fields.
    """
    from pinecone.preview.models import PreviewIndexModel

    schema = (
        PreviewSchemaBuilder()
        .add_dense_vector_field("embedding", dimension=4, metric="cosine")
        .build()
    )
    cleanup_preview_indexes.append(preview_index_name)
    client.preview.indexes.create(name=preview_index_name, schema=schema)

    def _is_ready(m: object) -> bool:
        return isinstance(m, PreviewIndexModel) and m.status.state == "Ready"

    poll_until(
        lambda: client.preview.indexes.describe(preview_index_name),
        _is_ready,
        timeout=300,
        interval=5,
        description=f"index {preview_index_name} ready",
    )

    # §14: PreviewIndexModel.__repr__ and _repr_html_
    described = client.preview.indexes.describe(preview_index_name)

    index_repr = repr(described)
    assert index_repr.startswith("PreviewIndexModel("), (
        f"PreviewIndexModel.__repr__ must start with 'PreviewIndexModel(', got: {index_repr!r}"
    )
    assert f"name={preview_index_name!r}" in index_repr, (
        f"PreviewIndexModel.__repr__ must include name=, got: {index_repr!r}"
    )
    assert "status=" in index_repr, (
        f"PreviewIndexModel.__repr__ must include status=, got: {index_repr!r}"
    )
    assert "host=" in index_repr, (
        f"PreviewIndexModel.__repr__ must include host=, got: {index_repr!r}"
    )
    assert "deletion_protection=" in index_repr, (
        f"PreviewIndexModel.__repr__ must include deletion_protection=, got: {index_repr!r}"
    )

    index_html = described._repr_html_()
    assert isinstance(index_html, str) and len(index_html) > 0, (
        "PreviewIndexModel._repr_html_() must return a non-empty string"
    )
    assert "PreviewIndexModel" in index_html, (
        f"PreviewIndexModel._repr_html_() must contain 'PreviewIndexModel', got: {index_html[:300]!r}"
    )

    # §14: PreviewDocumentSearchResponse.__repr__ and _repr_html_
    idx = client.preview.index(name=preview_index_name)
    response = idx.documents.search(
        namespace=preview_namespace,
        top_k=5,
        score_by=[PreviewDenseVectorQuery(field="embedding", values=[0.1, 0.2, 0.3, 0.4])],
        include_fields=["*"],
    )

    # Format: SearchResponse(matches=N, namespace='...', usage=...)
    search_repr = repr(response)
    assert search_repr.startswith("SearchResponse("), (
        f"PreviewDocumentSearchResponse.__repr__ must start with 'SearchResponse(', got: {search_repr!r}"
    )
    assert "matches=" in search_repr, (
        f"PreviewDocumentSearchResponse.__repr__ must include matches=, got: {search_repr!r}"
    )
    assert "namespace=" in search_repr, (
        f"PreviewDocumentSearchResponse.__repr__ must include namespace=, got: {search_repr!r}"
    )
    assert "usage=" in search_repr, (
        f"PreviewDocumentSearchResponse.__repr__ must include usage=, got: {search_repr!r}"
    )

    search_html = response._repr_html_()
    assert isinstance(search_html, str) and len(search_html) > 0, (
        "PreviewDocumentSearchResponse._repr_html_() must return a non-empty string"
    )
    assert "SearchResponse" in search_html, (
        f"PreviewDocumentSearchResponse._repr_html_() must contain 'SearchResponse', "
        f"got: {search_html[:300]!r}"
    )


# ---------------------------------------------------------------------------
# test_sparse_vector_schema_and_query_accepted — §1 add_sparse_vector_field + §6 PreviewSparseVectorQuery
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_sparse_vector_schema_and_query_accepted(
    client: Pinecone,
    preview_index_name: str,
    cleanup_preview_indexes: list[str],
    preview_namespace: str,
    require_preview: None,
) -> None:
    """Verify §1 add_sparse_vector_field() + §6 PreviewSparseVectorQuery / PreviewSparseValues.

    The pre-written test_sparse_vector_search mixes sparse + FTS fields (requiring
    dedicated capacity). This test uses ONLY a sparse vector field (OnDemand) to
    isolate the sparse-vector code path.

    Verifies:
    - add_sparse_vector_field("sparse_embedding") builds schema dict with type="sparse_vector"
    - Index creation with sparse-only schema is accepted by the API
    - Upsert of 2 documents with sparse_embedding dicts is accepted (upserted_count==2)
    - search() with PreviewSparseVectorQuery + PreviewSparseValues returns 200 OK
    - Response is PreviewDocumentSearchResponse with correct namespace and usage
    - 0 matches is acceptable (OnDemand eventual consistency)
    """
    from pinecone.preview.models import PreviewIndexModel

    # §1: add_sparse_vector_field builds correct schema dict
    schema = PreviewSchemaBuilder().add_sparse_vector_field("sparse_embedding").build()
    assert schema["fields"]["sparse_embedding"]["type"] == "sparse_vector", (
        f"add_sparse_vector_field must produce type='sparse_vector', got: {schema['fields']['sparse_embedding']}"
    )

    cleanup_preview_indexes.append(preview_index_name)
    client.preview.indexes.create(name=preview_index_name, schema=schema)

    def _is_ready(m: object) -> bool:
        return isinstance(m, PreviewIndexModel) and m.status.state == "Ready"

    poll_until(
        lambda: client.preview.indexes.describe(preview_index_name),
        _is_ready,
        timeout=300,
        interval=5,
        description=f"sparse index {preview_index_name} ready",
    )

    idx = client.preview.index(name=preview_index_name)
    upsert_resp = idx.documents.upsert(
        namespace=preview_namespace,
        documents=[
            {"_id": "doc-1", "sparse_embedding": {"indices": [0, 42], "values": [0.8, 0.3]}},
            {"_id": "doc-2", "sparse_embedding": {"indices": [1, 99], "values": [0.1, 0.1]}},
        ],
    )
    assert upsert_resp.upserted_count == 2, (
        f"expected upserted_count=2, got {upsert_resp.upserted_count}"
    )

    # §6: PreviewSparseVectorQuery with PreviewSparseValues — verify serialized + accepted by API
    response = idx.documents.search(
        namespace=preview_namespace,
        top_k=5,
        score_by=[
            PreviewSparseVectorQuery(
                field="sparse_embedding",
                sparse_values=PreviewSparseValues(indices=[0, 42], values=[0.8, 0.3]),
            )
        ],
        include_fields=["*"],
    )

    assert isinstance(response, PreviewDocumentSearchResponse), (
        f"Expected PreviewDocumentSearchResponse, got {type(response)}"
    )
    assert response.namespace == preview_namespace, (
        f"response.namespace {response.namespace!r} != {preview_namespace!r}"
    )
    assert response.usage is not None, "response.usage must not be None on a successful search"
    assert isinstance(response.usage.read_units, int), (
        f"read_units must be int, got {type(response.usage.read_units)}"
    )
    assert response.usage.read_units >= 0
    assert isinstance(response.matches, list), (
        f"response.matches must be a list, got {type(response.matches)}"
    )


# ---------------------------------------------------------------------------
# test_preview_document_model_attributes_after_fts_search — §7 PreviewDocument
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_preview_document_model_attributes_after_fts_search(
    client: Pinecone,
    fts_index_state: tuple[str, str] | None,
    require_preview: None,
) -> None:
    """Verify PreviewDocument attribute access after a real FTS search with actual matches.

    Spec §7 defines PreviewDocument with: .id property, ._id alias, .score float,
    __getattr__ dynamic field access, .get(key), .get(key, default) for missing keys,
    .to_dict() returning a dict with _id/score/fields, and .to_json() as valid JSON.

    The pre-existing test_full_text_search_only_flow only checks _id, score, isinstance.
    PVT-024 documented this gap — OnDemand searches return 0 results. Uses the
    session-scoped fts_index_state fixture which creates and warms up one FTS+dedicated
    index once per session, avoiding per-test cold-start latency.
    """
    import json

    if fts_index_state is None:
        pytest.skip("FTS index unavailable — preview endpoint unreachable or cold-start timeout")

    host, namespace = fts_index_state
    idx = client.preview.index(host=host)

    response = idx.documents.search(
        namespace=namespace,
        top_k=5,
        score_by=[PreviewTextQuery(fields=["text"], query="ancient Rome")],
        include_fields=["text", "year"],
    )

    assert len(response.matches) > 0, (
        "Expected at least one FTS match for 'ancient Rome' (session fixture confirmed searchable)"
    )
    doc = response.matches[0]

    # §7: id and _id alias must both return the same non-empty string
    assert isinstance(doc.id, str), f"doc.id must be str, got {type(doc.id)}"
    assert doc._id == doc.id, f"doc._id must alias doc.id: {doc._id!r} != {doc.id!r}"
    assert len(doc.id) > 0, "doc.id must be non-empty"

    # §7: score is a positive float for a real FTS match
    assert doc.score is not None, "doc.score must not be None for an FTS match"
    assert isinstance(doc.score, float), f"doc.score must be float, got {type(doc.score)}"
    assert doc.score > 0, f"doc.score must be positive for a real match, got {doc.score}"

    # §7: __getattr__ dynamic field access
    text_val = doc.text
    assert isinstance(text_val, str), f"doc.text via __getattr__ must be str, got {type(text_val)}"
    year_val = doc.year
    # Integer filterable fields arrive as JSON numbers; msgspec dict[str, Any] may
    # deserialize them as int or float depending on the server's wire representation.
    assert isinstance(year_val, (int, float)), (
        f"doc.year via __getattr__ must be a number, got {type(year_val)}"
    )

    # §7: .get(key) returns same value as attribute access
    assert doc.get("text") == text_val, (
        f"doc.get('text') must equal doc.text: {doc.get('text')!r} != {text_val!r}"
    )
    assert doc.get("year") == year_val, (
        f"doc.get('year') must equal doc.year: {doc.get('year')!r} != {year_val!r}"
    )

    # §7: .get(missing_key, default) returns the default
    sentinel = object()
    assert doc.get("nonexistent_field", sentinel) is sentinel, (
        "doc.get('nonexistent_field', sentinel) must return default for missing field"
    )

    # §7: to_dict() returns a plain dict with the raw API data.
    # to_dict() passes through whatever the server returns unchanged — the
    # server emits "_id" and "_score" as the wire keys.
    d = doc.to_dict()
    assert isinstance(d, dict), f"doc.to_dict() must return dict, got {type(d)}"
    id_key = "_id" if "_id" in d else "id"
    assert id_key in d, f"to_dict() must include an id key ('_id' or 'id'), keys: {set(d.keys())}"
    assert "_score" in d, f"to_dict() must include '_score', keys: {set(d.keys())}"
    assert "text" in d, (
        f"to_dict() must include 'text' (from include_fields), keys: {set(d.keys())}"
    )
    assert d[id_key] == doc.id, f"to_dict()[{id_key!r}] must equal doc.id: {d[id_key]!r}"
    assert d["_score"] == doc.score, "to_dict()['_score'] must equal doc.score"

    # §7: to_json() returns valid JSON round-trippable to same content as to_dict()
    j = doc.to_json()
    assert isinstance(j, str), f"doc.to_json() must return str, got {type(j)}"
    parsed = json.loads(j)
    assert isinstance(parsed, dict), "doc.to_json() must parse to a dict"
    assert parsed[id_key] == doc.id, f"to_json() parsed {id_key!r} must equal doc.id"
    assert parsed["_score"] == doc.score, "to_json() parsed _score must equal doc.score"


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_search_text_query_multiple_fields(
    client: Pinecone,
    fts_index_state: tuple[str, str] | None,
    require_preview: None,
) -> None:
    """PreviewTextQuery with multiple fields is accepted and returns a valid response."""
    if fts_index_state is None:
        pytest.skip("FTS index unavailable — preview endpoint unreachable or cold-start timeout")

    host, namespace = fts_index_state
    idx = client.preview.index(host=host)

    response = idx.documents.search(
        namespace=namespace,
        top_k=5,
        score_by=[PreviewTextQuery(fields=["text", "year"], query="ancient Rome")],
    )

    assert isinstance(response, PreviewDocumentSearchResponse)
