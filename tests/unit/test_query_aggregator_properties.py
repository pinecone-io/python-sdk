"""Property-based tests for QueryResultsAggregator.

The aggregator merges per-namespace results with a bounded heap; these tests
check it against a brute-force reference (collect everything, stable-sort by
score, take top_k) for randomly generated namespaces, scores, and metrics.
Each match is given an id equal to its global insertion index so the result
can be compared exactly, including the insertion-order tie-break.
"""

from __future__ import annotations

from hypothesis import example, given
from hypothesis import strategies as st

from pinecone.models.vectors.query_aggregator import QueryResultsAggregator
from pinecone.models.vectors.responses import QueryResponse
from pinecone.models.vectors.usage import Usage
from pinecone.models.vectors.vector import ScoredVector

_metrics = st.sampled_from(["cosine", "euclidean", "dotproduct"])
_scores = st.lists(
    st.floats(allow_nan=False, allow_infinity=False, width=32),
    max_size=20,
)
_namespaces = st.lists(_scores, max_size=6)


def _reference_top_ids(namespaces: list[list[float]], metric: str, top_k: int) -> list[str]:
    bigger_better = metric in ("cosine", "dotproduct")
    entries: list[tuple[float, int]] = []
    order = 0
    for ns_scores in namespaces:
        for score in ns_scores:
            key = -score if bigger_better else score
            entries.append((key, order))
            order += 1
    entries.sort(key=lambda e: (e[0], e[1]))
    return [str(order) for _key, order in entries[:top_k]]


@given(namespaces=_namespaces, metric=_metrics, top_k=st.integers(min_value=1, max_value=30))
@example(namespaces=[], metric="cosine", top_k=1)
@example(namespaces=[[0.9, 0.9, 0.9]], metric="cosine", top_k=2)
@example(namespaces=[[0.1], [0.1]], metric="euclidean", top_k=1)
def test_aggregator_matches_brute_force_reference(
    namespaces: list[list[float]], metric: str, top_k: int
) -> None:
    aggregator = QueryResultsAggregator(metric=metric, top_k=top_k)
    order = 0
    for i, ns_scores in enumerate(namespaces):
        matches = []
        for score in ns_scores:
            matches.append(ScoredVector(id=str(order), score=score))
            order += 1
        aggregator.add_results(f"ns{i}", QueryResponse(matches=matches, namespace=f"ns{i}"))

    result_ids = [m.id for m in aggregator.get_results().matches]
    assert result_ids == _reference_top_ids(namespaces, metric, top_k)


@given(namespaces=_namespaces, metric=_metrics, top_k=st.integers(min_value=1, max_value=30))
def test_aggregator_result_count_is_bounded(
    namespaces: list[list[float]], metric: str, top_k: int
) -> None:
    aggregator = QueryResultsAggregator(metric=metric, top_k=top_k)
    total = 0
    for i, ns_scores in enumerate(namespaces):
        matches = [ScoredVector(id=f"{i}-{j}", score=s) for j, s in enumerate(ns_scores)]
        total += len(matches)
        aggregator.add_results(f"ns{i}", QueryResponse(matches=matches, namespace=f"ns{i}"))

    assert len(aggregator.get_results().matches) == min(top_k, total)


@given(namespaces=_namespaces, metric=_metrics, top_k=st.integers(min_value=1, max_value=30))
def test_aggregator_results_are_sorted_by_relevance(
    namespaces: list[list[float]], metric: str, top_k: int
) -> None:
    aggregator = QueryResultsAggregator(metric=metric, top_k=top_k)
    for i, ns_scores in enumerate(namespaces):
        matches = [ScoredVector(id=f"{i}-{j}", score=s) for j, s in enumerate(ns_scores)]
        aggregator.add_results(f"ns{i}", QueryResponse(matches=matches, namespace=f"ns{i}"))

    scores = [m.score for m in aggregator.get_results().matches]
    if metric in ("cosine", "dotproduct"):
        assert scores == sorted(scores, reverse=True)
    else:
        assert scores == sorted(scores)


@given(
    read_units=st.lists(
        st.one_of(st.none(), st.integers(min_value=0, max_value=10**6)), max_size=6
    ),
    metric=_metrics,
)
def test_aggregator_sums_read_units_across_namespaces(
    read_units: list[int | None], metric: str
) -> None:
    aggregator = QueryResultsAggregator(metric=metric, top_k=10)
    expected_total = 0
    expected_ns_count = 0
    for i, ru in enumerate(read_units):
        usage = None if ru is None else Usage(read_units=ru)
        if usage is not None:
            expected_total += ru or 0
            expected_ns_count += 1
        aggregator.add_results(f"ns{i}", QueryResponse(matches=[], namespace=f"ns{i}", usage=usage))

    result = aggregator.get_results()
    assert result.usage.read_units == expected_total
    assert len(result.ns_usage) == expected_ns_count
