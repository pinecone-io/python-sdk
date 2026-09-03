"""One ranking out of several namespaces: the merged result, and the merger itself."""

from __future__ import annotations

import heapq
from typing import Any

from msgspec import Struct, field

from pinecone.models._mixin import StructDictMixin
from pinecone.models.vectors.responses import QueryResponse
from pinecone.models.vectors.usage import Usage
from pinecone.models.vectors.vector import ScoredVector


class QueryResultsAggregatorInvalidTopKError(ValueError):
    def __init__(self, top_k: int) -> None:
        super().__init__(f"Invalid top_k value {top_k}. top_k must be at least 1.")


class QueryNamespacesResults(StructDictMixin, Struct, kw_only=True):
    """One merged ranking drawn from several namespaces, as ``query_namespaces`` returns it.

    Reads like a :class:`~pinecone.models.vectors.responses.QueryResponse`: ``matches`` is
    already interleaved and ordered, so ``matches[0]`` is the best hit found anywhere, and
    each element is a :class:`~pinecone.models.vectors.vector.ScoredVector` you read as
    ``.id``, ``.score``, ``.values`` and ``.metadata``. What it does not carry is a
    ``namespace`` field, because the matches came from different ones — keep your own
    mapping from ID to namespace if you need to know where a hit lived.

    Attributes:
        matches (list[ScoredVector]): The merged top-k across every namespace queried,
            ordered by the ``metric`` the query named.
        usage (Usage): Read units summed over all the namespace queries.
        ns_usage (dict[str, Usage]): Read units for each namespace, keyed by namespace
            name, for attributing cost to one namespace rather than the fan-out.

    Examples:
        .. code-block:: python

            results = idx.query_namespaces(
                vector=[0.012, -0.087, 0.153],
                namespaces=["articles-en", "articles-fr"],
                metric="cosine",
                top_k=5,
            )
            for match in results.matches:
                print(match.id, match.score)
            print(results.usage.read_units, results.ns_usage)
    """

    matches: list[ScoredVector] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    ns_usage: dict[str, Usage] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        """Read a field by name, so ``results["matches"]`` works as well as ``.matches``.

        Raises:
            KeyError: If *key* is not one of this model's fields.
        """
        if key not in self.__struct_fields__:
            raise KeyError(key)
        return getattr(self, key)

    def __contains__(self, key: object) -> bool:
        """Report whether *key* names a field on this result."""
        return key in self.__struct_fields__


_VALID_METRICS = frozenset({"cosine", "euclidean", "dotproduct"})


class QueryResultsAggregator:
    """Merges per-namespace query responses into a single top-k ranking.

    ``query_namespaces`` uses this internally, so reach for it directly only when you run
    the per-namespace queries yourself — fanning them out concurrently, or mixing in
    results you already had. Feed each response in with :meth:`add_results`, then call
    :meth:`get_results` once; the aggregator is single-use and refuses further input after
    that.

    Which direction counts as "better" comes from *metric*, so it must match the field you
    queried: ``cosine`` and ``dotproduct`` rank higher scores first, ``euclidean`` ranks
    lower scores first. Get it wrong and you get a valid-looking ranking that is exactly
    backwards. Equal scores keep the order they were added in.

    Args:
        metric (str): The metric the queries ranked by — ``"cosine"``, ``"euclidean"``, or
            ``"dotproduct"``. Keyword-only.
        top_k (int): How many matches to keep across all namespaces. Defaults to ``10``.
            Keyword-only.

    Raises:
        ValueError: If *metric* is not one of the three, or *top_k* is below 1.

    Examples:
        >>> from pinecone.models.vectors.query_aggregator import QueryResultsAggregator
        >>> from pinecone.models.vectors.responses import QueryResponse
        >>> from pinecone import ScoredVector
        >>> aggregator = QueryResultsAggregator(metric="cosine", top_k=2)
        >>> aggregator.add_results(
        ...     "articles-en",
        ...     QueryResponse(matches=[ScoredVector(id="article-101", score=0.42)]),
        ... )
        >>> aggregator.add_results(
        ...     "articles-fr",
        ...     QueryResponse(matches=[ScoredVector(id="article-207", score=0.91)]),
        ... )
        >>> [match.id for match in aggregator.get_results().matches]
        ['article-207', 'article-101']

    .. seealso::
       ``Index.query_namespaces`` — the one call that fans the query out and merges for you.
    """

    __slots__ = (
        "_counter",
        "_finalized",
        "_heap",
        "_is_bigger_better",
        "_metric",
        "_ns_usage",
        "_read_units",
        "_top_k",
    )

    def __init__(self, *, metric: str, top_k: int = 10) -> None:
        if metric not in _VALID_METRICS:
            raise ValueError(
                f"Invalid metric {metric!r}. Must be one of: {', '.join(sorted(_VALID_METRICS))}"
            )
        if top_k < 1:
            raise QueryResultsAggregatorInvalidTopKError(top_k)

        self._metric = metric
        self._top_k = top_k
        self._heap: list[tuple[float, int, ScoredVector]] = []
        self._counter: int = 0
        self._finalized: bool = False
        self._read_units: int = 0
        self._ns_usage: dict[str, Usage] = {}
        self._is_bigger_better: bool = metric in ("cosine", "dotproduct")

    def add_results(self, namespace: str, response: QueryResponse) -> None:
        """Fold one namespace's query response into the merge.

        Call once per namespace, in any order — the ranking does not depend on the order
        you add them, only on the scores. Matches beyond ``top_k`` are dropped as you go,
        so adding many namespaces does not grow memory with the total number of matches.

        Args:
            namespace (str): The namespace this response came from; used as the key in
                :attr:`QueryNamespacesResults.ns_usage`, e.g. ``"articles-en"``.
            response (QueryResponse): What ``query`` returned for that namespace.

        Raises:
            ValueError: If called after :meth:`get_results` — the merge is closed at that
                point, so build a new aggregator instead.
        """
        if self._finalized:
            raise ValueError("Cannot add results after get_results()")

        if response.usage is not None:
            self._read_units += response.usage.read_units or 0
            self._ns_usage[namespace] = response.usage

        for match in response.matches:
            if self._is_bigger_better:
                key = -match.score
            else:
                key = match.score
            heapq.heappush(self._heap, (key, self._counter, match))
            self._counter += 1

        if len(self._heap) > self._top_k:
            self._heap = heapq.nsmallest(self._top_k, self._heap)
            heapq.heapify(self._heap)

    def get_results(self) -> QueryNamespacesResults:
        """Close the merge and return the combined ranking.

        Closes the aggregator: a later :meth:`add_results` raises. Calling this again
        returns the same ranking.

        Returns:
            :class:`QueryNamespacesResults` with ``matches`` (the merged top-k, best
            first), ``usage`` (read units summed over every namespace) and ``ns_usage``
            (read units per namespace).
        """
        self._finalized = True
        sorted_entries = sorted(self._heap)
        matches = [entry[2] for entry in sorted_entries[: self._top_k]]
        return QueryNamespacesResults(
            matches=matches,
            usage=Usage(read_units=self._read_units),
            ns_usage=self._ns_usage,
        )
