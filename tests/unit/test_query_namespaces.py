"""Unit tests for Index.query_namespaces."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from pinecone import Index
from pinecone.errors.exceptions import ValidationError
from pinecone.models.vectors.responses import QueryResponse
from pinecone.models.vectors.usage import Usage
from pinecone.models.vectors.vector import ScoredVector

INDEX_HOST = "test-index-abc1234.svc.us-east1-gcp.pinecone.io"


class _BoolAmbiguousVector:
    def __init__(self, values: list[float]) -> None:
        self._values = values

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, index: int) -> float:
        return self._values[index]

    def __bool__(self) -> bool:
        raise ValueError("truth value is ambiguous")


def _make_index() -> Index:
    return Index(host=INDEX_HOST, api_key="test-key")


def _make_query_response(
    matches: list[ScoredVector],
    namespace: str = "",
    read_units: int = 5,
) -> QueryResponse:
    return QueryResponse(
        matches=matches,
        namespace=namespace,
        usage=Usage(read_units=read_units),
    )


def _scored(id: str, score: float) -> ScoredVector:
    return ScoredVector(id=id, score=score)


class TestQueryNamespacesFanOut:
    def test_query_namespaces_fans_out(self) -> None:
        """Mock Index.query to return a QueryResponse with 2 matches, call
        query_namespaces with 3 namespaces, verify query was called 3 times."""
        idx = _make_index()
        response = _make_query_response(
            [_scored("v1", 0.9), _scored("v2", 0.8)],
        )

        with patch.object(idx, "query", return_value=response) as mock_query:
            idx.query_namespaces(
                vector=[0.1, 0.2, 0.3],
                namespaces=["ns1", "ns2", "ns3"],
                metric="cosine",
                top_k=10,
            )
            assert mock_query.call_count == 3
            called_namespaces = sorted(c.kwargs["namespace"] for c in mock_query.call_args_list)
            assert called_namespaces == ["ns1", "ns2", "ns3"]


class TestQueryNamespacesTieBreaking:
    def test_tie_breaking_follows_input_namespace_order(self) -> None:
        """With tied scores, result order must match input namespace order
        regardless of completion order. Earlier-listed namespaces are made
        slowest so completion order is the reverse of input order; the
        aggregator's insertion-order tie-break is only deterministic when
        the caller adds results in input order."""
        idx = _make_index()

        delays = {"ns1": 0.03, "ns2": 0.02, "ns3": 0.01, "ns4": 0.0}

        def query_side_effect(**kwargs: object) -> QueryResponse:
            ns = str(kwargs["namespace"])
            time.sleep(delays[ns])
            return _make_query_response([_scored(f"{ns}-v0", 1.0), _scored(f"{ns}-v1", 1.0)])

        with patch.object(idx, "query", side_effect=query_side_effect):
            result = idx.query_namespaces(
                vector=[0.1, 0.2],
                namespaces=["ns1", "ns2", "ns3", "ns4"],
                metric="cosine",
                top_k=8,
            )

        assert [m.id for m in result.matches] == [
            "ns1-v0",
            "ns1-v1",
            "ns2-v0",
            "ns2-v1",
            "ns3-v0",
            "ns3-v1",
            "ns4-v0",
            "ns4-v1",
        ]


class TestQueryNamespacesMerge:
    def test_query_namespaces_merges_results(self) -> None:
        """Mock Index.query to return different scores per namespace, verify
        final results are sorted by score (cosine: descending)."""
        idx = _make_index()

        def query_side_effect(**kwargs: object) -> QueryResponse:
            ns = kwargs["namespace"]
            if ns == "ns1":
                return _make_query_response([_scored("a", 0.9), _scored("b", 0.7)])
            return _make_query_response([_scored("c", 0.95), _scored("d", 0.6)])

        with patch.object(idx, "query", side_effect=query_side_effect):
            result = idx.query_namespaces(
                vector=[0.1, 0.2],
                namespaces=["ns1", "ns2"],
                metric="cosine",
                top_k=3,
            )
            ids = [m.id for m in result.matches]
            assert ids == ["c", "a", "b"]


class TestQueryNamespacesValidation:
    def test_query_namespaces_empty_namespaces_raises(self) -> None:
        idx = _make_index()
        with pytest.raises(ValidationError, match="namespaces must be a non-empty list"):
            idx.query_namespaces(
                vector=[0.1, 0.2],
                namespaces=[],
                metric="cosine",
            )

    def test_query_namespaces_empty_vector_raises(self) -> None:
        idx = _make_index()
        with pytest.raises(
            ValidationError, match="at least one of 'vector' or 'sparse_vector' must be provided"
        ):
            idx.query_namespaces(
                vector=[],
                namespaces=["ns1"],
                metric="cosine",
            )

    def test_query_namespaces_does_not_bool_check_vector(self) -> None:
        idx = _make_index()
        response = _make_query_response([_scored("v1", 0.9)])
        vector = _BoolAmbiguousVector([0.1, 0.2])

        with patch.object(idx, "query", return_value=response) as mock_query:
            idx.query_namespaces(
                vector=vector,
                namespaces=["ns1"],
                metric="cosine",
            )

        assert mock_query.call_count == 1
        assert mock_query.call_args.kwargs["vector"] is vector

    def test_query_namespaces_no_vector_no_sparse_raises(self) -> None:
        idx = _make_index()
        with pytest.raises(
            ValidationError, match="at least one of 'vector' or 'sparse_vector' must be provided"
        ):
            idx.query_namespaces(
                namespaces=["ns1"],
                metric="dotproduct",
            )

    def test_query_namespaces_invalid_metric_raises(self) -> None:
        idx = _make_index()
        with pytest.raises(ValidationError, match="Invalid metric 'badmetric'"):
            idx.query_namespaces(
                vector=[0.1, 0.2],
                namespaces=["ns1"],
                metric="badmetric",
            )

    def test_query_namespaces_invalid_metric_is_validation_error(self) -> None:
        from pinecone.errors.exceptions import PineconeError

        idx = _make_index()
        with pytest.raises(PineconeError):
            idx.query_namespaces(
                vector=[0.1, 0.2],
                namespaces=["ns1"],
                metric="invalid",
            )

    def test_query_namespaces_sparse_only_succeeds(self) -> None:
        """Passing only sparse_vector (no vector) should not raise validation error."""
        idx = _make_index()
        response = _make_query_response([_scored("v1", 0.9)])

        with patch.object(idx, "query", return_value=response) as mock_query:
            idx.query_namespaces(
                sparse_vector={"indices": [0, 1], "values": [0.5, 0.8]},
                namespaces=["ns1"],
                metric="dotproduct",
            )
            assert mock_query.call_count == 1
            call_kwargs = mock_query.call_args.kwargs
            assert "vector" not in call_kwargs
            assert call_kwargs["sparse_vector"] == {"indices": [0, 1], "values": [0.5, 0.8]}


class TestQueryNamespacesDedup:
    def test_query_namespaces_deduplicates_namespaces(self) -> None:
        """Call with namespaces=["a", "b", "a"], verify query is called only twice."""
        idx = _make_index()
        response = _make_query_response([_scored("v1", 0.5)])

        with patch.object(idx, "query", return_value=response) as mock_query:
            idx.query_namespaces(
                vector=[0.1],
                namespaces=["a", "b", "a"],
                metric="cosine",
            )
            assert mock_query.call_count == 2
            called_namespaces = sorted(c.kwargs["namespace"] for c in mock_query.call_args_list)
            assert called_namespaces == ["a", "b"]


class TestQueryNamespacesDefaultTopK:
    def test_query_namespaces_default_top_k(self) -> None:
        """Don't pass top_k, verify aggregator uses 10 (mock query to return
        15 matches each, verify result has 10)."""
        idx = _make_index()
        many_matches = [_scored(f"v{i}", 0.9 - i * 0.01) for i in range(15)]
        response = _make_query_response(many_matches)

        with patch.object(idx, "query", return_value=response):
            result = idx.query_namespaces(
                vector=[0.1, 0.2],
                namespaces=["ns1", "ns2"],
                metric="cosine",
            )
            assert len(result.matches) == 10


class TestQueryNamespacesFilter:
    def test_query_namespaces_passes_filter(self) -> None:
        """Pass a filter, verify each query call receives it."""
        idx = _make_index()
        response = _make_query_response([_scored("v1", 0.5)])
        test_filter = {"genre": {"$eq": "comedy"}}

        with patch.object(idx, "query", return_value=response) as mock_query:
            idx.query_namespaces(
                vector=[0.1],
                namespaces=["ns1", "ns2"],
                metric="cosine",
                filter=test_filter,
            )
            for c in mock_query.call_args_list:
                assert c.kwargs["filter"] == test_filter


class TestQueryNamespacesDRNParams:
    def test_query_namespaces_forwards_scan_factor(self) -> None:
        """Pass scan_factor, verify each query call receives it."""
        idx = _make_index()
        response = _make_query_response([_scored("v1", 0.5)])

        with patch.object(idx, "query", return_value=response) as mock_query:
            idx.query_namespaces(
                vector=[0.1],
                namespaces=["ns1"],
                metric="cosine",
                scan_factor=2.0,
            )
            assert mock_query.call_count == 1
            assert mock_query.call_args_list[0].kwargs["scan_factor"] == 2.0

    def test_query_namespaces_forwards_max_candidates(self) -> None:
        """Pass max_candidates, verify each query call receives it."""
        idx = _make_index()
        response = _make_query_response([_scored("v1", 0.5)])

        with patch.object(idx, "query", return_value=response) as mock_query:
            idx.query_namespaces(
                vector=[0.1],
                namespaces=["ns1"],
                metric="cosine",
                max_candidates=5000,
            )
            assert mock_query.call_count == 1
            assert mock_query.call_args_list[0].kwargs["max_candidates"] == 5000

    def test_query_namespaces_drn_params_default_none(self) -> None:
        """Call without DRN params, verify they default to None."""
        idx = _make_index()
        response = _make_query_response([_scored("v1", 0.5)])

        with patch.object(idx, "query", return_value=response) as mock_query:
            idx.query_namespaces(
                vector=[0.1],
                namespaces=["ns1"],
                metric="cosine",
            )
            assert mock_query.call_args_list[0].kwargs["scan_factor"] is None
            assert mock_query.call_args_list[0].kwargs["max_candidates"] is None
