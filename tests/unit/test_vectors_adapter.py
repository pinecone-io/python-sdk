"""Unit tests for VectorsAdapter."""

from __future__ import annotations

import msgspec
import pytest

from pinecone._internal.adapters.vectors_adapter import VectorsAdapter
from pinecone.errors.exceptions import PineconeError, ResponseParsingError
from pinecone.models.vectors.responses import (
    DescribeIndexStatsResponse,
    FetchByMetadataResponse,
    FetchResponse,
    ListResponse,
    QueryResponse,
    UpdateResponse,
    UpsertResponse,
)


class TestToUpsertResponse:
    """Tests for to_upsert_response."""

    def test_basic_upsert(self) -> None:
        data = msgspec.json.encode({"upsertedCount": 10})
        result = VectorsAdapter.to_upsert_response(data)
        assert isinstance(result, UpsertResponse)
        assert result.upserted_count == 10

    def test_upsert_zero_count(self) -> None:
        data = msgspec.json.encode({"upsertedCount": 0})
        result = VectorsAdapter.to_upsert_response(data)
        assert result.upserted_count == 0

    def test_upsert_forward_compatibility(self) -> None:
        data = msgspec.json.encode({"upsertedCount": 5, "futureField": "ignored"})
        result = VectorsAdapter.to_upsert_response(data)
        assert result.upserted_count == 5


class TestToQueryResponse:
    """Tests for to_query_response."""

    def test_basic_query(self) -> None:
        data = msgspec.json.encode(
            {
                "matches": [
                    {"id": "v1", "score": 0.95},
                    {"id": "v2", "score": 0.80},
                ],
                "namespace": "test-ns",
                "usage": {"readUnits": 5},
            }
        )
        result = VectorsAdapter.to_query_response(data)
        assert isinstance(result, QueryResponse)
        assert len(result.matches) == 2
        assert result.matches[0].id == "v1"
        assert result.matches[0].score == 0.95
        assert result.namespace == "test-ns"
        assert result.usage is not None
        assert result.usage.read_units == 5

    def test_query_with_values_and_metadata(self) -> None:
        data = msgspec.json.encode(
            {
                "matches": [
                    {
                        "id": "v1",
                        "score": 0.9,
                        "values": [0.1, 0.2, 0.3],
                        "sparseValues": {"indices": [0, 2], "values": [0.5, 0.7]},
                        "metadata": {"genre": "comedy"},
                    }
                ],
                "namespace": "ns",
            }
        )
        result = VectorsAdapter.to_query_response(data)
        match = result.matches[0]
        assert match.values == [0.1, 0.2, 0.3]
        assert match.sparse_values is not None
        assert match.sparse_values.indices == [0, 2]
        assert match.sparse_values.values == [0.5, 0.7]
        assert match.metadata == {"genre": "comedy"}

    def test_query_strips_deprecated_results_field(self) -> None:
        data = msgspec.json.encode(
            {
                "results": [{"some": "deprecated_data"}],
                "matches": [{"id": "v1", "score": 0.5}],
                "namespace": "ns",
            }
        )
        result = VectorsAdapter.to_query_response(data)
        assert len(result.matches) == 1
        assert result.matches[0].id == "v1"

    def test_query_null_namespace_becomes_empty_string(self) -> None:
        data = msgspec.json.encode(
            {
                "matches": [],
                "namespace": None,
            }
        )
        result = VectorsAdapter.to_query_response(data)
        assert result.namespace == ""

    def test_query_missing_namespace_defaults_empty(self) -> None:
        data = msgspec.json.encode({"matches": []})
        result = VectorsAdapter.to_query_response(data)
        assert result.namespace == ""

    def test_query_empty_matches(self) -> None:
        data = msgspec.json.encode({"matches": [], "namespace": "ns"})
        result = VectorsAdapter.to_query_response(data)
        assert result.matches == []

    def test_query_no_usage(self) -> None:
        data = msgspec.json.encode({"matches": [], "namespace": "ns"})
        result = VectorsAdapter.to_query_response(data)
        assert result.usage is None

    def test_query_preserves_api_order_ascending(self) -> None:
        """SDK must NOT reorder matches — euclidean backends return ascending scores."""
        data = msgspec.json.encode(
            {
                "matches": [
                    {"id": "v1", "score": 0.1},
                    {"id": "v2", "score": 0.5},
                    {"id": "v3", "score": 0.9},
                ],
                "namespace": "ns",
            }
        )
        result = VectorsAdapter.to_query_response(data)
        assert [m.id for m in result.matches] == ["v1", "v2", "v3"]
        assert [m.score for m in result.matches] == [0.1, 0.5, 0.9]

    def test_query_preserves_api_order_descending(self) -> None:
        """SDK must NOT reorder matches — cosine backends return descending scores."""
        data = msgspec.json.encode(
            {
                "matches": [
                    {"id": "v1", "score": 1.0},
                    {"id": "v2", "score": 0.9839},
                    {"id": "v3", "score": 0.9735},
                ],
                "namespace": "ns",
            }
        )
        result = VectorsAdapter.to_query_response(data)
        assert [m.id for m in result.matches] == ["v1", "v2", "v3"]
        assert result.matches[0].score == pytest.approx(1.0)
        assert result.matches[1].score == pytest.approx(0.9839)
        assert result.matches[2].score == pytest.approx(0.9735)


class TestToFetchResponse:
    """Tests for to_fetch_response."""

    def test_basic_fetch(self) -> None:
        data = msgspec.json.encode(
            {
                "vectors": {
                    "id-1": {"id": "id-1", "values": [1.0, 1.5]},
                    "id-2": {"id": "id-2", "values": [2.0, 1.0]},
                },
                "namespace": "test-ns",
                "usage": {"readUnits": 1},
            }
        )
        result = VectorsAdapter.to_fetch_response(data)
        assert isinstance(result, FetchResponse)
        assert len(result.vectors) == 2
        assert result.vectors["id-1"].id == "id-1"
        assert result.vectors["id-1"].values == [1.0, 1.5]
        assert result.namespace == "test-ns"
        assert result.usage is not None
        assert result.usage.read_units == 1

    def test_fetch_empty_vectors(self) -> None:
        data = msgspec.json.encode({"vectors": {}, "namespace": "ns", "usage": {"readUnits": 0}})
        result = VectorsAdapter.to_fetch_response(data)
        assert result.vectors == {}

    def test_fetch_with_sparse_and_metadata(self) -> None:
        data = msgspec.json.encode(
            {
                "vectors": {
                    "v1": {
                        "id": "v1",
                        "values": [0.1],
                        "sparseValues": {"indices": [3], "values": [0.9]},
                        "metadata": {"key": "val"},
                    }
                },
                "namespace": "ns",
            }
        )
        result = VectorsAdapter.to_fetch_response(data)
        v = result.vectors["v1"]
        assert v.sparse_values is not None
        assert v.sparse_values.indices == [3]
        assert v.metadata == {"key": "val"}

    def test_fetch_forward_compatibility(self) -> None:
        data = msgspec.json.encode(
            {
                "vectors": {},
                "namespace": "ns",
                "futureField": True,
            }
        )
        result = VectorsAdapter.to_fetch_response(data)
        assert result.namespace == "ns"

    def test_fetch_sparse_only_vector_has_empty_values(self) -> None:
        data = msgspec.json.encode(
            {
                "vectors": {
                    "v1": {
                        "id": "v1",
                        "values": [],
                        "sparseValues": {"indices": [3], "values": [0.9]},
                    }
                },
                "namespace": "ns",
            }
        )
        result = VectorsAdapter.to_fetch_response(data)
        v = result.vectors["v1"]
        assert v.values == []
        assert v.sparse_values is not None

    def test_fetch_vector_with_neither_values_nor_sparse_raises_pinecone_error(self) -> None:
        data = msgspec.json.encode(
            {"vectors": {"v1": {"id": "v1", "values": [], "sparseValues": None}}}
        )
        with pytest.raises(ResponseParsingError) as exc_info:
            VectorsAdapter.to_fetch_response(data)
        assert isinstance(exc_info.value, PineconeError)
        assert isinstance(exc_info.value.cause, msgspec.ValidationError)


class TestToFetchByMetadataResponse:
    """Tests for to_fetch_by_metadata_response."""

    def test_basic_fetch_by_metadata(self) -> None:
        data = msgspec.json.encode(
            {
                "vectors": {"id-1": {"id": "id-1", "values": [1.0, 1.5]}},
                "namespace": "test-ns",
            }
        )
        result = VectorsAdapter.to_fetch_by_metadata_response(data)
        assert isinstance(result, FetchByMetadataResponse)
        assert result.vectors["id-1"].values == [1.0, 1.5]

    def test_fetch_by_metadata_sparse_only_vector_has_empty_values(self) -> None:
        data = msgspec.json.encode(
            {
                "vectors": {
                    "v1": {
                        "id": "v1",
                        "values": [],
                        "sparseValues": {"indices": [3], "values": [0.9]},
                    }
                }
            }
        )
        result = VectorsAdapter.to_fetch_by_metadata_response(data)
        v = result.vectors["v1"]
        assert v.values == []
        assert v.sparse_values is not None

    def test_fetch_by_metadata_vector_with_neither_values_nor_sparse_raises_pinecone_error(
        self,
    ) -> None:
        data = msgspec.json.encode(
            {"vectors": {"v1": {"id": "v1", "values": [], "sparseValues": None}}}
        )
        with pytest.raises(ResponseParsingError) as exc_info:
            VectorsAdapter.to_fetch_by_metadata_response(data)
        assert isinstance(exc_info.value, PineconeError)
        assert isinstance(exc_info.value.cause, msgspec.ValidationError)


class TestToStatsResponse:
    """Tests for to_stats_response."""

    def test_basic_stats(self) -> None:
        data = msgspec.json.encode(
            {
                "namespaces": {
                    "": {"vectorCount": 50000},
                    "ns2": {"vectorCount": 30000},
                },
                "dimension": 1024,
                "indexFullness": 0.4,
                "totalVectorCount": 80000,
            }
        )
        result = VectorsAdapter.to_stats_response(data)
        assert isinstance(result, DescribeIndexStatsResponse)
        assert len(result.namespaces) == 2
        assert result.namespaces[""].vector_count == 50000
        assert result.namespaces["ns2"].vector_count == 30000
        assert result.dimension == 1024
        assert result.index_fullness == 0.4
        assert result.total_vector_count == 80000

    def test_stats_with_metric_and_vector_type(self) -> None:
        data = msgspec.json.encode(
            {
                "namespaces": {},
                "dimension": 256,
                "indexFullness": 0.0,
                "totalVectorCount": 0,
                "metric": "cosine",
                "vectorType": "dense",
            }
        )
        result = VectorsAdapter.to_stats_response(data)
        assert result.metric == "cosine"
        assert result.vector_type == "dense"

    def test_stats_empty_namespaces(self) -> None:
        data = msgspec.json.encode(
            {
                "namespaces": {},
                "dimension": 128,
                "indexFullness": 0.0,
                "totalVectorCount": 0,
            }
        )
        result = VectorsAdapter.to_stats_response(data)
        assert result.namespaces == {}


class TestToListResponse:
    """Tests for to_list_response."""

    def test_basic_list(self) -> None:
        data = msgspec.json.encode(
            {
                "vectors": [{"id": "doc1#abc"}, {"id": "doc1#def"}],
                "namespace": "ns",
                "usage": {"readUnits": 1},
            }
        )
        result = VectorsAdapter.to_list_response(data)
        assert isinstance(result, ListResponse)
        assert len(result.vectors) == 2
        assert result.vectors[0].id == "doc1#abc"
        assert result.namespace == "ns"

    def test_list_with_pagination(self) -> None:
        data = msgspec.json.encode(
            {
                "vectors": [{"id": "v1"}],
                "pagination": {"next": "abc123token"},
                "namespace": "ns",
            }
        )
        result = VectorsAdapter.to_list_response(data)
        assert result.pagination is not None
        assert result.pagination.next == "abc123token"

    def test_list_without_pagination(self) -> None:
        data = msgspec.json.encode(
            {
                "vectors": [{"id": "v1"}],
                "namespace": "ns",
            }
        )
        result = VectorsAdapter.to_list_response(data)
        assert result.pagination is None

    def test_list_empty_vectors(self) -> None:
        data = msgspec.json.encode({"vectors": [], "namespace": "ns"})
        result = VectorsAdapter.to_list_response(data)
        assert result.vectors == []


class TestToUpdateResponse:
    """Tests for to_update_response."""

    def test_update_with_matched_records(self) -> None:
        data = msgspec.json.encode({"matchedRecords": 42})
        result = VectorsAdapter.to_update_response(data)
        assert isinstance(result, UpdateResponse)
        assert result.matched_records == 42

    def test_update_without_matched_records(self) -> None:
        data = msgspec.json.encode({})
        result = VectorsAdapter.to_update_response(data)
        assert result.matched_records is None

    def test_update_forward_compatibility(self) -> None:
        data = msgspec.json.encode({"matchedRecords": 10, "newField": "test"})
        result = VectorsAdapter.to_update_response(data)
        assert result.matched_records == 10


class TestToDeleteResponse:
    """Tests for to_delete_response."""

    def test_delete_returns_none(self) -> None:
        data = msgspec.json.encode({})
        result = VectorsAdapter.to_delete_response(data)
        assert result is None
