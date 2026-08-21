"""Unit tests for vector and response models."""

from __future__ import annotations

from typing import Any

import msgspec
import pytest

from pinecone.errors import PineconeError, PineconeValueError
from pinecone.models.vectors.responses import (
    DescribeIndexStatsResponse,
    FetchResponse,
    ListItem,
    ListResponse,
    NamespaceSummary,
    QueryResponse,
    ResponseInfo,
    UpdateResponse,
    UpsertResponse,
)
from pinecone.models.vectors.sparse import SparseValues
from pinecone.models.vectors.usage import Usage
from pinecone.models.vectors.vector import ScoredVector, Vector
from tests.factories import (
    make_describe_index_stats_response,
    make_fetch_response,
    make_query_response,
    make_upsert_response,
)


class TestSparseValues:
    def test_construct(self) -> None:
        sv = SparseValues(indices=[0, 3, 7], values=[0.1, 0.2, 0.3])
        assert sv.indices == [0, 3, 7]
        assert sv.values == [0.1, 0.2, 0.3]

    def test_from_dict(self) -> None:
        data: dict[str, Any] = {"indices": [1, 2], "values": [0.5, 0.6]}
        sv = msgspec.convert(data, SparseValues)
        assert sv.indices == [1, 2]
        assert sv.values == [0.5, 0.6]

    def test_from_dict_via_classmethod(self) -> None:
        sv = SparseValues.from_dict({"indices": [0, 1], "values": [1.0, 2.0]})
        assert sv.indices == [0, 1]
        assert sv.values == [1.0, 2.0]


class TestVector:
    def test_minimal(self) -> None:
        v = Vector(id="vec-1", values=[1.0, 2.0])
        assert v.id == "vec-1"
        assert v.values == [1.0, 2.0]
        assert v.sparse_values is None
        assert v.metadata is None

    def test_with_values(self) -> None:
        v = Vector(id="vec-2", values=[1.0, 2.0, 3.0])
        assert v.values == [1.0, 2.0, 3.0]

    def test_with_sparse_values(self) -> None:
        sv = SparseValues(indices=[0], values=[1.0])
        v = Vector(id="vec-3", sparse_values=sv)
        assert v.sparse_values is not None
        assert v.sparse_values.indices == [0]

    def test_with_metadata(self) -> None:
        v = Vector(id="vec-4", values=[0.1], metadata={"genre": "comedy", "year": 2024})
        assert v.metadata == {"genre": "comedy", "year": 2024}

    def test_empty_vector_raises(self) -> None:
        # Vector with neither values nor sparse_values raises at construction time
        # (claim model-0042).
        with pytest.raises(PineconeValueError, match="values or sparse_values"):
            Vector(id="empty")

    def test_empty_vector_raises_pinecone_value_error(self) -> None:
        """``upsert`` documents ``PineconeValueError`` for a malformed vector element.

        The exact-type assertion is the load-bearing one: a plain
        ``pytest.raises(ValueError)`` passes whether or not this site raises the
        Pinecone type, because ``PineconeValueError`` subclasses ``ValueError``.
        """
        with pytest.raises(PineconeValueError) as exc_info:
            Vector(id="empty")
        assert type(exc_info.value) is PineconeValueError
        assert isinstance(exc_info.value, ValueError)

    def test_empty_vector_caught_as_pinecone_error(self) -> None:
        with pytest.raises(PineconeError):
            Vector(id="empty")

    def test_empty_vector_from_dict_raises_pinecone_value_error(self) -> None:
        with pytest.raises(PineconeValueError) as exc_info:
            Vector.from_dict({"id": "empty"})
        assert type(exc_info.value) is PineconeValueError

    def test_from_dict(self) -> None:
        data: dict[str, Any] = {
            "id": "vec-5",
            "values": [0.1, 0.2],
            "sparseValues": {"indices": [0], "values": [1.0]},
            "metadata": {"key": "val"},
        }
        v = msgspec.convert(data, Vector)
        assert v.id == "vec-5"
        assert v.values == [0.1, 0.2]
        assert v.sparse_values is not None
        assert v.metadata == {"key": "val"}

    def test_from_dict_via_classmethod(self) -> None:
        v = Vector.from_dict({"id": "v1", "values": [0.1, 0.2]})
        assert v.id == "v1"
        assert v.values == [0.1, 0.2]
        assert v.sparse_values is None
        assert v.metadata is None

    def test_from_dict_sparse_only_via_classmethod(self) -> None:
        v = Vector.from_dict(
            {"id": "v2", "sparse_values": {"indices": [0, 3], "values": [1.0, 2.0]}}
        )
        assert v.values == []
        assert v.sparse_values is not None
        assert v.sparse_values.indices == [0, 3]
        assert v.sparse_values.values == [1.0, 2.0]

    def test_from_dict_with_all_fields_via_classmethod(self) -> None:
        v = Vector.from_dict(
            {
                "id": "v3",
                "values": [0.5],
                "sparse_values": {"indices": [1], "values": [0.9]},
                "metadata": {"tag": "test"},
            }
        )
        assert v.id == "v3"
        assert v.values == [0.5]
        assert v.sparse_values is not None
        assert v.sparse_values.indices == [1]
        assert v.sparse_values.values == [0.9]
        assert v.metadata == {"tag": "test"}

    def test_from_dict_explicit_none_sparse_values(self) -> None:
        """Explicit sparse_values=None exercises the .get() is not None false branch."""
        v = Vector.from_dict({"id": "v3", "values": [0.1], "sparse_values": None})
        assert v.sparse_values is None

    def test_from_dict_missing_values_key(self) -> None:
        v = Vector.from_dict({"id": "v4", "sparse_values": {"indices": [0], "values": [1.0]}})
        assert v.values == []


class TestScoredVector:
    def test_minimal(self) -> None:
        sv = ScoredVector(id="match-1", score=0.95)
        assert sv.id == "match-1"
        assert sv.score == 0.95
        assert sv.values == []
        assert sv.sparse_values is None
        assert sv.metadata is None

    def test_with_all_fields(self) -> None:
        sparse = SparseValues(indices=[1], values=[0.5])
        sv = ScoredVector(
            id="match-2",
            score=0.8,
            values=[1.0, 2.0],
            sparse_values=sparse,
            metadata={"tag": "test"},
        )
        assert sv.score == 0.8
        assert sv.values == [1.0, 2.0]
        assert sv.sparse_values is not None
        assert sv.metadata == {"tag": "test"}


class TestUsage:
    def test_defaults(self) -> None:
        u = Usage()
        assert u.read_units is None
        assert u.write_units is None

    def test_with_values(self) -> None:
        u = Usage(read_units=5, write_units=10)
        assert u.read_units == 5
        assert u.write_units == 10


class TestUpsertResponse:
    def test_construct(self) -> None:
        r = UpsertResponse(upserted_count=42)
        assert r.upserted_count == 42

    def test_from_dict(self) -> None:
        data = make_upsert_response(upsertedCount=100)
        r = msgspec.convert(data, UpsertResponse)
        assert r.upserted_count == 100


class TestQueryResponse:
    def test_defaults(self) -> None:
        r = QueryResponse()
        assert r.matches == []
        assert r.namespace == ""
        assert r.usage is None

    def test_namespace_defaults_to_empty_string(self) -> None:
        """Per unified-rs-0013: null namespace becomes empty string."""
        data: dict[str, Any] = {"matches": [], "usage": {"readUnits": 1}}
        r = msgspec.convert(data, QueryResponse)
        assert r.namespace == ""

    def test_from_dict_with_matches(self) -> None:
        data = make_query_response()
        r = msgspec.convert(data, QueryResponse)
        assert len(r.matches) == 2
        assert r.matches[0].id == "vec-1"
        assert r.matches[0].score == 0.95
        assert r.matches[1].values == [0.4, 0.5, 0.6]
        assert r.namespace == "test-namespace"
        assert r.usage is not None
        assert r.usage.read_units == 5


class TestFetchResponse:
    def test_defaults(self) -> None:
        r = FetchResponse()
        assert r.vectors == {}
        assert r.namespace == ""
        assert r.usage is None

    def test_from_dict(self) -> None:
        data = make_fetch_response()
        r = msgspec.convert(data, FetchResponse)
        assert len(r.vectors) == 2
        assert r.vectors["id-1"].id == "id-1"
        assert r.vectors["id-1"].values == [1.0, 1.5]
        assert r.namespace == "test-namespace"
        assert r.usage is not None
        assert r.usage.read_units == 1


class TestDescribeIndexStatsResponse:
    def test_defaults(self) -> None:
        r = DescribeIndexStatsResponse()
        assert r.namespaces == {}
        assert r.dimension is None
        assert r.index_fullness == 0.0
        assert r.total_vector_count == 0

    def test_from_dict(self) -> None:
        data = make_describe_index_stats_response()
        r = msgspec.convert(data, DescribeIndexStatsResponse)
        assert len(r.namespaces) == 2
        assert r.namespaces["ns1"].vector_count == 100
        assert r.namespaces["ns2"].vector_count == 200
        assert r.dimension == 128
        assert r.index_fullness == 0.5
        assert r.total_vector_count == 300


class TestResponseInfo:
    def test_default(self) -> None:
        r = ResponseInfo()
        assert r.request_id is None

    def test_with_request_id(self) -> None:
        r = ResponseInfo(raw_headers={"x-pinecone-request-id": "abc-123"})
        assert r.request_id == "abc-123"


class TestNamespaceSummary:
    def test_default(self) -> None:
        ns = NamespaceSummary()
        assert ns.vector_count == 0

    def test_with_count(self) -> None:
        ns = NamespaceSummary(vector_count=500)
        assert ns.vector_count == 500


class TestListResponseIteration:
    """Tests for __iter__, __len__, and integer __getitem__ on ListResponse."""

    def test_list_response_iteration(self) -> None:
        item1 = ListItem(id="vec-1")
        item2 = ListItem(id="vec-2")
        response = ListResponse(vectors=[item1, item2], namespace="ns")
        collected = list(response)
        assert collected == [item1, item2]

    def test_list_response_len(self) -> None:
        response = ListResponse(
            vectors=[ListItem(id="a"), ListItem(id="b"), ListItem(id="c")],
            namespace="ns",
        )
        assert len(response) == 3

    def test_list_response_int_index(self) -> None:
        item1 = ListItem(id="first")
        item2 = ListItem(id="second")
        response = ListResponse(vectors=[item1, item2], namespace="ns")
        assert response[0] is item1
        assert response[1] is item2
        with pytest.raises(IndexError):
            response[5]

    def test_list_response_empty_iteration(self) -> None:
        response = ListResponse()
        assert list(response) == []
        assert len(response) == 0

    def test_list_response_string_access_still_works(self) -> None:
        response = ListResponse(namespace="test-ns")
        assert response["namespace"] == "test-ns"
        with pytest.raises(KeyError):
            response["bogus"]

    def test_list_response_contains_item(self) -> None:
        item = ListItem(id="vec-1")
        response = ListResponse(vectors=[item], namespace="ns")
        assert item in response  # item membership via __iter__
        assert "vectors" in response  # field-name check preserved
        assert "missing" not in response


class TestBracketAccess:
    """Tests for __getitem__ bracket access on data plane response models."""

    def test_upsert_response_bracket_access(self) -> None:
        r = UpsertResponse(upserted_count=42)
        assert r["upserted_count"] == 42

    def test_upsert_response_missing_key(self) -> None:
        r = UpsertResponse(upserted_count=42)
        with pytest.raises(KeyError, match="nonexistent"):
            r["nonexistent"]

    def test_query_response_bracket_access(self) -> None:
        r = QueryResponse(namespace="ns1")
        assert r["namespace"] == "ns1"
        assert r["matches"] == []
        assert r["usage"] is None

    def test_query_response_missing_key(self) -> None:
        r = QueryResponse()
        with pytest.raises(KeyError, match="bad_key"):
            r["bad_key"]

    def test_fetch_response_bracket_access(self) -> None:
        r = FetchResponse(namespace="ns1")
        assert r["namespace"] == "ns1"
        assert r["vectors"] == {}
        assert r["usage"] is None

    def test_fetch_response_missing_key(self) -> None:
        r = FetchResponse()
        with pytest.raises(KeyError, match="missing"):
            r["missing"]

    def test_describe_index_stats_response_bracket_access(self) -> None:
        r = DescribeIndexStatsResponse(dimension=128, total_vector_count=300)
        assert r["dimension"] == 128
        assert r["total_vector_count"] == 300
        assert r["index_fullness"] == 0.0
        assert r["namespaces"] == {}

    def test_describe_index_stats_response_missing_key(self) -> None:
        r = DescribeIndexStatsResponse()
        with pytest.raises(KeyError, match="nope"):
            r["nope"]

    def test_list_response_bracket_access(self) -> None:
        r = ListResponse(namespace="ns1")
        assert r["namespace"] == "ns1"
        assert r["vectors"] == []
        assert r["pagination"] is None

    def test_list_response_missing_key(self) -> None:
        r = ListResponse()
        with pytest.raises(KeyError, match="bogus"):
            r["bogus"]

    def test_update_response_bracket_access(self) -> None:
        r = UpdateResponse(matched_records=5)
        assert r["matched_records"] == 5

    def test_update_response_bracket_access_none(self) -> None:
        r = UpdateResponse()
        assert r["matched_records"] is None

    def test_update_response_missing_key(self) -> None:
        r = UpdateResponse()
        with pytest.raises(KeyError, match="invalid"):
            r["invalid"]
