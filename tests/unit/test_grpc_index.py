"""Unit tests for the GrpcIndex wrapper class."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest
import respx

from pinecone.errors.exceptions import (
    ApiError,
    ConflictError,
    ForbiddenError,
    NotFoundError,
    PineconeConnectionError,
    PineconeTimeoutError,
    ServiceError,
    UnauthorizedError,
    ValidationError,
)
from pinecone.grpc import GrpcIndex
from pinecone.grpc.future import PineconeFuture
from pinecone.models.vectors.responses import (
    DescribeIndexStatsResponse,
    FetchResponse,
    ListResponse,
    QueryResponse,
    UpdateResponse,
    UpsertRecordsResponse,
    UpsertResponse,
)
from pinecone.models.vectors.search import SearchRecordsResponse
from pinecone.models.vectors.vector import Vector

# The GrpcChannel import is a lazy import inside __init__, so we need to
# create a fake module to patch into sys.modules.
_MOCK_GRPC_MODULE_PATH = "pinecone._grpc"


def _make_grpc_index(mock_channel: MagicMock, **kwargs: Any) -> GrpcIndex:
    """Helper to create a GrpcIndex with a mocked GrpcChannel."""
    mock_module = MagicMock()
    mock_module.GrpcChannel.return_value = mock_channel
    with patch.dict("sys.modules", {_MOCK_GRPC_MODULE_PATH: mock_module}):
        return GrpcIndex(
            host=kwargs.pop("host", "test-index-abc123.svc.pinecone.io"),
            api_key=kwargs.pop("api_key", "test-api-key"),
            **kwargs,
        )


@pytest.fixture
def mock_channel() -> MagicMock:
    """Create a mock GrpcChannel."""
    return MagicMock()


@pytest.fixture
def grpc_index(mock_channel: MagicMock) -> GrpcIndex:
    """Create a GrpcIndex with a mocked GrpcChannel."""
    return _make_grpc_index(mock_channel)


class TestGrpcIndexInit:
    """Tests for GrpcIndex initialization."""

    def test_init_with_explicit_api_key(self, mock_channel: MagicMock) -> None:
        mock_module = MagicMock()
        mock_module.GrpcChannel.return_value = mock_channel
        with patch.dict("sys.modules", {_MOCK_GRPC_MODULE_PATH: mock_module}):
            idx = GrpcIndex(
                host="test-index.svc.pinecone.io",
                api_key="my-key",
            )
        assert idx.host == "https://test-index.svc.pinecone.io"
        mock_module.GrpcChannel.assert_called_once()
        call_args = mock_module.GrpcChannel.call_args
        assert call_args[0][1] == "my-key"  # api_key

    def test_init_with_env_api_key(
        self, mock_channel: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PINECONE_API_KEY", "env-key")
        mock_module = MagicMock()
        mock_module.GrpcChannel.return_value = mock_channel
        with patch.dict("sys.modules", {_MOCK_GRPC_MODULE_PATH: mock_module}):
            GrpcIndex(host="test-index.svc.pinecone.io")
        call_args = mock_module.GrpcChannel.call_args
        assert call_args[0][1] == "env-key"

    def test_init_no_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("PINECONE_API_KEY", raising=False)
        with pytest.raises(ValidationError, match="No API key provided"):
            GrpcIndex(host="test-index.svc.pinecone.io")

    def test_init_empty_host_raises(self) -> None:
        with pytest.raises(ValidationError):
            GrpcIndex(host="", api_key="test-key")

    def test_init_custom_timeouts(self, mock_channel: MagicMock) -> None:
        mock_module = MagicMock()
        mock_module.GrpcChannel.return_value = mock_channel
        with patch.dict("sys.modules", {_MOCK_GRPC_MODULE_PATH: mock_module}):
            GrpcIndex(
                host="test-index.svc.pinecone.io",
                api_key="test-key",
                timeout=60.0,
                connect_timeout=5.0,
            )
        call_args = mock_module.GrpcChannel.call_args
        assert call_args[0][5] == 60.0  # timeout
        assert call_args[0][6] == 5.0  # connect_timeout

    def test_init_insecure(self, mock_channel: MagicMock) -> None:
        mock_module = MagicMock()
        mock_module.GrpcChannel.return_value = mock_channel
        with patch.dict("sys.modules", {_MOCK_GRPC_MODULE_PATH: mock_module}):
            GrpcIndex(
                host="test-index.svc.pinecone.io",
                api_key="test-key",
                secure=False,
            )
        call_args = mock_module.GrpcChannel.call_args
        assert call_args[0][0].startswith("http://")  # endpoint
        assert call_args[0][4] is False  # secure


class TestUpsert:
    """Tests for GrpcIndex.upsert()."""

    def test_upsert_basic(self, grpc_index: GrpcIndex, mock_channel: MagicMock) -> None:
        mock_channel.upsert.return_value = {"upserted_count": 2}

        result = grpc_index.upsert(
            vectors=[
                {"id": "v1", "values": [0.1, 0.2]},
                {"id": "v2", "values": [0.3, 0.4]},
            ],
        )

        assert isinstance(result, UpsertResponse)
        assert result.upserted_count == 2
        mock_channel.upsert.assert_called_once()

    def test_upsert_with_namespace(self, grpc_index: GrpcIndex, mock_channel: MagicMock) -> None:
        mock_channel.upsert.return_value = {"upserted_count": 1}

        grpc_index.upsert(
            vectors=[{"id": "v1", "values": [0.1]}],
            namespace="my-ns",
        )

        call_args = mock_channel.upsert.call_args
        assert call_args[0][1] == "my-ns"

    def test_upsert_with_vector_objects(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        mock_channel.upsert.return_value = {"upserted_count": 1}

        grpc_index.upsert(
            vectors=[Vector(id="v1", values=[0.1, 0.2])],
        )

        call_args = mock_channel.upsert.call_args
        vec_dicts = call_args[0][0]
        assert vec_dicts[0]["id"] == "v1"
        assert vec_dicts[0]["values"] == [0.1, 0.2]

    def test_upsert_with_tuples(self, grpc_index: GrpcIndex, mock_channel: MagicMock) -> None:
        mock_channel.upsert.return_value = {"upserted_count": 1}

        grpc_index.upsert(
            vectors=[("v1", [0.1, 0.2])],
        )

        call_args = mock_channel.upsert.call_args
        vec_dicts = call_args[0][0]
        assert vec_dicts[0]["id"] == "v1"

    def test_upsert_with_vector_objects_preserves_sparse_values(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        from pinecone.models.vectors.sparse import SparseValues

        mock_channel.upsert.return_value = {"upserted_count": 1}

        grpc_index.upsert(
            vectors=[
                Vector(
                    id="v1",
                    values=[0.1, 0.2],
                    sparse_values=SparseValues(indices=[0, 2], values=[0.9, 0.1]),
                )
            ]
        )

        call_args = mock_channel.upsert.call_args
        vec_dict = call_args[0][0][0]
        assert vec_dict["sparse_values"] == {"indices": [0, 2], "values": [0.9, 0.1]}

    def test_upsert_with_vector_objects_preserves_metadata(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        mock_channel.upsert.return_value = {"upserted_count": 1}

        grpc_index.upsert(vectors=[Vector(id="v1", values=[0.1], metadata={"topic": "ai"})])

        call_args = mock_channel.upsert.call_args
        vec_dict = call_args[0][0][0]
        assert vec_dict["metadata"] == {"topic": "ai"}

    def test_upsert_empty_namespace_passes_none(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        mock_channel.upsert.return_value = {"upserted_count": 1}

        grpc_index.upsert(vectors=[{"id": "v1", "values": [0.1]}])

        call_args = mock_channel.upsert.call_args
        assert call_args[0][1] is None  # empty string -> None for GrpcChannel


class TestQuery:
    """Tests for GrpcIndex.query()."""

    def test_query_by_vector(self, grpc_index: GrpcIndex, mock_channel: MagicMock) -> None:
        mock_channel.query.return_value = {
            "matches": [
                {"id": "v1", "score": 0.95, "values": [0.1, 0.2], "metadata": {"key": "val"}},
                {"id": "v2", "score": 0.85, "values": [], "metadata": None},
            ],
            "namespace": "test-ns",
            "usage": {"read_units": 5},
        }

        result = grpc_index.query(top_k=2, vector=[0.1, 0.2], namespace="test-ns")

        assert isinstance(result, QueryResponse)
        assert len(result.matches) == 2
        assert result.matches[0].id == "v1"
        assert result.matches[0].score == 0.95
        assert result.namespace == "test-ns"
        assert result.usage is not None
        assert result.usage.read_units == 5

    def test_query_by_id(self, grpc_index: GrpcIndex, mock_channel: MagicMock) -> None:
        mock_channel.query.return_value = {
            "matches": [{"id": "v2", "score": 0.9, "values": [], "metadata": None}],
            "namespace": "",
        }

        result = grpc_index.query(top_k=1, id="v1")

        assert len(result.matches) == 1
        mock_channel.query.assert_called_once_with(
            1,
            vector=None,
            id="v1",
            namespace=None,
            filter=None,
            include_values=False,
            include_metadata=False,
            sparse_vector=None,
            scan_factor=None,
            max_candidates=None,
            timeout_s=None,
        )

    def test_query_validates_top_k(self, grpc_index: GrpcIndex) -> None:
        with pytest.raises(ValidationError, match="top_k must be between"):
            grpc_index.query(top_k=0, vector=[0.1])

    def test_query_validates_both_vector_and_id(self, grpc_index: GrpcIndex) -> None:
        with pytest.raises(ValidationError, match="not both"):
            grpc_index.query(top_k=1, vector=[0.1], id="v1")

    def test_query_validates_neither_vector_nor_id(self, grpc_index: GrpcIndex) -> None:
        with pytest.raises(ValidationError, match="At least one of vector, id, or sparse_vector"):
            grpc_index.query(top_k=1)

    def test_query_with_filter(self, grpc_index: GrpcIndex, mock_channel: MagicMock) -> None:
        mock_channel.query.return_value = {"matches": [], "namespace": ""}

        grpc_index.query(
            top_k=5,
            vector=[0.1],
            filter={"genre": {"$eq": "drama"}},
        )

        call_kwargs = mock_channel.query.call_args
        assert call_kwargs[1]["filter"] == {"genre": {"$eq": "drama"}}

    def test_query_no_usage(self, grpc_index: GrpcIndex, mock_channel: MagicMock) -> None:
        mock_channel.query.return_value = {"matches": [], "namespace": ""}

        result = grpc_index.query(top_k=1, vector=[0.1])
        assert result.usage is None

    def test_query_with_sparse_values(self, grpc_index: GrpcIndex, mock_channel: MagicMock) -> None:
        mock_channel.query.return_value = {
            "matches": [
                {
                    "id": "v1",
                    "score": 0.9,
                    "values": [],
                    "sparse_values": {"indices": [0, 2], "values": [0.5, 0.3]},
                    "metadata": None,
                }
            ],
            "namespace": "",
        }

        result = grpc_index.query(top_k=1, vector=[0.1])
        assert result.matches[0].sparse_values is not None
        assert result.matches[0].sparse_values.indices == [0, 2]
        assert result.matches[0].sparse_values.values == [0.5, 0.3]


class TestFetch:
    """Tests for GrpcIndex.fetch()."""

    def test_fetch_basic(self, grpc_index: GrpcIndex, mock_channel: MagicMock) -> None:
        mock_channel.fetch.return_value = {
            "vectors": {
                "v1": {"values": [0.1, 0.2], "metadata": {"key": "val"}},
                "v2": {"values": [0.3, 0.4], "metadata": None},
            },
            "namespace": "test-ns",
            "usage": {"read_units": 2},
        }

        result = grpc_index.fetch(ids=["v1", "v2"], namespace="test-ns")

        assert isinstance(result, FetchResponse)
        assert len(result.vectors) == 2
        assert result.vectors["v1"].id == "v1"
        assert result.vectors["v1"].values == [0.1, 0.2]
        assert result.vectors["v1"].metadata == {"key": "val"}
        assert result.namespace == "test-ns"
        assert result.usage is not None
        assert result.usage.read_units == 2

    def test_fetch_empty_ids_raises(self, grpc_index: GrpcIndex) -> None:
        with pytest.raises(ValidationError, match="ids must be a non-empty list"):
            grpc_index.fetch(ids=[])

    def test_fetch_missing_vectors(self, grpc_index: GrpcIndex, mock_channel: MagicMock) -> None:
        mock_channel.fetch.return_value = {
            "vectors": {},
            "namespace": "",
        }

        result = grpc_index.fetch(ids=["nonexistent"])
        assert len(result.vectors) == 0

    def test_fetch_inflates_sparse_values_from_response(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        from pinecone.models.vectors.sparse import SparseValues

        mock_channel.fetch.return_value = {
            "vectors": {
                "v1": {
                    "values": [0.1, 0.2],
                    "sparse_values": {"indices": [0, 2], "values": [0.5, 0.3]},
                    "metadata": None,
                }
            }
        }

        result = grpc_index.fetch(ids=["v1"])

        assert result.vectors["v1"].sparse_values == SparseValues(indices=[0, 2], values=[0.5, 0.3])


class TestDelete:
    """Tests for GrpcIndex.delete()."""

    def test_delete_by_ids(self, grpc_index: GrpcIndex, mock_channel: MagicMock) -> None:
        mock_channel.delete.return_value = {}

        grpc_index.delete(ids=["v1", "v2"])

        mock_channel.delete.assert_called_once_with(
            ids=["v1", "v2"],
            delete_all=False,
            namespace=None,
            filter=None,
            timeout_s=None,
        )

    def test_delete_by_ids_with_namespace(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        mock_channel.delete.return_value = {}

        grpc_index.delete(ids=["qadg-v1"], namespace="ns-abc12345")

        # delete_all must be False and namespace must be forwarded as-is;
        # the proto3 false default for delete_all must not become True.
        mock_channel.delete.assert_called_once_with(
            ids=["qadg-v1"],
            delete_all=False,
            namespace="ns-abc12345",
            filter=None,
            timeout_s=None,
        )

    def test_delete_all(self, grpc_index: GrpcIndex, mock_channel: MagicMock) -> None:
        mock_channel.delete.return_value = {}

        grpc_index.delete(delete_all=True, namespace="my-ns")

        mock_channel.delete.assert_called_once_with(
            ids=None,
            delete_all=True,
            namespace="my-ns",
            filter=None,
            timeout_s=None,
        )

    def test_delete_by_filter(self, grpc_index: GrpcIndex, mock_channel: MagicMock) -> None:
        mock_channel.delete.return_value = {}

        grpc_index.delete(filter={"genre": {"$eq": "obsolete"}})

        mock_channel.delete.assert_called_once_with(
            ids=None,
            delete_all=False,
            namespace=None,
            filter={"genre": {"$eq": "obsolete"}},
            timeout_s=None,
        )

    def test_delete_no_mode_raises(self, grpc_index: GrpcIndex) -> None:
        with pytest.raises(ValidationError, match="Must specify one"):
            grpc_index.delete()

    def test_delete_multiple_modes_raises(self, grpc_index: GrpcIndex) -> None:
        with pytest.raises(ValidationError, match="Cannot combine"):
            grpc_index.delete(ids=["v1"], delete_all=True)


class TestUpdate:
    """Tests for GrpcIndex.update()."""

    def test_update_by_id(self, grpc_index: GrpcIndex, mock_channel: MagicMock) -> None:
        mock_channel.update.return_value = {"matched_records": 1}

        result = grpc_index.update(id="v1", values=[0.1, 0.2])

        assert isinstance(result, UpdateResponse)
        assert result.matched_records == 1

    def test_update_with_metadata(self, grpc_index: GrpcIndex, mock_channel: MagicMock) -> None:
        mock_channel.update.return_value = {}

        result = grpc_index.update(id="v1", set_metadata={"key": "new_val"})

        assert result.matched_records is None
        call_kwargs = mock_channel.update.call_args[1]
        assert call_kwargs["set_metadata"] == {"key": "new_val"}

    def test_update_with_sparse_values_model(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        from pinecone.models.vectors.sparse import SparseValues

        mock_channel.update.return_value = {}

        grpc_index.update(
            id="v1",
            sparse_values=SparseValues(indices=[0, 2], values=[0.5, 0.3]),
        )

        call_kwargs = mock_channel.update.call_args[1]
        assert call_kwargs["sparse_values"] == {"indices": [0, 2], "values": [0.5, 0.3]}

    def test_update_with_sparse_values_dict(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        mock_channel.update.return_value = {}

        sv_dict: dict[str, Any] = {"indices": [1, 3], "values": [0.7, 0.1]}
        grpc_index.update(id="v1", sparse_values=sv_dict)

        call_kwargs = mock_channel.update.call_args[1]
        assert call_kwargs["sparse_values"] == sv_dict

    def test_update_validates_both_id_and_filter(self, grpc_index: GrpcIndex) -> None:
        with pytest.raises(ValidationError, match="not both"):
            grpc_index.update(id="v1", filter={"key": "val"})

    def test_update_validates_neither_id_nor_filter(self, grpc_index: GrpcIndex) -> None:
        with pytest.raises(ValidationError, match="got neither"):
            grpc_index.update(values=[0.1])

    def test_update_by_filter_with_dry_run(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        mock_channel.update.return_value = {"matched_records": 42}

        result = grpc_index.update(
            filter={"genre": {"$eq": "drama"}},
            set_metadata={"year": 2020},
            dry_run=True,
        )

        assert result.matched_records == 42
        call_kwargs = mock_channel.update.call_args[1]
        assert call_kwargs["dry_run"] is True


class TestList:
    """Tests for GrpcIndex.list() and list_paginated()."""

    def test_list_paginated_basic(self, grpc_index: GrpcIndex, mock_channel: MagicMock) -> None:
        mock_channel.list.return_value = {
            "vectors": [{"id": "v1"}, {"id": "v2"}],
            "namespace": "test-ns",
            "usage": {"read_units": 1},
        }

        result = grpc_index.list_paginated(namespace="test-ns")

        assert isinstance(result, ListResponse)
        assert len(result.vectors) == 2
        assert result.vectors[0].id == "v1"
        assert result.namespace == "test-ns"
        assert result.usage is not None
        assert result.pagination is None

    def test_list_paginated_with_pagination(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        mock_channel.list.return_value = {
            "vectors": [{"id": "v1"}],
            "namespace": "",
            "pagination": {"next": "token123"},
        }

        result = grpc_index.list_paginated()

        assert result.pagination is not None
        assert result.pagination.next == "token123"

    def test_list_auto_paginates(self, grpc_index: GrpcIndex, mock_channel: MagicMock) -> None:
        mock_channel.list.side_effect = [
            {
                "vectors": [{"id": "v1"}],
                "namespace": "",
                "pagination": {"next": "page2"},
            },
            {
                "vectors": [{"id": "v2"}],
                "namespace": "",
            },
        ]

        pages = list(grpc_index.list())

        assert len(pages) == 2
        assert pages[0].vectors[0].id == "v1"
        assert pages[1].vectors[0].id == "v2"

    def test_list_stops_on_empty(self, grpc_index: GrpcIndex, mock_channel: MagicMock) -> None:
        mock_channel.list.return_value = {
            "vectors": [],
            "namespace": "",
        }

        pages = list(grpc_index.list())
        assert len(pages) == 0

    def test_list_returns_iterator(self, grpc_index: GrpcIndex, mock_channel: MagicMock) -> None:
        mock_channel.list.return_value = {
            "vectors": [{"id": "v1"}],
            "namespace": "",
        }

        result = grpc_index.list()
        assert isinstance(result, Iterator)


class TestDescribeIndexStats:
    """Tests for GrpcIndex.describe_index_stats()."""

    def test_describe_basic(self, grpc_index: GrpcIndex, mock_channel: MagicMock) -> None:
        mock_channel.describe_index_stats.return_value = {
            "namespaces": {
                "ns1": {"vector_count": 100},
                "ns2": {"vector_count": 200},
            },
            "dimension": 128,
            "index_fullness": 0.5,
            "total_vector_count": 300,
            "metric": "cosine",
            "vector_type": "dense",
            "memory_fullness": 0.3,
            "storage_fullness": 0.4,
        }

        result = grpc_index.describe_index_stats()

        assert isinstance(result, DescribeIndexStatsResponse)
        assert len(result.namespaces) == 2
        assert result.namespaces["ns1"].vector_count == 100
        assert result.namespaces["ns2"].vector_count == 200
        assert result.dimension == 128
        assert result.index_fullness == 0.5
        assert result.total_vector_count == 300
        assert result.metric == "cosine"
        assert result.vector_type == "dense"
        assert result.memory_fullness == 0.3
        assert result.storage_fullness == 0.4

    def test_describe_with_filter(self, grpc_index: GrpcIndex, mock_channel: MagicMock) -> None:
        mock_channel.describe_index_stats.return_value = {
            "namespaces": {},
            "dimension": 64,
            "index_fullness": 0.0,
            "total_vector_count": 0,
        }

        grpc_index.describe_index_stats(filter={"genre": {"$eq": "drama"}})

        mock_channel.describe_index_stats.assert_called_once_with(
            filter={"genre": {"$eq": "drama"}},
            timeout_s=None,
        )

    def test_describe_minimal_response(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        mock_channel.describe_index_stats.return_value = {
            "namespaces": {},
            "index_fullness": 0.0,
            "total_vector_count": 0,
        }

        result = grpc_index.describe_index_stats()

        assert result.dimension is None
        assert result.metric is None
        assert result.vector_type is None
        assert result.memory_fullness is None
        assert result.storage_fullness is None


class TestGrpcIndexClose:
    """Tests for GrpcIndex.close()."""

    def test_close_shuts_down_executor_before_channel(self, mock_channel: MagicMock) -> None:
        call_order: list[str] = []

        mock_executor = MagicMock()
        mock_executor.shutdown.side_effect = lambda wait: call_order.append(
            f"shutdown(wait={wait})"
        )
        mock_channel.close.side_effect = lambda: call_order.append("channel.close()")

        mock_http = MagicMock()
        mock_http.close.side_effect = lambda: call_order.append("http.close()")

        idx = _make_grpc_index(mock_channel)
        idx._executor = mock_executor
        idx._http = mock_http

        idx.close()

        assert call_order == ["shutdown(wait=True)", "http.close()", "channel.close()"]
        mock_executor.shutdown.assert_called_once_with(wait=True)
        mock_http.close.assert_called_once()


class TestGrpcFutureReturningMethods:
    """Tests for GrpcIndex.*_async() future-returning methods."""

    def test_upsert_async_returns_pinecone_future(self, grpc_index: GrpcIndex) -> None:
        mock_submit = MagicMock(return_value=MagicMock())
        grpc_index._executor.submit = mock_submit  # type: ignore[method-assign]

        vectors = [{"id": "v1", "values": [0.1, 0.2]}]
        result = grpc_index.upsert_async(vectors=vectors, namespace="ns", timeout=12.5)

        assert isinstance(result, PineconeFuture)
        assert mock_submit.call_args[0][0] == grpc_index.upsert
        assert mock_submit.call_args.kwargs == {
            "vectors": vectors,
            "namespace": "ns",
            "timeout": 12.5,
        }

    def test_query_async_returns_pinecone_future(self, grpc_index: GrpcIndex) -> None:
        mock_submit = MagicMock(return_value=MagicMock())
        grpc_index._executor.submit = mock_submit  # type: ignore[method-assign]

        vec = [0.1] * 4
        result = grpc_index.query_async(top_k=5, vector=vec, filter={"a": 1}, timeout=7.0)

        assert isinstance(result, PineconeFuture)
        assert mock_submit.call_args[0][0] == grpc_index.query
        assert mock_submit.call_args.kwargs == {
            "top_k": 5,
            "vector": vec,
            "id": None,
            "namespace": "",
            "filter": {"a": 1},
            "include_values": False,
            "include_metadata": False,
            "sparse_vector": None,
            "scan_factor": None,
            "max_candidates": None,
            "timeout": 7.0,
        }

    def test_fetch_async_returns_pinecone_future(self, grpc_index: GrpcIndex) -> None:
        mock_submit = MagicMock(return_value=MagicMock())
        grpc_index._executor.submit = mock_submit  # type: ignore[method-assign]

        result = grpc_index.fetch_async(ids=["a", "b"], namespace="ns", timeout=3.0)

        assert isinstance(result, PineconeFuture)
        assert mock_submit.call_args[0][0] == grpc_index.fetch
        assert mock_submit.call_args.kwargs == {
            "ids": ["a", "b"],
            "namespace": "ns",
            "timeout": 3.0,
        }

    def test_delete_async_returns_pinecone_future(self, grpc_index: GrpcIndex) -> None:
        mock_submit = MagicMock(return_value=MagicMock())
        grpc_index._executor.submit = mock_submit  # type: ignore[method-assign]

        result = grpc_index.delete_async(
            ids=["a"],
            delete_all=False,
            filter=None,
            namespace="ns",
            timeout=4.0,
        )

        assert isinstance(result, PineconeFuture)
        assert mock_submit.call_args[0][0] == grpc_index.delete
        assert mock_submit.call_args.kwargs == {
            "ids": ["a"],
            "delete_all": False,
            "filter": None,
            "namespace": "ns",
            "timeout": 4.0,
        }

    def test_upsert_async_timeout_defaults_to_none(self, grpc_index: GrpcIndex) -> None:
        mock_submit = MagicMock(return_value=MagicMock())
        grpc_index._executor.submit = mock_submit  # type: ignore[method-assign]

        grpc_index.upsert_async(vectors=[{"id": "v1", "values": [0.1]}])

        assert mock_submit.call_args.kwargs["timeout"] is None

    def test_update_async_returns_pinecone_future(self, grpc_index: GrpcIndex) -> None:
        mock_submit = MagicMock(return_value=MagicMock())
        grpc_index._executor.submit = mock_submit  # type: ignore[method-assign]
        result = grpc_index.update_async(id="v1", values=[0.1, 0.2])
        assert isinstance(result, PineconeFuture)
        assert mock_submit.call_args[0][0] == grpc_index.update
        assert mock_submit.call_args.kwargs["id"] == "v1"
        assert mock_submit.call_args.kwargs["values"] == [0.1, 0.2]


class TestGrpcExceptionPropagation:
    """Tests that typed exceptions from the Rust gRPC layer propagate unchanged.

    The old _call_channel wrapper is gone. Rust's status_to_py_err now raises
    typed pinecone.errors exceptions directly. Python just calls
    self._channel.method(...) and the exception propagates unmodified.
    """

    def test_connection_error_propagates_unchanged(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        exc = PineconeConnectionError("gRPC UNAVAILABLE: connection refused")
        mock_channel.upsert.side_effect = exc

        with pytest.raises(PineconeConnectionError) as exc_info:
            grpc_index.upsert(vectors=[{"id": "v1", "values": [0.1]}])

        assert exc_info.value is exc

    def test_not_found_error_propagates_unchanged(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        exc = NotFoundError(
            message="index 'foo' does not exist",
            status_code=404,
            body={"error": {"code": "NOT_FOUND", "message": "index 'foo' does not exist"}},
            error_code="NOT_FOUND",
        )
        mock_channel.fetch.side_effect = exc

        with pytest.raises(NotFoundError) as exc_info:
            grpc_index.fetch(ids=["v1"])

        assert exc_info.value is exc
        assert exc_info.value.status_code == 404

    def test_api_error_propagates_unchanged(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        """gRPC INVALID_ARGUMENT maps to ApiError(status_code=400) at the Rust layer."""
        exc = ApiError(
            "Vector dimension 3 does not match the dimension of the index 2",
            400,
            {
                "error": {
                    "code": "INVALID_ARGUMENT",
                    "message": "Vector dimension 3 does not match the dimension of the index 2",
                }
            },
            error_code="INVALID_ARGUMENT",
        )
        mock_channel.upsert.side_effect = exc

        with pytest.raises(ApiError) as exc_info:
            grpc_index.upsert(vectors=[{"id": "v1", "values": [0.1, 0.2, 0.3]}])

        assert exc_info.value is exc
        assert exc_info.value.status_code == 400

    def test_timeout_error_propagates_unchanged(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        exc = PineconeTimeoutError("deadline exceeded after 20s")
        mock_channel.query.side_effect = exc

        with pytest.raises(PineconeTimeoutError) as exc_info:
            grpc_index.query(top_k=1, vector=[0.1])

        assert exc_info.value is exc

    def test_cancelled_timeout_expired_propagates_as_pinecone_timeout_error(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        # tonic converts per-request req.set_timeout() expiry via TimeoutExpired into
        # Status::cancelled("Timeout expired") instead of DeadlineExceeded.  The Rust
        # transport maps that specific Cancelled+message to PineconeTimeoutError so that
        # callers see the typed subclass rather than the base PineconeError.
        exc = PineconeTimeoutError("Timeout expired")
        mock_channel.query.side_effect = exc

        with pytest.raises(PineconeTimeoutError) as exc_info:
            grpc_index.query(top_k=1, vector=[0.1])

        assert exc_info.value is exc
        assert str(exc_info.value) == "Timeout expired"

    def test_unauthorized_error_propagates_unchanged(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        exc = UnauthorizedError(
            message="invalid api key",
            status_code=401,
            body={"error": {"code": "UNAUTHENTICATED", "message": "invalid api key"}},
            error_code="UNAUTHENTICATED",
        )
        mock_channel.query.side_effect = exc

        with pytest.raises(UnauthorizedError) as exc_info:
            grpc_index.query(top_k=1, vector=[0.1])

        assert exc_info.value is exc

    def test_forbidden_error_propagates_unchanged(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        exc = ForbiddenError(
            message="access denied",
            status_code=403,
            body={"error": {"code": "PERMISSION_DENIED", "message": "access denied"}},
            error_code="PERMISSION_DENIED",
        )
        mock_channel.describe_index_stats.side_effect = exc

        with pytest.raises(ForbiddenError) as exc_info:
            grpc_index.describe_index_stats()

        assert exc_info.value is exc

    def test_conflict_error_propagates_unchanged(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        exc = ConflictError(
            message="resource already exists",
            status_code=409,
            body={"error": {"code": "ALREADY_EXISTS", "message": "resource already exists"}},
            error_code="ALREADY_EXISTS",
        )
        mock_channel.upsert.side_effect = exc

        with pytest.raises(ConflictError) as exc_info:
            grpc_index.upsert(vectors=[{"id": "v1", "values": [0.1]}])

        assert exc_info.value is exc

    def test_service_error_propagates_unchanged(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        exc = ServiceError(
            message="internal server error",
            status_code=500,
            body={"error": {"code": "INTERNAL", "message": "internal server error"}},
            error_code="INTERNAL",
        )
        mock_channel.fetch.side_effect = exc

        with pytest.raises(ServiceError) as exc_info:
            grpc_index.fetch(ids=["v1"])

        assert exc_info.value is exc

    def test_response_parsing_error_propagates_unchanged(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        """ResponseParsingError from Rust proto decode propagates unchanged."""
        from pinecone.errors.exceptions import ResponseParsingError

        exc = ResponseParsingError("vector missing 'id'")
        mock_channel.upsert.side_effect = exc

        with pytest.raises(ResponseParsingError) as exc_info:
            grpc_index.upsert(vectors=[{"id": "v1", "values": [0.1]}])

        assert exc_info.value is exc

    def test_no_double_bracket_in_propagated_not_found_error(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        """Verify the old double-bracket bug is gone.

        Previously, _call_channel would call str(exc) on a Rust-raised
        NotFoundError (producing "[404 NOT_FOUND] index not found") and use
        that string as the message for a new NotFoundError, yielding
        "[404 NOT_FOUND] [404 NOT_FOUND] index not found" in str().
        Now the typed exception from Rust propagates unchanged, so there is
        exactly one bracket pair.
        """
        exc = NotFoundError(
            message="index 'foo' does not exist",
            status_code=404,
            body={"error": {"code": "NOT_FOUND", "message": "index 'foo' does not exist"}},
            error_code="NOT_FOUND",
        )
        mock_channel.fetch.side_effect = exc

        with pytest.raises(NotFoundError) as exc_info:
            grpc_index.fetch(ids=["v1"])

        assert str(exc_info.value).count("[") <= 1


# ---------------------------------------------------------------------------
# Records API — REST fallback in GrpcIndex
# ---------------------------------------------------------------------------

_INDEX_HOST = "test-index-abc123.svc.pinecone.io"
_INDEX_HOST_HTTPS = f"https://{_INDEX_HOST}"
_UPSERT_RECORDS_URL = f"{_INDEX_HOST_HTTPS}/records/namespaces/test-ns/upsert"
_SEARCH_URL = f"{_INDEX_HOST_HTTPS}/records/namespaces/test-ns/search"

_SEARCH_RESPONSE: dict[str, object] = {
    "result": {
        "hits": [
            {"_id": "doc-1", "_score": 0.95, "fields": {"text": "vector databases"}},
            {"_id": "doc-2", "_score": 0.80, "fields": {"text": "similarity search"}},
        ]
    },
    "usage": {"read_units": 3, "embed_total_tokens": 12},
}


class TestGrpcIndexUpsertRecords:
    """GrpcIndex.upsert_records() delegates to REST (NDJSON) endpoint."""

    @respx.mock
    def test_upsert_records_basic(self, mock_channel: MagicMock) -> None:
        respx.post(_UPSERT_RECORDS_URL).mock(return_value=httpx.Response(201))
        idx = _make_grpc_index(mock_channel, host=_INDEX_HOST)

        result = idx.upsert_records(
            namespace="test-ns",
            records=[{"_id": "r1", "text": "hello"}, {"_id": "r2", "text": "world"}],
        )

        assert isinstance(result, UpsertRecordsResponse)
        assert result.record_count == 2

    @respx.mock
    def test_upsert_records_ndjson_content_type(self, mock_channel: MagicMock) -> None:
        route = respx.post(_UPSERT_RECORDS_URL).mock(return_value=httpx.Response(201))
        idx = _make_grpc_index(mock_channel, host=_INDEX_HOST)
        idx.upsert_records(namespace="test-ns", records=[{"_id": "r1", "text": "hello"}])

        assert route.calls.last.request.headers["Content-Type"] == "application/x-ndjson"

    @respx.mock
    def test_upsert_records_ndjson_body(self, mock_channel: MagicMock) -> None:
        route = respx.post(_UPSERT_RECORDS_URL).mock(return_value=httpx.Response(201))
        idx = _make_grpc_index(mock_channel, host=_INDEX_HOST)
        idx.upsert_records(
            namespace="test-ns",
            records=[{"_id": "r1", "text": "hello"}, {"_id": "r2", "text": "world"}],
        )

        body = route.calls.last.request.content.decode("utf-8")
        lines = body.strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["_id"] == "r1"
        assert json.loads(lines[1])["_id"] == "r2"

    @respx.mock
    def test_upsert_records_id_alias_normalized(self, mock_channel: MagicMock) -> None:
        """Records with 'id' key are normalized to '_id' in the NDJSON body."""
        route = respx.post(_UPSERT_RECORDS_URL).mock(return_value=httpx.Response(201))
        idx = _make_grpc_index(mock_channel, host=_INDEX_HOST)
        idx.upsert_records(namespace="test-ns", records=[{"id": "r1", "text": "hello"}])

        body = route.calls.last.request.content.decode("utf-8")
        parsed = json.loads(body.strip())
        assert parsed["_id"] == "r1"
        assert "id" not in parsed

    def test_upsert_records_empty_namespace_raises(self, mock_channel: MagicMock) -> None:
        idx = _make_grpc_index(mock_channel, host=_INDEX_HOST)
        with pytest.raises(ValidationError, match="non-empty"):
            idx.upsert_records(namespace="", records=[{"_id": "r1"}])

    def test_upsert_records_non_string_namespace_raises(self, mock_channel: MagicMock) -> None:
        idx = _make_grpc_index(mock_channel, host=_INDEX_HOST)
        with pytest.raises(ValidationError, match="namespace must be a string"):
            idx.upsert_records(namespace=42, records=[{"_id": "r1"}])  # type: ignore[arg-type]

    def test_upsert_records_empty_records_raises(self, mock_channel: MagicMock) -> None:
        idx = _make_grpc_index(mock_channel, host=_INDEX_HOST)
        with pytest.raises(ValidationError, match="non-empty"):
            idx.upsert_records(namespace="test-ns", records=[])

    def test_upsert_records_missing_id_raises(self, mock_channel: MagicMock) -> None:
        idx = _make_grpc_index(mock_channel, host=_INDEX_HOST)
        with pytest.raises(ValidationError, match=r"_id.*id"):
            idx.upsert_records(namespace="test-ns", records=[{"text": "no id here"}])


class TestGrpcIndexSearch:
    """GrpcIndex.search() delegates to REST endpoint."""

    @respx.mock
    def test_search_with_text_inputs(self, mock_channel: MagicMock) -> None:
        respx.post(_SEARCH_URL).mock(return_value=httpx.Response(200, json=_SEARCH_RESPONSE))
        idx = _make_grpc_index(mock_channel, host=_INDEX_HOST)

        response = idx.search(namespace="test-ns", top_k=5, inputs={"text": "vector search"})

        assert isinstance(response, SearchRecordsResponse)
        assert len(response.result.hits) == 2
        assert response.result.hits[0].id == "doc-1"
        assert response.result.hits[0].score == pytest.approx(0.95)
        assert response.usage.read_units == 3

    @respx.mock
    def test_search_request_body_inputs(self, mock_channel: MagicMock) -> None:
        route = respx.post(_SEARCH_URL).mock(
            return_value=httpx.Response(200, json=_SEARCH_RESPONSE)
        )
        idx = _make_grpc_index(mock_channel, host=_INDEX_HOST)
        idx.search(namespace="test-ns", top_k=5, inputs={"text": "vector search"})

        import orjson

        body = orjson.loads(route.calls.last.request.content)
        assert body["query"]["top_k"] == 5
        assert body["query"]["inputs"] == {"text": "vector search"}

    @respx.mock
    def test_search_with_vector(self, mock_channel: MagicMock) -> None:
        route = respx.post(_SEARCH_URL).mock(
            return_value=httpx.Response(200, json=_SEARCH_RESPONSE)
        )
        idx = _make_grpc_index(mock_channel, host=_INDEX_HOST)
        idx.search(namespace="test-ns", top_k=5, vector=[0.1, 0.2, 0.3])

        import orjson

        body = orjson.loads(route.calls.last.request.content)
        assert body["query"]["vector"] == [0.1, 0.2, 0.3]

    @respx.mock
    def test_search_with_rerank(self, mock_channel: MagicMock) -> None:
        route = respx.post(_SEARCH_URL).mock(
            return_value=httpx.Response(200, json=_SEARCH_RESPONSE)
        )
        idx = _make_grpc_index(mock_channel, host=_INDEX_HOST)
        idx.search(
            namespace="test-ns",
            top_k=5,
            inputs={"text": "query"},
            rerank={"model": "bge-reranker-v2-m3", "rank_fields": ["text"], "top_n": 3},
        )

        import orjson

        body = orjson.loads(route.calls.last.request.content)
        assert body["rerank"]["model"] == "bge-reranker-v2-m3"

    def test_search_empty_namespace_raises(self, mock_channel: MagicMock) -> None:
        idx = _make_grpc_index(mock_channel, host=_INDEX_HOST)
        with pytest.raises(ValidationError, match="non-empty"):
            idx.search(namespace="", top_k=5, inputs={"text": "query"})

    def test_search_top_k_zero_raises(self, mock_channel: MagicMock) -> None:
        idx = _make_grpc_index(mock_channel, host=_INDEX_HOST)
        with pytest.raises(ValidationError, match="top_k"):
            idx.search(namespace="test-ns", top_k=0, inputs={"text": "query"})

    def test_search_no_query_source_raises(self, mock_channel: MagicMock) -> None:
        idx = _make_grpc_index(mock_channel, host=_INDEX_HOST)
        with pytest.raises(ValidationError, match="inputs, vector, or id"):
            idx.search(namespace="test-ns", top_k=5)

    def test_search_rerank_missing_model_raises(self, mock_channel: MagicMock) -> None:
        idx = _make_grpc_index(mock_channel, host=_INDEX_HOST)
        with pytest.raises(ValidationError, match="model"):
            idx.search(
                namespace="test-ns",
                top_k=5,
                inputs={"text": "q"},
                rerank={"rank_fields": ["text"]},
            )

    def test_search_non_string_namespace_raises(self, mock_channel: MagicMock) -> None:
        idx = _make_grpc_index(mock_channel, host=_INDEX_HOST)
        with pytest.raises(ValidationError, match="namespace must be a string"):
            idx.search(namespace=123, top_k=5, inputs={"text": "q"})  # type: ignore[arg-type]

    def test_search_rerank_missing_rank_fields_raises(self, mock_channel: MagicMock) -> None:
        idx = _make_grpc_index(mock_channel, host=_INDEX_HOST)
        with pytest.raises(ValidationError, match="rank_fields"):
            idx.search(
                namespace="test-ns",
                top_k=5,
                inputs={"text": "q"},
                rerank={"model": "bge-reranker-v2-m3"},
            )

    @respx.mock
    def test_search_with_id_forwarded_to_body(self, mock_channel: MagicMock) -> None:
        import orjson

        route = respx.post(_SEARCH_URL).mock(
            return_value=httpx.Response(200, json=_SEARCH_RESPONSE)
        )
        idx = _make_grpc_index(mock_channel, host=_INDEX_HOST)
        idx.search(namespace="test-ns", top_k=3, id="rec-1")

        body = orjson.loads(route.calls.last.request.content)
        assert body["query"]["id"] == "rec-1"

    @respx.mock
    def test_search_with_filter_forwarded_to_body(self, mock_channel: MagicMock) -> None:
        import orjson

        route = respx.post(_SEARCH_URL).mock(
            return_value=httpx.Response(200, json=_SEARCH_RESPONSE)
        )
        idx = _make_grpc_index(mock_channel, host=_INDEX_HOST)
        filter_dict = {"topic": {"$eq": "ai"}}
        idx.search(namespace="test-ns", top_k=5, inputs={"text": "q"}, filter=filter_dict)

        body = orjson.loads(route.calls.last.request.content)
        assert body["query"]["filter"] == filter_dict

    @respx.mock
    def test_search_with_match_terms_forwarded_to_body(self, mock_channel: MagicMock) -> None:
        import orjson

        route = respx.post(_SEARCH_URL).mock(
            return_value=httpx.Response(200, json=_SEARCH_RESPONSE)
        )
        idx = _make_grpc_index(mock_channel, host=_INDEX_HOST)
        match_terms = {"strategy": "all", "terms": ["ai"]}
        idx.search(namespace="test-ns", top_k=5, inputs={"text": "q"}, match_terms=match_terms)

        body = orjson.loads(route.calls.last.request.content)
        assert body["query"]["match_terms"] == match_terms

    @respx.mock
    def test_search_with_fields_forwarded_at_body_root(self, mock_channel: MagicMock) -> None:
        import orjson

        route = respx.post(_SEARCH_URL).mock(
            return_value=httpx.Response(200, json=_SEARCH_RESPONSE)
        )
        idx = _make_grpc_index(mock_channel, host=_INDEX_HOST)
        idx.search(namespace="test-ns", top_k=5, inputs={"text": "q"}, fields=["text", "title"])

        body = orjson.loads(route.calls.last.request.content)
        assert body["fields"] == ["text", "title"]
        assert "fields" not in body["query"]

    @respx.mock
    def test_search_accepts_legacy_query_body(self, mock_channel: MagicMock) -> None:
        import orjson

        route = respx.post(_SEARCH_URL).mock(
            return_value=httpx.Response(200, json=_SEARCH_RESPONSE)
        )
        idx = _make_grpc_index(mock_channel, host=_INDEX_HOST)
        query = {"inputs": {"text": "q"}, "top_k": 5, "filter": {"topic": {"$eq": "ai"}}}
        idx.search(namespace="test-ns", query=query, fields=["text", "title"])

        body = orjson.loads(route.calls.last.request.content)
        assert body["query"] == query
        assert body["fields"] == ["text", "title"]

    @respx.mock
    def test_search_records_alias(self, mock_channel: MagicMock) -> None:
        """search_records() is an alias for search() and produces the same result."""
        respx.post(_SEARCH_URL).mock(return_value=httpx.Response(200, json=_SEARCH_RESPONSE))
        idx = _make_grpc_index(mock_channel, host=_INDEX_HOST)

        response = idx.search_records(namespace="test-ns", top_k=5, inputs={"text": "query"})

        assert isinstance(response, SearchRecordsResponse)
        assert len(response.result.hits) == 2


class TestContextManager:
    """GrpcIndex context manager (__enter__ / __exit__) closes all resources."""

    def test_grpc_index_enter_exit_closes_resources(self, mock_channel: MagicMock) -> None:
        idx = _make_grpc_index(mock_channel)

        mock_http_close = MagicMock()
        mock_channel_close = MagicMock()
        mock_executor_shutdown = MagicMock()

        idx._http.close = mock_http_close
        idx._channel.close = mock_channel_close
        idx._executor.shutdown = mock_executor_shutdown

        with idx:
            pass

        mock_http_close.assert_called_once_with()
        mock_channel_close.assert_called_once_with()
        mock_executor_shutdown.assert_called_once_with(wait=True)
