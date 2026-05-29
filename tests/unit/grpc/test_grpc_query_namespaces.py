"""Unit tests for GrpcIndex.query_namespaces() and query_namespaces_async()."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pinecone.errors.exceptions import ValidationError
from pinecone.grpc import GrpcIndex
from pinecone.grpc.future import PineconeFuture
from pinecone.models.vectors.query_aggregator import QueryNamespacesResults
from pinecone.models.vectors.responses import QueryResponse
from pinecone.models.vectors.vector import ScoredVector

_MOCK_GRPC_MODULE_PATH = "pinecone._grpc"


def _make_grpc_index() -> tuple[GrpcIndex, MagicMock]:
    mock_channel = MagicMock()
    mock_module = MagicMock()
    mock_module.GrpcChannel.return_value = mock_channel
    with patch.dict("sys.modules", {_MOCK_GRPC_MODULE_PATH: mock_module}):
        idx = GrpcIndex(
            host="test-index-abc123.svc.pinecone.io",
            api_key="test-api-key",
        )
    return idx, mock_channel


def _query_response(namespace: str, score: float) -> QueryResponse:
    return QueryResponse(
        matches=[ScoredVector(id="v1", score=score, values=[], sparse_values=None, metadata=None)],
        namespace=namespace,
        usage=None,
    )


def test_query_namespaces_fans_out() -> None:
    """query_namespaces fans out one query() call per namespace and merges results."""
    idx, mock_channel = _make_grpc_index()
    # channel.query is called by GrpcIndex.query()
    mock_channel.query.return_value = {"matches": [{"id": "v1", "score": 0.9}], "namespace": "ns1"}

    result = idx.query_namespaces(
        vector=[0.1, 0.2, 0.3],
        namespaces=["ns1", "ns2"],
        metric="cosine",
        top_k=5,
    )

    assert mock_channel.query.call_count == 2
    assert isinstance(result, QueryNamespacesResults)
    assert len(result.matches) > 0


def test_query_namespaces_empty_namespaces() -> None:
    """query_namespaces raises ValidationError when namespaces is empty."""
    idx, _ = _make_grpc_index()
    with pytest.raises(ValidationError, match="namespaces must be a non-empty list"):
        idx.query_namespaces(
            vector=[0.1],
            namespaces=[],
            metric="cosine",
            top_k=5,
        )


def test_query_namespaces_invalid_metric() -> None:
    """query_namespaces raises ValidationError for an unrecognized metric."""
    idx, _ = _make_grpc_index()
    with pytest.raises(ValidationError, match="Invalid metric"):
        idx.query_namespaces(
            vector=[0.1],
            namespaces=["ns1"],
            metric="invalid",
            top_k=5,
        )


def test_query_namespaces_no_vector_or_sparse_raises() -> None:
    """query_namespaces raises ValidationError if neither vector nor sparse_vector is given."""
    idx, _ = _make_grpc_index()
    with pytest.raises(ValidationError, match="at least one of"):
        idx.query_namespaces(
            namespaces=["ns1"],
            metric="cosine",
            top_k=5,
        )


def test_query_namespaces_deduplicates_namespaces() -> None:
    """Duplicate namespaces are collapsed to a single query call each."""
    idx, mock_channel = _make_grpc_index()
    mock_channel.query.return_value = {"matches": [], "namespace": "ns1"}

    idx.query_namespaces(
        vector=[0.1],
        namespaces=["ns1", "ns1", "ns1"],
        metric="cosine",
        top_k=5,
    )

    assert mock_channel.query.call_count == 1


def test_query_namespaces_async_returns_future() -> None:
    """query_namespaces_async returns a PineconeFuture wrapping QueryNamespacesResults."""
    idx, mock_channel = _make_grpc_index()
    mock_channel.query.return_value = {"matches": [{"id": "v1", "score": 0.9}], "namespace": "ns1"}

    future = idx.query_namespaces_async(
        vector=[0.1, 0.2, 0.3],
        namespaces=["ns1"],
        metric="cosine",
        top_k=5,
    )

    assert isinstance(future, PineconeFuture)
    result = future.result()
    assert isinstance(result, QueryNamespacesResults)
