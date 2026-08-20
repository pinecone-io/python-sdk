"""Unit tests for GrpcIndex.query() — validation parity and filter forwarding."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pinecone.errors.exceptions import ValidationError
from pinecone.grpc import GrpcIndex

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


def test_grpc_query_sparse_only_accepted() -> None:
    """Sparse-only query (no vector, no id) must be accepted and forwarded to channel."""
    idx, mock_channel = _make_grpc_index()
    mock_channel.query.return_value = {"matches": [], "namespace": ""}

    idx.query(top_k=5, sparse_vector={"indices": [1], "values": [0.5]})

    mock_channel.query.assert_called_once()
    call_kwargs = mock_channel.query.call_args[1]
    assert call_kwargs["sparse_vector"] == {"indices": [1], "values": [0.5]}
    assert call_kwargs["vector"] is None
    assert call_kwargs["id"] is None


def test_grpc_query_vector_id_sparse_none_rejected() -> None:
    """Calling query() with no vector, no id, and no sparse_vector raises ValidationError."""
    idx, _ = _make_grpc_index()
    with pytest.raises(
        ValidationError,
        match="At least one of vector, id, or sparse_vector must be provided",
    ):
        idx.query(top_k=5)


def test_query_top_k_upper_bound_grpc() -> None:
    """top_k > 10000 must raise ValidationError client-side without a network call."""
    idx, mock_channel = _make_grpc_index()
    with pytest.raises(ValidationError, match="top_k must be between 1 and 10000"):
        idx.query(top_k=10001, vector=[0.1])
    mock_channel.query.assert_not_called()


def test_query_top_k_lower_bound_grpc() -> None:
    """top_k < 1 must still raise ValidationError client-side."""
    idx, mock_channel = _make_grpc_index()
    with pytest.raises(ValidationError, match="top_k must be between 1 and 10000"):
        idx.query(top_k=0, vector=[0.1])
    mock_channel.query.assert_not_called()


def test_query_top_k_at_max_boundary_accepted() -> None:
    """top_k=10000 is the maximum allowed value and must be accepted."""
    idx, mock_channel = _make_grpc_index()
    mock_channel.query.return_value = {"matches": [], "namespace": ""}
    idx.query(top_k=10000, vector=[0.1])
    mock_channel.query.assert_called_once()


def test_query_with_sparse_values_model_forwards_as_dict() -> None:
    """SparseValues model is converted to plain dict before being forwarded to GrpcChannel."""
    from pinecone.models.vectors.sparse import SparseValues

    idx, mock_channel = _make_grpc_index()
    mock_channel.query.return_value = {"matches": [], "namespace": ""}

    sv = SparseValues(indices=[1, 4], values=[0.5, 0.2])
    idx.query(top_k=5, sparse_vector=sv)

    mock_channel.query.assert_called_once()
    call_kwargs = mock_channel.query.call_args[1]
    assert call_kwargs["sparse_vector"] == {"indices": [1, 4], "values": [0.5, 0.2]}


def test_query_boolean_filter_true_forwarded_as_bool() -> None:
    """Filter with boolean True value is forwarded to channel.query as a Python bool.

    This is a regression guard for CI-0050: the gRPC path must forward
    boolean filter values as Python bool objects (not as ints/floats) so the
    Rust GrpcChannel can encode them as protobuf BoolValue rather than
    NumberValue.  Covers unified-filter-0006.
    """
    idx, mock_channel = _make_grpc_index()
    mock_channel.query.return_value = {"matches": [], "namespace": ""}

    idx.query(
        top_k=10,
        vector=[0.1, 0.2],
        filter={"active": {"$eq": True}},
        namespace="ns-test",
    )

    mock_channel.query.assert_called_once()
    call_kwargs = mock_channel.query.call_args[1]
    filt = call_kwargs["filter"]
    assert filt is not None, "filter must be forwarded to channel"
    assert filt == {"active": {"$eq": True}}, "filter dict must be passed unchanged"
    bool_val = filt["active"]["$eq"]
    assert type(bool_val) is bool, f"$eq value must be bool, got {type(bool_val)}"
    assert bool_val is True, "$eq value must be True"


def test_query_boolean_filter_false_forwarded_as_bool() -> None:
    """Filter with boolean False value is forwarded to channel.query as a Python bool.

    Covers the False branch of CI-0050 / unified-filter-0006.
    """
    idx, mock_channel = _make_grpc_index()
    mock_channel.query.return_value = {"matches": [], "namespace": ""}

    idx.query(
        top_k=10,
        vector=[0.3, 0.4],
        filter={"active": {"$eq": False}},
        namespace="ns-test",
    )

    mock_channel.query.assert_called_once()
    call_kwargs = mock_channel.query.call_args[1]
    filt = call_kwargs["filter"]
    assert filt is not None
    bool_val = filt["active"]["$eq"]
    assert type(bool_val) is bool, f"$eq value must be bool, got {type(bool_val)}"
    assert bool_val is False, "$eq value must be False"


def test_query_no_filter_forwards_none() -> None:
    """When no filter is supplied, channel.query receives filter=None."""
    idx, mock_channel = _make_grpc_index()
    mock_channel.query.return_value = {"matches": [], "namespace": ""}

    idx.query(top_k=5, vector=[0.1, 0.2])

    call_kwargs = mock_channel.query.call_args[1]
    assert call_kwargs["filter"] is None, "filter should be None when not specified"


def test_grpc_query_id_and_vector_rejected_with_the_rest_wording() -> None:
    idx, mock_channel = _make_grpc_index()
    with pytest.raises(ValidationError, match="id is mutually exclusive with vector"):
        idx.query(top_k=1, id="vec-1", vector=[0.1, 0.2])
    mock_channel.query.assert_not_called()


def test_grpc_query_id_and_sparse_vector_rejected() -> None:
    idx, mock_channel = _make_grpc_index()
    with pytest.raises(ValidationError, match="id is mutually exclusive with sparse_vector"):
        idx.query(top_k=1, id="vec-1", sparse_vector={"indices": [1], "values": [0.5]})
    mock_channel.query.assert_not_called()


def test_grpc_query_hybrid_vector_and_sparse_accepted() -> None:
    """The two literal forms combine into a hybrid query, exactly as on REST."""
    idx, mock_channel = _make_grpc_index()
    mock_channel.query.return_value = {"matches": [], "namespace": ""}

    idx.query(top_k=1, vector=[0.1, 0.2], sparse_vector={"indices": [1], "values": [0.5]})

    mock_channel.query.assert_called_once()


def test_grpc_query_id_charset_validated_like_rest() -> None:
    """A query-by-id is held to the same 1-512 ASCII no-NUL rule as fetch ids."""
    idx, mock_channel = _make_grpc_index()
    with pytest.raises(ValidationError, match="id must contain only ASCII characters"):
        idx.query(top_k=1, id="naïve")
    mock_channel.query.assert_not_called()


def test_grpc_query_overlong_id_rejected() -> None:
    idx, mock_channel = _make_grpc_index()
    with pytest.raises(ValidationError, match="id exceeds the maximum length of 512"):
        idx.query(top_k=1, id="a" * 513)
    mock_channel.query.assert_not_called()
