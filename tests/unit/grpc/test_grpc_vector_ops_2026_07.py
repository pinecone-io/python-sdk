"""2026-07 vector operations on the gRPC lane: validation parity with REST.

A caller who switches transports must not get a differently worded rejection,
so this module does not restate any expected text. It binds the REST lane's own
cross-lane fixtures — :data:`tests.unit.test_vector_op_validation.QUERY_TRUTH_TABLE`
(the spec's ``anyOf``/``not`` truth table for ``query``) and
:data:`~tests.unit.test_vector_op_validation.VECTOR_OP_VALIDATION_CASES` (the
exact rejection text for every tightened rule) — to a :class:`GrpcIndex`,
following the precedent ``test_grpc_namespace_2026_07.py`` set for the
namespace operations. The two clients expose the same vector method
signatures, so the very same callable runs against both, and the message
compared against is the one REST is asserted to produce.

Every rejection must fire in Python before the call crosses into the Rust
channel: the channel here is a mock, and a case that reached it would fail the
``method_calls`` assertion (and, against a real channel, would surface a worse
tonic-worded error instead of the REST text).

The API-version half mirrors ``TestApiVersionMetadata`` in the namespace
module: the Rust ``MetadataInterceptor`` attaches
``x-pinecone-api-version: 2026-07`` to every request on the channel (asserted
in ``rust/src/transport.rs``), and what is assertable from Python is that the
version handed to the channel is 2026-07 and that each vector rpc issues its
call on that same channel.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from pinecone._internal.constants import DATA_PLANE_API_VERSION
from pinecone.errors.exceptions import ValidationError
from pinecone.grpc import GrpcIndex
from tests.unit.test_vector_op_validation import (
    QUERY_TRUTH_TABLE,
    VECTOR_OP_VALIDATION_CASES,
    Invoke,
    query_kwargs,
)

_MOCK_GRPC_MODULE_PATH = "pinecone._grpc"


def _grpc_index(mock_channel: MagicMock) -> GrpcIndex:
    mock_module = MagicMock()
    mock_module.GrpcChannel.return_value = mock_channel
    with patch.dict("sys.modules", {_MOCK_GRPC_MODULE_PATH: mock_module}):
        return GrpcIndex(host="test-index-abc123.svc.pinecone.io", api_key="test-api-key")


@pytest.fixture
def mock_channel() -> MagicMock:
    """A channel whose replies terminate.

    ``list()`` follows ``pagination.next`` until it is ``None``, and a bare
    ``MagicMock`` answers every ``get("next")`` with a truthy mock — so a
    regression that stopped validating ``limit`` would make the generator
    paginate forever instead of failing. Handing back real dicts keeps a
    regression a fast failure. Arming a reply is not a call, so the
    "rejected before the channel" assertions still hold.
    """
    channel = MagicMock()
    channel.query.return_value = {"matches": [], "namespace": ""}
    channel.fetch.return_value = {"vectors": {}, "namespace": ""}
    channel.fetch_by_metadata.return_value = {"vectors": {}, "namespace": ""}
    channel.delete.return_value = {}
    channel.update.return_value = {}
    channel.list.return_value = {"vectors": [], "namespace": ""}
    channel.describe_index_stats.return_value = {
        "namespaces": {},
        "index_fullness": 0.0,
        "total_vector_count": 0,
    }
    channel.upsert.return_value = {"upserted_count": 1}
    return channel


@pytest.fixture
def grpc_index(mock_channel: MagicMock) -> GrpcIndex:
    return _grpc_index(mock_channel)


# ---------------------------------------------------------------------------
# The query anyOf/not truth table, bound to GrpcIndex
# ---------------------------------------------------------------------------


class TestQueryTruthTable:
    @pytest.mark.parametrize(
        ("has_vector", "has_id", "has_sparse", "accepted"),
        [pytest.param(v, i, s, ok, id=case_id) for case_id, v, i, s, ok in QUERY_TRUTH_TABLE],
    )
    def test_query_selector_truth_table(
        self,
        grpc_index: GrpcIndex,
        mock_channel: MagicMock,
        has_vector: bool,
        has_id: bool,
        has_sparse: bool,
        accepted: bool,
    ) -> None:
        kwargs = query_kwargs(has_vector, has_id, has_sparse)

        if accepted:
            grpc_index.query(**kwargs)
            assert mock_channel.query.called
        else:
            with pytest.raises(ValidationError):
                grpc_index.query(**kwargs)
            assert not mock_channel.method_calls, (
                "rejection must happen before the call reaches the channel"
            )

    def test_the_truth_table_is_not_empty(self) -> None:
        """A silently-emptied table would make every parity case vacuous."""
        assert len(QUERY_TRUTH_TABLE) == 8


# ---------------------------------------------------------------------------
# The cross-lane rejection table, bound to GrpcIndex
# ---------------------------------------------------------------------------


class TestValidationParity:
    @pytest.mark.parametrize(
        ("invoke", "expected"),
        [pytest.param(fn, msg, id=case_id) for case_id, fn, msg in VECTOR_OP_VALIDATION_CASES],
    )
    def test_rejected_before_the_channel_with_the_rest_message(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock, invoke: Invoke, expected: str
    ) -> None:
        with pytest.raises(ValidationError) as excinfo:
            invoke(grpc_index)  # type: ignore[arg-type]

        assert str(excinfo.value) == expected
        assert not mock_channel.method_calls, (
            "validation must reject before the call reaches the channel"
        )

    def test_the_table_is_not_empty(self) -> None:
        assert len(VECTOR_OP_VALIDATION_CASES) >= 30


# ---------------------------------------------------------------------------
# Rejections the ticket names explicitly, pinned against the channel mock
# ---------------------------------------------------------------------------


class TestRejectedWithoutANetworkCall:
    """AC: empty-filter and update(filter+values) never reach the Rust channel."""

    @pytest.mark.parametrize(
        "invoke",
        [
            pytest.param(lambda idx: idx.delete(filter={}), id="delete_empty_filter"),
            pytest.param(
                lambda idx: idx.update(filter={}, set_metadata={"year": 2020}),
                id="update_empty_filter",
            ),
            pytest.param(
                lambda idx: idx.fetch_by_metadata(filter={}),
                id="fetch_by_metadata_empty_filter",
            ),
            pytest.param(
                lambda idx: idx.update(filter={"genre": {"$eq": "drama"}}, values=[0.1, 0.2]),
                id="update_filter_and_values",
            ),
            pytest.param(
                lambda idx: idx.update(
                    filter={"genre": {"$eq": "drama"}},
                    sparse_values={"indices": [1], "values": [0.5]},
                ),
                id="update_filter_and_sparse_values",
            ),
        ],
    )
    def test_rejected_before_the_channel(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock, invoke: Any
    ) -> None:
        with pytest.raises(ValidationError):
            invoke(grpc_index)
        assert not mock_channel.method_calls


# ---------------------------------------------------------------------------
# API version metadata on the vector rpcs
# ---------------------------------------------------------------------------


class TestApiVersionMetadata:
    """The vector rpcs must go out carrying ``x-pinecone-api-version: 2026-07``.

    The Rust ``MetadataInterceptor`` attaches that header to every request on
    the channel — asserted in ``rust/src/transport.rs``
    (``interceptor_attaches_all_metadata_headers``). What is assertable from
    Python is the other half: the version string handed to the channel really
    is 2026-07, and each vector rpc issues its call on that same channel
    rather than one built some other way.
    """

    def test_data_plane_version_is_2026_07(self) -> None:
        assert DATA_PLANE_API_VERSION == "2026-07"

    def test_channel_is_constructed_with_the_2026_07_version(self) -> None:
        mock_module = MagicMock()
        mock_module.GrpcChannel.return_value = MagicMock()
        with patch.dict("sys.modules", {_MOCK_GRPC_MODULE_PATH: mock_module}):
            GrpcIndex(host="test-index-abc123.svc.pinecone.io", api_key="test-api-key")

        args, kwargs = mock_module.GrpcChannel.call_args
        assert args[2] == DATA_PLANE_API_VERSION == "2026-07", (
            f"GrpcChannel was not given the 2026-07 API version: {args!r} {kwargs!r}"
        )

    @pytest.mark.parametrize(
        ("call", "channel_attr"),
        [
            pytest.param(
                lambda idx: idx.upsert(vectors=[("v1", [0.1, 0.2])]),
                "upsert",
                id="Upsert",
            ),
            pytest.param(
                lambda idx: idx.query(top_k=1, vector=[0.1, 0.2]),
                "query",
                id="Query",
            ),
            pytest.param(
                lambda idx: idx.fetch(ids=["v1"]),
                "fetch",
                id="Fetch",
            ),
            pytest.param(
                lambda idx: idx.fetch_by_metadata(filter={"genre": {"$eq": "comedy"}}),
                "fetch_by_metadata",
                id="FetchByMetadata",
            ),
            pytest.param(
                lambda idx: idx.delete(ids=["v1"]),
                "delete",
                id="Delete",
            ),
            pytest.param(
                lambda idx: idx.update(id="v1", values=[0.1, 0.2]),
                "update",
                id="Update",
            ),
            pytest.param(
                lambda idx: idx.list_paginated(),
                "list",
                id="List",
            ),
            pytest.param(
                lambda idx: idx.describe_index_stats(),
                "describe_index_stats",
                id="DescribeIndexStats",
            ),
        ],
    )
    def test_vector_rpcs_use_the_versioned_channel(
        self, mock_channel: MagicMock, call: Any, channel_attr: str
    ) -> None:
        mock_module = MagicMock()
        mock_module.GrpcChannel.return_value = mock_channel
        with patch.dict("sys.modules", {_MOCK_GRPC_MODULE_PATH: mock_module}):
            idx = GrpcIndex(host="test-index-abc123.svc.pinecone.io", api_key="test-api-key")

        call(idx)

        assert getattr(mock_channel, channel_attr).called, f"{channel_attr} was not called"
        assert mock_module.GrpcChannel.call_args[0][2] == DATA_PLANE_API_VERSION
