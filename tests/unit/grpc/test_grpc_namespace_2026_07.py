"""2026-07 namespace operations on the gRPC lane: size_bytes and validation parity.

Two things have to hold for the gRPC lane to be interchangeable with REST here.

``size_bytes`` has to survive the whole path. It is ``uint64`` on the wire
(``db_data_2026-07.proto:379-380``), the Rust converter
``namespace_description_to_py_dict`` puts it in the dict it hands to Python, and
:func:`_dict_to_namespace_description` reads it back out. The Rust half of that
(no truncation crossing PyO3, the key always emitted) is covered by the unit
tests in ``rust/src/transport.rs``; this module covers the Python half and the
resulting model.

Validation has to reject the same inputs with the same words. gRPC has no URL
path, so #119's percent-encoding concern does not apply, but the name, prefix,
limit and schema rules do — and a caller who switches transports must not get a
differently worded rejection. Rather than restating the expected text, this
module binds :data:`tests.unit.test_namespace_validation.VALIDATION_CASES` — the
REST lane's own table — to a :class:`GrpcIndex`. The two clients expose the same
namespace method signatures, so the very same callable runs against both, and
the message compared against is the one REST is asserted to produce.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given
from hypothesis import strategies as st

from pinecone._internal.adapters.vectors_adapter import VectorsAdapter
from pinecone._internal.constants import DATA_PLANE_API_VERSION
from pinecone.errors.exceptions import ValidationError
from pinecone.grpc import GrpcIndex, _dict_to_namespace_description
from pinecone.models.namespaces.models import NamespaceDescription
from tests.factories import (
    make_namespace_description_grpc_dict,
    make_namespace_description_response,
)
from tests.unit.test_namespace_validation import VALIDATION_CASES, Invoke

_MOCK_GRPC_MODULE_PATH = "pinecone._grpc"

UINT64_MAX = 2**64 - 1


def _grpc_index(mock_channel: MagicMock) -> GrpcIndex:
    mock_module = MagicMock()
    mock_module.GrpcChannel.return_value = mock_channel
    with patch.dict("sys.modules", {_MOCK_GRPC_MODULE_PATH: mock_module}):
        return GrpcIndex(host="test-index-abc123.svc.pinecone.io", api_key="test-api-key")


@pytest.fixture
def mock_channel() -> MagicMock:
    """A channel whose namespace replies terminate.

    ``list_namespaces`` is a generator that follows ``pagination.next`` until it
    is ``None``, and a bare ``MagicMock`` answers every ``get("next")`` with a
    truthy mock — so a regression that stopped validating ``limit`` would make
    the generator paginate forever instead of failing. Handing back real dicts
    keeps a regression a fast failure rather than a hung run. Arming a reply is
    not a call, so the "rejected before the channel" assertions still hold.
    """
    channel = MagicMock()
    channel.list_namespaces.return_value = {"namespaces": [], "total_count": 0}
    channel.describe_namespace.return_value = {"name": "ns"}
    channel.create_namespace.return_value = {"name": "ns"}
    channel.delete_namespace.return_value = None
    return channel


@pytest.fixture
def grpc_index(mock_channel: MagicMock) -> GrpcIndex:
    return _grpc_index(mock_channel)


# ---------------------------------------------------------------------------
# Validation parity with the REST lane
# ---------------------------------------------------------------------------


class TestValidationParity:
    @pytest.mark.parametrize(
        ("invoke", "expected"),
        [pytest.param(fn, msg, id=case_id) for case_id, fn, msg in VALIDATION_CASES],
    )
    def test_rejected_before_the_channel_with_the_rest_message(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock, invoke: Invoke, expected: str
    ) -> None:
        with pytest.raises(ValidationError) as excinfo:
            invoke(grpc_index)  # type: ignore[arg-type]

        assert expected in str(excinfo.value)
        assert not mock_channel.method_calls, (
            "validation must reject before the call reaches the channel"
        )

    def test_the_table_is_not_empty(self) -> None:
        """A silently-emptied table would make every parity case vacuous."""
        assert len(VALIDATION_CASES) >= 20

    def test_reserved_default_is_describable_and_deletable(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        """``__default__`` is rejected only by create, exactly as on REST."""
        mock_channel.describe_namespace.return_value = {"name": "__default__"}
        assert grpc_index.describe_namespace(name="__default__").name == "__default__"

        mock_channel.delete_namespace.return_value = None
        assert grpc_index.delete_namespace(name="__default__") is None

        with pytest.raises(ValidationError, match="reserved and cannot be created"):
            grpc_index.create_namespace(name="__default__")

    @pytest.mark.parametrize(
        "name",
        [
            pytest.param(" ", id="single_space"),
            pytest.param("a" * 512, id="at_length_limit"),
            pytest.param("ns-with.dots_and~tilde", id="punctuation"),
            pytest.param("\x01", id="lowest_legal_code_point"),
            pytest.param("\x7f", id="highest_legal_code_point"),
        ],
    )
    def test_names_the_old_grpc_rules_would_have_refused(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock, name: str
    ) -> None:
        mock_channel.create_namespace.return_value = {"name": name}
        assert grpc_index.create_namespace(name=name).name == name
        mock_channel.create_namespace.assert_called_once_with(name, None, timeout_s=None)

    @pytest.mark.parametrize("limit", [1, 50, 100])
    def test_limits_at_the_boundary_are_accepted(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock, limit: int
    ) -> None:
        mock_channel.list_namespaces.return_value = {"namespaces": [], "total_count": 0}
        grpc_index.list_namespaces_paginated(limit=limit)
        assert mock_channel.list_namespaces.call_args.kwargs["limit"] == limit

    def test_bool_limit_is_rejected_like_rest(self, grpc_index: GrpcIndex) -> None:
        """``True`` is an ``int`` in Python; REST refuses it and so must gRPC."""
        with pytest.raises(ValidationError, match="limit must be an integer, got bool"):
            grpc_index.list_namespaces_paginated(limit=True)

    def test_empty_prefix_is_accepted(self, grpc_index: GrpcIndex, mock_channel: MagicMock) -> None:
        mock_channel.list_namespaces.return_value = {"namespaces": [], "total_count": 0}
        grpc_index.list_namespaces_paginated(prefix="")
        assert mock_channel.list_namespaces.call_args.kwargs["prefix"] == ""


# ---------------------------------------------------------------------------
# size_bytes through the Python converter
# ---------------------------------------------------------------------------


class TestSizeBytesConverter:
    def test_size_bytes_is_read_from_the_channel_dict(self) -> None:
        ns = _dict_to_namespace_description(make_namespace_description_grpc_dict())
        assert ns.size_bytes == 1048576

    def test_missing_size_bytes_defaults_to_zero(self) -> None:
        payload = make_namespace_description_grpc_dict()
        del payload["size_bytes"]
        assert _dict_to_namespace_description(payload).size_bytes == 0

    def test_empty_dict_defaults_to_zero(self) -> None:
        assert _dict_to_namespace_description({}).size_bytes == 0
        assert _dict_to_namespace_description({}) == NamespaceDescription()

    @given(size_bytes=st.integers(min_value=0, max_value=UINT64_MAX))
    def test_converter_never_truncates(self, size_bytes: int) -> None:
        ns = _dict_to_namespace_description({"name": "ns", "size_bytes": size_bytes})
        assert ns.size_bytes == size_bytes

    def test_zero_is_reported_verbatim(self) -> None:
        """A 0 from a never-written namespace is passed through, not treated as absent."""
        ns = _dict_to_namespace_description({"name": "ns", "size_bytes": 0})
        assert ns.size_bytes == 0
        assert ns.to_dict()["size_bytes"] == 0


class TestSizeBytesThroughGrpcIndex:
    def test_describe_namespace_surfaces_size_bytes(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        mock_channel.describe_namespace.return_value = make_namespace_description_grpc_dict(
            size_bytes=9_007_199_254_740_993
        )
        ns = grpc_index.describe_namespace(name="movies-en")
        assert ns.size_bytes == 9_007_199_254_740_993

    def test_create_namespace_surfaces_size_bytes(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        mock_channel.create_namespace.return_value = make_namespace_description_grpc_dict(
            size_bytes=0
        )
        assert grpc_index.create_namespace(name="movies-en").size_bytes == 0

    def test_list_namespaces_paginated_surfaces_size_bytes(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        mock_channel.list_namespaces.return_value = {
            "namespaces": [
                make_namespace_description_grpc_dict(name="a", size_bytes=0),
                make_namespace_description_grpc_dict(name="b", size_bytes=UINT64_MAX),
            ],
            "total_count": 2,
        }
        page = grpc_index.list_namespaces_paginated()
        assert [ns.size_bytes for ns in page.namespaces] == [0, UINT64_MAX]

    def test_list_namespaces_generator_surfaces_size_bytes(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        mock_channel.list_namespaces.return_value = {
            "namespaces": [make_namespace_description_grpc_dict(size_bytes=4096)],
            "total_count": 1,
        }
        pages = list(grpc_index.list_namespaces())
        assert [ns.size_bytes for page in pages for ns in page.namespaces] == [4096]


# ---------------------------------------------------------------------------
# REST / gRPC model parity on the shared fixture
# ---------------------------------------------------------------------------


class TestRestGrpcModelParity:
    def test_same_namespace_yields_the_same_model(self) -> None:
        """The two wire shapes differ; the models they decode to must not."""
        from_rest = VectorsAdapter.to_namespace_description(
            json.dumps(make_namespace_description_response()).encode()
        )
        from_grpc = _dict_to_namespace_description(make_namespace_description_grpc_dict())

        assert from_rest == from_grpc
        assert from_rest.size_bytes == from_grpc.size_bytes == 1048576

    @pytest.mark.parametrize("size_bytes", [0, 1, 2**32, 2**32 + 1, UINT64_MAX])
    def test_parity_holds_across_the_uint64_range(self, size_bytes: int) -> None:
        rest = make_namespace_description_response(size_bytes=size_bytes)
        grpc = make_namespace_description_grpc_dict(size_bytes=size_bytes)

        from_rest = VectorsAdapter.to_namespace_description(json.dumps(rest).encode())
        from_grpc = _dict_to_namespace_description(grpc)

        assert from_rest == from_grpc
        assert from_grpc.size_bytes == size_bytes

    def test_parity_holds_when_optional_members_are_absent(self) -> None:
        rest: dict[str, Any] = {"name": "ns", "record_count": 7, "size_bytes": 4096}
        grpc: dict[str, Any] = {"name": "ns", "record_count": 7, "size_bytes": 4096}

        from_rest = VectorsAdapter.to_namespace_description(json.dumps(rest).encode())
        from_grpc = _dict_to_namespace_description(grpc)

        assert from_rest == from_grpc
        assert from_rest.schema is None
        assert from_rest.indexed_fields is None

    def test_parity_holds_when_size_bytes_is_absent_from_both(self) -> None:
        rest = make_namespace_description_response()
        grpc = make_namespace_description_grpc_dict()
        del rest["size_bytes"]
        del grpc["size_bytes"]

        from_rest = VectorsAdapter.to_namespace_description(json.dumps(rest).encode())
        from_grpc = _dict_to_namespace_description(grpc)

        assert from_rest == from_grpc
        assert from_grpc.size_bytes == 0
        assert from_grpc.name == "test-namespace"
        assert from_grpc.record_count == 42

    def test_the_two_fixtures_are_not_the_same_dict(self) -> None:
        """Parity would be vacuous if the factories had been collapsed into one shape."""
        rest = make_namespace_description_response()
        grpc = make_namespace_description_grpc_dict()
        assert rest["indexed_fields"] == {"fields": ["genre", "year"]}
        assert grpc["indexed_fields"] == ["genre", "year"]
        assert rest != grpc


# ---------------------------------------------------------------------------
# API version metadata
# ---------------------------------------------------------------------------


class TestApiVersionMetadata:
    """The namespace rpcs must go out carrying ``x-pinecone-api-version: 2026-07``.

    The Rust ``MetadataInterceptor`` attaches that header to every request on the
    channel — asserted in ``rust/src/transport.rs``
    (``interceptor_attaches_all_metadata_headers``). What is assertable from
    Python is the other half: the version string handed to the channel really is
    2026-07, and each namespace method issues its call on that same channel
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
        assert "2026-07" in args or kwargs.get("api_version") == "2026-07", (
            f"GrpcChannel was not given the 2026-07 API version: {args!r} {kwargs!r}"
        )

    @pytest.mark.parametrize(
        ("call", "channel_attr"),
        [
            pytest.param(
                lambda idx: idx.create_namespace(name="ns"),
                "create_namespace",
                id="CreateNamespace",
            ),
            pytest.param(
                lambda idx: idx.describe_namespace(name="ns"),
                "describe_namespace",
                id="DescribeNamespace",
            ),
            pytest.param(
                lambda idx: idx.delete_namespace(name="ns"),
                "delete_namespace",
                id="DeleteNamespace",
            ),
            pytest.param(
                lambda idx: idx.list_namespaces_paginated(),
                "list_namespaces",
                id="ListNamespaces",
            ),
        ],
    )
    def test_namespace_rpcs_use_the_versioned_channel(self, call: Any, channel_attr: str) -> None:
        channel = MagicMock()
        channel.list_namespaces.return_value = {"namespaces": [], "total_count": 0}
        channel.delete_namespace.return_value = None
        channel.create_namespace.return_value = {"name": "ns"}
        channel.describe_namespace.return_value = {"name": "ns"}

        mock_module = MagicMock()
        mock_module.GrpcChannel.return_value = channel
        with patch.dict("sys.modules", {_MOCK_GRPC_MODULE_PATH: mock_module}):
            idx = GrpcIndex(host="test-index-abc123.svc.pinecone.io", api_key="test-api-key")

        call(idx)

        assert getattr(channel, channel_attr).called, f"{channel_attr} was not called"
        assert mock_module.GrpcChannel.call_args[0][2] == DATA_PLANE_API_VERSION
