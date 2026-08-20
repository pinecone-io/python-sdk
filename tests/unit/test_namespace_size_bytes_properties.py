"""Property and cross-lane parity tests for ``NamespaceDescription.size_bytes``.

``size_bytes`` is ``uint64`` on the wire (db_data_2026-07.proto:379-380) and
``integer/int64`` in the OAS (db_data_2026-07.oas.yaml:2125-2129), so the whole
unsigned 64-bit range has to survive decoding: msgspec must not raise, and the
value must not truncate through the ``int`` annotation.
"""

from __future__ import annotations

import json
from typing import Any

from hypothesis import given
from hypothesis import strategies as st

from pinecone._internal.adapters.vectors_adapter import VectorsAdapter
from pinecone.grpc import _dict_to_namespace_description
from pinecone.models.namespaces.models import NamespaceDescription
from tests.factories import (
    make_namespace_description_grpc_dict,
    make_namespace_description_response,
)

UINT64_MAX = 2**64 - 1

_size_bytes = st.integers(min_value=0, max_value=UINT64_MAX)
_schemas = st.one_of(
    st.none(),
    st.just({"fields": {}}),
    st.just({"fields": {"genre": {"filterable": True}}}),
)
_indexed_fields = st.one_of(
    st.none(),
    st.just({"fields": []}),
    st.just({"fields": ["genre", "year"]}),
)


@st.composite
def namespace_payloads(draw: st.DrawFn) -> dict[str, Any]:
    """Build a namespace JSON payload with optional members present or absent."""
    payload: dict[str, Any] = {
        "name": draw(st.text(alphabet=st.characters(min_codepoint=1, max_codepoint=127))),
        "record_count": draw(st.integers(min_value=0, max_value=UINT64_MAX)),
    }
    schema = draw(_schemas)
    if schema is not None:
        payload["schema"] = schema
    indexed = draw(_indexed_fields)
    if indexed is not None:
        payload["indexed_fields"] = indexed
    if draw(st.booleans()):
        payload["size_bytes"] = draw(_size_bytes)
    return payload


@given(payload=namespace_payloads())
def test_size_bytes_roundtrips_for_any_payload_permutation(payload: dict[str, Any]) -> None:
    ns = VectorsAdapter.to_namespace_description(json.dumps(payload).encode())
    assert ns.size_bytes == payload.get("size_bytes", 0)
    assert ns.to_dict()["size_bytes"] == ns.size_bytes


@given(size_bytes=_size_bytes)
def test_size_bytes_never_truncates(size_bytes: int) -> None:
    data = f'{{"name": "ns", "size_bytes": {size_bytes}}}'.encode()
    assert VectorsAdapter.to_namespace_description(data).size_bytes == size_bytes


@given(payload=namespace_payloads())
def test_list_namespaces_size_bytes_roundtrips(payload: dict[str, Any]) -> None:
    body = json.dumps({"namespaces": [payload], "total_count": 1}).encode()
    response = VectorsAdapter.to_list_namespaces_response(body)
    assert response.namespaces[0].size_bytes == payload.get("size_bytes", 0)


class TestRestGrpcParity:
    """The shared fixture both lanes decode.

    #121 plumbed size_bytes through the gRPC channel; the parity cases that
    exercise it across the uint64 range live alongside the rest of the gRPC
    namespace coverage, in tests/unit/grpc/test_grpc_namespace_2026_07.py.
    """

    def test_parity_without_size_bytes(self) -> None:
        rest_payload = make_namespace_description_response()
        grpc_payload = make_namespace_description_grpc_dict()
        del rest_payload["size_bytes"]
        del grpc_payload["size_bytes"]

        from_rest = VectorsAdapter.to_namespace_description(json.dumps(rest_payload).encode())
        from_grpc = _dict_to_namespace_description(grpc_payload)

        assert from_rest == from_grpc
        assert from_rest.size_bytes == 0

    def test_parity_field_sets_match(self) -> None:
        """Both lanes build the same model, so neither can drift a field the other lacks."""
        from_rest = VectorsAdapter.to_namespace_description(
            json.dumps(make_namespace_description_response()).encode()
        )
        from_grpc = _dict_to_namespace_description(make_namespace_description_grpc_dict())
        assert from_rest.keys() == from_grpc.keys()
        assert "size_bytes" in from_rest

    def test_rest_lane_reads_size_bytes_from_the_fixture(self) -> None:
        from_rest = VectorsAdapter.to_namespace_description(
            json.dumps(make_namespace_description_response()).encode()
        )
        assert from_rest.size_bytes == 1048576
        assert from_rest.name == "test-namespace"
        assert from_rest.record_count == 42
        assert from_rest.indexed_fields is not None
        assert from_rest.indexed_fields.fields == ["genre", "year"]

    def test_grpc_dict_shape_still_decodes_indexed_fields(self) -> None:
        from_grpc = _dict_to_namespace_description(make_namespace_description_grpc_dict())
        assert from_grpc.indexed_fields is not None
        assert from_grpc.indexed_fields.fields == ["genre", "year"]

    def test_default_model_matches_empty_grpc_dict(self) -> None:
        assert _dict_to_namespace_description({}) == NamespaceDescription()
