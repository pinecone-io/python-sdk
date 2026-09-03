"""Unit tests for namespace response models and adapter decoding."""

from __future__ import annotations

import json

import pytest

from pinecone._internal.adapters.vectors_adapter import VectorsAdapter
from pinecone.models.namespaces.models import (
    IndexedFields,
    ListNamespacesResponse,
    NamespaceDescription,
    NamespaceFieldConfig,
    NamespaceSchema,
)
from pinecone.models.vectors.responses import Pagination
from tests.factories import make_namespace_description_response


class TestNamespaceDescription:
    def test_namespace_description_basic(self) -> None:
        ns = NamespaceDescription(name="ns1", record_count=100)
        assert ns.name == "ns1"
        assert ns.record_count == 100

    def test_namespace_description_with_schema(self) -> None:
        ns = NamespaceDescription(
            name="ns1",
            record_count=50,
            schema=NamespaceSchema(fields={"genre": NamespaceFieldConfig(filterable=True)}),
        )
        assert ns.schema is not None
        assert ns.schema.fields["genre"].filterable is True

    def test_namespace_description_bracket_access(self) -> None:
        ns = NamespaceDescription(name="ns1", record_count=100)
        assert ns["name"] == "ns1"
        assert ns["record_count"] == 100
        with pytest.raises(KeyError):
            ns["missing"]

    def test_namespace_description_defaults(self) -> None:
        ns = NamespaceDescription()
        assert ns.name == ""
        assert ns.record_count == 0
        assert ns.schema is None
        assert ns.indexed_fields is None
        assert ns.size_bytes == 0

    def test_namespace_description_size_bytes(self) -> None:
        ns = NamespaceDescription(name="ns1", record_count=100, size_bytes=1048576)
        assert ns.size_bytes == 1048576

    def test_namespace_description_size_bytes_bracket_access(self) -> None:
        ns = NamespaceDescription(name="ns1", size_bytes=2048)
        assert ns["size_bytes"] == 2048
        assert ns.get("size_bytes") == 2048

    def test_namespace_description_size_bytes_contains(self) -> None:
        ns = NamespaceDescription(name="ns1")
        assert "size_bytes" in ns
        assert "size_byte" not in ns

    def test_namespace_description_size_bytes_in_keys_and_items(self) -> None:
        ns = NamespaceDescription(name="ns1", size_bytes=99)
        assert ns.keys() == ("name", "record_count", "schema", "indexed_fields", "size_bytes")
        assert ("size_bytes", 99) in ns.items()
        assert list(ns) == list(ns.keys())
        assert len(ns) == 5

    def test_namespace_description_repr_includes_size_bytes(self) -> None:
        ns = NamespaceDescription(name="ns1", record_count=3, size_bytes=4096)
        assert "size_bytes=4096" in repr(ns)

    def test_namespace_description_size_bytes_uint64_max(self) -> None:
        """uint64 on the wire; Python ints must carry the full range without truncating."""
        uint64_max = 2**64 - 1
        ns = NamespaceDescription(name="ns1", size_bytes=uint64_max)
        assert ns.size_bytes == uint64_max

    def test_namespace_description_with_indexed_fields(self) -> None:
        ns = NamespaceDescription(
            name="ns1",
            indexed_fields=IndexedFields(fields=["genre", "year"]),
        )
        assert ns.indexed_fields is not None
        assert ns.indexed_fields.fields == ["genre", "year"]


class TestListNamespacesResponse:
    def test_list_namespaces_response_with_pagination(self) -> None:
        response = ListNamespacesResponse(
            namespaces=[NamespaceDescription(name="ns1", record_count=10)],
            pagination=Pagination(next="token123"),
            total_count=1,
        )
        assert response.pagination is not None
        assert response.pagination.next == "token123"

    def test_list_namespaces_response_last_page(self) -> None:
        response = ListNamespacesResponse(
            namespaces=[NamespaceDescription(name="ns1")],
            pagination=None,
            total_count=1,
        )
        assert response.pagination is None

    def test_list_namespaces_response_total_count(self) -> None:
        response = ListNamespacesResponse(total_count=25)
        assert response.total_count == 25

    def test_list_namespaces_response_bracket_access(self) -> None:
        response = ListNamespacesResponse(total_count=5)
        assert response["total_count"] == 5
        assert response["namespaces"] == []
        with pytest.raises(KeyError):
            response["missing"]

    def test_list_namespaces_response_iteration(self) -> None:
        ns1 = NamespaceDescription(name="ns1", record_count=10)
        ns2 = NamespaceDescription(name="ns2", record_count=20)
        response = ListNamespacesResponse(namespaces=[ns1, ns2], total_count=2)
        collected = list(response)
        assert collected == [ns1, ns2]

    def test_list_namespaces_response_len(self) -> None:
        response = ListNamespacesResponse(
            namespaces=[
                NamespaceDescription(name="a"),
                NamespaceDescription(name="b"),
                NamespaceDescription(name="c"),
            ],
            total_count=3,
        )
        assert len(response) == 3

    def test_list_namespaces_response_int_index(self) -> None:
        ns1 = NamespaceDescription(name="first", record_count=1)
        ns2 = NamespaceDescription(name="second", record_count=2)
        response = ListNamespacesResponse(namespaces=[ns1, ns2], total_count=2)
        assert response[0] is ns1
        assert response[1] is ns2
        with pytest.raises(IndexError):
            response[5]

    def test_list_namespaces_response_empty_iteration(self) -> None:
        response = ListNamespacesResponse()
        assert list(response) == []
        assert len(response) == 0

    def test_list_namespaces_response_contains_item(self) -> None:
        ns = NamespaceDescription(name="ns1", record_count=10)
        response = ListNamespacesResponse(namespaces=[ns], total_count=1)
        assert ns in response  # item membership via __iter__
        assert "namespaces" in response  # field-name check preserved
        assert "missing" not in response


class TestVectorsAdapterNamespaces:
    def test_adapter_decode_namespace_description(self) -> None:
        data = (
            b'{"name": "ns1", "record_count": 500,'
            b' "schema": {"fields": {"genre": {"filterable": true}}}}'
        )
        ns = VectorsAdapter.to_namespace_description(data)
        assert ns.name == "ns1"
        assert ns.record_count == 500
        assert ns.schema is not None
        assert ns.schema.fields["genre"].filterable is True

    def test_adapter_decode_list_namespaces(self) -> None:
        data = (
            b'{"namespaces": [{"name": "ns1", "record_count": 100},'
            b' {"name": "ns2", "record_count": 200}],'
            b' "pagination": {"next": "tok"}, "total_count": 2}'
        )
        response = VectorsAdapter.to_list_namespaces_response(data)
        assert len(response.namespaces) == 2
        assert response.namespaces[0].name == "ns1"
        assert response.namespaces[0].record_count == 100
        assert response.namespaces[1].name == "ns2"
        assert response.namespaces[1].record_count == 200
        assert response.pagination is not None
        assert response.pagination.next == "tok"
        assert response.total_count == 2

    def test_adapter_decode_list_namespaces_no_pagination(self) -> None:
        data = b'{"namespaces": [{"name": "ns1", "record_count": 10}], "total_count": 1}'
        response = VectorsAdapter.to_list_namespaces_response(data)
        assert response.pagination is None
        assert len(response.namespaces) == 1

    def test_adapter_decode_namespace_description_string_record_count(self) -> None:
        """API returns record_count as a string-encoded integer; adapter must coerce it."""
        data = b'{"name": "ns1", "record_count": "0"}'
        ns = VectorsAdapter.to_namespace_description(data)
        assert ns.name == "ns1"
        assert ns.record_count == 0
        assert isinstance(ns.record_count, int)

    def test_adapter_decode_list_namespaces_string_record_count(self) -> None:
        """API returns record_count as a string in list responses; adapter must coerce it."""
        data = (
            b'{"namespaces": [{"name": "ns1", "record_count": "42"},'
            b' {"name": "ns2", "record_count": "0"}],'
            b' "total_count": 2}'
        )
        response = VectorsAdapter.to_list_namespaces_response(data)
        assert response.namespaces[0].record_count == 42
        assert isinstance(response.namespaces[0].record_count, int)
        assert response.namespaces[1].record_count == 0
        assert isinstance(response.namespaces[1].record_count, int)


class TestNamespaceDescriptionSizeBytesDecode:
    def test_decode_size_bytes_present(self) -> None:
        data = json.dumps(make_namespace_description_response()).encode()
        ns = VectorsAdapter.to_namespace_description(data)
        assert ns.size_bytes == 1048576

    def test_decode_size_bytes_absent_defaults_to_zero(self) -> None:
        """2025-10 fixtures omit size_bytes; decoding them must not raise."""
        payload = make_namespace_description_response()
        del payload["size_bytes"]
        ns = VectorsAdapter.to_namespace_description(json.dumps(payload).encode())
        assert ns.size_bytes == 0
        assert ns.name == "test-namespace"
        assert ns.record_count == 42

    def test_decode_size_bytes_explicit_zero(self) -> None:
        """A server-sent 0 is a real value, not a missing field — both land on 0."""
        data = json.dumps(make_namespace_description_response(size_bytes=0)).encode()
        ns = VectorsAdapter.to_namespace_description(data)
        assert ns.size_bytes == 0

    def test_decode_size_bytes_string_encoded(self) -> None:
        """The namespace API string-encodes integers; lax decoding must coerce this one too."""
        data = b'{"name": "ns1", "record_count": "0", "size_bytes": "1048576"}'
        ns = VectorsAdapter.to_namespace_description(data)
        assert ns.size_bytes == 1048576
        assert isinstance(ns.size_bytes, int)

    def test_decode_size_bytes_uint64_max(self) -> None:
        uint64_max = 2**64 - 1
        data = f'{{"name": "ns1", "size_bytes": {uint64_max}}}'.encode()
        ns = VectorsAdapter.to_namespace_description(data)
        assert ns.size_bytes == uint64_max

    def test_decode_list_namespaces_size_bytes(self) -> None:
        payload = {
            "namespaces": [
                make_namespace_description_response(name="ns1", size_bytes=100),
                make_namespace_description_response(name="ns2"),
            ],
            "total_count": 2,
        }
        del payload["namespaces"][1]["size_bytes"]  # type: ignore[index]
        response = VectorsAdapter.to_list_namespaces_response(json.dumps(payload).encode())
        assert response.namespaces[0].size_bytes == 100
        assert response.namespaces[1].size_bytes == 0
