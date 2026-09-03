"""2026-07 conformance for the 12 db_data_grpc rpcs, captured on a real channel.

Every test here drives the SDK's genuine gRPC stack — :class:`GrpcIndex` over
the Rust tonic transport — against the in-process ``VectorService`` in
``_grpc_harness.py``, listening on a loopback ephemeral port. Nothing is
mocked between the SDK call and the assertions:

1. **rpc method** — ``claim.assert_grpc_request`` receives the ``:path`` the
   server's interceptor saw on the wire (e.g. ``/VectorService/Upsert``).
2. **api version** — ``claim.assert_api_version`` receives the invocation
   metadata the server saw; the ``x-pinecone-api-version: 2026-07`` pair in it
   was attached by ``rust/src/transport.rs``'s ``MetadataInterceptor`` from the
   SDK's own ``DATA_PLANE_API_VERSION`` constant, which is deliberately never
   passed by these tests.
3. **schema round-trip** — requests are decoded server-side by protoc-generated
   code regenerated each session from the vendored
   ``rust/proto/db_data_2026-07.proto`` (an implementation independent of the
   client's prost codec) and their fields are asserted; responses are
   protoc-built messages from that same proto, decoded by the real client, and
   the returned models are asserted field-by-field before
   ``claim.assert_roundtrip`` proves the model loses nothing. ``Delete`` and
   ``DeleteNamespace`` answer with the fieldless ``DeleteResponse {}``, so they
   assert the SDK returned ``None`` via ``claim.assert_no_response_body``.

Float values are chosen to be exact in f32 (0.5, 0.25, 0.75) so equality
assertions survive the proto ``float`` fields; metadata numbers travel as
doubles and come back as Python floats, hence ``2026.0``.

#121's mock-pagination trap does not arise here: the paginated claims call the
single-page ``*_paginated`` methods, and the server's responses terminate by
construction.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from google.protobuf import struct_pb2

from pinecone.grpc import GrpcIndex
from pinecone.models.namespaces.models import ListNamespacesResponse, NamespaceDescription
from pinecone.models.vectors.responses import (
    DescribeIndexStatsResponse,
    FetchByMetadataResponse,
    FetchResponse,
    ListResponse,
    QueryResponse,
    UpdateResponse,
    UpsertResponse,
)
from tests.unit.conformance import api_op
from tests.unit.conformance._grpc_harness import VectorServiceHarness

NAMESPACE = "conformance-ns"
VALUES = [0.5, 0.25]
SPARSE_INDICES = [17, 42]
SPARSE_VALUES = [0.75, 0.5]
METADATA: dict[str, Any] = {"genre": "documentary", "year": 2026.0}
FILTER: dict[str, Any] = {"genre": {"$eq": "documentary"}}

VECTOR_PAYLOAD: dict[str, Any] = {
    "id": "vec-1",
    "values": VALUES,
    "sparseValues": {"indices": SPARSE_INDICES, "values": SPARSE_VALUES},
    "metadata": METADATA,
}
NAMESPACE_DESCRIPTION_PAYLOAD: dict[str, Any] = {
    "name": NAMESPACE,
    "record_count": 42,
    "schema": {"fields": {"genre": {"filterable": True}}},
    "indexed_fields": {"fields": ["genre"]},
    "size_bytes": 1048576,
}


def _struct(data: dict[str, Any]) -> struct_pb2.Struct:
    s = struct_pb2.Struct()
    s.update(data)
    return s


@pytest.fixture(scope="module")
def harness() -> Iterator[VectorServiceHarness]:
    h = VectorServiceHarness()
    h.start()
    yield h
    h.stop()


@pytest.fixture(scope="module")
def index(harness: VectorServiceHarness) -> Iterator[GrpcIndex]:
    idx = GrpcIndex(host=harness.host, api_key="conformance-api-key", secure=False)
    yield idx
    idx.close()


@pytest.fixture
def server(harness: VectorServiceHarness) -> VectorServiceHarness:
    harness.reset()
    return harness


def _proto_vector(pb2: Any) -> Any:
    return pb2.Vector(
        id="vec-1",
        values=VALUES,
        sparse_values=pb2.SparseValues(indices=SPARSE_INDICES, values=SPARSE_VALUES),
        metadata=_struct(METADATA),
    )


def _proto_namespace_description(pb2: Any) -> Any:
    return pb2.NamespaceDescription(
        name=NAMESPACE,
        record_count=42,
        size_bytes=1048576,
        schema=pb2.MetadataSchema(fields={"genre": pb2.MetadataFieldProperties(filterable=True)}),
        indexed_fields=pb2.IndexedFields(fields=["genre"]),
    )


@api_op("db_data_grpc:Upsert")
def test_upsert(claim: Any, server: VectorServiceHarness, index: GrpcIndex) -> None:
    pb2 = server.pb2
    server.respond("Upsert", pb2.UpsertResponse(upserted_count=2))

    response = index.upsert(
        vectors=[
            {
                "id": "vec-1",
                "values": VALUES,
                "sparse_values": {"indices": SPARSE_INDICES, "values": SPARSE_VALUES},
                "metadata": METADATA,
            },
            ("vec-2", VALUES),
        ],
        namespace=NAMESPACE,
    )

    call = server.single_call()
    assert call.request.namespace == NAMESPACE
    assert [v.id for v in call.request.vectors] == ["vec-1", "vec-2"]
    assert list(call.request.vectors[0].values) == VALUES
    assert list(call.request.vectors[0].sparse_values.indices) == SPARSE_INDICES
    assert list(call.request.vectors[0].sparse_values.values) == SPARSE_VALUES
    assert call.request.vectors[0].metadata.fields["genre"].string_value == "documentary"
    assert call.request.vectors[0].metadata.fields["year"].number_value == 2026.0
    assert response.upserted_count == 2

    claim.assert_grpc_request(call.method)
    claim.assert_api_version(call.metadata)
    claim.assert_roundtrip(UpsertResponse, {"upsertedCount": 2}, optional_absent=[])


@api_op("db_data_grpc:Query")
def test_query(claim: Any, server: VectorServiceHarness, index: GrpcIndex) -> None:
    pb2 = server.pb2
    match = pb2.ScoredVector(
        id="vec-1",
        score=0.75,
        values=VALUES,
        sparse_values=pb2.SparseValues(indices=SPARSE_INDICES, values=SPARSE_VALUES),
        metadata=_struct(METADATA),
    )
    server.respond(
        "Query",
        pb2.QueryResponse(matches=[match], namespace=NAMESPACE, usage=pb2.Usage(read_units=5)),
    )

    response = index.query(
        top_k=10,
        vector=VALUES,
        namespace=NAMESPACE,
        filter=FILTER,
        include_values=True,
        include_metadata=True,
    )

    call = server.single_call()
    assert call.request.top_k == 10
    assert list(call.request.vector) == VALUES
    assert call.request.namespace == NAMESPACE
    assert call.request.include_values is True
    assert call.request.include_metadata is True
    assert call.request.filter.fields["genre"].struct_value.fields["$eq"].string_value == (
        "documentary"
    )
    assert response.matches[0].id == "vec-1"
    assert response.matches[0].score == 0.75
    assert response.matches[0].values == VALUES
    assert response.matches[0].metadata == METADATA
    assert response.namespace == NAMESPACE
    assert response.usage is not None and response.usage.read_units == 5

    claim.assert_grpc_request(call.method)
    claim.assert_api_version(call.metadata)
    claim.assert_roundtrip(
        QueryResponse,
        {
            "matches": [{**VECTOR_PAYLOAD, "score": 0.75}],
            "namespace": NAMESPACE,
            "usage": {"readUnits": 5},
        },
        optional_absent=["usage"],
    )


@api_op("db_data_grpc:Fetch")
def test_fetch(claim: Any, server: VectorServiceHarness, index: GrpcIndex) -> None:
    pb2 = server.pb2
    server.respond(
        "Fetch",
        pb2.FetchResponse(
            vectors={"vec-1": _proto_vector(pb2)},
            namespace=NAMESPACE,
            usage=pb2.Usage(read_units=1),
        ),
    )

    response = index.fetch(ids=["vec-1"], namespace=NAMESPACE)

    call = server.single_call()
    assert list(call.request.ids) == ["vec-1"]
    assert call.request.namespace == NAMESPACE
    assert response.vectors["vec-1"].id == "vec-1"
    assert response.vectors["vec-1"].values == VALUES
    assert response.vectors["vec-1"].metadata == METADATA
    assert response.namespace == NAMESPACE
    assert response.usage is not None and response.usage.read_units == 1

    claim.assert_grpc_request(call.method)
    claim.assert_api_version(call.metadata)
    claim.assert_roundtrip(
        FetchResponse,
        {"vectors": {"vec-1": VECTOR_PAYLOAD}, "namespace": NAMESPACE, "usage": {"readUnits": 1}},
        optional_absent=["usage"],
    )


@api_op("db_data_grpc:FetchByMetadata")
def test_fetch_by_metadata(claim: Any, server: VectorServiceHarness, index: GrpcIndex) -> None:
    pb2 = server.pb2
    server.respond(
        "FetchByMetadata",
        pb2.FetchByMetadataResponse(
            vectors={"vec-1": _proto_vector(pb2)},
            namespace=NAMESPACE,
            usage=pb2.Usage(read_units=2),
            pagination=pb2.Pagination(next="page-2"),
        ),
    )

    response = index.fetch_by_metadata(filter=FILTER, namespace=NAMESPACE, limit=10)

    call = server.single_call()
    assert call.request.namespace == NAMESPACE
    assert call.request.limit == 10
    assert call.request.filter.fields["genre"].struct_value.fields["$eq"].string_value == (
        "documentary"
    )
    assert response.vectors["vec-1"].metadata == METADATA
    assert response.pagination is not None and response.pagination.next == "page-2"
    assert response.usage is not None and response.usage.read_units == 2

    claim.assert_grpc_request(call.method)
    claim.assert_api_version(call.metadata)
    claim.assert_roundtrip(
        FetchByMetadataResponse,
        {
            "vectors": {"vec-1": VECTOR_PAYLOAD},
            "namespace": NAMESPACE,
            "usage": {"readUnits": 2},
            "pagination": {"next": "page-2"},
        },
        optional_absent=["usage", "pagination"],
    )


@api_op("db_data_grpc:Delete")
def test_delete(claim: Any, server: VectorServiceHarness, index: GrpcIndex) -> None:
    server.respond("Delete", server.pb2.DeleteResponse())

    returned = index.delete(ids=["vec-1", "vec-2"], namespace=NAMESPACE)

    call = server.single_call()
    assert list(call.request.ids) == ["vec-1", "vec-2"]
    assert call.request.delete_all is False
    assert call.request.namespace == NAMESPACE

    claim.assert_grpc_request(call.method)
    claim.assert_api_version(call.metadata)
    claim.assert_no_response_body(returned)


@api_op("db_data_grpc:Update")
def test_update(claim: Any, server: VectorServiceHarness, index: GrpcIndex) -> None:
    pb2 = server.pb2
    server.respond("Update", pb2.UpdateResponse(matched_records=1))

    response = index.update(
        id="vec-1",
        values=VALUES,
        set_metadata={"genre": "drama"},
        namespace=NAMESPACE,
    )

    call = server.single_call()
    assert call.request.id == "vec-1"
    assert list(call.request.values) == VALUES
    assert call.request.set_metadata.fields["genre"].string_value == "drama"
    assert call.request.namespace == NAMESPACE
    assert not call.request.HasField("dry_run")
    assert response.matched_records == 1

    claim.assert_grpc_request(call.method)
    claim.assert_api_version(call.metadata)
    claim.assert_roundtrip(
        UpdateResponse, {"matchedRecords": 1}, optional_absent=["matchedRecords"]
    )


@api_op("db_data_grpc:List")
def test_list(claim: Any, server: VectorServiceHarness, index: GrpcIndex) -> None:
    pb2 = server.pb2
    server.respond(
        "List",
        pb2.ListResponse(
            vectors=[pb2.ListItem(id="vec-1")],
            pagination=pb2.Pagination(next="page-2"),
            namespace=NAMESPACE,
            usage=pb2.Usage(read_units=1),
        ),
    )

    response = index.list_paginated(prefix="vec-", limit=50, namespace=NAMESPACE)

    call = server.single_call()
    assert call.request.prefix == "vec-"
    assert call.request.limit == 50
    assert call.request.namespace == NAMESPACE
    assert [item.id for item in response.vectors] == ["vec-1"]
    assert response.pagination is not None and response.pagination.next == "page-2"
    assert response.namespace == NAMESPACE
    assert response.usage is not None and response.usage.read_units == 1

    claim.assert_grpc_request(call.method)
    claim.assert_api_version(call.metadata)
    claim.assert_roundtrip(
        ListResponse,
        {
            "vectors": [{"id": "vec-1"}],
            "pagination": {"next": "page-2"},
            "namespace": NAMESPACE,
            "usage": {"readUnits": 1},
        },
        optional_absent=["pagination", "usage"],
    )


@api_op("db_data_grpc:DescribeIndexStats")
def test_describe_index_stats(claim: Any, server: VectorServiceHarness, index: GrpcIndex) -> None:
    pb2 = server.pb2
    server.respond(
        "DescribeIndexStats",
        pb2.DescribeIndexStatsResponse(
            namespaces={NAMESPACE: pb2.NamespaceSummary(vector_count=80000)},
            dimension=1024,
            index_fullness=0.5,
            total_vector_count=80000,
            metric="cosine",
            vector_type="dense",
            memory_fullness=0.25,
            storage_fullness=0.75,
        ),
    )

    response = index.describe_index_stats(filter=FILTER)

    call = server.single_call()
    assert call.request.filter.fields["genre"].struct_value.fields["$eq"].string_value == (
        "documentary"
    )
    assert response.namespaces[NAMESPACE].vector_count == 80000
    assert response.dimension == 1024
    assert response.index_fullness == 0.5
    assert response.total_vector_count == 80000
    assert response.metric == "cosine"
    assert response.vector_type == "dense"
    assert response.memory_fullness == 0.25
    assert response.storage_fullness == 0.75

    claim.assert_grpc_request(call.method)
    claim.assert_api_version(call.metadata)
    claim.assert_roundtrip(
        DescribeIndexStatsResponse,
        {
            "namespaces": {NAMESPACE: {"vectorCount": 80000}},
            "dimension": 1024,
            "indexFullness": 0.5,
            "totalVectorCount": 80000,
            "metric": "cosine",
            "vectorType": "dense",
            "memoryFullness": 0.25,
            "storageFullness": 0.75,
        },
        optional_absent=["metric", "vectorType", "memoryFullness", "storageFullness"],
    )


@api_op("db_data_grpc:ListNamespaces")
def test_list_namespaces(claim: Any, server: VectorServiceHarness, index: GrpcIndex) -> None:
    pb2 = server.pb2
    server.respond(
        "ListNamespaces",
        pb2.ListNamespacesResponse(
            namespaces=[_proto_namespace_description(pb2)],
            pagination=pb2.Pagination(next="page-2"),
            total_count=1,
        ),
    )

    response = index.list_namespaces_paginated(prefix="conf", limit=10)

    call = server.single_call()
    assert call.request.prefix == "conf"
    assert call.request.limit == 10
    assert response.namespaces[0].name == NAMESPACE
    assert response.namespaces[0].record_count == 42
    assert response.namespaces[0].size_bytes == 1048576
    assert response.pagination is not None and response.pagination.next == "page-2"
    assert response.total_count == 1

    claim.assert_grpc_request(call.method)
    claim.assert_api_version(call.metadata)
    claim.assert_roundtrip(
        ListNamespacesResponse,
        {
            "namespaces": [NAMESPACE_DESCRIPTION_PAYLOAD],
            "pagination": {"next": "page-2"},
            "total_count": 1,
        },
        optional_absent=["pagination"],
    )


@api_op("db_data_grpc:DescribeNamespace")
def test_describe_namespace(claim: Any, server: VectorServiceHarness, index: GrpcIndex) -> None:
    server.respond("DescribeNamespace", _proto_namespace_description(server.pb2))

    response = index.describe_namespace(name=NAMESPACE)

    call = server.single_call()
    assert call.request.namespace == NAMESPACE
    assert response.name == NAMESPACE
    assert response.record_count == 42
    assert response.size_bytes == 1048576
    assert response.schema is not None
    assert response.schema.fields["genre"].filterable is True
    assert response.indexed_fields is not None
    assert response.indexed_fields.fields == ["genre"]

    claim.assert_grpc_request(call.method)
    claim.assert_api_version(call.metadata)
    claim.assert_roundtrip(
        NamespaceDescription,
        NAMESPACE_DESCRIPTION_PAYLOAD,
        optional_absent=["schema", "indexed_fields"],
    )


@api_op("db_data_grpc:DeleteNamespace")
def test_delete_namespace(claim: Any, server: VectorServiceHarness, index: GrpcIndex) -> None:
    server.respond("DeleteNamespace", server.pb2.DeleteResponse())

    returned = index.delete_namespace(name=NAMESPACE)

    call = server.single_call()
    assert call.request.namespace == NAMESPACE

    claim.assert_grpc_request(call.method)
    claim.assert_api_version(call.metadata)
    claim.assert_no_response_body(returned)


@api_op("db_data_grpc:CreateNamespace")
def test_create_namespace(claim: Any, server: VectorServiceHarness, index: GrpcIndex) -> None:
    server.respond("CreateNamespace", _proto_namespace_description(server.pb2))

    response = index.create_namespace(
        name=NAMESPACE, schema={"fields": {"genre": {"filterable": True}}}
    )

    call = server.single_call()
    assert call.request.name == NAMESPACE
    assert call.request.schema.fields["genre"].filterable is True
    assert response.name == NAMESPACE
    assert response.record_count == 42
    assert response.size_bytes == 1048576
    assert response.schema is not None
    assert response.schema.fields["genre"].filterable is True

    claim.assert_grpc_request(call.method)
    claim.assert_api_version(call.metadata)
    claim.assert_roundtrip(
        NamespaceDescription,
        NAMESPACE_DESCRIPTION_PAYLOAD,
        optional_absent=["schema", "indexed_fields"],
    )
