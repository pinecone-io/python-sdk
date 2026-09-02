"""Integration tests for error paths (sync / REST + gRPC).

Tests verify that the SDK raises typed, human-readable exceptions rather than
raw HTTP errors or generic exceptions.

As of 2026-07 ``indexes.list()`` returns a lazy ``Paginator`` and issues no HTTP
request until consumed, so a test expecting a transport error from a paginated
operation must drive the paginator (``.to_list()``) inside the ``pytest.raises``
block — otherwise nothing is sent and nothing raises.
"""

from __future__ import annotations

from typing import Any

import pytest

from pinecone import GrpcIndex, Index, Pinecone, PineconeValueError
from pinecone.errors import (
    ApiError,
    ConflictError,
    NotFoundError,
    PineconeError,
    PineconeTimeoutError,
    UnauthorizedError,
)
from tests.integration.conftest import cleanup_resource, poll_until, unique_name
from tests.integration.index_shapes import DENSE_FIELD, MANAGED_AWS, dense_schema

_DENSE_SCHEMA_2D = dense_schema(2)
_DENSE_SCHEMA_3D = dense_schema(3)

# ---------------------------------------------------------------------------
# error-bad-api-key
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_bad_api_key_raises_typed_exception() -> None:
    """Pinecone(api_key="invalid") + indexes.list() raises UnauthorizedError (not raw HTTP error)."""
    bad_client = Pinecone(api_key="invalid-key-12345")
    with pytest.raises(UnauthorizedError) as exc_info:
        bad_client.indexes.list().to_list()

    err = exc_info.value
    assert isinstance(err, ApiError)
    assert err.status_code == 401
    # Error message must be human-readable (non-empty)
    assert str(err)


@pytest.mark.integration
def test_bad_api_key_error_message_is_human_readable() -> None:
    """UnauthorizedError from a bad API key has a non-empty, informative message."""
    bad_client = Pinecone(api_key="totally-wrong-key-xyz")
    with pytest.raises(UnauthorizedError) as exc_info:
        bad_client.indexes.list().to_list()

    err = exc_info.value
    # Message should exist and not just be a raw status code
    msg = str(err)
    assert len(msg) > 0
    # Should not be only a number
    assert not msg.strip().isdigit()


# ---------------------------------------------------------------------------
# error-nonexistent-index
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_describe_nonexistent_index_raises_not_found(client: Pinecone) -> None:
    """indexes.describe() on a non-existent name raises NotFoundError (typed, status_code=404)."""
    with pytest.raises(NotFoundError) as exc_info:
        client.indexes.describe("index-that-does-not-exist-xyz")

    err = exc_info.value
    assert isinstance(err, ApiError)
    assert err.status_code == 404
    # Error message must be human-readable (non-empty, not just a number)
    msg = str(err)
    assert len(msg) > 0
    assert not msg.strip().isdigit()


@pytest.mark.integration
def test_delete_nonexistent_index_raises_not_found(client: Pinecone) -> None:
    """indexes.delete() on a non-existent name raises NotFoundError (typed, status_code=404)."""
    with pytest.raises(NotFoundError) as exc_info:
        client.indexes.delete("index-that-does-not-exist-xyz")

    err = exc_info.value
    assert isinstance(err, ApiError)
    assert err.status_code == 404


# ---------------------------------------------------------------------------
# error-dimension-mismatch
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_dimension_mismatch_raises_typed_error_rest(client: Pinecone) -> None:
    """Upserting a 3-dim vector into a 2-dim field raises ApiError (status_code=400, REST sync).

    The write goes through the documents API: a schema-bearing index rejects the
    vector write endpoints outright, so upsert(vectors=...) would return 400 for
    the wrong reason and the assertion would hold no matter what dimension was
    sent. The correctly-sized document at the end is the control that keeps this
    test honest — it must succeed, proving the 400 above is about the dimension.
    """
    name = unique_name("idx")
    try:
        client.indexes.create(
            name=name,
            schema=_DENSE_SCHEMA_2D,
            deployment=MANAGED_AWS,
            timeout=300,
        )
        index = client.index(name=name)

        with pytest.raises(ApiError) as exc_info:
            index.documents.upsert(
                namespace="dim-ns",
                documents=[{"_id": "dim-v1", DENSE_FIELD: [0.1, 0.2, 0.3]}],
            )

        err = exc_info.value
        assert err.status_code == 400
        msg = str(err)
        assert "dimension" in msg.lower()
        assert not msg.strip().isdigit()

        index.documents.upsert(
            namespace="dim-ns",
            documents=[{"_id": "dim-ok", DENSE_FIELD: [0.1, 0.2]}],
        )
    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


@pytest.mark.integration
def test_grpc_vector_write_rejected_on_schema_index(client: Pinecone) -> None:
    """A gRPC vector-API write to a schema-bearing index raises ApiError (status_code=400).

    Replaces a gRPC dimension-mismatch check that 2026-07 makes unreachable:
    every index carries a document schema, GrpcIndex exposes no documents write
    method, and the vector write path is refused before any dimension is
    examined. The refusal itself is what is assertable here, so this pins the
    message that points callers at the documents API.
    """
    name = unique_name("idx")
    try:
        client.indexes.create(
            name=name,
            schema=_DENSE_SCHEMA_2D,
            deployment=MANAGED_AWS,
            timeout=300,
        )
        index = client.index(name=name, grpc=True)

        with pytest.raises(ApiError) as exc_info:
            index.upsert(vectors=[{"id": "dim-v1", "values": [0.1, 0.2]}])

        err = exc_info.value
        assert err.status_code == 400
        msg = str(err)
        assert "documents" in msg.lower()
        assert not msg.strip().isdigit()
    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# error-duplicate-index
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_duplicate_index_raises_conflict_error(client: Pinecone) -> None:
    """Creating an index with a name that already exists raises ConflictError (status_code=409)."""
    name = unique_name("idx")
    try:
        client.indexes.create(
            name=name,
            schema=_DENSE_SCHEMA_2D,
            deployment=MANAGED_AWS,
            timeout=300,
        )

        with pytest.raises(ConflictError) as exc_info:
            client.indexes.create(
                name=name,
                schema=_DENSE_SCHEMA_2D,
                deployment=MANAGED_AWS,
                timeout=-1,  # skip waiting — index already exists
            )

        err = exc_info.value
        assert isinstance(err, ApiError)
        assert err.status_code == 409
        msg = str(err)
        assert len(msg) > 0
        assert not msg.strip().isdigit()
    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# error-invalid-host  (unified-index-0043)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_invalid_index_host_raises_value_error() -> None:
    """Index and GrpcIndex raise PineconeValueError for hosts without a dot or 'localhost'.

    Verifies unified-index-0043: host URL validation fires at construction time,
    before any network call is attempted. A host string must contain a dot or
    the substring 'localhost' to be considered a plausible URL.
    """
    # REST Index: no-dot host rejected
    with pytest.raises(PineconeValueError):
        Index(host="nodot", api_key="testkey")

    # REST Index: empty string rejected
    with pytest.raises(PineconeValueError):
        Index(host="", api_key="testkey")

    # REST Index: whitespace-only rejected
    with pytest.raises(PineconeValueError):
        Index(host="   ", api_key="testkey")

    # GrpcIndex: same validation applies
    with pytest.raises(PineconeValueError):
        GrpcIndex(host="nodot", api_key="testkey")

    with pytest.raises(PineconeValueError):
        GrpcIndex(host="", api_key="testkey")


# ---------------------------------------------------------------------------
# error-query-validation  (unified-vec-0038, unified-vec-0039, unified-vec-0040)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# error-invalid-index-name  (unified-index-0045)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_create_index_invalid_name_rest(client: Pinecone) -> None:
    """indexes.create() rejects invalid index names before any API call (REST sync).

    Verifies unified-index-0045: the SDK raises PineconeValueError for names
    that are too long (>45 characters) or contain disallowed characters (anything
    other than lowercase letters, digits, and hyphens). Validation fires
    synchronously in require_valid_resource_name() before any HTTP request is
    made, so no index resource is created and no cleanup is required.

    A valid ``schema`` must be supplied: create() validates the schema before
    the name, so omitting it would make the raise come from the missing-schema
    guard and prove nothing about name validation.
    """
    invalid_names = ["a" * 46, "MyIndex", "my_index", "my.index", "my index"]

    for name in invalid_names:
        with pytest.raises(PineconeValueError):
            client.indexes.create(name=name, schema=_DENSE_SCHEMA_2D)


# error-invalid-deployment-dict  (unified-index-0044)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_create_index_empty_deployment_dict_rejected_client_side() -> None:
    """indexes.create(deployment={}) raises PineconeValueError before any API call.

    Covers one client-side half of unified-index-0044. Uses a fake host so no
    network call is required.
    """
    client = Pinecone(api_key="testkey", host="https://fake-control.pinecone.io")

    with pytest.raises(PineconeValueError):
        client.indexes.create(name="test-idx-deployment", schema=_DENSE_SCHEMA_2D, deployment={})


@pytest.mark.integration
@pytest.mark.parametrize("deployment_type", ["MANAGED", "Serverless", "serverless"])
def test_create_index_invalid_deployment_type_rejected_client_side(deployment_type: str) -> None:
    """A bad ``deployment_type`` raises PineconeValueError before any API call.

    The discriminator match is case-sensitive, so ``"MANAGED"`` is as invalid
    as an unknown value. Asserting the narrow ``PineconeValueError`` (not its
    ``ValueError`` base) is the point: the two are indistinguishable to
    ``pytest.raises(ValueError)``. Uses a fake host — no network call.
    """
    client = Pinecone(api_key="testkey", host="https://fake-control.pinecone.io")

    with pytest.raises(PineconeValueError) as exc_info:
        client.indexes.create(
            name="test-idx-deployment",
            schema=_DENSE_SCHEMA_2D,
            deployment={"deployment_type": deployment_type, "cloud": "aws", "region": "us-east-1"},
        )
    assert type(exc_info.value) is PineconeValueError


@pytest.mark.integration
def test_create_index_unrecognized_deployment_key_rejected_by_server(client: Pinecone) -> None:
    """A non-empty deployment dict with no valid discriminator is rejected with 422.

    Covers the server-side half of unified-index-0044. A deployment dict is a
    ``deployment_type``-discriminated union deserialized with
    ``deny_unknown_fields``, so a dict carrying no ``deployment_type`` key is
    syntactically valid JSON that cannot be deserialized into the target type —
    which the control plane answers with 422, not 400.

    Both dicts below are non-empty, so the SDK forwards them and the rejection
    is genuinely the server's. No index resource is created, so no cleanup is
    required.
    """
    bad_deployments: list[dict[str, Any]] = [
        {"invalid": {"cloud": "aws", "region": "us-east-1"}},
        {"SERVERLESS": {"cloud": "aws", "region": "us-east-1"}},
    ]

    for deployment in bad_deployments:
        with pytest.raises(ApiError) as exc_info:
            client.indexes.create(
                name="test-idx-deployment",
                schema=_DENSE_SCHEMA_2D,
                deployment=deployment,
                timeout=-1,
            )
        assert exc_info.value.status_code == 422
        assert str(exc_info.value)


@pytest.mark.integration
def test_query_input_validation_rest() -> None:
    """query() client-side validation raises PineconeValueError before any API call (REST sync).

    Uses a fake host so no real index or network call is required; all checks
    fire synchronously before the HTTP request would be made.

    Verifies:
    - unified-vec-0038: top_k < 1 is rejected
    - unified-vec-0039: both vector and id supplied is rejected
    - unified-vec-0039: neither vector nor id is rejected
    - unified-vec-0040: positional arguments raise PineconeValueError
    """
    index = Index(host="fake-index.svc.pinecone.io", api_key="testkey")

    # unified-vec-0038: top_k=0 rejected
    with pytest.raises(PineconeValueError):
        index.query(top_k=0, vector=[0.1, 0.2])

    # unified-vec-0038: negative top_k rejected
    with pytest.raises(PineconeValueError):
        index.query(top_k=-5, vector=[0.1, 0.2])

    # unified-vec-0039: both vector and id rejected
    with pytest.raises(PineconeValueError):
        index.query(top_k=5, vector=[0.1, 0.2], id="some-id")

    # unified-vec-0039: neither vector nor id rejected
    with pytest.raises(PineconeValueError):
        index.query(top_k=5)

    # unified-vec-0040: positional arguments raise a clear PineconeValueError
    with pytest.raises(PineconeValueError, match="keyword-only"):
        index.query([0.1, 0.2], 5)  # type: ignore[misc]


@pytest.mark.integration
def test_update_input_validation_rest() -> None:
    """update() client-side validation raises PineconeValueError before any API call (REST sync).

    Uses a fake host so no real index or network call is required; all checks
    fire synchronously before the HTTP request would be made.

    Verifies:
    - unified-vec-0042: both id and filter rejected
    - unified-vec-0042: neither id nor filter rejected
    - update() uses keyword-only params (PineconeValueError on positional args)
    """
    index = Index(host="fake-index.svc.pinecone.io", api_key="testkey")

    # unified-vec-0042: both id and filter rejected
    with pytest.raises(PineconeValueError):
        index.update(id="some-id", filter={"genre": {"$eq": "drama"}}, set_metadata={"x": 1})

    # unified-vec-0042: neither id nor filter rejected
    with pytest.raises(PineconeValueError):
        index.update(set_metadata={"x": 1})

    # update() uses keyword-only params — positional call raises a clear PineconeValueError
    with pytest.raises(PineconeValueError, match="keyword-only"):
        index.update("some-id")  # type: ignore[misc]


@pytest.mark.integration
def test_fetch_empty_ids_list_raises_value_error() -> None:
    """fetch(ids=[]) raises PineconeValueError before any API call (REST sync).

    Uses a fake host so no real index or network call is required; the empty-list
    check fires synchronously before the HTTP request would be made.
    """
    index = Index(host="fake-index.svc.pinecone.io", api_key="testkey")

    with pytest.raises(PineconeValueError, match="ids"):
        index.fetch(ids=[])


@pytest.mark.integration
def test_query_input_validation_grpc() -> None:
    """query() client-side validation raises PineconeValueError before any gRPC call.

    All validations fire before the gRPC channel call, so no real server is needed.

    Verifies:
    - unified-vec-0038: top_k < 1 is rejected
    - unified-vec-0039: both vector and id supplied is rejected
    - unified-vec-0039: neither vector nor id is rejected
    - unified-vec-0040: positional arguments raise PineconeValueError
    """
    index = GrpcIndex(host="fake-index.svc.pinecone.io", api_key="testkey")

    # unified-vec-0038: top_k=0 rejected
    with pytest.raises(PineconeValueError):
        index.query(top_k=0, vector=[0.1, 0.2])

    # unified-vec-0038: negative top_k rejected
    with pytest.raises(PineconeValueError):
        index.query(top_k=-3, vector=[0.1, 0.2])

    # unified-vec-0039: both vector and id rejected
    with pytest.raises(PineconeValueError):
        index.query(top_k=5, vector=[0.1, 0.2], id="some-id")

    # unified-vec-0039: neither vector nor id rejected
    with pytest.raises(PineconeValueError):
        index.query(top_k=5)

    # unified-vec-0040: positional arguments raise a clear PineconeValueError
    with pytest.raises(PineconeValueError, match="keyword-only"):
        index.query([0.1, 0.2], 5)  # type: ignore[misc]


@pytest.mark.integration
def test_update_input_validation_grpc() -> None:
    """update() client-side validation raises PineconeValueError before any gRPC call.

    Uses a fake host so no real server is needed; validation fires before
    the gRPC channel is called.

    Verifies:
    - unified-vec-0042: both id and filter rejected
    - unified-vec-0042: neither id nor filter rejected
    """
    index = GrpcIndex(host="fake-index.svc.pinecone.io", api_key="testkey")

    # unified-vec-0042: both id and filter rejected
    with pytest.raises(PineconeValueError):
        index.update(id="some-id", filter={"genre": {"$eq": "drama"}}, set_metadata={"x": 1})

    # unified-vec-0042: neither id nor filter rejected
    with pytest.raises(PineconeValueError):
        index.update(set_metadata={"x": 1})


# ---------------------------------------------------------------------------
# namespace-name-must-be-string — REST sync
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_namespace_name_must_be_string_rest() -> None:
    """create_namespace(), describe_namespace(), and delete_namespace() raise
    PineconeValueError when the name parameter is not a string.

    Validation fires client-side before any HTTP request, so a fake host is
    sufficient — no real index or API call is needed.

    Verifies:
    - unified-ns-0011: Namespace operations require the namespace parameter to be a string.
    """
    # Fake host: contains a dot so it passes the host URL format check.
    index = Index(host="fake-index.svc.pinecone.io", api_key="testkey")

    non_string_values = [42, None, ["my-ns"], True]

    for bad_name in non_string_values:
        # create_namespace rejects non-string name
        with pytest.raises(PineconeValueError, match="string"):
            index.create_namespace(name=bad_name)  # type: ignore[arg-type]

        # describe_namespace rejects non-string name
        with pytest.raises(PineconeValueError, match="string"):
            index.describe_namespace(name=bad_name)  # type: ignore[arg-type]

        # delete_namespace rejects non-string name
        with pytest.raises(PineconeValueError, match="string"):
            index.delete_namespace(name=bad_name)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# error-exception-attributes  (unified-http-0017)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_api_error_exposes_status_reason_headers_body_rest(client: Pinecone) -> None:
    """ApiError subclasses expose status_code, reason, headers, and body for error diagnosis.

    Verifies unified-http-0017: all API exception objects carry diagnostic
    fields populated from the HTTP response.  Two real API call failure paths
    are exercised:

    1. UnauthorizedError (401) — bad API key
    2. NotFoundError (404)     — describe a nonexistent index

    For each:
    - status_code is an int matching the HTTP status
    - reason is a non-empty string (HTTP reason phrase, e.g. "Unauthorized")
    - headers is a non-empty dict (at minimum Content-Type is always present)
    - body attribute exists and is either a dict or None
    """
    # --- 1. UnauthorizedError (401) from a bad API key ---
    bad_client = Pinecone(api_key="invalid-key-for-attribute-test")
    with pytest.raises(UnauthorizedError) as exc_info:
        bad_client.indexes.list().to_list()

    err = exc_info.value
    # status_code is correct int
    assert err.status_code == 401
    assert isinstance(err.status_code, int)
    # reason is a non-empty string
    assert err.reason is not None
    assert isinstance(err.reason, str)
    assert len(err.reason) > 0
    # headers is a non-empty dict (API always returns at least Content-Type)
    assert err.headers is not None
    assert isinstance(err.headers, dict)
    assert len(err.headers) > 0
    # body is either a dict or None (attribute must exist)
    assert err.body is None or isinstance(err.body, dict)

    # --- 2. NotFoundError (404) from describing a nonexistent index ---
    with pytest.raises(NotFoundError) as exc_info2:
        client.indexes.describe("index-does-not-exist-attr-test")

    err2 = exc_info2.value
    assert err2.status_code == 404
    assert isinstance(err2.status_code, int)
    assert err2.reason is not None
    assert isinstance(err2.reason, str)
    assert len(err2.reason) > 0
    assert err2.headers is not None
    assert isinstance(err2.headers, dict)
    assert len(err2.headers) > 0
    assert err2.body is None or isinstance(err2.body, dict)


# ---------------------------------------------------------------------------
# exception-catch-hierarchy — REST sync
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_exception_catch_hierarchy_rest(client: Pinecone) -> None:
    """SDK exceptions are catchable via their base class hierarchy (REST sync).

    Verifies:
    - unified-err-0001: All SDK exceptions inherit from PineconeError, allowing
      any SDK-raised exception to be caught with a single `except PineconeError:`
      handler.
    - unified-err-0003: PineconeValueError inherits from both PineconeError and
      ValueError; PineconeTypeError inherits from both PineconeError and TypeError.

    No resources are created or cleaned up. Validation errors fire client-side
    before any HTTP request. The HTTP error variant exercises a real 404.
    """
    index = Index(host="fake-index.svc.pinecone.io", api_key="testkey")

    # --- unified-err-0003: PineconeValueError is catchable as ValueError ---
    caught = False
    try:
        index.query(top_k=0, vector=[0.1, 0.2])  # top_k < 1 raises PineconeValueError
    except ValueError:
        caught = True
    assert caught, "PineconeValueError must be catchable as ValueError (unified-err-0003)"

    # --- unified-err-0001: PineconeValueError is catchable as PineconeError ---
    caught = False
    try:
        index.query(top_k=0, vector=[0.1, 0.2])
    except PineconeError:
        caught = True
    assert caught, "PineconeValueError must be catchable as PineconeError (unified-err-0001)"

    # --- unified-err-0003: PineconeTypeError is catchable as TypeError ---
    caught = False
    try:
        client.inference.embed(
            model="multilingual-e5-large",
            inputs=42,  # type: ignore[arg-type]
        )
    except TypeError:
        caught = True
    assert caught, "PineconeTypeError must be catchable as TypeError (unified-err-0003)"

    # --- unified-err-0001: PineconeTypeError is catchable as PineconeError ---
    caught = False
    try:
        client.inference.embed(
            model="multilingual-e5-large",
            inputs=42,  # type: ignore[arg-type]
        )
    except PineconeError:
        caught = True
    assert caught, "PineconeTypeError must be catchable as PineconeError (unified-err-0001)"

    # --- unified-err-0001: ApiError (HTTP 404) is catchable as PineconeError ---
    caught = False
    try:
        client.indexes.describe("index-that-does-not-exist-hierarchy-xyz")
    except PineconeError:
        caught = True
    assert caught, (
        "NotFoundError (ApiError subclass) must be catchable as PineconeError (unified-err-0001)"
    )


# ---------------------------------------------------------------------------
# error-grpc-deadline-exceeded  (grpc-timeout-0001)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(600)
def test_grpc_query_too_short_timeout_raises(client: Pinecone) -> None:
    """A gRPC query with a sub-millisecond per-call timeout raises PineconeTimeoutError.

    Verifies that:
    1. The per-call timeout knob on GrpcIndex.query() is wired through to the
       Rust transport (timeout_s= parameter on GrpcChannelProtocol.query).
    2. When the deadline fires, the Rust channel raises an exception containing
       DEADLINE_EXCEEDED, which the Rust transport maps to PineconeTimeoutError.
    3. A subsequent query with a generous timeout succeeds — confirming the
       timeout is per-call and does not permanently break the channel.

    Seeding goes through the REST documents API because GrpcIndex has no
    documents write method and a schema-bearing index refuses gRPC vector
    writes. gRPC *reads* are unaffected, which is all this test needs.
    """
    name = unique_name("idx")
    namespace = "grpc-timeout-ns"
    try:
        client.indexes.create(
            name=name,
            schema=_DENSE_SCHEMA_3D,
            deployment=MANAGED_AWS,
            timeout=300,
        )
        client.index(name=name).documents.upsert(
            namespace=namespace,
            documents=[
                {"_id": "t1", DENSE_FIELD: [0.1, 0.2, 0.3]},
                {"_id": "t2", DENSE_FIELD: [0.4, 0.5, 0.6]},
                {"_id": "t3", DENSE_FIELD: [0.7, 0.8, 0.9]},
            ],
        )
        grpc_idx = client.index(name=name, grpc=True)

        # Wait until the upserted vectors are queryable
        poll_until(
            lambda: grpc_idx.query(vector=[0.1, 0.2, 0.3], top_k=3, namespace=namespace),
            lambda r: len(r.matches) > 0,
            timeout=60,
            description="vectors queryable via gRPC",
        )

        # Sub-microsecond deadline — must fire before the server responds
        with pytest.raises(PineconeTimeoutError):
            grpc_idx.query(vector=[0.1, 0.2, 0.3], top_k=3, namespace=namespace, timeout=0.000001)

        # Generous timeout: proves the channel is healthy and the knob is per-call
        result = grpc_idx.query(vector=[0.1, 0.2, 0.3], top_k=3, namespace=namespace, timeout=30)
        assert result.matches is not None
    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")
