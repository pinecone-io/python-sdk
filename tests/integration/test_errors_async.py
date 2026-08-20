"""Integration tests for error paths (async / REST async).

Tests verify that the async SDK raises typed, human-readable exceptions rather
than raw HTTP errors or generic exceptions.

As of 2026-07 ``indexes.list()`` returns a lazy ``AsyncPaginator`` and is no
longer a coroutine, so it must not be awaited directly; a test expecting a
transport error from a paginated operation awaits the paginator's ``.to_list()``
inside the ``pytest.raises`` block — otherwise nothing is sent and nothing raises.
"""

from __future__ import annotations

import pytest

from pinecone import AsyncIndex, AsyncPinecone, PineconeValueError
from pinecone.errors import ApiError, ConflictError, NotFoundError, PineconeError, UnauthorizedError
from pinecone.models.indexes.specs import ServerlessSpec
from tests.integration.conftest import async_cleanup_resource, unique_name

# ---------------------------------------------------------------------------
# error-bad-api-key
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_bad_api_key_raises_typed_exception_async() -> None:
    """AsyncPinecone(api_key="invalid") + indexes.list() raises UnauthorizedError (not raw HTTP error)."""
    async with AsyncPinecone(api_key="invalid-key-12345") as bad_client:
        with pytest.raises(UnauthorizedError) as exc_info:
            await bad_client.indexes.list().to_list()

    err = exc_info.value
    assert isinstance(err, ApiError)
    assert err.status_code == 401
    # Error message must be human-readable (non-empty)
    assert str(err)


@pytest.mark.integration
@pytest.mark.anyio
async def test_bad_api_key_error_message_is_human_readable_async() -> None:
    """UnauthorizedError from a bad API key has a non-empty, informative message."""
    async with AsyncPinecone(api_key="totally-wrong-key-xyz") as bad_client:
        with pytest.raises(UnauthorizedError) as exc_info:
            await bad_client.indexes.list().to_list()

    err = exc_info.value
    msg = str(err)
    assert len(msg) > 0
    assert not msg.strip().isdigit()


# ---------------------------------------------------------------------------
# error-nonexistent-index
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_describe_nonexistent_index_raises_not_found_async(
    async_client: AsyncPinecone,
) -> None:
    """indexes.describe() on a non-existent name raises NotFoundError (typed, status_code=404)."""
    with pytest.raises(NotFoundError) as exc_info:
        await async_client.indexes.describe("index-that-does-not-exist-xyz")

    err = exc_info.value
    assert isinstance(err, ApiError)
    assert err.status_code == 404
    # Error message must be human-readable (non-empty, not just a number)
    msg = str(err)
    assert len(msg) > 0
    assert not msg.strip().isdigit()


@pytest.mark.integration
@pytest.mark.anyio
async def test_delete_nonexistent_index_raises_not_found_async(
    async_client: AsyncPinecone,
) -> None:
    """indexes.delete() on a non-existent name raises NotFoundError (typed, status_code=404)."""
    with pytest.raises(NotFoundError) as exc_info:
        await async_client.indexes.delete("index-that-does-not-exist-xyz")

    err = exc_info.value
    assert isinstance(err, ApiError)
    assert err.status_code == 404


# ---------------------------------------------------------------------------
# error-dimension-mismatch
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_dimension_mismatch_raises_typed_error_async(
    async_client: AsyncPinecone,
) -> None:
    """Upsert a 3-dim vector into a 2-dim index raises ApiError (status_code=400, REST async)."""
    name = unique_name("idx")
    try:
        await async_client.indexes.create(
            name=name,
            dimension=2,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )
        # Populate host cache so pc.index(name=...) can resolve it
        await async_client.indexes.describe(name)
        index = await async_client.index(name=name)

        with pytest.raises(ApiError) as exc_info:
            await index.upsert(vectors=[{"id": "dim-v1", "values": [0.1, 0.2, 0.3]}])

        err = exc_info.value
        assert err.status_code == 400
        msg = str(err)
        assert len(msg) > 0
        assert not msg.strip().isdigit()
    finally:
        await async_cleanup_resource(lambda: async_client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# error-duplicate-index
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_duplicate_index_raises_conflict_error_async(
    async_client: AsyncPinecone,
) -> None:
    """Creating an index with a name that already exists raises ConflictError (status_code=409, REST async)."""
    name = unique_name("idx")
    try:
        await async_client.indexes.create(
            name=name,
            dimension=2,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )

        with pytest.raises(ConflictError) as exc_info:
            await async_client.indexes.create(
                name=name,
                dimension=2,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                timeout=-1,  # skip waiting — index already exists
            )

        err = exc_info.value
        assert isinstance(err, ApiError)
        assert err.status_code == 409
        msg = str(err)
        assert len(msg) > 0
        assert not msg.strip().isdigit()
    finally:
        await async_cleanup_resource(lambda: async_client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# error-invalid-host  (unified-index-0043 + unified-index-0048)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_invalid_index_host_raises_value_error_async() -> None:
    """AsyncIndex raises PineconeValueError for hosts without a dot or 'localhost'.

    Verifies unified-index-0043 for the async REST transport: host URL
    validation fires at construction time, before any network call.

    Also verifies unified-index-0048: AsyncPinecone raises NotImplementedError
    when proxy_headers are supplied (not yet supported for the async client).
    """
    # AsyncIndex: no-dot host rejected
    with pytest.raises(PineconeValueError):
        AsyncIndex(host="nodot", api_key="testkey")

    # AsyncIndex: empty string rejected
    with pytest.raises(PineconeValueError):
        AsyncIndex(host="", api_key="testkey")

    # unified-index-0048: proxy_headers not yet supported for async client
    with pytest.raises(NotImplementedError):
        AsyncPinecone(api_key="testkey", proxy_headers={"X-Proxy-Auth": "secret"})


# ---------------------------------------------------------------------------
# error-invalid-index-name  (unified-index-0045)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_create_index_invalid_name_async(async_client: AsyncPinecone) -> None:
    """async indexes.create() rejects invalid index names before any API call.

    Verifies unified-index-0045 for the async transport: validate_create_inputs()
    raises PineconeValueError synchronously before the first await, so no real
    HTTP request is made and no index resource needs cleanup.

    Cases checked:
    - name with 46 characters (one over the 45-character limit)
    - name with uppercase letters
    - name with an underscore
    - name with a dot
    - name with a space
    """
    spec = ServerlessSpec(cloud="aws", region="us-east-1")

    # 46-character name — one character over the 45-character limit
    long_name = "a" * 46
    with pytest.raises(PineconeValueError):
        await async_client.indexes.create(
            name=long_name,
            dimension=2,
            spec=spec,
        )

    # uppercase letters are not allowed
    with pytest.raises(PineconeValueError):
        await async_client.indexes.create(
            name="MyIndex",
            dimension=2,
            spec=spec,
        )

    # underscore is not allowed (only hyphens)
    with pytest.raises(PineconeValueError):
        await async_client.indexes.create(
            name="my_index",
            dimension=2,
            spec=spec,
        )

    # dot is not allowed
    with pytest.raises(PineconeValueError):
        await async_client.indexes.create(
            name="my.index",
            dimension=2,
            spec=spec,
        )

    # space is not allowed
    with pytest.raises(PineconeValueError):
        await async_client.indexes.create(
            name="my index",
            dimension=2,
            spec=spec,
        )


# ---------------------------------------------------------------------------
# error-invalid-spec-dict-key  (unified-index-0044)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_create_index_invalid_spec_dict_key_async(async_client: AsyncPinecone) -> None:
    """async indexes.create() with a spec dict missing a recognized key raises PineconeValueError.

    Verifies unified-index-0044 for the async transport: validation fires before
    any awaited HTTP request, so no index resource is created.
    """
    # empty spec dict
    with pytest.raises(PineconeValueError):
        await async_client.indexes.create(
            name="test-idx-spec",
            dimension=2,
            spec={},
        )

    # unrecognized key
    with pytest.raises(PineconeValueError):
        await async_client.indexes.create(
            name="test-idx-spec",
            dimension=2,
            spec={"invalid": {"cloud": "aws", "region": "us-east-1"}},
        )

    # case-sensitive: 'SERVERLESS' is not recognized
    with pytest.raises(PineconeValueError):
        await async_client.indexes.create(
            name="test-idx-spec",
            dimension=2,
            spec={"SERVERLESS": {"cloud": "aws", "region": "us-east-1"}},
        )


# ---------------------------------------------------------------------------
# error-query-validation  (unified-vec-0038, unified-vec-0039, unified-vec-0040)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_query_input_validation_async() -> None:
    """async query() client-side validation raises PineconeValueError before any API call.

    Uses a fake host so no real index or network call is required; validation
    fires synchronously inside the async function before the HTTP request.

    Verifies:
    - unified-vec-0038: top_k < 1 is rejected
    - unified-vec-0039: both vector and id supplied is rejected
    - unified-vec-0039: neither vector nor id is rejected
    - unified-vec-0040: positional arguments raise PineconeValueError
    """
    index = AsyncIndex(host="fake-index.svc.pinecone.io", api_key="testkey")
    try:
        # unified-vec-0038: top_k=0 rejected
        with pytest.raises(PineconeValueError):
            await index.query(top_k=0, vector=[0.1, 0.2])

        # unified-vec-0038: negative top_k rejected
        with pytest.raises(PineconeValueError):
            await index.query(top_k=-5, vector=[0.1, 0.2])

        # unified-vec-0039: both vector and id rejected
        with pytest.raises(PineconeValueError):
            await index.query(top_k=5, vector=[0.1, 0.2], id="some-id")

        # unified-vec-0039: neither vector nor id rejected
        with pytest.raises(PineconeValueError):
            await index.query(top_k=5)

        # unified-vec-0040: positional arguments raise a clear PineconeValueError
        with pytest.raises(PineconeValueError, match="keyword-only"):
            await index.query([0.1, 0.2], 5)  # type: ignore[misc]
    finally:
        await index.close()


@pytest.mark.integration
@pytest.mark.anyio
async def test_update_input_validation_async() -> None:
    """update() client-side validation raises PineconeValueError before any API call (async REST).

    Uses a fake host so no real index or network call is required; all checks
    fire synchronously before any await would be reached.

    Verifies:
    - unified-vec-0042: both id and filter rejected
    - unified-vec-0042: neither id nor filter rejected
    - update() uses keyword-only params (PineconeValueError on positional args)
    """
    index = AsyncIndex(host="fake-index.svc.pinecone.io", api_key="testkey")
    try:
        # unified-vec-0042: both id and filter rejected
        with pytest.raises(PineconeValueError):
            await index.update(
                id="some-id", filter={"genre": {"$eq": "drama"}}, set_metadata={"x": 1}
            )

        # unified-vec-0042: neither id nor filter rejected
        with pytest.raises(PineconeValueError):
            await index.update(set_metadata={"x": 1})

        # update() uses keyword-only params — positional call raises a clear PineconeValueError
        with pytest.raises(PineconeValueError, match="keyword-only"):
            await index.update("some-id")  # type: ignore[misc]
    finally:
        await index.close()


@pytest.mark.integration
@pytest.mark.anyio
async def test_fetch_empty_ids_list_raises_value_error_async() -> None:
    """fetch(ids=[]) raises PineconeValueError before any API call (async REST).

    Uses a fake host so no real index or network call is required; the empty-list
    check fires before any await would be reached.
    """
    index = AsyncIndex(host="fake-index.svc.pinecone.io", api_key="testkey")
    try:
        with pytest.raises(PineconeValueError, match="ids"):
            await index.fetch(ids=[])
    finally:
        await index.close()


# ---------------------------------------------------------------------------
# namespace-name-must-be-string — REST async
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_namespace_name_must_be_string_async() -> None:
    """create_namespace(), describe_namespace(), and delete_namespace() raise
    PineconeValueError when the name parameter is not a string (async REST).

    Validation fires client-side before any HTTP request, so a fake host is
    sufficient — no real index or API call is needed.

    Verifies:
    - unified-ns-0011: Namespace operations require the namespace parameter to be a string.
    """
    index = AsyncIndex(host="fake-index.svc.pinecone.io", api_key="testkey")
    try:
        non_string_values = [42, None, ["my-ns"], True]

        for bad_name in non_string_values:
            # create_namespace rejects non-string name
            with pytest.raises(PineconeValueError, match="string"):
                await index.create_namespace(name=bad_name)  # type: ignore[arg-type]

            # describe_namespace rejects non-string name
            with pytest.raises(PineconeValueError, match="string"):
                await index.describe_namespace(name=bad_name)  # type: ignore[arg-type]

            # delete_namespace rejects non-string name
            with pytest.raises(PineconeValueError, match="string"):
                await index.delete_namespace(name=bad_name)  # type: ignore[arg-type]
    finally:
        await index.close()


# ---------------------------------------------------------------------------
# error-exception-attributes  (unified-http-0017)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_api_error_exposes_status_reason_headers_body_async(
    async_client: AsyncPinecone,
) -> None:
    """ApiError subclasses expose status_code, reason, headers, and body (async REST).

    Verifies unified-http-0017: all API exception objects carry diagnostic
    fields populated from the HTTP response.  Two real API call failure paths
    are exercised:

    1. UnauthorizedError (401) — bad API key
    2. NotFoundError (404)     — describe a nonexistent index

    For each:
    - status_code is an int matching the HTTP status
    - reason is a non-empty string (HTTP reason phrase)
    - headers is a non-empty dict
    - body attribute exists and is either a dict or None
    """
    # --- 1. UnauthorizedError (401) from a bad API key ---
    async with AsyncPinecone(api_key="invalid-key-for-attribute-test") as bad_client:
        with pytest.raises(UnauthorizedError) as exc_info:
            await bad_client.indexes.list().to_list()

    err = exc_info.value
    assert err.status_code == 401
    assert isinstance(err.status_code, int)
    assert err.reason is not None
    assert isinstance(err.reason, str)
    assert len(err.reason) > 0
    assert err.headers is not None
    assert isinstance(err.headers, dict)
    assert len(err.headers) > 0
    assert err.body is None or isinstance(err.body, dict)

    # --- 2. NotFoundError (404) from describing a nonexistent index ---
    with pytest.raises(NotFoundError) as exc_info2:
        await async_client.indexes.describe("index-does-not-exist-attr-test")

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
# exception-catch-hierarchy — REST async
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_exception_catch_hierarchy_async(async_client: AsyncPinecone) -> None:
    """SDK exceptions are catchable via their base class hierarchy (async REST).

    Async transport parity for test_exception_catch_hierarchy_rest.

    Verifies:
    - unified-err-0001: All SDK exceptions inherit from PineconeError.
    - unified-err-0003: PineconeValueError inherits from both PineconeError and
      ValueError; PineconeTypeError inherits from both PineconeError and TypeError.
    """
    index = AsyncIndex(host="fake-index.svc.pinecone.io", api_key="testkey")

    # --- unified-err-0003: PineconeValueError is catchable as ValueError ---
    caught = False
    try:
        await index.query(top_k=0, vector=[0.1, 0.2])  # top_k < 1 raises PineconeValueError
    except ValueError:
        caught = True
    assert caught, "Async PineconeValueError must be catchable as ValueError (unified-err-0003)"

    # --- unified-err-0001: PineconeValueError is catchable as PineconeError ---
    caught = False
    try:
        await index.query(top_k=0, vector=[0.1, 0.2])
    except PineconeError:
        caught = True
    assert caught, "Async PineconeValueError must be catchable as PineconeError (unified-err-0001)"

    # --- unified-err-0003: PineconeTypeError is catchable as TypeError ---
    caught = False
    try:
        await async_client.inference.embed(
            model="multilingual-e5-large",
            inputs=42,  # type: ignore[arg-type]
        )
    except TypeError:
        caught = True
    assert caught, "Async PineconeTypeError must be catchable as TypeError (unified-err-0003)"

    # --- unified-err-0001: PineconeTypeError is catchable as PineconeError ---
    caught = False
    try:
        await async_client.inference.embed(
            model="multilingual-e5-large",
            inputs=42,  # type: ignore[arg-type]
        )
    except PineconeError:
        caught = True
    assert caught, "Async PineconeTypeError must be catchable as PineconeError (unified-err-0001)"

    # --- unified-err-0001: ApiError (HTTP 404) is catchable as PineconeError ---
    caught = False
    try:
        await async_client.indexes.describe("index-that-does-not-exist-hierarchy-xyz")
    except PineconeError:
        caught = True
    assert caught, (
        "Async NotFoundError (ApiError subclass) must be catchable as PineconeError "
        "(unified-err-0001)"
    )
    await index.close()
