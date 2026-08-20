"""Client-side validation for the 2026-07 vector operations, async lane.

The async twin of ``tests/unit/test_vector_op_validation.py`` (#122 ∥ #123). The
rules live once in ``pinecone/_internal/validation.py`` and the expected messages
live once in that file's ``VECTOR_OP_VALIDATION_CASES`` and ``QUERY_TRUTH_TABLE``
tables; here each case id is bound to an :class:`AsyncIndex` call instead of being
restated, so the two lanes cannot drift apart in wording — a message that changes
in one fails in both.

Spec, backend and simulator provenance for every rule asserted here is recorded in
the module docstring of ``tests/unit/test_vector_op_validation.py`` rather than
duplicated: this file adds no rule of its own, it only replays the REST lane's
rules against ``AsyncIndex``.

Two accepted divergences from "tighten everything" are pinned here as they are in
the REST lane, because a later reader is more likely to add them than to remove
them: ``query``'s ``filter`` is not emptiness-checked (``QueryRequest.filter``
carries no ``minProperties`` and the backend accepts an empty struct), and
``describe_index_stats`` forwards its filter rather than second-guessing whether
this index supports filtered stats.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import httpx
import orjson
import pytest
import respx
from hypothesis import given
from hypothesis import strategies as st

from pinecone import AsyncIndex
from pinecone._internal.constants import DATA_PLANE_API_VERSION
from pinecone.errors.exceptions import ApiError, ValidationError
from tests.unit.test_vector_op_validation import (
    FBM_LIMIT_MAX,
    FILTER,
    ID_MAX,
    LIST_LIMIT_MAX,
    OVERLONG,
    PREFIX_MAX,
    QUERY_TRUTH_TABLE,
    SPARSE,
    VECTOR,
    VECTOR_ID,
    VECTOR_OP_VALIDATION_CASES,
    query_kwargs,
)

INDEX_HOST = "async-vec-validation-abc123.svc.pinecone.io"
BASE_URL = f"https://{INDEX_HOST}"


@pytest.fixture
async def index() -> AsyncIterator[AsyncIndex]:
    client = AsyncIndex(host=INDEX_HOST, api_key="test-key")
    yield client
    await client.close()


def _query_response() -> dict[str, Any]:
    return {"matches": [], "namespace": "", "usage": {"readUnits": 1}}


def _fetch_response() -> dict[str, Any]:
    return {"vectors": {}, "namespace": "", "usage": {"readUnits": 1}}


def _list_response() -> dict[str, Any]:
    return {"vectors": [], "namespace": "", "usage": {"readUnits": 1}}


AsyncInvoke = Callable[[AsyncIndex], Awaitable[object]]

ASYNC_INVOCATIONS: dict[str, AsyncInvoke] = {
    "query_id_and_vector": lambda idx: idx.query(top_k=1, id=VECTOR_ID, vector=VECTOR),
    "query_id_and_sparse_vector": lambda idx: idx.query(
        top_k=1, id=VECTOR_ID, sparse_vector=SPARSE
    ),
    "query_no_selector": lambda idx: idx.query(top_k=1),
    "query_non_ascii_id": lambda idx: idx.query(top_k=1, id="naïve"),
    "query_overlong_id": lambda idx: idx.query(top_k=1, id=OVERLONG),
    "delete_no_selector": lambda idx: idx.delete(),
    "delete_ids_and_filter": lambda idx: idx.delete(ids=[VECTOR_ID], filter=FILTER),
    "delete_ids_and_delete_all": lambda idx: idx.delete(ids=[VECTOR_ID], delete_all=True),
    "delete_all_and_filter": lambda idx: idx.delete(delete_all=True, filter=FILTER),
    "delete_empty_filter": lambda idx: idx.delete(filter={}),
    "delete_empty_ids": lambda idx: idx.delete(ids=[]),
    "delete_nul_id": lambda idx: idx.delete(ids=["a\x00b"]),
    "update_filter_and_values": lambda idx: idx.update(filter=FILTER, values=VECTOR),
    "update_filter_and_sparse_values": lambda idx: idx.update(filter=FILTER, sparse_values=SPARSE),
    "update_empty_filter": lambda idx: idx.update(filter={}, set_metadata={"year": 2020}),
    "update_id_and_filter": lambda idx: idx.update(id=VECTOR_ID, filter=FILTER),
    "update_no_target": lambda idx: idx.update(values=VECTOR),
    "update_empty_id": lambda idx: idx.update(id="", values=VECTOR),
    "fetch_empty_ids": lambda idx: idx.fetch(ids=[]),
    "fetch_non_ascii_id": lambda idx: idx.fetch(ids=["ok", "naïve"]),
    "fetch_overlong_id": lambda idx: idx.fetch(ids=[OVERLONG]),
    "fetch_empty_id": lambda idx: idx.fetch(ids=[""]),
    "fetch_nul_id": lambda idx: idx.fetch(ids=["a\x00b"]),
    "fetch_by_metadata_empty_filter": lambda idx: idx.fetch_by_metadata(filter={}),
    "fetch_by_metadata_limit_zero": lambda idx: idx.fetch_by_metadata(filter=FILTER, limit=0),
    "fetch_by_metadata_limit_over_max": lambda idx: idx.fetch_by_metadata(
        filter=FILTER, limit=FBM_LIMIT_MAX + 1
    ),
    "list_limit_zero": lambda idx: idx.list_paginated(limit=0),
    "list_limit_over_max": lambda idx: idx.list_paginated(limit=LIST_LIMIT_MAX + 1),
    "list_prefix_non_ascii": lambda idx: idx.list_paginated(prefix="naïve"),
    "list_prefix_overlong": lambda idx: idx.list_paginated(prefix=OVERLONG),
    "list_prefix_nul": lambda idx: idx.list_paginated(prefix="a\x00b"),
    "list_generator_limit_over_max": lambda idx: anext(idx.list(limit=LIST_LIMIT_MAX + 1)),
}
"""One :class:`AsyncIndex` call per case id in the REST lane's rejection table."""

ACCEPTED_INVOCATIONS: dict[str, AsyncInvoke] = {
    "query": lambda idx: idx.query(top_k=1, vector=VECTOR),
    "fetch": lambda idx: idx.fetch(ids=["vec-1"]),
    "fetch_by_metadata": lambda idx: idx.fetch_by_metadata(filter=FILTER),
    "delete": lambda idx: idx.delete(ids=["vec-1"]),
    "update": lambda idx: idx.update(id="vec-1", values=VECTOR),
    "list_paginated": lambda idx: idx.list_paginated(limit=10),
    "describe_index_stats": lambda idx: idx.describe_index_stats(),
}
"""The minimal legal call for each operation in this ticket's op set."""


def test_every_shared_case_is_bound_to_an_async_call() -> None:
    """A rule the REST lane grows has to be exercised here too, or parity is untested."""
    shared = {case_id for case_id, _fn, _msg in VECTOR_OP_VALIDATION_CASES}
    assert set(ASYNC_INVOCATIONS) == shared, (
        f"unbound in async={sorted(shared - set(ASYNC_INVOCATIONS))}, "
        f"unknown to the shared table={sorted(set(ASYNC_INVOCATIONS) - shared)}"
    )


@pytest.mark.parametrize(
    ("case_id", "expected"),
    [pytest.param(case_id, msg, id=case_id) for case_id, _fn, msg in VECTOR_OP_VALIDATION_CASES],
)
async def test_rejected_before_any_http_call(
    index: AsyncIndex, case_id: str, expected: str, respx_mock: respx.MockRouter
) -> None:
    with pytest.raises(ValidationError) as excinfo:
        await ASYNC_INVOCATIONS[case_id](index)

    assert str(excinfo.value) == expected
    assert not respx_mock.calls, "validation must reject before the request goes out"


@pytest.mark.parametrize(
    ("has_vector", "has_id", "has_sparse", "accepted"),
    [pytest.param(v, i, s, ok, id=case_id) for case_id, v, i, s, ok in QUERY_TRUTH_TABLE],
)
async def test_query_selector_truth_table(
    index: AsyncIndex,
    has_vector: bool,
    has_id: bool,
    has_sparse: bool,
    accepted: bool,
    respx_mock: respx.MockRouter,
) -> None:
    route = respx_mock.post(f"{BASE_URL}/query").mock(
        return_value=httpx.Response(200, json=_query_response())
    )
    kwargs = query_kwargs(has_vector, has_id, has_sparse)

    if accepted:
        await index.query(**kwargs)
        assert route.called
    else:
        with pytest.raises(ValidationError):
            await index.query(**kwargs)
        assert not respx_mock.calls, "rejection must happen before the request goes out"


async def test_query_sends_exactly_the_selectors_it_was_given(
    index: AsyncIndex, respx_mock: respx.MockRouter
) -> None:
    """A hybrid query carries both literal forms and no ``id``."""
    route = respx_mock.post(f"{BASE_URL}/query").mock(
        return_value=httpx.Response(200, json=_query_response())
    )
    await index.query(top_k=3, vector=VECTOR, sparse_vector=SPARSE)

    body = orjson.loads(route.calls.last.request.content)
    assert body["vector"] == VECTOR
    assert body["sparseVector"] == SPARSE
    assert "id" not in body


@pytest.mark.parametrize(
    "vector_id",
    [
        pytest.param("a", id="single_char"),
        pytest.param("a" * ID_MAX, id="at_length_limit"),
        pytest.param("doc1#chunk2", id="hash_separator"),
        pytest.param(" ", id="single_space"),
        pytest.param("\x01", id="lowest_legal_code_point"),
        pytest.param("\x7f", id="highest_legal_code_point"),
        pytest.param("id/with/slashes", id="slashes_are_legal_in_a_query_param"),
    ],
)
async def test_fetch_accepts_legal_ids(
    index: AsyncIndex, vector_id: str, respx_mock: respx.MockRouter
) -> None:
    """The rules must not over-reject: every ID the spec admits has to reach the wire."""
    route = respx_mock.get(url__startswith=f"{BASE_URL}/vectors/fetch").mock(
        return_value=httpx.Response(200, json=_fetch_response())
    )
    await index.fetch(ids=[vector_id])

    assert route.calls.last.request.url.params.get_list("ids") == [vector_id]


async def test_fetch_accepts_many_ids_and_sends_them_all(
    index: AsyncIndex, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.get(url__startswith=f"{BASE_URL}/vectors/fetch").mock(
        return_value=httpx.Response(200, json=_fetch_response())
    )
    ids = [f"vec-{n}" for n in range(50)]
    await index.fetch(ids=ids)

    assert route.calls.last.request.url.params.get_list("ids") == ids


async def test_query_accepts_legal_id(index: AsyncIndex, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.post(f"{BASE_URL}/query").mock(
        return_value=httpx.Response(200, json=_query_response())
    )
    await index.query(top_k=1, id="a" * ID_MAX)

    assert route.called


async def test_update_accepts_legal_id_with_values(
    index: AsyncIndex, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.post(f"{BASE_URL}/vectors/update").mock(
        return_value=httpx.Response(200, json={})
    )
    await index.update(id="vec-1", values=VECTOR, sparse_values=SPARSE)

    body = orjson.loads(route.calls.last.request.content)
    assert body["id"] == "vec-1"
    assert body["values"] == VECTOR


async def test_update_by_filter_accepts_set_metadata_and_dry_run(
    index: AsyncIndex, respx_mock: respx.MockRouter
) -> None:
    """The legal by-filter shape: metadata only."""
    route = respx_mock.post(f"{BASE_URL}/vectors/update").mock(
        return_value=httpx.Response(200, json={"matchedRecords": 42})
    )
    result = await index.update(filter=FILTER, set_metadata={"year": 2020}, dry_run=True)

    body = orjson.loads(route.calls.last.request.content)
    assert body["filter"] == FILTER
    assert body["setMetadata"] == {"year": 2020}
    assert body["dryRun"] is True
    assert "values" not in body and "sparseValues" not in body
    assert result.matched_records == 42


async def test_delete_accepts_each_mode_alone(
    index: AsyncIndex, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.post(f"{BASE_URL}/vectors/delete").mock(
        return_value=httpx.Response(200, json={})
    )
    await index.delete(ids=["vec-1"])
    await index.delete(delete_all=True)
    await index.delete(filter=FILTER)

    bodies = [orjson.loads(call.request.content) for call in route.calls]
    assert bodies[0]["ids"] == ["vec-1"]
    assert bodies[1]["deleteAll"] is True
    assert bodies[2]["filter"] == FILTER


async def test_delete_all_false_is_not_a_selector(
    index: AsyncIndex, respx_mock: respx.MockRouter
) -> None:
    """``delete_all=False`` is the default, not a third mode, so ids stay legal alongside it."""
    route = respx_mock.post(f"{BASE_URL}/vectors/delete").mock(
        return_value=httpx.Response(200, json={})
    )
    await index.delete(ids=["vec-1"], delete_all=False)

    assert route.called


@pytest.mark.parametrize(
    "limit",
    [pytest.param(1, id="lower_bound"), pytest.param(LIST_LIMIT_MAX, id="upper_bound")],
)
async def test_list_limit_bounds_are_inclusive(
    index: AsyncIndex, limit: int, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.get(url__startswith=f"{BASE_URL}/vectors/list").mock(
        return_value=httpx.Response(200, json=_list_response())
    )
    await index.list_paginated(limit=limit)

    assert route.calls.last.request.url.params["limit"] == str(limit)


@pytest.mark.parametrize(
    "limit",
    [pytest.param(1, id="lower_bound"), pytest.param(FBM_LIMIT_MAX, id="upper_bound")],
)
async def test_fetch_by_metadata_limit_bounds_are_inclusive(
    index: AsyncIndex, limit: int, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.post(f"{BASE_URL}/vectors/fetch_by_metadata").mock(
        return_value=httpx.Response(200, json=_fetch_response())
    )
    await index.fetch_by_metadata(filter=FILTER, limit=limit)

    body = orjson.loads(route.calls.last.request.content)
    assert body["limit"] == limit


async def test_empty_list_prefix_is_accepted_and_sent(
    index: AsyncIndex, respx_mock: respx.MockRouter
) -> None:
    """``^[\\x01-\\x7F]*$`` — unlike an ID, the prefix may be empty; it matches everything."""
    route = respx_mock.get(url__startswith=f"{BASE_URL}/vectors/list").mock(
        return_value=httpx.Response(200, json=_list_response())
    )
    await index.list_paginated(prefix="")

    assert route.calls.last.request.url.params["prefix"] == ""


async def test_list_prefix_at_length_limit_is_accepted(
    index: AsyncIndex, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.get(url__startswith=f"{BASE_URL}/vectors/list").mock(
        return_value=httpx.Response(200, json=_list_response())
    )
    await index.list_paginated(prefix="a" * PREFIX_MAX)

    assert route.called


async def test_query_accepts_an_empty_filter(
    index: AsyncIndex, respx_mock: respx.MockRouter
) -> None:
    """``QueryRequest.filter`` has no ``minProperties``; tightening it would exceed the server."""
    route = respx_mock.post(f"{BASE_URL}/query").mock(
        return_value=httpx.Response(200, json=_query_response())
    )
    await index.query(top_k=1, vector=VECTOR, filter={})

    assert route.called


async def test_describe_index_stats_forwards_a_non_empty_filter(
    index: AsyncIndex, respx_mock: respx.MockRouter
) -> None:
    """Whether filtering is supported depends on the index, which the client cannot know."""
    route = respx_mock.post(f"{BASE_URL}/describe_index_stats").mock(
        return_value=httpx.Response(200, json={"namespaces": {}, "totalVectorCount": 0})
    )
    await index.describe_index_stats(filter=FILTER)

    assert orjson.loads(route.calls.last.request.content)["filter"] == FILTER


async def test_describe_index_stats_surfaces_the_server_rejection_text(
    index: AsyncIndex, respx_mock: respx.MockRouter
) -> None:
    """Serverless and Starter 4xx a non-empty filter; the body text has to reach the caller."""
    message = (
        "Serverless and Starter indexes do not support describing index stats "
        "with metadata filtering."
    )
    respx_mock.post(f"{BASE_URL}/describe_index_stats").mock(
        return_value=httpx.Response(400, json={"code": 3, "message": message, "details": []})
    )
    with pytest.raises(ApiError) as excinfo:
        await index.describe_index_stats(filter=FILTER)

    assert excinfo.value.status_code == 400
    assert message in str(excinfo.value)


async def test_describe_index_stats_accepts_an_empty_filter(
    index: AsyncIndex, respx_mock: respx.MockRouter
) -> None:
    """The backend only rejects a *non-empty* filter, so an empty one must not be blocked."""
    route = respx_mock.post(f"{BASE_URL}/describe_index_stats").mock(
        return_value=httpx.Response(200, json={"namespaces": {}, "totalVectorCount": 0})
    )
    await index.describe_index_stats(filter={})

    assert route.called


@pytest.mark.parametrize(
    ("method", "path", "response", "case_id"),
    [
        pytest.param("POST", "/query", _query_response(), "query", id="queryVectors"),
        pytest.param("GET", "/vectors/fetch", _fetch_response(), "fetch", id="fetchVectors"),
        pytest.param(
            "POST",
            "/vectors/fetch_by_metadata",
            _fetch_response(),
            "fetch_by_metadata",
            id="fetch_vectors_by_metadata",
        ),
        pytest.param("POST", "/vectors/delete", {}, "delete", id="deleteVectors"),
        pytest.param("POST", "/vectors/update", {}, "update", id="updateVector"),
        pytest.param("GET", "/vectors/list", _list_response(), "list_paginated", id="listVectors"),
        pytest.param(
            "POST",
            "/describe_index_stats",
            {"namespaces": {}, "totalVectorCount": 0},
            "describe_index_stats",
            id="describeIndexStats",
        ),
    ],
)
async def test_api_version_header_on_every_vector_op(
    index: AsyncIndex,
    method: str,
    path: str,
    response: dict[str, Any],
    case_id: str,
    respx_mock: respx.MockRouter,
) -> None:
    route = respx_mock.request(method, url__startswith=f"{BASE_URL}{path}").mock(
        return_value=httpx.Response(200, json=response)
    )
    await ACCEPTED_INVOCATIONS[case_id](index)

    assert route.calls.last.request.headers["X-Pinecone-Api-Version"] == DATA_PLANE_API_VERSION
    assert DATA_PLANE_API_VERSION == "2026-07"


_legal_chars = st.characters(min_codepoint=1, max_codepoint=127)
_legal_ids = st.text(alphabet=_legal_chars, min_size=1, max_size=ID_MAX)
_illegal_ids = st.one_of(
    st.just(""),
    st.text(alphabet=_legal_chars, min_size=ID_MAX + 1, max_size=ID_MAX + 40),
    st.text(alphabet=st.characters(min_codepoint=128), min_size=1, max_size=20),
    st.text(alphabet=_legal_chars, max_size=20).map(lambda s: s + "\x00"),
)


@given(vector_id=_legal_ids)
def test_every_legal_id_passes_fetch_validation(vector_id: str) -> None:
    async def check() -> None:
        with respx.mock(assert_all_called=False) as mock:
            route = mock.get(url__startswith=f"{BASE_URL}/vectors/fetch").mock(
                return_value=httpx.Response(200, json=_fetch_response())
            )
            client = AsyncIndex(host=INDEX_HOST, api_key="test-key")
            try:
                await client.fetch(ids=[vector_id])
            finally:
                await client.close()
            assert route.called

    asyncio.run(check())


@given(vector_id=_illegal_ids)
def test_every_illegal_id_is_rejected_before_http(vector_id: str) -> None:
    async def check() -> None:
        with respx.mock as mock:
            client = AsyncIndex(host=INDEX_HOST, api_key="test-key")
            try:
                with pytest.raises(ValidationError):
                    await client.fetch(ids=[vector_id])
                with pytest.raises(ValidationError):
                    await client.delete(ids=[vector_id])
            finally:
                await client.close()
            assert not mock.calls

    asyncio.run(check())


@given(prefix=st.text(alphabet=_legal_chars, max_size=PREFIX_MAX))
def test_every_legal_prefix_passes_list_validation(prefix: str) -> None:
    async def check() -> None:
        with respx.mock(assert_all_called=False) as mock:
            route = mock.get(url__startswith=f"{BASE_URL}/vectors/list").mock(
                return_value=httpx.Response(200, json=_list_response())
            )
            client = AsyncIndex(host=INDEX_HOST, api_key="test-key")
            try:
                await client.list_paginated(prefix=prefix)
            finally:
                await client.close()
            assert route.called

    asyncio.run(check())


@given(limit=st.integers(min_value=1, max_value=LIST_LIMIT_MAX))
def test_every_legal_list_limit_passes_validation(limit: int) -> None:
    async def check() -> None:
        with respx.mock as mock:
            mock.get(url__startswith=f"{BASE_URL}/vectors/list").mock(
                return_value=httpx.Response(200, json=_list_response())
            )
            client = AsyncIndex(host=INDEX_HOST, api_key="test-key")
            try:
                await client.list_paginated(limit=limit)
            finally:
                await client.close()

    asyncio.run(check())


@given(
    limit=st.one_of(
        st.integers(max_value=0), st.integers(min_value=LIST_LIMIT_MAX + 1, max_value=10**6)
    )
)
def test_every_illegal_list_limit_is_rejected_before_http(limit: int) -> None:
    async def check() -> None:
        with respx.mock as mock:
            client = AsyncIndex(host=INDEX_HOST, api_key="test-key")
            try:
                with pytest.raises(
                    ValidationError, match=f"limit must be between 1 and {LIST_LIMIT_MAX}"
                ):
                    await client.list_paginated(limit=limit)
            finally:
                await client.close()
            assert not mock.calls

    asyncio.run(check())


@given(has_vector=st.booleans(), has_id=st.booleans(), has_sparse=st.booleans())
def test_query_selectors_match_the_spec_truth_table(
    has_vector: bool, has_id: bool, has_sparse: bool
) -> None:
    """Randomized presence combinations must land exactly on 2026-07's anyOf/not."""
    spec_allows = (has_vector or has_id or has_sparse) and not (
        has_id and (has_vector or has_sparse)
    )

    async def check() -> None:
        with respx.mock(assert_all_called=False) as mock:
            route = mock.post(f"{BASE_URL}/query").mock(
                return_value=httpx.Response(200, json=_query_response())
            )
            client = AsyncIndex(host=INDEX_HOST, api_key="test-key")
            try:
                kwargs = query_kwargs(has_vector, has_id, has_sparse)
                if spec_allows:
                    await client.query(**kwargs)
                    assert route.called
                else:
                    with pytest.raises(ValidationError):
                        await client.query(**kwargs)
                    assert not mock.calls
            finally:
                await client.close()

    asyncio.run(check())


@given(
    values=st.one_of(st.none(), st.just(VECTOR)),
    sparse_values=st.one_of(st.none(), st.just(SPARSE)),
)
def test_update_by_filter_rejects_any_vector_payload(
    values: list[float] | None, sparse_values: dict[str, Any] | None
) -> None:
    """A by-filter update accepts metadata and nothing else, whichever value form is passed."""
    carries_vector_data = values is not None or sparse_values is not None

    async def check() -> None:
        with respx.mock(assert_all_called=False) as mock:
            route = mock.post(f"{BASE_URL}/vectors/update").mock(
                return_value=httpx.Response(200, json={"matchedRecords": 1})
            )
            client = AsyncIndex(host=INDEX_HOST, api_key="test-key")
            try:
                if carries_vector_data:
                    with pytest.raises(ValidationError, match="metadata-only"):
                        await client.update(
                            filter=FILTER, values=values, sparse_values=sparse_values
                        )
                    assert not mock.calls
                else:
                    await client.update(filter=FILTER, set_metadata={"year": 2020})
                    assert route.called
            finally:
                await client.close()

    asyncio.run(check())
