"""Client-side validation and error paths for the 2026-07 namespace operations, async lane.

The async twin of ``tests/unit/test_namespace_validation.py`` (#119 ∥ #120). The
rules live once in ``pinecone/_internal/validation.py`` and the expected messages
live once in that file's ``VALIDATION_CASES`` table; here each case id is bound to
an :class:`AsyncIndex` call instead of being restated, so the two lanes cannot
drift apart in wording — a message that changes in one fails in both.

Spec basis (``apis`` @ 5f808858):

- ``db_data_2026-07.oas.yaml:975-984`` — describe path param: ``^[\\x01-\\x7F]+$``,
  1-512, ``__default__`` legal
- ``db_data_2026-07.oas.yaml:1004-1015`` — describe 404, and a 429 rate limited
  per index independently of the other namespace operations
- ``db_data_2026-07.oas.yaml:838-865`` — list ``limit`` 1-100, ``prefix``
  ``^[\\x01-\\x7F]*$`` / 512
- ``db_data_2026-07.oas.yaml:2032-2071`` — ``CreateNamespaceRequest.name``;
  ``__default__`` reserved; ``filterable`` required with ``enum: [true]``

Backend basis (``pinecone-db`` @ f6fd0a40): ``pc-validation/src/data_plane/mod.rs:66-82``
(length before charset), ``pc-validation/src/error.rs:146`` (``__default__`` cannot be
created), ``pc-metadata-filtering/src/metadata/compiler.rs:745`` (the ``filterable:
false`` message), ``pc-settings/src/settings.rs:261-262`` (512 / 100 limits).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable

import httpx
import orjson
import pytest
import respx
from hypothesis import given
from hypothesis import strategies as st

from pinecone import AsyncIndex
from pinecone.errors.exceptions import NotFoundError, RateLimitError, ValidationError
from tests.factories import make_namespace_description_response
from tests.unit.test_namespace_validation import LIMIT_MAX, NAME_MAX, VALIDATION_CASES

INDEX_HOST = "async-ns-validation-abc123.svc.pinecone.io"
BASE_URL = f"https://{INDEX_HOST}"
NS_URL = f"{BASE_URL}/namespaces"


@pytest.fixture
async def index() -> AsyncIterator[AsyncIndex]:
    client = AsyncIndex(host=INDEX_HOST, api_key="test-key")
    yield client
    await client.close()


# ---------------------------------------------------------------------------
# The async binding of the cross-lane rule table
# ---------------------------------------------------------------------------

AsyncInvoke = Callable[[AsyncIndex], Awaitable[object]]

ASYNC_INVOCATIONS: dict[str, AsyncInvoke] = {
    "create_empty_name": lambda idx: idx.create_namespace(name=""),
    "create_non_ascii_name": lambda idx: idx.create_namespace(name="naïve"),
    "create_nul_name": lambda idx: idx.create_namespace(name="pinec\x00one"),
    "create_overlong_name": lambda idx: idx.create_namespace(name="a" * (NAME_MAX + 1)),
    "create_reserved_default": lambda idx: idx.create_namespace(name="__default__"),
    "create_schema_filterable_omitted": lambda idx: idx.create_namespace(
        name="ns", schema={"fields": {"genre": {}}}
    ),
    "create_schema_filterable_false": lambda idx: idx.create_namespace(
        name="ns", schema={"fields": {"genre": {"filterable": False}}}
    ),
    "create_schema_filterable_null": lambda idx: idx.create_namespace(
        name="ns", schema={"fields": {"genre": {"filterable": None}}}
    ),
    "create_schema_missing_fields": lambda idx: idx.create_namespace(name="ns", schema={}),
    "describe_empty_name": lambda idx: idx.describe_namespace(name=""),
    "describe_non_ascii_name": lambda idx: idx.describe_namespace(name="naïve"),
    "describe_overlong_name": lambda idx: idx.describe_namespace(name="a" * (NAME_MAX + 1)),
    "delete_empty_name": lambda idx: idx.delete_namespace(name=""),
    "delete_nul_name": lambda idx: idx.delete_namespace(name="a\x00b"),
    "list_limit_zero": lambda idx: idx.list_namespaces_paginated(limit=0),
    "list_limit_negative": lambda idx: idx.list_namespaces_paginated(limit=-1),
    "list_limit_over_max": lambda idx: idx.list_namespaces_paginated(limit=LIMIT_MAX + 1),
    "list_prefix_non_ascii": lambda idx: idx.list_namespaces_paginated(prefix="naïve"),
    "list_prefix_overlong": lambda idx: idx.list_namespaces_paginated(prefix="a" * (NAME_MAX + 1)),
    "list_generator_limit_over_max": lambda idx: anext(idx.list_namespaces(limit=LIMIT_MAX + 1)),
}


def test_every_shared_case_is_bound_to_an_async_call() -> None:
    """A rule the REST lane grows has to be exercised here too, or parity is untested."""
    shared = {case_id for case_id, _fn, _msg in VALIDATION_CASES}
    assert set(ASYNC_INVOCATIONS) == shared, (
        f"unbound in async={sorted(shared - set(ASYNC_INVOCATIONS))}, "
        f"unknown to the shared table={sorted(set(ASYNC_INVOCATIONS) - shared)}"
    )


@pytest.mark.parametrize(
    ("case_id", "expected"),
    [pytest.param(case_id, msg, id=case_id) for case_id, _fn, msg in VALIDATION_CASES],
)
async def test_rejected_before_any_http_call(
    index: AsyncIndex, case_id: str, expected: str, respx_mock: respx.MockRouter
) -> None:
    with pytest.raises(ValidationError) as excinfo:
        await ASYNC_INVOCATIONS[case_id](index)

    assert expected in str(excinfo.value)
    assert not respx_mock.calls, "validation must reject before the request goes out"


# ---------------------------------------------------------------------------
# Accepted inputs the old client-side rules would have refused
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("__default__", id="reserved_default_is_describable"),
        pytest.param(" ", id="single_space"),
        pytest.param("a" * NAME_MAX, id="at_length_limit"),
        pytest.param("ns-with.dots_and~tilde", id="punctuation"),
        pytest.param("\x01", id="lowest_legal_code_point"),
        pytest.param("\x7f", id="highest_legal_code_point"),
    ],
)
async def test_describe_accepts_legal_names(
    index: AsyncIndex, name: str, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.get(url__startswith=NS_URL).mock(
        return_value=httpx.Response(200, json=make_namespace_description_response(name=name))
    )
    result = await index.describe_namespace(name=name)

    assert result.name == name
    assert route.called


@pytest.mark.parametrize(
    ("name", "encoded"),
    [
        pytest.param("a/b", "a%2Fb", id="slash_cannot_change_the_route"),
        pytest.param("a b", "a%20b", id="space"),
        pytest.param("a?b#c", "a%3Fb%23c", id="query_and_fragment_delimiters"),
        pytest.param("100%", "100%25", id="percent"),
        pytest.param("\x01", "%01", id="control_character"),
        pytest.param("\x7f", "%7F", id="delete_character"),
    ],
)
async def test_namespace_is_percent_encoded_as_one_path_segment(
    index: AsyncIndex, name: str, encoded: str, respx_mock: respx.MockRouter
) -> None:
    """Legal names are not all URL-safe, and the name is one path segment either way.

    ``^[\\x01-\\x7F]+$`` admits ``/``, ``?``, ``#``, ``%`` and the C0 controls.
    Interpolated raw, ``/`` silently addresses a different route and a control
    character makes httpx reject the URL outright, so neither reaches the server
    as the name the caller passed.
    """
    describe = respx_mock.get(f"{NS_URL}/{encoded}").mock(
        return_value=httpx.Response(200, json=make_namespace_description_response(name=name))
    )
    delete = respx_mock.delete(f"{NS_URL}/{encoded}").mock(
        return_value=httpx.Response(200, json={})
    )

    assert (await index.describe_namespace(name=name)).name == name
    await index.delete_namespace(name=name)

    assert describe.called and delete.called


async def test_create_rejects_default_but_describe_and_delete_accept_it(
    index: AsyncIndex, respx_mock: respx.MockRouter
) -> None:
    """``__default__`` is reserved only on create; it is the way to name the default namespace."""
    with pytest.raises(ValidationError, match="reserved and cannot be created"):
        await index.create_namespace(name="__default__")

    respx_mock.get(f"{NS_URL}/__default__").mock(
        return_value=httpx.Response(
            200, json=make_namespace_description_response(name="__default__")
        )
    )
    respx_mock.delete(f"{NS_URL}/__default__").mock(return_value=httpx.Response(200, json={}))

    assert (await index.describe_namespace(name="__default__")).name == "__default__"
    assert await index.delete_namespace(name="__default__") is None


async def test_limit_bounds_are_inclusive(index: AsyncIndex, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.get(NS_URL).mock(
        return_value=httpx.Response(200, json={"namespaces": [], "total_count": 0})
    )
    await index.list_namespaces_paginated(limit=1)
    await index.list_namespaces_paginated(limit=LIMIT_MAX)

    assert [call.request.url.params["limit"] for call in route.calls] == ["1", str(LIMIT_MAX)]


async def test_empty_prefix_is_accepted_and_sent(
    index: AsyncIndex, respx_mock: respx.MockRouter
) -> None:
    """``^[\\x01-\\x7F]*$`` — unlike a name, the prefix may be empty; it matches everything."""
    route = respx_mock.get(NS_URL).mock(
        return_value=httpx.Response(200, json={"namespaces": [], "total_count": 0})
    )
    await index.list_namespaces_paginated(prefix="")

    assert route.calls.last.request.url.params["prefix"] == ""


async def test_schema_with_all_fields_filterable_is_sent(
    index: AsyncIndex, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.post(NS_URL).mock(
        return_value=httpx.Response(200, json=make_namespace_description_response(name="ns"))
    )
    await index.create_namespace(
        name="ns",
        schema={"fields": {"genre": {"filterable": True}, "year": {"filterable": True}}},
    )

    body = orjson.loads(route.calls.last.request.content)
    assert body["schema"] == {
        "fields": {"genre": {"filterable": True}, "year": {"filterable": True}}
    }


# ---------------------------------------------------------------------------
# size_bytes reaches the caller on every operation that returns a description
# ---------------------------------------------------------------------------


async def test_create_namespace_exposes_size_bytes(
    index: AsyncIndex, respx_mock: respx.MockRouter
) -> None:
    respx_mock.post(NS_URL).mock(
        return_value=httpx.Response(
            200, json=make_namespace_description_response(name="ns", size_bytes=1048576)
        )
    )
    assert (await index.create_namespace(name="ns")).size_bytes == 1048576


async def test_describe_namespace_exposes_size_bytes(
    index: AsyncIndex, respx_mock: respx.MockRouter
) -> None:
    respx_mock.get(f"{NS_URL}/ns").mock(
        return_value=httpx.Response(
            200, json=make_namespace_description_response(name="ns", size_bytes=2**63)
        )
    )
    assert (await index.describe_namespace(name="ns")).size_bytes == 2**63


async def test_list_namespaces_exposes_size_bytes_on_every_page(
    index: AsyncIndex, respx_mock: respx.MockRouter
) -> None:
    first = {
        "namespaces": [
            make_namespace_description_response(name="ns-a", size_bytes=10),
            make_namespace_description_response(name="ns-b", size_bytes=0),
        ],
        "pagination": {"next": "page-2"},
        "total_count": 3,
    }
    second = {
        "namespaces": [make_namespace_description_response(name="ns-c", size_bytes=999)],
        "total_count": 3,
    }
    respx_mock.get(NS_URL).mock(
        side_effect=[httpx.Response(200, json=first), httpx.Response(200, json=second)]
    )

    sizes = [ns.size_bytes async for page in index.list_namespaces() for ns in page.namespaces]
    assert sizes == [10, 0, 999]


async def test_size_bytes_defaults_to_zero_when_server_omits_it(
    index: AsyncIndex, respx_mock: respx.MockRouter
) -> None:
    """A pre-2026-07 server sends no ``size_bytes``; the model's default has to hold."""
    respx_mock.get(f"{NS_URL}/ns").mock(
        return_value=httpx.Response(200, json={"name": "ns", "record_count": 7})
    )
    assert (await index.describe_namespace(name="ns")).size_bytes == 0


# ---------------------------------------------------------------------------
# Server error paths
# ---------------------------------------------------------------------------


async def test_describe_namespace_404_raises_not_found(
    index: AsyncIndex, respx_mock: respx.MockRouter
) -> None:
    respx_mock.get(f"{NS_URL}/missing").mock(
        return_value=httpx.Response(
            404, json={"code": 5, "message": "Namespace not found", "details": []}
        )
    )
    with pytest.raises(NotFoundError) as excinfo:
        await index.describe_namespace(name="missing")

    assert excinfo.value.status_code == 404


async def test_describe_namespace_429_raises_rate_limit_with_retry_after(
    index: AsyncIndex, respx_mock: respx.MockRouter
) -> None:
    """describeNamespace is rate limited per index, independently of the other namespace ops."""
    respx_mock.get(f"{NS_URL}/busy").mock(
        return_value=httpx.Response(
            429,
            headers={"Retry-After": "30"},
            json={"code": 8, "message": "Too many requests", "details": []},
        )
    )
    with pytest.raises(RateLimitError) as excinfo:
        await index.describe_namespace(name="busy")

    assert excinfo.value.status_code == 429
    assert excinfo.value.retry_after == 30


async def test_delete_namespace_404_raises_not_found(
    index: AsyncIndex, respx_mock: respx.MockRouter
) -> None:
    respx_mock.delete(f"{NS_URL}/missing").mock(
        return_value=httpx.Response(
            404, json={"code": 5, "message": "Namespace not found", "details": []}
        )
    )
    with pytest.raises(NotFoundError):
        await index.delete_namespace(name="missing")


# ---------------------------------------------------------------------------
# Property: the client-side name rule is exactly ASCII / no-NUL / 1-512
# ---------------------------------------------------------------------------

_legal_chars = st.characters(min_codepoint=1, max_codepoint=127)
_legal_names = st.text(alphabet=_legal_chars, min_size=1, max_size=NAME_MAX)
_illegal_names = st.one_of(
    st.just(""),
    st.text(alphabet=_legal_chars, min_size=NAME_MAX + 1, max_size=NAME_MAX + 40),
    st.text(alphabet=st.characters(min_codepoint=128), min_size=1, max_size=20),
    st.text(alphabet=_legal_chars, max_size=20).map(lambda s: s + "\x00"),
)


@given(name=_legal_names)
def test_every_legal_name_passes_client_validation(name: str) -> None:
    async def check() -> None:
        with respx.mock(assert_all_called=False) as mock:
            route = mock.get(url__startswith=NS_URL).mock(
                return_value=httpx.Response(
                    200, json=make_namespace_description_response(name=name)
                )
            )
            client = AsyncIndex(host=INDEX_HOST, api_key="test-key")
            try:
                assert (await client.describe_namespace(name=name)).name == name
            finally:
                await client.close()
            assert route.called

    asyncio.run(check())


@given(name=_illegal_names)
def test_every_illegal_name_is_rejected_before_http(name: str) -> None:
    async def check() -> None:
        with respx.mock as mock:
            client = AsyncIndex(host=INDEX_HOST, api_key="test-key")
            try:
                with pytest.raises(ValidationError):
                    await client.describe_namespace(name=name)
                with pytest.raises(ValidationError):
                    await client.delete_namespace(name=name)
                with pytest.raises(ValidationError):
                    await client.create_namespace(name=name)
            finally:
                await client.close()
            assert not mock.calls

    asyncio.run(check())


@given(limit=st.integers(min_value=1, max_value=LIMIT_MAX))
def test_every_legal_limit_passes_client_validation(limit: int) -> None:
    async def check() -> None:
        with respx.mock as mock:
            mock.get(NS_URL).mock(
                return_value=httpx.Response(200, json={"namespaces": [], "total_count": 0})
            )
            client = AsyncIndex(host=INDEX_HOST, api_key="test-key")
            try:
                await client.list_namespaces_paginated(limit=limit)
            finally:
                await client.close()

    asyncio.run(check())


@given(
    limit=st.one_of(st.integers(max_value=0), st.integers(min_value=LIMIT_MAX + 1, max_value=10**6))
)
def test_every_illegal_limit_is_rejected_before_http(limit: int) -> None:
    async def check() -> None:
        with respx.mock as mock:
            client = AsyncIndex(host=INDEX_HOST, api_key="test-key")
            try:
                with pytest.raises(
                    ValidationError, match=f"limit must be between 1 and {LIMIT_MAX}"
                ):
                    await client.list_namespaces_paginated(limit=limit)
            finally:
                await client.close()
            assert not mock.calls

    asyncio.run(check())
