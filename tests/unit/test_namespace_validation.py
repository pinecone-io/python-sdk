"""Client-side validation and error paths for the 2026-07 namespace operations.

The rules under test are the ones 2026-07 spells out on the wire — name charset
and length, the reserved ``__default__`` name, the ``limit`` range, and the
``filterable: true`` requirement on metadata schemas. Each is rejected before any
HTTP request goes out, so a caller gets a message naming the argument and quoting
the offending value instead of a raw 400 from the server.

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
(length before charset), ``:984-1010`` (create validates name, then rejects the
default namespace, then the schema), ``pc-validation/src/error.rs:146``
(``__default__`` cannot be created), ``pc-metadata-filtering/src/metadata/compiler.rs:745``
(the ``filterable: false`` message, quoted verbatim in the client-side error),
``pc-settings/src/settings.rs:261-262`` (``max_namespace_length`` 512,
``max_list_limit`` 100).

The ``VALIDATION_CASES`` table is the parity fixture the async (#120) and gRPC
(#121) lanes bind their own callables to: identical inputs must produce identical
``ValidationError`` messages in every lane, so the expected text lives here once
rather than being restated per lane.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx
import pytest
import respx
from hypothesis import given
from hypothesis import strategies as st

from pinecone import Index
from pinecone.errors.exceptions import NotFoundError, RateLimitError, ValidationError
from tests.factories import make_namespace_description_response

INDEX_HOST = "ns-validation-abc123.svc.pinecone.io"
BASE_URL = f"https://{INDEX_HOST}"
NS_URL = f"{BASE_URL}/namespaces"

NAME_MAX = 512
LIMIT_MAX = 100


@pytest.fixture
def index() -> Any:
    client = Index(host=INDEX_HOST, api_key="test-key")
    yield client
    client.close()


# ---------------------------------------------------------------------------
# The cross-lane rule table
# ---------------------------------------------------------------------------

Invoke = Callable[[Index], object]

VALIDATION_CASES: list[tuple[str, Invoke, str]] = [
    (
        "create_empty_name",
        lambda idx: idx.create_namespace(name=""),
        "name must be a non-empty string; namespace names must be 1-512 characters, got ''",
    ),
    (
        "create_non_ascii_name",
        lambda idx: idx.create_namespace(name="naïve"),
        "name must contain only ASCII characters (code points 1-127), got 'naïve'",
    ),
    (
        "create_nul_name",
        lambda idx: idx.create_namespace(name="pinec\x00one"),
        "name must not contain the NUL character, got 'pinec\\x00one'",
    ),
    (
        "create_overlong_name",
        lambda idx: idx.create_namespace(name="a" * (NAME_MAX + 1)),
        f"name must be 1-{NAME_MAX} characters, got {NAME_MAX + 1}",
    ),
    (
        "create_reserved_default",
        lambda idx: idx.create_namespace(name="__default__"),
        "name='__default__' is reserved and cannot be created",
    ),
    (
        "create_schema_filterable_omitted",
        lambda idx: idx.create_namespace(name="ns", schema={"fields": {"genre": {}}}),
        "schema['fields']['genre']['filterable'] must be True, got omitted: "
        "Field 'genre' is set to filterable: false. Only filterable: true is supported. "
        "To avoid indexing the field, omit it from the list of fields.",
    ),
    (
        "create_schema_filterable_false",
        lambda idx: idx.create_namespace(
            name="ns", schema={"fields": {"genre": {"filterable": False}}}
        ),
        "schema['fields']['genre']['filterable'] must be True, got False: "
        "Field 'genre' is set to filterable: false. Only filterable: true is supported. "
        "To avoid indexing the field, omit it from the list of fields.",
    ),
    (
        "create_schema_filterable_null",
        lambda idx: idx.create_namespace(
            name="ns", schema={"fields": {"genre": {"filterable": None}}}
        ),
        "schema['fields']['genre']['filterable'] must be True, got None: ",
    ),
    (
        "create_schema_missing_fields",
        lambda idx: idx.create_namespace(name="ns", schema={}),
        "schema must contain a 'fields' key, got keys []",
    ),
    (
        "describe_empty_name",
        lambda idx: idx.describe_namespace(name=""),
        "name must be a non-empty string; namespace names must be 1-512 characters, got ''",
    ),
    (
        "describe_non_ascii_name",
        lambda idx: idx.describe_namespace(name="naïve"),
        "name must contain only ASCII characters (code points 1-127), got 'naïve'",
    ),
    (
        "describe_overlong_name",
        lambda idx: idx.describe_namespace(name="a" * (NAME_MAX + 1)),
        f"name must be 1-{NAME_MAX} characters, got {NAME_MAX + 1}",
    ),
    (
        "delete_empty_name",
        lambda idx: idx.delete_namespace(name=""),
        "name must be a non-empty string; namespace names must be 1-512 characters, got ''",
    ),
    (
        "delete_nul_name",
        lambda idx: idx.delete_namespace(name="a\x00b"),
        "name must not contain the NUL character, got 'a\\x00b'",
    ),
    (
        "list_limit_zero",
        lambda idx: idx.list_namespaces_paginated(limit=0),
        f"limit must be between 1 and {LIMIT_MAX}, got 0",
    ),
    (
        "list_limit_negative",
        lambda idx: idx.list_namespaces_paginated(limit=-1),
        f"limit must be between 1 and {LIMIT_MAX}, got -1",
    ),
    (
        "list_limit_over_max",
        lambda idx: idx.list_namespaces_paginated(limit=LIMIT_MAX + 1),
        f"limit must be between 1 and {LIMIT_MAX}, got {LIMIT_MAX + 1}",
    ),
    (
        "list_prefix_non_ascii",
        lambda idx: idx.list_namespaces_paginated(prefix="naïve"),
        "prefix must contain only ASCII characters (code points 1-127), got 'naïve'",
    ),
    (
        "list_prefix_overlong",
        lambda idx: idx.list_namespaces_paginated(prefix="a" * (NAME_MAX + 1)),
        f"prefix must be at most {NAME_MAX} characters, got {NAME_MAX + 1}",
    ),
    (
        "list_generator_limit_over_max",
        lambda idx: next(iter(idx.list_namespaces(limit=LIMIT_MAX + 1))),
        f"limit must be between 1 and {LIMIT_MAX}, got {LIMIT_MAX + 1}",
    ),
]


@pytest.mark.parametrize(
    ("invoke", "expected"),
    [pytest.param(fn, msg, id=case_id) for case_id, fn, msg in VALIDATION_CASES],
)
@respx.mock
def test_rejected_before_any_http_call(
    index: Index, invoke: Invoke, expected: str, respx_mock: respx.MockRouter
) -> None:
    with pytest.raises(ValidationError) as excinfo:
        invoke(index)

    assert expected in str(excinfo.value)
    assert not respx_mock.calls, "validation must reject before the request goes out"


def test_every_message_names_argument_and_value() -> None:
    """Every rejection has to be actionable without reading the SDK source."""
    for case_id, _fn, message in VALIDATION_CASES:
        first_token = message.split()[0]
        assert first_token.split("[")[0].split("=")[0] in {"name", "prefix", "limit", "schema"}, (
            f"{case_id}: message does not open by naming the argument: {message!r}"
        )


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
@respx.mock
def test_describe_accepts_legal_names(index: Index, name: str) -> None:
    route = respx.get(url__startswith=NS_URL).mock(
        return_value=httpx.Response(200, json=make_namespace_description_response(name=name))
    )
    assert index.describe_namespace(name=name).name == name
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
@respx.mock
def test_namespace_is_percent_encoded_as_one_path_segment(
    index: Index, name: str, encoded: str
) -> None:
    """Legal names are not all URL-safe, and the name is one path segment either way.

    ``^[\\x01-\\x7F]+$`` admits ``/``, ``?``, ``#``, ``%`` and the C0 controls.
    Interpolated raw, ``/`` silently addresses a different route and a control
    character makes httpx reject the URL outright, so neither reaches the server
    as the name the caller passed.
    """
    describe = respx.get(f"{NS_URL}/{encoded}").mock(
        return_value=httpx.Response(200, json=make_namespace_description_response(name=name))
    )
    delete = respx.delete(f"{NS_URL}/{encoded}").mock(return_value=httpx.Response(200, json={}))

    assert index.describe_namespace(name=name).name == name
    index.delete_namespace(name=name)

    assert describe.called and delete.called


@respx.mock
def test_create_rejects_default_but_describe_and_delete_accept_it(index: Index) -> None:
    """``__default__`` is reserved only on create; it is the way to name the default namespace."""
    with pytest.raises(ValidationError, match="reserved and cannot be created"):
        index.create_namespace(name="__default__")

    respx.get(f"{NS_URL}/__default__").mock(
        return_value=httpx.Response(
            200, json=make_namespace_description_response(name="__default__")
        )
    )
    respx.delete(f"{NS_URL}/__default__").mock(return_value=httpx.Response(200, json={}))

    assert index.describe_namespace(name="__default__").name == "__default__"
    assert index.delete_namespace(name="__default__") is None


@respx.mock
def test_limit_bounds_are_inclusive(index: Index) -> None:
    route = respx.get(NS_URL).mock(
        return_value=httpx.Response(200, json={"namespaces": [], "total_count": 0})
    )
    index.list_namespaces_paginated(limit=1)
    index.list_namespaces_paginated(limit=LIMIT_MAX)
    assert [call.request.url.params["limit"] for call in route.calls] == ["1", str(LIMIT_MAX)]


@respx.mock
def test_empty_prefix_is_accepted_and_sent(index: Index) -> None:
    """``^[\\x01-\\x7F]*$`` — unlike a name, the prefix may be empty; it matches everything."""
    route = respx.get(NS_URL).mock(
        return_value=httpx.Response(200, json={"namespaces": [], "total_count": 0})
    )
    index.list_namespaces_paginated(prefix="")
    assert route.calls.last.request.url.params["prefix"] == ""


@respx.mock
def test_schema_with_all_fields_filterable_is_sent(index: Index) -> None:
    route = respx.post(NS_URL).mock(
        return_value=httpx.Response(200, json=make_namespace_description_response(name="ns"))
    )
    index.create_namespace(
        name="ns",
        schema={"fields": {"genre": {"filterable": True}, "year": {"filterable": True}}},
    )

    import orjson

    body = orjson.loads(route.calls.last.request.content)
    assert body["schema"] == {
        "fields": {"genre": {"filterable": True}, "year": {"filterable": True}}
    }


# ---------------------------------------------------------------------------
# size_bytes reaches the caller on every operation that returns a description
# ---------------------------------------------------------------------------


@respx.mock
def test_create_namespace_exposes_size_bytes(index: Index) -> None:
    respx.post(NS_URL).mock(
        return_value=httpx.Response(
            200, json=make_namespace_description_response(name="ns", size_bytes=1048576)
        )
    )
    assert index.create_namespace(name="ns").size_bytes == 1048576


@respx.mock
def test_describe_namespace_exposes_size_bytes(index: Index) -> None:
    respx.get(f"{NS_URL}/ns").mock(
        return_value=httpx.Response(
            200, json=make_namespace_description_response(name="ns", size_bytes=2**63)
        )
    )
    assert index.describe_namespace(name="ns").size_bytes == 2**63


@respx.mock
def test_list_namespaces_exposes_size_bytes_on_every_page(index: Index) -> None:
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
    respx.get(NS_URL).mock(
        side_effect=[httpx.Response(200, json=first), httpx.Response(200, json=second)]
    )

    sizes = [ns.size_bytes for page in index.list_namespaces() for ns in page.namespaces]
    assert sizes == [10, 0, 999]


@respx.mock
def test_size_bytes_defaults_to_zero_when_server_omits_it(index: Index) -> None:
    """A pre-2026-07 server sends no ``size_bytes``; the model's default has to hold."""
    respx.get(f"{NS_URL}/ns").mock(
        return_value=httpx.Response(200, json={"name": "ns", "record_count": 7})
    )
    assert index.describe_namespace(name="ns").size_bytes == 0


# ---------------------------------------------------------------------------
# Server error paths
# ---------------------------------------------------------------------------


@respx.mock
def test_describe_namespace_404_raises_not_found(index: Index) -> None:
    respx.get(f"{NS_URL}/missing").mock(
        return_value=httpx.Response(
            404, json={"code": 5, "message": "Namespace not found", "details": []}
        )
    )
    with pytest.raises(NotFoundError) as excinfo:
        index.describe_namespace(name="missing")

    assert excinfo.value.status_code == 404


@respx.mock
def test_describe_namespace_429_raises_rate_limit_with_retry_after(index: Index) -> None:
    """describeNamespace is rate limited per index, independently of the other namespace ops."""
    respx.get(f"{NS_URL}/busy").mock(
        return_value=httpx.Response(
            429,
            headers={"Retry-After": "30"},
            json={"code": 8, "message": "Too many requests", "details": []},
        )
    )
    with pytest.raises(RateLimitError) as excinfo:
        index.describe_namespace(name="busy")

    assert excinfo.value.status_code == 429
    assert excinfo.value.retry_after == 30


@respx.mock
def test_delete_namespace_404_raises_not_found(index: Index) -> None:
    respx.delete(f"{NS_URL}/missing").mock(
        return_value=httpx.Response(
            404, json={"code": 5, "message": "Namespace not found", "details": []}
        )
    )
    with pytest.raises(NotFoundError):
        index.delete_namespace(name="missing")


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
    with respx.mock(assert_all_called=False) as mock:
        route = mock.get(url__startswith=NS_URL).mock(
            return_value=httpx.Response(200, json=make_namespace_description_response(name=name))
        )
        client = Index(host=INDEX_HOST, api_key="test-key")
        try:
            assert client.describe_namespace(name=name).name == name
        finally:
            client.close()
        assert route.called


@given(name=_illegal_names)
def test_every_illegal_name_is_rejected_before_http(name: str) -> None:
    with respx.mock as mock:
        client = Index(host=INDEX_HOST, api_key="test-key")
        try:
            with pytest.raises(ValidationError):
                client.describe_namespace(name=name)
            with pytest.raises(ValidationError):
                client.delete_namespace(name=name)
            with pytest.raises(ValidationError):
                client.create_namespace(name=name)
        finally:
            client.close()
        assert not mock.calls


@given(limit=st.integers(min_value=1, max_value=LIMIT_MAX))
def test_every_legal_limit_passes_client_validation(limit: int) -> None:
    with respx.mock as mock:
        mock.get(NS_URL).mock(
            return_value=httpx.Response(200, json={"namespaces": [], "total_count": 0})
        )
        client = Index(host=INDEX_HOST, api_key="test-key")
        try:
            client.list_namespaces_paginated(limit=limit)
        finally:
            client.close()


@given(
    limit=st.one_of(st.integers(max_value=0), st.integers(min_value=LIMIT_MAX + 1, max_value=10**6))
)
def test_every_illegal_limit_is_rejected_before_http(limit: int) -> None:
    with respx.mock as mock:
        client = Index(host=INDEX_HOST, api_key="test-key")
        try:
            with pytest.raises(ValidationError, match=f"limit must be between 1 and {LIMIT_MAX}"):
                client.list_namespaces_paginated(limit=limit)
        finally:
            client.close()
        assert not mock.calls
