"""A caller-supplied ``.`` or ``..`` must not collapse the request path.

``quote(value, safe="")`` — the encoding the path parameters in this client
apply — deliberately leaves ``.`` alone, because ``.`` is an RFC 3986 unreserved
character. httpx then normalizes the URL on construction, and that normalization
includes RFC 3986 ``remove_dot_segments``. So a value of ``.`` or ``..`` survives
escaping and is then collapsed away:

- ``describe_namespace(name="..")`` requested ``/`` instead of ``/namespaces/..``
- ``describe_namespace(name=".")`` requested ``/namespaces`` — the *list* route

The failure is silent and it addresses a different endpoint rather than 404ing,
so a caller could read a resource they did not name. The fix is at the boundary,
in ``_prepare_path``, wired into every forwarding method of ``HTTPClient`` and
``AsyncHTTPClient``, so it covers path parameters on every surface including the
sites that apply no encoding of their own.

Every test here asserts the actual request URL (``request.url.raw_path``) rather
than a helper's return value, so no row can pass vacuously.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from typing import Any

import httpx
import pytest
import respx
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from pinecone import AsyncIndex, Index, Pinecone, PineconeAsyncio
from pinecone._internal.config import PineconeConfig
from pinecone._internal.http_client import AsyncHTTPClient, HTTPClient, _prepare_path

INDEX_HOST = "dot-segment-385-abc123.svc.pinecone.io"
BASE_URL = f"https://{INDEX_HOST}"


PREPARE_PATH_CASES = [
    pytest.param("/namespaces/..", "/namespaces/%2E%2E", id="the_reported_defect"),
    pytest.param("/namespaces/.", "/namespaces/%2E", id="single_dot_segment"),
    pytest.param("/namespaces/ok", "/namespaces/ok", id="ordinary_value_untouched"),
    pytest.param("/namespaces/a..b", "/namespaces/a..b", id="dots_not_a_whole_segment"),
    pytest.param("/namespaces/..a", "/namespaces/..a", id="leading_dots_not_a_segment"),
    pytest.param("/namespaces/my.index", "/namespaces/my.index", id="legitimate_dotted_name"),
    pytest.param("/namespaces/...", "/namespaces/...", id="three_dots_is_ordinary"),
    pytest.param("/namespaces/%2E%2E", "/namespaces/%2E%2E", id="encoded_not_double_encoded"),
    pytest.param("/namespaces/%252E", "/namespaces/%252E", id="double_quoted_untouched"),
    pytest.param("/namespaces/%2F", "/namespaces/%2F", id="encoded_slash_untouched"),
    pytest.param("/a/../b", "/a/%2E%2E/b", id="interior_dot_segment"),
    pytest.param("/a/./b/../c", "/a/%2E/b/%2E%2E/c", id="several_dot_segments"),
    pytest.param("/..", "/%2E%2E", id="path_is_only_dotdot"),
    pytest.param("/.", "/%2E", id="path_is_only_dot"),
    pytest.param("/namespaces/./", "/namespaces/%2E/", id="trailing_slash"),
    pytest.param("", "", id="empty_path"),
    pytest.param("/", "/", id="root_path"),
    pytest.param("/indexes/naïve", "/indexes/naïve", id="unicode_passes_through"),
    pytest.param("/indexes/x?filter=.", "/indexes/x?filter=.", id="query_dot_untouched"),
    pytest.param("/indexes/./x?a=./b", "/indexes/%2E/x?a=./b", id="path_fixed_query_kept"),
    pytest.param("/indexes/x#.", "/indexes/x#.", id="fragment_dot_untouched"),
    pytest.param("/indexes/./x#./y", "/indexes/%2E/x#./y", id="path_fixed_fragment_kept"),
    pytest.param("/indexes/./x?q=1#./f", "/indexes/%2E/x?q=1#./f", id="both_delimiters"),
    pytest.param("/indexes/x?a=/./b", "/indexes/x?a=/./b", id="query_dot_segment_untouched"),
    pytest.param("/indexes/x?a=/../b", "/indexes/x?a=/../b", id="query_dotdot_segment_untouched"),
    pytest.param("/indexes/x#/../y", "/indexes/x#/../y", id="fragment_dot_segment_untouched"),
    pytest.param("/indexes/./my.index", "/indexes/%2E/my.index", id="dot_segment_beside_dotted"),
    pytest.param("https://h.example.com/x/..", "https://h.example.com/x/%2E%2E", id="absolute"),
    pytest.param("relative/..", "relative/%2E%2E", id="no_leading_slash"),
    pytest.param("..", "%2E%2E", id="bare_relative_dotdot"),
    pytest.param(".", "%2E", id="bare_relative_dot"),
    pytest.param("./x", "%2E/x", id="bare_relative_leading_dot"),
]


@pytest.mark.parametrize(("path", "expected"), PREPARE_PATH_CASES)
def test_prepare_path_encodes_only_whole_dot_segments(path: str, expected: str) -> None:
    """Shapes past what today's callers send: encoded segments, queries, fragments.

    Centralizing encoding means inheriting inputs no current call site produces,
    and those are the ones that fail silently, so the table probes them too.
    """
    assert _prepare_path(path) == expected


@pytest.mark.parametrize(("path", "expected"), PREPARE_PATH_CASES)
def test_prepare_path_is_idempotent(path: str, expected: str) -> None:
    assert _prepare_path(expected) == expected


@pytest.mark.parametrize(
    "path", [None, 12, b"/x/..", ("/x/..",)], ids=["none", "int", "bytes", "tuple"]
)
def test_prepare_path_passes_non_str_through(path: Any) -> None:
    """Only ``str`` is rewritten, pinning the contract that call sites build strings.

    An ``httpx.URL`` is the shape that cannot be protected: it normalizes on
    construction, so its dot segments are gone before the boundary sees it.
    """
    assert _prepare_path(path) is path


def test_httpx_url_normalization_is_the_mechanism() -> None:
    """Pin the httpx behaviour the fix defeats, so an upstream change is visible.

    The third assertion is why ``%2E`` works: httpx does not percent-decode
    before removing dot segments, so an encoded segment survives.
    """
    client = httpx.Client(base_url=BASE_URL)

    assert client.build_request("GET", "/namespaces/..").url.raw_path == b"/"
    assert client.build_request("GET", "/namespaces/.").url.raw_path == b"/namespaces"
    assert client.build_request("GET", "/namespaces/%2E%2E").url.raw_path == b"/namespaces/%2E%2E"


@pytest.fixture
def http_client() -> Iterator[HTTPClient]:
    client = HTTPClient(PineconeConfig(api_key="test-key", host=BASE_URL), api_version="2026-07")
    yield client
    client.close()


@pytest.fixture
async def async_http_client() -> AsyncIterator[AsyncHTTPClient]:
    client = AsyncHTTPClient(
        PineconeConfig(api_key="test-key", host=BASE_URL), api_version="2026-07"
    )
    yield client
    await client.close()


DOT_VALUES = [pytest.param(".", "%2E", id="dot"), pytest.param("..", "%2E%2E", id="dotdot")]


@pytest.mark.parametrize(("value", "encoded"), DOT_VALUES)
@pytest.mark.parametrize("verb", ["get", "post", "put", "patch", "delete"])
@respx.mock
def test_sync_client_verbs_preserve_a_dot_segment(
    http_client: HTTPClient, verb: str, value: str, encoded: str
) -> None:
    route = respx.route().mock(return_value=httpx.Response(200, json={}))

    getattr(http_client, verb)(f"/namespaces/{value}")

    assert route.calls.last.request.url.raw_path.decode() == f"/namespaces/{encoded}"


@pytest.mark.parametrize(("value", "encoded"), DOT_VALUES)
@respx.mock
def test_sync_client_stream_preserves_a_dot_segment(
    http_client: HTTPClient, value: str, encoded: str
) -> None:
    route = respx.route().mock(return_value=httpx.Response(200, json={}))

    with http_client.stream("GET", f"/namespaces/{value}"):
        pass

    assert route.calls.last.request.url.raw_path.decode() == f"/namespaces/{encoded}"


@pytest.mark.parametrize(("value", "encoded"), DOT_VALUES)
@pytest.mark.parametrize("verb", ["get", "post", "put", "patch", "delete"])
async def test_async_client_verbs_preserve_a_dot_segment(
    async_http_client: AsyncHTTPClient,
    verb: str,
    value: str,
    encoded: str,
    respx_mock: respx.MockRouter,
) -> None:
    route = respx_mock.route().mock(return_value=httpx.Response(200, json={}))

    await getattr(async_http_client, verb)(f"/namespaces/{value}")

    assert route.calls.last.request.url.raw_path.decode() == f"/namespaces/{encoded}"


@pytest.mark.parametrize(("value", "encoded"), DOT_VALUES)
async def test_async_client_stream_preserves_a_dot_segment(
    async_http_client: AsyncHTTPClient, value: str, encoded: str, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.route().mock(return_value=httpx.Response(200, json={}))

    async with async_http_client.stream("GET", f"/namespaces/{value}"):
        pass

    assert route.calls.last.request.url.raw_path.decode() == f"/namespaces/{encoded}"


@respx.mock
def test_post_fast_path_normalizes_and_caches_the_encoded_path() -> None:
    """``post`` builds and caches its own ``httpx.URL``, bypassing ``build_request``.

    Two identical calls exercise the cache-miss and cache-hit branches, so the
    URL cache cannot be keyed on the un-normalized path.
    """
    route = respx.route().mock(return_value=httpx.Response(200, json={}))
    client = HTTPClient(PineconeConfig(api_key="test-key", host=BASE_URL), api_version="2026-07")
    try:
        client.post("/namespaces/..", json={})
        client.post("/namespaces/..", json={})
    finally:
        client.close()

    assert [call.request.url.raw_path.decode() for call in route.calls] == [
        "/namespaces/%2E%2E",
        "/namespaces/%2E%2E",
    ]


Op = Callable[[Any, str], Any]
AsyncOp = Callable[[Any, str], Awaitable[Any]]

CONTROL_PLANE_OPS: list[tuple[str, Op, str]] = [
    ("index_describe", lambda pc, v: pc.describe_index(v), "/indexes/{}"),
    ("index_delete", lambda pc, v: pc.delete_index(v), "/indexes/{}"),
    ("index_configure", lambda pc, v: pc.configure_index(v, tags={"a": "b"}), "/indexes/{}"),
    ("collection_describe", lambda pc, v: pc.describe_collection(v), "/collections/{}"),
    ("collection_delete", lambda pc, v: pc.delete_collection(v), "/collections/{}"),
    ("backup_describe", lambda pc, v: pc.describe_backup(backup_id=v), "/backups/{}"),
    ("backup_delete", lambda pc, v: pc.delete_backup(backup_id=v), "/backups/{}"),
    ("restore_job_describe", lambda pc, v: pc.describe_restore_job(job_id=v), "/restore-jobs/{}"),
    (
        "index_create_backup",
        lambda pc, v: pc.create_backup(index_name=v, backup_name="b"),
        "/indexes/{}/backups",
    ),
    (
        "index_list_backups",
        lambda pc, v: list(pc.indexes.list_backups(index_name=v)),
        "/indexes/{}/backups",
    ),
    ("inference_get_model", lambda pc, v: pc.inference.get_model(model=v), "/models/{}"),
    (
        "backup_schedule_describe",
        lambda pc, v: pc.backup_schedules.describe(schedule_id=v),
        "/backup-schedules/{}",
    ),
    (
        "backup_schedule_delete",
        lambda pc, v: pc.backup_schedules.delete(schedule_id=v),
        "/backup-schedules/{}",
    ),
    (
        "backup_schedule_history",
        lambda pc, v: pc.backup_schedules.history(schedule_id=v),
        "/backup-schedules/{}/history",
    ),
    (
        "backup_schedule_list",
        lambda pc, v: pc.backup_schedules.list(index_name=v),
        "/indexes/{}/backup-schedules",
    ),
    (
        "assistant_describe",
        lambda pc, v: pc.assistants.describe(assistant_name=v),
        "/assistant/assistants/{}",
    ),
    (
        "assistant_delete",
        lambda pc, v: pc.assistants.delete(assistant_name=v),
        "/assistant/assistants/{}",
    ),
    (
        "assistant_update",
        lambda pc, v: pc.assistants.update(assistant_name=v, instructions="x"),
        "/assistant/assistants/{}",
    ),
]

DATA_PLANE_OPS: list[tuple[str, Op, str]] = [
    ("namespace_describe", lambda ix, v: ix.describe_namespace(name=v), "/namespaces/{}"),
    ("namespace_delete", lambda ix, v: ix.delete_namespace(name=v), "/namespaces/{}"),
    ("import_describe", lambda ix, v: ix.describe_import(v), "/bulk/imports/{}"),
    ("import_cancel", lambda ix, v: ix.cancel_import(v), "/bulk/imports/{}"),
    (
        "records_upsert",
        lambda ix, v: ix.upsert_records(namespace=v, records=[{"_id": "1", "text": "t"}]),
        "/records/namespaces/{}/upsert",
    ),
    (
        "records_search",
        lambda ix, v: ix.search_records(namespace=v, query={"inputs": {"text": "t"}, "top_k": 1}),
        "/records/namespaces/{}/search",
    ),
    (
        "documents_upsert",
        lambda ix, v: ix.documents.upsert(namespace=v, documents=[{"_id": "1"}]),
        "/namespaces/{}/documents/upsert",
    ),
    (
        "documents_search",
        lambda ix, v: ix.documents.search(namespace=v, score_by=["_score"], top_k=1),
        "/namespaces/{}/documents/search",
    ),
    (
        "documents_fetch",
        lambda ix, v: ix.documents.fetch(namespace=v, ids=["1"]),
        "/namespaces/{}/documents/fetch",
    ),
    (
        "documents_delete",
        lambda ix, v: ix.documents.delete(namespace=v, ids=["1"]),
        "/namespaces/{}/documents/delete",
    ),
    (
        "documents_update",
        lambda ix, v: ix.documents.update(namespace=v, documents=[{"_id": "1"}]),
        "/namespaces/{}/documents/update",
    ),
    (
        "documents_list",
        lambda ix, v: list(ix.documents.list(namespace=v)),
        "/namespaces/{}/documents/list",
    ),
]

ASYNC_CONTROL_PLANE_OPS: list[tuple[str, AsyncOp, str]] = [
    ("index_describe", lambda pc, v: pc.describe_index(v), "/indexes/{}"),
    ("index_delete", lambda pc, v: pc.delete_index(v), "/indexes/{}"),
    ("collection_describe", lambda pc, v: pc.describe_collection(v), "/collections/{}"),
    ("collection_delete", lambda pc, v: pc.delete_collection(v), "/collections/{}"),
    ("backup_describe", lambda pc, v: pc.describe_backup(backup_id=v), "/backups/{}"),
    ("backup_delete", lambda pc, v: pc.delete_backup(backup_id=v), "/backups/{}"),
    ("restore_job_describe", lambda pc, v: pc.describe_restore_job(job_id=v), "/restore-jobs/{}"),
    (
        "backup_schedule_describe",
        lambda pc, v: pc.backup_schedules.describe(schedule_id=v),
        "/backup-schedules/{}",
    ),
    (
        "assistant_describe",
        lambda pc, v: pc.assistants.describe(assistant_name=v),
        "/assistant/assistants/{}",
    ),
    ("inference_get_model", lambda pc, v: pc.inference.get_model(model=v), "/models/{}"),
]

ASYNC_DATA_PLANE_OPS: list[tuple[str, AsyncOp, str]] = [
    ("namespace_describe", lambda ix, v: ix.describe_namespace(name=v), "/namespaces/{}"),
    ("namespace_delete", lambda ix, v: ix.delete_namespace(name=v), "/namespaces/{}"),
    ("import_describe", lambda ix, v: ix.describe_import(v), "/bulk/imports/{}"),
    ("import_cancel", lambda ix, v: ix.cancel_import(v), "/bulk/imports/{}"),
    (
        "records_upsert",
        lambda ix, v: ix.upsert_records(namespace=v, records=[{"_id": "1", "text": "t"}]),
        "/records/namespaces/{}/upsert",
    ),
    (
        "documents_fetch",
        lambda ix, v: ix.documents.fetch(namespace=v, ids=["1"]),
        "/namespaces/{}/documents/fetch",
    ),
    (
        "documents_delete",
        lambda ix, v: ix.documents.delete(namespace=v, ids=["1"]),
        "/namespaces/{}/documents/delete",
    ),
]


def _params(ops: list[tuple[str, Any, str]]) -> list[Any]:
    return [pytest.param(op, template, id=name) for name, op, template in ops]


def _ignoring_the_stub_response_body() -> contextlib.AbstractContextManager[None]:
    """The stub body is not a valid payload for any operation under test.

    The claim being made is about the request line, so a deserialization failure
    on the way back out carries no information and is discarded.
    """
    return contextlib.suppress(Exception)


@pytest.mark.parametrize(("value", "encoded"), DOT_VALUES)
@pytest.mark.parametrize(("invoke", "template"), _params(CONTROL_PLANE_OPS))
@respx.mock
def test_control_plane_op_addresses_the_named_resource(
    invoke: Op, template: str, value: str, encoded: str
) -> None:
    respx.route().mock(return_value=httpx.Response(200, json={}))
    client = Pinecone(api_key="test-key")

    with _ignoring_the_stub_response_body():
        invoke(client, value)

    assert respx.calls, "no request was issued, so this row would prove nothing"
    assert respx.calls[0].request.url.raw_path.decode() == template.format(encoded)


@pytest.mark.parametrize(("value", "encoded"), DOT_VALUES)
@pytest.mark.parametrize(("invoke", "template"), _params(DATA_PLANE_OPS))
@respx.mock
def test_data_plane_op_addresses_the_named_namespace(
    invoke: Op, template: str, value: str, encoded: str
) -> None:
    respx.route().mock(return_value=httpx.Response(200, json={}))
    index = Index(host=INDEX_HOST, api_key="test-key")

    with _ignoring_the_stub_response_body():
        invoke(index, value)

    assert respx.calls, "no request was issued, so this row would prove nothing"
    assert respx.calls[0].request.url.raw_path.decode() == template.format(encoded)


@pytest.mark.parametrize(("value", "encoded"), DOT_VALUES)
@pytest.mark.parametrize(("invoke", "template"), _params(ASYNC_CONTROL_PLANE_OPS))
async def test_async_control_plane_op_addresses_the_named_resource(
    invoke: AsyncOp, template: str, value: str, encoded: str, respx_mock: respx.MockRouter
) -> None:
    respx_mock.route().mock(return_value=httpx.Response(200, json={}))
    client = PineconeAsyncio(api_key="test-key")

    with _ignoring_the_stub_response_body():
        await invoke(client, value)
    with _ignoring_the_stub_response_body():
        await client.close()

    assert respx_mock.calls, "no request was issued, so this row would prove nothing"
    assert respx_mock.calls[0].request.url.raw_path.decode() == template.format(encoded)


@pytest.mark.parametrize(("value", "encoded"), DOT_VALUES)
@pytest.mark.parametrize(("invoke", "template"), _params(ASYNC_DATA_PLANE_OPS))
async def test_async_data_plane_op_addresses_the_named_namespace(
    invoke: AsyncOp, template: str, value: str, encoded: str, respx_mock: respx.MockRouter
) -> None:
    respx_mock.route().mock(return_value=httpx.Response(200, json={}))
    index = AsyncIndex(host=INDEX_HOST, api_key="test-key")

    with _ignoring_the_stub_response_body():
        await invoke(index, value)
    with _ignoring_the_stub_response_body():
        await index.close()

    assert respx_mock.calls, "no request was issued, so this row would prove nothing"
    assert respx_mock.calls[0].request.url.raw_path.decode() == template.format(encoded)


PATH_VALUES = st.one_of(
    st.sampled_from([".", "..", "...", "a..b", "..a", "/", "%2F", "%2E", "", "a/b", "my.index"]),
    st.text(alphabet=st.characters(min_codepoint=1, max_codepoint=0x2FFF), max_size=12),
)


def _assert_not_collapsed(prefix: str, value: str, path: str) -> None:
    assert path.startswith(prefix), f"{value!r} collapsed the path to {path!r}"
    assert len(path.split("/")) >= len(prefix.rstrip("/").split("/")) + 1, (
        f"{value!r} lost a path segment: {path!r}"
    )


@settings(max_examples=200, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(value=PATH_VALUES)
@example(value="..")
@example(value=".")
@example(value="/")
@example(value="")
@example(value="ünïcødé")
def test_namespace_request_is_never_collapsed(value: str) -> None:
    """Whatever the value, the request still addresses ``/namespaces/<something>``.

    ``..`` and ``.`` are pinned as explicit examples because a cached
    counterexample is what originally exposed this, and a different seed would
    otherwise be free to stop finding it.

    The segment count is asserted alongside, so no value can reach a sibling of
    ``/namespaces/{name}`` either.
    """
    with respx.mock(assert_all_called=False) as router:
        router.route().mock(return_value=httpx.Response(200, json={}))
        index = Index(host=INDEX_HOST, api_key="test-key")
        try:
            with _ignoring_the_stub_response_body():
                index.describe_namespace(name=value)
        finally:
            index.close()
        calls = list(router.calls)

    if not calls:
        return
    path = calls[0].request.url.raw_path.decode()
    _assert_not_collapsed("/namespaces/", value, path)
    assert path.count("/") == 2, f"{value!r} injected a path segment: {path!r}"


@settings(max_examples=200, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(value=PATH_VALUES)
@example(value="..")
@example(value=".")
@example(value="/")
@example(value="")
@example(value="ünïcødé")
def test_index_request_is_never_collapsed(value: str) -> None:
    """Same invariant on a control-plane route, plus the stronger one #417 bought.

    ``/indexes/{name}`` percent-encodes its parameter, so the name occupies
    exactly one segment and no value can reach ``/indexes/{name}/backups`` or
    any other sibling route.
    """
    with respx.mock(assert_all_called=False) as router:
        router.route().mock(return_value=httpx.Response(200, json={}))
        client = Pinecone(api_key="test-key")
        with _ignoring_the_stub_response_body():
            client.describe_index(value)
        calls = list(router.calls)

    if not calls:
        return
    path = calls[0].request.url.raw_path.decode()
    _assert_not_collapsed("/indexes/", value, path)
    assert path.count("/") == 2, f"{value!r} injected a path segment: {path!r}"
