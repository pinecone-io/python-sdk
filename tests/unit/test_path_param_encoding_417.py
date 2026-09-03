"""Every path parameter the SDK interpolates is percent-encoded (#417, #460).

#385 fixed the ``.``/``..`` half of this defect class at the HTTP boundary. The
boundary cannot fix the other half: by the time ``_prepare_path`` sees
``/indexes/a/b`` it can no longer tell whether ``a/b`` was one caller-supplied
segment or two structural ones. So ``/``-injection has to be encoded at the call
site, and these routes have real siblings for an injected segment to reach --
``/indexes/{name}``, ``/indexes/{name}/backups`` and
``/indexes/{name}/backup-schedules`` are all live, so ``describe(name="x/backups")``
listed backups instead of describing an index. The assistants surface had the
same shape and a caller-chosen name to reach it with: ``/chat/{name}``,
``/chat/{name}/context`` and ``/chat/{name}/chat/completions`` are all live, so
``chat(assistant_name="x/context")`` retrieved context instead of chatting. That
is not a judgement about what a name may contain; it is a failure to send the
name we were given.

Three kinds of test live here, and only the first states the invariant:

- property tests over arbitrary values, asserting the request addressed the
  endpoint the method names -- the value as exactly one segment, and decoding
  that segment returns the value. One drives the public operations; one drives
  the encoding alone, so it covers sites nobody has written yet.
- per-operation tables asserting ``request.url.raw_path``, the actual request
  line, so no row can pass on a helper's return value;
- an AST scan of the ``pinecone/`` tree, derived at run time, that fails when an
  unencoded site appears anywhere. #417 converted the DB control-plane sites and
  #460 the assistants and admin ones, so the scan now asserts the tree is clean
  rather than matching an inventory of exceptions.
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import inspect
import io
import pathlib
import re
from collections import Counter
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any
from urllib.parse import quote, unquote

import httpx
import pytest
import respx
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from pinecone import Pinecone, PineconeAsyncio
from pinecone._internal.http_client import _prepare_path
from pinecone.admin.admin import _OAUTH_URL, Admin
from tests.unit.test_import_id_path_encoding import ENCODING_CASES

# ---------------------------------------------------------------------------
# Tree-derived inventory of unencoded path-parameter sites
# ---------------------------------------------------------------------------

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[2] / "pinecone"

_ENCODER_NAMES = frozenset({"quote", "_encode_document_namespace"})

Site = tuple[str, str, str]
"""``(module path, path template with one ``{}`` per interpolation, value expression)``."""


def _called_name(call: ast.Call) -> str | None:
    func = call.func
    if isinstance(func, ast.Attribute):
        return func.attr
    return getattr(func, "id", None)


def _names_bound_to_an_encoder(tree: ast.Module) -> set[str]:
    """Locals holding an already-encoded segment, e.g. ``segment = quote(ns, safe="")``."""
    bound: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and _called_name(node.value) in _ENCODER_NAMES
        ):
            bound.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return bound


def _is_encoded(expr: ast.expr, encoder_locals: set[str]) -> bool:
    if isinstance(expr, ast.Call):
        return _called_name(expr) in _ENCODER_NAMES
    return isinstance(expr, ast.Name) and expr.id in encoder_locals


def _template(node: ast.JoinedStr) -> str:
    return "".join(
        str(part.value) if isinstance(part, ast.Constant) else "{}" for part in node.values
    )


def raw_path_param_sites(source: str, module: str) -> Counter[Site]:
    """Every interpolation into a path-shaped f-string that is not encoded.

    A path-shaped f-string is one whose first literal chunk starts with ``/``.
    Counted rather than collected into a set, so a second site carrying a
    template already on the books is still a new site.
    """
    tree = ast.parse(source)
    encoder_locals = _names_bound_to_an_encoder(tree)
    found: Counter[Site] = Counter()
    for node in ast.walk(tree):
        if not isinstance(node, ast.JoinedStr):
            continue
        first = node.values[0]
        if not (isinstance(first, ast.Constant) and str(first.value).startswith("/")):
            continue
        for part in node.values:
            if isinstance(part, ast.FormattedValue) and not _is_encoded(part.value, encoder_locals):
                found[(module, _template(node), ast.unparse(part.value))] += 1
    return found


def scan_package() -> Counter[Site]:
    found: Counter[Site] = Counter()
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        module = path.relative_to(PACKAGE_ROOT.parent).as_posix()
        found += raw_path_param_sites(path.read_text(), module)
    return found


CONVERTED_MODULES = (
    "pinecone/_client.py",
    "pinecone/admin/api_keys.py",
    "pinecone/admin/invites.py",
    "pinecone/admin/organizations.py",
    "pinecone/admin/projects.py",
    "pinecone/admin/role_bindings.py",
    "pinecone/admin/service_accounts.py",
    "pinecone/admin/users.py",
    "pinecone/async_client/assistants.py",
    "pinecone/async_client/backup_schedules.py",
    "pinecone/async_client/backups.py",
    "pinecone/async_client/collections.py",
    "pinecone/async_client/indexes.py",
    "pinecone/async_client/pinecone.py",
    "pinecone/async_client/restore_jobs.py",
    "pinecone/client/assistants.py",
    "pinecone/client/backup_schedules.py",
    "pinecone/client/backups.py",
    "pinecone/client/collections.py",
    "pinecone/client/indexes.py",
    "pinecone/client/restore_jobs.py",
)


def test_no_unencoded_path_param_site_anywhere() -> None:
    """A raw site anywhere in ``pinecone/`` fails here rather than shipping.

    #417 left an inventory of known-raw sites for #460 to work through. That
    inventory is empty, so the assertion is now the flat one: no exceptions.
    """
    found = scan_package()
    assert not found, (
        f"unencoded path-parameter site(s); encode with quote(value, safe=''): {sorted(found)}"
    )


@pytest.mark.parametrize("module", CONVERTED_MODULES)
def test_converted_module_has_no_unencoded_site_left(module: str) -> None:
    found = scan_package()
    assert not [site for site in found if site[0] == module], (
        f"{module} was converted by #417 but still has raw sites: "
        f"{sorted(s for s in found if s[0] == module)}"
    )


# ---------------------------------------------------------------------------
# Controls: the scanner must fail on the pre-fix shape, not just pass on the fix
# ---------------------------------------------------------------------------

_RAW = 'def f(self, name):\n    self._http.get(f"/indexes/{name}")\n'
_ENCODED = "def f(self, name):\n    self._http.get(f\"/indexes/{quote(name, safe='')}\")\n"
_ENCODED_LOCAL = (
    "def f(self, ns):\n    seg = _encode_document_namespace(ns)\n"
    '    self._http.get(f"/namespaces/{seg}/documents/fetch")\n'
)
_NOT_A_PATH = 'def f(self, name):\n    logger.info(f"index {name} described")\n'
_TWO_RAW_ONE_TEMPLATE = 'def f(self, a, b):\n    self._http.get(f"/files/{a}/{b}")\n'


def test_scanner_flags_the_pre_fix_shape() -> None:
    assert raw_path_param_sites(_RAW, "m.py") == Counter({("m.py", "/indexes/{}", "name"): 1})


def test_scanner_accepts_a_quoted_site() -> None:
    assert raw_path_param_sites(_ENCODED, "m.py") == Counter()


def test_scanner_accepts_a_segment_encoded_into_a_local() -> None:
    assert raw_path_param_sites(_ENCODED_LOCAL, "m.py") == Counter()


def test_scanner_ignores_an_f_string_that_is_not_a_path() -> None:
    assert raw_path_param_sites(_NOT_A_PATH, "m.py") == Counter()


def test_scanner_counts_each_interpolation_separately() -> None:
    """Two parameters in one template are two sites, so fixing one is still a failure."""
    assert raw_path_param_sites(_TWO_RAW_ONE_TEMPLATE, "m.py") == Counter(
        {("m.py", "/files/{}/{}", "a"): 1, ("m.py", "/files/{}/{}", "b"): 1}
    )


@pytest.mark.parametrize("module", CONVERTED_MODULES)
def test_scanner_would_still_catch_a_regression_in_a_converted_module(module: str) -> None:
    """Run the scanner over real source with the fix textually undone.

    The obvious control -- assert the scanner finds *something* in the tree --
    would pass today only because sites remain deferred, and would start failing
    the moment the last one is encoded. So the non-vacuity check reverts the
    encoding in real module source instead: it proves the scanner matches this
    codebase's actual shapes, and it keeps proving it after the tree is clean.
    """
    source = (PACKAGE_ROOT.parent / module).read_text()
    reverted = re.sub(r"quote\((.+?), safe=''\)", r"\1", source)
    assert reverted != source, f"{module} has no encoded site to revert"
    assert raw_path_param_sites(reverted, module), (
        f"the scanner missed every un-encoded site in {module}"
    )


# ---------------------------------------------------------------------------
# Per-operation request-line assertions
# ---------------------------------------------------------------------------

Op = Callable[[Any, str], Any]

_SCHEDULE = {"frequency": "daily", "retention_days": 7}


async def _drain(paginator: Any) -> list[Any]:
    return [item async for item in paginator]


def _consume(paginator: Any) -> Any:
    """Walk a paginator on whichever lane produced it.

    Returning the coroutine unawaited lets one op table drive both lanes: the
    async test awaits whatever the row hands back, and the sync test does not.
    """
    if hasattr(paginator, "__aiter__"):
        return _drain(paginator)
    return list(paginator)


OPS: list[tuple[str, Op, str]] = [
    ("index_describe", lambda pc, v: pc.indexes.describe(v), "/indexes/{}"),
    ("index_delete", lambda pc, v: pc.indexes.delete(v), "/indexes/{}"),
    ("index_configure", lambda pc, v: pc.indexes.configure(v, tags={"a": "b"}), "/indexes/{}"),
    (
        "index_create_backup",
        lambda pc, v: pc.indexes.create_backup(index_name=v, name="b"),
        "/indexes/{}/backups",
    ),
    (
        "index_list_backups",
        lambda pc, v: _consume(pc.indexes.list_backups(index_name=v)),
        "/indexes/{}/backups",
    ),
    ("index_describe_backup", lambda pc, v: pc.indexes.describe_backup(v), "/backups/{}"),
    ("collection_describe", lambda pc, v: pc.collections.describe(v), "/collections/{}"),
    ("collection_delete", lambda pc, v: pc.collections.delete(v), "/collections/{}"),
    (
        "backup_create",
        lambda pc, v: pc.backups.create(index_name=v, name="b"),
        "/indexes/{}/backups",
    ),
    ("backup_list", lambda pc, v: pc.backups.list(index_name=v), "/indexes/{}/backups"),
    ("backup_describe", lambda pc, v: pc.backups.describe(backup_id=v), "/backups/{}"),
    ("backup_delete", lambda pc, v: pc.backups.delete(backup_id=v), "/backups/{}"),
    (
        "schedule_create",
        lambda pc, v: pc.backup_schedules.create(index_name=v, name="s", **_SCHEDULE),
        "/indexes/{}/backup-schedules",
    ),
    (
        "schedule_list",
        lambda pc, v: pc.backup_schedules.list(index_name=v),
        "/indexes/{}/backup-schedules",
    ),
    (
        "schedule_iter",
        lambda pc, v: _consume(pc.backup_schedules.iter_schedules(index_name=v)),
        "/indexes/{}/backup-schedules",
    ),
    (
        "schedule_describe",
        lambda pc, v: pc.backup_schedules.describe(schedule_id=v),
        "/backup-schedules/{}",
    ),
    (
        "schedule_update",
        lambda pc, v: pc.backup_schedules.update(schedule_id=v, enabled=False),
        "/backup-schedules/{}",
    ),
    (
        "schedule_delete",
        lambda pc, v: pc.backup_schedules.delete(schedule_id=v),
        "/backup-schedules/{}",
    ),
    (
        "schedule_history",
        lambda pc, v: pc.backup_schedules.history(schedule_id=v),
        "/backup-schedules/{}/history",
    ),
    (
        "schedule_iter_history",
        lambda pc, v: _consume(pc.backup_schedules.iter_history(schedule_id=v)),
        "/backup-schedules/{}/history",
    ),
    ("restore_job_describe", lambda pc, v: pc.restore_jobs.describe(job_id=v), "/restore-jobs/{}"),
    (
        "create_index_from_backup",
        lambda pc, v: pc.create_index_from_backup(name="i", backup_id=v, timeout=-1),
        "/backups/{}/create-index",
    ),
]


def _params(ops: list[tuple[str, Any, str]]) -> list[Any]:
    return [pytest.param(op, template, id=name) for name, op, template in ops]


def _ignoring_the_stub_response_body() -> contextlib.AbstractContextManager[None]:
    """The stub body is not a valid payload for any operation under test.

    The claim is about the request line, so a deserialization failure on the way
    back out carries no information and is discarded.
    """
    return contextlib.suppress(Exception)


@pytest.fixture
def client() -> Pinecone:
    return Pinecone(api_key="test-key")


@pytest.fixture
async def async_client() -> AsyncIterator[PineconeAsyncio]:
    pc = PineconeAsyncio(api_key="test-key")
    yield pc
    await pc.close()


@pytest.mark.parametrize(("value", "encoded"), ENCODING_CASES)
@pytest.mark.parametrize(("invoke", "template"), _params(OPS))
@respx.mock
def test_op_sends_the_value_as_one_path_segment(
    client: Pinecone, invoke: Op, template: str, value: str, encoded: str
) -> None:
    respx.route().mock(return_value=httpx.Response(200, json={}))

    with _ignoring_the_stub_response_body():
        invoke(client, value)

    assert respx.calls, "no request was issued, so this row would prove nothing"
    assert respx.calls[0].request.url.raw_path.decode() == template.format(encoded)


@pytest.mark.parametrize(("value", "encoded"), ENCODING_CASES)
@pytest.mark.parametrize(("invoke", "template"), _params(OPS))
async def test_async_op_sends_the_value_as_one_path_segment(
    async_client: PineconeAsyncio,
    invoke: Op,
    template: str,
    value: str,
    encoded: str,
    respx_mock: respx.MockRouter,
) -> None:
    respx_mock.route().mock(return_value=httpx.Response(200, json={}))

    with _ignoring_the_stub_response_body():
        await invoke(async_client, value)

    assert respx_mock.calls, "no request was issued, so this row would prove nothing"
    assert respx_mock.calls[0].request.url.raw_path.decode() == template.format(encoded)


# ---------------------------------------------------------------------------
# Probing shapes today's callers do not send
# ---------------------------------------------------------------------------


@respx.mock
def test_a_str_subclass_is_encoded_as_its_str_data(client: Pinecone) -> None:
    """``str`` subclasses reach path parameters here -- #444 removed the coercion
    that used to sit in front of one, on the grounds that the boundary covers it.

    ``quote`` encodes the instance's ``str`` data, so a ``(str, Enum)`` member's
    value is what lands on the wire, not its ``repr`` and not its member name.
    """

    class Name(str):
        pass

    respx.route().mock(return_value=httpx.Response(200, json={}))

    with _ignoring_the_stub_response_body():
        client.indexes.describe(Name("a/b"))

    assert respx.calls[0].request.url.raw_path.decode() == "/indexes/a%2Fb"


@respx.mock
def test_backups_list_coerces_a_non_str_index_name(client: Pinecone) -> None:
    """The one site pair that had no ``str`` guard keeps accepting a non-``str``.

    Every other converted site already raised on a non-``str``, because
    ``require_non_empty`` calls ``.strip()`` on it. These two never did, so bare
    ``quote`` would have turned a call that worked into a ``TypeError`` -- a new
    client-side refusal, which the minimal-validation decision forbids
    regardless of what the annotation says. Coercing sends what the caller
    asked for and leaves the verdict to the server.

    Deleting the ``str()`` is what this test exists to catch.
    """
    respx.route().mock(return_value=httpx.Response(200, json={}))

    with _ignoring_the_stub_response_body():
        client.backups.list(index_name=1234)  # type: ignore[arg-type]

    assert respx.calls[0].request.url.raw_path.decode() == "/indexes/1234/backups"


async def test_async_backups_list_coerces_a_non_str_index_name(
    async_client: PineconeAsyncio, respx_mock: respx.MockRouter
) -> None:
    """The async mirror; the two lanes must refuse and accept the same inputs."""
    respx_mock.route().mock(return_value=httpx.Response(200, json={}))

    with _ignoring_the_stub_response_body():
        await async_client.backups.list(index_name=1234)  # type: ignore[arg-type]

    assert respx_mock.calls[0].request.url.raw_path.decode() == "/indexes/1234/backups"


@respx.mock
def test_backups_list_without_an_index_name_is_unchanged(client: Pinecone) -> None:
    """The unscoped listing takes the other branch, so it gains no segment."""
    respx.route().mock(return_value=httpx.Response(200, json={}))

    with _ignoring_the_stub_response_body():
        client.backups.list()

    assert respx.calls[0].request.url.raw_path.decode() == "/backups"


@pytest.mark.parametrize("value", ["", "   "])
def test_an_empty_name_is_still_rejected_before_encoding(client: Pinecone, value: str) -> None:
    """Encoding is added ahead of the existing guard, not in place of it."""
    with pytest.raises(Exception, match="non-empty"):
        client.indexes.describe(value)


PATH_VALUES = st.one_of(
    st.sampled_from(
        [
            ".",
            "..",
            "...",
            "/",
            "//",
            "a/b",
            "/leading",
            "trailing/",
            "%2F",
            "%2E",
            "%",
            "%%",
            "?",
            "#",
            "?a=1",
            "#frag",
            "a?b#c/d",
            "a b",
            "\x01",
            "\x7f",
            "\t",
            "ünïcødé",
            "日本語",
            "🔥",
            "x" * 512,
            "a" * 80 + "/" + "b" * 80,
        ]
    ),
    st.text(min_size=1, max_size=24),
    st.text(alphabet=st.characters(min_codepoint=1, max_codepoint=0x2FFF), min_size=1, max_size=16),
)

_NASTY = (
    example(value=".."),
    example(value="."),
    example(value="/"),
    example(value="a/b"),
    example(value="%2F"),
    example(value="%"),
    example(value="?"),
    example(value="#"),
    example(value=""),
    example(value="   "),
    example(value="ünïcødé"),
    example(value="🔥"),
)


def _apply(decorators: tuple[Any, ...], fn: Any) -> Any:
    for decorator in reversed(decorators):
        fn = decorator(fn)
    return fn


def assert_addresses_the_named_endpoint(template: str, value: str, path: str) -> None:
    """The request addressed *template* with *value* filling its one parameter.

    Structural first, then a round-trip: the value must occupy exactly one
    segment (so no route with a different segment count can be reached), and
    percent-decoding that segment must give back the caller's value. The
    round-trip is what a mere "is it escaped" check cannot express -- it is
    satisfied only by encoding that is both sufficient and lossless.
    """
    prefix, suffix = template.split("{}")
    assert path.startswith(prefix), f"{value!r} left the route prefix: {path!r}"
    assert path.endswith(suffix), f"{value!r} left the route suffix: {path!r}"
    segment = path[len(prefix) : len(path) - len(suffix)]
    assert "/" not in segment, f"{value!r} became {len(segment.split('/'))} segments: {path!r}"
    assert unquote(segment) == value, f"{value!r} arrived as {unquote(segment)!r}: {path!r}"


PROPERTY_OPS = [
    op
    for op in OPS
    if op[0]
    in {
        "index_describe",
        "index_create_backup",
        "backup_describe",
        "schedule_history",
        "create_index_from_backup",
        "collection_delete",
    }
]


@pytest.fixture(scope="module")
def shared_client() -> Pinecone:
    """Built once per module: rebuilding it per Hypothesis example is #422."""
    return Pinecone(api_key="test-key")


@pytest.fixture(scope="module")
def shared_async_client() -> Iterator[tuple[asyncio.AbstractEventLoop, PineconeAsyncio]]:
    loop = asyncio.new_event_loop()
    pc = PineconeAsyncio(api_key="test-key")
    yield loop, pc
    loop.run_until_complete(pc.close())
    loop.run_until_complete(loop.shutdown_asyncgens())
    loop.close()


def _drive(invoke: Any, target: Any, value: str) -> list[Any]:
    with respx.mock(assert_all_called=False) as router:
        router.route().mock(return_value=httpx.Response(200, json={}))
        with _ignoring_the_stub_response_body():
            invoke(target, value)
        return [call.request.url.raw_path.decode() for call in router.calls]


def _drive_async(
    loop: asyncio.AbstractEventLoop, invoke: Any, target: Any, value: str
) -> list[Any]:
    async def main() -> None:
        with _ignoring_the_stub_response_body():
            await invoke(target, value)

    with respx.mock(assert_all_called=False) as router:
        router.route().mock(return_value=httpx.Response(200, json={}))
        loop.run_until_complete(main())
        return [call.request.url.raw_path.decode() for call in router.calls]


def _check(template: str, value: str, paths: list[Any]) -> None:
    """Both branches are asserted, so no example can pass by issuing nothing.

    An empty or whitespace-only name is refused before any request is built;
    every other value must reach the wire, and reach it intact.
    """
    if not value.strip():
        assert not paths, f"{value!r} is rejected before the request, yet one went out: {paths}"
        return
    assert paths, f"{value!r} is a legal value but produced no request"
    assert_addresses_the_named_endpoint(template, value, paths[0])


@pytest.mark.parametrize(("invoke", "template"), _params(PROPERTY_OPS))
@_apply(
    _NASTY,
    lambda fn: settings(
        max_examples=200, suppress_health_check=[HealthCheck.function_scoped_fixture]
    )(given(value=PATH_VALUES)(fn)),
)
def test_any_value_addresses_the_named_endpoint(
    shared_client: Pinecone, invoke: Op, template: str, value: str
) -> None:
    _check(template, value, _drive(invoke, shared_client, value))


@pytest.mark.parametrize(("invoke", "template"), _params(PROPERTY_OPS))
@_apply(
    _NASTY,
    lambda fn: settings(
        max_examples=200, suppress_health_check=[HealthCheck.function_scoped_fixture]
    )(given(value=PATH_VALUES)(fn)),
)
def test_any_value_addresses_the_named_endpoint_async(
    shared_async_client: tuple[asyncio.AbstractEventLoop, PineconeAsyncio],
    invoke: Op,
    template: str,
    value: str,
) -> None:
    loop, pc = shared_async_client
    _check(template, value, _drive_async(loop, invoke, pc, value))


@pytest.fixture(scope="module")
def url_builder() -> Iterator[httpx.Client]:
    """One client for the whole module; ``build_request`` opens no socket.

    Constructing it costs ~3ms because it builds an ``ssl.SSLContext`` (#421),
    against ~0.015ms for the request it is used to build. Per example that ratio
    is 200x and it leaks an unclosed client per example, which is how the first
    version of this test became the slowest in the suite and then went flaky on
    CI. Hoisting it out of ``@given`` is #422's fix.
    """
    client = httpx.Client(base_url="https://h.example.com")
    yield client
    client.close()


@settings(max_examples=1000)
@_apply(_NASTY, given(value=PATH_VALUES))
def test_the_encoding_itself_survives_any_value(url_builder: httpx.Client, value: str) -> None:
    """The contract every site relies on, asserted without going through a site.

    This is the property that covers the sites nobody has written yet: paired
    with the AST guard above -- which proves every converted site applies this
    exact encoding -- it says the invariant holds for the whole population
    rather than for the operations the tables below happen to enumerate.

    ``quote`` alone is not enough and that is the point of #385: it leaves ``.``
    unescaped because ``.`` is unreserved, and httpx then applies RFC 3986
    ``remove_dot_segments`` on the way out. So the composition under test is the
    call site's ``quote`` followed by the boundary's ``_prepare_path``, checked
    against a real ``httpx.Request`` so httpx's own normalization is included
    rather than assumed away.
    """
    request = url_builder.build_request(
        "GET", _prepare_path(f"/indexes/{quote(value, safe='')}/backups")
    )
    assert_addresses_the_named_endpoint("/indexes/{}/backups", value, request.url.raw_path.decode())


@settings(max_examples=500)
@_apply(_NASTY, given(value=PATH_VALUES))
def test_the_encoding_is_stable_under_reapplication(value: str) -> None:
    """``_prepare_path`` is idempotent over already-encoded segments.

    Every forwarding method on the client applies it, so a path that passes
    through two of them must not be encoded twice -- that would deliver ``%252E``
    where the caller asked for ``.``.
    """
    once = _prepare_path(f"/indexes/{quote(value, safe='')}/backups")
    assert _prepare_path(once) == once


# ---------------------------------------------------------------------------
# The assistants and admin surfaces (#460)
# ---------------------------------------------------------------------------

_MESSAGES = [{"content": "hi"}]
_FIXED_ASSISTANT = "an-assistant"
_FIXED_FILE = "file-abc123"
_FIXED_OPERATION = "op-abc123"


async def _await_then_drain(awaitable: Any) -> list[Any]:
    stream = await awaitable
    return [chunk async for chunk in stream]


def _consume_stream(result: Any) -> Any:
    """Drain a streaming chat on whichever lane produced it.

    The async lane returns a coroutine that resolves to the stream, so unlike
    :func:`_consume` this cannot dispatch on ``__aiter__`` -- there is nothing
    to inspect until the coroutine has been awaited.
    """
    if inspect.isawaitable(result):
        return _await_then_drain(result)
    return list(result)


def _seeded(pc: Any, assistant_name: str) -> Any:
    """Return ``pc.assistants`` with a data-plane client already cached for *name*.

    ``_data_plane_http`` otherwise describes the assistant first to learn its
    host, which would put a second request on the wire ahead of the one under
    test and build a fresh TLS context per Hypothesis example (#421). The
    control-plane client stands in for the cached one: it carries the same
    ``/assistant`` path prefix a real data-plane client does, so the request
    line has the shape the operation actually produces.
    """
    assistants = pc.assistants
    assistants._data_plane_clients.clear()
    assistants._data_plane_clients[assistant_name] = assistants._http
    return assistants


ASSISTANT_OPS: list[tuple[str, Op, str]] = [
    (
        "assistant_describe",
        lambda pc, v: pc.assistants.describe(name=v),
        "/assistant/assistants/{}",
    ),
    (
        "assistant_update",
        lambda pc, v: pc.assistants.update(name=v, instructions="be brief"),
        "/assistant/assistants/{}",
    ),
    (
        "assistant_delete",
        lambda pc, v: pc.assistants.delete(name=v, timeout=-1),
        "/assistant/assistants/{}",
    ),
    (
        "assistant_chat",
        lambda pc, v: _seeded(pc, v).chat(assistant_name=v, messages=_MESSAGES),
        "/assistant/chat/{}",
    ),
    (
        "assistant_chat_streaming",
        lambda pc, v: _consume_stream(
            _seeded(pc, v).chat(assistant_name=v, messages=_MESSAGES, stream=True)
        ),
        "/assistant/chat/{}",
    ),
    (
        "assistant_chat_completions",
        lambda pc, v: _seeded(pc, v).chat_completions(assistant_name=v, messages=_MESSAGES),
        "/assistant/chat/{}/chat/completions",
    ),
    (
        "assistant_chat_completions_streaming",
        lambda pc, v: _consume_stream(
            _seeded(pc, v).chat_completions(assistant_name=v, messages=_MESSAGES, stream=True)
        ),
        "/assistant/chat/{}/chat/completions",
    ),
    (
        "assistant_context",
        lambda pc, v: _seeded(pc, v).context(assistant_name=v, query="q"),
        "/assistant/chat/{}/context",
    ),
    (
        "assistant_list_files_page",
        lambda pc, v: _seeded(pc, v).list_files_page(assistant_name=v),
        "/assistant/files/{}",
    ),
    (
        "assistant_upload_file",
        lambda pc, v: _seeded(pc, v).upload_file(
            assistant_name=v, file_stream=io.BytesIO(b"x"), file_name="a.txt", timeout=-1
        ),
        "/assistant/files/{}",
    ),
    (
        "assistant_upsert_file_varying_the_assistant",
        lambda pc, v: _seeded(pc, v).upload_file(
            assistant_name=v,
            file_stream=io.BytesIO(b"x"),
            file_name="a.txt",
            file_id=_FIXED_FILE,
            timeout=-1,
        ),
        f"/assistant/files/{{}}/{_FIXED_FILE}",
    ),
    (
        "assistant_upsert_file_varying_the_file",
        lambda pc, v: _seeded(pc, _FIXED_ASSISTANT).upload_file(
            assistant_name=_FIXED_ASSISTANT,
            file_stream=io.BytesIO(b"x"),
            file_name="a.txt",
            file_id=v,
            timeout=-1,
        ),
        f"/assistant/files/{_FIXED_ASSISTANT}/{{}}",
    ),
    (
        "assistant_delete_file_varying_the_assistant",
        lambda pc, v: _seeded(pc, v).delete_file(assistant_name=v, file_id=_FIXED_FILE, timeout=-1),
        f"/assistant/files/{{}}/{_FIXED_FILE}",
    ),
    (
        "assistant_describe_file_varying_the_assistant",
        lambda pc, v: _seeded(pc, v).describe_file(assistant_name=v, file_id=_FIXED_FILE),
        f"/assistant/files/{{}}/{_FIXED_FILE}",
    ),
    (
        "assistant_describe_file_varying_the_file",
        lambda pc, v: _seeded(pc, _FIXED_ASSISTANT).describe_file(
            assistant_name=_FIXED_ASSISTANT, file_id=v
        ),
        f"/assistant/files/{_FIXED_ASSISTANT}/{{}}",
    ),
    (
        "assistant_delete_file_varying_the_file",
        lambda pc, v: _seeded(pc, _FIXED_ASSISTANT).delete_file(
            assistant_name=_FIXED_ASSISTANT, file_id=v, timeout=-1
        ),
        f"/assistant/files/{_FIXED_ASSISTANT}/{{}}",
    ),
    (
        "assistant_list_operations_page",
        lambda pc, v: _seeded(pc, v).list_operations_page(assistant_name=v),
        "/assistant/operations/{}",
    ),
    (
        "assistant_describe_operation_varying_the_assistant",
        lambda pc, v: _seeded(pc, v).describe_operation(
            assistant_name=v, operation_id=_FIXED_OPERATION
        ),
        f"/assistant/operations/{{}}/{_FIXED_OPERATION}",
    ),
    (
        "assistant_describe_operation_varying_the_operation",
        lambda pc, v: _seeded(pc, _FIXED_ASSISTANT).describe_operation(
            assistant_name=_FIXED_ASSISTANT, operation_id=v
        ),
        f"/assistant/operations/{_FIXED_ASSISTANT}/{{}}",
    ),
]


@pytest.mark.parametrize(("value", "encoded"), ENCODING_CASES)
@pytest.mark.parametrize(("invoke", "template"), _params(ASSISTANT_OPS))
@respx.mock
def test_assistant_op_sends_the_value_as_one_path_segment(
    client: Pinecone, invoke: Op, template: str, value: str, encoded: str
) -> None:
    respx.route().mock(return_value=httpx.Response(200, json={}))

    with _ignoring_the_stub_response_body():
        invoke(client, value)

    assert respx.calls, "no request was issued, so this row would prove nothing"
    assert respx.calls[0].request.url.raw_path.decode() == template.format(encoded)


@pytest.mark.parametrize(("value", "encoded"), ENCODING_CASES)
@pytest.mark.parametrize(("invoke", "template"), _params(ASSISTANT_OPS))
async def test_async_assistant_op_sends_the_value_as_one_path_segment(
    async_client: PineconeAsyncio,
    invoke: Op,
    template: str,
    value: str,
    encoded: str,
    respx_mock: respx.MockRouter,
) -> None:
    respx_mock.route().mock(return_value=httpx.Response(200, json={}))

    with _ignoring_the_stub_response_body():
        await invoke(async_client, value)

    assert respx_mock.calls, "no request was issued, so this row would prove nothing"
    assert respx_mock.calls[0].request.url.raw_path.decode() == template.format(encoded)


ADMIN_OPS: list[tuple[str, Op, str]] = [
    (
        "api_key_list",
        lambda ad, v: ad.api_keys.list(project_id=v),
        "/admin/projects/{}/api-keys",
    ),
    (
        "api_key_create",
        lambda ad, v: ad.api_keys.create(project_id=v, name="k"),
        "/admin/projects/{}/api-keys",
    ),
    ("api_key_describe", lambda ad, v: ad.api_keys.describe(api_key_id=v), "/admin/api-keys/{}"),
    (
        "api_key_update",
        lambda ad, v: ad.api_keys.update(api_key_id=v, name="k"),
        "/admin/api-keys/{}",
    ),
    ("api_key_delete", lambda ad, v: ad.api_keys.delete(api_key_id=v), "/admin/api-keys/{}"),
    ("invite_describe", lambda ad, v: ad.invites.describe(invite_id=v), "/admin/invites/{}"),
    ("invite_delete", lambda ad, v: ad.invites.delete(invite_id=v), "/admin/invites/{}"),
    ("invite_resend", lambda ad, v: ad.invites.resend(invite_id=v), "/admin/invites/{}/resend"),
    (
        "organization_describe",
        lambda ad, v: ad.organizations.describe(organization_id=v),
        "/admin/organizations/{}",
    ),
    (
        "organization_update",
        lambda ad, v: ad.organizations.update(organization_id=v, name="o"),
        "/admin/organizations/{}",
    ),
    (
        "organization_delete",
        lambda ad, v: ad.organizations.delete(organization_id=v),
        "/admin/organizations/{}",
    ),
    ("project_describe", lambda ad, v: ad.projects.describe(project_id=v), "/admin/projects/{}"),
    (
        "project_update",
        lambda ad, v: ad.projects.update(project_id=v, name="p"),
        "/admin/projects/{}",
    ),
    ("project_delete", lambda ad, v: ad.projects.delete(project_id=v), "/admin/projects/{}"),
    (
        "role_binding_describe",
        lambda ad, v: ad.role_bindings.describe(role_binding_id=v),
        "/admin/role-bindings/{}",
    ),
    (
        "role_binding_delete",
        lambda ad, v: ad.role_bindings.delete(role_binding_id=v),
        "/admin/role-bindings/{}",
    ),
    (
        "service_account_describe",
        lambda ad, v: ad.service_accounts.describe(service_account_id=v),
        "/admin/service-accounts/{}",
    ),
    (
        "service_account_update",
        lambda ad, v: ad.service_accounts.update(service_account_id=v, name="s"),
        "/admin/service-accounts/{}",
    ),
    (
        "service_account_delete",
        lambda ad, v: ad.service_accounts.delete(service_account_id=v),
        "/admin/service-accounts/{}",
    ),
    (
        "service_account_rotate_secret",
        lambda ad, v: ad.service_accounts.rotate_secret(service_account_id=v),
        "/admin/service-accounts/{}/rotate-secret",
    ),
    ("user_describe", lambda ad, v: ad.users.describe(user_id=v), "/admin/users/{}"),
    ("user_delete", lambda ad, v: ad.users.delete(user_id=v), "/admin/users/{}"),
]


def _build_admin() -> Admin:
    """Mint the Bearer token against a stub so construction issues no live call.

    The exchange happens in ``Admin.__init__`` and has to be routed separately
    from the operation under test: a catch-all returning the stub body would
    hand the exchange a payload it cannot parse.
    """
    with respx.mock(assert_all_called=False) as router:
        router.post(_OAUTH_URL).mock(
            return_value=httpx.Response(
                200, json={"access_token": "t", "token_type": "Bearer", "expires_in": 1800}
            )
        )
        return Admin(client_id="i", client_secret="s")


@pytest.fixture
def admin() -> Iterator[Admin]:
    client = _build_admin()
    yield client
    client.close()


@pytest.mark.parametrize(("value", "encoded"), ENCODING_CASES)
@pytest.mark.parametrize(("invoke", "template"), _params(ADMIN_OPS))
@respx.mock
def test_admin_op_sends_the_value_as_one_path_segment(
    admin: Admin, invoke: Op, template: str, value: str, encoded: str
) -> None:
    respx.route().mock(return_value=httpx.Response(200, json={}))

    with _ignoring_the_stub_response_body():
        invoke(admin, value)

    assert respx.calls, "no request was issued, so this row would prove nothing"
    assert respx.calls[0].request.url.raw_path.decode() == template.format(encoded)


@pytest.mark.parametrize("value", ["", "   "])
def test_an_empty_admin_id_is_still_rejected_before_encoding(admin: Admin, value: str) -> None:
    """Encoding is added ahead of the existing guard, not in place of it."""
    with pytest.raises(Exception, match="non-empty"):
        admin.projects.describe(project_id=value)


@respx.mock
def test_an_assistant_name_is_coerced_rather_than_refused(client: Pinecone) -> None:
    """A non-``str`` still reaches the wire as its ``str`` data.

    No assistants path parameter has ever had a ``str`` guard in front of it --
    the f-string coerced whatever it was handed, so a ``UUID`` file id or an
    ``int`` name worked. Bare ``quote`` would turn those calls into a
    ``TypeError``: a new client-side refusal, which the minimal-validation
    decision forbids regardless of what the annotation says.

    Deleting the ``str()`` is what this test exists to catch.
    """
    respx.route().mock(return_value=httpx.Response(200, json={}))

    with _ignoring_the_stub_response_body():
        client.assistants.describe(name=1234)  # type: ignore[arg-type]

    assert respx.calls[0].request.url.raw_path.decode() == "/assistant/assistants/1234"


async def test_an_async_assistant_name_is_coerced_rather_than_refused(
    async_client: PineconeAsyncio, respx_mock: respx.MockRouter
) -> None:
    """The async mirror; the two lanes must refuse and accept the same inputs."""
    respx_mock.route().mock(return_value=httpx.Response(200, json={}))

    with _ignoring_the_stub_response_body():
        await async_client.assistants.describe(name=1234)  # type: ignore[arg-type]

    assert respx_mock.calls[0].request.url.raw_path.decode() == "/assistant/assistants/1234"


_READY_ASSISTANT = {
    "name": "n",
    "status": "Ready",
    "created_at": "2025-01-15T12:00:00Z",
    "updated_at": "2025-01-15T12:00:00Z",
    "metadata": {},
    "instructions": None,
    "host": "n-abc123.svc.pinecone.io",
    "region": "us",
}


@respx.mock
def test_create_polls_the_named_assistant() -> None:
    """``create`` builds its own path for the readiness poll.

    The poll is a second site with the same template as ``describe``, reached
    only when ``create`` returns without ``timeout=-1``, so a row in the table
    above cannot cover it -- the request under test is the second one.
    """
    respx.route().mock(return_value=httpx.Response(200, json=_READY_ASSISTANT))
    client = Pinecone(api_key="test-key")

    client.assistants.create(name="a/b")

    assert respx.calls[1].request.url.raw_path.decode() == "/assistant/assistants/a%2Fb"


async def test_async_create_polls_the_named_assistant(respx_mock: respx.MockRouter) -> None:
    """The async mirror."""
    respx_mock.route().mock(return_value=httpx.Response(200, json=_READY_ASSISTANT))
    pc = PineconeAsyncio(api_key="test-key")

    await pc.assistants.create(name="a/b")
    await pc.close()

    assert respx_mock.calls[1].request.url.raw_path.decode() == "/assistant/assistants/a%2Fb"


def _named(ops: list[tuple[str, Op, str]], names: set[str]) -> list[tuple[str, Op, str]]:
    return [op for op in ops if op[0] in names]


ASSISTANT_PROPERTY_OPS = _named(
    ASSISTANT_OPS,
    {
        "assistant_describe",
        "assistant_chat",
        "assistant_context",
        "assistant_describe_file_varying_the_assistant",
        "assistant_describe_file_varying_the_file",
    },
)

ADMIN_PROPERTY_OPS = _named(
    ADMIN_OPS,
    {"api_key_list", "project_describe", "service_account_rotate_secret", "invite_resend"},
)


def _check_unguarded(template: str, value: str, paths: list[Any]) -> None:
    """The assistants variant of :func:`_check`: nothing is refused client-side.

    No assistants path parameter is validated before the request is built, so
    every value -- the empty string included -- has to reach the wire and
    address the named endpoint. There is no second branch to assert.
    """
    assert paths, f"{value!r} produced no request"
    assert_addresses_the_named_endpoint(template, value, paths[0])


@pytest.mark.parametrize(("invoke", "template"), _params(ASSISTANT_PROPERTY_OPS))
@_apply(
    _NASTY,
    lambda fn: settings(
        max_examples=200, suppress_health_check=[HealthCheck.function_scoped_fixture]
    )(given(value=PATH_VALUES)(fn)),
)
def test_any_value_addresses_the_named_assistant_endpoint(
    shared_client: Pinecone, invoke: Op, template: str, value: str
) -> None:
    _check_unguarded(template, value, _drive(invoke, shared_client, value))


@pytest.mark.parametrize(("invoke", "template"), _params(ASSISTANT_PROPERTY_OPS))
@_apply(
    _NASTY,
    lambda fn: settings(
        max_examples=200, suppress_health_check=[HealthCheck.function_scoped_fixture]
    )(given(value=PATH_VALUES)(fn)),
)
def test_any_value_addresses_the_named_assistant_endpoint_async(
    shared_async_client: tuple[asyncio.AbstractEventLoop, PineconeAsyncio],
    invoke: Op,
    template: str,
    value: str,
) -> None:
    loop, pc = shared_async_client
    _check_unguarded(template, value, _drive_async(loop, invoke, pc, value))


@pytest.fixture(scope="module")
def shared_admin() -> Iterator[Admin]:
    """Built once per module: rebuilding it per Hypothesis example is #422."""
    client = _build_admin()
    yield client
    client.close()


@pytest.mark.parametrize(("invoke", "template"), _params(ADMIN_PROPERTY_OPS))
@_apply(
    _NASTY,
    lambda fn: settings(
        max_examples=200, suppress_health_check=[HealthCheck.function_scoped_fixture]
    )(given(value=PATH_VALUES)(fn)),
)
def test_any_value_addresses_the_named_admin_endpoint(
    shared_admin: Admin, invoke: Op, template: str, value: str
) -> None:
    _check(template, value, _drive(invoke, shared_admin, value))
