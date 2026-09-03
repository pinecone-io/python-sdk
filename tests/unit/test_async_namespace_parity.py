"""Sync/async parity for the 2026-07 namespace operations (#119 ∥ #120).

``Index`` and ``AsyncIndex`` validate through the same callables in
``pinecone/_internal/validation.py`` and decode through the same adapter, so the
two transports should differ only in ``await``. These tests hold them to that on
the axes a transport port can quietly break: identical request snapshots on the
wire (method, path — including the percent-encoded namespace segment — query,
body, and the 2026-07 version header), identical signatures, and byte-identical
exception types and messages for the full client-side rejection matrix.

Follows the pattern of ``tests/unit/test_async_documents_parity.py``.

One divergence is asserted rather than papered over: ``list_namespaces`` returns
``Iterator[ListNamespacesResponse]`` in sync and ``AsyncIterator[...]`` in async,
which is the whole point of the async lane, so its return annotation is compared
modulo that one word.
"""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from typing import Any

import httpx
import orjson
import pytest
import respx

from pinecone._internal.constants import API_VERSION_HEADER
from pinecone.async_client.async_index import AsyncIndex
from pinecone.index import Index
from tests.factories import make_namespace_description_response

INDEX_HOST = "ns-parity-abc123.svc.us-east-1-aws.pinecone.io"
BASE_URL = f"https://{INDEX_HOST}"
NS_URL = f"{BASE_URL}/namespaces"

NS = "live ns/v1"
ENCODED_NS = "live%20ns%2Fv1"

NAME_MAX = 512
LIMIT_MAX = 100

_METHODS = [
    "create_namespace",
    "delete_namespace",
    "describe_namespace",
    "list_namespaces",
    "list_namespaces_paginated",
]

_CALLS: dict[str, dict[str, Any]] = {
    "create_namespace": {"name": NS, "schema": {"fields": {"genre": {"filterable": True}}}},
    "describe_namespace": {"name": NS},
    "delete_namespace": {"name": NS},
    "list_namespaces_paginated": {"prefix": "prod-", "limit": 10, "pagination_token": "tok-1"},
}

_ERROR_CASES: list[tuple[str, dict[str, Any]]] = [
    ("create_namespace", {"name": ""}),
    ("create_namespace", {"name": "naïve"}),
    ("create_namespace", {"name": "pinec\x00one"}),
    ("create_namespace", {"name": "a" * (NAME_MAX + 1)}),
    ("create_namespace", {"name": "__default__"}),
    ("create_namespace", {"name": 123}),
    ("create_namespace", {"name": None}),
    ("create_namespace", {"name": "ns", "schema": {}}),
    ("create_namespace", {"name": "ns", "schema": []}),
    ("create_namespace", {"name": "ns", "schema": {"fields": None}}),
    ("create_namespace", {"name": "ns", "schema": {"fields": []}}),
    ("create_namespace", {"name": "ns", "schema": {"fields": {}, "extra": 1}}),
    ("create_namespace", {"name": "ns", "schema": {"fields": {"genre": {}}}}),
    ("create_namespace", {"name": "ns", "schema": {"fields": {"genre": {"filterable": False}}}}),
    ("create_namespace", {"name": "ns", "schema": {"fields": {"genre": {"filterable": None}}}}),
    ("create_namespace", {"name": "ns", "schema": {"fields": {"genre": "yes"}}}),
    ("describe_namespace", {"name": ""}),
    ("describe_namespace", {"name": "naïve"}),
    ("describe_namespace", {"name": "a\x00b"}),
    ("describe_namespace", {"name": "a" * (NAME_MAX + 1)}),
    ("describe_namespace", {"name": 123}),
    ("describe_namespace", {"name": "a", "namespace": "b"}),
    ("delete_namespace", {"name": ""}),
    ("delete_namespace", {"name": "naïve"}),
    ("delete_namespace", {"name": "a\x00b"}),
    ("delete_namespace", {"name": "a" * (NAME_MAX + 1)}),
    ("delete_namespace", {"name": 123}),
    ("delete_namespace", {"name": "a", "namespace": "b"}),
    ("list_namespaces_paginated", {"limit": 0}),
    ("list_namespaces_paginated", {"limit": -1}),
    ("list_namespaces_paginated", {"limit": LIMIT_MAX + 1}),
    ("list_namespaces_paginated", {"limit": True}),
    ("list_namespaces_paginated", {"limit": 1.5}),
    ("list_namespaces_paginated", {"prefix": "naïve"}),
    ("list_namespaces_paginated", {"prefix": "a\x00b"}),
    ("list_namespaces_paginated", {"prefix": "a" * (NAME_MAX + 1)}),
    ("list_namespaces_paginated", {"prefix": 123}),
]


@pytest.fixture
def sync_index() -> Iterator[Index]:
    client = Index(host=INDEX_HOST, api_key="parity-key")
    yield client
    client.close()


@pytest.fixture
async def async_index() -> AsyncIterator[AsyncIndex]:
    client = AsyncIndex(host=INDEX_HOST, api_key="parity-key")
    yield client
    await client.close()


def _register_routes() -> None:
    respx.post(NS_URL).mock(
        return_value=httpx.Response(200, json=make_namespace_description_response(name=NS))
    )
    respx.get(f"{NS_URL}/{ENCODED_NS}").mock(
        return_value=httpx.Response(200, json=make_namespace_description_response(name=NS))
    )
    respx.delete(f"{NS_URL}/{ENCODED_NS}").mock(return_value=httpx.Response(200, json={}))
    respx.get(NS_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "namespaces": [make_namespace_description_response(name=NS)],
                "total_count": 1,
            },
        )
    )


def _snapshot(request: httpx.Request) -> dict[str, Any]:
    return {
        "method": request.method,
        "raw_path": request.url.raw_path.decode(),
        "query": dict(request.url.params),
        "body": orjson.loads(request.content) if request.content else None,
        "api_version": request.headers[API_VERSION_HEADER],
    }


def _raised(call: Callable[[], object]) -> tuple[type[BaseException], str]:
    try:
        call()
    except Exception as exc:
        return type(exc), str(exc)
    raise AssertionError("expected the call to raise, it returned instead")


async def _raised_async(call: Callable[[], Awaitable[object]]) -> tuple[type[BaseException], str]:
    try:
        await call()
    except Exception as exc:
        return type(exc), str(exc)
    raise AssertionError("expected the call to raise, it returned instead")


@pytest.mark.parametrize("method_name", sorted(_CALLS))
@respx.mock
async def test_request_snapshot_parity(
    method_name: str, sync_index: Index, async_index: AsyncIndex
) -> None:
    _register_routes()
    kwargs = _CALLS[method_name]

    getattr(sync_index, method_name)(**kwargs)
    sync_snapshot = _snapshot(respx.calls.last.request)

    await getattr(async_index, method_name)(**kwargs)
    async_snapshot = _snapshot(respx.calls.last.request)

    assert len(respx.calls) == 2, "each transport must have issued exactly one request"
    assert async_snapshot == sync_snapshot
    assert async_snapshot["api_version"] == "2026-07"


@respx.mock
async def test_list_namespaces_generator_snapshot_parity(
    sync_index: Index, async_index: AsyncIndex
) -> None:
    _register_routes()

    next(iter(sync_index.list_namespaces(prefix="prod-", limit=10)))
    sync_snapshot = _snapshot(respx.calls.last.request)

    await anext(async_index.list_namespaces(prefix="prod-", limit=10))
    async_snapshot = _snapshot(respx.calls.last.request)

    assert async_snapshot == sync_snapshot
    assert async_snapshot["api_version"] == "2026-07"


@respx.mock
async def test_encoded_namespace_segment_parity(sync_index: Index, async_index: AsyncIndex) -> None:
    """Both lanes address the name as one percent-encoded path segment, not a raw route."""
    _register_routes()

    sync_index.describe_namespace(name=NS)
    sync_index.delete_namespace(name=NS)
    await async_index.describe_namespace(name=NS)
    await async_index.delete_namespace(name=NS)

    paths = [call.request.url.raw_path.decode() for call in respx.calls]
    assert paths == [f"/namespaces/{ENCODED_NS}"] * 4


@pytest.mark.parametrize("method_name", _METHODS)
def test_parameter_parity(method_name: str) -> None:
    sync_params = dict(inspect.signature(getattr(Index, method_name)).parameters)
    async_params = dict(inspect.signature(getattr(AsyncIndex, method_name)).parameters)

    assert set(sync_params) == set(async_params), (
        f"{method_name}: parameter names differ — "
        f"sync-only={set(sync_params) - set(async_params)}, "
        f"async-only={set(async_params) - set(sync_params)}"
    )

    for name, sync_param in sync_params.items():
        async_param = async_params[name]
        assert sync_param.kind == async_param.kind, (
            f"{method_name}.{name}: kind differs (sync={sync_param.kind}, async={async_param.kind})"
        )
        assert sync_param.default == async_param.default, (
            f"{method_name}.{name}: default differs "
            f"(sync={sync_param.default!r}, async={async_param.default!r})"
        )
        assert str(sync_param.annotation) == str(async_param.annotation), (
            f"{method_name}.{name}: annotation differs "
            f"(sync={sync_param.annotation}, async={async_param.annotation})"
        )


@pytest.mark.parametrize("method_name", _METHODS)
def test_return_annotation_parity_modulo_async_iterator(method_name: str) -> None:
    sync_return = str(inspect.signature(getattr(Index, method_name)).return_annotation)
    async_return = str(inspect.signature(getattr(AsyncIndex, method_name)).return_annotation)

    assert async_return.replace("AsyncIterator", "Iterator") == sync_return, (
        f"{method_name}: return annotation differs (sync={sync_return}, async={async_return})"
    )


@pytest.mark.parametrize("method_name", _METHODS)
def test_keyword_only_parity(method_name: str) -> None:
    for cls in (Index, AsyncIndex):
        params = inspect.signature(getattr(cls, method_name)).parameters
        positional = [
            name
            for name, param in params.items()
            if name != "self"
            and param.kind not in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.VAR_KEYWORD)
        ]
        assert positional == [], (
            f"{cls.__name__}.{method_name} must be keyword-only, found {positional}"
        )


@pytest.mark.parametrize(
    ("method_name", "kwargs"),
    _ERROR_CASES,
    ids=[f"{name}-{i}" for i, (name, _) in enumerate(_ERROR_CASES)],
)
async def test_validation_error_parity(
    method_name: str,
    kwargs: dict[str, Any],
    sync_index: Index,
    async_index: AsyncIndex,
) -> None:
    sync_type, sync_message = _raised(lambda: getattr(sync_index, method_name)(**kwargs))
    async_type, async_message = await _raised_async(
        lambda: getattr(async_index, method_name)(**kwargs)
    )

    assert async_type is sync_type
    assert async_message == sync_message


async def test_list_namespaces_generator_validation_error_parity(
    sync_index: Index, async_index: AsyncIndex
) -> None:
    """The generators validate on first iteration, and say the same thing when they do."""
    sync_type, sync_message = _raised(
        lambda: next(iter(sync_index.list_namespaces(limit=LIMIT_MAX + 1)))
    )
    async_type, async_message = await _raised_async(
        lambda: anext(async_index.list_namespaces(limit=LIMIT_MAX + 1))
    )

    assert async_type is sync_type
    assert async_message == sync_message
