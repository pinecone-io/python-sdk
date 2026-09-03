"""Sync/async parity for the 2026-07 vector operations (#122 ∥ #123).

``Index`` and ``AsyncIndex`` validate through the same callables in
``pinecone/_internal/validation.py`` and decode through the same adapter, so the
two transports should differ only in ``await``. These tests hold them to that on
the axes a transport port can quietly break: identical request snapshots on the
wire (method, path, query, body, and the 2026-07 version header), identical
signatures, and byte-identical exception types and messages for the full
client-side rejection matrix.

The rejection matrix is not restated here. ``VECTOR_OP_VALIDATION_CASES`` holds a
callable per case, and because the two classes are signature-identical each one is
applied to *both* clients — the sync call directly, the async call awaited. So the
compared expression is literally the same source text in both lanes, and a lane
that grows a rule the other lacks fails here without anyone remembering to add a
case. ``list_generator_limit_over_max`` is the one case that cannot be shared, its
sync form driving the generator with ``next(iter(...))``; it gets its own test.

Follows the pattern of ``tests/unit/test_async_namespace_parity.py``, which #119 ∥
#120 set for the namespace operations.
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
from pinecone.errors.exceptions import ValidationError
from pinecone.index import Index
from tests.unit.test_vector_op_validation import (
    FILTER,
    LIST_LIMIT_MAX,
    SPARSE,
    VECTOR,
    VECTOR_OP_VALIDATION_CASES,
)

INDEX_HOST = "vec-parity-abc123.svc.us-east-1-aws.pinecone.io"
BASE_URL = f"https://{INDEX_HOST}"

_METHODS = [
    "delete",
    "describe_index_stats",
    "fetch",
    "fetch_by_metadata",
    "list",
    "list_paginated",
    "query",
    "update",
    "upsert",
]

_CALLS: dict[str, dict[str, Any]] = {
    "delete": {"ids": ["vec-1", "vec-2"], "namespace": "prod"},
    "describe_index_stats": {"filter": FILTER},
    "fetch": {"ids": ["vec-1", "vec-2"], "namespace": "prod"},
    "fetch_by_metadata": {
        "filter": FILTER,
        "namespace": "prod",
        "limit": 25,
        "pagination_token": "tok-1",
    },
    "list_paginated": {
        "prefix": "doc1#",
        "limit": 10,
        "pagination_token": "tok-1",
        "namespace": "prod",
    },
    "query": {
        "top_k": 3,
        "vector": VECTOR,
        "sparse_vector": SPARSE,
        "filter": FILTER,
        "namespace": "prod",
        "include_values": True,
        "include_metadata": True,
    },
    "update": {
        "id": "vec-1",
        "values": VECTOR,
        "sparse_values": SPARSE,
        "set_metadata": {"year": 2020},
        "namespace": "prod",
    },
    "upsert": {
        "vectors": [{"id": "vec-1", "values": VECTOR, "metadata": {"year": 2020}}],
        "namespace": "prod",
    },
}
"""One legal call per operation, exercising as many arguments as each one takes."""


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
    respx.post(f"{BASE_URL}/query").mock(
        return_value=httpx.Response(
            200, json={"matches": [], "namespace": "prod", "usage": {"readUnits": 1}}
        )
    )
    respx.get(url__startswith=f"{BASE_URL}/vectors/fetch").mock(
        return_value=httpx.Response(
            200, json={"vectors": {}, "namespace": "prod", "usage": {"readUnits": 1}}
        )
    )
    respx.post(f"{BASE_URL}/vectors/fetch_by_metadata").mock(
        return_value=httpx.Response(
            200, json={"vectors": {}, "namespace": "prod", "usage": {"readUnits": 1}}
        )
    )
    respx.post(f"{BASE_URL}/vectors/delete").mock(return_value=httpx.Response(200, json={}))
    respx.post(f"{BASE_URL}/vectors/update").mock(
        return_value=httpx.Response(200, json={"matchedRecords": 1})
    )
    respx.post(f"{BASE_URL}/vectors/upsert").mock(
        return_value=httpx.Response(200, json={"upsertedCount": 1})
    )
    respx.get(url__startswith=f"{BASE_URL}/vectors/list").mock(
        return_value=httpx.Response(
            200, json={"vectors": [{"id": "vec-1"}], "namespace": "prod", "usage": {"readUnits": 1}}
        )
    )
    respx.post(f"{BASE_URL}/describe_index_stats").mock(
        return_value=httpx.Response(200, json={"namespaces": {}, "totalVectorCount": 0})
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
async def test_list_generator_snapshot_parity(sync_index: Index, async_index: AsyncIndex) -> None:
    _register_routes()

    next(iter(sync_index.list(prefix="doc1#", limit=10, namespace="prod")))
    sync_snapshot = _snapshot(respx.calls.last.request)

    await anext(async_index.list(prefix="doc1#", limit=10, namespace="prod"))
    async_snapshot = _snapshot(respx.calls.last.request)

    assert async_snapshot == sync_snapshot
    assert async_snapshot["api_version"] == "2026-07"


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


SHARED_CASES = [
    (case_id, invoke)
    for case_id, invoke, _msg in VECTOR_OP_VALIDATION_CASES
    if case_id != "list_generator_limit_over_max"
]
"""Every rejection case whose call expression works verbatim against both classes."""


@pytest.mark.parametrize(
    ("case_id", "invoke"), SHARED_CASES, ids=[case_id for case_id, _fn in SHARED_CASES]
)
async def test_validation_error_parity(
    case_id: str,
    invoke: Callable[[Any], object],
    sync_index: Index,
    async_index: AsyncIndex,
) -> None:
    """The same call expression, both lanes, byte-identical type and message."""
    sync_type, sync_message = _raised(lambda: invoke(sync_index))
    async_type, async_message = await _raised_async(
        lambda: invoke(async_index)  # type: ignore[arg-type,return-value]
    )

    assert issubclass(sync_type, ValidationError), (
        f"{case_id}: expected a client-side rejection, got {sync_type.__name__}: {sync_message}"
    )
    assert async_type is sync_type
    assert async_message == sync_message


async def test_list_generator_validation_error_parity(
    sync_index: Index, async_index: AsyncIndex
) -> None:
    """The generators validate on first iteration, and say the same thing when they do."""
    sync_type, sync_message = _raised(lambda: next(iter(sync_index.list(limit=LIST_LIMIT_MAX + 1))))
    async_type, async_message = await _raised_async(
        lambda: anext(async_index.list(limit=LIST_LIMIT_MAX + 1))
    )

    assert issubclass(sync_type, ValidationError)
    assert async_type is sync_type
    assert async_message == sync_message
