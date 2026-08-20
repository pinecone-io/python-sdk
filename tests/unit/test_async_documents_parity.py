"""Sync/async parity for the graduated documents data plane (#132 ∥ #134).

``Index`` and ``AsyncIndex`` build their document requests through the same
``pinecone/_internal/documents_helpers.py`` builders and decode through the same
``DocumentsAdapter`` envelopes, so the two transports should differ only in
``await``. These tests hold them to that on the axes a transport port can quietly
break: identical request snapshots on the wire (method, path — including the
URL-encoded namespace segment — query, body, and the 2026-07 version header) for
identical arguments, identical signatures and return annotations, and identical
exception types and messages for the full client-side rejection matrix, which is
where the mutual-exclusion rules live.

Follows the pattern of ``tests/unit/test_async_inference_parity.py``.

Three divergences are asserted rather than papered over. The positional-misuse
guard names the owning class (``Index.upsert_documents`` vs
``AsyncIndex.upsert_documents``), so its message differs by design; that case is
checked for a shared suffix instead of byte-equality. ``batch_upsert_documents``
runs on a thread pool in sync and an ``asyncio.Semaphore`` in async, so only its
signature is compared — the per-request body it emits is already covered by the
single-request ``upsert_documents`` snapshot. And ``list_documents`` returns a
``Paginator`` in sync and an ``AsyncPaginator`` in async, so its return
annotation is checked for equality modulo that one type name, and its request
snapshot is taken after draining the paginator rather than from a bare call.
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
from pinecone.models.documents.document import DocumentRecord, UpdateDocumentRecord
from pinecone.models.documents.score_by import (
    DenseVectorQuery,
    QueryStringQuery,
    SparseVectorQuery,
    TextQuery,
)
from pinecone.models.vectors.sparse import SparseValues

INDEX_HOST = "parity-index-abc123.svc.us-east-1-aws.pinecone.io"
BASE_URL = f"https://{INDEX_HOST}"
NS = "articles-en"
ENCODED_NS = "live%20ns%2Fv1"

_METHODS = [
    "batch_upsert_documents",
    "delete_documents",
    "fetch_documents",
    "list_documents",
    "search_documents",
    "update_documents",
    "upsert_documents",
]

_PAGINATED_METHODS = ["list_documents"]

_CALLS: dict[str, dict[str, Any]] = {
    "upsert_documents": {
        "namespace": "live ns/v1",
        "documents": [
            {"_id": "doc-1", "title": "Rome", "year": 2026},
            DocumentRecord({"_id": "doc-2", "title": "Carthage"}),
        ],
    },
    "search_documents": {
        "namespace": "live ns/v1",
        "top_k": 7,
        "score_by": [
            TextQuery(query="punic wars", fields=["content"]),
            QueryStringQuery(query="title:(rome)"),
        ],
        "include_fields": ["title"],
        "filter": {"category": {"$eq": "history"}},
    },
    "fetch_documents": {
        "namespace": "live ns/v1",
        "filter": {"category": {"$eq": "history"}},
        "include_fields": [],
        "pagination_token": "tok-1",
    },
    "delete_documents": {
        "namespace": "live ns/v1",
        "filter": {"category": {"$eq": "history"}},
    },
    "update_documents": {
        "namespace": "live ns/v1",
        "documents": [
            {"_id": "doc-1", "title": "Rome", "year": 2026},
            UpdateDocumentRecord({"_id": "doc-2", "_remove_fields": ["content"]}),
        ],
    },
}

_PAGINATED_CALLS: dict[str, dict[str, Any]] = {
    "list_documents": {
        "namespace": "live ns/v1",
        "prefix": "doc-",
        "limit": 20,
        "pagination_token": "tok-1",
    },
}

_ERROR_CASES: list[tuple[str, dict[str, Any]]] = [
    ("upsert_documents", {"namespace": "", "documents": [{"_id": "a"}]}),
    ("upsert_documents", {"namespace": "   ", "documents": [{"_id": "a"}]}),
    ("upsert_documents", {"namespace": 7, "documents": [{"_id": "a"}]}),
    ("upsert_documents", {"namespace": NS, "documents": []}),
    ("upsert_documents", {"namespace": NS, "documents": [{"_id": f"d{i}"} for i in range(1001)]}),
    ("upsert_documents", {"namespace": NS, "documents": [{"title": "no id"}]}),
    ("upsert_documents", {"namespace": NS, "documents": [{"_id": ""}]}),
    ("upsert_documents", {"namespace": NS, "documents": [{"_id": 42}]}),
    ("upsert_documents", {"namespace": NS, "documents": [{"_id": "x" * 513}]}),
    ("upsert_documents", {"namespace": NS, "documents": [{"_id": "ünïcode"}]}),
    ("upsert_documents", {"namespace": NS, "documents": ["doc-1"]}),
    ("upsert_documents", {"namespace": NS, "documents": [{"_id": "a"}, {"_id": "a"}]}),
    ("batch_upsert_documents", {"namespace": NS, "documents": [{"_id": "a"}], "batch_size": 0}),
    ("batch_upsert_documents", {"namespace": NS, "documents": [{"_id": "a"}], "batch_size": 1001}),
    (
        "batch_upsert_documents",
        {"namespace": NS, "documents": [{"_id": "a"}], "max_concurrency": 0},
    ),
    (
        "batch_upsert_documents",
        {"namespace": NS, "documents": [{"_id": "a"}], "max_concurrency": 65},
    ),
    ("batch_upsert_documents", {"namespace": "", "documents": [{"_id": "a"}]}),
    ("batch_upsert_documents", {"namespace": NS, "documents": [{"_id": "a"}, {"_id": "a"}]}),
    ("search_documents", {"namespace": NS, "top_k": 5, "score_by": []}),
    (
        "search_documents",
        {
            "namespace": NS,
            "top_k": 5,
            "score_by": [{"type": "query_string", "query": f"q{i}"} for i in range(101)],
        },
    ),
    (
        "search_documents",
        {
            "namespace": NS,
            "top_k": 5,
            "score_by": [
                DenseVectorQuery(field="embedding", values=[0.1]),
                TextQuery(query="rome", fields=["content"]),
            ],
        },
    ),
    (
        "search_documents",
        {
            "namespace": NS,
            "top_k": 5,
            "score_by": [
                SparseVectorQuery(
                    field="sparse", sparse_values=SparseValues(indices=[1], values=[0.5])
                ),
                QueryStringQuery(query="rome"),
            ],
        },
    ),
    ("search_documents", {"namespace": NS, "top_k": 0, "score_by": [QueryStringQuery(query="r")]}),
    ("search_documents", {"namespace": NS, "top_k": -1, "score_by": [QueryStringQuery(query="r")]}),
    (
        "search_documents",
        {"namespace": NS, "top_k": 10001, "score_by": [QueryStringQuery(query="r")]},
    ),
    ("search_documents", {"namespace": NS, "top_k": 5, "score_by": [{"type": "bogus"}]}),
    ("search_documents", {"namespace": "", "top_k": 5, "score_by": [QueryStringQuery(query="r")]}),
    ("fetch_documents", {"namespace": NS}),
    ("fetch_documents", {"namespace": NS, "ids": ["a"], "filter": {"x": {"$eq": 1}}}),
    ("fetch_documents", {"namespace": NS, "filter": {}}),
    ("fetch_documents", {"namespace": NS, "ids": ["a"], "pagination_token": "tok"}),
    ("fetch_documents", {"namespace": NS, "ids": [f"d{i}" for i in range(1001)]}),
    ("fetch_documents", {"namespace": "", "ids": ["a"]}),
    ("delete_documents", {"namespace": NS}),
    ("delete_documents", {"namespace": NS, "ids": ["a"], "filter": {"x": {"$eq": 1}}}),
    ("delete_documents", {"namespace": NS, "ids": ["a"], "delete_all": True}),
    ("delete_documents", {"namespace": NS, "filter": {"x": {"$eq": 1}}, "delete_all": True}),
    (
        "delete_documents",
        {"namespace": NS, "ids": ["a"], "filter": {"x": {"$eq": 1}}, "delete_all": True},
    ),
    ("delete_documents", {"namespace": NS, "filter": {}}),
    ("delete_documents", {"namespace": NS, "ids": [f"d{i}" for i in range(1001)]}),
    ("delete_documents", {"namespace": "", "delete_all": True}),
    ("update_documents", {"namespace": NS}),
    ("update_documents", {"namespace": NS, "documents": []}),
    ("update_documents", {"namespace": NS, "documents": [{"_id": f"d{i}"} for i in range(1001)]}),
    ("update_documents", {"namespace": NS, "documents": [{"title": "no id"}]}),
    ("update_documents", {"namespace": NS, "documents": [{"_id": ""}]}),
    ("update_documents", {"namespace": NS, "documents": [{"_id": "x" * 513}]}),
    ("update_documents", {"namespace": NS, "documents": [{"_id": "ünïcode"}]}),
    ("update_documents", {"namespace": NS, "documents": ["doc-1"]}),
    ("update_documents", {"namespace": NS, "documents": [{"_id": "a"}, {"_id": "a"}]}),
    (
        "update_documents",
        {"namespace": NS, "documents": [{"_id": "a", "t": 1, "_remove_fields": ["t"]}]},
    ),
    ("update_documents", {"namespace": NS, "documents": [{"_id": "a", "_remove_fields": "t"}]}),
    (
        "update_documents",
        {"namespace": NS, "documents": [{"_id": "a"}], "filter": {"x": {"$eq": 1}}},
    ),
    ("update_documents", {"namespace": NS, "documents": [{"_id": "a"}], "set_fields": {"y": 1}}),
    ("update_documents", {"namespace": NS, "documents": [{"_id": "a"}], "remove_fields": ["y"]}),
    ("update_documents", {"namespace": NS, "set_fields": {"y": 1}}),
    ("update_documents", {"namespace": NS, "remove_fields": ["y"]}),
    ("update_documents", {"namespace": NS, "filter": {"x": {"$eq": 1}}}),
    ("update_documents", {"namespace": NS, "filter": {}, "set_fields": {"y": 1}}),
    ("update_documents", {"namespace": "", "documents": [{"_id": "a"}]}),
    ("list_documents", {"namespace": ""}),
    ("list_documents", {"namespace": 7}),
    ("list_documents", {"namespace": NS, "limit": 0}),
    ("list_documents", {"namespace": NS, "limit": 101}),
    ("list_documents", {"namespace": NS, "prefix": "x" * 513}),
    ("list_documents", {"namespace": NS, "prefix": "ünïcode"}),
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
    respx.post(f"{BASE_URL}/namespaces/{ENCODED_NS}/documents/upsert").mock(
        return_value=httpx.Response(202, json={"upserted_count": 2}),
    )
    respx.post(f"{BASE_URL}/namespaces/{ENCODED_NS}/documents/search").mock(
        return_value=httpx.Response(
            200,
            json={"matches": [], "namespace": NS, "usage": {"read_units": 1}},
        ),
    )
    respx.post(f"{BASE_URL}/namespaces/{ENCODED_NS}/documents/fetch").mock(
        return_value=httpx.Response(
            200,
            json={"documents": {}, "namespace": NS, "usage": {"read_units": 1}},
        ),
    )
    respx.post(f"{BASE_URL}/namespaces/{ENCODED_NS}/documents/delete").mock(
        return_value=httpx.Response(202, json={}),
    )
    respx.post(f"{BASE_URL}/namespaces/{ENCODED_NS}/documents/update").mock(
        return_value=httpx.Response(202, json={}),
    )
    respx.post(f"{BASE_URL}/namespaces/{ENCODED_NS}/documents/list").mock(
        return_value=httpx.Response(
            200,
            json={"documents": [{"_id": "doc-1"}], "namespace": NS, "usage": {"read_units": 1}},
        ),
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
    assert async_snapshot["raw_path"].startswith(f"/namespaces/{ENCODED_NS}/documents/")


@pytest.mark.parametrize("method_name", sorted(_PAGINATED_CALLS))
@respx.mock
async def test_paginated_request_snapshot_parity(
    method_name: str, sync_index: Index, async_index: AsyncIndex
) -> None:
    _register_routes()
    kwargs = _PAGINATED_CALLS[method_name]

    sync_items = getattr(sync_index, method_name)(**kwargs).to_list()
    sync_snapshot = _snapshot(respx.calls.last.request)

    async_items = await getattr(async_index, method_name)(**kwargs).to_list()
    async_snapshot = _snapshot(respx.calls.last.request)

    assert len(respx.calls) == 2, "each transport must have issued exactly one request"
    assert async_snapshot == sync_snapshot
    assert async_snapshot["api_version"] == "2026-07"
    assert async_snapshot["raw_path"] == f"/namespaces/{ENCODED_NS}/documents/list"
    assert [item.id for item in async_items] == [item.id for item in sync_items]


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


@pytest.mark.parametrize(
    "method_name", [name for name in _METHODS if name not in _PAGINATED_METHODS]
)
def test_return_annotation_parity(method_name: str) -> None:
    sync_return = inspect.signature(getattr(Index, method_name)).return_annotation
    async_return = inspect.signature(getattr(AsyncIndex, method_name)).return_annotation

    assert str(sync_return) == str(async_return), (
        f"{method_name}: return annotation differs (sync={sync_return}, async={async_return})"
    )


@pytest.mark.parametrize("method_name", _PAGINATED_METHODS)
def test_paginated_return_annotation_differs_only_in_the_paginator_type(method_name: str) -> None:
    sync_return = str(inspect.signature(getattr(Index, method_name)).return_annotation)
    async_return = str(inspect.signature(getattr(AsyncIndex, method_name)).return_annotation)

    assert sync_return.startswith("Paginator[")
    assert async_return.startswith("AsyncPaginator[")
    assert sync_return.removeprefix("Paginator[") == async_return.removeprefix("AsyncPaginator[")


@pytest.mark.parametrize("method_name", _METHODS)
def test_keyword_only_parity(method_name: str) -> None:
    sync_params = inspect.signature(getattr(Index, method_name)).parameters
    positional = [
        name
        for name, param in sync_params.items()
        if name != "self" and param.kind is not inspect.Parameter.KEYWORD_ONLY
    ]
    assert positional == [], f"{method_name} must be keyword-only, found {positional}"


@pytest.mark.parametrize(
    "method_name,kwargs",
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


@pytest.mark.parametrize("method_name", _METHODS)
async def test_positional_misuse_message_parity_modulo_owner(
    method_name: str, sync_index: Index, async_index: AsyncIndex
) -> None:
    sync_type, sync_message = _raised(lambda: getattr(sync_index, method_name)(NS))
    async_type, async_message = await _raised_async(lambda: getattr(async_index, method_name)(NS))

    assert async_type is sync_type
    assert sync_message.startswith(f"Index.{method_name}()")
    assert async_message.startswith(f"AsyncIndex.{method_name}()")
    suffix = "() is a keyword-only method"
    assert sync_message.split(suffix, 1)[1] == async_message.split(suffix, 1)[1]
