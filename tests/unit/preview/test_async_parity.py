"""Async parity tests: verify AsyncPreviewDocuments and AsyncPreviewIndexes mirror their
sync counterparts exactly — same parameter names, kinds, defaults, and annotations —
and that validation logic, error messages, and batch_upsert return shapes are identical.
"""

from __future__ import annotations

import inspect
from collections.abc import Coroutine
from typing import Any, get_args, get_origin
from unittest.mock import AsyncMock, MagicMock

import pytest

from pinecone._internal.config import PineconeConfig
from pinecone.async_client.indexes import AsyncIndexes
from pinecone.client.indexes import Indexes
from pinecone.errors.exceptions import PineconeError, ValidationError
from pinecone.models.batch import BatchResult
from pinecone.models.pagination import AsyncPaginator, Paginator
from pinecone.preview.async_documents import AsyncPreviewDocuments
from pinecone.preview.async_index import AsyncPreviewIndex
from pinecone.preview.async_indexes import AsyncPreviewIndexes
from pinecone.preview.documents import PreviewDocuments
from pinecone.preview.index import PreviewIndex
from pinecone.preview.indexes import PreviewIndexes

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strip_coroutine(annotation: Any) -> Any:
    """Return T if annotation is Coroutine[Any, Any, T], else the annotation unchanged."""
    if get_origin(annotation) is Coroutine:
        args = get_args(annotation)
        if args:
            return args[2]
    return annotation


def _normalize_paginator(annotation: Any) -> Any:
    """Convert AsyncPaginator[T] → Paginator[T] for comparison purposes.

    Handles both string annotations (from ``from __future__ import annotations``)
    and resolved generic aliases.
    """
    if isinstance(annotation, str) and annotation.startswith("AsyncPaginator["):
        return "Paginator[" + annotation[len("AsyncPaginator[") :]
    if get_origin(annotation) is AsyncPaginator:
        inner = get_args(annotation)
        return Paginator[inner[0]] if inner else Paginator  # type: ignore[valid-type]
    return annotation


def _param_pairs(
    sync_cls: type, async_cls: type, method_name: str
) -> tuple[dict[str, inspect.Parameter], dict[str, inspect.Parameter]]:
    sync_sig = inspect.signature(getattr(sync_cls, method_name))
    async_sig = inspect.signature(getattr(async_cls, method_name))
    return dict(sync_sig.parameters), dict(async_sig.parameters)


def _assert_signature_parity(sync_cls: type, async_cls: type, method_name: str) -> None:
    sync_params, async_params = _param_pairs(sync_cls, async_cls, method_name)

    assert set(sync_params) == set(async_params), (
        f"{method_name}: parameter names differ — "
        f"sync-only={set(sync_params) - set(async_params)}, "
        f"async-only={set(async_params) - set(sync_params)}"
    )

    for name in sync_params:
        sp = sync_params[name]
        ap = async_params[name]

        assert sp.kind == ap.kind, (
            f"{method_name}.{name}: kind differs (sync={sp.kind}, async={ap.kind})"
        )
        assert sp.default == ap.default, (
            f"{method_name}.{name}: default differs (sync={sp.default!r}, async={ap.default!r})"
        )

        s_ann = sp.annotation if sp.annotation is not inspect.Parameter.empty else None
        a_ann = ap.annotation if ap.annotation is not inspect.Parameter.empty else None
        if s_ann is not None and a_ann is not None:
            assert s_ann == a_ann, (
                f"{method_name}.{name}: annotation differs (sync={s_ann}, async={a_ann})"
            )


def _assert_return_parity(sync_cls: type, async_cls: type, method_name: str) -> None:
    sync_sig = inspect.signature(getattr(sync_cls, method_name))
    async_sig = inspect.signature(getattr(async_cls, method_name))

    s_ret = sync_sig.return_annotation
    a_ret = async_sig.return_annotation
    if s_ret is inspect.Parameter.empty or a_ret is inspect.Parameter.empty:
        return

    a_ret_unwrapped = _normalize_paginator(_strip_coroutine(a_ret))
    s_ret_normalized = _normalize_paginator(s_ret)
    assert s_ret_normalized == a_ret_unwrapped, (
        f"{method_name}: return annotation differs "
        f"(sync={s_ret}, async unwrapped={a_ret_unwrapped})"
    )


# ---------------------------------------------------------------------------
# PreviewDocuments vs AsyncPreviewDocuments — signature parity
# ---------------------------------------------------------------------------

_DOCUMENTS_METHODS = ["upsert", "batch_upsert", "search", "fetch", "delete"]


@pytest.mark.parametrize("method_name", _DOCUMENTS_METHODS)
def test_documents_parameter_parity(method_name: str) -> None:
    _assert_signature_parity(PreviewDocuments, AsyncPreviewDocuments, method_name)


@pytest.mark.parametrize("method_name", _DOCUMENTS_METHODS)
def test_documents_return_annotation_parity(method_name: str) -> None:
    _assert_return_parity(PreviewDocuments, AsyncPreviewDocuments, method_name)


# ---------------------------------------------------------------------------
# PreviewIndexes vs AsyncPreviewIndexes — signature parity
# ---------------------------------------------------------------------------

_INDEXES_METHODS = [
    "create",
    "configure",
    "describe",
    "list",
    "exists",
    "delete",
    "create_backup",
    "list_backups",
]


@pytest.mark.parametrize("method_name", _INDEXES_METHODS)
def test_indexes_parameter_parity(method_name: str) -> None:
    _assert_signature_parity(PreviewIndexes, AsyncPreviewIndexes, method_name)


@pytest.mark.parametrize("method_name", _INDEXES_METHODS)
def test_indexes_return_annotation_parity(method_name: str) -> None:
    _assert_return_parity(PreviewIndexes, AsyncPreviewIndexes, method_name)


# ---------------------------------------------------------------------------
# Graduated Indexes vs AsyncIndexes (#133) — signature, return, and error
# parity for the whole 2026-07 control-plane surface, including the
# index-scoped backup methods (#114 explicitly left their async twins to
# #133; the sync copies shipped with #113).
# ---------------------------------------------------------------------------

_GRADUATED_INDEXES_METHODS = [
    "create",
    "create_for_model",
    "configure",
    "describe",
    "list",
    "exists",
    "delete",
    "create_backup",
    "list_backups",
    "describe_backup",
]


@pytest.mark.parametrize("method_name", _GRADUATED_INDEXES_METHODS)
def test_graduated_indexes_parameter_parity(method_name: str) -> None:
    _assert_signature_parity(Indexes, AsyncIndexes, method_name)


@pytest.mark.parametrize("method_name", _GRADUATED_INDEXES_METHODS)
def test_graduated_indexes_return_annotation_parity(method_name: str) -> None:
    _assert_return_parity(Indexes, AsyncIndexes, method_name)


def _make_sync_indexes() -> Indexes:
    return Indexes(http=MagicMock())


def _make_async_indexes() -> AsyncIndexes:
    return AsyncIndexes(http=AsyncMock())


def _sync_indexes_error(fn: Any, *args: Any, **kwargs: Any) -> str:
    with pytest.raises(PineconeError) as exc_info:
        fn(*args, **kwargs)
    return str(exc_info.value)


async def _async_indexes_error(fn: Any, *args: Any, **kwargs: Any) -> str:
    with pytest.raises(PineconeError) as exc_info:
        await fn(*args, **kwargs)
    return str(exc_info.value)


_GRADUATED_ERROR_CASES: dict[str, tuple[str, tuple[Any, ...], dict[str, Any]]] = {
    "create-missing-schema": ("create", (), {"name": "x"}),
    "create-empty-schema": ("create", (), {"schema": {}}),
    "create-legacy-spec": (
        "create",
        (),
        {"name": "movies", "dimension": 1536, "spec": {"serverless": {}}},
    ),
    "create-source-backup-id": (
        "create",
        (),
        {"schema": {"fields": {"e": {"type": "dense_vector"}}}, "source_backup_id": "bkp-1"},
    ),
    "create-invalid-deletion-protection": (
        "create",
        (),
        {"schema": {"fields": {"e": {"type": "dense_vector"}}}, "deletion_protection": "on"},
    ),
    "configure-no-params": ("configure", ("test-index",), {}),
    "configure-legacy-replicas": ("configure", ("test-index",), {"replicas": 4}),
    "configure-legacy-embed": ("configure", ("test-index",), {"embed": {"model": "m"}}),
    "configure-legacy-serverless-read-capacity": (
        "configure",
        ("test-index",),
        {"serverless_read_capacity": {"mode": "OnDemand"}},
    ),
    "configure-deployment-type": (
        "configure",
        ("test-index",),
        {"deployment": {"deployment_type": "pod"}},
    ),
    "describe-empty-name": ("describe", ("",), {}),
    "exists-empty-name": ("exists", ("",), {}),
    "delete-empty-name": ("delete", ("",), {}),
    "create-for-model-missing-model": (
        "create_for_model",
        (),
        {"name": "x", "cloud": "aws", "region": "us-east-1", "embed": {"field_map": {"text": "t"}}},
    ),
    "create-backup-empty-index-name": ("create_backup", ("",), {}),
    "list-backups-empty-index-name": ("list_backups", ("",), {}),
    "list-backups-zero-limit": ("list_backups", ("my-index",), {"limit": 0}),
    "describe-backup-empty-id": ("describe_backup", ("",), {}),
}


@pytest.mark.parametrize("case_name", sorted(_GRADUATED_ERROR_CASES))
async def test_graduated_indexes_error_message_parity(case_name: str) -> None:
    """Invalid input raises byte-identical messages on the sync and async lanes."""
    method_name, args, kwargs = _GRADUATED_ERROR_CASES[case_name]
    sync_indexes = _make_sync_indexes()
    async_indexes = _make_async_indexes()

    sync_msg = _sync_indexes_error(getattr(sync_indexes, method_name), *args, **kwargs)
    async_msg = await _async_indexes_error(getattr(async_indexes, method_name), *args, **kwargs)

    assert sync_msg == async_msg


# ---------------------------------------------------------------------------
# Validation parity — error messages must match between sync and async
# ---------------------------------------------------------------------------


def _make_config() -> PineconeConfig:
    return PineconeConfig(api_key="test-key", host="https://api.test.pinecone.io")


def _make_sync_docs() -> PreviewDocuments:
    config = _make_config()
    return PreviewDocuments(config=config, host="https://host.test")


def _make_async_docs() -> AsyncPreviewDocuments:
    config = _make_config()
    return AsyncPreviewDocuments(config=config, host="https://host.test")


def _sync_error(fn: Any, *args: Any, **kwargs: Any) -> str:
    with pytest.raises(ValidationError) as exc_info:
        fn(*args, **kwargs)
    return str(exc_info.value)


async def _async_error(fn: Any, *args: Any, **kwargs: Any) -> str:
    with pytest.raises(ValidationError) as exc_info:
        await fn(*args, **kwargs)
    return str(exc_info.value)


# fetch validation


async def test_fetch_empty_namespace_parity() -> None:
    sync = _make_sync_docs()
    async_d = _make_async_docs()
    sync_msg = _sync_error(sync.fetch, namespace="")
    async_msg = await _async_error(async_d.fetch, namespace="")
    assert sync_msg == async_msg


# search validation


async def test_search_empty_namespace_parity() -> None:
    sync = _make_sync_docs()
    async_d = _make_async_docs()
    sync_msg = _sync_error(sync.search, namespace="", top_k=10, score_by=[{"bm25": {}}])
    async_msg = await _async_error(async_d.search, namespace="", top_k=10, score_by=[{"bm25": {}}])
    assert sync_msg == async_msg


async def test_search_top_k_range_parity() -> None:
    sync = _make_sync_docs()
    async_d = _make_async_docs()
    sync_msg = _sync_error(sync.search, namespace="ns", top_k=0, score_by=[{"bm25": {}}])
    async_msg = await _async_error(async_d.search, namespace="ns", top_k=0, score_by=[{"bm25": {}}])
    assert sync_msg == async_msg


async def test_search_empty_score_by_parity() -> None:
    sync = _make_sync_docs()
    async_d = _make_async_docs()
    sync_msg = _sync_error(sync.search, namespace="ns", top_k=5, score_by=[])
    async_msg = await _async_error(async_d.search, namespace="ns", top_k=5, score_by=[])
    assert sync_msg == async_msg


# delete validation


async def test_delete_empty_namespace_parity() -> None:
    sync = _make_sync_docs()
    async_d = _make_async_docs()
    sync_msg = _sync_error(sync.delete, namespace="", ids=["a"])
    async_msg = await _async_error(async_d.delete, namespace="", ids=["a"])
    assert sync_msg == async_msg


async def test_delete_no_target_parity() -> None:
    sync = _make_sync_docs()
    async_d = _make_async_docs()
    sync_msg = _sync_error(sync.delete, namespace="ns")
    async_msg = await _async_error(async_d.delete, namespace="ns")
    assert sync_msg == async_msg


async def test_delete_ids_and_delete_all_parity() -> None:
    sync = _make_sync_docs()
    async_d = _make_async_docs()
    sync_msg = _sync_error(sync.delete, namespace="ns", ids=["a"], delete_all=True)
    async_msg = await _async_error(async_d.delete, namespace="ns", ids=["a"], delete_all=True)
    assert sync_msg == async_msg


# upsert validation


async def test_upsert_empty_namespace_parity() -> None:
    sync = _make_sync_docs()
    async_d = _make_async_docs()
    docs = [{"_id": "doc1", "text": "hello"}]
    sync_msg = _sync_error(sync.upsert, namespace="", documents=docs)
    async_msg = await _async_error(async_d.upsert, namespace="", documents=docs)
    assert sync_msg == async_msg


async def test_upsert_missing_id_parity() -> None:
    sync = _make_sync_docs()
    async_d = _make_async_docs()
    docs = [{"text": "hello"}]
    sync_msg = _sync_error(sync.upsert, namespace="ns", documents=docs)
    async_msg = await _async_error(async_d.upsert, namespace="ns", documents=docs)
    assert sync_msg == async_msg


async def test_upsert_duplicate_id_parity() -> None:
    sync = _make_sync_docs()
    async_d = _make_async_docs()
    docs = [{"_id": "dup", "text": "a"}, {"_id": "dup", "text": "b"}]
    sync_msg = _sync_error(sync.upsert, namespace="ns", documents=docs)
    async_msg = await _async_error(async_d.upsert, namespace="ns", documents=docs)
    assert sync_msg == async_msg


# batch_upsert validation


async def test_batch_upsert_batch_size_range_parity() -> None:
    sync = _make_sync_docs()
    async_d = _make_async_docs()
    docs = [{"_id": "d1"}]
    sync_msg = _sync_error(sync.batch_upsert, namespace="ns", documents=docs, batch_size=0)
    async_msg = await _async_error(
        async_d.batch_upsert, namespace="ns", documents=docs, batch_size=0
    )
    assert sync_msg == async_msg


async def test_batch_upsert_max_concurrency_range_parity() -> None:
    sync = _make_sync_docs()
    async_d = _make_async_docs()
    docs = [{"_id": "d1"}]
    sync_msg = _sync_error(sync.batch_upsert, namespace="ns", documents=docs, max_concurrency=0)
    async_msg = await _async_error(
        async_d.batch_upsert, namespace="ns", documents=docs, max_concurrency=0
    )
    assert sync_msg == async_msg


async def test_batch_upsert_max_workers_alias_parity() -> None:
    sync = _make_sync_docs()
    async_d = _make_async_docs()
    docs = [{"_id": "d1"}]
    sync_msg = _sync_error(sync.batch_upsert, namespace="ns", documents=docs, max_workers=0)
    async_msg = await _async_error(
        async_d.batch_upsert, namespace="ns", documents=docs, max_workers=0
    )
    assert sync_msg == async_msg, (
        f"Error messages for max_workers=0 via alias differ: sync={sync_msg!r}, async={async_msg!r}"
    )


# ---------------------------------------------------------------------------
# batch_upsert returns BatchResult with the same populated fields
# ---------------------------------------------------------------------------

_DOCS_FIXTURE = [{"_id": f"doc{i}", "text": f"text {i}"} for i in range(3)]

_UPSERT_RESPONSE = b'{"upserted_count": 3}'


def test_sync_batch_upsert_returns_batch_result() -> None:
    mock_response = MagicMock()
    mock_response.content = _UPSERT_RESPONSE

    docs_obj = _make_sync_docs()
    docs_obj._http = MagicMock()
    docs_obj._http.post.return_value = mock_response

    result = docs_obj.batch_upsert(
        namespace="ns",
        documents=_DOCS_FIXTURE,
        batch_size=10,
        max_concurrency=1,
        show_progress=False,
    )

    assert isinstance(result, BatchResult)
    assert result.total_item_count == 3
    assert result.successful_item_count == 3
    assert result.failed_item_count == 0
    assert result.total_batch_count == 1
    assert result.successful_batch_count == 1
    assert result.failed_batch_count == 0
    assert result.errors == []


async def test_async_batch_upsert_returns_batch_result() -> None:
    mock_response = MagicMock()
    mock_response.content = _UPSERT_RESPONSE

    docs_obj = _make_async_docs()
    docs_obj._http = AsyncMock()
    docs_obj._http.post.return_value = mock_response

    result = await docs_obj.batch_upsert(
        namespace="ns",
        documents=_DOCS_FIXTURE,
        batch_size=10,
        max_concurrency=1,
        show_progress=False,
    )

    assert isinstance(result, BatchResult)
    assert result.total_item_count == 3
    assert result.successful_item_count == 3
    assert result.failed_item_count == 0
    assert result.total_batch_count == 1
    assert result.successful_batch_count == 1
    assert result.failed_batch_count == 0
    assert result.errors == []


async def test_batch_upsert_same_fields_both_variants() -> None:
    """Both sync and async batch_upsert fill the same BatchResult fields."""
    mock_response = MagicMock()
    mock_response.content = _UPSERT_RESPONSE

    sync_docs = _make_sync_docs()
    sync_docs._http = MagicMock()
    sync_docs._http.post.return_value = mock_response
    sync_result = sync_docs.batch_upsert(
        namespace="ns",
        documents=_DOCS_FIXTURE,
        batch_size=10,
        max_concurrency=1,
        show_progress=False,
    )

    async_docs = _make_async_docs()
    async_docs._http = AsyncMock()
    async_docs._http.post.return_value = mock_response
    async_result = await async_docs.batch_upsert(
        namespace="ns",
        documents=_DOCS_FIXTURE,
        batch_size=10,
        max_concurrency=1,
        show_progress=False,
    )

    for field in [
        "total_item_count",
        "successful_item_count",
        "failed_item_count",
        "total_batch_count",
        "successful_batch_count",
        "failed_batch_count",
    ]:
        assert getattr(sync_result, field) == getattr(async_result, field), (
            f"BatchResult.{field} differs between sync and async"
        )


# ---------------------------------------------------------------------------
# PreviewIndex.documents and AsyncPreviewIndex.documents are correct types
# ---------------------------------------------------------------------------


def test_preview_index_documents_type() -> None:
    config = _make_config()
    idx = PreviewIndex(host="https://host.test", config=config)
    assert isinstance(idx.documents, PreviewDocuments)


def test_async_preview_index_documents_type() -> None:
    config = _make_config()
    idx = AsyncPreviewIndex(config=config, host="https://host.test")
    assert isinstance(idx.documents, AsyncPreviewDocuments)


@pytest.mark.asyncio
async def test_async_preview_index_documents_lazy_resolves_on_first_call() -> None:
    """idx.documents returns AsyncPreviewDocuments synchronously; host provider fires once."""
    import httpx
    import respx

    config = _make_config()
    call_count = 0

    async def _provider() -> str:
        nonlocal call_count
        call_count += 1
        return "https://lazy-host.svc.pinecone.io"

    idx = AsyncPreviewIndex(config=config, _host_provider=_provider)

    # .documents is available synchronously before any await
    assert isinstance(idx.documents, AsyncPreviewDocuments)
    assert call_count == 0

    # First data-plane call triggers host resolution exactly once
    with respx.mock:
        respx.post("https://lazy-host.svc.pinecone.io/namespaces/ns/documents/upsert").mock(
            return_value=httpx.Response(200, json={"upserted_count": 1})
        )
        await idx.documents.upsert(namespace="ns", documents=[{"_id": "a"}])

    assert call_count == 1

    # Second data-plane call reuses cached host — provider not called again
    with respx.mock:
        respx.post("https://lazy-host.svc.pinecone.io/namespaces/ns/documents/upsert").mock(
            return_value=httpx.Response(200, json={"upserted_count": 1})
        )
        await idx.documents.upsert(namespace="ns", documents=[{"_id": "b"}])

    assert call_count == 1
