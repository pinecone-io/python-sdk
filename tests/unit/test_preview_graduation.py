"""Graduation guards for the retired ``pinecone.preview`` namespace (#140).

The 2026-01.alpha preview surface graduated to the default entry points in
2026-07 and the ``pinecone/preview/`` package was deleted outright (fate
option (a) from #140): a stale preview code sample must fail loudly, in the
way documented by ``docs/migration/v10-2026-07-preview-graduation.md``.

Replaces ``tests/unit/preview/test_import_paths.py`` and
``tests/unit/preview/test_namespace.py`` as the post-graduation guards, and
carries the close-propagation coverage that lived in
``tests/unit/preview/test_close.py`` and ``test_preview_documents_close.py``
forward onto the graduated surfaces.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest
import respx

from pinecone import AsyncPinecone, Pinecone
from tests.factories import make_index_response

BASE_URL = "https://api.test.pinecone.io"
INDEX_HOST = "graduated-idx-abc123.svc.pinecone.io"
INDEX_URL = f"https://{INDEX_HOST}"

FETCH_DOCUMENTS_RESPONSE = {
    "documents": {"doc-1": {"_id": "doc-1", "title": "Rome"}},
    "namespace": "articles-en",
    "usage": {"read_units": 1},
}


class TestPreviewNamespaceIsGone:
    def test_import_pinecone_preview_raises_module_not_found(self) -> None:
        with pytest.raises(ModuleNotFoundError):
            import pinecone.preview  # noqa: F401

    def test_import_preview_submodules_raise_module_not_found(self) -> None:
        with pytest.raises(ModuleNotFoundError):
            from pinecone.preview.indexes import PreviewIndexes  # noqa: F401
        with pytest.raises(ModuleNotFoundError):
            from pinecone.preview.models import PreviewIndexModel  # noqa: F401
        with pytest.raises(ModuleNotFoundError):
            from pinecone.preview._internal.constants import (  # noqa: F401
                INDEXES_API_VERSION,
            )

    def test_sync_client_has_no_preview_attribute(self) -> None:
        pc = Pinecone(api_key="test-key-1234")
        assert not hasattr(Pinecone, "preview")
        with pytest.raises(AttributeError):
            _ = pc.preview  # type: ignore[attr-defined]

    def test_async_client_has_no_preview_attribute(self) -> None:
        pc = AsyncPinecone(api_key="test-key-1234")
        assert not hasattr(AsyncPinecone, "preview")
        with pytest.raises(AttributeError):
            _ = pc.preview  # type: ignore[attr-defined]

    def test_no_preview_symbols_at_top_level(self) -> None:
        import pinecone

        leaked = [
            name
            for name in {*pinecone.__all__, *pinecone._LAZY_IMPORTS}
            if name.startswith(("Preview", "AsyncPreview"))
        ]
        assert leaked == []

    def test_preview_api_version_constant_is_gone(self) -> None:
        from pinecone._internal import constants

        assert not hasattr(constants, "INDEXES_API_VERSION")
        assert "2026-01.alpha" not in vars(constants).values()


class TestGraduatedEntryPoints:
    def test_pc_indexes_serves_graduated_indexes_class(self) -> None:
        from pinecone.client.indexes import Indexes

        pc = Pinecone(api_key="test-key-1234")
        assert type(pc.indexes) is Indexes

    def test_async_pc_indexes_serves_graduated_indexes_class(self) -> None:
        from pinecone.async_client.indexes import AsyncIndexes

        pc = AsyncPinecone(api_key="test-key-1234")
        assert type(pc.indexes) is AsyncIndexes

    def test_pc_index_factory_exposes_documents_operations(self) -> None:
        from pinecone.index import Index

        pc = Pinecone(api_key="test-key-1234")
        index = pc.index(host=INDEX_HOST)
        assert type(index) is Index
        for op in (
            "upsert_documents",
            "batch_upsert_documents",
            "search_documents",
            "fetch_documents",
            "delete_documents",
            "update_documents",
            "list_documents",
        ):
            assert callable(getattr(index, op))
        index.close()

    @pytest.mark.asyncio
    async def test_async_pc_index_factory_exposes_documents_operations(self) -> None:
        from pinecone.async_client.async_index import AsyncIndex

        pc = AsyncPinecone(api_key="test-key-1234")
        index = await pc.index(host=INDEX_HOST)
        assert type(index) is AsyncIndex
        for op in (
            "upsert_documents",
            "batch_upsert_documents",
            "search_documents",
            "fetch_documents",
            "delete_documents",
            "update_documents",
            "list_documents",
        ):
            assert callable(getattr(index, op))
        await index.close()

    @respx.mock
    def test_pc_index_by_name_resolves_host_and_caches(self) -> None:
        route = respx.get(f"{BASE_URL}/indexes/test-index").mock(
            return_value=httpx.Response(200, json=make_index_response())
        )
        pc = Pinecone(api_key="test-key-1234", host=BASE_URL)
        index_a = pc.index(name="test-index")
        index_b = pc.index(name="test-index")
        assert route.call_count == 1
        index_a.close()
        index_b.close()


class TestApiVersionHeaderEndToEnd:
    @respx.mock
    def test_control_plane_call_sends_2026_07(self) -> None:
        route = respx.get(f"{BASE_URL}/indexes/test-index").mock(
            return_value=httpx.Response(200, json=make_index_response())
        )
        with Pinecone(api_key="test-key-1234", host=BASE_URL) as pc:
            info = pc.indexes.describe("test-index")
        assert info.name == "test-index"
        assert route.calls.last.request.headers["X-Pinecone-Api-Version"] == "2026-07"

    @respx.mock
    def test_data_plane_call_sends_2026_07(self) -> None:
        route = respx.post(f"{INDEX_URL}/namespaces/articles-en/documents/fetch").mock(
            return_value=httpx.Response(200, json=FETCH_DOCUMENTS_RESPONSE)
        )
        pc = Pinecone(api_key="test-key-1234")
        with pc.index(host=INDEX_HOST) as index:
            result = index.fetch_documents(namespace="articles-en", ids=["doc-1"])
        assert result.documents["doc-1"].title == "Rome"
        assert route.calls.last.request.headers["X-Pinecone-Api-Version"] == "2026-07"

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_control_plane_call_sends_2026_07(self) -> None:
        route = respx.get(f"{BASE_URL}/indexes/test-index").mock(
            return_value=httpx.Response(200, json=make_index_response())
        )
        async with AsyncPinecone(api_key="test-key-1234", host=BASE_URL) as pc:
            info = await pc.indexes.describe("test-index")
        assert info.name == "test-index"
        assert route.calls.last.request.headers["X-Pinecone-Api-Version"] == "2026-07"

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_data_plane_call_sends_2026_07(self) -> None:
        route = respx.post(f"{INDEX_URL}/namespaces/articles-en/documents/fetch").mock(
            return_value=httpx.Response(200, json=FETCH_DOCUMENTS_RESPONSE)
        )
        pc = AsyncPinecone(api_key="test-key-1234")
        async with await pc.index(host=INDEX_HOST) as index:
            result = await index.fetch_documents(namespace="articles-en", ids=["doc-1"])
        assert result.documents["doc-1"].title == "Rome"
        assert route.calls.last.request.headers["X-Pinecone-Api-Version"] == "2026-07"


class TestCloseAfterGraduation:
    def test_sync_close_closes_every_initialized_sub_client(self) -> None:
        pc = Pinecone(api_key="test-key-1234")
        _ = pc.inference
        _ = pc.assistants
        _ = pc.indexes
        with (
            patch.object(pc._http, "close") as http_close,
            patch.object(pc._inference, "close") as inference_close,
            patch.object(pc._assistants, "close") as assistants_close,
        ):
            pc.close()
        http_close.assert_called_once()
        inference_close.assert_called_once()
        assistants_close.assert_called_once()

    def test_sync_close_skips_uninitialized_sub_clients(self) -> None:
        pc = Pinecone(api_key="test-key-1234")
        assert pc._inference is None
        assert pc._assistants is None
        pc.close()

    @pytest.mark.asyncio
    async def test_async_close_closes_every_initialized_sub_client(self) -> None:
        pc = AsyncPinecone(api_key="test-key-1234")
        _ = pc.inference
        _ = pc.assistants
        _ = pc.indexes
        with (
            patch.object(pc._http, "close", new_callable=AsyncMock) as http_close,
            patch.object(pc._inference, "close", new_callable=AsyncMock) as inference_close,
            patch.object(pc._assistants, "close", new_callable=AsyncMock) as assistants_close,
        ):
            await pc.close()
        http_close.assert_called_once()
        inference_close.assert_called_once()
        assistants_close.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_close_skips_uninitialized_sub_clients(self) -> None:
        pc = AsyncPinecone(api_key="test-key-1234")
        assert pc._inference is None
        assert pc._assistants is None
        await pc.close()

    def test_index_close_closes_http_and_is_idempotent(self) -> None:
        pc = Pinecone(api_key="test-key-1234")
        index = pc.index(host=INDEX_HOST)
        with patch.object(index._http, "close") as mock_close:
            index.close()
            index.close()
        assert mock_close.call_count == 2

    def test_index_context_manager_closes_http(self) -> None:
        pc = Pinecone(api_key="test-key-1234")
        index = pc.index(host=INDEX_HOST)
        with patch.object(index._http, "close") as mock_close, index:
            pass
        mock_close.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_index_close_closes_http_and_is_idempotent(self) -> None:
        pc = AsyncPinecone(api_key="test-key-1234")
        index = await pc.index(host=INDEX_HOST)
        with patch.object(index._http, "close", new_callable=AsyncMock) as mock_close:
            await index.close()
            await index.close()
        assert mock_close.call_count == 2

    @pytest.mark.asyncio
    async def test_async_index_context_manager_closes_http(self) -> None:
        pc = AsyncPinecone(api_key="test-key-1234")
        index = await pc.index(host=INDEX_HOST)
        with patch.object(index._http, "close", new_callable=AsyncMock) as mock_close:
            async with index:
                pass
        mock_close.assert_called_once()
