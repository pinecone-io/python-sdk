"""Graduation guards for the retired ``pinecone.preview`` namespace (#140).

The 2026-01.alpha preview surface graduated to the default entry points in
2026-07 and the ``pinecone/preview/`` package was deleted outright (fate
option (a) from #140): a stale preview code sample must fail loudly, in the
way documented by ``docs/migration/v10-migration.md``.

Replaces ``tests/unit/preview/test_import_paths.py`` and
``tests/unit/preview/test_namespace.py`` as the post-graduation guards, and
carries the close-propagation coverage that lived in
``tests/unit/preview/test_close.py`` and ``test_preview_documents_close.py``
forward onto the graduated surfaces.

The absence guards below deliberately do NOT use
``pytest.raises(ModuleNotFoundError)``; see
``assert_module_is_really_gone`` for why an empty leftover directory would
otherwise defeat them (#326).
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import respx

from pinecone import AsyncPinecone, Pinecone
from tests.factories import make_index_response

BASE_URL = "https://api.test.pinecone.io"
INDEX_HOST = "graduated-idx-abc123.svc.pinecone.io"
INDEX_URL = f"https://{INDEX_HOST}"

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "pinecone"

FETCH_DOCUMENTS_RESPONSE = {
    "documents": {"doc-1": {"_id": "doc-1", "title": "Rome"}},
    "namespace": "articles-en",
    "usage": {"read_units": 1},
}


def _stale_empty_dir_message(dotted_name: str, directories: list[str]) -> str:
    listing = "\n".join(f"    rm -rf {d}" for d in directories) or "    (none reported)"
    return (
        f"`import {dotted_name}` SUCCEEDED, but as an implicit namespace package "
        f"(PEP 420) rather than as a real module.\n"
        f"\n"
        f"THE 2026-07 BRANCH IS FINE. YOUR WORKING TREE HAS A STALE EMPTY "
        f"DIRECTORY. Remove it and re-run:\n"
        f"\n"
        f"{listing}\n"
        f"\n"
        f"or, to sweep every stale empty directory at once:\n"
        f"\n"
        f"    find {PACKAGE_ROOT} -type d -empty -delete\n"
        f"\n"
        f"Why: #140 (PR #316) deleted every file in pinecone/preview/. Git removes "
        f"tracked files but LEAVES THE DIRECTORY behind, and `git status` never "
        f"reports an empty directory -- so a tree rebased across #316 keeps a "
        f"phantom pinecone/preview/ that Python imports as a namespace package with "
        f"__file__ = None. CI checks out fresh, so CI never sees this. That is why "
        f"the failure looks like the preview graduation is broken when the only "
        f"problem is a leftover directory on your disk."
    )


def assert_module_is_really_gone(dotted_name: str) -> None:
    """Assert ``dotted_name`` is absent, and not merely file-less.

    ``pytest.raises(ModuleNotFoundError)`` is NOT sufficient here, and this is
    the whole point of #326. When a package's files are deleted but its
    directory survives, Python's PEP 420 finder resolves the leftover empty
    directory as an *implicit namespace package*: the import succeeds, the
    module object has ``__file__ is None`` and ``__spec__.origin is None``, and
    ``__path__`` is a ``_NamespacePath`` listing the offending directories.

    So we assert on ``__spec__.origin`` rather than on the exception: a real
    module always has an origin (the path of its ``.py``/``__init__.py``), and
    a namespace package never does. That distinction is what tells a genuinely
    retired package apart from a phantom one -- and it lets the failure name
    the exact directory the developer has to delete.
    """
    try:
        module = importlib.import_module(dotted_name)
    except ModuleNotFoundError:
        return

    sys.modules.pop(dotted_name, None)

    spec = module.__spec__
    origin = spec.origin if spec is not None else None
    if origin is None and getattr(module, "__file__", None) is None:
        pytest.fail(_stale_empty_dir_message(dotted_name, [str(p) for p in module.__path__]))

    pytest.fail(
        f"`import {dotted_name}` SUCCEEDED and resolved to a real module at "
        f"{origin!r}. The preview namespace was retired by #140 (PR #316) and "
        f"must not exist. Do not re-add it -- see "
        f"docs/migration/v10-migration.md."
    )


class TestPreviewNamespaceIsGone:
    def test_import_pinecone_preview_is_really_gone(self) -> None:
        assert_module_is_really_gone("pinecone.preview")

    @pytest.mark.parametrize(
        "dotted_name",
        [
            "pinecone.preview.indexes",
            "pinecone.preview.models",
            "pinecone.preview._internal",
            "pinecone.preview._internal.constants",
            "pinecone.preview._internal.adapters",
        ],
    )
    def test_preview_submodules_are_really_gone(self, dotted_name: str) -> None:
        assert_module_is_really_gone(dotted_name)

    def test_no_stale_empty_directories_under_pinecone(self) -> None:
        """Catch the #326 hazard for *any* future package deletion, not just preview.

        Cheap generalization of the guard above: every empty directory under
        ``pinecone/`` is an importable namespace package waiting to confuse
        someone, and git will never mention it.
        """

        def holds_no_source(directory: Path) -> bool:
            return not any(
                entry.is_file() and "__pycache__" not in entry.parts
                for entry in directory.rglob("*")
            )

        candidates = sorted(
            path
            for path in PACKAGE_ROOT.rglob("*")
            if path.is_dir()
            and path.name != "__pycache__"
            and not any(part.startswith(".") for part in path.relative_to(REPO_ROOT).parts)
            and holds_no_source(path)
        )
        topmost = [
            str(path)
            for path in candidates
            if not any(other in path.parents for other in candidates)
        ]
        assert topmost == [], _stale_empty_dir_message("<a stale package>", topmost)

    def test_sync_client_has_no_preview_attribute(self) -> None:
        pc = Pinecone(api_key="test-key-1234")
        assert not hasattr(Pinecone, "preview")
        with pytest.raises(AttributeError, match=r"Pinecone\.preview was removed at 10\.0\.0"):
            _ = pc.preview  # type: ignore[attr-defined]

    def test_async_client_has_no_preview_attribute(self) -> None:
        pc = AsyncPinecone(api_key="test-key-1234")
        assert not hasattr(AsyncPinecone, "preview")
        with pytest.raises(AttributeError, match=r"AsyncPinecone\.preview was removed at 10\.0\.0"):
            _ = pc.preview  # type: ignore[attr-defined]

    @pytest.mark.parametrize("client", [Pinecone, AsyncPinecone], ids=["sync", "async"])
    def test_preview_error_names_its_replacements(self, client: type) -> None:
        pc = client(api_key="test-key-1234")
        with pytest.raises(AttributeError) as excinfo:
            _ = pc.preview
        message = str(excinfo.value)
        assert "pc.preview.indexes is now pc.indexes" in message
        assert "pc.preview.index(...) is now pc.index(...)" in message
        assert "https://sdk.pinecone.io/python/migration/v10-migration.html" in message

    @pytest.mark.parametrize("client", [Pinecone, AsyncPinecone], ids=["sync", "async"])
    def test_unrelated_missing_attribute_keeps_the_standard_message(self, client: type) -> None:
        """Only ``preview`` is special-cased; anything else must behave normally."""
        pc = client(api_key="test-key-1234")
        with pytest.raises(
            AttributeError, match=rf"{client.__name__}' object has no attribute 'nope'"
        ):
            _ = pc.nope
        assert getattr(pc, "nope", "fallback") == "fallback"
        assert not hasattr(pc, "nope")

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
        for op in ("upsert", "batch_upsert", "search", "fetch", "delete", "update", "list"):
            assert callable(getattr(index.documents, op))
        index.close()

    @pytest.mark.asyncio
    async def test_async_pc_index_factory_exposes_documents_operations(self) -> None:
        from pinecone.async_client.async_index import AsyncIndex

        pc = AsyncPinecone(api_key="test-key-1234")
        index = await pc.index(host=INDEX_HOST)
        assert type(index) is AsyncIndex
        for op in ("upsert", "batch_upsert", "search", "fetch", "delete", "update", "list"):
            assert callable(getattr(index.documents, op))
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
            result = index.documents.fetch(namespace="articles-en", ids=["doc-1"])
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
            result = await index.documents.fetch(namespace="articles-en", ids=["doc-1"])
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
