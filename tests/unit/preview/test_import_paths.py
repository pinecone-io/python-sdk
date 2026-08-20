"""§12 import-path smoke tests.

Verifies every listed path from spec/preview.md §12 resolves correctly and
that preview symbols are NOT accessible from the top-level pinecone namespace.
No network I/O — pure import-resolution and identity assertions.
"""

from __future__ import annotations

import inspect

import pytest


def test_preview_root_exports_namespace_classes() -> None:
    from pinecone.preview import AsyncPreview, Preview

    assert inspect.isclass(Preview)
    assert inspect.isclass(AsyncPreview)


def test_preview_root_exports_schema_builder() -> None:
    import pinecone.schema_builder as _sb
    from pinecone.preview import SchemaBuilder

    assert inspect.isclass(SchemaBuilder)
    assert SchemaBuilder is _sb.SchemaBuilder


def test_preview_root_all_lists_schema_builder() -> None:
    import pinecone.preview as _preview

    assert "SchemaBuilder" in _preview.__all__


def test_preview_models_exports_score_by_queries() -> None:
    import pinecone.preview.models.score_by as _sb
    from pinecone.preview.models import (
        PreviewDenseVectorQuery,
        PreviewQueryStringQuery,
        PreviewSparseVectorQuery,
        PreviewTextQuery,
    )

    assert inspect.isclass(PreviewTextQuery)
    assert inspect.isclass(PreviewQueryStringQuery)
    assert inspect.isclass(PreviewDenseVectorQuery)
    assert inspect.isclass(PreviewSparseVectorQuery)
    assert PreviewTextQuery is _sb.PreviewTextQuery
    assert PreviewQueryStringQuery is _sb.PreviewQueryStringQuery
    assert PreviewDenseVectorQuery is _sb.PreviewDenseVectorQuery
    assert PreviewSparseVectorQuery is _sb.PreviewSparseVectorQuery


def test_preview_models_exports_response_models() -> None:
    from pinecone.preview.models import (
        PreviewDocument,
        PreviewDocumentFetchResponse,
        PreviewDocumentSearchResponse,
        PreviewDocumentUpsertResponse,
        PreviewIndexModel,
        PreviewSchema,
        PreviewSparseValues,
    )

    assert inspect.isclass(PreviewIndexModel)
    assert inspect.isclass(PreviewSchema)
    assert inspect.isclass(PreviewDocument)
    assert inspect.isclass(PreviewDocumentSearchResponse)
    assert inspect.isclass(PreviewDocumentFetchResponse)
    assert inspect.isclass(PreviewDocumentUpsertResponse)
    assert inspect.isclass(PreviewSparseValues)


def test_preview_symbols_not_in_pinecone_top_level() -> None:
    import pinecone

    preview_symbols = {
        "AsyncPreview",
        "Preview",
        "PreviewDenseVectorQuery",
        "PreviewDocument",
        "PreviewDocumentFetchResponse",
        "PreviewDocumentSearchResponse",
        "PreviewDocumentUpsertResponse",
        "PreviewIndexModel",
        "PreviewQueryStringQuery",
        "PreviewSchema",
        "PreviewSparseValues",
        "PreviewSparseVectorQuery",
        "PreviewTextQuery",
        "SchemaBuilder",
    }
    for name in preview_symbols:
        assert name not in pinecone.__all__, f"{name!r} must not be in pinecone.__all__"
        assert name not in pinecone._LAZY_IMPORTS, f"{name!r} must not be in pinecone._LAZY_IMPORTS"
        with pytest.raises(AttributeError):
            getattr(pinecone, name)


def test_preview_namespace_access_pattern() -> None:
    from pinecone import AsyncPinecone, Pinecone

    assert isinstance(Pinecone.__dict__["preview"], property)
    assert isinstance(AsyncPinecone.__dict__["preview"], property)
