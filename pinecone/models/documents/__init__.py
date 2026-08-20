"""Document models subpackage with lazy loading."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pinecone.models.documents.document import (  # noqa: F401
        Document,
        DocumentRecord,
        UpdateDocumentRecord,
    )
    from pinecone.models.documents.requests import (  # noqa: F401
        DeleteDocumentsRequest,
        FetchDocumentsRequest,
        ListDocumentsRequest,
        SearchDocumentsRequest,
        UpdateDocumentsRequest,
        UpsertDocumentsRequest,
    )
    from pinecone.models.documents.responses import (  # noqa: F401
        DeleteDocumentsResponse,
        DocumentFetchUsage,
        DocumentListUsage,
        DocumentSearchUsage,
        FetchDocumentsResponse,
        ListDocumentsResponse,
        ListedDocumentRecord,
        SearchDocumentsResponse,
        UpdateDocumentsResponse,
        UpsertDocumentsResponse,
    )
    from pinecone.models.documents.score_by import (  # noqa: F401
        DenseVectorQuery,
        DocumentScoringMethod,
        QueryStringQuery,
        SparseVectorQuery,
        TextQuery,
    )

_LAZY_IMPORTS: dict[str, str] = {
    "Document": "pinecone.models.documents.document",
    "DocumentRecord": "pinecone.models.documents.document",
    "UpdateDocumentRecord": "pinecone.models.documents.document",
    "DenseVectorQuery": "pinecone.models.documents.score_by",
    "DocumentScoringMethod": "pinecone.models.documents.score_by",
    "QueryStringQuery": "pinecone.models.documents.score_by",
    "SparseVectorQuery": "pinecone.models.documents.score_by",
    "TextQuery": "pinecone.models.documents.score_by",
    "DeleteDocumentsRequest": "pinecone.models.documents.requests",
    "FetchDocumentsRequest": "pinecone.models.documents.requests",
    "ListDocumentsRequest": "pinecone.models.documents.requests",
    "SearchDocumentsRequest": "pinecone.models.documents.requests",
    "UpdateDocumentsRequest": "pinecone.models.documents.requests",
    "UpsertDocumentsRequest": "pinecone.models.documents.requests",
    "DeleteDocumentsResponse": "pinecone.models.documents.responses",
    "DocumentFetchUsage": "pinecone.models.documents.responses",
    "DocumentListUsage": "pinecone.models.documents.responses",
    "DocumentSearchUsage": "pinecone.models.documents.responses",
    "FetchDocumentsResponse": "pinecone.models.documents.responses",
    "ListDocumentsResponse": "pinecone.models.documents.responses",
    "ListedDocumentRecord": "pinecone.models.documents.responses",
    "SearchDocumentsResponse": "pinecone.models.documents.responses",
    "UpdateDocumentsResponse": "pinecone.models.documents.responses",
    "UpsertDocumentsResponse": "pinecone.models.documents.responses",
}

__all__ = list(_LAZY_IMPORTS.keys())


def __getattr__(name: str) -> Any:
    """Lazy-load document models on first access."""
    if name in _LAZY_IMPORTS:
        from importlib import import_module

        module = import_module(_LAZY_IMPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    import builtins

    return builtins.list({*globals(), *__all__, *_LAZY_IMPORTS})
