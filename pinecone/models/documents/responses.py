"""Document response models (2026-07 API).

Envelopes and usage models are typed ``msgspec.Struct`` classes. The two
envelopes that carry open-schema documents (search and fetch) hold
:class:`~pinecone.models.documents.document.Document` dict-wrappers, so
they are plain classes constructed via :meth:`from_dict` from the decoded
response body.
"""

from __future__ import annotations

from typing import Any

import msgspec
from msgspec import Struct

from pinecone.models._display import render_table
from pinecone.models.documents.document import Document
from pinecone.models.response_info import ResponseInfo
from pinecone.models.vectors.responses import Pagination

__all__ = [
    "DeleteDocumentsResponse",
    "DocumentFetchUsage",
    "DocumentListUsage",
    "DocumentSearchUsage",
    "FetchDocumentsResponse",
    "ListDocumentsResponse",
    "ListedDocumentRecord",
    "SearchDocumentsResponse",
    "UpdateDocumentsResponse",
    "UpsertDocumentsResponse",
]


class DocumentSearchUsage(Struct, kw_only=True, gc=False):
    """Usage information for the ``search_documents`` operation.

    Attributes:
        read_units: Number of read units consumed by the request.
    """

    read_units: int


class DocumentFetchUsage(Struct, kw_only=True, gc=False):
    """Usage information for the ``fetch_documents`` operation.

    Attributes:
        read_units: Number of read units consumed by the request.
    """

    read_units: int


class DocumentListUsage(Struct, kw_only=True, gc=False):
    """Usage information for the ``list_documents`` operation.

    Attributes:
        read_units: Number of read units consumed by the request.
    """

    read_units: int


class UpsertDocumentsResponse(Struct, kw_only=True):
    """Response from a document upsert operation.

    Attributes:
        upserted_count: Number of documents successfully upserted.
        response_info: HTTP response metadata (request ID and LSN headers),
            or ``None`` when not present.
    """

    upserted_count: int
    response_info: ResponseInfo | None = None


class DeleteDocumentsResponse(Struct, kw_only=True):
    """Response from a document delete operation.

    Attributes:
        matched_records: The number of documents that matched ``filter``
            when the delete was accepted. Only returned for a filtered
            delete — ``None`` for by-id and delete-all paths, and when the
            count could not be read in time. ``0`` means the filter matched
            no documents. The delete is applied asynchronously, so this is
            a point-in-time count rather than a guarantee of the number of
            documents ultimately deleted.
        response_info: HTTP response metadata (request ID and LSN headers),
            or ``None`` when not present.
    """

    matched_records: int | None = None
    response_info: ResponseInfo | None = None


class UpdateDocumentsResponse(Struct, kw_only=True):
    """Response from a document update operation.

    Attributes:
        matched_records: The number of documents that matched ``filter``
            when the update was accepted. Only returned for a filtered
            update — ``None`` for per-ID updates and when the count could
            not be read in time. The patch is applied asynchronously, so
            this is a point-in-time count rather than a guarantee of the
            number of documents ultimately patched.
        response_info: HTTP response metadata (request ID and LSN headers),
            or ``None`` when not present.
    """

    matched_records: int | None = None
    response_info: ResponseInfo | None = None


class ListedDocumentRecord(Struct, kw_only=True, gc=False):
    """A listed document containing only its ID.

    Attributes:
        id: The unique identifier of the document (wire field ``_id``).
    """

    id: str = msgspec.field(name="_id")

    @property
    def _id(self) -> str:
        return self.id


class ListDocumentsResponse(Struct, kw_only=True):
    """Response from a document list operation.

    Attributes:
        documents: The listed documents, in sorted order by ID.
        namespace: The namespace the documents were listed from.
        usage: API usage statistics.
        pagination: Pagination token for retrieving the next page, or
            ``None`` when there are no more results.
        response_info: HTTP response metadata (request ID and LSN headers),
            or ``None`` when not present.
    """

    documents: list[ListedDocumentRecord]
    namespace: str
    usage: DocumentListUsage
    pagination: Pagination | None = None
    response_info: ResponseInfo | None = None


class SearchDocumentsResponse:
    """Response from a document search operation.

    Attributes:
        matches: Matching documents, ordered from most to least similar.
        namespace: The namespace that was searched.
        usage: API usage statistics, or ``None`` when not returned.
        response_info: HTTP response metadata (request ID and LSN headers),
            or ``None`` when not present.
    """

    __slots__ = ("matches", "namespace", "response_info", "usage")
    matches: list[Document]
    namespace: str
    usage: DocumentSearchUsage | None
    response_info: ResponseInfo | None

    def __init__(
        self,
        matches: list[Document],
        namespace: str,
        usage: DocumentSearchUsage | None = None,
        response_info: ResponseInfo | None = None,
    ) -> None:
        object.__setattr__(self, "matches", matches)
        object.__setattr__(self, "namespace", namespace)
        object.__setattr__(self, "usage", usage)
        object.__setattr__(self, "response_info", response_info)

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], *, response_info: ResponseInfo | None = None
    ) -> SearchDocumentsResponse:
        """Build a response from a decoded ``search_documents`` body.

        Unknown fields on each match are preserved verbatim in the wrapped
        :class:`Document` objects.
        """
        raw_usage = data.get("usage")
        return cls(
            matches=[Document(match) for match in data.get("matches", [])],
            namespace=data.get("namespace", ""),
            usage=(
                DocumentSearchUsage(read_units=raw_usage["read_units"])
                if raw_usage is not None
                else None
            ),
            response_info=response_info,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the response in wire shape as a plain dictionary."""
        result: dict[str, Any] = {
            "matches": [match.to_dict() for match in self.matches],
            "namespace": self.namespace,
        }
        if self.usage is not None:
            result["usage"] = {"read_units": self.usage.read_units}
        return result

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SearchDocumentsResponse):
            return NotImplemented
        return (
            self.matches == other.matches
            and self.namespace == other.namespace
            and self.usage == other.usage
        )

    def __repr__(self) -> str:
        return (
            f"SearchDocumentsResponse(matches={len(self.matches)}, "
            f"namespace={self.namespace!r}, "
            f"usage={self.usage!r})"
        )

    def _repr_html_(self) -> str:
        rows: list[tuple[str, str | int | float]] = [
            ("Matches:", len(self.matches)),
            ("Namespace:", self.namespace),
        ]
        if self.usage is not None:
            rows.append(("Read Units:", self.usage.read_units))
        return render_table("SearchDocumentsResponse", rows)


class FetchDocumentsResponse:
    """Response from a document fetch operation.

    Attributes:
        documents: Map of document ID to fetched document. Only IDs that
            exist appear in the map.
        namespace: The namespace the documents were fetched from.
        usage: API usage statistics, or ``None`` when not returned.
        pagination: Pagination token for retrieving the next page of a
            filter-based fetch, or ``None`` when there are no more results
            (always ``None`` for by-id fetches).
        response_info: HTTP response metadata (request ID and LSN headers),
            or ``None`` when not present.
    """

    __slots__ = ("documents", "namespace", "pagination", "response_info", "usage")
    documents: dict[str, Document]
    namespace: str
    usage: DocumentFetchUsage | None
    pagination: Pagination | None
    response_info: ResponseInfo | None

    def __init__(
        self,
        documents: dict[str, Document],
        namespace: str,
        usage: DocumentFetchUsage | None = None,
        pagination: Pagination | None = None,
        response_info: ResponseInfo | None = None,
    ) -> None:
        object.__setattr__(self, "documents", documents)
        object.__setattr__(self, "namespace", namespace)
        object.__setattr__(self, "usage", usage)
        object.__setattr__(self, "pagination", pagination)
        object.__setattr__(self, "response_info", response_info)

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], *, response_info: ResponseInfo | None = None
    ) -> FetchDocumentsResponse:
        """Build a response from a decoded ``fetch_documents`` body.

        Unknown fields on each document are preserved verbatim in the
        wrapped :class:`Document` objects.
        """
        raw_usage = data.get("usage")
        raw_pagination = data.get("pagination")
        return cls(
            documents={doc_id: Document(doc) for doc_id, doc in data.get("documents", {}).items()},
            namespace=data.get("namespace", ""),
            usage=(
                DocumentFetchUsage(read_units=raw_usage["read_units"])
                if raw_usage is not None
                else None
            ),
            pagination=(
                Pagination(next=raw_pagination.get("next")) if raw_pagination is not None else None
            ),
            response_info=response_info,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the response in wire shape as a plain dictionary."""
        result: dict[str, Any] = {
            "documents": {doc_id: doc.to_dict() for doc_id, doc in self.documents.items()},
            "namespace": self.namespace,
        }
        if self.usage is not None:
            result["usage"] = {"read_units": self.usage.read_units}
        if self.pagination is not None:
            result["pagination"] = {"next": self.pagination.next}
        return result

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FetchDocumentsResponse):
            return NotImplemented
        return (
            self.documents == other.documents
            and self.namespace == other.namespace
            and self.usage == other.usage
            and self.pagination == other.pagination
        )

    def __repr__(self) -> str:
        return (
            f"FetchDocumentsResponse(documents={len(self.documents)}, "
            f"namespace={self.namespace!r}, usage={self.usage!r}, "
            f"pagination={self.pagination!r})"
        )
