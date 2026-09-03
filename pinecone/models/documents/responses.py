"""What the document operations hand back, one envelope per verb.

The two envelopes that carry documents — search and fetch — hold
:class:`~pinecone.models.documents.document.Document` objects, so a field the SDK has
never heard of comes through untouched. Read each document as ``doc.id``, ``doc.score``
and then your own fields.
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
    """What one document search cost.

    Attributes:
        read_units: Read units this call consumed.
    """

    read_units: int


class DocumentFetchUsage(Struct, kw_only=True, gc=False):
    """What one document fetch cost.

    Attributes:
        read_units: Read units this call consumed.
    """

    read_units: int


class DocumentListUsage(Struct, kw_only=True, gc=False):
    """What one document list cost.

    Attributes:
        read_units: Read units this call consumed.
    """

    read_units: int


class UpsertDocumentsResponse(Struct, kw_only=True):
    """What a document upsert wrote.

    Attributes:
        upserted_count: Documents the server accepted.
        response_info: HTTP response metadata (request ID and LSN headers), or ``None``
            when not present.
    """

    upserted_count: int
    response_info: ResponseInfo | None = None


class DeleteDocumentsResponse(Struct, kw_only=True):
    """Confirmation that a document delete was accepted, and what it matched.

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
    """Confirmation that a document update was accepted, and what it matched.

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
    """One document ID from a list, and nothing else.

    This is what iterating ``idx.documents.list(...)`` yields. A list walks the IDs in a
    namespace without reading the documents, so none of your fields are here — fetch the
    IDs you want to read. The identifier is ``entry.id``; ``entry._id`` is the same value
    under the JSON key name.

    Attributes:
        id: The document's identifier. JSON key ``_id``.

    Examples:
        .. code-block:: python

            for entry in idx.documents.list(namespace="articles-en", prefix="article-"):
                print(entry.id)
    """

    id: str = msgspec.field(name="_id")

    @property
    def _id(self) -> str:
        """Alias for :attr:`id`, matching the JSON key. Prefer :attr:`id`."""
        return self.id


class ListDocumentsResponse(Struct, kw_only=True):
    """One decoded page of a document list, as it comes off the wire.

    ``idx.documents.list`` does not hand this to you — it returns a
    :class:`~pinecone.models.pagination.Paginator` that consumes these pages and yields
    the :class:`ListedDocumentRecord` entries, following ``pagination`` for you. Read this
    model when you are driving the paging yourself.

    Attributes:
        documents: The ID entries on this page, sorted by ID.
        namespace: The namespace the IDs were listed from.
        usage: What this page cost.
        pagination: Token for the next page, or ``None`` when this is the last page.
        response_info: HTTP response metadata (request ID and LSN headers), or ``None``
            when not present.

    .. seealso::
       :doc:`/guides/pagination` — which pagination shape applies where, and the
       paginator that saves you writing the loop.
    """

    documents: list[ListedDocumentRecord]
    namespace: str
    usage: DocumentListUsage
    pagination: Pagination | None = None
    response_info: ResponseInfo | None = None


class SearchDocumentsResponse:
    """The ranked documents a search found.

    ``matches`` is already ordered, so ``matches[0]`` is the best hit. Each element is a
    :class:`~pinecone.models.documents.document.Document`: read ``doc.id`` and
    ``doc.score``, then your own fields by name. Which fields are present depends on the
    search's ``include_fields`` — by default only the ID and the score come back, so ask
    for the fields you intend to read. A search that matched nothing returns an empty
    ``matches`` rather than raising.

    Attributes:
        matches: The matching documents, most relevant first.
        namespace: The namespace that was searched.
        usage: What the search cost, or ``None`` when not returned.
        response_info: HTTP response metadata (request ID and LSN headers), or ``None``
            when not present.

    Examples:
        .. code-block:: python

            response = idx.documents.search(
                namespace="articles-en",
                score_by=[{"type": "text", "query": "vector search", "fields": ["title"]}],
                top_k=5,
                include_fields=["title"],
            )
            for doc in response.matches:
                print(doc.id, doc.score, doc.title)
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
        """Build a response from an already-decoded search body.

        Fields the SDK does not know about are kept verbatim on each wrapped
        :class:`~pinecone.models.documents.document.Document`.

        Args:
            data (dict[str, Any]): The decoded response body.
            response_info (ResponseInfo | None): HTTP response metadata to attach, or
                ``None``. Keyword-only.

        Returns:
            :class:`SearchDocumentsResponse` with one
            :class:`~pinecone.models.documents.document.Document` per match.
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
        """Return the response as a plain dict in its JSON shape.

        Reserved keys come back under their JSON names, so each document carries ``_id``
        and, for a search, ``_score``.
        """
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
    """The documents a fetch retrieved, keyed by ID.

    ``documents`` is a dict, so look a document up by the ID you asked for. An ID that
    does not exist is simply absent — fetching a missing ID is not an error — so test
    membership rather than indexing blind. Unlike a search, a fetch returns every field by
    default.

    Attributes:
        documents: Document ID to :class:`~pinecone.models.documents.document.Document`,
            for the requested IDs that exist.
        namespace: The namespace the documents were fetched from.
        usage: What the fetch cost, or ``None`` when not returned.
        pagination: Token for the next page of a fetch by filter, or ``None`` when this is
            the last page. Always ``None`` for a fetch by ID, which does not page.
        response_info: HTTP response metadata (request ID and LSN headers), or ``None``
            when not present.

    Examples:
        .. code-block:: python

            wanted = ["article-101", "article-102"]
            response = idx.documents.fetch(ids=wanted, namespace="articles-en")
            for doc_id, doc in response.documents.items():
                print(doc_id, doc.title)
            print("not stored:", [d for d in wanted if d not in response.documents])
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
        """Build a response from an already-decoded fetch body.

        Fields the SDK does not know about are kept verbatim on each wrapped
        :class:`~pinecone.models.documents.document.Document`.

        Args:
            data (dict[str, Any]): The decoded response body.
            response_info (ResponseInfo | None): HTTP response metadata to attach, or
                ``None``. Keyword-only.

        Returns:
            :class:`FetchDocumentsResponse` keyed by document ID.
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
        """Return the response as a plain dict in its JSON shape.

        Reserved keys come back under their JSON names, so each document carries ``_id``
        and, for a search, ``_score``.
        """
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
