"""Document request models (2026-07 API).

These models are the typed boundary for document operations. Each envelope
validates its selector and size constraints at construction, mirroring the
backend's error vocabulary, so malformed requests fail before any HTTP
request. All use ``omit_defaults=True`` so unset optional fields stay off
the wire.

Records passed as :class:`DocumentRecord` / :class:`UpdateDocumentRecord`
or as plain dicts are normalized to validated plain dicts in
``__post_init__``, so envelopes encode directly with ``msgspec.json``.
"""

from __future__ import annotations

from typing import Any

import msgspec
from msgspec import Struct

from pinecone.models.documents.document import DocumentRecord, UpdateDocumentRecord
from pinecone.models.documents.score_by import (
    DenseVectorQuery,
    DocumentScoringMethod,
    SparseVectorQuery,
)

__all__ = [
    "DeleteDocumentsRequest",
    "FetchDocumentsRequest",
    "ListDocumentsRequest",
    "SearchDocumentsRequest",
    "UpdateDocumentsRequest",
    "UpsertDocumentsRequest",
]

_MAX_DOCUMENTS_PER_REQUEST = 1000
_MAX_IDS_PER_REQUEST = 1000
_MAX_SCORING_METHODS = 100
_MAX_TOP_K = 10000
_MAX_LIST_LIMIT = 100
_MAX_PREFIX_LENGTH = 512

_FILTER_EMPTY_ERR = (
    "The 'filter' field must be a non-empty object of filter predicates, "
    'for example {"category": {"$eq": "news"}}. '
    "Remove 'filter' to select documents by ID instead."
)


def _validate_filter_not_empty(filter_: dict[str, Any] | None) -> None:
    if filter_ is not None and len(filter_) == 0:
        raise ValueError(_FILTER_EMPTY_ERR)


def _validate_id_count(count: int) -> None:
    if count > _MAX_IDS_PER_REQUEST:
        raise ValueError(
            f"Number of ids ({count}) exceeds the maximum limit of "
            f"{_MAX_IDS_PER_REQUEST} ids per request"
        )


class UpsertDocumentsRequest(Struct, kw_only=True, omit_defaults=True):
    """Request model for the ``upsert_documents`` operation.

    Attributes:
        documents: The documents to upsert into the namespace (1-1000).
            Items may be :class:`DocumentRecord` instances or plain dicts
            with an ``_id``; each is validated on construction.
    """

    documents: list[dict[str, Any] | DocumentRecord]

    def __post_init__(self) -> None:
        if not self.documents:
            raise ValueError("'documents' must contain at least one document to upsert.")
        if len(self.documents) > _MAX_DOCUMENTS_PER_REQUEST:
            raise ValueError(
                f"Number of documents ({len(self.documents)}) exceeds the maximum "
                f"limit of {_MAX_DOCUMENTS_PER_REQUEST} documents"
            )
        self.documents = [
            (item if isinstance(item, DocumentRecord) else DocumentRecord(item)).to_dict()
            for item in self.documents
        ]


class SearchDocumentsRequest(Struct, kw_only=True, omit_defaults=True):
    """Request model for the ``search_documents`` operation.

    Attributes:
        score_by: The scoring methods to rank documents by (1-100). Items
            may be typed :data:`DocumentScoringMethod` variants or plain
            dicts with a ``type`` key. A ``dense_vector`` or
            ``sparse_vector`` clause must appear alone; ``text`` and
            ``query_string`` clauses may be combined.
        top_k: The number of top-ranked documents to return (1-10000).
        include_fields: The document fields to include in the results.
            ``None`` omits the key, which the server treats the same as
            ``[]``: only ``_id`` and ``_score`` come back. ``["*"]`` returns
            every field.
        filter: A metadata filter expression to restrict the documents
            searched, or ``None``.
    """

    score_by: list[DocumentScoringMethod | dict[str, Any]]
    top_k: int
    include_fields: list[str] | None = None
    filter: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.score_by:
            raise ValueError("'score_by' must contain at least one scoring method.")
        if len(self.score_by) > _MAX_SCORING_METHODS:
            raise ValueError(
                f"Number of scoring methods exceeds the maximum limit of "
                f"{_MAX_SCORING_METHODS} methods"
            )
        normalized: list[DocumentScoringMethod | dict[str, Any]] = [
            item
            if not isinstance(item, dict)
            else msgspec.convert(item, type=DocumentScoringMethod)
            for item in self.score_by
        ]
        if len(normalized) > 1 and any(
            isinstance(item, (DenseVectorQuery, SparseVectorQuery)) for item in normalized
        ):
            raise ValueError(
                "'score_by' clauses of type 'text' and 'query_string' may be combined, "
                "but 'dense_vector' and 'sparse_vector' clauses must appear alone in "
                "'score_by'"
            )
        self.score_by = normalized
        if not 1 <= self.top_k <= _MAX_TOP_K:
            raise ValueError(f"'top_k' must be between 1 and {_MAX_TOP_K} (got {self.top_k}).")


class FetchDocumentsRequest(Struct, kw_only=True, omit_defaults=True):
    """Request model for the ``fetch_documents`` operation.

    Exactly one of ``ids`` or ``filter`` must be provided.

    Attributes:
        ids: Document IDs to fetch (1-1000). Mutually exclusive with
            ``filter``.
        filter: A non-empty metadata filter expression selecting the
            documents to fetch. Mutually exclusive with ``ids``.
        include_fields: The document fields to include in the response.
            ``None`` omits the key, which the server treats the same as
            ``[]`` or ``["*"]``: every field is returned.
        pagination_token: Token from a previous fetch response to retrieve
            the next page of matching documents; the server chooses the page
            size. Only valid together with ``filter``.
    """

    ids: list[str] | None = None
    filter: dict[str, Any] | None = None
    include_fields: list[str] | None = None
    pagination_token: str | None = None

    def __post_init__(self) -> None:
        if self.pagination_token is not None and self.filter is None:
            raise ValueError(
                "The 'pagination_token' field is only valid together with 'filter' in "
                "document fetch requests. Remove 'pagination_token', or replace 'ids' "
                "with a 'filter' to page through matching documents."
            )
        has_ids = bool(self.ids)
        if has_ids and self.filter is not None:
            raise ValueError(
                "The 'ids' and 'filter' fields are mutually exclusive in document fetch "
                "requests. Provide 'ids' to fetch specific documents, or 'filter' to "
                "fetch every document matching a metadata filter."
            )
        if not has_ids and self.filter is None:
            raise ValueError(
                "No 'ids' or 'filter' provided in the document fetch request. Provide "
                "at least one document ID in 'ids', or a metadata filter in 'filter'."
            )
        _validate_filter_not_empty(self.filter)
        if self.ids is not None:
            _validate_id_count(len(self.ids))


class DeleteDocumentsRequest(Struct, kw_only=True, omit_defaults=True):
    """Request model for the ``delete_documents`` operation.

    Exactly one of ``ids``, ``filter``, or ``delete_all`` must be provided.

    Attributes:
        ids: Document IDs to delete (1-1000). Mutually exclusive with
            ``filter`` and ``delete_all``.
        filter: A non-empty metadata filter expression selecting the
            documents to delete. Text-match operators are not supported
            here. Mutually exclusive with ``ids`` and ``delete_all``.
        delete_all: If ``True``, delete all documents in the namespace.
            Mutually exclusive with ``ids`` and ``filter``.
    """

    ids: list[str] | None = None
    filter: dict[str, Any] | None = None
    delete_all: bool | None = None

    def __post_init__(self) -> None:
        has_ids = bool(self.ids)
        wants_delete_all = bool(self.delete_all)
        if self.filter is not None and (has_ids or wants_delete_all):
            raise ValueError(
                "The 'filter' field is mutually exclusive with 'ids' and 'delete_all' "
                "in document delete requests. Provide exactly one of 'ids', 'filter', "
                "or 'delete_all'."
            )
        if self.filter is None:
            if has_ids and wants_delete_all:
                raise ValueError(
                    "The 'ids' and 'delete_all' fields are mutually exclusive in "
                    "document delete requests. Provide 'ids' to delete specific "
                    "documents, or 'delete_all' to delete every document in the "
                    "namespace."
                )
            if not has_ids and not wants_delete_all:
                raise ValueError(
                    "No 'ids', 'filter', or 'delete_all' provided in the document "
                    "delete request. Provide exactly one of them."
                )
        _validate_filter_not_empty(self.filter)
        if self.ids is not None:
            _validate_id_count(len(self.ids))


class UpdateDocumentsRequest(Struct, kw_only=True, omit_defaults=True):
    """Request model for the ``update_documents`` operation.

    Either ``documents`` (per-ID patches) or ``filter`` (with a non-empty
    ``set_fields`` and/or ``remove_fields``) must be provided; they are
    mutually exclusive. An empty ``set_fields`` or ``remove_fields`` asks
    for no change and is ignored.

    Attributes:
        documents: Per-document partial updates (1-1000). Items may be
            :class:`UpdateDocumentRecord` instances or plain dicts with an
            ``_id``; each is validated on construction. Mutually exclusive
            with ``filter`` and non-empty patch fields.
        filter: A non-empty metadata filter expression selecting the
            documents to patch. Text-match operators are not supported
            here. Mutually exclusive with ``documents``.
        set_fields: Fields to set on every document matching ``filter``,
            and the values to set them to.
        remove_fields: Names of the fields to remove from every document
            matching ``filter``.
    """

    documents: list[dict[str, Any] | UpdateDocumentRecord] | None = None
    filter: dict[str, Any] | None = None
    set_fields: dict[str, Any] | None = None
    remove_fields: list[str] | None = None

    def __post_init__(self) -> None:
        has_patch = bool(self.set_fields) or bool(self.remove_fields)
        if self.documents is not None and (self.filter is not None or has_patch):
            raise ValueError(
                "The 'documents' field is mutually exclusive with the by-filter fields "
                "('filter', 'set_fields', 'remove_fields') in document update requests. "
                "Provide per-document patches in 'documents', or a 'filter' together "
                "with 'set_fields' and/or 'remove_fields'."
            )
        if self.documents is None and self.filter is None:
            if has_patch:
                raise ValueError(
                    "The 'set_fields' and 'remove_fields' fields are only valid "
                    "together with 'filter' in document update requests. Add a "
                    "'filter' to select the documents to patch, or move the changes "
                    "into per-document patches in 'documents'."
                )
            raise ValueError(
                "No 'documents' or 'filter' provided in the document update request. "
                "Provide per-document patches in 'documents', or a 'filter' together "
                "with 'set_fields' and/or 'remove_fields'."
            )
        if self.filter is not None:
            _validate_filter_not_empty(self.filter)
            if not has_patch:
                raise ValueError(
                    "A document update by 'filter' must change something. Provide "
                    "'set_fields' and/or 'remove_fields' alongside 'filter'."
                )
        if self.documents is not None:
            if not self.documents:
                raise ValueError("'documents' must contain at least one document update.")
            if len(self.documents) > _MAX_DOCUMENTS_PER_REQUEST:
                raise ValueError(
                    f"Number of documents ({len(self.documents)}) exceeds the maximum "
                    f"limit of {_MAX_DOCUMENTS_PER_REQUEST} documents"
                )
            self.documents = [
                (
                    item if isinstance(item, UpdateDocumentRecord) else UpdateDocumentRecord(item)
                ).to_dict()
                for item in self.documents
            ]


class ListDocumentsRequest(Struct, kw_only=True, omit_defaults=True):
    """Request model for the ``list_documents`` operation.

    Attributes:
        prefix: Return only documents whose IDs begin with this prefix
            (ASCII, at most 512 characters), or ``None`` for no prefix
            filtering.
        limit: Maximum number of documents to return per page (1-100), or
            ``None`` to let the server choose the page size.
        pagination_token: Token from a previous list response to retrieve
            the next page of results.
    """

    prefix: str | None = None
    limit: int | None = None
    pagination_token: str | None = None

    def __post_init__(self) -> None:
        if self.prefix is not None:
            if len(self.prefix) > _MAX_PREFIX_LENGTH:
                raise ValueError(
                    f"'prefix' exceeds the maximum length of {_MAX_PREFIX_LENGTH} "
                    f"characters (got {len(self.prefix)})."
                )
            if any(not ("\x01" <= char <= "\x7f") for char in self.prefix):
                raise ValueError(
                    "'prefix' must contain only ASCII characters in the range "
                    "\\x01-\\x7F (no NUL byte, no non-ASCII characters)."
                )
        if self.limit is not None and not 1 <= self.limit <= _MAX_LIST_LIMIT:
            raise ValueError(f"'limit' must be between 1 and {_MAX_LIST_LIMIT} (got {self.limit}).")
