"""Shared request-building and validation helpers for document operations.

Both the sync :class:`~pinecone.index.Index` and the async twin build their
document request bodies through these helpers, so the two transports emit
byte-identical wire bodies and raise byte-identical validation errors.

Validation and serialization are delegated to the typed request models in
:mod:`pinecone.models.documents.requests`; these helpers adapt their plain
:class:`ValueError` vocabulary into the SDK's :class:`PineconeValueError`
and add the cross-record checks (duplicate ``_id``, per-position error
context) the models cannot see.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import quote

import msgspec

from pinecone.errors.exceptions import ValidationError
from pinecone.models.documents.document import DocumentRecord, UpdateDocumentRecord
from pinecone.models.documents.requests import (
    DeleteDocumentsRequest,
    FetchDocumentsRequest,
    ListDocumentsRequest,
    SearchDocumentsRequest,
    UpdateDocumentsRequest,
)
from pinecone.models.documents.score_by import DocumentScoringMethod

_MAX_DOCUMENTS_PER_REQUEST = 1000


def _encode_document_namespace(namespace: str) -> str:
    """Validate a documents-operation namespace and URL-encode it as a path segment.

    Document operations address the namespace in the URL path, so unlike the
    vector operations there is no empty-string default — the namespace must be
    a non-empty string, and characters that are not path-safe (``/``, spaces,
    ``%``, non-ASCII) are percent-encoded so they cannot change the route.

    Raises:
        :exc:`PineconeValueError`: If *namespace* is not a non-empty string.
    """
    if not isinstance(namespace, str):
        raise ValidationError("namespace must be a string")
    if not namespace or not namespace.strip():
        raise ValidationError("namespace must be a non-empty string")
    return quote(namespace, safe="")


def _validate_documents(
    documents: Sequence[Mapping[str, Any] | DocumentRecord],
    *,
    max_documents: int | None = _MAX_DOCUMENTS_PER_REQUEST,
) -> list[dict[str, Any]]:
    """Validate documents for upsert and normalize them to wire-shape dicts.

    A document list is accepted if and only if every element is a dict or
    :class:`DocumentRecord` whose ``_id`` is a non-empty ASCII string of at
    most 512 characters (``^[\\x01-\\x7F]+$``), unique within the list.

    Args:
        documents: The documents to validate.
        max_documents: Per-request cap (1000 for a single upsert request).
            ``None`` disables the cap for callers that split into batches.

    Returns:
        The documents as plain dicts, in input order.

    Raises:
        :exc:`PineconeValueError`: If the list is empty or over *max_documents*,
            or any document violates the ``_id`` contract — the message names
            the position of the offending document.
    """
    docs = list(documents)
    if not docs:
        raise ValidationError("documents must be a non-empty list")
    if max_documents is not None and len(docs) > max_documents:
        raise ValidationError(
            f"Number of documents ({len(docs)}) exceeds the maximum limit of "
            f"{max_documents} documents per request. Use batch_upsert_documents "
            "to split a larger list into multiple requests."
        )
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for position, doc in enumerate(docs):
        if isinstance(doc, DocumentRecord):
            item = doc.to_dict()
        elif isinstance(doc, Mapping):
            try:
                item = DocumentRecord(dict(doc)).to_dict()
            except ValueError as exc:
                raise ValidationError(f"Document at position {position}: {exc}") from exc
        else:
            raise ValidationError(
                f"Document at position {position} must be a dict or DocumentRecord, "
                f"got {type(doc).__name__}"
            )
        doc_id: str = item["_id"]
        if doc_id in seen:
            raise ValidationError(
                f"Document at position {position} has duplicate '_id' {doc_id!r}; "
                "document IDs must be unique within a request."
            )
        seen.add(doc_id)
        normalized.append(item)
    return normalized


def _validate_update_documents(
    documents: Sequence[Mapping[str, Any] | UpdateDocumentRecord],
) -> list[dict[str, Any]]:
    """Validate per-ID document patches and normalize them to wire-shape dicts.

    A patch list is accepted if and only if every element is a dict or
    :class:`UpdateDocumentRecord` whose ``_id`` satisfies the same contract
    :func:`_validate_documents` enforces, whose ``_remove_fields`` (when
    present) is a list of strings naming no field the same patch also sets,
    and whose ``_id`` is unique within the list.

    Duplicate ``_id`` values are rejected rather than sent: the server
    applies the patches in a request independently, so two patches to one
    document would resolve in an unspecified order. This mirrors the
    duplicate rule ``upsert_documents`` already applies.

    Returns:
        The patches as plain dicts, in input order.

    Raises:
        :exc:`PineconeValueError`: If the list is empty or over 1000 entries,
            or any patch is malformed — the message names the position of the
            offending patch.
    """
    docs = list(documents)
    if not docs:
        raise ValidationError("documents must be a non-empty list")
    if len(docs) > _MAX_DOCUMENTS_PER_REQUEST:
        raise ValidationError(
            f"Number of documents ({len(docs)}) exceeds the maximum limit of "
            f"{_MAX_DOCUMENTS_PER_REQUEST} documents per request."
        )
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for position, doc in enumerate(docs):
        if isinstance(doc, UpdateDocumentRecord):
            item = doc.to_dict()
        elif isinstance(doc, Mapping):
            try:
                item = UpdateDocumentRecord(dict(doc)).to_dict()
            except ValueError as exc:
                raise ValidationError(f"Document at position {position}: {exc}") from exc
        else:
            raise ValidationError(
                f"Document at position {position} must be a dict or UpdateDocumentRecord, "
                f"got {type(doc).__name__}"
            )
        doc_id: str = item["_id"]
        if doc_id in seen:
            raise ValidationError(
                f"Document at position {position} has duplicate '_id' {doc_id!r}; "
                "document IDs must be unique within a request."
            )
        seen.add(doc_id)
        normalized.append(item)
    return normalized


def _build_search_documents_body(
    *,
    score_by: Sequence[DocumentScoringMethod | Mapping[str, Any]],
    top_k: int,
    include_fields: Sequence[str] | None,
    filter: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build and validate a ``search_documents`` request body.

    Raises:
        :exc:`PineconeValueError`: If ``score_by`` is empty, over 100 clauses,
            combines a ``dense_vector``/``sparse_vector`` clause with any other
            clause, or contains an invalid clause; or if ``top_k`` is outside
            [1, 10000].
    """
    try:
        request = SearchDocumentsRequest(
            score_by=[dict(item) if isinstance(item, Mapping) else item for item in score_by],
            top_k=top_k,
            include_fields=list(include_fields) if include_fields is not None else None,
            filter=dict(filter) if filter is not None else None,
        )
    except (ValueError, msgspec.ValidationError) as exc:
        raise ValidationError(str(exc)) from exc
    body: dict[str, Any] = msgspec.to_builtins(request)
    return body


def _build_fetch_documents_body(
    *,
    ids: Sequence[str] | None,
    filter: Mapping[str, Any] | None,
    include_fields: Sequence[str] | None,
    pagination_token: str | None,
) -> dict[str, Any]:
    """Build and validate a ``fetch_documents`` request body.

    Raises:
        :exc:`PineconeValueError`: If both or neither of ``ids`` and ``filter``
            are provided, ``filter`` is empty, ``ids`` exceeds 1000 entries, or
            ``pagination_token`` is passed without ``filter``.
    """
    try:
        request = FetchDocumentsRequest(
            ids=list(ids) if ids is not None else None,
            filter=dict(filter) if filter is not None else None,
            include_fields=list(include_fields) if include_fields is not None else None,
            pagination_token=pagination_token,
        )
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc
    body: dict[str, Any] = msgspec.to_builtins(request)
    return body


def _build_delete_documents_body(
    *,
    ids: Sequence[str] | None,
    filter: Mapping[str, Any] | None,
    delete_all: bool,
) -> dict[str, Any]:
    """Build and validate a ``delete_documents`` request body.

    Raises:
        :exc:`PineconeValueError`: If zero or more than one of ``ids``,
            ``filter``, and ``delete_all`` are provided, ``filter`` is empty,
            or ``ids`` exceeds 1000 entries.
    """
    try:
        request = DeleteDocumentsRequest(
            ids=list(ids) if ids is not None else None,
            filter=dict(filter) if filter is not None else None,
            delete_all=True if delete_all else None,
        )
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc
    body: dict[str, Any] = msgspec.to_builtins(request)
    return body


def _build_update_documents_body(
    *,
    documents: Sequence[Mapping[str, Any] | UpdateDocumentRecord] | None,
    filter: Mapping[str, Any] | None,
    set_fields: Mapping[str, Any] | None,
    remove_fields: Sequence[str] | None,
) -> dict[str, Any]:
    """Build and validate an ``update_documents`` request body.

    Per-ID patches go through :func:`_validate_update_documents` first, so a
    malformed patch is reported with its position before the envelope's
    selector rules are applied.

    Raises:
        :exc:`PineconeValueError`: If ``documents`` is combined with any of
            the by-filter fields, neither ``documents`` nor ``filter`` is
            provided, ``set_fields``/``remove_fields`` are passed without
            ``filter``, ``filter`` is empty or carries no patch, or any
            per-ID patch is malformed.
    """
    normalized = _validate_update_documents(documents) if documents is not None else None
    try:
        request = UpdateDocumentsRequest(
            documents=list(normalized) if normalized is not None else None,
            filter=dict(filter) if filter is not None else None,
            set_fields=dict(set_fields) if set_fields is not None else None,
            remove_fields=list(remove_fields) if remove_fields is not None else None,
        )
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc
    body: dict[str, Any] = msgspec.to_builtins(request)
    return body


def _build_list_documents_body(
    *,
    prefix: str | None,
    limit: int | None,
    pagination_token: str | None,
) -> dict[str, Any]:
    """Build and validate a ``list_documents`` request body.

    Raises:
        :exc:`PineconeValueError`: If ``prefix`` is over 512 characters or
            contains a character outside ``\\x01``-``\\x7F``, or ``limit``
            falls outside 1-100.
    """
    try:
        request = ListDocumentsRequest(
            prefix=prefix,
            limit=limit,
            pagination_token=pagination_token,
        )
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc
    body: dict[str, Any] = msgspec.to_builtins(request)
    return body
