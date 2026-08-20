"""Adapters decoding document-operation responses (2026-07 API).

The search and fetch envelopes are decoded through internal ``msgspec``
Structs mirroring the 2026-07 wire schemas, then rewrapped in the public
open-schema response classes so unknown document fields survive verbatim.
Fields the spec marks required are required here too: a response missing
one raises :class:`ResponseParsingError` instead of being silently patched
over with an empty default.
"""

from __future__ import annotations

from typing import Any

import httpx
from msgspec import Struct

from pinecone._internal.adapters._decode import decode_response
from pinecone._internal.adapters.vectors_adapter import extract_response_info
from pinecone.models.documents.document import Document
from pinecone.models.documents.responses import (
    DeleteDocumentsResponse,
    DocumentFetchUsage,
    DocumentSearchUsage,
    FetchDocumentsResponse,
    SearchDocumentsResponse,
    UpsertDocumentsResponse,
)
from pinecone.models.vectors.responses import Pagination

__all__ = ["DocumentsAdapter"]


class _SearchDocumentsEnvelope(Struct, kw_only=True):
    matches: list[dict[str, Any]]
    namespace: str
    usage: DocumentSearchUsage


class _FetchDocumentsEnvelope(Struct, kw_only=True):
    documents: dict[str, dict[str, Any]]
    namespace: str
    usage: DocumentFetchUsage
    pagination: Pagination | None = None


class DocumentsAdapter:
    """Adapter for document operation responses."""

    @staticmethod
    def to_upsert_response(response: httpx.Response) -> UpsertDocumentsResponse:
        result = decode_response(response.content, UpsertDocumentsResponse)
        result.response_info = extract_response_info(response)
        return result

    @staticmethod
    def to_delete_response(response: httpx.Response) -> DeleteDocumentsResponse:
        result = decode_response(response.content, DeleteDocumentsResponse)
        result.response_info = extract_response_info(response)
        return result

    @staticmethod
    def to_search_response(response: httpx.Response) -> SearchDocumentsResponse:
        envelope = decode_response(response.content, _SearchDocumentsEnvelope)
        return SearchDocumentsResponse(
            matches=[Document(match) for match in envelope.matches],
            namespace=envelope.namespace,
            usage=envelope.usage,
            response_info=extract_response_info(response),
        )

    @staticmethod
    def to_fetch_response(response: httpx.Response) -> FetchDocumentsResponse:
        envelope = decode_response(response.content, _FetchDocumentsEnvelope)
        return FetchDocumentsResponse(
            documents={doc_id: Document(doc) for doc_id, doc in envelope.documents.items()},
            namespace=envelope.namespace,
            usage=envelope.usage,
            pagination=envelope.pagination,
            response_info=extract_response_info(response),
        )
