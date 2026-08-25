"""Shared HTTP response metadata carrier."""

from __future__ import annotations

from msgspec import Struct, field

from pinecone.models._mixin import StructDictMixin

__all__ = ["BatchResponseInfo", "ResponseInfo"]


class ResponseInfo(StructDictMixin, Struct, kw_only=True, gc=False):
    """HTTP response metadata carrier.

    Stores every HTTP response header returned by the server (keys
    lowercased) plus typed convenience properties for the headers the
    SDK promotes to first-class fields.

    Attributes:
        raw_headers (dict[str, str]): All HTTP response headers, keys
            normalized to lowercase. Defaults to an empty dict. Use this
            to read any header the server returns, including headers not
            surfaced by the typed properties below. Prefer the typed
            properties when available — wire header names may change,
            but property semantics are stable.
        request_id (str | None): Server-assigned request identifier read
            from ``x-pinecone-request-id``, or ``None`` if not present.
        lsn_reconciled (int | None): Log sequence number indicating how
            far the index has reconciled, parsed from
            ``x-pinecone-lsn-reconciled``. ``None`` when absent or when
            the header value is not a valid integer.
        lsn_committed (int | None): Log sequence number of the last
            committed write, parsed from ``x-pinecone-lsn-committed``.
            ``None`` when absent or non-integer.
    """

    raw_headers: dict[str, str] = field(default_factory=dict)

    @property
    def request_id(self) -> str | None:
        """Server-assigned request identifier from ``x-pinecone-request-id``.

        Returns:
            :class:`str` with the request ID, or ``None`` when the header
            is absent.
        """
        return self.raw_headers.get("x-pinecone-request-id")

    @property
    def lsn_reconciled(self) -> int | None:
        """Log sequence number indicating how far the index has reconciled.

        Parsed from the ``x-pinecone-lsn-reconciled`` response header.

        Returns:
            :class:`int` LSN, or ``None`` when the header is absent or
            its value is not a valid integer.
        """
        return _parse_int(self.raw_headers.get("x-pinecone-lsn-reconciled"))

    @property
    def lsn_committed(self) -> int | None:
        """Log sequence number of the last committed write.

        Parsed from the ``x-pinecone-lsn-committed`` response header.

        Returns:
            :class:`int` LSN, or ``None`` when the header is absent or
            its value is not a valid integer.
        """
        return _parse_int(self.raw_headers.get("x-pinecone-lsn-committed"))

    def is_reconciled(self, target: int) -> bool:
        """Return ``True`` when the reconciled LSN meets or exceeds *target*.

        Use this for read-your-writes consistency checks: pass the LSN from
        a previous write response to verify that the index has caught up to
        that write before issuing a query.

        Args:
            target (int): The LSN threshold to check against. Typically the
                :attr:`lsn_committed` value returned by a prior upsert or
                delete response.

        Returns:
            ``True`` if :attr:`lsn_reconciled` is not ``None`` and is
            greater than or equal to *target*; ``False`` otherwise.

        Examples:
            .. code-block:: python

                from pinecone import Pinecone

                pc = Pinecone(api_key="your-api-key")
                index = pc.index(host="product-search.svc.pinecone.io")
                upsert_resp = index.upsert_records(
                    namespace="electronics",
                    records=[{"id": "prod-42", "_text": "wireless headphones"}],
                )
                committed_lsn = upsert_resp.response_info.lsn_committed
                query_resp = index.search(
                    namespace="electronics",
                    inputs={"text": "headphones"},
                )
                query_resp.result.response_info.is_reconciled(committed_lsn)
        """
        lsn = self.lsn_reconciled
        return lsn is not None and lsn >= target


def _parse_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


class BatchResponseInfo(StructDictMixin, Struct, kw_only=True, gc=False):
    """Aggregate durability signal across a multi-request batch operation.

    A batch operation fans out into N underlying HTTP requests, each
    with its own response headers. ``BatchResponseInfo`` collapses the
    reconciliation signal across those requests into a single object
    that mirrors the read-your-writes API surface of :class:`ResponseInfo`.

    Does **not** carry ``raw_headers`` or ``request_id`` — there is no
    single source HTTP response to point at. Individual sub-request
    diagnostics are available via :attr:`BatchError.error` for failed
    batches.

    Attributes:
        lsn_reconciled (int | None): Maximum ``lsn_reconciled`` observed
            across successful sub-batches, or ``None`` when no
            successful batch reported this header. Use
            :meth:`is_reconciled` for durability checks.
        lsn_committed (int | None): Maximum ``lsn_committed`` observed
            across successful sub-batches, or ``None`` when no
            successful batch reported this header.

    Examples:
        .. code-block:: python

            from pinecone import Pinecone

            pc = Pinecone(api_key="your-api-key")
            index = pc.index(name="articles-en")
            documents = [
                {"_id": f"article-{i:05d}", "content": f"Article {i}"}
                for i in range(500)
            ]
            result = index.documents.batch_upsert(
                namespace="articles-en",
                documents=documents,
            )
            if result.response_info is not None:
                target_lsn = result.response_info.lsn_committed
                if result.response_info.is_reconciled(target_lsn):
                    pass  # all writes durable through target_lsn
    """

    lsn_reconciled: int | None = None
    lsn_committed: int | None = None

    def is_reconciled(self, target: int) -> bool:
        """Return True when the aggregate reconciled LSN meets or exceeds *target*."""
        lsn = self.lsn_reconciled
        return lsn is not None and lsn >= target
