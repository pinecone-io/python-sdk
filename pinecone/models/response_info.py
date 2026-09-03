"""The response metadata that rides along on data-plane results."""

from __future__ import annotations

from msgspec import Struct, field

from pinecone.models._mixin import StructDictMixin

__all__ = ["BatchResponseInfo", "ResponseInfo"]


class ResponseInfo(StructDictMixin, Struct, kw_only=True, gc=False):
    """What the server said about a data-plane call, beyond the result itself.

    Every data-plane response carries one as ``response_info``, and there are
    two reasons to reach for it. :attr:`request_id` is what a Pinecone support
    conversation asks you for. The two LSN properties are how you check
    read-your-writes: an upsert reports the log position it committed at, and
    a later read reports how far the index has caught up, so you can tell
    whether a query was allowed to see your write yet.

    Reads are eventually consistent, so a query issued immediately after a
    write can legitimately miss it. :meth:`is_reconciled` is the check that
    turns that from a guess into an answer.

    Attributes:
        raw_headers (dict[str, str]): All HTTP response headers, keys
            normalized to lowercase. Defaults to an empty dict. Use this
            to read any header the server returns, including headers not
            surfaced by the typed properties below. Prefer the typed
            properties when available — wire header names may change,
            but property semantics are stable.
        request_id (str | None): Identifier the server assigned this request.
            Quote it when reporting a problem. ``None`` when the header is
            absent.
        lsn_reconciled (int | None): How far the index has caught up, as a log
            position. ``None`` when the header is absent, so a ``None`` here
            means *unknown*, not position zero.
        lsn_committed (int | None): Log position this write landed at. ``None``
            when the header is absent — including on reads, which commit
            nothing.

    Examples:
        Write, then read back only once the index has caught up to the write:

        .. code-block:: python

            import time

            index = pc.index(name="product-search")
            written = index.documents.upsert(
                namespace="published",
                documents=[{"_id": "article-00042", "chunk_text": "Q3 revenue"}],
            )
            target = written.response_info.lsn_committed

            while True:
                fetched = index.documents.fetch(
                    namespace="published", ids=["article-00042"]
                )
                if target is None or fetched.response_info.is_reconciled(target):
                    break
                time.sleep(0.5)

    .. seealso::
       :class:`BatchResponseInfo` — the same durability signal for a bulk
       method, aggregated over the requests it made.
    """

    raw_headers: dict[str, str] = field(default_factory=dict)

    @property
    def request_id(self) -> str | None:
        """Identifier the server assigned this request.

        The one field worth logging on every call: it is what a Pinecone
        support conversation asks for, and the only handle on a single request
        after the fact.

        Returns:
            The request ID, or ``None`` when the server sent no such header.
        """
        return self.raw_headers.get("x-pinecone-request-id")

    @property
    def lsn_reconciled(self) -> int | None:
        """How far the index has caught up, as a log position.

        Compare it against a :attr:`lsn_committed` from an earlier write to
        find out whether that write is visible yet;
        :meth:`is_reconciled` does the comparison for you.

        Returns:
            The reconciled position, or ``None`` when the header is absent or
            not an integer. ``None`` means *unknown*, not position zero, so
            never treat it as "nothing reconciled".
        """
        return _parse_int(self.raw_headers.get("x-pinecone-lsn-reconciled"))

    @property
    def lsn_committed(self) -> int | None:
        """Log position the write on this response landed at.

        Keep it from an upsert or delete response and pass it to
        :meth:`is_reconciled` on a later read to check that the read saw the
        write.

        Returns:
            The committed position, or ``None`` when the header is absent or
            not an integer — including on a read, which commits nothing.
        """
        return _parse_int(self.raw_headers.get("x-pinecone-lsn-committed"))

    def is_reconciled(self, target: int) -> bool:
        """Has this response's index caught up to *target* yet?

        The read-your-writes check: pass the :attr:`lsn_committed` from an
        earlier write and get back whether the read that produced *this*
        response was able to see it.

        A ``False`` means "not yet, as of this response" — it is a reason to
        read again, not an error. ``False`` is also what you get when this
        response carried no reconciled position at all, since an unknown
        position cannot be shown to have caught up.

        Args:
            target (int): Log position to compare against, normally the
                :attr:`lsn_committed` of a prior upsert or delete. Guard
                against that being ``None`` before calling.

        Returns:
            ``True`` when :attr:`lsn_reconciled` is known and at least
            *target*; ``False`` otherwise.
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
    """The same durability signal as :class:`ResponseInfo`, for a whole batch.

    A bulk method makes many requests, each with its own headers, and reports
    one of these as its result's ``response_info``. It keeps the highest log
    position any successful sub-request reported, which is the position the
    whole batch is durable through.

    It has no ``raw_headers`` and no ``request_id``: there is no single
    response to point at. When a sub-batch failed, its own exception is on
    that failure's ``error`` attribute in the result's ``errors`` list.

    Attributes:
        lsn_reconciled (int | None): Highest reconciled position any
            successful sub-request reported, or ``None`` when none reported
            one.
        lsn_committed (int | None): Highest committed position any successful
            sub-request reported, or ``None`` when none reported one. Keep it
            to check a later read against.

    Examples:
        Keep the position a bulk write reached, to check a later read against:

        .. code-block:: python

            index = pc.index(name="product-search")
            result = index.documents.batch_upsert(
                namespace="published",
                documents=[
                    {"_id": f"article-{i:05d}", "chunk_text": f"Paragraph {i}"}
                    for i in range(500)
                ],
            )
            target = None
            if result.response_info is not None:
                target = result.response_info.lsn_committed
    """

    lsn_reconciled: int | None = None
    lsn_committed: int | None = None

    def is_reconciled(self, target: int) -> bool:
        """Has every successful sub-request caught up to *target*?

        The batch equivalent of :meth:`ResponseInfo.is_reconciled`, answered
        from the highest position any sub-request reported.
        """
        lsn = self.lsn_reconciled
        return lsn is not None and lsn >= target
