"""Per-call read and write unit consumption, as reported on data-plane responses."""

from __future__ import annotations

from msgspec import Struct


class Usage(Struct, rename="camel", kw_only=True, gc=False):
    """What one data-plane call cost, in read and write units.

    Reachable as ``usage`` on the read responses — :class:`QueryResponse
    <pinecone.models.vectors.responses.QueryResponse>`, :class:`FetchResponse
    <pinecone.models.vectors.responses.FetchResponse>`, :class:`ListResponse
    <pinecone.models.vectors.responses.ListResponse>` — where it is the per-call figure,
    not a running total. Only one side is normally populated: a read reports
    ``read_units`` and leaves ``write_units`` as ``None``.

    Attributes:
        read_units (int | None): Read units this call consumed, or ``None`` when the
            operation does not report them.
        write_units (int | None): Write units this call consumed, or ``None`` when the
            operation does not report them.
    """

    read_units: int | None = None
    write_units: int | None = None
