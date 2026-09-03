"""Backwards-compatibility shim for legacy ``PineconeGRPC``.

Provides a thin subclass of :class:`pinecone.Pinecone` that exposes
a legacy ``Index(name, host, **kwargs)`` factory returning a
:class:`GrpcIndex` (i.e. the same as
``pc.index(name, host, grpc=True)``). Preserved so pre-rewrite
callers using ``from pinecone.grpc import PineconeGRPC`` keep
working. New code should use::

    pc = Pinecone(api_key=...)
    idx = pc.index(name="my-index", grpc=True)

:meta private:
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pinecone._client import Pinecone
from pinecone.errors.exceptions import PineconeValueError

if TYPE_CHECKING:
    from pinecone.grpc import GrpcIndex


class PineconeGRPC(Pinecone):
    """Legacy gRPC client. Subclass of :class:`Pinecone`; data-plane
    calls via :meth:`Index` use gRPC instead of HTTP.

    :meta private:
    """

    def Index(self, name: str = "", host: str = "", **kwargs: Any) -> GrpcIndex:  # noqa: N802
        """Return a :class:`GrpcIndex` for a data plane connection.

        Legacy equivalent of ``pc.index(name=name, host=host, grpc=True)``,
        kept for callers using the ``PineconeGRPC().Index(...)`` constructor
        style.

        Args:
            name (str): Name of the index to connect to. The host is
                resolved via the control plane. Omit if ``host`` is given.
            host (str): Data plane host URL for the index, e.g. as returned
                by ``describe_index``. Skips the control-plane lookup. Omit
                if ``name`` is given.
            **kwargs: Accepted for compatibility with the REST ``Index()``
                signature. ``pool_threads`` is silently ignored; any other
                keyword argument raises ``TypeError``.

        Returns:
            :class:`GrpcIndex` connected to the resolved host.

        Raises:
            :exc:`PineconeValueError`: If neither ``name`` nor ``host`` is given.
            :exc:`TypeError`: If any keyword argument other than
                ``pool_threads`` is passed — the message lists the ones it did
                not recognize.

        Examples:

            .. code-block:: python

                from pinecone.grpc import PineconeGRPC

                pc = PineconeGRPC(api_key="YOUR_API_KEY")
                idx = pc.Index(host="article-search-abc123.svc.pinecone.io")
        """
        if not name and not host:
            raise PineconeValueError("Either name or host must be specified")
        kwargs.pop("pool_threads", None)
        if kwargs:
            raise TypeError(
                f"PineconeGRPC.Index() got unexpected keyword arguments: {sorted(kwargs)!r}"
            )
        return self.index(name=name, host=host, grpc=True)


__all__ = ["PineconeGRPC"]
