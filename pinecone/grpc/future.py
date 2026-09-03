"""PineconeFuture — a thin wrapper around concurrent.futures.Future.

Provides SDK-specific timeout defaults and exception translation so that
callers get :class:`~pinecone.errors.PineconeTimeoutError` instead of the
stdlib ``TimeoutError`` when a result is not ready in time.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future
from concurrent.futures import TimeoutError as _FuturesTimeoutError
from typing import Any, TypeVar

from pinecone.errors.exceptions import PineconeTimeoutError

_T = TypeVar("_T")

_DEFAULT_TIMEOUT: float = 5.0


class PineconeFuture(Future["_T"]):
    """A handle on a ``GrpcIndex.*_async()`` call that is already in flight.

    The call was handed to a background thread and the method returned
    immediately. Issue as many as you want, then collect them: call
    :meth:`result` to block for one, or pass the whole batch to
    :func:`concurrent.futures.as_completed` or
    :func:`concurrent.futures.wait`, both of which this class supports. Nothing
    is cancelled if you never collect a future — the request still reaches the
    server.

    This is threads, not :keyword:`await`. Nothing here is awaitable, and the
    surrounding function does not need to be ``async``. If your code is already
    running under asyncio,
    :class:`~pinecone.async_client.async_index.AsyncIndex` is the client you
    want instead: its methods are coroutines, so a pending request yields to
    the event loop rather than parking a worker thread.

    ``result()`` and ``exception()`` default to a **5 second** wait, short
    enough that an unfinished call raises rather than hanging; pass an explicit
    ``timeout=`` for anything slower, or ``timeout=None`` to block until the
    call settles.

    Examples:

        .. code-block:: python

            from pinecone.grpc import GrpcIndex

            idx = GrpcIndex(host="article-search-abc123.svc.pinecone.io", api_key="...")
            future = idx.upsert_async(vectors=[("article-101", [0.012, -0.087, 0.153])])
            print(future.result().upserted_count)

        Issuing several at once is the reason to prefer these over the blocking
        methods — the requests overlap instead of queueing:

        .. code-block:: python

            from concurrent.futures import as_completed

            futures = [
                idx.upsert_async(vectors=[("article-101", [0.012, -0.087, 0.153])]),
                idx.upsert_async(vectors=[("article-102", [0.045, 0.021, -0.064])]),
            ]
            for future in as_completed(futures):
                print(future.result().upserted_count)

    .. seealso::
       :class:`~pinecone.async_client.async_index.AsyncIndex` — the asyncio
       client, for code that awaits rather than joining threads. See
       :doc:`/guides/sync-vs-async`.
    """

    def __init__(self, underlying: Future[_T]) -> None:
        # Do NOT call super().__init__() — we delegate everything to the
        # underlying future.  We *do* need the internal state that Future
        # expects however, so we initialise ourselves as a bare Future and
        # then wire up callbacks so our own state mirrors the underlying one.
        super().__init__()
        self._underlying = underlying

        # Mirror terminal state from the underlying future into *self* so
        # that concurrent.futures infrastructure (as_completed / wait) which
        # inspects our internal condition/state sees the correct values.
        self._underlying.add_done_callback(self._propagate_state)

    # ------------------------------------------------------------------
    # State propagation
    # ------------------------------------------------------------------

    def _propagate_state(self, _fut: Future[_T]) -> None:
        """Copy the terminal state of the underlying future into *self*."""
        if self._underlying.cancelled():
            # Mark ourselves cancelled so wait/as_completed see it.
            super().cancel()
            super().set_running_or_notify_cancel()
        elif self._underlying.exception() is not None:
            try:
                super().set_exception(self._underlying.exception())
            except Exception:
                pass  # already in terminal state
        else:
            try:
                super().set_result(self._underlying.result(timeout=0))
            except Exception:
                pass  # already in terminal state

    # ------------------------------------------------------------------
    # Public interface — delegates to the underlying future
    # ------------------------------------------------------------------

    def result(self, timeout: float | None = _DEFAULT_TIMEOUT) -> _T:
        """Block until the call settles, then return what it returned.

        Args:
            timeout: Maximum seconds to wait, defaulting to 5.0. Pass ``None``
                to block until the call settles, however long that takes.

        Returns:
            Whatever the underlying ``GrpcIndex`` method would have returned
            had you called it directly — an
            :class:`~pinecone.models.vectors.responses.UpsertResponse` from ``upsert_async``, a
            :class:`~pinecone.models.vectors.responses.QueryResponse` from ``query_async``, and so
            on.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeTimeoutError`: If *timeout* elapses
                first. The call is still in flight and may yet reach the
                server; call :meth:`result` again to keep waiting.

        Examples:

            .. code-block:: python

                future = idx.upsert_async(vectors=[("article-101", [0.012, -0.087, 0.153])])
                print(future.result().upserted_count)

            A large batch usually needs more than the 5-second default:

            .. code-block:: python

                future = idx.upsert_async(vectors=large_batch)
                result = future.result(timeout=30.0)
        """
        try:
            return self._underlying.result(timeout=timeout)
        except _FuturesTimeoutError:
            raise PineconeTimeoutError("deadline exceeded") from None

    def exception(self, timeout: float | None = _DEFAULT_TIMEOUT) -> BaseException | None:
        """Block until the call settles, then return how it failed, or ``None``.

        Use this to inspect a failure without it propagating, where
        :meth:`result` would re-raise it.

        Args:
            timeout: Maximum seconds to wait, defaulting to 5.0. Pass ``None``
                to block until the call settles.

        Returns:
            The exception the call raised, or ``None`` if it succeeded.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeTimeoutError`: If *timeout* elapses
                before the call settles. This is the wait timing out, not the
                call failing.

        Examples:

            .. code-block:: python

                future = idx.upsert_async(vectors=[("article-101", [0.012, -0.087, 0.153])])
                error = future.exception(timeout=30.0)
                if error is not None:
                    print("upsert failed:", error)
        """
        try:
            return self._underlying.exception(timeout=timeout)
        except _FuturesTimeoutError:
            raise PineconeTimeoutError("deadline exceeded") from None

    def cancel(self) -> bool:
        """Try to cancel the call before a worker thread picks it up.

        Returns ``True`` only if the call had not started yet. Once it is
        running there is no way to recall it — you get ``False`` and the
        request still reaches the server, so treat a ``False`` here as "the
        write may land" rather than "nothing happened".

        Examples:

            .. code-block:: python

                future = idx.upsert_async(vectors=[("article-101", [0.012, -0.087, 0.153])])
                if not future.cancel():
                    future.result(timeout=30.0)
        """
        return self._underlying.cancel()

    def cancelled(self) -> bool:
        """Return ``True`` if the call was successfully cancelled."""
        return self._underlying.cancelled()

    def done(self) -> bool:
        """Return ``True`` if the call has completed or was cancelled."""
        return self._underlying.done()

    def running(self) -> bool:
        """Return ``True`` if the call is currently being executed."""
        return self._underlying.running()

    def add_done_callback(self, fn: Callable[..., Any]) -> None:
        """Run *fn* once the call settles, instead of blocking on it.

        *fn* receives this future as its only argument, and runs on the worker
        thread that finished the call — so keep it short, and do not call
        :meth:`result` on a *different* pending future from inside it. Adding a
        callback to a future that has already settled runs *fn* immediately, on
        the calling thread.

        Examples:

            .. code-block:: python

                def log_result(future):
                    print("upserted", future.result().upserted_count)

                idx.upsert_async(
                    vectors=[("article-101", [0.012, -0.087, 0.153])]
                ).add_done_callback(log_result)
        """
        self._underlying.add_done_callback(lambda _underlying: fn(self))
