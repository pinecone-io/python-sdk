"""Backcompat shim providing the legacy ``async_req=True`` execution model
as a per-instance opt-in on top of the canonical methods.

Importing this module does **not** import ``multiprocessing``. The
``multiprocessing.pool.ThreadPool`` import is deferred to first use
inside :meth:`_LegacyAsyncPool.submit`.

:meta private:
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from pinecone.errors.exceptions import PineconeValueError

logger = logging.getLogger(__name__)

_METHODS_TO_WRAP = ("upsert", "query", "describe_index_stats", "list_paginated")

# Default thread-pool size when the caller uses ``async_req=True`` without
# having passed ``pool_threads=N``. Sized to match typical HTTP connection-pool
# defaults (urllib3=10, httpx max_keepalive=20) — meaningful parallelism for
# I/O-bound work without being aggressive enough to trip rate limits on a
# fresh client.
_DEFAULT_POOL_THREADS = 10


class _LegacyAsyncPool:
    """Owns a lazy ``multiprocessing.pool.ThreadPool``.

    The pool is constructed on the first :meth:`submit` call, not at
    instance construction. ``multiprocessing.pool`` is imported only
    inside that construction path.
    """

    def __init__(self, num_threads: int) -> None:
        if not isinstance(num_threads, int) or num_threads < 1:
            raise PineconeValueError(
                f"pool_threads must be a positive integer, got {num_threads!r}"
            )
        self._num_threads = num_threads
        self._pool: Any = None

    def submit(self, func: Callable[..., Any], kwargs: dict[str, Any]) -> Any:
        if self._pool is None:
            from multiprocessing.pool import ThreadPool

            logger.debug(
                "Constructing legacy async_req ThreadPool(num_threads=%d)",
                self._num_threads,
            )
            self._pool = ThreadPool(self._num_threads)
        return self._pool.apply_async(func, kwds=kwargs)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None


def install_async_req_support(index: Any, pool_threads: int) -> None:
    """Install per-instance ``async_req=True`` wrappers on *index*.

    Replaces the bound methods listed in ``_METHODS_TO_WRAP`` on this
    *instance only* (the class is not touched). Each wrapper pops
    ``async_req`` from kwargs and dispatches via a lazy
    :class:`_LegacyAsyncPool` when truthy.

    The original (canonical) bound methods are stashed on the
    instance under attribute names of the form ``_canonical_<name>``
    so wrappers can call through.
    """
    pool = _LegacyAsyncPool(pool_threads)
    index._legacy_async_pool = pool
    index._legacy_async_pool_threads = pool_threads

    for name in _METHODS_TO_WRAP:
        canonical = getattr(index, name)
        setattr(index, f"_canonical_{name}", canonical)
        setattr(index, name, _build_wrapper(name, canonical, pool))


def _build_wrapper(
    name: str,
    canonical: Callable[..., Any],
    pool: _LegacyAsyncPool,
) -> Callable[..., Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        async_req = kwargs.pop("async_req", False)
        if not async_req:
            return canonical(*args, **kwargs)
        if name == "upsert" and kwargs.get("batch_size") is not None:
            raise PineconeValueError("async_req is not supported when batch_size is provided.")
        return pool.submit(canonical, kwargs)

    wrapper.__name__ = name
    wrapper.__qualname__ = f"<async_req wrapper>.{name}"
    return wrapper
