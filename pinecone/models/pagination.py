"""Lazy cursors over paged listing endpoints.

The canonical account of the mechanics lives on :class:`Paginator`; the
narrative version, with async examples, is :doc:`/guides/pagination`.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Awaitable, Callable, Generator
from typing import Generic, TypeVar

T = TypeVar("T")


class Page(Generic[T]):
    """One page of results from a paginated listing.

    You meet a ``Page`` only when walking a listing page by page with
    :meth:`Paginator.pages`; iterating a :class:`Paginator` directly yields the
    items and hides pages entirely. Not constructed directly.

    Attributes:
        items (list): The results on this page, in the order the server
            returned them.
        pagination_token (str | None): Opaque cursor naming the page *after*
            this one, or ``None`` when this is the last page. A page truncated
            by the paginator's ``limit`` also reports ``None`` here even though
            the server had more — resume from
            :attr:`Paginator.pagination_token` instead.

    Examples:
        >>> pages = pc.backup_schedules.iter_history(
        ...     schedule_id="e88f7273-42aa-47e9-af73-593827136867"
        ... ).pages()
        >>> first = next(pages)
        >>> len(first.items), first.has_more
        (1, True)

    .. seealso::
       :doc:`/guides/pagination` — walking, limiting, and resuming a listing.
    """

    def __init__(
        self,
        *,
        items: list[T],
        pagination_token: str | None,
    ) -> None:
        self.items = items
        self.pagination_token = pagination_token

    @property
    def has_more(self) -> bool:
        """Whether a page follows this one, i.e. whether it carries a token."""
        return self.pagination_token is not None

    def __repr__(self) -> str:
        return f"Page(items={self.items!r}, pagination_token={self.pagination_token!r})"


class Paginator(Generic[T]):
    """Lazy cursor over a listing endpoint that returns its results in pages.

    Returned by the SDK's sync list methods; not constructed directly. Nothing
    is requested until you iterate, and each page is fetched only once the
    previous one runs out, so a listing you stop reading early costs only the
    pages you actually consumed.

    Iterate it directly to get items and never think about pages. Use
    :meth:`pages` when the page boundary matters — checkpointing a long walk,
    or handing each response straight to a batch job. Use :meth:`to_list` when
    you want the whole listing in memory at once.

    A paginator is re-iterable, and every walk restarts from the token it was
    built with rather than continuing where the last one stopped.

    Args:
        fetch_page: Called with a pagination token (``None`` for the first
            page) and returns the matching :class:`Page`. Supplied by the list
            method that built this paginator.
        initial_token: Token to resume from, taken from an earlier walk's
            :attr:`pagination_token`; ``None`` starts at the first page.
        limit: Stop after this many items across all pages; ``None`` walks to
            the end of the listing.

    Examples:
        >>> for index in pc.indexes.list():
        ...     print(index.name, index.status.state)

    .. seealso::
       :doc:`/guides/pagination` — the same mechanics with async examples, plus
       the separate ``list_paginated`` interface for vector IDs.
    """

    def __init__(
        self,
        *,
        fetch_page: Callable[[str | None], Page[T]],
        initial_token: str | None = None,
        limit: int | None = None,
    ) -> None:
        self._fetch_page = fetch_page
        self._initial_token = initial_token
        self._limit = limit
        self._pagination_token: str | None = initial_token

    @property
    def pagination_token(self) -> str | None:
        """Cursor for the page after the one most recently fetched.

        Persist this to resume the walk in a later process — pass it back as
        the list method's ``pagination_token``. It reflects only what has been
        fetched so far: before you iterate it is whatever token the paginator
        was built with, and it is ``None`` once the walk reaches the last page.
        """
        return self._pagination_token

    def __iter__(self) -> Generator[T, None, None]:
        count = 0
        token: str | None = self._initial_token
        while True:
            page = self._fetch_page(token)
            self._pagination_token = page.pagination_token
            for item in page.items:
                if self._limit is not None and count >= self._limit:
                    return
                yield item
                count += 1
            if page.pagination_token is None:
                return
            token = page.pagination_token

    def pages(self) -> Generator[Page[T], None, None]:
        """Walk the listing one :class:`Page` at a time instead of item by item.

        When ``limit`` is set, yields whole pages until the remaining budget is
        smaller than the next page, then yields that page truncated and stops.
        The truncated page reports ``pagination_token=None``; to carry on
        later, resume from this paginator's own :attr:`pagination_token`, which
        still holds the server's cursor.

        Returns:
            :class:`~collections.abc.Generator` of :class:`Page`, each with an
            ``items`` list and a ``pagination_token`` naming the page after it.

        Examples:
            >>> runs = pc.backup_schedules.iter_history(
            ...     schedule_id="e88f7273-42aa-47e9-af73-593827136867"
            ... )
            >>> [(len(page.items), page.has_more) for page in runs.pages()]
            [(1, True), (1, False)]
        """
        count = 0
        token: str | None = self._initial_token
        while True:
            page = self._fetch_page(token)
            self._pagination_token = page.pagination_token
            if self._limit is not None:
                remaining = self._limit - count
                if remaining <= 0:
                    return
                if len(page.items) > remaining:
                    yield Page(items=page.items[:remaining], pagination_token=None)
                    return
                count += len(page.items)
            yield page
            if page.pagination_token is None:
                return
            token = page.pagination_token

    def to_list(self) -> list[T]:
        """Walk every remaining page and return all the items in one list.

        Every page is fetched before this returns and the whole listing is
        held in memory, so iterate the paginator instead when the listing is
        large or you may stop early.

        Returns:
            list of every item the walk produced, in server order.

        Examples:
            >>> runs = pc.backup_schedules.iter_history(
            ...     schedule_id="e88f7273-42aa-47e9-af73-593827136867"
            ... )
            >>> len(runs.to_list())
            2
        """
        return list(self)

    def __repr__(self) -> str:
        has_more = self._pagination_token is not None
        parts = [f"has_more={has_more!r}"]
        if self._limit is not None:
            parts.append(f"limit={self._limit!r}")
        return f"Paginator({', '.join(parts)})"


class AsyncPaginator(Generic[T]):
    """Lazy cursor over a paged listing, for use with ``async for``.

    What :class:`Paginator` is on :class:`~pinecone.Pinecone`, this is on
    :class:`~pinecone.AsyncPinecone`: returned by the async list methods, never
    constructed directly. The list method itself is not a coroutine — it hands
    back the paginator synchronously, and the awaiting happens as you walk it.

    Args:
        fetch_page: Awaitable called with a pagination token (``None`` for the
            first page), returning the matching :class:`Page`. Supplied by the
            list method that built this paginator.
        initial_token: Token to resume from, taken from an earlier walk's
            :attr:`pagination_token`; ``None`` starts at the first page.
        limit: Stop after this many items across all pages; ``None`` walks to
            the end of the listing.

    Examples:
        .. code-block:: python

            from pinecone import AsyncPinecone

            async with AsyncPinecone(api_key="your-api-key") as pc:
                async for index in pc.indexes.list():
                    print(index.name, index.status.state)

    .. seealso::
       :doc:`/guides/pagination` — walking, limiting, and resuming a listing.
    """

    def __init__(
        self,
        *,
        fetch_page: Callable[[str | None], Awaitable[Page[T]]],
        initial_token: str | None = None,
        limit: int | None = None,
    ) -> None:
        self._fetch_page = fetch_page
        self._initial_token = initial_token
        self._limit = limit
        self._pagination_token: str | None = initial_token

    @property
    def pagination_token(self) -> str | None:
        """Cursor for the page after the one most recently fetched.

        Persist this to resume the walk in a later process — pass it back as
        the list method's ``pagination_token``. It reflects only what has been
        fetched so far: before you iterate it is whatever token the paginator
        was built with, and it is ``None`` once the walk reaches the last page.
        """
        return self._pagination_token

    async def __aiter__(self) -> AsyncGenerator[T, None]:
        count = 0
        token: str | None = self._initial_token
        while True:
            page = await self._fetch_page(token)
            self._pagination_token = page.pagination_token
            for item in page.items:
                if self._limit is not None and count >= self._limit:
                    return
                yield item
                count += 1
            if page.pagination_token is None:
                return
            token = page.pagination_token

    async def pages(self) -> AsyncGenerator[Page[T], None]:
        """Walk the listing one :class:`Page` at a time instead of item by item.

        When ``limit`` is set, yields whole pages until the remaining budget is
        smaller than the next page, then yields that page truncated and stops.
        The truncated page reports ``pagination_token=None``; to carry on
        later, resume from this paginator's own :attr:`pagination_token`, which
        still holds the server's cursor.

        Returns:
            :class:`~collections.abc.AsyncGenerator` of :class:`Page`, each
            with an ``items`` list and a ``pagination_token`` naming the page
            after it.

        Examples:
            .. code-block:: python

                async with AsyncPinecone(api_key="your-api-key") as pc:
                    async for page in pc.indexes.list().pages():
                        print(len(page.items), page.has_more)
        """
        count = 0
        token: str | None = self._initial_token
        while True:
            page = await self._fetch_page(token)
            self._pagination_token = page.pagination_token
            if self._limit is not None:
                remaining = self._limit - count
                if remaining <= 0:
                    return
                if len(page.items) > remaining:
                    yield Page(items=page.items[:remaining], pagination_token=None)
                    return
                count += len(page.items)
            yield page
            if page.pagination_token is None:
                return
            token = page.pagination_token

    async def to_list(self) -> list[T]:
        """Walk every remaining page and return all the items in one list.

        Every page is fetched before this returns and the whole listing is
        held in memory, so iterate the paginator instead when the listing is
        large or you may stop early.

        Returns:
            list of every item the walk produced, in server order.

        Examples:
            .. code-block:: python

                async with AsyncPinecone(api_key="your-api-key") as pc:
                    indexes = await pc.indexes.list().to_list()
        """
        return [item async for item in self]

    def __repr__(self) -> str:
        has_more = self._pagination_token is not None
        parts = [f"has_more={has_more!r}"]
        if self._limit is not None:
            parts.append(f"limit={self._limit!r}")
        return f"AsyncPaginator({', '.join(parts)})"
