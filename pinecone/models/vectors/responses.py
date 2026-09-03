"""What the data-plane read and write operations hand back."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from msgspec import Struct

from pinecone.models._mixin import DictLikeStruct, StructDictMixin
from pinecone.models.batch import BatchError
from pinecone.models.response_info import ResponseInfo as ResponseInfo  # re-export
from pinecone.models.vectors.usage import Usage
from pinecone.models.vectors.vector import ScoredVector, Vector


class UpsertResponse(DictLikeStruct, Struct, rename="camel", kw_only=True, gc=False):
    """What an upsert wrote, and — for a batched upsert — what it failed to write.

    Which fields carry information depends on how you called upsert. Without
    ``batch_size`` the client sends one request, so ``upserted_count`` is the whole
    answer and every batch counter is ``0``. With ``batch_size`` the client splits the
    vectors into requests and sends them one at a time; a later request can fail after
    earlier ones succeeded, so the batch counters and ``errors`` describe a partial
    success and ``upserted_count`` covers only the batches that landed.

    Attributes:
        upserted_count (int): Vectors the server accepted. Equals ``total_item_count``
            when every batch succeeded, and for a non-batched call.
        response_info (ResponseInfo | None): HTTP response metadata (request ID, LSN
            values), or ``None`` if not populated.
        total_item_count (int): Vectors you submitted, across every batch. ``0`` for a
            non-batched call.
        failed_item_count (int): Vectors that were in a batch that failed. Not all of
            them were necessarily rejected individually — the batch is the unit.
        total_batch_count (int): Batches the client sent. ``0`` for a non-batched call.
        successful_batch_count (int): Batches the server accepted.
        failed_batch_count (int): Batches that failed.
        errors (list[BatchError]): One entry per failed batch, carrying the underlying
            error and the items that batch held. Empty when nothing failed.

    Examples:
        A single upsert reports one number.

        .. code-block:: python

            response = idx.upsert(vectors=[("article-101", [0.12, 0.34, 0.56])])
            print(response.upserted_count)

        A batched upsert can partly succeed, so check :attr:`has_errors` before treating
        ``upserted_count`` as the full count. :attr:`failed_items` flattens the items from
        every failed batch back into a list you can resubmit.

        .. code-block:: python

            response = idx.upsert(vectors=vectors, batch_size=100)
            if response.has_errors:
                print(response.upserted_count, "of", response.total_item_count)
                retry = idx.upsert(vectors=response.failed_items, batch_size=100)

    .. seealso::
       :doc:`/guides/performance` — choosing a ``batch_size``.
    """

    upserted_count: int
    response_info: ResponseInfo | None = None
    total_item_count: int = 0
    failed_item_count: int = 0
    total_batch_count: int = 0
    successful_batch_count: int = 0
    failed_batch_count: int = 0
    errors: list[BatchError] = []

    @property
    def _response_info(self) -> ResponseInfo | None:
        return self.response_info

    @property
    def has_errors(self) -> bool:
        """``True`` if any batch failed, so ``upserted_count`` is a partial count."""
        return len(self.errors) > 0

    @property
    def error_count(self) -> int:
        """Alias for :attr:`failed_item_count`, spelled as ``BatchResult`` spells it."""
        return self.failed_item_count

    @property
    def success_count(self) -> int:
        """Alias for :attr:`upserted_count`, spelled as ``BatchResult`` spells it."""
        return self.upserted_count

    @property
    def successful_item_count(self) -> int:
        """Alias for :attr:`upserted_count`, spelled as ``BatchResult`` spells it."""
        return self.upserted_count

    @property
    def failed_items(self) -> list[dict[str, Any]]:
        """Every item from every failed batch, flattened into one list you can resubmit.

        Empty when :attr:`has_errors` is ``False``. Items that were in a successful batch
        are never included, so passing this straight back to ``upsert`` retries only the
        writes that did not land.
        """
        items: list[dict[str, Any]] = []
        for error in self.errors:
            items.extend(error.items)
        return items

    def __repr__(self) -> str:
        if not self.has_errors and self.total_batch_count == 0:
            return f"UpsertResponse(upserted_count={self.upserted_count})"
        status = "PARTIAL FAILURE" if self.has_errors else "SUCCESS"
        return (
            f"UpsertResponse({status}: "
            f"{self.upserted_count}/{self.total_item_count} items, "
            f"{self.successful_batch_count}/{self.total_batch_count} batches)"
        )


class QueryResponse(DictLikeStruct, Struct, rename="camel", kw_only=True, gc=False):
    """The ranked matches a query found.

    Almost everything you want is in ``matches``: a list of
    :class:`~pinecone.models.vectors.vector.ScoredVector`, already ordered so
    ``matches[0]`` is the closest hit. Read each match through ``.id``, ``.score``,
    ``.values`` and ``.metadata``. The last two come back empty or ``None`` unless the
    query passed ``include_values=True`` / ``include_metadata=True``, so a missing value
    there is far more often an unset flag than an empty stored vector.

    A query that matched nothing returns an empty ``matches`` rather than raising, so
    check the length instead of catching an exception.

    Attributes:
        matches (list[ScoredVector]): The hits, ordered from most to least similar.
        namespace (str): The namespace that was queried; ``""`` for the default namespace.
        usage (Usage | None): Read units this query consumed, or ``None`` if not reported.
        response_info (ResponseInfo | None): HTTP response metadata (request ID, LSN
            values), or ``None`` if not populated.

    Examples:
        .. code-block:: python

            response = idx.query(
                top_k=5,
                vector=[0.012, -0.087, 0.153],
                namespace="articles-en",
                include_metadata=True,
            )
            for match in response.matches:
                print(match.id, match.score, match.metadata)

        No match is not an error:

        .. code-block:: python

            if not response.matches:
                print("nothing above the cutoff in", response.namespace)

    .. seealso::
       :class:`~pinecone.models.vectors.search.SearchRecordsResponse` — what ``search``
       returns instead, where the hits sit under ``result.hits`` and carry ``fields``
       rather than ``values`` and ``metadata``.
    """

    matches: list[ScoredVector] = []
    namespace: str | None = ""
    usage: Usage | None = None
    response_info: ResponseInfo | None = None

    @property
    def _response_info(self) -> ResponseInfo | None:
        return self.response_info

    def __post_init__(self) -> None:
        """Read a null ``namespace`` back as ``""``, so the default namespace has one spelling."""
        if self.namespace is None:
            self.namespace = ""


class FetchResponse(DictLikeStruct, Struct, rename="camel", kw_only=True, gc=False):
    """The vectors a fetch retrieved, keyed by ID.

    ``vectors`` is a dict, not a list, so look a vector up by the ID you asked for. An ID
    that does not exist in the namespace is simply absent from the dict — fetching a
    missing ID is not an error — so use ``.get()`` or test membership rather than
    indexing blind. Unlike a query, a fetch always returns values and metadata; there is
    nothing to opt into.

    Attributes:
        vectors (dict[str, Vector]): Vector ID to :class:`~pinecone.models.vectors.vector.Vector`,
            for the requested IDs that exist.
        namespace (str): The namespace the vectors were fetched from.
        usage (Usage | None): Read units this fetch consumed, or ``None`` if not reported.
        response_info (ResponseInfo | None): HTTP response metadata (request ID, LSN
            values), or ``None`` if not populated.

    Examples:
        .. code-block:: python

            wanted = ["article-101", "article-102"]
            response = idx.fetch(ids=wanted, namespace="articles-en")
            for vector_id, vector in response.vectors.items():
                print(vector_id, len(vector.values), vector.metadata)
            print("not stored:", [vid for vid in wanted if vid not in response.vectors])

    .. seealso::
       :class:`FetchByMetadataResponse` — what you get when you select the vectors by
       metadata filter rather than by ID, which can span more than one page.
    """

    vectors: dict[str, Vector] = {}
    namespace: str = ""
    usage: Usage | None = None
    response_info: ResponseInfo | None = None

    @property
    def _response_info(self) -> ResponseInfo | None:
        return self.response_info


class FetchByMetadataResponse(DictLikeStruct, Struct, rename="camel", kw_only=True, gc=False):
    """One page of the vectors matching a metadata filter, keyed by ID.

    Same shape as :class:`FetchResponse` plus ``pagination``: a filter can match more
    vectors than one response carries, so this is a page and not the whole answer. Keep
    calling with the token until ``pagination`` is ``None``.

    Attributes:
        vectors (dict[str, Vector]): Vector ID to :class:`~pinecone.models.vectors.vector.Vector`
            for the matches on this page.
        namespace (str): The namespace the vectors were fetched from.
        usage (Usage | None): Read units this page consumed, or ``None`` if not reported.
        pagination (Pagination | None): Token to pass as ``pagination_token`` for the next
            page, or ``None`` when this is the last page.
        response_info (ResponseInfo | None): HTTP response metadata (request ID, LSN
            values), or ``None`` if not populated.

    Examples:
        .. code-block:: python

            token = None
            while True:
                page = idx.fetch_by_metadata(
                    filter={"lang": "en"}, namespace="articles-en", pagination_token=token
                )
                for vector_id, vector in page.vectors.items():
                    print(vector_id, vector.metadata)
                if page.pagination is None:
                    break
                token = page.pagination.next

    .. seealso::
       :doc:`/guides/pagination` — the paging pattern used across the SDK.
    """

    vectors: dict[str, Vector] = {}
    namespace: str = ""
    usage: Usage | None = None
    pagination: Pagination | None = None
    response_info: ResponseInfo | None = None

    @property
    def _response_info(self) -> ResponseInfo | None:
        return self.response_info


class NamespaceSummary(StructDictMixin, Struct, rename="camel", kw_only=True, gc=False):
    """The per-namespace entry in :class:`DescribeIndexStatsResponse`.

    Attributes:
        vector_count (int): Vectors in this namespace.
    """

    vector_count: int = 0


class DescribeIndexStatsResponse(StructDictMixin, Struct, rename="camel", kw_only=True, gc=False):
    """How much is in an index, and how it is configured, as of this call.

    The usual reason to call ``describe_index_stats`` is to find out which namespaces
    exist and how many vectors each holds — ``namespaces`` answers both, and its keys are
    the namespace names you can pass to a query. Counts are eventually consistent, so a
    vector you just upserted may not be reflected yet.

    Attributes:
        namespaces (dict[str, NamespaceSummary]): Namespace name to its
            :class:`NamespaceSummary`. The default namespace appears under ``""``.
        dimension (int | None): Length of the dense vectors this index stores, or ``None``
            for an index with no dense field.
        index_fullness (float): How full the index is, from ``0.0`` to ``1.0``.
        total_vector_count (int): Vectors across every namespace.
        metric (str | None): The similarity function used when ranking, e.g. ``"cosine"``,
            or ``None`` if not reported.
        vector_type (str | None): ``"dense"`` or ``"sparse"``, or ``None`` if not reported.
        memory_fullness (float | None): How full memory is, or ``None`` if not reported.
        storage_fullness (float | None): How full storage is, or ``None`` if not reported.
        response_info (ResponseInfo | None): HTTP response metadata (request ID, LSN
            values), or ``None`` if not populated.

    Examples:
        .. code-block:: python

            stats = idx.describe_index_stats()
            print(stats.total_vector_count, stats.dimension)
            for name, summary in stats.namespaces.items():
                print(name or "(default)", summary.vector_count)
    """

    namespaces: dict[str, NamespaceSummary] = {}
    dimension: int | None = None
    index_fullness: float = 0.0
    total_vector_count: int = 0
    metric: str | None = None
    vector_type: str | None = None
    memory_fullness: float | None = None
    storage_fullness: float | None = None
    response_info: ResponseInfo | None = None

    @property
    def _response_info(self) -> ResponseInfo | None:
        return self.response_info

    def __repr__(self) -> str:
        parts = []
        if self.dimension is not None:
            parts.append(f"dimension={self.dimension!r}")
        parts.append(f"total_vector_count={self.total_vector_count!r}")
        if self.metric is not None:
            parts.append(f"metric={self.metric!r}")
        parts.append(f"namespaces={len(self.namespaces)!r}")
        return f"DescribeIndexStatsResponse({', '.join(parts)})"

    def __getitem__(self, key: str) -> Any:
        """Read a field by name, so ``stats["dimension"]`` works as well as ``stats.dimension``.

        Raises:
            KeyError: If *key* is not one of this model's fields.
        """
        if key not in self.__struct_fields__:
            raise KeyError(key)
        return getattr(self, key)

    def __contains__(self, key: object) -> bool:
        """Report whether *key* names a field, so ``"dimension" in stats`` works."""
        return key in self.__struct_fields__


class Pagination(StructDictMixin, Struct, kw_only=True, gc=False):
    """The cursor that carries you from one page of results to the next.

    Appears as ``pagination`` on every paged response. A ``None`` on the response, or a
    ``None`` in ``next``, both mean the page you are holding is the last one — that is
    the loop's exit condition, not an error.

    Attributes:
        next (str | None): Opaque token to pass back as the next call's
            ``pagination_token``, or ``None`` when there is no further page. Treat it as
            opaque: it is not an ID, an offset, or anything you can construct yourself.

    .. seealso::
       :doc:`/guides/pagination` — the paging loop, and the paginated helpers that run it
       for you.
    """

    next: str | None = None


class ListItem(StructDictMixin, Struct, kw_only=True, gc=False):
    """One entry in :attr:`ListResponse.vectors` — an ID and nothing else.

    ``list`` walks the IDs in a namespace without reading the vectors themselves, so
    there are no values or metadata here. Fetch the IDs you care about to get those.

    Attributes:
        id (str | None): The vector identifier, or ``None`` if the entry carried none.
    """

    id: str | None = None


class ListResponse(StructDictMixin, Struct, rename="camel", kw_only=True, gc=False):
    """One page of vector IDs from a namespace.

    Each element of ``vectors`` is a :class:`ListItem` carrying only an ``id``. The
    response is also directly iterable and sized, so ``for item in response`` and
    ``len(response)`` walk that same page.

    Attributes:
        vectors (list[ListItem]): The ID entries on this page.
        pagination (Pagination | None): Token for the next page, or ``None`` when this is
            the last page.
        namespace (str): The namespace the IDs were listed from.
        usage (Usage | None): Read units this page consumed, or ``None`` if not reported.
        response_info (ResponseInfo | None): HTTP response metadata (request ID, LSN
            values), or ``None`` if not populated.

    Examples:
        .. code-block:: python

            page = idx.list_paginated(prefix="article-", namespace="articles-en")
            for item in page.vectors:
                print(item.id)

    .. seealso::
       :doc:`/guides/pagination` — and ``Index.list``, which yields every page for you.
    """

    vectors: list[ListItem] = []
    pagination: Pagination | None = None
    namespace: str = ""
    usage: Usage | None = None
    response_info: ResponseInfo | None = None

    @property
    def _response_info(self) -> ResponseInfo | None:
        return self.response_info

    def __getitem__(self, key: int | str) -> Any:
        """Index into the page's items, or read a field by name.

        Args:
            key (int | str): An integer position in ``vectors``, or the name of a field
                on this response.

        Returns:
            The :class:`ListItem` at that position, or the named field's value.

        Raises:
            KeyError: If a string *key* does not name a field.
            IndexError: If an integer *key* is past the end of this page.
        """
        if isinstance(key, int):
            return self.vectors[key]
        if key not in self.__struct_fields__:
            raise KeyError(key)
        return getattr(self, key)

    def __contains__(self, key: object) -> bool:
        """Report field-name membership for a string, and item membership otherwise."""
        if isinstance(key, str):
            return key in self.__struct_fields__
        return key in self.vectors

    def __len__(self) -> int:
        return len(self.vectors)

    def __iter__(self) -> Iterator[ListItem]:  # type: ignore[override]
        return iter(self.vectors)


class UpsertRecordsResponse(StructDictMixin, Struct, kw_only=True, gc=False):
    """Acknowledgement that ``upsert_records`` was accepted.

    ``upsert_records`` embeds text server-side and the response body carries no counts, so
    ``record_count`` is what the client sent rather than what the server confirmed. Read
    it as "the request went out with this many records", and call
    ``describe_index_stats`` if you need a count the index vouches for.

    Attributes:
        record_count (int): Records the client submitted. A client-side count.
        response_info (ResponseInfo | None): HTTP response metadata (request ID, LSN
            values), or ``None`` if not populated.
    """

    record_count: int
    response_info: ResponseInfo | None = None

    @property
    def _response_info(self) -> ResponseInfo | None:
        return self.response_info

    def __getitem__(self, key: str) -> Any:
        """Read a field by name, so ``response["record_count"]`` works too.

        Raises:
            KeyError: If *key* is not one of this model's fields.
        """
        if key not in self.__struct_fields__:
            raise KeyError(key)
        return getattr(self, key)

    def __contains__(self, key: object) -> bool:
        """Report whether *key* names a field on this response."""
        return key in self.__struct_fields__


class UpdateResponse(DictLikeStruct, Struct, rename="camel", kw_only=True, gc=False):
    """Acknowledgement that an update was accepted, and how many vectors it matched.

    Attributes:
        matched_records (int | None): Vectors the update matched, or ``None`` when no
            count was reported. A by-filter update is the case that reports one; pass
            ``dry_run=True`` to get the count without applying the change. Updates apply
            asynchronously, so a count here is a point-in-time figure rather than a
            guarantee that the writes have landed.
        response_info (ResponseInfo | None): HTTP response metadata (request ID, LSN
            values), or ``None`` if not populated.
    """

    matched_records: int | None = None
    response_info: ResponseInfo | None = None

    @property
    def _response_info(self) -> ResponseInfo | None:
        return self.response_info
