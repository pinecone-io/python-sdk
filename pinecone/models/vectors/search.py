"""What ``Index.search`` sends and gets back: hits, usage, and the typed request keys."""

from __future__ import annotations

from typing import Any, Literal, TypedDict, overload

from msgspec import Struct

from pinecone.models._mixin import DictLikeStruct, StructDictMixin
from pinecone.models.response_info import ResponseInfo

__all__ = [
    "Hit",
    "RerankConfig",
    "SearchInputs",
    "SearchQuery",
    "SearchQueryVector",
    "SearchRecordsResponse",
    "SearchRerank",
    "SearchResult",
    "SearchUsage",
]


class _RerankConfigRequired(TypedDict):
    """Required fields of :class:`RerankConfig`."""

    model: str
    rank_fields: list[str]


class RerankConfig(_RerankConfigRequired, total=False):
    """The ``rerank`` argument of :meth:`~pinecone.Index.search`, as a typed dict.

    Reranking runs a second, slower model over the hits the search already found and
    reorders them, which usually buys precision at the top of the list at the cost of
    latency. Pass this as a plain dict — it is a :class:`~typing.TypedDict`, so your
    editor and type checker see the keys, but there is nothing to instantiate.

    ``model`` and ``rank_fields`` are required; the rest are optional.

    Attributes:
        model (str): The reranking model to use, e.g. ``"bge-reranker-v2-m3"``. The model
            you request may not be the model that serves the request; the response reports
            which one did.
        rank_fields (list[str]): The record fields the reranker reads, e.g. ``["chunk"]``.
            These must be fields the search returns.
        top_n (int): How many hits to keep after reranking. Defaults to ``top_k``, so set
            it lower to have the reranker narrow a wider candidate set.
        parameters (dict[str, Any]): Extra parameters the chosen model accepts.
        query (str): Text to rerank against, when it should differ from the search query —
            omit it and the search inputs are used.

    Examples:
        .. code-block:: python

            response = idx.search(
                namespace="articles-en",
                top_k=20,
                inputs={"text": "how do sparse indexes score matches"},
                rerank={"model": "bge-reranker-v2-m3", "rank_fields": ["chunk"], "top_n": 5},
            )
    """

    top_n: int
    parameters: dict[str, Any]
    query: str


class _SearchInputsRequired(TypedDict):
    """Required fields of :class:`SearchInputs`."""

    text: str


class SearchInputs(_SearchInputsRequired, total=False):
    """The ``inputs`` argument of :meth:`~pinecone.Index.search`, as a typed dict.

    Use this when you want the index to embed your query for you rather than sending a
    vector — the path available on indexes with integrated inference. Like
    :class:`RerankConfig` it is a :class:`~typing.TypedDict`, so pass a plain dict and let
    your editor check the keys.

    Attributes:
        text (str): The query text to embed server-side, e.g.
            ``"how do sparse indexes score matches"``.

    Examples:
        .. code-block:: python

            response = idx.search(
                namespace="articles-en",
                top_k=5,
                inputs={"text": "how do sparse indexes score matches"},
            )
    """


class SearchUsage(StructDictMixin, Struct, kw_only=True):
    """What one search cost, broken out by the work it did.

    Which fields are populated tells you which stages ran: ``embed_total_tokens`` appears
    only when the index embedded your text, and ``rerank_units`` only when you passed
    ``rerank``. Both being ``None`` is normal for a search that supplied its own vector.

    Attributes:
        read_units (int): Read units the search consumed.
        embed_total_tokens (int | None): Tokens embedded server-side, or ``None`` when the
            search did not embed anything.
        rerank_units (int | None): Rerank units consumed, or ``None`` when the search did
            not rerank.
    """

    read_units: int
    embed_total_tokens: int | None = None
    rerank_units: int | None = None


class Hit(StructDictMixin, Struct, kw_only=True, rename={"id_": "_id", "score_": "_score"}):
    """One search result: which record matched, how well, and the fields you asked for.

    Read a hit as ``hit.id``, ``hit.score`` and ``hit.fields``. The underscore-suffixed
    ``id_`` and ``score_`` exist because the wire names are ``_id`` and ``_score``, which
    Python would name-mangle inside a class; prefer the unsuffixed properties in your own
    code. Bracket access works too, under the unsuffixed names: ``hit["id"]``.

    ``fields`` holds your record's own data, so what is in it depends on the ``fields``
    argument the search passed — this is where a search differs from a query, which
    splits the same information across ``values`` and ``metadata``.

    Attributes:
        id_ (str): The record identifier; read it as ``hit.id``. Wire name ``_id``.
        score_ (float): How well the record matched; read it as ``hit.score``. Higher is
            better, and after reranking the scale is the reranker's, not the index's.
            Wire name ``_score``.
        fields (dict[str, Any]): The record fields the search returned, keyed by field
            name. Omitting the search's ``fields`` argument returns every field the record
            has, so narrow it when you only need one or two.

    Examples:
        .. code-block:: python

            response = idx.search(
                namespace="articles-en",
                top_k=5,
                inputs={"text": "how do sparse indexes score matches"},
                fields=["title", "chunk"],
            )
            for hit in response.result.hits:
                print(hit.id, hit.score, hit.fields["title"])
    """

    id_: str
    score_: float
    fields: dict[str, Any] = {}

    @property
    def id(self) -> str:
        """The record identifier. Prefer this over the wire-shaped ``id_``."""
        return self.id_

    @property
    def score(self) -> float:
        """How well the record matched. Prefer this over the wire-shaped ``score_``."""
        return self.score_

    def __getitem__(self, key: str) -> Any:
        """Read a field by name, accepting ``"id"`` and ``"score"`` for the properties.

        Raises:
            KeyError: If *key* is neither of those nor one of this model's fields.
        """
        if key == "id":
            return self.id_
        if key == "score":
            return self.score_
        if key not in self.__struct_fields__:
            raise KeyError(key)
        return getattr(self, key)

    def __contains__(self, key: object) -> bool:
        """Report whether *key* is readable here; ``"id"`` and ``"score"`` always are."""
        if key in ("id", "score"):
            return True
        return key in self.__struct_fields__

    def __repr__(self) -> str:
        return f"Hit(id={self.id!r}, score={self.score!r}, fields={self.fields!r})"


class SearchResult(StructDictMixin, Struct, kw_only=True):
    """The one-field wrapper around a search's hits.

    It exists because the response envelope nests them, which is why reading a search
    result is ``response.result.hits`` and not ``response.hits``.

    Attributes:
        hits (list[Hit]): The matching records, ordered best match first.
    """

    hits: list[Hit] = []


class SearchRecordsResponse(StructDictMixin, Struct, kw_only=True):
    """What ``search`` returns: the hits, nested one level down, plus what the call cost.

    The hits live at ``response.result.hits`` — the extra ``result`` step is the shape of
    the response envelope, and forgetting it is the usual first stumble here. Each hit is
    a :class:`Hit`, read as ``.id``, ``.score`` and ``.fields``. A search that matched
    nothing returns an empty ``hits`` list rather than raising.

    Attributes:
        result (SearchResult): The wrapper holding ``hits``.
        usage (SearchUsage): What the search cost, broken out by stage.
        response_info (ResponseInfo | None): HTTP response metadata (request ID, LSN
            values), or ``None`` if not populated.

    Examples:
        .. code-block:: python

            response = idx.search(
                namespace="articles-en",
                top_k=5,
                inputs={"text": "how do sparse indexes score matches"},
                fields=["title"],
            )
            for hit in response.result.hits:
                print(hit.id, hit.score, hit.fields["title"])
            print(response.usage.read_units)

    .. seealso::
       :class:`~pinecone.models.vectors.responses.QueryResponse` — what ``query`` returns
       instead, where the matches are at ``response.matches`` and carry ``values`` and
       ``metadata`` rather than ``fields``.
    """

    result: SearchResult
    usage: SearchUsage
    response_info: ResponseInfo | None = None

    @overload
    def __getitem__(self, key: Literal["result"]) -> SearchResult: ...

    @overload
    def __getitem__(self, key: Literal["usage"]) -> SearchUsage: ...

    @overload
    def __getitem__(self, key: str) -> Any: ...

    def __getitem__(self, key: str) -> Any:
        """Read a field by name, so ``response["result"]`` works as well as ``.result``.

        Raises:
            KeyError: If *key* is not one of this response's fields.
        """
        if key not in self.__struct_fields__:
            raise KeyError(key)
        return getattr(self, key)

    def __contains__(self, key: object) -> bool:
        """Report whether *key* names a field on this response."""
        return key in self.__struct_fields__


class SearchQuery(DictLikeStruct, Struct, kw_only=True, gc=False):
    """Query parameters for a search operation (legacy backcompat type).

    Attributes:
        inputs (dict[str, Any]): Search inputs (e.g. ``{"text": "hello"}``).
        top_k (int): Number of top results to return.
        filter (dict[str, Any] | None): Metadata filter to apply, or ``None`` for no filter.
        vector (dict[str, Any] | None): Explicit query vector, or ``None`` to use inputs.
        id (str | None): ID of a stored record to use as query vector, or ``None``.
        match_terms (dict[str, Any] | None): Full-text match terms, or ``None``.
    """

    inputs: dict[str, Any]
    top_k: int
    filter: dict[str, Any] | None = None
    vector: dict[str, Any] | None = None
    id: str | None = None
    match_terms: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a dict of non-None field values.

        Returns:
            Dictionary containing only the fields whose value is not ``None``.
            Required fields (``inputs``, ``top_k``) are always present; optional
            fields (``filter``, ``vector``, ``id``, ``match_terms``) are omitted
            when they are ``None``.

        Examples:
            >>> from pinecone.models.vectors.search import SearchQuery
            >>> query = SearchQuery(inputs={"text": "hello"}, top_k=10)
            >>> query.to_dict()
            {'inputs': {'text': 'hello'}, 'top_k': 10}
            >>> query_with_filter = SearchQuery(
            ...     inputs={"text": "hello"},
            ...     top_k=10,
            ...     filter={"genre": "action"},
            ... )
            >>> query_with_filter.to_dict()
            {'inputs': {'text': 'hello'}, 'top_k': 10, 'filter': {'genre': 'action'}}
        """
        return {f: getattr(self, f) for f in self.__struct_fields__ if getattr(self, f) is not None}

    as_dict = to_dict


class SearchQueryVector(DictLikeStruct, Struct, kw_only=True, gc=False):
    """Explicit dense/sparse query vector for search operations (legacy backcompat type).

    Attributes:
        values (list[float] | None): Dense vector values, or ``None`` if not provided.
        sparse_values (list[float] | None): Sparse vector values, or ``None`` if not provided.
        sparse_indices (list[int] | None): Sparse vector indices, or ``None`` if not provided.
    """

    values: list[float] | None = None
    sparse_values: list[float] | None = None
    sparse_indices: list[int] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a dict of non-None field values.

        Returns:
            Dictionary containing only the fields whose value is not ``None``.
            All fields (``values``, ``sparse_values``, ``sparse_indices``) are
            optional and omitted when ``None``.

        Examples:
            >>> from pinecone.models.vectors.search import SearchQueryVector
            >>> vec = SearchQueryVector(values=[0.1, 0.2, 0.3])
            >>> vec.to_dict()
            {'values': [0.1, 0.2, 0.3]}
            >>> vec_sparse = SearchQueryVector(
            ...     values=[0.1, 0.2],
            ...     sparse_values=[0.5],
            ...     sparse_indices=[3],
            ... )
            >>> vec_sparse.to_dict()
            {'values': [0.1, 0.2], 'sparse_values': [0.5], 'sparse_indices': [3]}
        """
        return {f: getattr(self, f) for f in self.__struct_fields__ if getattr(self, f) is not None}

    as_dict = to_dict


class SearchRerank(DictLikeStruct, Struct, kw_only=True, gc=False):
    """Reranking configuration for a search operation (legacy backcompat type).

    Attributes:
        model (str): Reranking model name (e.g. ``"bge-reranker-v2-m3"``).
        top_n (int | None): Number of top results after reranking, or ``None`` to use ``top_k``.
        rank_fields (list[str] | None): Record fields to rank on, or ``None``.
        parameters (dict[str, Any] | None): Model-specific parameters, or ``None``.
        query (str | None): Override query text for reranking, or ``None`` to infer from inputs.
    """

    model: str
    top_n: int | None = None
    rank_fields: list[str] | None = None
    parameters: dict[str, Any] | None = None
    query: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a dict of non-None field values.

        Returns:
            Dictionary containing only the fields whose value is not ``None``.
            The ``model`` field is always present; optional fields (``top_n``,
            ``rank_fields``, ``parameters``, ``query``) are omitted when ``None``.

        Examples:
            >>> from pinecone.models.vectors.search import SearchRerank
            >>> rerank = SearchRerank(model="bge-reranker-v2-m3")
            >>> rerank.to_dict()
            {'model': 'bge-reranker-v2-m3'}
            >>> rerank_full = SearchRerank(
            ...     model="bge-reranker-v2-m3",
            ...     top_n=5,
            ...     rank_fields=["text"],
            ...     query="hello world",
            ... )
            >>> d = rerank_full.to_dict()
            >>> d["model"]
            'bge-reranker-v2-m3'
            >>> d["top_n"]
            5
            >>> d["rank_fields"]
            ['text']
        """
        return {f: getattr(self, f) for f in self.__struct_fields__ if getattr(self, f) is not None}

    as_dict = to_dict
