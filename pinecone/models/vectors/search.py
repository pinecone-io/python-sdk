"""Search records response models."""

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
    """Typed configuration for the ``rerank`` parameter of :meth:`~pinecone.Index.search`.

    Required keys: ``model``, ``rank_fields``.
    All other keys are optional.

    Attributes:
        model (str): Reranking model name (e.g. ``"bge-reranker-v2-m3"``).
        rank_fields (list[str]): Record fields to rank on (e.g. ``["text"]``).
        top_n (int): Number of top results to return after reranking.
            Defaults to the value of ``top_k`` when omitted.
        parameters (dict[str, Any]): Model-specific parameters forwarded to
            the reranker. See the model documentation for supported keys.
        query (str): Override query text used for reranking.  When omitted the
            query is inferred from the search inputs.
    """

    top_n: int
    parameters: dict[str, Any]
    query: str


class _SearchInputsRequired(TypedDict):
    """Required fields of :class:`SearchInputs`."""

    text: str


class SearchInputs(_SearchInputsRequired, total=False):
    """Typed configuration for the ``inputs`` parameter of :meth:`~pinecone.Index.search`.

    Required keys: ``text``.

    Attributes:
        text (str): Text to embed server-side for the search query.
    """


class SearchUsage(StructDictMixin, Struct, kw_only=True):
    """Usage statistics for a search operation.

    Attributes:
        read_units (int): Number of read units consumed.
        embed_total_tokens (int | None): Total tokens used for embedding, or ``None``
            if the search did not use integrated embedding.
        rerank_units (int | None): Number of rerank units consumed, or ``None`` if the
            search did not use reranking.
    """

    read_units: int
    embed_total_tokens: int | None = None
    rerank_units: int | None = None


_WIRE_KEY_ALIASES = {"_id": "id", "_score": "score"}


class Hit(StructDictMixin, Struct, kw_only=True, rename={"id_": "_id", "score_": "_score"}):
    """A single search result hit.

    The API returns ``_id`` and ``_score`` as field names. These are mapped
    to ``id_`` and ``score_`` internally (to avoid Python name mangling),
    with convenience properties ``id`` and ``score`` for clean access.

    Attributes:
        id_ (str): The record identifier (wire name ``_id``).
        score_ (float): The similarity score (wire name ``_score``).
        fields (dict[str, Any]): Record fields included in the result.
    """

    id_: str
    score_: float
    fields: dict[str, Any] = {}

    @property
    def id(self) -> str:
        """Alias for ``id_`` to provide a cleaner API."""
        return self.id_

    @property
    def score(self) -> float:
        """Alias for ``score_`` to provide a cleaner API."""
        return self.score_

    def __getitem__(self, key: str) -> Any:
        """Support bracket access (e.g. ``hit['id']``)."""
        if key == "id":
            return self.id_
        if key == "score":
            return self.score_
        if key not in self.__struct_fields__:
            alias = _WIRE_KEY_ALIASES.get(key)
            if alias is not None:
                raise KeyError(
                    f"{key!r} is the wire name for this field; use hit[{alias!r}] or hit.{alias}"
                )
            raise KeyError(
                f"{key!r}; available keys are 'id', 'score', and 'fields' "
                "(record fields live in hit['fields'])"
            )
        return getattr(self, key)

    def __contains__(self, key: object) -> bool:
        """Support ``in`` operator (e.g. ``'id' in hit``)."""
        if key in ("id", "score"):
            return True
        return key in self.__struct_fields__

    def __repr__(self) -> str:
        return f"Hit(id={self.id!r}, score={self.score!r}, fields={self.fields!r})"


class SearchResult(StructDictMixin, Struct, kw_only=True):
    """The result wrapper containing hits.

    Attributes:
        hits (list[Hit]): List of search result hits.
    """

    hits: list[Hit] = []


class SearchRecordsResponse(StructDictMixin, Struct, kw_only=True):
    """Response from a search records operation.

    Attributes:
        result (SearchResult): Wrapper containing the list of hits.
        usage (SearchUsage): Usage statistics for the search operation.
        response_info (ResponseInfo | None): HTTP response metadata (request ID, LSN values), or
            ``None`` if not populated.
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
        """Support bracket access (e.g. ``response['result']``)."""
        if key not in self.__struct_fields__:
            raise KeyError(key)
        return getattr(self, key)

    def __contains__(self, key: object) -> bool:
        """Support ``in`` operator (e.g. ``'result' in response``)."""
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
            >>> from pinecone.db_data.dataclasses.search_query import SearchQuery
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
            >>> from pinecone.db_data.dataclasses.search_query_vector import SearchQueryVector
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
            >>> from pinecone.db_data.dataclasses.search_rerank import SearchRerank
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
