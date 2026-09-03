"""The vector record you upsert, and the scored match a query returns."""

from __future__ import annotations

from typing import Any

from msgspec import Struct

from pinecone.errors.exceptions import PineconeValueError
from pinecone.models._mixin import DictLikeStruct
from pinecone.models.vectors.sparse import SparseValues


class Vector(DictLikeStruct, Struct, rename="camel", gc=False):
    """One record in a vector-based index: an ID, its coordinates, and its metadata.

    A vector carries its coordinates in either or both of two representations, and which
    ones you populate is what makes a vector dense, sparse, or hybrid:

    * **Dense** — ``values``, a list of floats whose length equals the ``dimension`` of the
      index field it is written to, ranked by that field's ``metric``. This is the usual
      output of an embedding model, and the representation that finds records by meaning.
      Build a dense vector when you have an embedding.
    * **Sparse** — ``sparse_values``, which names only the non-zero dimensions as parallel
      ``indices`` and ``values`` lists and has no fixed ``dimension``. This is how
      keyword-based scoring such as BM25 is expressed, and it finds records by exact term.
      Build a sparse vector when the terms themselves matter.
    * **Hybrid** — both populated on the same vector, so one record is reachable by meaning
      and by term. The index has to declare a dense field and a sparse field for this to
      be accepted.

    At least one of the two must be populated. A sparse-only vector leaves ``values`` empty,
    and the empty dense list is still sent.

    Attributes:
        id (str): Unique identifier for the vector; the client rejects an ID that is not
            ASCII, is empty, is over 512 characters, or contains NUL. Use an ID you can
            recompute from your own data, e.g. ``"article-101"``.
        values (list[float]): Dense vector values. Empty for a sparse-only vector.
        sparse_values (SparseValues | None): Sparse component, or ``None`` for a dense-only
            vector.
        metadata (dict[str, Any] | None): Your own key-value pairs to filter on later, or
            ``None`` if none are attached. Each value must be a string, a number, a boolean,
            or a list of strings — a nested object, or a list with a non-string element, is
            rejected. A key whose value is ``None`` is dropped rather than rejected. Keys may
            not begin with ``$``, which is reserved for filter operators; every other key is
            accepted, including empty and non-ASCII keys.

    Raises:
        PineconeValueError: If neither ``values`` nor ``sparse_values`` is populated — a
            vector with no coordinates cannot be scored against anything.

    Examples:
        A dense vector. Pass the embedding your model produced; its length has to match the
        dimension of the field you upsert into, so the three floats here stand in for a
        full-length embedding.

        >>> from pinecone import Vector
        >>> dense = Vector(id="article-101", values=[0.12, 0.34, 0.56])
        >>> dense.sparse_values is None
        True

        A sparse vector, where each index is a term slot that scored non-zero. ``values``
        comes back empty because nothing dense was supplied.

        >>> from pinecone import SparseValues
        >>> sparse = Vector(
        ...     id="article-102",
        ...     sparse_values=SparseValues(indices=[10, 42, 913], values=[0.4, 0.9, 0.2]),
        ... )
        >>> sparse.values
        []

        A hybrid vector populates both, and can carry metadata to filter on.

        >>> hybrid = Vector(
        ...     id="article-103",
        ...     values=[0.12, 0.34, 0.56],
        ...     sparse_values=SparseValues(indices=[10, 42], values=[0.4, 0.9]),
        ...     metadata={"lang": "en", "published": True},
        ... )
        >>> len(hybrid.values), len(hybrid.sparse_values.indices)
        (3, 2)

    .. seealso::
       :class:`ScoredVector` — what a query returns for each match, which adds ``score``.
       :class:`~pinecone.models.documents.document.DocumentRecord` — the record type for
       schema-based indexes, which store JSON documents rather than raw vectors.
    """

    id: str
    values: list[float] = []
    sparse_values: SparseValues | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Require at least one of ``values`` or ``sparse_values`` to be populated."""
        if not self.values and self.sparse_values is None:
            raise PineconeValueError("Vector must have either values or sparse_values")

    @staticmethod
    def from_dict(vector_dict: dict[str, Any]) -> Vector:
        """Build a :class:`Vector` from a plain dict.

        Accepts the snake_case keys ``id``, ``values``, ``sparse_values`` and ``metadata``.
        Use it when your vectors arrive as dicts — from your own JSON, a dataframe row, or
        a previous ``to_dict()`` — and you want the same construction-time check that
        ``Vector(...)`` applies.

        Args:
            vector_dict (dict[str, Any]): The dict to convert. ``id`` is required; the rest
                are optional and default the same way the constructor does.

        Returns:
            :class:`Vector` with ``sparse_values`` decoded into a :class:`SparseValues`.

        Raises:
            KeyError: If ``id`` is absent.
            PineconeValueError: If neither ``values`` nor ``sparse_values`` is populated.

        Examples:
            >>> from pinecone import Vector
            >>> Vector.from_dict({"id": "article-101", "values": [0.12, 0.34, 0.56]}).id
            'article-101'
        """
        sparse: SparseValues | None = None
        if vector_dict.get("sparse_values") is not None:
            sparse = SparseValues.from_dict(vector_dict["sparse_values"])
        return Vector(
            id=vector_dict["id"],
            values=vector_dict.get("values", []),
            sparse_values=sparse,
            metadata=vector_dict.get("metadata"),
        )

    def __repr__(self) -> str:
        if len(self.values) > 5:
            preview = ", ".join(repr(v) for v in self.values[:3])
            values_str = f"[{preview}, ...{len(self.values) - 3} more]"
        else:
            values_str = repr(self.values)
        parts = [
            f"id={self.id!r}",
            f"values={values_str}",
            f"sparse_values={self.sparse_values!r}",
        ]
        if self.metadata is not None:
            parts.append(f"metadata={self.metadata!r}")
        return f"Vector({', '.join(parts)})"


class ScoredVector(DictLikeStruct, Struct, rename="camel", kw_only=True, gc=False):
    """One match from a query: the vector that was found, plus how close it was.

    Every element of
    :attr:`QueryResponse.matches <pinecone.models.vectors.responses.QueryResponse.matches>`
    is one of these, so ``id`` and ``score`` are always populated. ``values`` and
    ``metadata`` are not: a query omits both unless you ask for them, so an empty
    ``values`` or a ``None`` ``metadata`` usually means the request did not set the
    corresponding flag rather than that the stored vector lacks them.

    Attributes:
        id (str): Identifier of the matched vector, the same ID it was upserted under.
        score (float): How close the match is under the queried field's ``metric``. Higher
            is closer for ``cosine`` and ``dotproduct``; lower is closer for ``euclidean``.
            Compare scores only within one response — the scale depends on the metric and
            on your data, so no fixed threshold means "good" across indexes.
        values (list[float]): Dense values of the matched vector, or ``[]`` when the query
            did not pass ``include_values=True``.
        sparse_values (SparseValues | None): Sparse component of the matched vector, or
            ``None`` for a dense-only vector or when values were not requested.
        metadata (dict[str, Any] | None): The metadata stored with the vector, or ``None``
            when the query did not pass ``include_metadata=True`` or nothing was stored.
            Values follow the same grammar as :attr:`Vector.metadata`.

    Examples:
        A query returns these in ``matches``, ordered most similar first, so reading a
        result is the same two attributes every time.

        .. code-block:: python

            response = idx.query(top_k=5, vector=[0.012, -0.087, 0.153])
            for match in response.matches:
                print(match.id, match.score)

        Ask for metadata if you intend to read it — without the flag ``match.metadata`` is
        ``None`` even for vectors that have metadata stored.

        .. code-block:: python

            response = idx.query(
                top_k=5,
                vector=[0.012, -0.087, 0.153],
                namespace="articles-en",
                include_metadata=True,
            )
            for match in response.matches:
                print(match.id, match.metadata["lang"])
    """

    id: str
    score: float
    values: list[float] = []
    sparse_values: SparseValues | None = None
    metadata: dict[str, Any] | None = None

    def __repr__(self) -> str:
        if len(self.values) > 5:
            preview = ", ".join(repr(v) for v in self.values[:3])
            values_str = f"[{preview}, ...{len(self.values) - 3} more]"
        else:
            values_str = repr(self.values)
        parts = [
            f"id={self.id!r}",
            f"score={self.score!r}",
            f"values={values_str}",
        ]
        if self.sparse_values is not None:
            parts.append(f"sparse_values={self.sparse_values!r}")
        if self.metadata is not None:
            parts.append(f"metadata={self.metadata!r}")
        return f"ScoredVector({', '.join(parts)})"
