"""The six ways a document search can score a document, and the union over them.

Every ``score_by`` clause is one of these six, told apart by its ``type`` tag:
``text`` for BM25 keyword scoring, ``query_string`` for a Lucene expression,
``dense_vector`` for similarity to an embedding, ``sparse_vector`` for similarity to
a sparse vector, and ``embed`` and ``sparse_embed`` for similarity to query text that
Pinecone embeds with the model of a ``string`` field declared with that aspect. Pass
them as typed instances or as plain dicts with a ``type`` key. Either way each one
checks its required fields on construction, so a malformed clause is rejected at the
call site. A dict clause with a key the clause does not define, such as ``boost``, is
rejected rather than dropped, so an option that has no effect cannot go unnoticed.

``text`` and ``query_string`` clauses can be combined in one search to score on several
signals at once. The clauses are OR'd: a document matching any of them is returned, and
one matching several scores the sum of the scores from each clause it matches. Repeating
a clause, naming a field twice, or writing one query as both a ``text`` and a
``query_string`` clause scores it once, so repeating a clause does not up-weight it; a
``^N`` boost in a ``query_string`` clause does up-weight a term. A vector clause, and an
``embed`` or ``sparse_embed`` clause, has to stand alone.
"""

from __future__ import annotations

import warnings
from typing import Any

from msgspec import Struct

from pinecone.errors.exceptions import PineconeTypeError
from pinecone.models.vectors.sparse import SparseValues

__all__ = [
    "DenseVectorQuery",
    "DocumentScoringMethod",
    "EmbedQuery",
    "QueryStringQuery",
    "SparseEmbedQuery",
    "SparseVectorQuery",
    "TextQuery",
]


def _reject_bare_str(fields: object) -> None:
    if isinstance(fields, str):
        raise PineconeTypeError(
            f"'fields' must be a list of field names, not a single str; pass [{fields!r}]."
        )


def _single_field(kind: str, field: str | None, fields: list[str] | None) -> str | None:
    if fields is None:
        return field
    _reject_bare_str(fields)
    if field is not None:
        raise ValueError(f"{kind} scoring accepts 'field' or 'fields', but not both")
    if len(fields) != 1:
        raise ValueError(f"{kind} scoring requires specifying the name of exactly one field")
    return fields[0]


def _plain_list(raw: Any, name: str) -> Any:
    tolist = getattr(raw, "tolist", None)
    if tolist is None:
        return raw
    converted = tolist()
    if not isinstance(converted, list):
        raise PineconeTypeError(
            f"{name} must be a sequence or array of numbers, got {type(raw).__name__} "
            f"whose tolist() returned {type(converted).__name__}"
        )
    return converted


class TextQuery(
    Struct,
    tag="text",
    tag_field="type",
    kw_only=True,
    omit_defaults=True,
    forbid_unknown_fields=True,
):
    """Score documents by BM25 keyword relevance over named text fields.

    The clause to reach for when the words themselves matter (exact terms, names, codes)
    rather than meaning. Scoring is per field, so name every field you want searched.
    A document matches if the query matches in any one of the named fields, and its score
    is the sum of its BM25 scores in the fields that matched, so naming another field
    widens the results and can raise scores. The text is matched literally, so a ``^N``
    inside ``query`` is not a boost; to weight a term, use :class:`QueryStringQuery`.

    Attributes:
        query: The words to search for, e.g. ``"sparse index scoring"``.
        fields: The text fields to search across, e.g. ``["title", "chunk"]``. At least
            one is required.
        field: Deprecated single-field form of ``fields``. A value here is moved into
            ``fields`` with a :exc:`DeprecationWarning` and this attribute reads back as
            ``None``, so read ``fields`` whichever way you set it.

    Raises:
        ValueError: If ``fields`` ends up empty, or if both ``fields`` and ``field`` are
            given.

    Examples:
        >>> from pinecone.models.documents.score_by import TextQuery
        >>> TextQuery(query="sparse index scoring", fields=["title", "chunk"]).fields
        ['title', 'chunk']

    .. seealso::
       :class:`QueryStringQuery`: when you need boolean operators in the query itself.
    """

    query: str
    fields: list[str] | None = None
    field: str | None = None

    def __post_init__(self) -> None:
        if self.field is not None:
            if self.fields is not None:
                raise ValueError(
                    "TextQuery accepts `fields=[...]` or the deprecated `field=...`, but not both."
                )
            warnings.warn(
                "TextQuery `field` is deprecated; use `fields=[...]`.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.fields = [self.field]
            self.field = None
        _reject_bare_str(self.fields)
        if not self.fields:
            raise ValueError("Text scoring requires specifying at least one field")


class QueryStringQuery(
    Struct,
    tag="query_string",
    tag_field="type",
    kw_only=True,
    omit_defaults=True,
    forbid_unknown_fields=True,
):
    """Score documents by a Lucene query string, with ``AND``, ``OR`` and ``NOT``.

    Choose this over :class:`TextQuery` when the query itself needs structure: combining
    terms, excluding one, or scoping a clause to one field. Fields are named *inside* the
    query string as ``field_name:(clause)``; leave the qualifiers off and every
    text-searchable field is searched.

    Attributes:
        query: The Lucene expression, e.g. ``'title:(sparse OR hybrid) NOT draft'``.
            Weight a term with a ``^N`` boost after it, e.g. ``title:entropy^5``: the
            term's score is multiplied by ``N``.
        field: Not accepted. Name the field with a qualifier in ``query`` instead.
        fields: Not accepted. Name the fields with qualifiers in ``query`` instead.

    Raises:
        ValueError: If ``field`` or ``fields`` is given.

    Examples:
        >>> from pinecone.models.documents.score_by import QueryStringQuery
        >>> QueryStringQuery(query="title:(sparse OR hybrid) NOT draft").query
        'title:(sparse OR hybrid) NOT draft'
        >>> QueryStringQuery(query="title:entropy^5 abstract:entropy").query
        'title:entropy^5 abstract:entropy'
    """

    query: str
    field: str | None = None
    fields: list[str] | None = None

    def __post_init__(self) -> None:
        if self.field is not None or self.fields is not None:
            raise ValueError(
                "'score_by' clauses of type 'query_string' must not specify 'field' or "
                "'fields'; use Lucene-style field qualifiers (e.g. 'field_name:value') "
                "in the query string instead"
            )


class DenseVectorQuery(
    Struct,
    tag="dense_vector",
    tag_field="type",
    kw_only=True,
    omit_defaults=True,
    forbid_unknown_fields=True,
):
    """Score documents by dense vector similarity to a query embedding.

    The clause for finding documents by meaning. It scores against one field, so the
    field named has to be declared ``dense_vector`` in the index schema, and ``values``
    has to be as long as that field's ``dimension``. A dense clause cannot be combined
    with any other scoring method in the same search.

    Attributes:
        field: The dense vector field to score against, e.g. ``"your_vector"``.
        values: The query embedding, one float per dimension of that field: a list, or an
            array with a ``tolist()`` method such as a numpy array, which the clause
            converts to a list.
        fields: The same field as a one-element list, e.g. ``["your_vector"]``, in place
            of ``field``. It is moved into ``field`` and reads back as ``None``.

    Raises:
        ValueError: If ``field`` is empty, ``values`` is empty, ``fields`` does not hold
            exactly one name, or both ``field`` and ``fields`` are given.
        PineconeTypeError: If ``values`` is not a sequence or array, such as a single
            number or a 0-dimensional numpy array.

    Examples:
        The three floats here stand in for a full-length embedding.

        >>> from pinecone.models.documents.score_by import DenseVectorQuery
        >>> DenseVectorQuery(field="your_vector", values=[0.12, 0.34, 0.56]).field
        'your_vector'

    .. seealso::
       :class:`SparseVectorQuery`: the same idea against a sparse field.
    """

    values: list[float]
    field: str | None = None
    fields: list[str] | None = None

    def __post_init__(self) -> None:
        self.field = _single_field("dense_vector", self.field, self.fields)
        self.fields = None
        if not self.field:
            raise ValueError(
                "dense_vector scoring requires specifying the name of exactly one field"
            )
        self.values = _plain_list(self.values, "DenseVectorQuery values")
        if self.values is None:
            raise ValueError("dense_vector scoring requires a non-empty 'values' array")
        if not hasattr(self.values, "__len__"):
            raise PineconeTypeError(
                "DenseVectorQuery values must be a sequence or array of numbers, "
                f"got {type(self.values).__name__}"
            )
        if len(self.values) == 0:
            raise ValueError("dense_vector scoring requires a non-empty 'values' array")


class SparseVectorQuery(
    Struct,
    tag="sparse_vector",
    tag_field="type",
    kw_only=True,
    omit_defaults=True,
    forbid_unknown_fields=True,
):
    """Score documents by sparse vector similarity to a query sparse vector.

    The clause for finding documents by term overlap when you already hold a sparse
    encoding of the query. The field named has to be declared ``sparse_vector`` in the
    index schema, and, like a dense clause, this one cannot be combined with any other
    scoring method in the same search.

    Attributes:
        field: The sparse vector field to score against, e.g. ``"keywords"``.
        sparse_values: The query's sparse vector, as
            :class:`~pinecone.models.vectors.sparse.SparseValues`. Its ``indices`` and
            ``values`` may be lists or arrays with a ``tolist()`` method such as numpy
            arrays, which the clause converts to lists.
        fields: The same field as a one-element list, e.g. ``["keywords"]``, in place of
            ``field``. It is moved into ``field`` and reads back as ``None``.

    Raises:
        ValueError: If ``field`` is empty, ``fields`` does not hold exactly one name, or
            both ``field`` and ``fields`` are given.
        PineconeTypeError: If the ``indices`` or ``values`` of ``sparse_values`` has a
            ``tolist()`` method that does not return a list, such as a 0-dimensional numpy
            array.

    Examples:
        >>> from pinecone import SparseValues
        >>> from pinecone.models.documents.score_by import SparseVectorQuery
        >>> SparseVectorQuery(
        ...     field="keywords",
        ...     sparse_values=SparseValues(indices=[10, 42], values=[0.4, 0.9]),
        ... ).field
        'keywords'

    .. seealso::
       :class:`TextQuery`: when you have the words rather than a sparse encoding of them.
    """

    sparse_values: SparseValues
    field: str | None = None
    fields: list[str] | None = None

    def __post_init__(self) -> None:
        self.field = _single_field("sparse_vector", self.field, self.fields)
        self.fields = None
        if not self.field:
            raise ValueError(
                "sparse_vector scoring requires specifying the name of exactly one field"
            )
        if isinstance(self.sparse_values, SparseValues) and (
            hasattr(self.sparse_values.indices, "tolist")
            or hasattr(self.sparse_values.values, "tolist")
        ):
            self.sparse_values = SparseValues(
                indices=_plain_list(self.sparse_values.indices, "SparseValues indices"),
                values=_plain_list(self.sparse_values.values, "SparseValues values"),
            )


class EmbedQuery(
    Struct,
    tag="embed",
    tag_field="type",
    kw_only=True,
    omit_defaults=True,
    forbid_unknown_fields=True,
):
    """Score documents by dense similarity to query text that Pinecone embeds.

    The clause for searching a ``string`` field declared with ``embed``: pass the query
    as text, and Pinecone embeds it with that field's model and read parameters, then
    scores it against the field's stored embeddings. You never handle a vector. Like a
    dense clause, it has to stand alone in ``score_by``.

    It takes ``query`` text, not a vector: pass a vector with :class:`DenseVectorQuery`.

    Attributes:
        field: The ``string`` field that declares ``embed``, e.g. ``"body"``.
        query: The text to search for, e.g. ``"how do vector indexes scale"``.
        fields: The same field as a one-element list, e.g. ``["body"]``, in place of
            ``field``. It is moved into ``field`` and reads back as ``None``.
        parameters: Model parameters for embedding the query in this search, e.g.
            ``{"task_type": "fact_verification"}``. They override the field's
            ``read_parameters``. Only ``task_type`` can be set, and
            :meth:`pc.inference.list_models() <pinecone.client.inference.Inference.list_models>`
            lists the models that accept it. The dict is sent as given and the server
            checks it; leave it unset to embed with the field's ``read_parameters``.

    Raises:
        ValueError: If ``field`` or ``query`` is empty, ``fields`` does not hold exactly
            one name, or both ``field`` and ``fields`` are given.

    Examples:
        >>> from pinecone.models.documents.score_by import EmbedQuery
        >>> EmbedQuery(field="body", query="how do vector indexes scale").field
        'body'

        Choose the query task for one search on a field that embeds with a model that
        takes ``task_type``, such as ``gemini-embedding-2``:

        >>> EmbedQuery(
        ...     field="body",
        ...     query="Is the Eiffel Tower taller than 300 m?",
        ...     parameters={"task_type": "fact_verification"},
        ... ).parameters
        {'task_type': 'fact_verification'}

    .. seealso::
       :class:`DenseVectorQuery`: to score a ``dense_vector`` field with a vector you
       embedded yourself.
    """

    query: str
    field: str | None = None
    fields: list[str] | None = None
    parameters: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.field = _single_field("embed", self.field, self.fields)
        self.fields = None
        if not self.field:
            raise ValueError("embed scoring requires specifying the name of exactly one field")
        if not self.query:
            raise ValueError("embed scoring requires a non-empty 'query'")


class SparseEmbedQuery(
    Struct,
    tag="sparse_embed",
    tag_field="type",
    kw_only=True,
    omit_defaults=True,
    forbid_unknown_fields=True,
):
    """Score documents by sparse similarity to query text that Pinecone embeds.

    The clause for searching a ``string`` field declared with ``sparse_embed``: pass the
    query as text, and Pinecone embeds it with that field's sparse model and read
    parameters. It has to stand alone in ``score_by``.

    It takes ``query`` text, not a vector: pass one with :class:`SparseVectorQuery`.

    Attributes:
        field: The ``string`` field that declares ``sparse_embed``, e.g. ``"body"``.
        query: The text to search for, e.g. ``"vector database"``.
        fields: The same field as a one-element list, e.g. ``["body"]``, in place of
            ``field``. It is moved into ``field`` and reads back as ``None``.

    Raises:
        ValueError: If ``field`` or ``query`` is empty, ``fields`` does not hold exactly
            one name, or both ``field`` and ``fields`` are given.

    Examples:
        >>> from pinecone.models.documents.score_by import SparseEmbedQuery
        >>> SparseEmbedQuery(field="body", query="vector database").query
        'vector database'

    .. seealso::
       :class:`SparseVectorQuery`: to score a ``sparse_vector`` field with a sparse
       vector you encoded yourself.
    """

    query: str
    field: str | None = None
    fields: list[str] | None = None

    def __post_init__(self) -> None:
        self.field = _single_field("sparse_embed", self.field, self.fields)
        self.fields = None
        if not self.field:
            raise ValueError(
                "sparse_embed scoring requires specifying the name of exactly one field"
            )
        if not self.query:
            raise ValueError("sparse_embed scoring requires a non-empty 'query'")


DocumentScoringMethod = (
    TextQuery
    | QueryStringQuery
    | DenseVectorQuery
    | SparseVectorQuery
    | EmbedQuery
    | SparseEmbedQuery
)
