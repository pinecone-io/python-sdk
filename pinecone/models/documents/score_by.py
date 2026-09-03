"""Score-by query models for document search (2026-07 API).

``DocumentScoringMethod`` is the typed union of the four scoring method
variants, discriminated on the ``type`` tag. Each variant validates its
required fields at construction, using the backend's error vocabulary, so
invalid queries fail before any HTTP request.
"""

from __future__ import annotations

import warnings

from msgspec import Struct

from pinecone.models.vectors.sparse import SparseValues

__all__ = [
    "DenseVectorQuery",
    "DocumentScoringMethod",
    "QueryStringQuery",
    "SparseVectorQuery",
    "TextQuery",
]


class TextQuery(Struct, tag="text", tag_field="type", kw_only=True, omit_defaults=True):
    """Full-text (BM25) search query for scoring documents.

    Attributes:
        query: Search query string.
        fields: One or more text field names to search across.
        field: Deprecated alias for ``fields``. If provided, it is migrated
            to ``fields=[field]`` and a ``DeprecationWarning`` is emitted.
            Cleared to ``None`` after migration; read ``fields`` instead.
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
        if not self.fields:
            raise ValueError("Text scoring requires specifying at least one field")


class QueryStringQuery(
    Struct, tag="query_string", tag_field="type", kw_only=True, omit_defaults=True
):
    """Lucene query string search with boolean operators (AND, OR, NOT).

    Target a specific field with a field qualifier inside the query string
    (``field_name:(clause)``), or omit qualifiers to search against all
    text-searchable fields.

    Attributes:
        query: Query string with operators.
        field: Not accepted for this scoring type; always ``None``.
        fields: Not accepted for this scoring type; always ``None``.
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


class DenseVectorQuery(Struct, tag="dense_vector", tag_field="type", kw_only=True):
    """Dense vector similarity query for scoring documents.

    Attributes:
        field: Name of the field containing dense vectors to search.
        values: Query vector as a list of floats.
    """

    field: str
    values: list[float]

    def __post_init__(self) -> None:
        if not self.field:
            raise ValueError(
                "dense_vector scoring requires specifying the name of exactly one field"
            )
        if not self.values:
            raise ValueError("dense_vector scoring requires a non-empty 'values' array")


class SparseVectorQuery(Struct, tag="sparse_vector", tag_field="type", kw_only=True):
    """Sparse vector similarity query for scoring documents.

    Attributes:
        field: Name of the field containing sparse vectors to search.
        sparse_values: Sparse vector with indices and values.
    """

    field: str
    sparse_values: SparseValues

    def __post_init__(self) -> None:
        if not self.field:
            raise ValueError(
                "sparse_vector scoring requires specifying the name of exactly one field"
            )


DocumentScoringMethod = TextQuery | QueryStringQuery | DenseVectorQuery | SparseVectorQuery
