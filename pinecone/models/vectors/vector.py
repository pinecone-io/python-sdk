"""Vector and ScoredVector response models."""

from __future__ import annotations

from typing import Any

from msgspec import Struct

from pinecone.models._mixin import DictLikeStruct
from pinecone.models.vectors.sparse import SparseValues


class Vector(DictLikeStruct, Struct, rename="camel", gc=False):
    """A stored vector with optional sparse values and metadata.

    At least one of ``values`` or ``sparse_values`` must be populated. ``values`` is not
    required on its own: a sparse-only vector leaves it empty, and an empty dense array is
    still sent so the pair reads as populated.

    Attributes:
        id (str): Unique identifier for the vector. ASCII, 1 to 512 characters, no NUL.
        values (list[float]): Dense vector values as a list of floats. Empty for a sparse-only
            vector, and empty on a response whenever values were not returned.
        sparse_values (SparseValues | None): Sparse vector component, or ``None`` if the vector
            has no sparse values.
        metadata (dict[str, Any] | None): User-defined metadata key-value pairs, or ``None`` if
            no metadata is attached. Each value must be a string, a number, a boolean, or a
            list of strings — nested objects and lists with a non-string element are rejected.
            A key whose value is ``None`` is dropped by the server rather than rejected. Keys
            may not begin with ``$``, which is reserved for filter operators; every other key is
            accepted, including empty and non-ASCII keys. The field is typed ``Any`` rather than
            narrowed to that grammar so that decoding a response never fails on a value shape
            the server has started returning; requests are validated on the way out instead.
    """

    id: str
    values: list[float] = []
    sparse_values: SparseValues | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Require at least one of ``values`` or ``sparse_values`` to be populated."""
        if not self.values and self.sparse_values is None:
            raise ValueError("Vector must have either values or sparse_values")

    @staticmethod
    def from_dict(vector_dict: dict[str, Any]) -> Vector:
        """Construct a ``Vector`` from a plain dict representation."""
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
    """A vector match with similarity score from a query operation.

    Attributes:
        id (str): Unique identifier of the matched vector.
        score (float): Similarity score for this match.
        values (list[float]): Dense vector values, or an empty list if values were not
            requested.
        sparse_values (SparseValues | None): Sparse vector component, or ``None`` if the vector
            has no sparse values.
        metadata (dict[str, Any] | None): User-defined metadata key-value pairs, or ``None`` if
            metadata was not requested or not attached. Values follow the same grammar as
            :attr:`Vector.metadata`: string, number, boolean, or list of strings.
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
