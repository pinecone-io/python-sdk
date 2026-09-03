"""The sparse half of a vector: the non-zero dimensions, named one by one."""

from __future__ import annotations

from typing import Any

from msgspec import Struct

from pinecone.models._mixin import DictLikeStruct


class SparseValues(DictLikeStruct, Struct, rename="camel", gc=False):
    """A sparse vector, given as its non-zero dimensions and their weights.

    A dense vector lists a float for every ``dimension``; a sparse vector lists only the
    dimensions that are not zero, as two parallel lists of the same length. Sparse vectors
    have no declared ``dimension``, so any index is legal and two sparse vectors in the
    same field need not name the same ones. Use one wherever a sparse component is asked
    for: :attr:`Vector.sparse_values <pinecone.models.vectors.vector.Vector.sparse_values>`
    when upserting, and the ``sparse_vector`` argument when querying.

    Attributes:
        indices (list[int]): The dimensions that carry a weight, typically the term slots a
            sparse embedding model or BM25 encoder produced.
        values (list[float]): The weight for each entry of ``indices``, positionally. The
            two lists must be the same length.

    Examples:
        >>> from pinecone import SparseValues
        >>> sparse = SparseValues(indices=[10, 42, 913], values=[0.4, 0.9, 0.2])
        >>> dict(zip(sparse.indices, sparse.values))
        {10: 0.4, 42: 0.9, 913: 0.2}
    """

    indices: list[int]
    values: list[float]

    @staticmethod
    def from_dict(sparse_values_dict: dict[str, Any]) -> SparseValues:
        """Build a :class:`SparseValues` from a plain dict.

        Args:
            sparse_values_dict (dict[str, Any]): Dict with ``indices`` and ``values`` keys,
                both required.

        Returns:
            :class:`SparseValues` carrying those two lists.

        Raises:
            KeyError: If either ``indices`` or ``values`` is absent.

        Examples:
            >>> from pinecone import SparseValues
            >>> SparseValues.from_dict({"indices": [10, 42], "values": [0.4, 0.9]}).indices
            [10, 42]
        """
        return SparseValues(
            indices=sparse_values_dict["indices"],
            values=sparse_values_dict["values"],
        )

    def __repr__(self) -> str:
        if len(self.indices) > 5:
            idx_preview = ", ".join(repr(v) for v in self.indices[:3])
            indices_str = f"[{idx_preview}, ...{len(self.indices) - 3} more]"
        else:
            indices_str = repr(self.indices)
        if len(self.values) > 5:
            val_preview = ", ".join(repr(v) for v in self.values[:3])
            values_str = f"[{val_preview}, ...{len(self.values) - 3} more]"
        else:
            values_str = repr(self.values)
        return f"SparseValues(indices={indices_str}, values={values_str})"
