"""Embedding response models for the Inference API."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, cast, overload

import msgspec
from msgspec import Struct

from pinecone.models._mixin import DictLikeStruct, StructDictMixin


class EmbedUsage(StructDictMixin, Struct, kw_only=True):
    """Token usage information for an embedding request.

    Attributes:
        total_tokens: Total number of tokens processed.
    """

    total_tokens: int


class DenseEmbedding(DictLikeStruct, Struct, kw_only=True):
    """One embedding from a dense model, as a list of floats.

    ``values`` is the vector, ready to pass to
    :meth:`~pinecone.Index.upsert` or as a query vector. Its length is the
    model's output dimension, which
    :meth:`~pinecone.client.inference.Inference.get_model` reports as
    ``default_dimension``.

    Attributes:
        values: The embedding, one float per dimension.
        vector_type: Always ``"dense"``.
    """

    values: list[float]
    vector_type: str = "dense"

    def __repr__(self) -> str:
        if len(self.values) > 5:
            preview = ", ".join(repr(v) for v in self.values[:3])
            values_str = f"[{preview}, ...{len(self.values) - 3} more]"
        else:
            values_str = repr(self.values)
        return f"DenseEmbedding(values={values_str}, vector_type={self.vector_type!r})"


class SparseEmbedding(StructDictMixin, Struct, kw_only=True):
    """One embedding from a sparse model, stored as index/value pairs.

    There is no ``values`` field here — the vector lives in
    ``sparse_indices`` and ``sparse_values``, paired position by position.
    Reading ``.values`` on one of these hands back a dict-view method rather
    than a vector and raises nothing to warn you, so branch on the enclosing
    :class:`EmbeddingsList`'s ``vector_type`` when the model is not fixed in
    advance.

    Attributes:
        sparse_values: The non-zero values of the sparse embedding.
        sparse_indices: The index each of those values sits at.
        sparse_tokens: The token each index came from, when the model reports
            them; ``None`` otherwise.
        vector_type: Always ``"sparse"``.
    """

    sparse_values: list[float]
    sparse_indices: list[int]
    sparse_tokens: list[str] | None = None
    vector_type: str = "sparse"

    def __repr__(self) -> str:
        if len(self.sparse_indices) > 5:
            idx_preview = ", ".join(repr(v) for v in self.sparse_indices[:3])
            indices_str = f"[{idx_preview}, ...{len(self.sparse_indices) - 3} more]"
        else:
            indices_str = repr(self.sparse_indices)
        if len(self.sparse_values) > 5:
            val_preview = ", ".join(repr(v) for v in self.sparse_values[:3])
            values_str = f"[{val_preview}, ...{len(self.sparse_values) - 3} more]"
        else:
            values_str = repr(self.sparse_values)
        parts = [
            f"sparse_indices={indices_str}",
            f"sparse_values={values_str}",
            f"vector_type={self.vector_type!r}",
        ]
        if self.sparse_tokens is not None:
            parts.insert(2, f"sparse_tokens={self.sparse_tokens!r}")
        return f"SparseEmbedding({', '.join(parts)})"


Embedding = DenseEmbedding | SparseEmbedding


class EmbeddingsList(Struct, kw_only=True):
    """What :meth:`~pinecone.client.inference.Inference.embed` returns.

    One embedding per input, in the order the inputs were given. Iterating the
    list is the usual way in; integer indexing and ``len()`` reach the same
    items, and bracket access with a field name (``embeddings["model"]``) reads
    the fields below. Returned by the SDK rather than constructed by callers.

    Attributes:
        model: The model that served the request.
        vector_type: ``"dense"`` or ``"sparse"`` — which of
            :class:`DenseEmbedding` or :class:`SparseEmbedding` ``data`` holds,
            and so which fields each item carries.
        data: The embeddings themselves.
        usage: Token usage, as ``usage.total_tokens``.

    Examples:
        >>> from pinecone import Pinecone
        >>> pc = Pinecone(api_key="your-api-key")
        >>> embeddings = pc.inference.embed(
        ...     model="multilingual-e5-large",
        ...     inputs=[
        ...         "Vector databases index embeddings for similarity search.",
        ...         "Reranking reorders candidate results by relevance.",
        ...     ],
        ...     parameters={"input_type": "passage"},
        ... )
        >>> len(embeddings)
        2
        >>> [embedding.vector_type for embedding in embeddings]
        ['dense', 'dense']
        >>> embeddings["model"]
        'multilingual-e5-large'
    """

    model: str
    vector_type: str
    data: list[DenseEmbedding] | list[SparseEmbedding]
    usage: EmbedUsage

    @overload
    def __getitem__(self, key: int) -> DenseEmbedding | SparseEmbedding: ...

    @overload
    def __getitem__(self, key: str) -> Any: ...

    def __getitem__(self, key: int | str) -> Any:
        """Support integer indexing into data and string bracket access.

        Args:
            key: An integer index into ``data``, or a string field name.

        Returns:
            The embedding at the given index, or the field value.
        """
        if isinstance(key, int):
            return self.data[key]
        if key not in self.__struct_fields__:
            raise KeyError(key)
        return getattr(self, key)

    def __contains__(self, key: object) -> bool:
        """Support ``in`` for field names (str) and embedding membership."""
        if isinstance(key, str):
            return key in self.__struct_fields__
        return key in self.data

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self) -> Iterator[DenseEmbedding | SparseEmbedding]:
        return iter(self.data)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict representation of this object."""
        return cast(dict[str, Any], msgspec.to_builtins(self))

    def __getattr__(self, name: str) -> Any:
        """Raise AttributeError for unknown attributes (backward compat hook)."""
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __repr__(self) -> str:
        return (
            f"EmbeddingsList("
            f"model={self.model!r}, "
            f"vector_type={self.vector_type!r}, "
            f"count={len(self.data)}, "
            f"usage={self.usage!r})"
        )
