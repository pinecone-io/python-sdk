"""Rerank response models for the Inference API."""

from __future__ import annotations

from typing import Any, cast

import msgspec
from msgspec import Struct

from pinecone.models._mixin import StructDictMixin


class RerankUsage(StructDictMixin, Struct, kw_only=True):
    """Usage information for a rerank request.

    Attributes:
        rerank_units: Number of rerank units consumed.
    """

    rerank_units: int


class RankedDocument(StructDictMixin, Struct, kw_only=True):
    """One document and the score the reranker gave it.

    ``index`` is where the document sat in the request, not where it sits in
    the reordered result — it is what maps a result back onto the list you
    passed in.

    Attributes:
        index: The position this document held in the *documents* argument.
        score: The relevance the reranker assigned, higher being closer to the
            query.
        document: The document as sent, unless ``return_documents=False`` was
            passed, in which case ``None``.
    """

    index: int
    score: float
    document: dict[str, Any] | None = None

    def __repr__(self) -> str:
        if self.document is None:
            doc_str = "None"
        else:
            truncated = {
                k: (v[:80] + "..." if isinstance(v, str) and len(v) > 80 else v)
                for k, v in self.document.items()
            }
            doc_str = repr(truncated)
        return f"RankedDocument(index={self.index}, score={self.score}, document={doc_str})"


class RerankResult(Struct, kw_only=True):
    """What :meth:`~pinecone.client.inference.Inference.rerank` returns.

    Bracket access with a field name (``result["model"]``) reads the fields
    below. Returned by the SDK rather than constructed by callers.

    Attributes:
        model: The model that served the request, which is not always the one
            asked for — Pinecone may substitute a different model.
        data: The :class:`RankedDocument` results, ordered by descending
            ``score`` rather than by the order the documents were passed in.
        usage: Rerank usage, as ``usage.rerank_units``.

    Examples:
        >>> from pinecone import Pinecone
        >>> pc = Pinecone(api_key="your-api-key")
        >>> result = pc.inference.rerank(
        ...     model="bge-reranker-v2-m3",
        ...     query="Tell me about tech companies",
        ...     documents=["Apple is a fruit.", "Acme Inc. revolutionized tech."],
        ...     top_n=1,
        ... )
        >>> result.data[0].index
        1
        >>> result.data[0].document["text"]
        'Acme Inc. revolutionized tech.'
        >>> result["model"]
        'bge-reranker-v2-m3'
    """

    model: str
    data: list[RankedDocument]
    usage: RerankUsage

    def __getitem__(self, key: str) -> Any:
        """Support bracket access (e.g. result['model'])."""
        if key not in self.__struct_fields__:
            raise KeyError(key)
        return getattr(self, key)

    def __contains__(self, key: object) -> bool:
        """Support ``in`` operator (e.g. ``'model' in result``)."""
        return key in self.__struct_fields__

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict representation of this object."""
        return cast(dict[str, Any], msgspec.to_builtins(self))

    def __getattr__(self, name: str) -> Any:
        """Raise AttributeError for unknown attributes (backward compat hook)."""
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __repr__(self) -> str:
        return f"RerankResult(model={self.model!r}, count={len(self.data)}, usage={self.usage!r})"
