"""ModelInfoList wrapper for listing inference models."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, overload

from pinecone.models.inference.models import ModelInfo


class ModelInfoList:
    """What :meth:`~pinecone.client.inference.Inference.list_models` returns.

    Iterate it to reach each :class:`ModelInfo`; integer indexing, ``len()``
    and the string key ``["models"]`` work too. Returned by the SDK rather
    than constructed by callers.

    Attributes:
        models: The underlying list of :class:`ModelInfo` instances.

    Examples:
        >>> from pinecone import Pinecone
        >>> pc = Pinecone(api_key="your-api-key")
        >>> models = pc.inference.list_models()
        >>> for info in models:
        ...     print(info.model, info.type)
        multilingual-e5-large embed
        pinecone-sparse-english-v0 embed
        bge-reranker-v2-m3 rerank
    """

    def __init__(self, models: list[ModelInfo]) -> None:
        """Initialize a ModelInfoList.

        Args:
            models: List of :class:`ModelInfo` instances.
        """
        self._models = models

    @property
    def models(self) -> list[ModelInfo]:
        """Return the underlying list of models."""
        return self._models

    def names(self) -> list[str]:
        """Return just the model identifiers, in listing order.

        Returns:
            list[str]: The ``model`` field of each :class:`ModelInfo` — the
            names accepted by ``model=`` on
            :meth:`~pinecone.client.inference.Inference.embed` and
            :meth:`~pinecone.client.inference.Inference.rerank`.

        Examples:
            >>> from pinecone import Pinecone
            >>> pc = Pinecone(api_key="your-api-key")
            >>> models = pc.inference.list_models()
            >>> models.names()
            ['multilingual-e5-large', 'pinecone-sparse-english-v0', 'bge-reranker-v2-m3']
        """
        return [m.model for m in self._models]

    @overload
    def __getitem__(self, key: int) -> ModelInfo: ...

    @overload
    def __getitem__(self, key: str) -> list[ModelInfo]: ...

    def __getitem__(self, key: int | str) -> Any:
        """Support integer indexing and string key access.

        Args:
            key: An integer index into the models list, or the string
                ``"models"`` to get the full list.

        Returns:
            A :class:`ModelInfo` for integer keys, or ``list[ModelInfo]``
            for the ``"models"`` key.
        """
        if isinstance(key, int):
            return self._models[key]
        if key == "models":
            return self._models
        raise KeyError(key)

    def __len__(self) -> int:
        return len(self._models)

    def __iter__(self) -> Iterator[ModelInfo]:
        return iter(self._models)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict representation of this list."""
        return {"models": [m.to_dict() for m in self._models]}

    def __repr__(self) -> str:
        summaries = ", ".join(f"<model={m.model!r}, type={m.type!r}>" for m in self._models)
        return f"ModelInfoList([{summaries}])"
