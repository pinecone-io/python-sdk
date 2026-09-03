"""CollectionModel response model."""

from __future__ import annotations

from msgspec import Struct

from pinecone.models._mixin import StructDictMixin


class CollectionModel(StructDictMixin, Struct, kw_only=True):
    """One collection, as returned by
    :meth:`Collections.create() <pinecone.client.collections.Collections.create>`,
    :meth:`Collections.describe() <pinecone.client.collections.Collections.describe>`,
    and iteration over :class:`~pinecone.models.collections.list.CollectionList`.

    Only ``name``, ``status``, and ``environment`` are populated while the snapshot is
    still being built; read ``status`` before trusting the rest.

    Attributes:
        name: The name of the collection.
        status: Current status of the collection (e.g. ``"Ready"``,
            ``"Initializing"``, ``"Terminating"``).
        environment: Deployment environment where the collection is hosted.
        size: Space the snapshot occupies, in bytes — not a vector
            dimension. ``None`` until the collection is built.
        dimension: Dimensionality of vectors in the collection, or ``None``
            until the collection is built.
        vector_count: Number of vectors in the collection, or ``None`` until
            the collection is built.

    Examples:
        >>> col = pc.collections.describe("movie-embeddings-snapshot")
        >>> col.status, col.dimension, col.vector_count
        ('Ready', 1024, 99)
    """

    name: str
    status: str
    environment: str
    size: int | None = None
    dimension: int | None = None
    vector_count: int | None = None
