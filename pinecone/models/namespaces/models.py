"""Namespace response models."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, overload

from msgspec import Struct

from pinecone.models._mixin import StructDictMixin
from pinecone.models.vectors.responses import Pagination


class NamespaceFieldConfig(StructDictMixin, Struct, kw_only=True):
    """Configuration for a single metadata field in a namespace schema.

    ``filterable`` defaults to ``False`` only so that responses decode when the
    server omits the flag. As a *request* value it is invalid: 2026-07 rejects
    any field config that does not carry ``filterable: true``. To leave a field
    unindexed, omit it from ``fields`` rather than sending ``filterable=False``.

    Attributes:
        filterable: Whether the field is indexed and usable in filters.
    """

    filterable: bool = False


class NamespaceSchema(StructDictMixin, Struct, kw_only=True):
    """Schema configuration for a namespace's metadata index."""

    fields: dict[str, NamespaceFieldConfig] = {}


class IndexedFields(StructDictMixin, Struct, kw_only=True):
    """List of indexed metadata fields in a namespace."""

    fields: list[str] = []


class NamespaceDescription(StructDictMixin, Struct, kw_only=True):
    """Description of a namespace including name, record count, size, and schema.

    Attributes:
        name: The name of the namespace.
        record_count: The total number of records in the namespace.
        schema: Schema configuration for metadata indexing, or None.
        indexed_fields: List of indexed metadata fields, or None.
        size_bytes: The total size of the namespace's data, in bytes. This is an
            approximation, not an exact byte count: data written before size
            tracking was enabled reads as 0, and recently deleted data may still
            be counted until compaction converges the value. Defaults to 0,
            which also covers API versions before 2026-07 that omit the field —
            a 0 therefore does not by itself mean the namespace is empty.
    """

    name: str = ""
    record_count: int = 0
    schema: NamespaceSchema | None = None
    indexed_fields: IndexedFields | None = None
    size_bytes: int = 0

    def __getitem__(self, key: str) -> Any:
        """Support bracket access (e.g. ns['name'])."""
        if key not in self.__struct_fields__:
            raise KeyError(key)
        return getattr(self, key)

    def __contains__(self, key: object) -> bool:
        """Support ``in`` operator (e.g. ``'name' in ns``)."""
        return key in self.__struct_fields__


class ListNamespacesResponse(StructDictMixin, Struct, kw_only=True):
    """Response from a list namespaces operation.

    Attributes:
        namespaces: List of namespace descriptions in this page.
        pagination: Pagination token for the next page, or None if last page.
        total_count: Total number of namespaces matching the query.
    """

    namespaces: list[NamespaceDescription] = []
    pagination: Pagination | None = None
    total_count: int = 0

    @overload
    def __getitem__(self, key: int) -> NamespaceDescription: ...

    @overload
    def __getitem__(self, key: str) -> Any: ...

    def __getitem__(self, key: int | str) -> Any:
        """Support integer indexing into namespaces and string bracket access.

        Args:
            key: An integer index into ``namespaces``, or a string field name.

        Returns:
            The namespace at the given index, or the field value.
        """
        if isinstance(key, int):
            return self.namespaces[key]
        if key not in self.__struct_fields__:
            raise KeyError(key)
        return getattr(self, key)

    def __contains__(self, key: object) -> bool:
        """Support ``in`` for field names (str) and namespace membership."""
        if isinstance(key, str):
            return key in self.__struct_fields__
        return key in self.namespaces

    def __len__(self) -> int:
        return len(self.namespaces)

    def __iter__(self) -> Iterator[NamespaceDescription]:  # type: ignore[override]
        return iter(self.namespaces)
