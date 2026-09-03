"""What ``describe`` and ``list`` on ``index.namespaces`` hand back."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, overload

from msgspec import Struct

from pinecone.models._mixin import StructDictMixin
from pinecone.models.vectors.responses import Pagination


class NamespaceFieldConfig(StructDictMixin, Struct, kw_only=True):
    """Whether one metadata field is indexed for filtering.

    ``filterable`` defaults to ``False`` only so a response decodes when the server omits
    the flag. As a *request* value ``False`` is rejected — the only accepted value is
    ``True``. To leave a field unindexed, omit it from ``fields`` rather than sending
    ``filterable=False``.

    Attributes:
        filterable: Whether the field is indexed and can appear in a filter.
    """

    filterable: bool = False


class NamespaceSchema(StructDictMixin, Struct, kw_only=True):
    """Which metadata fields a namespace indexes for filtering.

    Attributes:
        fields: Field name to its :class:`NamespaceFieldConfig`. A field absent here is
            stored but cannot be filtered on.
    """

    fields: dict[str, NamespaceFieldConfig] = {}


class IndexedFields(StructDictMixin, Struct, kw_only=True):
    """The indexed metadata field names, without the per-field configuration.

    Attributes:
        fields: The names of the fields that can appear in a filter.
    """

    fields: list[str] = []


class NamespaceDescription(StructDictMixin, Struct, kw_only=True):
    """One namespace: its name, how much is in it, and which fields it indexes.

    Attributes:
        name: The namespace's name. ``""`` is the default namespace.
        record_count: Records in the namespace. Eventually consistent, so a record you
            just wrote may not be counted yet.
        schema: Which metadata fields are indexed for filtering, or ``None`` when the
            namespace has no schema.
        indexed_fields: The same field names without the per-field configuration, or
            ``None``.
        size_bytes: The total size of the namespace's data, in bytes. This is an
            approximation, not an exact byte count: data written before size
            tracking was enabled reads as 0, and recently deleted data may still
            be counted until compaction converges the value. Defaults to 0,
            which also covers responses that omit the field entirely — a 0 therefore
            does not by itself mean the namespace is empty.
    """

    name: str = ""
    record_count: int = 0
    schema: NamespaceSchema | None = None
    indexed_fields: IndexedFields | None = None
    size_bytes: int = 0

    def __getitem__(self, key: str) -> Any:
        """Read a field by name, so ``ns["name"]`` works as well as ``ns.name``.

        Raises:
            KeyError: If *key* is not one of this model's fields.
        """
        if key not in self.__struct_fields__:
            raise KeyError(key)
        return getattr(self, key)

    def __contains__(self, key: object) -> bool:
        """Report whether *key* names a field on this description."""
        return key in self.__struct_fields__


class ListNamespacesResponse(StructDictMixin, Struct, kw_only=True):
    """One page of namespace descriptions.

    Iterable and sized directly, so ``for ns in response`` and ``len(response)`` walk this
    page. ``total_count`` counts every matching namespace, not just this page, so compare
    the two to tell whether more pages remain — or just follow ``pagination`` until it is
    ``None``.

    Attributes:
        namespaces: The :class:`NamespaceDescription` entries on this page.
        pagination: Token for the next page, or ``None`` when this is the last page.
        total_count: Namespaces matching the request, across every page.

    .. seealso::
       :doc:`/guides/pagination` — the paging loop used across the SDK.
    """

    namespaces: list[NamespaceDescription] = []
    pagination: Pagination | None = None
    total_count: int = 0

    @overload
    def __getitem__(self, key: int) -> NamespaceDescription: ...

    @overload
    def __getitem__(self, key: str) -> Any: ...

    def __getitem__(self, key: int | str) -> Any:
        """Index into the page's namespaces, or read a field by name.

        Args:
            key (int | str): An integer position in ``namespaces``, or the name of a field
                on this response.

        Returns:
            The :class:`NamespaceDescription` at that position, or the named field's value.

        Raises:
            KeyError: If a string *key* does not name a field.
            IndexError: If an integer *key* is past the end of this page.
        """
        if isinstance(key, int):
            return self.namespaces[key]
        if key not in self.__struct_fields__:
            raise KeyError(key)
        return getattr(self, key)

    def __contains__(self, key: object) -> bool:
        """Report field-name membership for a string, and namespace membership otherwise."""
        if isinstance(key, str):
            return key in self.__struct_fields__
        return key in self.namespaces

    def __len__(self) -> int:
        return len(self.namespaces)

    def __iter__(self) -> Iterator[NamespaceDescription]:  # type: ignore[override]
        return iter(self.namespaces)
