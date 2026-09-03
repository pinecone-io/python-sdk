"""Namespace models package."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

_LAZY_IMPORTS: dict[str, str] = {
    "IndexedFields": "pinecone.models.namespaces.models",
    "ListNamespacesResponse": "pinecone.models.namespaces.models",
    "NamespaceDescription": "pinecone.models.namespaces.models",
    "NamespaceFieldConfig": "pinecone.models.namespaces.models",
    "NamespaceSchema": "pinecone.models.namespaces.models",
}

__all__ = list(_LAZY_IMPORTS.keys())

if TYPE_CHECKING:
    from pinecone.models.namespaces.models import (  # noqa: F401
        IndexedFields,
        ListNamespacesResponse,
        NamespaceDescription,
        NamespaceFieldConfig,
        NamespaceSchema,
    )


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        import importlib

        mod = importlib.import_module(_LAZY_IMPORTS[name])
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return list(_LAZY_IMPORTS.keys())
