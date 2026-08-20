"""Backup models subpackage with lazy loading."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pinecone.models.backups.list import BackupList, RestoreJobList  # noqa: F401
    from pinecone.models.backups.model import (  # noqa: F401
        BackupModel,
        CreateIndexFromBackupRequest,
        CreateIndexFromBackupResponse,
        RestoreJobModel,
    )

_LAZY_IMPORTS: dict[str, str] = {
    "BackupModel": "pinecone.models.backups.model",
    "RestoreJobModel": "pinecone.models.backups.model",
    "CreateIndexFromBackupRequest": "pinecone.models.backups.model",
    "CreateIndexFromBackupResponse": "pinecone.models.backups.model",
    "BackupList": "pinecone.models.backups.list",
    "RestoreJobList": "pinecone.models.backups.list",
}

__all__ = list(_LAZY_IMPORTS.keys())


def __getattr__(name: str) -> Any:
    """Lazy-load models on first access."""
    if name in _LAZY_IMPORTS:
        from importlib import import_module

        module = import_module(_LAZY_IMPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    import builtins

    return builtins.list({*globals(), *__all__, *_LAZY_IMPORTS})
