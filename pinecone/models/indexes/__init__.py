"""Index models, lazily imported.

:class:`~pinecone.models.indexes.index.IndexModel` is the entry point;
:class:`~pinecone.models.indexes.schema.IndexSchema` documents the field types
an index is built from.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pinecone.models.indexes.deployment import (  # noqa: F401
        ByocDeployment,
        IndexDeployment,
        ManagedDeployment,
        PodDeployment,
    )
    from pinecone.models.indexes.index import (  # noqa: F401
        ByocSpecInfo,
        IndexModel,
        IndexSpec,
        IndexStatus,
        IndexTags,
        ModelIndexEmbed,
        PodSpecInfo,
        ServerlessSpecInfo,
    )
    from pinecone.models.indexes.list import IndexList  # noqa: F401
    from pinecone.models.indexes.read_capacity import (  # noqa: F401
        ReadCapacityDedicatedConfig,
        ReadCapacityDedicatedResponse,
        ReadCapacityOnDemandResponse,
        ReadCapacityResponse,
        ReadCapacityStatus,
        ScalingConfigManual,
    )
    from pinecone.models.indexes.requests import (  # noqa: F401
        ConfigureIndexRequest,
        CreateIndexRequest,
    )
    from pinecone.models.indexes.schema import (  # noqa: F401
        BooleanField,
        DenseVectorField,
        FloatField,
        FullTextSearchConfig,
        IndexSchema,
        IndexSchemaField,
        IntegerField,
        LegacyMetadataField,
        NgramConfig,
        SemanticTextField,
        SparseVectorField,
        StringField,
        StringListField,
    )
    from pinecone.models.indexes.specs import (  # noqa: F401
        ByocSpec,
        EmbedConfig,
        IntegratedSpec,
        PodSpec,
        ServerlessSpec,
    )

_LAZY_IMPORTS: dict[str, str] = {
    "IndexModel": "pinecone.models.indexes.index",
    "IndexStatus": "pinecone.models.indexes.index",
    "IndexTags": "pinecone.models.indexes.index",
    "IndexSpec": "pinecone.models.indexes.index",
    "ServerlessSpecInfo": "pinecone.models.indexes.index",
    "PodSpecInfo": "pinecone.models.indexes.index",
    "ByocSpecInfo": "pinecone.models.indexes.index",
    "ModelIndexEmbed": "pinecone.models.indexes.index",
    "IndexList": "pinecone.models.indexes.list",
    "ByocDeployment": "pinecone.models.indexes.deployment",
    "IndexDeployment": "pinecone.models.indexes.deployment",
    "ManagedDeployment": "pinecone.models.indexes.deployment",
    "PodDeployment": "pinecone.models.indexes.deployment",
    "ReadCapacityDedicatedConfig": "pinecone.models.indexes.read_capacity",
    "ReadCapacityDedicatedResponse": "pinecone.models.indexes.read_capacity",
    "ReadCapacityOnDemandResponse": "pinecone.models.indexes.read_capacity",
    "ReadCapacityResponse": "pinecone.models.indexes.read_capacity",
    "ReadCapacityStatus": "pinecone.models.indexes.read_capacity",
    "ScalingConfigManual": "pinecone.models.indexes.read_capacity",
    "ConfigureIndexRequest": "pinecone.models.indexes.requests",
    "CreateIndexRequest": "pinecone.models.indexes.requests",
    "BooleanField": "pinecone.models.indexes.schema",
    "DenseVectorField": "pinecone.models.indexes.schema",
    "FloatField": "pinecone.models.indexes.schema",
    "FullTextSearchConfig": "pinecone.models.indexes.schema",
    "IndexSchema": "pinecone.models.indexes.schema",
    "IndexSchemaField": "pinecone.models.indexes.schema",
    "IntegerField": "pinecone.models.indexes.schema",
    "LegacyMetadataField": "pinecone.models.indexes.schema",
    "NgramConfig": "pinecone.models.indexes.schema",
    "SemanticTextField": "pinecone.models.indexes.schema",
    "SparseVectorField": "pinecone.models.indexes.schema",
    "StringField": "pinecone.models.indexes.schema",
    "StringListField": "pinecone.models.indexes.schema",
    "ServerlessSpec": "pinecone.models.indexes.specs",
    "PodSpec": "pinecone.models.indexes.specs",
    "ByocSpec": "pinecone.models.indexes.specs",
    "EmbedConfig": "pinecone.models.indexes.specs",
    "IntegratedSpec": "pinecone.models.indexes.specs",
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
