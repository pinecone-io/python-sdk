"""Shared 2026-07 index-create shapes for the integration suite.

2026-07 replaced ``dimension=`` / ``metric=`` / ``spec=`` on index create with
``schema=`` + ``deployment=`` (see ``docs/migration/v10-migration.md``).
Modules in this package create throwaway indexes; the shapes live here rather
than being hand-rolled once per module.

The dense field name follows ``tests/integration/test_indexes.py``: ``embedding``.
"""

from __future__ import annotations

from typing import Any

DENSE_FIELD = "embedding"

MANAGED_AWS: dict[str, Any] = {
    "deployment_type": "managed",
    "cloud": "aws",
    "region": "us-east-1",
}


def dense_schema(
    dimension: int, metric: str = "cosine", *, field: str = DENSE_FIELD
) -> dict[str, Any]:
    return {"fields": {field: {"type": "dense_vector", "dimension": dimension, "metric": metric}}}
