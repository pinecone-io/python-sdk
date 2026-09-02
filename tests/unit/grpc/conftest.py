from __future__ import annotations

from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True)
def _fresh_bulk_gate_registry() -> Iterator[None]:
    """The bulk gate registry is process-global; without a per-test reset, a
    gate throttled in one test would clamp admission in the next."""
    from pinecone._internal.bulk import get_registry

    get_registry()._reset()
    yield
    get_registry()._reset()
