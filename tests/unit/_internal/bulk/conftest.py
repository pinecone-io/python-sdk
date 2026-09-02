"""Every test in this package must leave every gate quiescent.

This single fixture converts the whole slot-leak bug class — the reproduced
submit-raise deadlock, granted-but-never-observed slots, double-release —
into loud failures at the end of whichever test leaked, with pytest-timeout
as the backstop for the leaks that deadlock instead.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from pinecone._internal.bulk.registry import get_registry


@pytest.fixture(autouse=True)
def _quiescent_registry() -> Iterator[None]:
    registry = get_registry()
    registry._reset()
    yield
    leaked = {
        key: (gate.inflight, gate._core.waiting)
        for key, gate in registry._gates.items()
        if not gate.quiescent()
    }
    registry._reset()
    assert not leaked, f"gates not quiescent at teardown: {leaked}"
