"""Registry behavior: key normalization inside the boundary (the #60 class),
quiescent-only eviction, and the end-to-end assertion issue #60 proves was
never written — a throttle reported via the callback path must move the
limit of the gate the bulk path registered."""

from __future__ import annotations

import hypothesis.strategies as st
from hypothesis import given

from pinecone._internal.bulk.registry import GateRegistry, host_key

CANONICAL = "test-index-abc1234.svc.us-east1-gcp.pinecone.io"

_scheme = st.sampled_from(["", "https://", "http://"])
_port = st.sampled_from(["", ":443", ":80", ":5081"])
_path = st.sampled_from(["", "/", "/vectors/upsert", "?q=1", "#frag"])
_case = st.booleans()
_dot = st.sampled_from(["", "."])


@given(scheme=_scheme, port=_port, path=_path, upper=_case, dot=_dot)
def test_host_key_equivalence_across_variants(
    scheme: str, port: str, path: str, upper: bool, dot: str
) -> None:
    host = CANONICAL.upper() if upper else CANONICAL
    variant = f"{scheme}{host}{dot}{port}{path}"
    assert host_key(variant) == CANONICAL


@given(scheme=_scheme, port=_port, path=_path, upper=_case, dot=_dot)
def test_host_key_is_idempotent(scheme: str, port: str, path: str, upper: bool, dot: str) -> None:
    host = CANONICAL.upper() if upper else CANONICAL
    variant = f"{scheme}{host}{dot}{port}{path}"
    assert host_key(host_key(variant)) == host_key(variant)


def test_variants_resolve_to_one_gate() -> None:
    registry = GateRegistry()
    a = registry.get(f"https://{CANONICAL}")
    b = registry.get(CANONICAL)
    c = registry.get(f"HTTP://{CANONICAL.upper()}:443/vectors/upsert")
    assert a is b is c


def test_reported_throttle_moves_the_registered_limit() -> None:
    registry = GateRegistry()
    gate = registry.get(f"https://{CANONICAL}")
    before = gate.limit
    registry.report_throttled(CANONICAL)
    assert gate.limit < before


def test_throttle_before_any_get_is_not_lost() -> None:
    registry = GateRegistry()
    registry.report_throttled(f"https://{CANONICAL}", pushback_seconds=None)
    gate = registry.get(CANONICAL)
    assert gate.limit < 64


def test_eviction_skips_non_quiescent_gates() -> None:
    registry = GateRegistry()
    import pinecone._internal.bulk.registry as reg_mod

    original = reg_mod._MAX_GATES
    reg_mod._MAX_GATES = 2
    try:
        busy = registry.get("busy.example.pinecone.io")
        _, slot = busy.acquire()
        assert slot is not None
        registry.get("idle-a.example.pinecone.io")
        registry.get("idle-b.example.pinecone.io")
        assert registry.get("busy.example.pinecone.io") is busy
        slot.release()
    finally:
        reg_mod._MAX_GATES = original
    registry._reset()


def test_fork_reset_rebuilds_without_taking_the_lock() -> None:
    registry = GateRegistry()
    registry.get(CANONICAL)
    registry._lock.acquire()
    registry._reset_unlocked()
    assert registry._gates == {}
    with registry._lock:
        pass
