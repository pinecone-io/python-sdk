"""Process-global registry of per-host gates.

Process-global (not per-client) because the limit is a statement about the
backend, not about a client object: two ``Pinecone()`` instances in one
process hitting one host share one backend cell, and per-client state let
each instance run the full limit (the #56 shape, one level up) and reset
adaptive state on client churn. The coupling errs fail-safe — shared state
can only under-load a host, never over-load it.

Host keys are normalized HERE, not at call sites: issue #60 happened even
though a normalization helper existed, because a helper offered at call
sites is exactly the shape that gets missed. An unnormalized caller cannot
miss a normalization applied inside ``get`` and ``report_throttled``.

Fork safety: gates hold a ``threading.Lock`` and live in-flight counts. A
forked child (gunicorn, multiprocessing) inheriting them would start life
with possibly-held locks and phantom in-flight slots — a permanent leak at
best, a deadlock at the floor at worst — so the registry resets itself in
the child via ``os.register_at_fork``.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict

from pinecone._internal.bulk.gate import HostGate

_MAX_GATES = 1024


def host_key(host: str) -> str:
    """One canonical key: bare lowercase hostname — no scheme, port, path,
    userinfo, or trailing FQDN dot. Throttle callbacks report bare hostnames;
    any variant registering elsewhere gets a gate that never sees a throttle."""
    bare = host.strip()
    lowered = bare.lower()
    for prefix in ("https://", "http://"):
        if lowered.startswith(prefix):
            bare = bare[len(prefix) :]
            break
    if "@" in bare:
        bare = bare.rsplit("@", 1)[1]
    for separator in ("/", "?", "#", ":"):
        bare = bare.split(separator, 1)[0]
    return bare.lower().rstrip(".")


class GateRegistry:
    __slots__ = ("_gates", "_lock")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._gates: OrderedDict[str, HostGate] = OrderedDict()

    def get(self, host: str) -> HostGate:
        key = host_key(host)
        with self._lock:
            gate = self._gates.get(key)
            if gate is None:
                gate = HostGate()
                self._evict_quiescent_locked()
                self._gates[key] = gate
            else:
                self._gates.move_to_end(key)
            return gate

    def report_throttled(self, host: str, pushback_seconds: float | None = None) -> None:
        key = host_key(host)
        with self._lock:
            gate = self._gates.get(key)
            if gate is None:
                gate = HostGate()
                self._evict_quiescent_locked()
                self._gates[key] = gate
            else:
                self._gates.move_to_end(key)
        gate.report_throttled(pushback_seconds)

    def _evict_quiescent_locked(self) -> None:
        """LRU eviction, but only among quiescent gates — evicting a gate
        with live in-flight counts splits the counter between the evicted
        object and its replacement (the pool-cache bug's shape recurring)."""
        if len(self._gates) < _MAX_GATES:
            return
        for key in list(self._gates):
            if self._gates[key].quiescent():
                del self._gates[key]
                return

    def _reset(self) -> None:
        with self._lock:
            self._gates.clear()

    def _reset_unlocked(self) -> None:
        """Fork-child reset: the lock state inherited across fork is
        undefined (another thread may have held it at fork time), so the
        child rebuilds without trying to take it."""
        self._lock = threading.Lock()
        self._gates = OrderedDict()


_registry = GateRegistry()


def get_registry() -> GateRegistry:
    return _registry


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_registry._reset_unlocked)
