"""Backwards-compatibility shim for :mod:`pinecone.inference.models`.

Kept as an importable package so that ``import pinecone.inference.models``
still succeeds for callers written against earlier releases. It re-exports
nothing: the names that used to live here have canonical homes under
:mod:`pinecone.models`.

:meta private:
"""

from __future__ import annotations

__all__: list[str] = []
