"""Backwards-compatibility shim for :mod:`pinecone.client.inference`.

Re-exports the class that used to live at :mod:`pinecone.inference.inference`
in earlier releases, so that code written against it keeps working. New code
should import from the canonical module.

:meta private:
"""

from __future__ import annotations

from pinecone.client.inference import Inference

__all__ = ["Inference"]
