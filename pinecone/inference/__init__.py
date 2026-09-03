"""Backwards-compatibility shim for :mod:`pinecone.inference`.

Re-exports classes that used to live at :mod:`pinecone.inference` in earlier
releases, so that code written against them keeps working. New code should
import from the canonical module.

:meta private:
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Backwards-compatibility re-exports (see docs/conventions/backcompat-shims.md)
# :meta private:
# ---------------------------------------------------------------------------
from pinecone.inference.inference import Inference as _Inference
from pinecone.inference.inference_asyncio import AsyncioInference as _AsyncioInference

Inference = _Inference
"""Backwards-compatibility alias. Canonical: :class:`pinecone.client.inference.Inference`.

:meta private:
"""
AsyncioInference = _AsyncioInference
"""Backwards-compatibility alias.

Canonical: :class:`pinecone.async_client.inference.AsyncInference`.

:meta private:
"""

__all__ = ["AsyncioInference", "Inference"]
