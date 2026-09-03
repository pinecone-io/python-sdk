"""Backwards-compatibility shim for :mod:`pinecone.async_client.inference`.

Re-exports the class that used to live at
:mod:`pinecone.inference.inference_asyncio` in earlier releases, so that code
written against it keeps working. New code should import from the canonical
module.

:meta private:
"""

from __future__ import annotations

from pinecone.async_client.inference import AsyncInference as AsyncioInference

__all__ = ["AsyncioInference"]
