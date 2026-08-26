"""Legacy alias for :class:`~pinecone.admin.api_keys.ApiKeys`, kept so
``from pinecone.admin.resources.api_key import ApiKeyResource`` keeps
working. Use :class:`~pinecone.admin.api_keys.ApiKeys` in new code.

:meta private:
"""

from __future__ import annotations

from pinecone.admin.api_keys import ApiKeys as ApiKeyResource

__all__ = ["ApiKeyResource"]
