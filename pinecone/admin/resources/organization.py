"""Legacy alias for :class:`~pinecone.admin.organizations.Organizations`,
kept so ``from pinecone.admin.resources.organization import
OrganizationResource`` keeps working. Use
:class:`~pinecone.admin.organizations.Organizations` in new code.

:meta private:
"""

from __future__ import annotations

from pinecone.admin.organizations import Organizations as OrganizationResource

__all__ = ["OrganizationResource"]
