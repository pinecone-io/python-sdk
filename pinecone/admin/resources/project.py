"""Legacy alias for :class:`~pinecone.admin.projects.Projects`, kept so
``from pinecone.admin.resources.project import ProjectResource`` keeps
working. Use :class:`~pinecone.admin.projects.Projects` in new code.

:meta private:
"""

from __future__ import annotations

from pinecone.admin.projects import Projects as ProjectResource

__all__ = ["ProjectResource"]
