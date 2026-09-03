"""One page of bulk imports, wrapped so it reads like a list."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from pinecone.models.imports.model import ImportModel

if TYPE_CHECKING:
    from pinecone.models.vectors.responses import Pagination


class ImportList:
    """One page of :class:`ImportModel` objects, iterable and sized like a list.

    What ``list_imports_paginated`` returns. Iterate it, index into it, or take
    ``len()``; ``pagination`` carries the token for the next page, and is ``None`` when
    this is the last one.

    Attributes:
        pagination: Token for the next page, or ``None`` when there are no more.

    Examples:
        .. code-block:: python

            page = idx.list_imports_paginated()
            for operation in page:
                print(operation.id, operation.status, operation.percent_complete)

    .. seealso::
       :doc:`/guides/bulk-ingest` — starting and monitoring imports.
    """

    def __init__(
        self,
        imports: list[ImportModel],
        *,
        pagination: Pagination | None = None,
    ) -> None:
        """Wrap a page of imports.

        Args:
            imports (list[ImportModel]): The imports on this page.
            pagination (Pagination | None): Token for the next page, or ``None`` when this
                is the last page. Keyword-only.
        """
        self._imports = imports
        self.pagination = pagination

    def __iter__(self) -> Iterator[ImportModel]:
        return iter(self._imports)

    def __len__(self) -> int:
        return len(self._imports)

    def __getitem__(self, index: int) -> ImportModel:
        return self._imports[index]

    def to_dict(self) -> dict[str, Any]:
        """Return the page as a plain, JSON-serializable dict.

        Returns:
            A dict whose ``"data"`` key holds one dict per import, each from
            :meth:`ImportModel.to_dict`. A ``"pagination"`` key is present only when
            there is a next page.

        Examples:
            >>> idx = pc.index(name="article-search")
            >>> idx.list_imports_paginated().to_dict()
            {'data': []}
        """
        result: dict[str, Any] = {"data": [i.to_dict() for i in self._imports]}
        if self.pagination is not None:
            result["pagination"] = self.pagination.to_dict()
        return result

    def __repr__(self) -> str:
        summaries = ", ".join(
            f"<id={i.id!r}, status={i.status!r}, percent={i.percent_complete}>"
            for i in self._imports
        )
        return f"ImportList([{summaries}])"
