"""Pagination response models for assistant list operations."""

from __future__ import annotations

from typing import Any

from msgspec import Struct

from pinecone.models._display import HtmlBuilder, abbreviate_list, safe_display
from pinecone.models.assistant._mixin import StructDictMixin
from pinecone.models.assistant.file_model import AssistantFileModel
from pinecone.models.assistant.model import AssistantModel
from pinecone.models.assistant.operation import OperationModel


class _Pagination(Struct, kw_only=True):
    """Wire-format pagination object nested in a list response."""

    next: str


class ListAssistantsResponse(StructDictMixin, Struct, kw_only=True):
    """One page of assistants, plus the token for the next one.

    Returned by :meth:`~pinecone.client.assistants.Assistants.list_page`. This
    is one page only — pass :attr:`next` back as ``pagination_token`` to
    advance, or call :meth:`~pinecone.client.assistants.Assistants.list`,
    which drives that loop for you and yields assistants directly.

    Attributes:
        assistants: The :class:`~pinecone.models.assistant.model.AssistantModel`
            objects on this page.
        pagination: The raw nested wire object. Read :attr:`next` instead.

    Examples:
        :attr:`next` is ``None`` on the last page, which is the loop's exit
        condition:

        >>> page = pc.assistants.list_page(page_size=10)
        >>> page.next is None
        True

    .. seealso::
       :doc:`/guides/pagination` — the continuation-token loop, and the
       paginator that drives it for you.
    """

    assistants: list[AssistantModel]
    pagination: _Pagination | None = None

    @property
    def next(self) -> str | None:
        """Token for the next page, or ``None`` when this is the last one.

        Pass a non-``None`` value back as the ``pagination_token`` argument of
        the same ``*_page`` method to fetch the following page.
        """
        return self.pagination.next if self.pagination is not None else None

    @property
    def next_token(self) -> str | None:
        """Alias for :attr:`next`. Prefer :attr:`next` in new code."""
        return self.next

    @safe_display
    def __repr__(self) -> str:
        return f"ListAssistantsResponse(count={len(self.assistants)}, next={self.next!r})"

    @safe_display
    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        if cycle:
            p.text("ListAssistantsResponse(...)")
            return
        preview = abbreviate_list(self.assistants, head=3, formatter=lambda a: a.name)
        with p.group(2, "ListAssistantsResponse(", ")"):
            p.breakable()
            p.text(f"count={len(self.assistants)},")
            p.breakable()
            p.text(f"next={self.next!r},")
            p.breakable()
            p.text(f"assistants={preview}")

    @safe_display
    def _repr_html_(self) -> str:
        next_display = self.next if self.next is not None else "—"
        builder = HtmlBuilder("ListAssistantsResponse")
        builder.row("Count:", len(self.assistants))
        builder.row("Next page token:", next_display)
        shown = self.assistants[:5]
        section_rows: list[tuple[str, Any]] = [(a.name, a.status) for a in shown]
        if len(self.assistants) > 5:
            section_rows.append(("...", f"{len(self.assistants) - 5} more"))
        builder.section("Assistants", section_rows)
        return builder.build()


class ListFilesResponse(StructDictMixin, Struct, kw_only=True):
    """One page of an assistant's files, plus the token for the next one.

    Returned by
    :meth:`~pinecone.client.assistants.Assistants.list_files_page`. This is
    one page only — pass :attr:`next` back as ``pagination_token`` to advance,
    or call :meth:`~pinecone.client.assistants.Assistants.list_files`, which
    drives that loop for you and yields files directly.

    Attributes:
        files: The
            :class:`~pinecone.models.assistant.file_model.AssistantFileModel`
            objects on this page.
        pagination: The raw nested wire object. Read :attr:`next` instead.

    Examples:
        A newly uploaded file appears here with ``status`` ``"Processing"``
        before it becomes ``"Available"``, so filter on it before treating a
        file as searchable:

        .. code-block:: python

            page = pc.assistants.list_files_page(
                assistant_name="acme-support-bot",
                page_size=10,
            )
            for file in page.files:
                print(file.name, file.status)

    .. seealso::
       :doc:`/guides/pagination` — the continuation-token loop, and the
       paginator that drives it for you.
    """

    files: list[AssistantFileModel]
    pagination: _Pagination | None = None

    @property
    def next(self) -> str | None:
        """Token for the next page, or ``None`` when this is the last one.

        Pass a non-``None`` value back as the ``pagination_token`` argument of
        the same ``*_page`` method to fetch the following page.
        """
        return self.pagination.next if self.pagination is not None else None

    @property
    def next_token(self) -> str | None:
        """Alias for :attr:`next`. Prefer :attr:`next` in new code."""
        return self.next

    @safe_display
    def __repr__(self) -> str:
        return f"ListFilesResponse(count={len(self.files)}, next={self.next!r})"

    @safe_display
    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        if cycle:
            p.text("ListFilesResponse(...)")
            return
        preview = abbreviate_list(self.files, head=3, formatter=lambda f: f.name)
        with p.group(2, "ListFilesResponse(", ")"):
            p.breakable()
            p.text(f"count={len(self.files)},")
            p.breakable()
            p.text(f"next={self.next!r},")
            p.breakable()
            p.text(f"files={preview}")

    @safe_display
    def _repr_html_(self) -> str:
        next_display = self.next if self.next is not None else "—"
        builder = HtmlBuilder("ListFilesResponse")
        builder.row("Count:", len(self.files))
        builder.row("Next page token:", next_display)
        shown = self.files[:5]
        section_rows: list[tuple[str, Any]] = [(f.name, f.status) for f in shown]
        if len(self.files) > 5:
            section_rows.append(("...", f"{len(self.files) - 5} more"))
        builder.section("Files", section_rows)
        return builder.build()


class ListOperationsResponse(StructDictMixin, Struct, kw_only=True):
    """One page of an assistant's operations, plus the token for the next one.

    Returned by
    :meth:`~pinecone.client.assistants.Assistants.list_operations_page`. This
    is one page only — pass :attr:`next` back as ``pagination_token`` to
    advance, or call
    :meth:`~pinecone.client.assistants.Assistants.list_operations`, which
    drives that loop for you and yields operations directly.

    Attributes:
        operations: The
            :class:`~pinecone.models.assistant.operation.OperationModel`
            objects on this page.
        pagination: The raw nested wire object. Read :attr:`next` instead.

    Examples:
        Operations are where file-processing progress and failure detail live,
        so this is the list to check when an upload has not become
        ``"Available"``:

        .. code-block:: python

            page = pc.assistants.list_operations_page(
                assistant_name="acme-support-bot",
                page_size=10,
            )
            for operation in page.operations:
                print(operation.operation_id, operation.status)
                if operation.status == "Failed":
                    print(operation.error)

    .. seealso::
       :doc:`/guides/pagination` — the continuation-token loop, and the
       paginator that drives it for you.
    """

    operations: list[OperationModel]
    pagination: _Pagination | None = None

    @property
    def next(self) -> str | None:
        """Token for the next page, or ``None`` when this is the last one.

        Pass a non-``None`` value back as the ``pagination_token`` argument of
        the same ``*_page`` method to fetch the following page.
        """
        return self.pagination.next if self.pagination is not None else None

    @property
    def next_token(self) -> str | None:
        """Alias for :attr:`next`. Prefer :attr:`next` in new code."""
        return self.next

    @safe_display
    def __repr__(self) -> str:
        return f"ListOperationsResponse(count={len(self.operations)}, next={self.next!r})"

    @safe_display
    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        if cycle:
            p.text("ListOperationsResponse(...)")
            return
        preview = abbreviate_list(self.operations, head=3, formatter=lambda o: o.operation_id)
        with p.group(2, "ListOperationsResponse(", ")"):
            p.breakable()
            p.text(f"count={len(self.operations)},")
            p.breakable()
            p.text(f"next={self.next!r},")
            p.breakable()
            p.text(f"operations={preview}")

    @safe_display
    def _repr_html_(self) -> str:
        next_display = self.next if self.next is not None else "—"
        builder = HtmlBuilder("ListOperationsResponse")
        builder.row("Count:", len(self.operations))
        builder.row("Next page token:", next_display)
        shown = self.operations[:5]
        section_rows: list[tuple[str, Any]] = [(o.operation_id, o.status) for o in shown]
        if len(self.operations) > 5:
            section_rows.append(("...", f"{len(self.operations) - 5} more"))
        builder.section("Operations", section_rows)
        return builder.build()
