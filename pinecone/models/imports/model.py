"""The status of a bulk import, and the handle you get when you start one."""

from __future__ import annotations

from msgspec import Struct

from pinecone.models._mixin import StructDictMixin


class ImportModel(StructDictMixin, Struct, kw_only=True, rename="camel"):
    """The current state of one bulk import, as ``describe_import`` reports it.

    A bulk import runs server-side after you start it, so this is the model you poll.
    Three fields answer the questions worth asking: ``status`` says whether it is still
    running, ``percent_complete`` says how far along, and ``records_imported`` says how
    much has actually landed. Poll until ``status`` is a terminal value — ``Completed``,
    ``Failed``, or ``Cancelled`` — and read ``error`` when it is ``Failed``.

    Attributes:
        id: Identifier of the import, the value to pass back to ``describe_import`` and
            ``cancel_import``.
        uri: Where the data is being read from.
        status: ``Pending``, ``InProgress``, ``Failed``, ``Completed`` or ``Cancelled``.
            The first two mean keep polling; the rest are terminal.
        created_at: When the import was created, as a timestamp string.
        finished_at: When the import stopped running, or ``None`` while it is still
            running.
        percent_complete: How far along the import is, or ``None`` before the server has a
            figure. Progress alone is not completion — ``status`` is the authority.
        records_imported: Records written so far, or ``None`` before the server has a
            figure. A ``Failed`` import can leave this above zero, so a failure does not
            imply nothing was written.
        error: Why the import failed, or ``None``. Populated only for ``Failed``.

    Examples:
        .. code-block:: python

            operation = idx.describe_import(id=import_id)
            print(operation.status, operation.percent_complete, operation.records_imported)
            if operation.status == "Failed":
                print(operation.error)

    .. seealso::
       :doc:`/guides/bulk-ingest` — preparing the source data and choosing an error mode.
    """

    id: str
    uri: str
    status: str
    created_at: str
    finished_at: str | None = None
    percent_complete: float | None = None
    records_imported: int | None = None
    error: str | None = None


class StartImportResponse(StructDictMixin, Struct, kw_only=True):
    """The handle ``start_import`` returns: an ID, and nothing else yet.

    Starting an import is asynchronous, so this comes back before any data has been read.
    Keep the ``id`` — it is the only way to check on the import afterwards, or to cancel
    it.

    Attributes:
        id: Identifier of the import just created. Pass it to ``describe_import`` to poll
            :class:`ImportModel`, or to ``cancel_import`` to stop it.

    .. seealso::
       :doc:`/guides/bulk-ingest` — the whole start-then-poll flow.
    """

    id: str
