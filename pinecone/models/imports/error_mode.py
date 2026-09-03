"""How a bulk import reacts when one record in the source data will not load."""

from __future__ import annotations

from enum import Enum


class ImportErrorMode(str, Enum):
    """What a bulk import does when one record fails: skip it, or stop.

    Pass this as ``error_mode`` on ``start_import``. The choice is about how you would
    rather find out about bad data: ``ABORT`` surfaces the first bad record immediately
    and imports nothing, which suits data you expect to be clean; ``CONTINUE`` loads
    everything else and leaves you to reconcile what is missing, which suits a large
    source you would rather not restart.

    Attributes:
        CONTINUE: Skip the failing record and keep importing the rest.
        ABORT: Stop the whole import at the first failing record. This is what you get by
            omitting ``error_mode``.

    Examples:
        .. code-block:: python

            from pinecone import ImportErrorMode

            operation = idx.start_import(
                uri="s3://my-bucket/articles/", error_mode=ImportErrorMode.CONTINUE
            )

    .. seealso::
       :doc:`/guides/bulk-ingest` — the source-data layout an import expects.
    """

    CONTINUE = "continue"
    ABORT = "abort"


__all__ = ["ImportErrorMode"]
