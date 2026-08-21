"""ImportErrorMode enum for bulk import operations.

Controls how the import pipeline handles individual record errors.
"""

from __future__ import annotations

from enum import Enum


class ImportErrorMode(str, Enum):
    """Behaviour when an individual record fails during a bulk import.

    Attributes:
        CONTINUE: Skip the failed record and continue importing remaining records.
        ABORT: Abort the entire import when any record fails. Omitting
            ``error_mode`` selects this.
    """

    CONTINUE = "continue"
    ABORT = "abort"


__all__ = ["ImportErrorMode"]
