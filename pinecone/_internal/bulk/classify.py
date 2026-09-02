"""The exception → (disposition, retryable) mapping. This table IS the
contract: ``BatchError.retryable`` is only trustworthy because these rules
are versioned and tested, so a consumer's ``if err.retryable: resend(...)``
loop cannot spin on a poison batch (a 400 that will never succeed).

Dispositions are an OPEN set — consumers must not match exhaustively.
Current values:

- ``rejected``  — the attempt completed with an error. The write may still
  have landed server-side (a response can be lost after the server applied
  it), so re-sending a retryable rejection is safe only because upserts are
  idempotent by vector id.
- ``unsent``    — a deadline expired before the batch was submitted; it was
  never sent at all.
- ``abandoned`` — the stall detector gave up on a backend that appears down;
  the batch was never sent.

``unsent`` and ``abandoned`` are assigned by the engine, which knows the
path taken; this module only classifies ``rejected`` errors' retryability.
"""

from __future__ import annotations

DISPOSITION_REJECTED = "rejected"
DISPOSITION_UNSENT = "unsent"
DISPOSITION_ABANDONED = "abandoned"

_RETRYABLE_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})


def is_retryable(error: BaseException) -> bool:
    """Whether re-sending the batch could plausibly succeed.

    Unknown exception types default to retryable: transient transport
    failures come in many shapes, and the known deterministic classes
    (validation, 4xx) are what poison-loop protection actually requires.
    """
    from pinecone.errors.exceptions import (
        ApiError,
        PineconeConnectionError,
        PineconeTimeoutError,
        PineconeTypeError,
        PineconeValueError,
    )

    if isinstance(error, (PineconeValueError, PineconeTypeError, TypeError, ValueError)):
        return False
    if isinstance(error, (PineconeConnectionError, PineconeTimeoutError)):
        return True
    if isinstance(error, ApiError):
        status = getattr(error, "status_code", None)
        if status is None:
            return True
        return status in _RETRYABLE_STATUS_CODES
    return True
