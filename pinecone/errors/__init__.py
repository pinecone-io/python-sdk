"""Import path for the SDK's exceptions.

``from pinecone.errors import NotFoundError`` and
``from pinecone import NotFoundError`` reach the same class. The classes are
defined in :mod:`pinecone.errors.exceptions`, and
:doc:`/guides/error-handling` covers which call produces which.
"""

from __future__ import annotations

from pinecone.errors.exceptions import (
    ApiError,
    ConflictError,
    FailedPreconditionError,
    ForbiddenError,
    IndexInitFailedError,
    IndexTerminatedError,
    NotFoundError,
    PaymentRequiredError,
    PineconeConnectionError,
    PineconeError,
    PineconeTimeoutError,
    PineconeTypeError,
    PineconeValueError,
    RateLimitError,
    RateLimitException,
    ResponseParsingError,
    ServiceError,
    UnauthorizedError,
    ValidationError,
)

__all__ = [
    "ApiError",
    "ConflictError",
    "FailedPreconditionError",
    "ForbiddenError",
    "IndexInitFailedError",
    "IndexTerminatedError",
    "NotFoundError",
    "PaymentRequiredError",
    "PineconeConnectionError",
    "PineconeError",
    "PineconeTimeoutError",
    "PineconeTypeError",
    "PineconeValueError",
    "RateLimitError",
    "RateLimitException",
    "ResponseParsingError",
    "ServiceError",
    "UnauthorizedError",
    "ValidationError",
]
