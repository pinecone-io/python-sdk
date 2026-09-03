"""Exception hierarchy for the Pinecone SDK."""

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
