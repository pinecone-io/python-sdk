from __future__ import annotations

from pinecone.errors.exceptions import PineconeTimeoutError


def test_pinecone_timeout_error_extends_builtin_timeout_error() -> None:
    err = PineconeTimeoutError("timed out")
    assert isinstance(err, TimeoutError)
    assert isinstance(err, PineconeTimeoutError)


def test_pinecone_timeout_error_caught_by_builtin_timeout_error() -> None:
    caught = False
    try:
        raise PineconeTimeoutError("timed out")
    except TimeoutError:
        caught = True
    assert caught


def test_rate_limit_error_extends_api_error() -> None:
    from pinecone.errors.exceptions import ApiError, PineconeError, RateLimitError

    err = RateLimitError("rate limited")
    assert isinstance(err, ApiError)
    assert isinstance(err, PineconeError)


def test_rate_limit_error_caught_by_api_error() -> None:
    from pinecone.errors.exceptions import ApiError, RateLimitError

    caught = False
    try:
        raise RateLimitError("rate limited")
    except ApiError:
        caught = True
    assert caught
