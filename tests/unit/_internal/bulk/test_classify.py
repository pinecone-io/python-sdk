"""The exception → retryable mapping is a versioned contract: exhaustive
tests over the exception taxonomy are what make ``BatchError.retryable``
trustworthy enough to drive a blind retry loop."""

from __future__ import annotations

import pytest

from pinecone._internal.bulk.classify import (
    DISPOSITION_ABANDONED,
    DISPOSITION_REJECTED,
    DISPOSITION_UNSENT,
    is_retryable,
)
from pinecone.errors.exceptions import (
    ApiError,
    PineconeConnectionError,
    PineconeTimeoutError,
    PineconeTypeError,
    PineconeValueError,
)


def _api_error(status: int) -> ApiError:
    return ApiError(message="err", status_code=status)


@pytest.mark.parametrize("status", [408, 425, 429, 500, 502, 503, 504])
def test_transient_statuses_are_retryable(status: int) -> None:
    assert is_retryable(_api_error(status)) is True


@pytest.mark.parametrize("status", [400, 401, 403, 404, 409, 413, 422])
def test_deterministic_statuses_are_not_retryable(status: int) -> None:
    assert is_retryable(_api_error(status)) is False


@pytest.mark.parametrize(
    "error",
    [
        PineconeConnectionError("connection refused"),
        PineconeTimeoutError("request timed out"),
        RuntimeError("unknown transport failure"),
        OSError("broken pipe"),
    ],
)
def test_transient_shapes_are_retryable(error: Exception) -> None:
    assert is_retryable(error) is True


@pytest.mark.parametrize(
    "error",
    [
        PineconeValueError("vector dimension mismatch"),
        PineconeTypeError("values must be a list of floats"),
        ValueError("bad value"),
        TypeError("bad type"),
    ],
)
def test_validation_shapes_are_poison(error: Exception) -> None:
    assert is_retryable(error) is False


def test_disposition_constants_are_distinct_strings() -> None:
    values = {DISPOSITION_REJECTED, DISPOSITION_UNSENT, DISPOSITION_ABANDONED}
    assert values == {"rejected", "unsent", "abandoned"}


def test_async_engine_classifies_retryable() -> None:
    """The table is only worth versioning if the engine actually consults it,
    so this drives a poison batch all the way through the async engine rather
    than calling ``is_retryable`` directly."""
    import asyncio

    from pinecone._internal.bulk.async_engine import bulk_execute_async
    from pinecone.errors.exceptions import PineconeValueError

    async def poison_op(batch: list[dict[str, object]]) -> object:
        raise PineconeValueError("dimension mismatch")

    async def run() -> object:
        return await bulk_execute_async(
            items=[{"id": "1"}, {"id": "2"}],
            operation=poison_op,
            batch_size=1,
            show_progress=False,
            host="classify-async.svc.pinecone.io",
        )

    result = asyncio.run(run())
    assert result.errors
    assert all(e.retryable is False for e in result.errors)


def test_batch_result_serialization_includes_counters() -> None:
    from pinecone._internal.bulk.engine import bulk_execute_sync

    result = bulk_execute_sync(
        items=[{"id": "1"}],
        operation=lambda b: {"upserted_count": len(b)},
        batch_size=1,
        show_progress=False,
        host="serialize-test.svc.pinecone.io",
    )
    d = result.to_dict()
    assert "throttle_event_count" in d
    assert "final_limit" in d
    assert "peak_inflight" in d
    assert d["final_limit"] == result.final_limit
