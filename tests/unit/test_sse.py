from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from pinecone._internal.sse import aiter_sse_data, iter_sse_data


@pytest.mark.parametrize(
    ("lines", "expected"),
    [
        (["data: first", "data: second", "", "data: third", ""], ["first\nsecond", "third"]),
        ([": heartbeat", "event: message", "id: 42", "retry: 1000", ""], []),
        (["data:", "", "data", ""], []),
        (["data:  spaced", ""], [" spaced"]),
        (["data: final"], ["final"]),
        (["data: first", "data: final"], ["first\nfinal"]),
        (["", "", "data: one", "", "", "data: two", ""], ["one", "two"]),
    ],
)
async def test_event_data_parity(lines: list[str], expected: list[str]) -> None:
    async def async_lines() -> AsyncIterator[str]:
        for line in lines:
            yield line

    assert list(iter_sse_data(lines)) == expected
    assert [data async for data in aiter_sse_data(async_lines())] == expected


def test_emits_completed_event_before_reading_next_event() -> None:
    def lines():
        yield "data: first"
        yield ""
        raise AssertionError("read beyond the completed event")

    assert next(iter_sse_data(lines())) == "first"


async def test_async_emits_completed_event_before_reading_next_event() -> None:
    async def lines() -> AsyncIterator[str]:
        yield "data: first"
        yield ""
        raise AssertionError("read beyond the completed event")

    assert await anext(aiter_sse_data(lines())) == "first"
