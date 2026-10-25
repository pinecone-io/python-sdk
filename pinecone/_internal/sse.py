"""Assemble SSE data fields before decoding application payloads."""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator, Iterable, Iterator


class _EventData:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def feed(self, line: str) -> str | None:
        if not line:
            return self.finish()
        field, separator, value = line.partition(":")
        if field == "data":
            self.lines.append(value.removeprefix(" ") if separator else "")
        return None

    def finish(self) -> str | None:
        if not self.lines:
            return None
        data = "\n".join(self.lines)
        self.lines.clear()
        return data


def iter_sse_data(lines: Iterable[str]) -> Iterator[str]:
    event = _EventData()
    for line in lines:
        data = event.feed(line)
        if data:
            yield data
    # Preserve support for the backend's final event without a blank separator.
    data = event.finish()
    if data:
        yield data


async def aiter_sse_data(lines: AsyncIterable[str]) -> AsyncIterator[str]:
    event = _EventData()
    async for line in lines:
        data = event.feed(line)
        if data:
            yield data
    data = event.finish()
    if data:
        yield data
