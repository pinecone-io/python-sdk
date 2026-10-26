"""Assistant SSE streams decode UTF-8 and strip only the initial byte order mark."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx
import orjson
import pytest
import respx

from pinecone import AsyncPinecone, Pinecone

BASE_URL = "https://api.test.pinecone.io"
ASSISTANT_NAME = "stream-encoding"


class ChunkedStream(httpx.SyncByteStream, httpx.AsyncByteStream):
    def __init__(self, parts: list[bytes]) -> None:
        self.parts = parts

    def __iter__(self) -> Iterator[bytes]:
        yield from self.parts

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for part in self.parts:
            yield part


def _response(method: str, variant: str) -> tuple[httpx.Response, str]:
    content = "你好 🌲\ufeff end"
    if method == "chat":
        chunk: dict[str, Any] = {
            "type": "content_chunk",
            "id": "chat-1",
            "delta": {"content": content},
        }
    else:
        chunk = {
            "id": "chat-1",
            "model": "gpt-4o",
            "choices": [{"index": 0, "delta": {"content": content}}],
        }
    body = b"data: " + orjson.dumps(chunk) + b"\n\n"
    if variant in ("bom", "split_bom", "bom_comment"):
        body = b"\xef\xbb\xbf" + (b": heartbeat\n\n" if variant == "bom_comment" else b"") + body
    elif variant in ("empty", "bom_empty"):
        body, content = (b"\xef\xbb\xbf" if variant == "bom_empty" else b""), ""
    elif variant in ("done", "bom_done"):
        body = (b"\xef\xbb\xbf" if variant == "bom_done" else b"") + b"data: [DONE]\n\n" + body
        content = ""
    body += b"data: [DONE]\n\n"
    parts = [body[:1], body[1:2], body[2:3], body[3:]] if variant == "split_bom" else [body]
    return (
        httpx.Response(
            200, headers={"Content-Type": "text/event-stream"}, stream=ChunkedStream(parts)
        ),
        content,
    )


def _routes(router: respx.MockRouter, method: str, response: httpx.Response) -> None:
    router.get(f"{BASE_URL}/assistant/assistants/{ASSISTANT_NAME}").respond(
        json={"name": ASSISTANT_NAME, "status": "Ready", "host": BASE_URL}
    )
    suffix = "/chat/completions" if method == "chat_completions" else ""
    router.post(f"{BASE_URL}/assistant/chat/{ASSISTANT_NAME}{suffix}").mock(return_value=response)


@pytest.mark.parametrize("method", ["chat", "chat_completions"])
@pytest.mark.parametrize(
    "variant",
    ["plain", "bom", "split_bom", "bom_comment", "empty", "done", "bom_empty", "bom_done"],
)
def test_sync_stream_encoding(method: str, variant: str) -> None:
    response, expected = _response(method, variant)
    with respx.mock() as router, Pinecone(api_key="test-key", host=BASE_URL) as pc:
        _routes(router, method, response)
        stream = getattr(pc.assistants, method)(
            assistant_name=ASSISTANT_NAME, messages=[{"content": "hello"}], stream=True
        )
        assert stream.collect() == expected


@pytest.mark.parametrize("method", ["chat", "chat_completions"])
@pytest.mark.parametrize(
    "variant",
    ["plain", "bom", "split_bom", "bom_comment", "empty", "done", "bom_empty", "bom_done"],
)
async def test_async_stream_encoding(method: str, variant: str) -> None:
    response, expected = _response(method, variant)
    with respx.mock() as router:
        _routes(router, method, response)
        async with AsyncPinecone(api_key="test-key", host=BASE_URL) as pc:
            stream = await getattr(pc.assistants, method)(
                assistant_name=ASSISTANT_NAME, messages=[{"content": "hello"}], stream=True
            )
            assert await stream.collect() == expected
