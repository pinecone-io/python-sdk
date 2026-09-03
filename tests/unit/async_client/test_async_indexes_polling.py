"""Unit tests for async create-readiness polling edge cases (2026-07 create).

Async mirror of tests/unit/client/test_indexes_polling.py, verifying the
polling loop awaits asyncio.sleep (no event-loop blocking) and raises the
same errors as the sync lane.
"""

from __future__ import annotations

import itertools
from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import respx

from pinecone._internal.config import PineconeConfig
from pinecone._internal.constants import CONTROL_PLANE_API_VERSION
from pinecone._internal.http_client import AsyncHTTPClient
from pinecone.async_client.indexes import _POLL_INTERVAL_SECONDS, AsyncIndexes
from pinecone.errors.exceptions import (
    ConflictError,
    IndexInitFailedError,
    IndexTerminatedError,
    PineconeTimeoutError,
)
from tests.factories import make_error_response, make_index_response

BASE_URL = "https://api.test.pinecone.io"

DENSE_SCHEMA: dict[str, Any] = {
    "fields": {"embedding": {"type": "dense_vector", "dimension": 1536, "metric": "cosine"}}
}


@pytest.fixture
async def async_http_client() -> AsyncGenerator[AsyncHTTPClient]:
    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    client = AsyncHTTPClient(config, CONTROL_PLANE_API_VERSION)
    yield client
    await client.close()


@pytest.fixture
def indexes(async_http_client: AsyncHTTPClient) -> AsyncIndexes:
    return AsyncIndexes(http=async_http_client)


def test_poll_interval_is_five_seconds() -> None:
    assert _POLL_INTERVAL_SECONDS == 5


@respx.mock
async def test_create_polling_awaits_asyncio_sleep_five_seconds(indexes: AsyncIndexes) -> None:
    """Polling awaits asyncio.sleep(5) — it never blocks the event loop."""
    respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(
            201, json=make_index_response(status={"ready": False, "state": "Initializing"})
        )
    )
    respx.get(f"{BASE_URL}/indexes/test-index").mock(
        side_effect=[
            httpx.Response(
                200, json=make_index_response(status={"ready": False, "state": "Initializing"})
            ),
            httpx.Response(200, json=make_index_response(status={"ready": True, "state": "Ready"})),
        ]
    )

    with patch(
        "pinecone._internal.indexes_helpers.asyncio.sleep", new_callable=AsyncMock
    ) as mock_sleep:
        await indexes.create(name="test-index", schema=DENSE_SCHEMA, timeout=300)

    assert mock_sleep.await_count == 1
    mock_sleep.assert_awaited_with(5)


@respx.mock
async def test_create_init_failed_raises_immediately(indexes: AsyncIndexes) -> None:
    respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(
            201, json=make_index_response(status={"ready": False, "state": "Initializing"})
        )
    )
    respx.get(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(
            200,
            json=make_index_response(status={"ready": False, "state": "InitializationFailed"}),
        )
    )

    with (
        patch("pinecone._internal.indexes_helpers.asyncio.sleep", new_callable=AsyncMock),
        pytest.raises(IndexInitFailedError) as exc_info,
    ):
        await indexes.create(name="test-index", schema=DENSE_SCHEMA, timeout=300)

    assert exc_info.value.index_name == "test-index"


@respx.mock
async def test_create_terminating_raises(indexes: AsyncIndexes) -> None:
    respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(
            201, json=make_index_response(status={"ready": False, "state": "Initializing"})
        )
    )
    respx.get(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(
            200, json=make_index_response(status={"ready": False, "state": "Terminating"})
        )
    )

    with (
        patch("pinecone._internal.indexes_helpers.asyncio.sleep", new_callable=AsyncMock),
        pytest.raises(IndexTerminatedError),
    ):
        await indexes.create(name="test-index", schema=DENSE_SCHEMA, timeout=300)


@respx.mock
async def test_create_timeout_raises(indexes: AsyncIndexes) -> None:
    respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(
            201, json=make_index_response(status={"ready": False, "state": "Initializing"})
        )
    )
    respx.get(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(
            200, json=make_index_response(status={"ready": False, "state": "Initializing"})
        )
    )

    with (
        patch("pinecone._internal.indexes_helpers.asyncio.sleep", new_callable=AsyncMock),
        patch(
            "pinecone._internal.indexes_helpers.time.monotonic",
            side_effect=itertools.count(start=0.0, step=0.5).__next__,
        ),
        pytest.raises(PineconeTimeoutError, match="not ready after"),
    ):
        await indexes.create(name="test-index", schema=DENSE_SCHEMA, timeout=1)


@respx.mock
async def test_create_duplicate_raises_conflict(indexes: AsyncIndexes) -> None:
    respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(409, json=make_error_response(409, "Index already exists"))
    )

    with pytest.raises(ConflictError):
        await indexes.create(name="existing-index", schema=DENSE_SCHEMA)
