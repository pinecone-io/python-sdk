"""Unit tests for create-readiness polling edge cases (2026-07 create)."""

from __future__ import annotations

import itertools
from typing import Any
from unittest.mock import patch

import httpx
import pytest
import respx

from pinecone._internal.config import PineconeConfig
from pinecone._internal.constants import CONTROL_PLANE_API_VERSION
from pinecone._internal.http_client import HTTPClient
from pinecone.client.indexes import _POLL_INTERVAL_SECONDS, Indexes
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
def http_client() -> HTTPClient:
    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    return HTTPClient(config, CONTROL_PLANE_API_VERSION)


@pytest.fixture
def indexes(http_client: HTTPClient) -> Indexes:
    return Indexes(http=http_client)


def test_poll_interval_is_five_seconds() -> None:
    assert _POLL_INTERVAL_SECONDS == 5


@respx.mock
def test_create_polling_sleeps_five_seconds(indexes: Indexes) -> None:
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

    with patch("pinecone._internal.indexes_helpers.time.sleep") as mock_sleep:
        indexes.create(name="test-index", schema=DENSE_SCHEMA, timeout=300)

    assert mock_sleep.call_count == 1
    mock_sleep.assert_called_with(5)


@respx.mock
def test_create_init_failed_raises_immediately(indexes: Indexes) -> None:
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
        patch("pinecone._internal.indexes_helpers.time.sleep"),
        pytest.raises(IndexInitFailedError) as exc_info,
    ):
        indexes.create(name="test-index", schema=DENSE_SCHEMA, timeout=300)

    assert exc_info.value.index_name == "test-index"


@respx.mock
def test_create_terminating_raises(indexes: Indexes) -> None:
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
        patch("pinecone._internal.indexes_helpers.time.sleep"),
        pytest.raises(IndexTerminatedError),
    ):
        indexes.create(name="test-index", schema=DENSE_SCHEMA, timeout=300)


@respx.mock
def test_create_timeout_raises(indexes: Indexes) -> None:
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
        patch("pinecone._internal.indexes_helpers.time.sleep"),
        patch(
            "pinecone._internal.indexes_helpers.time.monotonic",
            side_effect=itertools.count(start=0.0, step=0.5).__next__,
        ),
        pytest.raises(PineconeTimeoutError, match="not ready after"),
    ):
        indexes.create(name="test-index", schema=DENSE_SCHEMA, timeout=1)


@respx.mock
def test_create_duplicate_raises_conflict(indexes: Indexes) -> None:
    respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(409, json=make_error_response(409, "Index already exists"))
    )

    with pytest.raises(ConflictError):
        indexes.create(name="existing-index", schema=DENSE_SCHEMA)
